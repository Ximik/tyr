import Tyr.DiffEq.Integrate
import Tyr.DiffEq.Solver.RK4
import Tyr.DiffEq.Term
import Tyr.EventSkeleton.NamedVector
import Tyr.EventSkeleton.Physics
import Tyr.EventSkeleton.SceneGraph
import Tyr.EventSkeleton.Trace

/-!
# Drake Compass Gait Event-Skeleton Example

This is a solver-backed port of `../drake/examples/compass_gait`.
The continuous physics follows Drake's `CompassGait` plant:

* continuous state is `(stance, swing, stancedot, swingdot)`,
* stance dynamics solve the 2x2 manipulator equation,
* the foot-collision witness uses Drake's `max(collision, swing - stance)`
  scuffing guard,
* impact reset solves the floating-base impulse projection and then swaps the
  stance and swing legs.

The EventSkeleton layer records localized intervals and collision saltation
events around this executable physics path.
-/

namespace Tyr.EventSkeleton.Examples.CompassGait

open Tyr.EventSkeleton
open torch.DiffEq

private def pi : Float := 3.14159265358979323846

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/compass_gait/simulate.cc"
      concept := "builds the CompassGait + SceneGraph diagram, fixes zero hip torque, sets Drake's default initial continuous state, and advances the simulator to t=10"
    },
    {
      path := "../drake/examples/compass_gait/compass_gait.cc"
      concept := "defines mass matrix, bias term, foot-collision witness, impulse reset, toe update, and floating-base output"
    },
    {
      path := "../drake/examples/compass_gait/compass_gait.h"
      concept := "declares the CompassGait LeafSystem ports, collision witness function, energy output, floating-base output, and scalar-converting API"
    },
    {
      path := "../drake/examples/compass_gait/compass_gait_continuous_state.h"
      concept := "defines CompassGaitContinuousState coordinate order and BasicVector accessors"
    },
    {
      path := "../drake/examples/compass_gait/compass_gait_continuous_state.cc"
      concept := "defines CompassGaitContinuousStateIndices::GetCoordinateNames"
    },
    {
      path := "../drake/examples/compass_gait/compass_gait_geometry.h"
      concept := "connects the floating-base state output to SceneGraph geometry for Drake visualizer playback"
    },
    {
      path := "../drake/examples/compass_gait/compass_gait_geometry.cc"
      concept := "registers the ramp, left/right leg frames, hip, leg, and mass geometry, and emits two frame poses"
    },
    {
      path := "../drake/examples/compass_gait/test/compass_gait_geometry_test.cc"
      concept := "acceptance test for adding CompassGaitGeometry to a DiagramBuilder with CompassGait and SceneGraph"
    },
    {
      path := "../drake/examples/compass_gait/test/compass_gait_test.cc"
      concept := "checks energy conservation, hip-torque fixed point, collision guard, angular momentum, and outputs"
    },
    {
      path := "../drake/examples/compass_gait/compass_gait_params.h"
      concept := "defines hip/leg masses, leg length, leg center of mass, gravity, and ramp slope defaults"
    },
    {
      path := "../drake/examples/compass_gait/compass_gait_params.cc"
      concept := "defines CompassGaitParamsIndices::GetCoordinateNames"
    }
  ]

structure CompassParams where
  massHip : Float := 10.0
  massLeg : Float := 5.0
  lengthLeg : Float := 1.0
  centerOfMassLeg : Float := 0.5
  gravity : Float := 9.81
  slope : Float := 0.0525
  rootTol : Float := 1.0e-8
  stepSize : Float := 1.0e-3
  resetJacEps : Float := 1.0e-6
  deriving Repr, Inhabited

def params : CompassParams := {}

namespace CompassParams

def lowerBounds : Array (Option Float) :=
  #[some 0.0, some 0.0, some 0.0, some 0.0, some 0.0, some 0.0]

def upperBounds : Array (Option Float) :=
  #[none, none, none, none, none, some 1.5707]

def asArray (p : CompassParams) : Array Float :=
  #[p.massHip, p.massLeg, p.lengthLeg, p.centerOfMassLeg, p.gravity, p.slope]

def isValid (p : CompassParams) : Bool :=
  (asArray p).all (fun x => x.isFinite) &&
    p.massHip >= 0.0 &&
    p.massLeg >= 0.0 &&
    p.lengthLeg >= 0.0 &&
    p.centerOfMassLeg >= 0.0 &&
    p.gravity >= 0.0 &&
    p.slope >= 0.0 &&
    p.slope <= 1.5707

def fromArray? (xs : Array Float) : Except String CompassParams := do
  if xs.size != 6 then
    .error s!"CompassGaitParams expects 6 coordinates, got {xs.size}"
  let p : CompassParams := {
    massHip := xs[0]!
    massLeg := xs[1]!
    lengthLeg := xs[2]!
    centerOfMassLeg := xs[3]!
    gravity := xs[4]!
    slope := xs[5]!
  }
  if !p.isValid then
    .error s!"CompassGaitParams values are outside Drake's BasicVector domain: {reprStr xs}"
  pure p

end CompassParams

def compassGaitContinuousStateVectorBoundary : NamedVectorBoundary :=
  {
    typeName := "CompassGaitContinuousState"
    headerPath := "../drake/examples/compass_gait/compass_gait_continuous_state.h"
    implementationPath? := some "../drake/examples/compass_gait/compass_gait_continuous_state.cc"
    coordinateNames := #["stance", "swing", "stancedot", "swingdot"]
    defaults := #[0.0, 0.0, 0.0, 0.0]
    lowerBounds := #[none, none, none, none]
    upperBounds := #[none, none, none, none]
  }

def compassGaitParamsVectorBoundary : NamedVectorBoundary :=
  {
    typeName := "CompassGaitParams"
    headerPath := "../drake/examples/compass_gait/compass_gait_params.h"
    implementationPath? := some "../drake/examples/compass_gait/compass_gait_params.cc"
    coordinateNames := #[
      "mass_hip",
      "mass_leg",
      "length_leg",
      "center_of_mass_leg",
      "gravity",
      "slope"
    ]
    defaults := CompassParams.asArray params
    lowerBounds := CompassParams.lowerBounds
    upperBounds := CompassParams.upperBounds
  }

structure CompassState where
  stance : Float
  swing : Float
  stanceDot : Float
  swingDot : Float
  deriving Repr, Inhabited

instance : DiffEqSpace CompassState where
  add a b :=
    {
      stance := a.stance + b.stance
      swing := a.swing + b.swing
      stanceDot := a.stanceDot + b.stanceDot
      swingDot := a.swingDot + b.swingDot
    }
  sub a b :=
    {
      stance := a.stance - b.stance
      swing := a.swing - b.swing
      stanceDot := a.stanceDot - b.stanceDot
      swingDot := a.swingDot - b.swingDot
    }
  scale s x :=
    {
      stance := s * x.stance
      swing := s * x.swing
      stanceDot := s * x.stanceDot
      swingDot := s * x.swingDot
    }

instance : DiffEqSeminorm CompassState where
  rms x := max (max (Float.abs x.stance) (Float.abs x.swing))
    (max (Float.abs x.stanceDot) (Float.abs x.swingDot))

instance : DiffEqElem CompassState where
  abs x :=
    {
      stance := Float.abs x.stance
      swing := Float.abs x.swing
      stanceDot := Float.abs x.stanceDot
      swingDot := Float.abs x.swingDot
    }
  max a b :=
    {
      stance := max a.stance b.stance
      swing := max a.swing b.swing
      stanceDot := max a.stanceDot b.stanceDot
      swingDot := max a.swingDot b.swingDot
    }
  addScalar s x :=
    {
      stance := x.stance + s
      swing := x.swing + s
      stanceDot := x.stanceDot + s
      swingDot := x.swingDot + s
    }
  div a b :=
    {
      stance := a.stance / b.stance
      swing := a.swing / b.swing
      stanceDot := a.stanceDot / b.stanceDot
      swingDot := a.swingDot / b.swingDot
    }

structure CompassHybridState where
  cont : CompassState
  toe : Float := 0.0
  leftSupport : Bool := true
  deriving Repr, Inhabited

def initialState (_p : CompassParams := params) : CompassHybridState :=
  {
    cont := { stance := 0.0, swing := 0.0, stanceDot := 0.4, swingDot := -2.0 }
    toe := 0.0
    leftSupport := true
  }

def stateAsArray (x : CompassState) : Array Float :=
  #[x.stance, x.swing, x.stanceDot, x.swingDot]

def compassStateFinite (x : CompassState) : Bool :=
  x.stance.isFinite && x.swing.isFinite &&
    x.stanceDot.isFinite && x.swingDot.isFinite

def stateFromArray (xs : Array Float) : CompassState :=
  {
    stance := xs.getD 0 0.0
    swing := xs.getD 1 0.0
    stanceDot := xs.getD 2 0.0
    swingDot := xs.getD 3 0.0
  }

def stateFromArray? (xs : Array Float) : Except String CompassState := do
  if xs.size != 4 then
    .error s!"CompassGaitContinuousState expects 4 coordinates, got {xs.size}"
  let x := stateFromArray xs
  if !compassStateFinite x then
    .error s!"CompassGaitContinuousState values must be finite, got {reprStr xs}"
  pure x

def stateWithArray (x : CompassHybridState) (xs : Array Float) : CompassHybridState :=
  { x with cont := stateFromArray xs }

def legA (p : CompassParams) : Float :=
  p.lengthLeg - p.centerOfMassLeg

def legB (p : CompassParams) : Float :=
  p.centerOfMassLeg

def massMatrix (p : CompassParams) (x : CompassState) : (Float × Float) × (Float × Float) :=
  let m := p.massLeg
  let mh := p.massHip
  let a := legA p
  let b := legB p
  let l := p.lengthLeg
  let c := Float.cos (x.swing - x.stance)
  ((mh * l * l + m * (l * l + a * a), -m * l * b * c),
    (-m * l * b * c, m * b * b))

def dynamicsBiasTerm (p : CompassParams) (x : CompassState) : Float × Float :=
  let m := p.massLeg
  let mh := p.massHip
  let a := legA p
  let b := legB p
  let l := p.lengthLeg
  let g := p.gravity
  let s := Float.sin (x.stance - x.swing)
  let vst := x.stanceDot
  let vsw := x.swingDot
  (-m * l * b * vsw * vsw * s - (mh * l + m * (a + l)) * g * Float.sin x.stance,
    m * l * b * vst * vst * s + m * b * g * Float.sin x.swing)

/-- Solve the 2×2 mass-matrix system by Cramer's rule. -/
def acceleration (p : CompassParams) (torque : Float) (x : CompassState) : Float × Float :=
  let (b0, b1) := dynamicsBiasTerm p x
  let ((m00, m01), (m10, m11)) := massMatrix p x
  let r0 := -torque - b0
  let r1 := torque - b1
  let det := m00 * m11 - m01 * m10
  ((r0 * m11 - m01 * r1) / det, (m00 * r1 - m10 * r0) / det)

def derivative (p : CompassParams) (torque : Float) (x : CompassState) : CompassState :=
  let (stanceDdot, swingDdot) := acceleration p torque x
  {
    stance := x.stanceDot
    swing := x.swingDot
    stanceDot := stanceDdot
    swingDot := swingDdot
  }

def vectorFieldArray (p : CompassParams) (torque : Float) (x : CompassState) : Array Float :=
  stateAsArray (derivative p torque x)

def odeTerm (p : CompassParams) (torque : Float := 0.0) : ODETerm CompassState Unit :=
  { vectorField := fun _t x _ => derivative p torque x }

def kineticEnergy (p : CompassParams) (x : CompassState) : Float :=
  let m := p.massLeg
  let mh := p.massHip
  let l := p.lengthLeg
  let a := legA p
  let b := legB p
  let vst := x.stanceDot
  let vsw := x.swingDot
  0.5 * (mh * l * l + m * a * a) * vst * vst +
    0.5 * m * (l * l * vst * vst + b * b * vsw * vsw) -
      m * l * b * vst * vsw * Float.cos (x.swing - x.stance)

def potentialEnergy (p : CompassParams) (x : CompassHybridState) : Float :=
  let m := p.massLeg
  let mh := p.massHip
  let l := p.lengthLeg
  let a := legA p
  let b := legB p
  let g := p.gravity
  let yToe := -x.toe * Float.sin p.slope
  let yHip := yToe + l * Float.cos x.cont.stance
  m * g * (yToe + a * Float.cos x.cont.stance) +
    mh * g * yHip +
      m * g * (yHip - b * Float.cos x.cont.swing)

def totalEnergy (p : CompassParams) (x : CompassHybridState) : Float :=
  kineticEnergy p x.cont + potentialEnergy p x

def collisionHeightGuard (p : CompassParams) (x : CompassState) : Float :=
  2.0 * p.slope - x.stance - x.swing

def scuffingGuard (_p : CompassParams) (x : CompassState) : Float :=
  x.swing - x.stance

def footCollision (p : CompassParams) (x : CompassState) : Float :=
  max (collisionHeightGuard p x) (scuffingGuard p x)

def footCollisionGuardGrad (p : CompassParams) (x : CompassState) : Array Float :=
  if collisionHeightGuard p x >= scuffingGuard p x then
    #[-1.0, -1.0, 0.0, 0.0]
  else
    #[-1.0, 1.0, 0.0, 0.0]

def footCollisionEvent (p : CompassParams) : EventSpec CompassState Unit :=
  {
    condition := .real (fun _t x _ => footCollision p x)
    direction := some false
    terminate := true
    rootTol := p.rootTol
  }

def floatingMassMatrix (p : CompassParams) (x : CompassState) : Array (Array Float) :=
  let m := p.massLeg
  let mh := p.massHip
  let a := legA p
  let b := legB p
  let l := p.lengthLeg
  let cst := Float.cos x.stance
  let csw := Float.cos x.swing
  let c := Float.cos (x.swing - x.stance)
  let sst := Float.sin x.stance
  let ssw := Float.sin x.swing
  let common := m * a + m * l + mh * l
  #[
    #[2.0 * m + mh, 0.0, common * cst, -m * b * csw],
    #[0.0, 2.0 * m + mh, -common * sst, m * b * ssw],
    #[common * cst, -common * sst, m * a * a + (m + mh) * l * l, -m * l * b * c],
    #[-m * b * csw, m * b * ssw, -m * l * b * c, m * b * b]
  ]

def swingToeJacobian (p : CompassParams) (x : CompassState) : Array (Array Float) :=
  let l := p.lengthLeg
  let cst := Float.cos x.stance
  let csw := Float.cos x.swing
  let sst := Float.sin x.stance
  let ssw := Float.sin x.swing
  #[
    #[1.0, 0.0, l * cst, -l * csw],
    #[0.0, 1.0, -l * sst, l * ssw]
  ]

def postImpactVelocity? (p : CompassParams) (x : CompassState) :
    Except String (Array Float) := do
  let vPre := #[0.0, 0.0, x.stanceDot, x.swingDot]
  let projection ← VelocityProjection.project? (floatingMassMatrix p x) (swingToeJacobian p x) vPre
  pure projection.vPost

structure ResetResult where
  state : CompassHybridState
  vPost : Array Float
  deriving Repr, Inhabited

def applyReset? (p : CompassParams) (x : CompassHybridState) :
    Except String ResetResult := do
  let vPost ← postImpactVelocity? p x.cont
  let hipAngle := x.cont.swing - x.cont.stance
  let next : CompassHybridState := {
    cont := {
      stance := x.cont.swing
      swing := x.cont.stance
      stanceDot := vPost.getD 3 0.0
      swingDot := vPost.getD 2 0.0
    }
    toe := x.toe - 2.0 * p.lengthLeg * Float.sin (hipAngle / 2.0)
    leftSupport := !x.leftSupport
  }
  pure { state := next, vPost := vPost }

def perturbStateArray (xs : Array Float) (idx : Nat) (delta : Float) : Array Float :=
  xs.set! idx (xs.getD idx 0.0 + delta)

def resetJacobian? (p : CompassParams) (x : CompassHybridState) :
    Except String (Array (Array Float)) := do
  let eps := p.resetJacEps
  let xs := stateAsArray x.cont
  let mut cols : Array (Array Float) := #[]
  for j in [:4] do
    let plus ← applyReset? p (stateWithArray x (perturbStateArray xs j eps))
    let minus ← applyReset? p (stateWithArray x (perturbStateArray xs j (-eps)))
    let diff := FloatArray.scale (1.0 / (2.0 * eps))
      (FloatArray.sub (stateAsArray plus.state.cont) (stateAsArray minus.state.cont))
    cols := cols.push diff
  pure (FloatMatrix.transpose cols)

def stepSaltationData? (p : CompassParams) (torque : Float) (pre : CompassHybridState)
    (reset : ResetResult) :
    Except String SaltationData := do
  let resetJac ← resetJacobian? p pre
  let data :=
    SaltationData.mkFromFields
      resetJac
      (footCollisionGuardGrad p pre.cont)
      (vectorFieldArray p torque pre.cont)
      (vectorFieldArray p torque reset.state.cont)
  data.validateGamma
  pure data

def floatingBaseState (p : CompassParams) (x : CompassHybridState) : Array Float :=
  let left := if x.leftSupport then x.cont.stance else x.cont.swing
  let right := if x.leftSupport then x.cont.swing else x.cont.stance
  let leftDot := if x.leftSupport then x.cont.stanceDot else x.cont.swingDot
  let rightDot := if x.leftSupport then x.cont.swingDot else x.cont.stanceDot
  #[
    x.toe * Float.cos p.slope + p.lengthLeg * Float.sin x.cont.stance,
    0.0,
    -x.toe * Float.sin p.slope + p.lengthLeg * Float.cos x.cont.stance,
    0.0,
    left,
    0.0,
    right - left,
    x.cont.stanceDot * p.lengthLeg * Float.cos x.cont.stance,
    0.0,
    -x.cont.stanceDot * p.lengthLeg * Float.sin x.cont.stance,
    0.0,
    leftDot,
    0.0,
    rightDot - leftDot
  ]

/-! ## CompassGaitGeometry SceneGraph provider -/

def compassGaitGeometrySourceId : Nat := 7710
def compassGaitLeftLegFrameId : Nat := 7711
def compassGaitRightLegFrameId : Nat := 7712
def compassGaitRampGeometryId : Nat := 7713
def compassGaitHipGeometryId : Nat := 7714
def compassGaitLeftLegGeometryId : Nat := 7715
def compassGaitLeftLegMassGeometryId : Nat := 7716
def compassGaitRightLegGeometryId : Nat := 7717
def compassGaitRightLegMassGeometryId : Nat := 7718

def compassGaitGeometryStateInputVertex : VertexId := 7730
def compassGaitGeometryProviderVertex : VertexId := 7731
def compassGaitGeometryPoseOutputVertex : VertexId := 7732

private def compassGaitIllustrationProperties (rgba : SceneRgba) : SceneGeometryProperties :=
  { roles := #[.illustration], diffuseRgba? := some rgba }

def compassGaitLegMassRadius (p : CompassParams := params) : Float :=
  let hipMassRadius := 0.1
  Float.cbrt (Float.pow hipMassRadius 3.0 * p.massLeg / p.massHip)

def compassGaitGeometryProvider (p : CompassParams := params) : SceneGraphProvider :=
  let legMassRadius := compassGaitLegMassRadius p
  {
    sources := #[
      { id := compassGaitGeometrySourceId, name := "CompassGaitGeometry" }
    ]
    frames := #[
      {
        id := compassGaitLeftLegFrameId
        sourceId := compassGaitGeometrySourceId
        name := "left_leg"
      },
      {
        id := compassGaitRightLegFrameId
        sourceId := compassGaitGeometrySourceId
        name := "right_leg"
        parentFrameId? := some compassGaitLeftLegFrameId
      }
    ]
    geometries := #[
      {
        id := compassGaitRampGeometryId
        sourceId := compassGaitGeometrySourceId
        frameId? := none
        X_FG := {
          translation := { x := 0.0, y := 0.0, z := -5.0 }
          rotationAxis := SceneVec3.unitY
          rotationAngle := p.slope
        }
        shape := .box 100.0 1.0 10.0
        name := "ramp"
        properties :=
          compassGaitIllustrationProperties { r := 0.9297, g := 0.7930, b := 0.6758, a := 1.0 }
      },
      {
        id := compassGaitHipGeometryId
        sourceId := compassGaitGeometrySourceId
        frameId? := some compassGaitLeftLegFrameId
        X_FG := {
          translation := {}
          rotationAxis := SceneVec3.unitX
          rotationAngle := pi / 2.0
        }
        shape := .sphere 0.1
        name := "hip"
        properties := compassGaitIllustrationProperties { r := 0.0, g := 1.0, b := 0.0, a := 1.0 }
      },
      {
        id := compassGaitLeftLegGeometryId
        sourceId := compassGaitGeometrySourceId
        frameId? := some compassGaitLeftLegFrameId
        X_FG := { translation := { x := 0.0, y := 0.0, z := -p.lengthLeg / 2.0 } }
        shape := .cylinder 0.0075 p.lengthLeg
        name := "left_leg"
        properties := compassGaitIllustrationProperties { r := 1.0, g := 0.0, b := 0.0, a := 1.0 }
      },
      {
        id := compassGaitLeftLegMassGeometryId
        sourceId := compassGaitGeometrySourceId
        frameId? := some compassGaitLeftLegFrameId
        X_FG := { translation := { x := 0.0, y := 0.0, z := -p.centerOfMassLeg } }
        shape := .sphere legMassRadius
        name := "left_leg_mass"
        properties := compassGaitIllustrationProperties { r := 1.0, g := 0.0, b := 0.0, a := 1.0 }
      },
      {
        id := compassGaitRightLegGeometryId
        sourceId := compassGaitGeometrySourceId
        frameId? := some compassGaitRightLegFrameId
        X_FG := { translation := { x := 0.0, y := 0.0, z := -p.lengthLeg / 2.0 } }
        shape := .cylinder 0.0075 p.lengthLeg
        name := "right_leg"
        properties := compassGaitIllustrationProperties { r := 0.0, g := 0.0, b := 1.0, a := 1.0 }
      },
      {
        id := compassGaitRightLegMassGeometryId
        sourceId := compassGaitGeometrySourceId
        frameId? := some compassGaitRightLegFrameId
        X_FG := { translation := { x := 0.0, y := 0.0, z := -p.centerOfMassLeg } }
        shape := .sphere legMassRadius
        name := "right_leg_mass"
        properties := compassGaitIllustrationProperties { r := 0.0, g := 0.0, b := 1.0, a := 1.0 }
      }
    ]
    label := "CompassGaitGeometry SceneGraph provider"
  }

private def floatingBaseStateFinite (x : Array Float) : Bool :=
  x.all (fun xi => xi.isFinite)

def compassGaitGeometryPoseOutput
    (floatingState : Array Float) : SceneFramePoseVector :=
  {
    poses := #[
      {
        frameId := compassGaitLeftLegFrameId
        X_WF := ScenePose3.fromRollPitchYaw
          {
            x := floatingState.getD 0 0.0
            y := floatingState.getD 1 0.0
            z := floatingState.getD 2 0.0
          }
          (floatingState.getD 3 0.0)
          (floatingState.getD 4 0.0)
          (floatingState.getD 5 0.0)
      },
      {
        frameId := compassGaitRightLegFrameId
        X_WF := {
          translation := {}
          rotationAxis := SceneVec3.unitY
          rotationAngle := floatingState.getD 6 0.0
        }
      }
    ]
  }

private def compassGaitGeometryMove
    (target : VertexId) (label : String) (reads : Array VertexId := #[])
    (writes : Array VertexId := #[]) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[target]
    reads := reads
    writes := writes
    exactness := .exact
    label := label
  }

def compassGaitGeometryGraph : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex {
      id := compassGaitGeometryStateInputVertex
      kind := .state .boundary
      label := "CompassGaitGeometry floating_base_state input"
    }
    |>.addVertex {
      id := compassGaitGeometryProviderVertex
      kind := .state .boundary
      label := "CompassGaitGeometry registered SceneGraph source"
    }
    |>.addVertex {
      id := compassGaitGeometryPoseOutputVertex
      kind := .state .checkpoint
      label := "CompassGaitGeometry geometry_pose output"
    }
    |>.addMove (compassGaitGeometryMove compassGaitGeometryProviderVertex
      "Register ramp, left/right leg frames, hip, legs, and leg-mass geometry"
      #[] #[compassGaitGeometryProviderVertex])
    |>.addMove (compassGaitGeometryMove compassGaitGeometryPoseOutputVertex
      "OutputGeometryPose: floating_base_state -> left/right FramePoseVector"
      #[compassGaitGeometryStateInputVertex, compassGaitGeometryProviderVertex]
      #[compassGaitGeometryPoseOutputVertex])

structure CompassGaitGeometryResult where
  references : Array DrakeReference
  params : CompassParams
  inputPortName : String := "floating_base_state"
  inputPortSize : Nat := 14
  outputPortName : String := "geometry_pose"
  provider : SceneGraphProvider
  sampleFloatingBaseState : Array Float
  poses : SceneFramePoseVector
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildCompassGaitGeometry?
    (p : CompassParams := params)
    (floatingState : Array Float := floatingBaseState p (initialState p)) :
    Except String CompassGaitGeometryResult := do
  if !p.massHip.isFinite || p.massHip <= 0.0 then
    .error "CompassGaitGeometry requires positive finite hip mass"
  if !p.massLeg.isFinite || p.massLeg <= 0.0 then
    .error "CompassGaitGeometry requires positive finite leg mass"
  if !p.lengthLeg.isFinite || p.lengthLeg <= 0.0 then
    .error "CompassGaitGeometry requires positive finite leg length"
  if !p.centerOfMassLeg.isFinite || p.centerOfMassLeg <= 0.0 then
    .error "CompassGaitGeometry requires positive finite leg center of mass"
  if !p.slope.isFinite then
    .error "CompassGaitGeometry requires finite ramp slope"
  if floatingState.size != 14 then
    .error s!"CompassGaitGeometry floating_base_state must have size 14, got {floatingState.size}"
  if !floatingBaseStateFinite floatingState then
    .error "CompassGaitGeometry floating_base_state must be finite"
  let provider := compassGaitGeometryProvider p
  provider.validate?
  let poses := compassGaitGeometryPoseOutput floatingState
  poses.validate? provider
  pure {
    references := drakeReferences
    params := p
    provider := provider
    sampleFloatingBaseState := floatingState
    poses := poses
    graph := compassGaitGeometryGraph
    moves := compassGaitGeometryGraph.moves
  }

def angularMomentum (p : CompassParams) (x : CompassHybridState)
    (aboutStanceFoot : Bool) : Float :=
  let m := p.massLeg
  let mh := p.massHip
  let a := legA p
  let b := legB p
  let l := p.lengthLeg
  let cst := Float.cos x.cont.stance
  let csw := Float.cos x.cont.swing
  let sst := Float.sin x.cont.stance
  let ssw := Float.sin x.cont.swing
  let vst := x.cont.stanceDot
  let vsw := x.cont.swingDot
  let pStance := #[a * sst, a * cst]
  let vStance := #[a * cst * vst, -a * sst * vst]
  let pHip := #[l * sst, l * cst]
  let vHip := #[l * cst * vst, -l * sst * vst]
  let pSwing := FloatArray.sub pHip #[b * ssw, b * csw]
  let vSwing := FloatArray.sub vHip #[b * csw * vsw, -b * ssw * vsw]
  let origin :=
    if aboutStanceFoot then
      #[0.0, 0.0]
    else
      FloatArray.sub pHip #[l * ssw, l * csw]
  let cross (u v : Array Float) : Float :=
    u.getD 1 0.0 * v.getD 0 0.0 - u.getD 0 0.0 * v.getD 1 0.0
  cross (FloatArray.sub pStance origin) (FloatArray.scale m vStance) +
    cross (FloatArray.sub pHip origin) (FloatArray.scale mh vHip) +
      cross (FloatArray.sub pSwing origin) (FloatArray.scale m vSwing)

structure IntervalSolve where
  tStart : Float
  tAttempt : Float
  tAfter : Float
  stateAfter : CompassState
  result : Result
  deriving Repr, Inhabited

def compassSolver :=
  RK4.solver
    (Term := ODETerm CompassState Unit)
    (Y := CompassState)
    (VF := CompassState)
    (Args := Unit)

def solveInterval? (p : CompassParams) (torque : Float)
    (tStart tAttempt : Float) (x0 : CompassState) :
    Except String IntervalSolve := do
  let sol :=
    diffeqsolve
      (Term := ODETerm CompassState Unit)
      (Y := CompassState)
      (VF := CompassState)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      (odeTerm p torque) compassSolver tStart tAttempt (some p.stepSize) x0 ()
      (saveat := { t1 := true })
      (event := some (footCollisionEvent p))
  if !sol.result.isOkay then
    .error s!"compass-gait solve failed: {reprStr sol.result}"
  else
    match sol.ts, sol.ys with
    | some ts, some ys =>
        if ts.size == 0 || ys.size == 0 then
          .error "compass-gait solve did not save endpoint"
        else
          pure {
            tStart := tStart
            tAttempt := tAttempt
            tAfter := ts[ts.size - 1]!
            stateAfter := ys[ys.size - 1]!
            result := sol.result
          }
    | _, _ => .error "compass-gait solve did not save endpoint arrays"

structure StepRecord where
  eventIndex : Nat
  time : Float
  preState : CompassHybridState
  postState : CompassHybridState
  saltation : SaltationData
  angularMomentumBefore : Float
  angularMomentumAfter : Float
  deriving Repr, Inhabited

structure SimulationResult where
  references : Array DrakeReference
  finalTime : Float
  finalState : CompassHybridState
  steps : Array StepRecord
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def intervalSegment
    (idx : Nat)
    (solve : IntervalSolve)
    (madeJumpAfter : Bool) : AcceptedStepSegment :=
  {
    id := idx
    attemptIndex := idx
    tStart := solve.tStart
    tAttempt := solve.tAttempt
    tAfter := solve.tAfter
    madeJumpAfter := madeJumpAfter
    label :=
      if madeJumpAfter then
        s!"compass-gait localized stance interval {idx}"
      else
        s!"compass-gait terminal stance interval {idx}"
  }

def collisionEventVertex (idx : Nat) : VertexId :=
  700 + idx

def simulateLoop?
    (p : CompassParams)
    (torque : Float)
    (tFinal : Float)
    (fuel : Nat)
    (idx : Nat)
    (t : Float)
    (x : CompassHybridState)
    (trace : DynamicEventTrace)
    (steps : Array StepRecord) :
    Except String SimulationResult :=
  match fuel with
  | 0 => .error s!"compass-gait simulation exceeded collision budget at t={t}"
  | fuel' + 1 => do
      if t >= tFinal then
        trace.validate?
        pure {
          references := drakeReferences
          finalTime := t
          finalState := x
          steps := steps
          trace := trace
          moves := trace.moves
        }
      else
        let solve ← solveInterval? p torque t tFinal x.cont
        match solve.result with
        | Result.eventOccurred =>
            let pre : CompassHybridState :=
              { cont := solve.stateAfter, toe := x.toe, leftSupport := x.leftSupport }
            let reset ← applyReset? p pre
            let saltation ← stepSaltationData? p torque pre reset
            let segment := intervalSegment idx solve true
            let trace' :=
              trace
                |>.push (.interval segment)
                |>.push (.saltation (collisionEventVertex idx) saltation)
            let steps' := steps.push {
              eventIndex := idx
              time := solve.tAfter
              preState := pre
              postState := reset.state
              saltation := saltation
              angularMomentumBefore := angularMomentum p pre false
              angularMomentumAfter := angularMomentum p reset.state true
            }
            simulateLoop? p torque tFinal fuel' (idx + 1) solve.tAfter reset.state trace' steps'
        | Result.successful =>
            let segment := intervalSegment idx solve false
            let trace' := trace.push (.interval segment)
            trace'.validate?
            pure {
              references := drakeReferences
              finalTime := solve.tAfter
              finalState := { cont := solve.stateAfter, toe := x.toe, leftSupport := x.leftSupport }
              steps := steps
              trace := trace'
              moves := trace'.moves
            }
        | other =>
            .error s!"unexpected okay result from compass-gait solve: {reprStr other}"

def simulate? (p : CompassParams := params) (tFinal : Float := 10.0)
    (x0 : CompassHybridState := initialState p) (torque : Float := 0.0)
    (maxCollisions : Nat := 256) :
    Except String SimulationResult :=
  simulateLoop? p torque tFinal maxCollisions 0 0.0 x0 DynamicEventTrace.empty #[]

structure SimulateExecutableConfig where
  targetRealtimeRate : Float := 1.0
  accuracy : Float := 1.0e-4
  advanceTo : Float := 10.0
  inputTorque : Float := 0.0
  maxCollisions : Nat := 256
  initialContinuousState : CompassState :=
    { stance := 0.0, swing := 0.0, stanceDot := 0.4, swingDot := -2.0 }
  initialToe : Float := 0.0
  initialLeftSupport : Bool := true
  plantName : String := "compass_gait"
  includeSceneGraph : Bool := true
  includeDrakeVisualizer : Bool := true
  floatingBaseOutputPort : String := "floating_base_state"
  deriving Repr, Inhabited

namespace SimulateExecutableConfig

def initialHybridState (config : SimulateExecutableConfig) : CompassHybridState :=
  {
    cont := config.initialContinuousState
    toe := config.initialToe
    leftSupport := config.initialLeftSupport
  }

def validate? (config : SimulateExecutableConfig) (_p : CompassParams) :
    Except String Unit := do
  if config.plantName == "" then
    .error "compass-gait simulate plant name must be nonempty"
  if config.floatingBaseOutputPort == "" then
    .error "compass-gait geometry output port name must be nonempty"
  if !config.targetRealtimeRate.isFinite || config.targetRealtimeRate < 0.0 then
    .error s!"compass-gait simulate target realtime rate must be nonnegative and finite, got {config.targetRealtimeRate}"
  if !config.accuracy.isFinite || config.accuracy <= 0.0 then
    .error s!"compass-gait simulate accuracy must be positive and finite, got {config.accuracy}"
  if !config.advanceTo.isFinite || config.advanceTo <= 0.0 then
    .error s!"compass-gait simulate AdvanceTo time must be positive and finite, got {config.advanceTo}"
  if !config.inputTorque.isFinite then
    .error s!"compass-gait fixed input torque must be finite, got {config.inputTorque}"
  if config.maxCollisions == 0 then
    .error "compass-gait simulate collision budget must be positive"
  if !compassStateFinite config.initialContinuousState || !config.initialToe.isFinite then
    .error "compass-gait simulate initial state must be finite"
  if !config.includeSceneGraph then
    .error "compass-gait simulate.cc boundary should include SceneGraph geometry"
  if !config.includeDrakeVisualizer then
    .error "compass-gait simulate.cc boundary should include DrakeVisualizerd"

end SimulateExecutableConfig

def simulateExecutableConfig : SimulateExecutableConfig := {}

def simulateExecutableGraph
    (config : SimulateExecutableConfig := simulateExecutableConfig) :
    SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 7700, kind := .state .boundary, label := "../drake/examples/compass_gait/simulate.cc flags" }
    |>.addVertex { id := 7701, kind := .state .interior, label := config.plantName }
    |>.addVertex { id := 7702, kind := .state .interior, label := "CompassGaitGeometry + SceneGraph" }
    |>.addVertex { id := 7703, kind := .state .interior, label := "DrakeVisualizerd" }
    |>.addVertex { id := 7704, kind := .frozen, label := "fixed zero hip torque input" }
    |>.addVertex { id := 7705, kind := .state .checkpoint, label := "initial CompassGait context" }
    |>.addVertex { id := 7706, kind := .interval, label := s!"Simulator.AdvanceTo({config.advanceTo})" }
    |>.addVertex { id := 7707, kind := .state .checkpoint, label := "final CompassGait context" }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[7701]
      reads := #[7700]
      writes := #[7701]
      label := "DiagramBuilder.AddSystem<CompassGait>"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[7702]
      reads := #[7701]
      writes := #[7702]
      label := s!"CompassGaitGeometry::AddToBuilder from {config.floatingBaseOutputPort}"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[7703]
      reads := #[7702]
      writes := #[7703]
      label := "DrakeVisualizerd::AddToBuilder"
    }
    |>.addMove {
      kind := .freezeControl
      targets := #[7704]
      reads := #[7700]
      writes := #[7704]
      label := s!"FixValue input port 0 to Vector1d({config.inputTorque})"
    }
    |>.addMove {
      kind := .checkpointBoundary
      targets := #[7705]
      reads := #[7700, 7701, 7704]
      writes := #[7705]
      label := s!"set stance/swing rates and accuracy={config.accuracy}"
    }
    |>.addMove {
      kind := .intervalAdjoint
      targets := #[7706]
      reads := #[7701, 7704, 7705]
      writes := #[7707]
      cost := { work := config.advanceTo }
      label := "Simulator.AdvanceTo via compass-gait ODE, collision witness, impulse projection, and leg-swap reset"
    }
    |>.addMove {
      kind := .checkpointBoundary
      targets := #[7707]
      reads := #[7706]
      writes := #[7707]
      label := "store CompassGait final context checkpoint"
    }

structure SimulateExecutableBoundary where
  config : SimulateExecutableConfig
  initialState : CompassHybridState
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildSimulateExecutableBoundary?
    (p : CompassParams := params)
    (config : SimulateExecutableConfig := simulateExecutableConfig) :
    Except String SimulateExecutableBoundary := do
  config.validate? p
  let graph := simulateExecutableGraph config
  pure {
    config := config
    initialState := config.initialHybridState
    graph := graph
    moves := graph.moves
  }

def executeSimulateExecutable?
    (p : CompassParams := params)
    (config : SimulateExecutableConfig := simulateExecutableConfig) :
    Except String SimulationResult := do
  config.validate? p
  simulate? p config.advanceTo config.initialHybridState
    config.inputTorque config.maxCollisions

def continuousStateBoundaryVertex : VertexId := 7740

def paramsBoundaryVertex : VertexId := 7741

private def generatedVectorBoundaryMove
    (vertex : VertexId) (boundary : NamedVectorBoundary) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[vertex]
    reads := #[vertex]
    writes := #[vertex]
    exactness := .exact
    cost := { work := boundary.dimension.toFloat, memory := boundary.dimension.toFloat }
    label := s!"generated BasicVector boundary: {boundary.typeName}"
  }

structure CompassGaitResult where
  references : Array DrakeReference
  params : CompassParams
  continuousStateBoundary : NamedVectorBoundary
  paramsBoundary : NamedVectorBoundary
  geometry : CompassGaitGeometryResult
  executableBoundary : SimulateExecutableBoundary
  executableRun : SimulationResult
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildEndToEnd?
    (p : CompassParams := params)
    (config : SimulateExecutableConfig := simulateExecutableConfig) :
    Except String CompassGaitResult := do
  if !p.isValid then
    .error s!"CompassGait params are outside Drake's BasicVector domain: {reprStr (CompassParams.asArray p)}"
  compassGaitContinuousStateVectorBoundary.validate?
  compassGaitParamsVectorBoundary.validate?
  config.validate? p
  let executableBoundary ← buildSimulateExecutableBoundary? p config
  let geometry ← buildCompassGaitGeometry? p
    (floatingBaseState p executableBoundary.initialState)
  let executableRun ← executeSimulateExecutable? p config
  executableRun.trace.validate?
  let moves :=
    #[
      generatedVectorBoundaryMove continuousStateBoundaryVertex
        compassGaitContinuousStateVectorBoundary,
      generatedVectorBoundaryMove paramsBoundaryVertex
        compassGaitParamsVectorBoundary
    ] ++ geometry.moves ++ executableBoundary.moves ++ executableRun.moves
  pure {
    references := drakeReferences
    params := p
    continuousStateBoundary := compassGaitContinuousStateVectorBoundary
    paramsBoundary := compassGaitParamsVectorBoundary
    geometry := geometry
    executableBoundary := executableBoundary
    executableRun := executableRun
    moves := moves
  }

end Tyr.EventSkeleton.Examples.CompassGait
