import LeanTest
import Tyr.EventSkeleton.Examples.InclinedPlaneBody

namespace Tests.EventSkeletonInclinedPlaneBodyExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.InclinedPlaneBody

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertArrayNear
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertError {α : Type} (res : Except String α) (label : String) :
    IO String := do
  match res with
  | .ok _ => LeanTest.fail s!"{label}: expected error, got ok"
  | .error msg => pure msg

@[test]
def testDrakeReferencesAndDefaultsAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/inclined_plane_with_body/inclined_plane_with_body.cc"))
    "Example should reference Drake's inclined-plane runner"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/multibody/benchmarks/inclined_plane/inclined_plane_plant.cc"))
    "Example should reference Drake's inclined-plane plant helper"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/multibody/plant/test/inclined_plane_test.cc"))
    "Example should reference Drake's analytic inclined-plane test"

  LeanTest.assertTrue (params.bodyType == BodyType.sphere)
    "Drake runner default bodyB_type should be sphere"
  LeanTest.assertTrue (params.planeShape == PlaneShape.halfSpace)
    "Drake runner default should use a half-space inclined plane"
  LeanTest.assertEqual params.contactApproximation "lagged"
    "Drake runner default contact approximation should be lagged"
  LeanTest.assertTrue (approx params.angleDegrees 15.0 1.0e-12)
    s!"Default plane angle should be 15 deg, got {params.angleDegrees}"
  LeanTest.assertTrue (approx params.sphereRadius 0.04 1.0e-12)
    s!"Default sphere radius should be 0.04, got {params.sphereRadius}"
  LeanTest.assertTrue (approx params.blockLengthX 0.4 1.0e-12)
    s!"Default block x length should be 0.4, got {params.blockLengthX}"
  LeanTest.assertTrue (approx params.planeFriction.staticFriction 0.3 1.0e-12)
    s!"Default plane static friction should be 0.3, got {params.planeFriction.staticFriction}"
  LeanTest.assertTrue (approx params.combinedFriction.staticFriction 0.3 1.0e-12)
    s!"Identical body/plane surface frictions should combine to 0.3, got {params.combinedFriction.staticFriction}"

  let spherePlaneBox := params.planeBoxDimensions
  LeanTest.assertTrue (approx spherePlaneBox.x (20.0 * params.sphereRadius) 1.0e-12)
    s!"Sphere case plane box x dimension should be 20*r, got {spherePlaneBox.x}"
  LeanTest.assertTrue (approx spherePlaneBox.y (10.0 * params.sphereRadius) 1.0e-12)
    s!"Sphere case plane box y dimension should be 10*r, got {spherePlaneBox.y}"

@[test]
def testPlaneFrameAndSphereCandidateKinematics : IO Unit := do
  let n := params.planeNormalW
  let tx := params.planeTangentXW
  let ty := params.planeTangentYW
  LeanTest.assertTrue (approx n.norm 1.0 1.0e-12)
    s!"Plane normal should be unit length, got norm {n.norm}"
  LeanTest.assertTrue (approx tx.norm 1.0 1.0e-12)
    s!"Plane downhill tangent should be unit length, got norm {tx.norm}"
  LeanTest.assertTrue (approx (n.dot tx) 0.0 1.0e-12)
    s!"Plane normal and downhill tangent should be orthogonal, got {n.dot tx}"
  LeanTest.assertTrue (approx (n.dot ty) 0.0 1.0e-12)
    s!"Plane normal and lateral tangent should be orthogonal, got {n.dot ty}"

  let airborneCandidate := (contactCandidates params defaultState)[0]!
  LeanTest.assertTrue (airborneCandidate.signedDistance > params.sphereRadius)
    s!"Drake default initial state should start above the inclined plane, got distance {airborneCandidate.signedDistance}"
  let airborneSupport := selectedSupport params defaultState
  LeanTest.assertEqual airborneSupport.selectedLocalIndices.size 0
    "Airborne default state should not select a contact support"

  let contactCandidate := (contactCandidates params (contactingSphereState params))[0]!
  LeanTest.assertTrue (approx contactCandidate.signedDistance 0.0 1.0e-12)
    s!"Contacting sphere center should put the sphere surface on the plane, got distance {contactCandidate.signedDistance}"
  LeanTest.assertTrue (approx contactCandidate.normalVelocity 0.0 1.0e-12)
    s!"Stationary contacting sphere should have zero normal velocity, got {contactCandidate.normalVelocity}"

  let impactSupport := selectedSupport params (impactingSphereState params)
  LeanTest.assertEqual impactSupport.selectedLocalIndices.size 1
    "Impacting sphere should select exactly one point contact"
  let selected ← assertOk impactSupport.selectedCandidates? "impact support candidates"
  LeanTest.assertTrue (selected[0]!.mode == ContactMode.impacting)
    "Closing normal velocity should classify the contact as impacting"
  LeanTest.assertTrue (approx selected[0]!.normalVelocity (-1.0) 1.0e-12)
    s!"Impacting helper state should close at unit normal speed, got {selected[0]!.normalVelocity}"

@[test]
def testBodyVariantsGenerateStableContactCandidates : IO Unit := do
  let blockParams : InclinedPlaneParams := { params with bodyType := .block }
  let sphereBlockParams : InclinedPlaneParams := { params with bodyType := .blockWith4Spheres }
  let blockCandidates := contactCandidates blockParams defaultState
  let sphereCandidates := contactCandidates sphereBlockParams defaultState

  LeanTest.assertEqual blockCandidates.size 4
    "Block contact provider should expose its four lower corners"
  LeanTest.assertEqual sphereCandidates.size 4
    "block_with_4Spheres contact provider should expose four welded collision spheres"
  LeanTest.assertEqual blockCandidates[0]!.id 2000
    "Block corner candidate ids should be stable"
  LeanTest.assertEqual sphereCandidates[0]!.id 3000
    "Four-sphere candidate ids should be stable"
  LeanTest.assertTrue
    (blockCandidates.all (fun candidate =>
      candidate.normalJacobian.size == 6 &&
      candidate.tangentJacobian.size == 6 &&
      candidate.tangentJacobian2.size == 6))
    "Block candidates should expose full 3D contact Jacobian rows"
  LeanTest.assertTrue
    (sphereCandidates.all (fun candidate =>
      candidate.normalJacobian.size == 6 &&
      candidate.tangentJacobian.size == 6 &&
      candidate.tangentJacobian2.size == 6))
    "Four-sphere candidates should expose full 3D contact Jacobian rows"

  let dims := blockParams.planeBoxDimensions
  LeanTest.assertTrue (approx dims.x (8.0 * blockParams.blockLengthX) 1.0e-12)
    s!"Block case plane box x dimension should be 8*LBx, got {dims.x}"
  LeanTest.assertTrue (approx dims.y (8.0 * blockParams.blockLengthY) 1.0e-12)
    s!"Block case plane box y dimension should be 8*LBy, got {dims.y}"

@[test]
def testRollingSphereContactDynamicsMatchAnalyticInclinedPlane : IO Unit := do
  let p : InclinedPlaneParams := {
    params with
    sphereRadius := 0.05
    gravity := 9.81
    planeFriction := { staticFriction := 1.0, dynamicFriction := 0.5 }
    bodyFriction := { staticFriction := 1.0, dynamicFriction := 0.5 }
  }
  let support := selectedSupport p (contactingSphereState p)
  let (dx, forces) ← assertOk
    (contactDerivative? p support (contactingSphereState p))
    "rolling sphere contact derivative"

  LeanTest.assertEqual forces.size 1
    "Sphere-plane contact should produce one force"
  LeanTest.assertTrue (approx forces[0]!.normalForce (normalForceTotal p) 1.0e-10)
    s!"Normal force should balance gravity normal component, got {forces[0]!.normalForce}"
  LeanTest.assertTrue
    (approx forces[0]!.tangentForce (-rollingSphereFrictionMagnitude p) 1.0e-10)
    s!"Static friction should be uphill with Drake's rolling magnitude, got {forces[0]!.tangentForce}"

  let accelerationW : Vec3 := { x := dx.vx, y := dx.vy, z := dx.vz }
  LeanTest.assertTrue
    (approx (p.planeTangentXW.dot accelerationW) (rollingSphereAcceleration p) 1.0e-10)
    s!"Downhill acceleration should match solid-sphere rolling value, got {p.planeTangentXW.dot accelerationW}"
  LeanTest.assertTrue (approx (p.planeNormalW.dot accelerationW) 0.0 1.0e-10)
    s!"Contact force should cancel normal acceleration, got {p.planeNormalW.dot accelerationW}"
  LeanTest.assertTrue
    (approx dx.wy (rollingSphereAcceleration p / p.sphereRadius) 1.0e-10)
    s!"Angular acceleration should satisfy rolling wy_dot = vdot/r, got {dx.wy}"

  let speed := rollingSphereSpeedAfterVerticalDrop p 0.2
  LeanTest.assertTrue (speed > 0.0)
    s!"Analytic rolling speed after a drop should be positive, got {speed}"

private def blockFirstCornerContactState (p : InclinedPlaneParams) :
    BodyState :=
  let r : Vec3 := {
    x := -p.blockLengthX / 2.0
    y := -p.blockLengthY / 2.0
    z := -p.blockLengthZ / 2.0
  }
  let center := Vec3.scale (-(p.planeNormalW.dot r)) p.planeNormalW
  { defaultState with x := center.x, y := center.y, z := center.z }

@[test]
def testFullPhysicsPrimitiveProviderRecomputesDynamicSupportAndBodyType :
    IO Unit := do
  let provider := fullPhysicsPrimitiveProvider params
    "inclined plane provider recompute test"
  let airborne := defaultState
  let contact := impactingSphereState params

  let airbornePrimitive ← assertOk
    (provider.validPrimitivesAt? airborne)
    "inclined plane provider primitive at airborne state"
  let contactPrimitive ← assertOk
    (provider.validPrimitivesAt? contact)
    "inclined plane provider primitive at contact state"
  let airborneSupport ← assertOk
    (provider.supportAt? airborne)
    "inclined plane provider airborne support"
  let contactSupport ← assertOk
    (provider.supportAt? contact)
    "inclined plane provider contact support"
  let contactResult ← assertOk
    (provider.solveAt? contact 902)
    "inclined plane provider contact solve"
  let directResult ← assertOk
    (solveFullPhysics? params contact 903)
    "inclined plane direct contact solve"

  LeanTest.assertEqual airbornePrimitive.contactCandidates.size 1
    "Airborne provider primitive should still expose the runtime sphere candidate view"
  LeanTest.assertTrue
    (airbornePrimitive.contactCandidates[0]!.signedDistance >
      contactPrimitive.contactCandidates[0]!.signedDistance)
    "Provider should recompute signed distance from the current body state"
  LeanTest.assertEqual airbornePrimitive.contactForces.size 0
    "Airborne provider primitive should not retain inactive contact force scalars"
  LeanTest.assertEqual airborneSupport.selectedLocalIndices #[]
    "Airborne provider support should be empty"
  LeanTest.assertEqual contactPrimitive.contactForces.size 1
    "Contact provider primitive should retain one selected contact force"
  LeanTest.assertEqual contactSupport.selectedLocalIndices #[0]
    "Contact provider support should select the sphere-plane candidate"
  LeanTest.assertEqual contactResult.move.targets #[902]
    "Provider solve should use the supplied interval vertex"
  assertArrayNear contactResult.derivative.vdot directResult.derivative.vdot 1.0e-12
    "Provider solve should match the direct full-physics solve"

  let blockParams : InclinedPlaneParams := { params with bodyType := .block }
  let blockProvider := fullPhysicsPrimitiveProvider blockParams
    "inclined plane block provider recompute test"
  let blockState := blockFirstCornerContactState blockParams
  let blockPrimitive ← assertOk
    (blockProvider.validPrimitivesAt? blockState)
    "inclined plane provider primitive for block body"
  let blockSupport ← assertOk
    (blockProvider.supportAt? blockState)
    "inclined plane provider support for block body"

  LeanTest.assertEqual blockPrimitive.contactCandidates.size 4
    "Block provider primitive should expose all four lower-corner candidates"
  LeanTest.assertEqual
    (blockPrimitive.contactCandidates.map (fun candidate => candidate.id))
    #[2000, 2001, 2002, 2003]
    "Block provider candidate ids should preserve the body-type geometry"
  LeanTest.assertEqual blockSupport.selectedLocalIndices #[0, 1]
    "Block provider support should select the two downhill contacting corners"
  LeanTest.assertEqual blockPrimitive.contactForces.size 2
    "Block provider primitive should keep force scalars aligned with selected contacts"

  let badState := { contact with wx := 1.0 / 0.0 }
  let msg ← assertError
    (provider.validPrimitivesAt? badState)
    "inclined plane provider malformed state"
  LeanTest.assertTrue (msg.contains "state")
    s!"Malformed inclined-plane state should fail at provider validation, got {msg}"

@[test]
def testEndToEndTraceAndStepExecute : IO Unit := do
  let result ← assertOk (buildEndToEnd? params) "inclined-plane end-to-end"
  let _ ← assertOk result.trace.validate? "inclined-plane trace validation"
  LeanTest.assertEqual result.contactSupport.selectedLocalIndices.size 1
    "End-to-end sphere contact should retain one dynamic contact candidate"
  LeanTest.assertEqual result.runtimeSupport.selectedIds #[1000]
    "Runtime support should preserve the source candidate id"
  LeanTest.assertEqual result.contactForces.size 1
    "End-to-end contact solve should produce one contact force"
  LeanTest.assertTrue (result.contactForces[0]!.tangentForce < 0.0)
    s!"Rolling sphere friction should act uphill in tangent coordinates, got {result.contactForces[0]!.tangentForce}"
  LeanTest.assertEqual result.fullPhysics.support.selectedLocalIndices
    result.contactSupport.selectedLocalIndices
    "Full physics primitive should recompute the same current-state contact support"
  LeanTest.assertEqual result.fullPhysics.contactForces.size result.contactForces.size
    "Full physics primitive should consume the selected force scalars"
  assertArrayNear result.fullPhysics.generalizedContactForce
    (aggregateGeneralizedForce result.contactForces)
    1.0e-12
    "Full physics primitive should use the same J^T f generalized contact force"
  assertArrayNear result.fullPhysics.derivative.qdot
    result.contactState.velocityVector
    1.0e-12
    "Full physics primitive qdot should come from the current body velocity"
  assertArrayNear result.fullPhysics.derivative.vdot
    #[result.derivative.vx, result.derivative.vy, result.derivative.vz,
      result.derivative.wx, result.derivative.wy, result.derivative.wz]
    1.0e-12
    "Full physics primitive should match the analytic rolling-sphere acceleration"
  LeanTest.assertTrue (result.fullPhysics.supportMove.kind == SkeletonMoveKind.markMarginalize)
    "Dynamic contact support should be a separate support-elimination move"
  LeanTest.assertTrue (result.fullPhysics.move.kind == SkeletonMoveKind.intervalAdjoint)
    "Mass-matrix full physics solve should be an interval-adjoint move"
  LeanTest.assertEqual result.moves.size 5
    "Interval, branch aggregate, contact-support selection, and full physics solve should all be visible"
  LeanTest.assertTrue result.oneStepState.isFinite
    s!"One-step state should remain finite, got {reprStr result.oneStepState}"
  LeanTest.assertTrue (result.branchResult.value > 0.0)
    s!"Branch aggregate should produce a nonzero contact message value, got {result.branchResult.value}"

end Tests.EventSkeletonInclinedPlaneBodyExample
