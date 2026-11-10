import LeanTest
import Tyr.EventSkeleton.Examples.CartPole

namespace Tests.EventSkeletonCartPoleExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.CartPole

private def pi : Float := 3.14159265358979323846

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertError {α : Type} (res : Except String α) (label : String) : IO String := do
  match res with
  | .ok _ => LeanTest.fail s!"{label}: expected error"
  | .error msg => pure msg

private def assertArrayNear
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

private def solve2x2Symmetric (a b d r0 r1 : Float) : Float × Float :=
  let det := a * d - b * b
  ((r0 * d - b * r1) / det, (a * r1 - b * r0) / det)

@[test]
def testDrakeReferencesAndNamedVectorsAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/multibody/cart_pole/BUILD.bazel"))
    "Cart-pole example should reference Drake's Bazel example target"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/multibody/cart_pole/cart_pole.sdf"))
    "Cart-pole example should reference Drake's SDF model"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/multibody/cart_pole/cart_pole_params.cc"))
    "Cart-pole example should reference Drake's parameter coordinate implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/multibody/cart_pole/test/cart_pole_test.cc"))
    "Cart-pole example should reference Drake's hand-written dynamics tests"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/multibody/cart_pole/cart_pole_passive_simulation.cc"))
    "Cart-pole example should reference Drake's passive MultibodyPlant executable"
  LeanTest.assertEqual stateCoordinateNames #["x", "theta", "xdot", "thetadot"]
    "Cart-pole state coordinate order should match Drake q/v order"
  LeanTest.assertEqual inputCoordinateNames #["cart_force"]
    "Cart-pole input should expose the single slider actuator force"
  LeanTest.assertEqual parameterCoordinateNames #["mc", "mp", "l", "gravity"]
    "Cart-pole parameter coordinate order should match CartPoleParams"
  LeanTest.assertTrue params.isValid "Default CartPoleParams should match Drake's valid domain"
  LeanTest.assertTrue defaultInput.isValid "Passive cart-pole input should be valid zero force"

@[test]
def testCartPoleParamsVectorBoundaryMatchesDrakeGeneratedVector : IO Unit := do
  assertOk cartPoleParamsVectorBoundary.validate? "CartPoleParams vector boundary"
  LeanTest.assertEqual (CartPoleParamsVectorBoundary.dimension cartPoleParamsVectorBoundary) 4
    "CartPoleParams should expose four generated-vector coordinates"
  LeanTest.assertEqual (CartPoleParamsVectorBoundary.indexOf? cartPoleParamsVectorBoundary "mc") (some 0)
    "mc should be parameter coordinate 0"
  LeanTest.assertEqual (CartPoleParamsVectorBoundary.indexOf? cartPoleParamsVectorBoundary "mp") (some 1)
    "mp should be parameter coordinate 1"
  LeanTest.assertEqual (CartPoleParamsVectorBoundary.indexOf? cartPoleParamsVectorBoundary "l") (some 2)
    "l should be parameter coordinate 2"
  LeanTest.assertEqual (CartPoleParamsVectorBoundary.indexOf? cartPoleParamsVectorBoundary "gravity") (some 3)
    "gravity should be parameter coordinate 3"
  LeanTest.assertEqual cartPoleParamsVectorBoundary.defaults #[10.0, 1.0, 0.5, 9.81]
    "CartPoleParams defaults should match cart_pole_params.h"
  LeanTest.assertEqual cartPoleParamsVectorBoundary.isValidLowerBounds
    #[some 0.0, some 0.0, some 0.0, some 0.0]
    "CartPoleParams IsValid domain should be nonnegative for all coordinates"
  LeanTest.assertEqual cartPoleParamsVectorBoundary.elementLowerBounds
    #[none, none, none, none]
    "CartPoleParams BasicVector element bounds should remain unbounded below"
  LeanTest.assertEqual cartPoleParamsVectorBoundary.elementUpperBounds
    #[none, none, none, none]
    "CartPoleParams BasicVector element bounds should remain unbounded above"

  let p ← assertOk (CartPoleParams.fromArray? params.asArray)
    "CartPoleParams from array"
  LeanTest.assertEqual p.asArray params.asArray
    "CartPoleParams array round trip should preserve Drake coordinate order"
  let x ← assertOk (CartPoleState.fromArray? #[1.0, 2.0, 3.0, 4.0])
    "CartPoleState from array"
  LeanTest.assertEqual x.asArray #[1.0, 2.0, 3.0, 4.0]
    "CartPoleState array round trip should preserve q/v order"
  let u ← assertOk (CartPoleInput.fromArray? #[12.0])
    "CartPoleInput from array"
  LeanTest.assertEqual u.asArray #[12.0]
    "CartPoleInput array round trip should preserve cart force"

  let badDim ← assertError (CartPoleParams.fromArray? #[1.0, 2.0])
    "CartPoleParams dimension check"
  LeanTest.assertTrue (badDim.contains "expects 4")
    s!"CartPoleParams dimension error should mention expected size, got {badDim}"
  let badDomain ← assertError (CartPoleParams.fromArray? #[10.0, 1.0, -0.5, 9.81])
    "CartPoleParams domain check"
  LeanTest.assertTrue (badDomain.contains "IsValid domain")
    s!"CartPoleParams domain error should mention IsValid, got {badDomain}"

@[test]
def testCartPoleModelAssetBoundaryRecordsSdfIdentity : IO Unit := do
  assertOk cartPoleModelAssetBoundary.validate? "CartPole SDF asset boundary"
  LeanTest.assertEqual cartPoleModelAssetBoundary.modelName "CartPole"
    "CartPole SDF model name should be recorded"
  LeanTest.assertEqual cartPoleModelAssetBoundary.sdfPath
    "../drake/examples/multibody/cart_pole/cart_pole.sdf"
    "CartPole SDF source path should be recorded"
  LeanTest.assertEqual cartPoleModelAssetBoundary.packageUri cartPoleModelUri
    "CartPole package URI should match the passive executable"
  LeanTest.assertEqual cartPoleModelAssetBoundary.linkNames #["Cart", "Pole"]
    "CartPole SDF links should be explicit"
  LeanTest.assertEqual cartPoleModelAssetBoundary.jointNames #["CartSlider", "PolePin"]
    "CartPole SDF joints should be explicit"
  LeanTest.assertEqual cartPoleModelAssetBoundary.jointTypes #["prismatic", "revolute"]
    "CartPole SDF joint types should be explicit"
  LeanTest.assertEqual cartPoleModelAssetBoundary.actuatorNames #["CartSlider"]
    "CartPole should expose one slider actuator"
  LeanTest.assertEqual cartPoleModelAssetBoundary.jointAxes
    #[#[1.0, 0.0, 0.0], #[0.0, -1.0, 0.0]]
    "CartPole SDF joint axes should match CartSlider and PolePin"
  LeanTest.assertTrue (approx cartPoleModelAssetBoundary.cartMass 10.0 1.0e-12)
    s!"Cart mass should be 10kg, got {cartPoleModelAssetBoundary.cartMass}"
  LeanTest.assertTrue (approx cartPoleModelAssetBoundary.polePointMass 1.0 1.0e-12)
    s!"Pole point mass should be 1kg, got {cartPoleModelAssetBoundary.polePointMass}"
  LeanTest.assertTrue (approx cartPoleModelAssetBoundary.poleLength 0.5 1.0e-12)
    s!"Pole length should be 0.5m, got {cartPoleModelAssetBoundary.poleLength}"
  LeanTest.assertEqual cartPoleModelAssetBoundary.cartBoxSize #[0.24, 0.12, 0.12]
    "Cart visual box dimensions should match the SDF"
  LeanTest.assertTrue (approx cartPoleModelAssetBoundary.polePointMassRadius 0.05 1.0e-12)
    s!"Point mass visual radius should be 0.05, got {cartPoleModelAssetBoundary.polePointMassRadius}"
  LeanTest.assertTrue (approx cartPoleModelAssetBoundary.poleRodRadius 0.025 1.0e-12)
    s!"Pole rod visual radius should be 0.025, got {cartPoleModelAssetBoundary.poleRodRadius}"
  LeanTest.assertTrue (approx cartPoleModelAssetBoundary.poleRodLength 0.5 1.0e-12)
    s!"Pole rod visual length should be 0.5, got {cartPoleModelAssetBoundary.poleRodLength}"

@[test]
def testMultibodyCartPolePassiveBenchmarkBoundaryIsRecorded : IO Unit := do
  let result ← assertOk buildMultibodyCartPole?
    "multibody cart-pole passive benchmark"
  assertOk result.asset.validate? "CartPole SDF asset boundary"
  assertOk result.config.validate? "CartPole passive benchmark config"
  assertOk result.step.validate? "CartPole passive FullMultibodyPlantStep"
  assertOk result.trace.validate? "CartPole passive benchmark trace"

  LeanTest.assertEqual result.step.model.modelUri cartPoleModelUri
    "CartPole passive benchmark should parse Drake's package SDF URL"
  LeanTest.assertEqual result.step.model.numPositions 2
    "CartPole MultibodyPlant should have two positions"
  LeanTest.assertEqual result.step.model.numVelocities 2
    "CartPole MultibodyPlant should have two velocities"
  LeanTest.assertEqual result.step.model.numActuatedDofs 1
    "CartPole MultibodyPlant should have one slider actuator"
  LeanTest.assertEqual result.step.q0 #[0.0, 2.0]
    "CartPole passive benchmark should set x=0 and theta=2"
  LeanTest.assertEqual result.step.v0 #[0.0, 0.0]
    "CartPole passive benchmark should leave velocities at zero"
  LeanTest.assertEqual result.step.actuation #[0.0]
    "CartPole passive benchmark should fix zero actuation"
  LeanTest.assertTrue (approx result.config.targetRealtimeRate 1.0 1.0e-12)
    s!"CartPole target_realtime_rate should default to 1, got {result.config.targetRealtimeRate}"
  LeanTest.assertTrue (approx result.config.simulationTime 10.0 1.0e-12)
    s!"CartPole simulation_time should default to 10, got {result.config.simulationTime}"
  LeanTest.assertTrue (approx result.config.timeStep 0.0 1.0e-12)
    s!"CartPole time_step should default to continuous mode, got {result.config.timeStep}"
  LeanTest.assertTrue result.config.visualizationEnabled
    "CartPole passive benchmark should include Drake's default visualization boundary"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.label == "full-physics-step:multibody cart-pole passive benchmark plant"))
    "Move list should expose the CartPole full-plant passive benchmark boundary"
  LeanTest.assertTrue result.step.config.isContinuous
    "CartPole passive benchmark should use continuous MultibodyPlant when time_step is zero"

  let x0 : CartPoleState := { x := 0.0, theta := 2.0, xdot := 0.0, thetadot := 0.0 }
  let expectedDx ← assertOk (derivative? params defaultInput x0)
    "CartPole full physics expected derivative"
  LeanTest.assertTrue (result.fullPhysics.support.policy == SupportPolicy.fullSupport)
    "CartPole has no contact candidates, so support selection should be exact full support"
  LeanTest.assertEqual result.fullPhysics.support.totalCandidates 0
    "CartPole full physics should have no contact candidates"
  LeanTest.assertEqual result.fullPhysics.contactForces.size 0
    "CartPole full physics should not invent contact forces"
  LeanTest.assertTrue (result.fullPhysics.supportMove.exactness == MoveExactness.exact)
    "Empty full-support contact selection should be exact"
  assertArrayNear result.fullPhysics.equation.massMatrix[0]!
    (massMatrix params x0)[0]! 1.0e-12
    "Full physics mass matrix row 0 should come from the Drake hand formula"
  assertArrayNear result.fullPhysics.equation.massMatrix[1]!
    (massMatrix params x0)[1]! 1.0e-12
    "Full physics mass matrix row 1 should come from the Drake hand formula"
  assertArrayNear result.fullPhysics.equation.biasForces
    (dynamicsBiasTerm params x0) 1.0e-12
    "Full physics bias should use the CartPole C*v - tau_g primitive"
  assertArrayNear result.fullPhysics.derivative.qdot #[0.0, 0.0] 1.0e-12
    "Full physics qdot should match the passive benchmark initial velocity"
  assertArrayNear result.fullPhysics.derivative.vdot
    #[expectedDx.xdot, expectedDx.thetadot] 1.0e-12
    "Full physics vdot should match the exact CartPole derivative"

@[test]
def testFullPhysicsPrimitiveProviderRecomputesStateAndInput : IO Unit := do
  let provider := fullPhysicsPrimitiveProvider params
    "cart-pole full physics provider test"
  let x0 := defaultState
  let u0 := defaultInput
  let x1 : CartPoleState :=
    { x := 0.3, theta := 1.1, xdot := -0.4, thetadot := 0.7 }
  let u1 : CartPoleInput := { cartForce := 3.5 }

  let primitive0 ← assertOk
    (provider.validPrimitivesAt? (physicsState x0 u0))
    "cart-pole provider primitive at default state and input"
  let primitive1 ← assertOk
    (provider.validPrimitivesAt? (physicsState x1 u1))
    "cart-pole provider primitive at moved state and input"
  let result1 ← assertOk
    (provider.solveAt? (physicsState x1 u1) 5355)
    "cart-pole provider solve at moved state and input"
  let direct1 ← assertOk
    (solveFullPhysics? params u1 x1 5356 "cart-pole direct provider parity")
    "cart-pole direct solve for provider parity"

  LeanTest.assertTrue
    (FloatArray.maxAbsDiff primitive0.massMatrix[0]! primitive1.massMatrix[0]! > 1.0e-2)
    "Changing theta should recompute the state-dependent CartPole mass matrix"
  assertArrayNear primitive1.qdot (qdotAsArray x1) 1.0e-12
    "CartPole provider qdot should come from the current state velocity"
  assertArrayNear primitive1.actuationForces #[u1.cartForce, 0.0] 1.0e-12
    "CartPole provider actuation should come from the current input"
  LeanTest.assertTrue
    (FloatArray.maxAbsDiff primitive0.biasForces primitive1.biasForces > 1.0)
    "Changing theta and thetadot should recompute gravity/Coriolis bias"
  assertArrayNear result1.derivative.vdot direct1.derivative.vdot 1.0e-12
    "Provider solve should match the direct full-physics solve"
  LeanTest.assertEqual result1.move.targets #[5355]
    "Provider solve should use the supplied interval vertex"

  let badInput : CartPoleInput := { cartForce := 1.0 / 0.0 }
  let msg ← assertError
    (provider.validPrimitivesAt? (physicsState x1 badInput))
    "cart-pole provider malformed input"
  LeanTest.assertTrue (msg.contains "input")
    s!"Malformed CartPole input should fail at provider validation, got {msg}"

@[test]
def testMassMatrixAndBiasMatchDrakeFormula : IO Unit := do
  let x : CartPoleState := { x := 2.5, theta := pi / 3.0, xdot := -1.5, thetadot := 0.5 }
  let m := massMatrix params x
  let bias := dynamicsBiasTerm params x
  let c := Float.cos x.theta
  let s := Float.sin x.theta
  let offdiag := params.mp * params.l * c
  let expectedCoriolis0 := -params.mp * params.l * x.thetadot * x.thetadot * s
  let expectedGravity1 := -params.mp * params.gravity * params.l * s

  LeanTest.assertTrue (approx ((m[0]!).getD 0 0.0) (params.mc + params.mp) 1.0e-12)
    s!"M00 should match Drake formula, got {reprStr m}"
  LeanTest.assertTrue (approx ((m[0]!).getD 1 0.0) offdiag 1.0e-12)
    s!"M01 should match Drake formula, got {reprStr m}"
  LeanTest.assertTrue (approx ((m[1]!).getD 0 0.0) offdiag 1.0e-12)
    s!"M10 should match Drake formula, got {reprStr m}"
  LeanTest.assertTrue (approx ((m[1]!).getD 1 0.0) (params.mp * params.l * params.l) 1.0e-12)
    s!"M11 should match Drake formula, got {reprStr m}"

  LeanTest.assertTrue (approx (bias.getD 0 99.0) expectedCoriolis0 1.0e-12)
    s!"Bias row 0 should be C*v - tau_g, got {reprStr bias}"
  LeanTest.assertTrue (approx (bias.getD 1 99.0) (-expectedGravity1) 1.0e-12)
    s!"Bias row 1 should be C*v - tau_g, got {reprStr bias}"

@[test]
def testDynamicsAndImplicitResidualMatchDrakeShape : IO Unit := do
  let x : CartPoleState := { x := 2.5, theta := pi / 3.0, xdot := -1.5, thetadot := 0.5 }
  let u : CartPoleInput := { cartForce := 0.0 }
  let dyn ← assertOk (derivative? params u x) "cart-pole derivative"
  let residual := implicitResidual params u x dyn
  let c := Float.cos x.theta
  let s := Float.sin x.theta
  let a := params.mc + params.mp
  let b := params.mp * params.l * c
  let d := params.mp * params.l * params.l
  let rhs0 := u.cartForce + params.mp * params.l * x.thetadot * x.thetadot * s
  let rhs1 := -params.mp * params.gravity * params.l * s
  let (expectedXddot, expectedThetaddot) := solve2x2Symmetric a b d rhs0 rhs1

  LeanTest.assertTrue (FloatArray.maxAbsDiff residual #[0.0, 0.0, 0.0, 0.0] < 1.0e-11)
    s!"Implicit residual for exact derivatives should be near zero, got {reprStr residual}"
  LeanTest.assertTrue (approx dyn.x x.xdot 1.0e-12)
    s!"x derivative should equal xdot, got {dyn.x}"
  LeanTest.assertTrue (approx dyn.theta x.thetadot 1.0e-12)
    s!"theta derivative should equal thetadot, got {dyn.theta}"
  LeanTest.assertTrue (approx dyn.xdot expectedXddot 1.0e-11)
    s!"xdot derivative should match Drake formula, got {dyn.xdot}, expected {expectedXddot}"
  LeanTest.assertTrue (approx dyn.thetadot expectedThetaddot 1.0e-11)
    s!"thetadot derivative should match Drake formula, got {dyn.thetadot}, expected {expectedThetaddot}"

@[test]
def testEnergyMatchesPointMassPoleModel : IO Unit := do
  let down : CartPoleState := { x := 0.0, theta := 0.0, xdot := 0.0, thetadot := 0.0 }
  let up : CartPoleState := { x := 0.0, theta := pi, xdot := 0.0, thetadot := 0.0 }
  let moving : CartPoleState := { x := 0.0, theta := 0.0, xdot := 1.0, thetadot := 0.0 }

  LeanTest.assertTrue
    (approx (totalEnergy params down) (-params.mp * params.gravity * params.l) 1.0e-12)
    s!"Downward pole energy should be -mp*g*l, got {totalEnergy params down}"
  LeanTest.assertTrue
    (approx (totalEnergy params up) (params.mp * params.gravity * params.l) 1.0e-12)
    s!"Upright pole energy should be mp*g*l, got {totalEnergy params up}"
  LeanTest.assertTrue
    (approx (kineticEnergy params moving) (0.5 * (params.mc + params.mp)) 1.0e-12)
    s!"Unit cart velocity should carry cart plus pole translational kinetic energy, got {kineticEnergy params moving}"

@[test]
def testPassiveSimulationRecordsContinuousInterval : IO Unit := do
  let run ← assertOk
    (solvePassive? params { x := 0.0, theta := 2.0, xdot := 0.0, thetadot := 0.0 } 0.0 0.05 defaultInput)
    "cart-pole passive solve"
  match run.trace.validate? with
  | .error msg => LeanTest.fail s!"Cart-pole trace should validate: {msg}"
  | .ok () => pure ()
  LeanTest.assertTrue (approx run.t1 0.05 1.0e-12)
    s!"Cart-pole run should reach requested final time, got {run.t1}"
  LeanTest.assertEqual run.moves.size 2
    "A pure continuous cart-pole interval should project to interval and checkpoint moves"
  LeanTest.assertTrue (run.moves[0]!.kind == SkeletonMoveKind.intervalAdjoint)
    "First cart-pole move should be the interval adjoint"
  LeanTest.assertTrue (Float.isFinite run.finalState.x && Float.isFinite run.finalState.theta &&
      Float.isFinite run.finalState.xdot && Float.isFinite run.finalState.thetadot)
    s!"Cart-pole final state should be finite, got {reprStr run.finalState}"
  LeanTest.assertTrue (Float.isFinite run.finalEnergy)
    s!"Cart-pole final energy should be finite, got {run.finalEnergy}"

end Tests.EventSkeletonCartPoleExample
