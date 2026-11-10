import LeanTest
import Tyr.EventSkeleton.Examples.BouncingBall

namespace Tests.EventSkeletonBouncingBallExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.BouncingBall

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertError {α : Type} (res : Except String α) (label : String) :
    IO String := do
  match res with
  | .ok _ => LeanTest.fail s!"{label}: expected error, got ok"
  | .error msg => pure msg

private def countMoveKind (moves : Array SkeletonMove) (kind : SkeletonMoveKind) : Nat :=
  (moves.filter (fun move => move.kind == kind)).size

private def hasMoveLabel (moves : Array SkeletonMove) (needle : String) : Bool :=
  moves.any (fun move => move.label.contains needle)

private def testParams : BallParams :=
  { params with initialHeight := 1.0, initialVelocity := 0.0, restitution := 1.0 }

private def getTenSecondRun : IO SimulationResult := do
  assertOk (simulate? testParams 10.0 (initialState testParams) 128) "ten-second bouncing-ball run"

@[test]
def testDrakeReferencesAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/bouncing_ball/bouncing_ball.h"))
    "Example should reference Drake's BouncingBall system"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/bouncing_ball/bouncing_ball.cc"))
    "Example should reference Drake's default-scalar instantiation source"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/bouncing_ball/test/bouncing_ball_test.cc"))
    "Example should reference Drake's closed-form repeated-impact test"

@[test]
def testLeafSystemBoundaryMatchesDrakeDeclaration : IO Unit := do
  assertOk systemBoundary.validate? "BouncingBall system boundary"
  LeanTest.assertEqual stateCoordinateNames #["q", "v"]
    "Top-level Drake BouncingBall uses q/v continuous state coordinates"
  LeanTest.assertEqual systemBoundary.outputPortName "y0"
    "BouncingBall should expose the state output port"
  LeanTest.assertEqual systemBoundary.continuousStateCount 2
    "BouncingBall should declare two continuous state variables"
  LeanTest.assertEqual systemBoundary.positionCount 1
    "BouncingBall should declare one q coordinate"
  LeanTest.assertEqual systemBoundary.velocityCount 1
    "BouncingBall should declare one v coordinate"
  LeanTest.assertEqual systemBoundary.miscCount 0
    "BouncingBall should declare no z coordinates"
  LeanTest.assertEqual systemBoundary.witnessName "Signed distance"
    "BouncingBall witness should match Drake's named witness function"
  LeanTest.assertEqual systemBoundary.witnessDirection "kPositiveThenNonPositive"
    "BouncingBall witness should localize downward ground crossings"
  LeanTest.assertTrue systemBoundary.unrestrictedUpdateEvent
    "BouncingBall witness should dispatch an unrestricted update"
  LeanTest.assertEqual systemBoundary.defaultState #[10.0, 0.0]
    "BouncingBall default state should match SetDefaultState"
  LeanTest.assertTrue (approx systemBoundary.gravitationalAcceleration (-9.81) 1.0e-12)
    s!"BouncingBall gravity should be -9.81, got {systemBoundary.gravitationalAcceleration}"
  LeanTest.assertTrue systemBoundary.scalarConversionOnDefaultScalars
    "BouncingBall.cc should instantiate default scalar conversions"

  let roundTrip ← assertOk (stateFromArray? systemBoundary.defaultState)
    "BouncingBall default-state vector round trip"
  LeanTest.assertTrue (approx roundTrip.height 10.0 1.0e-12)
    s!"Default q should be 10, got {roundTrip.height}"
  LeanTest.assertTrue (approx roundTrip.velocity 0.0 1.0e-12)
    s!"Default v should be 0, got {roundTrip.velocity}"
  match stateFromArray? #[1.0] with
  | .ok value => LeanTest.fail s!"Short BouncingBall state vector should be rejected, got {reprStr value}"
  | .error _ => pure ()

@[test]
def testDynamicsAndImpactResetMatchDrakeModel : IO Unit := do
  let x0 := initialState params
  let dx0 := derivative params x0
  LeanTest.assertTrue (approx dx0.height params.initialVelocity 1.0e-12)
    s!"Height derivative should be velocity, got {dx0.height}"
  LeanTest.assertTrue (approx dx0.velocity params.gravity 1.0e-12)
    s!"Velocity derivative should be gravity, got {dx0.velocity}"

  let pre : BallState := { height := 0.0, velocity := -2.0 }
  let post := resetState params pre
  LeanTest.assertTrue (approx post.height 0.0 1.0e-12)
    s!"Impact reset should keep the localized height at zero, got {post.height}"
  LeanTest.assertTrue (approx post.velocity 2.0 1.0e-12)
    s!"Impact reset should reverse velocity for e=1, got {post.velocity}"

@[test]
def testFreeFlightFullPhysicsPrimitiveMatchesDerivative : IO Unit := do
  let x : BallState := { height := 3.0, velocity := -1.25 }
  let result ← assertOk (solveFreeFlightPrimitivePhysics? params x 404)
    "BouncingBall free-flight primitive physics"
  LeanTest.assertEqual result.support.totalCandidates 0
    "BouncingBall free flight has no contact candidates before witness reset"
  LeanTest.assertEqual result.contactForces.size 0
    "BouncingBall free flight should not synthesize contact forces"
  LeanTest.assertEqual result.derivative.qdot #[x.velocity]
    "Primitive qdot should be the current velocity"
  LeanTest.assertTrue (approx (result.derivative.vdot.getD 0 0.0) params.gravity 1.0e-12)
    s!"Primitive vdot should equal gravitational acceleration, got {result.derivative.vdot}"
  LeanTest.assertEqual result.equation.massMatrix #[#[1.0]]
    "The primitive mass matrix should be the scalar unit mass"
  LeanTest.assertEqual result.equation.biasForces #[-params.gravity]
    "Gravity should enter the manipulator primitive as a positive downward bias"
  LeanTest.assertEqual result.move.targets #[404]
    "The primitive full-physics move should target the supplied interval vertex"

@[test]
def testFullPhysicsPrimitiveProviderRecomputesState : IO Unit := do
  let provider := fullPhysicsPrimitiveProvider params
    "bouncing-ball provider recompute test"
  let x0 : BallState := { height := 3.0, velocity := -1.25 }
  let x1 : BallState := { height := 0.25, velocity := 2.75 }

  let primitive0 ← assertOk (provider.validPrimitivesAt? x0)
    "BouncingBall provider primitive at falling state"
  let primitive1 ← assertOk (provider.validPrimitivesAt? x1)
    "BouncingBall provider primitive at rising state"
  let support1 ← assertOk (provider.supportAt? x1)
    "BouncingBall provider support at rising state"
  let result1 ← assertOk (provider.solveAt? x1 405)
    "BouncingBall provider solve at rising state"
  let direct1 ← assertOk (solveFreeFlightPrimitivePhysics? params x1 405
    "bouncing-ball direct provider parity")
    "BouncingBall direct solve for provider parity"

  LeanTest.assertEqual primitive0.qdot #[x0.velocity]
    "Provider qdot should use the falling state's current velocity"
  LeanTest.assertEqual primitive1.qdot #[x1.velocity]
    "Provider qdot should recompute from the rising state's current velocity"
  LeanTest.assertEqual support1.selectedLocalIndices #[]
    "Free-flight provider should not synthesize contact support before the witness reset"
  LeanTest.assertEqual result1.derivative.qdot direct1.derivative.qdot
    "Provider solve qdot should match the direct primitive solve"
  LeanTest.assertEqual result1.derivative.vdot direct1.derivative.vdot
    "Provider solve acceleration should match the direct primitive solve"
  LeanTest.assertEqual result1.move.targets #[405]
    "Provider solve should target the supplied interval vertex"

  let badState : BallState := { height := 1.0 / 0.0, velocity := 0.0 }
  let msg ← assertError (provider.validPrimitivesAt? badState)
    "BouncingBall provider malformed state"
  LeanTest.assertTrue (msg.contains "state")
    s!"Malformed BouncingBall state should fail at provider validation, got {msg}"

@[test]
def testImpactSaltationContainsTimingAndRestitutionDerivative : IO Unit := do
  let pre : BallState := { height := 0.0, velocity := -2.0 }
  let data := impactSaltationData params pre
  LeanTest.assertTrue (approx data.gamma (-2.0) 1.0e-12)
    s!"Guard transversality gamma should be pre-impact velocity, got {data.gamma}"
  LeanTest.assertTrue (FloatArray.maxAbsDiff data.a #[4.0, 2.0 * params.gravity] < 1.0e-12)
    s!"Unexpected saltation a vector: {reprStr data.a}"
  let restitutionGrad ← assertOk (data.reverseTheta? #[0.0, 1.0]) "restitution VJP"
  LeanTest.assertEqual restitutionGrad.size 1
    "Restitution parameter VJP should have one component"
  LeanTest.assertTrue (approx restitutionGrad[0]! 2.0 1.0e-12)
    s!"d(v+)/d(e) should be -v-, got {reprStr restitutionGrad}"

@[test]
def testRepeatedSimulationMatchesClosedForm : IO Unit := do
  let result ← getTenSecondRun
  let expected := closedFormUnitRestitutionFromRest testParams testParams.initialHeight 10.0
  LeanTest.assertTrue (approx result.finalTime 10.0 1.0e-10)
    s!"Simulation should finish at t=10, got {result.finalTime}"
  LeanTest.assertTrue (approx result.finalState.height expected.height 2.0e-4)
    s!"Height should match Drake closed-form elastic-bounce solution, got {result.finalState.height}, expected {expected.height}"
  LeanTest.assertTrue (approx result.finalState.velocity expected.velocity 2.0e-4)
    s!"Velocity should match Drake closed-form elastic-bounce solution, got {result.finalState.velocity}, expected {expected.velocity}"
  LeanTest.assertEqual result.impacts.size 11
    "A 10-second q0=1 elastic run should contain eleven downward ground impacts"

@[test]
def testRepeatedSimulationRecordsEventSkeleton : IO Unit := do
  let result ← getTenSecondRun
  match result.trace.validate? with
  | .error msg => LeanTest.fail s!"Bouncing-ball trace should validate: {msg}"
  | .ok () => pure ()

  LeanTest.assertEqual result.trace.entries.size (2 * result.impacts.size + 1)
    "Trace should contain one localized interval and one saltation per impact, plus the terminal interval"
  LeanTest.assertEqual result.moves.size (4 * result.impacts.size + 2)
    "Each impact contributes interval/checkpoint and saltation/reset moves; terminal interval contributes interval/checkpoint"
  match result.trace.entries[0]! with
  | .interval segment =>
      LeanTest.assertTrue segment.localizedByEvent
        s!"First interval should be localized by the witness event, got {reprStr segment}"
      LeanTest.assertTrue segment.madeJumpAfter
        "First localized interval should end at an impact jump boundary"
  | other => LeanTest.fail s!"Expected first trace entry to be an interval, got {reprStr other}"
  match result.trace.entries[1]! with
  | .saltation vertex _ =>
      LeanTest.assertEqual vertex (impactVertex 0)
        "Second trace entry should be the first impact saltation vertex"
  | other => LeanTest.fail s!"Expected second trace entry to be a saltation event, got {reprStr other}"

@[test]
def testEndToEndResultCarriesLeafSystemPhysicsAndRegressionBoundary : IO Unit := do
  let result ← assertOk (buildEndToEnd? testParams 10.0 128)
    "BouncingBall end-to-end result"
  assertOk result.systemBoundary.validate?
    "BouncingBall end-to-end system boundary"
  assertOk result.simulation.trace.validate?
    "BouncingBall end-to-end trace validation"

  LeanTest.assertTrue (approx result.initialState.height testParams.initialHeight 1.0e-12)
    "End-to-end result should carry the Drake default release height"
  LeanTest.assertTrue (approx result.initialState.velocity 0.0 1.0e-12)
    "End-to-end result should carry release from rest"
  LeanTest.assertEqual result.simulation.impacts.size 11
    "End-to-end result should retain the repeated-impact Drake regression run"
  LeanTest.assertTrue (result.closedFormHeightError < 2.0e-4)
    s!"End-to-end height should match the closed-form regression, got error {result.closedFormHeightError}"
  LeanTest.assertTrue (result.closedFormVelocityError < 2.0e-4)
    s!"End-to-end velocity should match the closed-form regression, got error {result.closedFormVelocityError}"

  LeanTest.assertEqual result.freeFlightFullPhysics.move.targets #[fullPhysicsVertex]
    "End-to-end free-flight full-physics primitive should target the explicit physics vertex"
  LeanTest.assertEqual result.freeFlightFullPhysics.derivative.qdot #[0.0]
    "Initial free-flight qdot should be zero at release from rest"
  LeanTest.assertTrue
    (approx (result.freeFlightFullPhysics.derivative.vdot.getD 0 0.0)
      testParams.gravity 1.0e-12)
    "Initial free-flight acceleration should equal gravity"

  LeanTest.assertEqual (countMoveKind result.moves .localSchurBlock) 2
    "End-to-end result should add LeafSystem and closed-form-regression local boundaries"
  LeanTest.assertEqual (countMoveKind result.moves .markMarginalize) 1
    "End-to-end result should expose the full-physics support-selection move"
  LeanTest.assertTrue
    (hasMoveLabel result.moves "LeafSystem declaration")
    "End-to-end moves should include the Drake LeafSystem declaration boundary"
  LeanTest.assertTrue
    (hasMoveLabel result.moves "closed-form elastic repeated-impact regression")
    "End-to-end moves should include the Drake closed-form regression boundary"

  let msg ← assertError (buildEndToEnd? { testParams with restitution := 0.9 } 1.0 8)
    "non-unit restitution closed-form boundary"
  LeanTest.assertTrue (msg.contains "unit restitution")
    s!"Non-unit restitution should be rejected by the closed-form boundary, got {msg}"

end Tests.EventSkeletonBouncingBallExample
