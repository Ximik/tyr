import LeanTest
import Tyr.EventSkeleton.Examples.UrdfContact

namespace Tests.EventSkeletonUrdfContactExample

open LeanTest
open LeanUrdfTypeProvider
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.UrdfContact

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertArrayApprox
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

private def getResult : IO UrdfContactResult := do
  match buildEndToEnd? with
  | .ok result => pure result
  | .error msg => LeanTest.fail s!"URDF contact example failed to build: {msg}"

@[test]
def testUrdfProviderFeedsRobotMetadata : IO Unit := do
  LeanTest.assertEqual robot.name "vertical_contact_probe"
  LeanTest.assertEqual ContactProbeUrdf.linkNames #["world", "probe_link"]
  LeanTest.assertEqual ContactProbeUrdf.jointNames #["vertical_slide"]
  LeanTest.assertEqual ContactProbeUrdf.rootLinks #["world"]
  LeanTest.assertTrue (slideJoint.jointType == JointType.prismatic)
    "URDF joint should be prismatic"
  LeanTest.assertEqual slideJoint.axis
    ({ x := 0.0, y := 0.0, z := 1.0 } : Vector3)
  LeanTest.assertTrue (approx probeMass 1.25 1.0e-12)
    s!"Expected probe mass 1.25, got {probeMass}"
  LeanTest.assertTrue (approx contactRadius 0.05 1.0e-12)
    s!"Expected contact radius 0.05, got {contactRadius}"

@[test]
def testHybridOdeImpactStateIsDerivedFromUrdfContactGeometry : IO Unit := do
  LeanTest.assertTrue (approx initialState.position 0.4462 1.0e-12)
    s!"Expected initial position 0.4462, got {initialState.position}"
  LeanTest.assertTrue (approx preImpactState.position contactRadius 1.0e-12)
    s!"Expected impact position to equal contact radius, got {preImpactState.position}"
  LeanTest.assertTrue (approx preImpactState.velocity (-2.962) 1.0e-12)
    s!"Expected pre-impact velocity -2.962, got {preImpactState.velocity}"
  LeanTest.assertTrue (approx (contactGuard preImpactState) 0.0 1.0e-12)
    s!"Expected zero contact guard, got {contactGuard preImpactState}"
  LeanTest.assertTrue (approx postImpactState.velocity 1.1848 1.0e-12)
    s!"Expected post-impact velocity 1.1848, got {postImpactState.velocity}"

@[test]
def testUrdfContactCandidateFeedsFullPhysicsPrimitive : IO Unit := do
  let result ← getResult
  let candidates ← assertOk
    result.fullPhysics.support.selectedCandidates?
    "selected URDF full-physics contact candidates"
  LeanTest.assertEqual candidates.size 1
    "The threshold support should retain the URDF contact candidate"
  let candidate := candidates[0]!
  LeanTest.assertEqual candidate.bodyA "probe_link"
    "URDF candidate should record the moving collision body"
  LeanTest.assertEqual candidate.bodyB "world"
    "URDF candidate should record the ground body"
  assertArrayApprox candidate.point_W #[0.0, 0.0, contactRadius] 1.0e-12
    "URDF candidate should preserve the world contact point"
  assertArrayApprox candidate.normal_W #[0.0, 0.0, 1.0] 1.0e-12
    "URDF candidate should preserve the world contact normal"
  LeanTest.assertTrue (approx candidate.signedDistance 0.0 1.0e-12)
    s!"Expected contact guard at impact, got {candidate.signedDistance}"
  LeanTest.assertTrue (approx candidate.normalVelocity preImpactState.velocity 1.0e-12)
    s!"Expected candidate normal velocity from the prismatic joint, got {candidate.normalVelocity}"
  LeanTest.assertEqual candidate.normalJacobian #[1.0]
    "The one-DOF prismatic contact normal row should be dq/dz"

  LeanTest.assertEqual result.fullPhysics.equation.massMatrix fullPhysicsMassMatrix
    "Full physics should use the URDF inertial mass in M(q)"
  LeanTest.assertEqual result.fullPhysics.equation.biasForces fullPhysicsBiasForces
    "Gravity should enter the manipulator equation as a generalized bias force"
  let expectedNormal := fullPhysicsContactModel.normalDamping * (-preImpactState.velocity)
  LeanTest.assertTrue
    (approx result.fullPhysics.contactForces[0]!.normalForce expectedNormal 1.0e-12)
    s!"Expected damping contact force {expectedNormal}, got {result.fullPhysics.contactForces[0]!.normalForce}"
  assertArrayApprox result.fullPhysics.generalizedContactForce #[expectedNormal] 1.0e-12
    "Contact force should pass through J^T into generalized coordinates"
  let expectedAcceleration :=
    (expectedNormal - probeMass * params.gravity) / probeMass
  LeanTest.assertTrue
    (approx (result.fullPhysics.derivative.vdot.getD 0 0.0) expectedAcceleration 1.0e-12)
    s!"Expected M vdot = tau + J^T f - bias acceleration {expectedAcceleration}, got {result.fullPhysics.derivative.vdot}"
  LeanTest.assertTrue (result.fullPhysics.move.kind == SkeletonMoveKind.intervalAdjoint)
    "The full-physics step should expose an interval elimination move"
  LeanTest.assertTrue (result.fullPhysics.move.exactness == MoveExactness.exact)
    "The one-DOF URDF mass-matrix solve should be exact for the selected support"
  LeanTest.assertTrue (result.fullPhysics.supportMove.kind == SkeletonMoveKind.markMarginalize)
    "Threshold contact support should be represented separately from the dynamics solve"
  LeanTest.assertTrue (result.fullPhysics.supportMove.exactness == MoveExactness.controlledApproximation)
    "Threshold-selected contact support is a fixed-trace controlled approximation"

@[test]
def testUrdfFullPhysicsProviderRecomputesPrimitivesFromState : IO Unit := do
  let penetrating : ContactState := {
    position := contactRadius - 0.01
    velocity := -0.2
  }
  let penetratingSet ← assertOk
    (contactCandidateProvider.validCandidatesAt? penetrating (some 1))
    "penetrating URDF contact candidate set"
  LeanTest.assertEqual penetratingSet.candidates.size 1
    "URDF provider should expose one dynamic collision candidate"
  LeanTest.assertTrue
    (approx penetratingSet.candidates[0]!.signedDistance (-0.01) 1.0e-12)
    s!"Expected current-state penetration depth -0.01, got {penetratingSet.candidates[0]!.signedDistance}"
  LeanTest.assertTrue
    (approx penetratingSet.candidates[0]!.normalVelocity (-0.2) 1.0e-12)
    s!"Expected current-state normal velocity -0.2, got {penetratingSet.candidates[0]!.normalVelocity}"

  let penetratingSupport ← assertOk
    (fullPhysicsProvider.supportAt? penetrating)
    "penetrating URDF full-physics support"
  LeanTest.assertEqual penetratingSupport.selectedLocalIndices #[0]
    "Penetrating state should select the URDF contact candidate"
  let penetratingResult ← assertOk
    (fullPhysicsAt? penetrating 303)
    "penetrating URDF full-physics solve"
  LeanTest.assertEqual penetratingResult.move.targets #[303]
    "State-dependent full-physics solve should preserve the requested interval vertex"
  let expectedNormal :=
    fullPhysicsContactModel.normalStiffness * 0.01 +
      fullPhysicsContactModel.normalDamping * 0.2
  LeanTest.assertTrue
    (approx penetratingResult.contactForces[0]!.normalForce expectedNormal 1.0e-12)
    s!"Expected state-dependent normal force {expectedNormal}, got {penetratingResult.contactForces[0]!.normalForce}"
  let expectedAcceleration := (expectedNormal - probeMass * params.gravity) / probeMass
  LeanTest.assertTrue
    (approx (penetratingResult.derivative.vdot.getD 0 0.0) expectedAcceleration 1.0e-12)
    s!"Expected penetrating-state acceleration {expectedAcceleration}, got {penetratingResult.derivative.vdot}"
  let stepped ← assertOk
    (fullPhysicsEulerStep? penetrating 0.01 304)
    "penetrating URDF Euler step"
  LeanTest.assertTrue
    (approx stepped.position (penetrating.position + 0.01 * penetrating.velocity) 1.0e-12)
    s!"Euler position should use qdot from the full-physics solve, got {stepped.position}"
  LeanTest.assertTrue
    (approx stepped.velocity (penetrating.velocity + 0.01 * expectedAcceleration) 1.0e-12)
    s!"Euler velocity should use vdot from the full-physics solve, got {stepped.velocity}"

  let separated : ContactState := {
    position := contactRadius + 0.2
    velocity := 0.0
  }
  let separatedSupport ← assertOk
    (fullPhysicsProvider.supportAt? separated)
    "separated URDF full-physics support"
  LeanTest.assertEqual separatedSupport.selectedLocalIndices #[]
    "Separated state should recompute support to an empty active set"
  let separatedResult ← assertOk
    (fullPhysicsAt? separated 305)
    "separated URDF full-physics solve"
  LeanTest.assertEqual separatedResult.contactForces.size 0
    "Separated state should not synthesize contact forces"
  LeanTest.assertEqual separatedResult.generalizedContactForce #[0.0]
    "Separated state should have zero generalized contact force"
  LeanTest.assertTrue
    (approx (separatedResult.derivative.vdot.getD 0 0.0) (-params.gravity) 1.0e-12)
    s!"Separated full physics should fall under gravity, got {separatedResult.derivative.vdot}"

@[test]
def testUrdfContactTraceAndReverseMessages : IO Unit := do
  let result ← getResult
  LeanTest.assertTrue acceptedContactSegment.localizedByEvent
    "The contact segment should be localized before the attempted endpoint"
  LeanTest.assertTrue acceptedContactSegment.crossedJumpFlag
    "The contact segment should record the impact jump"

  LeanTest.assertEqual result.moves.size 7
    "Witness, contact-mode, support-selection, and full-physics entries should project to seven moves"
  LeanTest.assertTrue (result.moves[0]!.kind == SkeletonMoveKind.intervalAdjoint)
    "First move should eliminate the continuous interval"
  LeanTest.assertTrue (result.moves[1]!.kind == SkeletonMoveKind.checkpointBoundary)
    "Second move should store the interval boundary"
  LeanTest.assertTrue (result.moves[2]!.kind == SkeletonMoveKind.saltationTime)
    "Third move should eliminate witness timing"
  LeanTest.assertTrue (result.moves[3]!.kind == SkeletonMoveKind.resetTranspose)
    "Fourth move should apply the impact reset transpose"
  LeanTest.assertTrue (result.moves[4]!.kind == SkeletonMoveKind.markMarginalize)
    "Fifth move should eliminate the runtime-selected contact modes"
  LeanTest.assertTrue (result.moves[4]!.exactness == MoveExactness.controlledApproximation)
    "Top-k contact-mode support should be tagged as fixed-trace approximation"
  LeanTest.assertTrue (result.moves[5]!.kind == SkeletonMoveKind.markMarginalize)
    "Sixth move should record threshold support selection for the full-physics contact solve"
  LeanTest.assertTrue (result.moves[5]!.exactness == MoveExactness.controlledApproximation)
    "Threshold full-physics contact support should be tagged as a fixed-trace approximation"
  LeanTest.assertTrue (result.moves[6]!.kind == SkeletonMoveKind.intervalAdjoint)
    "Seventh move should execute the exact mass-matrix full-physics solve"
  LeanTest.assertTrue (result.moves[6]!.exactness == MoveExactness.exact)
    "The mass-matrix full-physics solve should remain exact for the selected support"

  let supports := result.trace.supports
  LeanTest.assertEqual supports.size 1
    "The trace should expose the dynamic contact-mode support"
  LeanTest.assertTrue (supports[0]!.selectedIds == #[0, 2])
    s!"Expected contact-mode IDs #[0, 2], got {supports[0]!.selectedIds}"

  let expectedAlpha := (-(params.gravity * (1.0 + params.restitution))) / preImpactState.velocity
  LeanTest.assertTrue (approx result.saltationAlpha expectedAlpha 1.0e-12)
    s!"Expected saltation alpha {expectedAlpha}, got {result.saltationAlpha}"
  assertArrayApprox result.preImpactAdjoint #[expectedAlpha, -params.restitution] 1.0e-12
    "Saltation should propagate the velocity cotangent to the pre-impact state"
  assertArrayApprox result.restitutionGrad #[-preImpactState.velocity] 1.0e-12
    "Reset-theta transpose should produce the restitution gradient"

  LeanTest.assertTrue (approx result.markMessage.value 2.6 1.0e-12)
    s!"Expected retained contact-mode value 2.6, got {result.markMessage.value}"
  assertArrayApprox result.markMessage.stateAdjoint #[-0.07, 0.04] 1.0e-12
    "Contact-mode elimination should include weighted messages plus probability sensitivity"

end Tests.EventSkeletonUrdfContactExample
