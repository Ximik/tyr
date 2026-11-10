import LeanTest
import Tyr.EventSkeleton.Examples.AllegroHand

namespace Tests.EventSkeletonAllegroHandExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.AllegroHand

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) :
    IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertError {α : Type} (res : Except String α) (label : String) :
    IO String := do
  match res with
  | .ok _ => LeanTest.fail s!"{label}: expected error, got ok"
  | .error msg => pure msg

private def assertArrayNear
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

private def diagonalSolveExpected
    (diagonal rhs : Array Float) : Array Float := Id.run do
  let mut out : Array Float := #[]
  for i in [:rhs.size] do
    out := out.push (rhs[i]! / diagonal[i]!)
  return out

@[test]
def testDrakeReferencesConstantsAndJointOrderingAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/allegro_hand/allegro_common.cc"))
    "Example should reference Drake's Allegro common helpers"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/allegro_hand/allegro_lcm.cc"))
    "Example should reference Drake's Allegro LCM systems"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/allegro_hand/joint_control/run_twisting_mug.cc"))
    "Example should reference Drake's twisting-mug commander"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/allegro_hand/joint_control/test/run_twisting_mug_test.py"))
    "Example should reference Drake's twisting-mug Python regression"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/allegro_hand/test/parse_test.cc"))
    "Example should reference Drake's SDF/URDF parser quantity test"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/allegro_hand/run_allegro_constant_load_demo.cc"))
    "Example should reference Drake's constant-load demo"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path.contains "simple_mug.sdf"))
    "Example should record the mug SDF used by the joint-control demo"

  LeanTest.assertEqual numJoints 16
  LeanTest.assertEqual fingerCount 4
  LeanTest.assertEqual jointsPerFinger 4
  LeanTest.assertTrue (approx hardwareStatusPeriod 0.003 1.0e-12)
    s!"Hardware status period should be 3 ms, got {hardwareStatusPeriod}"
  LeanTest.assertEqual statusChannel "ALLEGRO_STATUS"
  LeanTest.assertEqual commandChannel "ALLEGRO_COMMAND"
  LeanTest.assertEqual preferredJointOrdering
    #["joint_12", "joint_13", "joint_14", "joint_15",
      "joint_0", "joint_1", "joint_2", "joint_3",
      "joint_4", "joint_5", "joint_6", "joint_7",
      "joint_8", "joint_9", "joint_10", "joint_11"]
  LeanTest.assertEqual fingerTipLinks #["link_3", "link_7", "link_11", "link_15"]
  LeanTest.assertTrue (rightHandModelUri.contains "allegro_hand_description_right.sdf")
    "Right hand model URI should match Drake's default model"
  LeanTest.assertTrue (leftHandModelUri.contains "allegro_hand_description_left.sdf")
    "Left hand model URI should be available"
  LeanTest.assertTrue (rightHandUrdfModelUri.contains "allegro_hand_description_right.urdf")
    "Right hand URDF model URI should be available for Drake's parse test"
  LeanTest.assertTrue (leftHandUrdfModelUri.contains "allegro_hand_description_left.urdf")
    "Left hand URDF model URI should be available for Drake's parse test"

@[test]
def testParseTestBoundaryUsesParserPrimitiveQuantities : IO Unit := do
  let boundary := allegroParseTestBoundary
  let _ ← assertOk boundary.validate? "Allegro SDF/URDF parse-test boundary"
  LeanTest.assertEqual boundary.testPath
    "../drake/examples/allegro_hand/test/parse_test.cc"
  LeanTest.assertTrue
    (boundary.formats == #[AllegroModelFormat.sdf, AllegroModelFormat.urdf])
    "Parse-test boundary should instantiate SDF then URDF"
  LeanTest.assertEqual boundary.quantities.size 2

  let sdf := boundary.quantities[0]!
  LeanTest.assertEqual sdf.modelUris #[rightHandModelUri, leftHandModelUri]
  LeanTest.assertEqual sdf.numModelInstances 4
  LeanTest.assertEqual sdf.numActuators (2 * numJoints)
  LeanTest.assertEqual sdf.numJoints (2 * numJoints + 2)
  LeanTest.assertEqual sdf.numBodies (2 * 17 + 1)
  LeanTest.assertEqual sdf.modelInstances.size 2
  LeanTest.assertEqual sdf.modelInstances[0]!.numPositions 23
  LeanTest.assertEqual sdf.modelInstances[0]!.numVelocities 22
  LeanTest.assertEqual sdf.modelInstances[1]!.numPositions 23
  LeanTest.assertEqual sdf.modelInstances[1]!.numVelocities 22
  LeanTest.assertTrue sdf.finalized
    "Parser quantities should be observed after plant.Finalize"

  let urdf := boundary.quantities[1]!
  LeanTest.assertEqual urdf.modelUris #[rightHandUrdfModelUri, leftHandUrdfModelUri]
  LeanTest.assertEqual urdf.numModelInstances 4
  LeanTest.assertEqual urdf.numActuators (2 * numJoints)
  LeanTest.assertEqual urdf.numJoints (2 * 21 + 2)
  LeanTest.assertEqual urdf.numBodies (2 * 22 + 1)
  LeanTest.assertEqual urdf.modelInstances[0]!.numPositions 23
  LeanTest.assertEqual urdf.modelInstances[0]!.numVelocities 22
  LeanTest.assertEqual urdf.modelInstances[1]!.numPositions 23
  LeanTest.assertEqual urdf.modelInstances[1]!.numVelocities 22
  LeanTest.assertTrue urdf.finalized
    "URDF parser quantities should also be observed after plant.Finalize"

  let graph := boundary.graph
  LeanTest.assertEqual graph.vertices.size 3
  LeanTest.assertEqual graph.moves.size 2
  LeanTest.assertTrue
    (graph.moves.all (fun move => move.kind == SkeletonMoveKind.localSchurBlock))
    "Parser.AddModelsFromUrl + plant.Finalize should be an exact local primitive"
  LeanTest.assertTrue
    (graph.moves.any (fun move => move.label.contains "Allegro SDF"))
    "Parse-test graph should expose the SDF parser primitive"
  LeanTest.assertTrue
    (graph.moves.any (fun move => move.label.contains "Allegro URDF"))
    "Parse-test graph should expose the URDF parser primitive"

@[test]
def testSingleObjectSimulationBoundaryKeepsFullHandAndMugPlant : IO Unit := do
  let step := allegroJointControlPlantStep jointControlParams 30.0
  let _ ← assertOk step.validate? "Allegro single-object full plant step"
  LeanTest.assertEqual step.model.numPositions 23
    "The full plant should include 16 hand positions plus a 7-position floating mug"
  LeanTest.assertEqual step.model.numVelocities 22
    "The full plant should include 16 hand velocities plus a 6-velocity floating mug"
  LeanTest.assertEqual step.model.numActuatedDofs numJoints
    "Only the hand joints are actuated"
  LeanTest.assertEqual step.q0.size step.model.numPositions
  LeanTest.assertEqual step.v0.size step.model.numVelocities
  LeanTest.assertEqual step.actuation.size numJoints
  LeanTest.assertTrue (approx step.config.timeStep jointControlParams.mbpDiscreteUpdatePeriod 1.0e-12)
    s!"Plant time step should match Drake's mbp_discrete_update_period, got {step.config.timeStep}"
  LeanTest.assertTrue (step.config.contactApproximation == DiscreteContactApproximation.sap)
    "The full-plant boundary should use SAP-style discrete contact"
  LeanTest.assertEqual step.model.floatingBases.size 1
  LeanTest.assertEqual step.model.floatingBases[0]!.bodyName "simple_mug"
  LeanTest.assertEqual step.model.floatingBases[0]!.floatingPositionsStart numJoints
  LeanTest.assertEqual step.model.floatingBases[0]!.floatingVelocitiesStartInV numJoints
  LeanTest.assertEqual allegroJointControlActuationMap.velocityDim step.model.numVelocities
    "Allegro actuation map should target all hand+mug velocities"
  LeanTest.assertEqual allegroJointControlActuationMap.actuatorVelocityIndices.size numJoints
    "Allegro actuation map should contain exactly the hand actuators"
  LeanTest.assertEqual allegroJointControlActuationMap.actuatorVelocityIndices[0]! 0
    "Allegro first actuator should map to the first hand joint velocity"
  LeanTest.assertEqual
    allegroJointControlActuationMap.actuatorVelocityIndices[numJoints - 1]!
    (numJoints - 1)
    "Allegro last actuator should map before the floating mug velocities"
  LeanTest.assertTrue (approx (initialMugPoseQ.getD 0 0.0) 0.7071067811865476 1.0e-12)
    s!"Mug quaternion should record the pi/2 roll initialization, got {initialMugPoseQ}"
  LeanTest.assertTrue (approx (initialMugPoseQ.getD 4 0.0) 0.095 1.0e-12)
    s!"Mug x translation should match Drake's initial offset, got {initialMugPoseQ}"

@[test]
def testConstantLoadDemoUsesFullPlantPrimitivePhysics : IO Unit := do
  let result ← assertOk solveAllegroConstantLoadPrimitivePhysics?
    "Allegro constant-load primitive physics"
  let step := result.plantStep
  let _ ← assertOk step.validate? "Allegro constant-load full plant step"
  LeanTest.assertEqual step.model.modelUri rightHandModelUri
  LeanTest.assertEqual step.model.numPositions numJoints
  LeanTest.assertEqual step.model.numVelocities numJoints
  LeanTest.assertEqual step.model.numActuatedDofs numJoints
  LeanTest.assertEqual step.model.floatingBases.size 0
  LeanTest.assertTrue step.model.finalized
    "The constant-load hand plant should be finalized after parser setup"
  LeanTest.assertTrue (approx step.config.timeStep constantLoadParams.maxTimeStep 1.0e-12)
    s!"Constant-load max time step mismatch: {step.config.timeStep}"
  LeanTest.assertTrue (approx step.t1 constantLoadParams.simulationTime 1.0e-12)
    s!"Constant-load simulation horizon mismatch: {step.t1}"
  LeanTest.assertEqual step.q0 constantLoadInitialQ
  LeanTest.assertEqual step.v0 constantLoadInitialV
  LeanTest.assertEqual step.actuation (constantLoadActuation constantLoadParams)
  LeanTest.assertTrue (approx step.q0[1]! 0.5 1.0e-12)
    "Drake initializes joint_1 to 0.5"
  LeanTest.assertTrue (approx step.q0[6]! (-0.1) 1.0e-12)
    "Drake initializes joint_6 to -0.1"
  LeanTest.assertTrue (approx step.q0[11]! 0.5 1.0e-12)
    "Drake initializes joint_11 to 0.5"
  LeanTest.assertEqual constantLoadActuationMap.velocityDim numJoints
  LeanTest.assertEqual constantLoadActuationMap.actuatorVelocityIndices
    (Array.range numJoints)
  assertOk result.primitivePlant.validate?
    "Allegro constant-load primitive plant wrapper"
  LeanTest.assertEqual result.primitivePlant.intervalVertex constantLoadIntervalVertex
  LeanTest.assertEqual result.primitivePlant.primitives.massMatrix
    (FloatMatrix.diagonal result.provider.massDiagonal)
  assertArrayNear result.primitivePlant.primitives.actuationForces step.actuation 1.0e-12
    "Constant-load identity actuation should become generalized forces"
  assertArrayNear result.primitivePlant.primitives.biasForces
    result.provider.gravityBiasForces 1.0e-12
    "Default constant-load demo should keep the gravity bias provider"
  LeanTest.assertEqual result.fullPhysics.support.totalCandidates 0
  LeanTest.assertEqual result.fullPhysics.contactForces.size 0
  assertArrayNear result.fullPhysics.equation.generalizedForces step.actuation 1.0e-12
    "Generalized forces should be the constant load when there are no contacts"
  let expectedVdot :=
    diagonalSolveExpected result.provider.massDiagonal result.fullPhysics.derivative.rhs
  assertArrayNear result.fullPhysics.derivative.vdot expectedVdot 1.0e-12
    "Constant-load acceleration should solve M vdot = tau - bias"
  LeanTest.assertEqual result.graph.vertices.size 5
  LeanTest.assertEqual result.graph.moves.size 4
  LeanTest.assertTrue (result.graph.containsMoveKind .localSchurBlock)
    "Parser/weld and constant-vector setup should remain exact local blocks"
  LeanTest.assertTrue (result.graph.containsMoveKind .intervalAdjoint)
    "The constant-load hand advance should use the full-physics interval primitive"
  LeanTest.assertTrue
    (result.graph.moves.any (fun m =>
      m.label == "full-physics-step:allegro constant-load primitive full physics"))
    "The graph should expose the solved constant-load full-physics step"

@[test]
def testConstantLoadCanDisableGravityAndApplyUniformTorque : IO Unit := do
  let params :=
    { constantLoadParams with constantLoad := 0.01, addGravity := false }
  let result ← assertOk (solveAllegroConstantLoadPrimitivePhysics? params)
    "Allegro nonzero constant-load primitive physics"
  assertArrayNear result.plantStep.actuation (Array.replicate numJoints 0.01) 1.0e-12
    "ConstantVectorSource should output one uniform torque per actuator"
  assertArrayNear result.primitivePlant.primitives.biasForces zeroJoints 1.0e-12
    "Disabling gravity should remove the provider gravity bias from the primitive equation"
  assertArrayNear result.fullPhysics.equation.generalizedForces
    (Array.replicate numJoints 0.01) 1.0e-12
    "No-contact generalized forces should equal the nonzero constant load"
  let expectedVdot :=
    diagonalSolveExpected result.provider.massDiagonal result.fullPhysics.derivative.rhs
  assertArrayNear result.fullPhysics.derivative.vdot expectedVdot 1.0e-12
    "Uniform constant load should be solved through the primitive mass matrix"
  for vdot in result.fullPhysics.derivative.vdot do
    LeanTest.assertTrue (vdot > 0.0)
      s!"Every joint acceleration should be positive under positive load and zero bias, got {result.fullPhysics.derivative.vdot}"

@[test]
def testReflectedInertiaAndPidGainsMatchDrakeFormula : IO Unit := do
  let reflected ← assertOk jointControlParams.reflectedInertia?
    "Allegro reflected inertia"
  LeanTest.assertTrue (approx reflected (1.0e-6 * 369.0 * 369.0) 1.0e-12)
    s!"Reflected inertia mismatch: {reflected}"
  LeanTest.assertTrue
    (approx jointControlParams.fingerBodyInertia
      ((0.17 / 3.0) / 3.0 * 0.05 * 0.05) 1.0e-12)
    s!"Finger rod inertia estimate mismatch: {jointControlParams.fingerBodyInertia}"
  let iEff ← assertOk jointControlParams.effectiveFingerInertia?
    "Allegro effective finger inertia"
  let gains ← assertOk jointControlParams.positionControlledGains?
    "Allegro position-controlled gains"
  let gainProp := jointControlParams.pidFrequency * jointControlParams.pidFrequency * iEff
  let gainDer :=
    2.0 * jointControlParams.pidFrequency * iEff * jointControlParams.dampingRatio
  LeanTest.assertTrue (approx gains.kp[0]! (1.6 * gainProp) 1.0e-12)
    s!"Thumb Kp should be scaled by 1.6, got {gains.kp[0]!}"
  for i in [1:numJoints] do
    LeanTest.assertTrue (approx gains.kp[i]! gainProp 1.0e-12)
      s!"Kp[{i}] mismatch: {gains.kp[i]!} vs {gainProp}"
  LeanTest.assertTrue (gains.kd.all (fun kd => approx kd gainDer 1.0e-12))
    s!"Kd should use 2 * frequency * Ieff * zeta, got {gains.kd}"
  LeanTest.assertTrue (gains.ki.all (fun ki => approx ki 0.0 1.0e-12))
    s!"Ki should be zero, got {gains.ki}"

@[test]
def testCommandReceiverLatchSemanticsMatchDrakeLcmTest : IO Unit := do
  let initial ← assertOk (initialReceiverState samplePosition)
    "Allegro initial receiver state"
  assertArrayNear initial.commandedPosition samplePosition 1.0e-12
    "Initial receiver position should be configurable"
  assertArrayNear initial.commandedVelocity zeroJoints 1.0e-12
    "Initial receiver commanded velocities should be zero"

  let defaultUpdate ← assertOk (receiveCommand? initial {})
    "Allegro default command update"
  assertArrayNear defaultUpdate.commandedPosition samplePosition 1.0e-12
    "Default-constructed command should preserve the previous position"
  assertArrayNear defaultUpdate.commandedVelocity zeroJoints 1.0e-12
    "Command receiver should always output zero desired velocities"
  assertArrayNear defaultUpdate.commandedTorque zeroJoints 1.0e-12
    "Missing torque command should produce zero feedforward torque"

  let delta :=
    #[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008,
      0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016]
  let commandedPosition := FloatArray.add samplePosition delta
  let update ← assertOk
    (receiveCommand? initial { jointPosition := commandedPosition, jointTorque := sampleTorque })
    "Allegro non-default command update"
  assertArrayNear update.commandedPosition commandedPosition 1.0e-12
    "Command receiver should latch the command position"
  assertArrayNear update.commandedVelocity zeroJoints 1.0e-12
    "Command receiver should keep desired velocities zero after a command"
  assertArrayNear update.commandedTorque sampleTorque 1.0e-12
    "Command receiver should latch torque commands when present"

@[test]
def testStatusSenderFieldsMatchDrakeLcmTest : IO Unit := do
  let commandState := (FloatArray.scale 0.5 samplePosition) ++ zeroJoints
  let measuredState := samplePosition ++ sampleVelocity
  let status ← assertOk
    (statusMessage? 0.25 commandState measuredState sampleTorque)
    "Allegro status sender"
  LeanTest.assertTrue (approx status.utime 250000.0 1.0e-9)
    s!"Status sender should convert seconds to microseconds, got {status.utime}"
  assertArrayNear status.jointPositionCommanded
    (FloatArray.scale 0.5 samplePosition) 1.0e-12
    "Status message should contain commanded position"
  assertArrayNear status.jointPositionMeasured samplePosition 1.0e-12
    "Status message should contain measured position"
  assertArrayNear status.jointVelocityEstimated sampleVelocity 1.0e-12
    "Status message should contain estimated velocity"
  assertArrayNear status.jointTorqueCommanded sampleTorque 1.0e-12
    "Status message should contain commanded torque"

@[test]
def testMotionStateClassifierMatchesDrakeStuckFingerLogic : IO Unit := do
  let measuredState := samplePosition ++ sampleVelocity
  let status ← assertOk
    (statusMessage? hardwareStatusPeriod (graspHandPosition ++ zeroJoints)
      measuredState sampleTorque)
    "Allegro status for motion classification"
  let motion ← assertOk (classifyMotion? status) "Allegro motion classifier"
  LeanTest.assertEqual motion.jointStuck.size numJoints
  LeanTest.assertEqual motion.fingerStuck.size fingerCount
  LeanTest.assertTrue (motion.isFingerStuck 1)
    "Index finger should be stuck because joints 5..7 are below velocity threshold"
  LeanTest.assertFalse (motion.isFingerStuck 0)
    "Thumb should not be stuck because all four thumb joints are moving"
  LeanTest.assertFalse (motion.isFingerStuck 2)
    "Middle finger should not be stuck in this sample"
  LeanTest.assertTrue (motion.isFingerStuck 3)
    "Ring finger should be stuck because reverse motor torque is detected"
  LeanTest.assertTrue motion.isAnyHighFingerStuck
    "A high finger should be classified as stuck"
  LeanTest.assertFalse motion.isAllFingersStuck
    "Not all fingers are stuck in this sample"

@[test]
def testGraspOpenAndTwistingTargetsMatchDrakeCommander : IO Unit := do
  LeanTest.assertEqual (fingerGraspJointPosition 0) #[1.396, 0.85, 0.0, 1.3]
  LeanTest.assertEqual (fingerGraspJointPosition 1) #[0.08, 0.9, 0.75, 1.5]
  LeanTest.assertEqual (fingerGraspJointPosition 2) #[0.1, 0.9, 0.75, 1.5]
  LeanTest.assertEqual (fingerGraspJointPosition 3) #[0.12, 0.9, 0.75, 1.5]
  LeanTest.assertEqual (fingerOpenJointPosition 0) #[0.263, 1.1, 0.0, 0.0]
  LeanTest.assertEqual openHandPosition
    #[0.263, 1.1, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0]
  LeanTest.assertEqual closeThumbTarget[0]! 1.396
  LeanTest.assertEqual closeThumbTarget[1]! 0.3
  LeanTest.assertEqual twistingMugPlan.closedGrasp graspHandPosition
  LeanTest.assertTrue
    (approx twistingMugPlan.indexTwist[5]!
      (graspHandPosition[5]! + 0.6) 1.0e-12)
    "Index twist should add 0.6 to joint 5"
  LeanTest.assertTrue
    (approx twistingMugPlan.indexTwist[6]!
      (graspHandPosition[6]! + 0.18) 1.0e-12)
    "Index twist should add 0.18 to joint 6"
  LeanTest.assertTrue
    (approx twistingMugPlan.indexTwist[9]!
      (graspHandPosition[9]! + 0.1) 1.0e-12)
    "Both twist targets should add middle-finger pivot preload"
  LeanTest.assertTrue
    (approx twistingMugPlan.ringTwist[13]!
      (graspHandPosition[13]! + 0.6) 1.0e-12)
    "Ring twist should add 0.6 to joint 13"
  LeanTest.assertTrue
    (approx twistingMugPlan.ringTwist[15]!
      (graspHandPosition[15]! + 0.3) 1.0e-12)
    "Ring twist should add 0.3 to joint 15"

@[test]
def testEndToEndGraphKeepsClockedLcmAndLocalPhysicsBlocks : IO Unit := do
  let result ← assertOk buildEndToEnd? "Allegro hand end-to-end"
  LeanTest.assertEqual result.references.size drakeReferences.size
  let _ ← assertOk result.parseTest.validate?
    "Allegro end-to-end parse-test primitive boundary"
  LeanTest.assertEqual result.gains.kp.size numJoints
  LeanTest.assertEqual result.plantStep.model.numPositions 23
  LeanTest.assertEqual result.plantStep.model.numVelocities 22
  LeanTest.assertTrue (approx result.plantStep.t1 30.0 1.0e-12)
    s!"Default end-to-end full plant boundary should use the coupled sim horizon, got {result.plantStep.t1}"
  assertArrayNear result.plantStep.actuation result.controllerOutput.feedback 1.0e-12
    "Allegro plant step actuation should be the PID feedback torque"
  assertArrayNear result.controllerOutput.desiredPosition graspHandPosition 1.0e-12
    "Allegro PID desired positions should come from the latched grasp command"
  let massDiagonal ← assertOk
    (allegroJointControlMassDiagonal? result.params)
    "Allegro primitive mass diagonal"
  LeanTest.assertEqual massDiagonal.size result.plantStep.model.numVelocities
  LeanTest.assertTrue (approx (massDiagonal.getD 16 0.0) 0.000156 1.0e-12)
    s!"Floating mug angular inertia should come from simple_mug.sdf, got {massDiagonal.getD 16 0.0}"
  LeanTest.assertTrue (approx (massDiagonal.getD 21 0.0) mugMass 1.0e-12)
    s!"Floating mug translational mass should come from simple_mug.sdf, got {massDiagonal.getD 21 0.0}"
  assertOk result.primitivePlant.validate? "Allegro primitive plant wrapper"
  LeanTest.assertEqual result.primitivePlant.intervalVertex fullPhysicsIntervalVertex
  LeanTest.assertEqual result.primitivePlant.primitives.velocityDim
    result.plantStep.model.numVelocities
    "Allegro primitive should expose one equation per plant velocity"
  LeanTest.assertEqual result.primitivePlant.primitives.contactCandidates.size 0
  LeanTest.assertEqual result.fullPhysics.support.totalCandidates 0
  LeanTest.assertEqual result.fullPhysics.contactForces.size 0
  LeanTest.assertEqual result.fullPhysics.equation.massMatrix
    (FloatMatrix.diagonal massDiagonal)
  assertArrayNear
    (result.fullPhysics.equation.generalizedForces.extract 0 numJoints)
    result.controllerOutput.feedback 1.0e-12
    "Generalized hand forces should start with the PID feedback torques"
  assertArrayNear
    (result.fullPhysics.equation.generalizedForces.extract numJoints
      result.plantStep.model.numVelocities)
    (Array.replicate mugFloatingVelocities 0.0) 1.0e-12
    "Floating mug velocities should receive no direct actuation"
  LeanTest.assertEqual result.commandedReceiver.commandedState.size (2 * numJoints)
  LeanTest.assertEqual result.status.jointPositionMeasured.size numJoints
  LeanTest.assertEqual result.graph.vertices.size 11
  LeanTest.assertEqual result.graph.moves.size 6
  LeanTest.assertTrue (result.graph.containsMoveKind .clockedUpdate)
    "LCM command and status boundaries should be clocked updates"
  LeanTest.assertTrue (result.graph.containsMoveKind .localSchurBlock)
    "PID and motion-state algebra should be exact local blocks"
  LeanTest.assertTrue (result.graph.containsMoveKind .intervalAdjoint)
    "Allegro hand+mug plant dynamics should be represented as a primitive full-physics interval"
  LeanTest.assertTrue
    (result.graph.moves.any (fun m =>
      m.label == "full-physics-step:allegro hand+mug primitive full physics"))
    "End-to-end graph should include the Allegro primitive full-physics solve"
  LeanTest.assertTrue (result.graph.moves.all (fun m => m.exactness == MoveExactness.exact))
    "The Allegro controller/status skeleton is exact for the represented fixture"

@[test]
def testJointControlPrimitiveProviderRecomputesLatchedCommandAndPlantState :
    IO Unit := do
  let provider :=
    allegroJointControlFullPhysicsPrimitiveProvider
      "allegro joint-control dynamic provider test"
  let qHand :=
    #[0.03, -0.04, 0.05, -0.06, 0.07, -0.08, 0.09, -0.10,
      0.11, -0.12, 0.13, -0.14, 0.15, -0.16, 0.17, -0.18]
  let vHand :=
    #[0.20, -0.18, 0.16, -0.14, 0.12, -0.10, 0.08, -0.06,
      0.04, -0.02, 0.01, -0.03, 0.05, -0.07, 0.09, -0.11]
  let step :=
    { allegroJointControlPlantStep jointControlParams 3.0 with
      q0 := qHand ++ initialMugPoseQ
      v0 := vHand ++ Array.replicate mugFloatingVelocities 0.0
      actuation := zeroJoints }
  let command :=
    { jointPosition := FloatArray.add graspHandPosition
        (Array.replicate numJoints 0.02)
      jointTorque := sampleTorque }
  let initial ← assertOk (initialReceiverState zeroJoints)
    "Allegro dynamic provider initial receiver"
  let commanded ← assertOk (receiveCommand? initial command)
    "Allegro dynamic provider command receiver"
  let snapshot :=
    allegroJointControlPhysicsState jointControlParams step commanded
  let controllerOutput ← assertOk snapshot.controllerOutput?
    "Allegro dynamic provider controller output"
  let primitives ← assertOk (provider.validPrimitivesAt? snapshot)
    "Allegro dynamic provider primitives"
  let actuatedStep := { step with actuation := controllerOutput.feedback }
  let directPrimitives ← assertOk
    (allegroJointControlFullPhysicsPrimitives?
      jointControlParams actuatedStep controllerOutput
      "allegro direct dynamic primitive")
    "Allegro direct dynamic primitives"

  LeanTest.assertEqual primitives.massMatrix directPrimitives.massMatrix
    "Provider should use the current Allegro hand+mug mass primitive"
  assertArrayNear primitives.qdot step.v0 1.0e-12
    "Provider should use the current plant velocity as qdot"
  assertArrayNear
    (primitives.actuationForces.extract 0 numJoints)
    controllerOutput.feedback 1.0e-12
    "Provider should recompute PID feedback from the current latched command"
  assertArrayNear
    (primitives.actuationForces.extract numJoints step.model.numVelocities)
    (Array.replicate mugFloatingVelocities 0.0) 1.0e-12
    "Provider should not directly actuate the floating mug"
  let staleOutput ← assertOk
    (allegroJointControlPhysicsState jointControlParams
      (allegroJointControlPlantStep jointControlParams 30.0)
      { commanded with commandedPosition := graspHandPosition }).controllerOutput?
    "Allegro stale fixture controller output"
  LeanTest.assertTrue
    (FloatArray.maxAbsDiff controllerOutput.feedback staleOutput.feedback > 1.0e-4)
    "Dynamic provider output should not reuse the default fixture PID feedback"

  let fullPhysics ← assertOk (provider.solveAt? snapshot 8523)
    "Allegro dynamic provider full physics solve"
  LeanTest.assertEqual fullPhysics.move.targets #[8523]
    "Provider solve should target the supplied interval vertex"
  assertArrayNear fullPhysics.derivative.rhs primitives.actuationForces 1.0e-12
    "Zero-bias Allegro primitive should solve against the recomputed actuation"
  let massDiagonal ← assertOk
    (allegroJointControlMassDiagonal? jointControlParams)
    "Allegro dynamic provider mass diagonal"
  let expectedVdot :=
    diagonalSolveExpected massDiagonal fullPhysics.derivative.rhs
  assertArrayNear fullPhysics.derivative.vdot expectedVdot 1.0e-12
    "Provider solve should use the current Allegro mass primitive"

  let badMsg ← assertError
    (provider.validPrimitivesAt?
      { snapshot with
        commanded :=
          { commanded with commandedPosition := #[1.0] } })
    "Allegro provider malformed command"
  LeanTest.assertTrue (badMsg.contains "commanded position")
    s!"Malformed commanded position should fail at provider validation, got {badMsg}"

@[test]
def testRunTwistingMugPythonRegressionBoundaryUsesFullPhysicsAndLcmPath : IO Unit := do
  let result ← assertOk buildRunTwistingMugPythonTest?
    "Allegro run_twisting_mug Python regression boundary"
  let boundary := result.boundary
  LeanTest.assertEqual boundary.testPath
    "../drake/examples/allegro_hand/joint_control/test/run_twisting_mug_test.py"
  LeanTest.assertEqual boundary.simResource
    "drake/examples/allegro_hand/joint_control/allegro_single_object_simulation"
  LeanTest.assertEqual boundary.controlResource
    "drake/examples/allegro_hand/joint_control/run_twisting_mug"
  LeanTest.assertEqual boundary.testTmpdirEnv "TEST_TMPDIR"
  LeanTest.assertEqual boundary.lcmUrlEnv "LCM_DEFAULT_URL"
  LeanTest.assertTrue (boundary.lcmUrlSource.contains "sha256")
    "LCM URL should be derived from sha256(TEST_TMPDIR)"
  LeanTest.assertTrue (boundary.lcmUrlSource.contains "ttl=0")
    "LCM URL should preserve Drake's local ttl=0 setting"
  LeanTest.assertEqual boundary.onlySimCommand
    #["allegro_single_object_simulation", "--simulation_time=0.01"]
  LeanTest.assertEqual boundary.coupledSimCommand
    #["allegro_single_object_simulation", "--simulation_time=30"]
  LeanTest.assertEqual boundary.coupledControlCommand
    #["run_twisting_mug", "--max_cycles=1"]
  LeanTest.assertTrue (approx boundary.onlySimSimulationTime 0.01 1.0e-12)
    s!"Only-sim smoke horizon should be 0.01, got {boundary.onlySimSimulationTime}"
  LeanTest.assertTrue (approx boundary.coupledSimulationTime 30.0 1.0e-12)
    s!"Coupled sim horizon should be 30, got {boundary.coupledSimulationTime}"
  LeanTest.assertEqual boundary.maxCyclesDefault 1000000000
  LeanTest.assertEqual boundary.smokeMaxCycles 1
  LeanTest.assertEqual boundary.initialStatusWaitCount 60
  LeanTest.assertEqual boundary.handleSubscriptionsTimeoutMs 10
  LeanTest.assertTrue (approx boundary.statusPollPeriod 0.1 1.0e-12)
    s!"Python polling period should be 0.1, got {boundary.statusPollPeriod}"
  LeanTest.assertTrue (approx boundary.killTimeout 10.0 1.0e-12)
    s!"Python kill timeout should be 10, got {boundary.killTimeout}"
  LeanTest.assertTrue boundary.skipOnDarwin
    "Drake skips the coupled test on macOS"
  LeanTest.assertTrue boundary.skipInDebugBuild
    "Drake skips the coupled test in debug builds"
  LeanTest.assertEqual boundary.expectedControlReturnCode (some 0)
  LeanTest.assertTrue boundary.expectedSimStillRunningWhenControlExits
    "The sim should remain live when the one-cycle controller exits"
  LeanTest.assertEqual boundary.plantStep.model.numPositions 23
  LeanTest.assertTrue (approx boundary.plantStep.t1 30.0 1.0e-12)
    s!"Boundary full-plant step should use coupled sim horizon, got {boundary.plantStep.t1}"
  LeanTest.assertTrue (result.graph.containsMoveKind .checkpointBoundary)
    "Python unittest platform/debug skip should be represented as a boundary checkpoint"
  LeanTest.assertTrue (result.graph.containsMoveKind .intervalAdjoint)
    "Full Drake subprocess advances should be interval boundaries"
  LeanTest.assertTrue (result.graph.containsMoveKind .clockedUpdate)
    "Composed result should retain the LCM command/status clocked updates"
  LeanTest.assertTrue (result.graph.containsMoveKind .localSchurBlock)
    "Composed result should retain resource lookup, plant setup, PID, and controller algebra blocks"
  LeanTest.assertTrue
    (result.moves.any (fun m => m.label.contains "full-physics process"))
    "The coupled simulator process should be labeled as the full-physics primitive"
  LeanTest.assertTrue
    (result.moves.any (fun m =>
      m.label == "full-physics-step:allegro hand+mug primitive full physics"))
    "The composed regression boundary should retain the Allegro primitive full-physics solve"
  LeanTest.assertTrue
    (result.moves.any (fun m => m.label.contains "run_twisting_mug --max_cycles=1"))
    "The one-cycle controller process should be part of the boundary"
  LeanTest.assertEqual result.controller.plan.indexTwist twistingMugPlan.indexTwist
  LeanTest.assertEqual result.controller.plan.ringTwist twistingMugPlan.ringTwist

end Tests.EventSkeletonAllegroHandExample
