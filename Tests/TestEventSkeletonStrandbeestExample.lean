import LeanTest
import Tyr.EventSkeleton.Examples.Strandbeest

namespace Tests.EventSkeletonStrandbeestExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.Strandbeest

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

private def assertSome {α : Type} (value : Option α) (label : String) : IO α := do
  match value with
  | some x => pure x
  | none => LeanTest.fail s!"{label}: expected some, got none"

private def assertArrayNear
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

@[test]
def testReferencesDefaultsAndModeUrlsAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/strandbeest/BUILD.bazel"))
    "Strandbeest port should reference Drake's Bazel xacro_filegroup"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/strandbeest/run_with_motor.cc"))
    "Strandbeest port should reference Drake's run_with_motor.cc"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/strandbeest/test/run_with_motor_test.py"))
    "Strandbeest port should reference Drake's run_with_motor Python smoke test"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/strandbeest/model/Strandbeest.xacro"))
    "Strandbeest port should reference the root Xacro"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/strandbeest/model/StrandbeestConstraints.urdf.xacro"))
    "Strandbeest port should reference the generated constraints URDF xacro entry"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/strandbeest/model/StrandbeestBushings.urdf.xacro"))
    "Strandbeest port should reference the generated bushing URDF xacro entry"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/strandbeest/model/LegAssembly.xacro"))
    "Strandbeest port should reference per-leg loop closures"

  LeanTest.assertEqual params.simulationTime 20.0
    "Default simulation_time should match Drake's command-line flag"
  LeanTest.assertEqual params.initialVelocity 5.0
    "Default desired crank velocity should match Drake's flag"
  LeanTest.assertEqual params.mbtDt 5.0e-2
    "Default discrete plant time step should match Drake's flag"
  LeanTest.assertEqual params.penetrationAllowance 5.0e-3
    "Penetration allowance should match Drake's flag"
  LeanTest.assertEqual params.stictionTolerance 5.0e-2
    "Stiction tolerance should match Drake's flag"
  LeanTest.assertEqual params.selectedUrdfUrl
    "package://drake/examples/multibody/strandbeest/model/StrandbeestConstraints.urdf"
    "Default constraints mode should select the constraints URDF"
  LeanTest.assertEqual
    ({ params with mode := .bushings, mbtDt := 0.0 } : StrandbeestParams).selectedUrdfUrl
    "package://drake/examples/multibody/strandbeest/model/StrandbeestBushings.urdf"
    "Bushing mode should select the bushing URDF"

@[test]
def testXacroAssetCatalogRecordsGeneratedUrdfVariants : IO Unit := do
  assertOk validateStrandbeestExampleAssetCatalog?
    "Strandbeest example asset catalog"
  LeanTest.assertEqual strandbeestExampleAssets.size 10
    "Catalog should cover every file in Drake's Strandbeest example directory"
  LeanTest.assertEqual strandbeestXacroAssets.size 6
    "Catalog should include both generated entry xacros and the four shared xacro files"
  LeanTest.assertEqual strandbeestGeneratedUrdfEntryAssets.size 2
    "Catalog should identify the two generated URDF entry points"
  LeanTest.assertEqual strandbeestModelExpansionAssets.size 6
    "Only the xacro files should feed model expansion"

  let constraints ← assertSome
    (findStrandbeestExampleAsset? "model/StrandbeestConstraints.urdf.xacro")
    "constraints generated URDF entry"
  LeanTest.assertTrue
    (constraints.kind == StrandbeestExampleAssetKind.generatedUrdfEntry)
    "Constraints entry should be a generated URDF entry"
  LeanTest.assertTrue
    (constraints.loopClosureMode? == some LoopClosureMode.constraints)
    "Constraints entry should map to constraints loop-closure mode"
  LeanTest.assertEqual constraints.generatedUrdfUrl? (some LoopClosureMode.constraints.urdfUrl)
    "Constraints entry should record the generated URDF URL consumed by run_with_motor"
  LeanTest.assertTrue
    (constraints.localDependencies.contains "model/Strandbeest.xacro")
    "Constraints entry should depend on the root Strandbeest xacro"

  let bushings ← assertSome
    (findStrandbeestExampleAsset? "model/StrandbeestBushings.urdf.xacro")
    "bushing generated URDF entry"
  LeanTest.assertTrue
    (bushings.loopClosureMode? == some LoopClosureMode.bushings)
    "Bushing entry should map to bushing loop-closure mode"
  LeanTest.assertEqual bushings.generatedUrdfUrl? (some LoopClosureMode.bushings.urdfUrl)
    "Bushing entry should record the generated URDF URL consumed by run_with_motor"
  LeanTest.assertTrue
    (bushings.localDependencies.contains "model/Strandbeest.xacro")
    "Bushing entry should depend on the root Strandbeest xacro"

  let root ← assertSome
    (findStrandbeestExampleAsset? "model/Strandbeest.xacro")
    "root Strandbeest xacro asset"
  LeanTest.assertTrue
    (root.localDependencies.contains "model/Macros.xacro")
    "Root xacro should depend on Macros.xacro"
  LeanTest.assertTrue
    (root.localDependencies.contains "model/LegPair.xacro")
    "Root xacro should depend on LegPair.xacro"

  let pair ← assertSome
    (findStrandbeestExampleAsset? "model/LegPair.xacro")
    "leg-pair xacro asset"
  LeanTest.assertTrue
    (pair.localDependencies.contains "model/LegAssembly.xacro")
    "LegPair.xacro should depend on LegAssembly.xacro"

  let leg ← assertSome
    (findStrandbeestExampleAsset? "model/LegAssembly.xacro")
    "leg assembly xacro asset"
  LeanTest.assertTrue
    (leg.localDependencies.contains "model/Macros.xacro")
    "LegAssembly.xacro should depend on Macros.xacro"

  let runWithMotor ← assertSome
    (findStrandbeestExampleAsset? "run_with_motor.cc")
    "run_with_motor source asset"
  LeanTest.assertTrue
    (runWithMotor.localDependencies.contains "model/StrandbeestConstraints.urdf.xacro")
    "run_with_motor should record the constraints-model entry dependency"
  LeanTest.assertTrue
    (runWithMotor.localDependencies.contains "model/StrandbeestBushings.urdf.xacro")
    "run_with_motor should record the bushing-model entry dependency"

@[test]
def testRunWithMotorPythonSmokeBoundaryUsesFullPhysicsPath : IO Unit := do
  let smoke ← assertOk (buildRunWithMotorSmokeTest? runWithMotorSmokeTestBoundary 0.0)
    "Strandbeest run_with_motor_test.py boundary"
  let boundary := smoke.boundary
  LeanTest.assertEqual boundary.smokeTestPath
    "../drake/examples/multibody/strandbeest/test/run_with_motor_test.py"
    "Boundary should point at Drake's Python smoke test"
  LeanTest.assertEqual boundary.executablePath
    "examples/multibody/strandbeest/run_with_motor"
    "Boundary should preserve the subprocess executable path"
  LeanTest.assertEqual boundary.commandLine
    #["examples/multibody/strandbeest/run_with_motor", "--simulation_time=0.1"]
    "Boundary should preserve the exact subprocess.check_call command"
  LeanTest.assertTrue boundary.expectsNoCrash
    "Python smoke test asserts no crash by using subprocess.check_call"
  LeanTest.assertTrue (approx boundary.simulationTimeOverride 0.1 1.0e-12)
    s!"Python smoke test should override simulation_time to 0.1, got {boundary.simulationTimeOverride}"
  LeanTest.assertTrue (approx boundary.params.simulationTime 0.1 1.0e-12)
    s!"Full-physics boundary should run the 0.1s smoke simulation, got {boundary.params.simulationTime}"
  LeanTest.assertTrue (boundary.params.mode == LoopClosureMode.constraints)
    "Smoke test should exercise Drake's default constraints-mode plant"
  LeanTest.assertTrue (approx boundary.params.mbtDt 5.0e-2 1.0e-12)
    s!"Smoke test should keep Drake's default discrete plant step, got {boundary.params.mbtDt}"

  let flags := boundary.simulatorFlags
  LeanTest.assertTrue (approx flags.targetRealtimeRate 1.0 1.0e-12)
    s!"run_with_motor.cc sets realtime rate to 1, got {flags.targetRealtimeRate}"
  LeanTest.assertTrue (approx flags.accuracy 1.0e-2 1.0e-12)
    s!"run_with_motor.cc sets simulator accuracy to 1e-2, got {flags.accuracy}"
  LeanTest.assertTrue (approx flags.maxTimeStep 1.0e-1 1.0e-12)
    s!"run_with_motor.cc sets max time step to 1e-1, got {flags.maxTimeStep}"
  LeanTest.assertEqual flags.integrationScheme "implicit_euler"
    "run_with_motor.cc sets the simulator integration scheme to implicit_euler"
  LeanTest.assertTrue flags.printStatistics
    "run_with_motor.cc prints simulator statistics after AdvanceTo"

  LeanTest.assertEqual smoke.graph.vertices.size 5
    "Python smoke boundary should expose test path, executable, CLI flag, simulator flags, and no-crash interval"
  LeanTest.assertEqual smoke.graph.moves.size 2
    "Python smoke boundary should expose simulator gflags and subprocess interval moves"
  LeanTest.assertTrue (smoke.graph.containsMoveKind .intervalAdjoint)
    "subprocess.check_call should be represented as a smoke-run interval"
  LeanTest.assertTrue
    (smoke.moves.any (fun move =>
      move.label == "Python unittest subprocess.check_call run_with_motor smoke boundary"))
    "Move list should retain the Python smoke-test boundary"
  LeanTest.assertTrue
    (smoke.moves.any (fun move =>
      move.label == "full-physics-step:Strandbeest constraints primitive plant dynamics"))
    "Smoke boundary must use the primitive full-physics Strandbeest advance"
  LeanTest.assertTrue
    (smoke.moves.any (fun move =>
      move.label == "SAP ball-constraint full-physics loop closure boundary"))
    "Smoke boundary should retain the constraints-mode loop-closure primitive"
  LeanTest.assertTrue (approx smoke.result.step.t1 0.1 1.0e-12)
    s!"Full plant step should advance to the smoke-test horizon, got {smoke.result.step.t1}"
  LeanTest.assertEqual smoke.result.support.retainedCount 36
    "Full-support dynamic loop-closure set should still retain all Strandbeest constraints"
  assertOk smoke.result.primitivePlant.validate?
    "Smoke boundary primitive plant validation"
  assertOk smoke.result.fullPhysics.equation.validate?
    "Smoke boundary primitive full-physics equation"
  LeanTest.assertEqual smoke.result.fullPhysics.support.selectedLocalIndices.size 0
    "Smoke boundary should not encode ball constraints as contact support"
  LeanTest.assertEqual smoke.result.primitivePlant.primitives.bilateralConstraints.size 36
    "Smoke boundary should retain all 36 ball constraints as bilateral primitives"
  match smoke.result.fullPhysics.constraintSolve? with
  | some solve =>
      LeanTest.assertEqual solve.constraints.size 36
        "Smoke boundary should solve the retained ball-constraint primitive set"
      LeanTest.assertEqual solve.jacobian.size 108
        "Each Strandbeest ball constraint contributes three scalar closure rows"
  | none => LeanTest.fail "Smoke boundary should expose the bilateral constraint solve"

@[test]
def testModeValidationFollowsDrakePlantChoiceInvariant : IO Unit := do
  assertOk params.validate? "default constraints-mode validation"
  assertOk ({ params with mode := .bushings, mbtDt := 0.0 } : StrandbeestParams).validate?
    "continuous bushing-mode validation"
  match ({ params with mbtDt := 0.0 } : StrandbeestParams).validate? with
  | .ok _ => LeanTest.fail "Constraints mode should reject mbt_dt == 0"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "constraints mode")
        s!"Expected constraints-mode error, got {msg}"
  match ({ params with mode := .bushings } : StrandbeestParams).validate? with
  | .ok _ => LeanTest.fail "Bushing mode should reject nonzero mbt_dt"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "bushing mode")
        s!"Expected bushing-mode error, got {msg}"

@[test]
def testLegPairPhasingAndModelDimensionsComeFromXacro : IO Unit := do
  LeanTest.assertEqual legPairs.size 6
    "Strandbeest.xacro declares six positioned leg pairs"
  LeanTest.assertEqual legPairs[0]!.name "pair01"
    "First leg pair name should match Xacro"
  LeanTest.assertTrue (approx legPairs[0]!.offset 0.6 1.0e-12)
    s!"pair01 offset should be 0.6, got {legPairs[0]!.offset}"
  LeanTest.assertTrue (approx legPairs[1]!.phase (2.0 * 3.14159265358979323846 / 3.0) 1.0e-12)
    s!"pair02 phase should be 2*pi/3, got {legPairs[1]!.phase}"
  LeanTest.assertTrue (approx legPairs[5]!.offset (-0.6) 1.0e-12)
    s!"pair06 offset should be -0.6, got {legPairs[5]!.offset}"
  LeanTest.assertTrue (approx legPairs[5]!.phase (5.0 * 3.14159265358979323846 / 3.0) 1.0e-12)
    s!"pair06 phase should be 5*pi/3, got {legPairs[5]!.phase}"

  LeanTest.assertEqual perLegRevoluteJointCount 5
    "LegAssembly.xacro declares five revolute joints per leg"
  LeanTest.assertEqual jointsPerLegPair 12
    "Each pair has two five-joint legs plus two continuous crank joints"
  LeanTest.assertEqual numPositions 80
    "Free crossbar, crank joint, and six leg pairs should produce 80 positions"
  LeanTest.assertEqual numVelocities 79
    "Quaternion floating base has one fewer velocity than position"
  LeanTest.assertEqual numActuatedDofs 1
    "Only joint_crossbar_crank is actuated"

@[test]
def testLoopClosurePrimitiveSetsAreComplete : IO Unit := do
  LeanTest.assertEqual ballConstraints.size 36
    "Six pairs times six loop closures should produce 36 ball constraints"
  LeanTest.assertEqual bushingLoops.size 36
    "Six pairs times six loop closures should produce 36 bushing loops"

  let first := ballConstraints[0]!
  LeanTest.assertEqual first.bodyA "pair01_leg1_bar_f"
    "First per-leg constraint should connect bar_f"
  LeanTest.assertEqual first.bodyB "pair01_leg1_bar_g"
    "First per-leg constraint should connect bar_g"
  LeanTest.assertTrue (approx first.p_AP.z 0.394 1.0e-12)
    s!"First F-G p_AP should match Xacro, got {reprStr first.p_AP}"
  LeanTest.assertTrue (approx first.p_BQ.x 0.3624 1.0e-12)
    s!"First F-G p_BQ should match Xacro, got {reprStr first.p_BQ}"

  let pairClosure := ballConstraints[4]!
  LeanTest.assertEqual pairClosure.bodyA "pair01_bar_a"
    "Pair-level closure should connect pair bar_a"
  LeanTest.assertEqual pairClosure.bodyB "pair01_leg1_bar_c"
    "Pair-level closure should connect leg1 bar_c"
  LeanTest.assertTrue (approx pairClosure.p_AP.z 0.38 1.0e-12)
    s!"Pair-level A-C p_AP should match Xacro, got {reprStr pairClosure.p_AP}"

  let last := ballConstraints[35]!
  LeanTest.assertEqual last.bodyA "pair06_bar_a"
    "Last closure should belong to pair06"
  LeanTest.assertEqual last.bodyB "pair06_leg2_bar_c"
    "Last closure should be the pair06 leg2 A-C closure"
  LeanTest.assertTrue (approx last.p_AP.z (-0.38) 1.0e-12)
    s!"Last A-C p_AP should match Xacro, got {reprStr last.p_AP}"

  LeanTest.assertEqual bushingLoops[0]!.params.forceStiffness #[20000.0, 20000.0, 20000.0]
    "Bushing force stiffness should match Macros.xacro"
  LeanTest.assertEqual bushingLoops[0]!.params.torqueDamping #[2000.0, 2000.0, 0.0]
    "Bushing torque damping should match Macros.xacro"

@[test]
def testDesiredVelocityMotorAndInitialContext : IO Unit := do
  let motor := params.motor
  LeanTest.assertEqual motor.jointName "joint_crossbar_crank"
    "Motor should attach to Drake's crank joint"
  LeanTest.assertEqual motor.actuatorName "crossbar_crank_motor"
    "Motor should preserve Drake's transmission actuator name"
  LeanTest.assertEqual motor.velocityStartInV crankJointVelocityStartInV
    "Crank velocity should follow the floating-base velocity block"
  LeanTest.assertEqual motor.velocityIndex (numPositions + crankJointVelocityStartInV)
    "Controller observes plant state at q-size plus joint velocity start"
  LeanTest.assertTrue (motor.proportionalGain > 30.0 && motor.proportionalGain < 35.0)
    s!"Total-mass proportional gain should be estimated from Xacro PVC bodies, got {motor.proportionalGain}"
  LeanTest.assertTrue (approx (motor.torque 3.0) (motor.proportionalGain * 2.0) 1.0e-9)
    s!"Torque law should be gain*(desired-actual), got {motor.torque 3.0}"

  let q0 := initialPositions params
  let v0 := initialVelocities
  LeanTest.assertEqual q0.size numPositions
    "Initial q should match the full plant model"
  LeanTest.assertEqual v0.size numVelocities
    "Initial v should match the full plant model"
  LeanTest.assertEqual (q0.extract 0 4) #[1.0, 0.0, 0.0, 0.0]
    "Crossbar orientation should be fixed to unit quaternion"
  LeanTest.assertEqual (q0.extract 4 7) #[-2.0, 0.0, 1.35]
    "Crossbar translation should match Drake's IK constraint"
  LeanTest.assertTrue (approx (q0.getD crankJointPositionIndex 99.0) 0.0 1.0e-12)
    s!"Crank should start at top dead center, got {q0.getD crankJointPositionIndex 99.0}"

@[test]
def testFullPhysicsPrimitiveProviderRecomputesModeAndMotorInput :
    IO Unit := do
  let provider :=
    fullPhysicsPrimitiveProvider "Strandbeest dynamic provider test"

  let constraintsSnapshot := physicsState params 3.0
  let constraintsStep := constraintsSnapshot.step
  let constraintsPrimitives ← assertOk
    (provider.validPrimitivesAt? constraintsSnapshot)
    "Strandbeest constraints provider primitives"
  LeanTest.assertTrue
    (approx constraintsStep.actuation[0]! ((params.motor).torque 3.0) 1.0e-9)
    s!"Measured crank velocity should be recomputed into motor torque, got {constraintsStep.actuation}"
  LeanTest.assertTrue
    (approx
      (constraintsPrimitives.actuationForces.getD crankJointVelocityStartInV 0.0)
      constraintsStep.actuation[0]! 1.0e-9)
    "Provider should lift the current motor torque into the crank generalized force"
  LeanTest.assertEqual constraintsPrimitives.bilateralConstraints.size 36
    "Constraints mode provider should emit all ball closures as bilateral primitives"
  LeanTest.assertEqual constraintsPrimitives.generalizedForceContributions.size 0
    "Constraints mode provider should not emit bushing generalized-force contributions"
  let defaultPrimitives ← assertOk
    (provider.validPrimitivesAt? (physicsState params 0.0))
    "Strandbeest default motor provider primitives"
  LeanTest.assertTrue
    (Float.abs
      ((defaultPrimitives.actuationForces.getD crankJointVelocityStartInV 0.0) -
        (constraintsPrimitives.actuationForces.getD crankJointVelocityStartInV 0.0)) > 1.0e-6)
    "Provider output should change when measured motor velocity changes"
  let constraintsFull ← assertOk
    (provider.solveAt? constraintsSnapshot 8911)
    "Strandbeest constraints provider solve"
  LeanTest.assertEqual constraintsFull.move.targets #[8911]
    "Provider solve should target the supplied constraints interval vertex"
  match constraintsFull.constraintSolve? with
  | some solve =>
      LeanTest.assertEqual solve.constraints.size 36
        "Provider solve should retain every ball constraint"
      LeanTest.assertEqual solve.jacobian.size 108
        "Provider solve should assemble three scalar rows per ball constraint"
  | none => LeanTest.fail "Constraints provider solve should expose a bilateral constraint solve"

  let bushingParams := ({ params with mode := .bushings, mbtDt := 0.0 } : StrandbeestParams)
  let bushingSnapshot := physicsState bushingParams 2.0
  let (bushingStep, bushingPrimitives, bushingResults) ← assertOk
    (bushingSnapshot.primitivesWithLoopClosures?
      "Strandbeest bushing dynamic provider test")
    "Strandbeest bushing provider primitive bundle"
  LeanTest.assertTrue
    (approx bushingStep.actuation[0]! ((bushingParams.motor).torque 2.0) 1.0e-9)
    s!"Bushing provider should recompute motor torque, got {bushingStep.actuation}"
  LeanTest.assertEqual bushingPrimitives.bilateralConstraints.size 0
    "Bushing mode provider should not emit ideal ball constraints"
  LeanTest.assertEqual bushingPrimitives.generalizedForceContributions.size 1
    "Bushing mode provider should emit one summed loop-closure force contribution"
  LeanTest.assertEqual bushingResults.size 36
    "Bushing mode provider should evaluate every LinearBushingRollPitchYaw closure"
  let bushingFull ← assertOk
    (provider.solveAt? bushingSnapshot 8912)
    "Strandbeest bushing provider solve"
  LeanTest.assertEqual bushingFull.move.targets #[8912]
    "Provider solve should target the supplied bushing interval vertex"
  assertArrayNear bushingFull.generalizedPrimitiveForce
    (sumBushingGeneralizedForces bushingResults)
    1.0e-9
    "Bushing provider solve should expose the summed loop-closure force"

  let badMsg ← assertError
    (provider.validPrimitivesAt? (physicsState params (0.0 / 0.0)))
    "Strandbeest provider malformed measured velocity"
  LeanTest.assertTrue (badMsg.contains "measured motor velocity")
    s!"Malformed measured motor velocity should fail at provider validation, got {badMsg}"

@[test]
def testEndToEndBuildCarriesFullPhysicsBoundary : IO Unit := do
  let result ← assertOk (buildEndToEnd? params 0.0)
    "Strandbeest end-to-end build"
  assertOk result.trace.validate? "Strandbeest trace validation"
  assertOk result.step.validate? "Strandbeest full plant step"
  LeanTest.assertEqual result.assetCatalog.size 10
    "End-to-end result should carry the complete Strandbeest model asset catalog"
  LeanTest.assertEqual result.model.numPositions 80
    "Result model should expose full plant positions"
  LeanTest.assertEqual result.model.numVelocities 79
    "Result model should expose full plant velocities"
  LeanTest.assertEqual result.step.actuation.size 1
    "Full plant step should carry the motor actuation input"
  LeanTest.assertTrue (approx result.step.actuation[0]! (result.motor.torque 0.0) 1.0e-9)
    s!"Initial actuation should be desired-velocity motor torque, got {result.step.actuation}"
  assertOk result.primitivePlant.validate?
    "Strandbeest primitive plant validation"
  assertOk result.fullPhysics.equation.validate?
    "Strandbeest primitive full-physics equation"
  LeanTest.assertEqual result.primitivePlant.primitives.velocityDim numVelocities
    "Primitive full physics should expose the full plant velocity dimension"
  LeanTest.assertTrue
    (approx (result.primitivePlant.primitives.actuationForces.getD crankJointVelocityStartInV 0.0)
      result.step.actuation[0]! 1.0e-9)
    "Primitive full physics should lift the scalar motor torque into the crank generalized force"
  LeanTest.assertEqual result.fullPhysics.support.candidates.size 0
    "Constraints mode should not expose ideal ball constraints as contact candidates"
  LeanTest.assertEqual result.fullPhysics.contactForces.size 0
    "Constraints mode should not attach contact force bundles to bilateral closures"
  LeanTest.assertEqual result.primitivePlant.primitives.bilateralConstraints.size 36
    "Constraints mode should expose all ball closures as bilateral primitives"
  match result.fullPhysics.constraintSolve? with
  | some solve =>
      LeanTest.assertEqual solve.constraints.size 36
        "The full-physics solve should retain all ball constraints"
      LeanTest.assertEqual solve.jacobian.size 108
        "Each retained ball constraint should contribute three scalar rows"
      LeanTest.assertTrue
        (solve.multipliers.any (fun multiplier => !approx multiplier 0.0 1.0e-12))
        "Gravity/actuation should produce signed bilateral multipliers for the ideal closures"
      LeanTest.assertTrue
        (result.fullPhysics.generalizedConstraintForce.any
          (fun force => !approx force 0.0 1.0e-12))
        "Constraints mode should expose J^T lambda separately from contact forces"
      LeanTest.assertTrue
        (solve.constraintAccelerationAfter.all (fun a => approx a 0.0 1.0e-12))
        s!"Solved closure accelerations should stay at the target, got {solve.constraintAccelerationAfter}"
      assertArrayNear result.fullPhysics.derivative.vdot solve.acceleration 1.0e-12
        "Primitive full physics should return the constrained acceleration from the bilateral solve"
  | none => LeanTest.fail "Constraints mode should expose a bilateral constraint solve"
  LeanTest.assertTrue (result.config.contactApproximation == DiscreteContactApproximation.sap)
    "Constraints mode should keep Drake's SAP contact approximation"
  LeanTest.assertEqual result.support.retainedCount 36
    "Full-support loop closures should retain all 36 closures"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.label == "full-physics-step:Strandbeest constraints primitive plant dynamics"))
    "Move list should keep the primitive full plant solver boundary explicit"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.label == "SAP ball-constraint full-physics loop closure boundary"))
    "Constraints mode should include the SAP ball-constraint closure primitive"

@[test]
def testBushingModeBuildUsesContinuousPlantAndBushingBoundary : IO Unit := do
  let bushingParams := ({ params with mode := .bushings, mbtDt := 0.0 } : StrandbeestParams)
  let result ← assertOk (buildEndToEnd? bushingParams 1.0)
    "Strandbeest bushing-mode end-to-end build"
  assertOk result.step.validate? "Strandbeest bushing-mode full plant step"
  LeanTest.assertEqual result.assetCatalog.size 10
    "Bushing-mode result should carry the complete Strandbeest model asset catalog"
  LeanTest.assertEqual result.config.timeStep 0.0
    "Bushing mode should use Drake's continuous plant path"
  LeanTest.assertEqual result.step.model.modelUri
    "package://drake/examples/multibody/strandbeest/model/StrandbeestBushings.urdf"
    "Bushing mode should load the bushing URDF"
  LeanTest.assertEqual result.support.retainedCount 36
    "Bushing mode should retain all 36 bushing closures"
  assertOk result.primitivePlant.validate?
    "Strandbeest bushing primitive plant validation"
  assertOk result.fullPhysics.equation.validate?
    "Strandbeest bushing primitive full-physics equation"
  LeanTest.assertEqual result.bushingResults.size 36
    "Bushing mode should evaluate all LinearBushingRollPitchYaw primitive closures"
  LeanTest.assertEqual result.fullPhysics.support.candidates.size 0
    "Bushing mode should feed compliant loop forces directly instead of ball-constraint candidates"
  LeanTest.assertEqual result.primitivePlant.primitives.generalizedForceContributions.size 1
    "Bushing mode should feed loop closures as a primitive generalized-force contribution"
  assertArrayNear
    result.fullPhysics.generalizedPrimitiveForce
    (sumBushingGeneralizedForces result.bushingResults)
    1.0e-9
    "Bushing mode full physics should expose the summed loop-closure force separately from actuation"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.label == "LinearBushingRollPitchYaw full-physics loop closure boundary"))
    "Bushing mode should include the bushing closure primitive"
  LeanTest.assertTrue (approx result.step.actuation[0]! (result.motor.torque 1.0) 1.0e-9)
    s!"Measured crank velocity should affect the motor torque, got {result.step.actuation}"

end Tests.EventSkeletonStrandbeestExample
