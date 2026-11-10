import LeanTest
import Tyr.EventSkeleton.Examples.HardwareSim

namespace Tests.EventSkeletonHardwareSimExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.HardwareSim

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def maxAbsDiff (xs ys : Array Float) : Float := Id.run do
  let n := Nat.max xs.size ys.size
  let mut acc := 0.0
  for i in [:n] do
    let d := Float.abs (xs.getD i 0.0 - ys.getD i 0.0)
    if d > acc then
      acc := d
  return acc

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertSome {α : Type} (value : Option α) (label : String) : IO α := do
  match value with
  | some x => pure x
  | none => LeanTest.fail s!"{label}: expected some, got none"

private def assertError {α : Type} (res : Except String α) (label : String) :
    IO String := do
  match res with
  | .ok _ => LeanTest.fail s!"{label}: expected error, got ok"
  | .error msg => pure msg

private def countMoveKind (moves : Array SkeletonMove) (kind : SkeletonMoveKind) : Nat :=
  (moves.filter (fun move => move.kind == kind)).size

@[test]
def testDrakeReferencesAndScenarioDefaultsAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/hardware_sim/BUILD.bazel"))
    "HardwareSim example should reference Drake's Bazel targets"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/hardware_sim/README.md"))
    "HardwareSim example should reference Drake's README"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/hardware_sim/scenario.h"))
    "HardwareSim example should reference Drake's Scenario schema"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/hardware_sim/hardware_sim.cc"))
    "HardwareSim example should reference Drake's C++ setup sequence"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/hardware_sim/hardware_sim.py"))
    "HardwareSim example should reference Drake's Python setup sequence"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/hardware_sim/example_scenarios.yaml"))
    "HardwareSim example should reference Drake's Demo scenario"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/hardware_sim/robot_commander.py"))
    "HardwareSim example should reference Drake's robot commander"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/hardware_sim/test/hardware_sim_cc_test.py"))
    "HardwareSim example should reference Drake's C++ smoke test"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/hardware_sim/test/hardware_sim_py_test.py"))
    "HardwareSim example should reference Drake's Python smoke test"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/hardware_sim/test/robot_commander_test.py"))
    "HardwareSim example should reference Drake's robot commander test"

  assertOk defaultScenario.validate? "default scenario validation"
  LeanTest.assertEqual defaultScenario.randomSeed 0
    "Scenario default random seed should be deterministic zero"
  LeanTest.assertTrue (defaultScenario.simulationDuration > 0.0)
    "Scenario default simulation duration should be positive infinity"
  LeanTest.assertTrue (approx defaultScenario.simulatorConfig.maxStepSize 1.0e-3 1.0e-12)
    s!"Default simulator max step should be 1e-3, got {defaultScenario.simulatorConfig.maxStepSize}"
  LeanTest.assertTrue (approx defaultScenario.simulatorConfig.accuracy 1.0e-2 1.0e-12)
    s!"Default simulator accuracy should be 1e-2, got {defaultScenario.simulatorConfig.accuracy}"
  LeanTest.assertTrue (approx defaultScenario.simulatorConfig.targetRealtimeRate 1.0 1.0e-12)
    s!"Default target realtime rate should be 1, got {defaultScenario.simulatorConfig.targetRealtimeRate}"
  LeanTest.assertEqual defaultScenario.lcmBuses.size 1
    "Default scenario should include the default LCM bus"
  LeanTest.assertTrue (defaultScenario.hasBus "default")
    "Default LCM bus should be named default"

@[test]
def testOneOfEverythingMatchesDrakeSmokeScenario : IO Unit := do
  assertOk oneOfEverythingScenario.validate? "OneOfEverything scenario validation"
  LeanTest.assertEqual oneOfEverythingScenario.randomSeed 1
    "OneOfEverything should override random_seed"
  LeanTest.assertTrue (approx oneOfEverythingScenario.simulationDuration 3.14 1.0e-12)
    s!"OneOfEverything duration should be 3.14, got {oneOfEverythingScenario.simulationDuration}"
  LeanTest.assertTrue (approx oneOfEverythingScenario.simulatorConfig.targetRealtimeRate 5.0 1.0e-12)
    s!"OneOfEverything target realtime rate should be 5, got {oneOfEverythingScenario.simulatorConfig.targetRealtimeRate}"
  LeanTest.assertTrue (approx oneOfEverythingScenario.plantConfig.stictionTolerance 1.0e-2 1.0e-12)
    s!"OneOfEverything stiction tolerance should be 1e-2, got {oneOfEverythingScenario.plantConfig.stictionTolerance}"
  LeanTest.assertTrue
    (oneOfEverythingScenario.sceneGraphConfig.defaultProximityCompliance? == some HardwareComplianceType.compliant)
    "OneOfEverything should request compliant default proximity properties"
  LeanTest.assertEqual
    (oneOfEverythingScenario.directiveCountByKind HardwareDirectiveKind.addModel) 1
    "OneOfEverything should add one model"
  LeanTest.assertEqual oneOfEverythingScenario.lcmBuses.size 2
    "OneOfEverything should keep default bus and add extra_bus"
  LeanTest.assertTrue (oneOfEverythingScenario.hasBus "extra_bus")
    "OneOfEverything should declare extra_bus"
  LeanTest.assertEqual oneOfEverythingScenario.modelDrivers.size 1
    "OneOfEverything should configure one model driver"
  LeanTest.assertTrue (oneOfEverythingScenario.modelDrivers[0]!.config == HardwareDriverConfig.zeroForce)
    "OneOfEverything should use ZeroForceDriver"
  LeanTest.assertEqual oneOfEverythingScenario.cameras[0]!.lcmBus "extra_bus"
    "OneOfEverything camera should publish on extra_bus"
  LeanTest.assertEqual oneOfEverythingScenario.visualization.lcmBus "extra_bus"
    "OneOfEverything visualization should publish on extra_bus"
  LeanTest.assertTrue (approx oneOfEverythingScenario.visualization.publishPeriod 0.125 1.0e-12)
    s!"OneOfEverything visualization period should be 0.125, got {oneOfEverythingScenario.visualization.publishPeriod}"

@[test]
def testDemoScenarioPortsYamlModelDirectivesDriversAndInitialPositions : IO Unit := do
  assertOk demoScenario.validate? "Demo scenario validation"
  LeanTest.assertTrue
    (demoScenario.sceneGraphConfig.defaultProximityCompliance? == some HardwareComplianceType.compliant)
    "Demo should request compliant default proximity properties"
  LeanTest.assertEqual demoScenario.directives.size 9
    "Demo should record table, IIWA, WSG, pepper, frames, and weld directives"
  LeanTest.assertEqual
    (demoScenario.directiveCountByKind HardwareDirectiveKind.addModel) 4
    "Demo should add four models"
  LeanTest.assertEqual
    (demoScenario.directiveCountByKind HardwareDirectiveKind.addFrame) 2
    "Demo should add two named frames"
  LeanTest.assertEqual
    (demoScenario.directiveCountByKind HardwareDirectiveKind.addWeld) 3
    "Demo should add three welds"
  LeanTest.assertEqual demoIiwaJointPositions.size 7
    "Demo IIWA default positions should cover seven joints"
  LeanTest.assertTrue
    (demoDirectives.any (fun directive =>
      directive.name == "iiwa" &&
      directive.file == "package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf" &&
      directive.defaultJointPositions.size == 7))
    "Demo should load the primitive-collision IIWA with seven default joint positions"
  LeanTest.assertTrue (demoScenario.hasBus "driver_traffic")
    "Demo should declare the driver_traffic LCM bus"
  LeanTest.assertEqual driverTrafficBus.lcmUrl lcmUrl
    "Demo driver traffic bus should use the robot_commander LCM URL"
  LeanTest.assertEqual demoScenario.modelDrivers.size 2
    "Demo should configure IIWA and WSG drivers"
  LeanTest.assertTrue
    (demoScenario.modelDrivers.any (fun driver =>
      driver.modelName == "iiwa" &&
      driver.config == HardwareDriverConfig.iiwa "wsg" "driver_traffic"))
    "Demo IIWA driver should name the WSG hand and driver_traffic bus"
  LeanTest.assertTrue
    (demoScenario.modelDrivers.any (fun driver =>
      driver.modelName == "wsg" &&
      driver.config == HardwareDriverConfig.schunkWsg "driver_traffic"))
    "Demo WSG driver should use driver_traffic"
  LeanTest.assertEqual demoScenario.initialPosition.size 2
    "Demo should override both WSG finger sliding joints"
  LeanTest.assertTrue
    (demoScenario.initialPosition.any (fun q =>
      q.modelName == "wsg" && q.jointName == "left_finger_sliding_joint" &&
      q.positions == #[-0.02]))
    "Demo should set the left WSG finger initial position"

@[test]
def testSetupPlanMatchesDrakeHardwareSimOrdering : IO Unit := do
  let result ← assertOk buildDemoSmoke? "Demo smoke build"
  LeanTest.assertTrue (approx result.scenario.simulationDuration smokeDuration 1.0e-12)
    s!"Smoke scenario_text override should set duration to {smokeDuration}, got {result.scenario.simulationDuration}"
  let expected := #[
    HardwareSetupStepKind.addPlantAndSceneGraph,
    HardwareSetupStepKind.processDirectives,
    HardwareSetupStepKind.applyInitialPositions,
    HardwareSetupStepKind.finalizePlant,
    HardwareSetupStepKind.applyLcmBuses,
    HardwareSetupStepKind.applyDriverConfigs,
    HardwareSetupStepKind.applyCameraConfigs,
    HardwareSetupStepKind.applyVisualization,
    HardwareSetupStepKind.buildDiagram,
    HardwareSetupStepKind.applySimulatorConfig,
    HardwareSetupStepKind.setRandomContext,
    HardwareSetupStepKind.advanceTo
  ]
  LeanTest.assertTrue (result.plan.stepKinds == expected)
    s!"HardwareSim setup plan should follow Drake's setup order, got {reprStr result.plan.stepKinds}"
  LeanTest.assertEqual result.plan.steps[1]!.count demoScenario.directives.size
    "ProcessModelDirectives step should carry directive count"
  LeanTest.assertEqual result.plan.steps[5]!.count demoScenario.modelDrivers.size
    "ApplyDriverConfigs step should carry driver count"
  LeanTest.assertEqual result.plan.steps[6]!.count demoScenario.cameras.size
    "ApplyCameraConfig step should carry camera count"

@[test]
def testGraphvizSmokeAddsOptionalDiagramOutputBoundary : IO Unit := do
  let result ← assertOk buildGraphvizSmoke? "Graphviz smoke build"
  LeanTest.assertTrue result.plan.graphvizRequested
    "Graphviz smoke should mark graphviz as requested"
  LeanTest.assertTrue (result.plan.containsStep .writeGraphviz)
    "Graphviz smoke should include a writeGraphviz setup step"
  LeanTest.assertEqual result.graphvizOptions #[("plant/split", "I/O")]
    "Graphviz smoke should record Drake's split plant I/O option"
  LeanTest.assertTrue (result.plan.graph.containsMoveKind .checkpointBoundary)
    "Graphviz output should be represented as a checkpoint boundary"

@[test]
def testHardwareSimSkeletonGraphRecordsExternalHardwareBoundary : IO Unit := do
  let result ← assertOk buildOneOfEverythingSmoke? "OneOfEverything smoke build"
  LeanTest.assertTrue (result.plan.graph.containsMoveKind .localSchurBlock)
    "Plant/directive/driver/camera setup should be local provider blocks"
  LeanTest.assertTrue (result.plan.graph.containsMoveKind .freezeControl)
    "LCM buses should be represented as an external hardware boundary"
  LeanTest.assertTrue (result.plan.graph.containsMoveKind .intervalAdjoint)
    "Simulator.AdvanceTo should be represented as the simulation interval"
  LeanTest.assertTrue (result.plan.graph.containsMoveKind .checkpointBoundary)
    "Initial positions, visualization, and randomized context should be checkpoint boundaries"

@[test]
def testOneOfEverythingCanAdvanceThroughPendulumPhysicsPrimitive : IO Unit := do
  let result ← assertOk runOneOfEverythingPhysicsSmoke? "OneOfEverything primitive physics run"
  LeanTest.assertEqual result.modelName "alice"
    "The executable model should be the pendulum model from the Drake smoke scenario"
  LeanTest.assertTrue (result.kind == HardwareExecutableModelKind.pendulum)
    "Pendulum.urdf should lower to the pendulum dynamics primitive"
  LeanTest.assertTrue (approx result.t1 smokeDuration 1.0e-12)
    s!"Primitive physics run should advance to the smoke duration, got {result.t1}"
  LeanTest.assertEqual result.stateCoordinateNames #["theta", "thetadot"]
    "HardwareSim physics execution should expose the primitive's state coordinates"
  LeanTest.assertEqual result.initialState.size 2
    "Pendulum primitive state should be two-dimensional"
  LeanTest.assertEqual result.finalState.size 2
    "Pendulum primitive final state should be two-dimensional"
  LeanTest.assertTrue (approx (result.initialState.getD 0 0.0) 0.1 1.0e-12)
    s!"Named initial_position should seed theta, got {reprStr result.initialState}"
  LeanTest.assertTrue ((result.finalState.getD 1 0.0) < 0.0)
    s!"Positive theta with zero torque should accelerate downward, got final state {reprStr result.finalState}"
  LeanTest.assertTrue (result.moves.any (fun move => move.kind == SkeletonMoveKind.intervalAdjoint))
    "AdvanceTo should run through the existing DiffEq/EventSkeleton interval primitive"

@[test]
def testDemoPhysicsRunsIiwaPrimitiveAndReportsRemainingProviderModels : IO Unit := do
  let result ← assertOk (runScenarioPhysics? (smokeScenario demoScenario))
    "Demo IIWA primitive physics run"
  LeanTest.assertEqual result.modelName "iiwa"
    "Demo should execute the IIWA model from the scenario directives"
  LeanTest.assertTrue (result.kind == HardwareExecutableModelKind.iiwa)
    "IIWA directive should lower to the articulated IIWA primitive"
  LeanTest.assertEqual result.modelUri
    "package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
    "Demo should keep Drake's primitive-collision IIWA URI"
  LeanTest.assertEqual result.driverLabel "IiwaDriver(hand=wsg, bus=driver_traffic)"
    "Demo should lower the configured IiwaDriver boundary"
  LeanTest.assertTrue (approx result.t1 smokeDuration 1.0e-12)
    s!"Demo smoke primitive should advance to {smokeDuration}, got {result.t1}"
  LeanTest.assertEqual result.stateCoordinateNames
    Tyr.EventSkeleton.Examples.KukaIiwaArm.stateCoordinateNames
    "HardwareSim IIWA execution should expose Kuka state coordinates"
  LeanTest.assertEqual result.initialState.size 14
    "IIWA primitive state should include seven positions and seven velocities"
  LeanTest.assertEqual result.finalState.size 14
    "IIWA primitive final state should include seven positions and seven velocities"
  LeanTest.assertTrue (maxAbsDiff (result.initialState.extract 0 7) iiwaQ0 < 1.0e-12)
    s!"Demo IIWA initial positions should match scenario defaults, got {reprStr result.initialState}"
  LeanTest.assertTrue (maxAbsDiff result.finalState result.initialState < 1.0e-12)
    s!"First hardware command should gravity-hold the demo pose over the smoke horizon, got {reprStr result.finalState}"
  match result.fullPhysics? with
  | none => LeanTest.fail "Demo IIWA execution should expose the assembled full-physics primitive"
  | some fullPhysics =>
      LeanTest.assertEqual fullPhysics.contactForces.size 0
        "Demo IIWA primitive has no locally selected contact candidates yet"
      LeanTest.assertTrue (maxAbsDiff fullPhysics.generalizedForces fullPhysics.equation.biasForces < 1.0e-12)
        s!"Gravity-only IIWA driver should balance generalized forces against bias, got {reprStr fullPhysics.generalizedForces}"
      LeanTest.assertTrue (maxAbsDiff fullPhysics.derivative.vdot (Array.replicate 7 0.0) < 1.0e-12)
        s!"Gravity-held IIWA demo should have zero acceleration, got {reprStr fullPhysics.derivative.vdot}"
  match result.fullPlantStep? with
  | none => LeanTest.fail "Demo IIWA execution should expose a full MultibodyPlant step boundary"
  | some step =>
      assertOk step.validate? "Demo IIWA full plant step"
      LeanTest.assertEqual step.model.modelName "iiwa"
      LeanTest.assertTrue (approx step.t1 smokeDuration 1.0e-12)
        s!"Full plant boundary should use the smoke horizon, got {step.t1}"
  match result.primitivePlant? with
  | none => LeanTest.fail "Demo IIWA execution should expose the plant-bound primitive physics wrapper"
  | some primitivePlant =>
      assertOk primitivePlant.validate? "Demo IIWA primitive plant wrapper"
      LeanTest.assertEqual primitivePlant.step.model.modelName "iiwa"
      LeanTest.assertEqual primitivePlant.primitives.velocityDim 7
        "IIWA primitive should expose seven velocity equations"
      LeanTest.assertEqual primitivePlant.intervalVertex
        Tyr.EventSkeleton.Examples.KukaIiwaArm.fullPhysicsIntervalVertex
        "HardwareSim should reuse the IIWA full-physics interval vertex"
      LeanTest.assertTrue (maxAbsDiff primitivePlant.primitives.qdot primitivePlant.step.v0 < 1.0e-12)
        s!"Primitive qdot should be bound to the plant step velocity state, got {reprStr primitivePlant.primitives.qdot} vs {reprStr primitivePlant.step.v0}"
  LeanTest.assertTrue (result.moves.any (fun move => move.kind == SkeletonMoveKind.localSchurBlock))
    "Demo IIWA execution should retain the controller elimination move"
  LeanTest.assertTrue (result.moves.any (fun move => move.kind == SkeletonMoveKind.intervalAdjoint))
    "Demo IIWA execution should retain the full-physics interval move"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.label == "full-physics-step:iiwa torque-control full physics primitive"))
    "Demo IIWA execution should use the primitive full-physics interval move"
  LeanTest.assertFalse
    (result.moves.any (fun move => move.label.contains "Simulator.AdvanceTo primitive"))
    "Demo IIWA execution should not add a raw simulator-advance placeholder move"

  let unsupported := result.unsupportedModels
  LeanTest.assertFalse (unsupported.any (fun model => model.modelName == "iiwa"))
    s!"IIWA should no longer be reported unsupported, got {reprStr unsupported}"
  LeanTest.assertFalse (unsupported.any (fun model => model.modelName == "wsg"))
    s!"WSG should no longer be reported unsupported once the gripper primitive is registered, got {reprStr unsupported}"
  LeanTest.assertFalse (unsupported.any (fun model => model.modelName == "amazon_table"))
    s!"Registered table should lower through the scene-object primitive, got {reprStr unsupported}"
  LeanTest.assertFalse (unsupported.any (fun model => model.modelName == "bell_pepper"))
    s!"Registered pepper should lower through the scene-object primitive, got {reprStr unsupported}"

@[test]
def testDemoPhysicsAllRunsIiwaAndWsgPrimitives : IO Unit := do
  let result ← assertOk (runScenarioPhysicsAll? (smokeScenario demoScenario))
    "Demo all-primitive physics run"
  LeanTest.assertEqual result.executions.size 3
    "Demo should execute IIWA, WSG, and the registered pepper/table scene primitive"
  let iiwa ← assertSome
    (result.executions.find? (fun execution => execution.modelName == "iiwa"))
    "IIWA primitive execution"
  let wsg ← assertSome
    (result.executions.find? (fun execution => execution.modelName == "wsg"))
    "WSG primitive execution"
  let pepper ← assertSome
    (result.executions.find? (fun execution => execution.modelName == "bell_pepper"))
    "bell pepper scene primitive execution"
  LeanTest.assertTrue (iiwa.kind == HardwareExecutableModelKind.iiwa)
    "IIWA execution should keep the articulated manipulator kind"
  LeanTest.assertTrue (wsg.kind == HardwareExecutableModelKind.wsg)
    "WSG execution should lower to the gripper primitive kind"
  LeanTest.assertTrue (pepper.kind == HardwareExecutableModelKind.sceneFreeBody)
    "Bell pepper execution should lower to the scene free-body primitive kind"
  LeanTest.assertEqual wsg.modelUri
    "package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf"
    "WSG execution should keep Drake's Schunk WSG SDF URI"
  LeanTest.assertEqual wsg.driverLabel "SchunkWsgDriver(bus=driver_traffic)"
    "WSG execution should lower the configured SchunkWsgDriver boundary"
  LeanTest.assertEqual wsg.stateCoordinateNames
    #[
      "left_finger_sliding_joint",
      "right_finger_sliding_joint",
      "left_finger_sliding_joint_v",
      "right_finger_sliding_joint_v"
    ]
    "WSG execution should expose both finger slider coordinates and velocities"
  LeanTest.assertTrue (maxAbsDiff wsg.initialState #[-0.02, 0.02, 0.0, 0.0] < 1.0e-12)
    s!"WSG initial state should come from Demo initial_position overrides, got {reprStr wsg.initialState}"
  LeanTest.assertTrue (maxAbsDiff wsg.finalState wsg.initialState < 1.0e-12)
    s!"The first WSG driver target should hold the configured open pose, got {reprStr wsg.finalState}"
  match wsg.fullPhysics? with
  | none => LeanTest.fail "WSG execution should expose the assembled full-physics primitive"
  | some fullPhysics =>
      LeanTest.assertEqual fullPhysics.contactForces.size 0
        "HardwareSim WSG primitive should have no local scene contact candidates until a WSG-scene provider is connected"
      LeanTest.assertTrue (maxAbsDiff fullPhysics.derivative.vdot #[0.0, 0.0] < 1.0e-12)
        s!"WSG hold target should produce zero finger acceleration, got {reprStr fullPhysics.derivative.vdot}"
      LeanTest.assertTrue
        (fullPhysics.move.label == "full-physics-step:hardware_sim WSG gripper full physics primitive")
        s!"WSG execution should expose the gripper full-physics interval move, got {fullPhysics.move.label}"
  match wsg.fullPlantStep? with
  | none => LeanTest.fail "WSG execution should expose a full MultibodyPlant step boundary"
  | some step =>
      assertOk step.validate? "WSG full plant step"
      LeanTest.assertEqual step.model.numPositions 2
      LeanTest.assertEqual step.model.numVelocities 2
      LeanTest.assertEqual step.model.numActuatedDofs 2
      LeanTest.assertTrue (approx step.t1 smokeDuration 1.0e-12)
        s!"WSG full plant boundary should use the smoke horizon, got {step.t1}"
  match wsg.primitivePlant? with
  | none => LeanTest.fail "WSG execution should expose the plant-bound primitive physics wrapper"
  | some primitivePlant =>
      assertOk primitivePlant.validate? "WSG primitive plant wrapper"
      LeanTest.assertEqual primitivePlant.primitives.velocityDim 2
        "WSG primitive should expose two velocity equations"
      LeanTest.assertTrue (maxAbsDiff primitivePlant.primitives.qdot #[0.0, 0.0] < 1.0e-12)
        s!"WSG primitive qdot should bind to the initial finger velocities, got {reprStr primitivePlant.primitives.qdot}"

  LeanTest.assertEqual pepper.modelUri yellowBellPepperSdf
    "Pepper execution should keep Drake's bell pepper SDF URI"
  LeanTest.assertEqual pepper.driverLabel "SceneGraph free-body contact on amazon_table"
    "Pepper execution should record the table contact SceneGraph boundary"
  LeanTest.assertEqual pepper.stateCoordinateNames pepperTableStateCoordinateNames
    "Pepper execution should expose free-body translational coordinates and velocities"
  LeanTest.assertTrue (maxAbsDiff pepper.initialState #[0.0, 0.10, 0.20, 0.0, 0.0, 0.0] < 1.0e-12)
    s!"Pepper initial state should come from the default_free_body_pose translation, got {reprStr pepper.initialState}"
  LeanTest.assertTrue
    (approx (pepper.finalState.getD 2 0.0) 0.18083984375 1.0e-12 &&
      approx (pepper.finalState.getD 5 0.0) (-0.613125) 1.0e-12)
    s!"Pepper smoke final state should use free-fall kinematics above the table, got {reprStr pepper.finalState}"
  match pepper.fullPhysics? with
  | none => LeanTest.fail "Pepper execution should expose the assembled table-contact full-physics primitive"
  | some fullPhysics =>
      LeanTest.assertEqual fullPhysics.support.totalCandidates 1
        "Pepper SceneGraph query should expose one table contact candidate"
      LeanTest.assertEqual fullPhysics.contactForces.size 0
        "Pepper starts above the table, so the active support should be empty"
      LeanTest.assertEqual fullPhysics.generalizedPrimitiveForce
        #[0.0, 0.0, -pepperTableMass * pepperTableGravity]
        "Pepper free-body primitive should expose gravity as a primitive generalized force"
      LeanTest.assertTrue (maxAbsDiff fullPhysics.derivative.vdot #[0.0, 0.0, -pepperTableGravity] < 1.0e-12)
        s!"Pepper in-air acceleration should be gravity, got {reprStr fullPhysics.derivative.vdot}"
  match pepper.fullPlantStep? with
  | none => LeanTest.fail "Pepper execution should expose a full MultibodyPlant step boundary"
  | some step =>
      assertOk step.validate? "Pepper table full plant step"
      LeanTest.assertTrue step.hasContactEnvironment
        "Pepper plant step should record the table as a contact environment"
      LeanTest.assertEqual step.model.numPositions 3
      LeanTest.assertEqual step.model.numVelocities 3
      LeanTest.assertEqual step.model.numActuatedDofs 0
  match pepper.primitivePlant? with
  | none => LeanTest.fail "Pepper execution should expose the plant-bound primitive physics wrapper"
  | some primitivePlant =>
      assertOk primitivePlant.validate? "Pepper primitive plant wrapper"
      LeanTest.assertEqual primitivePlant.primitives.velocityDim 3
        "Pepper primitive should expose three translational velocity equations"

  let restingPrimitives ← assertOk
    (pepperTableFullPhysicsPrimitives?
      ({ bottomZ := 0.0 } : PepperTableState))
    "resting pepper table primitives"
  let restingFull ← assertOk (restingPrimitives.solve? 5604)
    "resting pepper table full physics"
  LeanTest.assertEqual restingFull.contactForces.size 1
    "Resting pepper should select the table contact"
  LeanTest.assertTrue
    (maxAbsDiff restingFull.generalizedContactForce
      #[0.0, 0.0, pepperTableMass * pepperTableGravity] < 1.0e-12)
    s!"Resting pepper contact should balance gravity through J^T f, got {reprStr restingFull.generalizedContactForce}"
  LeanTest.assertTrue (maxAbsDiff restingFull.derivative.vdot #[0.0, 0.0, 0.0] < 1.0e-12)
    s!"Resting pepper should have zero acceleration after contact support, got {reprStr restingFull.derivative.vdot}"

  LeanTest.assertFalse (result.unsupportedModels.any (fun model => model.modelName == "wsg"))
    s!"The multi-primitive run should not report WSG as unsupported, got {reprStr result.unsupportedModels}"
  LeanTest.assertFalse (result.unsupportedModels.any (fun model => model.modelName == "amazon_table"))
    s!"The table should lower through the registered scene primitive, got {reprStr result.unsupportedModels}"
  LeanTest.assertFalse (result.unsupportedModels.any (fun model => model.modelName == "bell_pepper"))
    s!"The pepper should lower through the registered scene primitive, got {reprStr result.unsupportedModels}"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.label == "full-physics-step:hardware_sim WSG gripper full physics primitive"))
    "All-primitive move list should include the WSG full-physics interval"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.label == "full-physics-step:hardware_sim bell pepper table free-body primitive"))
    "All-primitive move list should include the pepper/table full-physics interval"

@[test]
def testPrimitiveProvidersRecomputeWsgAndPepperState : IO Unit := do
  let wsgProvider :=
    wsgFullPhysicsPrimitiveProvider "hardware_sim WSG dynamic provider test"
  let held : WsgPrimitiveState := {}
  let moving : WsgPrimitiveState := {
    leftQ := -0.03
    rightQ := 0.01
    leftV := 0.5
    rightV := -0.25
  }
  let heldPrimitive ← assertOk (wsgProvider.validPrimitivesAt? held)
    "held WSG provider primitive"
  let movingPrimitive ← assertOk (wsgProvider.validPrimitivesAt? moving)
    "moving WSG provider primitive"
  LeanTest.assertTrue (maxAbsDiff heldPrimitive.actuationForces #[0.0, 0.0] < 1.0e-12)
    s!"Held WSG state should require no motor force, got {reprStr heldPrimitive.actuationForces}"
  LeanTest.assertTrue (maxAbsDiff movingPrimitive.qdot #[0.5, -0.25] < 1.0e-12)
    s!"WSG provider qdot should come from current finger velocities, got {reprStr movingPrimitive.qdot}"
  LeanTest.assertTrue
    (maxAbsDiff movingPrimitive.actuationForces #[99.5, 100.25] < 1.0e-12)
    s!"WSG provider actuation should recompute from current finger state, got {reprStr movingPrimitive.actuationForces}"
  let movingFull ← assertOk (wsgProvider.solveAt? moving 5613)
    "moving WSG provider full physics"
  LeanTest.assertEqual movingFull.support.totalCandidates 0
    "WSG provider should remain contact-free until a WSG-scene provider is connected"
  LeanTest.assertTrue
    (movingFull.move.label == "full-physics-step:hardware_sim WSG gripper full physics primitive")
    s!"WSG provider solve should expose the primitive interval move, got {movingFull.move.label}"

  let pepperProvider :=
    pepperTableFullPhysicsPrimitiveProvider
      "hardware_sim pepper dynamic provider test"
  let airborne : PepperTableState := { bottomZ := 0.20, vz := -0.5 }
  let resting : PepperTableState := { bottomZ := 0.0 }
  let airbornePrimitive ← assertOk (pepperProvider.validPrimitivesAt? airborne)
    "airborne pepper provider primitive"
  let restingPrimitive ← assertOk (pepperProvider.validPrimitivesAt? resting)
    "resting pepper provider primitive"
  LeanTest.assertTrue (maxAbsDiff airbornePrimitive.qdot #[0.0, 0.0, -0.5] < 1.0e-12)
    s!"Pepper provider qdot should come from current free-body velocity, got {reprStr airbornePrimitive.qdot}"
  LeanTest.assertEqual airbornePrimitive.contactForces.size 0
    "Airborne pepper should not precompute active table contact forces"
  LeanTest.assertEqual restingPrimitive.contactForces.size 1
    "Resting pepper should precompute the selected table support force"
  let restingSupport ← assertOk (pepperProvider.supportAt? resting)
    "resting pepper provider support"
  let restingSelected ← assertOk restingSupport.selectedCandidates?
    "resting pepper provider selected candidates"
  LeanTest.assertEqual restingSelected.size 1
    "Pepper provider should retain one dynamic bell-pepper/table contact"
  LeanTest.assertEqual restingSelected[0]!.bodyA "bell_pepper::contact_proxy_sphere"
    "Pepper provider should retain the bell-pepper contact proxy sphere"
  LeanTest.assertEqual restingSelected[0]!.bodyB "amazon_table::top_half_space"
    "Pepper provider should retain the amazon table half-space"
  LeanTest.assertTrue (approx restingSelected[0]!.signedDistance 0.0 1.0e-12)
    s!"Resting pepper contact should sit on the table surface, got {restingSelected[0]!.signedDistance}"
  let restingFull ← assertOk (pepperProvider.solveAt? resting 5614)
    "resting pepper provider full physics"
  LeanTest.assertTrue (maxAbsDiff restingFull.derivative.vdot #[0.0, 0.0, 0.0] < 1.0e-12)
    s!"Resting pepper provider should balance gravity through contact, got {reprStr restingFull.derivative.vdot}"

  let badWsg : WsgPrimitiveState := { leftQ := (0.0 / 0.0) }
  let wsgMsg ← assertError (wsgProvider.validPrimitivesAt? badWsg)
    "malformed WSG provider state"
  LeanTest.assertTrue (wsgMsg.contains "WSG state entry")
    s!"Malformed WSG state should fail at provider validation, got {wsgMsg}"
  let badPepper : PepperTableState := { bottomZ := (0.0 / 0.0) }
  let pepperMsg ← assertError (pepperProvider.validPrimitivesAt? badPepper)
    "malformed pepper provider state"
  LeanTest.assertTrue (pepperMsg.contains "pepper-table state entry")
    s!"Malformed pepper state should fail at provider validation, got {pepperMsg}"

@[test]
def testRobotCommanderMatchesDrakeCyclicCommandPublisher : IO Unit := do
  LeanTest.assertEqual lcmUrl "udpm://239.241.129.92:20185?ttl=0"
    "Robot commander should use the same non-default LCM URL as Demo"
  LeanTest.assertEqual iiwaCommandChannel "IIWA_COMMAND"
    "IIWA command channel should match robot_commander.py"
  LeanTest.assertEqual wsgCommandChannel "SCHUNK_WSG_COMMAND"
    "WSG command channel should match robot_commander.py"
  LeanTest.assertEqual iiwaQ0
    (demoIiwaJointPositions.map (fun q => q.positions.getD 0 0.0))
    "Robot commander IIWA_Q0 should stay in sync with Demo default joint positions"
  LeanTest.assertTrue (approx commandHz 20.0 1.0e-12)
    s!"COMMAND_HZ should be 20, got {commandHz}"
  LeanTest.assertTrue (approx cycleTime 10.0 1.0e-12)
    s!"CYCLE_TIME should be 10, got {cycleTime}"

  let first := firstUnitTestCommand
  LeanTest.assertTrue first.isFinite
    s!"First command should be finite, got {reprStr first}"
  LeanTest.assertTrue (approx first.sine 0.0 1.0e-12)
    s!"First command sine should be zero, got {first.sine}"
  LeanTest.assertTrue (maxAbsDiff first.iiwaJointPosition iiwaQ0 < 1.0e-12)
    s!"First IIWA command should equal IIWA_Q0, got {reprStr first.iiwaJointPosition}"
  LeanTest.assertTrue (approx first.wsgTargetPositionMm 20.0 1.0e-12)
    s!"First WSG command should be 20 mm, got {first.wsgTargetPositionMm}"

  let quarter := quarterCycleCommand
  LeanTest.assertTrue (approx quarter.sine 1.0 1.0e-12)
    s!"At 50 ticks, sine should be one for 10s*20Hz cycle, got {quarter.sine}"
  LeanTest.assertTrue
    (maxAbsDiff quarter.iiwaJointPosition (iiwaQ0.map (fun q => q + iiwaMaxDeflection)) < 1.0e-12)
    s!"Quarter-cycle IIWA command should add max deflection, got {reprStr quarter.iiwaJointPosition}"
  LeanTest.assertTrue (approx quarter.wsgTargetPositionMm 40.0 1.0e-12)
    s!"Quarter-cycle WSG command should be 40 mm, got {quarter.wsgTargetPositionMm}"

@[test]
def testEndToEndBuildUsesScenarioComputedPrimitiveExecutions : IO Unit := do
  let result ← assertOk buildEndToEnd?
    "HardwareSim end-to-end build"
  LeanTest.assertTrue (approx result.setup.scenario.simulationDuration smokeDuration 1.0e-12)
    "End-to-end build should use the Demo smoke horizon by default"
  LeanTest.assertTrue (result.setup.plan.containsStep HardwareSetupStepKind.advanceTo)
    "End-to-end setup should include the Simulator.AdvanceTo boundary"
  LeanTest.assertEqual result.executions.executions.size 3
    "End-to-end physics should discover IIWA, WSG, and the registered pepper/table scene primitive"
  LeanTest.assertTrue
    (result.executions.executions.any (fun execution =>
      execution.modelName == "iiwa" && execution.kind == HardwareExecutableModelKind.iiwa))
    "End-to-end discovery should lower the IIWA directive to the manipulator primitive"
  LeanTest.assertTrue
    (result.executions.executions.any (fun execution =>
      execution.modelName == "wsg" && execution.kind == HardwareExecutableModelKind.wsg))
    "End-to-end discovery should lower the WSG directive to the gripper primitive"
  LeanTest.assertTrue
    (result.executions.executions.any (fun execution =>
      execution.modelName == "bell_pepper" &&
        execution.kind == HardwareExecutableModelKind.sceneFreeBody))
    "End-to-end discovery should lower the registered table/pepper scene to a free-body contact primitive"
  LeanTest.assertEqual result.executions.unsupportedModels.size 0
    "End-to-end Demo should not report registered provider models as unsupported"
  LeanTest.assertEqual result.robotCommands.size 2
    "End-to-end result should retain the robot_commander command boundary samples"
  LeanTest.assertTrue (countMoveKind result.moves .freezeControl >= 1)
    "End-to-end schedule should include the external LCM hardware boundary"
  LeanTest.assertTrue (countMoveKind result.moves .intervalAdjoint >= 3)
    "End-to-end schedule should include primitive physics interval moves"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.label == "full-physics-step:hardware_sim WSG gripper full physics primitive"))
    "End-to-end schedule should include the dynamically computed WSG full-physics move"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.label == "full-physics-step:hardware_sim bell pepper table free-body primitive"))
    "End-to-end schedule should include the dynamically computed pepper/table full-physics move"

end Tests.EventSkeletonHardwareSimExample
