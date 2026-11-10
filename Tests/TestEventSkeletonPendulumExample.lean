import LeanTest
import Tyr.EventSkeleton.Examples.Pendulum

namespace Tests.EventSkeletonPendulumExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.Pendulum

private def pi : Float := 3.14159265358979323846

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertSome {α : Type} (value : Option α) (label : String) : IO α := do
  match value with
  | some x => pure x
  | none => LeanTest.fail s!"{label}: expected some"

private def assertError {α : Type} (res : Except String α) (label : String) : IO Unit := do
  match res with
  | .ok _ => LeanTest.fail s!"{label}: expected error"
  | .error _ => pure ()

private def assertArrayNear
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

@[test]
def testDrakeReferencesAndNamedVectorsAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/pendulum/pendulum_plant.cc"))
    "Pendulum example should reference Drake's plant implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/pendulum/pendulum_plant.h"))
    "Pendulum example should reference Drake's plant declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/pendulum/pendulum_input.h"))
    "Pendulum example should reference Drake's input vector declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/pendulum/pendulum_input.cc"))
    "Pendulum example should reference Drake's input coordinate names"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/pendulum/pendulum_state.cc"))
    "Pendulum example should reference Drake's state coordinate names"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/pendulum/pendulum_params.cc"))
    "Pendulum example should reference Drake's parameter coordinate names"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/pendulum/pendulum_parameters_derivatives.cc"))
    "Pendulum example should reference Drake's AutoDiff parameter-derivative executable"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/pendulum/test/pendulum_plant_test.cc"))
    "Pendulum example should reference Drake's plant regression test"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/pendulum/test/urdf_dynamics_test.cc"))
    "Pendulum example should reference Drake's URDF dynamics parity test"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/pendulum/pendulum_geometry.h"))
    "Pendulum example should reference Drake's PendulumGeometry declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/pendulum/pendulum_geometry.cc"))
    "Pendulum example should reference Drake's PendulumGeometry implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/pendulum/test/pendulum_geometry_test.cc"))
    "Pendulum example should reference Drake's PendulumGeometry acceptance test"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/pendulum/Pendulum.urdf"))
    "Pendulum example should reference Drake's URDF model"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/pendulum/lqr_simulation.cc"))
    "Pendulum example should reference Drake's LQR simulation executable"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/pendulum/energy_shaping_simulation.cc"))
    "Pendulum example should reference Drake's energy-shaping simulation executable"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/pendulum/print_symbolic_dynamics.cc"))
    "Pendulum example should reference Drake's symbolic dynamics printer"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/pendulum/trajectory_optimization_simulation.cc"))
    "Pendulum example should reference Drake's direct-collocation executable"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/multibody/pendulum/passive_simulation.cc"))
    "Pendulum example should reference Drake's MultibodyPlant passive benchmark executable"
  LeanTest.assertEqual stateCoordinateNames #["theta", "thetadot"]
    "Pendulum state coordinate order should match Drake's BasicVector"
  LeanTest.assertEqual inputCoordinateNames #["tau"]
    "Pendulum input coordinate order should match Drake's BasicVector"
  LeanTest.assertEqual parameterCoordinateNames #["mass", "length", "damping", "gravity"]
    "Pendulum parameter coordinate order should match Drake's BasicVector"
  LeanTest.assertTrue params.isValid "Default PendulumParams should match Drake's valid domain"
  LeanTest.assertTrue defaultInput.isValid "Default disconnected input should be valid zero torque"

@[test]
def testNamedVectorBoundariesMatchDrakeBasicVectors : IO Unit := do
  assertOk pendulumStateVectorBoundary.validate? "PendulumState named vector boundary"
  assertOk pendulumInputVectorBoundary.validate? "PendulumInput named vector boundary"
  assertOk pendulumParamsVectorBoundary.validate? "PendulumParams named vector boundary"

  LeanTest.assertEqual pendulumStateVectorBoundary.dimension 2
    "PendulumState should have two coordinates"
  LeanTest.assertEqual pendulumInputVectorBoundary.dimension 1
    "PendulumInput should have one coordinate"
  LeanTest.assertEqual pendulumParamsVectorBoundary.dimension 4
    "PendulumParams should have four coordinates"
  LeanTest.assertEqual (pendulumStateVectorBoundary.indexOf? "theta") (some 0)
    "theta should be coordinate 0"
  LeanTest.assertEqual (pendulumStateVectorBoundary.indexOf? "thetadot") (some 1)
    "thetadot should be coordinate 1"
  LeanTest.assertEqual (pendulumInputVectorBoundary.indexOf? "tau") (some 0)
    "tau should be coordinate 0"
  LeanTest.assertEqual (pendulumParamsVectorBoundary.indexOf? "mass") (some 0)
    "mass should be parameter coordinate 0"
  LeanTest.assertEqual (pendulumParamsVectorBoundary.indexOf? "length") (some 1)
    "length should be parameter coordinate 1"
  LeanTest.assertEqual (pendulumParamsVectorBoundary.indexOf? "damping") (some 2)
    "damping should be parameter coordinate 2"
  LeanTest.assertEqual (pendulumParamsVectorBoundary.indexOf? "gravity") (some 3)
    "gravity should be parameter coordinate 3"

  LeanTest.assertEqual defaultState.asArray #[0.0, 0.0]
    "PendulumState default constructor should zero theta and thetadot"
  LeanTest.assertEqual defaultInput.asArray #[0.0]
    "PendulumInput default constructor should zero tau"
  LeanTest.assertEqual params.asArray #[1.0, 0.5, 0.1, 9.81]
    "PendulumParams default constructor should match Drake defaults"
  LeanTest.assertTrue (pendulumParamsVectorBoundary.lowerBounds ==
      #[some 0.0, some 0.0, some 0.0, some 0.0])
    s!"PendulumParams should expose Drake's nonnegative element lower bounds, got {reprStr pendulumParamsVectorBoundary.lowerBounds}"
  LeanTest.assertTrue (pendulumParamsVectorBoundary.upperBounds == #[none, none, none, none])
    s!"PendulumParams should expose unbounded element upper bounds, got {reprStr pendulumParamsVectorBoundary.upperBounds}"

  let x ← assertOk (PendulumState.fromArray? #[0.25, -0.75]) "PendulumState array parse"
  let u ← assertOk (PendulumInput.fromArray? #[1.25]) "PendulumInput array parse"
  let p ← assertOk (PendulumParams.fromArray? #[2.0, 0.75, 0.2, 3.7]) "PendulumParams array parse"
  LeanTest.assertEqual x.asArray #[0.25, -0.75]
    "PendulumState should round-trip through arrays"
  LeanTest.assertEqual u.asArray #[1.25]
    "PendulumInput should round-trip through arrays"
  LeanTest.assertEqual p.asArray #[2.0, 0.75, 0.2, 3.7]
    "PendulumParams should round-trip through arrays"
  assertError (PendulumParams.fromArray? #[-1.0, 0.5, 0.1, 9.81])
    "PendulumParams should reject negative mass"

@[test]
def testPendulumGeometryProviderMatchesDrakeSceneGraphRegistration : IO Unit := do
  let result ← assertOk (buildPendulumGeometry? params defaultState)
    "PendulumGeometry provider"
  assertOk result.provider.validate? "PendulumGeometry SceneGraph provider"
  assertOk (result.poses.validate? result.provider) "PendulumGeometry pose output"
  LeanTest.assertEqual result.inputPortName "state"
    "PendulumGeometry should declare a vector input port named state"
  LeanTest.assertEqual result.inputPortSize 2
    "PendulumGeometry state input should use the PendulumState vector size"
  LeanTest.assertEqual result.outputPortName "geometry_pose"
    "PendulumGeometry should declare an abstract output port named geometry_pose"
  LeanTest.assertTrue result.hasDirectFeedthrough
    "PendulumGeometry should preserve Drake's direct-feedthrough acceptance check"
  LeanTest.assertEqual result.provider.sources.size 1
    "PendulumGeometry should register one SceneGraph source"
  LeanTest.assertEqual result.provider.frames.size 1
    "PendulumGeometry should register the arm frame"
  let frame ← assertSome (result.provider.frameById? pendulumArmFrameId)
    "pendulum arm frame lookup"
  LeanTest.assertEqual frame.name "arm"
    "PendulumGeometry frame should use Drake's arm name"
  LeanTest.assertEqual (result.provider.anchoredGeometries.map (fun g => g.id))
    #[pendulumBaseGeometryId]
    "PendulumGeometry should anchor only the base box"
  LeanTest.assertEqual result.provider.geometries.size 3
    "PendulumGeometry should register base, arm, and endpoint mass geometry"

  let base ← assertSome (result.provider.geometryById? pendulumBaseGeometryId)
    "pendulum base geometry lookup"
  LeanTest.assertEqual base.name "base"
    "Base geometry should preserve Drake's name"
  LeanTest.assertTrue (base.hasRole .illustration && base.hasRole .perception)
    "Base should carry illustration and perception roles"
  LeanTest.assertTrue
    (base.properties.diffuseRgba? ==
      some { r := 0.3, g := 0.6, b := 0.4, a := 1.0 })
    s!"Base should carry Drake's green diffuse color, got {reprStr base.properties.diffuseRgba?}"
  LeanTest.assertTrue (approx base.X_FG.translation.z 0.025 1.0e-12)
    s!"Base z-offset should match Drake's 0.025m pose, got {base.X_FG.translation.z}"
  match base.shape with
  | .box sx sy sz =>
      LeanTest.assertTrue
        (approx sx 0.05 1.0e-12 && approx sy 0.05 1.0e-12 && approx sz 0.05 1.0e-12)
        s!"Base box dimensions should be 0.05m cube, got {sx}, {sy}, {sz}"
  | other => LeanTest.fail s!"Pendulum base should be a box, got {reprStr other}"

  let arm ← assertSome (result.provider.geometryById? pendulumArmGeometryId)
    "pendulum arm geometry lookup"
  LeanTest.assertEqual arm.name "arm"
    "Arm geometry should preserve Drake's name"
  LeanTest.assertEqual arm.frameId? (some pendulumArmFrameId)
    "Arm geometry should attach to the arm frame"
  LeanTest.assertTrue
    (arm.properties.diffuseRgba? ==
      some { r := 0.9, g := 0.1, b := 0.0, a := 1.0 })
    s!"Arm should carry Drake's red diffuse color, got {reprStr arm.properties.diffuseRgba?}"
  LeanTest.assertTrue (approx arm.X_FG.translation.z (-params.length / 2.0) 1.0e-12)
    s!"Arm cylinder should be centered at -length/2, got {arm.X_FG.translation.z}"
  match arm.shape with
  | .cylinder radius length =>
      LeanTest.assertTrue
        (approx radius 0.01 1.0e-12 && approx length params.length 1.0e-12)
        s!"Arm cylinder should have radius 0.01 and default length, got {radius}, {length}"
  | other => LeanTest.fail s!"Pendulum arm should be a cylinder, got {reprStr other}"

  let bob ← assertSome (result.provider.geometryById? pendulumPointMassGeometryId)
    "pendulum point-mass geometry lookup"
  LeanTest.assertEqual bob.name "arm point mass"
    "Endpoint mass geometry should preserve Drake's name"
  LeanTest.assertEqual bob.frameId? (some pendulumArmFrameId)
    "Endpoint mass should attach to the arm frame"
  LeanTest.assertTrue
    (bob.properties.diffuseRgba? ==
      some { r := 0.0, g := 0.0, b := 1.0, a := 1.0 })
    s!"Endpoint mass should carry Drake's blue diffuse color, got {reprStr bob.properties.diffuseRgba?}"
  LeanTest.assertTrue (approx bob.X_FG.translation.z (-params.length) 1.0e-12)
    s!"Endpoint mass should sit at -length, got {bob.X_FG.translation.z}"
  match bob.shape with
  | .sphere radius =>
      LeanTest.assertTrue (approx radius (params.mass / 40.0) 1.0e-12)
        s!"Endpoint mass sphere radius should be mass/40, got {radius}"
  | other => LeanTest.fail s!"Pendulum endpoint mass should be a sphere, got {reprStr other}"

@[test]
def testPendulumGeometryPoseOutputMatchesDrakeYRotation : IO Unit := do
  let x : PendulumState := { theta := 0.2, thetadot := -0.3 }
  let result ← assertOk (buildPendulumGeometry? params x)
    "PendulumGeometry pose output"
  let pose ← assertSome (result.poses.poseForFrame? pendulumArmFrameId)
    "pendulum arm pose"
  LeanTest.assertTrue (pose.rotationAxis == SceneVec3.unitY)
    s!"Pendulum arm pose should be a Y-axis rotation, got {reprStr pose.rotationAxis}"
  LeanTest.assertTrue (approx pose.rotationAngle x.theta 1.0e-12)
    s!"Pendulum arm pose angle should equal theta, got {pose.rotationAngle}"
  LeanTest.assertTrue (approx pose.translation.x 0.0 1.0e-12 &&
      approx pose.translation.y 0.0 1.0e-12 &&
      approx pose.translation.z 0.0 1.0e-12)
    s!"Pendulum arm pose should not translate the frame, got {reprStr pose.translation}"

@[test]
def testPendulumGeometryGraphRecordsExactSceneGraphBoundary : IO Unit := do
  let result ← assertOk (buildPendulumGeometry? params defaultState)
    "PendulumGeometry graph"
  LeanTest.assertEqual result.moves.size 2
    "PendulumGeometry should expose registration and pose-output local moves"
  LeanTest.assertTrue (result.moves.all (fun move =>
      move.kind == SkeletonMoveKind.localSchurBlock &&
      move.exactness == MoveExactness.exact))
    "PendulumGeometry moves should be exact local SceneGraph blocks"
  LeanTest.assertTrue (result.moves.any (fun move =>
      move.targets == #[pendulumGeometryPoseOutputVertex] &&
      move.reads == #[pendulumGeometryStateInputVertex, pendulumGeometryProviderVertex] &&
      move.writes == #[pendulumGeometryPoseOutputVertex] &&
      move.label.contains "OutputGeometryPose"))
    "PendulumGeometry graph should record the theta-to-FramePoseVector move"

@[test]
def testManipulatorEquationMatchesDrakePlantFormula : IO Unit := do
  let x : PendulumState := { theta := 0.5, thetadot := 2.0 }
  let u : PendulumInput := { tau := 3.0 }
  let eq := manipulatorEquation params u x
  let dyn ← assertOk eq.solve? "pendulum manipulator equation"
  let expectedMass := params.mass * params.length * params.length
  let expectedBias :=
    params.mass * params.gravity * params.length * Float.sin x.theta +
      params.damping * x.thetadot
  let expectedAccel := (u.tau - expectedBias) / expectedMass

  LeanTest.assertTrue (approx ((eq.massMatrix[0]!).getD 0 0.0) expectedMass 1.0e-12)
    s!"Mass matrix should be m*l^2, got {reprStr eq.massMatrix}"
  LeanTest.assertTrue (approx (dyn.rhs.getD 0 99.0) (u.tau - expectedBias) 1.0e-12)
    s!"Manipulator RHS should be tau - gravity - damping, got {reprStr dyn.rhs}"
  LeanTest.assertTrue (approx (dyn.qdot.getD 0 99.0) x.thetadot 1.0e-12)
    s!"qdot should equal thetadot, got {reprStr dyn.qdot}"
  LeanTest.assertTrue (approx (dyn.vdot.getD 0 99.0) expectedAccel 1.0e-12)
    s!"thetadot derivative should match Drake formula, got {reprStr dyn.vdot}, expected {expectedAccel}"

@[test]
def testFullPhysicsPrimitiveProviderRecomputesStateAndInput : IO Unit := do
  let provider := fullPhysicsPrimitiveProvider params
    "pendulum full physics provider test"
  let x0 := defaultState
  let u0 := defaultInput
  let x1 : PendulumState := { theta := 0.5, thetadot := 2.0 }
  let u1 : PendulumInput := { tau := 3.0 }

  let primitive0 ← assertOk
    (provider.validPrimitivesAt? (physicsState x0 u0))
    "pendulum provider primitive at default state and input"
  let primitive1 ← assertOk
    (provider.validPrimitivesAt? (physicsState x1 u1))
    "pendulum provider primitive at moved state and input"
  let result1 ← assertOk
    (provider.solveAt? (physicsState x1 u1) 5155)
    "pendulum provider solve at moved state and input"
  let direct1 ← assertOk
    (solveFullPhysics? params u1 x1 5156 "pendulum direct provider parity")
    "pendulum direct solve for provider parity"

  assertArrayNear primitive1.massMatrix[0]! (massMatrix params)[0]! 1.0e-12
    "Pendulum provider mass matrix should come from params"
  assertArrayNear primitive1.qdot #[x1.thetadot] 1.0e-12
    "Pendulum provider qdot should come from the current state velocity"
  assertArrayNear primitive1.actuationForces #[u1.tau] 1.0e-12
    "Pendulum provider actuation should come from the current input"
  LeanTest.assertTrue
    (FloatArray.maxAbsDiff primitive0.biasForces primitive1.biasForces > 1.0)
    "Changing theta and thetadot should recompute gravity plus damping bias"
  assertArrayNear result1.derivative.vdot direct1.derivative.vdot 1.0e-12
    "Provider solve should match the direct full-physics solve"
  LeanTest.assertEqual result1.move.targets #[5155]
    "Provider solve should use the supplied interval vertex"

  let badInput : PendulumInput := { tau := 1.0 / 0.0 }
  assertError
    (provider.validPrimitivesAt? (physicsState x1 badInput))
    "pendulum provider malformed input"

@[test]
def testPendulumPlantScalarConversionAndMassDerivativeBoundary : IO Unit := do
  assertOk pendulumAutoDiffContextBoundary.validate? "PendulumPlant ToAutoDiff boundary"
  LeanTest.assertTrue (!pendulumPlantHasDirectFeedthrough)
    "PendulumPlant should preserve Drake's no-direct-feedthrough property"
  LeanTest.assertEqual pendulumAutoDiffContextBoundary.state.asArray #[42.0, 76.0]
    "ToAutoDiff boundary should copy Drake's test state values"
  LeanTest.assertEqual pendulumAutoDiffContextBoundary.derivativeSizes #[0, 0]
    "ToAutoDiff boundary should not initialize derivative vectors"

  let record ← assertOk forwardDynamicsMassDerivative?
    "Pendulum forward dynamics mass derivative"
  assertOk record.validate? "Pendulum mass derivative record"
  LeanTest.assertTrue (approx record.derivative.theta (-1.0) 1.0e-12)
    s!"theta_dot should equal the stored velocity, got {record.derivative.theta}"
  LeanTest.assertTrue (approx record.dthetaDotDm 0.0 1.0e-12)
    s!"theta_dot should be independent of mass, got {record.dthetaDotDm}"
  LeanTest.assertTrue (approx record.domegaDotDm (-0.4) 1.0e-12)
    s!"mass partial should match -(tau - b*omega)/(m^2*l^2), got {record.domegaDotDm}"

@[test]
def testEnergyMatchesDrakePlantChecks : IO Unit := do
  let bottom : PendulumState := { theta := 0.0, thetadot := 0.0 }
  let top : PendulumState := { theta := pi, thetadot := 0.0 }
  let horizontal : PendulumState := { theta := pi / 2.0, thetadot := 1.0 }

  LeanTest.assertTrue
    (approx (totalEnergy params bottom) (-params.mass * params.gravity * params.length) 1.0e-12)
    s!"Bottom energy should be -mgl, got {totalEnergy params bottom}"
  LeanTest.assertTrue
    (approx (totalEnergy params top) (params.mass * params.gravity * params.length) 1.0e-12)
    s!"Top energy should be mgl, got {totalEnergy params top}"
  LeanTest.assertTrue
    (approx (totalEnergy params horizontal)
      (0.5 * params.mass * params.length * params.length) 1.0e-12)
    s!"Horizontal unit-speed energy should be kinetic only, got {totalEnergy params horizontal}"

@[test]
def testPassiveSimulationRecordsContinuousInterval : IO Unit := do
  let run ← assertOk (solvePassive? params { theta := 0.1, thetadot := 0.0 } 0.0 0.05 defaultInput)
    "pendulum passive solve"
  match run.trace.validate? with
  | .error msg => LeanTest.fail s!"Pendulum trace should validate: {msg}"
  | .ok () => pure ()
  LeanTest.assertTrue (approx run.t1 0.05 1.0e-12)
    s!"Pendulum run should reach requested final time, got {run.t1}"
  LeanTest.assertEqual run.moves.size 2
    "A pure continuous pendulum interval should project to interval and checkpoint moves"
  LeanTest.assertTrue (run.moves[0]!.kind == SkeletonMoveKind.intervalAdjoint)
    "First pendulum move should be the interval adjoint"
  LeanTest.assertTrue (Float.isFinite run.finalState.theta && Float.isFinite run.finalState.thetadot)
    s!"Pendulum final state should be finite, got {reprStr run.finalState}"
  LeanTest.assertTrue (Float.isFinite run.finalEnergy)
    s!"Pendulum final energy should be finite, got {run.finalEnergy}"

@[test]
def testMultibodyBenchmarkPendulumBoundaryIsRecorded : IO Unit := do
  let result ← assertOk buildMultibodyPendulum?
    "multibody pendulum benchmark boundary"
  assertOk result.step.validate? "multibody pendulum full plant step"
  assertOk result.trace.validate? "multibody pendulum trace"

  let expectedScale := 2.0 * pi * Float.sqrt (params.length / params.gravity)
  LeanTest.assertEqual result.step.model.modelUri benchmarkPendulumFactory
    "Passive benchmark should record Drake's MakePendulumPlant factory boundary"
  LeanTest.assertEqual result.step.model.numPositions 1
    "Benchmark Pendulum MultibodyPlant has one position"
  LeanTest.assertEqual result.step.model.numVelocities 1
    "Benchmark Pendulum MultibodyPlant has one velocity"
  LeanTest.assertEqual result.step.model.numActuatedDofs 1
    "Benchmark Pendulum MultibodyPlant has one pin actuator"
  LeanTest.assertEqual result.step.q0 #[pi / 3.0]
    "Passive benchmark should initialize the pin angle to pi/3"
  LeanTest.assertEqual result.step.v0 #[0.0]
    "Passive benchmark should leave angular velocity at zero"
  LeanTest.assertEqual result.step.actuation #[0.0]
    "Passive benchmark should connect zero pin torque"
  LeanTest.assertTrue (approx result.referenceTimeScale expectedScale 1.0e-12)
    s!"Reference time scale should be 2*pi*sqrt(l/g), got {result.referenceTimeScale}"
  LeanTest.assertTrue (approx result.maxTimeStep (expectedScale / 100.0) 1.0e-12)
    s!"Max step should be reference_time_scale/100, got {result.maxTimeStep}"
  LeanTest.assertTrue (approx result.step.t1 (5.0 * expectedScale) 1.0e-12)
    s!"Simulation time should be five reference periods, got {result.step.t1}"
  LeanTest.assertTrue (result.config.integrationScheme == MultibodyIntegratorScheme.rungeKutta3)
    "Passive benchmark should default to Drake's runge_kutta3 flag"
  LeanTest.assertTrue (!result.config.integrationScheme.fixedStep)
    "Runge-Kutta3 should be recorded as a variable-step integrator"
  LeanTest.assertTrue (approx result.config.targetAccuracy 0.001 1.0e-12)
    s!"Target accuracy should match Drake's benchmark, got {result.config.targetAccuracy}"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.label == "full-physics-step:multibody pendulum passive benchmark plant"))
    "Move list should expose the full-plant benchmark as a primitive physics solve"
  let x0 : PendulumState := { theta := pi / 3.0, thetadot := 0.0 }
  let expectedDx ← assertOk (derivative? params defaultInput x0)
    "multibody pendulum expected full physics derivative"
  LeanTest.assertTrue (result.fullPhysics.support.policy == SupportPolicy.fullSupport)
    "Pendulum has no contact candidates, so support selection should be exact full support"
  LeanTest.assertEqual result.fullPhysics.support.totalCandidates 0
    "Pendulum full physics should have no contact candidates"
  LeanTest.assertEqual result.fullPhysics.contactForces.size 0
    "Pendulum full physics should not invent contact forces"
  LeanTest.assertTrue (result.fullPhysics.supportMove.exactness == MoveExactness.exact)
    "Empty full-support contact selection should be exact"
  assertArrayNear result.fullPhysics.equation.massMatrix[0]!
    (massMatrix params)[0]! 1.0e-12
    "Full physics mass matrix should come from the Drake pendulum formula"
  assertArrayNear result.fullPhysics.equation.biasForces
    #[biasTorque params x0] 1.0e-12
    "Full physics bias should use gravity plus damping torque"
  assertArrayNear result.fullPhysics.generalizedForces #[0.0] 1.0e-12
    "Passive benchmark should supply zero generalized force"
  assertArrayNear result.fullPhysics.derivative.qdot #[0.0] 1.0e-12
    "Full physics qdot should match the passive benchmark initial velocity"
  assertArrayNear result.fullPhysics.derivative.vdot
    #[expectedDx.thetadot] 1.0e-12
    "Full physics vdot should match the exact pendulum derivative"

@[test]
def testUrdfDynamicsParityBoundaryMatchesHandWrittenPlant : IO Unit := do
  let boundary ← assertOk buildUrdfDynamicsParity?
    "Pendulum URDF dynamics parity boundary"
  assertOk boundary.validate? "Pendulum URDF dynamics parity validation"
  LeanTest.assertEqual boundary.urdfUrl pendulumUrdfUrl
    "URDF parity boundary should use Drake's package URL"
  LeanTest.assertEqual boundary.plantModel.modelUri pendulumUrdfUrl
    "URDF plant model should point at Pendulum.urdf"
  LeanTest.assertEqual boundary.plantModel.numPositions 1
    "Pendulum.urdf plant should expose one generalized position"
  LeanTest.assertEqual boundary.plantModel.numVelocities 1
    "Pendulum.urdf plant should expose one generalized velocity"
  LeanTest.assertEqual boundary.plantModel.numActuatedDofs 1
    "Pendulum.urdf plant should expose one actuator"
  LeanTest.assertEqual boundary.numRandomizedDrakeSamples 100
    "Drake's URDF dynamics regression randomizes 100 states"
  LeanTest.assertEqual boundary.samples.size 5
    "Tyr fixture should retain deterministic representatives of the randomized Drake check"
  LeanTest.assertTrue (boundary.samples.all (fun sample => sample.maxAbsError <= boundary.tolerance))
    s!"URDF and hand plant derivatives should match, got {reprStr (boundary.samples.map (fun s => s.maxAbsError))}"
  let graph := boundary.graph
  LeanTest.assertTrue
    (graph.moves.any (fun move => move.label.contains "Parser.AddModelsFromUrl"))
    "URDF parity graph should keep the parser/finalize boundary visible"
  LeanTest.assertTrue
    (graph.moves.any (fun move => move.label.contains "Compare MultibodyPlant and PendulumPlant derivatives"))
    "URDF parity graph should keep the dynamics comparison boundary visible"

@[test]
def testLqrSimulationStabilizesUprightFixedPoint : IO Unit := do
  let lin := linearizationAboutUpright params
  LeanTest.assertEqual lin.A.size 2
    "Pendulum LQR linearization should expose a 2x2 A matrix"
  LeanTest.assertTrue (approx ((lin.A[1]!).getD 0 0.0) (params.gravity / params.length) 1.0e-12)
    s!"Upright linearization should have +g/l angle term, got {reprStr lin.A}"
  LeanTest.assertTrue
    (approx ((lin.B[1]!).getD 0 0.0) (1.0 / (params.mass * params.length * params.length)) 1.0e-12)
    s!"Input matrix should be inverse inertia, got {reprStr lin.B}"

  let gain ← assertOk (lqrGain? params lqrConfig) "pendulum LQR gain"
  LeanTest.assertTrue (gain.kTheta > 0.0 && gain.kThetadot > 0.0)
    s!"Upright LQR gain should apply positive feedback gains in error coordinates, got {reprStr gain}"
  let run ← assertOk simulateLqr? "pendulum LQR simulation"
  LeanTest.assertTrue (approx run.t1 10.0 1.0e-12)
    s!"LQR run should advance to Drake's 10s horizon, got {run.t1}"
  LeanTest.assertTrue (approx run.finalState.theta pi 1.0e-3)
    s!"LQR should stabilize theta near pi, got {reprStr run.finalState}"
  LeanTest.assertTrue (approx run.finalState.thetadot 0.0 1.0e-3)
    s!"LQR should stabilize thetadot near zero, got {reprStr run.finalState}"
  LeanTest.assertTrue (run.moves.any (fun move => move.kind == SkeletonMoveKind.localSchurBlock))
    "LQR controller should be recorded as a local controller block before interval adjoints"

@[test]
def testEnergyShapingSimulationImprovesUprightEnergyGap : IO Unit := do
  let x0 : PendulumState := { theta := 0.1, thetadot := 0.2 }
  let desiredEnergy := uprightEnergy params
  let initialGap := Float.abs (totalEnergy params x0 - desiredEnergy)
  let controllerTau := (energyShapingController params x0).tau
  LeanTest.assertTrue (Float.isFinite controllerTau)
    s!"Energy shaping controller should produce finite torque, got {controllerTau}"
  LeanTest.assertTrue
    (approx (energyShapingControllerDesiredEnergy params) (1.1 * params.mass * params.gravity * params.length) 1.0e-12)
    "Controller should use Drake's 1.1*m*g*l desired controller energy"

  let run ← assertOk (simulateEnergyShaping? params energyShapingConfig x0)
    "pendulum energy-shaping simulation"
  let finalGap := Float.abs (run.finalEnergy - desiredEnergy)
  LeanTest.assertTrue (finalGap < 0.5 * initialGap)
    s!"Energy shaping should reduce the upright energy gap by at least half, initial={initialGap}, final={finalGap}"
  LeanTest.assertTrue (run.moves.any (fun move => move.kind == SkeletonMoveKind.localSchurBlock))
    "Energy-shaping controller should be recorded as a local controller block"

@[test]
def testSymbolicDynamicsAndDirectCollocationMetadataMatchDrakeExecutables : IO Unit := do
  LeanTest.assertTrue symbolicDynamicsAgree
    "PendulumPlant and MultibodyPlant symbolic dynamics should agree in the recorded formula"
  LeanTest.assertEqual symbolicDynamics.pendulumPlantThetaDot "thetadot"
    "Symbolic theta derivative should be the velocity state"
  LeanTest.assertTrue (symbolicDynamics.pendulumPlantThetaDotDot.contains "sin(theta)")
    "Symbolic acceleration formula should retain the nonlinear gravity term"

  let spec := directCollocationSpec
  match spec.validate? with
  | .error msg => LeanTest.fail s!"Direct collocation spec should validate: {msg}"
  | .ok () => pure ()
  LeanTest.assertEqual spec.numTimeSamples 21
    "Direct collocation should use Drake's 21 knot points"
  LeanTest.assertTrue (approx spec.minimumTimeStep 0.2 1.0e-12)
    s!"Minimum time step should be 0.2, got {spec.minimumTimeStep}"
  LeanTest.assertTrue (approx spec.maximumTimeStep 0.5 1.0e-12)
    s!"Maximum time step should be 0.5, got {spec.maximumTimeStep}"
  LeanTest.assertTrue (approx spec.torqueLimit 3.0 1.0e-12)
    s!"Torque limit should be 3 Nm, got {spec.torqueLimit}"
  LeanTest.assertTrue (approx spec.runningCostR 10.0 1.0e-12)
    s!"Running input effort cost should use R=10, got {spec.runningCostR}"
  LeanTest.assertTrue (approx spec.minimumDuration 4.0 1.0e-12)
    s!"Minimum duration should be 20 intervals * 0.2s = 4s, got {spec.minimumDuration}"
  LeanTest.assertTrue (approx spec.maximumDuration 10.0 1.0e-12)
    s!"Maximum duration should be 20 intervals * 0.5s = 10s, got {spec.maximumDuration}"
  let graph := directCollocationGraph spec
  LeanTest.assertTrue
    (graph.moves.any
      (fun move =>
        move.kind == SkeletonMoveKind.localSchurBlock &&
        move.exactness == MoveExactness.controlledApproximation &&
        move.label.contains "direct collocation"))
    "Direct-collocation solve should remain visible as a controlled solver boundary"

end Tests.EventSkeletonPendulumExample
