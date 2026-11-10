import LeanTest
import Tyr.EventSkeleton.Examples.Atlas

namespace Tests.EventSkeletonAtlasExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.Atlas

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def setAtD (xs : Array Float) (idx : Nat) (value : Float) :
    Array Float :=
  if idx < xs.size then xs.set! idx value else xs

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
def testDrakeReferencesAndDefaultFlagsAreRecorded : IO Unit := do
  let result ← assertOk buildEndToEnd? "Atlas end-to-end build"
  LeanTest.assertTrue
    (result.references.any (fun ref => ref.path == "../drake/examples/atlas/BUILD.bazel"))
    "Atlas example should reference Drake's BUILD.bazel"
  LeanTest.assertTrue
    (result.references.any (fun ref => ref.path == "../drake/examples/atlas/atlas_run_dynamics.cc"))
    "Atlas example should reference Drake's passive dynamics executable"
  LeanTest.assertTrue
    (result.references.any (fun ref => ref.path == modelUri))
    "Atlas example should reference the convex-hull Atlas URDF"
  LeanTest.assertTrue (result.config.contactApproximation == DiscreteContactApproximation.sap)
    "Drake's default Atlas contact approximation flag is sap"
  LeanTest.assertTrue (result.config.isDiscrete)
    "Drake's default Atlas plant uses a positive discrete update period"
  LeanTest.assertTrue (approx result.config.timeStep 0.01 1.0e-12)
    s!"Atlas time step should match the Drake flag default, got {result.config.timeStep}"
  LeanTest.assertTrue (approx result.config.penetrationAllowance 1.0e-3 1.0e-15)
    s!"Atlas penetration allowance should match the Drake flag default, got {result.config.penetrationAllowance}"
  LeanTest.assertTrue (approx result.config.stictionTolerance 1.0e-3 1.0e-15)
    s!"Atlas stiction tolerance should match the Drake flag default, got {result.config.stictionTolerance}"

@[test]
def testAtlasBuildTargetAndRunDynamicsBoundaryMatchDrake : IO Unit := do
  let result ← assertOk buildEndToEnd? "Atlas end-to-end build"
  let _ ← assertOk (validateBuildTarget? result.buildTarget)
    "Atlas BUILD target metadata"
  let _ ← assertOk result.runDynamics.validate?
    "Atlas run dynamics executable metadata"

  LeanTest.assertTrue (result.buildTarget.kind == AtlasBuildTargetKind.ccBinary)
    "atlas_run_dynamics should be represented as a drake_cc_binary"
  LeanTest.assertEqual result.buildTarget.name "atlas_run_dynamics"
    "Atlas BUILD target name should match Drake"
  LeanTest.assertEqual result.buildTarget.srcs #["atlas_run_dynamics.cc"]
    "Atlas BUILD target should compile atlas_run_dynamics.cc"
  LeanTest.assertTrue (result.buildTarget.hasData "@drake_models//:atlas")
    "Atlas BUILD target should carry the Drake model runfile data"
  LeanTest.assertTrue (result.buildTarget.hasDep "//multibody/parsing")
    "Atlas executable should depend on multibody parsing"
  LeanTest.assertTrue (result.buildTarget.hasDep "//systems/analysis:simulator_gflags")
    "Atlas executable should depend on simulator_gflags"
  LeanTest.assertTrue (result.buildTarget.hasDep "//visualization:visualization_config_functions")
    "Atlas executable should depend on visualization_config_functions"
  LeanTest.assertTrue result.buildTarget.addTestRule
    "Atlas BUILD target should keep its smoke-test rule"
  LeanTest.assertTrue (result.buildTarget.hasTestRuleArg "--simulation_time=0.01")
    "Atlas smoke test should pass --simulation_time=0.01"
  LeanTest.assertTrue (result.buildTarget.hasTestRuleArg "--simulator_target_realtime_rate=0.0")
    "Atlas smoke test should disable realtime pacing"

  LeanTest.assertEqual result.runDynamics.executableName "atlas_run_dynamics"
    "Run boundary should name Drake's executable"
  LeanTest.assertEqual result.runDynamics.modelRunfileData "@drake_models//:atlas"
    "Run boundary should preserve Drake's model runfile data"
  LeanTest.assertTrue result.runDynamics.usesParserAddModelsFromUrl
    "Atlas run should load the URDF through Parser.AddModelsFromUrl"
  LeanTest.assertTrue result.runDynamics.addsDefaultVisualization
    "Atlas run should install Drake's default visualization"
  LeanTest.assertTrue result.runDynamics.usesMakeSimulatorFromGflags
    "Atlas run should construct the Simulator from gflags"
  LeanTest.assertTrue result.runDynamics.usageMentionsMeldis
    "Atlas usage text should keep the meldis launch boundary visible"
  LeanTest.assertEqual result.runDynamics.smokeTestArgs
    #["--simulation_time=0.01", "--simulator_target_realtime_rate=0.0"]
    "Run boundary smoke-test args should match BUILD.bazel"

@[test]
def testAtlasModelDimensionsAndFloatingPelvisConvention : IO Unit := do
  let result ← assertOk buildEndToEnd? "Atlas end-to-end build"
  LeanTest.assertEqual result.model.numPositions 37
    "Drake demands Atlas num_positions == 37"
  LeanTest.assertEqual result.model.numVelocities 36
    "Drake demands Atlas num_velocities == 36"
  LeanTest.assertEqual result.model.numActuatedDofs 30
    "Atlas actuated dofs are the non-floating velocities"
  LeanTest.assertEqual result.model.stateDim 73
    "Atlas continuous state size should be q + v"
  LeanTest.assertEqual result.stateCoordinateNames.size 73
    "Atlas coordinate names should cover q and v"
  LeanTest.assertEqual result.model.floatingBases.size 1
    "Atlas example checks one floating pelvis body"
  let pelvis := result.model.floatingBases[0]!
  LeanTest.assertEqual pelvis.bodyName "pelvis"
  LeanTest.assertTrue (pelvis.convention == FloatingBaseCoordinateConvention.quaternion)
    "Drake checks that pelvis has quaternion dofs"
  LeanTest.assertEqual pelvis.floatingPositionsStart 0
    "Drake checks pelvis floating positions start at q[0]"
  LeanTest.assertEqual pelvis.floatingVelocitiesStartInV 0
    "Drake checks pelvis floating velocities start at v[0]"

@[test]
def testAtlasInitialContextAndGroundContactPrimitive : IO Unit := do
  let result ← assertOk buildEndToEnd? "Atlas end-to-end build"
  LeanTest.assertEqual result.step.q0.size numPositions
    "Initial q should match Atlas num_positions"
  LeanTest.assertEqual result.step.v0.size numVelocities
    "Initial v should match Atlas num_velocities"
  LeanTest.assertEqual result.step.actuation.size numActuatedDofs
    "Fixed actuation should match plant.num_actuated_dofs"
  LeanTest.assertTrue (result.step.actuation.all (fun tau => approx tau 0.0 1.0e-15))
    "Drake fixes Atlas actuation to zero"
  LeanTest.assertTrue (approx (result.step.q0.getD 0 0.0) 1.0 1.0e-15)
    "Floating pelvis quaternion should start at identity qw = 1"
  LeanTest.assertTrue (approx (result.step.q0.getD 6 0.0) 0.95 1.0e-15)
    "Drake sets pelvis translation z to 0.95"
  LeanTest.assertTrue result.step.hasContactEnvironment
    "Atlas should register the ground half-space as a contact environment"
  LeanTest.assertTrue (approx result.ground.friction.staticFriction 1.0 1.0e-15)
    "Drake registers static friction 1.0 for the ground"
  LeanTest.assertTrue (approx result.ground.friction.dynamicFriction 1.0 1.0e-15)
    "The time-stepping Atlas example uses matching dynamic friction metadata"
  assertOk result.primitivePlant.validate? "Atlas primitive plant wrapper"
  LeanTest.assertEqual result.primitivePlant.intervalVertex 8705
    "Atlas primitive full-physics solve should target the simulator interval vertex"
  LeanTest.assertEqual result.primitivePlant.primitives.velocityDim numVelocities
    "Atlas primitive should expose one equation per velocity"
  LeanTest.assertEqual result.primitivePlant.primitives.contactCandidates.size 4
    "Atlas local primitive provider should expose four foot-ground contact candidate views"
  LeanTest.assertEqual result.fullPhysics.support.selectedLocalIndices #[0, 1, 2, 3]
    "Default pelvis height should put all Atlas foot support points on the ground"
  LeanTest.assertEqual result.fullPhysics.support.totalCandidates 4
    "Atlas support selection should preserve the source candidate count"
  LeanTest.assertTrue
    (result.fullPhysics.contactForces.all (fun force =>
      approx force.normalForce (atlasTotalMass * atlasGravity / 4.0) 1.0e-12))
    s!"Atlas static support should distribute body weight across feet, got {reprStr result.fullPhysics.contactForces}"
  LeanTest.assertEqual result.fullPhysics.generalizedPrimitiveForce atlasGravityGeneralizedForce
    "Atlas gravity should be visible as a primitive generalized force"
  LeanTest.assertTrue
    (approx (result.fullPhysics.generalizedContactForce.getD pelvisVzIndex 0.0)
      (atlasTotalMass * atlasGravity) 1.0e-12)
    s!"Atlas foot support should map normal forces through J^T, got {reprStr result.fullPhysics.generalizedContactForce}"
  LeanTest.assertEqual atlasPrimitiveMassDiagonal.size numVelocities
    "Atlas primitive mass scaffold should match plant velocity dimension"
  LeanTest.assertTrue (atlasPrimitiveMassDiagonal.all (fun m => m > 0.0))
    "Atlas primitive mass scaffold should be strictly positive"
  let generalizedActuation ← assertOk
    (atlasGeneralizedActuation? result.step)
    "Atlas generalized actuation mapping"
  LeanTest.assertEqual generalizedActuation.size numVelocities
    "Atlas generalized actuation should be padded to all velocities"
  LeanTest.assertTrue
    ((generalizedActuation.extract 0 6).all (fun tau => approx tau 0.0 1.0e-15))
    "Floating pelvis velocities should receive no direct actuation"
  LeanTest.assertEqual atlasActuationMap.velocityDim numVelocities
    "Atlas actuation map should target the full generalized velocity vector"
  LeanTest.assertEqual atlasActuationMap.actuatorVelocityIndices.size numActuatedDofs
    "Atlas actuation map should contain one velocity slot per actuator"
  LeanTest.assertEqual atlasActuationMap.actuatorVelocityIndices[0]! 6
    "Atlas first actuator should follow the six floating-pelvis velocities"
  LeanTest.assertEqual
    atlasActuationMap.actuatorVelocityIndices[atlasActuationMap.actuatorVelocityIndices.size - 1]!
    (numVelocities - 1)
    "Atlas last actuator should land on the final generalized velocity coordinate"
  LeanTest.assertTrue
    ((generalizedActuation.extract 6 numVelocities).all (fun tau => approx tau 0.0 1.0e-15))
    "Passive Atlas actuator inputs should remain zero"
  LeanTest.assertEqual result.fullPhysics.equation.massMatrix
    (FloatMatrix.diagonal atlasPrimitiveMassDiagonal)
  LeanTest.assertTrue
    (result.fullPhysics.derivative.vdot.all (fun a => approx a 0.0 1.0e-15))
    s!"Passive Atlas primitive should balance gravity through static foot support, got {reprStr result.fullPhysics.derivative.vdot}"

@[test]
def testAtlasContactSupportRecomputesFromPelvisHeight : IO Unit := do
  let raisedQ :=
    setAtD initialPositions pelvisZIndex 1.20
  let raisedStep := { atlasPlantStep with q0 := raisedQ }
  let provider :=
    atlasPassiveFullPhysicsPrimitiveProvider "raised Atlas primitive provider"
  let primitives ← assertOk
    (provider.validPrimitivesAt? raisedStep)
    "raised Atlas primitive provider"
  LeanTest.assertEqual primitives.contactCandidates.size 4
    "Raised Atlas still exposes candidate views from the provider"
  let support ← assertOk (provider.supportAt? raisedStep)
    "raised Atlas provider support"
  LeanTest.assertEqual support.selectedLocalIndices #[]
    "Raising the pelvis should recompute support to empty"
  let full ← assertOk (primitives.solve? 8799)
    "raised Atlas full physics solve"
  LeanTest.assertEqual full.support.selectedLocalIndices #[]
    "Full physics should use the recomputed empty support"
  LeanTest.assertEqual full.contactForces.size 0
    "No selected support should synthesize no static contact forces"
  LeanTest.assertTrue
    (approx (full.derivative.vdot.getD pelvisVzIndex 0.0) (-atlasGravity) 1.0e-12)
    s!"Unsupported Atlas pelvis should accelerate downward at gravity, got {reprStr full.derivative.vdot}"

@[test]
def testAtlasFullPhysicsPrimitiveProviderRecomputesStepStateAndActuation :
    IO Unit := do
  let provider :=
    atlasPassiveFullPhysicsPrimitiveProvider "Atlas dynamic provider test"
  let movingV := setAtD zeroVelocities pelvisVzIndex (-0.25)
  let drivenU := setAtD zeroActuation 0 12.0
  let movingStep := { atlasPlantStep with v0 := movingV, actuation := drivenU }
  let primitives ← assertOk (provider.validPrimitivesAt? movingStep)
    "Atlas dynamic provider primitives"
  LeanTest.assertTrue
    (approx (primitives.qdot.getD pelvisVzIndex 0.0) (-0.25) 1.0e-12)
    s!"Atlas provider qdot should come from the current plant step velocity, got {reprStr primitives.qdot}"
  LeanTest.assertTrue
    (approx (primitives.actuationForces.getD 6 0.0) 12.0 1.0e-12)
    s!"Atlas provider should map the first actuator after floating velocities, got {reprStr primitives.actuationForces}"
  LeanTest.assertTrue
    (approx (primitives.actuationForces.getD pelvisVzIndex 0.0) 0.0 1.0e-12)
    "Atlas provider should not map actuator force onto floating pelvis velocities"
  let support ← assertOk (provider.supportAt? movingStep)
    "Atlas dynamic provider support"
  LeanTest.assertEqual support.selectedLocalIndices #[0, 1, 2, 3]
    "Atlas provider should still select all feet at the default pelvis height"
  let badMsg ← assertError
    (provider.validPrimitivesAt? { atlasPlantStep with ground? := none })
    "Atlas provider missing ground"
  LeanTest.assertTrue (badMsg.contains "ground half-space")
    s!"Malformed Atlas step should fail at provider validation, got {badMsg}"

@[test]
def testAtlasGraphExposesFullPlantAdvanceBoundary : IO Unit := do
  let result ← assertOk buildEndToEnd? "Atlas end-to-end build"
  LeanTest.assertTrue (result.graph.containsMoveKind .localSchurBlock)
    "Parser/finalize and ground registration should stay visible as provider blocks"
  LeanTest.assertTrue (result.graph.containsMoveKind .freezeControl)
    "Zero actuation should be represented as a frozen control boundary"
  LeanTest.assertTrue (result.graph.containsMoveKind .intervalAdjoint)
    "Simulator.AdvanceTo should be represented as a full-plant interval"
  LeanTest.assertTrue (result.graph.containsMoveKind .checkpointBoundary)
    "Atlas final context should be checkpointed"
  LeanTest.assertTrue
    (result.graph.moves.any
      (fun move =>
        move.kind == .intervalAdjoint &&
        move.exactness == MoveExactness.exact &&
        move.label == "full-physics-step:atlas passive initial primitive provider"))
    "Atlas should expose the full plant advance primitive, not a simplified fake dynamics step"
  LeanTest.assertTrue
    (result.graph.moves.any
      (fun move =>
        move.kind == .markMarginalize &&
        move.label == "contact-support-selection:atlas passive initial primitive provider"))
    "Atlas graph should record the primitive contact-support selection boundary"
  LeanTest.assertTrue
    (result.graph.moves.any
      (fun move =>
        move.kind == .checkpointBoundary &&
        move.label.contains "@drake_models//:atlas"))
    "Atlas graph should expose BUILD/runfile resolution for the Drake model data"
  LeanTest.assertTrue
    (result.graph.moves.any
      (fun move =>
        move.kind == .localSchurBlock &&
        move.label.contains "AddDefaultVisualization"))
    "Atlas graph should expose Drake's default visualization boundary"
  LeanTest.assertTrue
    (result.graph.moves.any
      (fun move =>
        move.kind == .localSchurBlock &&
        move.label.contains "MakeSimulatorFromGflags"))
    "Atlas graph should expose simulator_gflags construction before AdvanceTo"
  LeanTest.assertTrue (result.executionStatus == AtlasExecutionStatus.primitivePhysicsSolved)
    "Current Atlas port should report that the local primitive physics provider executed"

end Tests.EventSkeletonAtlasExample
