import Tyr.EventSkeleton.Manipulator

/-!
# Drake Atlas Event-Skeleton Example

This ports the primitive boundary of `../drake/examples/atlas/atlas_run_dynamics.cc`.

The Drake example is not a small closed-form ODE.  It builds a parser-backed
`MultibodyPlant`, registers a ground half-space, finalizes the plant, fixes zero
actuation, sets the floating pelvis pose, and advances the simulator with
discrete contact enabled by default.  Here that whole plant advance is recorded
as a full multibody primitive.  A local foot-contact provider exposes the
support-selection and `J^T f` boundary that a later URDF collision compiler can
replace without changing the physics interface.
-/

namespace Tyr.EventSkeleton.Examples.Atlas

open Tyr.EventSkeleton

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/atlas/BUILD.bazel"
      concept := "declares atlas_run_dynamics as a drake_cc_binary with Drake model data, simulator gflags, visualization deps, and a smoke test"
    },
    {
      path := "../drake/examples/atlas/atlas_run_dynamics.cc"
      concept := "builds an Atlas MultibodyPlant, registers ground, checks floating pelvis layout, fixes zero actuation, and advances the simulator"
    },
    {
      path := "../drake/examples/atlas/README.md"
      concept := "describes the passive Atlas dynamics demo"
    },
    {
      path := "package://drake_models/atlas/atlas_convex_hull.urdf"
      concept := "convex-hull Atlas model loaded by Drake's parser"
    }
  ]

inductive AtlasBuildTargetKind where
  | ccBinary
  deriving Repr, BEq, Inhabited

structure AtlasBuildTarget where
  kind : AtlasBuildTargetKind := .ccBinary
  name : String := "atlas_run_dynamics"
  srcs : Array String := #["atlas_run_dynamics.cc"]
  data : Array String := #["@drake_models//:atlas"]
  deps : Array String := #[
    "//common:add_text_logging_gflags",
    "//multibody/parsing",
    "//systems/analysis:simulator",
    "//systems/analysis:simulator_gflags",
    "//systems/framework:diagram",
    "//visualization:visualization_config_functions",
    "@gflags"
  ]
  addTestRule : Bool := true
  testRuleArgs : Array String := #[
    "--simulation_time=0.01",
    "--simulator_target_realtime_rate=0.0"
  ]
  deriving Repr, BEq, Inhabited

namespace AtlasBuildTarget

def hasDep (target : AtlasBuildTarget) (dep : String) : Bool :=
  target.deps.any (fun actual => actual == dep)

def hasData (target : AtlasBuildTarget) (datum : String) : Bool :=
  target.data.any (fun actual => actual == datum)

def hasTestRuleArg (target : AtlasBuildTarget) (arg : String) : Bool :=
  target.testRuleArgs.any (fun actual => actual == arg)

def validate? (target : AtlasBuildTarget) : Except String Unit := do
  if target.kind != .ccBinary then
    .error "Atlas BUILD target should be a drake_cc_binary"
  if target.name != "atlas_run_dynamics" then
    .error s!"Atlas executable target should be atlas_run_dynamics, got {target.name}"
  if target.srcs != #["atlas_run_dynamics.cc"] then
    .error s!"Atlas executable should compile atlas_run_dynamics.cc, got {target.srcs}"
  if !target.hasData "@drake_models//:atlas" then
    .error "Atlas executable should include @drake_models//:atlas runfile data"
  if !target.hasDep "//multibody/parsing" then
    .error "Atlas executable should depend on Drake multibody parsing"
  if !target.hasDep "//systems/analysis:simulator_gflags" then
    .error "Atlas executable should depend on simulator_gflags"
  if !target.hasDep "//visualization:visualization_config_functions" then
    .error "Atlas executable should depend on visualization_config_functions"
  if !target.hasDep "@gflags" then
    .error "Atlas executable should expose gflags"
  if !target.addTestRule then
    .error "Atlas executable should keep BUILD.bazel add_test_rule=True"
  if !target.hasTestRuleArg "--simulation_time=0.01" then
    .error "Atlas smoke test should override --simulation_time=0.01"
  if !target.hasTestRuleArg "--simulator_target_realtime_rate=0.0" then
    .error "Atlas smoke test should override simulator target realtime rate to 0.0"

end AtlasBuildTarget

def buildTarget : AtlasBuildTarget := {}

def validateBuildTarget? (target : AtlasBuildTarget := buildTarget) :
    Except String Unit :=
  target.validate?

def modelUri : String :=
  "package://drake_models/atlas/atlas_convex_hull.urdf"

def simulationTime : Float := 2.0
def penetrationAllowance : Float := 1.0e-3
def stictionTolerance : Float := 1.0e-3
def mbpDiscreteUpdatePeriod : Float := 0.01
def atlasGravity : Float := 9.81
def atlasTotalMass : Float := 80.0

def numPositions : Nat := 37
def numVelocities : Nat := 36
def numActuatedDofs : Nat := 30

def pelvisBodyName : String := "pelvis"

def floatingPelvis : FloatingBaseModelInstance :=
  {
    bodyName := pelvisBodyName
    convention := .quaternion
    floatingPositionsStart := 0
    floatingVelocitiesStartInV := 0
  }

def plantConfig : MultibodyPlantConfigPrimitive :=
  {
    timeStep := mbpDiscreteUpdatePeriod
    penetrationAllowance := penetrationAllowance
    stictionTolerance := stictionTolerance
    contactApproximation := .sap
  }

def atlasModel : FullMultibodyPlantModel :=
  {
    modelName := "atlas"
    modelUri := modelUri
    numPositions := numPositions
    numVelocities := numVelocities
    numActuatedDofs := numActuatedDofs
    floatingBases := #[floatingPelvis]
    finalized := true
    label := "atlas-convex-hull"
  }

def ground : HalfSpaceContactEnvironment :=
  {
    visualName := "GroundVisualGeometry"
    collisionName := "GroundCollisionGeometry"
    friction := { staticFriction := 1.0, dynamicFriction := 1.0 }
  }

def zeroActuation : Array Float :=
  Array.replicate numActuatedDofs 0.0

def zeroVelocities : Array Float :=
  Array.replicate numVelocities 0.0

def pelvisInitialTranslation : Array Float :=
  #[0.0, 0.0, 0.95]

def pelvisZIndex : Nat := 6
def pelvisVxIndex : Nat := 3
def pelvisVyIndex : Nat := 4
def pelvisVzIndex : Nat := 5

def pelvisZFromStep (step : FullMultibodyPlantStep) : Float :=
  step.q0.getD pelvisZIndex 0.0

def pelvisVxFromStep (step : FullMultibodyPlantStep) : Float :=
  step.v0.getD pelvisVxIndex 0.0

def pelvisVyFromStep (step : FullMultibodyPlantStep) : Float :=
  step.v0.getD pelvisVyIndex 0.0

def pelvisVzFromStep (step : FullMultibodyPlantStep) : Float :=
  step.v0.getD pelvisVzIndex 0.0

/--
Drake stores a quaternion floating-base pose first.  The Atlas example checks
that the pelvis floating positions start at index zero before setting
`X_WP = Translation3d(0, 0, 0.95)`.
-/
def initialPositions : Array Float :=
  #[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95] ++
    Array.replicate numActuatedDofs 0.0

def jointPositionCoordinateNames : Array String := Id.run do
  let mut out : Array String := #[]
  for i in [:numActuatedDofs] do
    out := out.push s!"q_atlas_joint_{i}"
  return out

def jointVelocityCoordinateNames : Array String := Id.run do
  let mut out : Array String := #[]
  for i in [:numActuatedDofs] do
    out := out.push s!"v_atlas_joint_{i}"
  return out

def positionCoordinateNames : Array String :=
  #["pelvis_qw", "pelvis_qx", "pelvis_qy", "pelvis_qz",
    "pelvis_x", "pelvis_y", "pelvis_z"] ++ jointPositionCoordinateNames

def velocityCoordinateNames : Array String :=
  #["pelvis_wx", "pelvis_wy", "pelvis_wz",
    "pelvis_vx", "pelvis_vy", "pelvis_vz"] ++ jointVelocityCoordinateNames

def stateCoordinateNames : Array String :=
  positionCoordinateNames ++ velocityCoordinateNames

def atlasPlantStep : FullMultibodyPlantStep :=
  {
    model := atlasModel
    config := plantConfig
    q0 := initialPositions
    v0 := zeroVelocities
    actuation := zeroActuation
    t0 := 0.0
    t1 := simulationTime
    ground? := some ground
    label := "atlas-passive-full-multibody-plant"
  }

structure AtlasRunDynamicsBoundary where
  executableName : String := "atlas_run_dynamics"
  sourcePath : String := "../drake/examples/atlas/atlas_run_dynamics.cc"
  modelUri : String := Atlas.modelUri
  modelRunfileData : String := "@drake_models//:atlas"
  simulationTime : Float := Atlas.simulationTime
  smokeTestSimulationTime : Float := 0.01
  smokeTestTargetRealtimeRate : Float := 0.0
  penetrationAllowance : Float := Atlas.penetrationAllowance
  stictionTolerance : Float := Atlas.stictionTolerance
  mbpDiscreteUpdatePeriod : Float := Atlas.mbpDiscreteUpdatePeriod
  contactApproximation : DiscreteContactApproximation := .sap
  usesDiagramBuilder : Bool := true
  usesParserAddModelsFromUrl : Bool := true
  addsDefaultVisualization : Bool := true
  usesMakeSimulatorFromGflags : Bool := true
  usageMentionsMeldis : Bool := true
  smokeTestArgs : Array String := #[
    "--simulation_time=0.01",
    "--simulator_target_realtime_rate=0.0"
  ]
  deriving Repr, BEq, Inhabited

namespace AtlasRunDynamicsBoundary

def validate? (boundary : AtlasRunDynamicsBoundary) : Except String Unit := do
  if boundary.executableName != "atlas_run_dynamics" then
    .error s!"Atlas run dynamics executable name mismatch: {boundary.executableName}"
  if boundary.sourcePath != "../drake/examples/atlas/atlas_run_dynamics.cc" then
    .error s!"Atlas run dynamics source path mismatch: {boundary.sourcePath}"
  if boundary.modelUri != Atlas.modelUri then
    .error s!"Atlas run dynamics model URI mismatch: {boundary.modelUri}"
  if boundary.modelRunfileData != "@drake_models//:atlas" then
    .error s!"Atlas run dynamics should use @drake_models//:atlas runfile data, got {boundary.modelRunfileData}"
  if !boundary.simulationTime.isFinite || boundary.simulationTime <= 0.0 then
    .error s!"Atlas simulation_time must be positive and finite, got {boundary.simulationTime}"
  if !boundary.smokeTestSimulationTime.isFinite || boundary.smokeTestSimulationTime <= 0.0 then
    .error s!"Atlas smoke-test simulation time must be positive and finite, got {boundary.smokeTestSimulationTime}"
  if Float.abs (boundary.smokeTestSimulationTime - 0.01) > 1.0e-12 then
    .error s!"Atlas BUILD smoke test should use --simulation_time=0.01, got {boundary.smokeTestSimulationTime}"
  if !boundary.smokeTestTargetRealtimeRate.isFinite || boundary.smokeTestTargetRealtimeRate < 0.0 then
    .error s!"Atlas smoke-test target realtime rate must be nonnegative and finite, got {boundary.smokeTestTargetRealtimeRate}"
  if Float.abs boundary.smokeTestTargetRealtimeRate > 1.0e-12 then
    .error s!"Atlas BUILD smoke test should use --simulator_target_realtime_rate=0.0, got {boundary.smokeTestTargetRealtimeRate}"
  if !boundary.penetrationAllowance.isFinite || boundary.penetrationAllowance < 0.0 then
    .error s!"Atlas penetration_allowance must be nonnegative and finite, got {boundary.penetrationAllowance}"
  if !boundary.stictionTolerance.isFinite || boundary.stictionTolerance < 0.0 then
    .error s!"Atlas stiction_tolerance must be nonnegative and finite, got {boundary.stictionTolerance}"
  if !boundary.mbpDiscreteUpdatePeriod.isFinite || boundary.mbpDiscreteUpdatePeriod < 0.0 then
    .error s!"Atlas mbp_discrete_update_period must be nonnegative and finite, got {boundary.mbpDiscreteUpdatePeriod}"
  if boundary.contactApproximation != .sap then
    .error "Atlas default contact_approximation should be sap"
  if !boundary.usesDiagramBuilder || !boundary.usesParserAddModelsFromUrl ||
      !boundary.addsDefaultVisualization || !boundary.usesMakeSimulatorFromGflags ||
      !boundary.usageMentionsMeldis then
    .error "Atlas run dynamics should preserve DiagramBuilder, Parser.AddModelsFromUrl, AddDefaultVisualization, MakeSimulatorFromGflags, and meldis usage boundaries"
  if boundary.smokeTestArgs != #["--simulation_time=0.01", "--simulator_target_realtime_rate=0.0"] then
    .error s!"Atlas BUILD smoke-test args mismatch: {boundary.smokeTestArgs}"

end AtlasRunDynamicsBoundary

def runDynamicsBoundary : AtlasRunDynamicsBoundary := {}

def floatingPelvisPrimitiveMassDiagonal : Array Float :=
  -- A local primitive provider for the initial passive Atlas state.  The
  -- angular entries are positive inertial scaffolds; the translational entries
  -- use the Atlas total-mass scale so gravity produces a physical acceleration
  -- when the support set is empty.
  #[10.0, 10.0, 10.0, atlasTotalMass, atlasTotalMass, atlasTotalMass]

def jointPrimitiveMassDiagonal : Array Float :=
  Array.replicate numActuatedDofs 1.0

def atlasPrimitiveMassDiagonal : Array Float :=
  floatingPelvisPrimitiveMassDiagonal ++ jointPrimitiveMassDiagonal

def unitVelocityRow (idx : Nat) (value : Float := 1.0) : Array Float := Id.run do
  let mut row := Array.replicate numVelocities 0.0
  if idx < row.size then
    row := row.set! idx value
  return row

def atlasGravityGeneralizedForce : Array Float := Id.run do
  let mut force := Array.replicate numVelocities 0.0
  force := force.set! pelvisVzIndex (-atlasTotalMass * atlasGravity)
  return force

structure AtlasFootContactSpec where
  id : Nat
  bodyName : String
  label : String
  offsetX : Float
  offsetY : Float
  soleOffsetZ : Float := -0.95
  deriving Repr, Inhabited

def atlasFootContactSpecs : Array AtlasFootContactSpec :=
  #[
    { id := 0, bodyName := "l_foot", label := "left heel", offsetX := -0.10, offsetY := 0.09 },
    { id := 1, bodyName := "l_foot", label := "left toe", offsetX := 0.14, offsetY := 0.09 },
    { id := 2, bodyName := "r_foot", label := "right heel", offsetX := -0.10, offsetY := -0.09 },
    { id := 3, bodyName := "r_foot", label := "right toe", offsetX := 0.14, offsetY := -0.09 }
  ]

def atlasFootContactCandidate
    (step : FullMultibodyPlantStep) (spec : AtlasFootContactSpec) :
    ContactCandidate :=
  let footZ := pelvisZFromStep step + spec.soleOffsetZ
  ({
    id := spec.id
    bodyA := spec.bodyName
    bodyB := "ground"
    point_W := #[spec.offsetX, spec.offsetY, footZ]
    normal_W := #[0.0, 0.0, 1.0]
    signedDistance := footZ
    normalVelocity := pelvisVzFromStep step
    tangentVelocity := pelvisVxFromStep step
    tangentVelocity2 := pelvisVyFromStep step
    normalJacobian := unitVelocityRow pelvisVzIndex
    tangentJacobian := unitVelocityRow pelvisVxIndex
    tangentJacobian2 := unitVelocityRow pelvisVyIndex
    label := s!"Atlas {spec.label} ground support"
  } : ContactCandidate).withClassifiedMode penetrationAllowance stictionTolerance

def atlasFootContactCandidates
    (step : FullMultibodyPlantStep := atlasPlantStep) : Array ContactCandidate :=
  atlasFootContactSpecs.map (atlasFootContactCandidate step)

def atlasContactSupport
    (step : FullMultibodyPlantStep := atlasPlantStep) : ContactSupport :=
  ContactSupport.selectByDistance penetrationAllowance
    (atlasFootContactCandidates step)
    "Atlas convex-hull foot-ground support"
    |>.classifyCandidates penetrationAllowance stictionTolerance

def atlasStaticSupportContactForces?
    (support : ContactSupport) : Except String (Array ContactForceScalars) := do
  let selected ← support.selectedCandidates?
  if selected.isEmpty then
    pure #[]
  else
    let normalEach := atlasTotalMass * atlasGravity / selected.size.toFloat
    pure (selected.map (fun candidate =>
      ContactForceScalars.fromCandidate3D candidate normalEach 0.0 0.0))

def atlasActuationMap : GeneralizedActuationMap :=
  GeneralizedActuationMap.contiguousOffset
    numVelocities numActuatedDofs 6 "Atlas actuators after floating pelvis velocities"

def atlasGeneralizedActuation? (step : FullMultibodyPlantStep) :
    Except String (Array Float) := do
  atlasActuationMap.generalizedForcesFromStep? step

def atlasPassivePrimitivePrimitives?
    (step : FullMultibodyPlantStep := atlasPlantStep)
    (label : String := "atlas passive initial primitive provider") :
    Except String FullPhysicsPrimitives := do
  step.validate?
  if !step.hasContactEnvironment then
    .error "Atlas primitive provider expects the ground half-space contact environment"
  let generalizedActuation ← atlasGeneralizedActuation? step
  if atlasPrimitiveMassDiagonal.size != step.model.numVelocities then
    .error s!"Atlas primitive mass diagonal size {atlasPrimitiveMassDiagonal.size} != plant velocities {step.model.numVelocities}"
  let candidates := atlasFootContactCandidates step
  let support := atlasContactSupport step
  support.validateJacobianWidth? step.model.numVelocities
  let contactForces ← atlasStaticSupportContactForces? support
  pure {
    massMatrix := FloatMatrix.diagonal atlasPrimitiveMassDiagonal
    qdot := step.v0
    actuationForces := generalizedActuation
    biasForces := Array.replicate step.model.numVelocities 0.0
    generalizedForceContributions :=
      #[GeneralizedForceContribution.ofForce
          atlasGravityGeneralizedForce
          "Atlas gravity generalized force"
          "Atlas"]
    contactCandidates := candidates
    sourceContactCandidateCount? := some candidates.size
    supportPolicy := .threshold penetrationAllowance
    contactForceSource := .precomputed
    contactForces := contactForces
    distanceTol := penetrationAllowance
    tangentVelocityTol := stictionTolerance
    label := label
  }

def atlasPassiveFullPhysicsPrimitiveProvider
    (label : String := "atlas passive initial primitive provider") :
    FullPhysicsPrimitiveProvider FullMultibodyPlantStep :=
  {
    label := label
    primitivesAt? := fun step => atlasPassivePrimitivePrimitives? step label
  }

structure AtlasPrimitivePhysics where
  primitivePlant : FullPlantPrimitivePhysics
  fullPhysics : FullPhysicsResult
  deriving Repr, Inhabited

def solveAtlasPrimitivePhysics?
    (step : FullMultibodyPlantStep := atlasPlantStep)
    (intervalVertex : VertexId := 8705)
    (label : String := "atlas passive initial primitive provider") :
    Except String AtlasPrimitivePhysics := do
  let provider := atlasPassiveFullPhysicsPrimitiveProvider label
  let primitives ← provider.validPrimitivesAt? step
  let primitivePlant : FullPlantPrimitivePhysics := {
    step := step
    primitives := primitives
    intervalVertex := intervalVertex
    label := provider.label
  }
  let fullPhysics ← primitivePlant.solve?
  pure {
    primitivePlant := primitivePlant
    fullPhysics := fullPhysics
  }

inductive AtlasExecutionStatus where
  | fullPlantPrimitiveBoundary
  | primitivePhysicsSolved
  deriving Repr, BEq, Inhabited

namespace AtlasExecutionStatus

def label : AtlasExecutionStatus → String
  | .fullPlantPrimitiveBoundary => "full-plant-primitive-boundary"
  | .primitivePhysicsSolved => "primitive-physics-solved"

end AtlasExecutionStatus

def atlasGraph (fullPhysics : FullPhysicsResult) : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 8696, kind := .state .boundary, label := "BUILD.bazel atlas_run_dynamics" }
    |>.addVertex { id := 8697, kind := .state .boundary, label := "@drake_models//:atlas runfile data" }
    |>.addVertex { id := 8698, kind := .state .boundary, label := "AddDefaultVisualization" }
    |>.addVertex { id := 8699, kind := .state .boundary, label := "MakeSimulatorFromGflags" }
    |>.addVertex { id := 8700, kind := .state .boundary, label := "atlas_convex_hull_urdf" }
    |>.addVertex { id := 8701, kind := .state .interior, label := "MultibodyPlantConfig" }
    |>.addVertex { id := 8702, kind := .state .interior, label := "SceneGraph ground half-space" }
    |>.addVertex { id := 8703, kind := .frozen, label := "zero actuation input port" }
    |>.addVertex { id := 8704, kind := .state .checkpoint, label := "Atlas initial context" }
    |>.addVertex { id := 8705, kind := .interval, label := "Simulator.AdvanceTo Atlas full plant" }
    |>.addVertex { id := 8706, kind := .state .checkpoint, label := "Atlas final context" }
    |>.addMove {
      kind := .checkpointBoundary
      targets := #[8696]
      reads := #[8696]
      writes := #[8697]
      label := "BUILD.bazel resolves @drake_models//:atlas runfile data"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8698]
      reads := #[8701, 8702]
      writes := #[8698]
      label := "visualization::AddDefaultVisualization(&builder)"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8699]
      reads := #[8704]
      writes := #[8699, 8705]
      label := "MakeSimulatorFromGflags then AdvanceTo(FLAGS_simulation_time)"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8700, 8701]
      reads := #[8700, 8701]
      writes := #[8704]
      label := "Parser.AddModelsFromUrl + plant.Finalize"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8702]
      reads := #[8701, 8702]
      writes := #[8702]
      label := "register visual/collision half-space ground with Coulomb friction"
    }
    |>.addMove {
      kind := .freezeControl
      targets := #[8703]
      reads := #[8703]
      writes := #[8703]
      label := "FixValue zero Atlas actuation"
    }
    |>.addMove fullPhysics.supportMove
    |>.addMove fullPhysics.move
    |>.addMove {
      kind := .checkpointBoundary
      targets := #[8706]
      reads := #[8705]
      writes := #[8706]
      label := "store Atlas final context checkpoint"
    }

structure AtlasResult where
  references : Array DrakeReference
  buildTarget : AtlasBuildTarget
  runDynamics : AtlasRunDynamicsBoundary
  config : MultibodyPlantConfigPrimitive
  model : FullMultibodyPlantModel
  ground : HalfSpaceContactEnvironment
  step : FullMultibodyPlantStep
  primitivePlant : FullPlantPrimitivePhysics
  fullPhysics : FullPhysicsResult
  stateCoordinateNames : Array String
  graph : SkeletonGraph
  executionStatus : AtlasExecutionStatus
  deriving Repr, Inhabited

def buildEndToEnd? : Except String AtlasResult := do
  validateBuildTarget?
  runDynamicsBoundary.validate?
  let primitivePhysics ← solveAtlasPrimitivePhysics? atlasPlantStep
  pure {
    references := drakeReferences
    buildTarget := buildTarget
    runDynamics := runDynamicsBoundary
    config := plantConfig
    model := atlasModel
    ground := ground
    step := atlasPlantStep
    primitivePlant := primitivePhysics.primitivePlant
    fullPhysics := primitivePhysics.fullPhysics
    stateCoordinateNames := stateCoordinateNames
    graph := atlasGraph primitivePhysics.fullPhysics
    executionStatus := .primitivePhysicsSolved
  }

end Tyr.EventSkeleton.Examples.Atlas
