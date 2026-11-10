import Tyr.EventSkeleton.Manipulator

/-!
# Drake-Style Planar Gripper / Brick Example

This ports the planning-side physics of `../drake/examples/planar_gripper`.
The Drake example has three planar two-link fingers manipulating a brick.  The
concrete slice here covers the `GripperBrickHelper` metadata, fingertip
kinematics in the brick frame, planar fingertip/face contact geometry, linear
friction-cone checks, and the nonlinear brick static-equilibrium residual.

The important boundary is the reusable primitive: static equilibrium is
expressed as planar body-frame contact wrenches plus gravity, not as a dense
MultibodyPlant callback.  The provider supplies the finger/brick geometry and
assigned contacts.
-/

namespace Tyr.EventSkeleton.Examples.PlanarGripper

open Tyr.EventSkeleton

private def pi : Float := 3.14159265358979323846

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/planar_gripper/BUILD.bazel"
      concept := "declares models_filegroup, install_data, libraries, trajectory publisher, simulation executable, googletests, and lint tests"
    },
    {
      path := "../drake/examples/planar_gripper/README.md"
      concept := "documents the planar-gripper example boundary; Drake currently leaves detailed usage notes as a TODO in the simulation source"
    },
    {
      path := "../drake/examples/planar_gripper/gripper_brick.h"
      concept := "declares the three-finger gripper/brick helper, position indices, fingertip sphere radius, brick size, and fingertip-in-link2 offset"
    },
    {
      path := "../drake/examples/planar_gripper/gripper_brick.cc"
      concept := "loads the gripper and brick SDFs, welds the three finger bases, and extracts fingertip/brick collision geometry"
    },
    {
      path := "../drake/examples/planar_gripper/brick_static_equilibrium_constraint.cc"
      concept := "evaluates planar brick static equilibrium: body-frame gravity plus contact forces and x-axis contact torque"
    },
    {
      path := "../drake/examples/planar_gripper/brick_static_equilibrium_constraint.h"
      concept := "declares the 3-output nonlinear constraint over plant positions plus two planar contact-force variables per assigned contact"
    },
    {
      path := "../drake/examples/planar_gripper/test/brick_static_equilibrium_constraint_test.cc"
      concept := "checks constraint dimension, zero bounds, residual formula, and autodiff/numerical-gradient parity"
    },
    {
      path := "../drake/examples/planar_gripper/gripper_brick_planning_constraint_helper.cc"
      concept := "adds face-dependent planar friction-cone constraints and fingertip contact/no-sliding constraints"
    },
    {
      path := "../drake/examples/planar_gripper/gripper_brick_planning_constraint_helper.h"
      concept := "declares friction-cone, fingertip-in-face-contact, and rolling/no-sliding planning constraint helpers"
    },
    {
      path := "../drake/examples/planar_gripper/test/gripper_brick_planning_constraint_helper_test.cc"
      concept := "checks planar friction cones, shrunk face contact regions, and no-sliding rolling kinematics"
    },
    {
      path := "../drake/examples/planar_gripper/planar_gripper_common.cc"
      concept := "welds the three finger base frames on a circle of radius 0.201 at angles pi/3, -pi/3, and pi"
    },
    {
      path := "../drake/examples/planar_gripper/planar_gripper_common.h"
      concept := "declares kNumFingers, kNumJoints, WeldGripperFrames, ParseKeyframes, ReorderKeyframesForPlant, and X_WGripper"
    },
    {
      path := "../drake/examples/planar_gripper/test/planar_gripper_common_test.cc"
      concept := "checks keyframe reordering against plant velocity order and failure cases for bad row maps or plant dimensions"
    },
    {
      path := "../drake/examples/planar_gripper/test/gripper_brick_test.cc"
      concept := "checks GripperBrickHelper geometry constants and link-2 orientation against plant frame transforms"
    },
    {
      path := "../drake/examples/planar_gripper/planar_gripper_lcm.cc"
      concept := "encodes/decodes lcmt_planar_gripper_command and status messages with state, torque, and fingertip-force ports"
    },
    {
      path := "../drake/examples/planar_gripper/planar_gripper_lcm.h"
      concept := "declares planar gripper command/status encoder-decoder systems, port layouts, and forced publish/update boundaries"
    },
    {
      path := "../drake/examples/planar_gripper/run_planar_gripper_trajectory_publisher.cc"
      concept := "loads postures.txt, reorders keyframes for the control plant, builds a CubicShapePreserving trajectory source, and force-publishes commands on status-message times"
    },
    {
      path := "../drake/examples/planar_gripper/planar_gripper_simulation.cc"
      concept := "builds the full gripper+brick simulation plant, selects position or torque control, adds the floor, status publisher, force sensor, and Drake visualizer"
    },
    {
      path := "../drake/examples/planar_gripper/planar_manipuland_lcm.cc"
      concept := "encodes and decodes lcmt_planar_manipuland_status pose/velocity messages with a 10 ms discrete update"
    },
    {
      path := "../drake/examples/planar_gripper/planar_manipuland_lcm.h"
      concept := "declares PlanarManipulandStatusEncoder/Decoder ports and kPlanarManipulandStatusPeriod"
    },
    {
      path := "../drake/examples/planar_gripper/test/planar_manipuland_lcm_test.cc"
      concept := "checks planar manipuland status encoder/decoder passthrough under forced discrete update"
    },
    {
      path := "../drake/examples/planar_gripper/test/planar_gripper_lcm_test.cc"
      concept := "checks planar gripper command and status encoder/decoder passthrough for state, torque, and fingertip force fields"
    },
    {
      path := "../drake/examples/planar_gripper/postures.txt"
      concept := "contains the vertical-case static-equilibrium posture keyframes played back by the trajectory publisher"
    },
    {
      path := "../drake/examples/planar_gripper/planar_gripper.sdf"
      concept := "defines the controlled three-finger gripper plant used to extract joint velocity ordering"
    },
    {
      path := "../drake/examples/planar_gripper/planar_gripper.xacro"
      concept := "xacro source for the three-finger planar gripper SDF and joint/link naming convention"
    },
    {
      path := "../drake/examples/planar_gripper/planar_brick.sdf"
      concept := "defines the brick mass, y/z/x planar joints, and 0.07 x 0.1 x 0.1 box collision geometry"
    }
  ]

def planarGripperExampleRoot : String :=
  "../drake/examples/planar_gripper"

inductive PlanarGripperExampleAssetKind where
  | metadata
  | source
  | header
  | model
  | xacro
  | data
  | test
  deriving Repr, BEq, Inhabited

inductive PlanarGripperExampleAssetFormat where
  | bazel
  | markdown
  | cpp
  | header
  | sdf
  | xacro
  | txt
  deriving Repr, BEq, Inhabited

namespace PlanarGripperExampleAssetFormat

def matchesPath (format : PlanarGripperExampleAssetFormat)
    (path : String) : Bool :=
  match format with
  | .bazel => path == "BUILD.bazel"
  | .markdown => path.endsWith ".md"
  | .cpp => path.endsWith ".cc"
  | .header => path.endsWith ".h"
  | .sdf => path.endsWith ".sdf"
  | .xacro => path.endsWith ".xacro"
  | .txt => path.endsWith ".txt"

end PlanarGripperExampleAssetFormat

/--
File manifest for Drake's `examples/planar_gripper` tree.

The manifest is the model/build/data provider boundary.  It records which local
files feed model installation, trajectory publishing, simulation, and Drake's
regression tests.  Dynamics still lower into `FullPhysicsPrimitives`; the
catalog only keeps the Drake source closure visible and checkable.
-/
structure PlanarGripperExampleAsset where
  relativePath : String
  format : PlanarGripperExampleAssetFormat
  kind : PlanarGripperExampleAssetKind
  component : String
  feedsModelsFilegroup : Bool := false
  feedsSimulation : Bool := false
  feedsTrajectoryPublisher : Bool := false
  localDependencies : Array String := #[]
  concept : String := ""
  deriving Repr, Inhabited

namespace PlanarGripperExampleAsset

def fullPath (asset : PlanarGripperExampleAsset) : String :=
  planarGripperExampleRoot ++ "/" ++ asset.relativePath

def validate? (asset : PlanarGripperExampleAsset) : Except String Unit := do
  if asset.relativePath.isEmpty then
    .error "PlanarGripper asset path cannot be empty"
  if !asset.format.matchesPath asset.relativePath then
    .error s!"PlanarGripper asset {asset.relativePath}: format does not match path"
  if asset.component.isEmpty then
    .error s!"PlanarGripper asset {asset.relativePath}: component cannot be empty"
  if asset.concept.isEmpty then
    .error s!"PlanarGripper asset {asset.relativePath}: concept cannot be empty"
  if asset.feedsModelsFilegroup &&
      asset.format != .sdf && asset.format != .xacro then
    .error s!"PlanarGripper asset {asset.relativePath}: only SDF/Xacro assets should feed the models filegroup"
  match asset.kind with
  | .header =>
      if asset.format != .header then
        .error s!"PlanarGripper asset {asset.relativePath}: header kind must use header format"
  | .model =>
      if asset.format != .sdf then
        .error s!"PlanarGripper asset {asset.relativePath}: model kind must use SDF format"
  | .xacro =>
      if asset.format != .xacro then
        .error s!"PlanarGripper asset {asset.relativePath}: xacro kind must use xacro format"
  | .data =>
      if asset.format != .txt then
        .error s!"PlanarGripper asset {asset.relativePath}: data kind must use txt format"
  | .test =>
      if asset.format != .cpp then
        .error s!"PlanarGripper asset {asset.relativePath}: test kind must use C++ format"
  | _ => pure ()
  for dep in asset.localDependencies do
    if dep.isEmpty then
      .error s!"PlanarGripper asset {asset.relativePath}: local dependency cannot be empty"
    if dep == asset.relativePath then
      .error s!"PlanarGripper asset {asset.relativePath}: cannot depend on itself"

end PlanarGripperExampleAsset

def planarGripperExampleAssets : Array PlanarGripperExampleAsset :=
  #[
    {
      relativePath := "BUILD.bazel"
      format := .bazel
      kind := .metadata
      component := "build"
      localDependencies := #[
        "planar_gripper_common.cc",
        "planar_gripper_common.h",
        "planar_gripper_lcm.cc",
        "planar_gripper_lcm.h",
        "planar_manipuland_lcm.cc",
        "planar_manipuland_lcm.h",
        "brick_static_equilibrium_constraint.cc",
        "brick_static_equilibrium_constraint.h",
        "gripper_brick.cc",
        "gripper_brick.h",
        "gripper_brick_planning_constraint_helper.cc",
        "gripper_brick_planning_constraint_helper.h",
        "run_planar_gripper_trajectory_publisher.cc",
        "planar_gripper_simulation.cc",
        "postures.txt",
        "planar_gripper.sdf",
        "planar_gripper.xacro",
        "planar_brick.sdf",
        "test/planar_gripper_common_test.cc",
        "test/gripper_brick_test.cc",
        "test/brick_static_equilibrium_constraint_test.cc",
        "test/gripper_brick_planning_constraint_helper_test.cc",
        "test/planar_gripper_lcm_test.cc",
        "test/planar_manipuland_lcm_test.cc"
      ]
      concept := "Bazel models_filegroup, install_data, libraries, binaries, simulation smoke test, googletests, and lint tests"
    },
    {
      relativePath := "README.md"
      format := .markdown
      kind := .metadata
      component := "docs"
      concept := "top-level example documentation placeholder referenced by planar_gripper_simulation.cc TODO"
    },
    {
      relativePath := "planar_gripper_common.cc"
      format := .cpp
      kind := .source
      component := "common"
      localDependencies := #["planar_gripper_common.h"]
      concept := "welds finger bases and implements keyframe parsing/reordering helpers"
    },
    {
      relativePath := "planar_gripper_common.h"
      format := .header
      kind := .header
      component := "common"
      concept := "declares planar gripper constants, frame welding, keyframe parsing, and row reordering"
    },
    {
      relativePath := "planar_gripper_lcm.cc"
      format := .cpp
      kind := .source
      component := "lcm"
      localDependencies := #["planar_gripper_lcm.h"]
      concept := "encodes and decodes planar gripper command/status LCM messages"
    },
    {
      relativePath := "planar_gripper_lcm.h"
      format := .header
      kind := .header
      component := "lcm"
      concept := "declares planar gripper command and status encoder/decoder systems"
    },
    {
      relativePath := "planar_manipuland_lcm.cc"
      format := .cpp
      kind := .source
      component := "lcm"
      localDependencies := #["planar_manipuland_lcm.h"]
      concept := "encodes and decodes manipuland pose/velocity LCM messages"
    },
    {
      relativePath := "planar_manipuland_lcm.h"
      format := .header
      kind := .header
      component := "lcm"
      concept := "declares manipuland status encoder/decoder and update period"
    },
    {
      relativePath := "brick_static_equilibrium_constraint.cc"
      format := .cpp
      kind := .source
      component := "planning"
      localDependencies := #[
        "brick_static_equilibrium_constraint.h",
        "gripper_brick_planning_constraint_helper.h"
      ]
      concept := "evaluates static brick equilibrium residual and nonlinear constraint bounds"
    },
    {
      relativePath := "brick_static_equilibrium_constraint.h"
      format := .header
      kind := .header
      component := "planning"
      localDependencies := #["gripper_brick.h"]
      concept := "declares the brick static-equilibrium nonlinear constraint"
    },
    {
      relativePath := "gripper_brick.cc"
      format := .cpp
      kind := .source
      component := "planning"
      localDependencies := #[
        "gripper_brick.h",
        "planar_gripper_common.h",
        "planar_gripper.sdf",
        "planar_brick.sdf"
      ]
      concept := "loads gripper/brick models and extracts helper geometry and frame transforms"
    },
    {
      relativePath := "gripper_brick.h"
      format := .header
      kind := .header
      component := "planning"
      concept := "declares helper accessors for gripper/brick plant geometry"
    },
    {
      relativePath := "gripper_brick_planning_constraint_helper.cc"
      format := .cpp
      kind := .source
      component := "planning"
      localDependencies := #["gripper_brick_planning_constraint_helper.h"]
      concept := "implements friction-cone, fingertip-contact, and no-sliding helper constraints"
    },
    {
      relativePath := "gripper_brick_planning_constraint_helper.h"
      format := .header
      kind := .header
      component := "planning"
      localDependencies := #["gripper_brick.h"]
      concept := "declares planar friction, face-contact, and rolling/no-sliding constraint helpers"
    },
    {
      relativePath := "run_planar_gripper_trajectory_publisher.cc"
      format := .cpp
      kind := .source
      component := "trajectory_publisher"
      feedsTrajectoryPublisher := true
      localDependencies := #[
        "planar_gripper_common.h",
        "planar_gripper_lcm.h",
        "planar_gripper.sdf",
        "postures.txt"
      ]
      concept := "loads keyframes, builds the trajectory source, and publishes commands on status-message times"
    },
    {
      relativePath := "planar_gripper_simulation.cc"
      format := .cpp
      kind := .source
      component := "simulation"
      feedsSimulation := true
      localDependencies := #[
        "planar_gripper_common.h",
        "planar_gripper_lcm.h",
        "planar_gripper.sdf",
        "planar_brick.sdf",
        "postures.txt"
      ]
      concept := "builds full gripper+brick simulation plant, controller boundary, floor contact, and publishers"
    },
    {
      relativePath := "postures.txt"
      format := .txt
      kind := .data
      component := "trajectory_publisher"
      feedsSimulation := true
      feedsTrajectoryPublisher := true
      concept := "vertical-case static-equilibrium posture keyframes consumed by publisher and simulation"
    },
    {
      relativePath := "planar_gripper.sdf"
      format := .sdf
      kind := .model
      component := "model"
      feedsModelsFilegroup := true
      feedsSimulation := true
      feedsTrajectoryPublisher := true
      localDependencies := #["planar_gripper.xacro"]
      concept := "installed three-finger gripper model consumed by parser and trajectory publisher"
    },
    {
      relativePath := "planar_gripper.xacro"
      format := .xacro
      kind := .xacro
      component := "model"
      feedsModelsFilegroup := true
      concept := "source xacro for planar_gripper.sdf and finger/link naming convention"
    },
    {
      relativePath := "planar_brick.sdf"
      format := .sdf
      kind := .model
      component := "model"
      feedsModelsFilegroup := true
      feedsSimulation := true
      concept := "brick inertial, planar joint, box collision, and corner-sphere contact model"
    },
    {
      relativePath := "test/planar_gripper_common_test.cc"
      format := .cpp
      kind := .test
      component := "test"
      localDependencies := #[
        "planar_gripper_common.h",
        "planar_gripper.sdf",
        "planar_brick.sdf"
      ]
      concept := "regresses keyframe reordering and parser/plant dimension failures"
    },
    {
      relativePath := "test/gripper_brick_test.cc"
      format := .cpp
      kind := .test
      component := "test"
      localDependencies := #["gripper_brick.h"]
      concept := "regresses helper geometry constants and link-2 orientation"
    },
    {
      relativePath := "test/brick_static_equilibrium_constraint_test.cc"
      format := .cpp
      kind := .test
      component := "test"
      localDependencies := #["brick_static_equilibrium_constraint.h"]
      concept := "regresses static-equilibrium residuals and numerical-gradient parity"
    },
    {
      relativePath := "test/gripper_brick_planning_constraint_helper_test.cc"
      format := .cpp
      kind := .test
      component := "test"
      localDependencies := #["gripper_brick_planning_constraint_helper.h"]
      concept := "regresses friction cones, contact regions, and no-sliding constraints"
    },
    {
      relativePath := "test/planar_gripper_lcm_test.cc"
      format := .cpp
      kind := .test
      component := "test"
      localDependencies := #["planar_gripper_lcm.h"]
      concept := "regresses planar gripper command/status LCM passthrough"
    },
    {
      relativePath := "test/planar_manipuland_lcm_test.cc"
      format := .cpp
      kind := .test
      component := "test"
      localDependencies := #["planar_manipuland_lcm.h"]
      concept := "regresses manipuland status LCM passthrough"
    }
  ]

private def hasDuplicatePlanarGripperAssetPath
    (assets : Array PlanarGripperExampleAsset) : Bool := Id.run do
  for i in [:assets.size] do
    for j in [:(assets.size - i - 1)] do
      let k := i + j + 1
      if assets[i]!.relativePath == assets[k]!.relativePath then
        return true
  return false

def planarGripperExampleAssetPaths : Array String :=
  planarGripperExampleAssets.map (fun asset => asset.relativePath)

def planarGripperModelAssets : Array PlanarGripperExampleAsset :=
  planarGripperExampleAssets.filter
    (fun asset => asset.kind == .model || asset.kind == .xacro)

def planarGripperTestAssets : Array PlanarGripperExampleAsset :=
  planarGripperExampleAssets.filter (fun asset => asset.kind == .test)

def planarGripperSimulationAssets : Array PlanarGripperExampleAsset :=
  planarGripperExampleAssets.filter (fun asset => asset.feedsSimulation)

def planarGripperTrajectoryAssets : Array PlanarGripperExampleAsset :=
  planarGripperExampleAssets.filter (fun asset => asset.feedsTrajectoryPublisher)

def findPlanarGripperExampleAsset? (relativePath : String) :
    Option PlanarGripperExampleAsset :=
  planarGripperExampleAssets.find? (fun asset => asset.relativePath == relativePath)

def requiredPlanarGripperExampleAssetPaths : Array String :=
  #[
    "BUILD.bazel",
    "README.md",
    "brick_static_equilibrium_constraint.cc",
    "brick_static_equilibrium_constraint.h",
    "gripper_brick.cc",
    "gripper_brick.h",
    "gripper_brick_planning_constraint_helper.cc",
    "gripper_brick_planning_constraint_helper.h",
    "planar_brick.sdf",
    "planar_gripper.sdf",
    "planar_gripper.xacro",
    "planar_gripper_common.cc",
    "planar_gripper_common.h",
    "planar_gripper_lcm.cc",
    "planar_gripper_lcm.h",
    "planar_gripper_simulation.cc",
    "planar_manipuland_lcm.cc",
    "planar_manipuland_lcm.h",
    "postures.txt",
    "run_planar_gripper_trajectory_publisher.cc",
    "test/brick_static_equilibrium_constraint_test.cc",
    "test/gripper_brick_planning_constraint_helper_test.cc",
    "test/gripper_brick_test.cc",
    "test/planar_gripper_common_test.cc",
    "test/planar_gripper_lcm_test.cc",
    "test/planar_manipuland_lcm_test.cc"
  ]

def validatePlanarGripperExampleAssetCatalog? : Except String Unit := do
  if planarGripperExampleAssets.size != requiredPlanarGripperExampleAssetPaths.size then
    .error s!"PlanarGripper asset catalog size {planarGripperExampleAssets.size} != expected {requiredPlanarGripperExampleAssetPaths.size}"
  if hasDuplicatePlanarGripperAssetPath planarGripperExampleAssets then
    .error "PlanarGripper asset catalog contains duplicate relative paths"
  for asset in planarGripperExampleAssets do
    asset.validate?
  for path in requiredPlanarGripperExampleAssetPaths do
    if !(planarGripperExampleAssetPaths.contains path) then
      .error s!"PlanarGripper asset catalog missing {path}"
  for asset in planarGripperExampleAssets do
    for dep in asset.localDependencies do
      if !(planarGripperExampleAssetPaths.contains dep) then
        .error s!"PlanarGripper asset {asset.relativePath}: missing local dependency {dep}"
  if planarGripperModelAssets.size != 3 then
    .error s!"PlanarGripper catalog should expose 3 model/xacro assets, got {planarGripperModelAssets.size}"
  if planarGripperTestAssets.size != 6 then
    .error s!"PlanarGripper catalog should expose 6 googletest assets, got {planarGripperTestAssets.size}"
  if planarGripperSimulationAssets.size != 4 then
    .error s!"PlanarGripper catalog should expose 4 simulation assets, got {planarGripperSimulationAssets.size}"
  if planarGripperTrajectoryAssets.size != 3 then
    .error s!"PlanarGripper catalog should expose 3 trajectory-publisher assets, got {planarGripperTrajectoryAssets.size}"

inductive Finger where
  | finger1
  | finger2
  | finger3
  deriving Repr, BEq, Inhabited

namespace Finger

def label : Finger → String
  | .finger1 => "finger 1"
  | .finger2 => "finger 2"
  | .finger3 => "finger 3"

def ordinal : Finger → Nat
  | .finger1 => 1
  | .finger2 => 2
  | .finger3 => 3

end Finger

abbrev BrickFace := PlanarBoxFace

structure PlanarGripperParams where
  numPositions : Nat := 9
  numFingers : Nat := 3
  jointsPerFinger : Nat := 2
  gripperOriginToBaseDistance : Float := 0.201
  link1Length : Float := 0.085
  pL2Fingertip : PlanarVec2 := { y := 0.0, z := -0.0713 }
  fingerTipRadius : Float := 0.015
  brickMass : Float := 0.028
  brickSize : PlanarVec2 := { y := 0.1, z := 0.1 }
  brickThicknessX : Float := 0.07
  gravity : Float := 9.81
  staticFriction : Float := 0.6
  deriving Repr, Inhabited

def params : PlanarGripperParams := {}

structure PlanarGripperState where
  finger1Base : Float := 0.0
  finger1Mid : Float := 0.0
  finger2Base : Float := 0.0
  finger2Mid : Float := 0.0
  finger3Base : Float := 0.0
  finger3Mid : Float := 0.0
  brickY : Float := 0.0
  brickZ : Float := 0.0
  brickTheta : Float := 0.0
  deriving Repr, Inhabited

namespace PlanarGripperState

def asArray (q : PlanarGripperState) : Array Float :=
  #[
    q.finger1Base, q.finger1Mid,
    q.finger2Base, q.finger2Mid,
    q.finger3Base, q.finger3Mid,
    q.brickY, q.brickZ, q.brickTheta
  ]

def fromArray? (xs : Array Float) : Except String PlanarGripperState := do
  if xs.size != 9 then
    .error s!"planar gripper state expected 9 positions, got {xs.size}"
  pure {
    finger1Base := xs[0]!
    finger1Mid := xs[1]!
    finger2Base := xs[2]!
    finger2Mid := xs[3]!
    finger3Base := xs[4]!
    finger3Mid := xs[5]!
    brickY := xs[6]!
    brickZ := xs[7]!
    brickTheta := xs[8]!
  }

end PlanarGripperState

def drakeTestState : PlanarGripperState :=
  {
    finger1Base := 0.1
    finger1Mid := 0.3
    finger2Base := 0.3
    finger2Mid := -0.4
    finger3Base := -1.2
    finger3Mid := 0.5
    brickY := 1.3
    brickZ := -0.2
    brickTheta := 1.8
  }

def fingerBasePositionIndex : Finger → Nat
  | .finger1 => 0
  | .finger2 => 2
  | .finger3 => 4

def fingerMidPositionIndex : Finger → Nat
  | .finger1 => 1
  | .finger2 => 3
  | .finger3 => 5

def brickTranslateYPositionIndex : Nat := 6
def brickTranslateZPositionIndex : Nat := 7
def brickRevoluteXPositionIndex : Nat := 8

def planarGripperModelUrl : String :=
  "package://drake/examples/planar_gripper/planar_gripper.sdf"

def planarBrickModelUrl : String :=
  "package://drake/examples/planar_gripper/planar_brick.sdf"

def postureKeyframePath : String :=
  "drake/examples/planar_gripper/postures.txt"

def lcmStatusChannel : String := "PLANAR_GRIPPER_STATUS"
def lcmCommandChannel : String := "PLANAR_GRIPPER_COMMAND"
def gripperLcmStatusPeriod : Float := 0.010

def postureHeader : Array String :=
  #[
    "finger1_BaseJoint",
    "finger3_BaseJoint",
    "finger2_BaseJoint",
    "brick_translate_y_joint",
    "finger1_MidJoint",
    "finger3_MidJoint",
    "finger2_MidJoint",
    "brick_translate_z_joint",
    "brick_revolute_x_joint"
  ]

def postureKeyframeCount : Nat := 41

def fingerJointParseOrder : Array String :=
  #[
    "finger1_BaseJoint",
    "finger2_BaseJoint",
    "finger3_BaseJoint",
    "finger1_MidJoint",
    "finger2_MidJoint",
    "finger3_MidJoint"
  ]

def controlPlantJointOrder : Array String :=
  #[
    "finger1_BaseJoint",
    "finger1_MidJoint",
    "finger2_BaseJoint",
    "finger2_MidJoint",
    "finger3_BaseJoint",
    "finger3_MidJoint"
  ]

structure JointRowIndex where
  jointName : String := ""
  rowIndex : Nat := 0
  deriving Repr, BEq, Inhabited

structure ReorderedKeyframes where
  keyframes : Array (Array Float)
  rowIndexMap : Array JointRowIndex
  deriving Repr, Inhabited

def drakeCommonTestJointRowIndexMap : Array JointRowIndex :=
  #[
    { jointName := "finger1_BaseJoint", rowIndex := 3 },
    { jointName := "finger2_BaseJoint", rowIndex := 2 },
    { jointName := "finger3_BaseJoint", rowIndex := 4 },
    { jointName := "finger1_MidJoint", rowIndex := 0 },
    { jointName := "finger3_MidJoint", rowIndex := 5 },
    { jointName := "finger2_MidJoint", rowIndex := 1 }
  ]

def constantKeyframeRows (numRows numCols : Nat) : Array (Array Float) :=
  Id.run do
    let mut rows : Array (Array Float) := #[]
    for i in [:numRows] do
      rows := rows.push (Array.replicate numCols i.toFloat)
    return rows

def drakeCommonTestKeyframes : Array (Array Float) :=
  constantKeyframeRows (params.numFingers * params.jointsPerFinger) 4

def rowIndexFor? (rowMap : Array JointRowIndex) (jointName : String) :
    Except String Nat := do
  for entry in rowMap do
    if entry.jointName == jointName then
      return entry.rowIndex
  .error s!"row map does not contain joint {jointName}"

def reorderKeyframesForControlPlant?
    (keyframes : Array (Array Float))
    (rowMap : Array JointRowIndex)
    (plantNumPositions : Nat := params.numFingers * params.jointsPerFinger) :
    Except String ReorderedKeyframes := do
  let numJoints := params.numFingers * params.jointsPerFinger
  if keyframes.size != rowMap.size then
    .error s!"keyframe rows {keyframes.size} must match row-map size {rowMap.size}"
  if keyframes.size != numJoints then
    .error s!"keyframe rows {keyframes.size} must match planar gripper joints {numJoints}"
  if plantNumPositions != numJoints then
    .error s!"control plant positions {plantNumPositions} must match planar gripper joints {numJoints}"
  let mut rows : Array (Array Float) := #[]
  let mut newMap : Array JointRowIndex := #[]
  for i in [:controlPlantJointOrder.size] do
    let jointName := controlPlantJointOrder[i]!
    let oldIndex ← rowIndexFor? rowMap jointName
    if oldIndex >= keyframes.size then
      .error s!"row index {oldIndex} for joint {jointName} exceeds keyframe rows {keyframes.size}"
    rows := rows.push keyframes[oldIndex]!
    newMap := newMap.push { jointName := jointName, rowIndex := i }
  pure { keyframes := rows, rowIndexMap := newMap }

def rawFirstPosture : Array Float :=
  #[
    -0.277611, 0.947974, 0.281503, 0.00731802, -0.0613293,
    -1.21373, 0.201025, -0.0197339, -0.454859
  ]

def rawLastPosture : Array Float :=
  #[
    -0.531127, 0.181992, -0.507908, -0.0152193, 0.259681,
    -0.628283, 0.459655, 0.00143686, 2.28755
  ]

private def postureHeaderIndex? (name : String) : Option Nat := Id.run do
  for i in [:postureHeader.size] do
    if postureHeader[i]! == name then
      return some i
  return none

private def postureValue? (row : Array Float) (name : String) :
    Except String Float := do
  let index ←
    match postureHeaderIndex? name with
    | some index => pure index
    | none => .error s!"posture header does not contain {name}"
  if index >= row.size then
    .error s!"posture row has width {row.size}, cannot read {name} at column {index}"
  let value := row[index]!
  if !value.isFinite then
    .error s!"posture value {name} must be finite, got {value}"
  pure value

def extractPostureOrder? (row : Array Float) (order : Array String) :
    Except String (Array Float) := do
  if postureHeader.size != 9 then
    .error s!"postures header should have 9 entries, got {postureHeader.size}"
  if row.size != postureHeader.size then
    .error s!"posture row width {row.size} != header width {postureHeader.size}"
  let mut out : Array Float := #[]
  for name in order do
    out := out.push (← postureValue? row name)
  pure out

def firstParsedFingerKeyframe? : Except String (Array Float) :=
  extractPostureOrder? rawFirstPosture fingerJointParseOrder

def firstControlPlantKeyframe? : Except String (Array Float) :=
  extractPostureOrder? rawFirstPosture controlPlantJointOrder

def firstBrickInitialPose? : Except String (Array Float) :=
  extractPostureOrder? rawFirstPosture
    #["brick_translate_y_joint", "brick_translate_z_joint", "brick_revolute_x_joint"]

def fingerJointAngles (q : PlanarGripperState) : Finger → Float × Float
  | .finger1 => (q.finger1Base, q.finger1Mid)
  | .finger2 => (q.finger2Base, q.finger2Mid)
  | .finger3 => (q.finger3Base, q.finger3Mid)

def fingerBaseAngle : Finger → Float
  | .finger1 => pi / 3.0
  | .finger2 => -pi / 3.0
  | .finger3 => pi

def fingerLink2Orientation (finger : Finger) (q : PlanarGripperState) : Float :=
  let angles := fingerJointAngles q finger
  fingerBaseAngle finger + angles.1 + angles.2

def rotateXPlanar (theta : Float) (v : PlanarVec2) : PlanarVec2 :=
  {
    y := Float.cos theta * v.y - Float.sin theta * v.z
    z := Float.sin theta * v.y + Float.cos theta * v.z
  }

def fingerBasePosition
    (p : PlanarGripperParams := params)
    (finger : Finger) : PlanarVec2 :=
  rotateXPlanar (fingerBaseAngle finger)
    { y := 0.0, z := p.gripperOriginToBaseDistance }

def fingerTipWorld
    (p : PlanarGripperParams := params)
    (q : PlanarGripperState)
    (finger : Finger) : PlanarVec2 :=
  let angles := fingerJointAngles q finger
  let baseTheta := fingerBaseAngle finger + angles.1
  let link2Theta := baseTheta + angles.2
  let base := fingerBasePosition p finger
  let link1 := rotateXPlanar baseTheta { y := 0.0, z := -p.link1Length }
  let link2 := rotateXPlanar link2Theta p.pL2Fingertip
  base.add link1 |>.add link2

def worldToBrick (q : PlanarGripperState) (point_W : PlanarVec2) :
    PlanarVec2 :=
  let d := point_W.sub { y := q.brickY, z := q.brickZ }
  {
    y := Float.cos q.brickTheta * d.y + Float.sin q.brickTheta * d.z
    z := -Float.sin q.brickTheta * d.y + Float.cos q.brickTheta * d.z
  }

def fingerTipInBrickFrame
    (p : PlanarGripperParams := params)
    (q : PlanarGripperState)
    (finger : Finger) : PlanarVec2 :=
  worldToBrick q (fingerTipWorld p q finger)

def faceCoordinate (face : BrickFace) (point : PlanarVec2) : Float :=
  match face with
  | .posY | .negY => point.y
  | .posZ | .negZ => point.z

def tangentCoordinate (face : BrickFace) (point : PlanarVec2) : Float :=
  match face with
  | .posY | .negY => point.z
  | .posZ | .negZ => point.y

def faceTargetContactCoordinate
    (p : PlanarGripperParams := params)
    (face : BrickFace)
    (depth : Float) : Float :=
  match face with
  | .posY => p.brickSize.y / 2.0 - depth
  | .negY => -p.brickSize.y / 2.0 + depth
  | .posZ => p.brickSize.z / 2.0 - depth
  | .negZ => -p.brickSize.z / 2.0 + depth

def faceTangentHalfExtent
    (p : PlanarGripperParams := params)
    (face : BrickFace)
    (faceShrinkFactor : Float) : Float :=
  match face with
  | .posY | .negY => faceShrinkFactor * p.brickSize.z / 2.0
  | .posZ | .negZ => faceShrinkFactor * p.brickSize.y / 2.0

def fingerTipContactPointFromTip
    (p : PlanarGripperParams := params)
    (face : BrickFace)
    (tip_B : PlanarVec2) : PlanarVec2 :=
  face.contactPointFromFingerTip p.fingerTipRadius tip_B

def fingerTipContactResidualFromTip
    (p : PlanarGripperParams := params)
    (face : BrickFace)
    (depth : Float)
    (tip_B : PlanarVec2) : Float :=
  let contactPoint := fingerTipContactPointFromTip p face tip_B
  faceCoordinate face contactPoint - faceTargetContactCoordinate p face depth

def fingerTipInShrunkFaceRegionFromTip
    (p : PlanarGripperParams := params)
    (face : BrickFace)
    (faceShrinkFactor depth : Float)
    (tip_B : PlanarVec2)
    (tol : Float := 1.0e-8) : Bool :=
  let contactPoint := fingerTipContactPointFromTip p face tip_B
  Float.abs (fingerTipContactResidualFromTip p face depth tip_B) <= tol &&
    Float.abs (tangentCoordinate face contactPoint) <=
      faceTangentHalfExtent p face faceShrinkFactor + tol

def fingerTipContactResidual
    (p : PlanarGripperParams := params)
    (q : PlanarGripperState)
    (finger : Finger)
    (face : BrickFace)
    (depth : Float) : Float :=
  fingerTipContactResidualFromTip p face depth
    (fingerTipInBrickFrame p q finger)

def noSlidingResidualNegZ
    (p : PlanarGripperParams := params)
    (thetaFrom thetaTo : Float)
    (tipFrom_B tipTo_B : PlanarVec2) : Float :=
  (tipTo_B.y - tipFrom_B.y) + p.fingerTipRadius * (thetaTo - thetaFrom)

structure FingerFaceContact where
  finger : Finger
  face : BrickFace
  force_B : PlanarVec2 := {}
  deriving Repr, Inhabited

namespace FingerFaceContact

def label (contact : FingerFaceContact) : String :=
  s!"{contact.finger.label}:{contact.face.label}"

end FingerFaceContact

def contactPointInBrickFrame
    (p : PlanarGripperParams := params)
    (q : PlanarGripperState)
    (contact : FingerFaceContact) : PlanarVec2 :=
  contact.face.contactPointFromFingerTip p.fingerTipRadius
    (fingerTipInBrickFrame p q contact.finger)

def contactWrench
    (p : PlanarGripperParams := params)
    (q : PlanarGripperState)
    (contact : FingerFaceContact) : PlanarContactWrench :=
  {
    point_B := contactPointInBrickFrame p q contact
    force_B := contact.force_B
    label := contact.label
  }

def staticEquilibrium
    (p : PlanarGripperParams := params)
    (q : PlanarGripperState)
    (contacts : Array FingerFaceContact) : PlanarStaticEquilibrium :=
  {
    mass := p.brickMass
    gravity := p.gravity
    theta := q.brickTheta
    contacts := contacts.map (contactWrench p q)
    label := "planar-gripper-brick-static-equilibrium"
  }

def staticEquilibriumResidual?
    (p : PlanarGripperParams := params)
    (q : PlanarGripperState)
    (contacts : Array FingerFaceContact) : Except String (Array Float) :=
  (staticEquilibrium p q contacts).residualArray?

def brickStaticEquilibriumGraph : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 7000, kind := .state .boundary, label := "q" }
    |>.addVertex { id := 7001, kind := .state .interior, label := "contact forces" }
    |>.addVertex { id := 7002, kind := .opaque, label := "static equilibrium residual" }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[7002]
      reads := #[7000, 7001]
      writes := #[7000]
      label := "planar-gripper-static-equilibrium-constraint"
    }

structure BrickStaticEquilibriumConstraintBoundary where
  contacts : Array FingerFaceContact
  numOutputs : Nat := 3
  numVars : Nat
  lowerBound : Array Float := #[0.0, 0.0, 0.0]
  upperBound : Array Float := #[0.0, 0.0, 0.0]
  residual : Array Float
  graph : SkeletonGraph
  deriving Repr, Inhabited

namespace BrickStaticEquilibriumConstraintBoundary

def validate? (boundary : BrickStaticEquilibriumConstraintBoundary) :
    Except String Unit := do
  if boundary.numOutputs != 3 then
    .error s!"brick static equilibrium constraint outputs {boundary.numOutputs} != 3"
  if boundary.numVars != params.numPositions + 2 * boundary.contacts.size then
    .error s!"brick static equilibrium constraint vars {boundary.numVars} inconsistent with contacts {boundary.contacts.size}"
  if boundary.lowerBound.size != boundary.numOutputs ||
      boundary.upperBound.size != boundary.numOutputs ||
      boundary.residual.size != boundary.numOutputs then
    .error "brick static equilibrium bounds and residual must have one entry per output"
  if !boundary.lowerBound.all (fun x => x == 0.0) ||
      !boundary.upperBound.all (fun x => x == 0.0) then
    .error "brick static equilibrium constraint bounds should be zero"

end BrickStaticEquilibriumConstraintBoundary

def buildBrickStaticEquilibriumConstraint?
    (p : PlanarGripperParams := params)
    (q : PlanarGripperState)
    (contacts : Array FingerFaceContact) :
    Except String BrickStaticEquilibriumConstraintBoundary := do
  let residual ← staticEquilibriumResidual? p q contacts
  let boundary : BrickStaticEquilibriumConstraintBoundary := {
    contacts := contacts
    numVars := p.numPositions + 2 * contacts.size
    residual := residual
    graph := brickStaticEquilibriumGraph
  }
  boundary.validate?
  pure boundary

def allForcesInFrictionCone
    (p : PlanarGripperParams := params)
    (contacts : Array FingerFaceContact)
    (tol : Float := 1.0e-12) : Bool :=
  contacts.all
    (fun contact => contact.face.inFrictionCone p.staticFriction contact.force_B tol)

def drakeStaticEquilibriumTestContacts : Array FingerFaceContact :=
  #[
    {
      finger := .finger1
      face := .posZ
      force_B := { y := 2.5, z := -0.4 }
    },
    {
      finger := .finger3
      face := .posY
      force_B := { y := -0.2, z := -0.6 }
    }
  ]

def symmetricSupportState : PlanarGripperState :=
  {
    finger1Base := 0.0
    finger1Mid := 0.0
    finger2Base := 0.0
    finger2Mid := 0.0
    finger3Base := 0.0
    finger3Mid := 0.0
    brickY := 0.0
    brickZ := 0.0
    brickTheta := 0.0
  }

def symmetricSupportContacts (p : PlanarGripperParams := params) :
    Array FingerFaceContact :=
  let halfWeight := p.brickMass * p.gravity / 2.0
  #[
    {
      finger := .finger1
      face := .negZ
      force_B := { y := 0.0, z := halfWeight }
    },
    {
      finger := .finger2
      face := .negZ
      force_B := { y := 0.0, z := halfWeight }
    }
  ]

def symmetricSupportWrenches (p : PlanarGripperParams := params) :
    Array PlanarContactWrench :=
  let halfWeight := p.brickMass * p.gravity / 2.0
  #[
    {
      point_B := { y := -0.02, z := -p.brickSize.z / 2.0 }
      force_B := { y := 0.0, z := halfWeight }
      label := "left-bottom-support"
    },
    {
      point_B := { y := 0.02, z := -p.brickSize.z / 2.0 }
      force_B := { y := 0.0, z := halfWeight }
      label := "right-bottom-support"
    }
  ]

def symmetricSupportEquilibrium
    (p : PlanarGripperParams := params) : PlanarStaticEquilibrium :=
  {
    mass := p.brickMass
    gravity := p.gravity
    theta := 0.0
    contacts := symmetricSupportWrenches p
    label := "symmetric-bottom-support"
  }

def symmetricSupportResidual? (p : PlanarGripperParams := params) :
    Except String (Array Float) :=
  (symmetricSupportEquilibrium p).residualArray?

def contactGraph : SkeletonGraph :=
  brickStaticEquilibriumGraph

inductive TrajectoryInterpolation where
  | cubicShapePreserving
  deriving Repr, BEq, Inhabited

namespace TrajectoryInterpolation

def drakeName : TrajectoryInterpolation → String
  | .cubicShapePreserving => "PiecewisePolynomial::CubicShapePreserving"

end TrajectoryInterpolation

structure TrajectoryPublisherConfig where
  gripperModelUrl : String := planarGripperModelUrl
  keyframePath : String := postureKeyframePath
  keyframeDt : Float := 0.1
  interpolation : TrajectoryInterpolation := .cubicShapePreserving
  derivativeOrder : Nat := 1
  numFingers : Nat := 3
  numJoints : Nat := 6
  statusChannel : String := lcmStatusChannel
  commandChannel : String := lcmCommandChannel
  statusPeriod : Float := gripperLcmStatusPeriod
  waitForFirstStatus : Bool := true
  timeSource : String := "lcmt_planar_gripper_status.utime * 1e-6"
  commandEncoderStateInputPort : String := "state"
  commandEncoderTorqueInputPort : String := "torque"
  commandEncoderOutputPort : String := "lcmt_gripper_command"
  statusDecoderInputPort : String := "lcmt_planar_gripper_status"
  deriving Repr, Inhabited

namespace TrajectoryPublisherConfig

def validate? (config : TrajectoryPublisherConfig) : Except String Unit := do
  if config.gripperModelUrl == "" then
    .error "planar gripper trajectory publisher model URL must be nonempty"
  if config.keyframePath == "" then
    .error "planar gripper trajectory publisher keyframe path must be nonempty"
  if !config.keyframeDt.isFinite || config.keyframeDt <= 0.0 then
    .error s!"planar gripper keyframe_dt must be positive and finite, got {config.keyframeDt}"
  if config.numFingers != params.numFingers then
    .error s!"trajectory publisher num_fingers {config.numFingers} != {params.numFingers}"
  if config.numJoints != config.numFingers * params.jointsPerFinger then
    .error s!"trajectory publisher num_joints {config.numJoints} is inconsistent with num_fingers={config.numFingers}"
  if config.derivativeOrder != 1 then
    .error s!"Drake trajectory source is constructed with one derivative, got {config.derivativeOrder}"
  if config.statusChannel == "" || config.commandChannel == "" then
    .error "planar gripper LCM channels must be nonempty"
  if config.statusChannel == config.commandChannel then
    .error "planar gripper status and command channels must be distinct"
  if !config.statusPeriod.isFinite || config.statusPeriod <= 0.0 then
    .error s!"planar gripper status period must be positive and finite, got {config.statusPeriod}"
  if !config.waitForFirstStatus then
    .error "trajectory publisher must wait for the first lcmt_planar_gripper_status"
  if config.commandEncoderStateInputPort != "state" ||
      config.commandEncoderTorqueInputPort != "torque" ||
      config.commandEncoderOutputPort != "lcmt_gripper_command" ||
      config.statusDecoderInputPort != "lcmt_planar_gripper_status" then
    .error "planar gripper LCM encoder/decoder port names should match Drake"

def stateOutputDim (config : TrajectoryPublisherConfig) : Nat :=
  config.numJoints * (config.derivativeOrder + 1)

def torqueDim (config : TrajectoryPublisherConfig) : Nat :=
  config.numJoints

def trajectoryDuration (config : TrajectoryPublisherConfig) : Float :=
  (postureKeyframeCount - 1).toFloat * config.keyframeDt

def keyframeTime (config : TrajectoryPublisherConfig) (i : Nat) : Float :=
  i.toFloat * config.keyframeDt

end TrajectoryPublisherConfig

def trajectoryPublisherConfig : TrajectoryPublisherConfig := {}

def zeroJointTorques (config : TrajectoryPublisherConfig := trajectoryPublisherConfig) :
    Array Float :=
  Array.replicate config.numJoints 0.0

def trajectoryPublisherGraph
    (config : TrajectoryPublisherConfig := trajectoryPublisherConfig) :
    SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 7010, kind := .state .boundary, label := config.statusChannel }
    |>.addVertex { id := 7011, kind := .state .boundary, label := config.keyframePath }
    |>.addVertex { id := 7012, kind := .state .interior, label := "reordered finger keyframes" }
    |>.addVertex { id := 7013, kind := .state .interior, label := "CubicShapePreserving TrajectorySource" }
    |>.addVertex { id := 7014, kind := .frozen, label := "zero joint torques" }
    |>.addVertex { id := 7015, kind := .state .interior, label := "GripperCommandEncoder" }
    |>.addVertex { id := 7016, kind := .state .boundary, label := config.commandChannel }
    |>.addVertex { id := 7017, kind := .eventTime, label := config.timeSource }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[7012]
      reads := #[7011]
      writes := #[7012]
      cost := { work := postureKeyframeCount.toFloat * config.numJoints.toFloat }
      label := "ParseKeyframes + ReorderKeyframesForPlant"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[7013]
      reads := #[7012]
      writes := #[7013]
      label := s!"TrajectorySource from {config.interpolation.drakeName} with {config.derivativeOrder} derivative"
    }
    |>.addMove {
      kind := .freezeControl
      targets := #[7014]
      writes := #[7014]
      label := "ConstantVectorSource zero gripper torques"
    }
    |>.addMove {
      kind := .checkpointBoundary
      targets := #[7017]
      reads := #[7010]
      writes := #[7017]
      label := "wait for first lcmt_planar_gripper_status and set context time"
    }
    |>.addMove {
      kind := .clockedUpdate
      targets := #[7017]
      reads := #[7010, 7013, 7014, 7015]
      writes := #[7016]
      label := s!"advance to status utime and force-publish {config.commandChannel}"
    }

structure TrajectoryPublisherBoundary where
  config : TrajectoryPublisherConfig
  header : Array String
  postureRows : Nat
  firstParsedFingerKeyframe : Array Float
  firstControlPlantKeyframe : Array Float
  firstBrickInitialPose : Array Float
  zeroTorques : Array Float
  assetCatalog : Array PlanarGripperExampleAsset
  graph : SkeletonGraph
  deriving Repr, Inhabited

def buildTrajectoryPublisher?
    (config : TrajectoryPublisherConfig := trajectoryPublisherConfig) :
    Except String TrajectoryPublisherBoundary := do
  config.validate?
  validatePlanarGripperExampleAssetCatalog?
  if postureHeader.size != 9 then
    .error s!"postures.txt header should contain 9 names, got {postureHeader.size}"
  if postureKeyframeCount == 0 then
    .error "postures.txt should contain at least one keyframe"
  let parsed ← firstParsedFingerKeyframe?
  let reordered ← firstControlPlantKeyframe?
  let brickPose ← firstBrickInitialPose?
  pure {
    config := config
    header := postureHeader
    postureRows := postureKeyframeCount
    firstParsedFingerKeyframe := parsed
    firstControlPlantKeyframe := reordered
    firstBrickInitialPose := brickPose
    zeroTorques := zeroJointTorques config
    assetCatalog := planarGripperExampleAssets
    graph := trajectoryPublisherGraph config
  }

inductive PlanarGripperSimulationOrientation where
  | vertical
  | horizontal
  deriving Repr, BEq, Inhabited

namespace PlanarGripperSimulationOrientation

def label : PlanarGripperSimulationOrientation → String
  | .vertical => "vertical"
  | .horizontal => "horizontal"

def gravityVector : PlanarGripperSimulationOrientation → Array Float
  | .vertical => #[0.0, 0.0, -9.81]
  | .horizontal => #[-9.81, 0.0, 0.0]

def fixesBrickBaseFrame : PlanarGripperSimulationOrientation → Bool
  | .vertical => true
  | .horizontal => false

def addsBrickTranslateXJoint : PlanarGripperSimulationOrientation → Bool
  | .vertical => false
  | .horizontal => true

def numPositions : PlanarGripperSimulationOrientation → Nat
  | .vertical => params.numPositions
  | .horizontal => params.numPositions + 1

def numVelocities : PlanarGripperSimulationOrientation → Nat
  | .vertical => params.numPositions
  | .horizontal => params.numPositions + 1

end PlanarGripperSimulationOrientation

inductive PlanarGripperSimulationControlMode where
  | positionControl
  | torqueControl
  deriving Repr, BEq, Inhabited

namespace PlanarGripperSimulationControlMode

def label : PlanarGripperSimulationControlMode → String
  | .positionControl => "position-control"
  | .torqueControl => "torque-control"

def usesInverseDynamicsController : PlanarGripperSimulationControlMode → Bool
  | .positionControl => true
  | .torqueControl => false

def usesDirectTorqueInput : PlanarGripperSimulationControlMode → Bool
  | .positionControl => false
  | .torqueControl => true

end PlanarGripperSimulationControlMode

structure PlanarManipulandStatusMessage where
  utime : Float := 0.0
  position : Array Float := #[0.0, 0.0]
  theta : Float := 0.0
  velocity : Array Float := #[0.0, 0.0]
  thetadot : Float := 0.0
  deriving Repr, Inhabited

namespace PlanarManipulandStatusMessage

def stateVector? (msg : PlanarManipulandStatusMessage) :
    Except String (Array Float) := do
  if msg.position.size != 2 then
    .error s!"planar manipuland status position size {msg.position.size} != 2"
  if msg.velocity.size != 2 then
    .error s!"planar manipuland status velocity size {msg.velocity.size} != 2"
  pure #[
    msg.position[0]!, msg.position[1]!, msg.theta,
    msg.velocity[0]!, msg.velocity[1]!, msg.thetadot
  ]

def fromState? (time : Float) (state : Array Float) :
    Except String PlanarManipulandStatusMessage := do
  if !time.isFinite then
    .error s!"planar manipuland status time must be finite, got {time}"
  if state.size != 6 then
    .error s!"planar manipuland status state size {state.size} != 6"
  pure {
    utime := time * 1000000.0
    position := #[state[0]!, state[1]!]
    theta := state[2]!
    velocity := #[state[3]!, state[4]!]
    thetadot := state[5]!
  }

def passthrough? (msg : PlanarManipulandStatusMessage) :
    Except String PlanarManipulandStatusMessage := do
  let state ← msg.stateVector?
  PlanarManipulandStatusMessage.fromState? (msg.utime / 1000000.0) state

end PlanarManipulandStatusMessage

def sampleManipulandStatus : PlanarManipulandStatusMessage :=
  {
    position := #[0.1, 0.2]
    theta := 0.3
    velocity := #[0.4, 0.5]
    thetadot := 0.6
  }

def planarManipulandStatusPeriod : Float := 0.010

structure PlanarGripperFloorPrimitive where
  visualName : String := "FloorVisualGeometry"
  collisionName : String := "FloorCollisionGeometry"
  radius : Float := 0.125
  height : Float := 0.001
  brickCornerSphereRadius : Float := 0.0001
  brickCornerSphereX : Float := -0.035
  penetration : Float := 1.0e-5
  friction : CoulombFriction := { staticFriction := 0.5, dynamicFriction := 0.5 }
  deriving Repr, Inhabited

namespace PlanarGripperFloorPrimitive

def sphereTipXOffset (floor : PlanarGripperFloorPrimitive) : Float :=
  floor.brickCornerSphereX - floor.brickCornerSphereRadius

def centerX (floor : PlanarGripperFloorPrimitive) : Float :=
  floor.sphereTipXOffset - floor.height / 2.0 + floor.penetration

def validate? (floor : PlanarGripperFloorPrimitive) : Except String Unit := do
  if floor.visualName.isEmpty || floor.collisionName.isEmpty then
    .error "planar gripper floor visual and collision names must be nonempty"
  if !floor.radius.isFinite || floor.radius <= 0.0 then
    .error s!"planar gripper floor radius must be positive and finite, got {floor.radius}"
  if !floor.height.isFinite || floor.height <= 0.0 then
    .error s!"planar gripper floor height must be positive and finite, got {floor.height}"
  if !floor.brickCornerSphereRadius.isFinite || floor.brickCornerSphereRadius < 0.0 then
    .error s!"planar gripper brick corner sphere radius must be nonnegative and finite, got {floor.brickCornerSphereRadius}"
  if !floor.penetration.isFinite || floor.penetration < 0.0 then
    .error s!"planar gripper brick-floor penetration must be nonnegative and finite, got {floor.penetration}"
  floor.friction.validate? floor.collisionName

def contactEnvironment (floor : PlanarGripperFloorPrimitive) :
    HalfSpaceContactEnvironment :=
  {
    visualName := floor.visualName
    collisionName := floor.collisionName
    friction := floor.friction
  }

end PlanarGripperFloorPrimitive

structure PlanarGripperSimulationConfig where
  targetRealtimeRate : Float := 1.0
  simulationTime : Float := 4.5
  timeStep : Float := 1.0e-3
  penetrationAllowance : Float := 1.0e-3
  floorStaticFriction : Float := 0.5
  floorKineticFriction : Float := 0.5
  brickFloorPenetration : Float := 1.0e-5
  orientation : PlanarGripperSimulationOrientation := .vertical
  visualizeContacts : Bool := false
  controlMode : PlanarGripperSimulationControlMode := .positionControl
  deriving Repr, Inhabited

namespace PlanarGripperSimulationConfig

def plantConfig (config : PlanarGripperSimulationConfig) :
    MultibodyPlantConfigPrimitive :=
  {
    timeStep := config.timeStep
    penetrationAllowance := config.penetrationAllowance
    stictionTolerance := 1.0e-3
    contactApproximation := .sap
  }

def floor (config : PlanarGripperSimulationConfig) :
    PlanarGripperFloorPrimitive :=
  {
    penetration := config.brickFloorPenetration
    friction := {
      staticFriction := config.floorStaticFriction
      dynamicFriction := config.floorKineticFriction
    }
  }

def validate? (config : PlanarGripperSimulationConfig) :
    Except String Unit := do
  config.plantConfig.validate?
  config.floor.validate?
  if !config.targetRealtimeRate.isFinite || config.targetRealtimeRate < 0.0 then
    .error s!"planar gripper target realtime rate must be nonnegative and finite, got {config.targetRealtimeRate}"
  if !config.simulationTime.isFinite || config.simulationTime <= 0.0 then
    .error s!"planar gripper simulation time must be positive and finite, got {config.simulationTime}"

def positionControlKp (_config : PlanarGripperSimulationConfig) :
    Array Float :=
  Array.replicate params.numFingers 1500.0 |>.flatMap (fun k => #[k, k])

def positionControlKd (_config : PlanarGripperSimulationConfig) :
    Array Float :=
  Array.replicate params.numFingers 500.0 |>.flatMap (fun k => #[k, k])

def positionControlKi (_config : PlanarGripperSimulationConfig) :
    Array Float :=
  Array.replicate params.numFingers 500.0 |>.flatMap (fun k => #[k, k])

end PlanarGripperSimulationConfig

def planarGripperParsedPlantQuantities : ParsedMultibodyPlantQuantities :=
  {
    modelUris := #[planarGripperModelUrl, planarBrickModelUrl]
    builtInModelInstances := 2
    numModelInstances := 4
    numActuators := params.numFingers * params.jointsPerFinger
    numJoints := 12
    numBodies := 17
    modelInstances := #[
      {
        name := "planar_gripper"
        modelUri := planarGripperModelUrl
        numPositions := params.numFingers * params.jointsPerFinger
        numVelocities := params.numFingers * params.jointsPerFinger
      },
      {
        name := "brick"
        modelUri := planarBrickModelUrl
        numPositions := 3
        numVelocities := 3
      }
    ]
    finalized := true
    label := "planar_gripper_simulation parser outputs"
  }

def planarGripperFullPlantModel
    (orientation : PlanarGripperSimulationOrientation) :
    FullMultibodyPlantModel :=
  {
    modelName := s!"planar_gripper_with_brick_{orientation.label}"
    modelUri := planarGripperModelUrl ++ " + " ++ planarBrickModelUrl
    numPositions := orientation.numPositions
    numVelocities := orientation.numVelocities
    numActuatedDofs := params.numFingers * params.jointsPerFinger
    finalized := true
    label := "planar_gripper_simulation full plant"
  }

def planarGripperControlPlantModel : FullMultibodyPlantModel :=
  {
    modelName := "planar_gripper_control_plant"
    modelUri := planarGripperModelUrl
    numPositions := params.numFingers * params.jointsPerFinger
    numVelocities := params.numFingers * params.jointsPerFinger
    numActuatedDofs := params.numFingers * params.jointsPerFinger
    finalized := true
    label := "planar_gripper_simulation control plant"
  }

def planarGripperSimulationInitialQ?
    (config : PlanarGripperSimulationConfig) : Except String (Array Float) := do
  let gripper ← firstControlPlantKeyframe?
  let brick ← firstBrickInitialPose?
  let q := gripper ++ brick
  if config.orientation.addsBrickTranslateXJoint then
    pure (q.push 0.0)
  else
    pure q

def planarGripperSimulationPlantStep?
    (config : PlanarGripperSimulationConfig) :
    Except String FullMultibodyPlantStep := do
  config.validate?
  let q0 ← planarGripperSimulationInitialQ? config
  let model := planarGripperFullPlantModel config.orientation
  let step : FullMultibodyPlantStep := {
    model := model
    config := config.plantConfig
    q0 := q0
    v0 := Array.replicate model.numVelocities 0.0
    actuation := zeroJointTorques trajectoryPublisherConfig
    t0 := 0.0
    t1 := config.simulationTime
    ground? := some config.floor.contactEnvironment
    label := "planar_gripper_simulation full plant advance"
  }
  step.validate?
  pure step

private def unitVector (n index : Nat) : Array Float := Id.run do
  let mut out := Array.replicate n 0.0
  if index < n then
    out := out.set! index 1.0
  return out

private def vectorWithEntry (n index : Nat) (value : Float) : Array Float :=
  (unitVector n index).map (fun x => value * x)

def brickPlanarInertia (p : PlanarGripperParams := params) : Float :=
  p.brickMass * (p.brickSize.y * p.brickSize.y +
    p.brickSize.z * p.brickSize.z) / 12.0

def planarGripperSimulationMassDiagonal
    (config : PlanarGripperSimulationConfig)
    (p : PlanarGripperParams := params) : Array Float :=
  let gripperDiag := Array.replicate (p.numFingers * p.jointsPerFinger) 1.0
  let brickPlanarDiag := #[p.brickMass, p.brickMass, brickPlanarInertia p]
  if config.orientation.addsBrickTranslateXJoint then
    gripperDiag ++ brickPlanarDiag ++ #[p.brickMass]
  else
    gripperDiag ++ brickPlanarDiag

def planarGripperSimulationMassMatrix
    (config : PlanarGripperSimulationConfig)
    (p : PlanarGripperParams := params) : Array (Array Float) :=
  FloatMatrix.diagonal (planarGripperSimulationMassDiagonal config p)

def planarGripperSimulationSupportVelocityIndex
    (config : PlanarGripperSimulationConfig) : Nat :=
  if config.orientation.addsBrickTranslateXJoint then 9 else brickTranslateZPositionIndex

def planarGripperSimulationTangentVelocityIndex
    (_config : PlanarGripperSimulationConfig) : Nat :=
  brickTranslateYPositionIndex

def planarGripperSimulationHorizontalPositionIndex : Nat := 9

structure PlanarGripperSimulationState where
  q : Array Float
  v : Array Float
  deriving Repr, Inhabited

namespace PlanarGripperSimulationState

def validate? (state : PlanarGripperSimulationState)
    (config : PlanarGripperSimulationConfig) : Except String Unit := do
  if state.q.size != config.orientation.numPositions then
    .error s!"planar gripper simulation state q size {state.q.size} != {config.orientation.numPositions}"
  if state.v.size != config.orientation.numVelocities then
    .error s!"planar gripper simulation state v size {state.v.size} != {config.orientation.numVelocities}"

def initial? (config : PlanarGripperSimulationConfig) :
    Except String PlanarGripperSimulationState := do
  let q ← planarGripperSimulationInitialQ? config
  pure { q := q, v := Array.replicate config.orientation.numVelocities 0.0 }

end PlanarGripperSimulationState

def planarGripperSimulationContactSignedDistance?
    (config : PlanarGripperSimulationConfig)
    (q : Array Float) : Except String Float := do
  if q.size != config.orientation.numPositions then
    .error s!"planar gripper contact q size {q.size} != {config.orientation.numPositions}"
  if config.orientation.addsBrickTranslateXJoint then
    pure ((q.getD planarGripperSimulationHorizontalPositionIndex 0.0) -
      config.floor.penetration)
  else
    pure (-config.floor.penetration)

def planarGripperSimulationContactPoint?
    (config : PlanarGripperSimulationConfig)
    (q : Array Float) : Except String (Array Float) := do
  if q.size != config.orientation.numPositions then
    .error s!"planar gripper contact q size {q.size} != {config.orientation.numPositions}"
  if config.orientation.addsBrickTranslateXJoint then
    pure #[
      q.getD planarGripperSimulationHorizontalPositionIndex 0.0 +
        config.floor.sphereTipXOffset,
      0.0,
      0.0
    ]
  else
    pure #[0.0, 0.0, -(params.brickSize.z / 2.0)]

def planarGripperSimulationContactCandidate?
    (config : PlanarGripperSimulationConfig)
    (q : Array Float)
    (qdot : Array Float) :
    Except String ContactCandidate := do
  if qdot.size != config.orientation.numVelocities then
    .error s!"planar gripper contact qdot size {qdot.size} != {config.orientation.numVelocities}"
  let n := config.orientation.numVelocities
  let normalIndex := planarGripperSimulationSupportVelocityIndex config
  let tangentIndex := planarGripperSimulationTangentVelocityIndex config
  let normalJac := unitVector n normalIndex
  let tangentJac := unitVector n tangentIndex
  let tangentJac2 := unitVector n brickRevoluteXPositionIndex
  let point_W ← planarGripperSimulationContactPoint? config q
  let signedDistance ← planarGripperSimulationContactSignedDistance? config q
  pure {
    id := 703600
    bodyA := "brick"
    bodyB := "FloorCollisionGeometry"
    point_W := point_W
    normal_W :=
      if config.orientation.addsBrickTranslateXJoint then
        #[1.0, 0.0, 0.0]
      else
        #[0.0, 0.0, 1.0]
    signedDistance := signedDistance
    normalVelocity := FloatArray.dot normalJac qdot
    tangentVelocity := FloatArray.dot tangentJac qdot
    tangentVelocity2 := FloatArray.dot tangentJac2 qdot
    normalJacobian := normalJac
    tangentJacobian := tangentJac
    tangentJacobian2 := tangentJac2
    mode := .sticking
    label := "planar-gripper-brick-floor-corner-contact"
  }

def planarGripperSimulationContactSupport
    (config : PlanarGripperSimulationConfig)
    (q qdot : Array Float) : Except String ContactSupport := do
  let candidate ← planarGripperSimulationContactCandidate? config q qdot
  pure <|
    ContactSupport.selectWithPolicy
        (.threshold config.penetrationAllowance)
        #[candidate]
        "planar-gripper-simulation-floor-support"
      |>.classifyCandidates config.penetrationAllowance
        config.plantConfig.stictionTolerance

def planarGripperSimulationGravityBias
    (config : PlanarGripperSimulationConfig)
    (p : PlanarGripperParams := params) : Array Float :=
  vectorWithEntry config.orientation.numVelocities
    (planarGripperSimulationSupportVelocityIndex config)
    (p.brickMass * p.gravity)

def planarGripperSimulationFloorForces
    (_config : PlanarGripperSimulationConfig)
    (support : ContactSupport)
    (p : PlanarGripperParams := params) :
    Except String (Array ContactForceScalars) := do
  let selected ← support.selectedCandidates?
  pure (selected.map (fun candidate =>
    ContactForceScalars.fromCandidate3D candidate
      (p.brickMass * p.gravity) 0.0 0.0))

def planarGripperPidGains
    (config : PlanarGripperSimulationConfig) : JointPidGains :=
  {
    kp := config.positionControlKp
    kd := config.positionControlKd
    ki := config.positionControlKi
    label := "planar-gripper-position-control"
  }

def planarGripperSimulationActuation?
    (config : PlanarGripperSimulationConfig)
    (q v : Array Float) :
    Except String (Array Float × Option JointPidOutput) := do
  let nActuated := params.numFingers * params.jointsPerFinger
  if q.size != config.orientation.numPositions then
    .error s!"planar gripper simulation q size {q.size} != {config.orientation.numPositions}"
  if v.size != config.orientation.numVelocities then
    .error s!"planar gripper simulation v size {v.size} != {config.orientation.numVelocities}"
  let qGripper := q.extract 0 nActuated
  let vGripper := v.extract 0 nActuated
  let (jointForces, controllerOutput?) ←
    match config.controlMode with
    | .torqueControl =>
        pure (zeroJointTorques trajectoryPublisherConfig, none)
    | .positionControl =>
        let input : JointPidInput := {
          estimatedState := qGripper ++ vGripper
          desiredState := qGripper ++ Array.replicate nActuated 0.0
          integralError := Array.replicate nActuated 0.0
          label := "planar-gripper-inverse-dynamics-position-control"
        }
        let output ← input.evaluate? (planarGripperPidGains config)
        pure (output.feedback, some output)
  pure
    (jointForces ++
      Array.replicate (config.orientation.numVelocities - nActuated) 0.0,
      controllerOutput?)

structure PlanarGripperSimulationPhysics where
  state : PlanarGripperSimulationState
  controllerOutput? : Option JointPidOutput := none
  primitivePlant : FullPlantPrimitivePhysics
  fullPhysics : FullPhysicsResult
  deriving Repr, Inhabited

def planarGripperSimulationFullPhysicsPrimitives?
    (config : PlanarGripperSimulationConfig)
    (q v : Array Float)
    (label : String := "planar_gripper_simulation primitive full physics") :
    Except String (FullPhysicsPrimitives × Option JointPidOutput) := do
  config.validate?
  let state : PlanarGripperSimulationState := { q := q, v := v }
  state.validate? config
  let (actuation, controllerOutput?) ← planarGripperSimulationActuation? config q v
  let support ← planarGripperSimulationContactSupport config q v
  support.validateJacobianWidth? config.orientation.numVelocities
  let forces ← planarGripperSimulationFloorForces config support
  pure ({
    massMatrix := planarGripperSimulationMassMatrix config
    qdot := v
    actuationForces := actuation
    biasForces := planarGripperSimulationGravityBias config
    contactCandidates := support.candidates
    sourceContactCandidateCount? := support.sourceCandidateCount?
    supportPolicy := .threshold config.penetrationAllowance
    contactForceSource := .precomputed
    contactForces := forces
    distanceTol := config.penetrationAllowance
    tangentVelocityTol := config.plantConfig.stictionTolerance
    label := label
  }, controllerOutput?)

def planarGripperSimulationPrimitiveProvider
    (config : PlanarGripperSimulationConfig)
    (label : String := "planar_gripper_simulation primitive provider") :
    FullPhysicsPrimitiveProvider PlanarGripperSimulationState :=
  {
    label := label
    primitivesAt? := fun state => do
      let (primitive, _) ←
        planarGripperSimulationFullPhysicsPrimitives?
          config state.q state.v label
      pure primitive
  }

def solvePlanarGripperSimulationFullPhysics?
    (plantStep : FullMultibodyPlantStep)
    (config : PlanarGripperSimulationConfig)
    (q v : Array Float)
    (intervalVertex : VertexId := 7037)
    (label : String := "planar_gripper_simulation primitive full physics") :
    Except String PlanarGripperSimulationPhysics := do
  let state : PlanarGripperSimulationState := { q := q, v := v }
  state.validate? config
  let provider := planarGripperSimulationPrimitiveProvider config label
  let primitive ← provider.validPrimitivesAt? state
  let (_, controllerOutput?) ← planarGripperSimulationActuation? config q v
  let primitivePlant : FullPlantPrimitivePhysics := {
    step := plantStep
    primitives := primitive
    intervalVertex := intervalVertex
    label := label
  }
  let fullPhysics ← primitivePlant.solve?
  pure {
    state := state
    controllerOutput? := controllerOutput?
    primitivePlant := primitivePlant
    fullPhysics := fullPhysics
  }

def solveInitialPlanarGripperSimulationFullPhysics?
    (config : PlanarGripperSimulationConfig)
    (plantStep : FullMultibodyPlantStep)
    (intervalVertex : VertexId := 7037) :
    Except String PlanarGripperSimulationPhysics := do
  let q ← planarGripperSimulationInitialQ? config
  let v := Array.replicate config.orientation.numVelocities 0.0
  solvePlanarGripperSimulationFullPhysics? plantStep config q v intervalVertex
    "planar_gripper_simulation full gripper+brick primitive"

def forceSensorJointNames : Array String :=
  #[
    "finger1_sensor_weldjoint",
    "finger2_sensor_weldjoint",
    "finger3_sensor_weldjoint"
  ]

structure PlanarGripperSimulationBoundary where
  config : PlanarGripperSimulationConfig
  parsedPlant : ParsedMultibodyPlantQuantities
  controlPlant : FullMultibodyPlantModel
  plantStep : FullMultibodyPlantStep
  controllerOutput? : Option JointPidOutput := none
  primitivePlant : FullPlantPrimitivePhysics
  fullPhysics : FullPhysicsResult
  floor : PlanarGripperFloorPrimitive
  initialGripperPosition : Array Float
  initialBrickPose : Array Float
  manipulandStatusPeriod : Float
  sampleManipulandRoundTrip : PlanarManipulandStatusMessage
  forceSensorJointNames : Array String
  forceSensorOutputDim : Nat
  initialState : PlanarGripperSimulationState
  assetCatalog : Array PlanarGripperExampleAsset
  graph : SkeletonGraph
  deriving Repr, Inhabited

namespace PlanarGripperSimulationBoundary

def validate? (boundary : PlanarGripperSimulationBoundary) :
    Except String Unit := do
  boundary.config.validate?
  boundary.parsedPlant.validate?
  boundary.controlPlant.validate?
  boundary.plantStep.validate?
  boundary.primitivePlant.validate?
  boundary.fullPhysics.equation.validate?
  boundary.floor.validate?
  validatePlanarGripperExampleAssetCatalog?
  if boundary.initialGripperPosition.size != params.numFingers * params.jointsPerFinger then
    .error s!"planar gripper simulation initial gripper position size {boundary.initialGripperPosition.size} != 6"
  if boundary.initialBrickPose.size != 3 then
    .error s!"planar gripper simulation initial brick pose size {boundary.initialBrickPose.size} != 3"
  if !boundary.manipulandStatusPeriod.isFinite || boundary.manipulandStatusPeriod <= 0.0 then
    .error s!"planar manipuland status period must be positive and finite, got {boundary.manipulandStatusPeriod}"
  if boundary.forceSensorJointNames.size != params.numFingers then
    .error s!"planar gripper force sensor joint count {boundary.forceSensorJointNames.size} != num_fingers {params.numFingers}"
  if boundary.forceSensorOutputDim != params.numFingers * 2 then
    .error s!"planar gripper force sensor output dim {boundary.forceSensorOutputDim} != 2*num_fingers"
  if boundary.fullPhysics.equation.qdot.size != boundary.plantStep.model.numVelocities then
    .error s!"planar gripper full-physics qdot size {boundary.fullPhysics.equation.qdot.size} != plant velocities {boundary.plantStep.model.numVelocities}"
  if boundary.fullPhysics.equation.massMatrix.size != boundary.plantStep.model.numVelocities then
    .error s!"planar gripper full-physics mass rows {boundary.fullPhysics.equation.massMatrix.size} != plant velocities {boundary.plantStep.model.numVelocities}"
  boundary.initialState.validate? boundary.config
  if boundary.initialState.q != boundary.plantStep.q0 then
    .error "planar gripper initial state q should match plant step q0"
  if boundary.initialState.v != boundary.plantStep.v0 then
    .error "planar gripper initial state v should match plant step v0"

end PlanarGripperSimulationBoundary

def planarGripperSimulationGraph
    (boundary : PlanarGripperSimulationBoundary) : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 7030, kind := .state .boundary, label := "../drake/examples/planar_gripper/planar_gripper_simulation.cc flags" }
    |>.addVertex { id := 7031, kind := .state .interior, label := "Parser gripper+brick full simulation plant" }
    |>.addVertex { id := 7032, kind := .state .interior, label := "Parser gripper-only control plant" }
    |>.addVertex { id := 7033, kind := .state .boundary, label := lcmCommandChannel }
    |>.addVertex { id := 7034, kind := .state .interior, label := "GripperCommandDecoder" }
    |>.addVertex { id := 7035, kind := .state .interior, label := boundary.config.controlMode.label }
    |>.addVertex { id := 7036, kind := .state .interior, label := "thin cylinder floor contact primitive" }
    |>.addVertex { id := 7037, kind := .interval, label := "Simulator.AdvanceTo planar gripper full plant" }
    |>.addVertex { id := 7038, kind := .state .interior, label := "ForceSensorEvaluator fy/fz extraction" }
    |>.addVertex { id := 7039, kind := .state .boundary, label := lcmStatusChannel }
    |>.addVertex { id := 7040, kind := .state .boundary, label := "PLANAR_MANIPULAND_STATUS" }
    |>.addMove
      (ParsedMultibodyPlantQuantities.parserMove 7031
        "Parser.AddModelsFromUrl gripper+brick; WeldGripperFrames; orientation-specific brick base; plant.Finalize")
    |>.addMove
      (ParsedMultibodyPlantQuantities.parserMove 7032
        "Parser.AddModelsFromUrl gripper-only control plant; WeldGripperFrames; control_plant.Finalize")
    |>.addMove {
      kind := .localSchurBlock
      targets := #[7036]
      reads := #[7031]
      writes := #[7036]
      label := "AddFloor from brick corner sphere, floor penetration, and Coulomb friction"
    }
    |>.addMove {
      kind := .clockedUpdate
      targets := #[7033]
      reads := #[7033]
      writes := #[7034]
      label := "LcmSubscriber PLANAR_GRIPPER_COMMAND + GripperCommandDecoder"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[7035]
      reads := #[7031, 7032, 7034]
      writes := #[7037]
      label :=
        if boundary.config.controlMode.usesInverseDynamicsController then
          "InverseDynamicsController Kp=1500 Ki=500 Kd=500 plus GeneralizedForceToActuationOrdering"
        else
          "direct torque command_decoder torques to plant actuation input"
    }
    |>.addMove boundary.fullPhysics.supportMove
    |>.addMove boundary.fullPhysics.move
    |>.addMove {
      kind := .localSchurBlock
      targets := #[7038]
      reads := #[7037]
      writes := #[7039]
      label := "ForceSensorEvaluator extracts planar fy/fz from three sensor weld joints"
    }
    |>.addMove {
      kind := .clockedUpdate
      targets := #[7039]
      reads := #[7037, 7038]
      writes := #[7039]
      label := "GripperStatusEncoder + LcmPublisher PLANAR_GRIPPER_STATUS at 10 ms"
    }
    |>.addMove {
      kind := .clockedUpdate
      targets := #[7040]
      reads := #[7040]
      writes := #[7040]
      label := "PlanarManipulandStatusDecoder forced/periodic update + encoder passthrough"
    }

def buildPlanarGripperSimulation?
    (config : PlanarGripperSimulationConfig := {}) :
    Except String PlanarGripperSimulationBoundary := do
  let plantStep ← planarGripperSimulationPlantStep? config
  let primitivePhysics ← solveInitialPlanarGripperSimulationFullPhysics? config plantStep
  let gripper ← firstControlPlantKeyframe?
  let brick ← firstBrickInitialPose?
  let manipulandRoundTrip ← sampleManipulandStatus.passthrough?
  let boundary : PlanarGripperSimulationBoundary := {
    config := config
    parsedPlant := planarGripperParsedPlantQuantities
    controlPlant := planarGripperControlPlantModel
    plantStep := plantStep
    controllerOutput? := primitivePhysics.controllerOutput?
    primitivePlant := primitivePhysics.primitivePlant
    fullPhysics := primitivePhysics.fullPhysics
    floor := config.floor
    initialGripperPosition := gripper
    initialBrickPose := brick
    manipulandStatusPeriod := planarManipulandStatusPeriod
    sampleManipulandRoundTrip := manipulandRoundTrip
    forceSensorJointNames := forceSensorJointNames
    forceSensorOutputDim := params.numFingers * 2
    initialState := primitivePhysics.state
    assetCatalog := planarGripperExampleAssets
    graph := SkeletonGraph.empty
  }
  let boundary := { boundary with graph := planarGripperSimulationGraph boundary }
  boundary.validate?
  pure boundary

structure PlanarGripperResult where
  references : Array DrakeReference
  assetCatalog : Array PlanarGripperExampleAsset
  params : PlanarGripperParams
  q : PlanarGripperState
  drakeTestResidual : Array Float
  symmetricResidual : Array Float
  graph : SkeletonGraph
  trajectoryPublisher : TrajectoryPublisherBoundary
  simulation : PlanarGripperSimulationBoundary
  deriving Repr, Inhabited

def buildEndToEnd? (p : PlanarGripperParams := params) :
    Except String PlanarGripperResult := do
  validatePlanarGripperExampleAssetCatalog?
  let drakeResidual ←
    staticEquilibriumResidual? p drakeTestState drakeStaticEquilibriumTestContacts
  let symmetricResidual ← symmetricSupportResidual? p
  let trajectoryPublisher ← buildTrajectoryPublisher?
  let simulation ← buildPlanarGripperSimulation?
  pure {
    references := drakeReferences
    assetCatalog := planarGripperExampleAssets
    params := p
    q := drakeTestState
    drakeTestResidual := drakeResidual
    symmetricResidual := symmetricResidual
    graph := contactGraph
    trajectoryPublisher := trajectoryPublisher
    simulation := simulation
  }

end Tyr.EventSkeleton.Examples.PlanarGripper
