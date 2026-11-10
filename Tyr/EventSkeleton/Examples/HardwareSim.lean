import Tyr.EventSkeleton.HardwareSim
import Tyr.EventSkeleton.Examples.Pendulum
import Tyr.EventSkeleton.Examples.KukaIiwaArm
import Tyr.EventSkeleton.Examples.SimpleGripper

/-!
# Drake Hardware Sim Event-Skeleton Example

This ports the typed scenario and setup sequence from
`../drake/examples/hardware_sim`.

The important behavior is the diagram boundary: a scenario describes model
directives, initial positions, LCM buses, drivers, cameras, visualization, and
simulator settings.  The EventSkeleton representation records that setup plan
as local provider blocks and fixed external hardware boundaries.

The executable physics path is intentionally primitive-driven.  When a scenario
contains a model family with a local primitive, `AdvanceTo` runs that primitive
through the existing DiffEq/EventSkeleton code.  Model families without local
primitives are reported as unsupported instead of being simulated by a fake
plant.
-/

namespace Tyr.EventSkeleton.Examples.HardwareSim

open Tyr.EventSkeleton

private def pi : Float := 3.14159265358979323846

def wsgDefaultFingerPosition : Float := 0.02

def amazonTableSdf : String :=
  "package://drake_models/manipulation_station/amazon_table_simplified.sdf"

def yellowBellPepperSdf : String :=
  "package://drake_models/veggies/yellow_bell_pepper_no_stem_low.sdf"

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/hardware_sim/BUILD.bazel"
      concept := "declares the hardware_sim binaries, scenario data, robot_commander, and smoke tests"
    },
    {
      path := "../drake/examples/hardware_sim/README.md"
      concept := "documents running the HardwareSim demo and the companion robot commander"
    },
    {
      path := "../drake/examples/hardware_sim/LICENSE-MIT-0.txt"
      concept := "records Drake's example-level license metadata"
    },
    {
      path := "../drake/examples/hardware_sim/scenario.h"
      concept := "defines the typed YAML Scenario schema shared by C++ and Python"
    },
    {
      path := "../drake/examples/hardware_sim/scenario.cc"
      concept := "loads a scenario from constructor defaults, scenario file, and scenario_text overrides"
    },
    {
      path := "../drake/examples/hardware_sim/hardware_sim.cc"
      concept := "builds plant, SceneGraph, directives, LCM buses, drivers, cameras, visualization, simulator config, random context, and optional Graphviz"
    },
    {
      path := "../drake/examples/hardware_sim/hardware_sim.py"
      concept := "Python implementation kept in sync with the C++ setup sequence"
    },
    {
      path := "../drake/examples/hardware_sim/example_scenarios.yaml"
      concept := "Demo scenario with IIWA, WSG, table, pepper, driver_traffic LCM bus, camera, drivers, and WSG initial position"
    },
    {
      path := "../drake/examples/hardware_sim/test/test_scenarios.yaml"
      concept := "Defaults and OneOfEverything smoke-test scenarios"
    },
    {
      path := "../drake/examples/hardware_sim/robot_commander.py"
      concept := "publishes cyclic IIWA and WSG commands on the hardware-style LCM channels"
    },
    {
      path := "../drake/examples/hardware_sim/test/hardware_sim_cc_test.py"
      concept := "runs Drake's C++ HardwareSim smoke tests over scenario files and graphviz"
    },
    {
      path := "../drake/examples/hardware_sim/test/hardware_sim_py_test.py"
      concept := "runs Drake's Python HardwareSim smoke tests over matching scenario behavior"
    },
    {
      path := "../drake/examples/hardware_sim/test/hardware_sim_test_common.py"
      concept := "shares HardwareSim test helpers for invoking the Drake example binaries"
    },
    {
      path := "../drake/examples/hardware_sim/test/robot_commander_test.py"
      concept := "checks robot_commander cyclic IIWA and WSG command publication"
    }
  ]

def defaultScenario : HardwareScenario :=
  { label := "Defaults" }

def pendulumDirective : HardwareModelDirective :=
  HardwareModelDirective.addModel
    "alice"
    "package://drake/examples/pendulum/Pendulum.urdf"

def oneOfEverythingScenario : HardwareScenario :=
  {
    label := "OneOfEverything"
    randomSeed := 1
    simulationDuration := 3.14
    simulatorConfig := { targetRealtimeRate := 5.0 }
    plantConfig := { stictionTolerance := 1.0e-2 }
    sceneGraphConfig := { defaultProximityCompliance? := some .compliant }
    directives := #[pendulumDirective]
    lcmBuses := #[
      { name := "default" },
      { name := "extra_bus" }
    ]
    modelDrivers := #[
      { modelName := "alice", config := .zeroForce }
    ]
    cameras := #[
      {
        key := "arbitrary_camera_name"
        name := "camera_0"
        lcmBus := "extra_bus"
      }
    ]
    visualization := {
      lcmBus := "extra_bus"
      publishPeriod := 0.125
      defaultIllustrationColor? := some { r := 0.8, g := 0.8, b := 0.8, a := 1.0 }
    }
  }

def demoIiwaJointPositions : Array HardwareJointPosition :=
  #[
    { jointName := "iiwa_joint_1", positions := #[-0.2] },
    { jointName := "iiwa_joint_2", positions := #[0.79] },
    { jointName := "iiwa_joint_3", positions := #[0.32] },
    { jointName := "iiwa_joint_4", positions := #[-1.76] },
    { jointName := "iiwa_joint_5", positions := #[-0.36] },
    { jointName := "iiwa_joint_6", positions := #[0.64] },
    { jointName := "iiwa_joint_7", positions := #[-0.73] }
  ]

def demoDirectives : Array HardwareModelDirective :=
  #[
    HardwareModelDirective.addModel
      "amazon_table"
      amazonTableSdf,
    HardwareModelDirective.addWeld "world" "amazon_table::amazon_table",
    HardwareModelDirective.addModel
      "iiwa"
      "package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
      demoIiwaJointPositions,
    HardwareModelDirective.addFrame
      "iiwa_on_world"
      "world"
      { y := -0.7, z := 0.1 }
      { z := 90.0 },
    HardwareModelDirective.addWeld "iiwa_on_world" "iiwa::base",
    HardwareModelDirective.addModel
      "wsg"
      "package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf",
    HardwareModelDirective.addFrame
      "wsg_on_iiwa"
      "iiwa_link_7"
      { z := 0.114 }
      { x := 90.0, z := 90.0 },
    HardwareModelDirective.addWeld "wsg_on_iiwa" "wsg::body",
    {
      kind := .addModel
      name := "bell_pepper"
      file := yellowBellPepperSdf
      parent := "amazon_table::amazon_table"
      translation := { y := 0.10, z := 0.20 }
    }
  ]

def driverTrafficBus : HardwareLcmBusConfig :=
  {
    name := "driver_traffic"
    lcmUrl := "udpm://239.241.129.92:20185?ttl=0"
  }

def demoScenario : HardwareScenario :=
  {
    label := "Demo"
    sceneGraphConfig := { defaultProximityCompliance? := some .compliant }
    directives := demoDirectives
    lcmBuses := #[
      { name := "default" },
      driverTrafficBus
    ]
    cameras := #[
      {
        key := "oracular_view"
        name := "camera_0"
        X_PB := {
          translation := { x := 1.5, y := 0.8, z := 1.25 }
          rotationAxis := SceneVec3.unitZ
          rotationAngle := 0.0
        }
      }
    ]
    modelDrivers := #[
      { modelName := "iiwa", config := .iiwa "wsg" "driver_traffic" },
      { modelName := "wsg", config := .schunkWsg "driver_traffic" }
    ]
    initialPosition := #[
      { modelName := "wsg", jointName := "left_finger_sliding_joint", positions := #[-0.02] },
      { modelName := "wsg", jointName := "right_finger_sliding_joint", positions := #[0.02] }
    ]
  }

def smokeDuration : Float := 0.0625

def smokeScenario (scenario : HardwareScenario) : HardwareScenario :=
  scenario.withSimulationDuration smokeDuration

def graphvizOptions : Array (String × String) :=
  #[("plant/split", "I/O")]

structure HardwareSimResult where
  references : Array DrakeReference
  scenario : HardwareScenario
  plan : HardwareSimulationPlan
  graphvizOptions : Array (String × String) := #[]
  deriving Repr, Inhabited

def buildScenario?
    (scenario : HardwareScenario)
    (graphvizRequested : Bool := false) : Except String HardwareSimResult := do
  let plan ← scenario.plan? graphvizRequested
  pure {
    references := drakeReferences
    scenario := scenario
    plan := plan
    graphvizOptions := if graphvizRequested then graphvizOptions else #[]
  }

def buildDefaultsSmoke? : Except String HardwareSimResult :=
  buildScenario? (smokeScenario defaultScenario)

def buildOneOfEverythingSmoke? : Except String HardwareSimResult :=
  buildScenario? (smokeScenario oneOfEverythingScenario)

def buildDemoSmoke? : Except String HardwareSimResult :=
  buildScenario? (smokeScenario demoScenario)

def buildGraphvizSmoke? : Except String HardwareSimResult :=
  buildScenario? (smokeScenario defaultScenario) true

/-! ## Primitive-driven `Simulator.AdvanceTo` -/

inductive HardwareExecutableModelKind where
  | pendulum
  | iiwa
  | wsg
  | sceneFreeBody
  deriving Repr, BEq, Inhabited

namespace HardwareExecutableModelKind

def label : HardwareExecutableModelKind → String
  | .pendulum => "pendulum"
  | .iiwa => "iiwa"
  | .wsg => "wsg"
  | .sceneFreeBody => "scene_free_body"

end HardwareExecutableModelKind

structure UnsupportedHardwareModel where
  modelName : String
  file : String
  reason : String
  deriving Repr, BEq, Inhabited

structure HardwarePhysicsExecution where
  scenario : HardwareScenario
  modelName : String
  kind : HardwareExecutableModelKind
  modelUri : String := ""
  driverLabel : String := ""
  t0 : Float
  t1 : Float
  stateCoordinateNames : Array String
  initialState : Array Float
  finalState : Array Float
  moves : Array SkeletonMove
  fullPhysics? : Option FullPhysicsResult := none
  fullPlantStep? : Option FullMultibodyPlantStep := none
  primitivePlant? : Option FullPlantPrimitivePhysics := none
  unsupportedModels : Array UnsupportedHardwareModel := #[]
  deriving Repr, Inhabited

private def pendulumUrdf : String :=
  "package://drake/examples/pendulum/Pendulum.urdf"

private def iiwaPrimitiveCollisionUrdf : String :=
  "package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"

private def schunkWsgWithTipSdf : String :=
  "package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf"

private def isAddModelDirective (directive : HardwareModelDirective) : Bool :=
  directive.kind == HardwareDirectiveKind.addModel

private def isAmazonTableDirective (directive : HardwareModelDirective) : Bool :=
  isAddModelDirective directive && directive.name == "amazon_table" &&
    directive.file == amazonTableSdf

private def isBellPepperDirective (directive : HardwareModelDirective) : Bool :=
  isAddModelDirective directive && directive.name == "bell_pepper" &&
    directive.file == yellowBellPepperSdf &&
    directive.parent == "amazon_table::amazon_table"

private def isPendulumDirective (directive : HardwareModelDirective) : Bool :=
  isAddModelDirective directive && directive.file == pendulumUrdf

private def isIiwaDirective (directive : HardwareModelDirective) : Bool :=
  isAddModelDirective directive &&
    (directive.file == iiwaPrimitiveCollisionUrdf ||
      directive.file == Tyr.EventSkeleton.Examples.KukaIiwaArm.iiwaModelUrl)

private def isWsgDirective (directive : HardwareModelDirective) : Bool :=
  isAddModelDirective directive && directive.file == schunkWsgWithTipSdf

private def isExecutableDirective (directive : HardwareModelDirective) : Bool :=
  isPendulumDirective directive || isIiwaDirective directive || isWsgDirective directive

private def executableDirectives (scenario : HardwareScenario) :
    Array HardwareModelDirective :=
  scenario.directives.filter isExecutableDirective

private def findAddModelDirective? (scenario : HardwareScenario) (modelName : String) :
    Option HardwareModelDirective :=
  scenario.directives.find? (fun directive =>
    isAddModelDirective directive && directive.name == modelName)

private def hasRegisteredPepperTableScene (scenario : HardwareScenario) : Bool :=
  (scenario.directives.any isAmazonTableDirective) &&
  (scenario.directives.any isBellPepperDirective)

private def isRegisteredSceneDirective
    (scenario : HardwareScenario)
    (directive : HardwareModelDirective) : Bool :=
  hasRegisteredPepperTableScene scenario &&
    (isAmazonTableDirective directive || isBellPepperDirective directive)

private def unsupportedReason (directive : HardwareModelDirective) : String :=
  if directive.file.contains "iiwa" then
    "IIWA needs an articulated multibody manipulator primitive"
  else if directive.file.contains "wsg" || directive.file.contains "schunk_wsg" then
    "Schunk WSG model URI is not registered with a gripper actuator/contact primitive"
  else if directive.file.contains ".sdf" then
    "SDF model currently contributes provider geometry only; no dynamics primitive is registered"
  else if directive.file.contains ".urdf" then
    "URDF model family has no registered dynamics primitive"
  else
    "model family has no registered dynamics primitive"

def unsupportedModelDirectives (scenario : HardwareScenario) :
    Array UnsupportedHardwareModel :=
  (scenario.directives.filter
    (fun directive =>
      isAddModelDirective directive &&
      !isExecutableDirective directive &&
      !isRegisteredSceneDirective scenario directive)).map
    (fun directive => {
      modelName := directive.name
      file := directive.file
      reason := unsupportedReason directive
    })

private def driverForModel? (scenario : HardwareScenario) (modelName : String) :
    Option HardwareDriverConfig :=
  (scenario.modelDrivers.find? (fun driver => driver.modelName == modelName)).map
    (fun driver => driver.config)

private def thetaFromPosition? (position : HardwareJointPosition) : Option Float :=
  if position.jointName == "theta" && !position.positions.isEmpty then
    some (position.positions.getD 0 0.0)
  else
    none

private def thetaFromPositions? (modelName : String) (positions : Array HardwareJointPosition) :
    Option Float := Id.run do
  for position in positions do
    if position.modelName == modelName || position.modelName.isEmpty then
      match thetaFromPosition? position with
      | some theta => return some theta
      | none => pure ()
  return none

private def pendulumInitialState
    (scenario : HardwareScenario)
    (directive : HardwareModelDirective) :
    Tyr.EventSkeleton.Examples.Pendulum.PendulumState :=
  let theta? :=
    match thetaFromPositions? directive.name scenario.initialPosition with
    | some theta => some theta
    | none => thetaFromPositions? directive.name directive.defaultJointPositions
  {
    theta := theta?.getD Tyr.EventSkeleton.Examples.Pendulum.defaultState.theta
    thetadot := 0.0
  }

private def jointValueFromPositions?
    (modelName jointName : String)
    (positions : Array HardwareJointPosition) : Option Float := Id.run do
  for position in positions do
    if (position.modelName == modelName || position.modelName.isEmpty) &&
        position.jointName == jointName && !position.positions.isEmpty then
      return some (position.positions.getD 0 0.0)
  return none

private def jointPositionsForNames?
    (modelName : String)
    (jointNames : Array String)
    (scenarioPositions defaultPositions : Array HardwareJointPosition) :
    Except String (Array Float) := do
  let mut out : Array Float := #[]
  for jointName in jointNames do
    let q? :=
      match jointValueFromPositions? modelName jointName scenarioPositions with
      | some q => some q
      | none => jointValueFromPositions? modelName jointName defaultPositions
    match q? with
    | some q => out := out.push q
    | none =>
        .error s!"model {modelName}: missing initial position for joint {jointName}"
  pure out

private def iiwaInitialState?
    (scenario : HardwareScenario)
    (directive : HardwareModelDirective) :
    Except String Tyr.EventSkeleton.Examples.KukaIiwaArm.IiwaState := do
  let q ← jointPositionsForNames? directive.name
    Tyr.EventSkeleton.Examples.KukaIiwaArm.jointNames
    scenario.initialPosition
    directive.defaultJointPositions
  let x : Tyr.EventSkeleton.Examples.KukaIiwaArm.IiwaState := {
    q := q
    v := Array.replicate Tyr.EventSkeleton.Examples.KukaIiwaArm.numJoints 0.0
  }
  x.validate? s!"hardware_sim {directive.name} initial state"
  pure x

structure WsgPrimitiveState where
  leftQ : Float := -wsgDefaultFingerPosition
  rightQ : Float := wsgDefaultFingerPosition
  leftV : Float := 0.0
  rightV : Float := 0.0
  deriving Repr, Inhabited

namespace WsgPrimitiveState

def asArray (x : WsgPrimitiveState) : Array Float :=
  #[x.leftQ, x.rightQ, x.leftV, x.rightV]

def q (x : WsgPrimitiveState) : Array Float :=
  #[x.leftQ, x.rightQ]

def v (x : WsgPrimitiveState) : Array Float :=
  #[x.leftV, x.rightV]

def validate? (x : WsgPrimitiveState) (label : String) : Except String Unit := do
  for i in [:x.asArray.size] do
    let value := x.asArray[i]!
    if !value.isFinite then
      .error s!"{label}: WSG state entry {i} must be finite, got {value}"

end WsgPrimitiveState

def wsgStateCoordinateNames : Array String :=
  #[
    "left_finger_sliding_joint",
    "right_finger_sliding_joint",
    "left_finger_sliding_joint_v",
    "right_finger_sliding_joint_v"
  ]

def pepperTableMass : Float := 0.15
def pepperTableGravity : Float := 9.81
def pepperContactRadius : Float := 0.035
def pepperTableContactStiffness : Float := 4000.0
def pepperTableContactDamping : Float := 8.0
def pepperTableFriction : Float := 0.5

def pepperTableStateCoordinateNames : Array String :=
  #[
    "bell_pepper_x",
    "bell_pepper_y",
    "bell_pepper_bottom_z",
    "bell_pepper_vx",
    "bell_pepper_vy",
    "bell_pepper_vz"
  ]

structure PepperTableState where
  x : Float := 0.0
  y : Float := 0.0
  bottomZ : Float := 0.20
  vx : Float := 0.0
  vy : Float := 0.0
  vz : Float := 0.0
  deriving Repr, Inhabited

namespace PepperTableState

def q (x : PepperTableState) : Array Float :=
  #[x.x, x.y, x.bottomZ]

def v (x : PepperTableState) : Array Float :=
  #[x.vx, x.vy, x.vz]

def asArray (x : PepperTableState) : Array Float :=
  x.q ++ x.v

def sphereCenter (x : PepperTableState) : SceneVec3 :=
  { x := x.x, y := x.y, z := x.bottomZ + pepperContactRadius }

def validate? (x : PepperTableState) (label : String) : Except String Unit := do
  for i in [:x.asArray.size] do
    let value := x.asArray[i]!
    if !value.isFinite then
      .error s!"{label}: pepper-table state entry {i} must be finite, got {value}"

end PepperTableState

def pepperTableSceneProvider : SceneGraphProvider :=
  {
    sources := #[
      { id := 910, name := "hardware_sim Demo table+pepper source" }
    ]
    geometries := #[
      {
        id := 9100
        sourceId := 910
        shape := .halfSpace SceneVec3.unitZ {}
        name := "amazon_table::top_half_space"
        properties := {
          roles := #[.proximity, .illustration]
          friction := {
            staticFriction := pepperTableFriction
            dynamicFriction := pepperTableFriction
          }
        }
      },
      {
        id := 9200
        sourceId := 910
        frameId? := some 0
        X_FG := { translation := { z := pepperContactRadius } }
        shape := .sphere pepperContactRadius
        name := "bell_pepper::contact_proxy_sphere"
        properties := {
          roles := #[.proximity, .illustration]
          friction := {
            staticFriction := pepperTableFriction
            dynamicFriction := pepperTableFriction
          }
        }
      }
    ]
    label := "hardware_sim Demo table+pepper SceneGraph primitive"
  }

def pepperTableContactModel : CompliantContactModel :=
  {
    normalStiffness := pepperTableContactStiffness
    normalDamping := pepperTableContactDamping
    tangentDamping := 0.2
    tangentDamping2 := 0.2
    friction := {
      staticFriction := pepperTableFriction
      dynamicFriction := pepperTableFriction
    }
    label := "hardware_sim bell pepper table contact"
  }

def pepperTableInitialState? (scenario : HardwareScenario) :
    Except String PepperTableState := do
  let table? := findAddModelDirective? scenario "amazon_table"
  let pepper? := findAddModelDirective? scenario "bell_pepper"
  match table?, pepper? with
  | some table, some pepper =>
      if !isAmazonTableDirective table then
        .error s!"scenario {scenario.label}: amazon_table is not the registered table SDF"
      if !isBellPepperDirective pepper then
        .error s!"scenario {scenario.label}: bell_pepper is not posed relative to amazon_table"
      let x : PepperTableState := {
        x := pepper.translation.x
        y := pepper.translation.y
        bottomZ := pepper.translation.z
        vx := 0.0
        vy := 0.0
        vz := 0.0
      }
      x.validate? s!"scenario {scenario.label} bell_pepper initial state"
      pure x
  | _, _ =>
      .error s!"scenario {scenario.label}: registered scene primitive requires amazon_table and bell_pepper directives"

def pepperTableContactCandidate? (x : PepperTableState) :
    Except String ContactCandidate :=
  sphereHalfSpaceContactCandidate?
    pepperTableSceneProvider
    9200
    9100
    x.sphereCenter
    x.vz
    #[0.0, 0.0, 1.0]
    #[1.0, 0.0, 0.0]
    #[0.0, 1.0, 0.0]
    "hardware_sim bell pepper table contact candidate"

def pepperTableContactCandidateSet? (x : PepperTableState) :
    Except String ContactCandidateSet := do
  let candidate ← pepperTableContactCandidate? x
  let set := ContactCandidateSet.ofArray #[candidate]
    "hardware_sim bell pepper table contact candidate set"
  set.validate? (some 3)
  pure set

private def pepperTableContactForces? (support : ContactSupport) :
    Except String (Array ContactForceScalars) := do
  let selected ← support.selectedCandidates?
  let mut forces : Array ContactForceScalars := #[]
  for candidate in selected do
    forces := forces.push {
      candidateId := candidate.id
      normalForce := pepperTableMass * pepperTableGravity
      tangentForce := 0.0
      tangentForce2 := 0.0
      mode := candidate.mode
      label := "hardware_sim bell pepper static support force"
    }
  pure forces

def pepperTableFullPhysicsPrimitives? (x : PepperTableState) :
    Except String FullPhysicsPrimitives := do
  x.validate? "hardware_sim bell pepper table primitive state"
  let set ← pepperTableContactCandidateSet? x
  let support :=
    set.selectByDistance 0.0 "hardware_sim bell pepper active table contacts"
      |>.classifyCandidates 0.0 1.0e-9
  let forces ← pepperTableContactForces? support
  pure {
    massMatrix := FloatMatrix.diagonal
      #[pepperTableMass, pepperTableMass, pepperTableMass]
    qdot := x.v
    actuationForces := #[0.0, 0.0, 0.0]
    generalizedForceContributions :=
      #[GeneralizedForceContribution.ofForce
          #[0.0, 0.0, -pepperTableMass * pepperTableGravity]
          "bell pepper gravity generalized force"
          "HardwareSim"]
    contactCandidates := set.candidates
    sourceContactCandidateCount? := some set.totalCandidates
    supportPolicy := .threshold 0.0
    contactForceSource := .precomputed
    contactForces := forces
    distanceTol := 0.0
    tangentVelocityTol := 1.0e-9
    label := "hardware_sim bell pepper table free-body primitive"
  }

def pepperTableFullPhysicsPrimitiveProvider
    (label : String := "hardware_sim bell pepper table full physics provider") :
    FullPhysicsPrimitiveProvider PepperTableState :=
  {
    label := label
    primitivesAt? := fun x => pepperTableFullPhysicsPrimitives? x
  }

private def pepperTablePlantStep
    (scenario : HardwareScenario)
    (x0 : PepperTableState) :
    FullMultibodyPlantStep :=
  {
    model := {
      modelName := "bell_pepper"
      modelUri := yellowBellPepperSdf
      numPositions := 3
      numVelocities := 3
      numActuatedDofs := 0
      label := "hardware_sim Demo bell pepper free-body primitive model"
    }
    config := {
      timeStep := scenario.plantConfig.timeStep
      penetrationAllowance := 0.0
      stictionTolerance := scenario.plantConfig.stictionTolerance
      contactApproximation := .sap
    }
    q0 := x0.q
    v0 := x0.v
    actuation := #[]
    t0 := 0.0
    t1 := scenario.simulationDuration
    ground? := some {
      visualName := "amazon_table::top"
      collisionName := "amazon_table::top_half_space"
      friction := {
        staticFriction := pepperTableFriction
        dynamicFriction := pepperTableFriction
      }
    }
    label := "hardware_sim Demo bell pepper table full plant primitive"
  }

private def pepperTableFinalState
    (x0 : PepperTableState)
    (dt : Float) : Array Float :=
  let zFree := x0.bottomZ + x0.vz * dt - 0.5 * pepperTableGravity * dt * dt
  let vzFree := x0.vz - pepperTableGravity * dt
  let bottomZ := max 0.0 zFree
  let vz := if zFree <= 0.0 then 0.0 else vzFree
  #[x0.x + x0.vx * dt, x0.y + x0.vy * dt, bottomZ, x0.vx, x0.vy, vz]

private def wsgInitialState?
    (scenario : HardwareScenario)
    (directive : HardwareModelDirective) :
    Except String WsgPrimitiveState := do
  let left :=
    match jointValueFromPositions? directive.name "left_finger_sliding_joint"
        scenario.initialPosition with
    | some q => q
    | none =>
        (jointValueFromPositions? directive.name "left_finger_sliding_joint"
          directive.defaultJointPositions).getD (-wsgDefaultFingerPosition)
  let right :=
    match jointValueFromPositions? directive.name "right_finger_sliding_joint"
        scenario.initialPosition with
    | some q => q
    | none =>
        (jointValueFromPositions? directive.name "right_finger_sliding_joint"
          directive.defaultJointPositions).getD wsgDefaultFingerPosition
  let x : WsgPrimitiveState := {
    leftQ := left
    rightQ := right
    leftV := 0.0
    rightV := 0.0
  }
  x.validate? s!"hardware_sim {directive.name} initial state"
  pure x

private def hardwareIiwaProvider :
    Tyr.EventSkeleton.Examples.KukaIiwaArm.IiwaMultibodyProviderData :=
  {
    Tyr.EventSkeleton.Examples.KukaIiwaArm.drakeTestProvider with
    modelUri := iiwaPrimitiveCollisionUrdf
    label := "hardware_sim Demo IIWA primitive provider"
  }

private def validatePendulumDriver?
    (scenario : HardwareScenario)
    (modelName : String) :
    Except String Tyr.EventSkeleton.Examples.Pendulum.PendulumInput := do
  match driverForModel? scenario modelName with
  | none => pure Tyr.EventSkeleton.Examples.Pendulum.defaultInput
  | some .zeroForce => pure Tyr.EventSkeleton.Examples.Pendulum.defaultInput
  | some driver =>
      .error s!"scenario {scenario.label}: model {modelName} uses unsupported driver {driver.label}; only ZeroForceDriver can currently lower to the pendulum primitive"

private def validateIiwaDriver?
    (scenario : HardwareScenario)
    (modelName : String) : Except String String := do
  match driverForModel? scenario modelName with
  | some (.iiwa hand bus) =>
      if hand.isEmpty then
        .error s!"scenario {scenario.label}: IiwaDriver for {modelName} must name a hand model"
      if !scenario.hasBus bus then
        .error s!"scenario {scenario.label}: IiwaDriver for {modelName} references missing LCM bus {bus}"
      pure s!"IiwaDriver(hand={hand}, bus={bus})"
  | some .zeroForce =>
      pure "ZeroForceDriver"
  | some driver =>
      .error s!"scenario {scenario.label}: model {modelName} uses unsupported driver {driver.label}; only IiwaDriver can lower to the IIWA primitive"
  | none =>
      pure "no-driver"

private def validateWsgDriver?
    (scenario : HardwareScenario)
    (modelName : String) : Except String String := do
  match driverForModel? scenario modelName with
  | some (.schunkWsg bus) =>
      if !scenario.hasBus bus then
        .error s!"scenario {scenario.label}: SchunkWsgDriver for {modelName} references missing LCM bus {bus}"
      pure s!"SchunkWsgDriver(bus={bus})"
  | some .zeroForce =>
      pure "ZeroForceDriver"
  | some driver =>
      .error s!"scenario {scenario.label}: model {modelName} uses unsupported driver {driver.label}; only SchunkWsgDriver can lower to the WSG primitive"
  | none =>
      pure "no-driver"

private def hardwareIiwaPlantStep
    (scenario : HardwareScenario)
    (directive : HardwareModelDirective)
    (x0 : Tyr.EventSkeleton.Examples.KukaIiwaArm.IiwaState)
    (controllerOutput : JointTorqueControllerOutput) :
    FullMultibodyPlantStep :=
  {
    model := {
      modelName := directive.name
      modelUri := directive.file
      numPositions := Tyr.EventSkeleton.Examples.KukaIiwaArm.numJoints
      numVelocities := Tyr.EventSkeleton.Examples.KukaIiwaArm.numJoints
      numActuatedDofs := Tyr.EventSkeleton.Examples.KukaIiwaArm.numJoints
      label := "hardware_sim Demo IIWA primitive model"
    }
    config := {
      timeStep := scenario.plantConfig.timeStep
      penetrationAllowance := 1.0e-3
      stictionTolerance := scenario.plantConfig.stictionTolerance
      contactApproximation := .sap
    }
    q0 := x0.q
    v0 := x0.v
    actuation := controllerOutput.controlTorque
    t0 := 0.0
    t1 := scenario.simulationDuration
    label := "hardware_sim Demo IIWA full plant primitive"
  }

private def wsgActuationForces (x : WsgPrimitiveState) : Array Float :=
  let kp := 10000.0
  let kd := 1.0
  #[
    kp * ((-wsgDefaultFingerPosition) - x.leftQ) - kd * x.leftV,
    kp * (wsgDefaultFingerPosition - x.rightQ) - kd * x.rightV
  ]

private def hardwareWsgPlantStep
    (scenario : HardwareScenario)
    (directive : HardwareModelDirective)
    (x0 : WsgPrimitiveState) :
    FullMultibodyPlantStep :=
  {
    model := {
      modelName := directive.name
      modelUri := directive.file
      numPositions := 2
      numVelocities := 2
      numActuatedDofs := 2
      label := "hardware_sim Demo WSG primitive model"
    }
    config := {
      timeStep := scenario.plantConfig.timeStep
      penetrationAllowance := 1.0e-3
      stictionTolerance := scenario.plantConfig.stictionTolerance
      contactApproximation := .sap
    }
    q0 := x0.q
    v0 := x0.v
    actuation := wsgActuationForces x0
    t0 := 0.0
    t1 := scenario.simulationDuration
    label := "hardware_sim Demo WSG full plant primitive"
  }

private def wsgFullPhysicsPrimitives (x0 : WsgPrimitiveState) :
    FullPhysicsPrimitives :=
  {
    massMatrix := Tyr.EventSkeleton.FloatMatrix.diagonal #[
      Tyr.EventSkeleton.Examples.SimpleGripper.params.fingerMass,
      Tyr.EventSkeleton.Examples.SimpleGripper.params.fingerMass
    ]
    qdot := x0.v
    actuationForces := wsgActuationForces x0
    biasForces := #[]
    contactCandidates := #[]
    supportPolicy := .fullSupport
    contactForceSource := .precomputed
    contactForces := #[]
    label := "hardware_sim WSG gripper full physics primitive"
  }

def wsgFullPhysicsPrimitives? (x0 : WsgPrimitiveState) :
    Except String FullPhysicsPrimitives := do
  x0.validate? "hardware_sim WSG primitive state"
  pure (wsgFullPhysicsPrimitives x0)

def wsgFullPhysicsPrimitiveProvider
    (label : String := "hardware_sim WSG full physics provider") :
    FullPhysicsPrimitiveProvider WsgPrimitiveState :=
  {
    label := label
    primitivesAt? := fun x => wsgFullPhysicsPrimitives? x
  }

private def wsgFinalState
    (x0 : WsgPrimitiveState)
    (derivative : ManipulatorDerivative)
    (dt : Float) : Array Float :=
  let q1 := FloatArray.add x0.q (FloatArray.scale dt derivative.qdot)
  let v1 := FloatArray.add x0.v (FloatArray.scale dt derivative.vdot)
  q1 ++ v1

private def iiwaFinalState
    (x0 : Tyr.EventSkeleton.Examples.KukaIiwaArm.IiwaState)
    (derivative : ManipulatorDerivative)
    (dt : Float) : Array Float :=
  let q1 := FloatArray.add x0.q (FloatArray.scale dt derivative.qdot)
  let v1 := FloatArray.add x0.v (FloatArray.scale dt derivative.vdot)
  q1 ++ v1

private def runIiwaPhysics?
    (scenario : HardwareScenario)
    (directive : HardwareModelDirective) :
    Except String HardwarePhysicsExecution := do
  let driverLabel ← validateIiwaDriver? scenario directive.name
  let x0 ← iiwaInitialState? scenario directive
  let desired : Tyr.EventSkeleton.Examples.KukaIiwaArm.IiwaState := {
    q := x0.q
    v := Array.replicate Tyr.EventSkeleton.Examples.KukaIiwaArm.numJoints 0.0
  }
  let controllerOutput ←
    Tyr.EventSkeleton.Examples.KukaIiwaArm.evaluateTorqueController?
      Tyr.EventSkeleton.Examples.KukaIiwaArm.gravityOnlyGains
      hardwareIiwaProvider
      x0
      desired
      Tyr.EventSkeleton.Examples.KukaIiwaArm.zeroTorque
  let primitives ←
    Tyr.EventSkeleton.Examples.KukaIiwaArm.fullPhysicsPrimitivesFromController?
      hardwareIiwaProvider x0 controllerOutput
  let plantStep := hardwareIiwaPlantStep scenario directive x0 controllerOutput
  let primitivePlant : FullPlantPrimitivePhysics := {
    step := plantStep
    primitives := primitives
    intervalVertex := Tyr.EventSkeleton.Examples.KukaIiwaArm.fullPhysicsIntervalVertex
    label := "hardware_sim Demo IIWA primitive plant"
  }
  let fullPhysics ← primitivePlant.solve?
  let finalState := iiwaFinalState x0 fullPhysics.derivative scenario.simulationDuration
  pure {
    scenario := scenario
    modelName := directive.name
    kind := .iiwa
    modelUri := directive.file
    driverLabel := driverLabel
    t0 := 0.0
    t1 := scenario.simulationDuration
    stateCoordinateNames := Tyr.EventSkeleton.Examples.KukaIiwaArm.stateCoordinateNames
    initialState := x0.asArray
    finalState := finalState
    moves :=
      Tyr.EventSkeleton.Examples.KukaIiwaArm.controllerGraph.moves ++
      #[fullPhysics.supportMove, fullPhysics.move]
    fullPhysics? := some fullPhysics
    fullPlantStep? := some plantStep
    primitivePlant? := some primitivePlant
    unsupportedModels := unsupportedModelDirectives scenario
  }

private def runWsgPhysics?
    (scenario : HardwareScenario)
    (directive : HardwareModelDirective) :
    Except String HardwarePhysicsExecution := do
  let driverLabel ← validateWsgDriver? scenario directive.name
  let x0 ← wsgInitialState? scenario directive
  let primitive ←
    (wsgFullPhysicsPrimitiveProvider
      "hardware_sim Demo WSG full physics provider").validPrimitivesAt? x0
  let plantStep := hardwareWsgPlantStep scenario directive x0
  let primitivePlant : FullPlantPrimitivePhysics := {
    step := plantStep
    primitives := primitive
    intervalVertex := 5603
    label := "hardware_sim Demo WSG primitive plant"
  }
  let fullPhysics ← primitivePlant.solve?
  let finalState := wsgFinalState x0 fullPhysics.derivative scenario.simulationDuration
  pure {
    scenario := scenario
    modelName := directive.name
    kind := .wsg
    modelUri := directive.file
    driverLabel := driverLabel
    t0 := 0.0
    t1 := scenario.simulationDuration
    stateCoordinateNames := wsgStateCoordinateNames
    initialState := x0.asArray
    finalState := finalState
    moves := #[fullPhysics.supportMove, fullPhysics.move]
    fullPhysics? := some fullPhysics
    fullPlantStep? := some plantStep
    primitivePlant? := some primitivePlant
    unsupportedModels := unsupportedModelDirectives scenario
  }

private def hardwareSceneIntervalVertex : VertexId := 5604

private def runPepperTablePhysics?
    (scenario : HardwareScenario) :
    Except String HardwarePhysicsExecution := do
  let x0 ← pepperTableInitialState? scenario
  let primitives ←
    (pepperTableFullPhysicsPrimitiveProvider
      "hardware_sim Demo bell pepper table full physics provider").validPrimitivesAt? x0
  let plantStep := pepperTablePlantStep scenario x0
  let primitivePlant : FullPlantPrimitivePhysics := {
    step := plantStep
    primitives := primitives
    intervalVertex := hardwareSceneIntervalVertex
    label := "hardware_sim Demo bell pepper table primitive plant"
  }
  let fullPhysics ← primitivePlant.solve?
  pure {
    scenario := scenario
    modelName := "bell_pepper"
    kind := .sceneFreeBody
    modelUri := yellowBellPepperSdf
    driverLabel := "SceneGraph free-body contact on amazon_table"
    t0 := 0.0
    t1 := scenario.simulationDuration
    stateCoordinateNames := pepperTableStateCoordinateNames
    initialState := x0.asArray
    finalState := pepperTableFinalState x0 scenario.simulationDuration
    moves := #[fullPhysics.supportMove, fullPhysics.move]
    fullPhysics? := some fullPhysics
    fullPlantStep? := some plantStep
    primitivePlant? := some primitivePlant
    unsupportedModels := unsupportedModelDirectives scenario
  }

private def runPendulumPhysics?
    (scenario : HardwareScenario)
    (directive : HardwareModelDirective) :
    Except String HardwarePhysicsExecution := do
  let input ← validatePendulumDriver? scenario directive.name
  let params : Tyr.EventSkeleton.Examples.Pendulum.PendulumParams :=
    { Tyr.EventSkeleton.Examples.Pendulum.params with
      stepSize := scenario.simulatorConfig.maxStepSize }
  let x0 := pendulumInitialState scenario directive
  let run ← Tyr.EventSkeleton.Examples.Pendulum.solvePassive?
    params x0 0.0 scenario.simulationDuration input
  pure {
    scenario := scenario
    modelName := directive.name
    kind := .pendulum
    modelUri := directive.file
    driverLabel := (driverForModel? scenario directive.name).map (fun driver => driver.label)
      |>.getD "no-driver"
    t0 := run.t0
    t1 := run.t1
    stateCoordinateNames := Tyr.EventSkeleton.Examples.Pendulum.stateCoordinateNames
    initialState := Tyr.EventSkeleton.Examples.Pendulum.stateAsArray run.initialState
    finalState := Tyr.EventSkeleton.Examples.Pendulum.stateAsArray run.finalState
    moves := run.moves
    unsupportedModels := unsupportedModelDirectives scenario
  }

private def runExecutableDirective?
    (scenario : HardwareScenario)
    (directive : HardwareModelDirective) :
    Except String HardwarePhysicsExecution := do
  if isPendulumDirective directive then
    runPendulumPhysics? scenario directive
  else if isIiwaDirective directive then
    runIiwaPhysics? scenario directive
  else if isWsgDirective directive then
    runWsgPhysics? scenario directive
  else
    .error s!"scenario {scenario.label}: directive {directive.name} has no executable physics primitive"

structure HardwarePhysicsExecutionSet where
  scenario : HardwareScenario
  executions : Array HardwarePhysicsExecution
  unsupportedModels : Array UnsupportedHardwareModel := #[]
  moves : Array SkeletonMove := #[]
  deriving Repr, Inhabited

def runScenarioPhysics?
    (scenario : HardwareScenario) :
    Except String HardwarePhysicsExecution := do
  scenario.validate?
  if !Float.isFinite scenario.simulationDuration then
    .error s!"scenario {scenario.label}: executable physics requires a finite simulation_duration"
  if scenario.simulationDuration < 0.0 then
    .error s!"scenario {scenario.label}: simulation_duration must be nonnegative"
  let directives := executableDirectives scenario
  if directives.isEmpty then
    if hasRegisteredPepperTableScene scenario then
      runPepperTablePhysics? scenario
    else
      let unsupported := unsupportedModelDirectives scenario
      .error s!"scenario {scenario.label}: no executable physics primitive is registered; unsupported model count = {unsupported.size}"
  else
    let directive := directives[0]!
    runExecutableDirective? scenario directive

def runScenarioPhysicsAll?
    (scenario : HardwareScenario) :
    Except String HardwarePhysicsExecutionSet := do
  scenario.validate?
  if !Float.isFinite scenario.simulationDuration then
    .error s!"scenario {scenario.label}: executable physics requires a finite simulation_duration"
  if scenario.simulationDuration < 0.0 then
    .error s!"scenario {scenario.label}: simulation_duration must be nonnegative"
  let directives := executableDirectives scenario
  if directives.isEmpty && !hasRegisteredPepperTableScene scenario then
    let unsupported := unsupportedModelDirectives scenario
    .error s!"scenario {scenario.label}: no executable physics primitive is registered; unsupported model count = {unsupported.size}"
  let mut executions : Array HardwarePhysicsExecution := #[]
  let mut moves : Array SkeletonMove := #[]
  for directive in directives do
    let execution ← runExecutableDirective? scenario directive
    executions := executions.push execution
    moves := moves ++ execution.moves
  if hasRegisteredPepperTableScene scenario then
    let sceneExecution ← runPepperTablePhysics? scenario
    executions := executions.push sceneExecution
    moves := moves ++ sceneExecution.moves
  pure {
    scenario := scenario
    executions := executions
    unsupportedModels := unsupportedModelDirectives scenario
    moves := moves
  }

def oneOfEverythingPhysicsScenario : HardwareScenario :=
  {
    smokeScenario oneOfEverythingScenario with
    initialPosition := #[
      { modelName := "alice", jointName := "theta", positions := #[0.1] }
    ]
  }

def runOneOfEverythingPhysicsSmoke? : Except String HardwarePhysicsExecution :=
  runScenarioPhysics? oneOfEverythingPhysicsScenario

/-! ## robot_commander.py -/

def lcmUrl : String := "udpm://239.241.129.92:20185?ttl=0"
def iiwaCommandChannel : String := "IIWA_COMMAND"
def wsgCommandChannel : String := "SCHUNK_WSG_COMMAND"
def commandHz : Float := 20.0
def cycleTime : Float := 10.0
def iiwaMaxDeflection : Float := 0.4
def wsgQ0 : Float := wsgDefaultFingerPosition
def wsgMaxDeflection : Float := 0.02

def iiwaQ0 : Array Float :=
  #[-0.2, 0.79, 0.32, -1.76, -0.36, 0.64, -0.73]

structure RobotCommand where
  commandIndex : Nat
  sine : Float
  iiwaJointPosition : Array Float
  wsgTargetPositionMm : Float
  iiwaChannel : String := iiwaCommandChannel
  wsgChannel : String := wsgCommandChannel
  lcmUrl : String := lcmUrl
  deriving Repr, Inhabited

namespace RobotCommand

def isFinite (cmd : RobotCommand) : Bool :=
  Float.isFinite cmd.sine &&
  Float.isFinite cmd.wsgTargetPositionMm &&
  cmd.iiwaJointPosition.all Float.isFinite

end RobotCommand

def commandPhase (i : Nat) : Float :=
  2.0 * pi * (Float.ofNat i) / (cycleTime * commandHz)

def robotCommandAt (i : Nat) : RobotCommand :=
  let sine := Float.sin (commandPhase i)
  {
    commandIndex := i
    sine := sine
    iiwaJointPosition := iiwaQ0.map (fun q => q + sine * iiwaMaxDeflection)
    wsgTargetPositionMm := 1000.0 * (wsgQ0 + sine * wsgMaxDeflection)
  }

def firstUnitTestCommand : RobotCommand :=
  robotCommandAt 0

def quarterCycleCommand : RobotCommand :=
  robotCommandAt 50

structure HardwareSimEndToEndResult where
  references : Array DrakeReference
  setup : HardwareSimResult
  executions : HardwarePhysicsExecutionSet
  robotCommands : Array RobotCommand
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildEndToEnd?
    (scenario : HardwareScenario := smokeScenario demoScenario)
    (graphvizRequested : Bool := false) :
    Except String HardwareSimEndToEndResult := do
  let setup ← buildScenario? scenario graphvizRequested
  let executions ← runScenarioPhysicsAll? scenario
  let moves := setup.plan.graph.moves ++ executions.moves
  pure {
    references := drakeReferences
    setup := setup
    executions := executions
    robotCommands := #[firstUnitTestCommand, quarterCycleCommand]
    moves := moves
  }

end Tyr.EventSkeleton.Examples.HardwareSim
