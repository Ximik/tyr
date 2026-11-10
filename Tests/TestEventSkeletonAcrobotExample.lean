import LeanTest
import Tyr.EventSkeleton.Examples.Acrobot

namespace Tests.EventSkeletonAcrobotExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.Acrobot

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def pi : Float := 3.14159265358979323846

private def twoPi : Float := 2.0 * pi

private def wrapTo (x low high : Float) : Float := Id.run do
  let width := high - low
  if width <= 0.0 then
    return x
  let mut y := x
  for _ in [:32] do
    if y < low then
      y := y + width
    else if y >= high then
      y := y - width
  return y

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertError {α : Type} (res : Except String α) (label : String) : IO String := do
  match res with
  | .ok _ => LeanTest.fail s!"{label}: expected error"
  | .error msg => pure msg

private def assertArrayNear
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

private def assertFullPhysicsMatches
    (result : FullPhysicsResult)
    (x : AcrobotState)
    (u : AcrobotInput)
    (label : String) : IO Unit := do
  let expectedDx ← assertOk (derivative? params u x) s!"{label} expected derivative"
  LeanTest.assertTrue (result.support.policy == SupportPolicy.fullSupport)
    s!"{label}: no-contact plant should use exact full support"
  LeanTest.assertEqual result.support.totalCandidates 0
    s!"{label}: no contact candidates expected"
  LeanTest.assertEqual result.contactForces.size 0
    s!"{label}: no contact forces expected"
  LeanTest.assertTrue (result.supportMove.exactness == MoveExactness.exact)
    s!"{label}: empty full-support selection should be exact"
  assertArrayNear result.equation.massMatrix[0]!
    (massMatrix params x)[0]! 1.0e-12
    s!"{label}: mass matrix row 0 should come from the Acrobot formula"
  assertArrayNear result.equation.massMatrix[1]!
    (massMatrix params x)[1]! 1.0e-12
    s!"{label}: mass matrix row 1 should come from the Acrobot formula"
  assertArrayNear result.equation.biasForces
    (dynamicsBiasTerm params x) 1.0e-12
    s!"{label}: bias should use Coriolis, gravity, and damping terms"
  assertArrayNear result.generalizedForces
    (inputGeneralizedForces u) 1.0e-12
    s!"{label}: generalized force should match elbow actuation"
  assertArrayNear result.derivative.qdot
    (qdotAsArray x) 1.0e-12
    s!"{label}: qdot should match the benchmark generalized velocity"
  assertArrayNear result.derivative.vdot
    #[expectedDx.theta1dot, expectedDx.theta2dot] 1.0e-12
    s!"{label}: vdot should match the exact Acrobot derivative"

@[test]
def testDrakeReferencesAndNamedVectorsAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/Acrobot.urdf"))
    "Acrobot example should reference Drake's URDF model"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/Acrobot.sdf"))
    "Acrobot example should reference Drake's SDF model"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/Acrobot_no_collision.urdf"))
    "Acrobot example should reference Drake's no-collision URDF variant"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/acrobot_plant.cc"))
    "Acrobot example should reference Drake's plant implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/acrobot_plant.h"))
    "Acrobot example should reference Drake's plant declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/acrobot_geometry.h"))
    "Acrobot example should reference Drake's AcrobotGeometry declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/acrobot_geometry.cc"))
    "Acrobot example should reference Drake's AcrobotGeometry implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/test/acrobot_geometry_test.cc"))
    "Acrobot example should reference Drake's AcrobotGeometry acceptance test"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/test/acrobot_plant_test.cc"))
    "Acrobot example should reference Drake's plant tests"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/run_passive.cc"))
    "Acrobot example should reference Drake's passive executable"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/run_lqr.cc"))
    "Acrobot example should reference Drake's LQR executable"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/run_lqr_w_estimator.cc"))
    "Acrobot example should reference Drake's estimator-in-the-loop LQR executable"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/run_plant_w_lcm.cc"))
    "Acrobot example should reference Drake's LCM plant executable"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/acrobot_lcm.h"))
    "Acrobot example should reference Drake's LCM system declarations"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/acrobot_lcm.cc"))
    "Acrobot example should reference Drake's LCM system implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/acrobot_state.cc"))
    "Acrobot example should reference Drake's state coordinate implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/acrobot_input.cc"))
    "Acrobot example should reference Drake's input coordinate implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/acrobot_params.cc"))
    "Acrobot example should reference Drake's parameter coordinate implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/run_swing_up.cc"))
    "Acrobot example should reference Drake's swing-up executable"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/spong_controller.h"))
    "Acrobot example should reference Drake's Spong controller"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/spong_controller_params.cc"))
    "Acrobot example should reference Drake's Spong controller parameter index source"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/spong_controller_w_lcm.cc"))
    "Acrobot example should reference Drake's LCM Spong controller executable"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/spong_sim.cc"))
    "Acrobot example should reference Drake's C++ Spong scenario runner"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/spong_sim.py"))
    "Acrobot example should reference Drake's Python Spong scenario runner"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/optimizer_demo.py"))
    "Acrobot example should reference Drake's optimizer demo"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/test/run_swing_up_traj_optimization.cc"))
    "Acrobot example should reference Drake's swing-up direct-collocation regression"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/test/spong_sim_lib_py_test.py"))
    "Acrobot example should reference Drake's Python Spong library regression"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/test/spong_sim_main_test.py"))
    "Acrobot example should reference Drake's Spong subprocess regression"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/test/acrobot_io_test.py"))
    "Acrobot example should reference Drake's scenario IO tests"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/test/acrobot_lcm_msg_generator.cc"))
    "Acrobot example should reference Drake's LCM message generator"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/test/metrics_test.py"))
    "Acrobot example should reference Drake's optimizer metric tests"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/test/multibody_dynamics_test.cc"))
    "Acrobot example should reference Drake's hand-written vs MultibodyPlant dynamics regression"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/acrobot/test/example_stochastic_scenario.yaml"))
    "Acrobot example should reference Drake's stochastic optimizer scenario"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/multibody/acrobot/passive_simulation.cc"))
    "Acrobot example should reference Drake's MultibodyPlant passive benchmark executable"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/multibody/acrobot/run_lqr.cc"))
    "Acrobot example should reference Drake's MultibodyPlant LQR benchmark executable"
  LeanTest.assertEqual stateCoordinateNames #["theta1", "theta2", "theta1dot", "theta2dot"]
    "Acrobot state coordinate order should match Drake's BasicVector"
  LeanTest.assertEqual inputCoordinateNames #["tau"]
    "Acrobot input coordinate order should match Drake's BasicVector"
  LeanTest.assertEqual parameterCoordinateNames
    #["m1", "m2", "l1", "l2", "lc1", "lc2", "Ic1", "Ic2", "b1", "b2", "gravity"]
    "Acrobot parameter coordinate order should match Drake's BasicVector"
  LeanTest.assertEqual spongParameterCoordinateNames #["k_e", "k_p", "k_d", "balancing_threshold"]
    "Spong controller parameter coordinate order should match Drake's BasicVector"
  assertOk
    (spongControllerParamsIndicesBoundary.validate?
      spongParameterCoordinateNames "SpongControllerParamsIndices")
    "Spong controller params index boundary"
  LeanTest.assertEqual spongControllerParamsIndicesBoundary.sourcePath
    "../drake/examples/acrobot/spong_controller_params.cc"
    "Spong parameter coordinate names should be tied to the concrete .cc source"
  LeanTest.assertTrue params.isValid "Default AcrobotParams should match Drake's valid domain"
  LeanTest.assertTrue defaultInput.isValid "Disconnected acrobot input should be valid zero torque"
  LeanTest.assertTrue spongControllerParams.isValid
    "Default Spong controller parameters should match Drake's valid domain"

@[test]
def testGeneratedVectorBoundariesMatchDrakeBasicVectors : IO Unit := do
  assertOk acrobotStateVectorBoundary.validate? "AcrobotState named-vector boundary"
  assertOk acrobotInputVectorBoundary.validate? "AcrobotInput named-vector boundary"
  assertOk acrobotParamsVectorBoundary.validate? "AcrobotParams named-vector boundary"

  LeanTest.assertEqual (AcrobotNamedVectorBoundary.dimension acrobotStateVectorBoundary) 4
    "AcrobotState should expose Drake's four coordinates"
  LeanTest.assertEqual (AcrobotNamedVectorBoundary.dimension acrobotInputVectorBoundary) 1
    "AcrobotInput should expose Drake's scalar elbow torque"
  LeanTest.assertEqual (AcrobotNamedVectorBoundary.dimension acrobotParamsVectorBoundary) 11
    "AcrobotParams should expose Drake's eleven named parameters"

  LeanTest.assertEqual (AcrobotNamedVectorBoundary.indexOf? acrobotStateVectorBoundary "theta1") (some 0)
    "theta1 should be state coordinate 0"
  LeanTest.assertEqual (AcrobotNamedVectorBoundary.indexOf? acrobotStateVectorBoundary "theta2") (some 1)
    "theta2 should be state coordinate 1"
  LeanTest.assertEqual (AcrobotNamedVectorBoundary.indexOf? acrobotStateVectorBoundary "theta1dot") (some 2)
    "theta1dot should be state coordinate 2"
  LeanTest.assertEqual (AcrobotNamedVectorBoundary.indexOf? acrobotStateVectorBoundary "theta2dot") (some 3)
    "theta2dot should be state coordinate 3"
  LeanTest.assertEqual (AcrobotNamedVectorBoundary.indexOf? acrobotInputVectorBoundary "tau") (some 0)
    "tau should be the only AcrobotInput coordinate"
  LeanTest.assertEqual (AcrobotNamedVectorBoundary.indexOf? acrobotParamsVectorBoundary "m1") (some 0)
    "m1 should be parameter coordinate 0"
  LeanTest.assertEqual (AcrobotNamedVectorBoundary.indexOf? acrobotParamsVectorBoundary "gravity") (some 10)
    "gravity should be the final AcrobotParams coordinate"

  LeanTest.assertEqual acrobotStateVectorBoundary.defaults #[0.0, 0.0, 0.0, 0.0]
    "AcrobotState defaults should be zero"
  LeanTest.assertEqual acrobotInputVectorBoundary.defaults #[0.0]
    "AcrobotInput default should be disconnected zero torque"
  LeanTest.assertEqual acrobotParamsVectorBoundary.defaults params.asArray
    "AcrobotParams defaults should match Drake's generated-vector defaults"

  let x ← assertOk (AcrobotState.fromArray? #[1.0, 2.0, 3.0, 4.0])
    "AcrobotState from BasicVector array"
  LeanTest.assertEqual x.asArray #[1.0, 2.0, 3.0, 4.0]
    "AcrobotState array round trip should preserve coordinate order"
  let u ← assertOk (AcrobotInput.fromArray? #[-5.0])
    "AcrobotInput from BasicVector array"
  LeanTest.assertEqual u.asArray #[-5.0]
    "AcrobotInput array round trip should preserve tau"
  let p ← assertOk (AcrobotParams.fromArray? params.asArray)
    "AcrobotParams from BasicVector array"
  LeanTest.assertEqual p.asArray params.asArray
    "AcrobotParams array round trip should preserve parameter order"

  let badStateDim ← assertError (AcrobotState.fromArray? #[1.0, 2.0])
    "AcrobotState dimension check"
  LeanTest.assertTrue (badStateDim.contains "expects 4")
    s!"AcrobotState dimension error should mention expected size, got {badStateDim}"
  let badParamsDomain ← assertError (AcrobotParams.fromArray? ({ params with Ic1 := -1.0 }).asArray)
    "AcrobotParams BasicVector domain check"
  LeanTest.assertTrue (badParamsDomain.contains "BasicVector domain")
    s!"AcrobotParams domain error should mention BasicVector domain, got {badParamsDomain}"

@[test]
def testAcrobotModelAssetBoundaryRecordsUrdfAndSdfIdentity : IO Unit := do
  assertOk acrobotModelAssetBoundary.validate? "Acrobot URDF/SDF asset boundary"
  LeanTest.assertEqual acrobotModelAssetBoundary.modelName "Acrobot"
    "Model asset boundary should retain Drake's model name"
  LeanTest.assertEqual acrobotModelAssetBoundary.urdfPath
    "../drake/examples/acrobot/Acrobot.urdf"
    "Model asset boundary should point at Drake's URDF"
  LeanTest.assertEqual acrobotModelAssetBoundary.sdfPath
    "../drake/examples/acrobot/Acrobot.sdf"
    "Model asset boundary should point at Drake's SDF"
  LeanTest.assertEqual acrobotModelAssetBoundary.noCollisionUrdfPath
    "../drake/examples/acrobot/Acrobot_no_collision.urdf"
    "Model asset boundary should point at Drake's no-collision URDF"
  LeanTest.assertEqual acrobotModelAssetBoundary.linkNames
    #["base_link", "upper_link", "lower_link"]
    "URDF/SDF link order should remain explicit"
  LeanTest.assertEqual acrobotModelAssetBoundary.jointNames
    #["base_weld", "shoulder", "elbow"]
    "URDF/SDF joint order should remain explicit"
  LeanTest.assertEqual acrobotModelAssetBoundary.actuatedJointNames #["elbow"]
    "Only the elbow joint should be actuated"
  LeanTest.assertEqual acrobotModelAssetBoundary.transmissionNames #["elbow_trans"]
    "The elbow transmission name should match Drake's URDF"
  LeanTest.assertEqual acrobotModelAssetBoundary.frameNames #["hand"]
    "The hand frame should be recorded from the model file"
  LeanTest.assertEqual acrobotModelAssetBoundary.jointAxes
    #[#[0.0, 1.0, 0.0], #[0.0, 1.0, 0.0]]
    "Shoulder and elbow axes should both be +Y"
  LeanTest.assertEqual acrobotModelAssetBoundary.baseBoxSize #[0.2, 0.2, 0.2]
    "URDF/SDF base visual box size should be recorded"
  LeanTest.assertTrue (approx acrobotModelAssetBoundary.visualCylinderRadius 0.05 1.0e-12)
    s!"URDF visual cylinder radius should be 0.05, got {acrobotModelAssetBoundary.visualCylinderRadius}"
  LeanTest.assertTrue (approx acrobotModelAssetBoundary.urdfUpperVisualLength 1.1 1.0e-12)
    s!"URDF upper visual length should be 1.1, got {acrobotModelAssetBoundary.urdfUpperVisualLength}"
  LeanTest.assertTrue (approx acrobotModelAssetBoundary.urdfLowerVisualLength 2.1 1.0e-12)
    s!"URDF lower visual length should be 2.1, got {acrobotModelAssetBoundary.urdfLowerVisualLength}"
  LeanTest.assertEqual acrobotModelAssetBoundary.handFrameOffset #[0.0, 0.0, -2.1]
    "URDF hand frame offset should be recorded"

@[test]
def testAcrobotLcmMessageBoundaryRoundTripsStateAndCommand : IO Unit := do
  let sample ← assertOk buildAcrobotLcmIoSample?
    "Acrobot LCM sender/receiver sample"
  assertOk sample.boundary.validate? "Acrobot LCM IO boundary"
  LeanTest.assertEqual sample.boundary.stateMessageType "lcmt_acrobot_x"
    "State LCM message type should match Drake"
  LeanTest.assertEqual sample.boundary.commandMessageType "lcmt_acrobot_u"
    "Command LCM message type should match Drake"
  LeanTest.assertEqual sample.boundary.stateReceiverInputPort "lcmt_acrobot_x"
    "AcrobotStateReceiver should receive lcmt_acrobot_x"
  LeanTest.assertEqual sample.boundary.stateReceiverOutputPort "acrobot_state"
    "AcrobotStateReceiver should output acrobot_state"
  LeanTest.assertEqual sample.boundary.commandSenderInputPort "elbow_torque"
    "AcrobotCommandSender should read the elbow_torque vector port"
  LeanTest.assertEqual sample.boundary.commandSenderOutputPort "lcmt_acrobot_u"
    "AcrobotCommandSender should emit lcmt_acrobot_u"
  LeanTest.assertEqual sample.boundary.commandReceiverInputPort "lcmt_acrobot_u"
    "AcrobotCommandReceiver should receive lcmt_acrobot_u"
  LeanTest.assertEqual sample.boundary.commandReceiverOutputPort "elbow_torque"
    "AcrobotCommandReceiver should output elbow_torque"
  LeanTest.assertEqual sample.boundary.stateSenderInputPort "acrobot_state"
    "AcrobotStateSender should read acrobot_state"
  LeanTest.assertEqual sample.boundary.stateSenderOutputPort "lcmt_acrobot_x"
    "AcrobotStateSender should emit lcmt_acrobot_x"
  LeanTest.assertEqual sample.boundary.stateVectorSize 4
    "Acrobot LCM state conversion should have vector size 4"
  LeanTest.assertEqual sample.boundary.commandVectorSize 1
    "Acrobot LCM command conversion should have vector size 1"

  LeanTest.assertEqual sample.stateMessage.theta1 1.0
    "State sender should map theta1 into lcmt_acrobot_x"
  LeanTest.assertEqual sample.stateMessage.theta2 2.0
    "State sender should map theta2 into lcmt_acrobot_x"
  LeanTest.assertEqual sample.stateMessage.theta1Dot 3.0
    "State sender should map theta1dot into lcmt_acrobot_x.theta1Dot"
  LeanTest.assertEqual sample.stateMessage.theta2Dot 4.0
    "State sender should map theta2dot into lcmt_acrobot_x.theta2Dot"
  LeanTest.assertEqual sample.stateOut.asArray sample.stateIn.asArray
    "State receiver should round-trip the state BasicVector"
  LeanTest.assertEqual sample.commandMessage.tau (-5.0)
    "Command sender should map elbow torque into lcmt_acrobot_u.tau"
  LeanTest.assertEqual sample.commandOut.asArray sample.commandIn.asArray
    "Command receiver should round-trip the input BasicVector"
  LeanTest.assertTrue (sample.move.kind == SkeletonMoveKind.localSchurBlock)
    "LCM conversions should be represented as exact local boundary moves"
  LeanTest.assertTrue (sample.move.label.contains "LCM message conversion")
    s!"LCM move label should identify the message conversion boundary, got {sample.move.label}"

@[test]
def testAcrobotGeometryProviderMatchesDrakeSceneGraphRegistration : IO Unit := do
  let result ← assertOk (buildAcrobotGeometry? params defaultState)
    "AcrobotGeometry provider"
  assertOk result.provider.validate? "AcrobotGeometry SceneGraph provider"
  assertOk (result.poses.validate? result.provider) "AcrobotGeometry pose output"
  LeanTest.assertEqual result.inputPortName "state"
    "AcrobotGeometry should declare a vector input port named state"
  LeanTest.assertEqual result.inputPortSize 4
    "AcrobotGeometry state input should be a BasicVector of size 4"
  LeanTest.assertEqual result.outputPortName "geometry_pose"
    "AcrobotGeometry should declare an abstract output port named geometry_pose"
  LeanTest.assertEqual result.provider.sources.size 1
    "AcrobotGeometry should register one SceneGraph source"
  let frameNames := result.provider.frames.map (fun frame => frame.name)
  LeanTest.assertEqual frameNames #["upper_link", "lower_link"]
    "AcrobotGeometry should register Drake's upper/lower link frames"
  LeanTest.assertEqual result.provider.anchoredGeometries.size 1
    "AcrobotGeometry should anchor exactly the green base box"
  LeanTest.assertEqual result.provider.shapeNames #["box", "cylinder", "cylinder"]
    "AcrobotGeometry should register base box plus two link cylinders"

  match result.provider.geometryById? acrobotBaseGeometryId with
  | some base =>
      LeanTest.assertEqual base.name "base"
        "Base illustration geometry should use Drake's name"
      LeanTest.assertEqual base.frameId? none
        "Base geometry should be anchored"
      match base.shape with
      | .box sx sy sz =>
          LeanTest.assertTrue (approx sx 0.2 1.0e-12 && approx sy 0.2 1.0e-12 &&
              approx sz 0.2 1.0e-12)
            s!"Base box dimensions should be 0.2m cubed, got {reprStr base.shape}"
      | _ => LeanTest.fail s!"Base geometry should be a box, got {reprStr base.shape}"
      LeanTest.assertTrue
        (base.properties.diffuseRgba? ==
          some { r := 0.0, g := 1.0, b := 0.0, a := 1.0 })
        s!"Base should carry Drake's green illustration role, got {reprStr base.properties.diffuseRgba?}"
  | none => LeanTest.fail "AcrobotGeometry provider should contain base geometry"

  match result.provider.geometryById? acrobotUpperLinkGeometryId with
  | some upper =>
      LeanTest.assertEqual upper.name "upper_link"
        "Upper link geometry should use Drake's name"
      LeanTest.assertEqual upper.frameId? (some acrobotUpperLinkFrameId)
        "Upper link geometry should attach to the upper_link frame"
      match upper.shape with
      | .cylinder radius length =>
          LeanTest.assertTrue (approx radius 0.05 1.0e-12)
            s!"Upper link cylinder radius should be 0.05, got {radius}"
          LeanTest.assertTrue (approx length params.l1 1.0e-12)
            s!"Upper link cylinder length should be l1, got {length}"
      | _ => LeanTest.fail s!"Upper link geometry should be a cylinder, got {reprStr upper.shape}"
      LeanTest.assertTrue (approx upper.X_FG.translation.y 0.15 1.0e-12)
        s!"Upper link visual y-offset should be 0.15, got {upper.X_FG.translation.y}"
      LeanTest.assertTrue (approx upper.X_FG.translation.z (-params.l1 / 2.0) 1.0e-12)
        s!"Upper link visual z-offset should be -l1/2, got {upper.X_FG.translation.z}"
      LeanTest.assertTrue
        (upper.properties.diffuseRgba? ==
          some { r := 1.0, g := 0.0, b := 0.0, a := 1.0 })
        s!"Upper link should carry Drake's red illustration role, got {reprStr upper.properties.diffuseRgba?}"
  | none => LeanTest.fail "AcrobotGeometry provider should contain upper link geometry"

  match result.provider.geometryById? acrobotLowerLinkGeometryId with
  | some lower =>
      LeanTest.assertEqual lower.name "lower_link"
        "Lower link geometry should use Drake's name"
      LeanTest.assertEqual lower.frameId? (some acrobotLowerLinkFrameId)
        "Lower link geometry should attach to the lower_link frame"
      match lower.shape with
      | .cylinder radius length =>
          LeanTest.assertTrue (approx radius 0.05 1.0e-12)
            s!"Lower link cylinder radius should be 0.05, got {radius}"
          LeanTest.assertTrue (approx length params.l2 1.0e-12)
            s!"Lower link cylinder length should be l2, got {length}"
      | _ => LeanTest.fail s!"Lower link geometry should be a cylinder, got {reprStr lower.shape}"
      LeanTest.assertTrue (approx lower.X_FG.translation.y 0.25 1.0e-12)
        s!"Lower link visual y-offset should be 0.25, got {lower.X_FG.translation.y}"
      LeanTest.assertTrue (approx lower.X_FG.translation.z (-params.l2 / 2.0) 1.0e-12)
        s!"Lower link visual z-offset should be -l2/2, got {lower.X_FG.translation.z}"
      LeanTest.assertTrue
        (lower.properties.diffuseRgba? ==
          some { r := 0.0, g := 0.0, b := 1.0, a := 1.0 })
        s!"Lower link should carry Drake's blue illustration role, got {reprStr lower.properties.diffuseRgba?}"
  | none => LeanTest.fail "AcrobotGeometry provider should contain lower link geometry"

@[test]
def testAcrobotGeometryPoseOutputMatchesDrakeKinematics : IO Unit := do
  let x : AcrobotState := { theta1 := 0.3, theta2 := -0.7, theta1dot := 1.0, theta2dot := -2.0 }
  let result ← assertOk (buildAcrobotGeometry? params x)
    "AcrobotGeometry pose output"
  match result.poses.poseForFrame? acrobotUpperLinkFrameId,
      result.poses.poseForFrame? acrobotLowerLinkFrameId with
  | some upper, some lower =>
      LeanTest.assertTrue (upper.rotationAxis == SceneVec3.unitY)
        s!"Upper link pose should be a Y-axis rotation, got {reprStr upper.rotationAxis}"
      LeanTest.assertTrue (approx upper.rotationAngle x.theta1 1.0e-12)
        s!"Upper link pose angle should be theta1, got {upper.rotationAngle}"
      LeanTest.assertTrue (upper.translation == SceneVec3.zero)
        s!"Upper link world-frame translation should be identity, got {reprStr upper.translation}"
      LeanTest.assertTrue (lower.rotationAxis == SceneVec3.unitY)
        s!"Lower link pose should be a Y-axis rotation, got {reprStr lower.rotationAxis}"
      LeanTest.assertTrue (approx lower.rotationAngle (x.theta1 + x.theta2) 1.0e-12)
        s!"Lower link pose angle should be theta1 + theta2, got {lower.rotationAngle}"
      LeanTest.assertTrue (approx lower.translation.x (-params.l1 * Float.sin x.theta1) 1.0e-12)
        s!"Lower link x translation should be -l1 sin(theta1), got {lower.translation.x}"
      LeanTest.assertTrue (approx lower.translation.y 0.0 1.0e-12)
        s!"Lower link y translation should be zero, got {lower.translation.y}"
      LeanTest.assertTrue (approx lower.translation.z (-params.l1 * Float.cos x.theta1) 1.0e-12)
        s!"Lower link z translation should be -l1 cos(theta1), got {lower.translation.z}"
  | _, _ => LeanTest.fail "AcrobotGeometry should emit upper and lower link poses"

@[test]
def testAcrobotGeometryGraphRecordsExactSceneGraphBoundary : IO Unit := do
  let result ← assertOk (buildAcrobotGeometry? params defaultState)
    "AcrobotGeometry graph"
  LeanTest.assertEqual result.moves.size 2
    "AcrobotGeometry should expose registration and pose-output local moves"
  LeanTest.assertEqual (result.moves.filter (fun move => move.kind == .localSchurBlock)).size 2
    "AcrobotGeometry moves should be exact local SceneGraph blocks"
  LeanTest.assertTrue (result.moves.all (fun move => move.exactness == .exact))
    "AcrobotGeometry registration and pose output are deterministic exact moves"
  LeanTest.assertTrue (result.moves.any (fun move =>
      move.targets == #[acrobotGeometryPoseOutputVertex] &&
      move.reads == #[acrobotGeometryStateInputVertex, acrobotGeometryProviderVertex] &&
      move.writes == #[acrobotGeometryPoseOutputVertex] &&
      move.label.contains "FramePoseVector"))
    "Pose-output move should read state and provider vertices and write geometry_pose"

@[test]
def testMassMatrixAndBiasMatchDrakePlantFormula : IO Unit := do
  let x : AcrobotState := { theta1 := 0.1, theta2 := 0.2, theta1dot := 0.3, theta2dot := 0.4 }
  let m := massMatrix params x
  let bias := dynamicsBiasTerm params x
  let c2 := Float.cos x.theta2
  let i1 := params.Ic1 + params.m1 * params.lc1 * params.lc1
  let i2 := params.Ic2 + params.m2 * params.lc2 * params.lc2
  let h := params.m2 * params.l1 * params.lc2
  let m12 := i2 + h * c2

  LeanTest.assertTrue (approx ((m[0]!).getD 0 0.0)
      (i1 + i2 + params.m2 * params.l1 * params.l1 + 2.0 * h * c2) 1.0e-12)
    s!"M00 should match Drake formula, got {reprStr m}"
  LeanTest.assertTrue (approx ((m[0]!).getD 1 0.0) m12 1.0e-12)
    s!"M01 should match Drake formula, got {reprStr m}"
  LeanTest.assertTrue (approx ((m[1]!).getD 0 0.0) m12 1.0e-12)
    s!"M10 should match Drake formula, got {reprStr m}"
  LeanTest.assertTrue (approx ((m[1]!).getD 1 0.0) i2 1.0e-12)
    s!"M11 should match Drake formula, got {reprStr m}"

  let s1 := Float.sin x.theta1
  let s2 := Float.sin x.theta2
  let s12 := Float.sin (x.theta1 + x.theta2)
  let expected0 :=
    -2.0 * h * s2 * x.theta2dot * x.theta1dot +
      -h * s2 * x.theta2dot * x.theta2dot +
      params.gravity * params.m1 * params.lc1 * s1 +
      params.gravity * params.m2 * (params.l1 * s1 + params.lc2 * s12) +
      params.b1 * x.theta1dot
  let expected1 :=
    h * s2 * x.theta1dot * x.theta1dot +
      params.gravity * params.m2 * params.lc2 * s12 +
      params.b2 * x.theta2dot
  LeanTest.assertTrue (approx (bias.getD 0 99.0) expected0 1.0e-12)
    s!"Bias row 0 should match Drake formula, got {reprStr bias}"
  LeanTest.assertTrue (approx (bias.getD 1 99.0) expected1 1.0e-12)
    s!"Bias row 1 should match Drake formula, got {reprStr bias}"

@[test]
def testFullPhysicsPrimitiveProviderRecomputesStateAndInput : IO Unit := do
  let provider := fullPhysicsPrimitiveProvider params
    "acrobot full physics provider test"
  let x0 := defaultState
  let u0 := defaultInput
  let x1 : AcrobotState :=
    { theta1 := 0.1, theta2 := 0.2, theta1dot := 0.3, theta2dot := 0.4 }
  let u1 : AcrobotInput := { tau := 1.7 }

  let primitive0 ← assertOk
    (provider.validPrimitivesAt? (physicsState x0 u0))
    "acrobot provider primitive at default state and input"
  let primitive1 ← assertOk
    (provider.validPrimitivesAt? (physicsState x1 u1))
    "acrobot provider primitive at moved state and input"
  let result1 ← assertOk
    (provider.solveAt? (physicsState x1 u1) 5258)
    "acrobot provider solve at moved state and input"
  let direct1 ← assertOk
    (solveFullPhysics? params u1 x1 5259 "acrobot direct provider parity")
    "acrobot direct solve for provider parity"

  LeanTest.assertTrue
    (FloatArray.maxAbsDiff primitive0.massMatrix[0]! primitive1.massMatrix[0]! > 1.0e-3)
    "Changing theta2 should recompute the state-dependent Acrobot mass matrix"
  assertArrayNear primitive1.qdot (qdotAsArray x1) 1.0e-12
    "Acrobot provider qdot should come from the current state velocity"
  assertArrayNear primitive1.actuationForces (inputGeneralizedForces u1) 1.0e-12
    "Acrobot provider actuation should come from the current input"
  LeanTest.assertTrue
    (FloatArray.maxAbsDiff primitive0.biasForces primitive1.biasForces > 1.0)
    "Changing angles and velocities should recompute Coriolis, gravity, and damping bias"
  assertFullPhysicsMatches result1 x1 u1
    "Acrobot provider full physics"
  assertArrayNear result1.derivative.vdot direct1.derivative.vdot 1.0e-12
    "Provider solve should match the direct full-physics solve"
  LeanTest.assertEqual result1.move.targets #[5258]
    "Provider solve should use the supplied interval vertex"

  let badState : AcrobotState := { x1 with theta2dot := 1.0 / 0.0 }
  let msg ← assertError
    (provider.validPrimitivesAt? (physicsState badState u1))
    "acrobot provider malformed state"
  LeanTest.assertTrue (msg.contains "state")
    s!"Malformed Acrobot state should fail at provider validation, got {msg}"

@[test]
def testManipulatorEquationAndImplicitResidualMatchDrakeShape : IO Unit := do
  let x : AcrobotState := { theta1 := 0.1, theta2 := 0.2, theta1dot := 0.3, theta2dot := 0.4 }
  let u : AcrobotInput := { tau := -0.32 }
  let dyn ← assertOk (derivative? params u x) "acrobot derivative"
  let residual := implicitResidual params u x dyn
  LeanTest.assertTrue (FloatArray.maxAbsDiff residual #[0.0, 0.0, 0.0, 0.0] < 1.0e-11)
    s!"Implicit residual for exact derivatives should be near zero, got {reprStr residual}"
  LeanTest.assertTrue (approx dyn.theta1 x.theta1dot 1.0e-12)
    s!"theta1 derivative should equal theta1dot, got {dyn.theta1}"
  LeanTest.assertTrue (approx dyn.theta2 x.theta2dot 1.0e-12)
    s!"theta2 derivative should equal theta2dot, got {dyn.theta2}"
  LeanTest.assertTrue (Float.isFinite dyn.theta1dot && Float.isFinite dyn.theta2dot)
    s!"Acrobot accelerations should be finite, got {reprStr dyn}"

@[test]
def testMitParameterSetterValuesArePreserved : IO Unit := do
  LeanTest.assertTrue (mitParams.m1 != params.m1)
    "MIT parameter setter should change m1 from the default"
  LeanTest.assertTrue (approx mitParams.m1 2.4367 1.0e-12)
    s!"MIT m1 should match Drake setter, got {mitParams.m1}"
  LeanTest.assertTrue (approx mitParams.Ic1 (-4.7443) 1.0e-12)
    s!"MIT Ic1 should preserve Drake's negative identified inertia, got {mitParams.Ic1}"
  LeanTest.assertTrue (!mitParams.isValid)
    "MIT sys-id params intentionally violate the nonnegative inertia BasicVector domain"
  LeanTest.assertTrue mitParams.isFiniteForDynamics
    "MIT sys-id params should still be finite for dynamics evaluation"

@[test]
def testPassiveSimulationRecordsContinuousInterval : IO Unit := do
  let run ← assertOk
    (solvePassive? params { theta1 := 0.1, theta2 := 0.2, theta1dot := 0.0, theta2dot := 0.0 } 0.0 0.05 defaultInput)
    "acrobot passive solve"
  match run.trace.validate? with
  | .error msg => LeanTest.fail s!"Acrobot trace should validate: {msg}"
  | .ok () => pure ()
  LeanTest.assertTrue (approx run.t1 0.05 1.0e-12)
    s!"Acrobot run should reach requested final time, got {run.t1}"
  LeanTest.assertEqual run.moves.size 2
    "A pure continuous acrobot interval should project to interval and checkpoint moves"
  LeanTest.assertTrue (run.moves[0]!.kind == SkeletonMoveKind.intervalAdjoint)
    "First acrobot move should be the interval adjoint"
  LeanTest.assertTrue (Float.isFinite run.finalState.theta1 && Float.isFinite run.finalState.theta2 &&
      Float.isFinite run.finalState.theta1dot && Float.isFinite run.finalState.theta2dot)
    s!"Acrobot final state should be finite, got {reprStr run.finalState}"
  LeanTest.assertTrue (Float.isFinite run.finalEnergy)
    s!"Acrobot final energy should be finite, got {run.finalEnergy}"

@[test]
def testMultibodyBenchmarkAcrobotBoundariesAreRecorded : IO Unit := do
  let result ← assertOk buildMultibodyAcrobot?
    "multibody acrobot benchmark boundary"
  assertOk result.passiveStep.validate? "multibody acrobot passive full plant step"
  assertOk result.lqrStep.validate? "multibody acrobot LQR full plant step"
  assertOk result.trace.validate? "multibody acrobot trace"

  LeanTest.assertEqual result.passiveStep.model.modelUri benchmarkAcrobotFactory
    "Passive benchmark should record Drake's MakeAcrobotPlant factory boundary"
  LeanTest.assertEqual result.passiveStep.model.numPositions 2
    "Benchmark Acrobot MultibodyPlant has two positions"
  LeanTest.assertEqual result.passiveStep.model.numVelocities 2
    "Benchmark Acrobot MultibodyPlant has two velocities"
  LeanTest.assertEqual result.passiveStep.model.numActuatedDofs 1
    "Benchmark Acrobot MultibodyPlant has one actuated elbow DOF"
  LeanTest.assertEqual result.passiveStep.q0 #[1.0, 1.0]
    "Passive benchmark should initialize shoulder and elbow angles to 1 rad"
  LeanTest.assertEqual result.passiveStep.v0 #[0.0, 0.0]
    "Passive benchmark should leave velocities at zero"
  LeanTest.assertEqual result.passiveStep.actuation #[0.0]
    "Passive benchmark should connect zero elbow torque"
  LeanTest.assertTrue (approx result.passiveStep.t1 10.0 1.0e-12)
    s!"Passive benchmark should advance to 10s, got {result.passiveStep.t1}"
  LeanTest.assertTrue (approx result.config.passiveMaxTimeStep 0.01 1.0e-12)
    s!"Passive max step should be simulation_time/1000, got {result.config.passiveMaxTimeStep}"
  LeanTest.assertTrue (result.config.passiveIntegrationScheme == MultibodyIntegratorScheme.rungeKutta3)
    "Passive benchmark should default to Drake's runge_kutta3 flag"
  LeanTest.assertTrue (!result.config.passiveIntegrationScheme.fixedStep)
    "Runge-Kutta3 should be recorded as a variable-step integrator"
  LeanTest.assertTrue (approx result.config.targetAccuracy 0.001 1.0e-12)
    s!"Target accuracy should match Drake's benchmark, got {result.config.targetAccuracy}"

  LeanTest.assertEqual result.lqrStep.model.modelUri benchmarkAcrobotSdfUrl
    "LQR benchmark should parse Drake's benchmark acrobot SDF URL"
  LeanTest.assertTrue (approx result.lqrStep.config.timeStep 1.0e-3 1.0e-15)
    s!"Time-stepping LQR plant should use 1ms discrete updates, got {result.lqrStep.config.timeStep}"
  LeanTest.assertTrue (approx result.lqrStep.t1 3.0 1.0e-12)
    s!"LQR benchmark should advance each rollout to 3s, got {result.lqrStep.t1}"
  LeanTest.assertTrue (approx (result.lqrStep.q0.getD 0 0.0) pi 1.0e-12)
    s!"LQR nominal shoulder angle should be pi, got {result.lqrStep.q0}"
  LeanTest.assertTrue (approx (result.lqrStep.q0.getD 1 99.0) 0.0 1.0e-12)
    s!"LQR nominal elbow angle should be zero, got {result.lqrStep.q0}"
  LeanTest.assertEqual result.randomInitialConditions.rollouts 5
    "Drake's MultibodyPlant LQR demo runs five randomized contexts"
  LeanTest.assertTrue (approx result.randomInitialConditions.shoulderMean pi 1.0e-12)
    s!"Random shoulder mean should be pi, got {result.randomInitialConditions.shoulderMean}"
  LeanTest.assertTrue (approx result.randomInitialConditions.shoulderStddev 0.02 1.0e-12)
    s!"Random shoulder stddev should be 0.02, got {result.randomInitialConditions.shoulderStddev}"
  LeanTest.assertTrue (approx result.randomInitialConditions.elbowStddev 0.05 1.0e-12)
    s!"Random elbow stddev should be 0.05, got {result.randomInitialConditions.elbowStddev}"
  LeanTest.assertEqual result.lqrQ #[10.0, 10.0, 1.0, 1.0]
    "LQR benchmark should record Drake's Q diagonal"
  LeanTest.assertEqual result.lqrR #[1.0]
    "LQR benchmark should record Drake's scalar R"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.label == "full-physics-step:multibody acrobot passive benchmark plant"))
    "Move list should expose the passive benchmark as a primitive physics solve"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.label == "full-physics-step:multibody acrobot LQR benchmark plant"))
    "Move list should expose the LQR benchmark as a primitive physics solve"
  let passiveX0 : AcrobotState :=
    { theta1 := 1.0, theta2 := 1.0, theta1dot := 0.0, theta2dot := 0.0 }
  assertFullPhysicsMatches result.passiveFullPhysics passiveX0 defaultInput
    "passive benchmark full physics"
  let lqrX0 : AcrobotState :=
    { theta1 := pi, theta2 := 0.0, theta1dot := 0.0, theta2dot := 0.0 }
  assertFullPhysicsMatches result.lqrFullPhysics lqrX0 defaultInput
    "LQR benchmark full physics"

@[test]
def testExternalAcrobotLcmEstimatorAndOptimizerBoundaries : IO Unit := do
  let result ← assertOk buildAcrobotExternalBoundaries?
    "acrobot LCM estimator optimizer boundaries"
  assertOk result.trace.validate? "acrobot external boundary trace"
  assertOk result.lcmPlant.validate? "acrobot LCM plant boundary"
  assertOk result.lcmController.validate? "acrobot LCM controller boundary"
  assertOk result.lcmIo.boundary.validate? "acrobot LCM message conversion boundary"
  assertOk result.estimatorLqr.validate? "acrobot estimator LQR boundary"
  assertOk result.deterministicScenario.validate? "acrobot deterministic scenario"
  assertOk result.stochasticScenario.validate? "acrobot stochastic scenario"
  assertOk result.optimizer.validate? "acrobot optimizer config"

  LeanTest.assertEqual result.channels.stateEstimateChannel "acrobot_xhat"
    "LCM state estimate channel should match Drake"
  LeanTest.assertEqual result.channels.commandChannel "acrobot_u"
    "LCM command channel should match Drake"
  LeanTest.assertTrue (approx result.lcmPlant.initialState.theta1 0.1 1.0e-12 &&
      approx result.lcmPlant.initialState.theta2 0.1 1.0e-12)
    s!"LCM plant initial condition should match run_plant_w_lcm, got {reprStr result.lcmPlant.initialState}"
  LeanTest.assertTrue (approx result.lcmController.commandPublishPeriod 1.0e-3 1.0e-15)
    s!"LCM Spong controller should publish commands every 1ms, got {result.lcmController.commandPublishPeriod}"
  LeanTest.assertEqual result.lcmIo.stateOut.asArray result.lcmIo.stateIn.asArray
    "External LCM boundary should round-trip Acrobot state messages"
  LeanTest.assertEqual result.lcmIo.commandOut.asArray result.lcmIo.commandIn.asArray
    "External LCM boundary should round-trip Acrobot command messages"

  LeanTest.assertTrue (approx result.estimatorLqr.simulationTime 5.0 1.0e-12)
    s!"Estimator LQR simulation time should match Drake's flag default, got {result.estimatorLqr.simulationTime}"
  LeanTest.assertTrue (approx result.estimatorLqr.maximumStepSize 0.01 1.0e-12)
    s!"Estimator LQR max step should be 0.01, got {result.estimatorLqr.maximumStepSize}"
  LeanTest.assertTrue result.estimatorLqr.fixedStepMode
    "Estimator LQR should record Drake's fixed-step integrator mode"
  LeanTest.assertEqual result.estimatorLqr.processNoiseDiagonal #[1.0, 1.0, 1.0, 1.0]
    "Estimator process noise W should be identity"
  LeanTest.assertEqual result.estimatorLqr.measurementNoiseDiagonal #[0.1, 0.1]
    "Encoder measurement noise V should be 0.1 * I"
  LeanTest.assertTrue result.estimatorLqr.logsTrueAndEstimatedState
    "Estimator demo should log both true and estimated state"

  LeanTest.assertTrue (approx result.deterministicScenario.tFinal 30.0 1.0e-12)
    s!"Deterministic scenario t_final should be 30s, got {result.deterministicScenario.tFinal}"
  LeanTest.assertTrue (approx result.deterministicScenario.tapePeriod 0.05 1.0e-12)
    s!"Scenario tape_period should be 0.05s, got {result.deterministicScenario.tapePeriod}"
  LeanTest.assertTrue (!result.deterministicScenario.isStochastic)
    "example_scenario.yaml should be deterministic"
  LeanTest.assertTrue result.stochasticScenario.isStochastic
    "example_stochastic_scenario.yaml should retain UniformVector sampled fields"
  match result.stochasticScenario.controllerParams with
  | .uniform lo hi =>
      LeanTest.assertEqual lo #[4.0, 40.0, 4.0, 0.9e3]
        "Stochastic controller parameter lower bounds should match Drake YAML"
      LeanTest.assertEqual hi #[6.0, 60.0, 6.0, 1.1e3]
        "Stochastic controller parameter upper bounds should match Drake YAML"
  | .deterministic _ =>
      LeanTest.fail "Stochastic controller params should use UniformVector bounds"
  match result.stochasticScenario.initialState with
  | .uniform lo hi =>
      LeanTest.assertEqual lo #[1.1, -0.1, -0.1, -0.1]
        "Stochastic initial-state lower bounds should match Drake YAML"
      LeanTest.assertEqual hi #[1.3, 0.1, 0.1, 0.1]
        "Stochastic initial-state upper bounds should match Drake YAML"
  | .deterministic _ =>
      LeanTest.fail "Stochastic initial state should use UniformVector bounds"

  LeanTest.assertTrue
    (result.stochasticScenario.support.policy == SupportPolicy.sampled 0)
    s!"Stochastic scenario support should be recorded as a sampled mark, got {reprStr result.stochasticScenario.support}"
  LeanTest.assertTrue
    (result.stochasticScenario.support.exactness == MoveExactness.unbiasedEstimator)
    "Sampled stochastic scenario support should carry unbiased-estimator exactness"
  LeanTest.assertTrue (result.optimizer.metric == AcrobotOptimizerMetric.ensembleCost)
    "optimizer_demo should default to ensemble_cost"
  LeanTest.assertEqual result.optimizer.ensembleSize 10
    "optimizer_demo should default to ten seeds per metric evaluation"
  LeanTest.assertEqual result.optimizer.numEvaluations 250
    "optimizer_demo should default to 250 optimizer function evaluations"
  LeanTest.assertEqual result.optimizer.totalRollouts 2500
    "Total rollout budget should be ensemble_size * num_evaluations"
  LeanTest.assertEqual result.optimizer.seedIds #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    "optimizer_demo should evaluate seeds 1 through ensemble_size"
  LeanTest.assertTrue
    (result.moves.any (fun move => move.kind == SkeletonMoveKind.markScoreSample))
    "Acrobot external moves should expose stochastic scenario sampling"
  LeanTest.assertTrue
    (result.moves.any (fun move => move.label.contains "LCM message conversion"))
    "Acrobot external moves should expose LCM sender/receiver conversion boundaries"
  LeanTest.assertTrue
    (result.moves.any (fun move => move.kind == SkeletonMoveKind.branchAggregate))
    "Acrobot external moves should expose optimizer ensemble branch aggregation"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.label == "spong_sim C++/Python scenario runner and state tape output"))
    "Acrobot external moves should expose the scenario runner boundary"

@[test]
def testAcrobotRegressionBoundariesRecordSpongAndTrajectoryOptimizationHarnesses : IO Unit := do
  let result ← assertOk buildAcrobotRegressionBoundaries?
    "acrobot regression boundaries"
  LeanTest.assertEqual result.spongParams.names spongParameterCoordinateNames
    "Spong parameter index boundary should preserve Drake coordinate names"
  LeanTest.assertEqual result.spongParams.sourcePath
    "../drake/examples/acrobot/spong_controller_params.cc"

  LeanTest.assertEqual result.libraryPy.testPath
    "../drake/examples/acrobot/test/spong_sim_lib_py_test.py"
  LeanTest.assertEqual result.libraryPy.functionName
    "examples.acrobot.spong_sim.simulate"
  LeanTest.assertEqual result.libraryPy.initialState #[0.01, 0.02, 0.03, 0.04]
  LeanTest.assertEqual result.libraryPy.controllerParams #[0.001, 0.002, 0.003, 0.004]
  LeanTest.assertTrue (approx result.libraryPy.tFinal 30.0 1.0e-12)
    s!"Spong library t_final should be 30, got {result.libraryPy.tFinal}"
  LeanTest.assertTrue (approx result.libraryPy.tapePeriod 0.05 1.0e-12)
    s!"Spong library tape_period should be 0.05, got {result.libraryPy.tapePeriod}"
  LeanTest.assertEqual result.libraryPy.expectedRows 4
  LeanTest.assertEqual result.libraryPy.expectedCols 601

  LeanTest.assertTrue (result.mainPy.backend == SpongSimBackend.py)
    s!"Expected Python backend, got {reprStr result.mainPy.backend}"
  LeanTest.assertEqual result.mainPy.executableResource
    "drake/examples/acrobot/spong_sim_main_py"
  LeanTest.assertEqual result.mainPy.helpCommand #["spong_sim_main_py", "--help"]
  LeanTest.assertEqual result.mainPy.expectedHelpReturnCode 0
  LeanTest.assertFalse result.mainPy.supportsStochasticScenario
    "Python backend should skip stochastic scenario support"
  LeanTest.assertEqual result.mainPy.deterministicCommand
    #["spong_sim_main_py", "--scenario", "example_scenario.yaml", "--output", "output.yaml"]

  LeanTest.assertTrue (result.mainCc.backend == SpongSimBackend.cc)
    s!"Expected C++ backend, got {reprStr result.mainCc.backend}"
  LeanTest.assertEqual result.mainCc.executableResource
    "drake/examples/acrobot/spong_sim_main_cc"
  LeanTest.assertEqual result.mainCc.helpCommand #["spong_sim_main_cc", "--help"]
  LeanTest.assertEqual result.mainCc.expectedHelpReturnCode 1
  LeanTest.assertTrue result.mainCc.supportsStochasticScenario
    "C++ backend should run and dump the stochastic scenario"
  LeanTest.assertEqual result.mainCc.stochasticCommand
    #["spong_sim_main_cc", "--scenario", "example_stochastic_scenario.yaml", "--output", "output.yaml",
      "--dump_scenario", "scenario_out.yaml"]
  LeanTest.assertEqual result.mainCc.expectedDumpControllerParams 4
  LeanTest.assertEqual result.mainCc.expectedDumpInitialState 4

  let swing := result.swingUpTrajectoryOptimization
  LeanTest.assertEqual swing.testPath
    "../drake/examples/acrobot/test/run_swing_up_traj_optimization.cc"
  LeanTest.assertTrue swing.requiresSnopt
    "Swing-up trajectory optimization should record the SNOPT availability gate"
  LeanTest.assertEqual swing.snoptUnavailableReturnCode 0
  LeanTest.assertEqual swing.solverFailureReturnCode 1
  LeanTest.assertEqual swing.numTimeSamples 21
  LeanTest.assertTrue (approx swing.minimumTimeStep 0.05 1.0e-12)
    s!"Direct collocation min dt should be 0.05, got {swing.minimumTimeStep}"
  LeanTest.assertTrue (approx swing.maximumTimeStep 0.2 1.0e-12)
    s!"Direct collocation max dt should be 0.2, got {swing.maximumTimeStep}"
  LeanTest.assertTrue swing.equalTimeIntervals
    "Direct collocation should add equal time interval constraints"
  LeanTest.assertTrue (approx swing.torqueLimit 8.0 1.0e-12)
    s!"Swing-up torque limit should be 8, got {swing.torqueLimit}"
  LeanTest.assertEqual swing.initialState #[0.0, 0.0, 0.0, 0.0]
  LeanTest.assertTrue (approx (swing.goalState.getD 0 0.0) pi 1.0e-12)
    s!"Swing-up goal shoulder should be pi, got {swing.goalState}"
  LeanTest.assertEqual swing.finiteHorizonQ #[10.0, 10.0, 1.0, 1.0]
  LeanTest.assertEqual swing.finiteHorizonR #[1.0]
  LeanTest.assertTrue (approx swing.terminalTolerance 0.1 1.0e-12)
    s!"Terminal tolerance should be 0.1, got {swing.terminalTolerance}"
  LeanTest.assertTrue swing.visualizerEnabled
    "Swing-up trajectory optimization should retain the SceneGraph/DrakeVisualizer playback boundary"

  LeanTest.assertTrue (result.graph.containsMoveKind .checkpointBoundary)
    "SNOPT availability should be represented as a checkpoint boundary"
  LeanTest.assertTrue (result.graph.containsMoveKind .localSchurBlock)
    "Direct collocation, SNOPT, and finite-horizon LQR should be local solver/controller blocks"
  LeanTest.assertTrue (result.graph.containsMoveKind .intervalAdjoint)
    "Spong subprocess runs and trajectory playback should be interval boundaries"
  LeanTest.assertTrue
    (result.moves.any (fun move => move.label.contains "spong_sim_main_cc"))
    "C++ Spong main regression should remain visible as a boundary move"
  LeanTest.assertTrue
    (result.moves.any (fun move => move.label.contains "DirectCollocation"))
    "Swing-up direct collocation should remain visible as a boundary move"

@[test]
def testBalancingLqrLinearizationAndControllerMode : IO Unit := do
  let lin := linearizationAboutUpright params
  LeanTest.assertEqual lin.A.size 4
    "Acrobot upright linearization should expose a 4x4 A matrix"
  LeanTest.assertTrue (approx ((lin.A[0]!).getD 2 0.0) 1.0 1.0e-12)
    s!"Acrobot A should map theta1dot into theta1 derivative, got {reprStr lin.A}"
  LeanTest.assertTrue (approx ((lin.A[1]!).getD 3 0.0) 1.0 1.0e-12)
    s!"Acrobot A should map theta2dot into theta2 derivative, got {reprStr lin.A}"
  LeanTest.assertTrue (((lin.B[3]!).getD 0 0.0) > 0.0)
    s!"Elbow torque should positively affect theta2 acceleration at the upright, got {reprStr lin.B}"
  LeanTest.assertEqual balancingLqrData.K.size 4
    "Balancing LQR data should expose Drake's one-row 4-state gain"

  let near ← assertOk (spongController? params spongControllerParams balancingLqrData runLqrInitialState)
    "near-upright Spong controller"
  LeanTest.assertTrue (near.mode == SpongMode.balancing)
    s!"Near-upright state should select LQR balancing mode, got {reprStr near.mode}"
  LeanTest.assertTrue (Float.abs near.tau <= 20.0)
    s!"Acrobot torque should be saturated within Drake bounds, got {near.tau}"

  let swing ← assertOk (spongController? params spongControllerParams balancingLqrData runSwingUpInitialState)
    "near-downward Spong controller"
  LeanTest.assertTrue (swing.mode == SpongMode.swingUp)
    s!"Swing-up initial state should select Spong energy mode, got {reprStr swing.mode}"
  LeanTest.assertTrue (Float.isFinite swing.energyTorque && Float.isFinite swing.partialFeedbackTorque)
    s!"Spong swing-up terms should be finite, got {reprStr swing}"

@[test]
def testLqrSimulationStabilizesNearUprightState : IO Unit := do
  let run ← assertOk simulateLqr? "acrobot LQR simulation"
  LeanTest.assertTrue (approx run.t1 10.0 1.0e-12)
    s!"Acrobot LQR run should advance to Drake's 10s horizon, got {run.t1}"
  LeanTest.assertTrue (run.modeSummary.balancingSteps == controllerSimulationConfig.steps)
    s!"LQR rollout should stay in balancing mode, got {reprStr run.modeSummary}"
  LeanTest.assertTrue (run.modeSummary.support.policy == SupportPolicy.deterministicPick SpongMode.balancing.id)
    s!"LQR mode support should record deterministic balancing, got {reprStr run.modeSummary.support}"
  LeanTest.assertTrue (approx (wrapTo run.finalState.theta1 0.0 twoPi) pi 1.0e-3)
    s!"LQR should stabilize theta1 near upright, got {reprStr run.finalState}"
  LeanTest.assertTrue (approx (wrapTo run.finalState.theta2 (-pi) pi) 0.0 1.0e-3)
    s!"LQR should stabilize theta2 near zero, got {reprStr run.finalState}"
  LeanTest.assertTrue (approx run.finalState.theta1dot 0.0 1.0e-3)
    s!"LQR should stabilize theta1dot near zero, got {reprStr run.finalState}"
  LeanTest.assertTrue (approx run.finalState.theta2dot 0.0 1.0e-3)
    s!"LQR should stabilize theta2dot near zero, got {reprStr run.finalState}"
  LeanTest.assertTrue (run.moves.any (fun move => move.kind == SkeletonMoveKind.localSchurBlock))
    "LQR controller should be recorded as a local solver/controller block"

@[test]
def testSpongSwingUpSimulationReachesUprightAndRecordsDynamicModes : IO Unit := do
  let run ← assertOk simulateSwingUp? "acrobot Spong swing-up simulation"
  LeanTest.assertTrue (approx run.t1 10.0 1.0e-12)
    s!"Spong run should advance to Drake's 10s horizon, got {run.t1}"
  LeanTest.assertTrue (run.modeSummary.totalSteps == controllerSimulationConfig.steps)
    s!"Spong mode summary should account for every step, got {reprStr run.modeSummary}"
  LeanTest.assertTrue run.modeSummary.sawBothModes
    s!"Spong run should dynamically switch from swing-up to balancing, got {reprStr run.modeSummary}"
  LeanTest.assertTrue (run.modeSummary.support.policy == SupportPolicy.threshold spongControllerParams.balancingThreshold)
    s!"Spong dynamic mode support should record threshold selection, got {reprStr run.modeSummary.support}"
  LeanTest.assertTrue (run.moves.any
      (fun move => move.kind == SkeletonMoveKind.localSchurBlock &&
        move.exactness == MoveExactness.controlledApproximation))
    "Spong dynamic mode switch should be visible as a fixed-trace controller block"
  LeanTest.assertTrue (approx (wrapTo run.finalState.theta1 0.0 twoPi) pi 1.0e-2)
    s!"Spong swing-up should reach theta1 upright modulo 2*pi, got {reprStr run.finalState}"
  LeanTest.assertTrue (approx (wrapTo run.finalState.theta2 (-pi) pi) 0.0 1.0e-2)
    s!"Spong swing-up should reach theta2 near zero modulo 2*pi, got {reprStr run.finalState}"
  LeanTest.assertTrue (Float.abs run.finalState.theta1dot < 0.1)
    s!"Spong swing-up should damp theta1dot below Drake's check, got {reprStr run.finalState}"
  LeanTest.assertTrue (Float.abs run.finalState.theta2dot < 0.1)
    s!"Spong swing-up should damp theta2dot below Drake's check, got {reprStr run.finalState}"

end Tests.EventSkeletonAcrobotExample
