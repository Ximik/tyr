import LeanTest
import Tyr.EventSkeleton.Examples.Quadrotor

namespace Tests.EventSkeletonQuadrotorExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.Quadrotor

private def pi : Float := 3.14159265358979323846

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
  | none => LeanTest.fail s!"{label}: expected some"

@[test]
def testDrakeReferencesAndStateConventionAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/quadrotor/quadrotor_plant.cc"))
    "Quadrotor example should reference Drake's explicit plant implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/quadrotor/test/quadrotor_dynamics_test.cc"))
    "Quadrotor example should reference Drake's MultibodyPlant parity test"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/quadrotor/run_quadrotor_lqr.cc"))
    "Quadrotor example should reference Drake's LQR hover executable"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/quadrotor/warehouse.sdf"))
    "Quadrotor example should reference Drake's warehouse environment model"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/quadrotor/office.urdf"))
    "Quadrotor example should reference Drake's office environment model"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/quadrotor/quadrotor_geometry.h"))
    "Quadrotor example should reference Drake's QuadrotorGeometry declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/quadrotor/quadrotor_geometry.cc"))
    "Quadrotor example should reference Drake's QuadrotorGeometry implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/quadrotor/test/quadrotor_geometry_test.cc"))
    "Quadrotor example should reference Drake's QuadrotorGeometry acceptance test"
  LeanTest.assertEqual stateCoordinateNames
    #["x", "y", "z", "roll", "pitch", "yaw",
      "xdot", "ydot", "zdot", "rolldot", "pitchdot", "yawdot"]
    "Quadrotor state order should match Drake's 12-coordinate convention"
  LeanTest.assertEqual inputCoordinateNames
    #["propeller_force_0", "propeller_force_1", "propeller_force_2", "propeller_force_3"]
    "Quadrotor input order should match Drake's four propeller force port"
  LeanTest.assertEqual parameterCoordinateNames
    #["m", "L", "ixx", "iyy", "izz", "kF", "kM", "gravity"]
    "Quadrotor parameter names should preserve Drake defaults"
  LeanTest.assertTrue params.isValid "Default quadrotor params should be valid"
  LeanTest.assertTrue (stateIsValid defaultState) "Default quadrotor state should be valid"
  LeanTest.assertTrue (inputIsValid defaultInput) "Disconnected input should be modeled as finite zero force"

@[test]
def testQuadrotorEnvironmentAssetsAndWarehouseProvider : IO Unit := do
  assertOk validateQuadrotorEnvironmentAssets? "quadrotor environment catalog"
  let warehouse ← assertSome
    (findQuadrotorEnvironmentAsset? "../drake/examples/quadrotor/warehouse.sdf")
    "warehouse asset"
  LeanTest.assertEqual warehouse.packageUri
    "package://drake/examples/quadrotor/warehouse.sdf"
  LeanTest.assertEqual warehouse.modelName "room_w_lidar"
  LeanTest.assertEqual warehouse.obstacleBoxes.size 8
    "Warehouse SDF should expose floor, four walls, and three slalom boxes"
  LeanTest.assertEqual warehouse.linkNames
    #["bottom_wall", "left_wall", "right_wall", "back_wall", "front_wall",
      "slalom_1", "slalom_2", "slalom_3"]
  let bottom := warehouse.obstacleBoxes[0]!
  LeanTest.assertEqual bottom.name "bottom_wall"
  LeanTest.assertTrue (approx bottom.center.x 3.5 1.0e-12 &&
      approx bottom.center.y (-0.25) 1.0e-12 &&
      approx bottom.center.z (-1.6) 1.0e-12)
    s!"Warehouse floor pose should match SDF, got {reprStr bottom.center}"
  LeanTest.assertTrue (approx bottom.size.x 10.0 1.0e-12 &&
      approx bottom.size.y 4.5 1.0e-12 &&
      approx bottom.size.z 0.2 1.0e-12)
    s!"Warehouse floor box should match SDF, got {reprStr bottom.size}"

  let office ← assertSome
    (findQuadrotorEnvironmentAsset? "../drake/examples/quadrotor/office.urdf")
    "office asset"
  LeanTest.assertEqual office.packageUri
    "package://drake/examples/quadrotor/office.urdf"
  LeanTest.assertEqual office.modelName "office"
  LeanTest.assertEqual office.obstacleBoxes.size 16
    "Office URDF should expose all wall, table, cabinet, and drawer boxes"
  LeanTest.assertEqual office.jointNames.size 15
    "Office URDF should preserve its fixed-joint tree"
  LeanTest.assertEqual office.materialNames #["Brown", "White", "Grey", "Red"]
    "Office URDF material names should be recorded"

  let provider := environmentSceneGraphProvider warehouse
  assertOk provider.validate? "warehouse SceneGraph provider"
  LeanTest.assertEqual provider.geometries.size warehouse.obstacleBoxes.size
    "Environment provider should emit one anchored geometry per obstacle box"
  let floorGeom ← assertSome
    (provider.geometryById? quadrotorEnvironmentGeometryIdBase)
    "warehouse floor geometry"
  LeanTest.assertEqual floorGeom.name "bottom_wall"
  LeanTest.assertTrue floorGeom.isAnchored
    "Static warehouse geometry should be anchored in world"
  LeanTest.assertTrue (floorGeom.hasRole .illustration && floorGeom.hasRole .proximity)
    "Warehouse boxes should expose both visualization and proximity roles"
  match floorGeom.shape with
  | .box sx sy sz =>
      LeanTest.assertTrue
        (approx sx 10.0 1.0e-12 && approx sy 4.5 1.0e-12 &&
          approx sz 0.2 1.0e-12)
        s!"Warehouse floor geometry should preserve SDF box dimensions, got {sx}, {sy}, {sz}"
  | other => LeanTest.fail s!"Warehouse floor should be a box geometry, got {reprStr other}"

@[test]
def testWarehouseContactCandidateProviderUsesCurrentState : IO Unit := do
  let x := mkState 0.0 (-0.25) (-1.4) 0.0 0.0 0.0
    0.1 0.0 (-0.3) 0.0 0.0 0.0
  let set ← assertOk
    (warehouseContactCandidateProvider.validCandidatesAt? x (some 6))
    "warehouse contact candidates"
  LeanTest.assertEqual set.totalCandidates 8
    "Warehouse provider should preserve the SDF obstacle count"
  LeanTest.assertEqual set.candidates.size 8
    "Warehouse provider should emit one candidate view per obstacle"
  let floor ← assertSome
    (set.candidates.find? (fun candidate => candidate.bodyB == "bottom_wall"))
    "warehouse floor candidate"
  LeanTest.assertTrue (approx floor.signedDistance 0.1 1.0e-12)
    s!"Point 10cm above the floor top should have distance 0.1, got {floor.signedDistance}"
  LeanTest.assertEqual floor.normal_W #[0.0, 0.0, 1.0]
    "Floor contact normal should point upward"
  LeanTest.assertTrue (approx floor.normalVelocity (-0.3) 1.0e-12)
    s!"Floor normal velocity should be zdot, got {floor.normalVelocity}"
  LeanTest.assertEqual floor.normalJacobian #[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    "Warehouse point contact should expose a six-velocity translational Jacobian row"

  let support ← assertOk
    (warehouseContactCandidateProvider.supportAt? x (.threshold 0.11)
      0.11 1.0e-6 (some 6) "warehouse near-floor support")
    "warehouse near-floor support"
  let selected ← assertOk support.selectedCandidates?
    "selected warehouse candidates"
  LeanTest.assertEqual selected.size 1
    "Only the floor should be inside the 11cm contact-support threshold"
  LeanTest.assertEqual selected[0]!.id floor.id
    "Support selection should preserve the floor candidate's stable id"
  LeanTest.assertTrue (selected[0]!.mode == ContactMode.impacting)
    "Negative normal velocity should classify the near-floor candidate as impacting"

@[test]
def testQuadrotorGeometryProviderMatchesDrakeSceneGraphRegistration : IO Unit := do
  let result ← assertOk (buildQuadrotorGeometry? nominalHoverState)
    "QuadrotorGeometry provider"
  assertOk result.provider.validate? "QuadrotorGeometry SceneGraph provider"
  assertOk (result.poses.validate? result.provider) "QuadrotorGeometry pose output"
  LeanTest.assertEqual result.inputPortName "state"
    "QuadrotorGeometry should declare a vector input port named state"
  LeanTest.assertEqual result.inputPortSize 12
    "QuadrotorGeometry state input should be a BasicVector of size 12"
  LeanTest.assertEqual result.outputPortName "geometry_pose"
    "QuadrotorGeometry should declare an abstract output port named geometry_pose"
  LeanTest.assertEqual result.modelUri quadrotorModelUri
    "QuadrotorGeometry should retain Drake's Skydio package model URI"
  LeanTest.assertEqual result.provider.sources.size 1
    "QuadrotorGeometry should register one SceneGraph source"
  LeanTest.assertEqual result.provider.frames.size 1
    "QuadrotorGeometry should expose Drake's single quadrotor body frame"
  let frame ← assertSome (result.provider.frameById? quadrotorBodyFrameId)
    "quadrotor body frame lookup"
  LeanTest.assertEqual frame.name "base_link"
    "Quadrotor body frame should preserve the Skydio base-link convention"
  let body ← assertSome (result.provider.geometryById? quadrotorBodyGeometryId)
    "quadrotor model geometry lookup"
  LeanTest.assertEqual body.frameId? (some quadrotorBodyFrameId)
    "Quadrotor model geometry should attach to the body frame"
  LeanTest.assertTrue (body.hasRole .illustration)
    "Quadrotor model geometry should carry the visualization role"
  match body.shape with
  | .model uri =>
      LeanTest.assertEqual uri quadrotorModelUri
        "Quadrotor model shape should point at Drake's Skydio URDF"
  | other => LeanTest.fail s!"QuadrotorGeometry body should be a model resource, got {reprStr other}"

@[test]
def testQuadrotorGeometryPoseOutputMatchesDrakeRollPitchYaw : IO Unit := do
  let x := mkState 1.2 (-0.4) 2.5 0.2 (-0.3) 0.4
    0.7 (-0.6) 0.5 0.11 (-0.12) 0.13
  let result ← assertOk (buildQuadrotorGeometry? x)
    "QuadrotorGeometry pose output"
  let pose ← assertSome (result.poses.poseForFrame? quadrotorBodyFrameId)
    "quadrotor body frame pose"
  LeanTest.assertTrue (approx pose.translation.x (x.getD 0 99.0) 1.0e-12)
    s!"Quadrotor pose x should match state[0], got {pose.translation.x}"
  LeanTest.assertTrue (approx pose.translation.y (x.getD 1 99.0) 1.0e-12)
    s!"Quadrotor pose y should match state[1], got {pose.translation.y}"
  LeanTest.assertTrue (approx pose.translation.z (x.getD 2 99.0) 1.0e-12)
    s!"Quadrotor pose z should match state[2], got {pose.translation.z}"
  let attitude := rpy x
  let expectedX := attitude.rotateBodyToWorld { x := 1.0 }
  let expectedY := attitude.rotateBodyToWorld { y := 1.0 }
  let expectedZ := attitude.rotateBodyToWorld { z := 1.0 }
  let actualX := pose.rotateVector SceneVec3.unitX
  let actualY := pose.rotateVector SceneVec3.unitY
  let actualZ := pose.rotateVector SceneVec3.unitZ
  LeanTest.assertTrue (maxAbsDiff actualX.toArray expectedX.asArray < 1.0e-12)
    s!"Quadrotor pose should encode Drake RollPitchYaw rotation for X, got {reprStr actualX}"
  LeanTest.assertTrue (maxAbsDiff actualY.toArray expectedY.asArray < 1.0e-12)
    s!"Quadrotor pose should encode Drake RollPitchYaw rotation for Y, got {reprStr actualY}"
  LeanTest.assertTrue (maxAbsDiff actualZ.toArray expectedZ.asArray < 1.0e-12)
    s!"Quadrotor pose should encode Drake RollPitchYaw rotation for Z, got {reprStr actualZ}"

@[test]
def testQuadrotorGeometryGraphRecordsExactSceneGraphBoundary : IO Unit := do
  let result ← assertOk (buildQuadrotorGeometry? nominalHoverState)
    "QuadrotorGeometry graph"
  LeanTest.assertEqual result.moves.size 2
    "QuadrotorGeometry should expose registration and pose-output local moves"
  LeanTest.assertTrue (result.moves.all (fun move =>
      move.kind == SkeletonMoveKind.localSchurBlock &&
      move.exactness == MoveExactness.exact))
    "QuadrotorGeometry moves should be exact local SceneGraph blocks"
  LeanTest.assertTrue (result.moves.all (fun move => move.cost.work == 0.0))
    "QuadrotorGeometry registration and pose output are deterministic exact moves"
  LeanTest.assertTrue (result.moves.any (fun move =>
      move.targets == #[quadrotorGeometryPoseOutputVertex] &&
      move.reads == #[quadrotorGeometryStateInputVertex, quadrotorGeometryProviderVertex] &&
      move.writes == #[quadrotorGeometryPoseOutputVertex] &&
      move.label.contains "OutputGeometryPose"))
    "QuadrotorGeometry graph should record the state-to-FramePoseVector move"

@[test]
def testDisconnectedInputDropsFromRestLikeDrakePlant : IO Unit := do
  let dx ← assertOk (derivative? params defaultInput defaultState) "quadrotor no-input derivative"
  LeanTest.assertTrue (maxAbsDiff dx
      #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -params.gravity, 0.0, 0.0, 0.0] < 1.0e-12)
    s!"Disconnected quadrotor input should produce free-fall derivative, got {reprStr dx}"

@[test]
def testHoverInputCancelsGravityAtLevelPose : IO Unit := do
  let dx ← assertOk (derivative? params (hoverInput params) defaultState) "quadrotor hover derivative"
  LeanTest.assertTrue (maxAbsDiff dx (Array.replicate 12 0.0) < 1.0e-12)
    s!"Hover input at level pose should be an equilibrium derivative, got {reprStr dx}"

@[test]
def testFullPhysicsPrimitiveAssemblesRigidBodyForceSolve : IO Unit := do
  let x := mkState 0.2 (-0.1) 1.3 0.15 (-0.2) 0.25
    0.4 (-0.3) 0.2 0.11 (-0.07) 0.09
  let u : QuadrotorInput := #[1.1, 1.4, 1.2, 1.3]
  let full ← assertOk (solveFullPhysics? params u x)
    "quadrotor full-physics solve"
  LeanTest.assertEqual full.equation.massMatrix (massMatrix params)
    "Quadrotor full physics should expose mass plus diagonal body inertia"
  LeanTest.assertEqual full.derivative.qdot (poseQdot x)
    "Quadrotor qdot should be translational velocity plus RPY rate"
  LeanTest.assertEqual full.generalizedPrimitiveForce (gravityGeneralizedForce params)
    "Gravity should enter as a primitive generalized force"
  LeanTest.assertEqual full.equation.biasForces (gyroscopicBiasForce params x)
    "Gyroscopic w x Iw should enter as the full-physics bias term"
  LeanTest.assertEqual full.generalizedForces
    (FloatArray.add (rotorGeneralizedForce params u x)
      (gravityGeneralizedForce params))
    "Rotor thrust/torque and primitive gravity should compose before subtracting bias"

  let a := translationalAcceleration params u x
  let alpha := bodyAngularAcceleration params u x
  LeanTest.assertTrue
    (maxAbsDiff full.derivative.vdot
      #[a.x, a.y, a.z, alpha.x, alpha.y, alpha.z] < 1.0e-9)
    s!"Full physics vdot should match the direct rigid-body solve, got {full.derivative.vdot}"

  let rpyDDt ← assertOk
    ((rpy x).rpyDDtFromBodyAngularAcceleration? (rpyDt x) alpha)
    "quadrotor RPY acceleration map"
  let dx ← assertOk (derivative? params u x)
    "quadrotor derivative from full physics"
  LeanTest.assertTrue
    (maxAbsDiff dx
      #[x.getD 6 0.0, x.getD 7 0.0, x.getD 8 0.0,
        x.getD 9 0.0, x.getD 10 0.0, x.getD 11 0.0,
        a.x, a.y, a.z, rpyDDt.x, rpyDDt.y, rpyDDt.z] < 1.0e-9)
    s!"Derivative should compose full-physics acceleration with RPY kinematics, got {reprStr dx}"

@[test]
def testFullPhysicsPrimitiveProviderRecomputesStateAndInput : IO Unit := do
  let provider := fullPhysicsPrimitiveProvider params
    "quadrotor full physics provider test"
  let x0 := defaultState
  let u0 := defaultInput
  let x1 := mkState 0.25 (-0.15) 1.1 0.12 (-0.08) 0.2
    0.35 (-0.2) 0.15 0.08 (-0.05) 0.06
  let u1 : QuadrotorInput := #[1.1, 1.4, 1.2, 1.3]

  let primitive0 ← assertOk
    (provider.validPrimitivesAt? (physicsState x0 u0))
    "quadrotor provider primitive at default state and input"
  let primitive1 ← assertOk
    (provider.validPrimitivesAt? (physicsState x1 u1))
    "quadrotor provider primitive at moved state and input"
  let result1 ← assertOk
    (provider.solveAt? (physicsState x1 u1) 5311)
    "quadrotor provider solve at moved state and input"
  let direct1 ← assertOk
    (solveFullPhysics? params u1 x1 "quadrotor direct provider parity")
    "quadrotor direct solve for provider parity"

  LeanTest.assertEqual primitive0.massMatrix (massMatrix params)
    "Quadrotor provider should expose the same diagonal mass and inertia matrix"
  LeanTest.assertEqual primitive1.massMatrix (massMatrix params)
    "Quadrotor provider mass matrix should stay parameter-derived after state changes"
  LeanTest.assertTrue (maxAbsDiff primitive0.qdot (poseQdot x0) < 1.0e-12)
    s!"Provider qdot at default state should come from the default state, got {reprStr primitive0.qdot}"
  LeanTest.assertTrue (maxAbsDiff primitive1.qdot (poseQdot x1) < 1.0e-12)
    s!"Provider qdot at moved state should come from the moved state, got {reprStr primitive1.qdot}"
  LeanTest.assertTrue
    (maxAbsDiff primitive0.actuationForces primitive1.actuationForces > 1.0)
    "Changing rotor input and attitude should change provider actuation generalized force"
  LeanTest.assertTrue
    (maxAbsDiff primitive0.biasForces primitive1.biasForces > 1.0e-6)
    "Changing angular rates should change provider gyroscopic bias"
  LeanTest.assertTrue
    (maxAbsDiff result1.derivative.vdot direct1.derivative.vdot < 1.0e-12)
    s!"Provider solve should match direct full-physics solve, provider={reprStr result1.derivative.vdot}, direct={reprStr direct1.derivative.vdot}"
  LeanTest.assertEqual result1.move.targets #[5311]
    "Provider solve should use the supplied interval vertex"

  match provider.validPrimitivesAt? (physicsState x1 #[1.0, 2.0]) with
  | .ok _ => LeanTest.fail "Quadrotor provider should reject malformed rotor input"
  | .error _ => pure ()

@[test]
def testRotorMomentAndRpyKinematicsMatchDrakeConventions : IO Unit := do
  let u := #[1.0, 2.0, 3.0, 4.0]
  let tau := bodyTorque params u
  LeanTest.assertTrue (approx tau.x (params.armLength * (2.0 - 4.0)) 1.0e-12)
    s!"Mx should use L*(rotor1 - rotor3), got {reprStr tau}"
  LeanTest.assertTrue (approx tau.y (params.armLength * (3.0 - 1.0)) 1.0e-12)
    s!"My should use L*(rotor2 - rotor0), got {reprStr tau}"
  LeanTest.assertTrue (approx tau.z (params.momentConstant * (1.0 - 2.0 + 3.0 - 4.0)) 1.0e-12)
    s!"Mz should alternate rotor torque signs, got {reprStr tau}"

  let requestedTau : Vec3 := { x := 0.03, y := -0.02, z := 0.004 }
  let totalThrust := 8.5
  let allocated := allocateRotorInput params totalThrust requestedTau
  let allocatedThrust := (rotorBodyForces params allocated).foldl (fun acc f => acc + f) 0.0
  let allocatedTau := bodyTorque params allocated
  LeanTest.assertTrue (approx allocatedThrust totalThrust 1.0e-12)
    s!"Rotor allocation should preserve requested total thrust, got {allocatedThrust}"
  LeanTest.assertTrue (maxAbsDiff allocatedTau.asArray requestedTau.asArray < 1.0e-12)
    s!"Rotor allocation should preserve requested body torque, got {reprStr allocatedTau}"

  let r : Rpy := { roll := 0.2, pitch := -0.3, yaw := 0.4 }
  let qd : Vec3 := { x := 0.7, y := -0.2, z := 0.5 }
  let w := r.bodyAngularVelocityFromRpyDt qd
  let recovered ← assertOk (r.rpyDtFromBodyAngularVelocity? w) "rpy inverse kinematics"
  LeanTest.assertTrue (maxAbsDiff recovered.asArray qd.asArray < 1.0e-12)
    s!"RPY body angular velocity map should invert away from singularity, got {reprStr recovered}"

@[test]
def testPassiveSimulationRecordsContinuousDrop : IO Unit := do
  let t1 := 0.05
  let z0 := 0.051
  let run ← assertOk
    (solvePassive? params (mkState 0.0 0.0 z0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0) 0.0 t1)
    "quadrotor passive solve"
  match run.trace.validate? with
  | .error msg => LeanTest.fail s!"Quadrotor trace should validate: {msg}"
  | .ok () => pure ()
  let expectedZ := z0 - 0.5 * params.gravity * t1 * t1
  let expectedZdot := -params.gravity * t1
  LeanTest.assertTrue (approx run.t1 t1 1.0e-12)
    s!"Quadrotor run should reach requested final time, got {run.t1}"
  LeanTest.assertEqual run.moves.size 4
    "Trace moves plus full-physics support and interval moves should be exposed"
  LeanTest.assertTrue (run.moves[0]!.kind == SkeletonMoveKind.intervalAdjoint)
    "First quadrotor move should be the interval adjoint"
  LeanTest.assertTrue (run.moves[2]!.kind == SkeletonMoveKind.markMarginalize)
    "Quadrotor full physics should expose empty support selection as a mark move"
  LeanTest.assertTrue (run.moves[3]!.kind == SkeletonMoveKind.intervalAdjoint)
    "Quadrotor full physics should expose the rigid-body force solve as an interval move"
  LeanTest.assertEqual run.fullPhysics.generalizedPrimitiveForce
    (gravityGeneralizedForce params)
    "Passive simulation should retain the initial full-physics gravity primitive"
  LeanTest.assertTrue (approx (run.finalState.getD 2 99.0) expectedZ 1.0e-8)
    s!"Passive level quadrotor should follow z(t)=z0-1/2*g*t^2, got {reprStr run.finalState}"
  LeanTest.assertTrue (approx (run.finalState.getD 8 99.0) expectedZdot 1.0e-8)
    s!"Passive level quadrotor should follow zdot(t)=-g*t, got {reprStr run.finalState}"
  LeanTest.assertTrue (Float.isFinite run.finalEnergy)
    s!"Quadrotor final energy should be finite, got {run.finalEnergy}"

@[test]
def testLqrMetadataAndHoverControllerMatchDrakeSetup : IO Unit := do
  let cfg := lqrConfig
  let cost := lqrCostMetadata params cfg
  LeanTest.assertEqual cost.qDiagonal.size 12
    "Drake quadrotor LQR uses a 12x12 Q matrix"
  LeanTest.assertEqual cost.rDiagonal.size 4
    "Drake quadrotor LQR uses a 4x4 R matrix"
  LeanTest.assertTrue (cost.qDiagonal.take 6 |>.all (fun q => approx q 10.0 1.0e-12))
    s!"Pose coordinates should have Drake's 10x LQR weight, got {reprStr cost.qDiagonal}"
  LeanTest.assertTrue (cost.qDiagonal.extract 6 12 |>.all (fun q => approx q 1.0 1.0e-12))
    s!"Velocity coordinates should have unit LQR weight, got {reprStr cost.qDiagonal}"
  LeanTest.assertTrue (cost.rDiagonal.all (fun r => approx r 1.0 1.0e-12))
    s!"Input coordinates should have unit LQR weight, got {reprStr cost.rDiagonal}"
  LeanTest.assertTrue (maxAbsDiff cost.nominalState nominalHoverState < 1.0e-12)
    s!"Nominal hover state should place the quadrotor at z=1, got {reprStr cost.nominalState}"
  LeanTest.assertTrue (maxAbsDiff cost.nominalInput (hoverInput params) < 1.0e-12)
    s!"Nominal input should be hover thrust, got {reprStr cost.nominalInput}"

  let uHover := lqrController params cfg nominalHoverState
  let dx ← assertOk (derivative? params uHover nominalHoverState)
    "quadrotor LQR hover derivative"
  LeanTest.assertTrue (maxAbsDiff uHover (hoverInput params) < 1.0e-12)
    s!"LQR controller should return hover input at the nominal state, got {reprStr uHover}"
  LeanTest.assertTrue (maxAbsDiff dx (Array.replicate 12 0.0) < 1.0e-12)
    s!"Closed-loop nominal hover should be an equilibrium, got {reprStr dx}"

@[test]
def testLqrSimulationStabilizesNominalHover : IO Unit := do
  let run ← assertOk (simulateLqr? params lqrConfig)
    "quadrotor LQR simulation"
  match run.trace.validate? with
  | .error msg => LeanTest.fail s!"Quadrotor LQR trace should validate: {msg}"
  | .ok () => pure ()
  LeanTest.assertTrue (approx run.t1 7.0 1.0e-12)
    s!"Drake LQR run should advance to the seven-second trial duration, got {run.t1}"
  LeanTest.assertEqual run.samples.size (lqrConfig.steps + 1)
    "LQR rollout should include the initial state and every RK4 step"
  LeanTest.assertTrue (run.moves.any (fun move =>
      move.kind == SkeletonMoveKind.localSchurBlock &&
      move.exactness == MoveExactness.controlledApproximation &&
      move.label.contains "LQR"))
    "LQR controller should be recorded as a local solver/controller block"
  LeanTest.assertTrue (run.moves.any (fun move => move.kind == SkeletonMoveKind.intervalAdjoint))
    "LQR rollout should still expose the continuous interval-adjoint primitive"
  LeanTest.assertTrue (run.moves.any (fun move => move.kind == SkeletonMoveKind.markMarginalize))
    "LQR rollout should retain the final full-physics support-selection move"
  LeanTest.assertEqual run.finalFullPhysics.derivative.qdot (poseQdot run.finalState)
    "LQR result should retain a full-physics solve at the final controller state"
  LeanTest.assertTrue (maxAbsDiff run.finalState nominalHoverState < 5.0e-2)
    s!"Closed-loop quadrotor should converge near nominal hover, got {reprStr run.finalState}"
  LeanTest.assertTrue (maxAbsDiff run.finalInput (hoverInput params) < 5.0e-2)
    s!"Final controller output should be near hover input, got {reprStr run.finalInput}"

end Tests.EventSkeletonQuadrotorExample
