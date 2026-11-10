import LeanTest
import Tyr.EventSkeleton.Examples.SceneGraph

namespace Tests.EventSkeletonSceneGraphExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.SceneGraph

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
  | none => LeanTest.fail s!"{label}: expected some, got none"

private def assertError {α : Type} (res : Except String α) (needle label : String) :
    IO Unit := do
  match res with
  | .ok _ => LeanTest.fail s!"{label}: expected error containing {needle}, got ok"
  | .error msg =>
      LeanTest.assertTrue (msg.contains needle)
        s!"{label}: expected error containing {needle}, got {msg}"

private def assertArrayApprox
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

@[test]
def testDrakeReferencesAndBouncingBallProviderRoles : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/scene_graph/bouncing_ball_plant.cc"))
    "SceneGraph example should reference Drake's bouncing-ball plant"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/scene_graph/bouncing_ball_plant.h"))
    "SceneGraph example should reference Drake's bouncing-ball plant declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/scene_graph/bouncing_ball_vector.h"))
    "SceneGraph example should reference Drake's BouncingBallVector declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/scene_graph/bouncing_ball_vector.cc"))
    "SceneGraph example should reference Drake's BouncingBallVector coordinate names"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/scene_graph/solar_system.cc"))
    "SceneGraph example should reference Drake's solar-system registration example"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/scene_graph/solar_system.h"))
    "SceneGraph example should reference Drake's SolarSystem declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/scene_graph/solar_system_run_dynamics.cc"))
    "SceneGraph example should reference Drake's solar-system dynamics executable"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/scene_graph/simple_contact_surface_vis.cc"))
    "SceneGraph example should reference Drake's contact-surface visualization query"

  let provider := bouncingBallSceneGraph bouncingBallParams
  assertOk provider.validate? "bouncing-ball scene graph validation"
  LeanTest.assertEqual provider.sources.size 3
    "Bouncing-ball scene should register ball1, ball2, and anchored sources"
  LeanTest.assertEqual provider.frames.size 2
    "Bouncing-ball scene should register one moving frame per ball"
  LeanTest.assertEqual provider.geometries.size 3
    "Bouncing-ball scene should register two spheres and one anchored half-space"

  let ball ← assertSome (provider.geometryById? ball1GeometryId) "ball geometry lookup"
  LeanTest.assertTrue (ball.hasRole .illustration)
    "Ball geometry should have illustration role"
  LeanTest.assertTrue (ball.hasRole .proximity)
    "Ball geometry should have proximity role"
  LeanTest.assertTrue (ball.hasRole .perception)
    "Ball geometry should have perception role"
  LeanTest.assertEqual ball.properties.renderLabel? (some ball1GeometryId)
    "Perception properties should retain Drake-style render label"
  LeanTest.assertEqual (provider.numGeometriesForFrameWithRole ball1FrameId .proximity) 1
    "Frame-role query should find the ball's proximity geometry"
  LeanTest.assertEqual provider.anchoredGeometries.size 1
    "Ground half-space should be the only anchored geometry"

@[test]
def testSceneGraphAssetCatalogRecordsMeshDependencyClosure : IO Unit := do
  assertOk validateSceneGraphExampleAssetCatalog?
    "SceneGraph example asset catalog"
  assertOk validateSolarSceneGraphAssetUsage?
    "Solar SceneGraph asset usage"
  LeanTest.assertEqual sceneGraphExampleAssets.size 19
  LeanTest.assertEqual sceneGraphGeometryAssets.size 16
  LeanTest.assertEqual sceneGraphSolarAssets.size 12
  LeanTest.assertTrue
    (sceneGraphExampleAssetPaths.any (fun path =>
      path == "../drake/examples/scene_graph/sun.sdf"))
    "SceneGraph asset catalog should include Drake's sun SDF wrapper"
  LeanTest.assertTrue
    (sceneGraphExampleAssetPaths.any (fun path =>
      path == "../drake/examples/scene_graph/rainbow_checker.png"))
    "SceneGraph asset catalog should include the OBJ material texture"

  let sunGltf ← assertSome (findSceneGraphExampleAsset? "sun.gltf")
    "sun glTF asset"
  LeanTest.assertTrue (sunGltf.kind == SceneGraphExampleAssetKind.mesh)
    "sun.gltf should be recorded as a mesh asset"
  LeanTest.assertEqual sunGltf.dependencies #["sun.bin", "sun.png", "sun.ktx2"]
    "sun.gltf should record its binary and texture dependencies"
  LeanTest.assertTrue sunGltf.feedsSceneGraph
    "sun.gltf should feed SceneGraph geometry"

  let sunSdf ← assertSome (findSceneGraphExampleAsset? "sun.sdf")
    "sun SDF asset"
  LeanTest.assertTrue (sunSdf.kind == SceneGraphExampleAssetKind.model)
    "sun.sdf should be recorded as a model asset"
  LeanTest.assertEqual sunSdf.dependencies #["sun.gltf"]
    "sun.sdf should record its glTF visual dependency"

  let cuboctaObj ← assertSome
    (findSceneGraphExampleAsset? "../drake/examples/scene_graph/cuboctahedron_with_hole.obj")
    "cuboctahedron OBJ asset by full path"
  LeanTest.assertTrue (cuboctaObj.kind == SceneGraphExampleAssetKind.mesh)
    "cuboctahedron OBJ should be recorded as a mesh asset"
  LeanTest.assertTrue
    (cuboctaObj.dependencies.any (fun dep => dep == "cuboctahedron_with_hole.mtl"))
    s!"cuboctahedron OBJ should record material dependency, got {cuboctaObj.dependencies}"
  LeanTest.assertTrue
    (cuboctaObj.dependencies.any (fun dep => dep == "rainbow_checker.png"))
    s!"cuboctahedron OBJ should record texture dependency, got {cuboctaObj.dependencies}"

@[test]
def testBouncingBallVectorBoundaryMatchesDrakeNamedVector : IO Unit := do
  assertOk bouncingBallVectorSpec.validate? "BouncingBallVector spec validation"
  LeanTest.assertEqual bouncingBallVectorSpec.vectorName "BouncingBallVector"
    "SceneGraph bouncing-ball state should retain Drake's generated-vector name"
  LeanTest.assertEqual bouncingBallVectorSpec.numCoordinates 2
    "BouncingBallVector should have two coordinates"
  LeanTest.assertEqual bouncingBallVectorSpec.qCount 1
    "BouncingBallVector should declare one position coordinate"
  LeanTest.assertEqual bouncingBallVectorSpec.vCount 1
    "BouncingBallVector should declare one velocity coordinate"
  LeanTest.assertEqual bouncingBallVectorSpec.zCount 0
    "BouncingBallVector should have no miscellaneous continuous state"
  LeanTest.assertEqual bouncingBallVectorSpec.coordinateNames #["z", "zdot"]
    "BouncingBallVector coordinate names should match Drake"
  LeanTest.assertEqual bouncingBallVectorZIndex 0
    "BouncingBallVector z index should match Drake"
  LeanTest.assertEqual bouncingBallVectorZdotIndex 1
    "BouncingBallVector zdot index should match Drake"
  LeanTest.assertEqual (bouncingBallVectorSpec.coordinateName? 0) (some "z")
    "Coordinate-name lookup should expose z at index 0"
  LeanTest.assertEqual (bouncingBallVectorSpec.coordinateName? 1) (some "zdot")
    "Coordinate-name lookup should expose zdot at index 1"

  let x : BouncingBallSceneState := { z := 0.4, zdot := -1.2 }
  LeanTest.assertEqual x.toArray #[0.4, -1.2]
    "BouncingBallSceneState should lower to Drake BasicVector row order"
  let roundTrip ← assertOk (BouncingBallSceneState.ofArray? #[0.4, -1.2])
    "BouncingBallVector array round trip"
  LeanTest.assertTrue (roundTrip == x)
    s!"BouncingBallVector array conversion should preserve z and zdot, got {reprStr roundTrip}"
  LeanTest.assertEqual (x.withZ 0.7).toArray #[0.7, -1.2]
    "with_z semantics should update only z"
  LeanTest.assertEqual (x.withZdot 2.5).toArray #[0.4, 2.5]
    "with_zdot semantics should update only zdot"
  LeanTest.assertEqual x.serialize #[("z", 0.4), ("zdot", -1.2)]
    "Serialize should visit z then zdot with Drake's field names"
  assertError (BouncingBallSceneState.ofArray? #[1.0])
    "input size 1 != 2"
    "BouncingBallVector should reject short BasicVector payloads"
  let nanValue := 0.0 / 0.0
  assertError (BouncingBallSceneState.ofArray? #[nanValue, 0.0])
    "values must be finite"
    "BouncingBallVector should reject NaN payloads"

@[test]
def testSceneGraphQueryEmitsPointPairAndContactCandidateViews : IO Unit := do
  let provider := bouncingBallSceneGraph bouncingBallParams
  let contactState : BouncingBallSceneState := { z := 0.03, zdot := -0.2 }
  let candidate ← assertOk (contactCandidate? provider 1 contactState)
    "sphere-half-space contact candidate"
  LeanTest.assertTrue (approx candidate.signedDistance (-0.02) 1.0e-12)
    s!"Sphere center at z=0.03 with radius 0.05 should penetrate by 0.02, got {candidate.signedDistance}"
  LeanTest.assertTrue (approx candidate.normalVelocity (-0.2) 1.0e-12)
    s!"Candidate should preserve vertical velocity as normal velocity, got {candidate.normalVelocity}"
  LeanTest.assertEqual candidate.bodyA "ball"
    "Candidate should identify the moving sphere geometry"
  LeanTest.assertEqual candidate.bodyB "ground"
    "Candidate should identify the anchored half-space geometry"
  assertArrayApprox candidate.point_W #[0.25, 0.25, -0.02] 1.0e-12
    "Candidate should expose the world contact point on the sphere surface"
  assertArrayApprox candidate.normal_W #[0.0, 0.0, 1.0] 1.0e-12
    "Candidate should expose the half-space contact normal"
  LeanTest.assertEqual candidate.normalJacobian #[1.0]
    "Candidate should expose the generalized-velocity normal row"
  LeanTest.assertEqual candidate.tangentJacobian #[0.0]
    "Candidate should expose a tangent row with the same generalized width"

  let pair? ← assertOk (pointPairPenetration? bouncingBallParams 1 contactState)
    "point-pair penetration query"
  let pair ← assertSome pair? "penetrating point pair"
  assertOk pair.validate? "point-pair validation"
  LeanTest.assertEqual pair.idA ball1GeometryId
    "Point pair should identify the sphere geometry"
  LeanTest.assertEqual pair.idB groundGeometryId
    "Point pair should identify the ground geometry"
  LeanTest.assertTrue (approx pair.depth 0.02 1.0e-12)
    s!"Point pair depth should be 0.02, got {pair.depth}"

  let separated? ← assertOk (pointPairPenetration? bouncingBallParams 1 ball1InitialState)
    "separated point-pair query"
  match separated? with
  | some pair => LeanTest.fail s!"Separated initial ball should not report penetration, got {reprStr pair}"
  | none => pure ()

@[test]
def testHuntCrossleyDynamicsUsesProviderContactView : IO Unit := do
  LeanTest.assertTrue
    (approx (bouncingBallParams.drakeStiffnessFromStaticPenetration 0.001) 981.0 1.0e-12)
    "Drake stiffness should be m*g/0.001 = 981 N/m"

  let freeDx ← assertOk (derivative? bouncingBallParams 1 ball1InitialState)
    "free-space derivative"
  LeanTest.assertTrue (approx freeDx.normalForce 0.0 1.0e-12)
    s!"Separated ball should have zero contact force, got {freeDx.normalForce}"
  LeanTest.assertTrue (approx freeDx.zddot (-bouncingBallParams.gravity) 1.0e-12)
    s!"Separated ball should accelerate by gravity, got {freeDx.zddot}"

  let contactState : BouncingBallSceneState := { z := 0.03, zdot := -0.1 }
  let contactDx ← assertOk (derivative? bouncingBallParams 1 contactState)
    "penetrating derivative"
  LeanTest.assertTrue (approx contactDx.normalForce 19.62 1.0e-10)
    s!"Depth 0.02 with k=981 should produce 19.62 N, got {contactDx.normalForce}"
  LeanTest.assertTrue (approx contactDx.zddot 186.39 1.0e-10)
    s!"Net acceleration should be (-m*g+fN)/m = 186.39, got {contactDx.zddot}"
  LeanTest.assertTrue contactDx.isFinite
    s!"Contact derivative should be finite, got {reprStr contactDx}"

@[test]
def testSceneGraphQueryFeedsFullPhysicsPrimitives : IO Unit := do
  let p := bouncingBallParams
  let provider := bouncingBallSceneGraph p
  let contactState : BouncingBallSceneState := { z := 0.03, zdot := -0.1 }
  let candidate ← assertOk (contactCandidate? provider 1 contactState)
    "bouncing-ball contact candidate"
  let query : SceneContactQueryResult := {
    providerLabel := provider.label
    candidates := ContactCandidateSet.ofArray #[candidate] "bouncing-ball solver candidates"
    label := "bouncing-ball scene contact query"
  }
  let model : CompliantContactModel := {
    normalStiffness := p.stiffness
    normalDamping := p.stiffness * p.dissipation
    tangentDamping := 0.0
    friction := CoulombFriction.frictionless
    label := "bouncing-ball Hunt-Crossley primitive"
  }
  let primitives ← assertOk
    (fullPhysicsPrimitivesFromSceneContactQuery?
      query
      #[#[p.mass]]
      #[contactState.zdot]
      #[0.0]
      #[p.mass * p.gravity]
      (.threshold p.distanceTol)
      .compliantModel
      #[]
      model
      p.distanceTol
      1.0e-9
      "bouncing-ball scene full physics")
    "SceneGraph query to full-physics primitive adapter"
  let result ← assertOk (primitives.solve? 40)
    "SceneGraph query full-physics solve"
  LeanTest.assertEqual result.support.selectedLocalIndices #[0]
    "The full-physics primitive should select the penetrating SceneGraph candidate"
  LeanTest.assertEqual result.support.totalCandidates 1
    "The full-physics primitive should preserve the dynamic SceneGraph candidate count"
  LeanTest.assertTrue (approx result.contactForces[0]!.normalForce 19.62 1.0e-10)
    s!"Full physics should compute the same normal force from primitive contact rows, got {result.contactForces[0]!.normalForce}"
  LeanTest.assertEqual result.generalizedContactForce.size 1
    "The primitive J^T f boundary should produce one generalized force for BouncingBallVector.v"
  LeanTest.assertTrue (approx (result.derivative.vdot.getD 0 0.0) 186.39 1.0e-10)
    s!"Full physics should solve M vdot = J^T f - gravity, got {result.derivative.vdot}"
  LeanTest.assertEqual result.move.label "full-physics-step:bouncing-ball scene full physics"
    "The SceneGraph-backed solve should still be represented as the standard full-physics interval move"

  let pair? ← assertOk (pointPairPenetration? p 1 contactState)
    "bouncing-ball point-pair penetration"
  let pair ← assertSome pair? "penetrating point pair"
  let pointPairOnlyQuery : SceneContactQueryResult := {
    providerLabel := provider.label
    pointPairs := #[pair]
    useStrictHydro := false
    label := "point-pair-only query"
  }
  assertError
    (fullPhysicsPrimitivesFromSceneContactQuery?
      pointPairOnlyQuery
      #[#[p.mass]]
      #[contactState.zdot]
      #[0.0]
      #[p.mass * p.gravity])
    "ContactCandidate views"
    "point-pair-only SceneGraph query full physics"

@[test]
def testSceneGraphFullPhysicsPrimitiveProviderRecomputesBallState :
    IO Unit := do
  let p := bouncingBallParams
  let provider :=
    bouncingBallSceneFullPhysicsPrimitiveProvider
      "bouncing-ball SceneGraph dynamic provider test"
  let airborne :=
    bouncingBallScenePhysicsState p 1 { z := 0.30, zdot := 0.1 }
  let contact :=
    bouncingBallScenePhysicsState p 1 { z := 0.03, zdot := -0.1 }
  let airbornePrimitives ← assertOk (provider.validPrimitivesAt? airborne)
    "airborne SceneGraph provider primitives"
  let contactPrimitives ← assertOk (provider.validPrimitivesAt? contact)
    "contact SceneGraph provider primitives"
  LeanTest.assertEqual airbornePrimitives.contactCandidates.size 1
    "SceneGraph provider should expose the current dynamic candidate view"
  LeanTest.assertTrue (approx (airbornePrimitives.qdot.getD 0 0.0) 0.1 1.0e-12)
    s!"SceneGraph provider qdot should come from the airborne state, got {reprStr airbornePrimitives.qdot}"
  LeanTest.assertTrue (approx (contactPrimitives.qdot.getD 0 0.0) (-0.1) 1.0e-12)
    s!"SceneGraph provider qdot should come from the contact state, got {reprStr contactPrimitives.qdot}"
  let airborneSupport ← assertOk (provider.supportAt? airborne)
    "airborne SceneGraph provider support"
  LeanTest.assertEqual airborneSupport.selectedLocalIndices #[]
    "Airborne ball state should recompute to empty support"
  let contactSupport ← assertOk (provider.supportAt? contact)
    "contact SceneGraph provider support"
  LeanTest.assertEqual contactSupport.selectedLocalIndices #[0]
    "Penetrating ball state should recompute to selected support"
  let full ← assertOk (provider.solveAt? contact 41)
    "contact SceneGraph provider full physics"
  LeanTest.assertTrue (approx full.contactForces[0]!.normalForce 19.62 1.0e-10)
    s!"Provider full physics should compute Hunt-Crossley normal force, got {full.contactForces[0]!.normalForce}"
  LeanTest.assertTrue (approx (full.derivative.vdot.getD 0 0.0) 186.39 1.0e-10)
    s!"Provider full physics should solve M vdot from current SceneGraph contact, got {reprStr full.derivative.vdot}"
  LeanTest.assertEqual full.move.label
    "full-physics-step:bouncing-ball SceneGraph dynamic provider test"
    "Provider solve should expose the standard full-physics interval move"

  assertError
    (provider.validPrimitivesAt?
      { params := p, ballIndex := 3, state := contact.state })
    "ballIndex 1 or 2"
    "invalid SceneGraph provider ball index"
  assertError
    (provider.validPrimitivesAt?
      (bouncingBallScenePhysicsState p 1 { z := (0.0 / 0.0), zdot := 0.0 }))
    "state must be finite"
    "invalid SceneGraph provider state"

@[test]
def testBouncingBallEndToEndGraphAndSupport : IO Unit := do
  let result ← assertOk (buildBouncingBall? bouncingBallParams)
    "bouncing-ball scene graph end-to-end"
  LeanTest.assertEqual result.vectorSpec.coordinateNames bouncingBallVectorCoordinateNames
    "End-to-end result should expose Drake BouncingBallVector coordinate names"
  LeanTest.assertEqual result.ball1StateVector ball1InitialState.toArray
    "End-to-end result should expose ball1 state in BouncingBallVector row order"
  LeanTest.assertEqual result.ball2StateVector ball2InitialState.toArray
    "End-to-end result should expose ball2 state in BouncingBallVector row order"
  LeanTest.assertEqual result.support.candidates.size 2
    "Forward pass should evaluate both ball-ground candidate pairs"
  LeanTest.assertEqual result.support.selectedLocalIndices.size 0
    "Default initial heights should not retain any active contact"
  let runtime ← assertOk result.support.toRuntimeSupport? "contact support runtime conversion"
  LeanTest.assertEqual runtime.totalCandidates? (some 2)
    "Runtime support should remember the dynamic candidate count"
  LeanTest.assertEqual runtime.selectedIds #[]
    "Separated contact support should expose no selected ids"

  let pose ← assertSome (result.poses.poseForFrame? ball2FrameId)
    "ball2 pose lookup"
  LeanTest.assertTrue (approx pose.translation.x (-0.25) 1.0e-12)
    s!"Ball2 x pose should be -0.25, got {pose.translation.x}"
  LeanTest.assertTrue (approx pose.translation.y (-0.25) 1.0e-12)
    s!"Ball2 y pose should be -0.25, got {pose.translation.y}"
  LeanTest.assertTrue (approx pose.translation.z 0.3 1.0e-12)
    s!"Ball2 z pose should be 0.3, got {pose.translation.z}"

  LeanTest.assertEqual result.moves.size 3
    "SceneGraph bouncing-ball graph should include query, interval, and pose-output moves"
  LeanTest.assertTrue (result.graph.containsMoveKind .localSchurBlock)
    "SceneGraph query should be represented as a local provider block"
  LeanTest.assertTrue (result.graph.containsMoveKind .intervalAdjoint)
    "Compliant dynamics should retain interval adjoint move"
  LeanTest.assertTrue (result.graph.containsMoveKind .checkpointBoundary)
    "Pose output should be represented as a checkpoint boundary"

@[test]
def testSolarSystemProviderExercisesSceneGraphShapeRegistry : IO Unit := do
  let result ← assertOk buildSolarSystem? "solar system scene graph"
  LeanTest.assertEqual result.assetCatalog.size sceneGraphExampleAssets.size
  assertOk result.provider.validate? "solar provider validation"
  LeanTest.assertEqual result.provider.sources.size 1
    "Solar system should register one geometry source"
  LeanTest.assertEqual result.provider.frames.size solarBodyCount
    "Solar system should register seven orbit frames"
  LeanTest.assertEqual result.provider.geometries.size 14
    "Solar system should register anchored sun/post, bodies, arms, rings, and satellites"
  LeanTest.assertEqual result.provider.anchoredGeometries.size 2
    "Sun and post should be anchored geometry"
  LeanTest.assertEqual (result.provider.numGeometriesForFrameWithRole earthFrameId .illustration) 3
    "Earth orbit frame should own Earth plus two arm geometries"

  let shapeNames := result.provider.shapeNames
  for shapeName in #["sphere", "cylinder", "mesh", "convex", "box", "capsule"] do
    LeanTest.assertTrue (shapeNames.any (fun name => name == shapeName))
      s!"Solar system should exercise {shapeName} geometry"

  let marsRings ← assertSome (result.provider.geometryById? 1261)
    "Mars rings geometry lookup"
  match marsRings.shape with
  | .mesh uri scale _ =>
      LeanTest.assertEqual uri "../drake/examples/scene_graph/planet_rings.obj"
        "Mars rings should point at Drake's mesh asset"
      LeanTest.assertTrue (approx scale 0.24 1.0e-12)
        s!"Mars rings mesh scale should match Mars size, got {scale}"
  | other => LeanTest.fail s!"Mars rings should be mesh geometry, got {reprStr other}"

  let sun ← assertSome (result.provider.geometryById? 1201)
    "Sun geometry lookup"
  match sun.shape with
  | .mesh uri scale supportingFiles =>
      LeanTest.assertEqual uri "../drake/examples/scene_graph/sun.gltf"
        "Sun should point at Drake's glTF mesh asset"
      LeanTest.assertTrue (approx scale 1.0 1.0e-12)
        s!"Sun glTF scale should be 1.0, got {scale}"
      LeanTest.assertEqual supportingFiles #["sun.bin", "sun.png", "sun.ktx2"]
        "Sun glTF should carry binary and texture dependencies"
  | other => LeanTest.fail s!"Sun should be mesh geometry, got {reprStr other}"

@[test]
def testSolarSystemStateDerivativeAndFramePoseOutput : IO Unit := do
  let result ← assertOk buildSolarSystem? "solar system scene graph"
  LeanTest.assertEqual result.defaultState.size 14
    "Solar state should contain seven angles and seven rates"
  LeanTest.assertEqual result.derivative.size 14
    "Solar derivative should preserve state size"
  LeanTest.assertTrue (approx (result.derivative.getD 0 0.0) (2.0 * pi / 5.0) 1.0e-12)
    s!"Earth derivative should use 5-second revolution rate, got {result.derivative.getD 0 0.0}"
  LeanTest.assertTrue (approx (result.derivative.getD 6 0.0) (2.0 * pi / 1.1) 1.0e-12)
    s!"Phobos derivative should use 1.1-second revolution rate, got {result.derivative.getD 6 0.0}"
  for i in [:solarBodyCount] do
    LeanTest.assertTrue (approx (result.derivative.getD (i + solarBodyCount) 1.0) 0.0 1.0e-12)
      s!"Solar angular accelerations should be zero, got {result.derivative.getD (i + solarBodyCount) 1.0}"

  LeanTest.assertEqual result.framePoses.poses.size solarBodyCount
    "Solar frame pose output should contain one pose per orbit frame"
  let lunaPose ← assertSome (result.framePoses.poseForFrame? lunaFrameId)
    "Luna orbit pose"
  LeanTest.assertTrue (approx lunaPose.rotationAngle (pi / 2.0) 1.0e-12)
    s!"Luna initial rotation should be pi/2, got {lunaPose.rotationAngle}"
  let phobosPose ← assertSome (result.framePoses.poseForFrame? phobosFrameId)
    "Phobos orbit pose"
  LeanTest.assertTrue (approx phobosPose.rotationAxis.z (-1.0) 1.0e-12)
    s!"Phobos should orbit around the negated z axis, got {reprStr phobosPose.rotationAxis}"

@[test]
def testSolarSystemRunDynamicsExecutableBoundary : IO Unit := do
  let result ← assertOk buildSolarRunDynamics?
    "solar_system_run_dynamics executable boundary"
  LeanTest.assertEqual result.assetCatalog.size sceneGraphExampleAssets.size
  LeanTest.assertTrue (approx result.params.simulationTime 13.0 1.0e-12)
    s!"Default simulation_time should match Drake, got {result.params.simulationTime}"
  LeanTest.assertTrue (approx result.params.maximumStepSize 0.002 1.0e-12)
    s!"Simulator maximum step size should match Drake, got {result.params.maximumStepSize}"
  LeanTest.assertTrue (approx result.params.targetRealtimeRate 1.0 1.0e-12)
    s!"Target realtime rate should match Drake, got {result.params.targetRealtimeRate}"
  LeanTest.assertTrue result.params.addDrakeVisualizer
    "DrakeVisualizerd::AddToBuilder should be represented"
  LeanTest.assertTrue result.params.addMeshcatVisualizer
    "MeshcatVisualizer::AddToBuilder should be represented"
  LeanTest.assertEqual result.sourceId solarSourceId
    "SolarSystem source_id should feed SceneGraph's source pose port"
  LeanTest.assertEqual result.provider.label "scene_graph_solar_system"
  LeanTest.assertEqual result.provider.frames.size solarBodyCount
  LeanTest.assertEqual result.initialFramePoses.poses.size solarBodyCount
  LeanTest.assertEqual result.defaultState solarDefaultState
  LeanTest.assertEqual result.initialDerivative.size result.defaultState.size
  LeanTest.assertEqual result.graph.vertices.size 8
  LeanTest.assertEqual result.moves.size 4
  LeanTest.assertTrue (result.graph.containsMoveKind .localSchurBlock)
    "DiagramBuilder system creation should be an exact local block"
  LeanTest.assertTrue (result.graph.containsMoveKind .checkpointBoundary)
    "Pose-port wiring and visualizer sinks should be checkpoint boundaries"
  LeanTest.assertTrue (result.graph.containsMoveKind .intervalAdjoint)
    "Simulator.AdvanceTo should be represented as the executable interval"
  LeanTest.assertTrue
    (result.moves.any (fun m => m.label.contains "SolarSystem.geometry_pose_output"))
    "Graph should record the pose-output to SceneGraph source-pose connection"
  LeanTest.assertTrue
    (result.moves.any (fun m => m.label.contains "DrakeVisualizerd"))
    "Graph should record Drake visualizer setup"
  LeanTest.assertTrue
    (result.moves.any (fun m => m.label.contains "MeshcatVisualizer"))
    "Graph should record Meshcat visualizer setup"
  LeanTest.assertTrue
    (result.moves.any (fun m => m.label.contains "AdvanceTo"))
    "Graph should record the simulator advance boundary"
  LeanTest.assertTrue
    (result.moves.any (fun m =>
      approx m.cost.work (13.0 / 0.002) 1.0e-9))
    "Simulator interval cost should record the default step-count scale"
  LeanTest.assertTrue (result.moves.all (fun m => m.exactness == MoveExactness.exact))
    "The solar-system executable boundary should be exact for the represented diagram"

@[test]
def testSimpleContactSurfaceVisProviderAndFlags : IO Unit := do
  let result ← assertOk buildContactSurfaceVis?
    "simple_contact_surface_vis provider boundary"
  assertOk result.provider.validate? "contact-surface provider validation"
  LeanTest.assertTrue (approx result.params.simulationTime 10.0 1.0e-12)
    s!"Default simulation_time should match Drake, got {result.params.simulationTime}"
  LeanTest.assertTrue (approx result.params.length 1.0 1.0e-12)
    s!"Default hydroelastic length should match Drake, got {result.params.length}"
  LeanTest.assertTrue result.params.rigidCylinders
    "Default contact-surface visualization should register rigid hydroelastic cylinders"
  LeanTest.assertTrue result.params.useStrictHydro
    "Default contact-surface visualization should use strict hydroelastic queries"
  LeanTest.assertTrue
    (result.params.surfaceRepresentation == HydroelasticSurfaceRepresentation.triangle)
    "Default contact-surface visualization should use triangle surfaces"
  LeanTest.assertTrue (approx result.params.targetRealtimeRate 1.0 1.0e-12)
    s!"Default realtime target should be 1, got {result.params.targetRealtimeRate}"
  LeanTest.assertTrue (approx result.params.maximumStepSize 0.002 1.0e-12)
    s!"Maximum step size should match Drake, got {result.params.maximumStepSize}"
  LeanTest.assertTrue (approx result.params.publishPeriod (1.0 / 64.0) 1.0e-12)
    s!"CONTACT_RESULTS publish period should be 1/64, got {result.params.publishPeriod}"

  LeanTest.assertEqual result.provider.sources.size 2
    "Contact-surface example should register moving_ball and world sources"
  LeanTest.assertEqual result.provider.frames.size 2
    "Contact-surface example should register moving_frame and double_can frames"
  LeanTest.assertEqual result.provider.geometries.size 4
    "Contact-surface example should register ball, box, can1, and can2 geometries"
  let movingFrame ← assertSome (result.provider.frameById? contactSurfaceMovingBallFrameId)
    "moving frame lookup"
  LeanTest.assertEqual movingFrame.frameGroup 1
    "MovingBall should retain Drake frame group 1"
  let canFrame ← assertSome (result.provider.frameById? contactSurfaceCanFrameId)
    "double_can frame lookup"
  LeanTest.assertEqual canFrame.frameGroup 2
    "FixedCylinders should retain Drake frame group 2"

  let ball ← assertSome (result.provider.geometryById? contactSurfaceBallGeometryId)
    "moving ball geometry"
  match ball.shape with
  | .sphere radius =>
      LeanTest.assertTrue (approx radius 1.0 1.0e-12)
        s!"Moving ball radius should be 1, got {radius}"
  | other => LeanTest.fail s!"Moving ball should be a sphere, got {reprStr other}"
  match ball.properties.diffuseRgba? with
  | some rgba =>
      LeanTest.assertTrue (approx rgba.a 0.25 1.0e-12)
        s!"Moving ball illustration alpha should be 0.25, got {rgba.a}"
  | none => LeanTest.fail "Moving ball should have Drake-style illustration color"
  match ball.properties.hydroelastic? with
  | some (.compliant resolution modulus) =>
      LeanTest.assertTrue (approx resolution 1.0 1.0e-12)
        s!"Moving ball compliant resolution hint should be 1, got {resolution}"
      LeanTest.assertTrue (approx modulus 1.0e8 1.0e-6)
        s!"Moving ball hydroelastic modulus should be 1e8, got {modulus}"
  | other => LeanTest.fail s!"Moving ball should have compliant hydroelastic metadata, got {reprStr other}"

  let box ← assertSome (result.provider.geometryById? contactSurfaceBoxGeometryId)
    "anchored box geometry"
  match box.shape with
  | .box sx sy sz =>
      LeanTest.assertTrue (approx sx 10.0 1.0e-12 && approx sy 10.0 1.0e-12 && approx sz 10.0 1.0e-12)
        s!"Anchored box dimensions should be 10, got {(sx, sy, sz)}"
  | other => LeanTest.fail s!"Anchored world geometry should be a box, got {reprStr other}"
  LeanTest.assertTrue (approx box.X_FG.rotationAngle (pi / 4.0) 1.0e-12)
    s!"Anchored box should be rotated pi/4 about x, got {box.X_FG.rotationAngle}"
  LeanTest.assertTrue (approx box.X_FG.translation.z (-((Float.sqrt 2.0) * 10.0 / 2.0)) 1.0e-12)
    s!"Anchored box z translation should match Drake, got {box.X_FG.translation.z}"
  match box.properties.hydroelastic? with
  | some (.rigid resolution) =>
      LeanTest.assertTrue (approx resolution 10.0 1.0e-12)
        s!"Anchored box rigid resolution hint should be edge length 10, got {resolution}"
  | other => LeanTest.fail s!"Anchored box should have rigid hydroelastic metadata, got {reprStr other}"

  for id in #[contactSurfaceCan1GeometryId, contactSurfaceCan2GeometryId] do
    let can ← assertSome (result.provider.geometryById? id)
      s!"can geometry {id}"
    match can.shape with
    | .cylinder radius length =>
        LeanTest.assertTrue (approx radius 0.5 1.0e-12 && approx length 1.0 1.0e-12)
          s!"Can should be Cylinder(0.5, 1.0), got radius={radius}, length={length}"
    | other => LeanTest.fail s!"Can geometry should be a cylinder, got {reprStr other}"
    match can.properties.hydroelastic? with
    | some (.rigid resolution) =>
        LeanTest.assertTrue (approx resolution 0.5 1.0e-12)
          s!"Rigid can resolution hint should be 0.5, got {resolution}"
    | other => LeanTest.fail s!"Rigid can should have rigid hydroelastic metadata, got {reprStr other}"

@[test]
def testSimpleContactSurfaceVisStrictHydroAndFallbackSemantics : IO Unit := do
  let strict ← assertOk buildContactSurfaceVis?
    "strict simple_contact_surface_vis"
  LeanTest.assertTrue strict.contactResult.query.useStrictHydro
    "Default ContactResultMaker should call ComputeContactSurfaces"
  LeanTest.assertEqual strict.contactResult.hydroelasticPatches.size 3
    "Strict rigid-cylinder mode should report box, can1, and can2 hydroelastic patches"
  LeanTest.assertEqual strict.contactResult.pointPairs.size 0
    "Strict hydroelastic mode should not include fallback point-pair contacts"
  LeanTest.assertTrue
    (strict.contactResult.query.representation == HydroelasticSurfaceRepresentation.triangle)
    "Default strict result should use triangle surface representation"
  LeanTest.assertEqual strict.contactResult.query.totalPrimitiveContacts 6
    "Query should expose three patches plus three solver-facing candidate views"
  LeanTest.assertEqual strict.contactResult.candidateSet.totalCandidates 3
    "Hydroelastic patches should project to three solver-facing contact candidates"
  let support := strict.contactResult.query.hydroelasticSupport 0.0
  LeanTest.assertEqual support.selectedLocalIndices.size 3
    "All positive-area hydroelastic patches should be selected by zero area threshold"
  let selectedForces ← assertOk support.selectedContactForces?
    "selected hydroelastic contact forces"
  LeanTest.assertEqual selectedForces.size 3
    "Hydroelastic patch support should feed full-physics force primitives"
  LeanTest.assertTrue (approx selectedForces[0]!.normalForce 1.2 1.0e-12)
    s!"First fake Drake contact force should be 1.2, got {selectedForces[0]!.normalForce}"

  let polygonResult ← assertOk (buildContactSurfaceVis?
      { contactSurfaceVisParams with polygons := true })
    "polygon simple_contact_surface_vis"
  LeanTest.assertTrue
    (polygonResult.contactResult.query.representation == HydroelasticSurfaceRepresentation.polygon)
    "FLAGS_polygons should switch the contact surface representation to polygon"

  assertError
    (buildContactSurfaceVis? { contactSurfaceVisParams with rigidCylinders := false })
    "strict hydroelastic requires rigid cylinder"
    "strict non-rigid cylinder hydroelastic query"

  let hybrid ← assertOk (buildContactSurfaceVis?
      { contactSurfaceVisParams with rigidCylinders := false, hybrid := true })
    "hybrid fallback simple_contact_surface_vis"
  LeanTest.assertTrue (!hybrid.contactResult.query.useStrictHydro)
    "FLAGS_hybrid should use ComputeContactSurfacesWithFallback"
  LeanTest.assertEqual hybrid.contactResult.hydroelasticPatches.size 1
    "Hybrid non-rigid-cylinder mode should retain only the rigid box hydroelastic patch"
  LeanTest.assertEqual hybrid.contactResult.pointPairs.size 2
    "Hybrid non-rigid-cylinder mode should report point-pair fallback contacts for the cans"
  LeanTest.assertTrue hybrid.contactResult.query.hasFallbackPointPairs
    "Hybrid query result should explicitly record fallback point pairs"
  LeanTest.assertEqual hybrid.contactResult.candidateSet.totalCandidates 3
    "Hybrid query should still expose all contacts through solver-facing candidates"
  let can1 ← assertSome (hybrid.provider.geometryById? contactSurfaceCan1GeometryId)
    "hybrid can1 lookup"
  LeanTest.assertTrue can1.properties.hydroelastic?.isNone
    "Non-rigid fallback can should omit hydroelastic metadata"
  LeanTest.assertTrue (can1.hasRole .proximity)
    "Non-rigid fallback can should still retain the proximity role"

@[test]
def testSimpleContactSurfaceVisGraphAndMovingBallBoundary : IO Unit := do
  let state : ContactSurfaceMovingBallState := { z := 0.25, zdot := 0.0 }
  let result ← assertOk
    (buildContactSurfaceVis?
      { contactSurfaceVisParams with realTime := false, forceFullName := true }
      state
      (pi / 2.0))
    "simple_contact_surface_vis graph and moving-ball boundary"
  LeanTest.assertTrue (approx result.params.targetRealtimeRate 0.0 1.0e-12)
    s!"FLAGS_real_time=false should set target realtime rate to 0, got {result.params.targetRealtimeRate}"
  assertArrayApprox result.movingBallDerivative #[1.0, 0.0] 1.0e-12
    "MovingBall derivatives should be [sin(t), 0]"
  LeanTest.assertTrue (approx result.movingBallPose.X_WF.translation.z 0.25 1.0e-12)
    s!"MovingBall pose output should translate by state z, got {result.movingBallPose.X_WF.translation.z}"
  LeanTest.assertTrue (approx result.contactResult.timestampMicros ((pi / 2.0) * 1.0e6) 1.0e-6)
    s!"Contact result timestamp should be context time in microseconds, got {result.contactResult.timestampMicros}"
  LeanTest.assertEqual result.contactResult.publishChannel contactSurfaceLcmChannel
    "Contact result should publish on CONTACT_RESULTS"
  LeanTest.assertEqual result.graph.vertices.size 9
    "Contact-surface graph should include flags, diagram, scene graph, systems, sinks, and simulator interval"
  LeanTest.assertEqual result.moves.size 5
    "Contact-surface graph should include build, pose, query, publisher, and simulator moves"
  LeanTest.assertTrue (result.graph.containsMoveKind .localSchurBlock)
    "SceneGraph and ContactResultMaker should be local provider blocks"
  LeanTest.assertTrue (result.graph.containsMoveKind .checkpointBoundary)
    "Pose and publisher wiring should be checkpoint boundaries"
  LeanTest.assertTrue (result.graph.containsMoveKind .intervalAdjoint)
    "Simulator.AdvanceTo should be the executable interval"
  LeanTest.assertTrue
    (result.moves.any (fun m => m.label.contains "ContactResultMaker"))
    "Graph should record ContactResultMaker query-object boundary"
  LeanTest.assertTrue
    (result.moves.any (fun m => m.label.contains "CONTACT_RESULTS"))
    "Graph should record the LCM contact-results publisher"
  LeanTest.assertTrue
    (result.moves.any (fun m => approx m.cost.work (10.0 / 0.002) 1.0e-9))
    "Simulator interval cost should record the default step-count scale"
  let firstPatch := result.contactResult.hydroelasticPatches[0]!
  LeanTest.assertTrue (firstPatch.bodyA.contains "MovingBall::ball")
    s!"force_full_name should emit unique moving-ball body name, got {firstPatch.bodyA}"

@[test]
def testSceneGraphExampleEndToEnd : IO Unit := do
  let result ← assertOk buildEndToEnd? "SceneGraph example end-to-end"
  LeanTest.assertEqual result.bouncingBall.provider.geometries.size 3
    "End-to-end result should include the bouncing-ball provider"
  LeanTest.assertEqual result.solarSystem.provider.frames.size solarBodyCount
    "End-to-end result should include the solar-system provider"
  LeanTest.assertEqual result.solarRunDynamics.provider.geometries.size
    result.solarSystem.provider.geometries.size
    "End-to-end result should reuse the solar-system provider for the executable boundary"
  LeanTest.assertTrue (result.solarRunDynamics.graph.containsMoveKind .intervalAdjoint)
    "End-to-end SceneGraph result should include the solar-system simulator interval"
  LeanTest.assertEqual result.contactSurfaceVis.provider.geometries.size 4
    "End-to-end result should include the contact-surface visualization provider"
  LeanTest.assertTrue (result.contactSurfaceVis.graph.containsMoveKind .localSchurBlock)
    "End-to-end SceneGraph result should include the contact-surface query block"
  LeanTest.assertTrue (result.contactSurfaceVis.graph.containsMoveKind .intervalAdjoint)
    "End-to-end SceneGraph result should include the contact-surface simulator interval"

end Tests.EventSkeletonSceneGraphExample
