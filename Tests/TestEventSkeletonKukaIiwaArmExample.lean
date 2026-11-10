import LeanTest
import Tyr.EventSkeleton.Examples.KukaIiwaArm

namespace Tests.EventSkeletonKukaIiwaArmExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.KukaIiwaArm

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) :
    IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertError {α : Type} (res : Except String α) (label : String) :
    IO String := do
  match res with
  | .ok _ => LeanTest.fail s!"{label}: expected error, got ok"
  | .error msg => pure msg

private def assertSome {α : Type} (value : Option α) (label : String) :
    IO α := do
  match value with
  | some value => pure value
  | none => LeanTest.fail s!"{label}: expected some value"

private def assertArrayNear
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

@[test]
def testDrakeReferencesGainsAndPortLayoutAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/kuka_iiwa_arm/kuka_torque_controller.cc"))
    "Example should reference Drake's torque-controller implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/kuka_iiwa_arm/test/kuka_torque_controller_test.cc"))
    "Example should reference Drake's torque-controller tests"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/kuka_iiwa_arm/iiwa_common.cc"))
    "Example should reference Drake's Iiwa gain helper"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/kuka_iiwa_arm/iiwa_common.h"))
    "Example should reference Drake's Iiwa common helper declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/kuka_iiwa_arm/iiwa_lcm.cc"))
    "Example should reference Drake's Iiwa LCM implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/kuka_iiwa_arm/lcm_plan_interpolator.cc"))
    "Example should reference Drake's LCM plan interpolator"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/kuka_iiwa_arm/iiwa_controller.cc"))
    "Example should reference Drake's Iiwa controller executable"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/kuka_iiwa_arm/kuka_simulation.cc"))
    "Example should reference Drake's LCM-facing Kuka simulation executable"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/kuka_iiwa_arm/kuka_plan_runner.cc"))
    "Example should reference Drake's plan-runner executable"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/kuka_iiwa_arm/move_iiwa_ee.cc"))
    "Example should reference Drake's end-effector plan publisher"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/manipulation/kuka_iiwa/iiwa_constants.cc"))
    "Example should reference Drake's Iiwa constants"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path.contains "iiwa14_polytope_collision.urdf"))
    "Example should record the Iiwa URDF model loaded by Drake's test"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path.contains "iiwa14_no_collision.urdf"))
    "Example should record the no-collision Iiwa URDF used by the plan runner"
  LeanTest.assertTrue
    (iiwaExampleModelCatalogPaths.any (fun path =>
      path == "../drake/examples/kuka_iiwa_arm/models/objects/block_for_pick_and_place.urdf"))
    "Model catalog should include Drake's pick-and-place block"
  LeanTest.assertTrue
    (iiwaExampleModelCatalogPaths.any (fun path =>
      path == "../drake/examples/kuka_iiwa_arm/models/objects/open_top_box.urdf"))
    "Model catalog should include Drake's open-top box"
  LeanTest.assertTrue
    (iiwaExampleModelCatalogPaths.any (fun path =>
      path == "../drake/examples/kuka_iiwa_arm/models/table/extra_heavy_duty_table.sdf"))
    "Model catalog should include Drake's extra-heavy-duty table"
  LeanTest.assertTrue
    (iiwaExampleModelCatalogPaths.any (fun path =>
      path == "../drake/examples/kuka_iiwa_arm/models/desk/transcendesk55inch.sdf"))
    "Model catalog should include Drake's TranscenDesk model"
  for path in #[
      "../drake/examples/kuka_iiwa_arm/models/objects/block_for_pick_and_place_large_size.urdf",
      "../drake/examples/kuka_iiwa_arm/models/objects/block_for_pick_and_place_mid_size.urdf",
      "../drake/examples/kuka_iiwa_arm/models/objects/folding_table.urdf",
      "../drake/examples/kuka_iiwa_arm/models/objects/round_table.urdf",
      "../drake/examples/kuka_iiwa_arm/models/objects/simple_cuboid.urdf",
      "../drake/examples/kuka_iiwa_arm/models/objects/simple_cylinder.urdf",
      "../drake/examples/kuka_iiwa_arm/models/objects/yellow_post.urdf",
      "../drake/examples/kuka_iiwa_arm/models/table/extra_heavy_duty_table_surface_only_collision.sdf"
    ] do
    LeanTest.assertTrue (iiwaExampleModelCatalogPaths.any (fun candidate => candidate == path))
      s!"Model catalog should include Drake asset {path}"

  LeanTest.assertEqual numJoints 7
  LeanTest.assertEqual jointNames.size 7
  LeanTest.assertEqual stateCoordinateNames.size 14
  LeanTest.assertEqual torqueCoordinateNames.size 7
  LeanTest.assertEqual torqueControlledGains.stiffness
    #[1000.0, 1000.0, 1000.0, 500.0, 500.0, 500.0, 500.0]
  LeanTest.assertTrue
    (torqueControlledGains.dampingRatio.all (fun ratio => approx ratio 1.0 1.0e-12))
    "Torque-controlled Iiwa damping ratios should default to 1"
  LeanTest.assertTrue
    (positionControlledKd.all (fun kd => approx kd 20.0 1.0e-12))
    s!"Position-controlled Kd should be 2*sqrt(100)=20, got {positionControlledKd}"

@[test]
def testExampleModelCatalogRecordsManipulationAssets : IO Unit := do
  assertOk validateIiwaExampleModelCatalog? "Kuka model catalog"
  LeanTest.assertEqual iiwaExampleModelAssets.size 15
  LeanTest.assertEqual iiwaPhysicalModelAssets.size 13
  LeanTest.assertEqual iiwaExampleObjectAssets.size 10
  LeanTest.assertEqual iiwaExampleSupportAssets.size 3
  LeanTest.assertTrue
    (iiwaPhysicalModelAssets.all (fun asset =>
      asset.hasInertial && asset.hasCollisionGeometry && asset.hasVisualGeometry))
    "Every physical Kuka catalog asset should expose inertial, collision, and visual facts"

  let block ← assertSome
    (findIiwaExampleModelAsset? "objects/block_for_pick_and_place.urdf")
    "pick-and-place block asset"
  LeanTest.assertTrue (block.kind == IiwaExampleModelKind.object)
    "Pick-and-place block should be an object asset"
  LeanTest.assertEqual block.fullPath
    "../drake/examples/kuka_iiwa_arm/models/objects/block_for_pick_and_place.urdf"
  LeanTest.assertEqual block.modelName "simple_cuboid"
  match block.mass? with
  | some mass =>
      LeanTest.assertTrue (approx mass 0.1 1.0e-12)
        s!"Pick-and-place block mass should be 0.1kg, got {mass}"
  | none => LeanTest.fail "Pick-and-place block mass should be recorded"
  LeanTest.assertTrue (block.representativeShape.contains "0.059 0.059 0.199")
    s!"Pick-and-place block should record collision dimensions, got {block.representativeShape}"
  LeanTest.assertTrue (block.contactRole.contains "pick-and-place")
    s!"Pick-and-place block should record its contact role, got {block.contactRole}"

  let blackBox ← assertSome
    (findIiwaExampleModelAsset? "../drake/examples/kuka_iiwa_arm/models/objects/black_box.urdf")
    "black box asset by full path"
  LeanTest.assertEqual blackBox.modelName "simple_cuboid"
  match blackBox.mass? with
  | some mass =>
      LeanTest.assertTrue (approx mass 0.122 1.0e-12)
        s!"Black box mass should be 0.122kg, got {mass}"
  | none => LeanTest.fail "Black box mass should be recorded"

  let openTopBox ← assertSome
    (findIiwaExampleModelAsset? "objects/open_top_box.urdf")
    "open-top box asset"
  LeanTest.assertEqual openTopBox.modelName "open_top_box"
  LeanTest.assertTrue (openTopBox.representativeShape.contains "five box panels")
    s!"Open-top box should record its panel collision model, got {openTopBox.representativeShape}"

  let simpleCylinder ← assertSome
    (findIiwaExampleModelAsset? "objects/simple_cylinder.urdf")
    "simple cylinder asset"
  LeanTest.assertEqual simpleCylinder.modelName "simple_cylinder"
  LeanTest.assertTrue (simpleCylinder.representativeShape.contains "radius 0.0325")
    s!"Simple cylinder should record the Drake cylinder radius, got {simpleCylinder.representativeShape}"

  let table ← assertSome
    (findIiwaExampleModelAsset? "table/extra_heavy_duty_table_surface_only_collision.sdf")
    "surface-only table asset"
  LeanTest.assertTrue (table.kind == IiwaExampleModelKind.table)
    "Surface-only table should be categorized as a table support asset"
  LeanTest.assertTrue (table.contactRole.contains "tabletop-only")
    s!"Surface-only table should preserve tabletop-only collision semantics, got {table.contactRole}"

@[test]
def testExampleModelPrimitiveGeometryRecordsCollisionShapes : IO Unit := do
  assertOk validateIiwaExampleModelPrimitiveGeometry?
    "Kuka primitive geometry catalog"
  LeanTest.assertTrue (iiwaPrimitiveGeometryCatalog.size > iiwaPhysicalModelAssets.size)
    "Primitive geometry catalog should expose multiple geometry entries per physical asset"

  let largeBlock ← assertSome
    (findIiwaExampleModelAsset? "objects/block_for_pick_and_place_large_size.urdf")
    "large pick-and-place block asset"
  let largeGeometry := iiwaModelPrimitiveGeometry largeBlock
  LeanTest.assertEqual largeGeometry.size 10
    "Large pick-and-place block should expose visual box, inset collision box, and eight point probes"
  LeanTest.assertEqual (iiwaModelCollisionGeometry largeBlock).size 9
    "Large pick-and-place block collision geometry should keep box plus eight probes"
  let largeCollisionBox ← assertSome
    (largeGeometry.find? (fun geometry => geometry.name == "inset_collision_box"))
    "large block inset collision box"
  match largeCollisionBox.shape with
  | .box sx sy sz =>
      LeanTest.assertTrue
        (approx sx 0.059 1.0e-12 && approx sy 0.089 1.0e-12 &&
          approx sz 0.199 1.0e-12)
        s!"Large block inset collision dimensions should match URDF, got {sx}, {sy}, {sz}"
  | other => LeanTest.fail s!"Large block inset collision should be a box, got {reprStr other}"

  let blackBox ← assertSome
    (findIiwaExampleModelAsset? "objects/black_box.urdf")
    "black box asset"
  LeanTest.assertEqual (iiwaModelPrimitiveGeometry blackBox).size 14
    "Black box should include visual box, collision box, eight corner probes, and four center probes"

  let simpleCylinder ← assertSome
    (findIiwaExampleModelAsset? "objects/simple_cylinder.urdf")
    "simple cylinder asset"
  let cylinderCollision ← assertSome
    ((iiwaModelCollisionGeometry simpleCylinder).find? (fun geometry =>
      geometry.name == "collision_cylinder"))
    "simple cylinder collision primitive"
  match cylinderCollision.shape with
  | .cylinder radius length =>
      LeanTest.assertTrue
        (approx radius 0.0325 1.0e-12 && approx length 0.130 1.0e-12)
        s!"Simple cylinder dimensions should match URDF, got radius={radius}, length={length}"
  | other => LeanTest.fail s!"Simple cylinder collision should be a cylinder, got {reprStr other}"

  let surfaceOnlyTable ← assertSome
    (findIiwaExampleModelAsset? "table/extra_heavy_duty_table_surface_only_collision.sdf")
    "surface-only table asset"
  LeanTest.assertEqual (iiwaModelVisualGeometry surfaceOnlyTable).size 9
    "Surface-only table should retain all nine visual boxes"
  LeanTest.assertEqual (iiwaModelCollisionGeometry surfaceOnlyTable).size 1
    "Surface-only table should intentionally expose only tabletop collision geometry"
  let surface ← assertSome
    ((iiwaModelCollisionGeometry surfaceOnlyTable).find? (fun geometry =>
      geometry.name == "surface"))
    "surface-only table collision primitive"
  match surface.shape with
  | .box sx sy sz =>
      LeanTest.assertTrue
        (approx sx 0.7112 1.0e-12 && approx sy 0.762 1.0e-12 &&
          approx sz 0.057 1.0e-12 &&
          approx surface.X_LG.translation.z 0.736 1.0e-12)
        s!"Surface-only table collision should match SDF tabletop, got {sx}, {sy}, {sz} at {surface.X_LG.translation.z}"
  | other => LeanTest.fail s!"Surface-only table collision should be a box, got {reprStr other}"

  let fullTable ← assertSome
    (findIiwaExampleModelAsset? "table/extra_heavy_duty_table.sdf")
    "full table asset"
  LeanTest.assertEqual (iiwaModelCollisionGeometry fullTable).size 9
    "Full table should preserve leg, crossbar, and surface collision boxes"

@[test]
def testGravityOnlyControllerIgnoresDesiredStateWhenGainsAreZero : IO Unit := do
  let output ← assertOk
    (evaluateTorqueController? gravityOnlyGains)
    "gravity-only torque controller"
  assertArrayNear output.springTorque zeroTorque 1.0e-12
    "Zero stiffness should produce zero virtual spring torque"
  assertArrayNear output.dampingTorque zeroTorque 1.0e-12
    "Zero damping ratio should produce zero damping torque"
  assertArrayNear output.controlTorque drakeTestProvider.gravityCompensationTorque 1.0e-12
    "Gravity-only controller output should equal gravity compensation torque"

@[test]
def testVirtualSpringTorqueMatchesDrakePidComposition : IO Unit := do
  let output ← assertOk
    (evaluateTorqueController? drakeTestSpringGains)
    "spring torque controller"
  let q := drakeTestEstimatedState.q
  let qd := drakeTestDesiredState.q
  let mut expectedSpring : Array Float := #[]
  for i in [:numJoints] do
    expectedSpring :=
      expectedSpring.push
        (drakeTestSpringGains.stiffness[i]! * (qd[i]! - q[i]!))
  assertArrayNear output.springTorque expectedSpring 1.0e-12
    "Virtual spring should compute Kp * (q_desired - q)"
  assertArrayNear output.dampingTorque zeroTorque 1.0e-12
    "Spring test damping ratios are zero"
  assertArrayNear output.controlTorque
    (FloatArray.add drakeTestProvider.gravityCompensationTorque expectedSpring)
    1.0e-12
    "Controller should add gravity compensation and virtual spring torque"

@[test]
def testStateDependentDampingUsesMassMatrixDiagonal : IO Unit := do
  let output ← assertOk
    (evaluateTorqueController? drakeTestDampingGains)
    "state-dependent damping torque controller"
  let v := drakeTestEstimatedState.v
  let mut expectedDampingGains : Array Float := #[]
  let mut expectedDamping : Array Float := #[]
  for i in [:numJoints] do
    let gain :=
      drakeTestDampingGains.dampingRatio[i]! *
        2.0 *
        Float.sqrt
          (drakeTestProvider.massMatrixDiagonal[i]! *
            drakeTestDampingGains.stiffness[i]!)
    expectedDampingGains := expectedDampingGains.push gain
    expectedDamping := expectedDamping.push (-v[i]! * gain)
  assertArrayNear output.dampingGains expectedDampingGains 1.0e-12
    "Damping gain should be ratio * 2 * sqrt(Hii * stiffness)"
  assertArrayNear output.dampingTorque expectedDamping 1.0e-12
    "Damping torque should be -gain * measured velocity"

  let expectedControl :=
    FloatArray.add
      (FloatArray.add drakeTestProvider.gravityCompensationTorque output.springTorque)
      expectedDamping
  assertArrayNear output.controlTorque expectedControl 1.0e-12
    "Controller should add gravity, spring, and state-dependent damping torques"

@[test]
def testFullPhysicsAdvanceUsesExistingPrimitives : IO Unit := do
  let result ← assertOk buildEndToEnd?
    "kuka iiwa full-physics end-to-end"
  LeanTest.assertEqual result.modelCatalog.size iiwaExampleModelAssets.size
  assertOk result.plantStep.validate? "Iiwa full plant step"
  LeanTest.assertEqual result.plantStep.model.modelUri iiwaModelUrl
  LeanTest.assertEqual result.plantStep.model.numPositions numJoints
  LeanTest.assertEqual result.plantStep.model.numVelocities numJoints
  LeanTest.assertEqual result.plantStep.model.numActuatedDofs numJoints
  LeanTest.assertTrue (approx result.plantStep.config.timeStep iiwaLcmStatusPeriod 1.0e-12)
    s!"Full plant step should use the Iiwa status period, got {result.plantStep.config.timeStep}"
  assertArrayNear result.plantStep.q0 drakeTestEstimatedState.q 1.0e-12
    "Full plant step should start from measured q"
  assertArrayNear result.plantStep.v0 drakeTestEstimatedState.v 1.0e-12
    "Full plant step should start from measured v"
  assertArrayNear result.plantStep.actuation result.controllerOutput.controlTorque 1.0e-12
    "Full plant step actuation should be the torque controller output"
  let generalizedActuation ← assertOk
    (iiwaGeneralizedActuation? result.plantStep.actuation)
    "Iiwa generalized actuation map"
  assertArrayNear generalizedActuation result.controllerOutput.controlTorque 1.0e-12
    "Fixed-base Iiwa actuation map should be the identity"

  let primitives ← assertOk
    (fullPhysicsPrimitivesFromController? drakeTestProvider drakeTestEstimatedState result.controllerOutput)
    "Iiwa full physics primitive bundle"
  LeanTest.assertTrue (primitives.supportPolicy == SupportPolicy.fullSupport)
    "Iiwa provider should lower no-contact dynamics through a full-support primitive selection"
  LeanTest.assertTrue (primitives.contactForceSource == ContactForceSource.precomputed)
    "Iiwa provider should expose contact-force provenance at the primitive boundary"
  LeanTest.assertEqual primitives.contactCandidates.size 0
  LeanTest.assertEqual primitives.contactForces.size 0
  let primitiveEquation ← assertOk primitives.equation?
    "Iiwa full physics primitive equation"
  LeanTest.assertEqual primitiveEquation.massMatrix
    (FloatMatrix.diagonal drakeTestProvider.massMatrixDiagonal)
  LeanTest.assertEqual primitiveEquation.biasForces
    drakeTestProvider.gravityCompensationTorque

  LeanTest.assertEqual result.fullPhysics.equation.massMatrix
    (FloatMatrix.diagonal drakeTestProvider.massMatrixDiagonal)
  LeanTest.assertEqual result.fullPhysics.equation.biasForces
    drakeTestProvider.gravityCompensationTorque
  LeanTest.assertEqual result.fullPhysics.contactForces.size 0
  LeanTest.assertEqual result.fullPhysics.support.totalCandidates 0
  assertArrayNear result.fullPhysics.generalizedForces result.controllerOutput.controlTorque 1.0e-12
    "No active contacts means generalized forces equal controller actuation"
  let expectedRhs :=
    FloatArray.sub result.controllerOutput.controlTorque
      drakeTestProvider.gravityCompensationTorque
  assertArrayNear result.fullPhysics.derivative.rhs expectedRhs 1.0e-12
    "Full physics rhs should subtract gravity bias from gravity-compensated actuation"
  let mut expectedVdot : Array Float := #[]
  for i in [:numJoints] do
    expectedVdot :=
      expectedVdot.push (expectedRhs[i]! / drakeTestProvider.massMatrixDiagonal[i]!)
  assertArrayNear result.fullPhysics.derivative.vdot expectedVdot 1.0e-12
    "Full physics primitive should solve M vdot = tau - bias"
  LeanTest.assertTrue (result.fullPhysics.move.kind == SkeletonMoveKind.intervalAdjoint)
    "Full physics advance should be represented as an interval-adjoint move"
  LeanTest.assertTrue (result.fullPhysics.move.exactness == MoveExactness.exact)
    "No contacts or learned surrogate are used in this primitive plant step"

@[test]
def testFullPhysicsPrimitiveProviderRecomputesControllerAndPlantPrimitives :
    IO Unit := do
  let provider := fullPhysicsPrimitiveProvider torqueControlledGains
    "iiwa test dynamic full physics provider"
  let dynamicProvider : IiwaMultibodyProviderData :=
    { drakeTestProvider with
      massMatrixDiagonal := #[5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0]
      gravityCompensationTorque := #[2.0, -1.5, 1.0, -0.5, 0.25, -0.125, 0.0625]
      label := "iiwa dynamic test provider" }
  let estimated : IiwaState :=
    {
      q := #[0.05, -0.10, 0.15, -0.20, 0.25, -0.30, 0.35]
      v := #[0.20, -0.15, 0.10, -0.05, 0.04, -0.03, 0.02]
    }
  let desired : IiwaState :=
    {
      q := #[0.15, 0.00, -0.05, 0.05, -0.10, 0.10, -0.15]
      v := #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }
  let commandedTorque : Array Float :=
    #[0.5, -0.4, 0.3, -0.2, 0.1, -0.05, 0.025]
  let snapshot :=
    torqueControlPhysicsState dynamicProvider estimated desired commandedTorque
  let controller ← assertOk (snapshot.controllerOutput? torqueControlledGains)
    "dynamic Iiwa torque-control snapshot"
  let primitives ← assertOk (provider.validPrimitivesAt? snapshot)
    "dynamic Iiwa full physics provider"
  let directPrimitives ← assertOk
    (fullPhysicsPrimitivesFromController? dynamicProvider estimated controller)
    "direct dynamic Iiwa full physics primitives"
  LeanTest.assertEqual primitives.massMatrix directPrimitives.massMatrix
    "Provider should use the mass primitive from the current multibody snapshot"
  assertArrayNear primitives.qdot estimated.v 1.0e-12
    "Provider should use the current measured velocity as qdot"
  assertArrayNear primitives.actuationForces controller.controlTorque 1.0e-12
    "Provider should recompute controller torque from the current snapshot"
  assertArrayNear primitives.biasForces dynamicProvider.gravityCompensationTorque 1.0e-12
    "Provider should use the current gravity-bias primitive"
  let staleController ← assertOk
    (evaluateTorqueController? torqueControlledGains)
    "default Iiwa torque-controller output"
  LeanTest.assertTrue
    (FloatArray.maxAbsDiff primitives.actuationForces staleController.controlTorque > 1.0e-3)
    "Dynamic provider output should not reuse the default fixture controller torque"

  let fullPhysics ← assertOk (provider.solveAt? snapshot 8123)
    "dynamic Iiwa full physics solve"
  LeanTest.assertEqual fullPhysics.move.targets #[8123]
    "Provider solve should target the supplied interval vertex"
  let expectedRhs :=
    FloatArray.sub controller.controlTorque dynamicProvider.gravityCompensationTorque
  assertArrayNear fullPhysics.derivative.rhs expectedRhs 1.0e-12
    "Provider solve should assemble M vdot = tau(snapshot) - bias(snapshot)"
  let mut expectedVdot : Array Float := #[]
  for i in [:numJoints] do
    expectedVdot :=
      expectedVdot.push (expectedRhs[i]! / dynamicProvider.massMatrixDiagonal[i]!)
  assertArrayNear fullPhysics.derivative.vdot expectedVdot 1.0e-12
    "Provider solve should use the current mass primitive"

  let badMsg ← assertError
    (provider.validPrimitivesAt? { snapshot with commandedTorque := #[1.0] })
    "Iiwa provider malformed commanded torque"
  LeanTest.assertTrue (badMsg.contains "commandedTorque")
    s!"Malformed commanded torque should fail at provider validation, got {badMsg}"

@[test]
def testRuntimePlanInterpolatorAndMoveEeBoundaries : IO Unit := do
  let result ← assertOk buildRuntimeBoundaries?
    "kuka runtime boundaries"
  LeanTest.assertEqual result.modelCatalog.size iiwaExampleModelAssets.size
  assertOk result.planInterpolator.validate? "LcmPlanInterpolator boundary"
  assertOk result.moveEndEffector.validate? "move_iiwa_ee boundary"
  assertOk result.simulation.validate? "kuka_simulation boundary"
  assertOk result.planRunner.validate? "kuka_plan_runner boundary"
  LeanTest.assertEqual result.channels.statusChannel "IIWA_STATUS"
  LeanTest.assertEqual result.channels.commandChannel "IIWA_COMMAND"
  LeanTest.assertEqual result.channels.planChannel "COMMITTED_ROBOT_PLAN"
  LeanTest.assertTrue (result.planInterpolator.interpolatorType == IiwaPlanInterpolatorType.cubic)
    "Drake's Iiwa controller defaults to cubic interpolation"
  LeanTest.assertEqual result.planInterpolator.stateInputPort
    "status_receiver_lcmt_iiwa_status"
  LeanTest.assertEqual result.planInterpolator.planInputPort
    "plan_interpolator_plan"
  LeanTest.assertEqual result.planInterpolator.commandOutputPort
    "command_sender_lcmt_iiwa_command"
  LeanTest.assertTrue
    (approx result.planInterpolator.defaultPlanUpdateInterval 0.1 1.0e-12)
    s!"Expected 0.1s plan update interval, got {result.planInterpolator.defaultPlanUpdateInterval}"
  LeanTest.assertTrue
    (approx result.planInterpolator.statusPeriod 0.005 1.0e-12)
    s!"Expected 0.005s Iiwa status period, got {result.planInterpolator.statusPeriod}"
  LeanTest.assertTrue result.planInterpolator.initializeFromFirstStatus
    "LcmPlanInterpolator must be initialized from the first measured status"
  LeanTest.assertEqual result.planInterpolator.holdPlanStartTimeSource
    "first lcmt_iiwa_status utime"
  LeanTest.assertEqual result.planInterpolator.holdPlanQ0Source
    "first lcmt_iiwa_status joint_position_measured"
  LeanTest.assertEqual result.moveEndEffector.baseFrame "base"
  LeanTest.assertEqual result.moveEndEffector.endEffectorFrame "iiwa_link_ee"
  LeanTest.assertEqual result.moveEndEffector.ikSamples 100
  LeanTest.assertEqual result.moveEndEffector.jointVelocityLimits iiwaMaxJointVelocities
  LeanTest.assertTrue result.moveEndEffector.waitsForStatus
    "move_iiwa_ee waits for a measured Iiwa status before solving IK"
  LeanTest.assertEqual result.moveEndEffector.statusPositionField
    "joint_position_measured"
  LeanTest.assertTrue result.moveEndEffector.planFailureIsNonfatal
    "Drake's move_iiwa_ee reports planning failure and exits without crashing"
  LeanTest.assertEqual result.simulation.modelUri iiwaModelUrl
  LeanTest.assertEqual result.simulation.controllerSystem "InverseDynamicsController"
  LeanTest.assertFalse result.simulation.usesCommandedTorqueInput
    "Default kuka_simulation mode should be position control"
  LeanTest.assertTrue
    (approx result.simulation.simDt 0.003 1.0e-12)
    s!"Expected Drake's default sim_dt=0.003, got {result.simulation.simDt}"
  LeanTest.assertTrue
    (approx result.simulation.statusPeriod iiwaLcmStatusPeriod 1.0e-12)
    s!"Expected IIWA status period, got {result.simulation.statusPeriod}"
  LeanTest.assertTrue result.simulation.weldsBaseToWorld
    "kuka_simulation welds the Iiwa base frame to world"
  LeanTest.assertTrue result.simulation.usesSceneGraphVisualization
    "kuka_simulation applies visualization config through SceneGraph"
  LeanTest.assertFalse result.simulation.addsFloor
    "kuka_simulation records Drake's floor TODO instead of pretending a floor exists"

  let torqueSimulation :=
    { result.simulation with controlMode := KukaSimulationControlMode.torqueControl }
  assertOk torqueSimulation.validate? "torque-control kuka_simulation boundary"
  LeanTest.assertEqual torqueSimulation.controllerSystem "KukaTorqueController"
  LeanTest.assertTrue torqueSimulation.usesCommandedTorqueInput
    "Torque-control simulation should wire commanded torque into KukaTorqueController"

  LeanTest.assertEqual result.planRunner.modelUri iiwaNoCollisionModelUrl
  LeanTest.assertEqual result.planRunner.stopChannel "STOP"
  LeanTest.assertEqual result.planRunner.jointCount numJoints
  LeanTest.assertTrue result.planRunner.waitsForFirstStatus
    "kuka_plan_runner waits for first status before processing plans"
  LeanTest.assertEqual result.planRunner.minKnotPoints 2
  LeanTest.assertEqual result.planRunner.firstKnotSource "status.joint_position_commanded"
  LeanTest.assertEqual result.planRunner.interpolation
    "PiecewisePolynomial::CubicWithContinuousSecondDerivatives"
  LeanTest.assertTrue result.planRunner.replacesActivePlan
    "New robot plans should replace the active plan"
  LeanTest.assertTrue result.planRunner.stopDiscardsPlan
    "STOP should reset the active plan to none"
  LeanTest.assertTrue result.planRunner.ignoresUnknownJointNames
    "Plan runner should ignore joints that are not in the Iiwa plant"
  LeanTest.assertEqual result.planRunner.commandTimestampSource "status.utime"
  LeanTest.assertEqual result.planRunner.commandPositionField "joint_position"

  LeanTest.assertEqual result.graph.vertices.size 10
  LeanTest.assertEqual result.graph.moves.size 11
  LeanTest.assertTrue (result.graph.containsMoveKind .freezeControl)
    "Runtime graph should expose the initial hold-plan freeze"
  LeanTest.assertTrue (result.graph.containsMoveKind .clockedUpdate)
    "Runtime graph should expose the status-clocked command publish"
  LeanTest.assertTrue (result.graph.containsMoveKind .resetTranspose)
    "Runtime graph should expose STOP as a reset of the active plan"
  LeanTest.assertTrue (result.graph.containsMoveKind .intervalAdjoint)
    "Runtime graph should expose kuka_simulation as a full-physics plant interval"
  LeanTest.assertTrue
    (result.graph.moves.any (fun move =>
      move.label.contains "LcmPlanInterpolator cubic"))
    "Runtime graph should include the cubic plan-to-command adapter"
  LeanTest.assertTrue
    (result.graph.moves.any (fun move =>
      move.label.contains "move_iiwa_ee IK plan"))
    "Runtime graph should include the end-effector IK plan publisher"
  LeanTest.assertTrue
    (result.graph.moves.any (fun move =>
      move.label.contains "kuka_simulation position-control"))
    "Runtime graph should include Drake's default position-control simulation path"
  LeanTest.assertTrue
    (result.graph.moves.any (fun move =>
      move.label.contains "kuka_plan_runner replaces active plan"))
    "Runtime graph should include plan-replacement semantics"
  LeanTest.assertTrue
    (result.graph.moves.any (fun move =>
      move.label.contains "STOP reset discards active plan"))
    "Runtime graph should include STOP reset semantics"

@[test]
def testEndToEndGraphKeepsControllerAsExactLocalBlock : IO Unit := do
  let result ← assertOk buildEndToEnd?
    "kuka iiwa torque-controller end-to-end"
  LeanTest.assertEqual result.references.size drakeReferences.size
  LeanTest.assertEqual result.modelCatalog.size iiwaExampleModelAssets.size
  LeanTest.assertEqual result.provider.massMatrixDiagonal.size numJoints
  LeanTest.assertEqual result.controllerOutput.controlTorque.size numJoints
  LeanTest.assertEqual result.graph.vertices.size 8
  LeanTest.assertEqual result.graph.moves.size 2
  LeanTest.assertTrue (result.graph.containsMoveKind .localSchurBlock)
    "The algebraic torque controller should be exposed as an exact local block"
  LeanTest.assertTrue (result.graph.containsMoveKind .intervalAdjoint)
    "The closed-loop example should also expose the full-physics plant interval"
  LeanTest.assertEqual result.graph.moves[0]!.targets #[8103, 8104]
  LeanTest.assertEqual result.graph.moves[0]!.writes #[8105]
  LeanTest.assertTrue (result.graph.moves[0]!.exactness == MoveExactness.exact)
    "No learned or approximate move is needed for the controller algebra"
  LeanTest.assertEqual result.graph.moves[1]!.targets #[fullPhysicsIntervalVertex]
  LeanTest.assertEqual result.graph.moves[1]!.writes #[8107]
  LeanTest.assertTrue (result.graph.moves[1]!.exactness == MoveExactness.exact)
    "The fixture plant step is an exact composition of existing physics primitives"

end Tests.EventSkeletonKukaIiwaArmExample
