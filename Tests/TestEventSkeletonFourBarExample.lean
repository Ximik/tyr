import LeanTest
import Tyr.EventSkeleton.Examples.FourBar

namespace Tests.EventSkeletonFourBarExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.FourBar

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def piTest : Float := 3.14159265358979323846

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertSome {α : Type} (value : Option α) (label : String) : IO α := do
  match value with
  | some x => pure x
  | none => LeanTest.fail s!"{label}: expected some, got none"

private def assertArrayNear
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

private def getResult : IO FourBarResult := do
  match buildEndToEnd? params with
  | .ok result => pure result
  | .error msg => LeanTest.fail s!"Four-bar example failed to build: {msg}"

@[test]
def testDrakeReferencesAndMultibodyNamesAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/four_bar/four_bar.sdf"))
    "Example should reference Drake's four_bar.sdf model"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/four_bar/passive_simulation.cc"))
    "Example should reference Drake's passive_simulation.cc driver"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/four_bar/BUILD.bazel"))
    "Example should reference Drake's Bazel executable/data declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/four_bar/README.md"))
    "Example should reference Drake's four-bar README derivation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/four_bar/dev/four_bar_loop.sdf"))
    "Example should reference Drake's aspirational direct-loop SDF"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/four_bar/dev/four_bar_weld.sdf"))
    "Example should reference Drake's split-coupler weld-constraint SDF"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/four_bar/images/FourBarLinkageGeometry.png"))
    "Example should reference Drake's geometry diagram"

  LeanTest.assertEqual jointNames #["joint_WA", "joint_AB", "joint_WC"]
    "Joint names should preserve Drake's SDF revolute joint names"
  LeanTest.assertEqual bushingFrameNames #["Bc_bushing", "Cb_bushing"]
    "Bushing frame names should preserve Drake's cut loop frames"
  LeanTest.assertEqual inputCoordinateNames #["applied_torque"]
    "The port keeps Drake's single WA applied torque input"

@[test]
def testFourBarAssetCatalogRecordsModelVariantsAndDiagrams : IO Unit := do
  assertOk validateFourBarExampleAssetCatalog?
    "FourBar example asset catalog"
  LeanTest.assertEqual fourBarExampleAssets.size 8
    "Catalog should cover every file in Drake's four_bar example directory"
  LeanTest.assertEqual fourBarModelFilegroupAssets.size 3
    "Bazel models filegroup should include the main SDF and two dev SDF variants"
  LeanTest.assertEqual fourBarModelVariantAssets.size 3
    "Catalog should classify all three SDF modeling variants"
  LeanTest.assertEqual fourBarImageAssets.size 2
    "Catalog should include the README schematic and geometry diagrams"

  let main ← assertSome
    (findFourBarExampleAsset? "four_bar.sdf")
    "main four_bar.sdf asset"
  LeanTest.assertTrue main.activePassiveModel
    "four_bar.sdf should be the active passive_simulation model"
  LeanTest.assertTrue main.inModelFilegroup
    "four_bar.sdf should be included in the Bazel models filegroup"
  LeanTest.assertTrue
    (main.modelVariant? == some FourBarModelVariant.bushingCut)
    "four_bar.sdf should be the bushing-cut model variant"

  let loop ← assertSome
    (findFourBarExampleAsset? "dev/four_bar_loop.sdf")
    "direct-loop dev SDF asset"
  LeanTest.assertTrue
    (loop.modelVariant? == some FourBarModelVariant.directLoop)
    "dev/four_bar_loop.sdf should be the direct-loop variant"
  LeanTest.assertTrue loop.inModelFilegroup
    "dev/four_bar_loop.sdf should be swept into Drake's models filegroup"

  let weld ← assertSome
    (findFourBarExampleAsset? "dev/four_bar_weld.sdf")
    "split-coupler weld dev SDF asset"
  LeanTest.assertTrue
    (weld.modelVariant? == some FourBarModelVariant.splitCouplerWeld)
    "dev/four_bar_weld.sdf should be the split-coupler weld-constraint variant"
  LeanTest.assertTrue
    (weld.localDependencies.contains "dev/four_bar_loop.sdf")
    "weld variant should record that it is described relative to the direct-loop SDF"

  let build ← assertSome
    (findFourBarExampleAsset? "BUILD.bazel")
    "FourBar BUILD asset"
  LeanTest.assertTrue
    (build.localDependencies.contains "passive_simulation.cc")
    "BUILD.bazel should record the passive_simulation source dependency"
  LeanTest.assertTrue
    (build.localDependencies.contains "dev/four_bar_weld.sdf")
    "BUILD.bazel should record that the globbed models filegroup includes dev SDFs"

  let readme ← assertSome
    (findFourBarExampleAsset? "README.md")
    "FourBar README asset"
  LeanTest.assertTrue
    (readme.localDependencies.contains "images/FourBarLinkageGeometry.png")
    "README should record the geometry diagram dependency"

@[test]
def testFourBarModelAssetBoundaryRecordsSdfIdentity : IO Unit := do
  assertOk fourBarModelAssetBoundary.validate? "FourBar SDF asset boundary"
  LeanTest.assertEqual fourBarModelAssetBoundary.modelName "four_bar"
    "FourBar SDF model name should be recorded"
  LeanTest.assertEqual fourBarModelAssetBoundary.sdfPath
    "../drake/examples/multibody/four_bar/four_bar.sdf"
    "FourBar SDF source path should be recorded"
  LeanTest.assertEqual fourBarModelAssetBoundary.packageUri fourBarModelUri
    "FourBar package URI should match the passive executable"
  LeanTest.assertEqual fourBarModelAssetBoundary.linkNames #["A", "B", "C"]
    "FourBar SDF links should be explicit"
  LeanTest.assertEqual fourBarModelAssetBoundary.jointNames jointNames
    "FourBar SDF joints should be explicit"
  LeanTest.assertEqual fourBarModelAssetBoundary.jointTypes
    #["revolute", "revolute", "revolute"]
    "FourBar SDF joint types should be explicit"
  LeanTest.assertEqual fourBarModelAssetBoundary.jointAxes
    #[#[0.0, 1.0, 0.0], #[0.0, 1.0, 0.0], #[0.0, 1.0, 0.0]]
    "FourBar SDF joint axes should match the model y-axis revolutes"
  LeanTest.assertEqual fourBarModelAssetBoundary.actuatedJointNames #["joint_WA"]
    "Only joint_WA is actuated by Drake's parsed SDF effort defaults"
  LeanTest.assertEqual fourBarModelAssetBoundary.unactuatedJointNames
    #["joint_AB", "joint_WC"]
    "The SDF effort=0 joints should be recorded as unactuated"
  LeanTest.assertEqual fourBarModelAssetBoundary.bushingFrameNames bushingFrameNames
    "FourBar bushing frames should match Drake's cut-loop frames"
  LeanTest.assertEqual fourBarModelAssetBoundary.bushingFrameAttachedTo #["B", "C"]
    "Bushing frames should be attached to the two cut links"
  LeanTest.assertEqual fourBarModelAssetBoundary.bushingFramePoseInAttached
    #[
      #[4.0, 0.0, 0.0, -1.57079632679, 0.0, 0.0],
      #[4.0, 0.0, 0.0, -1.57079632679, 0.0, 0.0]
    ]
    "Bushing frame poses should match the SDF end-of-link roll offset"
  LeanTest.assertEqual fourBarModelAssetBoundary.linkBoxSize #[4.2, 0.1, 0.2]
    "FourBar link visual/inertial box dimensions should match the SDF"
  LeanTest.assertTrue (approx fourBarModelAssetBoundary.linkMass 20.0 1.0e-12)
    s!"FourBar link mass should be 20kg, got {fourBarModelAssetBoundary.linkMass}"
  LeanTest.assertTrue (approx fourBarModelAssetBoundary.linkIyy 29.46666666666666 1.0e-12)
    s!"FourBar link Iyy should match the SDF cuboid inertia, got {fourBarModelAssetBoundary.linkIyy}"

@[test]
def testMultibodyFourBarPassiveBenchmarkBoundaryIsRecorded : IO Unit := do
  let result ← assertOk buildMultibodyFourBar?
    "multibody four-bar passive benchmark"
  assertOk result.asset.validate? "FourBar SDF asset boundary"
  assertOk result.config.validate? "FourBar passive benchmark config"
  assertOk result.step.validate? "FourBar passive FullMultibodyPlantStep"
  assertOk result.trace.validate? "FourBar passive benchmark trace"
  LeanTest.assertEqual result.assetCatalog.size 8
    "FourBar passive benchmark result should carry the complete provider asset catalog"

  LeanTest.assertEqual result.step.model.modelUri fourBarModelUri
    "FourBar passive benchmark should parse Drake's package SDF URL"
  LeanTest.assertEqual result.step.model.numPositions 3
    "FourBar MultibodyPlant should have three positions"
  LeanTest.assertEqual result.step.model.numVelocities 3
    "FourBar MultibodyPlant should have three velocities"
  LeanTest.assertEqual result.step.model.numActuatedDofs 1
    "FourBar MultibodyPlant should expose the single WA actuator"
  LeanTest.assertEqual result.step.q0 #[initialQA, initialQB, initialQC]
    "FourBar passive benchmark should use Drake's loop-closing initial angles"
  LeanTest.assertEqual result.step.v0 #[params.initialVelocity, 0.0, 0.0]
    "FourBar passive benchmark should set qAdot from FLAGS_initial_velocity"
  LeanTest.assertEqual result.step.actuation #[params.appliedTorque]
    "FourBar passive benchmark should fix the applied torque input"
  LeanTest.assertTrue (approx result.config.targetRealtimeRate 1.0 1.0e-12)
    s!"FourBar target_realtime_rate should default to 1, got {result.config.targetRealtimeRate}"
  LeanTest.assertTrue (approx result.config.simulationTime 10.0 1.0e-12)
    s!"FourBar simulation_time should default to 10, got {result.config.simulationTime}"
  LeanTest.assertTrue (approx result.config.timeStep 0.0 1.0e-12)
    s!"FourBar time_step should default to continuous mode, got {result.config.timeStep}"
  LeanTest.assertTrue result.config.visualizationEnabled
    "FourBar passive benchmark should include Drake's default visualization boundary"
  LeanTest.assertTrue result.step.config.isContinuous
    "FourBar passive benchmark should use continuous MultibodyPlant when time_step is zero"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.label == "full-physics-step:multibody four-bar passive benchmark plant"))
    "Move list should expose the FourBar full-plant passive benchmark as a primitive physics solve"

  let x0 := defaultState params
  let (expectedDx, expectedBushing) ← assertOk (derivative? params x0)
    "four-bar expected derivative"
  let expectedGeneralizedForces :=
    FloatArray.add (appliedGeneralizedForces params) expectedBushing.generalizedForce
  let expectedBias := (gravityGeneralizedForces params x0).map (fun g => -g)
  let expectedRhs := FloatArray.sub expectedGeneralizedForces expectedBias
  let (primitive, _) ← assertOk (fullPhysicsPrimitives? params x0 "four-bar primitive force test")
    "four-bar primitive bundle"

  LeanTest.assertTrue (result.fullPhysics.support.policy == SupportPolicy.fullSupport)
    "FourBar has no point-contact candidates; the full-physics solve should use exact full support"
  LeanTest.assertEqual result.fullPhysics.support.totalCandidates 0
    "FourBar bushing full physics should not create contact candidates"
  LeanTest.assertEqual result.fullPhysics.contactForces.size 0
    "FourBar bushing full physics should not create contact force scalars"
  LeanTest.assertTrue (result.fullPhysics.supportMove.exactness == MoveExactness.exact)
    "Empty full-support selection should be exact"
  for i in [:3] do
    assertArrayNear result.fullPhysics.equation.massMatrix[i]!
      (massMatrix params x0)[i]! 1.0e-10
      s!"FourBar full physics mass matrix row {i} should come from the local link-Jacobian formula"
  assertArrayNear primitive.actuationForces (appliedGeneralizedForces params) 1.0e-12
    "FourBar primitive actuation should keep only the applied torque input"
  LeanTest.assertEqual primitive.generalizedForceContributions.size 1
    "FourBar bushing should be a named primitive force contribution"
  assertArrayNear primitive.generalizedForceContributions[0]!.force
    expectedBushing.generalizedForce 1.0e-8
    "FourBar primitive force contribution should carry the bushing force"
  assertArrayNear result.bushing.generalizedForce expectedBushing.generalizedForce 1.0e-8
    "FourBar full physics should reuse the local LinearBushingRollPitchYaw force"
  assertArrayNear result.fullPhysics.generalizedPrimitiveForce
    expectedBushing.generalizedForce 1.0e-8
    "FourBar full physics should expose primitive-generated generalized force separately"
  assertArrayNear result.fullPhysics.generalizedForces expectedGeneralizedForces 1.0e-8
    "FourBar generalized force should be applied torque plus bushing force"
  assertArrayNear result.fullPhysics.equation.biasForces expectedBias 1.0e-8
    "FourBar full physics should encode gravity as negative manipulator bias"
  assertArrayNear result.fullPhysics.derivative.rhs expectedRhs 1.0e-8
    "FourBar full physics RHS should equal gravity plus applied plus bushing forces"
  assertArrayNear result.fullPhysics.derivative.qdot x0.qdot 1.0e-12
    "FourBar qdot should match the benchmark initial generalized velocity"
  assertArrayNear result.fullPhysics.derivative.vdot
    #[expectedDx.qAdot, expectedDx.qBdot, expectedDx.qCdot] 1.0e-8
    "FourBar vdot should match the existing derivative solve"

@[test]
def testBushingFullPhysicsProviderRecomputesFromCurrentState : IO Unit := do
  let p := params
  let provider := bushingFullPhysicsPrimitiveProvider p
    "four-bar bushing provider test"
  let x0 := defaultState p
  let x1 : FourBarState := {
    x0 with
    qA := x0.qA + 0.02
    qBdot := 0.25
  }

  let primitive0 ← assertOk (provider.validPrimitivesAt? x0)
    "four-bar bushing provider at default state"
  let primitive1 ← assertOk (provider.validPrimitivesAt? x1)
    "four-bar bushing provider at perturbed state"
  let result0 ← assertOk (provider.solveAt? x0 5456)
    "four-bar bushing provider solve at default state"
  let result1 ← assertOk (provider.solveAt? x1 5457)
    "four-bar bushing provider solve at perturbed state"

  LeanTest.assertEqual primitive0.generalizedForceContributions.size 1
    "Bushing provider should expose the cut-loop bushing as a primitive generalized force"
  LeanTest.assertEqual primitive1.generalizedForceContributions.size 1
    "Bushing provider should recompute the bushing primitive at the perturbed state"
  assertArrayNear primitive0.qdot x0.qdot 1.0e-12
    "Bushing provider qdot should come from the default state"
  assertArrayNear primitive1.qdot x1.qdot 1.0e-12
    "Bushing provider qdot should come from the perturbed state"
  LeanTest.assertTrue
    (FloatArray.maxAbsDiff primitive0.generalizedForceContributions[0]!.force
      primitive1.generalizedForceContributions[0]!.force > 1.0)
    "Perturbing the state should change the recomputed bushing generalized force"
  LeanTest.assertTrue
    (FloatArray.maxAbsDiff result0.derivative.vdot result1.derivative.vdot > 1.0e-4)
    "Provider solves should differ after recomputing mass, bias, and bushing force"
  LeanTest.assertEqual result0.contactForces.size 0
    "FourBar bushing provider should not synthesize point-contact forces"
  LeanTest.assertEqual result1.contactForces.size 0
    "FourBar bushing provider should stay contact-free after perturbation"

@[test]
def testIdealLoopClosureUsesBilateralConstraintPrimitive : IO Unit := do
  let p := params
  let x := defaultState p
  let primitive ← assertOk (idealLoopFullPhysicsPrimitives? p x
    "four-bar ideal loop test")
    "four-bar ideal loop primitive bundle"
  let provider := idealLoopFullPhysicsPrimitiveProvider p
    "four-bar ideal loop provider test"
  let result ← assertOk (provider.solveAt? x 5458)
    "four-bar ideal loop provider solve"
  let directResult ← assertOk (solveIdealLoopFullPhysics? p x 5459
    "four-bar ideal loop direct solve")
    "four-bar ideal loop direct solve"
  let solve ← assertSome result.constraintSolve?
    "four-bar ideal loop bilateral solve"

  LeanTest.assertEqual primitive.generalizedForceContributions.size 0
    "Ideal loop closure should not use the bushing force primitive"
  LeanTest.assertEqual primitive.bilateralConstraints.size 1
    "Ideal loop closure should lower into one bilateral constraint primitive"
  LeanTest.assertEqual primitive.bilateralConstraints[0]!.rowCount 2
    "Planar four-bar loop should constrain x/z closure rows and omit the identically-zero y row"
  for row in primitive.bilateralConstraints[0]!.jacobian do
    LeanTest.assertEqual row.size 3
      "Ideal loop closure rows should have one entry per generalized velocity"
  LeanTest.assertEqual primitive.bilateralConstraints[0]!.targetAcceleration #[]
    "Empty targetAcceleration should request zero acceleration through the primitive default"
  assertArrayNear solve.targetAcceleration #[0.0, 0.0] 1.0e-12
    "Bilateral solve should expand the empty target to zero x/z acceleration"
  assertArrayNear solve.constraintAccelerationAfter #[0.0, 0.0] 1.0e-8
    "Bilateral solve should enforce the loop acceleration target"
  LeanTest.assertTrue
    (FloatArray.maxAbsDiff solve.constraintAccelerationBefore
      solve.constraintAccelerationAfter > 1.0e-6)
    "The constraint should actively correct the unconstrained acceleration"
  LeanTest.assertEqual solve.multipliers.size 2
    "Two loop rows should produce two bilateral multipliers"
  LeanTest.assertEqual result.generalizedConstraintForce.size 3
    "Loop constraint force should be generalized back into the three coordinates"
  LeanTest.assertTrue (result.generalizedConstraintForce.all Float.isFinite)
    s!"Loop constraint generalized force should be finite, got {result.generalizedConstraintForce}"
  LeanTest.assertEqual result.generalizedPrimitiveForce #[0.0, 0.0, 0.0]
    "Ideal loop closure should leave primitive generalized forces empty"
  LeanTest.assertEqual result.contactForces.size 0
    "Ideal loop closure should not synthesize point-contact forces"
  assertArrayNear directResult.derivative.vdot result.derivative.vdot 1.0e-12
    "Direct ideal-loop solve should use the same provider-backed primitive equations"

@[test]
def testInitialGeometryClosesLoopAtDrakeReadmeAngles : IO Unit := do
  let p := params
  let x := defaultState p
  let sqrt15 := Float.sqrt 15.0
  LeanTest.assertTrue (approx initialQA 1.318116071652818 1.0e-15)
    s!"qA should be atan2(sqrt(15), 1), got {initialQA}"
  LeanTest.assertTrue (approx initialQB (piTest - initialQA) 1.0e-15)
    s!"qB should be pi - qA, got {initialQB}"
  LeanTest.assertTrue (approx initialQC initialQB 1.0e-15)
    s!"qC should match qB, got {initialQC}"

  let bc := endpointBc p x
  let cb := endpointCb p x
  LeanTest.assertTrue (approx bc.x (-3.0) 1.0e-9)
    s!"Bc endpoint x should match README geometry, got {bc.x}"
  LeanTest.assertTrue (approx bc.z sqrt15 1.0e-9)
    s!"Bc endpoint z should match README geometry, got {bc.z}"
  LeanTest.assertTrue (approx cb.x (-3.0) 1.0e-9)
    s!"Cb endpoint x should match README geometry, got {cb.x}"
  LeanTest.assertTrue (approx cb.z sqrt15 1.0e-9)
    s!"Cb endpoint z should match README geometry, got {cb.z}"
  LeanTest.assertTrue (loopClosureErrorNorm p x < 1.0e-9)
    s!"Initial four-bar loop should be closed, got error {reprStr (loopClosureError p x)}"

@[test]
def testBushingParametersMatchDrakePlanarRevoluteClosure : IO Unit := do
  let b := params.bushingParams
  LeanTest.assertEqual b.forceStiffness #[30000.0, 30000.0, 30000.0]
    "Drake uses the same translational stiffness in x/y/z"
  LeanTest.assertEqual b.forceDamping #[1500.0, 1500.0, 1500.0]
    "Drake uses the same translational damping in x/y/z"
  LeanTest.assertEqual b.torqueStiffness #[30000.0, 30000.0, 0.0]
    "The cut joint closes roll/pitch but leaves yaw free for planar revolute motion"
  LeanTest.assertEqual b.torqueDamping #[1500.0, 1500.0, 0.0]
    "The cut joint damps roll/pitch but leaves yaw free for planar revolute motion"

@[test]
def testBushingVelocityPenaltyUsesLoopClosureJacobian : IO Unit := do
  let p := params
  let x := defaultState p
  let state := bushingState p x
  let result ← assertOk (bushingResult? p x) "four-bar bushing"
  let sqrt15 := Float.sqrt 15.0
  let expectedVX := -3.0 * sqrt15
  let expectedVZ := -9.0

  LeanTest.assertTrue (approx (state.translationVelocityError.getD 0 99.0) expectedVX 1.0e-9)
    s!"Loop x velocity should come from J(q) qdot, got {state.translationVelocityError}"
  LeanTest.assertTrue (approx (state.translationVelocityError.getD 1 99.0) 0.0 1.0e-12)
    s!"The planar example should have zero y loop velocity, got {state.translationVelocityError}"
  LeanTest.assertTrue (approx (state.translationVelocityError.getD 2 99.0) expectedVZ 1.0e-9)
    s!"Loop z velocity should come from J(q) qdot, got {state.translationVelocityError}"

  LeanTest.assertTrue (approx (result.force.getD 0 0.0) (-(p.forceDamping * expectedVX)) 1.0e-4)
    s!"Bushing x force should be damping against loop velocity, got {result.force}"
  LeanTest.assertTrue (approx (result.force.getD 1 99.0) 0.0 1.0e-12)
    s!"Bushing y force should remain zero for the planar loop, got {result.force}"
  LeanTest.assertTrue (approx (result.force.getD 2 0.0) (-(p.forceDamping * expectedVZ)) 1.0e-4)
    s!"Bushing z force should be damping against loop velocity, got {result.force}"
  LeanTest.assertTrue (approx (result.torque.getD 2 99.0) 0.0 1.0e-12)
    s!"Yaw torque should be free even when yaw rate is nonzero, got {result.torque}"

@[test]
def testMassMatrixAndDerivativeAreFinite : IO Unit := do
  let p := params
  let x := defaultState p
  let m := massMatrix p x
  LeanTest.assertEqual m.size 3
    "The four-bar port has three generalized coordinates"
  for row in m do
    LeanTest.assertEqual row.size 3
      "Mass matrix rows should match the generalized-coordinate dimension"
  LeanTest.assertTrue (approx ((m[0]!).getD 1 0.0) ((m[1]!).getD 0 0.0) 1.0e-9)
    s!"Mass matrix should be symmetric, got {m}"
  LeanTest.assertTrue (approx ((m[0]!).getD 2 0.0) ((m[2]!).getD 0 0.0) 1.0e-9)
    s!"Mass matrix should be symmetric, got {m}"
  LeanTest.assertTrue (approx ((m[1]!).getD 2 0.0) ((m[2]!).getD 1 0.0) 1.0e-9)
    s!"Mass matrix should be symmetric, got {m}"
  LeanTest.assertTrue ((m[0]!).getD 0 0.0 > 0.0 && (m[1]!).getD 1 0.0 > 0.0 && (m[2]!).getD 2 0.0 > 0.0)
    s!"Mass matrix diagonal should be positive, got {m}"

  let (dx, bushing) ← assertOk (derivative? p x) "four-bar derivative"
  LeanTest.assertTrue (dx.isFinite)
    s!"Derivative should be finite, got {reprStr dx}"
  LeanTest.assertTrue (approx dx.qA p.initialVelocity 1.0e-12)
    s!"First derivative of qA should be qAdot, got {dx.qA}"
  LeanTest.assertTrue (approx dx.qB 0.0 1.0e-12)
    s!"First derivative of qB should be qBdot, got {dx.qB}"
  LeanTest.assertTrue (approx dx.qC 0.0 1.0e-12)
    s!"First derivative of qC should be qCdot, got {dx.qC}"
  LeanTest.assertTrue (bushing.dissipationPower > 0.0)
    "The initial WA angular velocity should produce bushing damping power"

@[test]
def testEndToEndTraceAndShortRolloutExecute : IO Unit := do
  let result ← getResult
  let _ ← assertOk (result.trace.validate?) "four-bar trace validation"
  let _ ← assertOk (result.fullPlant.trace.validate?) "four-bar full-plant trace validation"
  LeanTest.assertEqual result.assetCatalog.size 8
    "End-to-end FourBar result should carry the complete provider asset catalog"
  LeanTest.assertEqual result.fullPlant.assetCatalog.size 8
    "Nested full-plant result should carry the same provider asset catalog"
  LeanTest.assertEqual result.trace.moves.size 2
    "One continuous interval should expand to interval-adjoint and checkpoint-boundary moves"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.label == "full-physics-step:multibody four-bar passive benchmark plant"))
    "End-to-end result should include the full-plant FourBar boundary"
  LeanTest.assertTrue (result.rolloutState.isFinite)
    s!"Short rollout state should be finite, got {reprStr result.rolloutState}"
  LeanTest.assertTrue (result.oneStepState.qA > result.initialState.qA)
    s!"Positive initial qAdot should advance qA, got qA0={result.initialState.qA}, qA1={result.oneStepState.qA}"
  LeanTest.assertTrue (Float.isFinite result.initialEnergy && Float.isFinite result.oneStepEnergy)
    s!"Energy diagnostics should be finite, got {result.initialEnergy}, {result.oneStepEnergy}"
  LeanTest.assertTrue (Float.isFinite result.loopError.x && Float.isFinite result.loopError.z)
    s!"Loop diagnostics should be finite, got {reprStr result.loopError}"
  LeanTest.assertTrue (Float.isFinite result.loopVelocity.x && Float.isFinite result.loopVelocity.z)
    s!"Loop velocity diagnostics should be finite, got {reprStr result.loopVelocity}"

end Tests.EventSkeletonFourBarExample
