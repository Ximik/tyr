import LeanTest
import Tyr.EventSkeleton.Examples.SimpleGripper

namespace Tests.EventSkeletonSimpleGripperExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.SimpleGripper

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (result : Except String α) (label : String) :
    IO α := do
  match result with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: {msg}"

private def assertArrayNear
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  LeanTest.assertEqual actual.size expected.size
    s!"{label}: size mismatch, got {actual.size}, expected {expected.size}"
  for i in [:actual.size] do
    LeanTest.assertTrue (approx actual[i]! expected[i]! tol)
      s!"{label}[{i}]: got {actual[i]!}, expected {expected[i]!}"

private def sumNormalForce (forces : Array ContactForceScalars) : Float :=
  forces.foldl (fun acc force => acc + force.normalForce) 0.0

private def maxNormalForce (forces : Array ContactForceScalars) : Float :=
  forces.foldl
    (fun acc force => if force.normalForce > acc then force.normalForce else acc)
    0.0

@[test]
def testDrakeReferencesPlantMetadataAndCouplerAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/simple_gripper/simple_gripper.cc"))
    "The example should reference Drake's simple_gripper.cc driver"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/simple_gripper/simple_gripper.sdf"))
    "The example should reference the gripper SDF"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/simple_gripper/simple_mug.sdf"))
    "The example should reference the mug SDF"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/simple_gripper/BUILD.bazel"))
    "The example should reference Drake's Bazel executable/data declaration"

  let plant := plantSummary params
  LeanTest.assertEqual plant.modelNames #["simple_gripper", "simple_mug"]
  LeanTest.assertEqual plant.actuatorNames #["translate_joint", "left_slider"]
  LeanTest.assertEqual plant.jointRecords.size 4
  LeanTest.assertTrue ((translateAxis plant) == ({ x := 0.0, y := 0.0, z := 1.0 } : Vec3))
    "The Drake SDF translate joint should move along world/model z"
  LeanTest.assertTrue ((gravityVectorForTranslateAxis (translateAxis plant)) == ({} : Vec3))
    "Drake disables gravity for vertical forced gripper motion"

  assertOk plant.coupler.validate? "coupler validation"
  LeanTest.assertTrue (approx plant.coupler.gearRatio (-1.0) 1.0e-12)
    s!"Expected default coupler gear ratio -1, got {plant.coupler.gearRatio}"
  LeanTest.assertTrue (approx (gripActuationForce params) 20.0 1.0e-12)
    s!"Virtual-work grip actuation should be 2G for rho=-1, got {gripActuationForce params}"

  let x0 := initialState params
  LeanTest.assertTrue
    (approx (plant.coupler.positionResidual x0.leftQ x0.rightQ) 0.0 1.0e-12)
    "Initial finger positions should satisfy left = rho * right"
  LeanTest.assertTrue
    (approx x0.translateV (-(params.amplitude * driveOmega params)) 1.0e-12)
    s!"Initial translate velocity should match Drake sine setup, got {x0.translateV}"

@[test]
def testModelAssetBoundaryRecordsSdfAndGeneratedGeometryInputs : IO Unit := do
  assertOk simpleGripperModelAssetBoundary.validate? "simple gripper SDF asset boundary"
  LeanTest.assertEqual simpleGripperModelAssetBoundary.gripperPackageUri simpleGripperModelUri
    "Gripper package URI should match Drake's parser URL"
  LeanTest.assertEqual simpleGripperModelAssetBoundary.mugPackageUri simpleMugModelUri
    "Mug package URI should match Drake's parser URL"
  LeanTest.assertEqual simpleGripperModelAssetBoundary.linkNames
    #["y_translate_link", "body", "left_finger", "right_finger", "simple_mug"]
    "SDF link names should include the gripper links and free mug body"
  LeanTest.assertEqual simpleGripperModelAssetBoundary.jointNames
    #["weld_base", "translate_joint", "left_slider", "right_slider"]
    "SDF joint names should preserve Drake's gripper joint layout"
  LeanTest.assertEqual simpleGripperModelAssetBoundary.actuatedJointNames
    #["translate_joint", "left_slider"]
    "The translate and left finger joints should be actuated"
  LeanTest.assertEqual simpleGripperModelAssetBoundary.unactuatedJointNames
    #["right_slider"]
    "The right finger slider should remain unactuated and coupled"
  LeanTest.assertTrue
    (simpleGripperModelAssetBoundary.translateJointAxis ==
      ({ x := 0.0, y := 0.0, z := 1.0 } : Vec3))
    "The SDF translate joint should be vertical"
  LeanTest.assertEqual simpleGripperModelAssetBoundary.bodyBoxSize
    #[0.146, 0.0725, 0.049521]
    "Body visual box should match the gripper SDF"
  LeanTest.assertEqual simpleGripperModelAssetBoundary.fingerBoxSize
    #[0.007, 0.081, 0.028]
    "Finger visual boxes should match the gripper SDF"
  LeanTest.assertTrue (approx simpleGripperModelAssetBoundary.mugCylinderRadius 0.04 1.0e-12)
    s!"Mug cylinder radius should match the SDF, got {simpleGripperModelAssetBoundary.mugCylinderRadius}"
  LeanTest.assertTrue (approx simpleGripperModelAssetBoundary.padMinorRadius 0.006 1.0e-12)
    s!"Generated pad sphere radius should be 6mm, got {simpleGripperModelAssetBoundary.padMinorRadius}"
  LeanTest.assertTrue (approx simpleGripperModelAssetBoundary.padMajorRadius 0.014 1.0e-12)
    s!"Generated pad ring major radius should be 14mm, got {simpleGripperModelAssetBoundary.padMajorRadius}"

@[test]
def testMultibodySimpleGripperFullPlantBoundaryIsRecorded : IO Unit := do
  let result ← assertOk buildMultibodySimpleGripper?
    "multibody simple gripper full plant"
  assertOk result.asset.validate? "simple gripper SDF asset boundary"
  assertOk result.parsedPlant.validate? "simple gripper parsed plant quantities"
  assertOk result.config.validate? "simple gripper full plant config"
  assertOk result.step.validate? "simple gripper FullMultibodyPlantStep"
  assertOk result.fullPhysics.equation.validate? "simple gripper primitive full physics equation"
  assertOk result.trace.validate? "simple gripper full plant trace"

  LeanTest.assertEqual result.parsedPlant.modelUris
    #[simpleGripperModelUri, simpleMugModelUri]
    "Both SDF parser URLs should be explicit"
  LeanTest.assertEqual result.parsedPlant.numModelInstances 4
    "Parsed plant should include world/default plus two parsed model instances"
  LeanTest.assertEqual result.parsedPlant.numActuators 2
    "Drake demands two actuators"
  LeanTest.assertEqual result.parsedPlant.numBodies 6
    "Plant should contain world, four gripper bodies, and the mug body"
  LeanTest.assertEqual result.step.model.modelUri combinedModelUri
    "Full plant model URI should record the two parser inputs"
  LeanTest.assertEqual result.step.model.numPositions 10
    "Three gripper joints plus quaternion floating mug should give ten positions"
  LeanTest.assertEqual result.step.model.numVelocities 9
    "Three gripper joints plus floating mug velocity should give nine velocities"
  LeanTest.assertEqual result.step.model.numActuatedDofs 2
    "Full plant should expose translate and left-slider actuators"
  LeanTest.assertEqual result.step.model.floatingBases.size 1
    "The simple mug should be the single floating body"
  LeanTest.assertEqual result.step.model.floatingBases[0]!.bodyName "simple_mug"
    "The floating base should be attached to the mug body"
  LeanTest.assertEqual result.step.model.floatingBases[0]!.floatingPositionsStart 3
    "Mug quaternion positions should start after the three gripper joint positions"
  LeanTest.assertEqual result.step.model.floatingBases[0]!.floatingVelocitiesStartInV 3
    "Mug spatial velocities should start after the three gripper joint velocities"
  LeanTest.assertTrue result.step.config.isDiscrete
    "Default simple_gripper Drake plant should be discrete"
  LeanTest.assertTrue (approx result.step.config.timeStep 1.0e-3 1.0e-15)
    s!"Discrete update period should match FLAGS_mbp_discrete_update_period, got {result.step.config.timeStep}"
  LeanTest.assertTrue (result.step.config.contactApproximation == .sap)
    "Default contact approximation should be SAP"
  LeanTest.assertEqual result.step.q0.size 10
    "Full plant q0 should include three joints and one quaternion floating body"
  LeanTest.assertTrue (approx (result.step.q0.getD 0 99.0) 0.0 1.0e-12)
    s!"Translate joint should start at zero, got {result.step.q0}"
  LeanTest.assertTrue (approx (result.step.q0.getD 1 99.0) (-(params.gripWidth / 2.0)) 1.0e-12)
    s!"Left slider should start at -grip_width/2, got {result.step.q0}"
  LeanTest.assertTrue (approx (result.step.q0.getD 2 99.0) (params.gripWidth / 2.0) 1.0e-12)
    s!"Right slider should start at grip_width/2, got {result.step.q0}"
  LeanTest.assertTrue (approx (result.step.q0.getD 3 99.0) 0.0 1.0e-12)
    s!"Default mug yaw pi should have near-zero quaternion w, got {result.step.q0}"
  LeanTest.assertTrue (approx (result.step.q0.getD 6 99.0) 1.0 1.0e-12)
    s!"Default mug yaw pi should have unit z quaternion component, got {result.step.q0}"
  LeanTest.assertEqual result.step.v0.size 9
    "Full plant v0 should include three joint rates and one floating body velocity"
  LeanTest.assertTrue (approx (result.step.v0.getD 0 99.0) (initialTranslateVelocity params) 1.0e-12)
    s!"Translate joint rate should match Drake's harmonic initial velocity, got {result.step.v0}"
  LeanTest.assertEqual result.step.actuation #[0.0, gripActuationForce params]
    "At t=0 the harmonic translate force is zero and grip force is constant"
  LeanTest.assertEqual result.pads.size 16
    "Full plant boundary should record the 16 generated ring-pad spheres"
  LeanTest.assertEqual result.fullPhysics.support.candidates.size 16
    "Primitive full physics should dynamically expose the generated pad candidates"
  LeanTest.assertEqual result.fullPhysics.support.selectedLocalIndices.size 16
    "Primitive full physics should select the same near-contact support at the initial state"
  LeanTest.assertEqual result.fullPhysics.contactForces.size 16
    "Primitive full physics should produce one scalar force bundle per selected pad"
  LeanTest.assertTrue (result.fullPhysics.supportMove.exactness == .controlledApproximation)
    "Threshold-selected contacts should be marked as a fixed-trace approximation"
  LeanTest.assertTrue (result.fullPhysics.move.exactness == .exact)
    "The mass-matrix primitive solve itself should be exact for the selected support"
  LeanTest.assertEqual result.fullPhysics.equation.massMatrix
    (coupledReducedMassMatrix params)
    "Primitive full physics should use the coupler-reduced mass matrix"
  assertArrayNear result.fullPhysics.equation.qdot
    (coupledReducedVelocity params (initialState params)) 1.0e-12
    "Primitive full physics qdot"
  assertArrayNear result.fullPhysics.equation.generalizedForces
    (coupledReducedActuationForces params (commandAt params 0.0)) 1.0e-12
    "Initial primitive full physics generalized force should be pure actuation"
  let initialStep ← assertOk (physicsStep? params (initialState params) 0.0)
    "initial coupled primitive step"
  assertArrayNear result.fullPhysics.derivative.vdot
    initialStep.reducedSolve.derivative.vdot 1.0e-12
    "Initial primitive full physics acceleration should match the existing coupled solve"
  assertArrayNear result.fullPhysicsDerivative.asArray
    initialStep.derivative.asArray 1.0e-12
    "Lifted primitive derivative should match the coupled three-coordinate derivative"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.label == "full-physics-step:simple_gripper contact benchmark plant"))
    "Move list should expose the SimpleGripper primitive full-physics benchmark solve"

@[test]
def testGeneratedRingPadsAndDynamicContactCandidates : IO Unit := do
  let pads := generatedPadSpheres params
  LeanTest.assertEqual pads.size 16
    "Default Drake ring_samples=8 should generate 8 pads per finger"
  LeanTest.assertEqual pads[0]!.id 1000
  LeanTest.assertEqual pads[0]!.bodyName "left_finger"
  LeanTest.assertEqual pads[8]!.id 2000
  LeanTest.assertEqual pads[8]!.bodyName "right_finger"
  LeanTest.assertTrue (approx pads[0]!.centerInFinger.x params.padOffset 1.0e-12)
    s!"Left pad x offset should match Drake pad_offset, got {pads[0]!.centerInFinger.x}"
  LeanTest.assertTrue (approx pads[8]!.centerInFinger.x (-params.padOffset) 1.0e-12)
    s!"Right pad x offset should be negative pad_offset, got {pads[8]!.centerInFinger.x}"
  LeanTest.assertTrue
    (approx pads[0]!.centerInFinger.y
      (params.padMajorRadius + params.padTorusCenterY) 1.0e-12)
    s!"First sample should lie at torus center y plus major radius, got {pads[0]!.centerInFinger.y}"

  let support := activeSupport params (initialState params)
  assertOk (support.validateJacobianWidth? 3) "initial support jacobian width"
  LeanTest.assertEqual support.candidates.size 16
  LeanTest.assertEqual support.selectedLocalIndices.size 16
    "The penetration allowance should retain all near ring-pad candidates"
  let runtime ← assertOk support.toRuntimeSupport? "initial runtime support"
  LeanTest.assertEqual runtime.selectedIds.size 16
  LeanTest.assertTrue (runtime.policy == SupportPolicy.threshold params.penetrationAllowance)
    "Support should record that runtime contacts were selected by threshold"

  let leftNearest := (contactCandidates params (closingState params))[2]!
  let rightNearest := (contactCandidates params (closingState params))[10]!
  LeanTest.assertTrue (leftNearest.mode == ContactMode.sliding)
    s!"Compressed left pad should slide under vertical sine velocity, got {reprStr leftNearest.mode}"
  LeanTest.assertTrue (rightNearest.mode == ContactMode.sliding)
    s!"Compressed right pad should slide under vertical sine velocity, got {reprStr rightNearest.mode}"
  LeanTest.assertTrue (leftNearest.signedDistance < 0.0)
    s!"Compressed left pad should penetrate the mug, got {leftNearest.signedDistance}"
  LeanTest.assertTrue (rightNearest.signedDistance < 0.0)
    s!"Compressed right pad should penetrate the mug, got {rightNearest.signedDistance}"

@[test]
def testSimpleGripperProvidersRecomputeContactsAndTimedActuation : IO Unit := do
  let p := params
  let x0 := initialState p
  let closing := closingState p
  let rawProvider := contactCandidateProvider p
    "simple gripper raw provider test"
  let reducedProvider := coupledReducedContactCandidateProvider p
    "simple gripper reduced provider test"
  let fullProvider := fullPhysicsPrimitiveProvider p
    "simple gripper timed full physics provider test"

  let rawInitial ← assertOk
    (rawProvider.validCandidatesAt? x0 (some 3))
    "raw simple gripper candidates at initial state"
  let rawClosing ← assertOk
    (rawProvider.validCandidatesAt? closing (some 3))
    "raw simple gripper candidates at closing state"
  LeanTest.assertEqual rawInitial.candidates.size 16
    "Raw provider should expose one candidate per generated ring pad"
  LeanTest.assertEqual rawClosing.candidates.size 16
    "Raw provider should keep the generated pad count after state changes"
  let initialMin ←
    match rawInitial.minimumSignedDistance? with
    | some d => pure d
    | none => LeanTest.fail "initial raw provider should expose a minimum signed distance"
  let closingMin ←
    match rawClosing.minimumSignedDistance? with
    | some d => pure d
    | none => LeanTest.fail "closing raw provider should expose a minimum signed distance"
  LeanTest.assertTrue (closingMin < initialMin)
    s!"Closing the gripper should reduce minimum signed distance, initial={initialMin}, closing={closingMin}"
  LeanTest.assertTrue (closingMin < 0.0)
    s!"Closing provider state should produce penetrating candidates, got {closingMin}"

  let reducedInitial ← assertOk
    (reducedProvider.validCandidatesAt? x0 (some 2))
    "coupler-reduced candidates at initial state"
  let reducedSupport ← assertOk
    (reducedProvider.supportAt? closing (.threshold p.penetrationAllowance)
      p.penetrationAllowance p.stictionTolerance (some 2)
      "coupler-reduced closing support")
    "coupler-reduced support at closing state"
  LeanTest.assertEqual reducedInitial.candidates[0]!.normalJacobian.size 2
    "Coupler-reduced provider should expose two-column contact Jacobians"
  LeanTest.assertEqual reducedSupport.selectedLocalIndices.size 16
    "Closing reduced support should be recomputed from the current state"

  let quarterPeriod := 1.0 / (4.0 * p.frequency)
  let primitive0 ← assertOk
    (fullProvider.validPrimitivesAt? (timedState p x0 0.0))
    "timed full-physics provider at t=0"
  let primitiveQuarter ← assertOk
    (fullProvider.validPrimitivesAt? (timedState p x0 quarterPeriod))
    "timed full-physics provider at quarter period"
  assertArrayNear primitive0.actuationForces
    #[0.0, (gripperCoupler p).gearRatio * gripActuationForce p] 1.0e-12
    "Initial timed provider actuation"
  LeanTest.assertTrue
    (approx (primitiveQuarter.actuationForces.getD 0 0.0)
      (harmonicForceAmplitude p) 1.0e-9)
    s!"Quarter-period translate actuation should be harmonic force amplitude, got {primitiveQuarter.actuationForces}"
  LeanTest.assertTrue
    (FloatArray.maxAbsDiff primitive0.actuationForces primitiveQuarter.actuationForces > 1.0)
    "Timed provider should recompute sine actuation from t"

  let closingResult ← assertOk
    (fullProvider.solveAt? (timedState p closing 0.0) 5115)
    "timed provider full-physics solve at closing state"
  LeanTest.assertEqual closingResult.support.selectedLocalIndices.size 16
    "Timed provider solve should recompute closing support"
  LeanTest.assertTrue
    (maxNormalForce closingResult.contactForces > 100.0)
    s!"Timed provider closing solve should produce contact force, got {maxNormalForce closingResult.contactForces}"
  let closingReference ← assertOk (physicsStep? p closing 0.0)
    "closing reference physics step"
  assertArrayNear closingResult.derivative.vdot
    closingReference.reducedSolve.derivative.vdot
    1.0e-9
    "Timed full-physics provider should match the existing reduced solve"

@[test]
def testEndToEndTraceAndCoupledPhysicsStepExecute : IO Unit := do
  let result ← assertOk (buildEndToEnd? params) "simple gripper end-to-end"

  LeanTest.assertEqual result.trace.moves.size 3
    "Interval plus runtime contact branch should project to three moves"
  LeanTest.assertTrue (result.trace.moves[0]!.kind == SkeletonMoveKind.intervalAdjoint)
    "First move should eliminate the integration interval"
  LeanTest.assertTrue (result.trace.moves[1]!.kind == SkeletonMoveKind.checkpointBoundary)
    "Second move should record the interval boundary"
  LeanTest.assertTrue (result.trace.moves[2]!.kind == SkeletonMoveKind.branchAggregate)
    "Third move should aggregate the dynamically selected contact support"
  LeanTest.assertTrue (result.trace.moves[2]!.exactness == MoveExactness.controlledApproximation)
    "Threshold-selected contacts should be marked as a fixed-trace approximation"
  assertOk result.fullPlant.step.validate? "simple gripper full plant step"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.label == "full-physics-step:simple_gripper contact benchmark plant"))
    "End-to-end result should include the SimpleGripper primitive full-physics solve"

  let supports := result.trace.supports
  LeanTest.assertEqual supports.size 1
  LeanTest.assertEqual supports[0]!.selectedIds.size 16
  LeanTest.assertEqual supports[0]!.totalCandidates? (some 16)

  LeanTest.assertTrue
    (sumNormalForce result.initialStep.contactForces == 0.0)
    "Initial separated-but-near pads should not produce penalty normal force"
  LeanTest.assertTrue
    (maxNormalForce result.closingStep.contactForces > 100.0)
    s!"Compressed grasp should produce positive penalty normal force, got {maxNormalForce result.closingStep.contactForces}"
  LeanTest.assertTrue
    (result.closingStep.generalizedContactForce.getD 0 0.0 > 0.0)
    s!"Vertical sliding friction should push on the translate coordinate, got {result.closingStep.generalizedContactForce}"

  let closingFull ← assertOk
    (solveFullPhysics? params (closingState params) 0.0 5114
      "simple_gripper closing contact primitive")
    "closing primitive full physics solve"
  assertOk closingFull.equation.validate? "closing primitive full physics equation"
  LeanTest.assertEqual closingFull.support.selectedLocalIndices.size 16
    "Closing primitive solve should recompute the runtime support from the compressed state"
  LeanTest.assertTrue
    (maxNormalForce closingFull.contactForces > 100.0)
    s!"Closing primitive solve should recompute positive contact forces, got {maxNormalForce closingFull.contactForces}"
  assertArrayNear closingFull.generalizedContactForce
    #[result.closingStep.generalizedContactForce.getD 0 0.0,
      (gripperCoupler params).gearRatio * result.closingStep.generalizedContactForce.getD 1 0.0 +
        result.closingStep.generalizedContactForce.getD 2 0.0]
    1.0e-9
    "Closing primitive generalized contact force should be the coupler-reduced J^T f"
  assertArrayNear closingFull.derivative.vdot
    result.closingStep.reducedSolve.derivative.vdot 1.0e-9
    "Closing primitive full physics acceleration should match the dynamic coupled solve"

  let eq := result.closingStep.reducedSolve.equation
  LeanTest.assertEqual eq.massMatrix.size 2
  LeanTest.assertTrue
    (approx ((eq.massMatrix[0]!).getD 0 0.0) params.gripperMovingMass 1.0e-12)
    "Reduced translate mass should preserve Drake's gripper moving mass"
  LeanTest.assertTrue
    (approx ((eq.massMatrix[1]!).getD 1 0.0)
      (params.fingerMass + params.fingerMass) 1.0e-12)
    "Coupled finger coordinate should contain both finger masses for rho=-1"
  LeanTest.assertTrue
    (Float.isFinite (result.closingStep.derivative.translateV))
    s!"Translate acceleration should be finite, got {result.closingStep.derivative.translateV}"

end Tests.EventSkeletonSimpleGripperExample
