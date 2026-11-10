import LeanTest
import Tyr.EventSkeleton.Examples.Rod2D

namespace Tests.EventSkeletonRod2DExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.Rod2D

private def pi : Float := 3.14159265358979323846
private def sqrt2Over2 : Float := 0.70710678118654752440

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertError {α : Type} (res : Except String α) (label : String) : IO String := do
  match res with
  | .ok _ => LeanTest.fail s!"{label}: expected error"
  | .error msg => pure msg

private def assertSome {α : Type} (x : Option α) (label : String) : IO α := do
  match x with
  | some value => pure value
  | none => LeanTest.fail s!"{label}: expected some"

private def assertArrayApprox
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

private def assertStateApprox
    (actual expected : RodState)
    (tol : Float)
    (label : String) : IO Unit := do
  assertArrayApprox (stateAsArray actual) (stateAsArray expected) tol label

private def hasDrakeReference (path : String) : Bool :=
  drakeReferences.any (fun ref => ref.path == path)

private def assertAllDrakeReferences (paths : Array String) : IO Unit := do
  for path in paths do
    LeanTest.assertTrue (hasDrakeReference path)
      s!"Rod2D example should reference {path}"

private def hasPackageMove
    (moves : Array SkeletonMove)
    (label : String)
    (kind : SkeletonMoveKind)
    (exactness : MoveExactness) : Bool :=
  moves.any (fun move =>
    move.label == label && move.kind == kind && move.exactness == exactness)

@[test]
def testDrakeReferencesAreRecorded : IO Unit := do
  assertAllDrakeReferences #[
    "../drake/examples/rod2d/rod2d.cc",
    "../drake/examples/rod2d/rod2d.h",
    "../drake/examples/rod2d/test/rod2d_test.cc",
    "../drake/examples/rod2d/rod2d_geometry.h",
    "../drake/examples/rod2d/rod2d_geometry.cc",
    "../drake/examples/rod2d/rod2d_sim.cc",
    "../drake/examples/rod2d/constraint_problem_data.h",
    "../drake/examples/rod2d/constraint_solver.cc",
    "../drake/examples/rod2d/constraint_solver.h",
    "../drake/examples/rod2d/test/constraint_solver_test.cc",
    "../drake/examples/rod2d/rod2d_state_vector.cc"
  ]
  assertOk validateRod2dExampleAssets? "Rod2D package asset catalog"
  LeanTest.assertEqual
    (rod2dDocumentationAssets.map (fun asset => asset.relativePath))
    #["README.md", "images/colliding-boxes.png"]
    "Rod2D documentation boundary should include README and image sidecar"

@[test]
def testRod2dStateVectorBoundaryMatchesDrakeCoordinateLayout : IO Unit := do
  assertOk rod2dStateVectorBoundary.validate? "Rod2dStateVector boundary"
  LeanTest.assertEqual rod2dStateVectorBoundary.dimension 6
    "Rod2dStateVector should expose Drake's six coordinates"
  LeanTest.assertEqual (rod2dStateVectorBoundary.indexOf? "x") (some 0)
    "Rod2dStateVector x index should match Drake"
  LeanTest.assertEqual (rod2dStateVectorBoundary.indexOf? "y") (some 1)
    "Rod2dStateVector y index should match Drake"
  LeanTest.assertEqual (rod2dStateVectorBoundary.indexOf? "theta") (some 2)
    "Rod2dStateVector theta index should match Drake"
  LeanTest.assertEqual (rod2dStateVectorBoundary.indexOf? "xdot") (some 3)
    "Rod2dStateVector xdot index should match Drake"
  LeanTest.assertEqual (rod2dStateVectorBoundary.indexOf? "ydot") (some 4)
    "Rod2dStateVector ydot index should match Drake"
  LeanTest.assertEqual (rod2dStateVectorBoundary.indexOf? "thetadot") (some 5)
    "Rod2dStateVector thetadot index should match Drake"
  LeanTest.assertEqual rod2dStateVectorBoundary.defaults #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    "Rod2dStateVector defaults should be zero"

  let input := #[1.0, 2.0, 3.0, 5.0, 7.0, 11.0]
  let state ← assertOk (stateFromArray? input) "Rod2dStateVector decode"
  LeanTest.assertEqual (stateAsArray state) input
    "Rod2dStateVector decode should preserve Drake's coordinate order"
  let msg ← assertError (stateFromArray? #[1.0, 2.0]) "Rod2dStateVector short decode"
  LeanTest.assertTrue (msg.contains "expects 6 coordinates")
    s!"short Rod2dStateVector decode should report coordinate count, got {msg}"

@[test]
def testRod2dGeometryProviderMatchesDrakeSceneGraphRegistration : IO Unit := do
  let result ← assertOk (buildRod2dGeometry? params (defaultState params))
    "Rod2dGeometry provider"
  assertOk result.provider.validate? "Rod2dGeometry SceneGraph provider"
  assertOk (result.poses.validate? result.provider) "Rod2dGeometry pose output"
  LeanTest.assertEqual result.inputPortName "state"
    "Rod2dGeometry should declare Drake's state input"
  LeanTest.assertEqual result.inputPortSize 6
    "Rod2dGeometry input should use Drake's six-element rod state"
  LeanTest.assertEqual result.outputPortName "geometry_pose"
    "Rod2dGeometry should declare an abstract output port named geometry_pose"
  LeanTest.assertTrue (approx result.radius rod2dDefaultVisualRadius 1.0e-12)
    s!"Rod2dGeometry default visual radius should match rod2d_sim, got {result.radius}"
  LeanTest.assertEqual result.provider.sources.size 1
    "Rod2dGeometry should register one SceneGraph source"
  LeanTest.assertEqual result.provider.frames.size 1
    "Rod2dGeometry should register one body frame"
  LeanTest.assertEqual result.provider.geometries.size 1
    "Rod2dGeometry should register one cylinder geometry"
  LeanTest.assertEqual result.provider.shapeNames #["cylinder"]
    "Rod2dGeometry should register a cylinder"
  LeanTest.assertTrue result.provider.anchoredGeometries.isEmpty
    "Rod2dGeometry should not anchor the rod visual geometry"

  let source ← assertSome (result.provider.sourceById? rod2dGeometrySourceId)
    "rod2d source lookup"
  LeanTest.assertEqual source.name "rod2d"
    "Rod2dGeometry source should preserve Drake's source name"
  let frame ← assertSome (result.provider.frameById? rod2dGeometryFrameId)
    "rod2d frame lookup"
  LeanTest.assertEqual frame.name "rod2d"
    "Rod2dGeometry frame should preserve Drake's frame name"
  let geometry ← assertSome (result.provider.geometryById? rod2dGeometryId)
    "rod2d cylinder geometry lookup"
  LeanTest.assertEqual geometry.name "rod2d"
    "Rod2dGeometry cylinder should preserve Drake's geometry name"
  LeanTest.assertEqual geometry.frameId? (some rod2dGeometryFrameId)
    "Rod2dGeometry cylinder should attach to the rod frame"
  LeanTest.assertTrue (geometry.X_FG == ScenePose3.identity)
    s!"Rod2dGeometry cylinder should use identity X_FG, got {reprStr geometry.X_FG}"
  LeanTest.assertTrue (geometry.properties.diffuseRgba? ==
      some { r := 0.7, g := 0.7, b := 0.7, a := 1.0 })
    s!"Rod2dGeometry cylinder should carry Drake's grey diffuse color, got {reprStr geometry.properties.diffuseRgba?}"
  LeanTest.assertTrue (geometry.hasRole .illustration)
    "Rod2dGeometry cylinder should carry the illustration role"
  match geometry.shape with
  | .cylinder radius length =>
      LeanTest.assertTrue
        (approx radius rod2dDefaultVisualRadius 1.0e-12 &&
          approx length (2.0 * params.halfLength) 1.0e-12)
        s!"Rod2dGeometry cylinder should have sim radius and full rod length, got {radius}, {length}"
  | other => LeanTest.fail s!"Rod2dGeometry should register a cylinder, got {reprStr other}"

@[test]
def testRod2dGeometryPoseOutputMatchesDrakeStateMapping : IO Unit := do
  let x : RodState :=
    {
      x := 1.2
      y := 2.3
      theta := 0.4
      xdot := -0.1
      ydot := 0.2
      thetadot := 0.3
    }
  let result ← assertOk (buildRod2dGeometry? params x)
    "Rod2dGeometry pose output"
  let pose ← assertSome (result.poses.poseForFrame? rod2dGeometryFrameId)
    "rod2d frame pose"
  LeanTest.assertTrue (approx pose.translation.x x.x 1.0e-12 &&
      approx pose.translation.y 0.0 1.0e-12 &&
      approx pose.translation.z x.y 1.0e-12)
    s!"Rod2dGeometry should map 2D x/y to 3D x/z, got {reprStr pose.translation}"
  LeanTest.assertTrue (pose.rotationAxis == SceneVec3.unitY)
    s!"Rod2dGeometry should rotate the cylinder around world Y, got {reprStr pose.rotationAxis}"
  LeanTest.assertTrue (approx pose.rotationAngle (x.theta + pi / 2.0) 1.0e-12)
    s!"Rod2dGeometry should use theta + pi/2, got {pose.rotationAngle}"
  let cylinderAxis_W := pose.rotateVector SceneVec3.unitZ
  let expectedAngle := x.theta + pi / 2.0
  LeanTest.assertTrue
    (approx cylinderAxis_W.x (Float.sin expectedAngle) 1.0e-12 &&
      approx cylinderAxis_W.z (Float.cos expectedAngle) 1.0e-12)
    s!"Rod2dGeometry Y rotation should orient the cylinder axis like Drake, got {reprStr cylinderAxis_W}"

@[test]
def testRod2dGeometryGraphRecordsExactSceneGraphBoundary : IO Unit := do
  let result ← assertOk (buildRod2dGeometry? params (defaultState params))
    "Rod2dGeometry graph"
  LeanTest.assertEqual result.moves.size 2
    "Rod2dGeometry should expose registration and pose-output local moves"
  LeanTest.assertTrue (result.moves.all (fun move =>
      move.kind == SkeletonMoveKind.localSchurBlock &&
      move.exactness == MoveExactness.exact))
    "Rod2dGeometry moves should be exact local SceneGraph blocks"
  LeanTest.assertTrue (result.moves.any (fun move =>
      move.targets == #[rod2dGeometryProviderVertex] &&
      move.writes == #[rod2dGeometryProviderVertex] &&
      move.label.contains "Register rod2d"))
    "Rod2dGeometry graph should record the provider registration move"
  LeanTest.assertTrue (result.moves.any (fun move =>
      move.targets == #[rod2dGeometryPoseOutputVertex] &&
      move.reads == #[rod2dGeometryStateInputVertex, rod2dGeometryProviderVertex] &&
      move.writes == #[rod2dGeometryPoseOutputVertex] &&
      move.label.contains "OutputGeometryPose"))
    "Rod2dGeometry graph should record the state-to-FramePoseVector move"

@[test]
def testDefaultPainleveStateAndSupportMatchDrake : IO Unit := do
  let x0 := defaultState params
  LeanTest.assertTrue (approx x0.x (params.halfLength * sqrt2Over2) 1.0e-12)
    s!"Default x should match Drake's half_length * sqrt(2)/2, got {x0.x}"
  LeanTest.assertTrue (approx x0.y (params.halfLength * sqrt2Over2) 1.0e-12)
    s!"Default y should match Drake's half_length * sqrt(2)/2, got {x0.y}"
  LeanTest.assertTrue (approx x0.theta (pi / 4.0) 1.0e-12)
    s!"Default theta should be pi/4, got {x0.theta}"
  LeanTest.assertTrue (approx x0.xdot (-1.0) 1.0e-12)
    s!"Default xdot should be -1, got {x0.xdot}"

  let support := selectedSupport params x0
  let runtime ← assertOk support.toRuntimeSupport? "rod2d default support"
  LeanTest.assertEqual runtime.selectedIds #[410]
    "Only the lower left endpoint should touch the halfspace in Drake's default state"
  LeanTest.assertTrue (!isImpacting params x0)
    "Drake's default state touches while sliding, but is not impacting along the normal"

  let selected ← assertOk support.selectedCandidates? "rod2d default selected endpoint"
  LeanTest.assertTrue (selected[0]!.mode == ContactMode.sliding)
    s!"Expected default touching endpoint to classify as sliding, got {reprStr selected[0]!.mode}"

@[test]
def testEndpointKinematicsAndJacobiansMatchDrakeRows : IO Unit := do
  let x0 := defaultState params
  let left := candidateForEndpoint params x0 .left
  let right := candidateForEndpoint params x0 .right

  LeanTest.assertTrue (approx left.signedDistance 0.0 1.0e-12)
    s!"Left endpoint should lie on the halfspace, got distance {left.signedDistance}"
  LeanTest.assertTrue (right.signedDistance > 1.0)
    s!"Right endpoint should be above the halfspace, got distance {right.signedDistance}"
  LeanTest.assertTrue (approx (left.normalJacobian.getD 2 0.0) (-sqrt2Over2) 1.0e-12)
    s!"Normal Jacobian rotational entry should be endpoint rx, got {reprStr left.normalJacobian}"
  LeanTest.assertTrue (approx (left.tangentJacobian.getD 2 0.0) sqrt2Over2 1.0e-12)
    s!"Tangent Jacobian rotational entry should be -endpoint ry, got {reprStr left.tangentJacobian}"
  LeanTest.assertTrue (approx left.tangentVelocity (-1.0) 1.0e-12)
    s!"Default left endpoint tangent velocity should be -1, got {left.tangentVelocity}"
  LeanTest.assertTrue (approx left.normalVelocity 0.0 1.0e-12)
    s!"Default left endpoint normal velocity should be zero, got {left.normalVelocity}"

@[test]
def testCompliantContactForceUsesDrakeHuntCrossleyStribeckShape : IO Unit := do
  let penetrating := { defaultState params with y := (defaultState params).y - 0.01 }
  let forces := compliantForces params penetrating
  let leftForce := forces[0]!
  LeanTest.assertTrue (approx leftForce.normalForce 100.0 1.0e-9)
    s!"One centimeter penetration should produce k*h = 100N normal force, got {leftForce.normalForce}"
  LeanTest.assertTrue (leftForce.tangentForce > 0.0)
    s!"Friction should oppose the negative slip velocity, got {leftForce.tangentForce}"
  LeanTest.assertTrue (leftForce.torque > 0.0)
    s!"Off-center contact force should produce positive torque for this state, got {leftForce.torque}"

  let dx := continuousDerivative params penetrating
  LeanTest.assertTrue (dx.xdot > 0.0)
    s!"Positive friction should accelerate x velocity upward, got {dx.xdot}"
  LeanTest.assertTrue (dx.ydot > 0.0)
    s!"Normal force should overcome gravity for this penetration, got {dx.ydot}"
  LeanTest.assertTrue (dx.thetadot > 0.0)
    s!"Contact torque should produce positive angular acceleration, got {dx.thetadot}"

@[test]
def testRod2dFullPhysicsPrimitiveMatchesContinuousDerivative : IO Unit := do
  let penetrating := { defaultState params with y := (defaultState params).y - 0.01 }
  let applied : SpatialForce2D := { fx := 3.0, fy := -4.0, tau := 1.5 }
  let localDerivative := continuousDerivative params penetrating applied
  let fullDerivative ← assertOk
    (continuousDerivativeFromFullPhysics? params penetrating applied)
    "rod2d full physics derivative"
  assertStateApprox fullDerivative localDerivative 1.0e-10
    "Shared FullPhysicsEquation path should match Rod2D's Drake-specific compliant derivative"

  let result ← assertOk (solveFullPhysics? params penetrating applied)
    "rod2d full physics primitive"
  let candidates ← assertOk result.support.selectedCandidates?
    "rod2d full physics selected candidates"
  LeanTest.assertEqual (candidates.map (fun c => c.id)) #[410]
    "Threshold support should select the penetrating lower endpoint"
  let localContact := aggregateContactForce (compliantForces params penetrating)
  assertArrayApprox result.generalizedContactForce (spatialForceAsArray localContact) 1.0e-10
    "Full physics should map Rod2D contact scalars through J^T"
  assertArrayApprox result.generalizedForces
    (FloatArray.add (spatialForceAsArray applied) (spatialForceAsArray localContact))
    1.0e-10
    "Full physics should compose applied forces and contact forces before bias subtraction"
  LeanTest.assertTrue (result.supportMove.kind == SkeletonMoveKind.markMarginalize)
    "Runtime support selection should remain a separate elimination move"
  LeanTest.assertTrue (result.supportMove.exactness == MoveExactness.controlledApproximation)
    "Threshold-selected support should be tagged as a fixed-trace approximation"
  LeanTest.assertTrue (result.move.kind == SkeletonMoveKind.intervalAdjoint)
    "The mass-matrix dynamics solve should be the interval-adjoint primitive"
  LeanTest.assertTrue (result.move.exactness == MoveExactness.exact)
    "The Rod2D full-physics solve should be exact for the selected support"

@[test]
def testRod2dFullPhysicsPrimitiveProviderRecomputesContactSupportAndInput :
    IO Unit := do
  let provider := fullPhysicsPrimitiveProvider params
    "rod2d provider recompute test"
  let separated := fallingState
  let penetrating := { defaultState params with y := (defaultState params).y - 0.01 }
  let applied : SpatialForce2D := { fx := 3.0, fy := -4.0, tau := 1.5 }

  let separatedPrimitive ← assertOk
    (provider.validPrimitivesAt? (physicsState separated))
    "rod2d provider primitive at separated state"
  let separatedSupport ← assertOk
    (provider.supportAt? (physicsState separated))
    "rod2d provider support at separated state"
  let penetratingPrimitive ← assertOk
    (provider.validPrimitivesAt? (physicsState penetrating applied))
    "rod2d provider primitive at penetrating state"
  let penetratingSupport ← assertOk
    (provider.supportAt? (physicsState penetrating applied))
    "rod2d provider support at penetrating state"
  let providerResult ← assertOk
    (provider.solveAt? (physicsState penetrating applied) 4321)
    "rod2d provider solve at penetrating state"
  let directResult ← assertOk
    (solveFullPhysics? params penetrating applied 4321)
    "rod2d direct solve for provider parity"

  LeanTest.assertEqual separatedPrimitive.contactCandidates.size 2
    "Rod2D provider should expose both endpoint candidates even when separated"
  LeanTest.assertEqual separatedPrimitive.contactForces.size 0
    "Rod2D provider should not synthesize contact force scalars for separated support"
  LeanTest.assertEqual separatedSupport.selectedLocalIndices #[]
    "Separated Rod2D provider support should be empty"
  LeanTest.assertEqual separatedSupport.sourceCandidateCount? (some 2)
    "Separated Rod2D provider support should preserve total endpoint candidate count"

  LeanTest.assertEqual penetratingPrimitive.qdot (velocityAsArray penetrating)
    "Rod2D provider qdot should recompute from the current state velocity"
  LeanTest.assertEqual penetratingPrimitive.actuationForces (spatialForceAsArray applied)
    "Rod2D provider actuation should recompute from the current applied force"
  LeanTest.assertEqual penetratingPrimitive.contactCandidates.size 2
    "Penetrating Rod2D provider primitive should keep both endpoint candidates"
  LeanTest.assertEqual penetratingPrimitive.contactForces.size 1
    "Penetrating Rod2D provider should retain one selected contact force"
  LeanTest.assertEqual penetratingSupport.selectedLocalIndices #[0]
    "Penetrating Rod2D provider should select the lower endpoint"
  let selected ← assertOk penetratingSupport.selectedCandidates?
    "rod2d provider selected penetrating endpoint"
  LeanTest.assertEqual (selected.map (fun c => c.id)) #[410]
    "Rod2D provider should preserve stable endpoint contact ids"
  assertArrayApprox providerResult.derivative.vdot directResult.derivative.vdot 1.0e-12
    "Rod2D provider solve should match the direct primitive solve"
  assertArrayApprox providerResult.generalizedForces directResult.generalizedForces 1.0e-12
    "Rod2D provider generalized forces should match the direct primitive solve"
  LeanTest.assertEqual providerResult.move.targets #[4321]
    "Rod2D provider solve should target the supplied interval vertex"

  let badState := { penetrating with ydot := 1.0 / 0.0 }
  let msg ← assertError
    (provider.validPrimitivesAt? (physicsState badState applied))
    "rod2d provider malformed state"
  LeanTest.assertTrue (msg.contains "state")
    s!"Malformed Rod2D state should fail at provider validation, got {msg}"
  let badApplied : SpatialForce2D := { applied with fx := 1.0 / 0.0 }
  let msg ← assertError
    (provider.validPrimitivesAt? (physicsState penetrating badApplied))
    "rod2d provider malformed applied force"
  LeanTest.assertTrue (msg.contains "applied force")
    s!"Malformed Rod2D applied force should fail at provider validation, got {msg}"

@[test]
def testImpactProjectionEnforcesEndpointNormalVelocity : IO Unit := do
  let touchingIncoming := { defaultState params with ydot := -1.0 }
  let impact ← assertOk (projectImpact? params .normalOnly touchingIncoming)
    "rod2d normal impact projection"
  let contacts ← assertOk impact.support.selectedCandidates? "rod2d impact selected contacts"
  let jac := impactJacobianRows .normalOnly contacts
  let postConstraintVelocity := FloatMatrix.matVec jac (velocityAsArray impact.postState)

  LeanTest.assertEqual impact.runtimeSupport.selectedIds #[410]
    "Impact projection should retain the touching left endpoint"
  LeanTest.assertTrue (approx (postConstraintVelocity.getD 0 99.0) 0.0 1.0e-12)
    s!"Post-impact endpoint normal velocity should be zero, got {reprStr postConstraintVelocity}"
  LeanTest.assertTrue (impact.postState.ydot > touchingIncoming.ydot)
    "Normal impact projection should make the vertical velocity less downward"

@[test]
def testSustainedContactLcpCancelsRestingContactLoads : IO Unit := do
  let p := { params with mass := 2.0, muCoulomb := 15.0, muStatic := 15.0 }
  let x := restingVerticalState p
  let solve ← assertOk
    (sustainedContactSolve? p x { fx := 100.0, tau := 100.0 })
    "rod2d sustained contact solve"

  LeanTest.assertEqual solve.data.runtimeSupport.selectedIds #[410]
    "Sustained contact should use the dynamically selected lower endpoint"
  LeanTest.assertEqual solve.data.nonSlidingContacts #[0]
    "Resting vertical contact should be represented as a non-sliding contact"
  LeanTest.assertTrue (solve.data.slidingContacts.isEmpty)
    "Resting vertical contact should not be classified as sliding"
  LeanTest.assertTrue
    ((solve.moves.map (fun move => move.kind)) ==
      #[SkeletonMoveKind.branchAggregate, SkeletonMoveKind.localSchurBlock])
    "The solve should expose dynamic support selection followed by local solver elimination"

  let fN := solve.packedConstraintForce.getD 0 0.0
  let fF := solve.packedConstraintForce.getD 1 0.0
  LeanTest.assertTrue (approx fN 19.62 1.0e-6)
    s!"Normal force should balance the 2kg rod weight, got {fN}"
  LeanTest.assertTrue (approx fF (-100.0) 1.0e-6)
    s!"Static friction should cancel the horizontal contact-point force, got {fF}"
  LeanTest.assertTrue (FloatArray.maxAbsDiff solve.acceleration #[0.0, 0.0, 0.0] < 1.0e-6)
    s!"Sustained-contact acceleration should be zero, got {reprStr solve.acceleration}"
  LeanTest.assertTrue (solve.lcpSolution.maxComplementarity < 1.0e-7)
    s!"LCP solution should satisfy complementarity, got {solve.lcpSolution.maxComplementarity}"

@[test]
def testConstraintProblemDataSupportsDuplicatedContactRows : IO Unit := do
  let p := { params with mass := 2.0, muCoulomb := 15.0, muStatic := 15.0 }
  let x := restingVerticalState p
  let data ← assertOk
    (constraintAccelProblemData? p x { fx := 100.0, tau := 100.0 } true 2 3)
    "rod2d duplicated sustained-contact data"

  LeanTest.assertEqual data.contactCount 3
    "Two duplicated contact points should expand one selected endpoint to three solver normal rows"
  LeanTest.assertEqual data.frictionDirectionCount 12
    "Three non-sliding contacts with four friction directions each should expose twelve tangent rows"
  LeanTest.assertEqual data.nonSlidingContacts #[0, 1, 2]
    "Duplicated resting contacts should all remain non-sliding"
  LeanTest.assertTrue data.slidingContacts.isEmpty
    "Duplicated resting contacts should not create sliding rows"
  LeanTest.assertEqual data.r #[4, 4, 4]
    "Each duplicated non-sliding contact should record four friction directions"
  LeanTest.assertEqual data.kF.size 12
    "Friction bias vector should have one entry per duplicated tangent row"
  LeanTest.assertEqual data.runtimeSupport.selectedIds #[410, 1410, 2410]
    "Duplicated contact rows should carry stable expanded candidate ids"
  LeanTest.assertEqual data.runtimeSupport.totalCandidates? (some 3)
    "Runtime support should describe the expanded solver candidate set"
  LeanTest.assertTrue data.useComplementarityProblemSolver
    "Duplicated data should preserve the Drake complementarity-solver flag"

  let directData ← assertOk
    (constraintAccelProblemData? p x { fx := 100.0, tau := 100.0 } false 2 3)
    "rod2d duplicated direct sustained-contact data"
  LeanTest.assertTrue (!directData.useComplementarityProblemSolver)
    "Constraint problem data should preserve the non-complementarity solver flag too"

@[test]
def testContactFrameForcesRecoverPackedSingleDirectionRows : IO Unit := do
  let p := { params with mass := 2.0, muCoulomb := 15.0, muStatic := 15.0 }
  let x := restingVerticalState p
  let data ← assertOk
    (constraintAccelProblemData? p x { fx := 100.0, tau := 100.0 } true 2 0)
    "rod2d duplicated single-direction sustained-contact data"
  let solve ← assertOk (solveSustainedContact? data)
    "rod2d duplicated single-direction sustained-contact solve"
  let frames ← assertOk (contactFrameForces? data solve.packedConstraintForce)
    "rod2d contact-frame force conversion"

  LeanTest.assertEqual (frames.map (fun force => force.candidateId)) #[410, 1410, 2410]
    "Contact-frame forces should preserve expanded candidate ids"
  let normalSum := frames.foldl (fun acc force => acc + force.normalForce) 0.0
  let tangentSum := frames.foldl (fun acc force => acc + force.tangentForce) 0.0
  LeanTest.assertTrue (approx normalSum 19.62 1.0e-6)
    s!"Duplicated normal rows should share the weight-balancing force, got {normalSum}"
  LeanTest.assertTrue (approx tangentSum (-100.0) 1.0e-6)
    s!"Duplicated tangent rows should share the applied horizontal load, got {tangentSum}"
  LeanTest.assertTrue (FloatArray.maxAbsDiff solve.acceleration #[0.0, 0.0, 0.0] < 1.0e-6)
    s!"Duplicated contact solve should still produce static equilibrium, got {reprStr solve.acceleration}"

@[test]
def testContactFrameForceConversionRejectsDuplicatedFrictionDirections : IO Unit := do
  let p := { params with mass := 2.0, muCoulomb := 15.0, muStatic := 15.0 }
  let x := restingVerticalState p
  let data ← assertOk
    (constraintAccelProblemData? p x { fx := 100.0, tau := 100.0 } true 0 1)
    "rod2d duplicated-friction sustained-contact data"
  let solve ← assertOk (solveSustainedContact? data)
    "rod2d duplicated-friction sustained-contact solve"
  let msg ← assertError (contactFrameForces? data solve.packedConstraintForce)
    "rod2d duplicated-friction contact-frame conversion"
  LeanTest.assertTrue (msg.contains "one friction direction")
    s!"Duplicated friction directions should be rejected at frame-force conversion, got {msg}"

@[test]
def testSustainedContactFastExitLeavesSeparatingMotionUnforced : IO Unit := do
  let data ← assertOk
    (constraintAccelProblemData? params { restingVerticalState params with ydot := 1.0 })
    "rod2d separating sustained contact data"
  let data := { data with kN := #[Float.abs params.gravity] }
  let solve ← assertOk (solveSustainedContact? data)
    "rod2d sustained contact fast exit"

  LeanTest.assertTrue solve.lcpMatrix.isEmpty
    "Positive normal acceleration should fast-exit without forming an LCP"
  LeanTest.assertTrue (solve.packedConstraintForce.all (fun f => approx f 0.0 1.0e-12))
    s!"Fast-exit contact force should be zero, got {reprStr solve.packedConstraintForce}"
  LeanTest.assertTrue
    ((solve.moves.map (fun move => move.kind)) == #[SkeletonMoveKind.localSchurBlock])
    "Fast-exit should still record the solver boundary as a local elimination"

@[test]
def testDiffEqRunLocalizesFirstImpactAndRecordsTrace : IO Unit := do
  let run ← assertOk (solveToFirstImpact? params fallingState 1.0)
    "rod2d first impact run"
  let lower := candidateForEndpoint params fallingState (lowerEndpoint fallingState)
  let expected :=
    (lower.normalVelocity +
      Float.sqrt (lower.normalVelocity * lower.normalVelocity +
        2.0 * Float.abs params.gravity * lower.signedDistance)) /
      Float.abs params.gravity

  LeanTest.assertTrue (approx run.eventTime expected 1.0e-3)
    s!"First impact time should match the ballistic endpoint root, got {run.eventTime}, expected {expected}"
  LeanTest.assertTrue (approx (minimumSignedDistance params run.eventState) 0.0 1.0e-5)
    s!"Localized event state should lie on the halfspace, got distance {minimumSignedDistance params run.eventState}"
  LeanTest.assertEqual run.impact.runtimeSupport.selectedIds #[410]
    "The falling test state should hit the left endpoint first"
  LeanTest.assertTrue
    (approx (run.impact.projection.constraintVelocityAfter.getD 0 99.0) 0.0 1.0e-10)
    s!"Projected normal velocity should be zero, got {reprStr run.impact.projection.constraintVelocityAfter}"

  match run.trace.validate? with
  | .error msg => LeanTest.fail s!"Rod2D trace should validate: {msg}"
  | .ok () => pure ()
  LeanTest.assertEqual run.moves.size 3
    "Localized interval plus dynamic contact support branch should project to three moves"
  LeanTest.assertTrue (run.moves[0]!.kind == SkeletonMoveKind.intervalAdjoint)
    "First move should eliminate the ballistic interval"
  LeanTest.assertTrue (run.moves[2]!.kind == SkeletonMoveKind.branchAggregate)
    "Last move should aggregate the dynamically selected contact support"

@[test]
def testEndToEndResultCarriesRod2dPackageArtifacts : IO Unit := do
  let result ← assertOk buildEndToEnd? "rod2d end-to-end result"
  LeanTest.assertEqual
    (result.assetCatalog.map (fun asset => asset.relativePath))
    rod2dExampleAssetPaths
    "Rod2D end-to-end result should carry the package asset catalog"
  LeanTest.assertEqual
    (result.documentationAssets.map (fun asset => asset.relativePath))
    #["README.md", "images/colliding-boxes.png"]
    "Rod2D documentation boundary should include README and image sidecar"
  LeanTest.assertTrue
    (hasPackageMove result.packageMoves "rod2d README image documentation boundary"
      SkeletonMoveKind.localSchurBlock MoveExactness.exact)
    "Rod2D end-to-end result should expose the README image as an exact metadata boundary"

end Tests.EventSkeletonRod2DExample
