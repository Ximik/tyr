import LeanTest
import Tyr.EventSkeleton.Examples.RollingSphere

namespace Tests.EventSkeletonRollingSphereExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.RollingSphere

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertError {α : Type} (res : Except String α) (label : String) :
    IO String := do
  match res with
  | .ok _ => LeanTest.fail s!"{label}: expected error, got ok"
  | .error msg => pure msg

private def getResult : IO RollingSphereResult := do
  match buildEndToEnd? params with
  | .ok result => pure result
  | .error msg => LeanTest.fail s!"Rolling sphere example failed to build: {msg}"

@[test]
def testDrakeReferencesAndFreeBodyShapeAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/rolling_sphere/populate_ball_plant.cc"))
    "Example should reference Drake's rolling-sphere plant population path"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/rolling_sphere/populate_ball_plant.h"))
    "Example should reference Drake's rolling-sphere plant population declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/rolling_sphere/rolling_sphere_run_dynamics.cc"))
    "Example should reference Drake's rolling-sphere dynamics runner"

  LeanTest.assertEqual positionCoordinateCount 7
    "Drake MultibodyPlant exposes a free sphere with quaternion plus xyz positions"
  LeanTest.assertEqual velocityCoordinateCount 6
    "Drake MultibodyPlant exposes free-body spatial velocity"
  LeanTest.assertTrue preservesDrakeAngularVelocityFlagUnits
    "The default state should preserve Drake's runtime angular-velocity flag convention"

  let state := defaultState params
  LeanTest.assertTrue (approx state.z params.radius 1.0e-12)
    s!"Default z0 should put the sphere at the ground contact surface, got {state.z}"
  LeanTest.assertTrue (approx state.vx 1.5 1.0e-12)
    s!"Default vx should match Drake's flag default, got {state.vx}"
  LeanTest.assertTrue (approx state.wy (-360.0) 1.0e-12)
    s!"Default wy should match Drake's runtime SpatialVelocity value, got {state.wy}"

@[test]
def testSolidSphereInertiaAndVisualSpotsMatchDrakePlant : IO Unit := do
  LeanTest.assertTrue (approx (solidSphereRotationalInertia params) 1.0e-4 1.0e-12)
    s!"Solid sphere inertia should be 2/5 m r^2, got {solidSphereRotationalInertia params}"
  LeanTest.assertTrue (approx (visualSpotRadius params) 0.01 1.0e-12)
    s!"Visual spot radius should be 0.2 * radius, got {visualSpotRadius params}"
  LeanTest.assertTrue (approx (visualSpotRadialOffset params) 0.0455 1.0e-12)
    s!"Visual spot radial offset should avoid z-fighting, got {visualSpotRadialOffset params}"

  let spots := visualSpots params
  LeanTest.assertEqual spots.size 6
    "Drake adds six colored spot cylinders to make rotation visible"
  LeanTest.assertEqual (spots.map (fun spot => spot.name))
    #["sphere_x+", "sphere_x-", "sphere_y+", "sphere_y-", "sphere_z+", "sphere_z-"]
    "Spot names should preserve Drake's visual geometry names"
  LeanTest.assertTrue
    (spots[0]!.color == spots[1]!.color &&
      approx spots[0]!.color.r 1.0 1.0e-12 &&
      approx spots[0]!.color.g 0.0 1.0e-12 &&
      approx spots[0]!.color.b 0.0 1.0e-12)
    "The +/-x spots should be red"
  LeanTest.assertTrue
    (spots[4]!.color == spots[5]!.color &&
      approx spots[4]!.color.r 0.0 1.0e-12 &&
      approx spots[4]!.color.g 0.0 1.0e-12 &&
      approx spots[4]!.color.b 1.0 1.0e-12)
    "The +/-z spots should be blue"

@[test]
def testContactModelCompatibilityMatchesDrakeReadme : IO Unit := do
  let hydroDefault := { params with contactModel := ContactModelChoice.hydroelastic }
  LeanTest.assertTrue (groundPairResolution hydroDefault).usesHydroelastic
    "Rigid ground plus compliant sphere should support hydroelastic surface contact"

  let hydroRigidSphere := { hydroDefault with rigidSphere := true }
  LeanTest.assertTrue (groundPairResolution hydroRigidSphere).isUnsupported
    "Rigid ground plus rigid sphere should be unsupported for pure hydroelastic contact"

  let hybridRigidSphere := { hydroRigidSphere with contactModel := ContactModelChoice.hybrid }
  LeanTest.assertTrue ((groundPairResolution hybridRigidSphere) == ContactPairResolution.pointFallback)
    "Hybrid contact should fall back to point contact for rigid-rigid sphere-ground contact"

  let hydroWithWall := { hydroDefault with addWall := true }
  match wallPairResolution? hydroWithWall with
  | some resolution =>
      LeanTest.assertTrue resolution.isUnsupported
        s!"Compliant wall plus compliant sphere should be unsupported for pure hydroelastic contact, got {reprStr resolution}"
  | none => LeanTest.fail "Expected optional wall contact resolution when addWall is true"

  let hybridWithWall := { hydroWithWall with contactModel := ContactModelChoice.hybrid }
  match wallPairResolution? hybridWithWall with
  | some ContactPairResolution.pointFallback => pure ()
  | other =>
      LeanTest.fail
        s!"Hybrid contact should fall back for compliant-compliant wall contact, got {reprStr other}"

@[test]
def testResolvedContactPrimitivesAreDynamicHydroAndFallbackProducts : IO Unit := do
  let hydroDefault := { params with contactModel := ContactModelChoice.hydroelastic }
  let hydroResolved ← assertOk
    (resolvedContactPrimitives? hydroDefault (defaultState hydroDefault))
    "hydroelastic default resolved primitives"
  LeanTest.assertEqual hydroResolved.hydroelasticPatches.size 1
    "Supported strict hydro contact should emit a hydroelastic patch primitive"
  LeanTest.assertEqual hydroResolved.pointCandidates.size 0
    "Strict hydro contact should not emit point candidates for supported hydro pairs"
  LeanTest.assertEqual hydroResolved.fallbackCandidates.size 0
    "Strict hydro contact should not emit fallback candidates"

  let hydroSolver ← assertOk
    (solverContactsForResolved? hydroDefault hydroResolved)
    "hydroelastic solver contacts"
  LeanTest.assertEqual hydroSolver.candidates.size 1
    "A hydroelastic patch should lower to one solver-facing ContactCandidate"
  LeanTest.assertEqual hydroSolver.forces.size 1
    "A hydroelastic patch should lower to selected contact-force scalars"
  LeanTest.assertTrue (hydroSolver.candidates[0]!.label.contains "hydroelastic_patch")
    s!"Expected hydroelastic patch candidate label, got {hydroSolver.candidates[0]!.label}"
  LeanTest.assertTrue (approx hydroSolver.forces[0]!.normalForce 0.981 1.0e-12)
    s!"Hydro support force should still use the full-physics support primitive, got {hydroSolver.forces[0]!.normalForce}"

  let hydroWithWall := { hydroDefault with addWall := true }
  let separatedWall ← assertOk
    (resolvedContactPrimitives? hydroWithWall (defaultState hydroWithWall))
    "strict hydro with separated unsupported wall"
  LeanTest.assertEqual separatedWall.hydroelasticPatches.size 1
    "A separated unsupported wall pair should not throw before it is active"

  let wallHitState :=
    { defaultState hydroWithWall with
      x := wallRightFaceX hydroWithWall + hydroWithWall.radius
      vx := -1.0
    }
  match resolvedContactPrimitives? hydroWithWall wallHitState with
  | .ok value =>
      LeanTest.fail
        s!"Strict hydro should reject active compliant-compliant wall contact, got {reprStr value}"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "hydroelastic contact requires")
        s!"Expected strict hydro unsupported message, got {msg}"

  let hybridWithWall := { hydroWithWall with contactModel := ContactModelChoice.hybrid }
  let hybridResolved ← assertOk
    (resolvedContactPrimitives? hybridWithWall wallHitState)
    "hybrid wall-hit resolved primitives"
  LeanTest.assertEqual hybridResolved.hydroelasticPatches.size 1
    "Hybrid should keep the supported sphere-ground hydroelastic patch"
  LeanTest.assertEqual hybridResolved.fallbackCandidates.size 1
    "Hybrid should emit a point fallback candidate for active compliant-compliant wall contact"
  LeanTest.assertEqual hybridResolved.fallbackCandidates[0]!.id ContactSurface.wall.candidateId
    "The fallback candidate should be the wall contact"

  let hybridSolver ← assertOk
    (solverContactsForResolved? hybridWithWall hybridResolved)
    "hybrid wall-hit solver contacts"
  LeanTest.assertEqual hybridSolver.candidates.size 2
    "Hybrid wall hit should lower both hydro and fallback products into solver candidates"
  LeanTest.assertEqual hybridSolver.forces.size 2
    "Hybrid wall hit should compute selected force scalars for both active contacts"

@[test]
def testGroundCandidateProvidesFull3DPointContactRows : IO Unit := do
  let state := defaultState params
  let candidate := groundContactCandidate params state
  LeanTest.assertTrue (approx candidate.signedDistance 0.0 1.0e-12)
    s!"Default sphere should be exactly touching the ground, got distance {candidate.signedDistance}"
  LeanTest.assertTrue (approx candidate.normalVelocity 0.0 1.0e-12)
    s!"Default normal velocity should be zero, got {candidate.normalVelocity}"
  LeanTest.assertTrue (approx candidate.tangentVelocity 19.5 1.0e-12)
    s!"Default x slip should include vx - r * wy, got {candidate.tangentVelocity}"
  LeanTest.assertTrue (approx candidate.tangentVelocity2 0.0 1.0e-12)
    s!"Default y slip should be zero, got {candidate.tangentVelocity2}"
  LeanTest.assertTrue (candidate.mode == ContactMode.sliding)
    s!"Default contact should classify as sliding, got {reprStr candidate.mode}"

  let support := activeSupport params state
  let _ ← assertOk
    (support.validateJacobianWidth? velocityCoordinateCount)
    "rolling-sphere support width"
  let rows ← assertOk (support.constraintJacobianRows? true)
    "full 3D contact rows"
  LeanTest.assertEqual rows
    #[groundNormalJacobian, groundTangentXJacobian params, groundTangentYJacobian params]
    "A 3D point contact should expose normal plus two tangent Jacobian rows"

@[test]
def testPhysicsStepUsesMassMatrixContactForcesAndFriction : IO Unit := do
  let result ← getResult
  LeanTest.assertEqual result.runtimeSupport.selectedIds #[100]
    "The default support should retain the sphere-ground contact candidate"
  LeanTest.assertEqual result.contactForces.size 1
    "The default state should produce one active contact force"

  let force := result.contactForces[0]!
  LeanTest.assertTrue (approx force.normalForce 0.981 1.0e-12)
    s!"Ground support normal force should balance m*g, got {force.normalForce}"
  LeanTest.assertTrue (approx force.tangentForce (-0.2943) 1.0e-12)
    s!"Coulomb friction should oppose positive x slip, got {force.tangentForce}"
  LeanTest.assertTrue (approx (force.generalizedForce.getD 4 0.0) 0.014715 1.0e-12)
    s!"Friction should produce positive y torque, got {reprStr force.generalizedForce}"

  LeanTest.assertTrue (approx result.derivative.vz 0.0 1.0e-12)
    s!"Normal support should cancel gravity at the contact surface, got {result.derivative.vz}"
  LeanTest.assertTrue (approx result.derivative.vx (-2.943) 1.0e-12)
    s!"Friction should decelerate the translational x velocity, got {result.derivative.vx}"
  LeanTest.assertTrue (approx result.derivative.wy 147.15 1.0e-9)
    s!"Friction torque should drive wy toward rolling, got {result.derivative.wy}"

  LeanTest.assertTrue (result.oneStepState.vx < result.state.vx)
    "One physics step should reduce positive x velocity"
  LeanTest.assertTrue (result.oneStepState.wy > result.state.wy)
    "One physics step should make the negative y spin less negative"
  LeanTest.assertTrue (result.rolloutState.vx < result.oneStepState.vx)
    "The executable rollout should keep recomputing contact friction across steps"

@[test]
def testFullPhysicsPrimitiveAssemblesRollingDynamics : IO Unit := do
  let result ← getResult
  LeanTest.assertEqual result.fullPhysics.equation.massMatrix (massMatrix params)
    "The rolling-sphere full-physics primitive should expose the Drake-style mass matrix"
  LeanTest.assertEqual result.fullPhysics.derivative.qdot result.state.velocityVector
    "The primitive qdot should be the current spatial velocity"
  LeanTest.assertEqual result.fullPhysics.support.totalCandidates 1
    "The default state should dynamically generate one contact candidate"
  LeanTest.assertEqual result.fullPhysics.support.selectedLocalIndices #[0]
    "Threshold support should select the touching ground candidate"
  LeanTest.assertEqual result.fullPhysics.generalizedPrimitiveForce
    (gravityGeneralizedForce params)
    "Gravity should enter as an explicit primitive generalized force"
  LeanTest.assertEqual result.fullPhysics.generalizedContactForce
    (aggregateContactGeneralizedForce result.contactForces)
    "Contact scalars should map through the shared J^T f primitive"
  LeanTest.assertEqual result.fullPhysics.generalizedForces
    (appliedGeneralizedForce params result.contactForces)
    "Full physics should compose primitive gravity and contact before solving"
  LeanTest.assertEqual result.fullPhysics.contactForces.size result.contactForces.size
    "The common full-physics result should retain the selected contact-force scalars"
  LeanTest.assertTrue
    (result.fullPhysics.contactForces[0]!.candidateId == result.contactForces[0]!.candidateId)
    "Precomputed scalar forces should stay aligned with selected dynamic candidates"
  LeanTest.assertTrue
    (approx (result.fullPhysics.derivative.vdot.getD 0 0.0) result.derivative.vx 1.0e-12 &&
      approx (result.fullPhysics.derivative.vdot.getD 2 0.0) result.derivative.vz 1.0e-12 &&
      approx (result.fullPhysics.derivative.vdot.getD 4 0.0) result.derivative.wy 1.0e-12)
    s!"Full physics acceleration should match the exposed derivative, got {result.fullPhysics.derivative.vdot}"

@[test]
def testFullPhysicsPrimitiveProviderRecomputesDynamicContactSupport :
    IO Unit := do
  let provider := fullPhysicsPrimitiveProvider params
    "rolling sphere provider recompute test"
  let activeState := defaultState params
  let separatedState := { activeState with z := params.radius + 0.25 }

  let activePrimitive ← assertOk
    (provider.validPrimitivesAt? activeState)
    "rolling sphere provider primitive at active state"
  let separatedPrimitive ← assertOk
    (provider.validPrimitivesAt? separatedState)
    "rolling sphere provider primitive at separated state"
  let separatedSupport ← assertOk
    (provider.supportAt? separatedState)
    "rolling sphere provider separated support"
  let separatedResult ← assertOk
    (provider.solveAt? separatedState 402)
    "rolling sphere provider separated solve"

  LeanTest.assertEqual activePrimitive.contactCandidates.size 1
    "Active provider primitive should expose the current ground contact candidate"
  LeanTest.assertTrue
    (approx activePrimitive.contactCandidates[0]!.signedDistance 0.0 1.0e-12)
    s!"Active provider candidate should be touching, got {activePrimitive.contactCandidates[0]!.signedDistance}"
  LeanTest.assertEqual activePrimitive.contactForces.size 1
    "Active provider primitive should retain one precomputed contact force"
  LeanTest.assertTrue
    (approx activePrimitive.contactForces[0]!.normalForce 0.981 1.0e-12)
    s!"Active provider normal force should balance weight, got {activePrimitive.contactForces[0]!.normalForce}"

  LeanTest.assertEqual separatedPrimitive.contactCandidates.size 1
    "Separated provider primitive should still expose the runtime candidate view"
  LeanTest.assertTrue
    (approx separatedPrimitive.contactCandidates[0]!.signedDistance 0.25 1.0e-12)
    s!"Separated provider should recompute distance from the new state, got {separatedPrimitive.contactCandidates[0]!.signedDistance}"
  LeanTest.assertEqual separatedPrimitive.contactForces.size 0
    "Separated provider primitive should not retain inactive contact force scalars"
  LeanTest.assertEqual separatedSupport.selectedLocalIndices #[]
    "Separated provider support should be recomputed as empty"
  LeanTest.assertEqual separatedResult.move.targets #[402]
    "Provider solve should use the supplied interval vertex"
  LeanTest.assertEqual separatedResult.generalizedContactForce
    (Array.replicate velocityCoordinateCount 0.0)
    "Separated provider solve should assemble zero generalized contact force"

  let wallParams := { params with addWall := true }
  let wallProvider := fullPhysicsPrimitiveProvider wallParams
    "rolling sphere wall provider recompute test"
  let wallHitState :=
    { defaultState wallParams with
      x := wallRightFaceX wallParams + wallParams.radius
      vx := -1.0
    }
  let wallPrimitive ← assertOk
    (wallProvider.validPrimitivesAt? wallHitState)
    "rolling sphere provider primitive at wall-hit state"
  let wallSupport ← assertOk
    (wallProvider.supportAt? wallHitState)
    "rolling sphere provider wall-hit support"

  LeanTest.assertEqual wallPrimitive.contactCandidates.size 2
    "Wall provider primitive should dynamically expose ground and wall candidates"
  LeanTest.assertEqual wallSupport.selectedLocalIndices #[0, 1]
    "Wall-hit provider support should select both active contact candidates"
  LeanTest.assertEqual
    (wallPrimitive.contactForces.map (fun force => force.candidateId))
    #[ContactSurface.ground.candidateId, ContactSurface.wall.candidateId]
    "Wall-hit provider forces should stay aligned with selected candidate ids"
  LeanTest.assertTrue
    (approx wallPrimitive.contactForces[1]!.normalForce wallParams.dissipation 1.0e-12)
    s!"Wall normal force should come from closing velocity damping, got {wallPrimitive.contactForces[1]!.normalForce}"

  let badState := { activeState with vz := 1.0 / 0.0 }
  let msg ← assertError
    (provider.validPrimitivesAt? badState)
    "rolling sphere provider malformed state"
  LeanTest.assertTrue (msg.contains "state")
    s!"Malformed rolling sphere state should fail at provider validation, got {msg}"

@[test]
def testVelocityProjectionUsesNormalAndTwoTangents : IO Unit := do
  let result ← getResult
  LeanTest.assertTrue (approx (result.normalProjection.constraintVelocityAfter.getD 0 99.0) 0.0 1.0e-12)
    s!"Normal projection should enforce zero normal velocity, got {reprStr result.normalProjection.constraintVelocityAfter}"

  LeanTest.assertEqual result.stickingProjection.constraintVelocityBefore.size 3
    "Sticking projection should include normal plus two tangent rows"
  LeanTest.assertTrue (approx (result.stickingProjection.constraintVelocityBefore.getD 1 0.0) 19.5 1.0e-12)
    s!"Before projection, x slip should match the contact candidate, got {reprStr result.stickingProjection.constraintVelocityBefore}"
  LeanTest.assertTrue
    (result.stickingProjection.constraintVelocityAfter.all (fun value => Float.abs value < 1.0e-9))
    s!"Projection should enforce all 3D point-contact rows, got {reprStr result.stickingProjection.constraintVelocityAfter}"

@[test]
def testTraceRecordsDynamicSupportBoundary : IO Unit := do
  let result ← getResult
  match result.trace.validate? with
  | .error msg => LeanTest.fail s!"Trace should validate: {msg}"
  | .ok () => pure ()

  LeanTest.assertEqual result.moves.size 5
    "Trace moves plus full-physics support and interval moves should be exposed"
  LeanTest.assertTrue (result.moves[0]!.kind == SkeletonMoveKind.intervalAdjoint)
    "First move should eliminate the contact-aware interval"
  LeanTest.assertTrue (result.moves[1]!.kind == SkeletonMoveKind.checkpointBoundary)
    "Second move should retain the step boundary"
  LeanTest.assertTrue (result.moves[2]!.kind == SkeletonMoveKind.branchAggregate)
    "Third move should aggregate the dynamic contact support"
  LeanTest.assertTrue (result.moves[2]!.exactness == MoveExactness.controlledApproximation)
    "Thresholded contact support is a fixed-trace approximation"
  LeanTest.assertTrue (result.moves[3]!.kind == SkeletonMoveKind.markMarginalize)
    "The full-physics primitive should expose support selection as a mark move"
  LeanTest.assertTrue (result.moves[4]!.kind == SkeletonMoveKind.intervalAdjoint)
    "The full-physics primitive should expose the mass-matrix solve as an interval move"

  LeanTest.assertEqual result.branchData.children.size 1
    "Default support branch should have one retained contact child"
  LeanTest.assertTrue (result.branchResult.value > 0.0)
    s!"Contact branch should carry a positive physics message, got {result.branchResult.value}"

end Tests.EventSkeletonRollingSphereExample
