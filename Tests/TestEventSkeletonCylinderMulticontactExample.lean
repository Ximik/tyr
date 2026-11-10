import LeanTest
import Tyr.EventSkeleton.Examples.CylinderMulticontact

namespace Tests.EventSkeletonCylinderMulticontactExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.CylinderMulticontact

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def sumNormalForces (forces : Array ContactForce) : Float :=
  forces.foldl (fun acc force => acc + force.normalForce) 0.0

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertError {α : Type} (res : Except String α) (label : String) :
    IO String := do
  match res with
  | .ok _ => LeanTest.fail s!"{label}: expected error, got ok"
  | .error msg => pure msg

private def assertArrayNear
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

private def getResult : IO CylinderMulticontactResult := do
  match buildEndToEnd? with
  | .ok result => pure result
  | .error msg => LeanTest.fail s!"Cylinder multicontact example failed to build: {msg}"

@[test]
def testDrakeReferencesAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/cylinder_with_multicontact/populate_cylinder_plant.cc"))
    "Example should reference Drake's multicontact geometry population path"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/cylinder_with_multicontact/populate_cylinder_plant.h"))
    "Example should reference Drake's multicontact geometry population declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/cylinder_with_multicontact/cylinder_run_dynamics.cc"))
    "Example should reference Drake's multicontact dynamics runner"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/cylinder_with_multicontact/test/populate_cylinder_plant_test.cc"))
    "Example should reference Drake's inertia and plant-dimension regression"

@[test]
def testCylinderGeneratesDynamicContactCandidates : IO Unit := do
  let x := preContactState params
  let candidates := contactCandidates params x
  LeanTest.assertEqual candidates.size 20
    "Drake cylinder example uses ten contact spheres on each rim"

  let batch := contactCandidateBatch params x
  LeanTest.assertEqual batch.size 20
    "Packed contact batch should keep one source row per rim sphere"
  let candidateSet ← assertOk (contactCandidateSet? params x)
    "packed cylinder candidate set"
  LeanTest.assertEqual candidateSet.totalCandidates 20
    "Materialized candidate set should preserve the source candidate count"
  LeanTest.assertEqual candidateSet.candidates.size candidates.size
    "Full materialization should expose the same number of candidate views"

  let allSupport := allActiveSupport params x
  LeanTest.assertEqual allSupport.selectedLocalIndices.size 10
    "At first impact, all ten bottom-rim spheres should lie on the ground"
  let allRuntime ← assertOk allSupport.toRuntimeSupport? "all active support"
  LeanTest.assertEqual allRuntime.selectedIds
    #[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
    "Active support should expose stable bottom-rim source IDs"

  let selected ← assertOk allSupport.selectedCandidates? "selected all contacts"
  LeanTest.assertTrue
    (selected.all (fun candidate => candidate.mode == ContactMode.impacting))
    s!"Expected every active bottom contact to classify as impacting, got {reprStr (selected.map (fun c => c.mode))}"

@[test]
def testPackedCylinderBatchFiltersBeforeSupportSelection : IO Unit := do
  let x := preContactState params
  let retained ← assertOk
    ((contactCandidateBatch params x).retainedByDistance? params.penetrationAllowance
      "packed retained cylinder contacts")
    "distance-retained packed contacts"
  LeanTest.assertEqual retained.candidates.size 10
    "Distance retention should materialize only the bottom-rim candidates at impact"
  LeanTest.assertEqual retained.totalCandidates 20
    "Packed retention should preserve the full source candidate count"
  LeanTest.assertEqual (retained.candidates.map (fun candidate => candidate.id))
    #[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
    "Distance-retained contacts should keep stable bottom-rim IDs"

  let closest ← assertOk
    ((contactCandidateBatch params x).retainedClosestK? params.supportBudget
      "packed closest cylinder contacts")
    "closest-k packed contacts"
  LeanTest.assertEqual closest.candidates.size params.supportBudget
    "Closest-k retention should materialize only the configured contact budget"
  LeanTest.assertEqual closest.totalCandidates 20
    "Closest-k retention should still report the full source candidate count"
  LeanTest.assertEqual (closest.candidates.map (fun candidate => candidate.id))
    #[2000, 2001, 2002, 2003]
    "Closest-k packed retention should preserve deterministic bottom-rim IDs"

@[test]
def testBudgetedSupportKeepsStableIdsAndPhysicsPayload : IO Unit := do
  let result ← getResult
  LeanTest.assertEqual result.runtimeSupport.selectedIds #[2000, 2001, 2002, 2003]
    "Budgeted closest-k support should keep the first four closest bottom-rim contact IDs"
  LeanTest.assertEqual result.runtimeSupport.totalCandidates? (some 20)
    "Runtime support should preserve the full candidate count"
  LeanTest.assertTrue (result.runtimeSupport.exactness == MoveExactness.controlledApproximation)
    "Budgeted support is an explicit fixed-trace approximation"

  LeanTest.assertEqual result.contactForces.size params.supportBudget
    "The force kernel should evaluate exactly the retained contact support"
  LeanTest.assertTrue
    (result.contactForces.all (fun force => force.normalForce > 0.0))
    s!"Expected positive normal forces for closing contacts, got {reprStr result.contactForces}"
  LeanTest.assertTrue
    (result.contactForces.all (fun force => force.tangentForceX < 0.0))
    s!"Expected friction to oppose positive x slip, got {reprStr result.contactForces}"

@[test]
def testPhysicsStepRecomputesSupportAndUpdatesState : IO Unit := do
  let result ← getResult
  LeanTest.assertTrue (result.derivative.centerZ < 0.0)
    s!"Position derivative should still be the incoming vertical velocity, got {result.derivative.centerZ}"
  LeanTest.assertTrue (result.derivative.vz > 0.0)
    s!"Contact forces should produce upward vertical acceleration, got {result.derivative.vz}"
  LeanTest.assertTrue (result.derivative.vx < 0.0)
    s!"Friction should decelerate positive x velocity, got {result.derivative.vx}"

  LeanTest.assertTrue (result.oneStepState.centerZ < result.state.centerZ)
    "The first explicit step should continue moving downward before contact acceleration changes velocity"
  LeanTest.assertTrue (result.oneStepState.vz > result.state.vz)
    "The first explicit step should make vertical velocity less negative"
  LeanTest.assertTrue (result.oneStepState.vx < result.state.vx)
    "The first explicit step should reduce forward slip velocity"
  LeanTest.assertTrue (!approx result.rolloutState.centerZ result.oneStepState.centerZ 1.0e-12)
    "The rollout should recompute dynamic support across multiple steps, not reuse a single terminal state"

@[test]
def testFullPhysicsPrimitiveAssemblesPenaltyContactStep : IO Unit := do
  let result ← getResult
  LeanTest.assertEqual result.fullPhysics.equation.massMatrix (massMatrix params)
    "The full-physics primitive should expose the solid-cylinder mass matrix"
  LeanTest.assertEqual result.fullPhysics.derivative.qdot (velocityVector result.state)
    "The primitive qdot should be the current generalized velocity"
  LeanTest.assertEqual result.fullPhysics.support.totalCandidates 20
    "The full-physics support should retain the dynamic source candidate count"
  let selectedIds ← assertOk result.fullPhysics.support.selectedIds?
    "full-physics selected ids"
  LeanTest.assertEqual selectedIds result.runtimeSupport.selectedIds
    "Full physics should use the same closest-k dynamic support as the branch payload"
  LeanTest.assertEqual result.fullPhysics.generalizedPrimitiveForce
    (gravityGeneralizedForce params)
    "Gravity should be represented as a primitive generalized force"
  LeanTest.assertEqual result.fullPhysics.generalizedContactForce
    (aggregateContactWrench result.contactForces).asArray
    "Selected contact scalars should map through the shared J^T f boundary"
  LeanTest.assertEqual result.fullPhysics.generalizedForces
    (FloatArray.add (gravityGeneralizedForce params)
      (aggregateContactWrench result.contactForces).asArray)
    "Full physics should compose gravity and contact forces before the mass solve"
  LeanTest.assertEqual result.fullPhysics.contactForces.size result.contactForces.size
    "The full-physics result should retain one scalar force record per selected contact"
  LeanTest.assertTrue
    (result.fullPhysics.contactForces.all (fun force =>
      result.runtimeSupport.selectedIds.contains force.candidateId))
    s!"Scalar force records should stay aligned with runtime support, got {reprStr result.fullPhysics.contactForces}"
  LeanTest.assertTrue
    (approx (result.fullPhysics.derivative.vdot.getD 0 0.0) result.derivative.vx 1.0e-12 &&
      approx (result.fullPhysics.derivative.vdot.getD 2 0.0) result.derivative.vz 1.0e-12 &&
      approx (result.fullPhysics.derivative.vdot.getD 3 0.0) result.derivative.wx 1.0e-12)
    s!"Full physics acceleration should match the exposed derivative, got {result.fullPhysics.derivative.vdot}"

@[test]
def testFullPhysicsPrimitiveProviderRecomputesPackedDynamicSupport :
    IO Unit := do
  let x := preContactState params
  let budgetedProvider := fullPhysicsPrimitiveProvider params
    CylinderSupportMode.budgeted
    "cylinder budgeted provider recompute test"
  let allActiveProvider := fullPhysicsPrimitiveProvider params
    CylinderSupportMode.allActive
    "cylinder all-active provider recompute test"

  let budgetedPrimitive ← assertOk
    (budgetedProvider.validPrimitivesAt? x)
    "cylinder budgeted provider primitive at first contact"
  let budgetedSupport ← assertOk
    (budgetedProvider.supportAt? x)
    "cylinder budgeted provider support at first contact"
  let budgetedResult ← assertOk
    (budgetedProvider.solveAt? x 302)
    "cylinder budgeted provider solve at first contact"
  let directSupport ← assertOk (budgetedSupport? params x)
    "cylinder direct budgeted support"
  let (directResult, _) ← assertOk
    (solveFullPhysics? params directSupport x)
    "cylinder direct budgeted full physics"

  LeanTest.assertEqual budgetedPrimitive.contactCandidates.size params.supportBudget
    "Budgeted provider primitive should materialize only retained packed candidates"
  LeanTest.assertEqual budgetedPrimitive.sourceContactCandidateCount? (some 20)
    "Budgeted provider primitive should preserve full packed source candidate count"
  let budgetedIds ← assertOk budgetedSupport.selectedIds?
    "cylinder budgeted provider selected ids"
  LeanTest.assertEqual budgetedIds #[2000, 2001, 2002, 2003]
    "Budgeted provider should recompute deterministic bottom-rim support ids"
  LeanTest.assertEqual budgetedPrimitive.contactForces.size params.supportBudget
    "Budgeted provider primitive should keep one scalar force per retained contact"
  LeanTest.assertTrue
    (budgetedPrimitive.contactForces.all (fun force => force.normalForce > 0.0))
    s!"Budgeted provider should compute positive normal forces, got {reprStr budgetedPrimitive.contactForces}"
  LeanTest.assertEqual budgetedResult.move.targets #[302]
    "Provider solve should use the supplied interval vertex"
  assertArrayNear budgetedResult.derivative.vdot directResult.derivative.vdot 1.0e-12
    "Provider solve should match direct budgeted full physics"

  let airbornePrimitive ← assertOk
    (allActiveProvider.validPrimitivesAt? initialState)
    "cylinder all-active provider primitive before contact"
  let airborneSupport ← assertOk
    (allActiveProvider.supportAt? initialState)
    "cylinder all-active provider support before contact"
  LeanTest.assertEqual airbornePrimitive.contactCandidates.size 0
    "All-active provider should not materialize separated contacts before impact"
  LeanTest.assertEqual airbornePrimitive.sourceContactCandidateCount? (some 20)
    "All-active provider should still report the full packed source candidate count"
  LeanTest.assertEqual airborneSupport.selectedLocalIndices #[]
    "All-active provider support should be empty before impact"
  LeanTest.assertEqual airbornePrimitive.contactForces.size 0
    "All-active provider should not synthesize force scalars before impact"

  let restingPrimitive ← assertOk
    (allActiveProvider.validPrimitivesAt? (restingState params))
    "cylinder all-active provider primitive at rest"
  let restingSupport ← assertOk
    (allActiveProvider.supportAt? (restingState params))
    "cylinder all-active provider support at rest"
  LeanTest.assertEqual restingPrimitive.contactCandidates.size params.contactsPerRim
    "All-active provider should materialize every bottom-rim contact at rest"
  let restingIds ← assertOk restingSupport.selectedIds?
    "cylinder all-active provider resting selected ids"
  LeanTest.assertEqual restingIds
    #[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
    "All-active provider should preserve all bottom-rim stable ids at rest"

  let badState := { x with vz := 1.0 / 0.0 }
  let msg ← assertError
    (budgetedProvider.validPrimitivesAt? badState)
    "cylinder provider malformed state"
  LeanTest.assertTrue (msg.contains "state")
    s!"Malformed cylinder state should fail at provider validation, got {msg}"

@[test]
def testDiffEqRunLocalizesContactAndRecomputesSupport : IO Unit := do
  let result ← getResult
  let run := result.diffEqRun
  LeanTest.assertTrue (minimumCandidateDistance params initialState > 0.0)
    "Initial state should start above the ground before the event solver localizes impact"
  LeanTest.assertTrue (approx run.eventTime (impactTime params) 1.0e-4)
    s!"DiffEq event time should match the analytic free-fall impact time, got {run.eventTime}"
  LeanTest.assertTrue (approx (minimumCandidateDistance params run.eventState) 0.0 1.0e-5)
    s!"Localized event state should lie on the contact surface, got distance {minimumCandidateDistance params run.eventState}"
  LeanTest.assertEqual run.runtimeSupport.selectedIds #[2000, 2001, 2002, 2003]
    "Localized event state should regenerate the same stable bottom-rim support IDs"
  LeanTest.assertEqual run.contactForces.size params.supportBudget
    "The localized contact run should feed the retained support into the force kernel"
  LeanTest.assertEqual run.fullPhysics.support.totalCandidates 20
    "The localized full-physics solve should preserve the regenerated candidate count"
  LeanTest.assertTrue (run.derivative.vz > 0.0)
    s!"Post-event derivative should include upward contact acceleration, got {run.derivative.vz}"
  LeanTest.assertTrue (run.postStepState.vz > run.eventState.vz)
    "A physics step from the localized event state should make vertical velocity less negative"

@[test]
def testSustainedContactUsesCoupledNormalLcp : IO Unit := do
  let result ← getResult
  let solve := result.sustainedContact
  LeanTest.assertTrue
    (solve.runtimeSupport.selectedIds ==
      #[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009])
    s!"Resting sustained solve should consume the dynamically generated bottom-rim contacts, got {solve.runtimeSupport.selectedIds}"
  LeanTest.assertTrue (solve.runtimeSupport.exactness == MoveExactness.controlledApproximation)
    "Resting contacts are dynamically selected by penetration allowance before the exact LCP block"
  LeanTest.assertEqual solve.problem.massMatrix (massMatrix params)
    "Sustained solve should use the solid-cylinder mass matrix from Drake's inertia model"
  LeanTest.assertEqual solve.problem.normalJacobian.size params.contactsPerRim
    "Sustained solve should assemble one normal row per active rim sphere"

  LeanTest.assertTrue
    (solve.lcpResult.normalForces.all (fun f => f >= -1.0e-9))
    s!"Normal LCP forces should be nonnegative, got {solve.lcpResult.normalForces}"
  LeanTest.assertTrue
    (solve.lcpResult.normalMotionAfter.all (fun a => a >= -1.0e-7))
    s!"Post-contact normal accelerations should be nonnegative, got {solve.lcpResult.normalMotionAfter}"
  LeanTest.assertTrue (solve.lcpResult.solution.maxComplementarity < 1.0e-7)
    s!"Normal LCP should satisfy complementarity, got {solve.lcpResult.solution.maxComplementarity}"
  LeanTest.assertTrue
    (approx (sumNormalForces solve.contactForces) (params.mass * params.gravity) 1.0e-7)
    s!"Normal forces should balance the cylinder weight, got {sumNormalForces solve.contactForces}"
  LeanTest.assertTrue (approx solve.derivative.vz 0.0 1.0e-7)
    s!"Sustained contact acceleration should cancel gravity in z, got {solve.derivative.vz}"
  LeanTest.assertEqual solve.moves.size 2
    "Sustained solve should expose support aggregation plus local LCP elimination"
  LeanTest.assertTrue (solve.moves[0]!.kind == SkeletonMoveKind.branchAggregate)
    "First sustained move should aggregate dynamic rim contacts"
  LeanTest.assertTrue (solve.moves[1]!.kind == SkeletonMoveKind.localSchurBlock)
    "Second sustained move should eliminate the coupled contact LCP"

@[test]
def testTraceProjectsMulticontactBranchMoves : IO Unit := do
  let result ← getResult
  match result.trace.validate? with
  | .error msg => LeanTest.fail s!"Trace should validate: {msg}"
  | .ok () => pure ()

  LeanTest.assertEqual result.moves.size 5
    "Trace moves plus full-physics support and interval moves should be exposed"
  LeanTest.assertTrue (result.moves[0]!.kind == SkeletonMoveKind.intervalAdjoint)
    "First move should eliminate the free-flight interval"
  LeanTest.assertTrue (result.moves[1]!.kind == SkeletonMoveKind.checkpointBoundary)
    "Second move should retain the event boundary checkpoint"
  LeanTest.assertTrue (result.moves[2]!.kind == SkeletonMoveKind.branchAggregate)
    "Third move should aggregate retained contact candidates"
  LeanTest.assertTrue (result.moves[2]!.exactness == MoveExactness.controlledApproximation)
    "Closest-k contact branch should be marked as approximate"
  LeanTest.assertTrue (result.moves[3]!.kind == SkeletonMoveKind.markMarginalize)
    "The full-physics primitive should expose support selection as a mark move"
  LeanTest.assertTrue (result.moves[4]!.kind == SkeletonMoveKind.intervalAdjoint)
    "The full-physics primitive should expose the mass-matrix solve as an interval move"

  LeanTest.assertEqual result.branchData.children.size params.supportBudget
    "Branch payload should have one child per retained contact"
  LeanTest.assertTrue (result.branchResult.value > 0.0)
    s!"Expected positive contact branch value, got {result.branchResult.value}"

@[test]
def testDiffEqTraceRecordsLocalizedIntervalAndBranch : IO Unit := do
  let result ← getResult
  match result.diffEqRun.trace.validate? with
  | .error msg => LeanTest.fail s!"DiffEq trace should validate: {msg}"
  | .ok () => pure ()

  LeanTest.assertEqual result.diffEqRun.moves.size 5
    "Localized DiffEq trace plus full-physics support and interval moves should be exposed"
  match result.diffEqRun.trace.entries[0]! with
  | .interval segment =>
      LeanTest.assertTrue (segment.tAfter < segment.tAttempt)
        s!"Event-localized interval should end before the attempted horizon, got {reprStr segment}"
      LeanTest.assertTrue segment.madeJumpAfter
        "The localized interval should mark a post-event jump boundary"
  | other =>
      LeanTest.fail s!"Expected first DiffEq trace entry to be an interval, got {reprStr other}"
  LeanTest.assertTrue (result.diffEqRun.moves[0]!.kind == SkeletonMoveKind.intervalAdjoint)
    "First localized move should eliminate the DiffEq interval"
  LeanTest.assertTrue (result.diffEqRun.moves[2]!.kind == SkeletonMoveKind.branchAggregate)
    "Last localized move should aggregate regenerated contact candidates"
  LeanTest.assertTrue (result.diffEqRun.moves[3]!.kind == SkeletonMoveKind.markMarginalize)
    "Localized full physics should expose support selection as a mark move"
  LeanTest.assertTrue (result.diffEqRun.moves[4]!.kind == SkeletonMoveKind.intervalAdjoint)
    "Localized full physics should expose the mass-matrix solve as an interval move"

end Tests.EventSkeletonCylinderMulticontactExample
