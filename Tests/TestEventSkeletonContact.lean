import LeanTest
import Tyr.EventSkeleton

namespace Tests.EventSkeletonContact

open LeanTest
open Tyr.EventSkeleton

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

private def leftEndpoint : ContactCandidate :=
  {
    id := 10
    signedDistance := -0.001
    normalVelocity := -0.2
    tangentVelocity := 0.3
    normalJacobian := #[0.0, 1.0, -0.5]
    tangentJacobian := #[1.0, 0.0, 0.1]
    label := "left endpoint"
  }

private def rightEndpoint : ContactCandidate :=
  {
    id := 11
    signedDistance := 0.02
    normalVelocity := 0.0
    tangentVelocity := 0.0
    normalJacobian := #[0.0, 1.0, 0.5]
    tangentJacobian := #[1.0, 0.0, -0.1]
    label := "right endpoint"
  }

private def stickingPatch : ContactCandidate :=
  {
    id := 20
    signedDistance := 0.0
    normalVelocity := 0.0
    tangentVelocity := 1.0e-8
    normalJacobian := #[0.0, 1.0]
    tangentJacobian := #[1.0, 0.0]
    label := "sticking patch"
  }

private structure DynamicContactState where
  signedDistance : Float
  normalVelocity : Float
  tangentVelocity : Float := 0.0
  deriving Repr, Inhabited

private def dynamicCandidateProvider :
    ContactCandidateProvider DynamicContactState :=
  {
    label := "dynamic packed contacts"
    candidatesAt? := fun state =>
      .ok {
        candidates := #[
          { leftEndpoint with
            signedDistance := state.signedDistance
            normalVelocity := state.normalVelocity
            tangentVelocity := state.tangentVelocity },
          rightEndpoint
        ]
        sourceCandidateCount? := some 8
        label := "retained dynamic contacts"
      }
  }

@[test]
def testContactSupportSelectsRuntimeContactIds : IO Unit := do
  let support :=
    ContactSupport.selectByDistance 0.0 #[leftEndpoint, rightEndpoint]
      "rod endpoint threshold"
  let runtime ← assertOk support.toRuntimeSupport? "contact support runtime conversion"

  LeanTest.assertTrue (runtime.policy == SupportPolicy.threshold 0.0)
    "Distance-selected support should be tagged as threshold support"
  LeanTest.assertEqual runtime.selectedIds #[10]
    "Runtime support should expose stable contact candidate IDs, not local indices"
  LeanTest.assertEqual runtime.totalCandidates? (some 2)
    "Runtime support should record total candidate count"
  LeanTest.assertTrue (runtime.exactness == MoveExactness.controlledApproximation)
    "Threshold contact support should be a fixed-trace approximation"

@[test]
def testContactModeClassificationDistinguishesImpactStickSlide : IO Unit := do
  let classified :=
    { policy := .fullSupport
      candidates := #[leftEndpoint, rightEndpoint, stickingPatch]
      selectedLocalIndices := #[0, 2]
      label := "classification" : ContactSupport }
      |>.classifyCandidates 0.0 1.0e-6

  LeanTest.assertTrue (classified.candidates[0]!.mode == ContactMode.impacting)
    s!"Expected closing contact to be impacting, got {reprStr classified.candidates[0]!.mode}"
  LeanTest.assertTrue (classified.candidates[1]!.mode == ContactMode.separated)
    s!"Expected positive-distance contact to be separated, got {reprStr classified.candidates[1]!.mode}"
  LeanTest.assertTrue (classified.candidates[2]!.mode == ContactMode.sticking)
    s!"Expected near-zero tangent velocity to be sticking, got {reprStr classified.candidates[2]!.mode}"

@[test]
def testContactSupportValidatesJacobianWidthAndIndices : IO Unit := do
  let support :=
    ContactSupport.selectTopK 2 #[0, 1] #[leftEndpoint, rightEndpoint]
      "rod endpoint top-k"
  assertOk (support.validateJacobianWidth? 3) "contact Jacobian width"

  let runtime ← assertOk support.toRuntimeSupport? "top-k contact support conversion"
  LeanTest.assertEqual runtime.selectedIds #[10, 11]
    "Top-k contact support should preserve selected contact IDs in order"

  let badSupport : ContactSupport := {
    policy := .topK 1
    candidates := #[leftEndpoint]
    selectedLocalIndices := #[3]
    label := "bad contact support"
  }
  match badSupport.toRuntimeSupport? with
  | .ok _ => LeanTest.fail "Out-of-bounds selected contact index should fail"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "out of bounds")
        s!"Expected out-of-bounds diagnostic, got: {msg}"

  let badJacobian :=
    { leftEndpoint with tangentJacobian := #[1.0, 0.0] }
  match badJacobian.validateJacobianWidth? 3 with
  | .ok () => LeanTest.fail "Wrong-width tangent Jacobian should fail"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "tangent Jacobian width")
        s!"Expected tangent Jacobian diagnostic, got: {msg}"

@[test]
def testContactCandidateSetPreservesProviderDiagnosticsAndStableIds : IO Unit := do
  let set : ContactCandidateSet := {
    candidates := #[rightEndpoint, leftEndpoint]
    sourceCandidateCount? := some 5
    label := "packed broadphase contacts"
  }
  assertOk (set.validate? (some 3)) "contact candidate set validation"
  LeanTest.assertEqual set.totalCandidates 5
    "Candidate set should preserve the provider's source candidate count"
  LeanTest.assertTrue (set.minimumSignedDistance? == some leftEndpoint.signedDistance)
    s!"Candidate set should expose minimum signed distance, got {reprStr set.minimumSignedDistance?}"

  let support := set.selectWithPolicy (.deterministicPick leftEndpoint.id)
    "stable id selection"
  LeanTest.assertEqual support.selectedLocalIndices #[1]
    "Deterministic support selection should resolve stable ids into local candidate indices"
  LeanTest.assertEqual support.totalCandidates 5
    "Support should retain provider-level source candidate diagnostics"
  let runtime ← assertOk support.toRuntimeSupport? "candidate-set runtime support"
  LeanTest.assertEqual runtime.selectedIds #[leftEndpoint.id]
    "Runtime support should publish the stable candidate id selected from the set"
  LeanTest.assertEqual runtime.totalCandidates? (some 5)
    "Runtime support should keep the provider source count"

  let badCount : ContactCandidateSet :=
    { set with sourceCandidateCount? := some 1 }
  match badCount.validate? with
  | .ok () => LeanTest.fail "Source candidate count below retained candidates should fail"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "source candidate count")
        s!"Expected source-count diagnostic, got: {msg}"

  let duplicateIds : ContactCandidateSet :=
    { set with
      candidates := #[leftEndpoint, { rightEndpoint with id := leftEndpoint.id }]
      sourceCandidateCount? := some 2 }
  match duplicateIds.validate? with
  | .ok () => LeanTest.fail "Duplicate stable contact ids should fail"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "duplicate candidate id")
        s!"Expected duplicate-id diagnostic, got: {msg}"

@[test]
def testContactCandidateProviderRecomputesDynamicViewsAndSupport :
    IO Unit := do
  let active : DynamicContactState := {
    signedDistance := -0.002
    normalVelocity := -0.4
    tangentVelocity := 0.1
  }
  let activeSet ← assertOk
    (dynamicCandidateProvider.validCandidatesAt? active (some 3))
    "active contact provider"
  LeanTest.assertEqual activeSet.totalCandidates 8
    "Provider diagnostics should preserve the broadphase source count"
  LeanTest.assertTrue
    (approx activeSet.candidates[0]!.signedDistance (-0.002) 1.0e-12)
    s!"Provider should recompute signed distance from state, got {activeSet.candidates[0]!.signedDistance}"

  let support ← assertOk
    (dynamicCandidateProvider.supportAt? active (.threshold 0.0)
      0.0 1.0e-6 (some 3) "state support")
    "active provider support"
  LeanTest.assertEqual support.selectedLocalIndices #[0]
    "Threshold support should select the active penetrating contact"
  LeanTest.assertTrue (support.candidates[0]!.mode == ContactMode.impacting)
    "Provider support should classify the current candidate mode"

  let runtime ← assertOk
    (dynamicCandidateProvider.runtimeSupportAt? active (.threshold 0.0)
      0.0 1.0e-6 (some 3) "state support")
    "active provider runtime support"
  LeanTest.assertEqual runtime.selectedIds #[leftEndpoint.id]
    "Runtime support should expose stable ids from provider-generated candidates"
  LeanTest.assertEqual runtime.totalCandidates? (some 8)
    "Runtime support should preserve the provider source count"

  let separated : DynamicContactState := {
    signedDistance := 0.25
    normalVelocity := 0.0
  }
  let separatedSupport ← assertOk
    (dynamicCandidateProvider.supportAt? separated (.threshold 0.0)
      0.0 1.0e-6 (some 3) "separated support")
    "separated provider support"
  LeanTest.assertEqual separatedSupport.selectedLocalIndices #[]
    "Changing state should recompute support and drop separated contacts"

  let badProvider : ContactCandidateProvider DynamicContactState := {
    label := "bad dynamic provider"
    candidatesAt? := fun _ =>
      .ok {
        candidates := #[
          { leftEndpoint with tangentJacobian := #[1.0, 0.0] }
        ]
        sourceCandidateCount? := some 1
        label := "bad retained contacts"
      }
  }
  let msg ← assertError
    (badProvider.validCandidatesAt? active (some 3))
    "bad provider validation"
  LeanTest.assertTrue (msg.contains "tangent Jacobian width")
    s!"Expected provider boundary validation to catch bad Jacobian width, got {msg}"

@[test]
def testContactPrimitivesAssembleConstraintRowsAndGeneralizedForces : IO Unit := do
  let support :=
    ContactSupport.selectTopK 2 #[0, 1] #[leftEndpoint, rightEndpoint]
      "primitive support"
  let normalRows ← assertOk
    (support.constraintJacobianRows? false)
    "normal-only constraint rows"
  LeanTest.assertEqual normalRows #[leftEndpoint.normalJacobian, rightEndpoint.normalJacobian]
    "Normal-only projection should expose one Jacobian row per selected candidate"

  let fullRows ← assertOk
    (support.constraintJacobianRows? true)
    "normal-and-tangent constraint rows"
  LeanTest.assertEqual fullRows
    #[leftEndpoint.normalJacobian, leftEndpoint.tangentJacobian,
      rightEndpoint.normalJacobian, rightEndpoint.tangentJacobian]
    "Sticking projection should expose normal and tangent rows in support order"

  let scalarForce := ContactForceScalars.fromCandidate leftEndpoint 7.0 (-2.0)
  let generalized := scalarForce.generalizedForce leftEndpoint
  LeanTest.assertEqual generalized #[-2.0, 7.0, -3.7]
    "J^T f should assemble normal and tangent force components in generalized coordinates"

  let total := sumGeneralizedForces #[
    generalized,
    rightEndpoint.generalizedForce 1.0 3.0
  ]
  LeanTest.assertEqual total #[1.0, 8.0, -3.5]
    "Generalized contact forces should sum coordinate-wise with zero-padding semantics"

end Tests.EventSkeletonContact
