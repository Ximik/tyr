import LeanTest
import Tyr.Mctx.Batched

namespace Tests.MctxSearchIntegrity

open torch.mctx

private def assertApprox (actual expected : Float) (label : String) : IO Unit :=
  LeanTest.assertTrue (actual.isFinite && Float.abs (actual - expected) < 1e-10)
    s!"{label}: expected {expected}, got {actual}"

private def first : RootActionSelectionFn Nat Unit := fun _ _ _ => 0
private def firstInterior : InteriorActionSelectionFn Nat Unit := fun _ _ _ _ => 0

private def root (value : Float := 0.0) : RootFnOutput Nat :=
  { priorLogits := #[0.0], value := value, embedding := 0 }

private def assertAllocationInvariant (tree : Tree Nat Unit) : IO Unit := do
  LeanTest.assertTrue (tree.numAllocated ≤ tree.nodeVisits.size) "Allocation respects capacity"
  LeanTest.assertEqual tree.nextNodeIndex tree.numAllocated "Next slot follows the allocation counter"
  for i in [:tree.nodeVisits.size] do
    if i < tree.numAllocated then
      LeanTest.assertTrue (tree.nodeVisits[i]! > 0) "Allocated prefix contains no unused slots"
    else
      LeanTest.assertEqual tree.nodeVisits[i]! 0 "Unallocated suffix has zero visits"

-- The three full-edge evaluations have different rewards, discounts and
-- continuation values. Their complete returns are 6, 8 and 5 respectively.
private def varyingStep (key : UInt64) : RecurrentFnOutput :=
  let (reward, discount, value) :=
    (#[(1.0, 0.5, 10.0), (-2.0, 2.0, 5.0), (7.0, -0.25, 8.0)] :
      Array (Float × Float × Float)).getD ((key.toNat - 1) % 3) (0.0, 0.0, 0.0)
  { reward := reward, discount := discount, value := value, priorLogits := #[0.0] }

private def varying : RecurrentFn Unit Nat := fun _ key _ state =>
  (varyingStep key, state + 1)

private def varyingBatched : BatchedRecurrentFn Unit Nat := fun _ key _ states =>
  let step := varyingStep key
  ({ reward := states.map (fun _ => step.reward),
     discount := states.map (fun _ => step.discount),
     value := states.map (fun _ => step.value),
     priorLogits := states.map (fun _ => step.priorLogits) }, states.map (· + 1))

@[test] def testMctxRootOnlyCapacityBacksUpEveryReturn : IO Unit := do
  for requestedCapacity in #[0, 1] do
    let initial := instantiateTreeFromRootWithCapacity (root 3.0) requestedCapacity #[false] ()
    LeanTest.assertEqual initial.nodeVisits.size 1 "A root slot is retained even at requested capacity zero"
    let tree := searchWithTree () 0 initial varying first firstInterior 3 (some 8)
    LeanTest.assertEqual tree.nodeVisits #[4] "Every simulation increments root visits"
    LeanTest.assertEqual tree.childrenVisits #[#[3]] "Every transient edge evaluation counts"
    LeanTest.assertEqual tree.childrenIndex #[#[none]] "Capacity exhaustion does not allocate a child"
    assertApprox tree.summary.value 5.5 "Root includes initial value and returns 6,8,5"
    assertApprox (tree.qvalues 0)[0]! (19.0 / 3.0) "Transient Q averages complete returns"
    assertAllocationInvariant tree
  let batch := searchBatchedWithTrees () 0
    #[instantiateTreeFromRootWithCapacity (root 3.0) 0 #[false] (),
      instantiateTreeFromRootWithCapacity (root 3.0) 1 #[false] ()]
    varyingBatched first firstInterior 3
  for tree in batch.trees do
    LeanTest.assertEqual tree.nodeVisits #[4] "Batched root-only search counts every simulation"
    LeanTest.assertEqual tree.childrenVisits #[#[3]] "Batched transient edge visits"
    assertApprox tree.summary.value 5.5 "Batched transient root return"
    assertApprox (tree.qvalues 0)[0]! (19.0 / 3.0) "Batched transient Q"
    assertAllocationInvariant tree

@[test] def testMctxFullTreeRetainsUnallocatedRootAction : IO Unit := do
  let initial := instantiateTreeFromRootWithCapacity
    ({ priorLogits := #[0.0, 0.0], value := 0.0, embedding := 0 } : RootFnOutput Nat)
    2 #[false, false] ()
  let recurrent : RecurrentFn Unit Nat := fun _ _ action state =>
    ({ reward := if action == 0 then 1.0 else 100.0, discount := 0.5,
       value := 10.0, priorLogits := #[0.0, 0.0] }, state + 1)
  let select : RootActionSelectionFn Nat Unit := fun _ tree _ =>
    if tree.nodeVisits[0]! == 1 then 0 else 1
  let tree := searchWithTree () 0 initial recurrent select firstInterior 4 (some 1)
  LeanTest.assertEqual tree.nodeVisits #[5, 1] "Full capacity must not discard requested simulations"
  LeanTest.assertEqual tree.childrenVisits[0]! #[1, 3] "Both root actions retain their evaluations"
  LeanTest.assertEqual tree.childrenIndex[0]! #[some 1, none] "Hard capacity is preserved"
  assertApprox (tree.qvalues 0)[0]! 6.0 "Allocated action Q"
  assertApprox (tree.qvalues 0)[1]! 105.0 "Unallocated action includes its continuation value"
  assertApprox tree.summary.value 64.2 "All four returns enter the root mean"
  assertAllocationInvariant tree

private def nestedRecurrent : RecurrentFn Unit Nat := fun _ key _ state =>
  if state == 0 then
    ({ reward := 2.0, discount := 0.5, value := 10.0, priorLogits := #[0.0] }, 1)
  else
    (varyingStep key, state + 1)

private def nestedBatched : BatchedRecurrentFn Unit Nat := fun _ key _ states =>
  let outputs := states.map fun state => nestedRecurrent () key 0 state
  ({ reward := outputs.map (·.1.reward), discount := outputs.map (·.1.discount),
     value := outputs.map (·.1.value), priorLogits := outputs.map (·.1.priorLogits) }, outputs.map (·.2))

private def initialFullTree : Tree Nat Unit :=
  searchWithTree () 0 (instantiateTreeFromRootWithCapacity (root 3.0) 2 #[false] ())
    nestedRecurrent first firstInterior 1 (some 3)

private def assertNestedBackup (tree : Tree Nat Unit) : IO Unit := do
  LeanTest.assertEqual tree.nodeVisits #[5, 4] "Every ancestor receives each transient rollout"
  LeanTest.assertEqual tree.childrenVisits #[#[4], #[3]] "Ancestor and transient edge counts"
  LeanTest.assertEqual tree.childrenIndex #[#[some 1], #[none]] "No allocation beyond capacity"
  assertApprox tree.nodeValues[1]! 7.25 "Child mean includes its initial 10 and returns 6,8,5"
  -- Root rollout returns are 7 initially, followed by 5,6,4.5. Passing the
  -- child's running mean upstream instead of each raw return gives a different answer.
  assertApprox tree.summary.value 5.1 "Root backs up raw returns through reward and discount"
  assertApprox (tree.qvalues 1)[0]! (19.0 / 3.0) "Transient edge averages varying complete returns"
  assertApprox (tree.qvalues 0)[0]! 5.625 "Allocated ancestor edge tracks its child's mean"
  assertAllocationInvariant tree

@[test] def testMctxFullNonrootEdgeBacksUpRawReturns : IO Unit := do
  let tree := searchWithTree () 0 initialFullTree nestedRecurrent first firstInterior 3 (some 3)
  assertNestedBackup tree
  let batch := searchBatchedWithTrees () 0 #[initialFullTree] nestedBatched first firstInterior 3 (some 3)
  assertNestedBackup batch.trees[0]!

@[test] def testMctxTransientReturnTraversesMultipleAncestors : IO Unit := do
  let recurrent : RecurrentFn Unit Nat := fun _ key _ state =>
    if state == 0 then
      ({ reward := 1.0, discount := 0.5, value := 4.0, priorLogits := #[0.0] }, 1)
    else if state == 1 then
      ({ reward := 2.0, discount := 0.25, value := 8.0, priorLogits := #[0.0] }, 2)
    else
      (varyingStep key, state + 1)
  let initial := searchWithTree () 0
    (instantiateTreeFromRootWithCapacity (root 0.0) 3 #[false] ())
    recurrent first firstInterior 2 (some 5)
  let tree := searchWithTree () 0 initial recurrent first firstInterior 3 (some 5)
  LeanTest.assertEqual tree.nodeVisits #[6, 5, 4] "Transient rollouts visit the whole ancestor chain"
  LeanTest.assertEqual tree.childrenVisits #[#[5], #[4], #[3]] "All ancestor edges count each rollout"
  assertApprox tree.nodeValues[2]! 6.75 "Deepest node averages its initial value and transient returns"
  assertApprox tree.nodeValues[1]! 3.75 "Middle ancestor applies its own reward and discount"
  assertApprox tree.nodeValues[0]! (14.375 / 6.0) "Root receives each twice-discounted raw return"
  assertApprox (tree.qvalues 2)[0]! (19.0 / 3.0) "Deep transient Q"
  assertApprox (tree.qvalues 1)[0]! 3.6875 "Middle Q follows the deepest node mean"
  assertApprox (tree.qvalues 0)[0]! 2.875 "Root Q follows the middle node mean"
  assertAllocationInvariant tree

@[test] def testMctxTransientEdgeCanAllocateAfterReroot : IO Unit := do
  let full := searchWithTree () 0 initialFullTree nestedRecurrent first firstInterior 3 (some 3)
  let rerooted := getSubtree full 0
  LeanTest.assertEqual rerooted.numAllocated 1 "Rerooting frees the discarded ancestor's slot"
  LeanTest.assertEqual rerooted.embeddings[0]! 1 "Selected child becomes the new root"
  assertApprox (rerooted.qvalues 0)[0]! (19.0 / 3.0) "Reroot preserves transient return statistics"
  let tree := searchWithTree () 0 rerooted nestedRecurrent first firstInterior 1 (some 3)
  LeanTest.assertEqual tree.childrenIndex[0]! #[some 1] "Transient edge now receives a real node"
  LeanTest.assertEqual tree.nodeVisits #[5, 1] "New child starts with one model evaluation"
  LeanTest.assertEqual tree.childrenVisits[0]! #[4] "Historical edge visits remain counted"
  assertApprox tree.rawValues[1]! 10.0 "New child raw continuation value"
  assertApprox tree.childrenValues[0]![0]! 10.0 "Old complete-return mean is not reused as continuation V"
  assertApprox (tree.qvalues 0)[0]! 6.0 "Allocated Q uses reward plus discounted fresh continuation"
  assertApprox tree.summary.value 7.0 "Root mean retains earlier returns and adds the new return"
  assertAllocationInvariant tree

@[test] def testMctxAllocatedDepthCutoffRetainsModelReevaluation : IO Unit := do
  let tree := searchWithTree () 0
    (instantiateTreeFromRootWithCapacity (root 3.0) 4 #[false] ()) varying first firstInterior 3 (some 1)
  LeanTest.assertEqual tree.numAllocated 2 "Re-evaluating a cutoff node allocates no extra slots"
  LeanTest.assertEqual tree.nodeVisits #[4, 3, 0, 0] "Repeated model evaluation increments cutoff visits"
  assertApprox tree.nodeValues[1]! 8.0 "Allocated cutoff retains upstream latest model value semantics"
  assertApprox (tree.qvalues 0)[0]! 5.0 "Allocated cutoff uses the latest evaluated transition"
  assertApprox tree.summary.value 5.5 "Root still averages each complete rollout"
  assertAllocationInvariant tree

private def chain : RecurrentFn Unit Nat := fun _ _ _ state =>
  ({ reward := Float.ofNat (state + 1), discount := 1.0, value := 0.0,
     priorLogits := #[0.0] }, state + 1)

private def chainBatched : BatchedRecurrentFn Unit Nat := fun _ _ _ states =>
  ({ reward := states.map (fun state => Float.ofNat (state + 1)),
     discount := states.map (fun _ => 1.0), value := states.map (fun _ => 0.0),
     priorLogits := states.map (fun _ => #[0.0]) }, states.map (· + 1))

private def assertSameTree (actual expected : Tree Nat Unit) : IO Unit := do
  LeanTest.assertEqual actual.numAllocated expected.numAllocated "Batched allocation matches independent search"
  LeanTest.assertEqual actual.nodeVisits expected.nodeVisits "Batched node visits match independent search"
  LeanTest.assertEqual actual.childrenVisits expected.childrenVisits "Batched edge visits match independent search"
  LeanTest.assertEqual actual.childrenIndex expected.childrenIndex "Batched topology matches independent search"
  LeanTest.assertEqual actual.nodeValues expected.nodeValues "Batched values match independent search"
  LeanTest.assertEqual actual.childrenValues expected.childrenValues "Batched edge values match independent search"
  LeanTest.assertEqual actual.embeddings expected.embeddings "Batched embeddings match independent search"

@[test] def testMctxBatchedDepthDefaultsArePerTree : IO Unit := do
  let small := instantiateTreeFromRootWithCapacity (root 0.0) 2 #[false] ()
  let large := instantiateTreeFromRootWithCapacity (root 0.0) 5 #[false] ()
  for depth in #[none, some 1, some 3] do
    let expectedSmall := searchWithTree () 0 small chain first firstInterior 3 depth
    let expectedLarge := searchWithTree () 0 large chain first firstInterior 3 depth
    let forward := searchBatchedWithTrees () 0 #[small, large] chainBatched first firstInterior 3 depth
    let reverse := searchBatchedWithTrees () 0 #[large, small] chainBatched first firstInterior 3 depth
    assertSameTree forward.trees[0]! expectedSmall
    assertSameTree forward.trees[1]! expectedLarge
    assertSameTree reverse.trees[0]! expectedLarge
    assertSameTree reverse.trees[1]! expectedSmall
    if depth.isNone then
      assertApprox expectedSmall.summary.value 0.75 "Small tree's default depth is one"
      assertApprox expectedLarge.summary.value 2.5 "Large tree's default permits three steps"

@[test] def testMctxAllocationCounterResetAndSubtree : IO Unit := do
  let initial := instantiateTreeFromRootWithCapacity (root 0.0) 5 #[false] ()
  LeanTest.assertEqual initial.numAllocated 1 "Constructor counts the root"
  let tree := searchWithTree () 0 initial chain first firstInterior 3
  LeanTest.assertEqual tree.numAllocated 4 "Each new expansion allocates one node"
  assertAllocationInvariant tree
  let subtree := getSubtree tree 0
  LeanTest.assertEqual subtree.numAllocated 3 "Subtree allocation equals retained node count"
  LeanTest.assertEqual subtree.parents[0]! none "Extracted root has no parent"
  assertAllocationInvariant subtree
  let reset := resetSearchTree tree
  LeanTest.assertEqual reset.numAllocated 0 "Reset clears the allocation counter"
  LeanTest.assertEqual reset.nodeVisits.size 5 "Reset preserves capacity"
  assertAllocationInvariant reset
  let restarted := updateTreeWithRoot reset (root 7.0) #[false] ()
  LeanTest.assertEqual restarted.numAllocated 1 "Reinitializing a reset tree counts the root"
  assertApprox restarted.summary.value 7.0 "Reinitialization restores root value"
  assertAllocationInvariant restarted
  let missing := getSubtree tree 99
  LeanTest.assertEqual missing.numAllocated 0 "An absent child returns a reset tree"
  assertAllocationInvariant missing
  let empty : Tree Nat Unit := default
  let recovered := updateTreeWithRoot empty (root 7.0) #[false] ()
  LeanTest.assertEqual recovered.nodeVisits.size 1 "Updating an empty tree supplies a root slot"
  assertAllocationInvariant recovered

def run : IO Unit := do
  testMctxRootOnlyCapacityBacksUpEveryReturn
  testMctxFullTreeRetainsUnallocatedRootAction
  testMctxFullNonrootEdgeBacksUpRawReturns
  testMctxTransientReturnTraversesMultipleAncestors
  testMctxTransientEdgeCanAllocateAfterReroot
  testMctxAllocatedDepthCutoffRetainsModelReevaluation
  testMctxBatchedDepthDefaultsArePerTree
  testMctxAllocationCounterResetAndSubtree

end Tests.MctxSearchIntegrity
