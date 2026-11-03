import LeanTest
import Tyr.MctxDag

open torch.mctxdag

private def approx (a b : Float) (tol : Float := 1e-6) : Bool :=
  Float.abs (a - b) < tol

private def deepestLeaf [BEq K] [Hashable K] (tree : DagTree S K E) : Nat × Nat := Id.run do
  let mut stack : Array (Nat × Nat) := #[(ROOT_INDEX, 0)]
  let mut bestNode := ROOT_INDEX
  let mut bestDepth := 0
  let mut bestVisits := tree.nodeVisits.getD ROOT_INDEX 0

  while !stack.isEmpty do
    let some (node, depth) := stack.back? | break
    stack := stack.pop

    let mut hasChild := false
    for a in [:tree.numActions] do
      if let some child := (tree.childrenIndex.getD node #[]).getD a none then
        hasChild := true
        stack := stack.push (child, depth + 1)

    if !hasChild then
      let visits := tree.nodeVisits.getD node 0
      if depth > bestDepth || (depth = bestDepth && visits > bestVisits) then
        bestNode := node
        bestDepth := depth
        bestVisits := visits

  return (bestNode, bestDepth)

private def chooseLeastVisitedAction [BEq K] [Hashable K]
    (tree : DagTree S K E) (nodeIndex : Nat) : Nat :=
  let visits := tree.childrenVisits.getD nodeIndex #[]
  if visits.isEmpty then
    0
  else
    let init : Nat × Nat := (0, visits.getD 0 0)
    let (best, _) := (List.range visits.size).foldl (init := init) fun (acc : Nat × Nat) a =>
      let c := visits.getD a acc.2
      if c < acc.2 then (a, c) else acc
    best

/-- Structure every searched, rerooted or reset DAG must have: edges point at
    allocated nodes and have been visited, rows past `numAllocated` are empty,
    and every allocated node is reachable from the root. -/
private def assertDagWellFormed [BEq K] [Hashable K] (tree : DagTree S K E) (label : String) :
    IO Unit := do
  let n := tree.numAllocated
  LeanTest.assertTrue (n ≤ tree.capacity) s!"{label}: allocation within capacity"
  for i in [:tree.capacity] do
    let row := tree.childrenIndex.getD i #[]
    let visits := tree.childrenVisits.getD i #[]
    LeanTest.assertEqual row.size tree.numActions s!"{label}: node {i} has one edge per action"
    for a in [:row.size] do
      if i ≥ n then
        LeanTest.assertEqual row[a]! none s!"{label}: unallocated node {i} has no edges"
        LeanTest.assertEqual (visits.getD a 1) 0 s!"{label}: unallocated node {i} has no edge visits"
      else if let some child := row[a]! then
        LeanTest.assertTrue (child < n)
          s!"{label}: edge {i}/{a} points at an allocated node, got {child}"
        LeanTest.assertTrue (visits.getD a 0 > 0) s!"{label}: allocated edge {i}/{a} was visited"
    if i ≥ n then
      LeanTest.assertEqual (tree.nodeVisits.getD i 1) 0 s!"{label}: unallocated node {i} has no visits"
  if n > 0 then
    let mut seen := (Array.replicate n false).set! ROOT_INDEX true
    let mut stack := #[ROOT_INDEX]
    while !stack.isEmpty do
      let some node := stack.back? | break
      stack := stack.pop
      for child in tree.childrenIndex.getD node #[] do
        if let some c := child then
          if !seen.getD c true then
            seen := seen.set! c true
            stack := stack.push c
    LeanTest.assertTrue (seen.all id) s!"{label}: every allocated node is reachable from the root"

@[test]
def testMctxDagTranspositionReuse : IO Unit := do
  let root : RootFnOutput UInt64 := {
    priorLogits := #[0.0, 0.0]
    value := 0.0
    embedding := 1
  }

  let recurrent : RecurrentFn Unit UInt64 := fun _ _ action _ =>
    let nextEmb : UInt64 := if action = 0 then 777 else 777
    ({ reward := 0.0, discount := 1.0, priorLogits := #[0.0, 0.0], value := 0.0 }, nextEmb)

  let rootFn : RootActionSelectionFn UInt64 UInt64 Unit := fun _ tree nodeIndex =>
    chooseLeastVisitedAction tree nodeIndex
  let interiorFn : InteriorActionSelectionFn UInt64 UInt64 Unit := fun _ tree nodeIndex _ =>
    chooseLeastVisitedAction tree nodeIndex

  let tree0 := instantiateDagTreeFromRoot root root.embedding 4 #[false, false] ()
  let tree := searchWithDag
    (params := ())
    (rngKey := 0)
    (tree := tree0)
    (recurrentFn := recurrent)
    (keyFn := id)
    (rootActionSelectionFn := rootFn)
    (interiorActionSelectionFn := interiorFn)
    (numSimulations := 2)

  LeanTest.assertEqual tree.numAllocated 2
    "Two root actions reaching same key should share one child node"

  let c0 := (tree.childrenIndex.getD ROOT_INDEX #[]).getD 0 none
  let c1 := (tree.childrenIndex.getD ROOT_INDEX #[]).getD 1 none
  LeanTest.assertTrue (c0 = c1 && c0 = some 1)
    s!"Expected both root actions to point to shared node 1, got ({c0}, {c1})"
  assertDagWellFormed tree "transposition"

private def assertKeyMap (tree : DagTree Nat Nat Unit) : IO Unit := do
  LeanTest.assertEqual tree.keyToNode.size tree.numAllocated "one key per allocated DAG node"
  for i in [:tree.numAllocated] do
    LeanTest.assertEqual tree.keyToNode[tree.keys[i]!]? (some i)
      s!"DAG key map must identify node {i}"

@[test]
def testMctxDagWellFormedCheckRejectsBrokenTrees : IO Unit := do
  let root : RootFnOutput Nat := { priorLogits := #[0.0, 0.0], value := 0.0, embedding := 0 }
  let recurrent : RecurrentFn Unit Nat := fun _ _ action embedding =>
    ({ reward := 0.0, discount := 1.0, priorLogits := #[0.0, 0.0], value := 0.0 },
      embedding * 2 + action + 1)
  let tree := searchDag () 0 root recurrent id (fun _ _ _ => 0) (fun _ _ _ _ => 0) 2
  assertDagWellFormed tree "valid search"
  let rejects := fun (broken : DagTree Nat Nat Unit) => do
    try
      assertDagWellFormed broken "broken"
      pure false
    catch _ => pure true
  let dangling := { tree with childrenIndex := tree.childrenIndex.set! 0 #[some tree.capacity, none] }
  let unvisited := { tree with childrenVisits := tree.childrenVisits.set! 0 #[0, 0] }
  let orphan := { tree with childrenIndex := tree.childrenIndex.set! 0 #[none, none] }
  LeanTest.assertTrue (← rejects dangling) "an edge past numAllocated is rejected"
  LeanTest.assertTrue (← rejects unvisited) "an allocated edge without visits is rejected"
  LeanTest.assertTrue (← rejects orphan) "an allocated node unreachable from the root is rejected"

@[test]
def testMctxDagRerootEarlierTransposition : IO Unit := do
  let root : RootFnOutput Nat := { priorLogits := #[0.0, 0.0], value := 0.0, embedding := 0 }
  let recurrent : RecurrentFn Unit Nat := fun _ _ action embedding =>
    let next := if embedding == 0 then (if action == 0 then 1 else 2)
      else if embedding == 2 then 1 else 3
    ({ reward := 0.0, discount := 1.0, priorLogits := #[0.0, 0.0], value := Float.ofNat next }, next)
  let rootFn : RootActionSelectionFn Nat Nat Unit := fun _ t _ =>
    if t.nodeVisits[0]! == 1 then 0 else 1
  let interiorFn : InteriorActionSelectionFn Nat Nat Unit := fun _ _ _ _ => 0
  -- A is allocated before B, but B reaches A: root -> A, root -> B -> A.
  let tree := searchDag () 0 root recurrent id rootFn interiorFn 3 (maxDepth := some 3)
  let carried := getSubtree tree 1
  LeanTest.assertEqual carried.keys[0]! 2 "the selected child B must become the root"
  LeanTest.assertEqual carried.embeddings[0]! 2 "reroot must retain B's embedding"
  LeanTest.assertEqual carried.childrenIndex[0]![0]! (some 1) "B must retain its edge to A"
  LeanTest.assertEqual carried.keys[1]! 1 "the earlier transposition A must be retained"
  LeanTest.assertEqual carried.numAllocated 2 "discard the unreachable previous root"
  assertKeyMap carried
  assertDagWellFormed tree "reroot-earlier search"
  assertDagWellFormed carried "reroot-earlier carried"
  let refreshed := updateDagTreeWithRoot carried { root with embedding := 2, value := 2.0 }
    2 #[false, false] ()
  let resumed := searchWithDag () 0 refreshed recurrent id (fun _ _ _ => 0) interiorFn 1
    (maxDepth := some 3)
  LeanTest.assertEqual resumed.keys[0]! 2 "continued search must stay rooted at B"
  LeanTest.assertEqual resumed.nodeVisits[0]! (carried.nodeVisits[0]! + 1)
  LeanTest.assertEqual resumed.childrenIndex[1]![0]! (some 2) "continued search expands A's child"
  LeanTest.assertEqual resumed.keys[2]! 3
  assertKeyMap resumed
  assertDagWellFormed resumed "reroot-earlier resumed"

@[test]
def testMctxDagRerootCycleReachingPreviousRoot : IO Unit := do
  let root : RootFnOutput Nat := { priorLogits := #[0.0], value := 0.0, embedding := 0 }
  let recurrent : RecurrentFn Unit Nat := fun _ _ _ embedding =>
    let next := if embedding == 0 then 1 else 0
    ({ reward := 0.0, discount := 1.0, priorLogits := #[0.0], value := Float.ofNat next }, next)
  let rootFn : RootActionSelectionFn Nat Nat Unit := fun _ _ _ => 0
  let interiorFn : InteriorActionSelectionFn Nat Nat Unit := fun _ _ _ _ => 0
  let tree := searchDag () 0 root recurrent id rootFn interiorFn 2 (maxDepth := some 2)
  let carried := getSubtree tree 0
  LeanTest.assertEqual carried.keys[0]! 1 "the selected child stays first even in a cycle"
  LeanTest.assertEqual carried.keys[1]! 0 "the reachable previous root must be retained"
  LeanTest.assertEqual carried.childrenIndex[0]![0]! (some 1)
  LeanTest.assertEqual carried.childrenIndex[1]![0]! (some 0) "cycle back edge must be remapped"
  LeanTest.assertEqual carried.nodeVisits[0]! tree.nodeVisits[1]!
  assertKeyMap carried
  assertDagWellFormed tree "cycle search"
  assertDagWellFormed carried "cycle carried"
  let refreshed := updateDagTreeWithRoot carried { root with embedding := 1, value := 1.0 }
    1 #[false] ()
  let resumed := searchWithDag () 0 refreshed recurrent id rootFn interiorFn 1 (maxDepth := some 1)
  LeanTest.assertEqual resumed.keys[0]! 1
  LeanTest.assertEqual resumed.numAllocated 2 "following a cycle must not allocate duplicate keys"
  LeanTest.assertEqual resumed.nodeVisits[0]! (carried.nodeVisits[0]! + 1)
  assertKeyMap resumed
  assertDagWellFormed resumed "cycle resumed"

private def capacityRoot : RootFnOutput Nat := {
  priorLogits := #[0.0], value := 0.0, embedding := 0
}

private def capacityRecurrent : RecurrentFn Unit Nat := fun _ key _ embedding =>
  if embedding == 0 then
    ({ reward := 1.0, discount := 0.5, priorLogits := #[0.0], value := 2.0 }, 1)
  else if key % 2 == 0 then
    ({ reward := 3.0, discount := 0.25, priorLogits := #[0.0], value := 8.0 }, 2)
  else
    ({ reward := 5.0, discount := 0.5, priorLogits := #[0.0], value := 10.0 }, 2)

private def runCapacitySearch (tree : DagTree Nat Nat Unit) (key : UInt64) (simulations : Nat) :=
  searchWithDag () key tree capacityRecurrent id (fun _ _ _ => 0) (fun _ _ _ _ => 0)
    simulations (maxDepth := some 3)

@[test]
def testMctxDagRootOnlyCapacityBacksUpEvaluatedReturns : IO Unit := do
  for requestedCapacity in #[0, 1] do
    let tree := instantiateDagTreeFromRootWithCapacity capacityRoot 0 requestedCapacity #[false] ()
    let searched := runCapacitySearch tree 0 3
    LeanTest.assertEqual searched.capacity 1 "zero requested capacity must retain one root slot"
    LeanTest.assertEqual searched.numAllocated 1 "the hard node capacity must be preserved"
    LeanTest.assertEqual searched.nodeVisits[0]! 4 "every evaluated rollout visits the root"
    LeanTest.assertEqual searched.childrenVisits[0]![0]! 3
    LeanTest.assertEqual searched.childrenIndex[0]![0]! none "no child exists at capacity"
    LeanTest.assertTrue (approx searched.nodeValues[0]! 1.5) "root mean includes three returns of 2"
    LeanTest.assertTrue (approx (searched.qvalues 0)[0]! 2.0) "Q must use evaluated r + discount * V"
    LeanTest.assertEqual searched.rawValues[0]! 0.0 "transient leaves must not overwrite root raw value"
    assertKeyMap searched
    assertDagWellFormed searched s!"root-only capacity {requestedCapacity}"

@[test]
def testMctxDagFullNonrootCapacityBacksUpVaryingReturns : IO Unit := do
  let tree := instantiateDagTreeFromRootWithCapacity capacityRoot 0 2 #[false] ()
  let searched := runCapacitySearch tree 0 3
  LeanTest.assertEqual searched.capacity 2
  LeanTest.assertEqual searched.numAllocated 2
  LeanTest.assertEqual searched.nodeVisits #[4, 3] "root and full nonroot each receive their rollouts"
  LeanTest.assertEqual searched.childrenVisits #[#[3], #[2]]
  LeanTest.assertEqual searched.childrenIndex #[#[some 1], #[none]]
  -- The transient returns at node 1 are 3 + .25*8 = 5 and 5 + .5*10 = 10.
  LeanTest.assertTrue (approx (searched.qvalues 1)[0]! 7.5) "transient Q averages full returns"
  LeanTest.assertTrue (approx searched.nodeValues[1]! (17.0 / 3.0)) "nonroot mean includes its initial value 2"
  -- Root rollouts are 2, 1 + .5*5 = 3.5, and 1 + .5*10 = 6.
  LeanTest.assertTrue (approx searched.nodeValues[0]! 2.875) "ancestors receive each raw rollout return"
  LeanTest.assertTrue (approx (searched.qvalues 0)[0]! (23.0 / 6.0)) "allocated edges retain child-value semantics"
  LeanTest.assertEqual searched.rawValues[1]! 2.0
  assertKeyMap searched
  assertDagWellFormed searched "full nonroot capacity"

@[test]
def testMctxDagTransientEdgeAllocatesAfterReroot : IO Unit := do
  let tree := instantiateDagTreeFromRootWithCapacity capacityRoot 0 2 #[false] ()
  let searched := runCapacitySearch tree 0 2
  let carried := getSubtree searched 0
  LeanTest.assertEqual carried.numAllocated 1 "reroot frees the unreachable old root's slot"
  LeanTest.assertTrue (approx (carried.qvalues 0)[0]! 5.0)
  let resumed := runCapacitySearch carried 2 1
  LeanTest.assertEqual resumed.numAllocated 2
  LeanTest.assertEqual resumed.childrenIndex[0]![0]! (some 1)
  LeanTest.assertEqual resumed.childrenVisits[0]![0]! 2 "historical transient edge visits are preserved"
  LeanTest.assertEqual resumed.nodeVisits #[3, 1]
  LeanTest.assertTrue (approx resumed.nodeValues[1]! 10.0) "new child starts at the current evaluated value"
  LeanTest.assertTrue (approx resumed.childrenValues[0]![0]! 10.0) "replace stored return-Q with allocated child V"
  LeanTest.assertTrue (approx (resumed.qvalues 0)[0]! 10.0)
  LeanTest.assertTrue (approx resumed.nodeValues[0]! (17.0 / 3.0)) "reroot preserves prior rollout history"
  assertKeyMap resumed
  assertDagWellFormed carried "transient reroot carried"
  assertDagWellFormed resumed "transient reroot resumed"

@[test]
def testMctxDagTransientEdgeLaterFindsExistingTransposition : IO Unit := do
  let root : RootFnOutput Nat := { priorLogits := #[0.0, 0.0], value := 0.0, embedding := 0 }
  let recurrent : RecurrentFn Unit Nat := fun _ key action _ =>
    if action == 0 then
      ({ reward := 0.0, discount := 1.0, priorLogits := #[0.0, 0.0], value := 10.0 }, 1)
    else if key == 2 then
      ({ reward := 2.0, discount := 0.5, priorLogits := #[0.0, 0.0], value := 4.0 }, 2)
    else
      ({ reward := 1.0, discount := 0.5, priorLogits := #[0.0, 0.0], value := 10.0 }, 1)
  let tree := instantiateDagTreeFromRootWithCapacity root 0 2 #[false, false] ()
  let searched := searchWithDag () 0 tree recurrent id
    (fun _ t _ => if t.nodeVisits[0]! == 1 then 0 else 1) (fun _ _ _ _ => 0) 3
    (maxDepth := some 1)
  LeanTest.assertEqual searched.numAllocated 2 "a full DAG can still reuse an existing transposition"
  LeanTest.assertEqual searched.childrenIndex[0]! #[some 1, some 1]
  LeanTest.assertEqual searched.childrenVisits[0]! #[1, 2]
  LeanTest.assertEqual searched.childrenValues[0]![1]! 10.0 "promoted edge stores child V, not its old return-Q"
  LeanTest.assertTrue (approx (searched.qvalues 0)[1]! 6.0) "new reward and discount apply to the shared child's value"
  LeanTest.assertTrue (approx searched.nodeValues[0]! 5.0) "rollout returns 10, 4, 6 are backed up once each"
  assertKeyMap searched
  assertDagWellFormed searched "transient finds transposition"

@[test]
def testMctxDagAlphaZeroPersistentSubtree : IO Unit := do
  let root1 : RootFnOutput UInt64 := {
    priorLogits := #[0.2, -0.3, 0.9, 0.1]
    value := -0.2
    embedding := 0
  }
  let recurrent : RecurrentFn Unit UInt64 := fun _ _ action emb =>
    let nextEmb := emb * 131 + UInt64.ofNat (action + 1)
    ({ reward := 0.05, discount := 0.95, priorLogits := #[0.0, 0.0, 0.0, 0.0], value := 0.1 }, nextEmb)

  let out1 := alphazeroPolicyDag
    (params := ())
    (rngKey := 11)
    (root := root1)
    (recurrentFn := recurrent)
    (keyFn := id)
    (numSimulations := 6)
    (searchTree := none)
    (maxNodes := some 64)
    (dirichletFraction := 0.0)

  let carried := getSubtree out1.searchTree out1.action
  let before := carried.nodeVisits.getD ROOT_INDEX 0

  -- The continuation is rooted at the carried child's own state.
  let root2 : RootFnOutput UInt64 := {
    priorLogits := root1.priorLogits
    value := 0.7
    embedding := carried.embeddings[ROOT_INDEX]!
  }

  let out2 := alphazeroPolicyDag
    (params := ())
    (rngKey := 19)
    (root := root2)
    (recurrentFn := recurrent)
    (keyFn := id)
    (numSimulations := 3)
    (searchTree := some carried)
    (maxNodes := some 64)
    (dirichletFraction := 0.0)

  let after := out2.searchTree.nodeVisits.getD ROOT_INDEX 0
  LeanTest.assertEqual after (before + 3)
    "Persistent subtree should be reused and extended by new simulations"
  assertDagWellFormed out1.searchTree "persistent first search"
  assertDagWellFormed carried "persistent carried"
  assertDagWellFormed out2.searchTree "persistent continued search"
  LeanTest.assertTrue (out2.searchTree.numAllocated ≥ carried.numAllocated)
    "continuing a search never drops carried nodes"
  LeanTest.assertTrue (carried.childrenIndex.any (·.any (·.isSome)))
    "the carried subtree has edges for the persistence checks below"
  for i in [:carried.numAllocated] do
    LeanTest.assertEqual out2.searchTree.keys[i]! carried.keys[i]! s!"carried node {i} keeps its key"
    for a in [:carried.numActions] do
      if let some before := carried.childrenIndex[i]![a]! then
        LeanTest.assertEqual out2.searchTree.childrenIndex[i]![a]! (some before)
          s!"carried edge {i}/{a} keeps pointing at node {before}"
  LeanTest.assertTrue (approx (out2.searchTree.rawValues.getD ROOT_INDEX 0.0) root2.value 1e-6)
    "Continuing search should refresh root raw value from new root output"

@[test]
def testMctxDagResetSearchTree : IO Unit := do
  let root : RootFnOutput UInt64 := {
    priorLogits := #[0.0, 1.0]
    value := 0.0
    embedding := 12
  }

  let recurrent : RecurrentFn Unit UInt64 := fun _ _ action emb =>
    ({ reward := 0.0, discount := 0.0, priorLogits := #[0.0, 0.0], value := 0.0 }, emb + UInt64.ofNat (action + 1))

  let out := muzeroPolicyDag
    (params := ())
    (rngKey := 0)
    (root := root)
    (recurrentFn := recurrent)
    (keyFn := id)
    (numSimulations := 2)
    (dirichletFraction := 0.0)

  let reset := resetSearchTree out.searchTree
  LeanTest.assertEqual reset.numAllocated 0 "Reset DAG tree should clear allocated nodes"
  LeanTest.assertTrue (reset.keyToNode.isEmpty) "Reset DAG tree should clear transposition table"
  LeanTest.assertTrue ((List.range reset.nodeVisits.size).all fun i => reset.nodeVisits.getD i 1 = 0)
    "Reset DAG tree should clear node visits"
  LeanTest.assertTrue (out.searchTree.numAllocated > 1) "the search being reset allocated children"
  assertDagWellFormed out.searchTree "before reset"
  assertDagWellFormed reset "after reset"
  LeanTest.assertEqual reset.capacity out.searchTree.capacity "reset keeps capacity"
  LeanTest.assertTrue (reset.childrenIndex.all (·.all (·.isNone))) "reset clears every edge"
  LeanTest.assertTrue (reset.childrenVisits.all (·.all (· == 0))) "reset clears every edge visit"

@[test]
def testMctxDagGumbelPolicyRespectsInvalidMask : IO Unit := do
  let root : RootFnOutput UInt64 := {
    priorLogits := #[0.0, -1.0, 2.0, 3.0]
    value := -5.0
    embedding := 42
  }
  let rewards : Array Float := #[20.0, 3.0, -1.0, 10.0]
  let recurrent : RecurrentFn Unit UInt64 := fun _ _ action emb =>
    let nextEmb := emb * 131 + UInt64.ofNat (action + 1)
    ({ reward := rewards.getD action 0.0, discount := 0.0, priorLogits := #[0.0, 0.0, 0.0, 0.0], value := 0.0 }, nextEmb)

  let invalid : Array Bool := #[true, false, false, true]
  let out := gumbelMuZeroPolicyDag
    (params := ())
    (rngKey := 17)
    (root := root)
    (recurrentFn := recurrent)
    (keyFn := id)
    (numSimulations := 17)
    (invalidActions := some invalid)
    (maxDepth := some 3)
    (qtransform := fun t i => qtransformCompletedByMixValue t i (valueScale := 0.05) (maxvisitInit := 60.0) (rescaleValues := true))
    (gumbelScale := 1.0)

  LeanTest.assertTrue (out.action = 1 || out.action = 2)
    s!"Expected valid action among considered arms, got {out.action}"

  let w := out.actionWeights
  LeanTest.assertTrue (w.getD 0 1.0 < 1e-8 && w.getD 3 1.0 < 1e-8)
    "Invalid actions should have near-zero weight"

  let summary := out.searchTree.summary
  LeanTest.assertEqual (summary.visitCounts.getD 0 999) 0 "Invalid action 0 should not be visited"
  LeanTest.assertEqual (summary.visitCounts.getD 3 999) 0 "Invalid action 3 should not be visited"

  let (_leaf, depth) := deepestLeaf out.searchTree
  LeanTest.assertTrue (depth ≤ 3) s!"Search depth should respect max_depth=3, got {depth}"
  assertDagWellFormed out.searchTree "gumbel masked"
  let rootEdges := out.searchTree.childrenIndex[ROOT_INDEX]!
  LeanTest.assertEqual rootEdges[0]! none "invalid action 0 is never expanded"
  LeanTest.assertEqual rootEdges[3]! none "invalid action 3 is never expanded"
  LeanTest.assertTrue (rootEdges[1]!.isSome || rootEdges[2]!.isSome)
    "some valid root action is expanded"

/-! Exploration integration: seed replay, legal support, and policy wiring. -/

private def explorationRoot : RootFnOutput UInt64 := {
  priorLogits := #[0.0, 0.0, 0.0, 0.0], value := 0.0, embedding := 0
}

private def explorationRecurrent : RecurrentFn Unit UInt64 := fun _ _ action embedding =>
  ({ reward := 0.0, discount := 0.0, priorLogits := explorationRoot.priorLogits, value := 0.0 }, embedding * 5 + action.toUInt64 + 1)

private def explorationPolicy (alphaZero : Bool) (seed : UInt64) (simulations : Nat)
    (temperature fraction alpha : Float) (root : RootFnOutput UInt64 := explorationRoot)
    (invalid : Option (Array Bool) := none) : PolicyOutput (DagTree UInt64 UInt64 Unit) :=
  if alphaZero then
    alphazeroPolicyDag () seed root explorationRecurrent id simulations
      (invalidActions := invalid) (temperature := temperature)
      (dirichletFraction := fraction) (dirichletAlpha := alpha)
  else
    muzeroPolicyDag () seed root explorationRecurrent id simulations
      (invalidActions := invalid) (temperature := temperature)
      (dirichletFraction := fraction) (dirichletAlpha := alpha)

@[test] def testMctxDagExplorationTemperatureAndReplay : IO Unit := do
  for alphaZero in #[false, true] do
    let first := explorationPolicy alphaZero 0 16 1.0 0.0 0.3
    let mut actions := #[]
    for seed in [:32] do
      let out := explorationPolicy alphaZero seed.toUInt64 16 1.0 0.0 0.3
      let replay := explorationPolicy alphaZero seed.toUInt64 16 1.0 0.0 0.3
      LeanTest.assertEqual out.action replay.action "same seed replays the selected action"
      LeanTest.assertEqual out.actionWeights replay.actionWeights "same seed replays policy weights"
      LeanTest.assertEqual out.searchTree.summary.visitCounts first.searchTree.summary.visitCounts
        "with noise disabled, action sampling must not alter deterministic search visits"
      LeanTest.assertEqual out.searchTree.childrenIndex first.searchTree.childrenIndex
        "with noise disabled, action sampling must not alter the searched structure"
      assertDagWellFormed out.searchTree s!"exploration seed {seed}"
      actions := actions.push out.action
      for temperature in #[0.0, -1.0] do
        let greedy := explorationPolicy alphaZero seed.toUInt64 16 temperature 0.0 0.3
        LeanTest.assertEqual greedy.action (torch.mctx.argmax greedy.actionWeights)
          "nonpositive temperature must select the visit-count argmax"
    LeanTest.assertTrue (actions.any (· != first.action))
      "positive-temperature policy must sample across different small seeds"

@[test] def testMctxDagExplorationZeroVisitsRespectMask : IO Unit := do
  -- Masking must happen before normalization: the invalid logit deliberately
  -- dominates every representable softmax weight of the legal actions.
  let root := { explorationRoot with priorLogits := #[1e100, 0.0, 1e100, 1.0] }
  let invalid := some #[true, false, true, false]
  for alphaZero in #[false, true] do
    for temperature in #[0.0, 1.0, 2.0] do
      for seed in [:16] do
        let out := explorationPolicy alphaZero seed.toUInt64 0 temperature 0.0 0.3 root invalid
        let prior := softmax out.searchTree.childrenPriorLogits[ROOT_INDEX]!
        let expected := softmax #[0.0, 1.0]
        LeanTest.assertTrue (approx prior[1]! expected[0]! && approx prior[3]! expected[1]!)
          "masking a huge invalid prior must preserve the legal prior ratio"
        LeanTest.assertTrue (out.action == 1 || out.action == 3) "zero visits must still select a legal action"
        LeanTest.assertEqual out.actionWeights[0]! 0.0 "invalid action zero has exactly zero weight"
        LeanTest.assertEqual out.actionWeights[2]! 0.0 "invalid action two has exactly zero weight"
        LeanTest.assertTrue (out.actionWeights.all (fun w => w.isFinite && w >= 0.0)) "weights are finite probabilities"
        LeanTest.assertTrue (approx (sum out.actionWeights) 1.0) "zero-visit legal weights sum to one"

@[test] def testMctxDagExplorationDirichletAlpha : IO Unit := do
  for alphaZero in #[false, true] do
    let priors := fun seed alpha =>
      softmax ((explorationPolicy alphaZero seed 0 0.0 1.0 alpha).searchTree.childrenPriorLogits[ROOT_INDEX]!)
    LeanTest.assertEqual (priors 7 0.3) (priors 7 0.3) "Dirichlet root noise replays by seed"
    LeanTest.assertTrue (priors 0 0.3 != priors 1 0.3) "nearby UInt64 seeds change root noise"
    let mut sparseConcentration := 0.0
    let mut denseConcentration := 0.0
    for seed in [:24] do
      sparseConcentration := sparseConcentration + sum ((priors seed.toUInt64 0.05).map (fun p => p * p))
      denseConcentration := denseConcentration + sum ((priors seed.toUInt64 10.0).map (fun p => p * p))
    LeanTest.assertTrue (sparseConcentration > denseConcentration + 8.0)
      "small Dirichlet alpha must produce more concentrated priors than large alpha"

@[test] def testMctxDagAlphaZeroVisitTemperatureDistribution : IO Unit := do
  -- Reuse fixed visits so this checks policy-level temperature wiring without
  -- introducing search randomness: P(action=0) is 0.9 at T=1, 0.75 at T=2.
  let base := instantiateDagTreeFromRootWithCapacity explorationRoot (0 : UInt64) 1 #[false, false, false, false] ()
  let carried := { base with childrenVisits := #[#[9, 1, 0, 0]], nodeVisits := #[10] }
  let draw := fun seed temperature =>
    (alphazeroPolicyDag () seed explorationRoot explorationRecurrent id 0
      (searchTree := some carried) (dirichletFraction := 0.0) (temperature := temperature)).action
  let mut cold := 0
  let mut hot := 0
  for seed in [:256] do
    let a := draw seed.toUInt64 1.0
    let b := draw seed.toUInt64 2.0
    LeanTest.assertTrue (a < 2 && b < 2) "unvisited actions carry no sampling mass"
    if a == 0 then cold := cold + 1
    if b == 0 then hot := hot + 1
  LeanTest.assertTrue (cold > hot + 16 && cold >= 200 && hot >= 150 && hot <= 220)
    s!"temperature must change categorical visit probabilities, got T1={cold}/256 T2={hot}/256"

@[test] def testMctxDagGumbelSmallSeedExploration : IO Unit := do
  let run := fun seed => gumbelMuZeroPolicyDag () seed explorationRoot explorationRecurrent id 0
  let first := run 0
  LeanTest.assertEqual (run 0).searchTree.extraData.rootGumbel first.searchTree.extraData.rootGumbel
    "same seed replays Gumbel draws"
  LeanTest.assertTrue ((run 1).searchTree.extraData.rootGumbel != first.searchTree.extraData.rootGumbel)
    "adjacent small seeds must change Gumbel draws"
  let actions := (List.range 32).toArray.map fun seed => (run seed.toUInt64).action
  LeanTest.assertTrue (actions.any (· != first.action)) "Gumbel exploration must change selected arms across seeds"
