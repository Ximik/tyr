import Std
import Tyr.MctxDag.Base

/-!
# Tyr.MctxDag.Tree

`DagTree S K E`, the DAG search-tree state: like `Mctx.Tree` but with a
`keyToNode` transposition map, so repeated states share nodes; includes
capacity, Q-value, summary, reset, and reachable-subgraph helpers.
-/

namespace torch.mctxdag

/-- Search tree state with transposition support through `keyToNode`. -/
structure DagTree (S K E : Type) [BEq K] [Hashable K] where
  nodeVisits : Array Nat
  rawValues : Array Float
  nodeValues : Array Float
  /-- `none` for an edge whose child is not allocated. -/
  childrenIndex : Array (Array (Option Nat))
  childrenPriorLogits : Array (Array Float)
  childrenVisits : Array (Array Nat)
  childrenRewards : Array (Array Float)
  childrenDiscounts : Array (Array Float)
  /-- Allocated edges store the child's value estimate. Evaluated edges that
      remain unallocated at capacity store their mean complete rollout return. -/
  childrenValues : Array (Array Float)
  embeddings : Array S
  keys : Array K
  rootInvalidActions : Array Bool
  extraData : E
  keyToNode : Std.HashMap K NodeIndex
  numAllocated : Nat
  deriving Repr

instance [BEq K] [Hashable K] [Inhabited S] [Inhabited K] [Inhabited E] : Inhabited (DagTree S K E) where
  default := {
    nodeVisits := #[]
    rawValues := #[]
    nodeValues := #[]
    childrenIndex := #[]
    childrenPriorLogits := #[]
    childrenVisits := #[]
    childrenRewards := #[]
    childrenDiscounts := #[]
    childrenValues := #[]
    embeddings := #[]
    keys := #[]
    rootInvalidActions := #[]
    extraData := default
    keyToNode := {}
    numAllocated := 0
  }

def ROOT_INDEX : Nat := 0
def DagTree.capacity [BEq K] [Hashable K] (tree : DagTree S K E) : Nat :=
  tree.nodeVisits.size

def DagTree.numActions [BEq K] [Hashable K] (tree : DagTree S K E) : Nat :=
  (tree.childrenIndex.getD ROOT_INDEX #[]).size

def DagTree.numSimulations [BEq K] [Hashable K] (tree : DagTree S K E) : Nat :=
  tree.capacity - 1

def DagTree.qvalues [BEq K] [Hashable K] (tree : DagTree S K E) (nodeIndex : Nat) : Array Float :=
  let rewards := tree.childrenRewards.getD nodeIndex #[]
  let discounts := tree.childrenDiscounts.getD nodeIndex #[]
  let values := tree.childrenValues.getD nodeIndex #[]
  let children := tree.childrenIndex.getD nodeIndex #[]
  let visits := tree.childrenVisits.getD nodeIndex #[]
  (List.range rewards.size).toArray.map fun a =>
    if (children.getD a none).isNone && visits.getD a 0 > 0 then
      values.getD a 0.0
    else
      rewards.getD a 0.0 + discounts.getD a 0.0 * values.getD a 0.0

private def visitProbsFromCounts (counts : Array Nat) (numActions : Nat) : Array Float :=
  let total := counts.foldl (init := 0) (· + ·)
  if total = 0 then
    if numActions = 0 then #[] else Array.replicate numActions (1.0 / numActions.toFloat)
  else
    counts.map (fun c => c.toFloat / total.toFloat)

def DagTree.summary [BEq K] [Hashable K] (tree : DagTree S K E) : SearchSummary :=
  let value := tree.nodeValues.getD ROOT_INDEX 0.0
  let visitCounts := tree.childrenVisits.getD ROOT_INDEX #[]
  let qvalues := tree.qvalues ROOT_INDEX
  let visitProbs := visitProbsFromCounts visitCounts tree.numActions
  {
    visitCounts := visitCounts
    visitProbs := visitProbs
    value := value
    qvalues := qvalues
  }

private def updateAt (xs : Array α) (i : Nat) (v : α) : Array α :=
  if i < xs.size then xs.set! i v else xs

private def updateAt2D (xs : Array (Array α)) (i j : Nat) (v : α) : Array (Array α) :=
  if i < xs.size then
    let row := xs.getD i #[]
    if j < row.size then
      xs.set! i (row.set! j v)
    else
      xs
  else
    xs

/-- Resets a DAG tree to the empty state, keeping capacity and `extraData`. -/
def resetSearchTree [Inhabited S] [Inhabited K] [BEq K] [Hashable K]
    (tree : DagTree S K E) : DagTree S K E :=
  let numNodes := tree.capacity
  let numActions := tree.numActions
  let edgeRow : Array (Option Nat) := Array.replicate numActions none
  let visitRow : Array Nat := Array.replicate numActions 0
  let floatRow : Array Float := Array.replicate numActions 0.0
  { tree with
    nodeVisits := Array.replicate numNodes 0
    rawValues := Array.replicate numNodes 0.0
    nodeValues := Array.replicate numNodes 0.0
    childrenIndex := Array.replicate numNodes edgeRow
    childrenPriorLogits := Array.replicate numNodes floatRow
    childrenVisits := Array.replicate numNodes visitRow
    childrenRewards := Array.replicate numNodes floatRow
    childrenDiscounts := Array.replicate numNodes floatRow
    childrenValues := Array.replicate numNodes floatRow
    embeddings := Array.replicate numNodes default
    keys := Array.replicate numNodes default
    rootInvalidActions := Array.replicate numActions false
    keyToNode := {}
    numAllocated := 0
  }

/-- Extracts the reachable subgraph from a selected root child action,
    remapping node indices to a compact `[0, n)` range. -/
def getSubtree [Inhabited S] [Inhabited K] [BEq K] [Hashable K]
    (tree : DagTree S K E)
    (childAction : Nat)
    : DagTree S K E := Id.run do
  let some rootChild := (tree.childrenIndex.getD ROOT_INDEX #[]).getD childAction none
    | return resetSearchTree tree
  let numNodes := tree.capacity
  let numActions := tree.numActions
  if rootChild >= numNodes then
    return resetSearchTree tree

  let mut keep : Array Bool := Array.replicate numNodes false
  let mut stack : Array Nat := #[rootChild]

  while !stack.isEmpty do
    let some node := stack.back? | break
    stack := stack.pop
    if node < numNodes && !(keep.getD node false) then
      keep := keep.set! node true
      for a in [:numActions] do
        if let some child := (tree.childrenIndex.getD node #[]).getD a none then
          stack := stack.push child

  -- A transposition can point to an earlier allocated node; a cycle can even
  -- reach the old root. The selected child must nevertheless become index 0.
  let mut retained : Array Nat := #[rootChild]
  for i in [:numNodes] do
    if i != rootChild && keep.getD i false then
      retained := retained.push i

  let mut oldToNew : Array (Option Nat) := Array.replicate numNodes none
  for newIdx in [:retained.size] do
    let oldIdx := retained.getD newIdx 0
    oldToNew := oldToNew.set! oldIdx (some newIdx)

  let edgeRow : Array (Option Nat) := Array.replicate numActions none
  let visitRow : Array Nat := Array.replicate numActions 0
  let floatRow : Array Float := Array.replicate numActions 0.0
  let mut out : DagTree S K E := {
    nodeVisits := Array.replicate numNodes 0
    rawValues := Array.replicate numNodes 0.0
    nodeValues := Array.replicate numNodes 0.0
    childrenIndex := Array.replicate numNodes edgeRow
    childrenPriorLogits := Array.replicate numNodes floatRow
    childrenVisits := Array.replicate numNodes visitRow
    childrenRewards := Array.replicate numNodes floatRow
    childrenDiscounts := Array.replicate numNodes floatRow
    childrenValues := Array.replicate numNodes floatRow
    embeddings := Array.replicate numNodes default
    keys := Array.replicate numNodes default
    rootInvalidActions := Array.replicate numActions false
    extraData := tree.extraData
    keyToNode := {}
    numAllocated := retained.size
  }

  let mut keyMap : Std.HashMap K NodeIndex := {}
  for newIdx in [:retained.size] do
    let oldIdx := retained.getD newIdx 0
    let k := tree.keys.getD oldIdx default
    out := { out with
      nodeVisits := updateAt out.nodeVisits newIdx (tree.nodeVisits.getD oldIdx 0)
      rawValues := updateAt out.rawValues newIdx (tree.rawValues.getD oldIdx 0.0)
      nodeValues := updateAt out.nodeValues newIdx (tree.nodeValues.getD oldIdx 0.0)
      embeddings := updateAt out.embeddings newIdx (tree.embeddings.getD oldIdx default)
      keys := updateAt out.keys newIdx k
    }
    keyMap := keyMap.insert k newIdx

    for a in [:numActions] do
      let childNew := ((tree.childrenIndex.getD oldIdx #[]).getD a none).bind (oldToNew.getD · none)
      out := { out with
        childrenIndex := updateAt2D out.childrenIndex newIdx a childNew
        childrenPriorLogits := updateAt2D out.childrenPriorLogits newIdx a
          ((tree.childrenPriorLogits.getD oldIdx #[]).getD a 0.0)
        childrenVisits := updateAt2D out.childrenVisits newIdx a
          ((tree.childrenVisits.getD oldIdx #[]).getD a 0)
        childrenRewards := updateAt2D out.childrenRewards newIdx a
          ((tree.childrenRewards.getD oldIdx #[]).getD a 0.0)
        childrenDiscounts := updateAt2D out.childrenDiscounts newIdx a
          ((tree.childrenDiscounts.getD oldIdx #[]).getD a 0.0)
        childrenValues := updateAt2D out.childrenValues newIdx a
          ((tree.childrenValues.getD oldIdx #[]).getD a 0.0)
      }

  return { out with keyToNode := keyMap }

end torch.mctxdag
