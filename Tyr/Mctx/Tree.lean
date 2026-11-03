import Tyr.Mctx.Base
import Tyr.Mctx.Math

/-!
# Tyr.Mctx.Tree

`Tree S E`, the unbatched search-tree state (Lean port of `mctx.Tree`):
flat arrays of node and edge statistics, with Q-value, root-summary,
reset, and subtree-extraction helpers.
-/

namespace torch.mctx

/-- Search tree state (unbatched Lean port of mctx.Tree). -/
structure Tree (S E : Type) where
  nodeVisits : Array Nat
  /-- Allocated nodes occupy `[0, numAllocated)`; all remaining visit counts
      are zero. Constructors and mutation helpers maintain this invariant.
      The default scans once for compatibility with manually constructed trees;
      manual record updates must maintain the counter explicitly. -/
  numAllocated : Nat := (nodeVisits.toList.takeWhile (· != 0)).length
  rawValues : Array Float
  nodeValues : Array Float
  /-- `none` for the root and for unallocated nodes. -/
  parents : Array (Option Nat)
  actionFromParent : Array (Option Nat)
  /-- `none` for an edge whose child is not allocated. -/
  childrenIndex : Array (Array (Option Nat))
  childrenPriorLogits : Array (Array Float)
  childrenVisits : Array (Array Nat)
  childrenRewards : Array (Array Float)
  childrenDiscounts : Array (Array Float)
  /-- For allocated children, their continuation value. For an unallocated
      edge with positive visits, the mean complete evaluated return `r + γV`.
      Such transient edges arise when the fixed node capacity is exhausted. -/
  childrenValues : Array (Array Float)
  embeddings : Array S
  rootInvalidActions : Array Bool
  extraData : E
  deriving Repr

instance [Inhabited S] [Inhabited E] : Inhabited (Tree S E) where
  default := {
    nodeVisits := #[]
    rawValues := #[]
    nodeValues := #[]
    parents := #[]
    actionFromParent := #[]
    childrenIndex := #[]
    childrenPriorLogits := #[]
    childrenVisits := #[]
    childrenRewards := #[]
    childrenDiscounts := #[]
    childrenValues := #[]
    embeddings := #[]
    rootInvalidActions := #[]
    extraData := default
  }

/-- Root node index. -/
def ROOT_INDEX : Nat := 0

/-- Number of action branches per node. -/
def Tree.numActions (tree : Tree S E) : Nat :=
  (tree.childrenIndex.getD ROOT_INDEX #[]).size

/-- Number of simulations represented in this tree capacity. -/
def Tree.numSimulations (tree : Tree S E) : Nat :=
  tree.nodeVisits.size - 1

/-- Q-values `r + gamma * v` for all actions from `nodeIndex`. -/
def Tree.qvalues (tree : Tree S E) (nodeIndex : Nat) : Array Float :=
  let rewards := tree.childrenRewards.getD nodeIndex #[]
  let discounts := tree.childrenDiscounts.getD nodeIndex #[]
  let values := tree.childrenValues.getD nodeIndex #[]
  (List.range rewards.size).toArray.map fun a =>
    if ((tree.childrenIndex.getD nodeIndex #[]).getD a none).isNone &&
        (tree.childrenVisits.getD nodeIndex #[]).getD a 0 > 0 then
      values.getD a 0.0
    else
      rewards.getD a 0.0 + discounts.getD a 0.0 * values.getD a 0.0

private def visitProbsFromCounts (counts : Array Nat) (numActions : Nat) : Array Float :=
  let total := counts.foldl (init := 0) (· + ·)
  if total = 0 then
    if numActions = 0 then #[] else Array.replicate numActions (1.0 / numActions.toFloat)
  else
    counts.map (fun c => c.toFloat / total.toFloat)

/-- Root summary statistics used by policies. -/
def Tree.summary (tree : Tree S E) : SearchSummary :=
  let value := tree.nodeValues.getD ROOT_INDEX 0.0
  let visitCounts := tree.childrenVisits.getD ROOT_INDEX #[]
  let qvalues := tree.qvalues ROOT_INDEX
  let visitProbs := visitProbsFromCounts visitCounts tree.numActions
  { visitCounts := visitCounts
  , visitProbs := visitProbs
  , value := value
  , qvalues := qvalues
  }

/-- Returns the next free node slot in constant time (or capacity if full). -/
def Tree.nextNodeIndex (tree : Tree S E) : Nat :=
  tree.numAllocated

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

/-- Resets a search tree in-place shape to empty/unvisited state. -/
def resetSearchTree [Inhabited S] (tree : Tree S E) : Tree S E :=
  let numNodes := tree.nodeVisits.size
  let numActions := tree.numActions
  let edgeRow : Array (Option Nat) := Array.replicate numActions none
  let natRow : Array Nat := Array.replicate numActions 0
  let floatRow : Array Float := Array.replicate numActions 0.0
  { tree with
    nodeVisits := Array.replicate numNodes 0
    numAllocated := 0
    rawValues := Array.replicate numNodes 0.0
    nodeValues := Array.replicate numNodes 0.0
    parents := Array.replicate numNodes none
    actionFromParent := Array.replicate numNodes none
    childrenIndex := Array.replicate numNodes edgeRow
    childrenPriorLogits := Array.replicate numNodes floatRow
    childrenVisits := Array.replicate numNodes natRow
    childrenRewards := Array.replicate numNodes floatRow
    childrenDiscounts := Array.replicate numNodes floatRow
    childrenValues := Array.replicate numNodes floatRow
    embeddings := Array.replicate numNodes default
    rootInvalidActions := Array.replicate numActions false
  }

/-- Extracts the subtree rooted at a root child action and remaps node indices. -/
def getSubtree [Inhabited S] (tree : Tree S E) (childAction : Nat) : Tree S E := Id.run do
  let some rootChild := (tree.childrenIndex.getD ROOT_INDEX #[]).getD childAction none
    | return resetSearchTree tree
  let numNodes := tree.nodeVisits.size
  let numActions := tree.numActions

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

  let mut retained : Array Nat := #[]
  for i in [:numNodes] do
    if keep.getD i false then
      retained := retained.push i

  let mut oldToNew : Array (Option Nat) := Array.replicate numNodes none
  for newIdx in [:retained.size] do
    let oldIdx := retained.getD newIdx 0
    oldToNew := oldToNew.set! oldIdx (some newIdx)

  let edgeRow : Array (Option Nat) := Array.replicate numActions none
  let natRow : Array Nat := Array.replicate numActions 0
  let floatRow : Array Float := Array.replicate numActions 0.0
  let mut out : Tree S E := {
    nodeVisits := Array.replicate numNodes 0
    numAllocated := retained.size
    rawValues := Array.replicate numNodes 0.0
    nodeValues := Array.replicate numNodes 0.0
    parents := Array.replicate numNodes none
    actionFromParent := Array.replicate numNodes none
    childrenIndex := Array.replicate numNodes edgeRow
    childrenPriorLogits := Array.replicate numNodes floatRow
    childrenVisits := Array.replicate numNodes natRow
    childrenRewards := Array.replicate numNodes floatRow
    childrenDiscounts := Array.replicate numNodes floatRow
    childrenValues := Array.replicate numNodes floatRow
    embeddings := Array.replicate numNodes default
    rootInvalidActions := Array.replicate numActions false
    extraData := tree.extraData
  }

  for newIdx in [:retained.size] do
    let oldIdx := retained.getD newIdx 0
    out := { out with
      nodeVisits := updateAt out.nodeVisits newIdx (tree.nodeVisits.getD oldIdx 0)
      rawValues := updateAt out.rawValues newIdx (tree.rawValues.getD oldIdx 0.0)
      nodeValues := updateAt out.nodeValues newIdx (tree.nodeValues.getD oldIdx 0.0)
      embeddings := updateAt out.embeddings newIdx (tree.embeddings.getD oldIdx default)
    }

    let parentNew : Option Nat :=
      if newIdx = ROOT_INDEX then none
      else (tree.parents.getD oldIdx none).bind (oldToNew.getD · none)
    out := { out with
      parents := updateAt out.parents newIdx parentNew
      actionFromParent := updateAt out.actionFromParent newIdx (
        if newIdx = ROOT_INDEX then none else tree.actionFromParent.getD oldIdx none
      )
    }

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

  return out

end torch.mctx
