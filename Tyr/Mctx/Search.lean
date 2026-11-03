import Tyr.Mctx.ActionSelection

/-!
# Tyr.Mctx.Search

Core MCTS loop over `Tree`: root instantiation, simulation, edge expansion
through the recurrent model, value backup, and the unbatched `search` /
`searchWithTree` entry points.
-/

namespace torch.mctx

private def updateAt (xs : Array α) (i : Nat) (v : α) : Array α :=
  if i < xs.size then xs.set! i v else xs

private def updateAt2D (xs : Array (Array α)) (i j : Nat) (v : α) : Array (Array α) :=
  if i < xs.size then
    let row := xs.getD i #[]
    if j < row.size then
      let row' := row.set! j v
      xs.set! i row'
    else
      xs
  else
    xs

/-- Updates one node's value/prior/embedding and increments visit count.
    `nodeIndex` must name an allocated node or the next free prefix slot. -/
def updateTreeNode
    (tree : Tree S E)
    (nodeIndex : Nat)
    (priorLogits : Array Float)
    (value : Float)
    (embedding : S)
    : Tree S E :=
  let newVisit := tree.nodeVisits.getD nodeIndex 0 + 1
  { tree with
    childrenPriorLogits := updateAt tree.childrenPriorLogits nodeIndex priorLogits
    rawValues := updateAt tree.rawValues nodeIndex value
    nodeValues := updateAt tree.nodeValues nodeIndex value
    nodeVisits := updateAt tree.nodeVisits nodeIndex newVisit
    numAllocated := if nodeIndex < tree.nodeVisits.size then
      max tree.numAllocated (nodeIndex + 1) else tree.numAllocated
    embeddings := updateAt tree.embeddings nodeIndex embedding
  }

/-- Initializes an empty tree and fills the root node. Capacity is at least one
    so even a requested zero capacity retains the root and can back up values. -/
def instantiateTreeFromRootWithCapacity
    [Inhabited S]
    (root : RootFnOutput S)
    (numNodes : Nat)
    (rootInvalidActions : Array Bool)
    (extraData : E)
    : Tree S E :=
  let numNodes := max 1 numNodes
  let numActions := root.priorLogits.size
  let edgeRow : Array (Option Nat) := Array.replicate numActions none
  let visitRow : Array Nat := Array.replicate numActions 0
  let floatRow : Array Float := Array.replicate numActions 0.0
  let tree : Tree S E := {
    nodeVisits := Array.replicate numNodes 0
    numAllocated := 0
    rawValues := Array.replicate numNodes 0.0
    nodeValues := Array.replicate numNodes 0.0
    parents := Array.replicate numNodes none
    actionFromParent := Array.replicate numNodes none
    childrenIndex := Array.replicate numNodes edgeRow
    childrenPriorLogits := Array.replicate numNodes floatRow
    childrenVisits := Array.replicate numNodes visitRow
    childrenRewards := Array.replicate numNodes floatRow
    childrenDiscounts := Array.replicate numNodes floatRow
    childrenValues := Array.replicate numNodes floatRow
    embeddings := updateAt (Array.replicate numNodes default) ROOT_INDEX root.embedding
    rootInvalidActions := rootInvalidActions
    extraData := extraData
  }
  updateTreeNode tree ROOT_INDEX root.priorLogits root.value root.embedding

/-- Initializes an empty tree and fills the root node.
    Capacity is `numSimulations + 1` (upstream mctx default). -/
def instantiateTreeFromRoot
    [Inhabited S]
    (root : RootFnOutput S)
    (numSimulations : Nat)
    (rootInvalidActions : Array Bool)
    (extraData : E)
    : Tree S E :=
  instantiateTreeFromRootWithCapacity root (numSimulations + 1) rootInvalidActions extraData

/-- Updates root priors/raw value/embedding in an existing tree. If the root was
    uninitialized, initializes root visit/value as well. -/
def updateTreeWithRoot
    [Inhabited S]
    (tree : Tree S E)
    (root : RootFnOutput S)
    (rootInvalidActions : Array Bool)
    (extraData : E)
    : Tree S E :=
  if tree.nodeVisits.isEmpty then
    instantiateTreeFromRootWithCapacity root 1 rootInvalidActions extraData
  else
    let rootUninitialized := tree.nodeVisits.getD ROOT_INDEX 0 = 0
    let tree := { tree with
      childrenPriorLogits := updateAt tree.childrenPriorLogits ROOT_INDEX root.priorLogits
      rawValues := updateAt tree.rawValues ROOT_INDEX root.value
      embeddings := updateAt tree.embeddings ROOT_INDEX root.embedding
      rootInvalidActions := rootInvalidActions
      extraData := extraData
    }
    if rootUninitialized then
      { tree with
        nodeValues := updateAt tree.nodeValues ROOT_INDEX root.value
        nodeVisits := updateAt tree.nodeVisits ROOT_INDEX 1
        numAllocated := max 1 tree.numAllocated
      }
    else
      tree

/-- Simulates from root until reaching an unvisited edge or depth cutoff. -/
def simulate
    (rngKey : UInt64)
    (tree : Tree S E)
    (actionSelectionFn : InteriorActionSelectionFn S E)
    (maxDepth : Nat)
    : NodeIndex × Action := Id.run do
  let mut nodeIndex := ROOT_INDEX
  let mut parentIndex := ROOT_INDEX
  let mut action : Action := 0
  let mut depth := 0
  let mut continuing := true

  while continuing do
    parentIndex := nodeIndex
    action := actionSelectionFn rngKey tree nodeIndex depth
    let nextDepth := depth + 1
    match (tree.childrenIndex.getD nodeIndex #[]).getD action none with
    | some nextNode =>
      if nextDepth < maxDepth then
        nodeIndex := nextNode
        depth := nextDepth
      else
        continuing := false
    | none => continuing := false

  return (parentIndex, action)

/-- Store an evaluated transition, allocating its child only when a node slot
    is available. Re-evaluation at an existing depth-cutoff node retains the
    upstream behavior of replacing that node's model value. -/
def expandWithStep
    (tree : Tree S E)
    (parentIndex : NodeIndex)
    (action : Action)
    (nextNodeIndex : NodeIndex)
    (step : RecurrentFnOutput)
    (nextEmbedding : S)
    : Tree S E :=
  let inBounds := nextNodeIndex < tree.nodeVisits.size
  let tree :=
    if inBounds then
      updateTreeNode tree nextNodeIndex step.priorLogits step.value nextEmbedding
    else
      tree
  let childIdx : Option Nat := if inBounds then some nextNodeIndex else none
  { tree with
    childrenIndex := updateAt2D tree.childrenIndex parentIndex action childIdx
    childrenRewards := updateAt2D tree.childrenRewards parentIndex action step.reward
    childrenDiscounts := updateAt2D tree.childrenDiscounts parentIndex action step.discount
    parents := updateAt tree.parents nextNodeIndex (some parentIndex)
    actionFromParent := updateAt tree.actionFromParent nextNodeIndex (some action)
  }

/-- Expands one `(parent, action)` edge and evaluates recurrent dynamics. -/
def expand
    [Inhabited S]
    (params : P)
    (rngKey : UInt64)
    (tree : Tree S E)
    (recurrentFn : RecurrentFn P S)
    (parentIndex : NodeIndex)
    (action : Action)
    (nextNodeIndex : NodeIndex)
    : Tree S E :=
  let embedding := tree.embeddings.getD parentIndex default
  let (step, nextEmbedding) := recurrentFn params rngKey action embedding
  expandWithStep tree parentIndex action nextNodeIndex step nextEmbedding

-- The rollout return is separate from the current node's running mean. This
-- matters when a full-capacity edge was backed up before traversing ancestors.
private def backwardFrom (tree : Tree S E) (leafIndex : NodeIndex)
    (rolloutValue : Float) : Tree S E := Id.run do
  let mut t := tree
  let mut index := leafIndex
  let mut leafValue := rolloutValue

  while index != ROOT_INDEX do
    match t.parents.getD index none, t.actionFromParent.getD index none with
    | some parent, some action =>
      let count := t.nodeVisits.getD parent 0
      let reward := (t.childrenRewards.getD parent #[]).getD action 0.0
      let discount := (t.childrenDiscounts.getD parent #[]).getD action 0.0
      leafValue := reward + discount * leafValue
      let parentValue :=
        ((t.nodeValues.getD parent 0.0) * Float.ofNat count + leafValue) /
          Float.ofNat (count + 1)
      let childValue := t.nodeValues.getD index 0.0
      let childCount := (t.childrenVisits.getD parent #[]).getD action 0 + 1

      t := { t with
        nodeValues := updateAt t.nodeValues parent parentValue
        nodeVisits := updateAt t.nodeVisits parent (count + 1)
        childrenValues := updateAt2D t.childrenValues parent action childValue
        childrenVisits := updateAt2D t.childrenVisits parent action childCount
      }
      index := parent
    | _, _ => index := ROOT_INDEX

  return t

/-- Backs up values from an allocated leaf to the root. -/
def backward (tree : Tree S E) (leafIndex : NodeIndex) : Tree S E :=
  backwardFrom tree leafIndex (tree.nodeValues.getD leafIndex 0.0)

/-- Back up a newly evaluated transition when the hard node capacity is full.
    Its edge remains unallocated, but visits and the mean complete return are
    recorded. Every ancestor receives this simulation's actual return rather
    than the parent's updated mean. -/
private def backwardUnallocated (tree : Tree S E) (parentIndex : NodeIndex)
    (action : Action) (step : RecurrentFnOutput) : Tree S E :=
  let value := step.reward + step.discount * step.value
  let edgeCount := (tree.childrenVisits.getD parentIndex #[]).getD action 0
  let oldValue := (tree.childrenValues.getD parentIndex #[]).getD action 0.0
  let edgeValue := (oldValue * edgeCount.toFloat + value) / (edgeCount + 1).toFloat
  let count := tree.nodeVisits.getD parentIndex 0
  let parentValue :=
    ((tree.nodeValues.getD parentIndex 0.0) * Float.ofNat count + value) / Float.ofNat (count + 1)
  let tree := { tree with
    nodeValues := updateAt tree.nodeValues parentIndex parentValue
    nodeVisits := updateAt tree.nodeVisits parentIndex (count + 1)
    childrenValues := updateAt2D tree.childrenValues parentIndex action edgeValue
    childrenVisits := updateAt2D tree.childrenVisits parentIndex action (edgeCount + 1)
  }
  backwardFrom tree parentIndex value

/-- Complete an evaluated simulation for both ordinary and capacity-limited
    searches. When a formerly transient edge gains a node slot after rerooting,
    the new child starts from the current model value; backup replaces the
    edge's transient complete-return mean with that child's continuation value.
    Historical edge visits and ancestor averages are retained. -/
def expandWithStepAndBackup (tree : Tree S E) (parentIndex : NodeIndex)
    (action : Action) (nextNodeIndex : NodeIndex) (step : RecurrentFnOutput)
    (nextEmbedding : S) : Tree S E :=
  let inBounds := nextNodeIndex < tree.nodeVisits.size
  let tree := expandWithStep tree parentIndex action nextNodeIndex step nextEmbedding
  if inBounds then backward tree nextNodeIndex
  else backwardUnallocated tree parentIndex action step

/-- Runs full MCTS search (unbatched Lean port). -/
def searchWithTree
    [Inhabited S]
    (params : P)
    (rngKey : UInt64)
    (tree : Tree S E)
    (recurrentFn : RecurrentFn P S)
    (rootActionSelectionFn : RootActionSelectionFn S E)
    (interiorActionSelectionFn : InteriorActionSelectionFn S E)
    (numSimulations : Nat)
    (maxDepth : Option Nat := none)
    : Tree S E := Id.run do
  let actionSelectionFn :=
    switchingActionSelectionWrapper rootActionSelectionFn interiorActionSelectionFn
  let depthCutoff := maxDepth.getD tree.numSimulations
  let mut tree := tree

  let mut sim := 0
  while sim < numSimulations do
    let key := rngKey + UInt64.ofNat (sim + 1)
    let (parentIndex, action) := simulate key tree actionSelectionFn depthCutoff
    let nextNodeIndex :=
      ((tree.childrenIndex.getD parentIndex #[]).getD action none).getD tree.nextNodeIndex
    let embedding := tree.embeddings.getD parentIndex default
    let (step, nextEmbedding) := recurrentFn params key action embedding
    tree := expandWithStepAndBackup tree parentIndex action nextNodeIndex step nextEmbedding
    sim := sim + 1

  return tree

/-- Runs full MCTS search (unbatched Lean port). -/
def search
    [Inhabited S]
    [Inhabited E]
    (params : P)
    (rngKey : UInt64)
    (root : RootFnOutput S)
    (recurrentFn : RecurrentFn P S)
    (rootActionSelectionFn : RootActionSelectionFn S E)
    (interiorActionSelectionFn : InteriorActionSelectionFn S E)
    (numSimulations : Nat)
    (maxDepth : Option Nat := none)
    (invalidActions : Option (Array Bool) := none)
    (extraData : E := default)
    : Tree S E := Id.run do
  let rootInvalid := invalidActions.getD (Array.replicate root.priorLogits.size false)
  let tree := instantiateTreeFromRoot root numSimulations rootInvalid extraData
  searchWithTree params rngKey tree recurrentFn rootActionSelectionFn interiorActionSelectionFn
    numSimulations maxDepth

end torch.mctx
