import Tyr.Mctx.Policies
import Tyr.Mctx.Sampling

/-!
# Tyr.Mctx.Batched

Batched MCTS API: `BatchedTree` holds one `Tree` per batch element, with
batched search, tree reset and subtree extraction, and batched
MuZero/AlphaZero/Gumbel-MuZero policies.
-/

namespace torch.mctx

structure BatchedTree (S E : Type) where
  trees : Array (Tree S E)
  deriving Repr

/-- Number of batch elements in the tree. -/
def BatchedTree.batchSize (tree : BatchedTree S E) : Nat :=
  tree.trees.size

/-- Root summaries for each batch element. -/
def BatchedTree.summary (tree : BatchedTree S E) : BatchedSearchSummary :=
  let summaries := tree.trees.map Tree.summary
  {
    visitCounts := summaries.map (·.visitCounts)
    visitProbs := summaries.map (·.visitProbs)
    value := summaries.map (·.value)
    qvalues := summaries.map (·.qvalues)
  }


private def treeAt! [Inhabited S] [Inhabited E] (trees : Array (Tree S E)) (i : Nat) : Tree S E :=
  match trees[i]? with
  | some t => t
  | none => panic! s!"batched tree index out of bounds: {i}"

private def invalidRow
    (invalidActions : Option (Array (Array Bool)))
    (batchIdx : Nat)
    (numActions : Nat)
    : Array Bool :=
  match invalidActions with
  | none => Array.replicate numActions false
  | some rows => rows.getD batchIdx (Array.replicate numActions false)

private def invalidRowOpt
    (invalidActions : Option (Array (Array Bool)))
    (batchIdx : Nat)
    : Option (Array Bool) :=
  match invalidActions with
  | none => none
  | some rows => some (rows.getD batchIdx #[])

private def batchedRootAt [Inhabited S] (root : BatchedRootFnOutput S) (batchIdx : Nat) : RootFnOutput S := {
  priorLogits := root.priorLogits.getD batchIdx #[]
  value := root.value.getD batchIdx 0.0
  embedding := root.embedding.getD batchIdx default
}

/-- Batched search continuation from a pre-initialized tree array. -/
def searchBatchedWithTrees
    [Inhabited S]
    [Inhabited E]
    (params : P)
    (rngKey : UInt64)
    (trees : Array (Tree S E))
    (recurrentFn : BatchedRecurrentFn P S)
    (rootActionSelectionFn : RootActionSelectionFn S E)
    (interiorActionSelectionFn : InteriorActionSelectionFn S E)
    (numSimulations : Nat)
    (maxDepth : Option Nat := none)
    : BatchedTree S E := Id.run do
  let batchSize := trees.size
  let actionSelectionFn :=
    switchingActionSelectionWrapper rootActionSelectionFn interiorActionSelectionFn
  let mut trees := trees
  let mut sim := 0
  while sim < numSimulations do
    let mut parentIndices : Array Nat := Array.mkEmpty batchSize
    let mut actions : Array Action := Array.mkEmpty batchSize
    let mut nextNodeIndices : Array Nat := Array.mkEmpty batchSize
    let mut parentEmbeddings : Array S := Array.mkEmpty batchSize

    for bi in [:batchSize] do
      let tree := treeAt! trees bi
      let depthCutoff := maxDepth.getD tree.numSimulations
      let simKey := rngKey + UInt64.ofNat ((sim + 1) * 1315423911 + bi)
      let (parentIndex, action) := simulate simKey tree actionSelectionFn depthCutoff
      let nextNodeIndex :=
        ((tree.childrenIndex.getD parentIndex #[]).getD action none).getD tree.nextNodeIndex
      parentIndices := parentIndices.push parentIndex
      actions := actions.push action
      nextNodeIndices := nextNodeIndices.push nextNodeIndex
      parentEmbeddings := parentEmbeddings.push (tree.embeddings.getD parentIndex default)

    let recKey := rngKey + UInt64.ofNat (sim + 1)
    let (stepBatch, nextEmbeddings) := recurrentFn params recKey actions parentEmbeddings

    for bi in [:batchSize] do
      let tree := treeAt! trees bi
      let parentIndex := parentIndices.getD bi ROOT_INDEX
      let action := actions.getD bi 0
      let nextNodeIndex := nextNodeIndices.getD bi tree.nodeVisits.size
      let step : RecurrentFnOutput := {
        reward := stepBatch.reward.getD bi 0.0
        discount := stepBatch.discount.getD bi 0.0
        priorLogits := stepBatch.priorLogits.getD bi #[]
        value := stepBatch.value.getD bi 0.0
      }
      let nextEmbedding := nextEmbeddings.getD bi default
      let tree := expandWithStepAndBackup tree parentIndex action nextNodeIndex step nextEmbedding
      trees := trees.set! bi tree

    sim := sim + 1

  return { trees := trees }

/-- Batched MCTS search. The implementation is batched at the API level and
    runs one recurrent function call per simulation with all batch elements. -/
def searchBatched
    [Inhabited S]
    [Inhabited E]
    (params : P)
    (rngKey : UInt64)
    (root : BatchedRootFnOutput S)
    (recurrentFn : BatchedRecurrentFn P S)
    (rootActionSelectionFn : RootActionSelectionFn S E)
    (interiorActionSelectionFn : InteriorActionSelectionFn S E)
    (numSimulations : Nat)
    (maxDepth : Option Nat := none)
    (invalidActions : Option (Array (Array Bool)) := none)
    (extraData : Option (Array E) := none)
    : BatchedTree S E := Id.run do
  let batchSize := root.value.size
  let depthCutoff := maxDepth.getD numSimulations

  let mut trees : Array (Tree S E) := Array.mkEmpty batchSize
  for bi in [:batchSize] do
    let rootRow := batchedRootAt root bi
    let invalid := invalidRow invalidActions bi rootRow.priorLogits.size
    let extra :=
      match extraData with
      | some rows => rows.getD bi default
      | none => default
    let tree := instantiateTreeFromRoot rootRow numSimulations invalid extra
    trees := trees.push tree

  return searchBatchedWithTrees
    params rngKey trees recurrentFn rootActionSelectionFn interiorActionSelectionFn
    numSimulations (some depthCutoff)

/-- Resets batched search trees to empty/unvisited state.
    If `selectBatch` is provided, only selected rows are reset. -/
def resetSearchTreeBatched
    [Inhabited S]
    [Inhabited E]
    (tree : BatchedTree S E)
    (selectBatch : Option (Array Bool) := none)
    : BatchedTree S E :=
  let trees := (List.range tree.trees.size).toArray.map fun bi =>
    let t := treeAt! tree.trees bi
    match selectBatch with
    | none => resetSearchTree t
    | some sel =>
      if sel.getD bi false then resetSearchTree t else t
  { trees := trees }

/-- Extracts one subtree per batch element using per-row root child actions. -/
def getSubtreeBatched
    [Inhabited S]
    [Inhabited E]
    (tree : BatchedTree S E)
    (childActions : Array Nat)
    : BatchedTree S E :=
  let trees := (List.range tree.trees.size).toArray.map fun bi =>
    let t := treeAt! tree.trees bi
    getSubtree t (childActions.getD bi 0)
  { trees := trees }

private def addArrays (a b : Array Float) : Array Float :=
  (List.range a.size).toArray.map fun i => a.getD i 0.0 + b.getD i 0.0

private def maskInvalidActionsRow
    (logits : Array Float)
    (invalidActions : Option (Array Bool))
    : Array Float :=
  maskInvalidActions logits invalidActions

/-- Batched MuZero policy. -/
def muzeroPolicyBatched
    [Inhabited S]
    (params : P)
    (rngKey : UInt64)
    (root : BatchedRootFnOutput S)
    (recurrentFn : BatchedRecurrentFn P S)
    (numSimulations : Nat)
    (invalidActions : Option (Array (Array Bool)) := none)
    (maxDepth : Option Nat := none)
    (qtransform : QTransform S Unit := qtransformByParentAndSiblings)
    (dirichletFraction : Float := 0.25)
    (dirichletAlpha : Float := 0.3)
    (pbCInit : Float := 1.25)
    (pbCBase : Float := 19652.0)
    (temperature : Float := 1.0)
    : BatchedPolicyOutput (BatchedTree S Unit) :=
  let batchSize := root.value.size
  let noisyPrior := (List.range batchSize).toArray.map fun bi =>
    let row := root.priorLogits.getD bi #[]
    let rowKey := Sampling.splitKey rngKey (UInt64.ofNat bi + 3)
    Sampling.rootLogits (Sampling.splitKey rowKey 0) row (invalidRowOpt invalidActions bi)
      dirichletFraction dirichletAlpha

  let root : BatchedRootFnOutput S := {
    priorLogits := noisyPrior
    value := root.value
    embedding := root.embedding
  }

  let interiorFn : InteriorActionSelectionFn S Unit := fun _ tree nodeIndex depth =>
    muzeroActionSelection tree nodeIndex depth qtransform pbCInit pbCBase
  let rootFn : RootActionSelectionFn S Unit := fun _ tree nodeIndex =>
    interiorFn 0 tree nodeIndex 0

  let searchTree := searchBatched
    params (Sampling.splitKey rngKey 1) root recurrentFn rootFn interiorFn
    numSimulations maxDepth invalidActions

  let summary := searchTree.summary
  let actionWeights := (List.range batchSize).toArray.map fun bi =>
    Sampling.normalizeWeights (summary.visitProbs.getD bi #[]) (invalidRowOpt invalidActions bi)
  let actions := (List.range batchSize).toArray.map fun bi =>
    let rowKey := Sampling.splitKey rngKey (UInt64.ofNat bi + 3)
    Sampling.sampleAction (Sampling.splitKey rowKey 2) (actionWeights.getD bi #[])
      temperature (invalidRowOpt invalidActions bi)

  {
    action := actions
    actionWeights := actionWeights
    searchTree := searchTree
  }

/-- Batched AlphaZero-style policy with optional tree continuation. -/
def alphazeroPolicyBatched
    [Inhabited S]
    (params : P)
    (rngKey : UInt64)
    (root : BatchedRootFnOutput S)
    (recurrentFn : BatchedRecurrentFn P S)
    (numSimulations : Nat)
    (searchTree : Option (BatchedTree S Unit) := none)
    (maxNodes : Option Nat := none)
    (invalidActions : Option (Array (Array Bool)) := none)
    (maxDepth : Option Nat := none)
    (qtransform : QTransform S Unit := qtransformByParentAndSiblings)
    (dirichletFraction : Float := 0.25)
    (dirichletAlpha : Float := 0.3)
    (pbCInit : Float := 1.25)
    (pbCBase : Float := 19652.0)
    (temperature : Float := 1.0)
    : BatchedPolicyOutput (BatchedTree S Unit) :=
  let batchSize := root.value.size
  let noisyPrior := (List.range batchSize).toArray.map fun bi =>
    let row := root.priorLogits.getD bi #[]
    let rowKey := Sampling.splitKey rngKey (UInt64.ofNat bi + 3)
    Sampling.rootLogits (Sampling.splitKey rowKey 0) row (invalidRowOpt invalidActions bi)
      dirichletFraction dirichletAlpha

  let root : BatchedRootFnOutput S := {
    priorLogits := noisyPrior
    value := root.value
    embedding := root.embedding
  }

  let interiorFn : InteriorActionSelectionFn S Unit := fun _ tree nodeIndex depth =>
    muzeroActionSelection tree nodeIndex depth qtransform pbCInit pbCBase
  let rootFn : RootActionSelectionFn S Unit := fun _ tree nodeIndex =>
    interiorFn 0 tree nodeIndex 0

  let capacity := maxNodes.getD (numSimulations + 1)
  let initialTrees : Array (Tree S Unit) :=
    match searchTree with
    | none =>
      (List.range batchSize).toArray.map fun bi =>
        let rootRow := batchedRootAt root bi
        let invalid := invalidRow invalidActions bi rootRow.priorLogits.size
        instantiateTreeFromRootWithCapacity rootRow capacity invalid ()
    | some bt =>
      (List.range batchSize).toArray.map fun bi =>
        let rootRow := batchedRootAt root bi
        let invalid := invalidRow invalidActions bi rootRow.priorLogits.size
        let fallback := instantiateTreeFromRootWithCapacity rootRow capacity invalid ()
        let existing := bt.trees.getD bi fallback
        updateTreeWithRoot existing rootRow invalid ()

  let searchTree := searchBatchedWithTrees
    params (Sampling.splitKey rngKey 1) initialTrees recurrentFn rootFn interiorFn numSimulations maxDepth

  let summary := searchTree.summary
  let actionWeights := (List.range batchSize).toArray.map fun bi =>
    Sampling.normalizeWeights (summary.visitProbs.getD bi #[]) (invalidRowOpt invalidActions bi)
  let actions := (List.range batchSize).toArray.map fun bi =>
    let rowKey := Sampling.splitKey rngKey (UInt64.ofNat bi + 3)
    Sampling.sampleAction (Sampling.splitKey rowKey 2) (actionWeights.getD bi #[])
      temperature (invalidRowOpt invalidActions bi)

  {
    action := actions
    actionWeights := actionWeights
    searchTree := searchTree
  }

/-- Batched Gumbel MuZero policy. -/
def gumbelMuZeroPolicyBatched
    [Inhabited S]
    (params : P)
    (rngKey : UInt64)
    (root : BatchedRootFnOutput S)
    (recurrentFn : BatchedRecurrentFn P S)
    (numSimulations : Nat)
    (invalidActions : Option (Array (Array Bool)) := none)
    (maxDepth : Option Nat := none)
    (qtransform : QTransform S GumbelMuZeroExtraData := qtransformCompletedByMixValue)
    (maxNumConsideredActions : Nat := 16)
    (gumbelScale : Float := 1.0)
    : BatchedPolicyOutput (BatchedTree S GumbelMuZeroExtraData) :=
  let batchSize := root.value.size

  let maskedPrior := (List.range batchSize).toArray.map fun bi =>
    let row := root.priorLogits.getD bi #[]
    maskInvalidActionsRow row (invalidRowOpt invalidActions bi)

  let root : BatchedRootFnOutput S := {
    priorLogits := maskedPrior
    value := root.value
    embedding := root.embedding
  }

  let gumbels := (List.range batchSize).toArray.map fun bi =>
    let rowKey := Sampling.splitKey rngKey (UInt64.ofNat bi + 3)
    Sampling.gumbel (Sampling.splitKey rowKey 0) (maskedPrior.getD bi #[]).size gumbelScale
  let extras : Array GumbelMuZeroExtraData := gumbels.mapIdx fun bi g =>
    let numConsidered := countConsideredActions maxNumConsideredActions g.size
      ((invalidRowOpt invalidActions bi).getD #[])
    { rootGumbel := g
      consideredVisitSchedule := some (ConsideredVisitSchedule.create numConsidered numSimulations) }

  let rootFn : RootActionSelectionFn S GumbelMuZeroExtraData := fun _ tree nodeIndex =>
    gumbelMuZeroRootActionSelection tree nodeIndex numSimulations maxNumConsideredActions qtransform
  let interiorFn : InteriorActionSelectionFn S GumbelMuZeroExtraData := fun _ tree nodeIndex _depth =>
    gumbelMuZeroInteriorActionSelection tree nodeIndex qtransform

  let searchTree := searchBatched
    params (Sampling.splitKey rngKey 1) root recurrentFn rootFn interiorFn
    numSimulations maxDepth invalidActions (extraData := some extras)

  let summary := searchTree.summary

  let actions := (List.range batchSize).toArray.map fun bi =>
    let tree := treeAt! searchTree.trees bi
    let visitCounts := summary.visitCounts.getD bi #[]
    let consideredVisit := visitCounts.foldl (init := 0) fun acc c => if c > acc then c else acc
    let completedQvalues := qtransform tree ROOT_INDEX
    let gumbel := gumbels.getD bi #[]
    let prior := maskedPrior.getD bi #[]
    let toArgmax := scoreConsidered consideredVisit gumbel prior completedQvalues visitCounts
    maskedArgmax toArgmax (invalidRowOpt invalidActions bi)

  let actionWeights := (List.range batchSize).toArray.map fun bi =>
    let tree := treeAt! searchTree.trees bi
    let completedQvalues := qtransform tree ROOT_INDEX
    let logits := addArrays (maskedPrior.getD bi #[]) completedQvalues
    let logits := maskInvalidActionsRow logits (invalidRowOpt invalidActions bi)
    Sampling.normalizeWeights (softmax logits) (invalidRowOpt invalidActions bi)

  {
    action := actions
    actionWeights := actionWeights
    searchTree := searchTree
  }

end torch.mctx
