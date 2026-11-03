import Tyr.MctxDag.QTransforms
import Tyr.Mctx.SeqHalving

/-!
# Tyr.MctxDag.ActionSelection

Action selection for DAG search over `DagTree`, mirroring
`Tyr.Mctx.ActionSelection`: PUCT (`muzeroActionSelection`) and Gumbel
MuZero root/interior selectors via sequential halving.
-/

namespace torch.mctxdag

/-- Q-transform function type. -/
abbrev QTransform (S K E : Type) [BEq K] [Hashable K] := DagTree S K E → NodeIndex → Array Float

/-- Root action selector signature. -/
abbrev RootActionSelectionFn (S K E : Type) [BEq K] [Hashable K] :=
  UInt64 → DagTree S K E → NodeIndex → Action

/-- Interior action selector signature. -/
abbrev InteriorActionSelectionFn (S K E : Type) [BEq K] [Hashable K] :=
  UInt64 → DagTree S K E → NodeIndex → Depth → Action

/-- Switches between root and interior selection by depth. -/
def switchingActionSelectionWrapper
    [BEq K]
    [Hashable K]
    (rootFn : RootActionSelectionFn S K E)
    (interiorFn : InteriorActionSelectionFn S K E)
    : InteriorActionSelectionFn S K E :=
  fun rng tree nodeIndex depth =>
    if depth = 0 then rootFn rng tree nodeIndex else interiorFn rng tree nodeIndex depth

private def addArrays (a b : Array Float) : Array Float :=
  (List.range a.size).toArray.map fun i => a.getD i 0.0 + b.getD i 0.0

/-- PUCT action selection used by MuZero. -/
def muzeroActionSelection
    [BEq K]
    [Hashable K]
    (tree : DagTree S K E)
    (nodeIndex : NodeIndex)
    (depth : Depth)
    (qtransform : QTransform S K E := qtransformByParentAndSiblings)
    (pbCInit : Float := 1.25)
    (pbCBase : Float := 19652.0)
    : Action :=
  let visitCounts := tree.childrenVisits.getD nodeIndex #[]
  let nodeVisit := Float.ofNat (tree.nodeVisits.getD nodeIndex 0)
  let pbC := pbCInit + Float.log ((nodeVisit + pbCBase + 1.0) / pbCBase)
  let priorProbs := softmax (tree.childrenPriorLogits.getD nodeIndex #[])
  let policyScore := (List.range priorProbs.size).toArray.map fun i =>
    Float.sqrt nodeVisit * pbC * priorProbs.getD i 0.0 /
      ((visitCounts.getD i 0).toFloat + 1.0)
  let valueScore := qtransform tree nodeIndex
  let toArgmax := addArrays valueScore policyScore
  let invalid := if depth = 0 then some tree.rootInvalidActions else none
  maskedArgmax toArgmax invalid

/-- Extra search metadata for Gumbel MuZero (DAG variant). -/
structure GumbelMuZeroExtraData where
  rootGumbel : Array Float
  consideredVisitSchedule : Option torch.mctx.ConsideredVisitSchedule := none
  deriving Repr, Inhabited

private def prepareArgmaxInput (probs : Array Float) (visitCounts : Array Nat) : Array Float :=
  let total := (visitCounts.foldl (init := 0) (· + ·)).toFloat
  (List.range probs.size).toArray.map fun i =>
    probs.getD i 0.0 - (visitCounts.getD i 0).toFloat / (1.0 + total)

/-- Root action selection for DAG Gumbel MuZero using sequential halving. -/
def gumbelMuZeroRootActionSelection
    [BEq K]
    [Hashable K]
    (tree : DagTree S K GumbelMuZeroExtraData)
    (nodeIndex : NodeIndex)
    (numSimulations : Nat)
    (maxNumConsideredActions : Nat)
    (qtransform : QTransform S K GumbelMuZeroExtraData := qtransformCompletedByMixValue)
    : Action :=
  let visitCounts := tree.childrenVisits.getD nodeIndex #[]
  let priorLogits := tree.childrenPriorLogits.getD nodeIndex #[]
  let completedQvalues := qtransform tree nodeIndex
  let numConsidered := torch.mctx.countConsideredActions
    maxNumConsideredActions priorLogits.size tree.rootInvalidActions
  let simulationIndex := visitCounts.foldl (init := 0) (· + ·)
  let consideredVisit := torch.mctx.consideredVisitAt tree.extraData.consideredVisitSchedule
    numConsidered numSimulations simulationIndex
  let toArgmax :=
    torch.mctx.scoreConsidered consideredVisit tree.extraData.rootGumbel priorLogits completedQvalues visitCounts
  maskedArgmax toArgmax (some tree.rootInvalidActions)

/-- Deterministic interior action selection for DAG Gumbel MuZero. -/
def gumbelMuZeroInteriorActionSelection
    [BEq K]
    [Hashable K]
    (tree : DagTree S K E)
    (nodeIndex : NodeIndex)
    (qtransform : QTransform S K E := qtransformCompletedByMixValue)
    : Action :=
  let visitCounts := tree.childrenVisits.getD nodeIndex #[]
  let logits := tree.childrenPriorLogits.getD nodeIndex #[]
  let completedQ := qtransform tree nodeIndex
  let probs := softmax (addArrays logits completedQ)
  let toArgmax := prepareArgmaxInput probs visitCounts
  argmax toArgmax

end torch.mctxdag
