import Tyr.Mctx.QTransforms
import Tyr.Mctx.SeqHalving

/-!
# Tyr.Mctx.ActionSelection

Action selection inside the search loop: PUCT (`muzeroActionSelection`),
Gumbel MuZero root/interior selection via sequential halving, and the
depth-switching wrapper.
-/

namespace torch.mctx

/-- Q-transform function type. -/
abbrev QTransform (S E : Type) := Tree S E → NodeIndex → Array Float

/-- Root action selector signature. -/
abbrev RootActionSelectionFn (S E : Type) :=
  UInt64 → Tree S E → NodeIndex → Action

/-- Interior action selector signature. -/
abbrev InteriorActionSelectionFn (S E : Type) :=
  UInt64 → Tree S E → NodeIndex → Depth → Action

/-- Switches between root and interior selection by depth. -/
def switchingActionSelectionWrapper
    (rootFn : RootActionSelectionFn S E)
    (interiorFn : InteriorActionSelectionFn S E)
    : InteriorActionSelectionFn S E :=
  fun rng tree nodeIndex depth =>
    if depth = 0 then rootFn rng tree nodeIndex else interiorFn rng tree nodeIndex depth

private def addArrays (a b : Array Float) : Array Float :=
  (List.range a.size).toArray.map fun i => a.getD i 0.0 + b.getD i 0.0

/-- PUCT action selection used by MuZero. -/
def muzeroActionSelection
    (tree : Tree S E)
    (nodeIndex : NodeIndex)
    (depth : Depth)
    (qtransform : QTransform S E := qtransformByParentAndSiblings)
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

/-- Extra search metadata for Gumbel MuZero. -/
structure GumbelMuZeroExtraData where
  rootGumbel : Array Float
  consideredVisitSchedule : Option ConsideredVisitSchedule := none
  deriving Repr, Inhabited

private def prepareArgmaxInput (probs : Array Float) (visitCounts : Array Nat) : Array Float :=
  let total := (visitCounts.foldl (init := 0) (· + ·)).toFloat
  (List.range probs.size).toArray.map fun i =>
    probs.getD i 0.0 - (visitCounts.getD i 0).toFloat / (1.0 + total)

/-- Root action selection for Gumbel MuZero using sequential halving. -/
def gumbelMuZeroRootActionSelection
    (tree : Tree S GumbelMuZeroExtraData)
    (nodeIndex : NodeIndex)
    (numSimulations : Nat)
    (maxNumConsideredActions : Nat)
    (qtransform : QTransform S GumbelMuZeroExtraData := qtransformCompletedByMixValue)
    : Action :=
  let visitCounts := tree.childrenVisits.getD nodeIndex #[]
  let priorLogits := tree.childrenPriorLogits.getD nodeIndex #[]
  let completedQvalues := qtransform tree nodeIndex
  let numConsidered := countConsideredActions
    maxNumConsideredActions priorLogits.size tree.rootInvalidActions
  let simulationIndex := visitCounts.foldl (init := 0) (· + ·)
  let consideredVisit := consideredVisitAt tree.extraData.consideredVisitSchedule
    numConsidered numSimulations simulationIndex
  let toArgmax := scoreConsidered consideredVisit tree.extraData.rootGumbel priorLogits completedQvalues visitCounts
  maskedArgmax toArgmax (some tree.rootInvalidActions)

/-- Deterministic interior action selection for Gumbel MuZero. -/
def gumbelMuZeroInteriorActionSelection
    (tree : Tree S E)
    (nodeIndex : NodeIndex)
    (qtransform : QTransform S E := qtransformCompletedByMixValue)
    : Action :=
  let visitCounts := tree.childrenVisits.getD nodeIndex #[]
  let logits := tree.childrenPriorLogits.getD nodeIndex #[]
  let completedQ := qtransform tree nodeIndex
  let probs := softmax (addArrays logits completedQ)
  let toArgmax := prepareArgmaxInput probs visitCounts
  argmax toArgmax

end torch.mctx
