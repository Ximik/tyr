import Tyr.MctxDag.Tree

/-!
# Tyr.MctxDag.QTransforms

Q-value transforms for `DagTree`, mirroring `Tyr.Mctx.QTransforms`:
min/max and parent/siblings normalization, the completed-Q transform, and
the Gumbel MuZero mixed value.
-/

namespace torch.mctxdag

private def completeQvalues (qvalues : Array Float) (visitCounts : Array Nat) (value : Float) : Array Float :=
  (List.range qvalues.size).toArray.map fun i =>
    if visitCounts.getD i 0 > 0 then qvalues.getD i 0.0 else value

private def rescaleQvalues (qvalues : Array Float) (epsilon : Float := 1e-8) : Array Float :=
  if qvalues.isEmpty then
    #[]
  else
    let minV := qvalues.foldl (init := qvalues.getD 0 0.0) fun acc x => if x < acc then x else acc
    let maxV := qvalues.foldl (init := qvalues.getD 0 0.0) fun acc x => if x > acc then x else acc
    let denom := if Float.abs (maxV - minV) < epsilon then epsilon else (maxV - minV)
    qvalues.map (fun q => (q - minV) / denom)

/-- Mixed value from Appendix D of Gumbel MuZero. -/
def computeMixedValue
    (rawValue : Float)
    (qvalues : Array Float)
    (visitCounts : Array Nat)
    (priorProbs : Array Float)
    : Float :=
  let sumVisitCounts : Float := (visitCounts.foldl (init := 0) (· + ·)).toFloat
  let tiny : Float := 1e-30
  let priorSafe := priorProbs.map (fun p => if p < tiny then tiny else p)
  let sumProbs := (List.range priorSafe.size).foldl (init := 0.0) fun acc i =>
    if visitCounts.getD i 0 > 0 then acc + priorSafe.getD i 0.0 else acc
  let denom := if sumProbs <= tiny then 1.0 else sumProbs
  let weightedQ := (List.range qvalues.size).foldl (init := 0.0) fun acc i =>
    if visitCounts.getD i 0 > 0 then
      acc + priorSafe.getD i 0.0 * qvalues.getD i 0.0 / denom
    else
      acc
  (rawValue + sumVisitCounts * weightedQ) / (sumVisitCounts + 1.0)

/-- Q-transform using known global min/max value bounds. -/
def qtransformByMinMax
    [BEq K]
    [Hashable K]
    (tree : DagTree S K E)
    (nodeIndex : Nat)
    (minValue maxValue : Float)
    : Array Float :=
  let qvalues := tree.qvalues nodeIndex
  let visits := tree.childrenVisits.getD nodeIndex #[]
  let denom :=
    let d := maxValue - minValue
    if Float.abs d < 1e-8 then 1e-8 else d
  (List.range qvalues.size).toArray.map fun i =>
    let q := if visits.getD i 0 > 0 then qvalues.getD i 0.0 else minValue
    (q - minValue) / denom

/-- Q-transform normalized by parent value and sibling extrema. -/
def qtransformByParentAndSiblings
    [BEq K]
    [Hashable K]
    (tree : DagTree S K E)
    (nodeIndex : Nat)
    (epsilon : Float := 1e-8)
    : Array Float :=
  let qvalues := tree.qvalues nodeIndex
  let visits := tree.childrenVisits.getD nodeIndex #[]
  let nodeValue := tree.nodeValues.getD nodeIndex 0.0
  let safeQ := (List.range qvalues.size).toArray.map fun i =>
    if visits.getD i 0 > 0 then qvalues.getD i 0.0 else nodeValue
  let minValue :=
    let m := safeQ.foldl (init := nodeValue) fun acc x => if x < acc then x else acc
    if nodeValue < m then nodeValue else m
  let maxValue :=
    let m := safeQ.foldl (init := nodeValue) fun acc x => if x > acc then x else acc
    if nodeValue > m then nodeValue else m
  let denom := if Float.abs (maxValue - minValue) < epsilon then epsilon else (maxValue - minValue)
  (List.range qvalues.size).toArray.map fun i =>
    let q := if visits.getD i 0 > 0 then qvalues.getD i 0.0 else minValue
    (q - minValue) / denom

/-- Completed-Q transform used by Gumbel MuZero style scoring. -/
def qtransformCompletedByMixValue
    [BEq K]
    [Hashable K]
    (tree : DagTree S K E)
    (nodeIndex : Nat)
    (valueScale : Float := 0.1)
    (maxvisitInit : Float := 50.0)
    (rescaleValues : Bool := true)
    (useMixedValue : Bool := true)
    (epsilon : Float := 1e-8)
    : Array Float :=
  let qvalues := tree.qvalues nodeIndex
  let visitCounts := tree.childrenVisits.getD nodeIndex #[]
  let rawValue := tree.rawValues.getD nodeIndex 0.0
  let priorProbs := softmax (tree.childrenPriorLogits.getD nodeIndex #[])
  let value :=
    if useMixedValue then
      computeMixedValue rawValue qvalues visitCounts priorProbs
    else
      rawValue
  let completed := completeQvalues qvalues visitCounts value
  let completed := if rescaleValues then rescaleQvalues completed epsilon else completed
  let maxVisit := (visitCounts.foldl (init := 0) fun acc c => if c > acc then c else acc).toFloat
  let visitScale := maxvisitInit + maxVisit
  completed.map (fun q => visitScale * valueScale * q)

end torch.mctxdag
