import Std
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Model.Flowfusion

/-!
  BranchingFlows-style abstractions (Lean port, minimal core).

  This module focuses on the combinatorial/structural pieces:
  - coalescence policies
  - coalescent forest sampling
  - branching state bookkeeping
  - simple RNG utilities

  It intentionally leaves base-process specifics (bridge/step/loss) as
  user-supplied functions. This keeps the API usable without a full
  Flowfusion port.
-/

namespace torch.branching

universe u

/-! ## RNG utilities (deterministic LCG) -/

structure Rng where
  state : UInt64
  deriving Repr

private def lcgNext (s : UInt64) : UInt64 :=
  s * 6364136223846793005 + 1442695040888963407

def Rng.next (r : Rng) : Rng := { state := lcgNext r.state }

def randUInt64 (r : Rng) : UInt64 × Rng :=
  let s := lcgNext r.state
  (s, { state := s })

private def u64Denom : Float :=
  Float.ofNat (Nat.pow 2 64)

def randFloat (r : Rng) : Float × Rng :=
  let (u, r') := randUInt64 r
  let f := (Float.ofNat u.toNat) / u64Denom
  (f, r')

def randNat (r : Rng) (n : Nat) : Nat × Rng :=
  if n = 0 then
    (0, r)
  else
    let (u, r') := randUInt64 r
    let n64 := UInt64.ofNat n
    let v := (u % n64).toNat
    (v, r')

def randBool (r : Rng) : Bool × Rng :=
  let (u, r') := randUInt64 r
  (u % 2 = 0, r')

def randBernoulli (r : Rng) (p : Float) : Bool × Rng :=
  let (u, r') := randFloat r
  (u < p, r')

def randBinomial (r : Rng) (n : Nat) (p : Float) : Nat × Rng := Id.run do
  let mut count := 0
  let mut rng := r
  for _ in [:n] do
    let (b, rng') := randBernoulli rng p
    rng := rng'
    if b then
      count := count + 1
  return (count, rng)

def randPoisson (r : Rng) (lambda : Float) : Nat × Rng :=
  if lambda <= 0 then
    (0, r)
  else
    Id.run do
    let L := Float.exp (-lambda)
    let mut k := 0
    let mut p := 1.0
    let mut rng := r
    while p > L do
      k := k + 1
      let (u, rng') := randFloat rng
      rng := rng'
      p := p * u
    return (Nat.pred k, rng)

def randExponential (r : Rng) : Float × Rng :=
  let (u, r') := randFloat r
  let u := if u <= 1e-12 then 1e-12 else u
  (-Float.log u, r')

def randNormal (r : Rng) : Float × Rng :=
  let (u1, r') := randFloat r
  let (u2, r'') := randFloat r'
  let u1 := if u1 <= 1.0e-12 then 1.0e-12 else u1
  let radius := Float.sqrt (-2.0 * Float.log u1)
  let angle := 2.0 * 3.14159265358979323846 * u2
  (radius * Float.cos angle, r'')

/-! ## Time distributions -/

structure TimeDist where
  cdf : Float → Float
  pdf : Float → Float
  quantile : Float → Float

namespace TimeDist

private def clamp01 (x : Float) : Float :=
  max 0.0 (min x 1.0)

private def bisectQuantile (cdf : Float → Float) (p : Float) (steps : Nat := 48) : Float := Id.run do
  let target := clamp01 p
  let mut lo := 0.0
  let mut hi := 1.0
  for _ in [:steps] do
    let mid := (lo + hi) / 2.0
    if cdf mid < target then
      lo := mid
    else
      hi := mid
  return (lo + hi) / 2.0

def uniform : TimeDist :=
  { cdf := fun t => clamp01 t,
    pdf := fun t => if t < 0.0 || t > 1.0 then 0.0 else 1.0,
    quantile := fun p => clamp01 p }

/-- Closed-form `Beta(1, 2)` distribution on `[0, 1]`, used by the Julia demo. -/
def betaOneTwo : TimeDist :=
  { cdf := fun t =>
      let x := clamp01 t
      1.0 - Float.pow (1.0 - x) 2.0,
    pdf := fun t =>
      if t < 0.0 || t > 1.0 then 0.0 else 2.0 * (1.0 - t),
    quantile := fun p =>
      let q := clamp01 p
      1.0 - Float.pow (1.0 - q) (1.0 / 2.0) }

/-- Closed-form `Beta(1, 3/2)` split-time distribution used by the QM9 setup. -/
def betaOneThreeHalves : TimeDist :=
  { cdf := fun t =>
      let x := clamp01 t
      1.0 - Float.pow (1.0 - x) (3.0 / 2.0),
    pdf := fun t =>
      if t < 0.0 || t > 1.0 then 0.0 else (3.0 / 2.0) * Float.pow (1.0 - t) (1.0 / 2.0),
    quantile := fun p =>
      let q := clamp01 p
      1.0 - Float.pow (1.0 - q) (2.0 / 3.0) }

/-- `Beta(2, 2)` distribution on `[0, 1]`, used by the QM9 DFM hazards. -/
def betaTwoTwo : TimeDist :=
  let cdf := fun t =>
    let x := clamp01 t
    3.0 * x * x - 2.0 * x * x * x
  { cdf := cdf,
    pdf := fun t =>
      if t < 0.0 || t > 1.0 then 0.0 else 6.0 * t * (1.0 - t),
    quantile := fun p => bisectQuantile cdf p }

def survival (dist : TimeDist) (t : Float) : Float :=
  max (1.0 - dist.cdf t) 0.0

def hazard (dist : TimeDist) (t : Float) : Float :=
  let s := dist.survival t
  if s > 0.0 then dist.pdf t / s else 0.0

/-- Density at `t` after truncating a time distribution to `[start, 1]`. -/
def truncatedPdfFrom (dist : TimeDist) (start t : Float) : Float :=
  let s := dist.survival start
  if s > 0.0 then dist.pdf t / s else 0.0

end TimeDist

private def clampProbability (p : Float) : Float :=
  max 0.0 (min p 1.0)

def defaultSplitTransform (x : Float) : Float :=
  Float.exp (max (-100.0) (min x 11.0))

private def sigmoid (x : Float) : Float :=
  1.0 / (1.0 + Float.exp (-x))

/-! ## Training loss helpers -/

def xlogy (x y : Float) : Float :=
  if x == 0.0 then 0.0 else x * Float.log y

def poissonBregmanLoss (mu count : Float) : Float :=
  mu - xlogy count mu

/-- Shifted Poisson Bregman loss used by `BranchingFlows.jl` for split counts. -/
def shiftedPoissonBregmanLoss (mu count : Float) : Float :=
  poissonBregmanLoss mu count - (count - xlogy count count)

def splitCountLoss (splitTransform : Float → Float) (logit : Float) (count : Nat) : Float :=
  shiftedPoissonBregmanLoss (splitTransform logit) count.toFloat

private def relu (x : Float) : Float :=
  if x < 0.0 then 0.0 else x

def softplus (x : Float) : Float :=
  Float.log (1.0 + Float.exp (-(Float.abs x))) + relu x

def logSigmoid (x : Float) : Float :=
  -softplus (-x)

/-- Stable logit binary cross entropy, matching the Julia deletion loss. -/
def logitBinaryCrossEntropy (logit : Float) (target : Bool) : Float :=
  let y := if target then 1.0 else 0.0
  (1.0 - y) * logit - logSigmoid logit

def maskedScaledMean (values : Array Float) (mask : Array Bool) (scale : Float := 1.0) : Float := Id.run do
  let mut total := 0.0
  let mut count := 0
  for i in [:values.size] do
    if mask.getD i false then
      total := total + values[i]! * scale
      count := count + 1
  return if count = 0 then 0.0 else total / count.toFloat

def splitCountLosses (splitTransform : Float → Float) (logits : Array Float) (targets : Array Nat) :
    Array Float :=
  logits.mapIdx (fun i logit => splitCountLoss splitTransform logit (targets.getD i 0))

def deletionLosses (logits : Array Float) (targets : Array Bool) : Array Float :=
  logits.mapIdx (fun i logit => logitBinaryCrossEntropy logit (targets.getD i false))

def maskedSplitCountLoss
    (splitTransform : Float → Float)
    (logits : Array Float)
    (targets : Array Nat)
    (mask : Array Bool)
    (scale : Float := 1.0) : Float :=
  maskedScaledMean (splitCountLosses splitTransform logits targets) mask scale

def maskedDeletionLoss
    (logits : Array Float)
    (targets mask : Array Bool)
    (scale : Float := 1.0) : Float :=
  maskedScaledMean (deletionLosses logits targets) mask scale

/-! ## Flow tree node -/

structure FlowNode (α : Type) where
  time : Float
  data : α
  weight : Nat
  group : Int
  branchable : Bool
  del : Bool
  id : Int
  flowable : Bool
  children : Array (FlowNode α) := #[]
  deriving Repr, Inhabited

namespace FlowNode

def leaf (time : Float) (data : α) (group : Int) (branchable del flowable : Bool) (id : Int) : FlowNode α :=
  { time, data, weight := 1, group, branchable, del, id, flowable, children := #[] }

def merge (time : Float) (data : α) (left right : FlowNode α) : FlowNode α :=
  { time,
    data,
    weight := left.weight + right.weight,
    group := left.group,
    branchable := true,
    del := false,
    id := 0,
    flowable := true,
    children := #[left, right] }

end FlowNode

/-! ## Branching state (single sequence) -/

structure BranchingState (α : Type) where
  state : Array α
  groupings : Array Int
  del : Array Bool
  ids : Array Int
  branchmask : Array Bool
  flowmask : Array Bool
  padmask : Array Bool
  deriving Repr, Inhabited

namespace BranchingState

def length (x : BranchingState α) : Nat := x.state.size

def mkDefault (state : Array α) (groupings : Array Int) : BranchingState α :=
  let n := state.size
  { state,
    groupings,
    del := Array.replicate n false,
    ids := (Array.range n).map (fun i => Int.ofNat (i + 1)),
    branchmask := Array.replicate n true,
    flowmask := Array.replicate n true,
    padmask := Array.replicate n true }

end BranchingState

/-! ## Anchor merging -/

class AnchorMerge (α : Type) where
  merge : α → α → Nat → Nat → α

def canonicalAnchorMerge [AnchorMerge α] (a b : α) (w1 w2 : Nat) : α :=
  AnchorMerge.merge a b w1 w2

def selectAnchorMerge [AnchorMerge α] (a b : α) (w1 w2 : Nat) (rng : Rng) : α × Rng :=
  let total := w1 + w2
  if total = 0 then
    (canonicalAnchorMerge a b w1 w2, rng)
  else
    let (u, rng') := randFloat rng
    let p := w1.toFloat / total.toFloat
    if u < p then
      (canonicalAnchorMerge a b 1 0, rng')
    else
      (canonicalAnchorMerge b a 0 1, rng')

instance : AnchorMerge Float where
  merge a b w1 w2 :=
    let total := (w1 + w2).toFloat
    if total == 0.0 then a else (a * w1.toFloat + b * w2.toFloat) / total

instance {σ : StaticSpec} : AnchorMerge (Tensor σ) where
  merge a b w1 w2 :=
    let total := (w1 + w2).toFloat
    if total == 0.0 then a
    else
      let wa := Tensor.Preserving.mulScalar a w1.toFloat
      let wb := Tensor.Preserving.mulScalar b w2.toFloat
      Tensor.Preserving.divScalar (Tensor.Preserving.add wa wb) total

instance [AnchorMerge α] [AnchorMerge β] : AnchorMerge (α × β) where
  merge a b w1 w2 :=
    (AnchorMerge.merge a.1 b.1 w1 w2, AnchorMerge.merge a.2 b.2 w1 w2)

instance [AnchorMerge α] {cmaskSpec lmaskSpec : StaticSpec} :
    AnchorMerge (torch.flowfusion.MaskedState α cmaskSpec lmaskSpec) where
  merge a b w1 w2 :=
    torch.flowfusion.maskLike (AnchorMerge.merge a.state b.state w1 w2) a

/-! ## Coalescence policies -/

abbrev GroupMins := Std.HashMap Int Nat

/-! ## Group-minimum helpers -/

inductive GroupMinsSpec where
  | none
  | uniform (min : Nat)
  | perGroup (mins : GroupMins)
  deriving Repr, Inhabited

private def defaultGroupMins (groupings : Array Int) (min : Nat) : GroupMins := Id.run do
  let mut mins : GroupMins := {}
  for g in groupings do
    match mins.get? g with
    | some _ => continue
    | none => mins := mins.insert g min
  mins

def resolveGroupMins (spec : GroupMinsSpec) (groupings : Array Int) : GroupMins :=
  match spec with
  | .none => defaultGroupMins groupings 1
  | .uniform min => defaultGroupMins groupings min
  | .perGroup mins => mins

def resolveGroupMinsBatch (default : GroupMinsSpec) (perItem : Array GroupMinsSpec)
    (groupings : Array (Array Int)) : Array GroupMins := Id.run do
  if perItem.isEmpty then
    return groupings.map (resolveGroupMins default)
  if perItem.size != groupings.size then
    return groupings.map (resolveGroupMins default)
  let mut out : Array GroupMins := #[]
  for i in [:groupings.size] do
    out := out.push (resolveGroupMins perItem[i]! groupings[i]!)
  out

def groupwiseMaxCoalescences (nodes : Array (FlowNode α)) : Nat := Id.run do
  let mut counts : Std.HashMap Int Nat := {}
  for n in nodes do
    if n.branchable then
      let c := counts.getD n.group 0
      counts := counts.insert n.group (c + 1)
  return counts.fold (init := 0) (fun acc _ c => acc + (Nat.pred c))

structure CoalescencePolicy (α : Type) where
  select : Array (FlowNode α) → Option GroupMins → Rng → Option (Nat × Nat) × Rng
  maxCoalescences : Array (FlowNode α) → Nat
  init : Array (FlowNode α) → Rng → Rng := fun _ r => r
  update : Array (FlowNode α) → Nat → Nat → Nat → Rng → Rng := fun _ _ _ _ r => r
  reorder : Array (FlowNode α) → Array (FlowNode α) := id
  shouldAppendOnSplit : Bool := false

section InhabitedOps

variable [Inhabited α]

def sequentialPairs (nodes : Array (FlowNode α)) : Array Nat := Id.run do
  let mut idx : Array Nat := #[]
  let n := nodes.size
  if n <= 1 then
    return idx
  for i in [:n-1] do
    let a := nodes[i]!
    let b := nodes[i+1]!
    if a.branchable && b.branchable && a.group == b.group then
      idx := idx.push i
  return idx

def eligibleSequentialPairStarts
    (nodes : Array (FlowNode α))
    (groupMins : Option GroupMins) :
    Array Nat := Id.run do
  let n := nodes.size
  if n <= 1 then
    return #[]
  let mut groupSizes : Std.HashMap Int Nat := {}
  if groupMins.isSome then
    for node in nodes do
      if node.branchable then
        let c := groupSizes.getD node.group 0
        groupSizes := groupSizes.insert node.group (c + 1)
  let mut eligible : Array Nat := #[]
  for i in [:n-1] do
    let a := nodes[i]!
    let b := nodes[i+1]!
    if a.branchable && b.branchable && a.group == b.group then
      let allowed :=
        match groupMins with
        | none => true
        | some mins => groupSizes.getD a.group 0 > mins.getD a.group 0
      if allowed then
        eligible := eligible.push i
  return eligible

def sequentialUniformSelect (nodes : Array (FlowNode α)) (groupMins : Option GroupMins) (rng : Rng)
    : Option (Nat × Nat) × Rng := Id.run do
  let eligible := eligibleSequentialPairStarts nodes groupMins
  if eligible.isEmpty then
    return (none, rng)
  else
    let (k, rng') := randNat rng eligible.size
    let i := eligible[k]!
    return (some (i, i+1), rng')

def sequentialUniformPolicy (α : Type) [Inhabited α] : CoalescencePolicy α :=
  { select := sequentialUniformSelect,
    maxCoalescences := fun nodes => (sequentialPairs nodes).size }

def sequentialUniformBlockMinSelect (blockMin : Nat) (nodes : Array (FlowNode α)) (rng : Rng)
    : Option (Nat × Nat) × Rng := Id.run do
  let n := nodes.size
  if n <= 1 then
    return (none, rng)
  let mut blockSizes : Std.HashMap Nat Nat := {}
  let mut block := 0
  for i in [:n-1] do
    let a := nodes[i]!
    let b := nodes[i+1]!
    if a.branchable && b.branchable && a.group == b.group then
      let c := blockSizes.getD block 0
      blockSizes := blockSizes.insert block (c + 1)
    else
      block := block + 1
  let mut eligible : Array Nat := #[]
  let mut block2 := 0
  for i in [:n-1] do
    let a := nodes[i]!
    let b := nodes[i+1]!
    if a.branchable && b.branchable && a.group == b.group then
      let size := blockSizes.getD block2 0
      if size > (blockMin - 1) then
        eligible := eligible.push i
    else
      block2 := block2 + 1
  if eligible.isEmpty then
    return (none, rng)
  else
    let (k, rng') := randNat rng eligible.size
    let i := eligible[k]!
    return (some (i, i+1), rng')

def sequentialUniformBlockMinPolicy (α : Type) [Inhabited α] (blockMin : Nat) : CoalescencePolicy α :=
  { select := fun nodes _ rng => sequentialUniformBlockMinSelect blockMin nodes rng,
    maxCoalescences := fun nodes => (sequentialPairs nodes).size }

private def weightedIndex (weights : Array Float) (rng : Rng) : Option Nat × Rng := Id.run do
  if weights.isEmpty then
    return (none, rng)
  let total := weights.foldl (init := 0.0) (fun acc w => acc + w)
  if total <= 0.0 then
    return (none, rng)
  let (u, rng') := randFloat rng
  let target := u * total
  let mut acc := 0.0
  for i in [:weights.size] do
    acc := acc + weights[i]!
    if acc >= target then
      return (some i, rng')
  return (some (weights.size - 1), rng')

def balancedSequentialSelect (alpha : Float) (nodes : Array (FlowNode α))
    (_groupMins : Option GroupMins) (rng : Rng)
    : Option (Nat × Nat) × Rng := Id.run do
  let alpha := if alpha < 0.0 then 0.0 else alpha
  let n := nodes.size
  if n <= 1 then
    return (none, rng)
  let mut eligible : Array Nat := #[]
  let mut weights : Array Float := #[]
  for i in [:n-1] do
    let a := nodes[i]!
    let b := nodes[i+1]!
    if a.branchable && b.branchable && a.group == b.group then
      eligible := eligible.push i
      let w := (a.weight + b.weight).toFloat
      weights := weights.push (Float.pow w (-alpha))
  let (k, rng') := weightedIndex weights rng
  match k with
  | none => return (none, rng')
  | some k =>
      let i := eligible[k]!
      return (some (i, i+1), rng')

def balancedSequentialPolicy (α : Type) [Inhabited α] (alpha : Float := 1.0) : CoalescencePolicy α :=
  { select := balancedSequentialSelect alpha,
    maxCoalescences := fun nodes => (sequentialPairs nodes).size }

def richGetRicherSequentialSelect (alpha : Float) (nodes : Array (FlowNode α))
    (groupMins : Option GroupMins) (rng : Rng)
    : Option (Nat × Nat) × Rng := Id.run do
  let alpha := if alpha < 0.0 then 0.0 else alpha
  let eligible := eligibleSequentialPairStarts nodes groupMins
  let mut weights : Array Float := #[]
  for i in eligible do
    let a := nodes[i]!
    let b := nodes[i+1]!
    weights := weights.push (Float.pow (a.weight + b.weight).toFloat alpha)
  let (k, rng') := weightedIndex weights rng
  match k with
  | none => return (none, rng')
  | some k =>
      let i := eligible[k]!
      return (some (i, i+1), rng')

def richGetRicherSequentialPolicy (α : Type) [Inhabited α] (alpha : Float := 1.0) :
    CoalescencePolicy α :=
  { select := richGetRicherSequentialSelect alpha,
    maxCoalescences := fun nodes => (sequentialPairs nodes).size }

private def approxFloat (a b atol rtol : Float) : Bool :=
  Float.abs (a - b) <= atol + rtol * Float.abs b

def sequentialProximitySelect
    (distance : α → α → Float)
    (tieAtol tieRtol : Float)
    (nodes : Array (FlowNode α))
    (groupMins : Option GroupMins)
    (rng : Rng) :
    Option (Nat × Nat) × Rng := Id.run do
  let eligible := eligibleSequentialPairStarts nodes groupMins
  if eligible.isEmpty then
    return (none, rng)
  let mut best? : Option Float := none
  let mut dists : Array Float := #[]
  for i in eligible do
    let d := distance nodes[i]!.data nodes[i+1]!.data
    dists := dists.push d
    best? :=
      match best? with
      | none => some d
      | some best => some (min best d)
  let best := best?.getD 0.0
  let mut tied : Array Nat := #[]
  for j in [:eligible.size] do
    if approxFloat dists[j]! best tieAtol tieRtol then
      tied := tied.push (eligible[j]!)
  if tied.isEmpty then
    return (none, rng)
  let (k, rng') := randNat rng tied.size
  let i := tied[k]!
  return (some (i, i+1), rng')

def sequentialProximityPolicy (α : Type) [Inhabited α]
    (distance : α → α → Float)
    (tieAtol : Float := 0.0)
    (tieRtol : Float := 0.0) :
    CoalescencePolicy α :=
  { select := sequentialProximitySelect distance tieAtol tieRtol,
    maxCoalescences := fun nodes => (sequentialPairs nodes).size }

private def deepLineagePairStarts
    (minCount targetTrunks : Nat)
    (nodes : Array (FlowNode α))
    (groupMins : Option GroupMins) :
    Array Nat := Id.run do
  let minCount := Nat.max 1 minCount
  let targetTrunks := Nat.max 1 targetTrunks
  let pairs := eligibleSequentialPairStarts nodes groupMins
  if pairs.isEmpty then
    return pairs
  let mut deepCounts : Std.HashMap Int Nat := {}
  for node in nodes do
    if node.branchable && node.weight >= minCount then
      let c := deepCounts.getD node.group 0
      deepCounts := deepCounts.insert node.group (c + 1)
  let mut filtered : Array Nat := #[]
  for i in pairs do
    let a := nodes[i]!
    let b := nodes[i+1]!
    let deepActive := deepCounts.getD a.group 0 >= targetTrunks
    if !deepActive || a.weight >= minCount || b.weight >= minCount then
      filtered := filtered.push i
  if filtered.isEmpty then pairs else filtered

def sequentialDeepLineageSelect
    (minCount targetTrunks : Nat)
    (nodes : Array (FlowNode α))
    (groupMins : Option GroupMins)
    (rng : Rng) :
    Option (Nat × Nat) × Rng := Id.run do
  let eligible := deepLineagePairStarts minCount targetTrunks nodes groupMins
  if eligible.isEmpty then
    return (none, rng)
  let (k, rng') := randNat rng eligible.size
  let i := eligible[k]!
  return (some (i, i+1), rng')

def sequentialDeepLineagePolicy (α : Type) [Inhabited α]
    (minCount : Nat := 2)
    (targetTrunks : Nat := 1) :
    CoalescencePolicy α :=
  { select := sequentialDeepLineageSelect minCount targetTrunks,
    maxCoalescences := fun nodes => (sequentialPairs nodes).size }

/-! ## Forest sampling -/

def nextSplitTime (dist : TimeDist) (W : Nat) (t0 : Float) (rng : Rng) : Float × Rng :=
  let m := (W - 1).toFloat
  let S0 := 1.0 - dist.cdf t0
  if S0 <= 0.0 then
    (t0, rng)
  else
    let (e, rng') := randExponential rng
    let sStar := S0 * Float.exp (-(e / m))
    let p := 1.0 - sStar
    let t := dist.quantile p
    (max t0 (min t 1.0), rng')

partial def sampleSplitTimes (dist : TimeDist) (node : FlowNode α) (t0 : Float) (rng : Rng)
    : FlowNode α × Array Float × Rng := Id.run do
  if node.weight <= 1 then
    return (node, #[], rng)
  else
    let (t, rng') := nextSplitTime dist node.weight t0 rng
    let mut times : Array Float := #[t]
    let mut children : Array (FlowNode α) := #[]
    let mut rng'' := rng'
    for child in node.children do
      let (child', ctimes, rng''') := sampleSplitTimes dist child t rng''
      rng'' := rng'''
      children := children.push child'
      times := times ++ ctimes
    return ({ node with time := t, children := children }, times, rng'')

private def eraseIdx (arr : Array α) (idx : Nat) : Array α := Id.run do
  let mut out : Array α := #[]
  for i in [:arr.size] do
    if i != idx then
      out := out.push arr[i]!
  out

def sampleForest
    (elements : Array α)
    (groupings : Array Int)
    (branchable : Array Bool)
    (flowable : Array Bool)
    (deleted : Array Bool)
    (ids : Array Int)
    (branchTimeDist : TimeDist)
    (policy : CoalescencePolicy α := sequentialUniformPolicy α)
    (coalescenceFactor : Float := 1.0)
    (merger : α → α → Nat → Nat → α)
    (groupMins : Option GroupMins := none)
    (rng : Rng := { state := 0 })
    : Array (FlowNode α) × Array Float × Rng := Id.run do
  let n := elements.size
  let mut nodes : Array (FlowNode α) := #[]
  for i in [:n] do
    nodes := nodes.push (FlowNode.leaf 1.0 elements[i]! groupings[i]! branchable[i]! deleted[i]! flowable[i]! ids[i]!)
  let rng := policy.init nodes rng
  let maxMerges := policy.maxCoalescences nodes
  let (sampledMerges, rng) := randBinomial rng maxMerges coalescenceFactor
  let mut rng := rng
  let mut nodesAcc := nodes
  for _ in [:sampledMerges] do
    let (sel, rng') := policy.select nodesAcc groupMins rng
    rng := rng'
    match sel with
    | none => break
    | some (i, j) =>
        let lo := min i j
        let hi := max i j
        let left := nodesAcc[lo]!
        let right := nodesAcc[hi]!
        let mergedData := merger left.data right.data left.weight right.weight
        let merged := FlowNode.merge 0.0 mergedData left right
        -- replace lo, remove hi
        nodesAcc := nodesAcc.set! lo merged
        nodesAcc := eraseIdx nodesAcc hi
        rng := policy.update nodesAcc lo hi lo rng
  nodesAcc := policy.reorder nodesAcc
  -- sample split times for each root
  let mut allTimes : Array Float := #[]
  let mut roots : Array (FlowNode α) := #[]
  for node in nodesAcc do
    let (node', times, rng') := sampleSplitTimes branchTimeDist node 0.0 rng
    rng := rng'
    roots := roots.push node'
    allTimes := allTimes ++ times
  (roots, allTimes, rng)

/-! ## Deletion insertion utilities -/

def groupCounts (groupings : Array Int) : Std.HashMap Int Nat := Id.run do
  let mut counts : Std.HashMap Int Nat := {}
  for g in groupings do
    let c := counts.getD g 0
    counts := counts.insert g (c + 1)
  counts

def uniformDelInsertions (x : BranchingState α) (delP : Float) (rng : Rng)
    : BranchingState α × Rng := Id.run do
  let n := x.state.size
  let mut rng := rng
  let mut newIndices : Array Nat := #[]
  let mut delFlags : Array Bool := #[]
  for i in [:n] do
    let eligible := x.flowmask[i]! && x.branchmask[i]!
    let (doDup, rng') := if eligible then randBernoulli rng delP else (false, rng)
    rng := rng'
    if doDup then
      newIndices := newIndices.push i
      newIndices := newIndices.push i
      let (chooseDup, rng'') := randBool rng
      rng := rng''
      -- exactly one of the two gets deletion
      delFlags := delFlags.push (!chooseDup)
      delFlags := delFlags.push chooseDup
    else
      newIndices := newIndices.push i
      delFlags := delFlags.push false
  let mut newState : Array α := #[]
  let mut groupings : Array Int := #[]
  let mut ids : Array Int := #[]
  let mut branchmask : Array Bool := #[]
  let mut flowmask : Array Bool := #[]
  let mut padmask : Array Bool := #[]
  for idx in newIndices do
    newState := newState.push x.state[idx]!
    groupings := groupings.push x.groupings[idx]!
    ids := ids.push x.ids[idx]!
    branchmask := branchmask.push x.branchmask[idx]!
    flowmask := flowmask.push x.flowmask[idx]!
    padmask := padmask.push x.padmask[idx]!
  ({ state := newState, groupings, del := delFlags, ids, branchmask, flowmask, padmask }, rng)

def fixedcountDelInsertions (x : BranchingState α) (numEvents : Nat) (rng : Rng)
    : BranchingState α × Rng := Id.run do
  let n := x.state.size
  if numEvents = 0 then
    return (x, rng)
  let eligible : Array Nat :=
    (Array.range n)
      |>.filter (fun i => x.flowmask[i]! && x.branchmask[i]!)
  if eligible.isEmpty then
    return (x, rng)
  let mut rng := rng
  let mut beforeFlags : Array (Array Bool) := Array.replicate n #[]
  let mut afterFlags : Array (Array Bool) := Array.replicate n #[]
  let mut origDel : Array Bool := Array.replicate n false
  for _ in [:numEvents] do
    let (k, rng') := randNat rng eligible.size
    rng := rng'
    let i := eligible[k]!
    let (before, rng'') := randBool rng
    rng := rng''
    let (useOrig, rng''') := randBool rng
    rng := rng'''
    if before then
      if useOrig && !(origDel[i]!) then
        origDel := origDel.set! i true
        beforeFlags := beforeFlags.set! i (beforeFlags[i]! |>.push false)
      else
        beforeFlags := beforeFlags.set! i (beforeFlags[i]! |>.push true)
    else
      if useOrig && !(origDel[i]!) then
        origDel := origDel.set! i true
        afterFlags := afterFlags.set! i (afterFlags[i]! |>.push false)
      else
        afterFlags := afterFlags.set! i (afterFlags[i]! |>.push true)
  let mut newState : Array α := #[]
  let mut groupings : Array Int := #[]
  let mut ids : Array Int := #[]
  let mut branchmask : Array Bool := #[]
  let mut flowmask : Array Bool := #[]
  let mut padmask : Array Bool := #[]
  let mut delFlags : Array Bool := #[]
  for i in [:n] do
    for flag in beforeFlags[i]! do
      newState := newState.push x.state[i]!
      groupings := groupings.push x.groupings[i]!
      ids := ids.push x.ids[i]!
      branchmask := branchmask.push x.branchmask[i]!
      flowmask := flowmask.push x.flowmask[i]!
      padmask := padmask.push x.padmask[i]!
      delFlags := delFlags.push flag
    newState := newState.push x.state[i]!
    groupings := groupings.push x.groupings[i]!
    ids := ids.push x.ids[i]!
    branchmask := branchmask.push x.branchmask[i]!
    flowmask := flowmask.push x.flowmask[i]!
    padmask := padmask.push x.padmask[i]!
    delFlags := delFlags.push origDel[i]!
    for flag in afterFlags[i]! do
      newState := newState.push x.state[i]!
      groupings := groupings.push x.groupings[i]!
      ids := ids.push x.ids[i]!
      branchmask := branchmask.push x.branchmask[i]!
      flowmask := flowmask.push x.flowmask[i]!
      padmask := padmask.push x.padmask[i]!
      delFlags := delFlags.push flag
  ({ state := newState, groupings, del := delFlags, ids, branchmask, flowmask, padmask }, rng)

def groupFixedcountDelInsertions (x : BranchingState α) (groupNumEvents : Std.HashMap Int Nat) (rng : Rng)
    : BranchingState α × Rng := Id.run do
  let n := x.state.size
  if groupNumEvents.isEmpty then
    return (x, rng)
  let mut rng := rng
  let mut beforeFlags : Array (Array Bool) := Array.replicate n #[]
  let mut afterFlags : Array (Array Bool) := Array.replicate n #[]
  let mut origDel : Array Bool := Array.replicate n false
  let mut actualEvents := 0
  for (g, numEvents) in groupNumEvents.toList do
    if numEvents = 0 then
      continue
    let eligible : Array Nat :=
      (Array.range n)
        |>.filter (fun i => x.flowmask[i]! && x.branchmask[i]! && x.groupings[i]! == g)
    if eligible.isEmpty then
      continue
    for _ in [:numEvents] do
      let (k, rng') := randNat rng eligible.size
      rng := rng'
      let i := eligible[k]!
      let (before, rng'') := randBool rng
      rng := rng''
      let (useOrig, rng''') := randBool rng
      rng := rng'''
      if before then
        if useOrig && !(origDel[i]!) then
          origDel := origDel.set! i true
          beforeFlags := beforeFlags.set! i (beforeFlags[i]! |>.push false)
        else
          beforeFlags := beforeFlags.set! i (beforeFlags[i]! |>.push true)
      else
        if useOrig && !(origDel[i]!) then
          origDel := origDel.set! i true
          afterFlags := afterFlags.set! i (afterFlags[i]! |>.push false)
        else
          afterFlags := afterFlags.set! i (afterFlags[i]! |>.push true)
      actualEvents := actualEvents + 1
  if actualEvents = 0 then
    return (x, rng)
  let mut newState : Array α := #[]
  let mut groupings : Array Int := #[]
  let mut ids : Array Int := #[]
  let mut branchmask : Array Bool := #[]
  let mut flowmask : Array Bool := #[]
  let mut padmask : Array Bool := #[]
  let mut delFlags : Array Bool := #[]
  for i in [:n] do
    for flag in beforeFlags[i]! do
      newState := newState.push x.state[i]!
      groupings := groupings.push x.groupings[i]!
      ids := ids.push x.ids[i]!
      branchmask := branchmask.push x.branchmask[i]!
      flowmask := flowmask.push x.flowmask[i]!
      padmask := padmask.push x.padmask[i]!
      delFlags := delFlags.push flag
    newState := newState.push x.state[i]!
    groupings := groupings.push x.groupings[i]!
    ids := ids.push x.ids[i]!
    branchmask := branchmask.push x.branchmask[i]!
    flowmask := flowmask.push x.flowmask[i]!
    padmask := padmask.push x.padmask[i]!
    delFlags := delFlags.push origDel[i]!
    for flag in afterFlags[i]! do
      newState := newState.push x.state[i]!
      groupings := groupings.push x.groupings[i]!
      ids := ids.push x.ids[i]!
      branchmask := branchmask.push x.branchmask[i]!
      flowmask := flowmask.push x.flowmask[i]!
      padmask := padmask.push x.padmask[i]!
      delFlags := delFlags.push flag
  ({ state := newState, groupings, del := delFlags, ids, branchmask, flowmask, padmask }, rng)

/-! ## Bridge helpers (generic) -/

structure Segment (α : Type) where
  Xt : α
  t : Float
  anchor : α
  descendants : Nat
  del : Bool
  branchable : Bool
  flowable : Bool
  group : Int
  lastCoalescence : Float
  id : Int
  deriving Repr

structure CoalescentFlow (P α : Type) where
  base : P
  branchTime : TimeDist
  splitTransform : Float → Float
  policy : CoalescencePolicy α
  deletionTime : TimeDist

namespace CoalescentFlow

def mkWithPolicy
    (base : P)
    (branchTime : TimeDist)
    (policy : CoalescencePolicy α)
    (deletionTime : TimeDist := TimeDist.uniform)
    (splitTransform : Float → Float := defaultSplitTransform) :
    CoalescentFlow P α :=
  { base, branchTime, splitTransform, policy, deletionTime }

def mkDefault
    [Inhabited α]
    (base : P)
    (branchTime : TimeDist)
    (deletionTime : TimeDist := TimeDist.uniform)
    (splitTransform : Float → Float := defaultSplitTransform) :
    CoalescentFlow P α :=
  mkWithPolicy base branchTime (sequentialUniformPolicy α) deletionTime splitTransform

end CoalescentFlow

/-! ## Forward-time generation helpers -/

structure BranchingStepPrediction (α : Type) where
  targets : Array α
  splitLogits : Array Float
  delLogits : Array Float
  deriving Repr, Inhabited

structure BranchingStepEvent where
  sourceIndex : Nat
  sourceId : Int
  group : Int
  splitCount : Nat
  deleted : Bool
  t0 : Float
  t1 : Float
  deriving Repr, Inhabited

structure BranchingStepResult (α : Type) where
  state : BranchingState α
  events : Array BranchingStepEvent
  deriving Repr, Inhabited

structure BranchingGenerateResult (α : Type) where
  finalState : BranchingState α
  trajectory : Array (BranchingState α)
  events : Array (Array BranchingStepEvent)
  times : Array Float
  deriving Repr, Inhabited

/-!
## Runtime lineage reconstruction

`BranchingState.ids` identifies conditional-tree anchors and is therefore not a
unique particle identity after a forward split.  The structures below assign
separate runtime identities to generated particles without changing the state
representation used by training.
-/

structure LineageParticle where
  particleId : Nat
  parentId? : Option Nat := none
  birthEventId? : Option Nat := none
  deriving Repr, Inhabited, BEq

inductive LineageEventKind where
  | split
  | delete
  deriving Repr, Inhabited, BEq

structure LineageEvent where
  eventId : Nat
  kind : LineageEventKind
  t0 : Float
  t1 : Float
  parentId : Nat
  childIds : Array Nat := #[]
  deriving Repr, Inhabited

structure LineageFrame where
  step : Nat
  time : Float
  particles : Array LineageParticle
  deriving Repr, Inhabited

structure BranchingLineageTrace where
  frames : Array LineageFrame
  events : Array LineageEvent
  deriving Repr, Inhabited

/--
Assign stable runtime particle identities to an already generated trajectory.

At a split the source identity terminates and `splitCount + 1` child identities
are created.  `appendOnSplit` must match the coalescence policy used for the
forward pass so that lineage entries stay aligned with state-array entries.
-/
def reconstructLineage
    (result : BranchingGenerateResult α)
    (appendOnSplit : Bool := false) :
    Except String BranchingLineageTrace := do
  if result.trajectory.isEmpty then
    throw "cannot reconstruct lineage for an empty trajectory"
  if result.times.size != result.trajectory.size then
    throw s!"lineage time/frame count mismatch: times={result.times.size}, frames={result.trajectory.size}"
  if result.events.size + 1 != result.trajectory.size then
    throw s!"lineage event/frame count mismatch: event_steps={result.events.size}, frames={result.trajectory.size}"
  let finalFrameSize := result.trajectory[result.trajectory.size - 1]!.state.size
  if result.finalState.state.size != finalFrameSize then
    throw s!"lineage final-state mismatch: final={result.finalState.state.size}, last_frame={finalFrameSize}"
  let initialSize := result.trajectory[0]!.state.size
  let initialParticles := (Array.range initialSize).map (fun i => { particleId := i + 1 })
  let mut current := initialParticles
  let mut frames : Array LineageFrame := #[{
    step := 0
    time := result.times.getD 0 0.0
    particles := initialParticles
  }]
  let mut lineageEvents : Array LineageEvent := #[]
  let mut nextParticleId := initialSize + 1
  let mut nextEventId := 1
  for step in [:result.trajectory.size - 1] do
    let stepEvents := result.events.getD step #[]
    let mut seenSources : Array Nat := #[]
    for event in stepEvents do
      if event.sourceIndex >= current.size then
        throw s!"lineage event source index {event.sourceIndex} is outside frame {step} of size {current.size}"
      if seenSources.contains event.sourceIndex then
        throw s!"multiple lineage events target source index {event.sourceIndex} at step {step}"
      if event.deleted && event.splitCount > 0 then
        throw s!"lineage event at step {step} cannot both delete and split source index {event.sourceIndex}"
      seenSources := seenSources.push event.sourceIndex
    let mut primary : Array LineageParticle := #[]
    let mut appended : Array LineageParticle := #[]
    for i in [:current.size] do
      let particle := current[i]!
      let event? := stepEvents.find? (fun event => event.sourceIndex == i)
      match event? with
      | none => primary := primary.push particle
      | some event =>
          let eventId := nextEventId
          nextEventId := nextEventId + 1
          if event.deleted then
            lineageEvents := lineageEvents.push {
              eventId
              kind := .delete
              t0 := event.t0
              t1 := event.t1
              parentId := particle.particleId
            }
          else if event.splitCount > 0 then
            let mut children : Array LineageParticle := #[]
            for _ in [:(event.splitCount + 1)] do
              children := children.push {
                particleId := nextParticleId
                parentId? := some particle.particleId
                birthEventId? := some eventId
              }
              nextParticleId := nextParticleId + 1
            primary := primary.push children[0]!
            for child in children.extract 1 children.size do
              if appendOnSplit then
                appended := appended.push child
              else
                primary := primary.push child
            lineageEvents := lineageEvents.push {
              eventId
              kind := .split
              t0 := event.t0
              t1 := event.t1
              parentId := particle.particleId
              childIds := children.map (fun child => child.particleId)
            }
          else
            primary := primary.push particle
    let nextParticles := primary ++ appended
    let expectedSize := result.trajectory[step + 1]!.state.size
    if nextParticles.size != expectedSize then
      throw s!"lineage/state size mismatch at step {step + 1}: lineage={nextParticles.size}, state={expectedSize}"
    current := nextParticles
    frames := frames.push {
      step := step + 1
      time := result.times.getD (step + 1) 0.0
      particles := current
    }
  return { frames, events := lineageEvents }

private def maskBaseStep (x : BranchingState α) (stepped : Array α) [Inhabited α] : Array α := Id.run do
  let mut out : Array α := #[]
  for i in [:x.state.size] do
    let old := x.state.getD i default
    if x.flowmask.getD i false then
      out := out.push (stepped.getD i old)
    else
      out := out.push old
  out

private def appendGeneratedElement
    (state : Array α)
    (groupings : Array Int)
    (ids : Array Int)
    (branchmask : Array Bool)
    (flowmask : Array Bool)
    (padmask : Array Bool)
    (value : α)
    (group : Int)
    (id : Int)
    (branchable flowable : Bool) :
    Array α × Array Int × Array Int × Array Bool × Array Bool × Array Bool :=
  (state.push value,
   groupings.push group,
   ids.push id,
   branchmask.push branchable,
   flowmask.push flowable,
   padmask.push true)

def branchingStep
    (baseStep : P → BranchingState α → Array α → Float → Float → Array α)
    (flow : CoalescentFlow P α)
    (x : BranchingState α)
    (prediction : BranchingStepPrediction α)
    (s1 s2 : Float)
    (splitAllowedAfterBaseStep : α → α → Bool := fun _ _ => true)
    (rng : Rng := { state := 0 })
    : BranchingStepResult α × Rng := Id.run do
  let n := x.state.size
  let dt := max (s2 - s1) 0.0
  let stepped := maskBaseStep x (baseStep flow.base x prediction.targets s1 s2)
  let splitDensity := flow.branchTime.truncatedPdfFrom s1 s1
  let baseDelP := 1.0 - Float.exp (-(flow.deletionTime.hazard s1) * dt)
  let mut rng := rng
  let mut splits : Array Nat := Array.replicate n 0
  let mut deleted : Array Bool := Array.replicate n false
  for i in [:n] do
    if x.branchmask.getD i false then
      let splitLogit := prediction.splitLogits.getD i 0.0
      let lambda := max (dt * flow.splitTransform splitLogit * splitDensity) 0.0
      let (k0, rng') := randPoisson rng lambda
      rng := rng'
      let old := x.state.getD i default
      let new := stepped.getD i old
      let k := if splitAllowedAfterBaseStep old new then k0 else 0
      splits := splits.set! i k
      let delLogit := prediction.delLogits.getD i (-100.0)
      let pDel := clampProbability (sigmoid delLogit * baseDelP)
      let (d, rng') := randBernoulli rng pDel
      rng := rng'
      if d then
        deleted := deleted.set! i true
        splits := splits.set! i 0

  let deletedCount := deleted.foldl (init := 0) (fun acc d => if d then acc + 1 else acc)
  if n > 0 && deletedCount == n then
    let deletedIdxs := (Array.range n).filter (fun i => deleted.getD i false)
    let (k, rng') := randNat rng deletedIdxs.size
    rng := rng'
    deleted := deleted.set! (deletedIdxs.getD k 0) false

  let mut events : Array BranchingStepEvent := #[]
  let mut primaryState : Array α := #[]
  let mut primaryGroups : Array Int := #[]
  let mut primaryIds : Array Int := #[]
  let mut primaryBranchmask : Array Bool := #[]
  let mut primaryFlowmask : Array Bool := #[]
  let mut primaryPadmask : Array Bool := #[]
  let mut appendedState : Array α := #[]
  let mut appendedGroups : Array Int := #[]
  let mut appendedIds : Array Int := #[]
  let mut appendedBranchmask : Array Bool := #[]
  let mut appendedFlowmask : Array Bool := #[]
  let mut appendedPadmask : Array Bool := #[]

  for i in [:n] do
    let splitCount := splits.getD i 0
    let wasDeleted := deleted.getD i false
    if splitCount > 0 || wasDeleted then
      events := events.push {
        sourceIndex := i
        sourceId := x.ids.getD i 0
        group := x.groupings.getD i 0
        splitCount := splitCount
        deleted := wasDeleted
        t0 := s1
        t1 := s2
      }
    if !wasDeleted then
      let value := stepped.getD i (x.state.getD i default)
      let group := x.groupings.getD i 0
      let id := x.ids.getD i 0
      let branchable := x.branchmask.getD i false
      let flowable := x.flowmask.getD i false
      let tuple :=
        appendGeneratedElement primaryState primaryGroups primaryIds primaryBranchmask primaryFlowmask primaryPadmask
          value group id branchable flowable
      primaryState := tuple.1
      primaryGroups := tuple.2.1
      primaryIds := tuple.2.2.1
      primaryBranchmask := tuple.2.2.2.1
      primaryFlowmask := tuple.2.2.2.2.1
      primaryPadmask := tuple.2.2.2.2.2
      for _ in [:splitCount] do
        if flow.policy.shouldAppendOnSplit then
          let tuple :=
            appendGeneratedElement appendedState appendedGroups appendedIds appendedBranchmask appendedFlowmask appendedPadmask
              value group id branchable flowable
          appendedState := tuple.1
          appendedGroups := tuple.2.1
          appendedIds := tuple.2.2.1
          appendedBranchmask := tuple.2.2.2.1
          appendedFlowmask := tuple.2.2.2.2.1
          appendedPadmask := tuple.2.2.2.2.2
        else
          let tuple :=
            appendGeneratedElement primaryState primaryGroups primaryIds primaryBranchmask primaryFlowmask primaryPadmask
              value group id branchable flowable
          primaryState := tuple.1
          primaryGroups := tuple.2.1
          primaryIds := tuple.2.2.1
          primaryBranchmask := tuple.2.2.2.1
          primaryFlowmask := tuple.2.2.2.2.1
          primaryPadmask := tuple.2.2.2.2.2

  let newState := primaryState ++ appendedState
  let newGroups := primaryGroups ++ appendedGroups
  let newIds := primaryIds ++ appendedIds
  let newBranchmask := primaryBranchmask ++ appendedBranchmask
  let newFlowmask := primaryFlowmask ++ appendedFlowmask
  let newPadmask := primaryPadmask ++ appendedPadmask
  let newDel := Array.replicate newState.size false
  let out : BranchingState α :=
    { state := newState, groupings := newGroups, del := newDel, ids := newIds,
      branchmask := newBranchmask, flowmask := newFlowmask, padmask := newPadmask }
  ({ state := out, events }, rng)

def branchingGenerate
    (baseStep : P → BranchingState α → Array α → Float → Float → Array α)
    (flow : CoalescentFlow P α)
    (x0 : BranchingState α)
    (model : Float → BranchingState α → BranchingStepPrediction α)
    (schedule : Array Float)
    (splitAllowedAfterBaseStep : α → α → Bool := fun _ _ => true)
    (rng : Rng := { state := 0 })
    : BranchingGenerateResult α × Rng := Id.run do
  if schedule.size <= 1 then
    return ({ finalState := x0, trajectory := #[x0], events := #[], times := schedule }, rng)
  let mut state := x0
  let mut trajectory : Array (BranchingState α) := #[x0]
  let mut events : Array (Array BranchingStepEvent) := #[]
  let mut rng := rng
  for i in [:schedule.size - 1] do
    let s1 := schedule[i]!
    let s2 := schedule[i + 1]!
    let prediction := model s1 state
    let (result, rng') :=
      branchingStep baseStep flow state prediction s1 s2 splitAllowedAfterBaseStep rng
    rng := rng'
    state := result.state
    trajectory := trajectory.push state
    events := events.push result.events
  ({ finalState := state, trajectory, events, times := schedule }, rng)

partial def treeBridge {P : Type u}
    (bridge : P → α → α → Float → Float → α)
    (base : P)
    (node : FlowNode α)
    (x0 : α)
    (targetT currentT : Float)
    (deletionDist : TimeDist)
    (rng : Rng)
    (sampleBridge? : Option (P → α → α → Float → Float → Rng → α × Rng) := none)
    : Array (Segment α) × Rng := Id.run do
  if !node.flowable then
    let seg : Segment α :=
      { Xt := node.data, t := targetT, anchor := node.data, descendants := node.weight,
        del := node.del, branchable := false, flowable := false, group := node.group,
        lastCoalescence := currentT, id := node.id }
    return (#[seg], rng)
  if node.time > targetT then
    -- deletion hazard
    let mut rng := rng
    let mut survive := true
    if node.del then
      let sCurr := max (1.0 - deletionDist.cdf currentT) 0.0
      let sTgt := max (1.0 - deletionDist.cdf targetT) 0.0
      let survRatio := if sCurr > 0.0 then sTgt / sCurr else 0.0
      let (u, rng') := randFloat rng
      rng := rng'
      survive := u >= (1.0 - survRatio)
    if survive then
      let (Xt, rng') :=
        match sampleBridge? with
        | some sampleBridge => sampleBridge base x0 node.data currentT targetT rng
        | none => (bridge base x0 node.data currentT targetT, rng)
      rng := rng'
      let seg : Segment α :=
        { Xt, t := targetT, anchor := node.data, descendants := node.weight,
          del := node.del, branchable := node.branchable, flowable := true, group := node.group,
          lastCoalescence := currentT, id := node.id }
      return (#[seg], rng)
    else
      return (#[], rng)
  else
    let mut out : Array (Segment α) := #[]
    let mut rng := rng
    let (nextX, rng') :=
      match sampleBridge? with
      | some sampleBridge => sampleBridge base x0 node.data currentT node.time rng
      | none => (bridge base x0 node.data currentT node.time, rng)
    rng := rng'
    for child in node.children do
      let (segs, rng') :=
        treeBridge bridge base child nextX targetT node.time deletionDist rng
          (sampleBridge? := sampleBridge?)
      rng := rng'
      out := out ++ segs
    return (out, rng)

def forestBridge {P : Type u}
    (bridge : P → α → α → Float → Float → α)
    (base : P)
    (x0Sampler : FlowNode α → α)
    (x1 : Array α)
    (t : Float)
    (groups : Array Int)
    (branchable : Array Bool)
    (flowable : Array Bool)
    (deleted : Array Bool)
    (branchTime : TimeDist)
    (deletionTime : TimeDist)
    (policy : CoalescencePolicy α)
    (merger : α → α → Nat → Nat → α)
    (groupMins : Option GroupMins := none)
    (coalescenceFactor : Float := 1.0)
    (useBranchingTimeProb : Float := 0.0)
    (maxLen : Option Nat := none)
    (maxResamples : Nat := 8)
    (rng : Rng := { state := 0 })
    (sampleBridge? : Option (P → α → α → Float → Float → Rng → α × Rng) := none)
    (sampleX0? : Option (FlowNode α → Rng → α × Rng) := none)
    : Array (Segment α) × Float × Rng := Id.run do
  let mut rng := rng
  let mut forest : Array (FlowNode α) := #[]
  let mut coalTimes : Array Float := #[]
  let mut tUsed := t
  let mut accept := false
  let mut attempts := 0
  while !accept && attempts <= maxResamples do
    let (forest', coalTimes', rng') :=
      sampleForest x1 groups branchable flowable deleted ((Array.range x1.size).map (fun i => Int.ofNat (i + 1)))
        branchTime policy coalescenceFactor merger groupMins rng
    rng := rng'
    let mut t' := t
    if coalTimes'.size > 0 then
      let (u, rng'') := randFloat rng
      rng := rng''
      if u < useBranchingTimeProb then
        let (k, rng''') := randNat rng coalTimes'.size
        rng := rng'''
        t' := coalTimes'[k]!
    let segCount := forest'.size + (coalTimes'.filter (fun τ => τ <= t')).size
    let ok :=
      match maxLen with
      | none => true
      | some m => segCount <= m
    forest := forest'
    coalTimes := coalTimes'
    tUsed := t'
    if ok then
      accept := true
    else
      attempts := attempts + 1
  let mut out : Array (Segment α) := #[]
  for root in forest do
    let (x0, rng') :=
      match sampleX0? with
      | some sampleX0 => sampleX0 root rng
      | none => (x0Sampler root, rng)
    rng := rng'
    let (segs, rng') :=
      treeBridge bridge base root x0 tUsed 0.0 deletionTime rng
        (sampleBridge? := sampleBridge?)
    rng := rng'
    out := out ++ segs
  (out, tUsed, rng)

structure BranchingBridgeResult (α : Type) where
  t : Array Float
  segments : Array (Array (Segment α))
  Xt : Array (BranchingState α)
  X1anchor : Array (Array α)
  descendants : Array (Array Nat)
  del : Array (Array Bool)
  splitsTarget : Array (Array Nat)
  prevCoalescence : Array (Array Float)
  deriving Repr

def branchingBridge {P : Type u}
    (bridge : P → α → α → Float → Float → α)
    (base : P)
    (x0Sampler : FlowNode α → α)
    (x1s : Array (BranchingState α))
    (times : Array Float)
    (branchTime : TimeDist)
    (deletionTime : TimeDist)
    (policy : CoalescencePolicy α)
    (merger : α → α → Nat → Nat → α)
    (groupMins : Option GroupMins := none)
    (coalescenceFactor : Float := 1.0)
    (useBranchingTimeProb : Float := 0.0)
    (maxLen : Option Nat := none)
    (maxResamples : Nat := 8)
    (lengthMins : GroupMinsSpec := .none)
    (lengthMinsPerItem : Array GroupMinsSpec := #[])
    (deletionPad : Float := 0.0)
    (x1Modifier : BranchingState α → BranchingState α := id)
    (rng : Rng := { state := 0 })
    (sampleBridge? : Option (P → α → α → Float → Float → Rng → α × Rng) := none)
    (sampleX0? : Option (FlowNode α → Rng → α × Rng) := none)
    : BranchingBridgeResult α × Rng := Id.run do
  let mut rng := rng
  let groupings := x1s.map (fun x => x.groupings)
  let resolvedMins : Array GroupMins :=
    match groupMins with
    | some mins => Array.replicate x1s.size mins
    | none => resolveGroupMinsBatch lengthMins lengthMinsPerItem groupings
  let mut x1s := x1s
  if deletionPad > 0.0 then
    let mut padded : Array (BranchingState α) := #[]
    for i in [:x1s.size] do
      let x1 := x1s[i]!
      let counts := groupCounts x1.groupings
      let mins := resolvedMins[i]!
      let mut groupNumEvents : Std.HashMap Int Nat := {}
      for (g, count) in counts.toList do
        let minLen := mins.getD g 1
        let floorLen := Nat.max count minLen
        let (k, rng') :=
          if deletionPad >= 1.0 then
            let deterministicExtra := floorLen - count
            let stochasticMean := max ((deletionPad - 1.0) * floorLen.toFloat) 0.0
            let (stochasticExtra, rng') := randPoisson rng stochasticMean
            (deterministicExtra + stochasticExtra, rng')
          else
            let target := deletionPad * floorLen.toFloat
            let lam := max (target - count.toFloat) 0.0
            randPoisson rng lam
        rng := rng'
        if k > 0 then
          groupNumEvents := groupNumEvents.insert g k
      let (x1', rng') := groupFixedcountDelInsertions x1 groupNumEvents rng
      rng := rng'
      padded := padded.push (x1Modifier x1')
    x1s := padded
  let mut out : Array (Array (Segment α)) := #[]
  let mut usedTimes : Array Float := #[]
  let mut XtStates : Array (BranchingState α) := #[]
  let mut anchors : Array (Array α) := #[]
  let mut descendants : Array (Array Nat) := #[]
  let mut delFlags : Array (Array Bool) := #[]
  let mut splitsTargets : Array (Array Nat) := #[]
  let mut prevCoals : Array (Array Float) := #[]
  for i in [:x1s.size] do
    let x1 := x1s[i]!
    let t := times[i]!
    let (segs, tUsed, rng') :=
      forestBridge bridge base x0Sampler x1.state t x1.groupings x1.branchmask x1.flowmask x1.del
        branchTime deletionTime policy merger (some (resolvedMins[i]!)) coalescenceFactor
        useBranchingTimeProb maxLen maxResamples rng
        (sampleBridge? := sampleBridge?) (sampleX0? := sampleX0?)
    rng := rng'
    out := out.push segs
    usedTimes := usedTimes.push tUsed
    let stateArray := segs.map (fun s => s.Xt)
    let groupArray := segs.map (fun s => s.group)
    let delArray := segs.map (fun s => s.del)
    let idArray := segs.map (fun s => s.id)
    let branchArray := segs.map (fun s => s.branchable)
    let flowArray := segs.map (fun s => s.flowable)
    let padArray := Array.replicate segs.size true
    XtStates := XtStates.push
      { state := stateArray, groupings := groupArray, del := delArray, ids := idArray,
        branchmask := branchArray, flowmask := flowArray, padmask := padArray }
    anchors := anchors.push (segs.map (fun s => s.anchor))
    descendants := descendants.push (segs.map (fun s => s.descendants))
    delFlags := delFlags.push delArray
    splitsTargets := splitsTargets.push (segs.map (fun s => Nat.pred s.descendants))
    prevCoals := prevCoals.push (segs.map (fun s => s.lastCoalescence))
  ({ t := usedTimes, segments := out, Xt := XtStates, X1anchor := anchors,
     descendants := descendants, del := delFlags, splitsTarget := splitsTargets,
     prevCoalescence := prevCoals }, rng)

end InhabitedOps

end torch.branching
