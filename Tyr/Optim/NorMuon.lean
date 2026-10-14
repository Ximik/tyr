/-
  Tyr/Optim/NorMuon.lean

  NorMuon Optimizer implementation for Tyr.

  NorMuon is an advanced optimizer combining:
  - Muon's orthogonalized momentum (via Polar Express)
  - Low-rank variance reduction (Adafactor-style)
  - Decoupled weight decay (AdamW-style: `p ← p - lr_eff * wd * p`)
  - Distributed training with reduce-scatter/all-gather

  Based on modded-nanogpt's NorMuon implementation.
-/
import Tyr.TensorStruct
import Tyr.Typed.Neural
import Tyr.Distributed
import Tyr.Optim.PolarExpress

namespace torch.Optim.NorMuon

open torch

/-- Parameter labels for grouping parameters with different update rules -/
inductive ParamLabel where
  | smearGate   : ParamLabel  -- Smear gate parameters
  | attnGate    : ParamLabel  -- Attention gate parameters
  | attn        : ParamLabel  -- Attention weight matrices
  | mlp         : ParamLabel  -- MLP weight matrices
  | embed       : ParamLabel  -- Token embeddings
  | lmHead      : ParamLabel  -- Language model head
  | scalars     : ParamLabel  -- Learnable scalar parameters
  | valueEmbed  : ParamLabel  -- Value embeddings
  deriving Repr, BEq, Hashable, Inhabited

/-- Labeled parameter with metadata for optimization -/
structure LabeledParam (σ : StaticSpec) where
  /-- The parameter tensor -/
  tensor : Tensor σ
  /-- Parameter label for grouping -/
  label : ParamLabel
  /-- Learning rate multiplier (1.0 = base LR) -/
  lrMul : Float := 1.0
  /-- Weight decay multiplier (1.0 = base WD, 0.0 = no WD) -/
  wdMul : Float := 1.0
  deriving Repr

/-- NorMuon configuration -/
structure Config where
  /-- Base learning rate -/
  lr : Float := 0.023
  /-- Weight decay (decoupled, AdamW-style; per-step decay is
      `effectiveLr * weightDecay * wdMul`, 0.0 disables) -/
  weightDecay : Float := 1.2
  /-- Momentum coefficient -/
  momentum : Float := 0.95
  /-- Second moment coefficient for variance reduction -/
  beta2 : Float := 0.95
  /-- Number of Newton-Schulz iterations for Polar Express -/
  numIters : UInt64 := 5
  /-- Whether to use distributed training -/
  distributed : Bool := false
  /-- World size for distributed (set automatically if distributed=true) -/
  worldSize : UInt64 := 1
  deriving Repr

instance : Inhabited Config where
  default := { lr := 0.023, weightDecay := 1.2, momentum := 0.95, beta2 := 0.95,
               numIters := 5, distributed := false, worldSize := 1 }

/-- State for a single parameter in NorMuon -/
structure ParamState (σ : StaticSpec) where
  /-- Momentum buffer (EMA of orthogonalized gradients) -/
  momentumBuffer : Option (Tensor σ) := none
  /-- Second moment for variance reduction (scalar; mean of squared gradients) -/
  secondMoment : Option (Tensor σ.sumSpec) := none
  /-- Step count for bias correction -/
  step : Nat := 0
  deriving Repr

/-- Initialize state for a parameter. -/
def initParamState {σ : StaticSpec} (_param : Tensor σ) : ParamState σ := {
  momentumBuffer := none
  secondMoment := none
  step := 0
}

instance {σ : StaticSpec} : TensorStruct (ParamState σ) where
  map f ps := { ps with
    momentumBuffer := ps.momentumBuffer.map (TensorStruct.map f),
    secondMoment := ps.secondMoment.map (TensorStruct.map f)
  }
  mapM f ps := do
    let momentumBuffer ← match ps.momentumBuffer with
      | some t => some <$> TensorStruct.mapM f t
      | none => pure none
    let secondMoment ← match ps.secondMoment with
      | some t => some <$> TensorStruct.mapM f t
      | none => pure none
    return { ps with momentumBuffer, secondMoment }
  zipWith f ps1 ps2 := {
    momentumBuffer := match ps1.momentumBuffer, ps2.momentumBuffer with
      | some t1, some t2 => some (TensorStruct.zipWith f t1 t2)
      | _, _ => none
    secondMoment := match ps1.secondMoment, ps2.secondMoment with
      | some t1, some t2 => some (TensorStruct.zipWith f t1 t2)
      | _, _ => none
    step := ps1.step
  }
  fold f acc ps :=
    let acc := match ps.momentumBuffer with
      | some t => TensorStruct.fold f acc t
      | none => acc
    match ps.secondMoment with
      | some t => TensorStruct.fold f acc t
      | none => acc

/-- Full NorMuon optimizer state -/
structure State where
  /-- Configuration -/
  config : Config
  /-- Per-parameter states, existentially packed over their specs -/
  paramStates : Array (Sigma ParamState)
  /-- Global step count -/
  globalStep : Nat := 0
  deriving Repr

/-- Initialize NorMuon state from configuration -/
def State.init (cfg : Config) : IO State := do
  return {
    config := cfg
    paramStates := #[]
    globalStep := 0
  }

/-- Apply variance reduction to orthogonalized gradient.

    Uses low-rank second moment estimation (Adafactor-style):
    v_t = beta2 * v_{t-1} + (1 - beta2) * mean(g^2)
    g_normalized = g / sqrt(v_t + eps)

    This reduces variance of the update while maintaining direction.
-/
def applyVarianceReduction {σ : StaticSpec} (grad : Tensor σ)
    (secondMoment : Option (Tensor σ.sumSpec))
    (beta2 : Float) (step : Nat) : IO (Tensor σ × Tensor σ.sumSpec) := do
  let eps : Float := 1e-8

  -- Compute mean of squared gradients over all entries
  let gradSquared := Mul.mul grad grad
  let numel := (TensorSpec.numelOfShape σ.shape).toFloat
  let newSecondMoment :=
    Tensor.Preserving.divScalar (Tensor.sumAll gradSquared) numel

  -- Update EMA of second moment
  let updatedSecondMoment ← match secondMoment with
    | some sm =>
      let b2 := Tensor.Preserving.mulScalar sm beta2
      let g2 := Tensor.Preserving.mulScalar newSecondMoment (1.0 - beta2)
      pure (Add.add b2 g2)
    | none => pure newSecondMoment

  -- Bias correction
  let biasCorrection := 1.0 - Float.pow beta2 (step + 1).toFloat
  let correctedMoment := Tensor.Preserving.divScalar updatedSecondMoment biasCorrection

  -- Normalize gradient by sqrt(second moment)
  let scale := 1.0 / Float.sqrt (Tensor.item correctedMoment + eps)
  let normalizedGrad := Tensor.Preserving.mulScalar grad scale

  return (normalizedGrad, updatedSecondMoment)

/-- Apply cautious weight decay.

    Only applies weight decay when gradient and parameter have the same sign:
    mask = sign(grad) == sign(param)
    update = grad + wd * mask * param

    This prevents weight decay from fighting the gradient.
-/
def applyCautiousWeightDecay {σ : StaticSpec} (param grad : Tensor σ) (wd : Float) :
    IO (Tensor σ) := do
  if wd == 0.0 then
    return grad
  else
    dist.cautiousUpdate param grad 1.0 wd

/-- Update momentum buffer with orthogonalized gradient.

    Matches nanochat Muon:
    buf_t = momentum * buf_{t-1} + (1 - momentum) * g
-/
def updateMomentum {σ : StaticSpec} (orthGrad : Tensor σ)
    (momentumBuffer : Option (Tensor σ)) (momentum : Float) : Tensor σ :=
  match momentumBuffer with
  | some buf =>
    let decayed := Tensor.Preserving.mulScalar buf momentum
    let scaled := Tensor.Preserving.mulScalar orthGrad (1.0 - momentum)
    Add.add decayed scaled
  | none => Tensor.Preserving.mulScalar orthGrad (1.0 - momentum)

/-- Compute aspect ratio scaling for a tensor shape.
    For 2D matrices: sqrt(max(1, height/width))
    For other tensors: 1.0

    This scales the learning rate for tall matrices (more rows than columns),
    following nanochat/modded-nanogpt's approach. -/
def aspectRatioScale (shape : Shape) : Float :=
  if shape.size >= 2 then
    let height := shape[shape.size - 2]!.toFloat
    let width := shape[shape.size - 1]!.toFloat
    if height > width then
      Float.sqrt (height / width)
    else
      1.0
  else
    1.0

/-- Single Muon-style update step for one parameter.

    Matches nanochat's Muon semantics:
    1. Orthogonalize gradient via Newton-Schulz
    2. Update momentum buffer
    3. Apply Nesterov-style blended update
    4. Apply aspect-ratio learning-rate scaling
    5. Apply decoupled weight decay (AdamW-style, matching DistAdam):
       `param ← param - effectiveLr * weightDecay * wdMul * param`
-/
def stepSingle {σ : StaticSpec} (param : Tensor σ) (grad : Tensor σ) (state : ParamState σ)
    (cfg : Config) (lrMul wdMul : Float) : IO (Tensor σ × ParamState σ) := do
  -- Match nanochat ordering:
  -- 1) momentum buffer update on raw gradients
  -- 2) Nesterov blend
  -- 3) orthogonalize blended update
  let gRaw := Tensor.detach grad
  let newMomentum := updateMomentum gRaw state.momentumBuffer cfg.momentum
  let nesterovGrad := Add.add (Tensor.Preserving.mulScalar gRaw (1.0 - cfg.momentum))
    (Tensor.Preserving.mulScalar newMomentum cfg.momentum)
  -- Zero gradients should stay zero; this avoids entering Newton-Schulz on
  -- exact zeros while preserving reference Muon semantics.
  let nesterovAbsMax := Tensor.item (Tensor.maxAll (Tensor.abs nesterovGrad))
  let aspectScale := aspectRatioScale σ.shape
  let effectiveLr := cfg.lr * lrMul * aspectScale
  let wd := cfg.weightDecay * wdMul
  -- Apply weight decay (decoupled, AdamW-style), matching DistAdam.
  let paramDecayed := if wd > 0.0 then
      let decay := Tensor.Preserving.mulScalar (Tensor.detach param) (effectiveLr * wd)
      Sub.sub (Tensor.detach param) decay
    else
      Tensor.detach param
  let update ← if nesterovAbsMax == 0.0 then
      pure nesterovGrad
    else do
      let orthGrad ← PolarExpress.muonOrthogonalize nesterovGrad cfg.numIters
      -- Match native Muon: scale in BF16, then restore the parameter dtype
      -- for subtraction. Never claim the BF16 kernel result has the input dtype.
      pure (Tensor.castLike param (Tensor.mulScalar (Tensor.detach orthGrad) effectiveLr))
  let newParam := Sub.sub paramDecayed update
  let newParam := Tensor.setRequiresGrad (Tensor.detach newParam) true

  let newState : ParamState σ := {
    momentumBuffer := some newMomentum
    secondMoment := none
    step := state.step + 1
  }

  return (newParam, newState)

/-- Distributed Muon step with owner-based updates.

    This mirrors nanochat DistMuon semantics:
    1. Average gradients across ranks.
    2. Only `ownerRank` computes the Muon update and advances state.
    3. Broadcast updated parameter (and momentum buffer) from owner.
-/
def stepDistributedOwner {σ : StaticSpec} (param : Tensor σ) (grad : Tensor σ)
    (state : ParamState σ)
    (cfg : Config) (lrMul wdMul : Float) (ownerRank : UInt64) :
    IO (Tensor σ × ParamState σ) := do
  let isDist ← dist.isInitialized
  if !cfg.distributed || !isDist then
    return (← stepSingle param grad state cfg lrMul wdMul)

  let worldSize ← dist.getWorldSize
  if worldSize <= 1 then
    return (← stepSingle param grad state cfg lrMul wdMul)

  let rank ← dist.getRank
  let gradAvg := Tensor.detach grad
  let gradAvg ← dist.allReduce gradAvg .avg

  let (ownerParam, ownerState) ←
    if rank == ownerRank then
      stepSingle param gradAvg state cfg lrMul wdMul
    else
      pure (param, state)

  let paramSynced := if rank == ownerRank then ownerParam else Tensor.detach param
  let paramSynced ← dist.broadcast paramSynced ownerRank

  let momentumSynced :=
    if rank == ownerRank then
      ownerState.momentumBuffer.getD (Tensor.zerosLike paramSynced)
    else
      Tensor.zerosLike paramSynced
  let momentumSynced ← dist.broadcast momentumSynced ownerRank

  let newParam := Tensor.setRequiresGrad (Tensor.detach paramSynced) true
  let newState : ParamState σ := {
    momentumBuffer := some momentumSynced
    secondMoment := ownerState.secondMoment
    step := state.step + 1
  }
  return (newParam, newState)

/-- Local (non-distributed) Muon step for a homogeneous parameter group. -/
private def stepGroupLocal {σ : StaticSpec}
    (params : Array (Tensor σ))
    (grads : Array (Tensor σ))
    (states : Array (ParamState σ))
    (cfg : Config) (lrMul wdMul : Float)
    : IO (Array (Tensor σ) × Array (ParamState σ)) := do
  let mut outParams : Array (Tensor σ) := #[]
  let mut outStates : Array (ParamState σ) := #[]
  for i in [:params.size] do
    let p := params[i]!
    let g := grads[i]?.getD (Tensor.zerosLike p)
    let st := states[i]?.getD (initParamState p)
    let (p', st') ← stepSingle p g st cfg lrMul wdMul
    outParams := outParams.push p'
    outStates := outStates.push st'
  return (outParams, outStates)

/-- Distributed Muon step for a homogeneous parameter group.

    This matches nanochat DistMuon group semantics:
    - Parameters are processed in blocks of `worldSize`.
    - For each block, rank `r` owns parameter `base + r` (if it exists).
    - Gradients are averaged with `reduce_scatter` over explicit per-rank lists.
    - Updated owner parameters are replicated with `all_gather`.
-/
def stepDistributedGroup {σ : StaticSpec}
    (params : Array (Tensor σ))
    (grads : Array (Tensor σ))
    (states : Array (ParamState σ))
    (cfg : Config) (lrMul wdMul : Float := 1.0)
    : IO (Array (Tensor σ) × Array (ParamState σ)) := do
  if params.isEmpty then
    return (params, states)

  let isDist ← dist.isInitialized
  if !cfg.distributed || !isDist then
    return (← stepGroupLocal params grads states cfg lrMul wdMul)

  let worldSize ← dist.getWorldSize
  if worldSize <= 1 then
    return (← stepGroupLocal params grads states cfg lrMul wdMul)

  let rank ← dist.getRank
  let ws := worldSize.toNat
  let rankNat := rank.toNat
  let zeroBuf := Tensor.zerosLike params[0]!
  let mut outParams := params
  let mut outStates := states
  let mut base : Nat := 0

  while base < params.size do
    -- Build per-rank inputs for reduce-scatter (pad with zeros for short tails).
    let mut rsInputs : Array (Tensor σ) := #[]
    for j in [:ws] do
      let idx := base + j
      let g := grads[idx]?.getD zeroBuf
      rsInputs := rsInputs.push g

    let ownerIdx := base + rankNat
    let gradOwner := if ownerIdx < params.size then Tensor.zerosLike params[ownerIdx]!
      else Tensor.zerosLike zeroBuf
    let gradOwner ← dist.reduceScatterList gradOwner rsInputs .avg

    let mut ownerParam := if ownerIdx < params.size then params[ownerIdx]! else zeroBuf
    let mut ownerState := states[ownerIdx]?.getD (initParamState ownerParam)
    if ownerIdx < params.size then
      let (p', st') ← stepSingle ownerParam gradOwner ownerState cfg lrMul wdMul
      ownerParam := p'
      ownerState := st'

    -- Gather updated owner parameters back to all ranks for this block.
    let mut gatherParams : Array (Tensor σ) := #[]
    for j in [:ws] do
      let idx := base + j
      let t := if idx < params.size then params[idx]! else zeroBuf
      gatherParams := gatherParams.push (Tensor.detach t)
    let agInput := if ownerIdx < params.size then ownerParam else zeroBuf
    gatherParams ← dist.allGatherList gatherParams agInput

    -- Gather owner momentum buffers to keep replicated checkpoint-compatible state.
    let mut gatherMomentum : Array (Tensor σ) := #[]
    for j in [:ws] do
      let idx := base + j
      let t := if idx < params.size then params[idx]! else zeroBuf
      gatherMomentum := gatherMomentum.push (Tensor.zerosLike t)
    let agMomentum := ownerState.momentumBuffer.getD (Tensor.zerosLike agInput)
    gatherMomentum ← dist.allGatherList gatherMomentum agMomentum

    for j in [:ws] do
      let idx := base + j
      if idx < params.size then
        let pSynced := Tensor.setRequiresGrad (Tensor.detach gatherParams[j]!) true
        outParams := outParams.set! idx pSynced
        let prevState := outStates[idx]?.getD (initParamState pSynced)
        let stSynced : ParamState σ := {
          momentumBuffer := some (Tensor.detach gatherMomentum[j]!)
          secondMoment := prevState.secondMoment
          step := prevState.step + 1
        }
        outStates := outStates.set! idx stSynced

    base := base + ws

  return (outParams, outStates)

/-- Step function for parameters using AdamW-like updates (non-orthogonalized).

    Used for embeddings, scalars, and other non-matrix parameters where
    orthogonalization doesn't apply.

    Note: weight decay here is coupled L2-style (`grad + wd * param`, so the
    decay is scaled by the learning rate through the momentum update), while
    `stepSingle` uses decoupled AdamW-style decay
    (`param ← param - effectiveLr * weightDecay * wdMul * param`). The two
    paths therefore apply `weightDecay` inconsistently; the coupled form also
    interacts with momentum, unlike true AdamW.
-/
def stepAdamLike {σ : StaticSpec} (param : Tensor σ) (grad : Tensor σ) (state : ParamState σ)
    (cfg : Config) (lrMul wdMul : Float) : IO (Tensor σ × ParamState σ) := do
  -- Simple SGD with momentum for non-matrix params
  let effectiveWd := cfg.weightDecay * wdMul
  let gradWithWd := if effectiveWd > 0 then
      Add.add grad (Tensor.Preserving.mulScalar param effectiveWd)
    else
      grad

  -- Update momentum
  let newMomentum := updateMomentum gradWithWd state.momentumBuffer cfg.momentum

  -- Apply update
  let effectiveLr := cfg.lr * lrMul
  let update := Tensor.Preserving.mulScalar newMomentum effectiveLr
  let newParam := Sub.sub param update
  let newParam := Tensor.setRequiresGrad (Tensor.detach newParam) true

  let newState : ParamState σ := {
    momentumBuffer := some newMomentum
    secondMoment := state.secondMoment
    step := state.step + 1
  }

  return (newParam, newState)

/-- Determine if a parameter should use orthogonalized updates.

    Matrix parameters (attention, MLP weights) use orthogonalization.
    Embeddings, scalars, and biases use standard momentum updates.
-/
def shouldOrthogonalize (label : ParamLabel) : Bool :=
  match label with
  | .smearGate => true
  | .attnGate => true
  | .attn => true
  | .mlp => true
  | .embed => false
  | .lmHead => false
  | .scalars => false
  | .valueEmbed => false

/-- Get default learning rate multiplier for a parameter label.

    Based on modded-nanogpt:
    - Embeddings: 75x (need larger updates)
    - Scalars: 5x
    - Others: 1x
-/
def defaultLrMul (label : ParamLabel) : Float :=
  match label with
  | .embed => 75.0
  | .valueEmbed => 75.0
  | .lmHead => 1.0
  | .scalars => 5.0
  | _ => 1.0

/-- Get default weight decay multiplier for a parameter label.

    Based on modded-nanogpt:
    - Scalars: 0x (no weight decay)
    - Others: 1x
-/
def defaultWdMul (label : ParamLabel) : Float :=
  match label with
  | .scalars => 0.0
  | _ => 1.0

/-- Momentum schedule used by nanochat base/mid training.

    Warmup only:
    - Warmup: 300 steps (0.85 -> baseMomentum)
    - Then constant at baseMomentum
-/
def getMomentum (step totalSteps : Nat) (baseMomentum : Float := 0.95)
    (warmupSteps : Nat := 300) (cooldownSteps : Nat := 50) : Float :=
  let _ := totalSteps
  let _ := cooldownSteps
  let minMomentum := 0.85
  if step < warmupSteps then
    -- Linear warmup
    let progress := step.toFloat / warmupSteps.toFloat
    minMomentum + progress * (baseMomentum - minMomentum)
  else
    baseMomentum

end torch.Optim.NorMuon
