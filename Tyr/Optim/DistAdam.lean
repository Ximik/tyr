/-
  Tyr/Optim/DistAdam.lean

  Distributed Adam optimizer for embedding parameters.

  DistAdam is used in modded-nanogpt for:
  - Token embeddings
  - Value embeddings
  - Language model head
  - Scalar parameters

  Features:
  - Standard Adam with bias correction
  - Type-safe parameter sharding via ShardedTensor
  - Proper reduce-scatter/all-gather pattern
  - Per-parameter learning rate multipliers
  - Decoupled weight decay

  Based on modded-nanogpt's DistAdam implementation.
-/
import Tyr.TensorStruct
import Tyr.Distributed
import Tyr.Sharding

namespace torch.Optim.DistAdam

open torch
open torch.Sharding

/-- DistAdam configuration -/
structure Config where
  /-- Base learning rate -/
  lr : Float := 0.008
  /-- First moment decay (beta1) -/
  beta1 : Float := 0.65
  /-- Second moment decay (beta2) -/
  beta2 : Float := 0.95
  /-- Epsilon for numerical stability -/
  eps : Float := 1e-8
  /-- Weight decay (decoupled, AdamW-style) -/
  weightDecay : Float := 0.005
  /-- Whether to use distributed training -/
  distributed : Bool := false
  deriving Repr

instance : Inhabited Config where
  default := { lr := 0.008, beta1 := 0.65, beta2 := 0.95, eps := 1e-8,
               weightDecay := 0.005, distributed := false }

/-- State for a single parameter in DistAdam -/
structure ParamState (σ : StaticSpec) where
  /-- First moment (mean of gradients) -/
  expAvg : Tensor σ
  /-- Second moment (mean of squared gradients) -/
  expAvgSq : Tensor σ
  /-- Step count for bias correction -/
  step : Nat := 0
  deriving Repr

/-- Full DistAdam optimizer state -/
structure State (α : Type) where
  /-- Configuration -/
  config : Config
  /-- Step count -/
  step : Nat := 0
  deriving Repr

/-- Initialize DistAdam state -/
def State.init (cfg : Config) : State α := {
  config := cfg
  step := 0
}

/-- Initialize parameter state with zeros -/
def initParamState {σ : StaticSpec} (param : Tensor σ) : ParamState σ := {
  expAvg := Tensor.zerosLike param
  expAvgSq := Tensor.zerosLike param
  step := 0
}

instance {σ : StaticSpec} : TensorStruct (ParamState σ) where
  map f ps := { ps with expAvg := TensorStruct.map f ps.expAvg, expAvgSq := TensorStruct.map f ps.expAvgSq }
  mapM f ps := do
    let expAvg ← TensorStruct.mapM f ps.expAvg
    let expAvgSq ← TensorStruct.mapM f ps.expAvgSq
    return { ps with expAvg, expAvgSq }
  zipWith f ps1 ps2 := {
    expAvg := TensorStruct.zipWith f ps1.expAvg ps2.expAvg
    expAvgSq := TensorStruct.zipWith f ps1.expAvgSq ps2.expAvgSq
    step := ps1.step
  }
  fold f acc ps :=
    let acc := TensorStruct.fold f acc ps.expAvg
    TensorStruct.fold f acc ps.expAvgSq

/-- Single Adam update step for one parameter.

    Standard Adam with bias correction:
    m_t = beta1 * m_{t-1} + (1 - beta1) * g
    v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    update = lr * m_hat / (sqrt(v_hat) + eps)
    param = param - update - lr * wd * param  (decoupled WD)
-/
def stepSingle {σ : StaticSpec} (param : Tensor σ) (grad : Tensor σ) (state : ParamState σ)
    (cfg : Config) (lrMul : Float := 1.0) (wdMul : Float := 1.0) : IO (Tensor σ × ParamState σ) := do
  let step := state.step + 1
  let beta1 := cfg.beta1
  let beta2 := cfg.beta2
  let eps := cfg.eps
  let lr := cfg.lr * lrMul
  let wd := cfg.weightDecay * wdMul

  -- Detach gradient for updates
  let g := Tensor.detach grad

  -- Update first moment: m = beta1 * m + (1 - beta1) * g
  let expAvg := Tensor.Preserving.add (Tensor.Preserving.mulScalar state.expAvg beta1)
    (Tensor.Preserving.mulScalar g (1.0 - beta1))

  -- Update second moment: v = beta2 * v + (1 - beta2) * g^2
  let gSquared := Tensor.Preserving.mul g g
  let expAvgSq := Tensor.Preserving.add (Tensor.Preserving.mulScalar state.expAvgSq beta2)
    (Tensor.Preserving.mulScalar gSquared (1.0 - beta2))

  -- Bias correction
  let biasCorrection1 := 1.0 - Float.pow beta1 step.toFloat
  let biasCorrection2 := 1.0 - Float.pow beta2 step.toFloat

  -- Compute step: lr * m_hat / (sqrt(v_hat) + eps)
  let mHat := Tensor.Preserving.divScalar expAvg biasCorrection1
  let vHat := Tensor.Preserving.divScalar expAvgSq biasCorrection2
  let denominator := Tensor.Preserving.addScalar (Tensor.Preserving.sqrt vHat) eps
  let update := Tensor.Preserving.div mHat denominator
  let scaledUpdate := Tensor.Preserving.mulScalar update lr

  -- Apply weight decay (decoupled)
  let paramDecayed := if wd > 0.0 then
      let decay := Tensor.Preserving.mulScalar (Tensor.detach param) (lr * wd)
      Tensor.Preserving.sub (Tensor.detach param) decay
    else
      Tensor.detach param

  -- Apply update
  let newParam := Tensor.Preserving.sub paramDecayed scaledUpdate
  let newParam ← Tensor.parameter newParam

  let newState : ParamState σ := {
    expAvg := expAvg
    expAvgSq := expAvgSq
    step := step
  }

  return (newParam, newState)

/-- Distributed Adam step with gradient aggregation.

    For distributed training:
    1. Reduce-scatter gradients across ranks
    2. Each rank updates its shard of parameters
    3. All-gather updated parameters

    This is more efficient than all-reduce as each rank only
    updates a portion of the parameters.
-/
private def canShardFirstDim (s : Shape) (worldSize : UInt64) : Bool :=
  worldSize > 0 && s.size > 0 && s[0]! % worldSize == 0

private def firstDimShardShape (s : Shape) (worldSize : UInt64) : Shape :=
  if s.size == 0 then
    s
  else
    #[s[0]! / worldSize] ++ s[1:].toArray

/-- Slice the leading axis of a tensor to `[start, start + len)`, re-validating
    the runtime handle against the computed target shape. -/
private def sliceFirstDim {σ : StaticSpec} (t : Tensor σ) (start len : UInt64)
    (shape' : Shape) : IO (Tensor (σ.withShape shape')) := do
  match Tensor.ofRaw (RawTensor.slice σ.shape t.raw 0 start len) with
  | .ok r => pure r
  | .error message => throw (IO.userError s!"DistAdam.stepDistributed slice: {message}")

def stepDistributed {σ : StaticSpec} (param : Tensor σ) (grad : Tensor σ) (state : ParamState σ)
    (cfg : Config) (lrMul : Float := 1.0) (wdMul : Float := 1.0) : IO (Tensor σ × ParamState σ) := do
  if !cfg.distributed then
    return (← stepSingle param grad state cfg lrMul wdMul)

  let isDist ← dist.isInitialized
  if !isDist then
    return (← stepSingle param grad state cfg lrMul wdMul)

  let worldSize ← dist.getWorldSize
  let rank ← dist.getRank
  if worldSize <= 1 then
    return (← stepSingle param grad state cfg lrMul wdMul)

  -- Match nanochat DistAdamW behavior where the first dimension must shard evenly.
  -- If not shardable, fall back to all-reduce + local step for compatibility.
  if !canShardFirstDim σ.shape worldSize then
    let gradReduced := Tensor.detach grad
    let gradReduced ← dist.allReduce gradReduced .avg
    return (← stepSingle param gradReduced state cfg lrMul wdMul)

  let step := state.step + 1
  let beta1 := cfg.beta1
  let beta2 := cfg.beta2
  let eps := cfg.eps
  let lr := cfg.lr * lrMul
  let wd := cfg.weightDecay * wdMul

  let firstDim := σ.shape[0]!
  let localRows := firstDim / worldSize
  let start := rank * localRows
  let localShape := firstDimShardShape σ.shape worldSize

  -- 1) materialize local parameter/state slices on the parameter device
  let paramLocal ← sliceFirstDim param start localRows localShape
  let expAvgBase := state.expAvg
  let expAvgSqBase := state.expAvgSq
  let expAvgLocal ← sliceFirstDim expAvgBase start localRows localShape
  let expAvgSqLocal ← sliceFirstDim expAvgSqBase start localRows localShape

  -- 2) reduce-scatter averaged gradient shard
  let gradLocal := (← dist.reduceScatter worldSize grad .avg)
  let gradLocal : Tensor (σ.withShape localShape) ←
    match Tensor.ofRaw gradLocal.raw with
    | .ok t => pure t
    | .error message => throw (IO.userError s!"DistAdam.stepDistributed gradient shard: {message}")
  let g := Tensor.detach gradLocal

  -- 3) local Adam update on shard
  let newExpAvg := Tensor.Preserving.add (Tensor.Preserving.mulScalar expAvgLocal beta1)
    (Tensor.Preserving.mulScalar g (1.0 - beta1))
  let gSquared := Tensor.Preserving.mul g g
  let newExpAvgSq := Tensor.Preserving.add (Tensor.Preserving.mulScalar expAvgSqLocal beta2)
    (Tensor.Preserving.mulScalar gSquared (1.0 - beta2))
  let biasCorrection1 := 1.0 - Float.pow beta1 step.toFloat
  let biasCorrection2 := 1.0 - Float.pow beta2 step.toFloat
  let mHat := Tensor.Preserving.divScalar newExpAvg biasCorrection1
  let vHat := Tensor.Preserving.divScalar newExpAvgSq biasCorrection2
  let denominator := Tensor.Preserving.addScalar (Tensor.Preserving.sqrt vHat) eps
  let update := Tensor.Preserving.div mHat denominator
  let scaledUpdate := Tensor.Preserving.mulScalar update lr

  let paramDecayed :=
    if wd > 0.0 then
      let decay := Tensor.Preserving.mulScalar paramLocal (lr * wd)
      Tensor.Preserving.sub paramLocal decay
    else
      paramLocal
  let newLocalParam := Tensor.Preserving.sub paramDecayed scaledUpdate

  -- 4) all-gather updated local shard and optimizer shards back to replicated form
  let gatherAs (label : String) (shard : Tensor (σ.withShape localShape)) : IO (Tensor σ) := do
    let gathered ← dist.allGather worldSize shard
    match Tensor.ofRaw (σ := σ) gathered.raw with
    | .ok t => pure t
    | .error message => throw (IO.userError s!"DistAdam.stepDistributed {label}: {message}")
  let gatheredParam ← gatherAs "parameter" newLocalParam
  let gatheredExpAvg ← gatherAs "first moment" newExpAvg
  let gatheredExpAvgSq ← gatherAs "second moment" newExpAvgSq

  let newParam ← Tensor.parameter (Tensor.detach gatheredParam)
  let newState : ParamState σ := {
    expAvg := gatheredExpAvg
    expAvgSq := gatheredExpAvgSq
    step := step
  }
  return (newParam, newState)

/-- Get learning rate multiplier for embedding parameters.

    modded-nanogpt uses 75x for embeddings.
-/
def embeddingLrMul : Float := 75.0

/-- Get learning rate multiplier for scalar parameters.

    modded-nanogpt uses 5x for scalars.
-/
def scalarLrMul : Float := 5.0

/-- Initialize a map of parameter states from a model structure -/
def initParamStates [TensorStruct α] (_model : α) : Array (Sigma ParamState) :=
  -- Simplified: in a full implementation, this would create states
  -- for each parameter in the model
  #[]

/-! ## Type-Safe Sharded Adam -/

/-- State for a sharded parameter in DistAdam.

    The optimizer state (moments) is sharded in the same way as the parameter,
    so each rank only stores and updates its portion.
-/
structure ShardedParamState (σ : StaticSpec) (rank worldSize : UInt64) where
  /-- First moment (sharded) -/
  expAvg : ShardedTensor σ rank worldSize
  /-- Second moment (sharded) -/
  expAvgSq : ShardedTensor σ rank worldSize
  /-- Step count for bias correction -/
  step : Nat := 0
  deriving Repr

/-- Initialize sharded parameter state with zeros -/
def initShardedParamState {σ : StaticSpec} {rank worldSize : UInt64}
    (_param : ShardedTensor σ rank worldSize) : ShardedParamState σ rank worldSize :=
  let shardSpec := σ.withShape (shardedShape σ.shape ⟨.first, worldSize, rank⟩)
  {
    expAvg := ⟨Tensor.zerosFor shardSpec, none, true⟩
    expAvgSq := ⟨Tensor.zerosFor shardSpec, none, true⟩
    step := 0
  }

/-- Adam step for a sharded parameter.

    The sharding pattern for DistAdam:
    1. fullGrad arrives from backward pass
    2. Reduce-scatter: each rank gets its gradient shard (summed across ranks)
    3. Each rank updates its parameter shard with its gradient shard
    4. State (moments) stays sharded - no communication needed
    5. Parameters are gathered when needed for forward pass

    This is more memory-efficient than all-reduce because:
    - Each rank only stores 1/worldSize of the parameter
    - Gradient communication is reduce-scatter (same bandwidth as all-reduce)
    - Parameter gather only happens once per forward pass
-/
def stepSharded {σ : StaticSpec} {rank worldSize : UInt64}
    (param : ShardedTensor σ rank worldSize)
    (fullGrad : Tensor σ)
    (state : ShardedParamState σ rank worldSize)
    (cfg : Config) (lrMul wdMul : Float := 1.0)
    : IO (ShardedTensor σ rank worldSize × ShardedParamState σ rank worldSize) := do

  let step := state.step + 1
  let beta1 := cfg.beta1
  let beta2 := cfg.beta2
  let eps := cfg.eps
  let lr := cfg.lr * lrMul
  let wd := cfg.weightDecay * wdMul

  -- 1. Reduce-scatter gradient to get this rank's portion
  let localGrad ← param.scatterGrad fullGrad

  -- Detach for optimizer updates
  let g := Tensor.detach localGrad

  -- 2. Update first moment: m = beta1 * m + (1 - beta1) * g
  let expAvg := Tensor.Preserving.add (Tensor.Preserving.mulScalar state.expAvg.shard beta1)
    (Tensor.Preserving.mulScalar g (1.0 - beta1))

  -- 3. Update second moment: v = beta2 * v + (1 - beta2) * g^2
  let gSquared := Tensor.Preserving.mul g g
  let expAvgSq := Tensor.Preserving.add (Tensor.Preserving.mulScalar state.expAvgSq.shard beta2)
    (Tensor.Preserving.mulScalar gSquared (1.0 - beta2))

  -- 4. Bias correction
  let biasCorrection1 := 1.0 - Float.pow beta1 step.toFloat
  let biasCorrection2 := 1.0 - Float.pow beta2 step.toFloat

  -- 5. Compute update: lr * m_hat / (sqrt(v_hat) + eps)
  let mHat := Tensor.Preserving.divScalar expAvg biasCorrection1
  let vHat := Tensor.Preserving.divScalar expAvgSq biasCorrection2
  let denominator := Tensor.Preserving.addScalar (Tensor.Preserving.sqrt vHat) eps
  let update := Tensor.Preserving.div mHat denominator
  let scaledUpdate := Tensor.Preserving.mulScalar update lr

  -- 6. Apply weight decay (decoupled) to local shard
  let paramLocal := Tensor.detach param.shard
  let paramDecayed := if wd > 0.0 then
      let decay := Tensor.Preserving.mulScalar paramLocal (lr * wd)
      Tensor.Preserving.sub paramLocal decay
    else
      paramLocal

  -- 7. Apply update to local shard
  let newLocal := Tensor.Preserving.sub paramDecayed scaledUpdate
  let newLocal ← Tensor.parameter newLocal

  -- 8. Update sharded parameter (mark cache as stale)
  let newParam := param.updateShard newLocal

  -- 9. Update sharded state
  let newState : ShardedParamState σ rank worldSize := {
    expAvg := state.expAvg.updateShard expAvg
    expAvgSq := state.expAvgSq.updateShard expAvgSq
    step := step
  }

  return (newParam, newState)

/-- Sharded embedding with integrated optimizer state.

    This bundles together:
    - The sharded embedding weights
    - The sharded Adam moments
    - Learning rate and weight decay multipliers

    Usage:
    ```lean
    let (output, newEmbed) ← embed.forward tokens
    -- ... backward pass produces fullGrad ...
    let newEmbed ← embed.step fullGrad cfg
    ```
-/
structure ShardedEmbeddingAdam (vocabSize dim : UInt64) (rank worldSize : UInt64)
    (dtype : DType := .Float32) (device : Device := .CPU) where
  /-- Sharded embedding weights -/
  weight : ShardedTensor ⟨#[vocabSize, dim], dtype, device⟩ rank worldSize
  /-- Sharded optimizer state -/
  optState : ShardedParamState ⟨#[vocabSize, dim], dtype, device⟩ rank worldSize
  /-- Learning rate multiplier (modded-nanogpt uses 75x) -/
  lrMul : Float := 75.0
  /-- Weight decay multiplier -/
  wdMul : Float := 1.0
  deriving Repr

/-- Initialize sharded embedding with Adam state -/
def ShardedEmbeddingAdam.init {vocabSize dim rank worldSize : UInt64} {d : DType} {dev : Device}
    (fullWeight : Tensor ⟨#[vocabSize, dim], d, dev⟩) (h : ValidShard rank worldSize)
    : ShardedEmbeddingAdam vocabSize dim rank worldSize d dev :=
  let shardedWeight := ShardedTensor.fromFull fullWeight h
  let optState := initShardedParamState shardedWeight
  {
    weight := shardedWeight
    optState := optState
    lrMul := 75.0
    wdMul := 1.0
  }

/-- Forward pass: gather weights and apply embedding lookup -/
def ShardedEmbeddingAdam.forward {vocabSize dim rank worldSize batch seq : UInt64}
    {d : DType} {dev : Device}
    (embed : ShardedEmbeddingAdam vocabSize dim rank worldSize d dev)
    (tokens : Tensor ⟨#[batch, seq], .Int64, dev⟩)
    : IO (Tensor ⟨#[batch, seq, dim], d, dev⟩ × ShardedEmbeddingAdam vocabSize dim rank worldSize d dev) := do
  let (fullWeight, newSharded) ← embed.weight.gather
  let output := Tensor.embedding tokens fullWeight
  return (output, { embed with weight := newSharded })

/-- Backward/update step: reduce-scatter grad and update local shard -/
def ShardedEmbeddingAdam.step {vocabSize dim rank worldSize : UInt64} {d : DType} {dev : Device}
    (embed : ShardedEmbeddingAdam vocabSize dim rank worldSize d dev)
    (fullGrad : Tensor ⟨#[vocabSize, dim], d, dev⟩)
    (cfg : Config)
    : IO (ShardedEmbeddingAdam vocabSize dim rank worldSize d dev) := do
  let (newWeight, newOptState) ← stepSharded
    embed.weight fullGrad embed.optState cfg embed.lrMul embed.wdMul
  return { embed with
    weight := newWeight
    optState := newOptState
  }

end torch.Optim.DistAdam
