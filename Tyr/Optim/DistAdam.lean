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
import Tyr.Module.Derive
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

derive_tensor_struct_for ParamState

/-- Single Adam update step for one parameter.

    Standard Adam with bias correction:
    m_t = beta1 * m_{t-1} + (1 - beta1) * g
    v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    update = lr * m_hat / (sqrt(v_hat) + eps)
    param = param - update - lr * wd * param  (decoupled WD)
-/
def stepSingle {σ : StaticSpec} [DType.RealFloating σ.dtype] (param : Tensor σ) (grad : Tensor σ) (state : ParamState σ)
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
  let expAvg := Tensor.add (Tensor.mulScalar state.expAvg beta1)
    (Tensor.mulScalar g (1.0 - beta1))

  -- Update second moment: v = beta2 * v + (1 - beta2) * g^2
  let gSquared := Tensor.mul g g
  let expAvgSq := Tensor.add (Tensor.mulScalar state.expAvgSq beta2)
    (Tensor.mulScalar gSquared (1.0 - beta2))

  -- Bias correction
  let biasCorrection1 := 1.0 - Float.pow beta1 step.toFloat
  let biasCorrection2 := 1.0 - Float.pow beta2 step.toFloat

  -- Compute step: lr * m_hat / (sqrt(v_hat) + eps)
  let mHat := Tensor.divScalar expAvg biasCorrection1
  let vHat := Tensor.divScalar expAvgSq biasCorrection2
  let denominator := Tensor.addScalar (Tensor.sqrt vHat) eps
  let update : Tensor _ := by simpa using Tensor.div mHat denominator
  let scaledUpdate := Tensor.mulScalar update lr

  -- Apply weight decay (decoupled)
  let paramDecayed := if wd > 0.0 then
      let decay := Tensor.mulScalar (Tensor.detach param) (lr * wd)
      Tensor.sub (Tensor.detach param) decay
    else
      Tensor.detach param

  -- Apply update
  let newParam := Tensor.sub paramDecayed scaledUpdate
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
private def canShardFirstDim (s : Shape) (worldSize : Nat) : Bool :=
  worldSize > 0 && s.size > 0 && s[0]! % worldSize == 0

private def firstDimShardShape (s : Shape) (worldSize : Nat) : Shape :=
  if s.size == 0 then
    s
  else
    #[s[0]! / worldSize] ++ s[1:].toArray

/-- Slice the leading axis of a tensor to `[start, start + len)`; the result is
    admitted at the caller's target shape. -/
private def sliceFirstDim {σ : StaticSpec} (t : Tensor σ) (start len : Nat)
    (shape' : Shape) : Tensor (σ.withShape shape') :=
  Tensor.ofRaw! (RawTensor.slice t.raw 0 start len)

def stepDistributed {σ : StaticSpec} [DType.RealFloating σ.dtype] (param : Tensor σ) (grad : Tensor σ) (state : ParamState σ)
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
  let paramLocal := sliceFirstDim param start localRows localShape
  let expAvgBase := state.expAvg
  let expAvgSqBase := state.expAvgSq
  let expAvgLocal := sliceFirstDim expAvgBase start localRows localShape
  let expAvgSqLocal := sliceFirstDim expAvgSqBase start localRows localShape

  -- 2) reduce-scatter averaged gradient shard
  let gradLocal := (← dist.reduceScatter worldSize grad .avg)
  let gradLocal : Tensor (σ.withShape localShape) ←
    pure (Tensor.ofRaw! gradLocal.raw)
  let g := Tensor.detach gradLocal

  -- 3) local Adam update on shard
  let newExpAvg := Tensor.add (Tensor.mulScalar expAvgLocal beta1)
    (Tensor.mulScalar g (1.0 - beta1))
  let gSquared := Tensor.mul g g
  let newExpAvgSq := Tensor.add (Tensor.mulScalar expAvgSqLocal beta2)
    (Tensor.mulScalar gSquared (1.0 - beta2))
  let biasCorrection1 := 1.0 - Float.pow beta1 step.toFloat
  let biasCorrection2 := 1.0 - Float.pow beta2 step.toFloat
  let mHat := Tensor.divScalar newExpAvg biasCorrection1
  let vHat := Tensor.divScalar newExpAvgSq biasCorrection2
  let denominator := Tensor.addScalar (Tensor.sqrt vHat) eps
  let update : Tensor _ := by simpa using Tensor.div mHat denominator
  let scaledUpdate := Tensor.mulScalar update lr

  let paramDecayed :=
    if wd > 0.0 then
      let decay := Tensor.mulScalar paramLocal (lr * wd)
      Tensor.sub paramLocal decay
    else
      paramLocal
  let newLocalParam := Tensor.sub paramDecayed scaledUpdate

  -- 4) all-gather updated local shard and optimizer shards back to replicated form
  let gatherAs (label : String) (shard : Tensor (σ.withShape localShape)) : IO (Tensor σ) := do
    let gathered ← dist.allGather worldSize shard
    pure (Tensor.ofRaw! (σ := σ) gathered.raw)
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

end torch.Optim.DistAdam
