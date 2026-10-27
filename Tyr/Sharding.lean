import Tyr.Distributed

/-!
# Tyr.Sharding

`Tyr.Sharding` provides type-aware tensor sharding for distributed data-parallel training.
It models local shards and full-tensor reconstruction with explicit rank/world information,
so sharding behavior is encoded in types and helper APIs.

## Major Components

- `ShardSpec`/`ShardDim` and shard-size/offset computation helpers.
- Type-level shape projection (`shardedShape`) for local shard tensors.
- `ShardedTensor` structure with cache/gather/scatter semantics.
- Validation helpers (`ValidShard`) and tensor-structure integration.

## Design Intent

- Keep sharding semantics explicit in the type/domain model.
- Support reduce-scatter/all-gather optimizer patterns efficiently.
- Enable local-shard update workflows with controlled full-tensor materialization.

## Scope

This module defines sharding data structures and core operations.
Distributed orchestration and optimizer policy should compose this with `Tyr.Distributed`.
-/

namespace torch.Sharding

open torch

/-! ## Sharding Specification -/

/-- Dimension along which to shard (for multi-dimensional tensors) -/
inductive ShardDim where
  | first : ShardDim   -- Shard along first dimension
  | last : ShardDim    -- Shard along last dimension
  | dim (n : Nat) : ShardDim
  deriving Repr, BEq, Inhabited

/-- Resolve a shard dimension to a concrete axis for a tensor of the given rank.
    Returns `none` when the axis is out of range, in which case sharding
    operations treat the tensor as replicated rather than sharded. -/
def ShardDim.resolve? (sd : ShardDim) (rank : Nat) : Option Nat :=
  let d := match sd with
    | .first => 0
    | .last => rank - 1
    | .dim n => n
  if d < rank then some d else none

/-- Specification for how a tensor is sharded across ranks -/
structure ShardSpec where
  /-- Which dimension to shard along -/
  shardDim : ShardDim := .first
  /-- Total number of shards (world size) -/
  numShards : UInt64
  /-- This shard's index (rank) -/
  shardIdx : UInt64
  deriving Repr, BEq

/-- Compute the shard size for a given full size -/
def shardSize (fullSize numShards shardIdx : UInt64) : UInt64 :=
  let baseSize := fullSize / numShards
  let remainder := fullSize % numShards
  if shardIdx < remainder then baseSize + 1 else baseSize

/-- Compute shard offset (starting index) -/
def shardOffset (fullSize numShards shardIdx : UInt64) : UInt64 :=
  let baseSize := fullSize / numShards
  let remainder := fullSize % numShards
  -- First `remainder` shards get one extra element
  if shardIdx < remainder then
    shardIdx * (baseSize + 1)
  else
    remainder * (baseSize + 1) + (shardIdx - remainder) * baseSize

/-! ## Type-level Shape Computation -/

/-- Compute the sharded shape from full shape.

    The axis selected by `spec.shardDim` is divided across `spec.numShards`
    shards; when the axis size is not evenly divisible, `shardSize` distributes
    the remainder over the first shards. If `spec.shardDim` does not resolve to
    a valid axis of `s`, the shape is returned unchanged (replicated tensor).
-/
def shardedShape (s : Shape) (spec : ShardSpec) : Shape :=
  match spec.shardDim.resolve? s.size with
  | some d => replaceAtDim s d (shardSize (s.getD d 0) spec.numShards spec.shardIdx)
  | none => s

/-! ## Sharded Tensor -/

/-- A tensor that is sharded across distributed ranks.

    The type encodes:
    - `σ`: the static spec of the full (gathered) tensor
    - `rank`: This process's rank (shard index)
    - `worldSize`: Total number of processes (shards)

    The actual local data is stored as a tensor with shape
    `shardedShape σ.shape spec`, retaining the full tensor's dtype and device policy.
-/
structure ShardedTensor (σ : StaticSpec) (rank worldSize : UInt64)
    (shardDim : ShardDim := .first) where
  /-- The local shard (this rank's portion) -/
  shard : Tensor (σ.withShape (shardedShape σ.shape ⟨shardDim, worldSize, rank⟩))
  /-- Cached full tensor (populated after gather) -/
  cachedFull : Option (Tensor σ) := none
  /-- Whether the cache is stale (shard was modified) -/
  cacheIsStale : Bool := true
  deriving Repr

/-- Proof obligation for valid sharding -/
abbrev ValidShard (rank worldSize : UInt64) := rank < worldSize

/-! ## Sharded Tensor Operations -/

/-- Create a sharded tensor by slicing a full tensor along `shardDim`.

    This is used during initialization when one rank has the full tensor
    and needs to extract its shard. If `shardDim` does not resolve to a valid
    axis of `σ.shape`, the full tensor is kept as the (replicated) shard.
    The slice's runtime shape is validated against the computed shard spec.
-/
def ShardedTensor.fromFull {σ : StaticSpec} {rank worldSize : UInt64}
    (full : Tensor σ) (_h : ValidShard rank worldSize)
    (shardDim : ShardDim := .first) : ShardedTensor σ rank worldSize shardDim :=
  -- Compute this rank's slice
  let spec : ShardSpec := ⟨shardDim, worldSize, rank⟩
  let shardShape := shardedShape σ.shape spec

  -- Slice along the sharded dimension
  let shardedRaw := match shardDim.resolve? σ.shape.size with
    | some d =>
      let fullDim := σ.shape.getD d 0
      let offset := shardOffset fullDim worldSize rank
      let size := shardSize fullDim worldSize rank
      RawTensor.slice σ.shape full.raw d.toUInt64 offset size
    | none => full.raw

  let shard : Tensor (σ.withShape shardShape) :=
    match Tensor.ofRaw shardedRaw with
    | .ok t => t
    | .error message => nativeInvariantFailure s!"ShardedTensor.fromFull: {message}"

  { shard := shard
    cachedFull := some full
    cacheIsStale := false
  }

/-- Permutation that moves axis `d` of a `rank`-D tensor to the front. -/
private def moveAxisToFrontPerm (rank d : Nat) : Array UInt64 :=
  (List.range rank).toArray.map fun i =>
    if i = 0 then d.toUInt64
    else if i ≤ d then (i - 1).toUInt64
    else i.toUInt64

/-- Inverse of `moveAxisToFrontPerm`: moves the front axis back to position `d`. -/
private def moveFrontToAxisPerm (rank d : Nat) : Array UInt64 :=
  (List.range rank).toArray.map fun i =>
    if i < d then (i + 1).toUInt64
    else if i = d then 0
    else i.toUInt64

/-- Gather all shards to reconstruct the full tensor.

    Uses the all-gather collective operation, concatenating the per-rank shards
    along the sharded dimension. All-gather requires equal-sized shards, so the
    sharded dimension must be divisible by `worldSize`; otherwise an error is
    thrown. The gathered handle is re-validated against the full spec before
    re-entering the typed structure. The result is cached for repeated access.
-/
def ShardedTensor.gather {σ : StaticSpec} {rank worldSize : UInt64} {sd : ShardDim}
    (sharded : ShardedTensor σ rank worldSize sd) : IO (Tensor σ × ShardedTensor σ rank worldSize sd) := do
  -- Check cache first
  if !sharded.cacheIsStale then
    match sharded.cachedFull with
    | some full => return (full, sharded)
    | none => pure ()

  -- Need to gather
  let full ← match sd.resolve? σ.shape.size with
    | none =>
      -- Not actually sharded: the shard is a full replica
      match Tensor.ofRaw (σ := σ) sharded.shard.raw with
      | .ok t => pure t
      | .error message => throw (IO.userError s!"ShardedTensor.gather: {message}")
    | some d =>
      let fullDim := σ.shape.getD d 0
      if fullDim % worldSize != 0 then
        throw (IO.userError
          s!"ShardedTensor.gather: sharded dimension of size {fullDim} is not divisible by worldSize {worldSize}")
      if d = 0 then
        -- allGather concatenates along dimension 0
        let gathered ← dist.allGather worldSize sharded.shard
        match Tensor.ofRaw (σ := σ) gathered.raw with
        | .ok t => pure t
        | .error message => throw (IO.userError s!"ShardedTensor.gather: {message}")
      else
        -- Move the sharded axis to the front, gather, then move it back
        let toFront := moveAxisToFrontPerm σ.shape.size d
        let fromFront := moveFrontToAxisPerm σ.shape.size d
        let gathered ← dist.allGather worldSize (Tensor.permute sharded.shard toFront)
        let unpermuted := Tensor.permute gathered fromFront
        match Tensor.ofRaw (σ := σ) unpermuted.raw with
        | .ok t => pure t
        | .error message => throw (IO.userError s!"ShardedTensor.gather: {message}")

  let newSharded := { sharded with
    cachedFull := some full
    cacheIsStale := false
  }
  return (full, newSharded)

/-- Scatter gradients to get this rank's portion.

    Used during backward pass: full gradient -> local gradient shard.
    Uses the reduce-scatter collective, which chunks the leading axis, so this
    is only valid for `.first`-sharded tensors whose leading dimension is
    divisible by `worldSize`; the result is re-validated against the computed
    shard spec and an error is thrown otherwise.
-/
def ShardedTensor.scatterGrad {σ : StaticSpec} {rank worldSize : UInt64} {sd : ShardDim}
    (_sharded : ShardedTensor σ rank worldSize sd)
    (fullGrad : Tensor σ) : IO (Tensor (σ.withShape (shardedShape σ.shape ⟨sd, worldSize, rank⟩))) := do
  let localRaw := (← dist.reduceScatter worldSize fullGrad .sum).raw
  match Tensor.ofRaw localRaw with
  | .ok t => pure t
  | .error message => throw (IO.userError
      s!"ShardedTensor.scatterGrad: reduce-scatter result does not match the shard spec: {message}")

/-- Update the local shard and mark cache as stale.

    This is called after the optimizer updates the local parameters.
-/
def ShardedTensor.updateShard {σ : StaticSpec} {rank worldSize : UInt64} {sd : ShardDim}
    (sharded : ShardedTensor σ rank worldSize sd)
    (newShard : Tensor (σ.withShape (shardedShape σ.shape ⟨sd, worldSize, rank⟩)))
    : ShardedTensor σ rank worldSize sd :=
  { sharded with
    shard := newShard
    cacheIsStale := true
  }

/-! ## Sharded Parameter Group -/

/-- A parameter that may or may not be sharded.

    Some parameters (like embeddings) are sharded for memory efficiency.
    Others (like small scalars) are replicated.
-/
inductive MaybeSharded (σ : StaticSpec) (rank worldSize : UInt64) where
  | sharded : ShardedTensor σ rank worldSize → MaybeSharded σ rank worldSize
  | replicated : Tensor σ → MaybeSharded σ rank worldSize
  deriving Repr

/-- Get the tensor for forward pass (gathers if sharded) -/
def MaybeSharded.get {σ : StaticSpec} {rank worldSize : UInt64}
    (p : MaybeSharded σ rank worldSize) : IO (Tensor σ × MaybeSharded σ rank worldSize) := do
  match p with
  | .replicated t => return (t, p)
  | .sharded st =>
    let (full, newSt) ← st.gather
    return (full, .sharded newSt)

/-- Update parameter with gradient -/
def MaybeSharded.updateWithGrad {σ : StaticSpec} {rank worldSize : UInt64}
    (p : MaybeSharded σ rank worldSize)
    (fullGrad : Tensor σ)
    (updateFn : ∀ {σLocal : StaticSpec}, Tensor σLocal → Tensor σLocal → IO (Tensor σLocal))
    : IO (MaybeSharded σ rank worldSize) := do
  match p with
  | .replicated t =>
    -- Replicated: all-reduce gradient, update locally
    let fullGrad ← dist.allReduce fullGrad .avg
    let newT ← updateFn t fullGrad
    return .replicated newT
  | .sharded st =>
    -- Sharded: reduce-scatter gradient, update local shard
    let localGrad ← st.scatterGrad fullGrad
    let newShard ← updateFn st.shard localGrad
    return .sharded (st.updateShard newShard)

/-! ## Sharded Model Parameters -/

/-- Sharded embedding table for distributed training.

    Embeddings are sharded along the vocabulary dimension.
    Each rank owns vocab_size/world_size tokens.
-/
structure ShardedEmbedding (vocabSize dim : UInt64) (rank worldSize : UInt64)
    (dtype : DType := .Float32) (device : Device := .CPU) where
  /-- Local portion of embedding table -/
  weight : ShardedTensor ⟨#[vocabSize, dim], dtype, device⟩ rank worldSize .first
  /-- Learning rate multiplier -/
  lrMul : Float := 75.0
  /-- Weight decay multiplier -/
  wdMul : Float := 1.0
  deriving Repr

/-- Forward pass for sharded embedding.

    1. Gather full embedding
    2. Apply embedding lookup
-/
def ShardedEmbedding.forward {vocabSize dim rank worldSize batch seq : UInt64}
    {d : DType} {dev : Device}
    (emb : ShardedEmbedding vocabSize dim rank worldSize d dev)
    (tokens : Tensor ⟨#[batch, seq], .Int64, dev⟩)
    : IO (Tensor ⟨#[batch, seq, dim], d, dev⟩ × ShardedEmbedding vocabSize dim rank worldSize d dev) := do
  let (fullWeight, newSharded) ← emb.weight.gather
  let output := Tensor.embedding tokens fullWeight
  return (output, { emb with weight := newSharded })

/-! ## Sharded Optimizer State -/

/-- Optimizer state that mirrors parameter sharding.

    If a parameter is sharded, its optimizer state (momentum, variance) is too.
-/
structure ShardedAdamState (σ : StaticSpec) (rank worldSize : UInt64) where
  /-- First moment (sharded if param is sharded) -/
  expAvg : MaybeSharded σ rank worldSize
  /-- Second moment (sharded if param is sharded) -/
  expAvgSq : MaybeSharded σ rank worldSize
  /-- Step count -/
  step : Nat
  deriving Repr

/-- Initialize sharded Adam state with zeros -/
def ShardedAdamState.init {σ : StaticSpec} {rank worldSize : UInt64}
    [DType.ConstantSupported σ.dtype σ.device]
    (param : MaybeSharded σ rank worldSize) : ShardedAdamState σ rank worldSize :=
  match param with
  | .replicated _ => {
      expAvg := .replicated (Tensor.zerosFor σ)
      expAvgSq := .replicated (Tensor.zerosFor σ)
      step := 0
    }
  | .sharded _ =>
      let shardSpec := σ.withShape (shardedShape σ.shape ⟨.first, worldSize, rank⟩)
      {
        expAvg := .sharded ⟨Tensor.zerosFor shardSpec, none, true⟩
        expAvgSq := .sharded ⟨Tensor.zerosFor shardSpec, none, true⟩
        step := 0
      }

/-! ## Helper Functions -/

/-- Check if distributed training is active -/
def isShardingActive : IO Bool := dist.isInitialized

/-- Get sharding info for current process -/
def getShardingInfo : IO (UInt64 × UInt64) := do
  let isActive ← isShardingActive
  if isActive then
    dist.getRankAndWorldSize
  else
    pure (0, 1)

/-- Create shard spec from current distributed state -/
def currentShardSpec (shardDim : ShardDim := .first) : IO ShardSpec := do
  let (rank, worldSize) ← getShardingInfo
  return ⟨shardDim, worldSize, rank⟩

end torch.Sharding
