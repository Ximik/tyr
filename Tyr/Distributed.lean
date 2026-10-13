import Tyr.TensorStruct

/-!
# Tyr.Distributed

`Tyr.Distributed` provides Lean bindings and helpers for multi-process training.
It wraps core PyTorch distributed collectives and process-group lifecycle operations,
plus utilities for tensor-tree synchronization patterns.

## Major Components

- Process-group lifecycle (`initProcessGroup`, rank/world queries, barriers, teardown).
- Collective communication (`allReduce`, broadcast, reduce-scatter, all-gather, gather/scatter).
- Async operation handles (`WorkHandle`) and wait semantics.
- TensorStruct-aware helpers for synchronizing nested parameter/gradient structures.

## Scope

This module exposes communication primitives and synchronization building blocks.
Higher-level distributed training logic (optimizer/pipeline policies) should compose this layer.
-/

namespace torch.dist

open torch

/-- Reduce operation for collective communications -/
inductive ReduceOp where
  | sum     : ReduceOp
  | avg     : ReduceOp
  | product : ReduceOp
  | min     : ReduceOp
  | max     : ReduceOp
  deriving Repr, BEq, Inhabited

/-- Convert ReduceOp to UInt8 for FFI -/
def ReduceOp.toUInt8 : ReduceOp → UInt8
  | sum => 0
  | avg => 1
  | product => 2
  | min => 3
  | max => 4

/-- Handle for async distributed operations -/
structure WorkHandle where
  id : UInt64
  deriving Repr, BEq, Inhabited

/-! ## Process Group Management -/

/-- Initialize the distributed process group.

    Args:
    - backend: "nccl" for GPU, "gloo" for CPU
    - masterAddr: Address of rank 0 process (e.g., "localhost")
    - masterPort: Port for rendezvous
    - rank: This process's rank (0 to worldSize-1)
    - worldSize: Total number of processes
-/
@[extern "lean_torch_dist_init_process_group"]
opaque initProcessGroup (backend masterAddr : @& String) (masterPort rank worldSize : UInt64) : IO Unit

/-- Set the current CUDA device for this process (typically LOCAL_RANK). -/
@[extern "lean_torch_dist_set_cuda_device"]
opaque setCudaDevice (device : UInt64) : IO Unit

/-- Get the rank of this process -/
@[extern "lean_torch_dist_get_rank"]
opaque getRank : IO UInt64

/-- Get the world size (total number of processes) -/
@[extern "lean_torch_dist_get_world_size"]
opaque getWorldSize : IO UInt64

/-- Check if distributed is initialized -/
@[extern "lean_torch_dist_is_initialized"]
opaque isInitialized : IO Bool

/-- Destroy the process group and clean up resources -/
@[extern "lean_torch_dist_destroy_process_group"]
opaque destroyProcessGroup : IO Unit

/-- Synchronize all processes (barrier) -/
@[extern "lean_torch_dist_barrier"]
opaque barrier : IO Unit

/-! ## Collective Operations -/

/-- All-reduce: sum/avg/etc tensors across all processes (in-place).

    Args:
    - tensor: Input/output tensor (modified in-place)
    - op: Reduce operation (sum, avg, etc.)
    - asyncOp: If true, returns work handle for async wait

    Returns work ID if async, 0 otherwise.
-/
@[extern "lean_torch_dist_all_reduce"]
private opaque allReduceImpl (tensor : @& RawTensor) (op : UInt8) (asyncOp : UInt8) : IO UInt64

/-- All-reduce (synchronous). The buffer is reduced in place; the static spec
    is untouched by the collective. -/
def allReduce {σ : StaticSpec} (tensor : Tensor σ) (op : ReduceOp := .sum) : IO Unit := do
  let _ ← allReduceImpl tensor.raw op.toUInt8 0
  return ()

/-- All-reduce (asynchronous) -/
def allReduceAsync {σ : StaticSpec} (tensor : Tensor σ) (op : ReduceOp := .sum) : IO WorkHandle := do
  let id ← allReduceImpl tensor.raw op.toUInt8 1
  return ⟨id⟩

/-- Broadcast tensor from source rank to all processes (in-place).

    Args:
    - tensor: Input/output tensor
    - srcRank: Source rank to broadcast from
-/
@[extern "lean_torch_dist_broadcast"]
private opaque broadcastImpl (tensor : @& RawTensor) (srcRank : UInt64) : IO Unit

/-- Broadcast a tensor from `srcRank` (in-place buffer update). -/
def broadcast {σ : StaticSpec} (tensor : Tensor σ) (srcRank : UInt64) : IO Unit :=
  broadcastImpl tensor.raw srcRank

/-- Reduce-scatter: reduce tensors and scatter results to all processes.

    Each process gets a different portion of the reduced tensor.
    Input is split into worldSize chunks, reduced, and each rank gets one chunk.
    The output buffer is written in place.

    Returns work ID if async, 0 otherwise.
-/
@[extern "lean_torch_dist_reduce_scatter_tensor"]
private opaque reduceScatterImpl (output : @& RawTensor) (input : @& RawTensor)
    (op : UInt8) (asyncOp : UInt8) : IO UInt64

/-- Reduce-scatter (synchronous), return-allocated: the input's leading axis is
    chunked across `worldSize` ranks, so the result spec is computed from the
    input spec — no independent output shape can be claimed. -/
def reduceScatter {σ : StaticSpec} (worldSize : UInt64) (input : Tensor σ)
    (op : ReduceOp := .sum)
    : IO (Tensor (σ.withShape (σ.shape.set! 0 (σ.shape.getD 0 0 / worldSize)))) := do
  let output := Tensor.zerosFor (σ.withShape (σ.shape.set! 0 (σ.shape.getD 0 0 / worldSize)))
  let _ ← reduceScatterImpl output.raw input.raw op.toUInt8 0
  return output

/-- Reduce-scatter from an explicit list of per-rank inputs.

    This mirrors `torch.distributed.reduce_scatter` where each rank contributes
    a list of tensors (one per rank). Every contribution and the output share
    one spec; the list length must equal world size.
-/
@[extern "lean_torch_dist_reduce_scatter_list"]
private opaque reduceScatterListImpl (output : @& RawTensor) (inputs : @& Array RawTensor)
    (op : UInt8) : IO Unit

/-- Reduce-scatter from an explicit list of per-rank inputs (one per rank),
    all sharing the output's spec. -/
def reduceScatterList {σ : StaticSpec} (output : Tensor σ) (inputs : Array (Tensor σ))
    (op : ReduceOp := .sum) : IO Unit :=
  reduceScatterListImpl output.raw (inputs.map Tensor.toTensor) op.toUInt8

/-- All-gather: gather tensors from all processes to all processes.
    The output buffer (leading axis = worldSize × input's) is written in place.

    Returns work ID if async, 0 otherwise.
-/
@[extern "lean_torch_dist_all_gather_into_tensor"]
private opaque allGatherImpl (output : @& RawTensor) (input : @& RawTensor)
    (asyncOp : UInt8) : IO UInt64

/-- All-gather (synchronous), return-allocated: the result concatenates the
    `worldSize` per-rank contributions along the leading axis, so its spec is
    computed from the input spec — no independent output shape can be claimed. -/
def allGather {σ : StaticSpec} (worldSize : UInt64) (input : Tensor σ)
    : IO (Tensor (σ.withShape (σ.shape.set! 0 (σ.shape.getD 0 0 * worldSize)))) := do
  let output := Tensor.zerosFor (σ.withShape (σ.shape.set! 0 (σ.shape.getD 0 0 * worldSize)))
  let _ ← allGatherImpl output.raw input.raw 0
  return output

/-- All-gather into an explicit list of output tensors.

    This mirrors `torch.distributed.all_gather` where each rank provides one
    input tensor and receives the gathered outputs in a list; every output
    shares the input's spec.
-/
@[extern "lean_torch_dist_all_gather_list"]
private opaque allGatherListImpl (outputs : @& Array RawTensor) (input : @& RawTensor) : IO Unit

/-- All-gather into an explicit list of output tensor buffers sharing the
    input's spec. -/
def allGatherList {σ : StaticSpec} (outputs : Array (Tensor σ)) (input : Tensor σ) : IO Unit :=
  allGatherListImpl (outputs.map Tensor.toTensor) input.raw

/-! ## Async Operation Handling -/

/-- Wait for an async operation to complete -/
@[extern "lean_torch_dist_wait"]
opaque wait (handle : WorkHandle) : IO Unit

/-- Check if an async operation is complete (non-blocking) -/
@[extern "lean_torch_dist_is_completed"]
private opaque isCompletedImpl (workId : UInt64) : IO Bool

def WorkHandle.isCompleted (handle : WorkHandle) : IO Bool :=
  isCompletedImpl handle.id

/-- Wait for multiple async operations -/
def waitAll (handles : Array WorkHandle) : IO Unit := do
  for h in handles do
    wait h

/-! ## Polar Express Operations -/

/-- Symmetric matrix multiplication: C = A @ A.T (raw handle boundary).

    For batched input [batch, rows, cols], output is [batch, rows, rows]
    For 2D input [rows, cols], output is [rows, rows]
-/
@[extern "lean_torch_xxt"]
private opaque xxtImpl (A : @& RawTensor) : IO RawTensor

/-- `A @ Aᵀ` for an indexed matrix: the result shape is computed from the
    input, not claimed independently. -/
def xxt {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (A : Tensor ⟨#[m, n], d, dev⟩) : IO (Tensor ⟨#[m, m], d, dev⟩) := do
  pure (.assumeSpec (← xxtImpl A.raw))

/-- Fused operation: C = beta * A + alpha * (A @ A.T)

    Used in Newton-Schulz iteration. Input must be square (last two dims equal).
-/
@[extern "lean_torch_ba_plus_cAA"]
private opaque baPlusCaaImpl (A : @& RawTensor) (alpha beta : Float) : IO RawTensor

/-- `beta * A + alpha * (A @ Aᵀ)`, spec-preserving. -/
def baPlusCaa {σ : StaticSpec} (A : Tensor σ) (alpha beta : Float) : IO (Tensor σ) := do
  pure (.assumeSpec (← baPlusCaaImpl A.raw alpha beta))

/-- Single Newton-Schulz iteration step (raw handle boundary).

    Computes: Y = a*X + b*X@X.T@X + c*X@X.T@X@X.T@X
-/
@[extern "lean_torch_newton_schulz_step"]
private opaque newtonSchulzStepImpl (X : @& RawTensor) (a b c : Float) : IO RawTensor

/-- Single Newton-Schulz iteration step, spec-preserving. -/
def newtonSchulzStep {σ : StaticSpec} (X : Tensor σ) (a b c : Float) : IO (Tensor σ) := do
  pure (.assumeSpec (← newtonSchulzStepImpl X.raw a b c))

/-- Polar Express orthogonalization via Newton-Schulz iterations (raw boundary).

    Approximates the matrix sign function for orthogonalization.
    Typically uses 5 iterations with precomputed stable coefficients.
-/
@[extern "lean_torch_polar_express"]
private opaque polarExpressImpl (G : @& RawTensor) (numIters : UInt64) : IO RawTensor

/-- Polar Express orthogonalization via Newton-Schulz iterations,
    spec-preserving. -/
def polarExpress {σ : StaticSpec} (G : Tensor σ) (numIters : UInt64 := 5) : IO (Tensor σ) := do
  pure (.assumeSpec (← polarExpressImpl G.raw numIters))

/-- Muon-style orthogonalized gradient (raw boundary).

    For [out, in] gradient:
    - If out > in: orthogonalize along rows (transpose, orthogonalize, transpose back)
    - If out <= in: orthogonalize directly
-/
@[extern "lean_torch_muon_orthogonalize"]
private opaque muonOrthogonalizeImpl (G : @& RawTensor) (numIters : UInt64) : IO RawTensor

/-- Muon-style orthogonalized gradient, spec-preserving (handles tall and wide
    matrices internally via transposition). -/
def muonOrthogonalize {σ : StaticSpec} (G : Tensor σ) (numIters : UInt64 := 5) : IO (Tensor σ) := do
  pure (.assumeSpec (← muonOrthogonalizeImpl G.raw numIters))

/-- Cautious weight decay update (raw boundary).

    Only applies weight decay when update and parameter have the same sign:
    mask = sign(update) == sign(param)
    newParam = param - lr * (update + wd * mask * param)
-/
@[extern "lean_torch_cautious_update"]
private opaque cautiousUpdateImpl (param update : @& RawTensor) (lr wd : Float) : IO RawTensor

/-- Cautious weight decay update, spec-preserving. -/
def cautiousUpdate {σ : StaticSpec} (param update : Tensor σ) (lr wd : Float) : IO (Tensor σ) := do
  pure (.assumeSpec (← cautiousUpdateImpl param.raw update.raw lr wd))

/-! ## Helper Functions -/

/-- Run a distributed computation with proper initialization and cleanup -/
def withDistributed (backend masterAddr : String) (masterPort rank worldSize : UInt64)
    (action : IO α) : IO α := do
  initProcessGroup backend masterAddr masterPort rank worldSize
  let result ← action
  destroyProcessGroup
  return result

/-- Get rank and world size as a pair -/
def getRankAndWorldSize : IO (UInt64 × UInt64) := do
  let rank ← getRank
  let worldSize ← getWorldSize
  return (rank, worldSize)

/-- Check if this is the master process (rank 0) -/
def isMaster : IO Bool := do
  let rank ← getRank
  return rank == 0

/-- Broadcast model parameters from rank 0 to all processes -/
def broadcastParams {α : Type} [TensorStruct α] (params : α) : IO α := do
  TensorStruct.mapM (fun t => do broadcast t 0; pure t) params

/-- All-reduce gradients across all processes.
    Uses the gradient field of tensors that have requires_grad=true.

    Args:
    - grads: Gradient structure (typically from TensorStruct.grads)
    - op: Reduce operation (default: average across ranks)

    After this call, all ranks have the same averaged gradients.
-/
def allReduceGrads {α : Type} [TensorStruct α] (grads : α) (op : ReduceOp := .avg) : IO α := do
  TensorStruct.mapM (fun t => do
    allReduce t op
    pure t
  ) grads

/-- All-reduce gradients asynchronously, returning work handles.
    Use `waitAll` to wait for completion before using gradients.
-/
def allReduceGradsAsync {α : Type} [TensorStruct α] (grads : α) (op : ReduceOp := .avg)
    : IO (α × Array WorkHandle) := do
  -- Use a StateT pattern to collect handles
  let handlesRef ← IO.mkRef #[]
  let result ← TensorStruct.mapM (fun t => do
    let h ← allReduceAsync t op
    handlesRef.modify (·.push h)
    pure t
  ) grads
  let handles ← handlesRef.get
  return (result, handles)

/-! ## Distributed Sampler -/

/-- Configuration for distributed data sampling.
    Ensures each rank gets a different portion of the dataset.
-/
structure DistributedSamplerConfig where
  /-- Total size of the dataset -/
  datasetSize : Nat
  /-- Rank of this process -/
  rank : UInt64
  /-- Total number of processes -/
  worldSize : UInt64
  /-- Random seed for shuffling -/
  seed : UInt64 := 42
  /-- Whether to shuffle indices -/
  shuffle : Bool := true
  /-- Whether to drop the last incomplete batch -/
  dropLast : Bool := false
  deriving Repr, Inhabited

/-- Distributed sampler for sharding data across ranks.

    Usage:
    ```lean
    let sampler := DistributedSampler.create cfg
    for idx in sampler.indices do
      let sample := dataset[idx]!
      -- process sample
    ```
-/
structure DistributedSampler where
  /-- Configuration -/
  config : DistributedSamplerConfig
  /-- Indices for this rank -/
  indices : Array Nat
  /-- Current epoch (for re-shuffling) -/
  epoch : Nat := 0
  deriving Repr

/-- LCG random number generator for deterministic shuffling -/
private def lcgNext (state : UInt64) : UInt64 :=
  state * 6364136223846793005 + 1442695040888963407

/-- Fisher-Yates shuffle -/
private def shuffleArray (arr : Array Nat) (seed : UInt64) : Array Nat := Id.run do
  if arr.size <= 1 then return arr
  let mut result := arr
  let mut state := seed
  for i in [:(arr.size - 1)] do
    state := lcgNext state
    let range := arr.size - i
    let j := i + (state % range.toUInt64).toNat
    let tmp := result[i]!
    result := result.set! i result[j]!
    result := result.set! j tmp
  return result

/-- Create a distributed sampler that shards data across ranks.

    Each rank gets datasetSize / worldSize indices, starting from
    rank * (datasetSize / worldSize).

    If shuffle is true, indices are shuffled deterministically
    using the seed (same shuffle on all ranks, different slices).
-/
def DistributedSampler.create (cfg : DistributedSamplerConfig) : DistributedSampler := Id.run do
  let totalSize := cfg.datasetSize
  let worldSize := cfg.worldSize.toNat
  let rank := cfg.rank.toNat

  -- Compute shard boundaries
  let samplesPerRank := totalSize / worldSize
  let remainder := totalSize % worldSize

  -- First 'remainder' ranks get one extra sample
  let startIdx := if rank < remainder then
      rank * (samplesPerRank + 1)
    else
      remainder * (samplesPerRank + 1) + (rank - remainder) * samplesPerRank

  let numSamples := if rank < remainder then samplesPerRank + 1 else samplesPerRank

  -- Generate all indices
  let allIndices := Array.range totalSize

  -- Shuffle if requested (same shuffle for all ranks)
  let allIndices := if cfg.shuffle then
      shuffleArray allIndices cfg.seed
    else
      allIndices

  -- Extract this rank's portion
  let indices := allIndices.extract startIdx (startIdx + numSamples)

  { config := cfg, indices := indices, epoch := 0 }

/-- Create a new sampler for the next epoch (re-shuffles if shuffle=true) -/
def DistributedSampler.nextEpoch (sampler : DistributedSampler) : DistributedSampler :=
  let newEpoch := sampler.epoch + 1
  let newSeed := sampler.config.seed + newEpoch.toUInt64
  let newConfig := { sampler.config with seed := newSeed }
  let newSampler := DistributedSampler.create newConfig
  { newSampler with epoch := newEpoch }

/-- Get the number of samples for this rank -/
def DistributedSampler.size (sampler : DistributedSampler) : Nat :=
  sampler.indices.size

/-- Get index at position in this rank's shard -/
def DistributedSampler.getIndex (sampler : DistributedSampler) (pos : Nat) : Option Nat :=
  sampler.indices[pos]?

end torch.dist
