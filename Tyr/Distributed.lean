import Tyr.TensorStruct
import Tyr.Inference.OwnedKV
import Tyr.ConstantSupport
import Tyr.RawTensor.Distributed

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
open torch.dist.Native

/-- Handle for async distributed operations -/
structure WorkHandle where
  id : UInt64
  deriving Repr, BEq, Inhabited

def wait (handle : WorkHandle) : IO Unit :=
  waitImpl handle.id

/-! ## Process Group Management -/

/-! ## Collective Operations -/

/-- All-reduce into an independent result, preserving the input tensor and its
    `requiresGrad` flag. A trainable result is a new leaf. -/
def allReduce {σ : StaticSpec} (tensor : Tensor σ) (op : ReduceOp := .sum) : IO (Tensor σ) := do
  let requiresGrad := tensor.requiresGrad
  let result ← data.cloneInferenceIO tensor
  let _ ← allReduceImpl result.raw op 0
  result.setRequiresGrad requiresGrad

/-- Launch all-reduce on an independent buffer. Execute the returned action once
    to await completion and obtain the result; no pending tensor is exposed. -/
def allReduceAsync {σ : StaticSpec} (tensor : Tensor σ) (op : ReduceOp := .sum) :
    IO (IO (Tensor σ)) := do
  let requiresGrad := tensor.requiresGrad
  let result ← data.cloneInferenceIO tensor
  let id ← allReduceImpl result.raw op 1
  return do
    wait ⟨id⟩
    result.setRequiresGrad requiresGrad

/-- Return the tensor broadcast from `srcRank`, preserving the input value and
    its `requiresGrad` flag. A trainable result is a new leaf. -/
def broadcast {σ : StaticSpec} (tensor : Tensor σ) (srcRank : Nat) : IO (Tensor σ) := do
  let requiresGrad := tensor.requiresGrad
  let result ← data.cloneInferenceIO tensor
  broadcastImpl result.raw srcRank
  result.setRequiresGrad requiresGrad

/-- Allocate private collective storage with the input's exact dtype and device. -/
private def collectiveOutput {σ : StaticSpec} (shape : Shape) (_input : Tensor σ) :
    IO (Tensor (σ.withShape shape)) := do
  IO.ofExcept (Shape.validateAllocation shape)
  let support ← σ.dtype.acquireConstantSupport σ.device
  let _ := support.down
  let template := Tensor.zerosFor { shape, dtype := σ.dtype, device := σ.device }
  let output ← data.cloneInferenceIO template
  match Tensor.ofRaw output.raw with
  | .ok result => pure result
  | .error message => throw (IO.userError s!"Collective output allocation: {message}")

/-- Reduce-scatter (synchronous), return-allocated: the input's leading axis is
    chunked across `worldSize` ranks, so the result spec is computed from the
    input spec — no independent output shape can be claimed. -/
def reduceScatter {σ : StaticSpec} (worldSize : Nat) (input : Tensor σ)
    (op : ReduceOp := .sum)
    : IO (Tensor (σ.withShape (σ.shape.set! 0 (σ.shape.getD 0 0 / worldSize)))) := do
  let output ← collectiveOutput (σ.shape.set! 0 (σ.shape.getD 0 0 / worldSize)) input
  let _ ← reduceScatterImpl output.raw input.raw op 0
  return output

/-- Reduce-scatter from per-rank inputs into an independent copy of `output`.
    All inputs and the output template retain their original values. -/
def reduceScatterList {σ : StaticSpec} (output : Tensor σ) (inputs : Array (Tensor σ))
    (op : ReduceOp := .sum) : IO (Tensor σ) := do
  let requiresGrad := output.requiresGrad
  let result ← data.cloneInferenceIO output
  reduceScatterListImpl result.raw (inputs.map Tensor.toTensor) op
  result.setRequiresGrad requiresGrad

/-- All-gather (synchronous), return-allocated: the result concatenates the
    `worldSize` per-rank contributions along the leading axis, so its spec is
    computed from the input spec — no independent output shape can be claimed. -/
def allGather {σ : StaticSpec} (worldSize : Nat) (input : Tensor σ)
    : IO (Tensor (σ.withShape (σ.shape.set! 0 (σ.shape.getD 0 0 * worldSize)))) := do
  let output ← collectiveOutput (σ.shape.set! 0 (σ.shape.getD 0 0 * worldSize)) input
  let _ ← allGatherImpl output.raw input.raw 0
  return output

/-- All-gather into independent copies of the output templates. Every returned
    entry has its own storage, including when templates share an input value. -/
def allGatherList {σ : StaticSpec} (outputs : Array (Tensor σ)) (input : Tensor σ) :
    IO (Array (Tensor σ)) := do
  let requiresGrad := outputs.map Tensor.requiresGrad
  let results ← outputs.mapM data.cloneInferenceIO
  allGatherListImpl (results.map Tensor.toTensor) input.raw
  results.mapIdxM fun i result => result.setRequiresGrad (requiresGrad[i]?.getD false)

/-! ## Async Operation Handling -/

/-- Observe completion without releasing the handle. `wait` remains required
    after successful polling; an unknown or already consumed handle is an error. -/
def WorkHandle.isCompleted (handle : WorkHandle) : IO Bool :=
  isCompletedImpl handle.id

/-- Wait for every async operation, then rethrow the first error if any wait
    failed. Later work handles are drained even when an earlier operation fails. -/
def waitAll (handles : Array WorkHandle) : IO Unit := do
  let mut firstError : Option IO.Error := none
  for h in handles do
    try
      wait h
    catch error =>
      if firstError.isNone then firstError := some error
  match firstError with
  | some error => throw error
  | none => pure ()

/-! ## Polar Express Operations -/

/-- Validate native results before exposing the indexed API. In particular,
floating scalar arithmetic can promote an integer input's dtype. -/
private def checkedResult {σ : StaticSpec} (operation : String) (dtype : DType)
    (device : Device) (raw : RawTensor) : IO (Tensor σ) :=
  let checked : Except String (Tensor σ) := do
    TensorSpec.checkDType dtype raw.dtype
    device.check raw.device
    Tensor.ofRaw raw
  match checked with
  | .ok tensor => pure tensor
  | .error message => throw (IO.userError s!"{operation}: {message}")

/-- `A @ Aᵀ` for an indexed matrix: the result shape is computed from the
    input, not claimed independently. -/
def xxt {m n : Nat} {d : DType} {dev : Device}
    (A : Tensor ⟨#[m, n], d, dev⟩) : IO (Tensor ⟨#[m, m], d, dev⟩) := do
  checkedResult "xxt" A.raw.dtype A.device (← xxtImpl A.raw)

/-- `beta * A + alpha * (A @ Aᵀ)`, spec-preserving. -/
def baPlusCaa {σ : StaticSpec} (A : Tensor σ) (alpha beta : Float) : IO (Tensor σ) := do
  checkedResult "baPlusCaa" A.raw.dtype A.device (← baPlusCaaImpl A.raw alpha beta)

/-- Single Newton-Schulz iteration step, spec-preserving. -/
def newtonSchulzStep {σ : StaticSpec} (X : Tensor σ) (a b c : Float) : IO (Tensor σ) := do
  checkedResult "newtonSchulzStep" X.raw.dtype X.device (← newtonSchulzStepImpl X.raw a b c)

/-- Polar Express orthogonalization via Newton-Schulz iterations,
    spec-preserving. -/
def polarExpress {σ : StaticSpec} (G : Tensor σ) (numIters : Nat := 5) : IO (Tensor σ) := do
  checkedResult "polarExpress" G.raw.dtype G.device (← polarExpressImpl G.raw numIters)

/-- Muon orthogonalization runs and returns BF16, including for Float32 inputs.
    Shape and placement are preserved; the output dtype is explicit. -/
def muonOrthogonalize {σ : StaticSpec} (G : Tensor σ) (numIters : Nat := 5) :
    IO (Tensor (σ.withDType .BFloat16)) := do
  checkedResult "muonOrthogonalize" .BFloat16 G.device (← muonOrthogonalizeImpl G.raw numIters)

/-- Cautious weight decay update, spec-preserving. -/
def cautiousUpdate {σ : StaticSpec} (param update : Tensor σ) (lr wd : Float) : IO (Tensor σ) := do
  checkedResult "cautiousUpdate" param.raw.dtype param.device (← cautiousUpdateImpl param.raw update.raw lr wd)

/-! ## Helper Functions -/

/-- Run a distributed computation and always clean up after a successful init.
    If both the action and cleanup fail, preserve the action error; after a
    successful action, report any cleanup error instead of returning its result. -/
def withDistributed (backend masterAddr : String) (masterPort rank worldSize : Nat)
    (action : IO α) : IO α := do
  initProcessGroup backend masterAddr masterPort rank worldSize
  let result ← action.toBaseIO
  let cleanup ← destroyProcessGroup.toBaseIO
  match result with
  | .error error => throw error
  | .ok value =>
    match cleanup with
    | .error error => throw error
    | .ok _ => pure value

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
  TensorStruct.mapM (fun t => broadcast t 0) params

/-- All-reduce gradient values across all processes, preserving the input tree.

    Args:
    - grads: Gradient structure (typically from TensorStruct.grads)
    - op: Reduce operation (default: average across ranks)

    After this call, all ranks have the same averaged gradients.
-/
def allReduceGrads {α : Type} [TensorStruct α] (grads : α) (op : ReduceOp := .avg) : IO α := do
  TensorStruct.mapM (fun t => allReduce t op) grads

/-- Launch all gradient reductions, returning an action that waits for every
    operation before exposing the independent result tree. Execute it once.
-/
def allReduceGradsAsync {α : Type} [TensorStruct α] (grads : α) (op : ReduceOp := .avg)
    : IO (IO α) := do
  let handlesRef ← IO.mkRef #[]
  let flagsRef ← IO.mkRef #[]
  try
    let result ← TensorStruct.mapM (fun t => do
      flagsRef.modify (·.push t.requiresGrad)
      let output ← data.cloneInferenceIO t
      let id ← allReduceImpl output.raw op 1
      handlesRef.modify (·.push (WorkHandle.mk id))
      pure output
    ) grads
    let handles ← handlesRef.get
    return do
      waitAll handles
      let flags ← flagsRef.get
      let restore : StateT Nat IO α := TensorStruct.mapM (fun output => do
        let index ← get
        set (index + 1)
        output.setRequiresGrad (flags[index]?.getD false)) result
      return (← restore.run 0).1
  catch error =>
    -- Drain any operations launched before a later leaf failed.
    for handle in (← handlesRef.get) do
      try wait handle catch _ => pure ()
    throw error

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
