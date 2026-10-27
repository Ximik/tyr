import Tyr.Basic

/-! Native process-group and collective bindings. The typed API is in
`Tyr.Distributed`; `Native` holds the raw-handle entry points it wraps. -/

namespace torch.dist

/-- Reduce operation for collective communications -/
inductive ReduceOp where
  | sum     : ReduceOp
  | avg     : ReduceOp
  | product : ReduceOp
  | min     : ReduceOp
  | max     : ReduceOp
  deriving Repr, BEq, Inhabited

/-- Initialize the distributed process group.

    Args:
    - backend: "nccl" for GPU, "gloo" for CPU
    - masterAddr: Address of rank 0 process (e.g., "localhost")
    - masterPort: Port for rendezvous
    - rank: This process's rank (0 to worldSize-1)
    - worldSize: Total number of processes

    Invalid bounds and unavailable/unknown backends fail before rendezvous.
    Initialization publishes process-group state only after construction succeeds.
-/
@[extern "lean_torch_dist_init_process_group"]
opaque initProcessGroup (backend masterAddr : @& String) (masterPort rank worldSize : UInt64) : IO Unit

/-- Set the current CUDA device for this process (typically LOCAL_RANK).
    Indices outside the native DeviceIndex range fail on every build. -/
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

/-- Await pending work and release all process-group resources. Cleanup attempts
    every pending work item and resets state even when waiting fails, then reports
    the first failure. Calling this while uninitialized is safe. -/
@[extern "lean_torch_dist_destroy_process_group"]
opaque destroyProcessGroup : IO Unit

/-- Synchronize all processes (barrier) -/
@[extern "lean_torch_dist_barrier"]
opaque barrier : IO Unit

namespace Native

/-- Await an async operation, reporting failure or abortion and consuming its
    handle. A second wait on the same handle is an error. -/
@[extern "lean_torch_dist_wait"]
opaque waitImpl (workId : UInt64) : IO Unit

/-- All-reduce: sum/avg/etc tensors across all processes (in-place).

    Args:
    - tensor: Input/output tensor (modified in-place)
    - op: Reduce operation (sum, avg, etc.)
    - asyncOp: If true, returns work handle for async wait

    Returns work ID if async, 0 otherwise.
-/
@[extern "lean_torch_dist_all_reduce"]
opaque allReduceImpl (tensor : @& RawTensor) (op : ReduceOp) (asyncOp : UInt8) : IO UInt64

/-- Broadcast tensor from source rank to all processes (in-place).

    Args:
    - tensor: Input/output tensor
    - srcRank: Source rank to broadcast from
-/
@[extern "lean_torch_dist_broadcast"]
opaque broadcastImpl (tensor : @& RawTensor) (srcRank : UInt64) : IO Unit

/-- Reduce-scatter: reduce tensors and scatter results to all processes.

    Each process gets a different portion of the reduced tensor.
    Input is split into worldSize chunks, reduced, and each rank gets one chunk.
    The output buffer is written in place.

    Returns work ID if async, 0 otherwise.
-/
@[extern "lean_torch_dist_reduce_scatter_tensor"]
opaque reduceScatterImpl (output : @& RawTensor) (input : @& RawTensor)
    (op : ReduceOp) (asyncOp : UInt8) : IO UInt64

/-- Reduce-scatter from an explicit list of per-rank inputs.

    This mirrors `torch.distributed.reduce_scatter` where each rank contributes
    a list of tensors (one per rank). Every contribution and the output share
    one spec; the list length must equal world size.
-/
@[extern "lean_torch_dist_reduce_scatter_list"]
opaque reduceScatterListImpl (output : @& RawTensor) (inputs : @& Array RawTensor)
    (op : ReduceOp) : IO Unit

/-- All-gather: gather tensors from all processes to all processes.
    The output buffer (leading axis = worldSize × input's) is written in place.

    Returns work ID if async, 0 otherwise.
-/
@[extern "lean_torch_dist_all_gather_into_tensor"]
opaque allGatherImpl (output : @& RawTensor) (input : @& RawTensor)
    (asyncOp : UInt8) : IO UInt64

/-- All-gather into an explicit list of output tensors.

    This mirrors `torch.distributed.all_gather` where each rank provides one
    input tensor and receives the gathered outputs in a list; every output
    shares the input's spec.
-/
@[extern "lean_torch_dist_all_gather_list"]
opaque allGatherListImpl (outputs : @& Array RawTensor) (input : @& RawTensor) : IO Unit

/-- Poll for successful completion without consuming the work handle.
    Pending work returns false; failed work or an unknown handle returns an error.
    Call `wait` even after a true result to synchronize and release the work. -/
@[extern "lean_torch_dist_is_completed"]
opaque isCompletedImpl (workId : UInt64) : IO Bool

/-- Symmetric matrix multiplication: C = A @ A.T (raw handle boundary).

    For batched input [batch, rows, cols], output is [batch, rows, rows]
    For 2D input [rows, cols], output is [rows, rows]
-/
@[extern "lean_torch_xxt"]
opaque xxtImpl (A : @& RawTensor) : IO RawTensor

/-- Fused operation: C = beta * A + alpha * (A @ A.T)

    Used in Newton-Schulz iteration. Input must be square (last two dims equal).
-/
@[extern "lean_torch_ba_plus_cAA"]
opaque baPlusCaaImpl (A : @& RawTensor) (alpha beta : Float) : IO RawTensor

/-- Single Newton-Schulz iteration step (raw handle boundary).

    Computes: Y = a*X + b*X@X.T@X + c*X@X.T@X@X.T@X
-/
@[extern "lean_torch_newton_schulz_step"]
opaque newtonSchulzStepImpl (X : @& RawTensor) (a b c : Float) : IO RawTensor

/-- Polar Express orthogonalization via Newton-Schulz iterations (raw boundary).

    Approximates the matrix sign function for orthogonalization.
    Typically uses 5 iterations with precomputed stable coefficients.
-/
@[extern "lean_torch_polar_express"]
opaque polarExpressImpl (G : @& RawTensor) (numIters : UInt64) : IO RawTensor

/-- Muon-style orthogonalized gradient (raw boundary).

    For [out, in] gradient:
    - If out > in: orthogonalize along rows (transpose, orthogonalize, transpose back)
    - If out <= in: orthogonalize directly
-/
@[extern "lean_torch_muon_orthogonalize"]
opaque muonOrthogonalizeImpl (G : @& RawTensor) (numIters : UInt64) : IO RawTensor

/-- Cautious weight decay update (raw boundary).

    Only applies weight decay when update and parameter have the same sign:
    mask = sign(update) == sign(param)
    newParam = param - lr * (update + wd * mask * param)
-/
@[extern "lean_torch_cautious_update"]
opaque cautiousUpdateImpl (param update : @& RawTensor) (lr wd : Float) : IO RawTensor

end Native

end torch.dist
