import Tyr.Typed.Tensor

/- Tyr/Inference/OwnedKV.lean

  Effectful inference-only tensor updates for KV-cache storage.
  Each update returns the destination value to use next; retained aliases
  preserve their values. Native updates consume the destination ownership
  token and reuse storage only when it is safe. -/

namespace torch.data

/-- Checked native slice update, consuming the destination reference. -/
@[extern "lean_torch_copy_slice_io"]
private opaque copySliceIORaw (dstShape srcShape : Shape) (dst : RawTensor)
    (dim start : UInt64) (src : @& RawTensor) : IO RawTensor

/-- Raw ABI for the effectful inference clone. The shape argument is passed
for ABI compatibility only; the native side never reads it. -/
@[extern "lean_torch_clone_inference_io"]
private opaque cloneInferenceIORaw (shape : Shape) (src : @& RawTensor) : IO RawTensor

/-- Return the destination with the requested slice replaced. Retained aliases
keep their original values. The native operation disables autograd for the copy. -/
def copySliceIO {σ σsrc : StaticSpec} (dst : Tensor σ) (dim start : UInt64)
    (src : @& Tensor σsrc) : IO (Tensor σ) := do
  let dtype := dst.raw.dtype
  let device := dst.raw.device
  let raw ← copySliceIORaw σ.shape σsrc.shape dst.raw dim start src.raw
  match Tensor.checkedUpdate? dtype device raw with
  | .ok result => pure result
  | .error message => throw (IO.userError s!"copySliceIO: {message}")

/-- Allocate an independent inference tensor copy, retaining the full static
spec. The IO effect prevents Lean from merging identical allocations when
distinct mutable buffers are required. -/
def cloneInferenceIO {σ : StaticSpec} (src : @& Tensor σ) : IO (Tensor σ) := do
  match src.checkedReplacement? (← cloneInferenceIORaw σ.shape src.raw) with
  | .ok tensor => pure tensor
  | .error message => throw (IO.userError s!"cloneInferenceIO: {message}")

end torch.data
