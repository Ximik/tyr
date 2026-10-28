import Tyr.Typed.Tensor
import Tyr.RawTensor.OwnedKV

/- Tyr/Inference/OwnedKV.lean

  Effectful inference-only tensor updates for KV-cache storage.
  Each update returns the destination value to use next; retained aliases
  preserve their values. Native updates consume the destination ownership
  token and reuse storage only when it is safe. -/

namespace torch.data

open torch.data.Native

/-- Return the destination with the requested slice replaced. Retained aliases
keep their original values. The native operation disables autograd for the copy. -/
def copySliceIO {σ σsrc : StaticSpec} (dst : Tensor σ) (dim start : Nat)
    (src : @& Tensor σsrc) : IO (Tensor σ) := do
  let dtype := dst.raw.dtype
  let device := dst.raw.device
  let raw ← copySliceIORaw dst.raw dim start src.raw
  match Tensor.checkedUpdate? dtype device raw with
  | .ok result => pure result
  | .error message => throw (IO.userError s!"copySliceIO: {message}")

/-- Allocate an independent inference tensor copy, retaining the full static
spec. The IO effect prevents Lean from merging identical allocations when
distinct mutable buffers are required. -/
def cloneInferenceIO {σ : StaticSpec} (src : @& Tensor σ) : IO (Tensor σ) := do
  match src.checkedReplacement? (← cloneInferenceIORaw src.raw) with
  | .ok tensor => pure tensor
  | .error message => throw (IO.userError s!"cloneInferenceIO: {message}")

end torch.data
