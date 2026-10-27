import Tyr.Basic

/-! Native KV-cache update bindings wrapped by `Tyr.Inference.OwnedKV`. -/

namespace torch.data.Native

/-- Checked native slice update, consuming the destination reference. -/
@[extern "lean_torch_copy_slice_io"]
opaque copySliceIORaw (dst : RawTensor)
    (dim start : UInt64) (src : @& RawTensor) : IO RawTensor

/-- Raw ABI for the effectful inference clone. The shape argument is passed
for ABI compatibility only; the native side never reads it. -/
@[extern "lean_torch_clone_inference_io"]
opaque cloneInferenceIORaw (src : @& RawTensor) : IO RawTensor

end torch.data.Native
