import Tyr.Typed.Tensor

/- Tyr/Inference/OwnedKV.lean

  Effectful inference-only buffer primitives for owned KV-cache storage,
  exposed on the indexed `Tensor σ`.

  The `@[extern]` entry points keep the existing C ABI (`cc/src/tyr.cpp`,
  checked against `cc/src/tyr_owned_kv_abi.h`): the native side ignores the
  shape arguments, borrows the tensor handles, and disables autograd for the
  copy. The typed wrappers assert the static spec relations — the destination
  and source each keep their own spec — while the native `copy_` still
  validates shapes/device/dtype at runtime. -/

namespace torch.data

/-- Raw ABI for the effectful slice write. The shape arguments are passed for
ABI compatibility only; the native side never reads them. -/
@[extern "lean_torch_copy_slice_io"]
private opaque copySliceIORaw (dstShape srcShape : Shape) (dst : @& RawTensor)
    (dim start : UInt64) (src : @& RawTensor) : IO Unit

/-- Raw ABI for the effectful inference clone. The shape argument is passed
for ABI compatibility only; the native side never reads it. -/
@[extern "lean_torch_clone_inference_io"]
private opaque cloneInferenceIORaw (shape : Shape) (src : @& RawTensor) : IO RawTensor

/-- Effectful inference-only slice write. Callers must own the destination and
serialize access; aliases observe this mutation. Unlike the legacy pure
`sliceScatterInplace`, the IO effect orders reads and writes explicitly.
The native operation disables autograd for the copy. -/
def copySliceIO {σ σsrc : StaticSpec} (dst : @& Tensor σ) (dim start : UInt64)
    (src : @& Tensor σsrc) : IO Unit :=
  copySliceIORaw σ.shape σsrc.shape dst.raw dim start src.raw

/-- Allocate an independent inference tensor copy, retaining the full static
spec. The IO effect prevents Lean from merging identical allocations when
distinct mutable buffers are required. -/
def cloneInferenceIO {σ : StaticSpec} (src : @& Tensor σ) : IO (Tensor σ) := do
  pure (Tensor.assumeSpec (← cloneInferenceIORaw σ.shape src.raw))

end torch.data
