/-
  Tyr/Model/Utils.lean

  Shared model utility functions used across multiple model families.
  These are higher-level than Tyr.Torch primitives and are specific to
  neural network initialization and token-level inference helpers.

  All utilities are typed: shapes (and, where the runtime fixes them,
  dtypes) are carried in the `Tensor` spec.
-/
import Tyr.Typed

namespace torch.Model

open torch

/-- Cast a freshly allocated Float32 tensor to `dtype`. A dynamic
    (`Unknown`) dtype keeps the Float32 handle, which the dynamic claim
    already admits. -/
private def castToDType {s : Shape} (t : DTensor s .Float32) (dtype : DType) : DTensor s dtype :=
  match dtype.safeTensorTag? with
  | some tag => .assumeSpec (RawTensor.castDType s t.raw tag)
  | none => .assumeSpec t.raw

/-- Kaiming-like weight init: randn scaled by sqrt(2 / fanIn).
    The runtime factory produces Float32; the result is cast to `dtype`. -/
def initWeight (shape : Shape) (fanIn : UInt64) (dtype : DType := .Float32) : IO (DTensor shape dtype) := do
  let std := Float.sqrt (2.0 / fanIn.toFloat)
  let w ← Tensor.randn shape
  let w := Tensor.Preserving.mulScalar w std
  pure (Tensor.setRequiresGrad (castToDType w dtype) true)

/-- Zero bias tensor with requires_grad=true. -/
def initBias (shape : Shape) (dtype : DType := .Float32) : DTensor shape dtype :=
  Tensor.setRequiresGrad (castToDType (Tensor.zeros shape) dtype) true

/-- Mark a loaded pretrained tensor as frozen (requires_grad=false). -/
def reqGradFalse {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  Tensor.setRequiresGrad t false

/-- Dequantize FP8 weights with blockwise inverse scales (128x128 blocks).
    weight: [out, in], scale_inv: [out/128, in/128]. Returns float32 weights.
    The block reshapes are runtime-validated (`reshapeChecked`): the block
    factors are symbolic UInt64 division, so no static numel proof exists. -/
def dequantizeFP8 {outDim inDim : UInt64}
    (weight : STensor #[outDim, inDim])
    (scaleInv : STensor #[outDim / 128, inDim / 128])
    : DTensor #[outDim, inDim] .Float32 :=
  let outBlocks := outDim / 128
  let inBlocks := inDim / 128
  let w := weight.toFloat32
  let s := scaleInv.toFloat32
  let w := w.reshapeChecked #[outBlocks, 128, inBlocks, 128]
  let s := s.reshapeChecked #[outBlocks, 1, inBlocks, 1]
  let s := s.expand #[outBlocks, 128, inBlocks, 128]
  let w := w * s
  w.reshapeChecked #[outDim, inDim]

/-- Dequantize FP8 expert weights with blockwise inverse scales per expert.
    weight: [experts, out, in], scale_inv: [experts, out/128, in/128]. -/
def dequantizeFP8Experts {experts outDim inDim : UInt64}
    (weight : STensor #[experts, outDim, inDim])
    (scaleInv : STensor #[experts, outDim / 128, inDim / 128])
    : DTensor #[experts, outDim, inDim] .Float32 :=
  let outBlocks := outDim / 128
  let inBlocks := inDim / 128
  let w := weight.toFloat32
  let s := scaleInv.toFloat32
  let w := w.reshapeChecked #[experts, outBlocks, 128, inBlocks, 128]
  let s := s.reshapeChecked #[experts, outBlocks, 1, inBlocks, 1]
  let s := s.expand #[experts, outBlocks, 128, inBlocks, 128]
  let w := w * s
  w.reshapeChecked #[experts, outDim, inDim]

end torch.Model
