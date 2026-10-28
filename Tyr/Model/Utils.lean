/-
  Tyr/Model/Utils.lean

  Shared model utility functions used across multiple model families.
  These are higher-level than Tyr.Torch primitives and are specific to
  neural network initialization and token-level inference helpers.

  All utilities are typed: shapes (and, where the runtime fixes them,
  dtypes) are carried in the `Tensor` spec.
-/
import Tyr.Typed
import Tyr.Typed.Neural
import Tyr.ConstantSupport

namespace torch.Model

variable {device : Device}
open torch

/-- Kaiming-like weight init: randn scaled by sqrt(2 / fanIn).
    The runtime factory produces Float32; the result is cast to `dtype`. -/
def initWeight (shape : Shape) (fanIn : Nat) (dtype : DType := .Float32) (device : Device := .CPU) : IO (Tensor { device := device, shape := shape, dtype := dtype }) := do
  let std := Float.sqrt (2.0 / fanIn.toFloat)
  let w ← Tensor.randn shape (device := device)
  let w ← IO.ofExcept (Tensor.mulScalar? w std)
  let w ← IO.ofExcept (w.castDType? dtype)
  Tensor.parameter w

/-- Zero bias tensor with requires_grad=true. -/
def initBias (shape : Shape) (dtype : DType := .Float32) (device : Device := .CPU) : IO (Tensor { device := device, shape := shape, dtype := dtype }) := do
  let support ← dtype.acquireConstantSupport device
  let _ : DType.ConstantSupported dtype device := support.down
  IO.ofExcept (Shape.validateAllocation shape)
  Tensor.parameter (Tensor.zerosFor { shape, dtype, device })

/-- Mark a loaded pretrained tensor as frozen (requires_grad=false). -/
def reqGradFalse {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  Tensor.detach t

/-- Blockwise FP8 scales cover whole 128x128 blocks; checked where the scales are loaded. -/
def checkFP8Blocks (name : String) (outDim inDim : Nat) : IO (PLift (128 ∣ outDim ∧ 128 ∣ inDim)) :=
  if h : 128 ∣ outDim ∧ 128 ∣ inDim then pure ⟨h⟩
  else throw <| IO.userError
    s!"{name}: FP8 block scales need dimensions divisible by 128, got {outDim}x{inDim}"

/-- Proves an FP8 block reshape from the `128 ∣ outDim ∧ 128 ∣ inDim` hypothesis in scope. -/
local macro "fp8_blocks" : tactic => `(tactic| (
  obtain ⟨⟨a, ha⟩, ⟨b, hb⟩⟩ := ‹128 ∣ _ ∧ 128 ∣ _›
  subst ha hb
  simp +zetaDelta [Nat.mul_div_cancel_left _ (by decide : 0 < 128), Nat.mul_assoc, Nat.mul_comm,
    Nat.mul_left_comm]))

/-- Dequantize FP8 weights with blockwise inverse scales (128x128 blocks).
    weight: [out, in], scale_inv: [out/128, in/128]. Returns float32 weights. -/
def dequantizeFP8 {outDim inDim : Nat} {weightDType scaleDType : DType}
    (weight : Tensor { device := device, shape := #[outDim, inDim], dtype := weightDType })
    (scaleInv : Tensor { device := device, shape := #[outDim / 128, inDim / 128], dtype := scaleDType })
    (hBlocks : 128 ∣ outDim ∧ 128 ∣ inDim)
    : Tensor { device := device, shape := #[outDim, inDim], dtype := .Float32 } :=
  let outBlocks := outDim / 128
  let inBlocks := inDim / 128
  let w := weight.toFloat32
  let s := scaleInv.toFloat32
  let w := w.reshape #[outBlocks, 128, inBlocks, 128] (by fp8_blocks)
  let s := s.reshape #[outBlocks, 1, inBlocks, 1]
  let s := s.expand #[outBlocks, 128, inBlocks, 128]
  let w := w * s
  w.reshape #[outDim, inDim] (by fp8_blocks)

/-- Dequantize FP8 expert weights with blockwise inverse scales per expert.
    weight: [experts, out, in], scale_inv: [experts, out/128, in/128]. -/
def dequantizeFP8Experts {experts outDim inDim : Nat} {weightDType scaleDType : DType}
    (weight : Tensor { device := device, shape := #[experts, outDim, inDim], dtype := weightDType })
    (scaleInv : Tensor { device := device, shape := #[experts, outDim / 128, inDim / 128], dtype := scaleDType })
    (hBlocks : 128 ∣ outDim ∧ 128 ∣ inDim)
    : Tensor { device := device, shape := #[experts, outDim, inDim], dtype := .Float32 } :=
  let outBlocks := outDim / 128
  let inBlocks := inDim / 128
  let w := weight.toFloat32
  let s := scaleInv.toFloat32
  let w := w.reshape #[experts, outBlocks, 128, inBlocks, 128] (by fp8_blocks)
  let s := s.reshape #[experts, outBlocks, 1, inBlocks, 1]
  let s := s.expand #[experts, outBlocks, 128, inBlocks, 128]
  let w := w * s
  w.reshape #[experts, outDim, inDim] (by fp8_blocks)

end torch.Model
