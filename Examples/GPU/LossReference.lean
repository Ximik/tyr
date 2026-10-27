import Tyr.Typed.Ops

namespace Examples.GPU.RunLoss

open torch

variable {device : Device}

/-- Float32 loss accumulation and a BF16 cotangent for BF16 training logits. -/
def referenceTraining {rows : UInt64}
    (logitsBase : Tensor { shape := #[rows, 3072], dtype := .BFloat16, device := device })
    (targets : Tensor { shape := #[rows], dtype := .Int64, device := device }) :
    IO (Tensor { shape := #[rows], dtype := .Float32, device := device } ×
      Tensor { shape := #[], dtype := .Float32, device := device } ×
      Tensor { shape := #[rows, 3072], dtype := .BFloat16, device := device }) := do
  let logits ← Tensor.parameter logitsBase
  let logitsF := logits.toFloat32
  let losses := Tensor.crossEntropyNone logitsF targets
  let mean := Tensor.crossEntropy logitsF targets
  let gradient ← Tensor.grad mean logits (Tensor.onesLike mean)
  return (losses, mean, gradient)

end Examples.GPU.RunLoss
