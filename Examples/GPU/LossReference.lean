import Tyr.Typed.Ops

namespace Examples.GPU.RunLoss

open torch

/-- Float32 loss accumulation and a BF16 cotangent for BF16 training logits. -/
def referenceTraining {rows : UInt64}
    (logitsBase : Tensor { shape := #[rows, 3072], dtype := .BFloat16 })
    (targets : Tensor { shape := #[rows], dtype := .Int64 }) :
    IO (Tensor { shape := #[rows], dtype := .Float32 } ×
      Tensor { shape := #[], dtype := .Float32 } ×
      Tensor { shape := #[rows, 3072], dtype := .BFloat16 }) := do
  let logits ← Tensor.parameter logitsBase
  let logitsF := logits.toFloat32
  let losses := Tensor.crossEntropyNone logitsF targets
  let mean := Tensor.crossEntropy logitsF targets
  let gradient := Tensor.grad mean logits (Tensor.onesLike mean)
  return (losses, mean, gradient)

end Examples.GPU.RunLoss
