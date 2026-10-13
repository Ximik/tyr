import Tyr.Typed.Ops

namespace Examples.GPU.RunLoss

open torch

/-- Float32 loss accumulation and a BF16 cotangent for BF16 training logits. -/
def referenceTraining {rows : UInt64}
    (logitsBase : DTensor #[rows, 3072] .BFloat16)
    (targets : DTensor #[rows] .Int64) := Id.run do
  let logits := Tensor.setRequiresGrad logitsBase true
  let logitsF := logits.toFloat32
  let losses := Tensor.crossEntropyNone logitsF targets
  let mean := Tensor.crossEntropy logitsF targets
  let gradient := Tensor.grad mean logits (Tensor.onesLike mean)
  return (losses, mean, gradient)

end Examples.GPU.RunLoss
