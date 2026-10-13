import Tyr.Typed.Ops

namespace Examples.GPU.RunOptimizer
open torch

/-- AdamW update with Float32 master weights/moments and BF16 model weights/gradients. -/
def referenceUpdate {elements : UInt64}
    (master : DTensor #[elements] .Float32) (grad : DTensor #[elements] .BFloat16)
    (moment1 moment2 : DTensor #[elements] .Float32)
    (learningRate beta1 beta2 epsilon weightDecay invBias1 invBias2 : Float) := Id.run do
  let grad32 := grad.toFloat32
  let expectedM := Tensor.Preserving.mulScalar moment1 beta1 +
    Tensor.Preserving.mulScalar grad32 (1.0 - beta1)
  let expectedV := Tensor.Preserving.mulScalar moment2 beta2 +
    Tensor.Preserving.mulScalar (grad32 * grad32) (1.0 - beta2)
  let mHat := Tensor.Preserving.mulScalar expectedM invBias1
  let vHat := Tensor.Preserving.mulScalar expectedV invBias2
  let invDenom := Tensor.pow (Tensor.sqrt vHat + epsilon) (-1.0)
  let update := mHat * invDenom + Tensor.Preserving.mulScalar master weightDecay
  let expectedMaster := master - Tensor.Preserving.mulScalar update learningRate
  return (expectedMaster, expectedMaster.toBFloat16, expectedM, expectedV)

end Examples.GPU.RunOptimizer
