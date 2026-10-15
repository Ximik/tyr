import Tyr.Typed.Ops

namespace Examples.GPU.RunOptimizer
open torch

variable {device : Device}

/-- AdamW update with Float32 master weights/moments and BF16 model weights/gradients. -/
def referenceUpdate {elements : UInt64}
    (master : Tensor { shape := #[elements], dtype := .Float32, device := device }) (grad : Tensor { shape := #[elements], dtype := .BFloat16, device := device })
    (moment1 moment2 : Tensor { shape := #[elements], dtype := .Float32, device := device })
    (learningRate beta1 beta2 epsilon weightDecay invBias1 invBias2 : Float) := Id.run do
  let grad32 := grad.toFloat32
  let expectedM := Tensor.mulScalar moment1 beta1 +
    Tensor.mulScalar grad32 (1.0 - beta1)
  let expectedV := Tensor.mulScalar moment2 beta2 +
    Tensor.mulScalar (grad32 * grad32) (1.0 - beta2)
  let mHat := Tensor.mulScalar expectedM invBias1
  let vHat := Tensor.mulScalar expectedV invBias2
  let invDenom := Tensor.pow (Tensor.sqrt vHat + epsilon) (-1.0)
  let update := mHat * invDenom + Tensor.mulScalar master weightDecay
  let expectedMaster := master - Tensor.mulScalar update learningRate
  return (expectedMaster, expectedMaster.toBFloat16, expectedM, expectedV)

end Examples.GPU.RunOptimizer
