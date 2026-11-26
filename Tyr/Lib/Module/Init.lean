/-
  Tyr/Lib/Module/Init.lean

  Parameter initialization.
-/
import Tyr.Core.Typed
import Tyr.Core.Typed.Neural
import Tyr.Core.FFI.ConstantSupport

namespace torch.Model

open torch

/-- Kaiming-like weight init: randn scaled by sqrt(2 / fanIn).
    The runtime factory produces Float32; the result is cast to `dtype`. -/
def initWeight (shape : Shape) (fanIn : Nat) (dtype : DType := .Float32) (device : Device := .CPU) : IO (Tensor { device := device, shape := shape, dtype := dtype }) := do
  let std := Float.sqrt (2.0 / fanIn.toFloat)
  let w ← Tensor.randn shape (device := device)
  let w := Tensor.mulScalar w std
  let w := w.castDType dtype
  Tensor.parameter w

/-- Zero bias tensor with requires_grad=true. -/
def initBias (shape : Shape) (dtype : DType := .Float32) (device : Device := .CPU) : IO (Tensor { device := device, shape := shape, dtype := dtype }) := do
  let support ← dtype.acquireConstantSupport device
  let _ : DType.ConstantSupported dtype device := support.down
  Tensor.parameter (Tensor.zerosFor { shape, dtype, device })

end torch.Model
