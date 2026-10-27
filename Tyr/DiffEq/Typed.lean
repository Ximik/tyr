import Tyr.DiffEq.Types
import Tyr.Typed.Ops

/-!
Differential-equation algebra for indexed tensors. Solver arithmetic retains
shape, dtype and device. Real-floating evidence makes the solver state
closed under its arithmetic operations.
-/

namespace torch.DiffEq

instance {σ : StaticSpec} [DType.RealFloating σ.dtype] : DiffEqSpace (Tensor σ) where
  add := Tensor.add
  sub := Tensor.sub
  scale := fun value tensor => Tensor.mulScalar tensor value

instance {σ : StaticSpec} [DType.RealFloating σ.dtype] : DiffEqElem (Tensor σ) where
  abs := fun t => by simpa using Tensor.abs t
  max := Tensor.maximum
  addScalar := fun value tensor => Tensor.addScalar tensor value
  div := fun a b => by simpa using Tensor.div a b

instance {σ : StaticSpec} [DType.RealFloating σ.dtype] : DiffEqSeminorm (Tensor σ) where
  rms tensor := Float.sqrt (
    Tensor.item (Tensor.sumAll (Tensor.mul tensor tensor)) /
      (TensorSpec.numelOfShape σ.shape).toFloat)

end torch.DiffEq
