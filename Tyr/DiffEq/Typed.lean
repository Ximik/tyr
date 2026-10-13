import Tyr.DiffEq.Types
import Tyr.Typed.Ops

/-!
Differential-equation algebra for indexed tensors. Solver arithmetic retains
shape, dtype and device policy. Scalar operations validate native dtype
preservation, so integer promotion cannot invalidate a solver state's index.
-/

namespace torch.DiffEq

instance {σ : StaticSpec} : DiffEqSpace (Tensor σ) where
  add := Add.add
  sub := Sub.sub
  scale := fun value tensor => Tensor.Preserving.mulScalar tensor value

instance {σ : StaticSpec} : DiffEqElem (Tensor σ) where
  abs := Tensor.abs
  max := Tensor.maximum
  addScalar := fun value tensor => Tensor.Preserving.addScalar tensor value
  div := Tensor.Preserving.div

instance {σ : StaticSpec} : DiffEqSeminorm (Tensor σ) where
  rms tensor := Float.sqrt (
    Tensor.item (Tensor.sumAll (Mul.mul tensor tensor)) /
      (TensorSpec.numelOfShape σ.shape).toFloat)

end torch.DiffEq
