/-
  Tyr/Lib/Module/Core.lean

  Equinox-style module typeclass for neural networks.
-/
import Tyr.Core.TensorStruct

namespace torch

/-! ## Module Typeclass

`Module M In Out` is a pure parameterized function `M → In → Out`. It extends
`torch.TensorStruct M`, so generic parameter transforms (detach, grads,
checkpoint traversal) apply independently of the forward API.

```lean
structure MyLinear (inDim outDim : Nat) where
  weight : Tensor { shape := #[outDim, inDim], dtype := .Float32, device := .CPU }
  bias : Option (Tensor { shape := #[outDim], dtype := .Float32, device := .CPU }) := none
  deriving TensorStruct

instance {inDim outDim batch : Nat} :
    Module (MyLinear inDim outDim) (Tensor { shape := #[batch, inDim], dtype := .Float32, device := .CPU }) (Tensor { shape := #[batch, outDim], dtype := .Float32, device := .CPU }) where
  forward m x :=
    let y := Tensor.linear x m.weight
    match m.bias with
    | some b => Tensor.add y (Tensor.broadcastTo b #[batch, outDim])
    | none => y
```
-/

/-- A pure module: a parameterized function with no side effects.
    The module type `M` must have a `torch.TensorStruct` instance for parameter traversal. -/
class Module (M : Type) (In : Type) (Out : Type) extends TensorStruct M where
  forward : M → In → Out

end torch
