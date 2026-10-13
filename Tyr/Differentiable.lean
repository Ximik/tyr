import Tyr.Typed.Ops

namespace torch

/-- A value with an associated cotangent representation and gradient extraction. -/
class differentiable (α : Type u) where
  cotangent_space : Type u
  grad : α → cotangent_space

/-- Tensor gradients retain the complete static specification. -/
instance {σ : StaticSpec} : differentiable (Tensor σ) where
  cotangent_space := Tensor σ
  grad := Tensor.gradOf

end torch
