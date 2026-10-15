/-
  Tyr/Modular/Norm.lean

  Modular norm typeclass for neural network weight spaces.

  Based on "Scalable Optimization in the Modular Norm" (NeurIPS 2024).
  https://arxiv.org/abs/2405.14813

  The modular norm is defined recursively on the weight space of any neural
  network architecture. Key properties:
  - Learning rates transfer across width and depth
  - Gradient Lipschitz constants follow a simple recursive formula
  - Normalized updates prevent unstable activation changes
-/
import Tyr.TensorStruct

namespace Tyr.Modular

open torch (TensorStruct)

/-- Maximum of two floats -/
def floatMax (a b : Float) : Float := if a > b then a else b

/-! ## Modular Norm Typeclass

The modular norm provides a principled way to measure and normalize weight updates.
For atomic modules:
- Linear: spectral norm (largest singular value)
- Embedding: max row norm
- Bias/Scale: ℓ₂ norm

For composed modules, norms combine via recursive formulas involving:
- ν (nu): input sensitivity bound ||∂f/∂x||
- μ (mu): weight sensitivity bound ||∂f/∂w||
-/

/-- A module with a defined norm on its weight space.

    The modular norm enables scale-invariant optimization where learning rates
    transfer across network width and depth.

    Mathematical properties:
    - `norm w` measures the "size" of weights in the modular norm
    - `dualNorm g` measures gradients in the dual norm
    - `nu` bounds the Lipschitz constant w.r.t. inputs
    - `mu` bounds the Lipschitz constant w.r.t. weights -/
class NormedModule (M : Type) extends TensorStruct M where
  /-- Modular norm on weight space: ||w||_M -/
  norm : M → Float

  /-- Dual norm for gradients: ||g||_M*.
      For spectral norm, the dual is the nuclear norm. -/
  dualNorm : M → Float

  /-- Input sensitivity: ||∂f/∂x|| ≤ ν.
      For linear layers, this equals the spectral norm. -/
  nu : M → Float

  /-- Weight sensitivity: ||∂f/∂w|| ≤ μ.
      For normalized modules, this is typically 1. -/
  mu : M → Float

  /-- Normalize weights to unit modular norm: w / ||w||_M -/
  normalize : M → M

  /-- Normalize in dual norm (for gradient updates).
      This "saturates" the norm constraint for optimal descent. -/
  normalizeDual : M → M

namespace NormedModule

/-- Check if module is well-normed (norm ≤ 1) -/
def isWellNormed [NormedModule M] (m : M) : Bool :=
  norm m ≤ 1.0

/-- Lipschitz constant for the full module.
    This is the product ν · μ for the composed network. -/
def lipschitzConstant [NormedModule M] (m : M) : Float :=
  nu m * mu m

/-- Scale weights by a constant factor -/
def scale [NormedModule M] [torch.FloatingTensorStruct M] (s : Float) (m : M) : M :=
  torch.FloatingTensorStruct.map (fun t => torch.Tensor.mulScalar t s) m

/-- Normalize a gradient update and apply with learning rate.
    Δw = -lr · normalizeDual(grad)
    This is the core of modular optimization. -/
def normalizedUpdate [NormedModule M] [torch.FloatingTensorStruct M] (lr : Float) (grad : M) : M :=
  scale (-lr) (normalizeDual grad)

end NormedModule

/-! ## Trivial Instances -/

/-- Float is a trivial 0-dimensional normed module.
    It contains no tensors, so TensorStruct operations are identity. -/
instance : NormedModule Float where
  map _ x := x  -- No tensors to map
  mapM _ x := pure x
  zipWith _ x _ := x  -- Just return first (they should be equal in valid use)
  fold _ init _ := init  -- No tensors to fold
  norm x := Float.abs x
  dualNorm x := Float.abs x
  nu _ := 1.0
  mu _ := 1.0
  normalize x := if x == 0.0 then 0.0 else x / Float.abs x
  normalizeDual x := if x == 0.0 then 0.0 else x / Float.abs x

/-- Static values have trivial zero norm (they don't contribute to optimization) -/
instance [Inhabited α] : NormedModule (torch.Static α) where
  map _ s := s
  mapM _ s := pure s
  zipWith _ s _ := s
  fold _ init _ := init
  norm _ := 0.0
  dualNorm _ := 0.0
  nu _ := 1.0  -- Identity for composition
  mu _ := 0.0  -- No contribution
  normalize s := s
  normalizeDual s := s

end Tyr.Modular
