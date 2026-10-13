/-
  Tyr/Modular/Compose.lean

  Composition rules for the modular norm.

  The modular norm is defined recursively via two operations:
  1. Sequential composition (M₂ ∘ M₁): layers applied in sequence
  2. Parallel concatenation (M₁, M₂): layers applied to different parts

  Key formulas from the paper:
  - Sequential: ||w||_{M₂∘M₁} = max(ν₂·||w₁||, ||w₂||)
  - Sequential: ν_{M₂∘M₁} = ν₁ · ν₂
  - Parallel: ||w|| = √(||w₁||² + ||w₂||²) for ℓ₂ combination
-/
import Tyr.Modular.Norm

namespace Tyr.Modular

open torch (TensorStruct)

/-! ## Product Types

Products represent either:
- Sequential composition: (M₁, M₂) where M₂ is applied after M₁
- Parallel composition: modules operating on different inputs

We provide a `Sequential` wrapper to distinguish composition semantics.
-/

/-- Marker for sequential composition: M₂ applied after M₁ -/
structure Sequential (M₁ M₂ : Type) where
  first : M₁
  second : M₂

namespace Sequential

def mk' (m₁ : M₁) (m₂ : M₂) : Sequential M₁ M₂ := ⟨m₁, m₂⟩

end Sequential

/-- TensorStruct for Sequential just combines the two modules -/
instance [TensorStruct M₁] [TensorStruct M₂] : TensorStruct (Sequential M₁ M₂) where
  map f s := ⟨TensorStruct.map f s.first, TensorStruct.map f s.second⟩
  mapM f s := do
    let first' ← TensorStruct.mapM f s.first
    let second' ← TensorStruct.mapM f s.second
    pure ⟨first', second'⟩
  zipWith f s1 s2 :=
    ⟨TensorStruct.zipWith f s1.first s2.first,
     TensorStruct.zipWith f s1.second s2.second⟩
  fold f init s :=
    let acc := TensorStruct.fold f init s.first
    TensorStruct.fold f acc s.second

/-! ## Sequential Composition

For M₂ ∘ M₁ (M₂ applied after M₁):

  ||w||_{M₂∘M₁} = max(ν₂ · ||w₁||_{M₁}, ||w₂||_{M₂})

This ensures that the "downstream" sensitivity ν₂ scales up the first layer's norm.

  ν_{M₂∘M₁} = ν₁ · ν₂  (sensitivities multiply)
  μ_{M₂∘M₁} = max(μ₁ · ν₂, μ₂)
-/

instance [NormedModule M₁] [NormedModule M₂] : NormedModule (Sequential M₁ M₂) where
  norm s :=
    let n₁ := NormedModule.norm s.first
    let n₂ := NormedModule.norm s.second
    let ν₂ := NormedModule.nu s.second
    floatMax (ν₂ * n₁) n₂

  dualNorm s :=
    let d₁ := NormedModule.dualNorm s.first
    let d₂ := NormedModule.dualNorm s.second
    let ν₂ := NormedModule.nu s.second
    -- Dual of max is sum with scaling
    ν₂ * d₁ + d₂

  nu s := NormedModule.nu s.first * NormedModule.nu s.second

  mu s :=
    let μ₁ := NormedModule.mu s.first
    let μ₂ := NormedModule.mu s.second
    let ν₂ := NormedModule.nu s.second
    floatMax (μ₁ * ν₂) μ₂

  normalize s :=
    -- Normalize both components to "saturate" the max
    ⟨NormedModule.normalize s.first, NormedModule.normalize s.second⟩

  normalizeDual s :=
    ⟨NormedModule.normalizeDual s.first, NormedModule.normalizeDual s.second⟩

/-! ## Parallel Composition (Product Types)

For parallel modules (M₁, M₂) operating on different inputs:

  ||(w₁, w₂)|| = √(||w₁||² + ||w₂||²)  (ℓ₂ combination)

This is appropriate when the outputs are concatenated or added.

For outputs that are separately processed:
  ||(w₁, w₂)|| = max(||w₁||, ||w₂||)  (ℓ∞ combination)
-/

/-- Parallel composition with ℓ₂ norm combination -/
instance [NormedModule M₁] [NormedModule M₂] : NormedModule (M₁ × M₂) where
  map f p := (TensorStruct.map f p.1, TensorStruct.map f p.2)
  mapM f p := do
    let m₁' ← TensorStruct.mapM f p.1
    let m₂' ← TensorStruct.mapM f p.2
    pure (m₁', m₂')
  zipWith f a b :=
    (TensorStruct.zipWith f a.1 b.1, TensorStruct.zipWith f a.2 b.2)
  fold f init p :=
    let acc := TensorStruct.fold f init p.1
    TensorStruct.fold f acc p.2

  -- ℓ₂ combination of norms
  norm p :=
    let n₁ := NormedModule.norm p.1
    let n₂ := NormedModule.norm p.2
    Float.sqrt (n₁ * n₁ + n₂ * n₂)

  dualNorm p :=
    let d₁ := NormedModule.dualNorm p.1
    let d₂ := NormedModule.dualNorm p.2
    Float.sqrt (d₁ * d₁ + d₂ * d₂)

  -- For parallel, take max sensitivity
  nu p := floatMax (NormedModule.nu p.1) (NormedModule.nu p.2)

  mu p :=
    let μ₁ := NormedModule.mu p.1
    let μ₂ := NormedModule.mu p.2
    Float.sqrt (μ₁ * μ₁ + μ₂ * μ₂)

  normalize p :=
    (NormedModule.normalize p.1, NormedModule.normalize p.2)

  normalizeDual p :=
    (NormedModule.normalizeDual p.1, NormedModule.normalizeDual p.2)

/-! ## Array and Vector Instances

For a homogeneous array of modules (like transformer blocks):
  ||[w₁, ..., wₙ]|| = max_i ||wᵢ|| · √n  (for sequential)
  ||[w₁, ..., wₙ]|| = √(Σᵢ ||wᵢ||²)    (for parallel)

We use ℓ₂ combination as the default for arrays.
-/

instance [NormedModule M] : NormedModule (Array M) where
  norm arr :=
    let norms := arr.map NormedModule.norm
    Float.sqrt (norms.foldl (fun acc n => acc + n * n) 0.0)

  dualNorm arr :=
    let duals := arr.map NormedModule.dualNorm
    Float.sqrt (duals.foldl (fun acc d => acc + d * d) 0.0)

  nu arr :=
    -- For sequential array, multiply sensitivities
    arr.foldl (fun acc m => acc * NormedModule.nu m) 1.0

  mu arr :=
    let mus := arr.map NormedModule.mu
    Float.sqrt (mus.foldl (fun acc μ => acc + μ * μ) 0.0)

  normalize arr := arr.map NormedModule.normalize

  normalizeDual arr := arr.map NormedModule.normalizeDual

instance [NormedModule M] {n : Nat} : NormedModule (torch.Vector n M) where
  norm v :=
    let norms := v.data.map NormedModule.norm
    Float.sqrt (norms.foldl (fun acc n => acc + n * n) 0.0)

  dualNorm v :=
    let duals := v.data.map NormedModule.dualNorm
    Float.sqrt (duals.foldl (fun acc d => acc + d * d) 0.0)

  nu v :=
    v.data.foldl (fun acc m => acc * NormedModule.nu m) 1.0

  mu v :=
    let mus := v.data.map NormedModule.mu
    Float.sqrt (mus.foldl (fun acc μ => acc + μ * μ) 0.0)

  normalize v := torch.Vector.map NormedModule.normalize v

  normalizeDual v := torch.Vector.map NormedModule.normalizeDual v

/-! ## Utility: Compose operator -/

/-- Compose two modules sequentially: M₂ after M₁ -/
def compose (m₁ : M₁) (m₂ : M₂) : Sequential M₁ M₂ := ⟨m₁, m₂⟩

infixr:90 " ⊛ " => compose

end Tyr.Modular
