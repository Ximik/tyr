/-
  Tyr/Modular/Atomic.lean

  NormedModule instances for atomic (leaf) neural network modules.

  Atomic modules are those whose norms are directly defined, not composed:
  - Linear: spectral norm (largest singular value)
  - LayerNorm: ℓ₂ norm of weight and bias
  - Embedding: max row norm

  These form the base cases for the recursive modular norm.
-/
import Tyr.Modular.Norm
import Tyr.Module.Linear
import Tyr.Module.LayerNorm

namespace Tyr.Modular

open torch

/-! ## Linear Layer

For a linear layer y = Wx, the modular norm is the spectral norm:
  ||W||_σ = σ_max(W)

The dual norm (for gradients) is the nuclear norm:
  ||G||_* = Σ σᵢ(G)

Input sensitivity ν = ||W||_σ since ||Wx||₂ ≤ ||W||_σ ||x||₂.
-/

instance (in_dim out_dim : UInt64) {d : DType} {dev : Device} :
    NormedModule (torch.Linear in_dim out_dim d dev) where
  norm m :=
    let weightNorm := Tensor.spectralNorm m.weight
    match m.bias with
    | some b => floatMax weightNorm (Tensor.l2Norm b)
    | none => weightNorm

  dualNorm m :=
    let weightDual := Tensor.nuclearNorm m.weight
    match m.bias with
    | some b => weightDual + Tensor.l2Norm b
    | none => weightDual

  nu m := Tensor.spectralNorm m.weight

  mu _ := 1.0  -- For well-normed modules

  normalize m :=
    let σ := Tensor.spectralNorm m.weight
    if σ == 0.0 then m else
    let normalizedWeight := Tensor.Preserving.mulScalar m.weight (1.0 / σ)
    match m.bias with
    | some b =>
      let bNorm := Tensor.l2Norm b
      let normalizedBias := if bNorm == 0.0 then b else Tensor.Preserving.mulScalar b (1.0 / bNorm)
      { m with weight := normalizedWeight, bias := some normalizedBias }
    | none => { m with weight := normalizedWeight }

  normalizeDual m :=
    -- For spectral norm, the dual normalization involves SVD projection
    -- For now, use Frobenius normalization as an approximation
    let fNorm := Tensor.frobeniusNorm m.weight
    if fNorm == 0.0 then m else
    let normalizedWeight := Tensor.Preserving.mulScalar m.weight (1.0 / fNorm)
    match m.bias with
    | some b =>
      let bNorm := Tensor.l2Norm b
      let normalizedBias := if bNorm == 0.0 then b else Tensor.Preserving.mulScalar b (1.0 / bNorm)
      { m with weight := normalizedWeight, bias := some normalizedBias }
    | none => { m with weight := normalizedWeight }

/-! ## LayerNorm

LayerNorm has learnable scale (weight) and shift (bias) parameters.
The modular norm combines them via ℓ₂:
  ||(γ, β)||_M = √(||γ||₂² + ||β||₂²)

Input sensitivity ν depends on the activation range, typically O(1).
-/

instance (dim : UInt64) {d : DType} {dev : Device} :
    NormedModule (torch.LayerNorm dim d dev) where
  norm m :=
    let wNorm := Tensor.l2Norm m.weight
    let bNorm := Tensor.l2Norm m.bias
    Float.sqrt (wNorm * wNorm + bNorm * bNorm)

  dualNorm m :=
    -- Dual of ℓ₂ is ℓ₂
    let wNorm := Tensor.l2Norm m.weight
    let bNorm := Tensor.l2Norm m.bias
    Float.sqrt (wNorm * wNorm + bNorm * bNorm)

  nu _ := 1.0  -- LayerNorm approximately preserves scale

  mu _ := 1.0

  normalize m :=
    -- Compute norm inline to avoid recursion
    let wNorm := Tensor.l2Norm m.weight
    let bNorm := Tensor.l2Norm m.bias
    let n := Float.sqrt (wNorm * wNorm + bNorm * bNorm)
    if n == 0.0 then m else
    { m with
      weight := Tensor.Preserving.mulScalar m.weight (1.0 / n)
      bias := Tensor.Preserving.mulScalar m.bias (1.0 / n) }

  normalizeDual m :=
    -- Compute dual norm inline
    let wNorm := Tensor.l2Norm m.weight
    let bNorm := Tensor.l2Norm m.bias
    let n := Float.sqrt (wNorm * wNorm + bNorm * bNorm)
    if n == 0.0 then m else
    { m with
      weight := Tensor.Preserving.mulScalar m.weight (1.0 / n)
      bias := Tensor.Preserving.mulScalar m.bias (1.0 / n) }

/-! ## Embedding Layer

For an embedding table E ∈ ℝ^{V×d}, the modular norm is the max row norm:
  ||E||_M = max_i ||e_i||₂

This ensures that no single embedding vector dominates.
The dual norm is the sum of row norms (ℓ₁ of row ℓ₂ norms).
-/

/-- Embedding table: maps discrete tokens to continuous vectors -/
structure Embedding (vocabSize dim : UInt64) (dtype : torch.DType := .Float32)
    (device : torch.Device := .CPU) where
  weight : torch.Tensor ⟨#[vocabSize, dim], dtype, device⟩

variable {dtype : torch.DType} {device : torch.Device}

namespace Embedding

instance : torch.TensorStruct (Embedding vocabSize dim dtype device) where
  map f e := { weight := torch.TensorStruct.map f e.weight }
  mapM f e := do
    let w' ← torch.TensorStruct.mapM f e.weight
    pure { weight := w' }
  zipWith f e1 e2 := { weight := torch.TensorStruct.zipWith f e1.weight e2.weight }
  fold f init e := torch.TensorStruct.fold f init e.weight

/-- Initialize with small random values -/
def init (vocabSize dim : UInt64) : IO (Embedding vocabSize dim) := do
  let std := 1.0 / Float.sqrt dim.toFloat
  let w ← torch.Tensor.randn #[vocabSize, dim]
  let weight := torch.Tensor.Preserving.mulScalar w std
  let weight ← torch.Tensor.parameter weight
  pure { weight }

/-- Lookup embeddings for a batch of token indices -/
def forward {vocabSize dim batch seq : UInt64}
    (emb : Embedding vocabSize dim dtype device)
    (indices : torch.Tensor ⟨#[batch, seq], .Int64, device⟩) : torch.Tensor ⟨#[batch, seq, dim], dtype, device⟩ :=
  torch.Tensor.embedding indices emb.weight

end Embedding

instance (vocabSize dim : UInt64) : NormedModule (Embedding vocabSize dim dtype device) where
  norm m := torch.Tensor.maxRowNorm m.weight

  dualNorm m :=
    -- Sum of row norms
    let norms := torch.Tensor.rowNorms m.weight
    torch.Tensor.item (torch.Tensor.sumAll norms)

  nu m := torch.Tensor.maxRowNorm m.weight

  mu _ := 1.0

  normalize m :=
    let n := torch.Tensor.maxRowNorm m.weight
    if n == 0.0 then m else
    { weight := torch.Tensor.Preserving.mulScalar m.weight (1.0 / n) }

  normalizeDual m :=
    -- Compute dual norm inline to avoid recursion
    let norms := torch.Tensor.rowNorms m.weight
    let n := torch.Tensor.item (torch.Tensor.sumAll norms)
    if n == 0.0 then m else
    { weight := torch.Tensor.Preserving.mulScalar m.weight (1.0 / n) }

end Tyr.Modular
