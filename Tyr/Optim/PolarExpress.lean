/-
  Tyr/Optim/PolarExpress.lean

  Polar Express orthogonalization for Muon/NorMuon optimizers.
  Implements the Newton-Schulz iteration for matrix sign approximation.

  Based on modded-nanogpt's Polar Express implementation:
  - Uses precomputed coefficients for numerical stability
  - Supports batched and unbatched matrices
  - Handles tall vs wide matrices differently
-/
import Tyr.Distributed

namespace torch.Optim.PolarExpress

open torch

/-- Configuration for Polar Express -/
structure Config where
  /-- Number of Newton-Schulz iterations (typically 5) -/
  numIters : Nat := 5
  /-- Whether to use split baddbmm for memory efficiency (advanced) -/
  splitBaddbmm : Bool := false
  deriving Repr, Inhabited

/-- Apply Polar Express orthogonalization to a gradient matrix.

    This approximates the matrix sign function via Newton-Schulz iterations,
    effectively orthogonalizing the gradient for more stable optimization.

    For a matrix G:
    1. Normalize: X = G / ||G||_F
    2. Iterate: X = a*X + b*X@X.T@X + c*X@X.T@X@X.T@X
    3. Return orthogonalized X

    Args:
    - G: Gradient matrix [batch?, m, n]
    - cfg: Configuration with numIters

    Returns: Orthogonalized gradient of the same spec
-/
def apply {σ : StaticSpec} [DType.RealFloating σ.dtype] (G : Tensor σ) (cfg : Config := {}) : IO (Tensor σ) := do
  dist.polarExpress G cfg.numIters

/-- Muon-style orthogonalization that handles tall vs wide matrices.

    For [out, in] gradient:
    - If out > in: orthogonalize along rows (G.T is used, then result transposed back)
    - If out <= in: orthogonalize directly

    This ensures the orthogonalization is always applied to the "short" dimension.
-/
def muonOrthogonalize {σ : StaticSpec} [DType.RealFloating σ.dtype] (G : Tensor σ) (numIters : Nat := 5) :
    IO (Tensor (σ.withDType .BFloat16)) := do
  dist.muonOrthogonalize G numIters

end torch.Optim.PolarExpress
