/-
  Tyr/Model/Flux/Config.lean

  Configuration for Flux diffusion model.
  Default values are for Klein 4B.
-/
import Tyr.Basic

namespace torch.flux

/-- Flux model configuration.
    Default values are for Klein 4B. -/
structure FluxConfig where
  /-- Hidden dimension -/
  hidden_size : Nat := 3072
  /-- Number of attention heads -/
  num_heads : Nat := 24
  /-- Head dimension -/
  head_dim : Nat := 128
  /-- Number of double-stream transformer blocks -/
  num_double_layers : Nat := 5
  /-- Number of single-stream transformer blocks -/
  num_single_layers : Nat := 20
  /-- Context (text) input dimension -/
  context_in_dim : Nat := 7680
  /-- Image input dimension (latent patch dim) -/
  in_channels : Nat := 128
  /-- Timestep embedding dimension -/
  time_dim : Nat := 256
  /-- MLP hidden dimension multiplier -/
  mlp_ratio : Float := 3.0
  /-- RoPE theta base -/
  theta : Nat := 2000
  /-- RoPE axes dimensions (for 4-axis RoPE) -/
  axes_dims : Array Nat := #[32, 32, 32, 32]
  deriving Repr, Inhabited

namespace FluxConfig

/-- Klein 4B configuration (Flux2 Klein uses mlp_ratio=3.0) -/
def klein4B : FluxConfig := {
  mlp_ratio := 3.0
}

/-- Compute MLP hidden dimension -/
def mlpHiddenDim (cfg : FluxConfig) : Nat :=
  (cfg.hidden_size.toFloat * cfg.mlp_ratio).toUInt64.toNat

/-- Total RoPE dimension (sum of axes) -/
def ropeDim (cfg : FluxConfig) : Nat :=
  cfg.axes_dims.foldl (· + ·) 0

end FluxConfig

end torch.flux
