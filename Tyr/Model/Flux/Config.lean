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
  /-- Half the head dimension; each head has `2 * half_head_dim` channels. -/
  half_head_dim : Nat := 64
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
  /-- MLP hidden dimension (`hidden_size * mlp_ratio`; Klein uses ratio 3) -/
  mlp_hidden_dim : Nat := 9216
  /-- RoPE theta base -/
  theta : Nat := 2000
  /-- Halves of the four RoPE axis dimensions -/
  half_axes_dims : Array Nat := #[16, 16, 16, 16]
  /-- Four RoPE axes whose halves sum to `half_head_dim`; checked when the config is built. -/
  axesValid : half_axes_dims.size = 4 ∧ half_axes_dims.toList.sum = half_head_dim := by decide
  /-- RoPE frequencies take `log theta`; checked when the config is built. -/
  thetaPos : 0 < theta := by decide
  deriving Repr


namespace FluxConfig

/-- Channels per attention head. -/
def head_dim (cfg : FluxConfig) : Nat := 2 * cfg.half_head_dim

/-- Klein 4B configuration (Flux2 Klein uses mlp_ratio=3.0) -/
def klein4B : FluxConfig := {}

/-- MLP hidden dimension -/
def mlpHiddenDim (cfg : FluxConfig) : Nat :=
  cfg.mlp_hidden_dim

end FluxConfig

end torch.flux
