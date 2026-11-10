/-
  Tyr/Model/Qwen/Config.lean

  Configuration for Qwen3 model architecture.
  Qwen3-4B is used as the text encoder for FLUX.2 Klein.
-/
import Tyr.Basic

namespace torch.qwen

/-- Qwen3 model configuration.
    Default values are for Qwen3-4B. -/
structure QwenConfig where
  /-- Vocabulary size -/
  vocab_size : Nat := 151936
  /-- Hidden dimension -/
  hidden_size : Nat := 2560
  /-- Intermediate (FFN) dimension -/
  intermediate_size : Nat := 6912
  /-- Number of transformer layers -/
  num_hidden_layers : Nat := 28
  /-- Number of attention heads -/
  num_attention_heads : Nat := 20
  /-- Number of key-value heads (for GQA) -/
  num_key_value_heads : Nat := 4
  /-- Half the head dimension; each head has `2 * half_head_dim` channels. -/
  half_head_dim : Nat := 64
  /-- RoPE theta base -/
  rope_theta : Float := 10000.0
  /-- RMSNorm epsilon -/
  rms_norm_eps : Float := 1e-6
  /-- Query heads form whole groups over a nonzero number of K/V heads; checked when the config is built. -/
  headsValid : 0 < num_attention_heads ∧ 0 < num_key_value_heads ∧
      num_key_value_heads ∣ num_attention_heads := by decide
  /-- Heads are nonempty; checked when the config is built. -/
  halfHeadDimPos : 0 < half_head_dim := by decide
  /-- Greedy decoding picks among `vocab_size` logits; checked when the config is built. -/
  vocabPos : 0 < vocab_size := by decide
  /-- RoPE frequencies and RMSNorm divide by these; checked when the config is built. -/
  ropeThetaPos : 0 < rope_theta := by native_decide
  rmsNormEpsPos : 0 < rms_norm_eps := by native_decide
  deriving Repr

namespace QwenConfig

/-- Channels per attention head. -/
def head_dim (cfg : QwenConfig) : Nat := 2 * cfg.half_head_dim

/-- Default Qwen3-4B configuration -/
def qwen3_4B : QwenConfig := {}

/-- Flux Klein text encoder configuration.
    This is based on Qwen3 but with different architecture parameters.
    Notable: 36 layers, 32 attention heads, 8 KV heads, larger intermediate size.
    Also uses Q/K norms in attention layers. -/
def fluxKleinTextEncoder : QwenConfig := {
  vocab_size := 151936
  hidden_size := 2560
  intermediate_size := 9728  -- larger than standard Qwen3-4B
  num_hidden_layers := 36    -- more layers than Qwen3-4B (28)
  num_attention_heads := 32  -- more heads than Qwen3-4B (20)
  num_key_value_heads := 8   -- more KV heads than Qwen3-4B (4)
  half_head_dim := 64        -- same
  rope_theta := 1000000.0    -- different base (10000 for standard)
  rms_norm_eps := 1e-6
}

end QwenConfig

end torch.qwen
