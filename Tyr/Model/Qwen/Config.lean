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
  /-- Head dimension (hidden_size / num_attention_heads) -/
  head_dim : Nat := 128
  /-- RoPE theta base -/
  rope_theta : Float := 10000.0
  /-- RMSNorm epsilon -/
  rms_norm_eps : Float := 1e-6
  /-- Maximum sequence length -/
  max_position_embeddings : Nat := 32768
  deriving Repr, Inhabited

namespace QwenConfig

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
  head_dim := 128            -- same
  rope_theta := 1000000.0    -- different base (10000 for standard)
  rms_norm_eps := 1e-6
  max_position_embeddings := 40960
}

/-- Compute the head dimension from hidden size and num heads -/
def computeHeadDim (cfg : QwenConfig) : Nat :=
  cfg.hidden_size / cfg.num_attention_heads

/-- Compute the KV dimension (num_kv_heads * head_dim) -/
def kvDim (cfg : QwenConfig) : Nat :=
  cfg.num_key_value_heads * cfg.head_dim

/-- Number of heads per KV group (for GQA) -/
def numHeadsPerKVGroup (cfg : QwenConfig) : Nat :=
  cfg.num_attention_heads / cfg.num_key_value_heads

end QwenConfig

end torch.qwen
