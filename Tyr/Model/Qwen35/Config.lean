/-
  Tyr/Model/Qwen35/Config.lean

  Configuration for the standalone Qwen3.5 text causal-LM implementation.
  Includes dense and MoE variants plus hybrid layer scheduling.
-/
import Tyr.Basic
import Tyr.TensorStruct
import Tyr.Tokenizer.Types

namespace torch.qwen35

/-- Per-layer token-mixer kind used by Qwen3.5 hybrid blocks. -/
inductive LayerType where
  | linearAttention
  | fullAttention
  deriving Repr, Inhabited, BEq

namespace LayerType

/-- Parse a HuggingFace `layer_types` string entry. -/
def ofString? (s : String) : Option LayerType :=
  if s == "linear_attention" then
    some .linearAttention
  else if s == "full_attention" then
    some .fullAttention
  else
    none

/-- Serialize to HuggingFace-style string. -/
def toString : LayerType → String
  | .linearAttention => "linear_attention"
  | .fullAttention => "full_attention"

end LayerType

instance : TensorStruct LayerType where
  map _ x := x
  mapM _ x := pure x
  zipWith _ x _ := x
  fold _ init _ := init

/-- Hybrid schedule: every `interval`th layer is full attention, the rest linear attention. -/
def defaultLayerTypes (numLayers interval : Nat) : Array LayerType :=
  Array.ofFn (n := numLayers) fun i =>
    if (i.val + 1) % interval == 0 then .fullAttention else .linearAttention

@[simp] theorem defaultLayerTypes_size (numLayers interval : Nat) :
    (defaultLayerTypes numLayers interval).size = numLayers := by
  simp [defaultLayerTypes]

/-- Qwen3.5 text model configuration.
    Dense checkpoints use `num_experts = 0`.
    MoE checkpoints set `num_experts > 0`. -/
structure Qwen35Config where
  vocab_size : Nat := 248320
  hidden_size : Nat := 4096
  intermediate_size : Nat := 12288
  num_hidden_layers : Nat := 32
  num_attention_heads : Nat := 16
  num_key_value_heads : Nat := 4
  head_dim : Nat := 256

  rope_theta : Float := 1000000.0
  /-- Half the leading channels of each head that RoPE rotates (`head_dim * partial_rotary_factor / 2`). -/
  half_rotary_dim : Nat := 32
  rms_norm_eps : Float := 1e-6

  linear_conv_kernel_dim : Nat := 4
  linear_key_head_dim : Nat := 128
  linear_value_head_dim : Nat := 128
  linear_num_key_heads : Nat := 16
  linear_num_value_heads : Nat := 32

  layer_types : Array LayerType := defaultLayerTypes num_hidden_layers 4

  moe_intermediate_size : Nat := 512
  shared_expert_intermediate_size : Nat := 512
  num_experts_per_tok : Nat := 8
  num_experts : Nat := 0

  tie_word_embeddings : Bool := false
  eos_token_id : Option tokenizer.TokenId := none
  /-- Linear-attention value heads are a whole multiple of key heads; checked when the config is built. -/
  linearKeyHeadsDivide : linear_num_key_heads ∣ linear_num_value_heads := by decide
  /-- Query heads are a whole multiple of K/V heads; checked when the config is built. -/
  kvHeadsDivide : num_key_value_heads ∣ num_attention_heads := by decide
  headsPos : 0 < num_attention_heads ∧ 0 < num_key_value_heads ∧ 0 < linear_num_key_heads := by decide
  /-- RoPE rotates a nonzero even prefix of each head; checked when the config is built. -/
  rotaryValid : 0 < half_rotary_dim ∧ 2 * half_rotary_dim ≤ head_dim := by decide
  /-- One mixer kind per decoder layer; checked when the config is built. -/
  layerTypesSize : layer_types.size = num_hidden_layers := by simp
  /-- The causal depthwise conv has a kernel; checked when the config is built. -/
  convKernelPos : 0 < linear_conv_kernel_dim := by decide
  /-- The depthwise conv has one group per channel, so its channel count is positive; checked
      when the config is built. -/
  linearKeyHeadDimPos : 0 < linear_key_head_dim := by decide
  /-- MoE routes each token to between one and `num_experts` experts; checked when the config is built. -/
  expertsValid : 0 < num_experts → 0 < num_experts_per_tok ∧ num_experts_per_tok ≤ num_experts := by decide
  /-- RoPE frequencies and RMSNorm divide by these; checked when the config is built. -/
  ropeThetaPos : 0 < rope_theta := by native_decide
  rmsNormEpsPos : 0 < rms_norm_eps := by native_decide
  /-- Generation stops on and embeds the eos token; checked when the config is built. -/
  eosInVocab : eos_token_id.all (·.toNat < vocab_size) = true := by decide
  deriving Repr


namespace Qwen35Config

/-- Default dense Qwen3.5-9B style config. -/
def qwen35_9B : Qwen35Config :=
  { num_experts := 0 }

/-- Default MoE Qwen3.5-35B-A3B style config. -/
def qwen35_35B_A3B : Qwen35Config := {
  vocab_size := 248320
  hidden_size := 2048
  intermediate_size := 0
  num_hidden_layers := 40
  num_attention_heads := 16
  num_key_value_heads := 2
  head_dim := 256
  rope_theta := 1000000.0
  half_rotary_dim := 32
  rms_norm_eps := 1e-6
  linear_conv_kernel_dim := 4
  linear_key_head_dim := 128
  linear_value_head_dim := 128
  linear_num_key_heads := 16
  linear_num_value_heads := 32
  moe_intermediate_size := 512
  shared_expert_intermediate_size := 512
  num_experts_per_tok := 8
  num_experts := 256
  tie_word_embeddings := false
  eos_token_id := none
}

/-- Default multimodal Qwen3.6-35B-A3B text config from Hugging Face `config.json`. -/
def qwen36_35B_A3B : Qwen35Config := {
  vocab_size := 248320
  hidden_size := 2048
  intermediate_size := 0
  num_hidden_layers := 40
  num_attention_heads := 16
  num_key_value_heads := 2
  head_dim := 256
  rope_theta := 10000000.0
  half_rotary_dim := 32
  rms_norm_eps := 1e-6
  linear_conv_kernel_dim := 4
  linear_key_head_dim := 128
  linear_value_head_dim := 128
  linear_num_key_heads := 16
  linear_num_value_heads := 32
  moe_intermediate_size := 512
  shared_expert_intermediate_size := 512
  num_experts_per_tok := 8
  num_experts := 256
  tie_word_embeddings := false
  eos_token_id := some 248044
}

/-- Default dense Qwen3.5-0.8B style config. -/
def qwen35_0_8B : Qwen35Config := {
  vocab_size := 248320
  hidden_size := 1024
  intermediate_size := 3584
  num_hidden_layers := 24
  num_attention_heads := 8
  num_key_value_heads := 2
  head_dim := 256
  rope_theta := 10000000.0
  half_rotary_dim := 32
  rms_norm_eps := 1e-6
  linear_conv_kernel_dim := 4
  linear_key_head_dim := 128
  linear_value_head_dim := 128
  linear_num_key_heads := 16
  linear_num_value_heads := 16
  num_experts := 0
  tie_word_embeddings := true
  eos_token_id := some 248044
}

/-- Linear-attention key projection dimension. -/
@[simp] def linearKeyDim (cfg : Qwen35Config) : Nat :=
  cfg.linear_num_key_heads * cfg.linear_key_head_dim

/-- Linear-attention value projection dimension. -/
@[simp] def linearValueDim (cfg : Qwen35Config) : Nat :=
  cfg.linear_num_value_heads * cfg.linear_value_head_dim

/-- Depthwise-conv channel count for linear-attention fused QKV path. -/
@[simp] def linearConvDim (cfg : Qwen35Config) : Nat :=
  (linearKeyDim cfg) * 2 + (linearValueDim cfg)

theorem linearConvDim_pos (cfg : Qwen35Config) : 0 < linearConvDim cfg := by
  have := Nat.mul_pos cfg.headsPos.2.2 cfg.linearKeyHeadDimPos
  simp only [linearConvDim, linearKeyDim]; omega

/-- Repeat factor to broadcast linear K/Q heads to V heads. -/
def linearKVRepeat (cfg : Qwen35Config) : Nat :=
  if cfg.linear_num_key_heads == 0 then 1 else cfg.linear_num_value_heads / cfg.linear_num_key_heads

theorem linear_key_heads_mul_repeat (cfg : Qwen35Config) :
    cfg.linear_num_key_heads * linearKVRepeat cfg = cfg.linear_num_value_heads := by
  have h := cfg.linearKeyHeadsDivide
  unfold linearKVRepeat
  split
  · simp_all [Nat.eq_zero_of_zero_dvd]
  · exact Nat.mul_div_cancel' h

/-- Whether this config uses MoE FFN blocks. -/
def isMoE (cfg : Qwen35Config) : Bool :=
  cfg.num_experts > 0

/-- Rotary dimension for partial RoPE. -/
def rotaryDim (cfg : Qwen35Config) : Nat :=
  2 * cfg.half_rotary_dim

/-- Number of rotary frequencies. -/
def rotaryHalfDim (cfg : Qwen35Config) : Nat :=
  cfg.half_rotary_dim

end Qwen35Config

abbrev Config := Qwen35Config

namespace Config

def qwen35_0_8B : Config := Qwen35Config.qwen35_0_8B

def qwen35_9B : Config := Qwen35Config.qwen35_9B

def qwen35_35B_A3B : Config := Qwen35Config.qwen35_35B_A3B

def qwen36_35B_A3B : Config := Qwen35Config.qwen36_35B_A3B

def isMoE (cfg : Config) : Bool := Qwen35Config.isMoE cfg

/-- Whether layer `i` uses full attention (otherwise linear attention). -/
def isFullAttentionLayer (cfg : Config) (i : Nat) : Bool :=
  match cfg.layer_types.getD i .linearAttention with
  | .fullAttention => true
  | .linearAttention => false

def rotaryDim (cfg : Config) : Nat := Qwen35Config.rotaryDim cfg

def rotaryHalfDim (cfg : Config) : Nat := Qwen35Config.rotaryHalfDim cfg

theorem rotaryDim_le (cfg : Config) : 2 * cfg.rotaryHalfDim ≤ cfg.head_dim := cfg.rotaryValid.2

@[simp] def linearKeyDim (cfg : Config) : Nat := Qwen35Config.linearKeyDim cfg

@[simp] def linearValueDim (cfg : Config) : Nat := Qwen35Config.linearValueDim cfg

@[simp] def linearConvDim (cfg : Config) : Nat := Qwen35Config.linearConvDim cfg

def linearKVRepeat (cfg : Config) : Nat := Qwen35Config.linearKVRepeat cfg

theorem linear_key_heads_mul_repeat (cfg : Config) :
    cfg.linear_num_key_heads * linearKVRepeat cfg = cfg.linear_num_value_heads :=
  Qwen35Config.linear_key_heads_mul_repeat cfg

end Config

end torch.qwen35
