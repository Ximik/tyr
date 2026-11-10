/-
  Tyr/Model/Laguna/Config.lean

  Configuration for the poolside Laguna text causal-LM implementation
  (Laguna-S-2.1: 118B-A8B MoE, interleaved full/sliding-window attention,
  per-head softplus attention output gating, sigmoid token-choice router).
-/
import Tyr.Basic
import Tyr.Tokenizer.Types
import Tyr.TensorStruct

namespace torch.laguna

/-- Per-layer attention kind used by Laguna hybrid blocks. -/
inductive LayerType where
  | fullAttention
  | slidingAttention
  deriving Repr, Inhabited, BEq

namespace LayerType

/-- Parse a HuggingFace `layer_types` string entry. -/
def ofString? (s : String) : Option LayerType :=
  if s == "full_attention" then
    some .fullAttention
  else if s == "sliding_attention" then
    some .slidingAttention
  else
    none

/-- Serialize to HuggingFace-style string. -/
def toString : LayerType → String
  | .fullAttention => "full_attention"
  | .slidingAttention => "sliding_attention"

end LayerType

instance : TensorStruct LayerType where
  map _ x := x
  mapM _ x := pure x
  zipWith _ x _ := x
  fold _ init _ := init

/-- Default layer schedule: full attention at layers 0, 4, 8, ... (i % 4 == 0),
    sliding-window attention elsewhere. -/
def defaultLayerTypes (numLayers : Nat) : Array LayerType :=
  Array.ofFn (n := numLayers) fun i => if i.val % 4 == 0 then .fullAttention else .slidingAttention

@[simp] theorem defaultLayerTypes_size (numLayers : Nat) : (defaultLayerTypes numLayers).size = numLayers := by
  simp [defaultLayerTypes]

/-- Laguna model configuration.
    Defaults match `poolside/Laguna-S-2.1-NVFP4` (118B-A8B). -/
structure LagunaConfig where
  vocab_size : Nat := 100352
  hidden_size : Nat := 3072
  intermediate_size : Nat := 12288
  num_hidden_layers : Nat := 48
  /-- Query heads on full-attention layers. -/
  num_attention_heads : Nat := 48
  /-- Query heads on sliding-window layers. -/
  num_attention_heads_sliding : Nat := 72
  num_key_value_heads : Nat := 8
  head_dim : Nat := 128

  rms_norm_eps : Float := 1e-6
  sliding_window : Nat := 512

  /-- RoPE base for sliding-window layers (plain rope). -/
  rope_theta_sliding : Float := 10000.0
  /-- Half the leading head channels RoPE rotates on sliding-window layers
      (`head_dim * partial_rotary_factor / 2`). -/
  half_rotary_dim_sliding : Nat := 64
  /-- RoPE base for full-attention layers (YaRN, partial rotary). -/
  rope_theta_full : Float := 500000.0
  /-- Half the leading head channels RoPE rotates on full-attention layers. -/
  half_rotary_dim_full : Nat := 32
  yarn_factor : Float := 32.0
  yarn_original_max_position_embeddings : Nat := 8192
  yarn_beta_fast : Float := 32.0
  yarn_beta_slow : Float := 1.0
  yarn_attention_factor : Float := 1.3465735902799727

  num_experts : Nat := 256
  num_experts_per_tok : Nat := 10
  moe_intermediate_size : Nat := 1024
  shared_expert_intermediate_size : Nat := 1024
  norm_topk_prob : Bool := true
  moe_routed_scaling_factor : Float := 2.5
  moe_router_logit_softcapping : Float := 0.0

  /-- Layers that use a dense MLP instead of the MoE block. -/
  mlp_only_layers : Array Nat := #[0]
  layer_types : Array LayerType := defaultLayerTypes num_hidden_layers

  eos_token_ids : Array tokenizer.TokenId := #[2, 24]

  /-- Sliding layers keep a ring cache of `sliding_window` tokens; checked when the config is built. -/
  slidingWindowPos : 0 < sliding_window := by decide
  /-- Query heads are whole multiples of K/V heads on both layer kinds; checked when the config is built. -/
  kvHeadsDivideFull : num_key_value_heads ∣ num_attention_heads := by decide
  kvHeadsDivideSliding : num_key_value_heads ∣ num_attention_heads_sliding := by decide
  headsPos : 0 < num_attention_heads ∧ 0 < num_attention_heads_sliding ∧ 0 < num_key_value_heads := by decide
  /-- Both layer kinds rotate a nonzero prefix of each head; checked when the config is built. -/
  rotaryValid : 0 < half_rotary_dim_full ∧ 2 * half_rotary_dim_full ≤ head_dim ∧
    0 < half_rotary_dim_sliding ∧ 2 * half_rotary_dim_sliding ≤ head_dim := by decide
  /-- One attention kind per layer; checked when the config is built. -/
  layerTypesSize : layer_types.size = num_hidden_layers := by simp
  /-- Dense-MLP layer indices name decoder layers; checked when the config is built. -/
  mlpOnlyLayersValid : ∀ i ∈ mlp_only_layers, i < num_hidden_layers := by decide
  /-- MoE routes each token to between one and `num_experts` experts; checked when the config is built. -/
  expertsValid : 0 < num_experts → 0 < num_experts_per_tok ∧ num_experts_per_tok ≤ num_experts := by decide
  /-- YaRN rescales from a nonzero original context; checked when the config is built. -/
  yarnContextPos : 0 < yarn_original_max_position_embeddings := by decide
  /-- RoPE frequencies, YaRN and RMSNorm divide by or take logs of these; checked when the
      config is built. -/
  floatsPos : 0 < rms_norm_eps ∧ 0 < rope_theta_sliding ∧ 0 < rope_theta_full ∧ 0 < yarn_factor := by native_decide
  /-- Generation pads finished rows with, and embeds, eos ids; checked when the config is built. -/
  eosInVocab : eos_token_ids.all (·.toNat < vocab_size) = true := by native_decide
  deriving Repr

namespace LagunaConfig

/-- Laguna-S-2.1 default config (NVFP4 checkpoint values). -/
def laguna_s_2_1 : LagunaConfig := {}

/-- Attention kind of layer `i`. -/
def layerType (cfg : LagunaConfig) (i : Nat) : LayerType :=
  cfg.layer_types.getD i .slidingAttention

/-- Query head count of layer `i` (48 full / 72 sliding for S-2.1). -/
def numHeadsForLayer (cfg : LagunaConfig) (i : Nat) : Nat :=
  match layerType cfg i with
  | .fullAttention => cfg.num_attention_heads
  | .slidingAttention => cfg.num_attention_heads_sliding

/-- Whether layer `i` uses the dense MLP (`mlp_only_layers`) instead of MoE. -/
def isDenseMlpLayer (cfg : LagunaConfig) (i : Nat) : Bool :=
  cfg.mlp_only_layers.any (· == i)

/-- Whether this config uses MoE FFN blocks at all. -/
def isMoE (cfg : LagunaConfig) : Bool :=
  cfg.num_experts > 0

/-- Queries per KV head on full-attention layers. -/
def numHeadsPerKVGroupFull (cfg : LagunaConfig) : Nat :=
  cfg.num_attention_heads / cfg.num_key_value_heads

/-- Queries per KV head on sliding-window layers. -/
def numHeadsPerKVGroupSliding (cfg : LagunaConfig) : Nat :=
  cfg.num_attention_heads_sliding / cfg.num_key_value_heads

/-- Rotary dims on full-attention layers (64 for partial_rotary 0.5, head_dim 128). -/
def rotaryDimFull (cfg : LagunaConfig) : Nat :=
  2 * cfg.half_rotary_dim_full

/-- Rotary dims on sliding-window layers (128 for partial_rotary 1.0). -/
def rotaryDimSliding (cfg : LagunaConfig) : Nat :=
  2 * cfg.half_rotary_dim_sliding

end LagunaConfig

abbrev Config := LagunaConfig

namespace Config

def laguna_s_2_1 : Config := LagunaConfig.laguna_s_2_1

def layerType (cfg : Config) (i : Nat) : LayerType := LagunaConfig.layerType cfg i

def numHeadsForLayer (cfg : Config) (i : Nat) : Nat :=
  LagunaConfig.numHeadsForLayer cfg i

def isDenseMlpLayer (cfg : Config) (i : Nat) : Bool := LagunaConfig.isDenseMlpLayer cfg i

def isMoE (cfg : Config) : Bool := LagunaConfig.isMoE cfg

def numHeadsPerKVGroupFull (cfg : Config) : Nat := LagunaConfig.numHeadsPerKVGroupFull cfg

def numHeadsPerKVGroupSliding (cfg : Config) : Nat := LagunaConfig.numHeadsPerKVGroupSliding cfg

def rotaryDimFull (cfg : Config) : Nat := LagunaConfig.rotaryDimFull cfg

def rotaryDimSliding (cfg : Config) : Nat := LagunaConfig.rotaryDimSliding cfg

theorem rotaryDimFull_le (cfg : Config) : 2 * cfg.half_rotary_dim_full ≤ cfg.head_dim := cfg.rotaryValid.2.1

theorem rotaryDimSliding_le (cfg : Config) : 2 * cfg.half_rotary_dim_sliding ≤ cfg.head_dim := cfg.rotaryValid.2.2.2

/-- Whether layer `i` uses full attention (otherwise sliding-window attention). -/
def isFullAttentionLayer (cfg : Config) (i : Nat) : Bool :=
  match cfg.layerType i with
  | .fullAttention => true
  | .slidingAttention => false

/-- Whether layer `i` has a sparse MoE block (otherwise a dense MLP). -/
def isMoELayer (cfg : Config) (i : Nat) : Bool :=
  cfg.isMoE && !cfg.isDenseMlpLayer i

end Config

end torch.laguna
