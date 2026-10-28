/-
  Tyr/Model/Laguna/Config.lean

  Configuration for the poolside Laguna text causal-LM implementation
  (Laguna-S-2.1: 118B-A8B MoE, interleaved full/sliding-window attention,
  per-head softplus attention output gating, sigmoid token-choice router).
-/
import Tyr.Basic
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
  max_position_embeddings : Nat := 262144
  sliding_window : Nat := 512

  /-- RoPE base for sliding-window layers (plain rope, full rotary). -/
  rope_theta_sliding : Float := 10000.0
  partial_rotary_sliding : Float := 1.0
  /-- RoPE base for full-attention layers (YaRN, partial rotary). -/
  rope_theta_full : Float := 500000.0
  partial_rotary_full : Float := 0.5
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
  layer_types : Array LayerType := #[]

  attention_bias : Bool := false
  hidden_act : String := "silu"
  use_cache : Bool := true
  tie_word_embeddings : Bool := false

  pad_token_id : Option UInt64 := some 9
  bos_token_id : Option UInt64 := some 2
  eos_token_ids : Array UInt64 := #[2, 24]
  deriving Repr, Inhabited

namespace LagunaConfig

/-- Laguna-S-2.1 default config (NVFP4 checkpoint values). -/
def laguna_s_2_1 : LagunaConfig := {}

/-- Default layer schedule: full attention at layers 0, 4, 8, ... (i % 4 == 0),
    sliding-window attention elsewhere. -/
def defaultLayerTypes (numLayers : Nat) : Array LayerType :=
  Id.run do
    let mut out : Array LayerType := Array.mkEmpty numLayers
    for i in [:numLayers] do
      out := out.push (if i % 4 == 0 then .fullAttention else .slidingAttention)
    return out

/-- Return `layer_types` if valid, otherwise synthesize the default schedule. -/
def normalizedLayerTypes (cfg : LagunaConfig) : Array LayerType :=
  if cfg.layer_types.size == cfg.num_hidden_layers then
    cfg.layer_types
  else
    defaultLayerTypes cfg.num_hidden_layers

/-- Config with `layer_types` normalized for runtime use. -/
def normalize (cfg : LagunaConfig) : LagunaConfig :=
  { cfg with layer_types := normalizedLayerTypes cfg }

/-- Attention kind of layer `i`. -/
def layerType (cfg : LagunaConfig) (i : Nat) : LayerType :=
  let lts := normalizedLayerTypes cfg
  lts.getD i .slidingAttention

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
  cfg.num_experts > 0 && cfg.num_experts_per_tok > 0

/-- Queries per KV head on full-attention layers. -/
def numHeadsPerKVGroupFull (cfg : LagunaConfig) : Nat :=
  if cfg.num_key_value_heads == 0 then 1 else cfg.num_attention_heads / cfg.num_key_value_heads

/-- Queries per KV head on sliding-window layers. -/
def numHeadsPerKVGroupSliding (cfg : LagunaConfig) : Nat :=
  if cfg.num_key_value_heads == 0 then 1 else cfg.num_attention_heads_sliding / cfg.num_key_value_heads

private def rotaryDimFor (headDim : Nat) (factor : Float) : Nat :=
  let raw := (headDim.toFloat * factor).toUInt64.toNat
  let base :=
    if raw == 0 then headDim
    else if raw > headDim then headDim
    else raw
  if base <= 2 then 2 else if base % 2 == 0 then base else base - 1

/-- Rotary dims on full-attention layers (64 for partial_rotary 0.5, head_dim 128). -/
def rotaryDimFull (cfg : LagunaConfig) : Nat :=
  rotaryDimFor cfg.head_dim cfg.partial_rotary_full

/-- Rotary dims on sliding-window layers (128 for partial_rotary 1.0). -/
def rotaryDimSliding (cfg : LagunaConfig) : Nat :=
  rotaryDimFor cfg.head_dim cfg.partial_rotary_sliding

end LagunaConfig

abbrev Config := LagunaConfig

namespace Config

def laguna_s_2_1 : Config := LagunaConfig.laguna_s_2_1

def normalize (cfg : Config) : Config := LagunaConfig.normalize cfg

def normalizedLayerTypes (cfg : Config) : Array LayerType :=
  LagunaConfig.normalizedLayerTypes cfg

def layerType (cfg : Config) (i : Nat) : LayerType := LagunaConfig.layerType cfg i

def numHeadsForLayer (cfg : Config) (i : Nat) : Nat :=
  LagunaConfig.numHeadsForLayer cfg i

def isDenseMlpLayer (cfg : Config) (i : Nat) : Bool := LagunaConfig.isDenseMlpLayer cfg i

def isMoE (cfg : Config) : Bool := LagunaConfig.isMoE cfg

def numHeadsPerKVGroupFull (cfg : Config) : Nat := LagunaConfig.numHeadsPerKVGroupFull cfg

def numHeadsPerKVGroupSliding (cfg : Config) : Nat := LagunaConfig.numHeadsPerKVGroupSliding cfg

def rotaryDimFull (cfg : Config) : Nat := LagunaConfig.rotaryDimFull cfg

def rotaryDimSliding (cfg : Config) : Nat := LagunaConfig.rotaryDimSliding cfg

end Config

end torch.laguna
