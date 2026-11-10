/-
  Tyr/Model/Gemma4/Config.lean

  Configuration for the standalone Gemma 4 text causal-LM implementation.
  Covers the public Gemma 4 family checkpoints routed through the text
  submodel of the official multimodal repos.
-/
import Tyr.Basic
import Tyr.TensorStruct
import Tyr.Tokenizer.Types

namespace torch.gemma4

/-- Per-layer attention schedule entry used by Gemma 4. -/
inductive LayerType where
  | slidingAttention
  | fullAttention
  deriving Repr, Inhabited, BEq, DecidableEq

namespace LayerType

def ofString? (s : String) : Option LayerType :=
  if s == "sliding_attention" then
    some .slidingAttention
  else if s == "full_attention" then
    some .fullAttention
  else
    none

def toString : LayerType → String
  | .slidingAttention => "sliding_attention"
  | .fullAttention => "full_attention"

end LayerType

instance : TensorStruct LayerType where
  map _ x := x
  mapM _ x := pure x
  zipWith _ x _ := x
  fold _ init _ := init

/-- Hybrid schedule: every `interval`th layer is full attention, and so is the last layer. -/
def defaultLayerTypes (numLayers interval : Nat) : Array LayerType :=
  Array.ofFn (n := numLayers) fun i =>
    if (i.val + 1) % interval == 0 || i.val + 1 == numLayers then .fullAttention else .slidingAttention

@[simp] theorem defaultLayerTypes_size (numLayers interval : Nat) :
    (defaultLayerTypes numLayers interval).size = numLayers := by
  simp [defaultLayerTypes]

/-- The last `shared` of `numLayers` layers reuse K/V from the last earlier layer of their
attention kind; every such layer needs one. -/
def kvSharingValid (types : Array LayerType) (numLayers shared : Nat) : Bool :=
  shared == 0 || (shared < numLayers && (List.range shared).all fun j =>
    (List.range (numLayers - shared)).any fun s => types[s]? == types[numLayers - shared + j]?)

/-- A full-attention layer size: the sliding-layer size, or a size of its own. -/
inductive FullSize where
  | same
  | distinct (size : PosNat)
  deriving Repr, Inhabited, BEq, DecidableEq

/-- The size given the sliding-layer size. -/
def FullSize.value (sliding : Nat) : FullSize → Nat
  | .same => sliding
  | .distinct size => size.val

/-- Gemma 4 text model configuration. -/
structure Gemma4Config where
  vocab_size : Nat := 262144
  hidden_size : Nat := 2560
  intermediate_size : Nat := 10240
  num_hidden_layers : Nat := 42
  num_attention_heads : Nat := 8
  num_key_value_heads : Nat := 2
  /-- K/V heads of full-attention layers. -/
  num_global_key_value_heads : FullSize := .same
  /-- Half the sliding-layer head dimension; those heads have `2 * half_head_dim` channels. -/
  half_head_dim : Nat := 128
  /-- Half the full-layer head dimension. -/
  global_half_head_dim : FullSize := .distinct 256

  sliding_window : Nat := 512
  sliding_rope_theta : Float := 10000.0
  full_rope_theta : Float := 1000000.0
  /-- Half the leading channels of each full-attention head that RoPE rotates
      (`full head dim * partial_rotary_factor / 2`). -/
  half_full_rotary_dim : Nat := 64
  rms_norm_eps : Float := 1e-6

  hidden_activation : String := "gelu_pytorch_tanh"
  attention_k_eq_v : Bool := false
  use_bidirectional_attention : String := ""

  layer_types : Array LayerType := defaultLayerTypes num_hidden_layers 6
  num_kv_shared_layers : Nat := 0
  use_double_wide_mlp : Bool := false

  hidden_size_per_layer_input : Nat := 0
  vocab_size_per_layer_input : Nat := 262144

  enable_moe_block : Bool := false
  num_experts : Nat := 0
  top_k_experts : Nat := 0
  moe_intermediate_size : Nat := 0

  tie_word_embeddings : Bool := true
  /-- `none` disables the final `cap * tanh (logits / cap)`. -/
  final_logit_softcapping : Option Float := some 30.0

  pad_token_id : Option tokenizer.TokenId := some 0
  eos_token_id : Array tokenizer.TokenId := #[1]

  /-- Query heads are whole multiples of the sliding-layer K/V heads; checked when the config is built. -/
  slidingKVHeadsDivide : num_key_value_heads ∣ num_attention_heads := by decide
  /-- Query heads are whole multiples of the full-layer K/V heads (`fullNumKVHeads`); checked when the config is built. -/
  fullKVHeadsDivide :
    num_global_key_value_heads.value num_key_value_heads ∣ num_attention_heads := by decide
  headsPos : 0 < num_attention_heads ∧ 0 < num_key_value_heads := by decide
  /-- Sliding-layer heads are nonempty; checked when the config is built. -/
  headDimPos : 0 < half_head_dim := by decide
  /-- Full layers rotate a nonzero prefix of each head; checked when the config is built. -/
  fullRotaryValid : 0 < half_full_rotary_dim ∧
    half_full_rotary_dim ≤ global_half_head_dim.value half_head_dim := by decide
  /-- Sliding layers attend over a nonempty window; checked when the config is built. -/
  slidingWindowPos : 0 < sliding_window := by decide
  /-- One attention kind per layer, the last one full; checked when the config is built. -/
  layerTypesValid : layer_types.size = num_hidden_layers ∧
    (num_hidden_layers = 0 ∨ layer_types.back? = some .fullAttention) := by simp [defaultLayerTypes]
  /-- KV-shared layers have earlier sources; checked when the config is built. -/
  kvSharing : kvSharingValid layer_types num_hidden_layers num_kv_shared_layers = true := by native_decide
  /-- The router picks `top_k_experts` of `num_experts`, and an enabled MoE block routes to a
      nonzero expert MLP; checked when the config is built. -/
  topKValid : top_k_experts ≤ num_experts := by decide
  moeValid : enable_moe_block = true → 0 < top_k_experts ∧ 0 < moe_intermediate_size := by decide
  /-- Per-layer inputs embed the same ids as the main embedding; checked when the config is built. -/
  perLayerVocab : 0 < hidden_size_per_layer_input → vocab_size ≤ vocab_size_per_layer_input := by decide
  /-- Only vision tokens may attend bidirectionally; checked when the config is built. -/
  bidirectionalSupported : use_bidirectional_attention = "" ∨ use_bidirectional_attention = "vision" := by decide
  /-- RoPE frequencies, RMSNorm and the logit softcap divide by these; checked when the config is built. -/
  floatsPos : 0 < sliding_rope_theta ∧ 0 < full_rope_theta ∧ 0 < rms_norm_eps := by native_decide
  softcapPos : final_logit_softcapping.all (0 < ·) = true := by native_decide
  /-- Pad and eos ids index the vocabulary; checked when the config is built. -/
  tokenIdsInVocab : pad_token_id.all (·.toNat < vocab_size) = true ∧
    eos_token_id.all (·.toNat < vocab_size) = true := by native_decide
  deriving Repr

namespace Gemma4Config

/-- Default Gemma 4 E4B-style config. -/
def gemma4_E4B : Gemma4Config :=
  {
    vocab_size := 262144
    hidden_size := 2560
    intermediate_size := 10240
    num_hidden_layers := 42
    num_attention_heads := 8
    num_key_value_heads := 2
    num_global_key_value_heads := .same
    half_head_dim := 128
    global_half_head_dim := .distinct 256
    sliding_window := 512
    sliding_rope_theta := 10000.0
    full_rope_theta := 1000000.0
    rms_norm_eps := 1e-6
    hidden_activation := "gelu_pytorch_tanh"
    attention_k_eq_v := false
    use_bidirectional_attention := ""
    layer_types := #[
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention
    ]
    num_kv_shared_layers := 18
    use_double_wide_mlp := false
    hidden_size_per_layer_input := 256
    vocab_size_per_layer_input := 262144
    enable_moe_block := false
    num_experts := 0
    top_k_experts := 0
    moe_intermediate_size := 0
    tie_word_embeddings := true
    final_logit_softcapping := some 30.0
    pad_token_id := some 0
    eos_token_id := #[1]
  }

/-- Default Gemma 4 E2B-style config. -/
def gemma4_E2B : Gemma4Config :=
  {
    gemma4_E4B with
    hidden_size := 1536
    intermediate_size := 6144
    num_hidden_layers := 35
    num_key_value_heads := 1
    layer_types := #[
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention
    ]
    num_kv_shared_layers := 20
    use_double_wide_mlp := true
    slidingKVHeadsDivide := by decide
    fullKVHeadsDivide := by decide
    layerTypesValid := by decide
    kvSharing := by native_decide
    headsPos := by decide
    topKValid := by decide
    moeValid := by decide
  }

/-- Default Gemma 4 26B-A4B-style config. -/
def gemma4_26B_A4B : Gemma4Config :=
  {
    vocab_size := 262144
    hidden_size := 2816
    intermediate_size := 2112
    num_hidden_layers := 30
    num_attention_heads := 16
    num_key_value_heads := 8
    num_global_key_value_heads := .distinct 2
    half_head_dim := 128
    global_half_head_dim := .distinct 256
    sliding_window := 1024
    sliding_rope_theta := 10000.0
    full_rope_theta := 1000000.0
    rms_norm_eps := 1e-6
    hidden_activation := "gelu_pytorch_tanh"
    attention_k_eq_v := true
    use_bidirectional_attention := "vision"
    layer_types := #[
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention
    ]
    num_kv_shared_layers := 0
    use_double_wide_mlp := false
    hidden_size_per_layer_input := 0
    vocab_size_per_layer_input := 262144
    enable_moe_block := true
    num_experts := 128
    top_k_experts := 8
    moe_intermediate_size := 704
    tie_word_embeddings := true
    final_logit_softcapping := some 30.0
    pad_token_id := some 0
    eos_token_id := #[1]
  }

/-- Default Gemma 4 31B-style config. -/
def gemma4_31B : Gemma4Config :=
  {
    gemma4_26B_A4B with
    hidden_size := 5376
    intermediate_size := 21504
    num_hidden_layers := 60
    num_attention_heads := 32
    num_key_value_heads := 16
    num_global_key_value_heads := .distinct 4
    enable_moe_block := false
    num_experts := 0
    top_k_experts := 0
    moe_intermediate_size := 0
    layer_types := #[
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention
    ]
    slidingKVHeadsDivide := by decide
    fullKVHeadsDivide := by decide
    layerTypesValid := by decide
    kvSharing := by native_decide
    headsPos := by decide
    topKValid := by decide
    moeValid := by decide
  }

/-- Channels per sliding-layer head. -/
def head_dim (cfg : Gemma4Config) : Nat := 2 * cfg.half_head_dim

def fullHalfHeadDim (cfg : Gemma4Config) : Nat :=
  cfg.global_half_head_dim.value cfg.half_head_dim

def fullHeadDim (cfg : Gemma4Config) : Nat :=
  2 * fullHalfHeadDim cfg

def maxHeadDim (cfg : Gemma4Config) : Nat :=
  if cfg.head_dim >= fullHeadDim cfg then cfg.head_dim else fullHeadDim cfg

def fullNumKVHeads (cfg : Gemma4Config) : Nat :=
  cfg.num_global_key_value_heads.value cfg.num_key_value_heads

def maxKVHeads (cfg : Gemma4Config) : Nat :=
  if cfg.num_key_value_heads >= fullNumKVHeads cfg then cfg.num_key_value_heads else fullNumKVHeads cfg

def maxAttentionDim (cfg : Gemma4Config) : Nat :=
  cfg.num_attention_heads * maxHeadDim cfg

def maxKVProjDim (cfg : Gemma4Config) : Nat :=
  maxKVHeads cfg * maxHeadDim cfg

def fullRotaryDim (cfg : Gemma4Config) : Nat :=
  2 * cfg.half_full_rotary_dim

def fullRotaryHalfDim (cfg : Gemma4Config) : Nat :=
  cfg.half_full_rotary_dim

def maxIntermediateSize (cfg : Gemma4Config) : Nat :=
  if cfg.use_double_wide_mlp then
    2 * cfg.intermediate_size
  else
    cfg.intermediate_size

def layerTypeAt (cfg : Gemma4Config) (layerIdx : Nat) : LayerType :=
  cfg.layer_types.getD layerIdx .fullAttention

def firstKVSharedLayer (cfg : Gemma4Config) : Nat :=
  if cfg.num_kv_shared_layers == 0 || cfg.num_kv_shared_layers >= cfg.num_hidden_layers then
    0
  else
    cfg.num_hidden_layers - cfg.num_kv_shared_layers

def isKVSharedLayer (cfg : Gemma4Config) (layerIdx : Nat) : Bool :=
  let first := firstKVSharedLayer cfg
  cfg.num_kv_shared_layers > 0 && first > 0 && layerIdx >= first

def layerIntermediateSize (cfg : Gemma4Config) (layerIdx : Nat) : Nat :=
  if cfg.use_double_wide_mlp && isKVSharedLayer cfg layerIdx then
    2 * cfg.intermediate_size
  else
    cfg.intermediate_size

theorem firstKVSharedLayer_le (cfg : Gemma4Config) : firstKVSharedLayer cfg ≤ cfg.num_hidden_layers := by
  unfold firstKVSharedLayer; split <;> omega

def sharedSourceLayer? (cfg : Gemma4Config) (layerIdx : Nat) : Option (Fin cfg.num_hidden_layers) :=
  if !(isKVSharedLayer cfg layerIdx) then
    none
  else
    Id.run do
      let target := layerTypeAt cfg layerIdx
      let mut src : Option (Fin cfg.num_hidden_layers) := none
      for h : i in [:firstKVSharedLayer cfg] do
        if layerTypeAt cfg i == target then
          src := some ⟨i, Nat.lt_of_lt_of_le h.upper (firstKVSharedLayer_le cfg)⟩
      return src

def isMoE (cfg : Gemma4Config) : Bool :=
  cfg.enable_moe_block

def hasPerLayerInput (cfg : Gemma4Config) : Bool :=
  cfg.hidden_size_per_layer_input > 0

end Gemma4Config

abbrev Config := Gemma4Config

namespace Config

def gemma4_E2B : Config := Gemma4Config.gemma4_E2B
def gemma4_E4B : Config := Gemma4Config.gemma4_E4B
def gemma4_26B_A4B : Config := Gemma4Config.gemma4_26B_A4B
def gemma4_31B : Config := Gemma4Config.gemma4_31B

def fullHalfHeadDim (cfg : Config) : Nat := Gemma4Config.fullHalfHeadDim cfg
def fullHeadDim (cfg : Config) : Nat := Gemma4Config.fullHeadDim cfg
def maxHeadDim (cfg : Config) : Nat := Gemma4Config.maxHeadDim cfg
def fullNumKVHeads (cfg : Config) : Nat := Gemma4Config.fullNumKVHeads cfg
def maxKVHeads (cfg : Config) : Nat := Gemma4Config.maxKVHeads cfg

theorem kvHeads_le_max (cfg : Config) : cfg.num_key_value_heads ≤ maxKVHeads cfg := by
  show _ ≤ Gemma4Config.maxKVHeads cfg; unfold Gemma4Config.maxKVHeads; split <;> omega

theorem fullKVHeads_le_max (cfg : Config) : fullNumKVHeads cfg ≤ maxKVHeads cfg := by
  show Gemma4Config.fullNumKVHeads cfg ≤ Gemma4Config.maxKVHeads cfg
  unfold Gemma4Config.maxKVHeads; split <;> omega

theorem headDim_le_max (cfg : Config) : cfg.head_dim ≤ maxHeadDim cfg := by
  show _ ≤ Gemma4Config.maxHeadDim cfg; unfold Gemma4Config.maxHeadDim; split <;> omega

theorem fullHeadDim_le_max (cfg : Config) : fullHeadDim cfg ≤ maxHeadDim cfg := by
  show Gemma4Config.fullHeadDim cfg ≤ Gemma4Config.maxHeadDim cfg
  unfold Gemma4Config.maxHeadDim; split <;> omega
def maxAttentionDim (cfg : Config) : Nat := Gemma4Config.maxAttentionDim cfg
def maxKVProjDim (cfg : Config) : Nat := Gemma4Config.maxKVProjDim cfg

theorem qSliding_le (cfg : Config) : cfg.num_attention_heads * cfg.head_dim ≤ maxAttentionDim cfg :=
  Nat.mul_le_mul_left _ cfg.headDim_le_max

theorem qFull_le (cfg : Config) : cfg.num_attention_heads * fullHeadDim cfg ≤ maxAttentionDim cfg :=
  Nat.mul_le_mul_left _ cfg.fullHeadDim_le_max

theorem kvSliding_le (cfg : Config) : cfg.num_key_value_heads * cfg.head_dim ≤ maxKVProjDim cfg :=
  Nat.mul_le_mul cfg.kvHeads_le_max cfg.headDim_le_max

theorem kvFull_le (cfg : Config) : fullNumKVHeads cfg * fullHeadDim cfg ≤ maxKVProjDim cfg :=
  Nat.mul_le_mul cfg.fullKVHeads_le_max cfg.fullHeadDim_le_max
def fullRotaryDim (cfg : Config) : Nat := Gemma4Config.fullRotaryDim cfg
def fullRotaryHalfDim (cfg : Config) : Nat := Gemma4Config.fullRotaryHalfDim cfg
def maxIntermediateSize (cfg : Config) : Nat := Gemma4Config.maxIntermediateSize cfg
def layerTypeAt (cfg : Config) (layerIdx : Nat) : LayerType := Gemma4Config.layerTypeAt cfg layerIdx
def firstKVSharedLayer (cfg : Config) : Nat := Gemma4Config.firstKVSharedLayer cfg
def isKVSharedLayer (cfg : Config) (layerIdx : Nat) : Bool := Gemma4Config.isKVSharedLayer cfg layerIdx
def layerIntermediateSize (cfg : Config) (layerIdx : Nat) : Nat := Gemma4Config.layerIntermediateSize cfg layerIdx
def sharedSourceLayer? (cfg : Config) (layerIdx : Nat) : Option (Fin cfg.num_hidden_layers) :=
  Gemma4Config.sharedSourceLayer? cfg layerIdx
def isMoE (cfg : Config) : Bool := Gemma4Config.isMoE cfg
def hasPerLayerInput (cfg : Config) : Bool := Gemma4Config.hasPerLayerInput cfg

end Config

end torch.gemma4
