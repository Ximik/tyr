/-
  Tyr/Model/Qwen/Weights.lean

  Weight loading for Qwen3-4B from SafeTensors format.
  Maps HuggingFace weight names to Tyr structure.
  Supports both single-file and sharded HuggingFace models.

  Consumes the exact `Tyr/SafeTensors/Exact` API: each load retains the
  stored dtype in a dependent package. The caller explicitly converts its
  tensor to the Float32 structure fields. FP8 weights and block scales keep
  independent dtype indices until the existing Float32 dequantization step.
-/
import Tyr.Typed
import Tyr.Log
import Tyr.SafeTensors.Exact
import Tyr.Model.Utils
import Tyr.Model.Qwen.Config
import Tyr.Model.Qwen.Model
import Tyr.Model.Qwen.Embedder

namespace torch.qwen

variable {device : Device}
open torch.Log
open torch.Model (dequantizeFP8 checkFP8Blocks)
open torch.safetensors (loadTensorExact loadTensorShardedExact
  loadOptionalTensorExact loadOptionalTensorShardedExact)

/-- Load a required tensor with its exact stored dtype from a sharded checkpoint. -/
private def loadRequiredSharded (modelDir : String) (name : String) (s : Shape)
    (device : Device := .CPU) : IO (SomeDTypeTensor s device) :=
  loadTensorShardedExact modelDir name s device

/-- Load a required tensor with its exact stored dtype from a single file. -/
private def loadRequired (path : String) (name : String) (s : Shape)
    (device : Device := .CPU) : IO (SomeDTypeTensor s device) :=
  loadTensorExact path name s device

/-- Load a (potentially FP8-quantized) linear weight from sharded SafeTensors.
    If a matching weight_scale_inv exists, dequantize to float32;
    otherwise cast the loaded weights to float32. -/
private def loadLinearWeightSharded (modelDir : String) (name : String) (outDim inDim : Nat)
    (device : Device := .CPU) : IO (Tensor { device := device, shape := #[outDim, inDim], dtype := .Float32 }) := do
  let w ← loadRequiredSharded (device := device) modelDir s!"{name}.weight" #[outDim, inDim]
  let scaleInvOpt ← loadOptionalTensorShardedExact (device := device) modelDir s!"{name}.weight_scale_inv" #[outDim / 128, inDim / 128]
  match scaleInvOpt with
  | some s =>
    let ⟨hBlocks⟩ ← checkFP8Blocks s!"{name}.weight_scale_inv" outDim inDim
    pure (dequantizeFP8 w.2 s.2 hBlocks)
  | none => pure w.2.toFloat32

/-- Load a (potentially FP8-quantized) linear weight from a single SafeTensors file.
    If a matching weight_scale_inv exists, dequantize to float32;
    otherwise cast the loaded weights to float32. -/
private def loadLinearWeight (path : String) (name : String) (outDim inDim : Nat)
    (device : Device := .CPU) : IO (Tensor { device := device, shape := #[outDim, inDim], dtype := .Float32 }) := do
  let w ← loadRequired (device := device) path s!"{name}.weight" #[outDim, inDim]
  let scaleInv ← loadOptionalTensorExact (device := device) path s!"{name}.weight_scale_inv" #[outDim / 128, inDim / 128]
  match scaleInv with
  | some s =>
    let ⟨hBlocks⟩ ← checkFP8Blocks s!"{name}.weight_scale_inv" outDim inDim
    pure (dequantizeFP8 w.2 s.2 hBlocks)
  | none => pure w.2.toFloat32

/-- Load a single attention layer from sharded SafeTensors.
    Optionally loads Q/K norms if present (for Flux Klein text encoder). -/
def loadAttentionSharded (modelDir : String) (layerIdx : Nat)
    (hidden_size num_heads num_kv_heads head_dim : Nat)
    (loadQKNorms : Bool := false)
    (device : Device := .CPU) : IO (QwenAttention (device := device) hidden_size num_heads num_kv_heads head_dim) := do
  let layerPrefix := s!"model.layers.{layerIdx}.self_attn"

  -- Load separate projections
  let q_proj ← loadLinearWeightSharded (device := device) modelDir s!"{layerPrefix}.q_proj" (num_heads * head_dim) hidden_size
  let k_proj ← loadLinearWeightSharded (device := device) modelDir s!"{layerPrefix}.k_proj" (num_kv_heads * head_dim) hidden_size
  let v_proj ← loadLinearWeightSharded (device := device) modelDir s!"{layerPrefix}.v_proj" (num_kv_heads * head_dim) hidden_size
  let o_proj ← loadLinearWeightSharded (device := device) modelDir s!"{layerPrefix}.o_proj" hidden_size (num_heads * head_dim)

  -- Optionally load Q/K norms (Flux Klein text encoder has these)
  let q_norm ← if loadQKNorms then
    loadOptionalTensorShardedExact (device := device) modelDir s!"{layerPrefix}.q_norm.weight" #[head_dim]
  else
    pure none

  let k_norm ← if loadQKNorms then
    loadOptionalTensorShardedExact (device := device) modelDir s!"{layerPrefix}.k_norm.weight" #[head_dim]
  else
    pure none

  pure {
    q_proj := Tensor.detach q_proj
    k_proj := Tensor.detach k_proj
    v_proj := Tensor.detach v_proj
    o_proj := Tensor.detach o_proj
    q_norm := q_norm.map (fun t => Tensor.detach t.2.toFloat32)
    k_norm := k_norm.map (fun t => Tensor.detach t.2.toFloat32)
  }

/-- Load a single MLP layer from sharded SafeTensors -/
def loadMLPSharded (modelDir : String) (layerIdx : Nat)
    (hidden_size intermediate_size : Nat)
    (device : Device := .CPU) : IO (QwenMLP (device := device) hidden_size intermediate_size) := do
  let layerPrefix := s!"model.layers.{layerIdx}.mlp"

  let gate_proj ← loadLinearWeightSharded (device := device) modelDir s!"{layerPrefix}.gate_proj" intermediate_size hidden_size
  let up_proj ← loadLinearWeightSharded (device := device) modelDir s!"{layerPrefix}.up_proj" intermediate_size hidden_size
  let down_proj ← loadLinearWeightSharded (device := device) modelDir s!"{layerPrefix}.down_proj" hidden_size intermediate_size

  pure {
    gate_proj := Tensor.detach gate_proj
    up_proj := Tensor.detach up_proj
    down_proj := Tensor.detach down_proj
  }

/-- Load RMSNorm weights from sharded SafeTensors -/
def loadRMSNormSharded (modelDir : String) (name : String) (dim : Nat)
    (device : Device := .CPU) : IO (RMSNorm (device := device) dim) := do
  let scale ← loadRequiredSharded (device := device) modelDir s!"{name}.weight" #[dim]
  pure { weight := Tensor.detach scale.2.toFloat32, eps := ⟨1e-6⟩ }

/-- Load a single transformer layer from sharded SafeTensors -/
def loadLayerSharded (modelDir : String) (layerIdx : Nat) (cfg : QwenConfig)
    (loadQKNorms : Bool := false)
    (device : Device := .CPU) : IO (QwenLayer (device := device) cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim cfg.intermediate_size) := do
  let layerPrefix := s!"model.layers.{layerIdx}"

  let input_layernorm ← loadRMSNormSharded (device := device) modelDir s!"{layerPrefix}.input_layernorm" cfg.hidden_size
  let self_attn ← loadAttentionSharded (device := device) modelDir layerIdx cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim loadQKNorms
  let post_attention_layernorm ← loadRMSNormSharded (device := device) modelDir s!"{layerPrefix}.post_attention_layernorm" cfg.hidden_size
  let mlp ← loadMLPSharded (device := device) modelDir layerIdx cfg.hidden_size cfg.intermediate_size

  pure { input_layernorm, self_attn, post_attention_layernorm, mlp }

/-- Load full Qwen3 model from sharded SafeTensors directory.
    modelDir should contain model.safetensors.index.json for sharded models,
    or model.safetensors for single-file models. -/
def loadQwen3ModelSharded (modelDir : String) (cfg : QwenConfig := QwenConfig.qwen3_4B)
    (loadQKNorms : Bool := false)
    (log : Handlers := {})
    (device : Device := .CPU) : IO (Qwen3Model (device := device) cfg) := do
  log.onInfo s!"Loading Qwen3 model from {modelDir}..."

  -- Token embeddings
  let embed_tokens ← loadRequiredSharded (device := device) modelDir "model.embed_tokens.weight" #[cfg.vocab_size, cfg.hidden_size]
  let embed_tokens := Tensor.detach embed_tokens.2.toFloat32
  log.onInfo "  Loaded embed_tokens"

  -- Layers
  let mut layers := #[]
  for i in [:cfg.num_hidden_layers] do
    let layer ← loadLayerSharded (device := device) modelDir i cfg loadQKNorms
    layers := layers.push layer
    if (i + 1) % 7 == 0 then
      log.onInfo s!"  Loaded {i + 1}/{cfg.num_hidden_layers} layers"

  -- Final norm
  let norm ← loadRMSNormSharded (device := device) modelDir "model.norm" cfg.hidden_size
  log.onInfo "  Loaded final norm"

  log.onInfo "Qwen3 model loaded successfully!"
  pure { embed_tokens, layers, norm }

-- Legacy functions for backwards compatibility with single-file models

/-- Load a single attention layer from SafeTensors (single file) -/
def loadAttention (path : String) (layerIdx : Nat)
    (hidden_size num_heads num_kv_heads head_dim : Nat)
    (device : Device := .CPU) : IO (QwenAttention (device := device) hidden_size num_heads num_kv_heads head_dim) := do
  let layerPrefix := s!"model.layers.{layerIdx}.self_attn"

  -- Qwen uses fused QKV in some versions, separate in others
  -- We'll load separate projections
  let q_proj ← loadLinearWeight (device := device) path s!"{layerPrefix}.q_proj" (num_heads * head_dim) hidden_size
  let k_proj ← loadLinearWeight (device := device) path s!"{layerPrefix}.k_proj" (num_kv_heads * head_dim) hidden_size
  let v_proj ← loadLinearWeight (device := device) path s!"{layerPrefix}.v_proj" (num_kv_heads * head_dim) hidden_size
  let o_proj ← loadLinearWeight (device := device) path s!"{layerPrefix}.o_proj" hidden_size (num_heads * head_dim)

  pure {
    q_proj := Tensor.detach q_proj
    k_proj := Tensor.detach k_proj
    v_proj := Tensor.detach v_proj
    o_proj := Tensor.detach o_proj
  }

/-- Load a single MLP layer from SafeTensors (single file) -/
def loadMLP (path : String) (layerIdx : Nat)
    (hidden_size intermediate_size : Nat)
    (device : Device := .CPU) : IO (QwenMLP (device := device) hidden_size intermediate_size) := do
  let layerPrefix := s!"model.layers.{layerIdx}.mlp"

  let gate_proj ← loadLinearWeight (device := device) path s!"{layerPrefix}.gate_proj" intermediate_size hidden_size
  let up_proj ← loadLinearWeight (device := device) path s!"{layerPrefix}.up_proj" intermediate_size hidden_size
  let down_proj ← loadLinearWeight (device := device) path s!"{layerPrefix}.down_proj" hidden_size intermediate_size

  pure {
    gate_proj := Tensor.detach gate_proj
    up_proj := Tensor.detach up_proj
    down_proj := Tensor.detach down_proj
  }

/-- Load RMSNorm weights from SafeTensors (single file) -/
def loadRMSNorm (path : String) (name : String) (dim : Nat)
    (device : Device := .CPU) : IO (RMSNorm (device := device) dim) := do
  let scale ← loadRequired (device := device) path s!"{name}.weight" #[dim]
  pure { weight := Tensor.detach scale.2.toFloat32, eps := ⟨1e-6⟩ }

/-- Load a single transformer layer from SafeTensors (single file) -/
def loadLayer (path : String) (layerIdx : Nat) (cfg : QwenConfig)
    (device : Device := .CPU) : IO (QwenLayer (device := device) cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim cfg.intermediate_size) := do
  let layerPrefix := s!"model.layers.{layerIdx}"

  let input_layernorm ← loadRMSNorm (device := device) path s!"{layerPrefix}.input_layernorm" cfg.hidden_size
  let self_attn ← loadAttention (device := device) path layerIdx cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim
  let post_attention_layernorm ← loadRMSNorm (device := device) path s!"{layerPrefix}.post_attention_layernorm" cfg.hidden_size
  let mlp ← loadMLP (device := device) path layerIdx cfg.hidden_size cfg.intermediate_size

  pure { input_layernorm, self_attn, post_attention_layernorm, mlp }

/-- Load full Qwen3 model from SafeTensors file (single file, legacy) -/
def loadQwen3Model (path : String) (cfg : QwenConfig := QwenConfig.qwen3_4B)
    (log : Handlers := {})
    (device : Device := .CPU) : IO (Qwen3Model (device := device) cfg) := do
  log.onInfo s!"Loading Qwen3 model from {path}..."

  -- Token embeddings
  let embed_tokens ← loadRequired (device := device) path "model.embed_tokens.weight" #[cfg.vocab_size, cfg.hidden_size]
  let embed_tokens := Tensor.detach embed_tokens.2.toFloat32
  log.onInfo "  Loaded embed_tokens"

  -- Layers
  let mut layers := #[]
  for i in [:cfg.num_hidden_layers] do
    let layer ← loadLayer (device := device) path i cfg
    layers := layers.push layer
    if (i + 1) % 7 == 0 then
      log.onInfo s!"  Loaded {i + 1}/{cfg.num_hidden_layers} layers"

  -- Final norm
  let norm ← loadRMSNorm (device := device) path "model.norm" cfg.hidden_size
  log.onInfo "  Loaded final norm"

  log.onInfo "Qwen3 model loaded successfully!"
  pure { embed_tokens, layers, norm }

/-- Load Qwen model as Flux text embedder (single file, legacy) -/
def loadQwenFluxEmbedder (path : String) (cfg : QwenConfig := QwenConfig.qwen3_4B)
    (max_seq : Nat := 512)
    (outputLayers : Array Nat := #[8, 17, 26])
    (log : Handlers := {})
    (device : Device := .CPU) : IO (QwenFluxEmbedder (device := device) cfg max_seq) := do
  let model ← loadQwen3Model (device := device) path cfg log
  QwenFluxEmbedder.fromModel cfg max_seq model outputLayers

/-- Load Qwen model as Flux text embedder from sharded directory.
    This is the preferred method for HuggingFace models with sharded weights. -/
def loadQwenFluxEmbedderSharded (modelDir : String) (cfg : QwenConfig := QwenConfig.fluxKleinTextEncoder)
    (max_seq : Nat := 512)
    (outputLayers : Array Nat := #[8, 17, 26])  -- Flux2 reference uses [9,18,27] with embedding state
    (log : Handlers := {})
    (device : Device := .CPU) : IO (QwenFluxEmbedder (device := device) cfg max_seq) := do
  -- Flux Klein text encoder has Q/K norms
  let model ← loadQwen3ModelSharded (device := device) modelDir cfg true log
  QwenFluxEmbedder.fromModel cfg max_seq model outputLayers

end torch.qwen
