/-
  Tyr/Model/Qwen/Weights.lean

  Weight loading for Qwen3-4B from SafeTensors format.
  Maps HuggingFace weight names to Tyr structure.
  Supports both single-file and sharded HuggingFace models.

  Each load retains the stored dtype in a dependent package. The caller
  explicitly converts its tensor to the Float32 structure fields. FP8 weights
  and block scales keep independent dtype indices until the existing Float32
  dequantization step.
-/
import Tyr.Typed
import Tyr.Log
import Tyr.SafeTensors.Source
import Tyr.Model.Utils
import Tyr.Model.Qwen.Config
import Tyr.Model.Qwen.Model
import Tyr.Model.Qwen.Embedder

namespace torch.qwen

variable {device : Device}
open torch.Log
open torch.Model (dequantizeFP8 checkFP8Blocks)
open torch.safetensors (Source)

/-- Load a (potentially FP8-quantized) linear weight.
    If a matching weight_scale_inv exists, dequantize to float32;
    otherwise cast the loaded weights to float32. -/
private def loadLinearWeight (src : Source) (name : String) (outDim inDim : Nat)
    (device : Device := .CPU) : IO (Tensor { device := device, shape := #[outDim, inDim], dtype := .Float32 }) := do
  let w ← src.load s!"{name}.weight" #[outDim, inDim] device
  match ← src.load? s!"{name}.weight_scale_inv" #[outDim / 128, inDim / 128] device with
  | some s =>
    let ⟨hBlocks⟩ ← checkFP8Blocks s!"{name}.weight_scale_inv" outDim inDim
    pure (dequantizeFP8 w.2 s.2 hBlocks)
  | none => pure w.2.toFloat32

/-- Load a single attention layer.
    Optionally loads Q/K norms if present (for Flux Klein text encoder). -/
def loadAttention (src : Source) (layerIdx : Nat)
    (hidden_size num_heads num_kv_heads head_dim : Nat)
    (loadQKNorms : Bool := false)
    (device : Device := .CPU) : IO (QwenAttention (device := device) hidden_size num_heads num_kv_heads head_dim) := do
  let layerPrefix := s!"model.layers.{layerIdx}.self_attn"

  let q_proj ← loadLinearWeight (device := device) src s!"{layerPrefix}.q_proj" (num_heads * head_dim) hidden_size
  let k_proj ← loadLinearWeight (device := device) src s!"{layerPrefix}.k_proj" (num_kv_heads * head_dim) hidden_size
  let v_proj ← loadLinearWeight (device := device) src s!"{layerPrefix}.v_proj" (num_kv_heads * head_dim) hidden_size
  let o_proj ← loadLinearWeight (device := device) src s!"{layerPrefix}.o_proj" hidden_size (num_heads * head_dim)

  -- Optionally load Q/K norms (Flux Klein text encoder has these)
  let q_norm ← if loadQKNorms then
    src.load? s!"{layerPrefix}.q_norm.weight" #[head_dim] device
  else
    pure none

  let k_norm ← if loadQKNorms then
    src.load? s!"{layerPrefix}.k_norm.weight" #[head_dim] device
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

def loadMLP (src : Source) (layerIdx : Nat)
    (hidden_size intermediate_size : Nat)
    (device : Device := .CPU) : IO (QwenMLP (device := device) hidden_size intermediate_size) := do
  let layerPrefix := s!"model.layers.{layerIdx}.mlp"

  let gate_proj ← loadLinearWeight (device := device) src s!"{layerPrefix}.gate_proj" intermediate_size hidden_size
  let up_proj ← loadLinearWeight (device := device) src s!"{layerPrefix}.up_proj" intermediate_size hidden_size
  let down_proj ← loadLinearWeight (device := device) src s!"{layerPrefix}.down_proj" hidden_size intermediate_size

  pure {
    gate_proj := Tensor.detach gate_proj
    up_proj := Tensor.detach up_proj
    down_proj := Tensor.detach down_proj
  }

def loadRMSNorm (src : Source) (name : String) (dim : Nat)
    (device : Device := .CPU) : IO (RMSNorm (device := device) dim) := do
  let scale ← src.load s!"{name}.weight" #[dim] device
  pure { weight := Tensor.detach scale.2.toFloat32, eps := ⟨1e-6⟩ }

def loadLayer (src : Source) (layerIdx : Nat) (cfg : QwenConfig)
    (loadQKNorms : Bool := false)
    (device : Device := .CPU) : IO (QwenLayer (device := device) cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim cfg.intermediate_size) := do
  let layerPrefix := s!"model.layers.{layerIdx}"

  let input_layernorm ← loadRMSNorm (device := device) src s!"{layerPrefix}.input_layernorm" cfg.hidden_size
  let self_attn ← loadAttention (device := device) src layerIdx cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim loadQKNorms
  let post_attention_layernorm ← loadRMSNorm (device := device) src s!"{layerPrefix}.post_attention_layernorm" cfg.hidden_size
  let mlp ← loadMLP (device := device) src layerIdx cfg.hidden_size cfg.intermediate_size

  pure { input_layernorm, self_attn, post_attention_layernorm, mlp }

def loadQwen3Model (src : Source) (cfg : QwenConfig := QwenConfig.qwen3_4B)
    (loadQKNorms : Bool := false)
    (log : Handlers := {})
    (device : Device := .CPU) : IO (Qwen3Model (device := device) cfg) := do
  log.onInfo s!"Loading Qwen3 model from {src.path}..."

  -- Token embeddings
  let embed_tokens ← src.load "model.embed_tokens.weight" #[cfg.vocab_size, cfg.hidden_size] device
  let embed_tokens := Tensor.detach embed_tokens.2.toFloat32
  log.onInfo "  Loaded embed_tokens"

  -- Layers
  let mut layers := #[]
  for i in [:cfg.num_hidden_layers] do
    let layer ← loadLayer (device := device) src i cfg loadQKNorms
    layers := layers.push layer
    if (i + 1) % 7 == 0 then
      log.onInfo s!"  Loaded {i + 1}/{cfg.num_hidden_layers} layers"

  -- Final norm
  let norm ← loadRMSNorm (device := device) src "model.norm" cfg.hidden_size
  log.onInfo "  Loaded final norm"

  log.onInfo "Qwen3 model loaded successfully!"
  pure { embed_tokens, layers, norm }

/-- Load a Qwen model as the Flux text embedder from a SafeTensors file or a
    directory of shards. -/
def loadQwenFluxEmbedder (path : String) (cfg : QwenConfig := QwenConfig.fluxKleinTextEncoder)
    (max_seq : Nat := 512)
    (outputLayers : Array Nat := #[8, 17, 26])  -- Flux2 reference uses [9,18,27] with embedding state
    (log : Handlers := {})
    (device : Device := .CPU) : IO (QwenFluxEmbedder (device := device) cfg max_seq) := do
  -- Flux Klein text encoder has Q/K norms
  let model ← loadQwen3Model (device := device) (← Source.open path) cfg true log
  QwenFluxEmbedder.fromModel cfg max_seq model outputLayers

end torch.qwen
