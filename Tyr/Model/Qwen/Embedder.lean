/-
  Tyr/Model/Qwen/Embedder.lean

  Flux text embedder using Qwen3.
  Extracts hidden states from specific layers and concatenates them.

  Fully typed: token ids are Int64, hidden states Float32.
-/
import Tyr.Torch
import Tyr.Typed
import Tyr.Typed.Neural
import Tyr.Model.Qwen.Config
import Tyr.Model.Qwen.Model
import Tyr.Model.Qwen.RoPE

namespace torch.qwen

variable {device : Device}
/-- Qwen-based text embedder for Flux.
    Extracts hidden states from specified layers and concatenates them. -/
structure QwenFluxEmbedder (cfg : QwenConfig) (max_seq : Nat) where
  /-- The Qwen3 model -/
  model : Qwen3Model (device := device) cfg
  layersSize : model.layers.size = cfg.num_hidden_layers
  /-- RoPE cache for position embeddings -/
  ropeCache : RoPECache (device := device) max_seq cfg.half_head_dim
  /-- Decoder layers whose outputs are concatenated, 0-indexed -/
  outputLayers : Array (Fin cfg.num_hidden_layers)

namespace QwenFluxEmbedder

/-- Create a text embedder from a Qwen3 model; `outputLayers` must be nonempty
    and name layers of `cfg`, and the model must have `cfg.num_hidden_layers` layers. -/
def fromModel (cfg : QwenConfig) (max_seq : Nat) (model : Qwen3Model (device := device) cfg)
    (outputLayers : Array Nat)
    : IO (QwenFluxEmbedder (device := device) cfg max_seq) := do
  let ⟨layersSize⟩ ← if h : model.layers.size = cfg.num_hidden_layers then pure (PLift.up h) else
    throw <| IO.userError s!"the model has {model.layers.size} layers, the config {cfg.num_hidden_layers}"
  if outputLayers.isEmpty then
    throw <| IO.userError "no output layers"
  let outputLayers ← outputLayers.mapM fun i =>
    if h : i < cfg.num_hidden_layers then pure (⟨i, h⟩ : Fin cfg.num_hidden_layers) else
      throw <| IO.userError s!"output layer {i} is not below num_hidden_layers = {cfg.num_hidden_layers}"
  let ropeCache ← RoPECache.init (device := device) max_seq cfg.half_head_dim cfg.rope_theta
  pure { model, layersSize, ropeCache, outputLayers }

/-- Concatenate equal-spec tensors along `dim`, validating the runtime
    result against the claimed output shape. -/
private def catAlong {s : Shape} {d : DType} {dev : Device}
    (tensors : Array (Tensor ⟨s, d, dev⟩)) (dim : Nat) (out : Shape)
    : Tensor ⟨out, d, dev⟩ :=
  match Tensor.catMany? (tensors.map Tensor.packShape) dim out with
  | .ok t => t
  | .error message => nativeInvariantFailure s!"{decl_name%}: {message}"

/-- Encode text tokens to Flux-compatible embeddings: the outputs of
    `outputLayers`, concatenated along the feature axis.
    Input: [batch, seq] token IDs and attention mask
    Output: [batch, seq, outputLayers.size * hidden_size] -/
def encodeMasked {batch seq : Nat} (cfg : QwenConfig) (max_seq : Nat)
    {maskDtype : DType}
    (embedder : QwenFluxEmbedder (device := device) cfg max_seq)
    (input_ids : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (attn_mask : Tensor ⟨#[batch, seq], maskDtype, device⟩)
    (hseq : seq ≤ max_seq := by omega)
    : Tensor { device := device, shape := #[batch, seq, embedder.outputLayers.size * cfg.hidden_size], dtype := .Float32 } :=
  let cos := embedder.ropeCache.cos.slice 0 0 seq
  let sin := embedder.ropeCache.sin.slice 0 0 seq
  let states := List.scanl
    (fun h layer => layer.forwardMasked cfg.headsValid.2.2 h cos sin attn_mask true)
    (Tensor.embedding input_ids embedder.model.embed_tokens) embedder.model.layers.toList
  have hStates : states.length = cfg.num_hidden_layers + 1 := by
    simp [states, List.length_scanl, embedder.layersSize]
  let selected := embedder.outputLayers.map fun i => states[i.val + 1]'(by omega)
  catAlong selected 2 #[batch, seq, embedder.outputLayers.size * cfg.hidden_size]

end QwenFluxEmbedder

end torch.qwen
