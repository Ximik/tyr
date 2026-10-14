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

/-- Configuration for Flux text embedding.
    Specifies which layers to extract hidden states from. -/
structure FluxEmbedderConfig where
  /-- Layer indices to extract (0-indexed) -/
  output_layers : Array UInt64 := #[8, 17, 26]
  /-- Maximum sequence length -/
  max_seq_len : UInt64 := 512
  deriving Repr, Inhabited

/-- Qwen-based text embedder for Flux.
    Extracts hidden states from specified layers and concatenates them. -/
structure QwenFluxEmbedder (cfg : QwenConfig) (max_seq : UInt64) where
  /-- The Qwen3 model -/
  model : Qwen3Model cfg
  /-- RoPE cache for position embeddings -/
  ropeCache : RoPECache max_seq cfg.head_dim
  /-- Which layers to extract -/
  outputLayers : Array UInt64

namespace QwenFluxEmbedder

/-- Compute output dimension: num_layers * hidden_size -/
def outputDim (cfg : QwenConfig) (numLayers : UInt64) : UInt64 :=
  numLayers * cfg.hidden_size

/-- Create a text embedder from a Qwen3 model -/
def fromModel (cfg : QwenConfig) (max_seq : UInt64) (model : Qwen3Model cfg)
    (outputLayers : Array UInt64 := #[8, 17, 26])
    : IO (QwenFluxEmbedder cfg max_seq) := do
  let ropeCache ← RoPECache.init max_seq cfg.head_dim cfg.rope_theta
  pure { model, ropeCache, outputLayers }

/-- Concatenate equal-spec tensors along `dim`, validating the runtime
    result against the claimed output shape (the legacy code reshaped the
    shape-erased cat result, which libtorch validated with a throw). -/
private def catAlong {s : Shape} {d : DType} {dev : DevicePolicy}
    (tensors : Array (Tensor ⟨s, d, dev⟩)) (dim : Int) (out : Shape)
    : Tensor ⟨out, d, dev⟩ :=
  match Tensor.catMany? (tensors.map Tensor.packShape) dim out with
  | .ok t => t
  | .error message => panic! s!"QwenFluxEmbedder cat: {message}"

/-- Encode text tokens to Flux-compatible embeddings.
    Input: [batch, seq] token IDs
    Output: [batch, seq, output_dim] where output_dim = num_layers * hidden_size

    For Qwen3-4B with layers [8, 17, 26]:
    output_dim = 3 * 2560 = 7680

    If `outputLayers` names a layer index that produced no hidden state, the
    concatenated tensor is smaller than the return type claims; `catAlong`
    panics on the mismatch, exactly where the legacy `reshape` threw. -/
def encode {batch seq : UInt64} (cfg : QwenConfig) (max_seq : UInt64)
    (embedder : QwenFluxEmbedder cfg max_seq)
    (input_ids : DTensor #[batch, seq] .Int64)
    : DTensor #[batch, seq, embedder.outputLayers.size.toUInt64 * cfg.hidden_size] .Float32 :=
  -- Slice RoPE cache to current sequence length
  let cos := embedder.ropeCache.cos.sliceChecked 0 0 seq #[seq, cfg.head_dim / 2]
  let sin := embedder.ropeCache.sin.sliceChecked 0 0 seq #[seq, cfg.head_dim / 2]

  -- Get hidden states from all layers
  let allHiddenStates := embedder.model.forwardWithHiddenStates cfg input_ids cos sin

  -- Extract specified layers using filterMap
  let selectedStates : Array (DTensor #[batch, seq, cfg.hidden_size] .Float32) :=
    embedder.outputLayers.filterMap fun layerIdx =>
      if h : layerIdx.toNat < allHiddenStates.size then
        some (allHiddenStates[layerIdx.toNat]'h)
      else
        none

  -- Concatenate selected states along dim 2
  if selectedStates.size > 0 then
    catAlong selectedStates 2 #[batch, seq, embedder.outputLayers.size.toUInt64 * cfg.hidden_size]
  else
    -- Fallback: return zeros if no layers specified
    Tensor.zeros #[batch, seq, embedder.outputLayers.size.toUInt64 * cfg.hidden_size]

def encodeMasked {batch seq : UInt64} (cfg : QwenConfig) (max_seq : UInt64)
    {maskDtype : DType}
    (embedder : QwenFluxEmbedder cfg max_seq)
    (input_ids : DTensor #[batch, seq] .Int64)
    (attn_mask : Tensor ⟨#[batch, seq], maskDtype, .any⟩)
    : DTensor #[batch, seq, embedder.outputLayers.size.toUInt64 * cfg.hidden_size] .Float32 :=
  -- Slice RoPE cache to current sequence length
  let cos := embedder.ropeCache.cos.sliceChecked 0 0 seq #[seq, cfg.head_dim / 2]
  let sin := embedder.ropeCache.sin.sliceChecked 0 0 seq #[seq, cfg.head_dim / 2]

  -- Get hidden states from all layers
  let allHiddenStates := embedder.model.forwardWithHiddenStatesMasked cfg input_ids cos sin attn_mask

  -- Extract specified layers using filterMap
  let selectedStates : Array (DTensor #[batch, seq, cfg.hidden_size] .Float32) :=
    embedder.outputLayers.filterMap fun layerIdx =>
      if h : layerIdx.toNat < allHiddenStates.size then
        some (allHiddenStates[layerIdx.toNat]'h)
      else
        none

  -- Concatenate selected states along dim 2
  if selectedStates.size > 0 then
    catAlong selectedStates 2 #[batch, seq, embedder.outputLayers.size.toUInt64 * cfg.hidden_size]
  else
    -- Fallback: return zeros if no layers specified
    Tensor.zeros #[batch, seq, embedder.outputLayers.size.toUInt64 * cfg.hidden_size]

end QwenFluxEmbedder

end torch.qwen
