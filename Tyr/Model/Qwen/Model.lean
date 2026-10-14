/-
  Tyr/Model/Qwen/Model.lean

  Full Qwen3 model for text encoding.

  Shape-indexed activations and weights share a dtype parameter. Normalization
  accumulates in Float32 and restores the input dtype for BF16 checkpoints.
-/
import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Module.RMSNorm
import Tyr.Model.Qwen.Config
import Tyr.Model.Qwen.Layer
import Tyr.Model.Qwen.RoPE

namespace torch.qwen

/-- Qwen3 model structure.
    Uses type-level parameters for dimensions. -/
structure Qwen3Model (cfg : QwenConfig) (dtype : DType := .Float32) where
  /-- Token embeddings: [vocab_size, hidden_size] -/
  embed_tokens : Tensor { shape := #[cfg.vocab_size, cfg.hidden_size], dtype := dtype }
  /-- Transformer layers -/
  layers : Array (QwenLayer cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim cfg.intermediate_size dtype)
  /-- Final layer norm -/
  norm : RMSNorm cfg.hidden_size dtype
  deriving TensorStruct

namespace Qwen3Model

variable {dtype : DType}

/-- Initialize a Qwen3 model with random weights -/
def init (cfg : QwenConfig) (dtype : DType := .Float32) : IO (Qwen3Model cfg dtype) := do
  -- Token embeddings
  let emb ← Tensor.randn #[cfg.vocab_size, cfg.hidden_size]
  let emb := emb.castDType dtype
  let std := 0.02  -- Standard embedding init
  let embed_tokens := Tensor.setRequiresGrad (Tensor.Preserving.mulScalar emb std) true

  -- Initialize layers
  let mut layers := #[]
  for _ in [:cfg.num_hidden_layers.toNat] do
    let layer ← QwenLayer.init cfg.hidden_size cfg.num_attention_heads
        cfg.num_key_value_heads cfg.head_dim cfg.intermediate_size cfg.rms_norm_eps dtype
    layers := layers.push layer

  -- Final norm
  let norm := RMSNorm.init cfg.hidden_size cfg.rms_norm_eps dtype

  pure { embed_tokens, layers, norm }

/-- Forward pass returning hidden states from all layers.
    Input: [batch, seq] token IDs
    Output: Array of [batch, seq, hidden_size] hidden states (one per layer + final)
    Rewritten in functional style to avoid Id.run issues with FFI shape tracking. -/
def forwardWithHiddenStates {batch seq : UInt64} (cfg : QwenConfig)
    (model : Qwen3Model cfg dtype)
    (input_ids : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (cos : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (sin : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    : Array (Tensor { shape := #[batch, seq, cfg.hidden_size], dtype := dtype }) :=
  -- Embed tokens
  let hidden := Tensor.embedding input_ids model.embed_tokens
  -- Process each layer, accumulating (current_hidden, collected_states)
  let (finalH, hiddenStates) := model.layers.foldl
    (fun (acc : Tensor { shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × Array (Tensor { shape := #[batch, seq, cfg.hidden_size], dtype := dtype })) layer =>
      let (h, states) := acc
      let h' := layer.forward h cos sin true
      (h', states.push h'))
    (hidden, #[])
  -- Add final normalized state
  let finalHidden : Tensor { shape := #[batch, seq, cfg.hidden_size], dtype := dtype } := model.norm.forward3dPreserving finalH
  hiddenStates.push finalHidden

def forwardWithHiddenStatesMasked {batch seq : UInt64} (cfg : QwenConfig)
    {maskDtype : DType}
    (model : Qwen3Model cfg dtype)
    (input_ids : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (cos : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (sin : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (attn_mask : Tensor ⟨#[batch, seq], maskDtype, .any⟩)
    : Array (Tensor { shape := #[batch, seq, cfg.hidden_size], dtype := dtype }) :=
  -- Embed tokens
  let hidden := Tensor.embedding input_ids model.embed_tokens
  -- Process each layer, accumulating (current_hidden, collected_states)
  let (finalH, hiddenStates) := model.layers.foldl
    (fun (acc : Tensor { shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × Array (Tensor { shape := #[batch, seq, cfg.hidden_size], dtype := dtype })) layer =>
      let (h, states) := acc
      let h' := layer.forwardMasked h cos sin attn_mask true
      (h', states.push h'))
    (hidden, #[])
  -- Add final normalized state
  let finalHidden : Tensor { shape := #[batch, seq, cfg.hidden_size], dtype := dtype } := model.norm.forward3dPreserving finalH
  hiddenStates.push finalHidden

/-- Forward pass returning only the final hidden state.
    Input: [batch, seq] token IDs
    Output: [batch, seq, hidden_size]
    Rewritten in functional style to avoid Id.run issues with FFI shape tracking. -/
def forward {batch seq : UInt64} (cfg : QwenConfig)
    (model : Qwen3Model cfg dtype)
    (input_ids : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (cos : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (sin : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    : Tensor { shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  -- Embed tokens
  let hidden := Tensor.embedding input_ids model.embed_tokens
  -- Process each layer using fold
  let finalH := model.layers.foldl
    (fun h layer => layer.forward h cos sin true)
    hidden
  -- Final normalization
  model.norm.forward3dPreserving finalH

def forwardMasked {batch seq : UInt64} (cfg : QwenConfig)
    {maskDtype : DType}
    (model : Qwen3Model cfg dtype)
    (input_ids : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (cos : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (sin : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (attn_mask : Tensor ⟨#[batch, seq], maskDtype, .any⟩)
    : Tensor { shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  -- Embed tokens
  let hidden := Tensor.embedding input_ids model.embed_tokens
  -- Process each layer using fold
  let finalH := model.layers.foldl
    (fun h layer => layer.forwardMasked h cos sin attn_mask true)
    hidden
  -- Final normalization
  model.norm.forward3dPreserving finalH

end Qwen3Model

end torch.qwen
