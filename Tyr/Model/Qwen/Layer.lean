/-
  Tyr/Model/Qwen/Layer.lean

  Transformer layer for Qwen3.
  Pre-norm architecture with RMSNorm.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Module.RMSNorm
import Tyr.Model.Qwen.Config
import Tyr.Model.Qwen.Attention
import Tyr.Model.Qwen.MLP

namespace torch.qwen

/-- Qwen transformer layer.
    Uses pre-norm architecture with RMSNorm. -/
structure QwenLayer (hidden_size num_heads num_kv_heads head_dim intermediate_size : UInt64) where
  /-- Input layer norm (before attention) -/
  input_layernorm : RMSNorm hidden_size
  /-- Self-attention module -/
  self_attn : QwenAttention hidden_size num_heads num_kv_heads head_dim
  /-- Post-attention layer norm (before MLP) -/
  post_attention_layernorm : RMSNorm hidden_size
  /-- MLP module -/
  mlp : QwenMLP hidden_size intermediate_size
  deriving TensorStruct

namespace QwenLayer

/-- Initialize a transformer layer -/
def init (hidden_size num_heads num_kv_heads head_dim intermediate_size : UInt64)
    (eps : Float := 1e-6)
    : IO (QwenLayer hidden_size num_heads num_kv_heads head_dim intermediate_size) := do
  let input_layernorm := RMSNorm.init hidden_size eps
  let self_attn ← QwenAttention.init hidden_size num_heads num_kv_heads head_dim
  let post_attention_layernorm := RMSNorm.init hidden_size eps
  let mlp ← QwenMLP.init hidden_size intermediate_size
  pure { input_layernorm, self_attn, post_attention_layernorm, mlp }

/-- Migration boundary: wrap a legacy runtime handle for the typed module
    interface, claiming Float32 (the legacy runtime dtype). Removed when this
    module migrates to indexed tensors. -/
private def typedF32 {s : Shape} (t : T s) : DTensor s .Float32 := .assumeSpec t

/-- Forward pass for a transformer layer.
    Input: [batch, seq, hidden_size]
    Output: [batch, seq, hidden_size]

    Pre-norm: x + attn(norm1(x)) + mlp(norm2(x + attn(...))) -/
def forward {batch seq hidden_size num_heads num_kv_heads head_dim intermediate_size : UInt64}
    (layer : QwenLayer hidden_size num_heads num_kv_heads head_dim intermediate_size)
    (x : T #[batch, seq, hidden_size])
    (cos : T #[seq, head_dim / 2])
    (sin : T #[seq, head_dim / 2])
    (is_causal : Bool := true)
    : T #[batch, seq, hidden_size] :=
  -- Pre-norm attention
  let residual := x
  let x := (layer.input_layernorm.forward3d (typedF32 x) rfl).raw
  let x := layer.self_attn.forward x cos sin is_causal
  let x := residual + x

  -- Pre-norm MLP
  let residual := x
  let x := (layer.post_attention_layernorm.forward3d (typedF32 x) rfl).raw
  let x := layer.mlp.forward x
  residual + x

/-- Incremental one-token transformer layer step with attention KV cache.
    Input/output: `[batch, 1, hidden_size]`. -/
def forwardStep {batch hidden_size num_heads num_kv_heads head_dim intermediate_size : UInt64}
    (layer : QwenLayer hidden_size num_heads num_kv_heads head_dim intermediate_size)
    (x : T #[batch, 1, hidden_size])
    (cos : T #[1, head_dim / 2])
    (sin : T #[1, head_dim / 2])
    (cache : QwenAttention.KVCache batch num_kv_heads head_dim)
    : T #[batch, 1, hidden_size] × QwenAttention.KVCache batch num_kv_heads head_dim :=
  -- Pre-norm attention.
  let residual1 := x
  let x1 := (layer.input_layernorm.forward3d (typedF32 x) rfl).raw
  let (attnOut, cache') := layer.self_attn.forwardStep x1 cos sin cache
  let h1 := residual1 + attnOut

  -- Pre-norm MLP.
  let residual2 := h1
  let h2 := (layer.post_attention_layernorm.forward3d (typedF32 h1) rfl).raw
  let h3 := layer.mlp.forward h2
  (residual2 + h3, cache')

def forwardMasked {batch seq hidden_size num_heads num_kv_heads head_dim intermediate_size : UInt64}
    (layer : QwenLayer hidden_size num_heads num_kv_heads head_dim intermediate_size)
    (x : T #[batch, seq, hidden_size])
    (cos : T #[seq, head_dim / 2])
    (sin : T #[seq, head_dim / 2])
    (attn_mask : T #[batch, seq])
    (is_causal : Bool := true)
    : T #[batch, seq, hidden_size] :=
  -- Pre-norm attention
  let residual := x
  let x := (layer.input_layernorm.forward3d (typedF32 x) rfl).raw
  let x := layer.self_attn.forwardMasked x cos sin attn_mask is_causal
  let x := residual + x

  -- Pre-norm MLP
  let residual := x
  let x := (layer.post_attention_layernorm.forward3d (typedF32 x) rfl).raw
  let x := layer.mlp.forward x
  residual + x

end QwenLayer

end torch.qwen
