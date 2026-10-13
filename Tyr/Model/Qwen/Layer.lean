/-
  Tyr/Model/Qwen/Layer.lean

  Transformer layer for Qwen3.
  Pre-norm architecture with RMSNorm.

  Fully typed: the legacy `typedF32` boundary bridge is gone. The forward
  math is Float32-pinned (`DTensor s .Float32`) because the typed RMSNorm
  upcasts to Float32 (`promote Float32 dtype`) and `linear3d` requires
  matching dtypes.
-/
import Tyr.Typed
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

/-- Forward pass for a transformer layer.
    Input: [batch, seq, hidden_size]
    Output: [batch, seq, hidden_size]

    Pre-norm: x + attn(norm1(x)) + mlp(norm2(x + attn(...))) -/
def forward {batch seq hidden_size num_heads num_kv_heads head_dim intermediate_size : UInt64}
    (layer : QwenLayer hidden_size num_heads num_kv_heads head_dim intermediate_size)
    (x : DTensor #[batch, seq, hidden_size] .Float32)
    (cos : DTensor #[seq, head_dim / 2] .Float32)
    (sin : DTensor #[seq, head_dim / 2] .Float32)
    (is_causal : Bool := true)
    : DTensor #[batch, seq, hidden_size] .Float32 :=
  -- Pre-norm attention
  let residual := x
  let x : DTensor #[batch, seq, hidden_size] .Float32 := layer.input_layernorm.forward3d x
  let x := layer.self_attn.forward x cos sin is_causal
  let x : DTensor #[batch, seq, hidden_size] .Float32 := residual + x

  -- Pre-norm MLP
  let residual := x
  let x : DTensor #[batch, seq, hidden_size] .Float32 := layer.post_attention_layernorm.forward3d x
  let x := layer.mlp.forward x
  residual + x

/-- Incremental one-token transformer layer step with attention KV cache.
    Input/output: `[batch, 1, hidden_size]`. -/
def forwardStep {batch hidden_size num_heads num_kv_heads head_dim intermediate_size : UInt64}
    (layer : QwenLayer hidden_size num_heads num_kv_heads head_dim intermediate_size)
    (x : DTensor #[batch, 1, hidden_size] .Float32)
    (cos : DTensor #[1, head_dim / 2] .Float32)
    (sin : DTensor #[1, head_dim / 2] .Float32)
    (cache : QwenAttention.KVCache batch num_kv_heads head_dim)
    : DTensor #[batch, 1, hidden_size] .Float32 × QwenAttention.KVCache batch num_kv_heads head_dim :=
  -- Pre-norm attention.
  let residual1 := x
  let x1 : DTensor #[batch, 1, hidden_size] .Float32 := layer.input_layernorm.forward3d x
  let (attnOut, cache') := layer.self_attn.forwardStep x1 cos sin cache
  let h1 : DTensor #[batch, 1, hidden_size] .Float32 := residual1 + attnOut

  -- Pre-norm MLP.
  let residual2 := h1
  let h2 : DTensor #[batch, 1, hidden_size] .Float32 := layer.post_attention_layernorm.forward3d h1
  let h3 := layer.mlp.forward h2
  (residual2 + h3, cache')

def forwardMasked {batch seq hidden_size num_heads num_kv_heads head_dim intermediate_size : UInt64}
    {maskDtype : DType}
    (layer : QwenLayer hidden_size num_heads num_kv_heads head_dim intermediate_size)
    (x : DTensor #[batch, seq, hidden_size] .Float32)
    (cos : DTensor #[seq, head_dim / 2] .Float32)
    (sin : DTensor #[seq, head_dim / 2] .Float32)
    (attn_mask : Tensor ⟨#[batch, seq], maskDtype, .any⟩)
    (is_causal : Bool := true)
    : DTensor #[batch, seq, hidden_size] .Float32 :=
  -- Pre-norm attention
  let residual := x
  let x : DTensor #[batch, seq, hidden_size] .Float32 := layer.input_layernorm.forward3d x
  let x := layer.self_attn.forwardMasked x cos sin attn_mask is_causal
  let x : DTensor #[batch, seq, hidden_size] .Float32 := residual + x

  -- Pre-norm MLP
  let residual := x
  let x : DTensor #[batch, seq, hidden_size] .Float32 := layer.post_attention_layernorm.forward3d x
  let x := layer.mlp.forward x
  residual + x

end QwenLayer

end torch.qwen
