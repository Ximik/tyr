/-
  Tyr/Model/Qwen/Layer.lean

  Transformer layer for Qwen3.
  Pre-norm architecture with RMSNorm.

  Shape-indexed activations and weights share a dtype parameter. Normalization
  accumulates in Float32 and restores the input dtype for BF16 checkpoints.
-/
import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Module.RMSNorm
import Tyr.Model.Qwen.Config
import Tyr.Model.Qwen.Attention
import Tyr.Model.Qwen.MLP

namespace torch.qwen

variable {device : Device}
/-- Qwen transformer layer.
    Uses pre-norm architecture with RMSNorm. -/
structure QwenLayer (hidden_size num_heads num_kv_heads head_dim intermediate_size : Nat) (dtype : DType := .Float32) where
  /-- Input layer norm (before attention) -/
  input_layernorm : RMSNorm (device := device) hidden_size dtype
  /-- Self-attention module -/
  self_attn : QwenAttention (device := device) hidden_size num_heads num_kv_heads head_dim dtype
  /-- Post-attention layer norm (before MLP) -/
  post_attention_layernorm : RMSNorm (device := device) hidden_size dtype
  /-- MLP module -/
  mlp : QwenMLP (device := device) hidden_size intermediate_size dtype
  deriving TensorStruct

namespace QwenLayer

variable {dtype : DType}
variable [DType.RealFloating dtype]

/-- Initialize a transformer layer -/
def init (hidden_size num_heads num_kv_heads head_dim intermediate_size : Nat)
    (eps : Float := 1e-6) (dtype : DType := .Float32)
    (device : Device := .CPU) : IO (QwenLayer (device := device) hidden_size num_heads num_kv_heads head_dim intermediate_size dtype) := do
  let input_layernorm ← RMSNorm.init (device := device) hidden_size eps dtype
  let self_attn ← QwenAttention.init (device := device) hidden_size num_heads num_kv_heads head_dim dtype
  let post_attention_layernorm ← RMSNorm.init (device := device) hidden_size eps dtype
  let mlp ← QwenMLP.init (device := device) hidden_size intermediate_size dtype
  pure { input_layernorm, self_attn, post_attention_layernorm, mlp }

/-- Forward pass for a transformer layer.
    Input: [batch, seq, hidden_size]
    Output: [batch, seq, hidden_size]

    Pre-norm: x + attn(norm1(x)) + mlp(norm2(x + attn(...))) -/
def forward {batch seq hidden_size num_heads num_kv_heads head_dim intermediate_size : Nat}
    (layer : QwenLayer (device := device) hidden_size num_heads num_kv_heads head_dim intermediate_size dtype)
    (x : Tensor { device := device, shape := #[batch, seq, hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, head_dim / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, head_dim / 2], dtype := .Float32 })
    (is_causal : Bool := true)
    : Tensor { device := device, shape := #[batch, seq, hidden_size], dtype := dtype } :=
  -- Pre-norm attention
  let residual := x
  let x : Tensor { device := device, shape := #[batch, seq, hidden_size], dtype := dtype } := layer.input_layernorm.forward3dPreserving x
  let x := layer.self_attn.forward x cos sin is_causal
  let x : Tensor { device := device, shape := #[batch, seq, hidden_size], dtype := dtype } := Tensor.add residual x

  -- Pre-norm MLP
  let residual := x
  let x : Tensor { device := device, shape := #[batch, seq, hidden_size], dtype := dtype } := layer.post_attention_layernorm.forward3dPreserving x
  let x := layer.mlp.forward x
  Tensor.add residual x

/-- Incremental one-token transformer layer step with attention KV cache.
    Input/output: `[batch, 1, hidden_size]`. -/
def forwardStep {batch hidden_size num_heads num_kv_heads head_dim intermediate_size : Nat}
    (layer : QwenLayer (device := device) hidden_size num_heads num_kv_heads head_dim intermediate_size dtype)
    (x : Tensor { device := device, shape := #[batch, 1, hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[1, head_dim / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[1, head_dim / 2], dtype := .Float32 })
    (cache : QwenAttention.KVCache (device := device) batch num_kv_heads head_dim dtype)
    : Tensor { device := device, shape := #[batch, 1, hidden_size], dtype := dtype } × QwenAttention.KVCache (device := device) batch num_kv_heads head_dim dtype :=
  -- Pre-norm attention.
  let residual1 := x
  let x1 : Tensor { device := device, shape := #[batch, 1, hidden_size], dtype := dtype } := layer.input_layernorm.forward3dPreserving x
  let (attnOut, cache') := layer.self_attn.forwardStep x1 cos sin cache
  let h1 : Tensor { device := device, shape := #[batch, 1, hidden_size], dtype := dtype } := Tensor.add residual1 attnOut

  -- Pre-norm MLP.
  let residual2 := h1
  let h2 : Tensor { device := device, shape := #[batch, 1, hidden_size], dtype := dtype } := layer.post_attention_layernorm.forward3dPreserving h1
  let h3 := layer.mlp.forward h2
  (Tensor.add residual2 h3, cache')

def forwardMasked {batch seq hidden_size num_heads num_kv_heads head_dim intermediate_size : Nat}
    {maskDtype : DType}
    (layer : QwenLayer (device := device) hidden_size num_heads num_kv_heads head_dim intermediate_size dtype)
    (x : Tensor { device := device, shape := #[batch, seq, hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, head_dim / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, head_dim / 2], dtype := .Float32 })
    (attn_mask : Tensor ⟨#[batch, seq], maskDtype, device⟩)
    (is_causal : Bool := true)
    : Tensor { device := device, shape := #[batch, seq, hidden_size], dtype := dtype } :=
  -- Pre-norm attention
  let residual := x
  let x : Tensor { device := device, shape := #[batch, seq, hidden_size], dtype := dtype } := layer.input_layernorm.forward3dPreserving x
  let x := layer.self_attn.forwardMasked x cos sin attn_mask is_causal
  let x : Tensor { device := device, shape := #[batch, seq, hidden_size], dtype := dtype } := Tensor.add residual x

  -- Pre-norm MLP
  let residual := x
  let x : Tensor { device := device, shape := #[batch, seq, hidden_size], dtype := dtype } := layer.post_attention_layernorm.forward3dPreserving x
  let x := layer.mlp.forward x
  Tensor.add residual x

end QwenLayer

end torch.qwen
