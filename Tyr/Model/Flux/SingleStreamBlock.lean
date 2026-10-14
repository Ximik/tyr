/-
  Tyr/Model/Flux/SingleStreamBlock.lean

  Single-stream transformer block for Flux.
  Processes concatenated image and text tokens together.
-/
import Tyr.Model.Flux.TypedOps
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Module.LayerNorm
import Tyr.Model.Flux.Config
import Tyr.Model.Flux.RoPE
import Tyr.Model.Flux.Modulation
import Tyr.Model.Flux.QKNorm

namespace torch.flux

/-- Single-stream transformer block.
    Processes concatenated image + text tokens with shared attention.
    Uses fused QKV + MLP projection for efficiency. -/
structure SingleStreamBlock (hidden_size num_heads head_dim mlp_hidden : UInt64) where
  /-- Pre-norm (LayerNorm) -/
  pre_norm : LayerNorm hidden_size (.Unknown "dynamic")
  /-- Fused projection for Q, K, V and MLP gate/up
      Q: [hidden, hidden], K: [hidden, hidden], V: [hidden, hidden]
      gate: [mlp_hidden, hidden], up: [mlp_hidden, hidden]
      Total: [hidden * 3 + mlp_hidden * 2, hidden] -/
  linear1 : Tensor { shape := #[num_heads * head_dim * 3 + mlp_hidden * 2, hidden_size] }
  /-- Fused output projection for attention + MLP down
      attn_out: [hidden, hidden], mlp_down: [hidden, mlp_hidden]
      Total: [hidden, hidden + mlp_hidden] -/
  linear2 : Tensor { shape := #[hidden_size, num_heads * head_dim + mlp_hidden] }
  /-- QK normalization -/
  norm : QKNorm head_dim
  deriving TensorStruct

namespace SingleStreamBlock

/-- Initialize single-stream block -/
def init (hidden_size num_heads head_dim mlp_hidden : UInt64)
    : IO (SingleStreamBlock hidden_size num_heads head_dim mlp_hidden) := do
  let pre_norm ← initLayerNorm hidden_size
  let qkv_dim := num_heads * head_dim * 3
  let mlp_in := mlp_hidden * 2
  let std := Float.sqrt (2.0 / hidden_size.toFloat)

  let l1 ← randomWeight #[qkv_dim + mlp_in, hidden_size]
  let l2 ← randomWeight #[hidden_size, num_heads * head_dim + mlp_hidden]

  let norm ← QKNorm.init head_dim
  pure {
    pre_norm
    linear1 := (← Tensor.parameter (Tensor.Preserving.mulScalar l1 std))
    linear2 := (← Tensor.parameter (Tensor.Preserving.mulScalar l2 std))
    norm
  }

/-- Forward pass for single-stream block.
    x: [batch, seq, hidden_size] - concatenated image + text tokens
    pe: RoPE embeddings
    mod: [batch, 3, hidden_size] - shift/scale/gate modulation
    Returns: [batch, seq, hidden_size] -/
def forward {batch seq hidden_size num_heads head_dim mlp_hidden : UInt64}
    (block : SingleStreamBlock hidden_size num_heads head_dim mlp_hidden)
    (x : Tensor { shape := #[batch, seq, hidden_size] })
    (pe : Tensor { shape := #[batch, 1, seq, head_dim / 2, 2, 2], dtype := .Float32 })  -- Shape depends on RoPE implementation
    (mod : Tensor { shape := #[batch, 3, hidden_size] })
    : Tensor { shape := #[batch, seq, hidden_size] } :=
  -- Extract shift, scale, gate from modulation
  let mod_shift := Tensor.reshapeChecked (sliceTyped mod 1 0 1) #[batch, hidden_size]
  let mod_scale := Tensor.reshapeChecked (sliceTyped mod 1 1 1) #[batch, hidden_size]
  let mod_gate := Tensor.reshapeChecked (sliceTyped mod 1 2 1) #[batch, hidden_size]

  -- Pre-norm with modulation
  let x_norm := block.pre_norm.forward3d x
  let x_mod := applyModulation x_norm mod_scale mod_shift

  -- Fused QKV + MLP projection
  let qkv_mlp := Tensor.linear3d x_mod block.linear1  -- [batch, seq, qkv_dim + mlp_in]

  -- Split into QKV and MLP parts
  let qkv_dim := num_heads * head_dim * 3
  let qkv := sliceTyped qkv_mlp 2 0 qkv_dim
  let mlp := sliceTyped qkv_mlp 2 qkv_dim (mlp_hidden * 2)

  -- Reshape QKV
  let qkv := Tensor.reshapeChecked qkv #[batch, seq, 3, num_heads, head_dim]
  let q := Tensor.reshapeChecked (sliceTyped qkv 2 0 1) #[batch, seq, num_heads, head_dim]
  let k := Tensor.reshapeChecked (sliceTyped qkv 2 1 1) #[batch, seq, num_heads, head_dim]
  let v := Tensor.reshapeChecked (sliceTyped qkv 2 2 1) #[batch, seq, num_heads, head_dim]

  -- QK normalization
  let (q, k) := block.norm.forward q k

  -- Apply RoPE
  let q := flux.applyRope q pe
  let k := flux.applyRope k pe

  -- Transpose for attention: [batch, num_heads, seq, head_dim]
  let q := transposeAttention q
  let k := transposeAttention k
  let v := transposeAttention v

  -- Scaled dot-product attention (non-causal for diffusion)
  let attn := Tensor.attentionGQA q k v 0.0 false

  -- Transpose back and reshape
  let attn := fromAttention attn
  let attn := Tensor.reshapeChecked attn #[batch, seq, num_heads * head_dim]

  -- MLP: SwiGLU
  let mlp := Tensor.reshapeChecked mlp #[batch, seq, 2, mlp_hidden]
  let gate := Tensor.reshapeChecked (sliceTyped mlp 2 0 1) #[batch, seq, mlp_hidden]
  let up := Tensor.reshapeChecked (sliceTyped mlp 2 1 1) #[batch, seq, mlp_hidden]
  let mlp_out := Tensor.Preserving.silu gate * up

  -- Concatenate attention and MLP outputs
  let combined : Tensor { shape := #[batch, seq, num_heads * head_dim + mlp_hidden] } := qwen35.typedCat attn mlp_out 2  -- [batch, seq, hidden + mlp_hidden]

  -- Final projection
  let out := Tensor.linear3d combined block.linear2  -- [batch, seq, hidden_size]

  -- Apply gate and residual
  let mod_gate := Tensor.unsqueeze mod_gate 1
  let mod_gate := Tensor.expand mod_gate #[batch, seq, hidden_size]
  x + mod_gate * out

end SingleStreamBlock

end torch.flux
