/-
  Tyr/Model/Flux/DoubleStreamBlock.lean

  Double-stream transformer block for Flux.
  Processes image and text tokens with separate paths but joint attention.
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

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]
/-- Self-attention layer for double-stream block -/
structure SelfAttention (hidden_size num_heads head_dim : Nat) (dtype : DType := .Float32) where
  /-- QKV projection: [hidden * 3, hidden] -/
  qkv : Tensor { dtype := dtype, device := device, shape := #[num_heads * head_dim * 3, hidden_size] }
  /-- QK normalization -/
  norm : QKNorm (device := device) (dtype := dtype) head_dim
  /-- Output projection: [hidden, hidden] -/
  proj : Tensor { dtype := dtype, device := device, shape := #[hidden_size, num_heads * head_dim] }
  deriving TensorStruct


/-- SwiGLU MLP for double-stream block -/
structure SwiGLUMLP (hidden_size mlp_hidden : Nat) (dtype : DType := .Float32) where
  /-- Linear in: gate and up projection fused -/
  w1 : Tensor { dtype := dtype, device := device, shape := #[mlp_hidden * 2, hidden_size] }
  /-- Linear out: down projection -/
  w2 : Tensor { dtype := dtype, device := device, shape := #[hidden_size, mlp_hidden] }
  deriving TensorStruct


/-- Double-stream transformer block.
    Processes image and text tokens with separate paths but joint attention. -/
structure DoubleStreamBlock (hidden_size num_heads head_dim mlp_hidden : Nat) (dtype : DType := .Float32) where
  /-- Image path normalization -/
  img_norm1 : LayerNorm (device := device) hidden_size dtype
  /-- Image self-attention -/
  img_attn : SelfAttention (device := device) (dtype := dtype) hidden_size num_heads head_dim
  /-- Image MLP normalization -/
  img_norm2 : LayerNorm (device := device) hidden_size dtype
  /-- Image MLP -/
  img_mlp : SwiGLUMLP (device := device) (dtype := dtype) hidden_size mlp_hidden

  /-- Text path normalization -/
  txt_norm1 : LayerNorm (device := device) hidden_size dtype
  /-- Text self-attention -/
  txt_attn : SelfAttention (device := device) (dtype := dtype) hidden_size num_heads head_dim
  /-- Text MLP normalization -/
  txt_norm2 : LayerNorm (device := device) hidden_size dtype
  /-- Text MLP -/
  txt_mlp : SwiGLUMLP (device := device) (dtype := dtype) hidden_size mlp_hidden

  deriving TensorStruct


namespace DoubleStreamBlock

/-- Initialize self-attention -/
def initSelfAttention (hidden_size num_heads head_dim : Nat)
    (device : Device := .CPU) (dtype : DType := .Float32) : IO (SelfAttention (device := device) (dtype := dtype) hidden_size num_heads head_dim) := do
  let floating ← IO.ofExcept (DType.realFloating? dtype)
  let _ := floating.down
  let std := Float.sqrt (2.0 / hidden_size.toFloat)
  let qkv ← randomWeight (dtype := dtype) (device := device) #[num_heads * head_dim * 3, hidden_size]
  let proj ← randomWeight (dtype := dtype) (device := device) #[hidden_size, num_heads * head_dim]
  let norm ← QKNorm.init (dtype := dtype) (device := device) head_dim
  pure {
    qkv := (← Tensor.parameter (Tensor.mulScalar qkv std))
    norm
    proj := (← Tensor.parameter (Tensor.mulScalar proj std))
  }

/-- Initialize SwiGLU MLP -/
def initMLP (hidden_size mlp_hidden : Nat)
    (device : Device := .CPU) (dtype : DType := .Float32) : IO (SwiGLUMLP (device := device) (dtype := dtype) hidden_size mlp_hidden) := do
  let floating ← IO.ofExcept (DType.realFloating? dtype)
  let _ := floating.down
  let std := Float.sqrt (2.0 / hidden_size.toFloat)
  let w1 ← randomWeight (dtype := dtype) (device := device) #[mlp_hidden * 2, hidden_size]
  let w2 ← randomWeight (dtype := dtype) (device := device) #[hidden_size, mlp_hidden]
  pure {
    w1 := (← Tensor.parameter (Tensor.mulScalar w1 std))
    w2 := (← Tensor.parameter (Tensor.mulScalar w2 std))
  }

/-- Initialize double-stream block -/
def init (hidden_size num_heads head_dim mlp_hidden : Nat)
    (device : Device := .CPU) (dtype : DType := .Float32) : IO (DoubleStreamBlock (device := device) (dtype := dtype) hidden_size num_heads head_dim mlp_hidden) := do
  let floating ← IO.ofExcept (DType.realFloating? dtype)
  let _ := floating.down
  let img_norm1 ← initLayerNorm (dtype := dtype) (device := device) hidden_size
  let img_attn ← initSelfAttention (dtype := dtype) (device := device) hidden_size num_heads head_dim
  let img_norm2 ← initLayerNorm (dtype := dtype) (device := device) hidden_size
  let img_mlp ← initMLP (dtype := dtype) (device := device) hidden_size mlp_hidden

  let txt_norm1 ← initLayerNorm (dtype := dtype) (device := device) hidden_size
  let txt_attn ← initSelfAttention (dtype := dtype) (device := device) hidden_size num_heads head_dim
  let txt_norm2 ← initLayerNorm (dtype := dtype) (device := device) hidden_size
  let txt_mlp ← initMLP (dtype := dtype) (device := device) hidden_size mlp_hidden

  pure {
    img_norm1, img_attn, img_norm2, img_mlp
    txt_norm1, txt_attn, txt_norm2, txt_mlp
  }

/-- Forward pass for double-stream block.
    img: [batch, img_seq, hidden_size]
    txt: [batch, txt_seq, hidden_size]
    vec: [batch, hidden_size] - timestep embedding
    pe: RoPE embeddings
    Returns: (img_out, txt_out) -/
def forward {batch img_seq txt_seq hidden_size num_heads head_dim mlp_hidden : Nat}
    (block : DoubleStreamBlock (device := device) (dtype := dtype) hidden_size num_heads head_dim mlp_hidden)
    (img : Tensor { dtype := dtype, device := device, shape := #[batch, img_seq, hidden_size] })
    (txt : Tensor { dtype := dtype, device := device, shape := #[batch, txt_seq, hidden_size] })
    (img_pe : Tensor { device := device, shape := #[batch, 1, img_seq, head_dim / 2, 2, 2], dtype := .Float32 })
    (txt_pe : Tensor { device := device, shape := #[batch, 1, txt_seq, head_dim / 2, 2, 2], dtype := .Float32 })  -- RoPE for image and text
    (mod_img : Tensor { dtype := dtype, device := device, shape := #[batch, 6, hidden_size] })
    (mod_txt : Tensor { dtype := dtype, device := device, shape := #[batch, 6, hidden_size] })
    : Tensor { dtype := dtype, device := device, shape := #[batch, img_seq, hidden_size] } × Tensor { dtype := dtype, device := device, shape := #[batch, txt_seq, hidden_size] } :=
  -- Split modulation into two triplets for attention and MLP
  let img_mod1 := sliceTyped mod_img 1 0 3
  let img_mod2 := sliceTyped mod_img 1 3 3
  let txt_mod1 := sliceTyped mod_txt 1 0 3
  let txt_mod2 := sliceTyped mod_txt 1 3 3

  -- Modulation order is (shift, scale, gate)
  let img_mod1_shift := Tensor.reshapeChecked (sliceTyped img_mod1 1 0 1) #[batch, hidden_size]
  let img_mod1_scale := Tensor.reshape (sliceTyped img_mod1 1 1 1) #[batch, hidden_size]
  let img_mod1_gate := Tensor.reshape (sliceTyped img_mod1 1 2 1) #[batch, hidden_size]
  let img_mod2_shift := Tensor.reshape (sliceTyped img_mod2 1 0 1) #[batch, hidden_size]
  let img_mod2_scale := Tensor.reshape (sliceTyped img_mod2 1 1 1) #[batch, hidden_size]
  let img_mod2_gate := Tensor.reshape (sliceTyped img_mod2 1 2 1) #[batch, hidden_size]

  let txt_mod1_shift := Tensor.reshape (sliceTyped txt_mod1 1 0 1) #[batch, hidden_size]
  let txt_mod1_scale := Tensor.reshape (sliceTyped txt_mod1 1 1 1) #[batch, hidden_size]
  let txt_mod1_gate := Tensor.reshape (sliceTyped txt_mod1 1 2 1) #[batch, hidden_size]
  let txt_mod2_shift := Tensor.reshape (sliceTyped txt_mod2 1 0 1) #[batch, hidden_size]
  let txt_mod2_scale := Tensor.reshape (sliceTyped txt_mod2 1 1 1) #[batch, hidden_size]
  let txt_mod2_gate := Tensor.reshapeChecked (sliceTyped txt_mod2 1 2 1) #[batch, hidden_size]

  -- Image path: norm1 + modulation
  let img_norm := block.img_norm1.forward3d img
  let img_mod := applyModulation img_norm img_mod1_scale img_mod1_shift

  -- Text path: norm1 + modulation
  let txt_norm := block.txt_norm1.forward3d txt
  let txt_mod := applyModulation txt_norm txt_mod1_scale txt_mod1_shift

  -- Project to Q, K, V for both
  let img_qkv := Tensor.linear3d img_mod block.img_attn.qkv
  let txt_qkv := Tensor.linear3d txt_mod block.txt_attn.qkv

  -- Reshape QKV
  let img_qkv := Tensor.reshapeChecked img_qkv #[batch, img_seq, 3, num_heads, head_dim]
  let txt_qkv := Tensor.reshapeChecked txt_qkv #[batch, txt_seq, 3, num_heads, head_dim]

  let img_q := Tensor.reshapeChecked (sliceTyped img_qkv 2 0 1) #[batch, img_seq, num_heads, head_dim]
  let img_k := Tensor.reshapeChecked (sliceTyped img_qkv 2 1 1) #[batch, img_seq, num_heads, head_dim]
  let img_v := Tensor.reshapeChecked (sliceTyped img_qkv 2 2 1) #[batch, img_seq, num_heads, head_dim]

  let txt_q := Tensor.reshapeChecked (sliceTyped txt_qkv 2 0 1) #[batch, txt_seq, num_heads, head_dim]
  let txt_k := Tensor.reshapeChecked (sliceTyped txt_qkv 2 1 1) #[batch, txt_seq, num_heads, head_dim]
  let txt_v := Tensor.reshapeChecked (sliceTyped txt_qkv 2 2 1) #[batch, txt_seq, num_heads, head_dim]

  -- QK normalization
  let (img_q, img_k) := block.img_attn.norm.forward img_q img_k
  let (txt_q, txt_k) := block.txt_attn.norm.forward txt_q txt_k

  -- Apply RoPE
  let img_q := flux.applyRope img_q img_pe
  let img_k := flux.applyRope img_k img_pe
  let txt_q := flux.applyRope txt_q txt_pe
  let txt_k := flux.applyRope txt_k txt_pe

  -- Concatenate for joint attention
  -- q: [batch, img_seq + txt_seq, num_heads, head_dim]
  let q : Tensor { dtype := dtype, device := device, shape := #[batch, txt_seq + img_seq, num_heads, head_dim] } := qwen35.typedCat txt_q img_q 1
  let k : Tensor { dtype := dtype, device := device, shape := #[batch, txt_seq + img_seq, num_heads, head_dim] } := qwen35.typedCat txt_k img_k 1
  let v : Tensor { dtype := dtype, device := device, shape := #[batch, txt_seq + img_seq, num_heads, head_dim] } := qwen35.typedCat txt_v img_v 1

  -- Transpose for attention
  let q := transposeAttention q
  let k := transposeAttention k
  let v := transposeAttention v

  -- Joint attention
  let attn := Tensor.attentionGQA q k v false

  -- Transpose back
  let attn := fromAttention attn
  -- Split back to img and txt
  let txt_attn := sliceTyped attn 1 0 txt_seq
  let img_attn := sliceTyped attn 1 txt_seq img_seq
  let img_attn := Tensor.reshapeChecked img_attn #[batch, img_seq, num_heads * head_dim]
  let txt_attn := Tensor.reshapeChecked txt_attn #[batch, txt_seq, num_heads * head_dim]

  -- Output projections
  let img_attn := Tensor.linear3d img_attn block.img_attn.proj
  let txt_attn := Tensor.linear3d txt_attn block.txt_attn.proj

  -- Apply gates
  let img_mod1_gate := Tensor.unsqueeze img_mod1_gate 1
  let img_mod1_gate := Tensor.expand img_mod1_gate #[batch, img_seq, hidden_size]
  let txt_mod1_gate := Tensor.unsqueeze txt_mod1_gate 1
  let txt_mod1_gate := Tensor.expand txt_mod1_gate #[batch, txt_seq, hidden_size]

  -- Residual + gated attention
  let img := Tensor.add img (Tensor.mul img_mod1_gate img_attn)
  let txt := Tensor.add txt (Tensor.mul txt_mod1_gate txt_attn)

  -- MLP paths with second modulation
  let img_mlp_in := block.img_norm2.forward3d img
  let img_mlp_in := applyModulation img_mlp_in img_mod2_scale img_mod2_shift
  let img_mlp_proj := Tensor.linear3d img_mlp_in block.img_mlp.w1
  let img_mlp_proj := Tensor.reshapeChecked img_mlp_proj #[batch, img_seq, 2, mlp_hidden]
  let img_gate := Tensor.reshapeChecked (sliceTyped img_mlp_proj 2 0 1) #[batch, img_seq, mlp_hidden]
  let img_up := Tensor.reshapeChecked (sliceTyped img_mlp_proj 2 1 1) #[batch, img_seq, mlp_hidden]
  let img_mlp_out := Tensor.mul (Tensor.silu img_gate) img_up
  let img_mlp_out := Tensor.linear3d img_mlp_out block.img_mlp.w2
  let img_mod2_gate := Tensor.unsqueeze img_mod2_gate 1
  let img_mod2_gate := Tensor.expand img_mod2_gate #[batch, img_seq, hidden_size]
  let img := Tensor.add img (Tensor.mul img_mod2_gate img_mlp_out)

  let txt_mlp_in := block.txt_norm2.forward3d txt
  let txt_mlp_in := applyModulation txt_mlp_in txt_mod2_scale txt_mod2_shift
  let txt_mlp_proj := Tensor.linear3d txt_mlp_in block.txt_mlp.w1
  let txt_mlp_proj := Tensor.reshapeChecked txt_mlp_proj #[batch, txt_seq, 2, mlp_hidden]
  let txt_gate := Tensor.reshapeChecked (sliceTyped txt_mlp_proj 2 0 1) #[batch, txt_seq, mlp_hidden]
  let txt_up := Tensor.reshapeChecked (sliceTyped txt_mlp_proj 2 1 1) #[batch, txt_seq, mlp_hidden]
  let txt_mlp_out := Tensor.mul (Tensor.silu txt_gate) txt_up
  let txt_mlp_out := Tensor.linear3d txt_mlp_out block.txt_mlp.w2
  let txt_mod2_gate := Tensor.unsqueeze txt_mod2_gate 1
  let txt_mod2_gate := Tensor.expand txt_mod2_gate #[batch, txt_seq, hidden_size]
  let txt := Tensor.add txt (Tensor.mul txt_mod2_gate txt_mlp_out)

  (img, txt)

end DoubleStreamBlock

end torch.flux
