/-
  Tyr/Lib/Attention.lean

  Grouped-query attention with optional per-head query/key RMS norms, optional
  query/key/value biases and a rotary position embedding, and its incremental
  key/value cache.
-/
import Tyr.Core.Typed.Neural
import Tyr.Core.TensorStruct
import Tyr.Lib.Module.Core
import Tyr.Core.TensorStruct.Derive

namespace torch

variable {device : Device}
/-- Attention projections with `num_kv_heads` key/value heads, each shared by
    `num_heads / num_kv_heads` query heads. -/
structure GroupedQueryAttention (hidden_size num_heads num_kv_heads head_dim : Nat) (dtype : DType := .Float32) where
  q_proj : Tensor { device := device, shape := #[num_heads * head_dim, hidden_size], dtype := dtype }
  k_proj : Tensor { device := device, shape := #[num_kv_heads * head_dim, hidden_size], dtype := dtype }
  v_proj : Tensor { device := device, shape := #[num_kv_heads * head_dim, hidden_size], dtype := dtype }
  o_proj : Tensor { device := device, shape := #[hidden_size, num_heads * head_dim], dtype := dtype }
  /-- Optional per-head RMS norm weights of queries and keys. -/
  q_norm : Option (Tensor { device := device, shape := #[head_dim], dtype := dtype }) := none
  k_norm : Option (Tensor { device := device, shape := #[head_dim], dtype := dtype }) := none
  /-- Optional Q/K/V projection biases. -/
  q_bias : Option (Tensor { device := device, shape := #[num_heads * head_dim], dtype := dtype }) := none
  k_bias : Option (Tensor { device := device, shape := #[num_kv_heads * head_dim], dtype := dtype }) := none
  v_bias : Option (Tensor { device := device, shape := #[num_kv_heads * head_dim], dtype := dtype }) := none
  deriving TensorStruct

namespace GroupedQueryAttention

variable {dtype : DType}
variable [DType.RealFloating dtype]

/-- Projections drawn from a normal distribution scaled by `sqrt (2 / hidden_size)`,
    without norms or biases. -/
def init (hidden_size num_heads num_kv_heads head_dim : Nat) (dtype : DType := .Float32) (device : Device := .CPU) : IO (GroupedQueryAttention (device := device) hidden_size num_heads num_kv_heads head_dim dtype) := do
  let floating ← IO.ofExcept (DType.realFloating? dtype)
  let _ := floating.down
  let std := Float.sqrt (2.0 / hidden_size.toFloat)

  let q ← Tensor.randn (device := device) #[num_heads * head_dim, hidden_size]
  let q := q.castDType dtype
  let k ← Tensor.randn (device := device) #[num_kv_heads * head_dim, hidden_size]
  let k := k.castDType dtype
  let v ← Tensor.randn (device := device) #[num_kv_heads * head_dim, hidden_size]
  let v := v.castDType dtype
  let o ← Tensor.randn (device := device) #[hidden_size, num_heads * head_dim]
  let o := o.castDType dtype

  pure {
    q_proj := (← Tensor.parameter (Tensor.mulScalar q std))
    k_proj := (← Tensor.parameter (Tensor.mulScalar k std))
    v_proj := (← Tensor.parameter (Tensor.mulScalar v std))
    o_proj := (← Tensor.parameter (Tensor.mulScalar o std))
  }

/-- Incremental KV cache for one attention layer: K/V buffers of capacity `cap`
    whose first `len` slots hold the tokens seen so far. -/
structure KVCache (batch num_kv_heads head_dim cap len : Nat) (dtype : DType := .Float32) where
  kStore : Tensor { device := device, shape := #[batch, num_kv_heads, cap, head_dim], dtype := dtype }
  vStore : Tensor { device := device, shape := #[batch, num_kv_heads, cap, head_dim], dtype := dtype }
  cap_pos : 0 < cap
  len_le : len ≤ cap

/-- Empty static KV cache with fixed capacity `maxLen`. -/
def initKVCache {batch num_kv_heads head_dim : Nat}
    (maxLen : Nat)
    (device : Device := Device.CPU) (dtype : DType := .Float32)
    [DType.ConstantSupported dtype device] (h : 0 < maxLen := by omega) : KVCache (device := device) batch num_kv_heads head_dim maxLen 0 dtype :=
  let k0 : Tensor { device := device, shape := #[batch, num_kv_heads, maxLen, head_dim], dtype := dtype } :=
    Tensor.zerosFor ⟨#[batch, num_kv_heads, maxLen, head_dim], dtype, device⟩
  let v0 : Tensor { device := device, shape := #[batch, num_kv_heads, maxLen, head_dim], dtype := dtype } :=
    Tensor.zerosFor ⟨#[batch, num_kv_heads, maxLen, head_dim], dtype, device⟩
  { kStore := k0, vStore := v0, cap_pos := h, len_le := Nat.zero_le _ }

private def project {batch seq inDim outDim : Nat}
    (x : Tensor { device := device, shape := #[batch, seq, inDim], dtype := dtype })
    (weight : Tensor { device := device, shape := #[outDim, inDim], dtype := dtype })
    (bias : Option (Tensor { device := device, shape := #[outDim], dtype := dtype }))
    : Tensor { device := device, shape := #[batch, seq, outDim], dtype := dtype } :=
  match bias with
  | some b => Tensor.affine3d x weight b
  | none => Tensor.linear3d x weight

/-- [batch, seq, n_head, head_dim] → [batch, n_head, seq, head_dim] (contiguous). -/
private def transposeForAttention {batch seq n_head head_dim : Nat} {d : DType} {dev : Device}
    (x : Tensor ⟨#[batch, seq, n_head, head_dim], d, dev⟩)
    : Tensor ⟨#[batch, n_head, seq, head_dim], d, dev⟩ :=
  Tensor.splitHeads x

/-- [batch, n_head, seq, head_dim] → [batch, seq, n_head, head_dim] (contiguous). -/
private def transposeFromAttention {batch n_head seq head_dim : Nat} {d : DType} {dev : Device}
    (x : Tensor ⟨#[batch, n_head, seq, head_dim], d, dev⟩)
    : Tensor ⟨#[batch, seq, n_head, head_dim], d, dev⟩ :=
  Tensor.joinHeads x

/-- Scaled dot-product attention with GQA; output shares the query spec. -/
private def sdpaGQA {batch n_head n_kv_head seq head_dim : Nat} {d : DType} {dev : Device} [DType.RealFloating d]
    (hkv : n_kv_head ∣ n_head)
    (query : Tensor ⟨#[batch, n_head, seq, head_dim], d, dev⟩)
    (key : Tensor ⟨#[batch, n_kv_head, seq, head_dim], d, dev⟩)
    (value : Tensor ⟨#[batch, n_kv_head, seq, head_dim], d, dev⟩)
    (isCausal : Bool)
    : Tensor ⟨#[batch, n_head, seq, head_dim], d, dev⟩ :=
  Tensor.attentionGQA query key value isCausal true (_heads := .ofDvd hkv)

/-- SDPA with GQA where query and KV sequence lengths differ. -/
private def sdpaGQAQKV {batch n_head n_kv_head q_seq kv_seq head_dim : Nat} {d : DType} {dev : Device} [DType.RealFloating d]
    (hkv : n_kv_head ∣ n_head)
    (query : Tensor ⟨#[batch, n_head, q_seq, head_dim], d, dev⟩)
    (key : Tensor ⟨#[batch, n_kv_head, kv_seq, head_dim], d, dev⟩)
    (value : Tensor ⟨#[batch, n_kv_head, kv_seq, head_dim], d, dev⟩)
    (isCausal : Bool)
    : Tensor ⟨#[batch, n_head, q_seq, head_dim], d, dev⟩ :=
  Tensor.attentionGQAQKV query key value isCausal true (_heads := .ofDvd hkv)

/-- SDPA with GQA and a [batch, seq] padding mask of any dtype. -/
private def sdpaGQAMask {batch n_head n_kv_head seq head_dim : Nat} {d dm : DType} {dev : Device} [DType.RealFloating d]
    (hkv : n_kv_head ∣ n_head)
    (query : Tensor ⟨#[batch, n_head, seq, head_dim], d, dev⟩)
    (key : Tensor ⟨#[batch, n_kv_head, seq, head_dim], d, dev⟩)
    (value : Tensor ⟨#[batch, n_kv_head, seq, head_dim], d, dev⟩)
    (mask : Tensor ⟨#[batch, seq], dm, dev⟩)
    (isCausal : Bool)
    : Tensor ⟨#[batch, n_head, seq, head_dim], d, dev⟩ :=
  Tensor.attentionGQAMask query key value mask isCausal true (_heads := .ofDvd hkv)

/-- `sliceScatter`: replace a slice of `input` along `dim` at `start` with
    `src`; the result preserves the input spec. -/
private def sliceScatterAlong {s srcShape : Shape} {d : DType} {dev : Device}
    (input : Tensor ⟨s, d, dev⟩) (dim : Nat) (start : Nat)
    (src : Tensor ⟨srcShape, d, dev⟩)
    (h : Shape.SliceScatterCompatible s srcShape dim := by
      first | exact of_decide_eq_true rfl | (simp <;> fail "incompatible slice scatter shapes"))
    (bound : start + srcShape.getD dim 0 ≤ s.getD dim 0 := by
      first | exact of_decide_eq_true rfl | omega | (simp <;> omega)) :
    Tensor ⟨s, d, dev⟩ :=
  Tensor.sliceScatter input dim start src h bound

/-- Apply RMS normalization along the last dimension (per-head norm).
    Input: [batch, seq, n_heads, head_dim], norm: [head_dim]
    Broadcasts norm across all heads. -/
private def applyHeadNorm {batch seq n_heads head_dim : Nat}
    (x : Tensor { device := device, shape := #[batch, seq, n_heads, head_dim], dtype := dtype })
    (norm : Tensor { device := device, shape := #[head_dim], dtype := dtype })
    (eps : Float := 1e-6)
    : Tensor { device := device, shape := #[batch, seq, n_heads, head_dim], dtype := dtype } :=
  Tensor.rmsNormPreserving x norm eps

/-- Projects `x`, applies the optional head norms and the rotary embedding
    `cos`/`sin` to queries and keys, attends, and projects back. -/
def forward {batch seq hidden_size num_heads num_kv_heads half : Nat}
    (attn : GroupedQueryAttention (device := device) hidden_size num_heads num_kv_heads (2 * half) dtype)
    (hkv : num_kv_heads ∣ num_heads)
    (x : Tensor { device := device, shape := #[batch, seq, hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, half], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, half], dtype := .Float32 })
    (is_causal : Bool := true)
    : Tensor { device := device, shape := #[batch, seq, hidden_size], dtype := dtype } :=
  -- Project to Q, K, V
  let q := project x attn.q_proj attn.q_bias  -- [batch, seq, num_heads * head_dim]
  let k := project x attn.k_proj attn.k_bias  -- [batch, seq, num_kv_heads * head_dim]
  let v := project x attn.v_proj attn.v_bias  -- [batch, seq, num_kv_heads * head_dim]

  -- Reshape to [batch, seq, num_heads, head_dim]
  let q := q.reshape #[batch, seq, num_heads, 2 * half]
  let k := k.reshape #[batch, seq, num_kv_heads, 2 * half]
  let v := v.reshape #[batch, seq, num_kv_heads, 2 * half]

  let q := match attn.q_norm with
    | some qn => applyHeadNorm q qn
    | none => q
  let k := match attn.k_norm with
    | some kn => applyHeadNorm k kn
    | none => k

  -- Apply RoPE to Q and K
  let q := Tensor.applyRotary q cos sin
  let k := Tensor.applyRotary k cos sin

  -- Transpose to [batch, num_heads, seq, head_dim]
  let q := transposeForAttention q
  let k := transposeForAttention k
  let v := transposeForAttention v

  -- Scaled dot-product attention with GQA
  let attn_out := sdpaGQA hkv q k v is_causal

  -- Transpose back to [batch, seq, num_heads, head_dim]
  let attn_out := transposeFromAttention attn_out

  -- Reshape to [batch, seq, hidden_size]
  let attn_out := attn_out.reshape #[batch, seq, num_heads * (2 * half)]

  -- Output projection
  Tensor.linear3d attn_out attn.o_proj

/-- `forward` for one new position: its keys and values are written into `cache`
    at `len`, and the query attends to the first `len + 1` cached positions. -/
def forwardStep {batch hidden_size num_heads num_kv_heads half cap len : Nat}
    (attn : GroupedQueryAttention (device := device) hidden_size num_heads num_kv_heads (2 * half) dtype)
    (hkv : num_kv_heads ∣ num_heads)
    (x : Tensor { device := device, shape := #[batch, 1, hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[1, half], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[1, half], dtype := .Float32 })
    (cache : KVCache (device := device) batch num_kv_heads (2 * half) cap len dtype)
    (hlen : len < cap)
    : Tensor { device := device, shape := #[batch, 1, hidden_size], dtype := dtype } × KVCache (device := device) batch num_kv_heads (2 * half) cap (len + 1) dtype :=
  -- Project current token to Q/K/V.
  let q0 := project x attn.q_proj attn.q_bias
  let k0 := project x attn.k_proj attn.k_bias
  let v0 := project x attn.v_proj attn.v_bias

  let q := q0.reshape #[batch, 1, num_heads, 2 * half]
  let k := k0.reshape #[batch, 1, num_kv_heads, 2 * half]
  let v := v0.reshape #[batch, 1, num_kv_heads, 2 * half]

  -- Apply optional per-head Q/K normalization.
  let q := match attn.q_norm with
    | some qn => applyHeadNorm q qn
    | none => q
  let k := match attn.k_norm with
    | some kn => applyHeadNorm k kn
    | none => k

  -- Apply RoPE to the single-step Q/K.
  let q := Tensor.applyRotary q cos sin
  let k := Tensor.applyRotary k cos sin

  -- Convert to attention layout and write new KV into preallocated cache.
  let qh := transposeForAttention q
  let kNew := transposeForAttention k
  let vNew := transposeForAttention v

  let kStore : Tensor { device := device, shape := #[batch, num_kv_heads, cap, 2 * half], dtype := dtype } :=
    Tensor.castLike kNew cache.kStore
  let vStore : Tensor { device := device, shape := #[batch, num_kv_heads, cap, 2 * half], dtype := dtype } :=
    Tensor.castLike vNew cache.vStore

  let kStore' := sliceScatterAlong kStore 2 len kNew
  let vStore' := sliceScatterAlong vStore 2 len vNew
  let kvLen : Nat := len + 1
  let kAll : Tensor { device := device, shape := #[batch, num_kv_heads, kvLen, 2 * half], dtype := dtype } :=
    kStore'.slice 2 0 kvLen
  let vAll : Tensor { device := device, shape := #[batch, num_kv_heads, kvLen, 2 * half], dtype := dtype } :=
    vStore'.slice 2 0 kvLen

  -- Use q_len=1, kv_len=(len+1). Causal masking is unnecessary because KV has no future tokens.
  let attnOut := sdpaGQAQKV hkv qh kAll vAll false
  let attnOut := transposeFromAttention attnOut
  let attnOut := attnOut.reshape #[batch, 1, num_heads * (2 * half)]
  let out := Tensor.linear3d attnOut attn.o_proj

  let cache' : KVCache (device := device) batch num_kv_heads (2 * half) cap (len + 1) dtype := {
    cap_pos := cache.cap_pos
    len_le := hlen
    kStore := kStore'
    vStore := vStore'
  }
  (out, cache')

/-- `forward` with a [batch, seq] padding mask. -/
def forwardMasked {batch seq hidden_size num_heads num_kv_heads half : Nat}
    {maskDtype : DType}
    (attn : GroupedQueryAttention (device := device) hidden_size num_heads num_kv_heads (2 * half) dtype)
    (hkv : num_kv_heads ∣ num_heads)
    (x : Tensor { device := device, shape := #[batch, seq, hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, half], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, half], dtype := .Float32 })
    (attn_mask : Tensor ⟨#[batch, seq], maskDtype, device⟩)
    (is_causal : Bool := true)
    : Tensor { device := device, shape := #[batch, seq, hidden_size], dtype := dtype } :=
  -- Project to Q, K, V
  let q := project x attn.q_proj attn.q_bias  -- [batch, seq, num_heads * head_dim]
  let k := project x attn.k_proj attn.k_bias  -- [batch, seq, num_kv_heads * head_dim]
  let v := project x attn.v_proj attn.v_bias  -- [batch, seq, num_kv_heads * head_dim]

  -- Reshape to [batch, seq, num_heads, head_dim]
  let q := q.reshape #[batch, seq, num_heads, 2 * half]
  let k := k.reshape #[batch, seq, num_kv_heads, 2 * half]
  let v := v.reshape #[batch, seq, num_kv_heads, 2 * half]

  let q := match attn.q_norm with
    | some qn => applyHeadNorm q qn
    | none => q
  let k := match attn.k_norm with
    | some kn => applyHeadNorm k kn
    | none => k

  -- Apply RoPE to Q and K
  let q := Tensor.applyRotary q cos sin
  let k := Tensor.applyRotary k cos sin

  -- Transpose to [batch, num_heads, seq, head_dim]
  let q := transposeForAttention q
  let k := transposeForAttention k
  let v := transposeForAttention v

  -- Scaled dot-product attention with GQA and padding mask
  let attn_out := sdpaGQAMask hkv q k v attn_mask is_causal

  -- Transpose back to [batch, seq, num_heads, head_dim]
  let attn_out := transposeFromAttention attn_out

  -- Reshape to [batch, seq, hidden_size]
  let attn_out := attn_out.reshape #[batch, seq, num_heads * (2 * half)]

  -- Output projection
  Tensor.linear3d attn_out attn.o_proj

end GroupedQueryAttention

end torch
