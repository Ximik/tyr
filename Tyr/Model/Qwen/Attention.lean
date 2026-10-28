/-
  Tyr/Model/Qwen/Attention.lean

  Grouped Query Attention (GQA) for Qwen3.
  Uses fewer key-value heads than query heads for efficiency.

  Indexed shapes and a shared dtype parameter preserve Float32 and BF16
  execution. Native attention operations live behind typed adapters.
-/
import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Qwen.Config
import Tyr.Model.Qwen.RoPE

namespace torch.qwen

variable {device : Device}
/-- Qwen attention with Grouped Query Attention (GQA).
    Query heads: num_attention_heads
    Key/Value heads: num_key_value_heads (typically 4 for Qwen3-4B)
    Each KV head is shared across multiple query heads. -/
structure QwenAttention (hidden_size num_heads num_kv_heads head_dim : Nat) (dtype : DType := .Float32) where
  /-- Query projection: [hidden_size, hidden_size] -/
  q_proj : Tensor { device := device, shape := #[num_heads * head_dim, hidden_size], dtype := dtype }
  /-- Key projection: [kv_dim, hidden_size] -/
  k_proj : Tensor { device := device, shape := #[num_kv_heads * head_dim, hidden_size], dtype := dtype }
  /-- Value projection: [kv_dim, hidden_size] -/
  v_proj : Tensor { device := device, shape := #[num_kv_heads * head_dim, hidden_size], dtype := dtype }
  /-- Output projection: [hidden_size, hidden_size] -/
  o_proj : Tensor { device := device, shape := #[hidden_size, num_heads * head_dim], dtype := dtype }
  /-- Optional Q norm (per-head normalization for Flux Klein text encoder) -/
  q_norm : Option (Tensor { device := device, shape := #[head_dim], dtype := dtype }) := none
  /-- Optional K norm (per-head normalization for Flux Klein text encoder) -/
  k_norm : Option (Tensor { device := device, shape := #[head_dim], dtype := dtype }) := none
  deriving TensorStruct

namespace QwenAttention

variable {dtype : DType}
variable [DType.RealFloating dtype]

/-- Initialize attention layers with random weights -/
def init (hidden_size num_heads num_kv_heads head_dim : Nat) (dtype : DType := .Float32) (device : Device := .CPU) : IO (QwenAttention (device := device) hidden_size num_heads num_kv_heads head_dim dtype) := do
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

/-- Incremental KV cache for one attention layer.
    Stores preallocated K/V buffers with capacity `maxLen`.

    `maxLen` is a runtime value (not a type index), so the stores are
    carried existentially over their shape and re-validated against the
    cache metadata by `expectStore` at every decode step — the typed
    adapter for rank changes without erasing the tensor specification. -/
structure KVCache (batch num_kv_heads head_dim : Nat) (dtype : DType := .Float32) where
  kStoreDyn : Σ (s : Shape), Tensor { device := device, shape := s, dtype := dtype }
  vStoreDyn : Σ (s : Shape), Tensor { device := device, shape := s, dtype := dtype }
  seq : Nat := 0
  maxLen : Nat := 0

/-- Empty static KV cache with fixed capacity `maxLen`. -/
def initKVCache {batch num_kv_heads head_dim : Nat}
    (maxLen : Nat)
    (device : Device := Device.CPU) (dtype : DType := .Float32)
    [DType.ConstantSupported dtype device] : KVCache (device := device) batch num_kv_heads head_dim dtype :=
  let k0 : Tensor { device := device, shape := #[batch, num_kv_heads, maxLen, head_dim], dtype := dtype } :=
    Tensor.zerosFor ⟨#[batch, num_kv_heads, maxLen, head_dim], dtype, device⟩
  let v0 : Tensor { device := device, shape := #[batch, num_kv_heads, maxLen, head_dim], dtype := dtype } :=
    Tensor.zerosFor ⟨#[batch, num_kv_heads, maxLen, head_dim], dtype, device⟩
  { kStoreDyn := ⟨_, k0⟩, vStoreDyn := ⟨_, v0⟩, seq := 0, maxLen := maxLen }

/-- Recover the statically-shaped store from its existential carrier,
    validating the runtime handle against the cache metadata. -/
private def expectStore {batch num_kv_heads head_dim : Nat}
    (store : Σ (s : Shape), Tensor { device := device, shape := s, dtype := dtype }) (maxLen : Nat)
    : Tensor { device := device, shape := #[batch, num_kv_heads, maxLen, head_dim], dtype := dtype } :=
  match Tensor.ofRaw store.2.raw with
  | .ok t => t
  | .error message => nativeInvariantFailure s!"QwenAttention.KVCache store: {message}"

/-! ## Extern adapters

Typed wrappers over existing attention externs that have no counterpart in
`Tyr.Typed.Ops` yet. Each re-asserts the output spec the extern is known to
produce (see `Tyr/Inference/KVCache.lean` for the same pattern). -/

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
private def sdpaGQA {batch n_head n_kv_head seq head_dim : Nat} {d : DType} {dev : Device}
    (query : Tensor ⟨#[batch, n_head, seq, head_dim], d, dev⟩)
    (key : Tensor ⟨#[batch, n_kv_head, seq, head_dim], d, dev⟩)
    (value : Tensor ⟨#[batch, n_kv_head, seq, head_dim], d, dev⟩)
    (isCausal : Bool) (enableGqa : Bool)
    : Tensor ⟨#[batch, n_head, seq, head_dim], d, dev⟩ :=
  Tensor.attentionGQA query key value isCausal enableGqa

/-- SDPA with GQA where query and KV sequence lengths differ. -/
private def sdpaGQAQKV {batch n_head n_kv_head q_seq kv_seq head_dim : Nat} {d : DType} {dev : Device}
    (query : Tensor ⟨#[batch, n_head, q_seq, head_dim], d, dev⟩)
    (key : Tensor ⟨#[batch, n_kv_head, kv_seq, head_dim], d, dev⟩)
    (value : Tensor ⟨#[batch, n_kv_head, kv_seq, head_dim], d, dev⟩)
    (isCausal : Bool) (enableGqa : Bool)
    : Tensor ⟨#[batch, n_head, q_seq, head_dim], d, dev⟩ :=
  Tensor.attentionGQAQKV query key value isCausal enableGqa

/-- SDPA with GQA and a [batch, seq] padding mask. The mask dtype stays
    generic: the extern accepts any runtime dtype and the legacy callers
    were dtype-erased here. -/
private def sdpaGQAMask {batch n_head n_kv_head seq head_dim : Nat} {d dm : DType} {dev : Device}
    (query : Tensor ⟨#[batch, n_head, seq, head_dim], d, dev⟩)
    (key : Tensor ⟨#[batch, n_kv_head, seq, head_dim], d, dev⟩)
    (value : Tensor ⟨#[batch, n_kv_head, seq, head_dim], d, dev⟩)
    (mask : Tensor ⟨#[batch, seq], dm, dev⟩)
    (isCausal : Bool) (enableGqa : Bool)
    : Tensor ⟨#[batch, n_head, seq, head_dim], d, dev⟩ :=
  Tensor.attentionGQAMask query key value mask isCausal enableGqa

/-- Experimental Tyr FlashAttention entry point; output shares the query spec. -/
private def flashAttn4d {batch nHead nKvHead qSeq kvSeq headDim : Nat} {d : DType} {dev : Device}
    (query : Tensor ⟨#[batch, nHead, qSeq, headDim], d, dev⟩)
    (key : Tensor ⟨#[batch, nKvHead, kvSeq, headDim], d, dev⟩)
    (value : Tensor ⟨#[batch, nKvHead, kvSeq, headDim], d, dev⟩)
    (enableGqa : Bool) : Tensor ⟨#[batch, nHead, qSeq, headDim], d, dev⟩ :=
  Tensor.flashAttention query key value enableGqa

/-- `sliceScatter`: replace a slice of `input` along `dim` at `start` with
    `src`; the result preserves the input spec. -/
private def sliceScatterAlong {s srcShape : Shape} {d : DType} {dev : Device}
    (input : Tensor ⟨s, d, dev⟩) (dim : Nat) (start : Nat)
    (src : Tensor ⟨srcShape, d, dev⟩) : Tensor ⟨s, d, dev⟩ :=
  Tensor.sliceScatter input dim start src

/-- Mean over one axis with the result shape validated at runtime. -/
private def meanDimTo {s : Shape} {d : DType} {dev : Device}
    (t : Tensor ⟨s, d, dev⟩) (dim : Nat) (out : Shape) (keepdim : Bool := false)
    : Tensor ⟨out, d, dev⟩ :=
  match Tensor.ofRaw (RawTensor.mean t.raw (some #[dim]) keepdim) with
  | .ok r => r
  | .error message => nativeInvariantFailure s!"QwenAttention.meanDim: {message}"

/-- Apply RMS normalization along the last dimension (per-head norm).
    Input: [batch, seq, n_heads, head_dim], norm: [head_dim]
    Broadcasts norm across all heads. -/
private def applyHeadNorm {batch seq n_heads head_dim : Nat}
    (x : Tensor { device := device, shape := #[batch, seq, n_heads, head_dim], dtype := dtype })
    (norm : Tensor { device := device, shape := #[head_dim], dtype := dtype })
    (eps : Float := 1e-6)
    : Tensor { device := device, shape := #[batch, seq, n_heads, head_dim], dtype := dtype } :=
  Tensor.rmsNormPreserving x norm eps

/-- Forward pass for attention.
    Input: [batch, seq, hidden_size]
    Output: [batch, seq, hidden_size] -/
def forward {batch seq hidden_size num_heads num_kv_heads head_dim : Nat}
    (attn : QwenAttention (device := device) hidden_size num_heads num_kv_heads head_dim dtype)
    (x : Tensor { device := device, shape := #[batch, seq, hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, head_dim / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, head_dim / 2], dtype := .Float32 })
    (is_causal : Bool := true)
    : Tensor { device := device, shape := #[batch, seq, hidden_size], dtype := dtype } :=
  -- Project to Q, K, V
  let q := Tensor.linear3d x attn.q_proj  -- [batch, seq, num_heads * head_dim]
  let k := Tensor.linear3d x attn.k_proj  -- [batch, seq, num_kv_heads * head_dim]
  let v := Tensor.linear3d x attn.v_proj  -- [batch, seq, num_kv_heads * head_dim]

  -- Reshape to [batch, seq, num_heads, head_dim]
  let q := q.reshape #[batch, seq, num_heads, head_dim]
  let k := k.reshape #[batch, seq, num_kv_heads, head_dim]
  let v := v.reshape #[batch, seq, num_kv_heads, head_dim]

  -- Apply Q/K norms if present (Flux Klein text encoder has these)
  let q := match attn.q_norm with
    | some qn => applyHeadNorm q qn
    | none => q
  let k := match attn.k_norm with
    | some kn => applyHeadNorm k kn
    | none => k

  -- Apply RoPE to Q and K
  let q := applyRotaryEmb q cos sin
  let k := applyRotaryEmb k cos sin

  -- Transpose to [batch, num_heads, seq, head_dim]
  let q := transposeForAttention q
  let k := transposeForAttention k
  let v := transposeForAttention v

  -- Scaled dot-product attention with GQA
  let attn_out := sdpaGQA q k v is_causal true

  -- Transpose back to [batch, seq, num_heads, head_dim]
  let attn_out := transposeFromAttention attn_out

  -- Reshape to [batch, seq, hidden_size]
  let attn_out := attn_out.reshape #[batch, seq, num_heads * head_dim]

  -- Output projection
  Tensor.linear3d attn_out attn.o_proj

/-- Incremental attention step with KV cache.
    Input query is one token `[batch,1,hidden]`; cache grows by one KV step.

    When `useTyrFlashAttn` is true (default false), routes the cached
    attention through `Tensor.flashAttention` instead of `scaledDotProductAttentionGQAQKV`.
    The C++ `tyr::flash_attn` dispatcher then routes shape-eligible decode
    problems (BF16 + qSeq=1 + head_dim=128 or 64 + GQA-valid) to the native
    TK kernel `tkMhaH100DecodeFwd[64]`, and falls back to PyTorch SDPA
    otherwise — so this flag is safe to flip on for any shape. -/
def forwardStep {batch hidden_size num_heads num_kv_heads head_dim : Nat}
    (attn : QwenAttention (device := device) hidden_size num_heads num_kv_heads head_dim dtype)
    (x : Tensor { device := device, shape := #[batch, 1, hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[1, head_dim / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[1, head_dim / 2], dtype := .Float32 })
    (cache : KVCache (device := device) batch num_kv_heads head_dim dtype)
    (useTyrFlashAttn : Bool := false)
    : Tensor { device := device, shape := #[batch, 1, hidden_size], dtype := dtype } × KVCache (device := device) batch num_kv_heads head_dim dtype :=
  -- Project current token to Q/K/V.
  let q0 := Tensor.linear3d x attn.q_proj
  let k0 := Tensor.linear3d x attn.k_proj
  let v0 := Tensor.linear3d x attn.v_proj

  let q := q0.reshape #[batch, 1, num_heads, head_dim]
  let k := k0.reshape #[batch, 1, num_kv_heads, head_dim]
  let v := v0.reshape #[batch, 1, num_kv_heads, head_dim]

  -- Apply optional per-head Q/K normalization.
  let q := match attn.q_norm with
    | some qn => applyHeadNorm q qn
    | none => q
  let k := match attn.k_norm with
    | some kn => applyHeadNorm k kn
    | none => k

  -- Apply RoPE to the single-step Q/K.
  let q := applyRotaryEmb q cos sin
  let k := applyRotaryEmb k cos sin

  -- Convert to attention layout and write new KV into preallocated cache.
  let qh := transposeForAttention q
  let kNew := transposeForAttention k
  let vNew := transposeForAttention v

  let kStore : Tensor { device := device, shape := #[batch, num_kv_heads, cache.maxLen, head_dim], dtype := dtype } :=
    Tensor.castLike kNew (expectStore cache.kStoreDyn cache.maxLen)
  let vStore : Tensor { device := device, shape := #[batch, num_kv_heads, cache.maxLen, head_dim], dtype := dtype } :=
    Tensor.castLike vNew (expectStore cache.vStoreDyn cache.maxLen)

  if cache.seq < cache.maxLen then
    let kStore' := sliceScatterAlong kStore 2 cache.seq kNew
    let vStore' := sliceScatterAlong vStore 2 cache.seq vNew
    let kvLen : Nat := cache.seq + 1
    let kAll := kStore'.sliceChecked 2 0 kvLen #[batch, num_kv_heads, kvLen, head_dim]
    let vAll := vStore'.sliceChecked 2 0 kvLen #[batch, num_kv_heads, kvLen, head_dim]

    -- Use q_len=1, kv_len=(seq+1). Causal masking is unnecessary because KV has no future tokens.
    let attnOut :=
      if useTyrFlashAttn then
        flashAttn4d qh kAll vAll true
      else
        sdpaGQAQKV qh kAll vAll false true
    let attnOut := transposeFromAttention attnOut
    let attnOut := attnOut.reshape #[batch, 1, num_heads * head_dim]
    let out := Tensor.linear3d attnOut attn.o_proj

    let cache' : KVCache (device := device) batch num_kv_heads head_dim dtype := {
      kStoreDyn := ⟨_, kStore'⟩
      vStoreDyn := ⟨_, vStore'⟩
      seq := kvLen
      maxLen := cache.maxLen
    }
    (out, cache')
  else
    let writePos : Nat :=
      if cache.maxLen == 0 then
        0
      else
        cache.maxLen - 1
    let kStore' := sliceScatterAlong kStore 2 writePos kNew
    let vStore' := sliceScatterAlong vStore 2 writePos vNew
    let kvLen : Nat := cache.maxLen
    let kAll := kStore'.sliceChecked 2 0 kvLen #[batch, num_kv_heads, kvLen, head_dim]
    let vAll := vStore'.sliceChecked 2 0 kvLen #[batch, num_kv_heads, kvLen, head_dim]
    let attnOut :=
      if useTyrFlashAttn then
        flashAttn4d qh kAll vAll true
      else
        sdpaGQAQKV qh kAll vAll false true
    let attnOut := transposeFromAttention attnOut
    let attnOut := attnOut.reshape #[batch, 1, num_heads * head_dim]
    let out := Tensor.linear3d attnOut attn.o_proj
    let cache' : KVCache (device := device) batch num_kv_heads head_dim dtype := {
      kStoreDyn := ⟨_, kStore'⟩
      vStoreDyn := ⟨_, vStore'⟩
      seq := cache.maxLen
      maxLen := cache.maxLen
    }
    (out, cache')

def forwardMasked {batch seq hidden_size num_heads num_kv_heads head_dim : Nat}
    {maskDtype : DType}
    (attn : QwenAttention (device := device) hidden_size num_heads num_kv_heads head_dim dtype)
    (x : Tensor { device := device, shape := #[batch, seq, hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, head_dim / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, head_dim / 2], dtype := .Float32 })
    (attn_mask : Tensor ⟨#[batch, seq], maskDtype, device⟩)
    (is_causal : Bool := true)
    : Tensor { device := device, shape := #[batch, seq, hidden_size], dtype := dtype } :=
  -- Project to Q, K, V
  let q := Tensor.linear3d x attn.q_proj  -- [batch, seq, num_heads * head_dim]
  let k := Tensor.linear3d x attn.k_proj  -- [batch, seq, num_kv_heads * head_dim]
  let v := Tensor.linear3d x attn.v_proj  -- [batch, seq, num_kv_heads * head_dim]

  -- Reshape to [batch, seq, num_heads, head_dim]
  let q := q.reshape #[batch, seq, num_heads, head_dim]
  let k := k.reshape #[batch, seq, num_kv_heads, head_dim]
  let v := v.reshape #[batch, seq, num_kv_heads, head_dim]

  -- Apply Q/K norms if present (Flux Klein text encoder has these)
  let q := match attn.q_norm with
    | some qn => applyHeadNorm q qn
    | none => q
  let k := match attn.k_norm with
    | some kn => applyHeadNorm k kn
    | none => k

  -- Apply RoPE to Q and K
  let q := applyRotaryEmb q cos sin
  let k := applyRotaryEmb k cos sin

  -- Transpose to [batch, num_heads, seq, head_dim]
  let q := transposeForAttention q
  let k := transposeForAttention k
  let v := transposeForAttention v

  -- Scaled dot-product attention with GQA and padding mask
  let attn_out := sdpaGQAMask q k v attn_mask is_causal true

  -- Transpose back to [batch, seq, num_heads, head_dim]
  let attn_out := transposeFromAttention attn_out

  -- Reshape to [batch, seq, hidden_size]
  let attn_out := attn_out.reshape #[batch, seq, num_heads * head_dim]

  -- Output projection
  Tensor.linear3d attn_out attn.o_proj

end QwenAttention

end torch.qwen
