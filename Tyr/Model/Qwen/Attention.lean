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

/-- Qwen attention with Grouped Query Attention (GQA).
    Query heads: num_attention_heads
    Key/Value heads: num_key_value_heads (typically 4 for Qwen3-4B)
    Each KV head is shared across multiple query heads. -/
structure QwenAttention (hidden_size num_heads num_kv_heads head_dim : UInt64) (dtype : DType := .Float32) where
  /-- Query projection: [hidden_size, hidden_size] -/
  q_proj : DTensor #[num_heads * head_dim, hidden_size] dtype
  /-- Key projection: [kv_dim, hidden_size] -/
  k_proj : DTensor #[num_kv_heads * head_dim, hidden_size] dtype
  /-- Value projection: [kv_dim, hidden_size] -/
  v_proj : DTensor #[num_kv_heads * head_dim, hidden_size] dtype
  /-- Output projection: [hidden_size, hidden_size] -/
  o_proj : DTensor #[hidden_size, num_heads * head_dim] dtype
  /-- Optional Q norm (per-head normalization for Flux Klein text encoder) -/
  q_norm : Option (DTensor #[head_dim] dtype) := none
  /-- Optional K norm (per-head normalization for Flux Klein text encoder) -/
  k_norm : Option (DTensor #[head_dim] dtype) := none
  deriving TensorStruct

namespace QwenAttention

variable {dtype : DType}

/-- Initialize attention layers with random weights -/
def init (hidden_size num_heads num_kv_heads head_dim : UInt64) (dtype : DType := .Float32) : IO (QwenAttention hidden_size num_heads num_kv_heads head_dim dtype) := do
  let std := Float.sqrt (2.0 / hidden_size.toFloat)

  let q ← Tensor.randn #[num_heads * head_dim, hidden_size]
  let q := q.castDType dtype
  let k ← Tensor.randn #[num_kv_heads * head_dim, hidden_size]
  let k := k.castDType dtype
  let v ← Tensor.randn #[num_kv_heads * head_dim, hidden_size]
  let v := v.castDType dtype
  let o ← Tensor.randn #[hidden_size, num_heads * head_dim]
  let o := o.castDType dtype

  pure {
    q_proj := Tensor.setRequiresGrad (Tensor.Preserving.mulScalar q std) true
    k_proj := Tensor.setRequiresGrad (Tensor.Preserving.mulScalar k std) true
    v_proj := Tensor.setRequiresGrad (Tensor.Preserving.mulScalar v std) true
    o_proj := Tensor.setRequiresGrad (Tensor.Preserving.mulScalar o std) true
  }

/-- Incremental KV cache for one attention layer.
    Stores preallocated K/V buffers with capacity `maxLen`.

    `maxLen` is a runtime value (not a type index), so the stores are
    carried existentially over their shape and re-validated against the
    cache metadata by `expectStore` at every decode step — the typed
    equivalent of the legacy `T #[]` erase-then-reshape dance. -/
structure KVCache (batch num_kv_heads head_dim : UInt64) (dtype : DType := .Float32) where
  kStoreDyn : Σ (s : Shape), DTensor s dtype
  vStoreDyn : Σ (s : Shape), DTensor s dtype
  seq : UInt64 := 0
  maxLen : UInt64 := 0

/-- Empty static KV cache with fixed capacity `maxLen`. -/
def initKVCache {batch num_kv_heads head_dim : UInt64}
    (maxLen : UInt64)
    (device : Device := Device.CPU) (dtype : DType := .Float32) : KVCache batch num_kv_heads head_dim dtype :=
  let k0 : DTensor #[batch, num_kv_heads, maxLen, head_dim] dtype :=
    (Tensor.zeros #[batch, num_kv_heads, maxLen, head_dim] false device).castDType dtype
  let v0 : DTensor #[batch, num_kv_heads, maxLen, head_dim] dtype :=
    (Tensor.zeros #[batch, num_kv_heads, maxLen, head_dim] false device).castDType dtype
  { kStoreDyn := ⟨_, k0⟩, vStoreDyn := ⟨_, v0⟩, seq := 0, maxLen := maxLen }

/-- Recover the statically-shaped store from its existential carrier,
    validating the runtime handle against the cache metadata. -/
private def expectStore {batch num_kv_heads head_dim : UInt64}
    (store : Σ (s : Shape), DTensor s dtype) (maxLen : UInt64)
    : DTensor #[batch, num_kv_heads, maxLen, head_dim] dtype :=
  match Tensor.ofRaw store.2.raw with
  | .ok t => t
  | .error message => panic! s!"QwenAttention.KVCache store: {message}"

/-! ## Extern adapters

Typed wrappers over existing attention externs that have no counterpart in
`Tyr.Typed.Ops` yet. Each re-asserts the output spec the extern is known to
produce (see `Tyr/Inference/KVCache.lean` for the same pattern). -/

/-- [batch, seq, n_head, head_dim] → [batch, n_head, seq, head_dim] (contiguous). -/
private def transposeForAttention {batch seq n_head head_dim : UInt64} {d : DType} {dev : DevicePolicy}
    (x : Tensor ⟨#[batch, seq, n_head, head_dim], d, dev⟩)
    : Tensor ⟨#[batch, n_head, seq, head_dim], d, dev⟩ :=
  Tensor.splitHeads x

/-- [batch, n_head, seq, head_dim] → [batch, seq, n_head, head_dim] (contiguous). -/
private def transposeFromAttention {batch n_head seq head_dim : UInt64} {d : DType} {dev : DevicePolicy}
    (x : Tensor ⟨#[batch, n_head, seq, head_dim], d, dev⟩)
    : Tensor ⟨#[batch, seq, n_head, head_dim], d, dev⟩ :=
  Tensor.joinHeads x

/-- Scaled dot-product attention with GQA; output shares the query spec. -/
private def sdpaGQA {batch n_head n_kv_head seq head_dim : UInt64} {d : DType} {dev : DevicePolicy}
    (query : Tensor ⟨#[batch, n_head, seq, head_dim], d, dev⟩)
    (key : Tensor ⟨#[batch, n_kv_head, seq, head_dim], d, dev⟩)
    (value : Tensor ⟨#[batch, n_kv_head, seq, head_dim], d, dev⟩)
    (dropoutP : Float) (isCausal : Bool) (enableGqa : Bool)
    : Tensor ⟨#[batch, n_head, seq, head_dim], d, dev⟩ :=
  Tensor.attentionGQA query key value dropoutP isCausal enableGqa

/-- SDPA with GQA where query and KV sequence lengths differ. -/
private def sdpaGQAQKV {batch n_head n_kv_head q_seq kv_seq head_dim : UInt64} {d : DType} {dev : DevicePolicy}
    (query : Tensor ⟨#[batch, n_head, q_seq, head_dim], d, dev⟩)
    (key : Tensor ⟨#[batch, n_kv_head, kv_seq, head_dim], d, dev⟩)
    (value : Tensor ⟨#[batch, n_kv_head, kv_seq, head_dim], d, dev⟩)
    (dropoutP : Float) (isCausal : Bool) (enableGqa : Bool)
    : Tensor ⟨#[batch, n_head, q_seq, head_dim], d, dev⟩ :=
  Tensor.attentionGQAQKV query key value dropoutP isCausal enableGqa

/-- SDPA with GQA and a [batch, seq] padding mask. The mask dtype stays
    generic: the extern accepts any runtime dtype and the legacy callers
    were dtype-erased here. -/
private def sdpaGQAMask {batch n_head n_kv_head seq head_dim : UInt64} {d dm : DType} {dev : DevicePolicy}
    (query : Tensor ⟨#[batch, n_head, seq, head_dim], d, dev⟩)
    (key : Tensor ⟨#[batch, n_kv_head, seq, head_dim], d, dev⟩)
    (value : Tensor ⟨#[batch, n_kv_head, seq, head_dim], d, dev⟩)
    (mask : Tensor ⟨#[batch, seq], dm, dev⟩)
    (dropoutP : Float) (isCausal : Bool) (enableGqa : Bool)
    : Tensor ⟨#[batch, n_head, seq, head_dim], d, dev⟩ :=
  Tensor.attentionGQAMask query key value mask dropoutP isCausal enableGqa

/-- Experimental Tyr FlashAttention entry point; output shares the query spec. -/
private def flashAttn4d {batch nHead nKvHead qSeq kvSeq headDim : UInt64} {d : DType} {dev : DevicePolicy}
    (query : Tensor ⟨#[batch, nHead, qSeq, headDim], d, dev⟩)
    (key : Tensor ⟨#[batch, nKvHead, kvSeq, headDim], d, dev⟩)
    (value : Tensor ⟨#[batch, nKvHead, kvSeq, headDim], d, dev⟩)
    (enableGqa : Bool) : Tensor ⟨#[batch, nHead, qSeq, headDim], d, dev⟩ :=
  Tensor.flashAttention query key value enableGqa

/-- `sliceScatter`: replace a slice of `input` along `dim` at `start` with
    `src`; the result preserves the input spec. -/
private def sliceScatterAlong {s srcShape : Shape} {d : DType} {dev : DevicePolicy}
    (input : Tensor ⟨s, d, dev⟩) (dim : UInt64) (start : UInt64)
    (src : Tensor ⟨srcShape, d, dev⟩) : Tensor ⟨s, d, dev⟩ :=
  Tensor.sliceScatter input dim start src

/-- Mean over one axis with the result shape validated at runtime. -/
private def meanDimTo {s : Shape} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨s, d, dev⟩) (dim : Nat) (out : Shape) (keepdim : Bool := false)
    : Tensor ⟨out, d, dev⟩ :=
  match Tensor.ofRaw (RawTensor.mean s t.raw (some #[dim.toUInt64]) keepdim) with
  | .ok r => r
  | .error message => panic! s!"QwenAttention.meanDim: {message}"

/-- Apply RMS normalization along the last dimension (per-head norm).
    Input: [batch, seq, n_heads, head_dim], norm: [head_dim]
    Broadcasts norm across all heads. -/
private def applyHeadNorm {batch seq n_heads head_dim : UInt64}
    (x : DTensor #[batch, seq, n_heads, head_dim] dtype)
    (norm : DTensor #[head_dim] dtype)
    (eps : Float := 1e-6)
    : DTensor #[batch, seq, n_heads, head_dim] dtype :=
  Tensor.rmsNormPreserving x norm eps

/-- Forward pass for attention.
    Input: [batch, seq, hidden_size]
    Output: [batch, seq, hidden_size] -/
def forward {batch seq hidden_size num_heads num_kv_heads head_dim : UInt64}
    (attn : QwenAttention hidden_size num_heads num_kv_heads head_dim dtype)
    (x : DTensor #[batch, seq, hidden_size] dtype)
    (cos : DTensor #[seq, head_dim / 2] .Float32)
    (sin : DTensor #[seq, head_dim / 2] .Float32)
    (is_causal : Bool := true)
    : DTensor #[batch, seq, hidden_size] dtype :=
  -- Project to Q, K, V
  let q := Tensor.linear3d x attn.q_proj  -- [batch, seq, num_heads * head_dim]
  let k := Tensor.linear3d x attn.k_proj  -- [batch, seq, num_kv_heads * head_dim]
  let v := Tensor.linear3d x attn.v_proj  -- [batch, seq, num_kv_heads * head_dim]

  -- Reshape to [batch, seq, num_heads, head_dim]
  let q := q.reshapeChecked #[batch, seq, num_heads, head_dim]
  let k := k.reshapeChecked #[batch, seq, num_kv_heads, head_dim]
  let v := v.reshapeChecked #[batch, seq, num_kv_heads, head_dim]

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
  let attn_out := sdpaGQA q k v 0.0 is_causal true

  -- Transpose back to [batch, seq, num_heads, head_dim]
  let attn_out := transposeFromAttention attn_out

  -- Reshape to [batch, seq, hidden_size]
  let attn_out := attn_out.reshapeChecked #[batch, seq, num_heads * head_dim]

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
def forwardStep {batch hidden_size num_heads num_kv_heads head_dim : UInt64}
    (attn : QwenAttention hidden_size num_heads num_kv_heads head_dim dtype)
    (x : DTensor #[batch, 1, hidden_size] dtype)
    (cos : DTensor #[1, head_dim / 2] .Float32)
    (sin : DTensor #[1, head_dim / 2] .Float32)
    (cache : KVCache batch num_kv_heads head_dim dtype)
    (useTyrFlashAttn : Bool := false)
    : DTensor #[batch, 1, hidden_size] dtype × KVCache batch num_kv_heads head_dim dtype :=
  -- Project current token to Q/K/V.
  let q0 := Tensor.linear3d x attn.q_proj
  let k0 := Tensor.linear3d x attn.k_proj
  let v0 := Tensor.linear3d x attn.v_proj

  let q := q0.reshapeChecked #[batch, 1, num_heads, head_dim]
  let k := k0.reshapeChecked #[batch, 1, num_kv_heads, head_dim]
  let v := v0.reshapeChecked #[batch, 1, num_kv_heads, head_dim]

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

  let kStore : DTensor #[batch, num_kv_heads, cache.maxLen, head_dim] dtype :=
    Tensor.castLike kNew (expectStore cache.kStoreDyn cache.maxLen)
  let vStore : DTensor #[batch, num_kv_heads, cache.maxLen, head_dim] dtype :=
    Tensor.castLike vNew (expectStore cache.vStoreDyn cache.maxLen)

  if cache.seq < cache.maxLen then
    let kStore' := sliceScatterAlong kStore 2 cache.seq kNew
    let vStore' := sliceScatterAlong vStore 2 cache.seq vNew
    let kvLen : UInt64 := cache.seq + 1
    let kAll := kStore'.sliceChecked 2 0 kvLen #[batch, num_kv_heads, kvLen, head_dim]
    let vAll := vStore'.sliceChecked 2 0 kvLen #[batch, num_kv_heads, kvLen, head_dim]

    -- Use q_len=1, kv_len=(seq+1). Causal masking is unnecessary because KV has no future tokens.
    let attnOut :=
      if useTyrFlashAttn then
        flashAttn4d qh kAll vAll true
      else
        sdpaGQAQKV qh kAll vAll 0.0 false true
    let attnOut := transposeFromAttention attnOut
    let attnOut := attnOut.reshapeChecked #[batch, 1, num_heads * head_dim]
    let out := Tensor.linear3d attnOut attn.o_proj

    let cache' : KVCache batch num_kv_heads head_dim dtype := {
      kStoreDyn := ⟨_, kStore'⟩
      vStoreDyn := ⟨_, vStore'⟩
      seq := kvLen
      maxLen := cache.maxLen
    }
    (out, cache')
  else
    let writePos : UInt64 :=
      if cache.maxLen == 0 then
        0
      else
        cache.maxLen - 1
    let kStore' := sliceScatterAlong kStore 2 writePos kNew
    let vStore' := sliceScatterAlong vStore 2 writePos vNew
    let kvLen : UInt64 := cache.maxLen
    let kAll := kStore'.sliceChecked 2 0 kvLen #[batch, num_kv_heads, kvLen, head_dim]
    let vAll := vStore'.sliceChecked 2 0 kvLen #[batch, num_kv_heads, kvLen, head_dim]
    let attnOut :=
      if useTyrFlashAttn then
        flashAttn4d qh kAll vAll true
      else
        sdpaGQAQKV qh kAll vAll 0.0 false true
    let attnOut := transposeFromAttention attnOut
    let attnOut := attnOut.reshapeChecked #[batch, 1, num_heads * head_dim]
    let out := Tensor.linear3d attnOut attn.o_proj
    let cache' : KVCache batch num_kv_heads head_dim dtype := {
      kStoreDyn := ⟨_, kStore'⟩
      vStoreDyn := ⟨_, vStore'⟩
      seq := cache.maxLen
      maxLen := cache.maxLen
    }
    (out, cache')

def forwardMasked {batch seq hidden_size num_heads num_kv_heads head_dim : UInt64}
    {maskDtype : DType}
    (attn : QwenAttention hidden_size num_heads num_kv_heads head_dim dtype)
    (x : DTensor #[batch, seq, hidden_size] dtype)
    (cos : DTensor #[seq, head_dim / 2] .Float32)
    (sin : DTensor #[seq, head_dim / 2] .Float32)
    (attn_mask : Tensor ⟨#[batch, seq], maskDtype, .any⟩)
    (is_causal : Bool := true)
    : DTensor #[batch, seq, hidden_size] dtype :=
  -- Project to Q, K, V
  let q := Tensor.linear3d x attn.q_proj  -- [batch, seq, num_heads * head_dim]
  let k := Tensor.linear3d x attn.k_proj  -- [batch, seq, num_kv_heads * head_dim]
  let v := Tensor.linear3d x attn.v_proj  -- [batch, seq, num_kv_heads * head_dim]

  -- Reshape to [batch, seq, num_heads, head_dim]
  let q := q.reshapeChecked #[batch, seq, num_heads, head_dim]
  let k := k.reshapeChecked #[batch, seq, num_kv_heads, head_dim]
  let v := v.reshapeChecked #[batch, seq, num_kv_heads, head_dim]

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
  let attn_out := sdpaGQAMask q k v attn_mask 0.0 is_causal true

  -- Transpose back to [batch, seq, num_heads, head_dim]
  let attn_out := transposeFromAttention attn_out

  -- Reshape to [batch, seq, hidden_size]
  let attn_out := attn_out.reshapeChecked #[batch, seq, num_heads * head_dim]

  -- Output projection
  Tensor.linear3d attn_out attn.o_proj

end QwenAttention

end torch.qwen
