/- Tyr/Inference/KVCache.lean

  Key-Value cache for autoregressive decode generation.

  Layout: each layer holds preallocated K, V buffers of shape
  `[batch, numKvHeads, maxSeqLen, headDim]`, allocated fresh per layer in
  `Cache.init` so no two layers share storage. During generation we write the
  new token's K, V at position `cache.currentLen` via `copySliceIO`
  (the C++ op `lean_torch_copy_slice_io` in `cc/src/tyr.cpp`
  does narrow + copy_ on the ORIGINAL buffer — no clone) and slice the cache
  to `[..., currentLen+1, ...]` for the attention call.

  Buffers are indexed tensors: shape, dtype and device policy are carried in
  the `Tensor` spec. The dtype defaults to BFloat16 (the `Cache.init`
  allocation dtype) and the device policy defaults to `.any` (the placement is
  a runtime `Device` argument); appends and views share the cache's spec
  statically, and still re-check the physical device/dtype at runtime so
  dynamically-graded (`Unknown`/`.any`) call sites keep the old error
  behavior. The sequence dimension of an attention view stays runtime
  (`validLen` is a value, not a constant), so views are produced by
  `Tensor.sliceChecked`, which validates the claimed spec against the native
  handle.

  Ownership contract: appends mutate the layer's buffers in place, so the
  `Cache`/`LayerCache` value passed to `LayerCache.append`,
  `Cache.appendLayer`, or `Cache.attendLayer` must not be reused after the
  call — the pre-append and post-append values share the mutated tensors, and
  the write is observable through the old value. The interface stays
  effectful: an updated cache is returned in IO and the caller
  threads it forward, discarding the old one. Access must be serialized by
  the caller; this low-level cache does not provide a lock. All in-tree consumers (the
  decode parity harness `Examples/GPU/RunMhaH100Decode.lean`) satisfy this.

  The attention call goes through `nn.tyrFlashAttn4d`, which the C++
  `tyr::flash_attn` operator dispatches to the native TK decode kernel
  (`tkMhaH100DecodeFwd`) when shape-eligible (BF16, qSeq=1,
  head_dim ∈ {64, 128, 256}, GQA-valid) and to PyTorch SDPA otherwise. The
  kernel handles non-multiple-of-64 `kvSeq` via a runtime tail mask, so the
  cache view never has to be padded.

  This module is the canonical home for the cache abstraction. NanoChat,
  Qwen, and any other generation-loop code should import from here. (The
  earlier copy at `Examples/NanoChat/Generator/KVCache.lean` is a thin
  re-export shim.) -/
import Tyr.Torch
import Tyr.Typed.Ops
import Tyr.Inference.OwnedKV
import Tyr.TensorStruct
import Tyr.Module.Derive

namespace torch.Generator.KVCache

open torch
open torch.data

/-- KV cache for a single attention layer -/
structure LayerCache (batch maxSeqLen numKvHeads headDim : UInt64)
    (dtype : DType := .BFloat16) (device : DevicePolicy := .any) where
  /-- Cached keys: [batch, num_kv_heads, max_seq_len, head_dim] -/
  keys : Tensor ⟨#[batch, numKvHeads, maxSeqLen, headDim], dtype, device⟩
  /-- Cached values: [batch, num_kv_heads, max_seq_len, head_dim] -/
  values : Tensor ⟨#[batch, numKvHeads, maxSeqLen, headDim], dtype, device⟩
  deriving Repr

/-- Full KV cache for all layers -/
structure Cache (numLayers batch maxSeqLen numKvHeads headDim : UInt64)
    (dtype : DType := .BFloat16) (device : DevicePolicy := .any) where
  /-- Per-layer KV caches -/
  layers : Array (LayerCache batch maxSeqLen numKvHeads headDim dtype device)
  /-- Current sequence length for each batch element -/
  seqLens : Array UInt64
  /-- Maximum sequence length this cache can hold -/
  maxLen : UInt64
  deriving Repr, Inhabited

/-- Initialize empty KV cache for given dimensions.
    `device` defaults to CPU; pass `Device.CUDA n` (or a non-default value) so
    the cache buffers live on the same device as the Q/K/V tensors that will be
    appended into them — `tyrFlashAttn4d` rejects cross-device inputs. -/
def Cache.init (numLayers batch maxSeqLen numKvHeads headDim : UInt64)
    (device : Device := Device.CPU)
    : IO (Cache numLayers batch maxSeqLen numKvHeads headDim) := do
  if numLayers == 0 || batch == 0 || maxSeqLen == 0 || numKvHeads == 0 || headDim == 0 then
    throw <| IO.userError "KV cache dimensions must be positive"
  -- Pure allocations may be commoned by Lean, even within a single layer.
  -- Each IO clone must execute independently, both for K/V and across layers.
  let template : Tensor ⟨#[batch, numKvHeads, maxSeqLen, headDim], .BFloat16, .any⟩ :=
    Tensor.toBFloat16 (Tensor.zeros #[batch, numKvHeads, maxSeqLen, headDim]
      (requiresGrad := false) (device := device))
  let mut layers : Array (LayerCache batch maxSeqLen numKvHeads headDim) := #[]
  for _ in [:numLayers.toNat] do
    let keys ← cloneInferenceIO template
    let values ← cloneInferenceIO template
    layers := layers.push { keys, values }
  let seqLens := Array.mk (List.replicate batch.toNat (0 : UInt64))
  return { layers, seqLens, maxLen := maxSeqLen }

private def Cache.validate
    (cache : Cache numLayers batch maxSeqLen numKvHeads headDim dtype device) : IO Unit := do
  if batch == 0 || cache.layers.size.toUInt64 != numLayers || cache.seqLens.size.toUInt64 != batch ||
      cache.maxLen != maxSeqLen then
    throw <| IO.userError "KV cache metadata shape mismatch"
  let pos := cache.seqLens[0]!
  if pos > maxSeqLen || cache.seqLens.any (· != pos) then
    throw <| IO.userError "KV cache requires equal, in-bounds sequence lengths"

/-- Increment sequence lengths for all batch elements -/
def Cache.incrementSeqLens (cache : Cache numLayers batch maxSeqLen numKvHeads headDim dtype device)
    : IO (Cache numLayers batch maxSeqLen numKvHeads headDim dtype device) := do
  cache.validate
  if cache.seqLens[0]! >= maxSeqLen then
    throw <| IO.userError "KV cache sequence length exceeds capacity"
  return { cache with seqLens := cache.seqLens.map (· + 1) }

/-- Get current sequence length (assumes all batch elements have same length) -/
def Cache.currentLen (cache : Cache numLayers batch maxSeqLen numKvHeads headDim dtype device)
    : UInt64 :=
  cache.seqLens[0]?.getD 0

/-- Check if cache has room for more tokens -/
def Cache.hasRoom (cache : Cache numLayers batch maxSeqLen numKvHeads headDim dtype device) : Bool :=
  cache.currentLen < cache.maxLen

/-- Get layer cache at index -/
def Cache.getLayer (cache : Cache numLayers batch maxSeqLen numKvHeads headDim dtype device)
    (idx : Nat) : Option (LayerCache batch maxSeqLen numKvHeads headDim dtype device) :=
  cache.layers[idx]?

/-- Set layer cache at index -/
def Cache.setLayer (cache : Cache numLayers batch maxSeqLen numKvHeads headDim dtype device)
    (idx : Nat) (layer : LayerCache batch maxSeqLen numKvHeads headDim dtype device)
    : Cache numLayers batch maxSeqLen numKvHeads headDim dtype device :=
  if idx < cache.layers.size then
    { cache with layers := cache.layers.set! idx layer }
  else
    cache

/-- Append a single new K, V token at runtime position `pos`.

    The new K and V tensors have shape `[batch, numKvHeads, 1, headDim]` (one
    token per batch row) — now static in the `Tensor` spec — and are written
    IN PLACE into the layer's cache buffers at the `pos`-th slot along the
    sequence dimension, via ordered `copySliceIO` calls (narrow + copy_ on the
    original tensor, no clone). The passed-in layer cache must not be reused
    after this call: old and new values share the mutated tensors, so the
    write is observable through the old value. The cache's `seqLens` is not
    touched here — call `Cache.incrementSeqLens` after appending to all
    layers of a step to advance position. -/
def LayerCache.append (lc : LayerCache batch maxSeqLen numKvHeads headDim dtype device)
    (newK newV : Tensor ⟨#[batch, numKvHeads, 1, headDim], dtype, device⟩) (pos : UInt64)
    : IO (LayerCache batch maxSeqLen numKvHeads headDim dtype device) := do
  if pos >= maxSeqLen then throw <| IO.userError "KV cache append exceeds capacity"
  -- Shapes are static in the specs; the physical device/dtype equality is
  -- still checked at runtime so dynamically-graded specs (`Unknown`, `.any`)
  -- keep the old failure mode.
  if lc.keys.raw.device != lc.values.raw.device || lc.keys.raw.dtype != lc.values.raw.dtype ||
      newK.raw.device != lc.keys.raw.device || newV.raw.device != lc.values.raw.device ||
      newK.raw.dtype != lc.keys.raw.dtype || newV.raw.dtype != lc.values.raw.dtype then
    throw <| IO.userError "KV cache append device/dtype mismatch"
  copySliceIO lc.keys 2 pos newK
  copySliceIO lc.values 2 pos newV
  return lc

/-- Slice the cache K and V tensors to cover positions `[0, validLen)`.

    Returns `(K_view, V_view)` of shape `[batch, numKvHeads, validLen, headDim]`,
    with the runtime `validLen` recorded in the spec (produced by
    `Tensor.sliceChecked`, which validates the native handle), suitable to
    pass directly to `tkMhaH100DecodeFwd` (which expects contiguous
    `[batch, numKvHeads, kv_seq, headDim]` and handles the tail mask when
    `validLen` is not a multiple of 64). -/
def LayerCache.attentionView (lc : LayerCache batch maxSeqLen numKvHeads headDim dtype device)
    (validLen : UInt64)
    : IO (Tensor ⟨#[batch, numKvHeads, validLen, headDim], dtype, device⟩
      × Tensor ⟨#[batch, numKvHeads, validLen, headDim], dtype, device⟩) := do
  if validLen > maxSeqLen then throw <| IO.userError "KV cache view exceeds capacity"
  return (Tensor.sliceChecked lc.keys 2 0 validLen #[batch, numKvHeads, validLen, headDim],
    Tensor.sliceChecked lc.values 2 0 validLen #[batch, numKvHeads, validLen, headDim])

/-- Append a single new K, V token to the layer cache at `cache.currentLen` and
    return the updated full cache (with that layer rewritten). The write is in
    place (see `LayerCache.append`): do not reuse the passed-in `cache` after
    this call. Does NOT advance seqLens — call `incrementSeqLens` once per
    generated token after appending to every layer for the step. -/
def Cache.appendLayer (cache : Cache numLayers batch maxSeqLen numKvHeads headDim dtype device)
    (layerIdx : Nat) (newK newV : Tensor ⟨#[batch, numKvHeads, 1, headDim], dtype, device⟩)
    : IO (Cache numLayers batch maxSeqLen numKvHeads headDim dtype device) := do
  cache.validate
  match cache.layers[layerIdx]? with
  | none => throw <| IO.userError "KV cache layer index out of range"
  | some lc => do
    let pos := cache.currentLen
    let lc ← lc.append newK newV pos
    return cache.setLayer layerIdx lc

/-- Typed adapter over the `tyr::flash_attn` runtime dispatcher
    (`nn.tyrFlashAttn4d`, the same extern the legacy call used): the output
    shares the query's shape, dtype and device policy. `Tyr.Typed.Ops` has no
    typed flash-attention op yet; this unwraps the indexed tensors at the
    extern boundary and re-asserts the output spec. -/
private def flashAttn4d {batch nHead nKvHead qSeq kvSeq headDim : UInt64}
    {dtype : DType} {device : DevicePolicy}
    (query : Tensor ⟨#[batch, nHead, qSeq, headDim], dtype, device⟩)
    (key : Tensor ⟨#[batch, nKvHead, kvSeq, headDim], dtype, device⟩)
    (value : Tensor ⟨#[batch, nKvHead, kvSeq, headDim], dtype, device⟩)
    (enableGqa : Bool) : Tensor ⟨#[batch, nHead, qSeq, headDim], dtype, device⟩ :=
  Tensor.assumeSpec (nn.tyrFlashAttn4d
    (batch := batch) (n_head := nHead) (n_kv_head := nKvHead)
    (q_seq := qSeq) (kv_seq := kvSeq) (head_dim := headDim)
    query.raw key.raw value.raw none 0.0 false none enableGqa)

/-- One step of cached decode-attention for a single layer.

    1. Write `newK`, `newV` (one token per batch row) into layer `layerIdx`'s
       cache at position `cache.currentLen`.
    2. Slice the layer's cache to the resulting valid length (`currentLen + 1`).
    3. Call `tyrFlashAttn4d` on `(newQ, K_view, V_view)`.

    Returns the updated full cache and the attention output for the new query
    token. The K/V append is in place (see `LayerCache.append`): do not reuse
    the passed-in `cache` after this call. The C++ `tyr::flash_attn`
    dispatcher routes to `tkMhaH100DecodeFwd` when the shape is
    decode-eligible (head_dim ∈ {64, 128, 256}, qSeq==1, BF16, GQA-valid),
    and falls back to PyTorch SDPA otherwise.

    Rejects invalid head counts, layer indices, capacity and (at runtime, for
    dynamically-graded specs) query device/dtype mismatches before writing;
    shapes are enforced statically by the `Tensor` specs.

    Does NOT advance `seqLens` — call `cache.incrementSeqLens` once per
    generated token after all layers have been processed for the step. -/
def Cache.attendLayer
    {numQHeads : UInt64}
    (cache : Cache numLayers batch maxSeqLen numKvHeads headDim dtype device)
    (layerIdx : Nat)
    (newQ : Tensor ⟨#[batch, numQHeads, 1, headDim], dtype, device⟩)
    (newK newV : Tensor ⟨#[batch, numKvHeads, 1, headDim], dtype, device⟩)
    (enableGqa : Bool := false)
    : IO (Cache numLayers batch maxSeqLen numKvHeads headDim dtype device
      × Tensor ⟨#[batch, numQHeads, 1, headDim], dtype, device⟩) := autograd.no_grad do
  let some lc := cache.layers[layerIdx]?
    | throw <| IO.userError "KV cache layer index out of range"
  if numQHeads == 0 || numKvHeads == 0 ||
      (if enableGqa then numQHeads % numKvHeads != 0 else numQHeads != numKvHeads) ||
      newQ.raw.device != lc.keys.raw.device || newQ.raw.dtype != lc.keys.raw.dtype then
    throw <| IO.userError "KV cache query heads, device or dtype mismatch"
  let cache' ← cache.appendLayer layerIdx newK newV
  let validLen := cache.currentLen + 1
  let (kView, vView) ← lc.attentionView validLen
  let output := flashAttn4d newQ kView vView enableGqa
  return (cache', output)

/-- TensorStruct instance for LayerCache -/
instance {batch maxSeqLen numKvHeads headDim : UInt64} {dtype : DType} {device : DevicePolicy}
    : TensorStruct (LayerCache batch maxSeqLen numKvHeads headDim dtype device) where
  map f c := { keys := TensorStruct.map f c.keys, values := TensorStruct.map f c.values }
  mapM f c := do pure { keys := ← TensorStruct.mapM f c.keys, values := ← TensorStruct.mapM f c.values }
  zipWith f c1 c2 := { keys := TensorStruct.zipWith f c1.keys c2.keys, values := TensorStruct.zipWith f c1.values c2.values }
  fold f init c := TensorStruct.fold f (TensorStruct.fold f init c.keys) c.values

/-- TensorStruct instance for Cache -/
instance {numLayers batch maxSeqLen numKvHeads headDim : UInt64} {dtype : DType}
    {device : DevicePolicy}
    : TensorStruct (Cache numLayers batch maxSeqLen numKvHeads headDim dtype device) where
  map f c := { c with layers := c.layers.map (TensorStruct.map f) }
  mapM f c := do
    let layers ← c.layers.mapM (TensorStruct.mapM f)
    pure { c with layers }
  zipWith f c1 c2 := { c1 with
    layers := Array.zipWith (TensorStruct.zipWith f) c1.layers c2.layers
  }
  fold f init c := c.layers.foldl (fun acc l => TensorStruct.fold f acc l) init

end torch.Generator.KVCache
