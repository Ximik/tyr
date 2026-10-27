/- Tyr/Inference/KVCache.lean

  Key-Value cache for autoregressive decode generation.

  Layout: each layer holds preallocated K, V buffers of shape
  `[batch, numKvHeads, maxSeqLen, headDim]`, allocated fresh per layer in
  `Cache.init` so no two layers share storage. During generation we write the
  new token's K, V at position `cache.currentLen` via `copySliceIO`
  and keep the returned buffers. Storage is reused when safe. We slice the cache
  to `[..., currentLen+1, ...]` for the attention call.

  Buffers are indexed tensors: shape, dtype and concrete device are carried in
  the `Tensor` spec. The dtype defaults to BFloat16 (the `Cache.init`
  allocation dtype); the runtime-selected `Device` is retained in the result
  type. Appends and views share the cache's spec statically, and still
  re-check physical device/dtype at native boundaries. The sequence dimension of an attention view stays runtime
  (`validLen` is a value, not a constant), so views are produced by
  `Tensor.sliceChecked`, which validates the claimed spec against the native
  handle.

  Appends return an updated cache in IO. Retaining an earlier cache or view
  preserves its values; callers thread the returned cache forward to observe
  new tokens. Native copy-on-write can reuse storage when no aliases remain.

  The attention call goes through `Tensor.flashAttention`, which the C++
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
import Tyr.ConstantSupport
import Tyr.Typed.Neural
import Tyr.Inference.OwnedKV
import Tyr.TensorStruct
import Tyr.Module.Derive

namespace torch.Generator.KVCache

open torch
open torch.data

/-- KV cache for a single attention layer -/
structure LayerCache (batch maxSeqLen numKvHeads headDim : UInt64)
    (dtype : DType := .BFloat16) (device : Device := .CPU) where
  /-- Cached keys: [batch, num_kv_heads, max_seq_len, head_dim] -/
  keys : Tensor ⟨#[batch, numKvHeads, maxSeqLen, headDim], dtype, device⟩
  /-- Cached values: [batch, num_kv_heads, max_seq_len, head_dim] -/
  values : Tensor ⟨#[batch, numKvHeads, maxSeqLen, headDim], dtype, device⟩
  deriving Repr

/-- Full KV cache for all layers -/
structure Cache (numLayers batch maxSeqLen numKvHeads headDim : UInt64)
    (dtype : DType := .BFloat16) (device : Device := .CPU) where
  /-- Per-layer KV caches -/
  layers : Array (LayerCache batch maxSeqLen numKvHeads headDim dtype device)
  /-- Current sequence length for each batch element -/
  seqLens : Array UInt64
  /-- Maximum sequence length this cache can hold -/
  maxLen : UInt64
  deriving Repr, Inhabited

/-- Initialize empty KV cache for given dimensions.
    `device` defaults to CPU; pass a device acquired with `Device.cuda n` so
    the cache buffers live on the same device as the Q/K/V tensors that will be
    appended into them — `tyrFlashAttn4d` rejects cross-device inputs. -/
def Cache.init (numLayers batch maxSeqLen numKvHeads headDim : UInt64)
    (device : Device := Device.CPU)
    : IO (Cache numLayers batch maxSeqLen numKvHeads headDim .BFloat16 device) := do
  if numLayers == 0 || batch == 0 || maxSeqLen == 0 || numKvHeads == 0 || headDim == 0 then
    throw <| IO.userError "KV cache dimensions must be positive"
  let support ← DType.BFloat16.acquireConstantSupport device
  let _ := support.down
  IO.ofExcept (Shape.validateAllocation #[batch, numKvHeads, maxSeqLen, headDim])
  -- The pure zero template can be shared. Each explicit IO clone owns
  -- independent cache storage, both for K/V and across layers.
  let template := Tensor.zerosFor
    ⟨#[batch, numKvHeads, maxSeqLen, headDim], .BFloat16, device⟩
  let mut layers : Array (LayerCache batch maxSeqLen numKvHeads headDim .BFloat16 device) := #[]
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
    into the returned layer's cache buffers at the `pos`-th slot along the
    sequence dimension. Storage is reused when safe; retained old caches and
    views preserve their values. The cache's `seqLens` is not
    touched here — call `Cache.incrementSeqLens` after appending to all
    layers of a step to advance position. -/
def LayerCache.append (lc : LayerCache batch maxSeqLen numKvHeads headDim dtype device)
    (newK newV : Tensor ⟨#[batch, numKvHeads, 1, headDim], dtype, device⟩) (pos : UInt64)
    : IO (LayerCache batch maxSeqLen numKvHeads headDim dtype device) := do
  let keys := lc.keys
  let values := lc.values
  if pos >= maxSeqLen then throw <| IO.userError "KV cache append exceeds capacity"
  -- Validate both sources and destinations before either update.
  -- Interoperability can introduce handles whose metadata violates an index.
  for result in [keys.validate, values.validate, newK.validate, newV.validate] do
    match result with
    | .ok () => pure ()
    | .error message => throw <| IO.userError s!"KV cache append: {message}"
  -- Dynamic dtype/device policies also require physical equality.
  if keys.raw.device != values.raw.device || keys.raw.dtype != values.raw.dtype ||
      newK.raw.device != keys.raw.device || newV.raw.device != values.raw.device ||
      newK.raw.dtype != keys.raw.dtype || newV.raw.dtype != values.raw.dtype then
    throw <| IO.userError "KV cache append device/dtype mismatch"
  let keys ← copySliceIO keys 2 pos newK
  let values ← copySliceIO values 2 pos newV
  return { keys, values }

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

/-- Remove a layer before updating it, using constant-time swaps and a pop. -/
@[noinline] private def takeLayerForUpdate
    (layers : Array (LayerCache batch maxSeqLen numKvHeads headDim dtype device))
    (index : Nat) : Option (LayerCache batch maxSeqLen numKvHeads headDim dtype device ×
      Array (LayerCache batch maxSeqLen numKvHeads headDim dtype device)) :=
  if h : index < layers.size then
    let selected := layers[index]
    -- Returning both values across this boundary forces removal before the
    -- caller can update selected; a pure slot-clear let can be floated later.
    let remaining := (layers.swapIfInBounds index (layers.size - 1)).pop
    some (selected, remaining)
  else none

/-- Append a single new K, V token at `cache.currentLen`, returning the updated
cache and preserving retained old caches and views. This does not advance
`seqLens`; call `incrementSeqLens` once per generated token after all layers. -/
def Cache.appendLayer (cache : Cache numLayers batch maxSeqLen numKvHeads headDim dtype device)
    (layerIdx : Nat) (newK newV : Tensor ⟨#[batch, numKvHeads, 1, headDim], dtype, device⟩)
    : IO (Cache numLayers batch maxSeqLen numKvHeads headDim dtype device) := do
  cache.validate
  let layers := cache.layers
  let seqLens := cache.seqLens
  let maxLen := cache.maxLen
  let pos := seqLens[0]?.getD 0
  let some (lc, layers) := takeLayerForUpdate layers layerIdx
    | throw <| IO.userError "KV cache layer index out of range"
  let updated ← lc.append newK newV pos
  -- Extraction swapped the selected layer to the end before popping it.
  -- Append its replacement and swap it back to restore the original order.
  let layers := layers.push updated
  let layers := layers.swapIfInBounds layerIdx (layers.size - 1)
  return { layers, seqLens, maxLen }

/-- One step of cached decode-attention for a single layer.

    1. Write `newK`, `newV` (one token per batch row) into layer `layerIdx`'s
       cache at position `cache.currentLen`.
    2. Slice the layer's cache to the resulting valid length (`currentLen + 1`).
    3. Call `tyrFlashAttn4d` on `(newQ, K_view, V_view)`.

    Returns the updated full cache and the attention output for the new query
    token. Retained input caches remain unchanged. The C++ `tyr::flash_attn`
    dispatcher routes to `tkMhaH100DecodeFwd` when the shape is
    decode-eligible (head_dim ∈ {64, 128, 256}, qSeq==1, BF16, GQA-valid),
    and falls back to PyTorch SDPA otherwise.

    Rejects invalid head counts, layer indices, capacity and (at runtime, for
    dynamically-graded specs) query device/dtype mismatches before writing;
    native shapes are revalidated before either cache buffer is written.

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
      × Tensor ⟨#[batch, numQHeads, 1, headDim], dtype, device⟩) := do
  let some lc := cache.layers[layerIdx]?
    | throw <| IO.userError "KV cache layer index out of range"
  if numQHeads == 0 || numKvHeads == 0 ||
      (if enableGqa then numQHeads % numKvHeads != 0 else numQHeads != numKvHeads) ||
      newQ.raw.device != lc.keys.raw.device || newQ.raw.dtype != lc.keys.raw.dtype then
    throw <| IO.userError "KV cache query heads, device or dtype mismatch"
  match newQ.validate with
  | .ok () => pure ()
  | .error message => throw <| IO.userError s!"KV cache query: {message}"
  let validLen := cache.currentLen + 1
  let cache' ← cache.appendLayer layerIdx newK newV
  let some updatedLayer := cache'.layers[layerIdx]?
    | throw <| IO.userError "KV cache layer index out of range after append"
  let (kView, vView) ← updatedLayer.attentionView validLen
  let output := Tensor.flashAttention (Tensor.detach newQ) kView vView enableGqa
  return (cache', output)

/-- TensorStruct instance for LayerCache -/
instance {batch maxSeqLen numKvHeads headDim : UInt64} {dtype : DType} {device : Device}
    : TensorStruct (LayerCache batch maxSeqLen numKvHeads headDim dtype device) where
  map f c := { keys := TensorStruct.map f c.keys, values := TensorStruct.map f c.values }
  mapM f c := do pure { keys := ← TensorStruct.mapM f c.keys, values := ← TensorStruct.mapM f c.values }
  zipWith f c1 c2 := { keys := TensorStruct.zipWith f c1.keys c2.keys, values := TensorStruct.zipWith f c1.values c2.values }
  fold f init c := TensorStruct.fold f (TensorStruct.fold f init c.keys) c.values

/-- TensorStruct instance for Cache -/
instance {numLayers batch maxSeqLen numKvHeads headDim : UInt64} {dtype : DType}
    {device : Device}
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
