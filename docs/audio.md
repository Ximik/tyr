# Audio capture, playback, and inference utilities

Two small support layers sit behind the voice demos:

- `Tyr/Audio/` — microphone capture and speaker feedback on macOS
  (AudioToolbox/AudioQueue), plus an unboxed sample buffer for hot audio loops.
- `Tyr/Inference/` — a shape-indexed KV cache for autoregressive decode loops.

Use them when building voice-driven or streaming applications on top of Tyr's
audio models. The models themselves (Whisper, Qwen3ASR, KittenTTS, Qwen3TTS,
SileroVAD) are covered in [Audio and speech models](models/audio-speech.md);
this chapter is about getting samples in and out of the process and about the
decode-time cache. Neither module is re-exported by the root `Tyr.lean`
aggregator — import the submodules directly.

## Architecture and main abstractions

### Audio: thin façades over C++ globals

All three audio modules are `@[extern]` bindings; every piece of state (the
AudioQueue, a sample FIFO, lifecycle mutexes) lives in C++:

| Lean module | C++ backing | Off macOS |
| --- | --- | --- |
| `Tyr/Audio/FloatBuffer.lean` | `cc/src/float_buffer.cpp` | works everywhere |
| `Tyr/Audio/AppleInput.lean` | `cc/src/apple_audio_input.mm` | `start` raises `IO.userError`; `read`/`readBuffer` return empty, `stop` is a no-op |
| `Tyr/Audio/AppleOutput.lean` | `cc/src/apple_audio_output.mm` | silent no-ops |

`AudioToolbox` is linked into `libTyrC` by `cc/Makefile`, so no extra build
flags are needed on macOS.

Capture is a **global singleton**: `AppleInput.start` opens an AudioQueue whose
callback pushes interleaved Float32 frames onto a `std::deque` FIFO, and
`read`/`readBuffer` drain it. The FIFO is capped at ~30 s of audio
(`cc/src/apple_audio_input.mm:46`); under sustained backpressure the oldest
audio is silently dropped. `stop` tears down the queue and clears the FIFO.
Only one capture session can be active at a time.

The currency at the model boundary is a 16 kHz mono waveform as `Array Float`
(see [Audio and speech models](models/audio-speech.md)). Inside capture loops,
samples flow through `FloatBuffer` to avoid per-element boxing of
`Array Float`; convert once with `FloatBuffer.toArray` when handing off to a
model frontend.

### Inference: a preallocated, shape-indexed KV cache

`Tyr/Inference/KVCache.lean` provides a key/value cache for transformer decode
loops. Each layer owns preallocated K and V buffers of shape
`[batch, numKvHeads, maxSeqLen, headDim]`; a decode step writes the new token's
K/V at position `cache.currentLen` (in place, via
`torch.data.sliceScatterInplace`), slices the
layer to the valid prefix, and runs attention through `nn.tyrFlashAttn4d`
(`Tyr/Torch.lean:1198`).

The C++ `tyr::flash_attn` operator behind that call routes to the
ThunderKittens H100 decode kernel (`tkMhaH100DecodeFwd`) when the step is
decode-eligible, and to PyTorch SDPA otherwise. Eligibility is decided in
`select_route` (`cc/src/tyr_ops.cpp:314`): CUDA on a Hopper-class device, BF16,
`qSeq = 1`, `headDim ∈ {64, 128, 256}`, GQA-valid head counts, no attention
mask, `dropout = 0`, non-causal, default scale. The kernel iterates
`ceil(kvSeq/64)` blocks and tail-masks the last one, so cache views are never
padded to a block multiple. See [GPU kernels](gpu/kernels.md) for the kernel
catalog.

One namespace quirk: the module is named `Tyr.Inference.KVCache` but declares
`namespace torch.Generator.KVCache` (`Tyr/Inference/KVCache.lean:24`), a
leftover from its NanoChat origin. Call sites write
`torch.Generator.KVCache.Cache`, or `open torch.Generator` and use
`KVCache.Cache`. The older `Examples/NanoChat/Generator/KVCache.lean` is a thin
re-export shim; new code should import `Tyr.Inference.KVCache` directly.

**Position bookkeeping is the caller's job.** `Cache.attendLayer` appends and
attends but does *not* advance the sequence position. Call
`cache.incrementSeqLens` exactly once per generated token, after every layer of
that step has been processed. `Cache.currentLen` reads `seqLens[0]` and assumes
uniform lengths across the batch.

Both cache structures have `TensorStruct` instances
(`Tyr/Inference/KVCache.lean:162`), so generic traversals (`map`, `mapM`,
`zipWith`, `fold`) — device moves, dtype casts — work over whole caches. See
[TensorStruct](core/tensorstruct.md).

## Key APIs

### `FloatBuffer` (`Tyr/Audio/FloatBuffer.lean`)

Opaque contiguous `Float64` buffer backed by native memory, registered as a
Lean external class with a finalizer. Mutation is gated behind `IO` and returns
the same buffer for chaining, enforcing linear use.

```lean
opaque FloatBuffer : Type := Unit

namespace FloatBuffer
  opaque mkEmpty (capacity : @& Nat) : IO FloatBuffer
  opaque size (buf : @& FloatBuffer) : Nat                      -- pure, O(1)
  opaque uget (buf : @& FloatBuffer) (i : @& Nat) : Float       -- unchecked!
  opaque rms (buf : @& FloatBuffer) : IO Float
  opaque push (buf : FloatBuffer) (x : Float) : IO FloatBuffer
  opaque append (buf : FloatBuffer) (src : @& FloatBuffer) : IO FloatBuffer
  opaque appendArray (buf : FloatBuffer) (arr : @& Array Float) : IO FloatBuffer
  opaque clear (buf : FloatBuffer) : IO FloatBuffer
  opaque keepLast (buf : FloatBuffer) (n : @& Nat) : IO FloatBuffer  -- memmove
  opaque toArray (buf : @& FloatBuffer) : IO (Array Float)      -- boxes; use at boundaries
end FloatBuffer
```

Two sharp edges: `uget` performs no bounds check, and the auto-derived
`Inhabited` instance is the `Unit` witness, not a valid native buffer — never
call an op on `(default : FloatBuffer)`.

### `Tyr.Audio.AppleInput` (`Tyr/Audio/AppleInput.lean`)

```lean
opaque start (sampleRate : UInt64 := 16000) (channels : UInt64 := 1)
    (bufferMs : UInt64 := 100) : IO Unit
opaque read (maxSamples : UInt64) (blockMs : UInt64 := 250) : IO (Array Float)
opaque readBuffer (maxSamples : UInt64) (blockMs : UInt64 := 250) : IO FloatBuffer
opaque stop : IO Unit
opaque rms (xs : @& Array Float) : IO Float
```

- `start` validates parameters and fails with `IO.userError` if the queue
  cannot start — e.g. missing microphone permission (`AudioQueueStart`).
- `read`/`readBuffer` poll the FIFO in 5 ms naps until at least one sample is
  available, capture stops, or the `blockMs` deadline passes; they then drain
  up to `maxSamples`. The result can be shorter than requested, or empty on
  timeout — always handle the empty case.
- `readBuffer` is the unboxed variant; prefer it in tight capture loops.
  `read` boxes every sample and is fine for hop-wise streaming.
- `rms` is a native-loop RMS over a boxed array (works on all platforms); the
  `FloatBuffer.rms` method covers the unboxed case.
- Off macOS, only `start` fails loudly (`IO.userError "audio input is only
  supported on macOS"`); `read`/`readBuffer` just return empty results, so a
  capture loop that never checks for `start` failure would spin silently.

### `Tyr.Audio.AppleOutput` (`Tyr/Audio/AppleOutput.lean`)

```lean
opaque beep (freqHz : Float) (durationMs : UInt64 := 120)
    (sampleRate : UInt64 := 44100) : IO Unit
opaque beepAsync (freqHz : Float) (durationMs : UInt64 := 120)
    (sampleRate : UInt64 := 44100) : IO Unit

def beepStart : IO Unit    -- 880 Hz, 120 ms — recording started
def beepDone : IO Unit     -- two 660 Hz, 80 ms beeps — transcription done
def beepNoSpeech : IO Unit -- 440 Hz, 100 ms — no speech detected
```

Blocking sine tones at 0.3 volume with a 5 ms fade envelope to avoid clicks;
off macOS they are silent no-ops, so UX cues simply vanish on Linux.
`beepAsync` plays on a detached thread and holds the output mutex for the whole
tone — do not mix it with synchronous `beep` in the same run unless you know
the async tone has finished (hazard documented at
`Tyr/Audio/AppleOutput.lean:8`).

### `torch.Generator.KVCache` (`Tyr/Inference/KVCache.lean`)

```lean
structure LayerCache (batch maxSeqLen numKvHeads headDim : UInt64) where
  keys : T #[batch, numKvHeads, maxSeqLen, headDim]
  values : T #[batch, numKvHeads, maxSeqLen, headDim]

structure Cache (numLayers batch maxSeqLen numKvHeads headDim : UInt64) where
  layers : Array (LayerCache batch maxSeqLen numKvHeads headDim)
  seqLens : Array UInt64
  maxLen : UInt64
```

| Operation | Signature (eliding indices) | Purpose |
| --- | --- | --- |
| `Cache.init` | `(numLayers batch maxSeqLen numKvHeads headDim : UInt64) (device : Device := Device.CPU) : Cache …` | zero-allocate both buffers per layer, cast to BF16 |
| `Cache.attendLayer` | `(layerIdx : Nat) (newQ : T #[batch, numQHeads, 1, headDim]) (newK newV : T #[batch, numKvHeads, 1, headDim]) (enableGqa : Bool := false) : Cache … × T #[batch, numQHeads, 1, headDim]` | append one token to layer `layerIdx`, then flash-attend over the valid prefix |
| `Cache.incrementSeqLens` | `Cache … → Cache …` | advance all positions by one; call once per token, after all layers |
| `Cache.currentLen` | `Cache … → UInt64` | current position (`seqLens[0]`) |
| `Cache.hasRoom` | `Cache … → Bool` | `currentLen < maxLen` |
| `Cache.appendLayer` | `(layerIdx : Nat) (newK newV : …) → Cache …` | append without attending |
| `Cache.getLayer` / `Cache.setLayer` | `…` | per-layer access |
| `LayerCache.append` | `(newK newV : T #[batch, numKvHeads, 1, headDim]) (pos : UInt64) → LayerCache …` | in-place `sliceScatterInplace` of one token at `pos`; the passed-in layer cache must not be reused |
| `LayerCache.attentionView` | `(validLen : UInt64) → (T #[…, validLen, …] × T #[…, validLen, …])` | unpadded K/V views over `[0, validLen)` |

Notes:

- Pass `(device := Device.CUDA n)` to `Cache.init` for GPU decode — the cache
  buffers must live on the same device as the Q/K/V appended into them, and the
  TK route additionally requires Hopper.
- The dtype is hardcoded to BF16 (`toBFloat16'` over `zeros`); FP16/FP32
  pipelines need a cast or their own cache.
- Appends mutate the cache buffers in place via `sliceScatterInplace`
  (`cc/src/tyr.cpp` `lean_torch_slice_scatter_along_dim_inplace`: narrow +
  copy_ on the original tensor, no clone), so a decode token costs O(1) per
  layer instead of an O(maxSeqLen) buffer copy. The trade-off is an ownership
  contract: a `Cache`/`LayerCache` value passed to `attendLayer` /
  `appendLayer` / `LayerCache.append` must not be reused after the call —
  old and new values share the mutated tensors. `Cache.init` allocates fresh
  buffers per layer so layers never alias each other's storage.
- An out-of-range `layerIdx` makes `attendLayer` panic with a message naming
  the index and the layer count — keep indices valid.
- The in-tree consumer is the GPU parity harness
  `Examples/GPU/RunMhaH100Decode.lean`. Model families under `Tyr/Model/`
  (Qwen, Whisper, …) currently define their own caches instead of reusing this
  one.

## Usage examples

### Recording with voice-activity detection

Reconstructed example (from `Examples/Whisper/VoiceMode.lean:163-231`, beeps
and `start`/`stop` at lines 255-256 and 368-393). The real demo records until
trailing silence, keeping a 500 ms pre-roll so speech onset is not clipped;
audio flows through unboxed `FloatBuffer`s and is boxed once at the model
boundary:

```lean
import Tyr.Audio.AppleInput
import Tyr.Audio.AppleOutput
import Tyr.Audio.FloatBuffer

/-- Record 16 kHz mono audio until 1.5 s of trailing silence. -/
def recordUntilSilence : IO (Array Float) := do
  let sampleRate : UInt64 := 16000
  let speechRmsThreshold := 0.008                 -- VoiceMode default
  let preRollSamples := sampleRate.toNat / 2      -- 500 ms
  Tyr.Audio.AppleOutput.beepStart                 -- cue the user
  Tyr.Audio.AppleInput.start sampleRate 1 100
  let mut buf ← FloatBuffer.mkEmpty (sampleRate.toNat * 10)
  let mut recording := false
  let mut silenceSamples : Nat := 0
  while buf.size < sampleRate.toNat * 30 do       -- 30 s hard cap
    let pcm ← Tyr.Audio.AppleInput.readBuffer (sampleRate / 10) 250  -- 100 ms chunks
    if pcm.size == 0 then continue                -- read timed out
    let level ← pcm.rms
    buf ← buf.append pcm
    if level >= speechRmsThreshold then
      recording := true
      silenceSamples := 0
    else if !recording then
      buf ← buf.keepLast preRollSamples           -- slide the pre-roll window
    else
      silenceSamples := silenceSamples + pcm.size
      if silenceSamples >= sampleRate.toNat * 3 / 2 then break  -- 1.5 s silence
  Tyr.Audio.AppleInput.stop
  if !recording then Tyr.Audio.AppleOutput.beepNoSpeech
  buf.toArray                                     -- box once, hand to the model
```

For hop-wise streaming recognition, read fixed-size hops with the boxed
`AppleInput.read` and push each hop into the model session, stopping capture in
both the success and `catch` branches — see
`Examples/Qwen3ASR/LiveMicTrueStream.lean:108-132`.

### Cached decode step

Reconstructed example (from `Examples/GPU/RunMhaH100Decode.lean:179-193`, the
cache-vs-no-cache parity harness). One layer, GQA, BF16 on CUDA:

```lean
import Tyr.Inference.KVCache

open torch
open torch.Generator

/-- One cached decode step for a single-layer model. -/
def decodeStep (cache : KVCache.Cache 1 batch maxLen kvHeads 128)
    (q : T #[batch, 32, 1, 128]) (k v : T #[batch, kvHeads, 1, 128])
    : KVCache.Cache 1 batch maxLen kvHeads 128 × T #[batch, 32, 1, 128] :=
  let (cache', out) := cache.attendLayer (numQHeads := 32) 0 q k v (enableGqa := true)
  (cache'.incrementSeqLens, out)   -- advance position once per token

-- Initialization (device must match the Q/K/V tensors):
--   let cache := KVCache.Cache.init 1 batch maxLen kvHeads 128 (device := Device.CUDA 0)
```

The harness checks this token-by-token path against a single dense
`scaledDotProductAttentionGQAQKV` over the full sequence and expects agreement
within BF16 tolerance — the canonical way to catch off-by-one append positions
and stale slice bounds when wiring a cache into a new model.

## Related guides

- [Audio and speech models](models/audio-speech.md) — the models these
  utilities feed: Whisper, Qwen3ASR, SileroVAD, KittenTTS, Qwen3TTS.
- [Tensors](core/tensors.md) — `T s`, `torch.data.slice` / `sliceScatter`, and
  the device model used by the cache.
- [TensorStruct](core/tensorstruct.md) — the generic traversal class behind the
  cache instances.
- [GPU kernels](gpu/kernels.md) — the ThunderKittens decode kernel and the
  dispatch layer behind `nn.tyrFlashAttn4d`.
- [Getting started](getting-started.md) — building and running the examples
  referenced above.

This chapter is a guide, not a symbol dump. Exhaustive, per-definition
documentation for `Tyr.Audio` and `Tyr.Inference` is generated by doc-gen4; see
`docbuild/` for the API reference build.
