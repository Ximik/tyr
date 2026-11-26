# Audio capture, playback, and inference utilities

Two small support layers sit behind the voice demos:

- `Tyr/Core/FFI/AppleAudioInput.lean`, `Tyr/Core/FFI/AppleAudioOutput.lean` — microphone
  capture and speaker feedback on macOS (AudioToolbox/AudioQueue).
- `Tyr/Inference/` — a shape-indexed KV cache for autoregressive decode loops.

Use them when building voice-driven or streaming applications on top of Tyr's
audio models. The models themselves (Whisper, Qwen3ASR, KittenTTS, Qwen3TTS,
SileroVAD) are covered in [Audio and speech models](models/audio-speech.md);
this chapter is about getting samples in and out of the process and about the
decode-time cache. Neither module is re-exported by the root `Tyr.lean`
aggregator — import the submodules directly.

## Architecture and main abstractions

### Audio: thin façades over C++ globals

Both audio modules are `@[extern]` bindings; every piece of state (the
AudioQueue, a sample FIFO, lifecycle mutexes) lives in C++:

| Lean module | C++ backing | Off macOS |
| --- | --- | --- |
| `Tyr/Core/FFI/AppleAudioInput.lean` | `cc/src/apple_audio_input.mm` | `start` raises `IO.userError`; `read`/`readBuffer` return empty, `stop` is a no-op |
| `Tyr/Core/FFI/AppleAudioOutput.lean` | `cc/src/apple_audio_output.mm` | silent no-ops |

`AudioToolbox` is linked into `libTyrC` by `cc/Makefile`, so no extra build
flags are needed on macOS.

Capture is a **global singleton**: `AppleInput.start` opens an AudioQueue whose
callback pushes interleaved Float32 frames onto a `std::deque` FIFO, and
`read`/`readBuffer` drain it. The FIFO is capped at ~30 s of audio
(`cc/src/apple_audio_input.mm`); under sustained backpressure the oldest
audio is silently dropped. `stop` tears down the queue and clears the FIFO.
Only one capture session can be active at a time.

The currency at the model boundary is a 16 kHz mono waveform as `Array Float`
(see [Audio and speech models](models/audio-speech.md)). Inside capture loops,
samples flow through core `FloatArray` (from `readBuffer`) to avoid per-element
boxing of `Array Float`; convert once with `.data` when handing off to a model
frontend.

### Inference: a preallocated, shape-indexed KV cache

`Tyr/Lib/Inference/KVCache.lean` provides a key/value cache for transformer decode
loops. Each layer owns preallocated K and V buffers of shape
`[batch, numKvHeads, maxSeqLen, headDim]`; a decode step writes the new token's
K/V at position `cache.currentLen` (in place, via
`torch.data.sliceScatterInplace`), slices the
layer to the valid prefix, and runs attention through
`Tensor.attentionGQAQKV` (libtorch scaled-dot-product attention).

One namespace quirk: the module is named `Tyr.Lib.Inference.KVCache` but declares
`namespace torch.Generator.KVCache` (`Tyr/Lib/Inference/KVCache.lean`), a
leftover from its NanoChat origin. Call sites write
`torch.Generator.KVCache.Cache`, or `open torch.Generator` and use
`KVCache.Cache`.

**Position bookkeeping is the caller's job.** `Cache.attendLayer` appends and
attends but does *not* advance the sequence position. Call
`cache.incrementSeqLens` exactly once per generated token, after every layer of
that step has been processed. `Cache.currentLen` reads `seqLens[0]` and assumes
uniform lengths across the batch.

Both cache structures have `TensorStruct` instances
(`Tyr/Lib/Inference/KVCache.lean`), so generic traversals (`map`, `mapM`,
`zipWith`, `fold`) preserve each leaf specification. Device conversion requires
a destination type and `TensorTreeCast` instance; dtype conversion uses the
checked cast APIs. See
[TensorStruct](core/tensorstruct.md).

## Key APIs

### `Tyr.Audio.AppleInput` (`Tyr/Core/FFI/AppleAudioInput.lean`)

```lean
opaque start (sampleRate : UInt64 := 16000) (channels : UInt64 := 1)
    (bufferMs : UInt64 := 100) : IO Unit
opaque read (maxSamples : UInt64) (blockMs : UInt64 := 250) : IO (Array Float)
opaque readBuffer (maxSamples : UInt64) (blockMs : UInt64 := 250) : IO FloatArray
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
- `rms` is a native-loop RMS over a boxed array (works on all platforms).
- Off macOS, only `start` fails loudly (`IO.userError "audio input is only
  supported on macOS"`); `read`/`readBuffer` just return empty results, so a
  capture loop that never checks for `start` failure would spin silently.

### `Tyr.Audio.AppleOutput` (`Tyr/Core/FFI/AppleAudioOutput.lean`)

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
`Tyr/Core/FFI/AppleAudioOutput.lean`).

### `torch.Generator.KVCache` (`Tyr/Lib/Inference/KVCache.lean`)

`LayerCache batch maxSeqLen numKvHeads headDim dtype device` stores indexed key
and value tensors of shape `#[batch, numKvHeads, maxSeqLen, headDim]`.
`Cache` adds a layer count, an array of layer caches, sequence lengths, and capacity.
The default dtype is BFloat16. The device parameter is concrete; `Cache.init`
defaults to CPU and indexes its result by the requested device.

| Operation | Behavior |
| --- | --- |
| `Cache.init` | Allocates independent BF16 buffers in `IO`, on the requested device |
| `Cache.attendLayer` | Appends one token and returns the updated cache and indexed attention output in `IO` |
| `Cache.appendLayer` | Appends without attending, in `IO` |
| `Cache.incrementSeqLens` | Validates and advances positions in `IO`; call once per token after all layers |
| `Cache.currentLen` | Inspect the current position |
| `LayerCache.append` | Writes a token through the explicit inference copy boundary |
| `LayerCache.attentionView` | Returns indexed K/V views with runtime-dependent `validLen`, in `IO` |

The cache's runtime device must match Q/K/V. Although the structure can represent
other dtype policies, the convenience initializer creates BF16 buffers. Appends
use the returned tensors from `copySliceIO`; an earlier cache or attention view
keeps its original values. Thread the returned cache into the next decode step.
Uniquely owned eligible CPU buffers can be reused; accelerator updates currently
copy. `Cache.init` allocates independent K/V storage for every layer.
Invalid capacity, position, and layer accesses produce IO errors.

The CPU/GPU parity harness compares cached decoding with full-sequence attention.
Model families may use their own cache implementations; this shared cache does
not automatically replace them.

## Usage examples

### Recording with voice-activity detection

Reconstructed example (from `Examples/Whisper/VoiceMode.lean`, beeps
and `start`/`stop` at lines 255-256 and 368-393). The real demo records until
trailing silence, keeping a 500 ms pre-roll so speech onset is not clipped;
audio flows through unboxed `FloatArray`s and is boxed once at the model
boundary. `appendSamples`, `keepLast` and `rms` are small `FloatArray` folds
defined in that file:

```lean
import Tyr.Core.FFI.AppleAudioInput
import Tyr.Core.FFI.AppleAudioOutput

/-- Record 16 kHz mono audio until 1.5 s of trailing silence. -/
def recordUntilSilence : IO (Array Float) := do
  let sampleRate : UInt64 := 16000
  let speechRmsThreshold := 0.008                 -- VoiceMode default
  let preRollSamples := sampleRate.toNat / 2      -- 500 ms
  Tyr.Audio.AppleOutput.beepStart                 -- cue the user
  Tyr.Audio.AppleInput.start sampleRate 1 100
  let mut buf := FloatArray.emptyWithCapacity (sampleRate.toNat * 10)
  let mut recording := false
  let mut silenceSamples : Nat := 0
  while buf.size < sampleRate.toNat * 30 do       -- 30 s hard cap
    let pcm ← Tyr.Audio.AppleInput.readBuffer (sampleRate / 10) 250  -- 100 ms chunks
    if pcm.size == 0 then continue                -- read timed out
    let level := rms pcm
    buf := appendSamples buf pcm
    if level >= speechRmsThreshold then
      recording := true
      silenceSamples := 0
    else if !recording then
      buf := keepLast buf preRollSamples          -- slide the pre-roll window
    else
      silenceSamples := silenceSamples + pcm.size
      if silenceSamples >= sampleRate.toNat * 3 / 2 then break  -- 1.5 s silence
  Tyr.Audio.AppleInput.stop
  if !recording then Tyr.Audio.AppleOutput.beepNoSpeech
  pure buf.data                                   -- box once, hand to the model
```

For hop-wise streaming recognition, read fixed-size hops with the boxed
`AppleInput.read` and push each hop into the model session, stopping capture in
both the success and `catch` branches — see
`Examples/Qwen3ASR/LiveMicTrueStream.lean`.

### Cached decode step

One indexed BF16 decode step with GQA:

```lean
import Tyr.Lib.Inference.KVCache
open torch torch.Generator

def decodeStep {batch maxLen kvHeads : UInt64}
    (cache : KVCache.Cache 1 batch maxLen kvHeads 128)
    (q : Tensor { shape := #[batch, 32, 1, 128], dtype := .BFloat16, device := .CPU })
    (k v : Tensor { shape := #[batch, kvHeads, 1, 128], dtype := .BFloat16, device := .CPU }) :
    IO (KVCache.Cache 1 batch maxLen kvHeads 128 ×
      Tensor { shape := #[batch, 32, 1, 128], dtype := .BFloat16, device := .CPU }) := do
  let (cache', out) ← cache.attendLayer (numQHeads := 32) 0 q k v (enableGqa := true)
  let cache' ← cache'.incrementSeqLens
  pure (cache', out)

-- let device ← Device.cuda 0
-- let cache ← KVCache.Cache.init 1 batch maxLen kvHeads 128 (device := device)
```

The harness checks this token-by-token path against a single dense
`Tensor.attentionGQAQKV` over the full sequence and expects agreement
within BF16 tolerance — the canonical way to catch off-by-one append positions
and stale slice bounds when wiring a cache into a new model.

## Related guides

- [Audio and speech models](models/audio-speech.md) — the models these
  utilities feed: Whisper, Qwen3ASR, SileroVAD, KittenTTS, Qwen3TTS.
- [Tensors](core/tensors.md) — `Tensor σ`, indexed slices and explicit cache copies, and
  the device model used by the cache.
- [TensorStruct](core/tensorstruct.md) — the generic traversal class behind the
  cache instances.
- [Getting started](getting-started.md) — building and running the examples
  referenced above.

This chapter is a guide, not a symbol dump. Exhaustive, per-definition
documentation for the Apple audio bindings and `Tyr.Inference` is generated by doc-gen4; see
`docbuild/` for the API reference build.
