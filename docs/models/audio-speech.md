# Audio and speech models

Tyr ships Lean-native ports of five audio model families under `Tyr/Model/`: **KittenTTS**
and **Qwen3TTS** for text-to-speech, **Qwen3ASR** and **Whisper** for speech recognition,
and **SileroVAD** for voice-activity detection. Use them when you want inference with
tensor shapes checked at elaboration time and no Python in the hot path; microphone
capture and playback live in the separate `Tyr/Audio` layer (see
[Audio capture and playback](../audio.md)). All five are inference ports loading
pretrained safetensors checkpoints, not training code.

## Architecture and main abstractions

All five families follow the same idiom as the rest of `Tyr/Model`:

The `Tensor { shape := shape, device := device }` fields shown below preserve their shape index
with a shared concrete device and dynamic dtype. They do not assert Float32;
checked loading and model adapters validate the metadata they require.

- A plain **config structure** (e.g. `WhisperConfig`, `Qwen3ASRConfig`) with field
  defaults, parsed from a checkpoint's `config.json` via
  `<Config>.loadFromPretrainedDir modelDir`.
- A **parameter structure indexed by the config**, with shapes computed from config
  fields at the type level and one shared device parameter. `deriving TensorStruct`
  supports parameter traversal and detachment. Loaders request a concrete device;
  converting an existing model uses `TensorStruct.castTo` with a target model type
  and a `TensorTreeCast` instance.
  See [TensorStruct](../core/tensorstruct.md).
- Weights loaded from safetensors with shape-checked names, sharded variants for
  multi-file checkpoints through the checked `Tyr.SafeTensors` loaders; see
  [Serialization](../serialization.md).
- A **pretrained resolution** step that accepts either a local directory or a
  HuggingFace repo id and downloads/caches through `Tyr.Hub`.

The common audio currency is a 16 kHz mono waveform as `Array Float`. WAV loading,
resampling, and log-mel features are shared through the Qwen3-ASR frontend
(`Tyr/Model/Qwen3ASR/Frontend.lean`), reused by SileroVAD, Whisper, and Qwen3TTS.

### KittenTTS (`torch.kittentts`)

A Kokoro/StyleTTS2-style single-utterance TTS port. Import `Tyr.Model.KittenTTS`.

```lean
structure Model (cfg : KittenTTSConfig) where
  bert : KittenAlbert cfg.nToken cfg.plbert
  bertEncoder : LinearNorm cfg.plbert.hiddenSize cfg.hiddenDim
  predictor : ProsodyPredictor cfg
  textEncoder : TextEncoder cfg
  decoder : Decoder cfg
  deriving TensorStruct
```

`Tyr/Model/KittenTTS/Model.lean:1693`. Pipeline: ALBERT phoneme encoder →
duration/prosody predictor (durations, F0/N curves) → text encoder → AdaIN decoder with
an hn-NSF + iSTFT generator. Configuration is layered: `KittenTTSConfig`
(`Tyr/Model/KittenTTS/Config.lean:139`) embeds `AlbertConfig` and `GeneratorConfig`;
`KittenTTSConfig.fullStyleDim cfg = 2 * cfg.styleDim` is the voice-style width.

Checkpoint typing happens at elaboration: `Tyr/Model/KittenTTS/Checkpoint.lean:14` runs
`safetensors_type_provider "Tyr/Model/KittenTTS/kokoro_v1.schema.json" as KokoroCheckpoint`,
generating typed structures for the whole Kokoro checkpoint from a checked-in schema;
`Weights.lean` maps those onto `Model cfg` (`Model.loadAutoConfig`).

The user-facing entry point is `PretrainedBundle` (`Tyr/Model/KittenTTS/Pretrained.lean:170`),
a record of `cfg`, `model`, phoneme `vocab`, and the source/revision/cache paths.
Input is **phoneme IDs**, not raw text — grapheme-to-phoneme conversion is out of scope
for this port; call sites pass IPA strings (e.g. `"həˈloʊ wɜːld"`). Voice styles are rows
of a `[510, 1, fullStyleDim]` safetensors table selected by phoneme count
(`PretrainedBundle.loadVoiceStyle`).

### Qwen3TTS (`torch.qwen3tts`)

A Qwen3-TTS codec-LM port: an autoregressive **talker** generates speech-tokenizer
codes, a **speech tokenizer decoder** turns codes into 24 kHz audio. Import
`Tyr.Model.Qwen3TTS`.

```lean
structure Qwen3TTSForConditionalGeneration (cfg : Qwen3TTSConfig) where
  talker : TalkerForConditionalGeneration cfg.talkerConfig
  speakerEncoder : Option (SpeakerEncoder cfg.speakerEncoderConfig) := none
```

`Tyr/Model/Qwen3TTS/Model.lean:25`. Pieces:

- **Talker** (`Tyr/Model/Qwen3TTS/Talker.lean`): `TalkerModel` (codec + text embeddings,
  text projection, transformer stack) plus `TalkerCodePredictor`, which predicts
  codebooks 1..N from codebook-0 context. Both reuse the Qwen LLM layer — `TalkerLayer
  cfg` is an abbrev for `qwen.QwenLayer` (Talker.lean:26).
- **Speaker encoder** (`SpeakerEncoder.lean`): ECAPA-TDNN producing
  `Tensor { shape := #[batch, cfg.speakerEncoderConfig.encDim], device := device }` embeddings from mel features; only
  present for `ttsModelType == "base"`. Used for voice cloning via
  `VoiceClonePromptItem` (Model.lean:17).
- **Speech tokenizer decoder** (`SpeechTokenizer.lean:263`): `SpeechTokenizer12HzDecoder`
  is a flat parameter record (split RVQ, causal pre-conv + transformer, upsampling conv
  stack; `outputSampleRate = 24000`, `decodeUpsampleRate = 1920`) decoding
  `[batch, 16, frames]` codes to `[batch, 1, frames * 1920]` waveforms — one shot
  (`decode`), chunked with left context (`decodeChunked`), or incrementally.
- **Speech tokenizer encoders** (`SpeechTokenizerEncoder.lean`,
  `SpeechTokenizer25HzEncoder.lean`): Lean-native audio → codes for the 12 Hz and 25 Hz
  variants (`encode`, `encodeMonoFrameMajor`), used for voice-clone reference codes.
- **Streaming** (`Tyr/Model/Qwen3TTS/Streaming.lean`): `streamFromTalkerInputs` drives
  frame-by-frame generation with per-frame and per-audio-chunk callbacks, optionally
  decoding in-process through a `SpeechTokenizer12HzDecoder.DecodeStreamState`. In-process
  streaming decode requires `cfg.talkerConfig.numCodeGroups == 16` (the upstream 12 Hz
  tokenizer); otherwise decoding falls back to the legacy Python bridge in
  `SpeechTokenizerBridge.lean`.

### Qwen3ASR (`torch.qwen3asr`)

A Qwen3-ASR "thinker" port covering offline transcription, streaming, and forced
alignment. Import `Tyr.Model.Qwen3ASR`.

```lean
variable {device : Device}

structure Qwen3ASRThinkerForConditionalGeneration (cfg : ThinkerConfig) where
  audioTower : AudioEncoder (device := device) cfg.audioConfig
  textModel : qwen.Qwen3Model (device := device) (TextQwenConfig cfg) (.Unknown "dynamic")
  audioProjectionWeight : Tensor { shape := #[cfg.textConfig.hiddenSize, cfg.audioConfig.outputDim], device := device }
  audioProjectionBias : Tensor { shape := #[cfg.textConfig.hiddenSize], device := device }
  lmHead : Tensor { shape := #[ThinkerLmVocabSize cfg, cfg.textConfig.hiddenSize], device := device }
  deriving TensorStruct

structure Qwen3ASRForConditionalGeneration (cfg : Qwen3ASRConfig) where
  thinker : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg.thinkerConfig
  supportLanguages : Array String := cfg.supportLanguages
  deriving TensorStruct
```

`Tyr/Model/Qwen3ASR/Model.lean:130` and `:893`. The audio encoder's output is projected
into the text hidden size and spliced into the token embedding stream; the text decoder
is a standard Qwen3 model (see [LLM models](llms.md)). Generation is greedy:
`generateGreedy`, `generateGreedyUncached`, and `generateGreedyWithPromptCache` over
per-layer `LayerKVCache` / reusable `StreamingPromptCache`.

Configs are layered (`Tyr/Model/Qwen3ASR/Config.lean`): `AudioEncoderConfig`,
`TextConfig` (with `toQwenConfig`), `ThinkerConfig`, `Qwen3ASRConfig`.
`ThinkerConfig.isForcedAligner` detects aligner checkpoints by `model_type`, and
`lmHeadOutDim` switches the LM head between vocab size and `classifyNum` accordingly.

The **frontend** (`Frontend.lean`) provides the unified audio input surface plus
Whisper-style log-mel extraction (`waveformToWhisperFeaturesDynamic`,
`normalizeAudioTo16kFromWav`, `normalizeAudioInputTo16k`):

```lean
inductive ASRAudioInput where
  | wavPath (path : String)
  | url (value : String)
  | base64 (value : String)
  | waveform (samples : Array Float) (sampleRate : UInt64)
```

The **offline API** (`Transcribe.lean`) returns
`ASRTranscription` (`language`, `text`, `timeStamps : Option ForcedAlignResult`) through
`transcribeWaveform(s)`, `transcribeWav(s)`, `transcribeAudioInput(s)`, and
`transcribeAudioSource(s)` — both free functions and method-style wrappers on
`Qwen3ASRForConditionalGeneration`. All take the model, a `tokenizer.qwen3.QwenTokenizer`,
and a `PreprocessorConfig`, plus optional `forcedAligner`, `context`, `language`,
`returnTimeStamps`, `maxNewTokens`, and `eosTokenIds` arguments. Long audio is chunked
internally (20 min max per chunk, 3 min when timestamps are requested).

The **streaming API** (`Streaming.lean`) is state-based. `ASRStreamingState` is created
by `initStreamingState` (window/hop sizes, context, forced language, decode mode),
advanced by `streamingTranscribe`, and closed by `finishStreamingTranscribe`:

```lean
inductive StreamingDecodeMode where
  | rollingWindow
  | fullAccumulation

abbrev StreamingDecodeFn := String → Array Float → IO String

def streamingTranscribe
    (tok : tokenizer.qwen3.QwenTokenizer) (decodeFn : StreamingDecodeFn)
    (pcm16k : Array Float) (state : ASRStreamingState) : IO ASRStreamingState
```

`decodeFn` abstracts the model call; the `...WithModel`, `...WithModelCached`, and
`...WithModelStateCached` families in the same file plug the Lean model in directly, with
`StreamingDecodeCache` carrying prompt/frontend/encoder caches across hops. On top,
`StreamModel.lean` provides a session-style wrapper — `StreamModel.loadFromPretrained`,
`newSession`, `pushAudio`, `flush` — returning `StreamStepOutput` with
`stableAppend`/`unstableText`/`fullText` from a text-consensus layer
(`Tyr.Text.StreamingConsensus`), optionally gated by a Silero VAD provider.
`Realtime.lean` is the lower-level coherent-transcript assembly.

The **forced aligner** (`Transcribe.lean:29`, `ForcedAligner.lean`) is a separate
checkpoint wrapped as `Qwen3ForcedAligner`; `alignWaveform`/`align`/`alignBatch` map a
transcript to word-level spans:

```lean
structure ForcedAlignItem where
  text : String
  startTime : Float
  endTime : Float

structure ForcedAlignResult where
  items : Array ForcedAlignItem
```

Official HF repos are `Qwen/Qwen3-ASR-0.6B` and `Qwen/Qwen3-ASR-1.7B`
(`Tyr/Model/Qwen3ASR/Pretrained.lean:25`).

### SileroVAD (`torch.silerovad`)

The tiny Silero voice-activity detector. Import `Tyr.Model.SileroVAD`. 16 kHz mono only,
512-sample chunks. `SileroVAD` (`Tyr/Model/SileroVAD/Model.lean:99`) is a flat parameter
record: STFT conv front-end, 4-layer conv encoder, hand-rolled LSTM cell (`VADLstmState`),
sigmoid probability head. `SileroVAD.init` builds random weights for shape tests;
`SileroVAD.load path` reads the pretrained checkpoint.

`SileroVADRuntime` (Model.lean:251) is the stateful streaming wrapper: `step rt chunk`
runs one 512-sample chunk and returns `IO (Float × SileroVADRuntime)`; `audioForward rt
audio` maps over a whole waveform (zero-padding the tail). On top, `Utils.lean` ports the
Silero segmentation heuristics: `timestampsFromProbabilities` under a `TimestampConfig`
(thresholds, min speech/silence durations, padding), `getSpeechTimestamps`,
`timestampsToSeconds`, the streaming `VADIterator` emitting `.start`/`.stop` boundaries,
and `collectChunks`/`dropChunks` to cut audio by timestamps.

### Whisper (`torch.whisper`)

A native Whisper encoder-decoder port with a whisper.cpp-style decode loop. Import
`Tyr.Model.Whisper`. `WhisperConfig` (`Tyr/Model/Whisper/Config.lean:10`) holds the usual
HF Whisper fields (`numMelBins`, `dModel`, `encoderLayers`, `decoderLayers`,
`maxTargetPositions`, suppress-token lists, ...). The model split:

```lean
variable {device : Device}

structure WhisperForConditionalGeneration (cfg : WhisperConfig) where
  model : WhisperModel (device := device) cfg      -- encoder + decoder stacks
  projOut : Tensor { shape := #[cfg.vocabSize, cfg.dModel], device := device }
  deriving TensorStruct
```

with `encode`, `decode`, and the incremental path `initLayerKVCaches` +
`precomputeCrossCaches` + `decodeStepWithCache` (`Tyr/Model/Whisper/Model.lean`).
`WhisperForConditionalGeneration.loadSharded modelDir cfg` loads HF-layout weights, ties
`projOut` to the token embedding when missing, and picks a device from `TYR_DEVICE`
(`cpu`/`cuda`/`mps`/`auto`), falling back to `getBestDevice`. Its result packages
that placement as `Σ device, WhisperForConditionalGeneration (device := device) cfg`.
Use `loadShardedAt` to request a device explicitly; bundles retain the device
alongside the indexed model.

The user API (`Tyr/Model/Whisper/Transcribe.lean`) is a bundle plus transcribe functions:

```lean
structure WhisperBundle where
  cfg : WhisperConfig
  device : Device
  model : WhisperForConditionalGeneration (device := device) cfg
  tok : tokenizer.qwen3.QwenTokenizer
  preprocessor : PreprocessorConfig

def loadFromPretrainedDir (modelDir : String) : IO WhisperBundle

def transcribeWav
    (model : WhisperForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer) (pre : PreprocessorConfig)
    (wavPath : String) (language : String := "en")
    (maxNewTokens : UInt64 := 128)      -- 0 = model maximum
    (noTimestamps : Bool := true) (decodeOpts : WhisperDecodeOptions := {})
    : IO WhisperTranscription
```

Note the tokenizer is the shared Qwen3 BPE tokenizer (`Tyr/Tokenizer/Qwen3`), not a
separate Whisper tokenizer. Decoding (`WhisperDecodeOptions`, Transcribe.lean:34) mirrors
whisper-cli: beam-first at temperature 0, temperature fallback with best-of, no-speech
and compression-ratio gating, rolling context across overlapping 30 s chunks.
`transcribeWaveform16k` is the in-memory variant; `WhisperTranscription` carries
`language`, `text`, and `tokenIds`.

## Key APIs

Loading:

| Family | Config | Weights | One-shot bundle |
|---|---|---|---|
| KittenTTS | `KittenTTSConfig.loadFromPretrainedDir` | `Model.loadAutoConfig path cfg` | `Model.loadFromPretrained source` → `PretrainedBundle` |
| Qwen3TTS | `Qwen3TTSConfig.loadFromPretrainedDir` | `Qwen3TTSForConditionalGeneration.loadSharded modelDir cfg device` | — |
| Qwen3ASR | `Qwen3ASRConfig.loadFromPretrainedDir` | `Qwen3ASRForConditionalGeneration.loadSharded modelDir cfg` | `Qwen3ASRForConditionalGeneration.loadFromPretrained source`; `StreamModel.loadFromPretrained source`; `Qwen3ForcedAligner.loadFromPretrained source` |
| SileroVAD | — (fixed shapes) | `SileroVAD.load path` | — |
| Whisper | `WhisperConfig.loadFromPretrainedDir` | `WhisperForConditionalGeneration.loadSharded modelDir cfg` | `loadFromPretrainedDir modelDir` → `WhisperBundle` |

Running:

- KittenTTS: `Model.synthesizeIds m inputIds refStyle speed`, `Model.predictDurations`;
  bundle-level `synthesizePhonemes`, `synthesizePhonemesToFile`, `predictPhonemeDurations`
  — returning `KittenTTSOutput` / `KittenTTSDurationPrediction`.
- Qwen3TTS: `generateFrame`, `generateCodes`, `generateCodesWithLengths`,
  `generateFromText`, `generateFromInstructText` on `Qwen3TTSForConditionalGeneration`;
  decoding via `SpeechTokenizer12HzDecoder.decodeFrameMajorToWav` /
  `decodeFrameMajorChunkedToWav`; streaming via `streamFromTalkerInputs` with
  `StreamingOptions`/`StreamingCallbacks`.
- Qwen3ASR: `transcribeWav(s)`, `transcribeWaveform(s)`, `transcribeAudioSource(s)` →
  `ASRTranscription`; streaming via `initStreamingState` / `streamingTranscribe` /
  `finishStreamingTranscribe` or the `StreamModel` session trio `newSession` /
  `pushAudio` / `flush`; alignment via `Qwen3ForcedAligner.alignWaveform` / `alignBatch`.
- SileroVAD: `SileroVADRuntime.init` / `step` / `audioForward`; `getSpeechTimestamps`,
  `timestampsFromProbabilities`, `VADIterator.init` / `step`; `readAudio path` for WAV
  input.
- Whisper: `transcribeWav`, `transcribeWaveform16k` with `WhisperDecodeOptions`.

## Usage examples

Reconstructed example (from `Examples/KittenTTSPretrained.lean`):

```lean
import Tyr.Model.KittenTTS

open torch torch.kittentts

def main (args : List String) : IO UInt32 := do
  let bundle ← Model.loadFromPretrained "hexgrad/Kokoro-82M"
  let out ← bundle.synthesizePhonemesToFile "həˈloʊ wɜːld" "af_heart" "out.wav" 1.0
  IO.println s!"frames={out.alignmentFrames} shape={out.audio.runtimeShape}"
  return 0
```

Reconstructed example (from `Examples/Qwen3TTS/EndToEnd.lean`):

```lean
import Tyr.Model.Qwen3TTS
import Tyr.Tokenizer.Qwen3

open torch torch.qwen3tts

let cfg ← Qwen3TTSConfig.loadFromPretrainedDir modelDir
let model ← Qwen3TTSForConditionalGeneration.loadSharded modelDir cfg device
let tok ← tokenizer.qwen3.loadTokenizer modelDir
-- example-local helper: folds text, language id, speaker embedding into talker inputs
let talkerInputs ← buildTalkerInputsEquivalent cfg model tokenIds languageId thinkMode speakerEmbed

-- offline: generate codec frames, then decode to 24 kHz WAV
let out ← TalkerForConditionalGeneration.generateCodesWithLengths
  cfg.talkerConfig model.talker talkerInputs maxFrames 2 temperature topK topP ...
let dec ← SpeechTokenizer12HzDecoder.loadFromDir s!"{modelDir}/speech_tokenizer" device
dec.decodeFrameMajorChunkedToWav codes16 "out.wav"

-- or true streaming: audio chunks appended to the WAV as frames are generated
let out ← model.streamFromTalkerInputs talkerInputs streamOpts (some dec)
  { onAudioChunk := fun chunk => Tensor.wavAppend chunk "out.wav" }
```

Reconstructed example (from `Examples/Qwen3ASR/Transcribe.lean`):

```lean
import Tyr.Model.Qwen3ASR
import Tyr.Model.Qwen3ASR.StreamModel

open torch.qwen3asr

-- offline, with optional forced-alignment timestamps
let modelDir ← hub.resolvePretrainedDir "Qwen/Qwen3-ASR-0.6B" {}
let cfg ← Qwen3ASRConfig.loadFromPretrainedDir modelDir
let tok ← tokenizer.qwen3.loadTokenizer modelDir
let pre ← PreprocessorConfig.loadFromPretrainedDir modelDir
let ⟨device, model⟩ ← Qwen3ASRForConditionalGeneration.loadSharded modelDir cfg
let outs ← model.transcribeWavs tok pre #["MLKDream.wav"]
  (languages := #[some "English"]) (returnTimeStamps := false)
IO.println (outs.getD 0 default).text

-- streaming, session style
let sm ← StreamModel.loadFromPretrained "Qwen/Qwen3-ASR-0.6B"
let mut ss ← newSession sm (chunkSec := 2.0) (hopSec := 0.5)
for chunk in pcm16kChunks do
  let (ss', step) ← pushAudio sm ss chunk
  ss := ss'
  if step.didDecode then IO.println step.fullText
let _ ← flush sm ss
```

Reconstructed example (from `Examples/Whisper/Transcribe.lean`):

```lean
import Tyr.Model.Whisper

open torch.whisper

let bundle ← loadFromPretrainedDir "weights/whisper-base.en"
let out ← transcribeWav bundle.model bundle.tok bundle.preprocessor
  "MLKDream.wav" "en" 0 true { beamSize := 1, noFallback := true }
IO.println out.text
```

Reconstructed example (from `Tests/TestSileroVAD.lean`):

```lean
import Tyr.Model.SileroVAD

open torch torch.silerovad

let model ← SileroVAD.load "silero_vad_16k.safetensors"
let rt := SileroVADRuntime.init model
let (probs, _) ← rt.audioForward wav16k
let tss ← timestampsFromProbabilities probs (audioLengthSamples := wav16k.size) {}
let (p, rt1) ← rt.step (Array.replicate 512 0.0)   -- single-chunk streaming form
```

For live microphone input, `Examples/Qwen3ASR/LiveMic.lean` and `Examples/Whisper/VoiceMode.lean`
combine these models with the `Tyr/Audio` capture layer — see
[Audio capture and playback](../audio.md).

## Related guides

- [Getting started](../getting-started.md) — build and run the examples
- [Tensors](../core/tensors.md) — the indexed `Tensor σ` used throughout these models
- [TensorStruct](../core/tensorstruct.md) — parameter traversal and typed model conversion
- [Serialization](../serialization.md) — safetensors loading and the type provider
- [LLM models](llms.md) — the Qwen layers reused by Qwen3TTS and Qwen3ASR
- [Audio capture and playback](../audio.md) — microphone/speaker IO (`Tyr/Audio`)
- [Examples and testing](../examples-and-testing.md) — the example executables cited above

Exhaustive per-symbol documentation for these modules is generated by doc-gen4 (see `docbuild/`).
