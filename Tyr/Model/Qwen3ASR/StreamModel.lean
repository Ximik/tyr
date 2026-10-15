import Tyr.Model.Qwen3ASR.ConfigIO
import Tyr.Model.Qwen3ASR.Pretrained
import Tyr.Model.Qwen3ASR.Weights
import Tyr.Model.Qwen3ASR.Streaming
import Tyr.Text.StreamingConsensus
import Tyr.Text.VADProvider
import Tyr.Tokenizer.Qwen3

namespace torch.qwen3asr

variable {device : Device}
/-- Separate streaming-native model wrapper.
    This is intentionally parallel to the existing model API so baseline paths
    keep zero additional runtime overhead. -/
structure StreamModel where
  cfg : Qwen3ASRConfig
  device : Device
  model : Qwen3ASRForConditionalGeneration (device := device) cfg
  tok : tokenizer.qwen3.QwenTokenizer
  preprocessor : PreprocessorConfig
  modelDir : String

/-- Streaming session mutable state.
    Carries rolling ASR state plus reusable decode prompt cache. -/
structure StreamSession (cfg : Qwen3ASRConfig) where
  chunkSec : Float := 2.0
  hopSec : Float := 0.5
  chunkSamples : Nat := 32000
  hopSamples : Nat := 8000
  asrState : ASRStreamingState := {}
  context : String := ""
  language : Option String := none
  textConsensus : Tyr.Text.ConsensusState := {}
  sileroVAD : Option Tyr.Text.SileroProvider := none
  decodeCache : Option (StreamingDecodeCache (device := device) cfg) := none
  promptCache : Option (StreamingPromptCache (device := device) cfg) := none
  encoderCachedFrames : UInt64 := 0
  decoderCachedTokens : UInt64 := 0

structure StreamStepOutput where
  didDecode : Bool := false
  stableAppend : String := ""
  unstableText : String := ""
  fullText : String := ""
  mode : String := "streaming_step"

private def toSamples (sec : Float) : Nat :=
  let n := ((sec * 16000.0) + 0.5).toUInt64.toNat
  if n == 0 then 1 else n

def loadFromPretrained
    (source : String)
    (revision : String := "main")
    (cacheDir : String := Hub.defaultCacheDir) : IO StreamModel := do
  let modelDir ← hub.resolvePretrainedDir source {
    revision := revision
    cacheDir := cacheDir
    includeTokenizer := true
    includePreprocessor := true
  }
  let cfg ← Qwen3ASRConfig.loadFromPretrainedDir modelDir
  let tok ← tokenizer.qwen3.loadTokenizer modelDir
  let preprocessor ← PreprocessorConfig.loadFromPretrainedDir modelDir
  let ⟨device, model⟩ ← Qwen3ASRForConditionalGeneration.loadSharded modelDir cfg
  pure { cfg, device, model, tok, preprocessor, modelDir }

def newSession
    (m : StreamModel)
    (chunkSec : Float := 2.0)
    (hopSec : Float := 0.5)
    (decodeMode : StreamingDecodeMode := .rollingWindow)
    (context : String := "")
    (language : Option String := none)
    (historyWindows : Nat := 6)
    (confirmWindows : Nat := 3)
    (rollbackTokens : Nat := 2)
    (freezeAfterSteps : Nat := 4)
    (mutableTailTokensWhileSpeech : Nat := 6)
    (sileroVADPath : Option String := none)
    : IO (StreamSession (device := m.device) m.cfg) := do
  if chunkSec <= 0.0 || hopSec <= 0.0 then
    throw <| IO.userError "chunkSec and hopSec must be > 0"
  let cCfg : Tyr.Text.ConsensusConfig := {
    historyWindows := historyWindows
    confirmWindows := confirmWindows
    rollbackTokens := rollbackTokens
    freezeAfterSteps := freezeAfterSteps
    mutableTailTokensWhileSpeech := mutableTailTokensWhileSpeech
  }
  let vad ←
    match sileroVADPath with
    | some p =>
      if p.isEmpty then
        pure none
      else
        pure (some (← Tyr.Text.initSileroProvider p))
    | none => pure none
  let asrState ← initStreamingState
    m.cfg.supportLanguages
    context
    language
    (chunkSizeSec := chunkSec)
    (stepSizeSec := hopSec)
    (decodeMode := decodeMode)
  pure {
    chunkSec := chunkSec
    hopSec := hopSec
    chunkSamples := toSamples chunkSec
    hopSamples := toSamples hopSec
    asrState := asrState
    context := context
    language := language
    textConsensus := { cfg := cCfg }
    sileroVAD := vad
  }

/-- Push PCM16k samples and optionally decode one step.
    Runs streaming decode at hop cadence and updates consensus with VAD
    stabilization signals. -/
def pushAudio
    (m : StreamModel)
    (s : StreamSession (device := m.device) m.cfg)
    (pcm16k : Array Float)
    (maxNewTokens : UInt64 := 128)
    : IO ((StreamSession (device := m.device) m.cfg) × StreamStepOutput) := do
  let mut s' := s
  let vadSignal : Tyr.Text.VADSignal ←
    match s'.sileroVAD with
    | some p =>
      let (p', sig) ← Tyr.Text.stepSileroProvider p pcm16k
      s' := { s' with sileroVAD := some p' }
      pure sig
    | none =>
      pure { speechActive := true, boundary := false }

  let beforeChunkId := s'.asrState.chunkId
  let (asrNext, decodeCacheNext) ← streamingTranscribeWithModelStateCached
    m.model
    m.tok
    m.preprocessor
    pcm16k
    s'.asrState
    (cache := s'.decodeCache)
    (maxNewTokens := maxNewTokens)
  let promptCacheNext := decodeCacheNext.bind (fun c => c.promptCache)
  s' := { s' with asrState := asrNext, decodeCache := decodeCacheNext, promptCache := promptCacheNext }
  let decodedSteps := asrNext.chunkId - beforeChunkId
  if decodedSteps == 0 then
    pure (s', { didDecode := false, stableAppend := "", unstableText := "", fullText := "", mode := "streaming_step" })
  else
    let ids := tokenizer.qwen3.encodeText m.tok asrNext.text
    let (cs', delta) := Tyr.Text.updateWithSignals
      s'.textConsensus ids vadSignal.speechActive vadSignal.boundary (tokenizer.qwen3.decodeText m.tok)
    let decodedSamples := decodedSteps * asrNext.stepSizeSamples
    let s'' := {
      s' with
      textConsensus := cs'
      encoderCachedFrames := s'.encoderCachedFrames + decodedSamples.toUInt64
      decoderCachedTokens := ids.size.toUInt64
    }
    pure (s'', {
      didDecode := true
      stableAppend := delta.stableAppend
      unstableText := delta.unstableText
      fullText := delta.fullText
      mode := "streaming_step"
    })

def flush
    (m : StreamModel)
    (s : StreamSession (device := m.device) m.cfg)
    (maxNewTokens : UInt64 := 128)
    : IO ((StreamSession (device := m.device) m.cfg) × StreamStepOutput) := do
  let beforeChunkId := s.asrState.chunkId
  let (asrNext, decodeCacheNext) ← finishStreamingTranscribeWithModelStateCached
    m.model
    m.tok
    m.preprocessor
    s.asrState
    (cache := s.decodeCache)
    (maxNewTokens := maxNewTokens)
  let promptCacheNext := decodeCacheNext.bind (fun c => c.promptCache)
  let s1 := { s with asrState := asrNext, decodeCache := decodeCacheNext, promptCache := promptCacheNext }
  if asrNext.chunkId == beforeChunkId then
    pure (s1, { didDecode := false, stableAppend := "", unstableText := "", fullText := "", mode := "streaming_step" })
  else
    let ids := tokenizer.qwen3.encodeText m.tok asrNext.text
    let (cs', delta) := Tyr.Text.updateWithSignals
      s1.textConsensus ids false true (tokenizer.qwen3.decodeText m.tok)
    let s' := {
      s1 with
      textConsensus := cs'
      decoderCachedTokens := ids.size.toUInt64
    }
    pure (s', {
      didDecode := true
      stableAppend := delta.stableAppend
      unstableText := delta.unstableText
      fullText := delta.fullText
      mode := "streaming_step"
    })

end torch.qwen3asr
