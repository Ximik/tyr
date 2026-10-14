/-
  Tyr/Model/Qwen3TTS/Model.lean

  Top-level Qwen3-TTS model in Lean4.
  Mirrors the upstream model split:
  - talker (codec generation)
  - optional speaker encoder (base model mode)
-/
import Tyr.Typed.Neural
import Tyr.Model.Qwen3TTS.Config
import Tyr.Model.Qwen3TTS.SpeakerEncoder
import Tyr.Model.Qwen3TTS.Talker

namespace torch.qwen3tts

/-- Voice clone prompt payload for one sample. -/
structure VoiceClonePromptItem (cfg : Qwen3TTSConfig) where
  refCode : Option (Sigma fun shape => Tensor { shape := shape, dtype := .Int64 }) := none
  refSpeakerEmbedding : Tensor { shape := #[cfg.speakerEncoderConfig.encDim] }
  xVectorOnlyMode : Bool := false
  iclMode : Bool := false
  deriving Inhabited

/-- Top-level Qwen3-TTS conditional generation model. -/
structure Qwen3TTSForConditionalGeneration (cfg : Qwen3TTSConfig) where
  talker : TalkerForConditionalGeneration cfg.talkerConfig
  speakerEncoder : Option (SpeakerEncoder cfg.speakerEncoderConfig) := none

namespace Qwen3TTSForConditionalGeneration

def init (cfg : Qwen3TTSConfig) : IO (Qwen3TTSForConditionalGeneration cfg) := do
  let talker ← TalkerForConditionalGeneration.init cfg.talkerConfig
  let speakerEncoder ←
    if cfg.ttsModelType == "base" then
      let enc ← SpeakerEncoder.init cfg.speakerEncoderConfig
      pure (some enc)
    else
      pure none
  pure { talker, speakerEncoder }

def getSupportedLanguages (_m : Qwen3TTSForConditionalGeneration cfg) : Array String :=
  Qwen3TTSConfig.supportedLanguages cfg

def getSupportedSpeakers (_m : Qwen3TTSForConditionalGeneration cfg) : Array String :=
  Qwen3TTSConfig.supportedSpeakers cfg

/-- Extract speaker embedding from mel features `[batch, frames, melDim]`.
    Features are aligned to the speaker encoder's weight dtype and device.
    The caller is responsible for mel preprocessing and sample-rate alignment. -/
def extractSpeakerEmbedding {batch frames : UInt64}
    (m : Qwen3TTSForConditionalGeneration cfg)
    (mel : Tensor { shape := #[batch, frames, cfg.speakerEncoderConfig.melDim] })
    : IO (Tensor { shape := #[batch, cfg.speakerEncoderConfig.encDim] }) := do
  match m.speakerEncoder with
  | some enc => do
      let mel := Tensor.castLike enc.tdnn0.weight mel
      let mel : Tensor { shape := #[batch, frames, cfg.speakerEncoderConfig.melDim] } :=
        if mel.device == enc.tdnn0.weight.device then mel else mel.to enc.tdnn0.weight.device
      pure (enc.forward mel)
  | none => throw (IO.userError "Speaker encoder is unavailable for this Qwen3-TTS model type")

/-- Build talker input embeddings from text token IDs (`[batch, textSeq]`).
    Prepends codec BOS embedding. -/
def buildTalkerInputsFromText {batch textSeq : UInt64}
    (m : Qwen3TTSForConditionalGeneration cfg)
    (textIds : Tensor { shape := #[batch, textSeq], dtype := .Int64 })
    : Tensor { shape := #[batch, 1 + textSeq, cfg.talkerConfig.hiddenSize] } :=
  TalkerModel.buildInputsFromText cfg.talkerConfig m.talker.model textIds

/-- Build talker input embeddings from instruct + text token IDs.
    Prepends codec BOS embedding. -/
def buildTalkerInputsFromInstructText {batch instructSeq textSeq : UInt64}
    (m : Qwen3TTSForConditionalGeneration cfg)
    (instructIds : Tensor { shape := #[batch, instructSeq], dtype := .Int64 })
    (textIds : Tensor { shape := #[batch, textSeq], dtype := .Int64 })
    : Tensor { shape := #[batch, 1 + (instructSeq + textSeq), cfg.talkerConfig.hiddenSize] } :=
  TalkerModel.buildInputsFromInstructText cfg.talkerConfig m.talker.model instructIds textIds

/-- Generate one codec frame (`numCodeGroups` tokens per sample) from talker inputs.
    Inputs are already-projected talker embeddings. -/
def generateFrame {batch seq : UInt64}
    (m : Qwen3TTSForConditionalGeneration cfg)
    (talkerInputs : Tensor { shape := #[batch, seq, cfg.talkerConfig.hiddenSize] })
    (attnMask : Option (Tensor { shape := #[batch, seq], dtype := .Int64 }) := none)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    : IO (Tensor { shape := #[batch, cfg.talkerConfig.numCodeGroups], dtype := .Int64 }) :=
  return (← TalkerForConditionalGeneration.generateFrame
    cfg.talkerConfig m.talker talkerInputs attnMask
    none true temperature topK topP).codes

/-- Generate multiple codec frames autoregressively from talker context.
    Returns `[batch, maxFrames, numCodeGroups]`. -/
def generateCodes {batch seq : UInt64}
    (m : Qwen3TTSForConditionalGeneration cfg)
    (talkerInputs : Tensor { shape := #[batch, seq, cfg.talkerConfig.hiddenSize] })
    (maxFrames : UInt64 := 256)
    (minNewTokens : UInt64 := 2)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    : IO (Tensor { shape := #[batch, maxFrames, cfg.talkerConfig.numCodeGroups], dtype := .Int64 }) :=
  TalkerForConditionalGeneration.generateCodes
    cfg.talkerConfig m.talker talkerInputs maxFrames minNewTokens temperature topK topP

/-- Generate multiple codec frames with EOS-aware stopping metadata.
    `codes` is padded to `[batch, maxFrames, numCodeGroups]`.
    `lengths` stores the first EOS step per sample (or `maxFrames` if no EOS). -/
def generateCodesWithLengths {batch seq : UInt64}
    (m : Qwen3TTSForConditionalGeneration cfg)
    (talkerInputs : Tensor { shape := #[batch, seq, cfg.talkerConfig.hiddenSize] })
    (maxFrames : UInt64 := 256)
    (minNewTokens : UInt64 := 2)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    : IO (TalkerForConditionalGeneration.CodeGenerationOutput batch maxFrames cfg.talkerConfig.numCodeGroups) :=
  TalkerForConditionalGeneration.generateCodesWithLengths
    cfg.talkerConfig m.talker talkerInputs maxFrames minNewTokens temperature topK topP

/-- End-to-end generation from text token IDs.
    Internally builds talker inputs and runs EOS-aware codec generation. -/
def generateFromText {batch textSeq : UInt64}
    (m : Qwen3TTSForConditionalGeneration cfg)
    (textIds : Tensor { shape := #[batch, textSeq], dtype := .Int64 })
    (maxFrames : UInt64 := 256)
    (minNewTokens : UInt64 := 2)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    : IO (TalkerForConditionalGeneration.CodeGenerationOutput batch maxFrames cfg.talkerConfig.numCodeGroups) := do
  let talkerInputs := buildTalkerInputsFromText m textIds
  generateCodesWithLengths m talkerInputs maxFrames minNewTokens temperature topK topP

/-- End-to-end generation from instruct+text token IDs.
    Internally builds talker inputs and runs EOS-aware codec generation. -/
def generateFromInstructText {batch instructSeq textSeq : UInt64}
    (m : Qwen3TTSForConditionalGeneration cfg)
    (instructIds : Tensor { shape := #[batch, instructSeq], dtype := .Int64 })
    (textIds : Tensor { shape := #[batch, textSeq], dtype := .Int64 })
    (maxFrames : UInt64 := 256)
    (minNewTokens : UInt64 := 2)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    : IO (TalkerForConditionalGeneration.CodeGenerationOutput batch maxFrames cfg.talkerConfig.numCodeGroups) := do
  let talkerInputs := buildTalkerInputsFromInstructText m instructIds textIds
  generateCodesWithLengths m talkerInputs maxFrames minNewTokens temperature topK topP

end Qwen3TTSForConditionalGeneration

end torch.qwen3tts
