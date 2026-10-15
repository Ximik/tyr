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

variable {device : Device} {dtypes : DTypeLayout}

/-- Voice clone prompt payload for one sample. -/
structure VoiceClonePromptItem (cfg : Qwen3TTSConfig) (dtype : ExactDType := float32DType) where
  refCode : Option (Sigma fun shape => Tensor { shape := shape, dtype := .Int64, device := device }) := none
  refSpeakerEmbedding : Tensor { shape := #[cfg.speakerEncoderConfig.encDim], dtype := dtype.val, device := device }
  xVectorOnlyMode : Bool := false
  iclMode : Bool := false
  deriving Nonempty

/-- Top-level Qwen3-TTS conditional generation model. -/
structure Qwen3TTSForConditionalGeneration (cfg : Qwen3TTSConfig) (dtypes : DTypeLayout := {}) where
  talker : TalkerForConditionalGeneration (dtype := dtypes.talker) (device := device) cfg.talkerConfig
  speakerEncoder : Option (SpeakerEncoder (dtype := dtypes.speaker) (device := device) cfg.speakerEncoderConfig) := none

namespace Qwen3TTSForConditionalGeneration

def init (cfg : Qwen3TTSConfig) (device : Device := .CPU) (dtypes : DTypeLayout := {}) : IO (Qwen3TTSForConditionalGeneration (device := device) cfg dtypes) := do
  let talker ← TalkerForConditionalGeneration.init (dtype := dtypes.talker) (device := device) cfg.talkerConfig
  let speakerEncoder ←
    if cfg.ttsModelType == "base" then
      let enc ← SpeakerEncoder.init (dtype := dtypes.speaker) (device := device) cfg.speakerEncoderConfig
      pure (some enc)
    else
      pure none
  pure { talker, speakerEncoder }

def getSupportedLanguages (_m : Qwen3TTSForConditionalGeneration (device := device) cfg dtypes) : Array String :=
  Qwen3TTSConfig.supportedLanguages cfg

def getSupportedSpeakers (_m : Qwen3TTSForConditionalGeneration (device := device) cfg dtypes) : Array String :=
  Qwen3TTSConfig.supportedSpeakers cfg

/-- Extract speaker embedding from mel features `[batch, frames, melDim]`.
    Features are aligned to the speaker encoder's weight dtype and device.
    The caller is responsible for mel preprocessing and sample-rate alignment. -/
def extractSpeakerEmbedding {batch frames : UInt64} {inputDType : DType}
    (m : Qwen3TTSForConditionalGeneration (device := device) cfg dtypes)
    (mel : Tensor { shape := #[batch, frames, cfg.speakerEncoderConfig.melDim], dtype := inputDType, device := device })
    : IO (Tensor { shape := #[batch, cfg.speakerEncoderConfig.encDim], dtype := dtypes.speaker.val, device := device }) := do
  match m.speakerEncoder with
  | some enc => do
      -- Mel features are an explicit preprocessing boundary. Conversion changes
      -- their dtype, while all speaker weights and activations share one index.
      let mel ← IO.ofExcept (mel.castDType? dtypes.speaker.val)
      pure (enc.forward mel)
  | none => throw (IO.userError "Speaker encoder is unavailable for this Qwen3-TTS model type")

/-- Build talker input embeddings from text token IDs (`[batch, textSeq]`).
    Prepends codec BOS embedding. -/
def buildTalkerInputsFromText {batch textSeq : UInt64}
    (m : Qwen3TTSForConditionalGeneration (device := device) cfg dtypes)
    (textIds : Tensor { shape := #[batch, textSeq], dtype := .Int64, device := device })
    : Tensor { shape := #[batch, 1 + textSeq, cfg.talkerConfig.hiddenSize], dtype := dtypes.talker.val, device := device } :=
  TalkerModel.buildInputsFromText cfg.talkerConfig m.talker.model textIds

/-- Build talker input embeddings from instruct + text token IDs.
    Prepends codec BOS embedding. -/
def buildTalkerInputsFromInstructText {batch instructSeq textSeq : UInt64}
    (m : Qwen3TTSForConditionalGeneration (device := device) cfg dtypes)
    (instructIds : Tensor { shape := #[batch, instructSeq], dtype := .Int64, device := device })
    (textIds : Tensor { shape := #[batch, textSeq], dtype := .Int64, device := device })
    : Tensor { shape := #[batch, 1 + (instructSeq + textSeq), cfg.talkerConfig.hiddenSize], dtype := dtypes.talker.val, device := device } :=
  TalkerModel.buildInputsFromInstructText cfg.talkerConfig m.talker.model instructIds textIds

/-- Generate one codec frame (`numCodeGroups` tokens per sample) from talker inputs.
    Inputs are already-projected talker embeddings. -/
def generateFrame {batch seq : UInt64}
    (m : Qwen3TTSForConditionalGeneration (device := device) cfg dtypes)
    (talkerInputs : Tensor { shape := #[batch, seq, cfg.talkerConfig.hiddenSize], dtype := dtypes.talker.val, device := device })
    (attnMask : Option (Tensor { shape := #[batch, seq], dtype := .Int64, device := device }) := none)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    : IO (Tensor { shape := #[batch, cfg.talkerConfig.numCodeGroups], dtype := .Int64, device := device }) :=
  return (← TalkerForConditionalGeneration.generateFrame
    cfg.talkerConfig m.talker talkerInputs attnMask
    none true temperature topK topP).codes

/-- Generate multiple codec frames autoregressively from talker context.
    Returns `[batch, maxFrames, numCodeGroups]`. -/
def generateCodes {batch seq : UInt64}
    (m : Qwen3TTSForConditionalGeneration (device := device) cfg dtypes)
    (talkerInputs : Tensor { shape := #[batch, seq, cfg.talkerConfig.hiddenSize], dtype := dtypes.talker.val, device := device })
    (maxFrames : UInt64 := 256)
    (minNewTokens : UInt64 := 2)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    : IO (Tensor { shape := #[batch, maxFrames, cfg.talkerConfig.numCodeGroups], dtype := .Int64, device := device }) :=
  TalkerForConditionalGeneration.generateCodes
    cfg.talkerConfig m.talker talkerInputs maxFrames minNewTokens temperature topK topP

/-- Generate multiple codec frames with EOS-aware stopping metadata.
    `codes` is padded to `[batch, maxFrames, numCodeGroups]`.
    `lengths` stores the first EOS step per sample (or `maxFrames` if no EOS). -/
def generateCodesWithLengths {batch seq : UInt64}
    (m : Qwen3TTSForConditionalGeneration (device := device) cfg dtypes)
    (talkerInputs : Tensor { shape := #[batch, seq, cfg.talkerConfig.hiddenSize], dtype := dtypes.talker.val, device := device })
    (maxFrames : UInt64 := 256)
    (minNewTokens : UInt64 := 2)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    : IO (TalkerForConditionalGeneration.CodeGenerationOutput (device := device) batch maxFrames cfg.talkerConfig.numCodeGroups) :=
  TalkerForConditionalGeneration.generateCodesWithLengths
    cfg.talkerConfig m.talker talkerInputs maxFrames minNewTokens temperature topK topP

/-- End-to-end generation from text token IDs.
    Internally builds talker inputs and runs EOS-aware codec generation. -/
def generateFromText {batch textSeq : UInt64}
    (m : Qwen3TTSForConditionalGeneration (device := device) cfg dtypes)
    (textIds : Tensor { shape := #[batch, textSeq], dtype := .Int64, device := device })
    (maxFrames : UInt64 := 256)
    (minNewTokens : UInt64 := 2)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    : IO (TalkerForConditionalGeneration.CodeGenerationOutput (device := device) batch maxFrames cfg.talkerConfig.numCodeGroups) := do
  let talkerInputs := buildTalkerInputsFromText m textIds
  generateCodesWithLengths m talkerInputs maxFrames minNewTokens temperature topK topP

/-- End-to-end generation from instruct+text token IDs.
    Internally builds talker inputs and runs EOS-aware codec generation. -/
def generateFromInstructText {batch instructSeq textSeq : UInt64}
    (m : Qwen3TTSForConditionalGeneration (device := device) cfg dtypes)
    (instructIds : Tensor { shape := #[batch, instructSeq], dtype := .Int64, device := device })
    (textIds : Tensor { shape := #[batch, textSeq], dtype := .Int64, device := device })
    (maxFrames : UInt64 := 256)
    (minNewTokens : UInt64 := 2)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    : IO (TalkerForConditionalGeneration.CodeGenerationOutput (device := device) batch maxFrames cfg.talkerConfig.numCodeGroups) := do
  let talkerInputs := buildTalkerInputsFromInstructText m instructIds textIds
  generateCodesWithLengths m talkerInputs maxFrames minNewTokens temperature topK topP

end Qwen3TTSForConditionalGeneration

end torch.qwen3tts
