/-
  Tyr/Model/Qwen3ASR/Processor.lean

  Lean-native port of the key algorithmic pieces from
  `processing_qwen3_asr.py` (without Python/HF dependencies):
  - multimodal special-token replacement
  - feature-length conversion helpers
  - chunk-index utilities
-/
import Tyr.Model.Qwen3ASR.Config
import Tyr.Tokenizer.Qwen3

namespace torch.qwen3asr

/-- Default processor kwargs mirrored from reference processor defaults. -/
structure ProcessorKwargs where
  textPadding : Bool := false
  textPaddingSide : String := "left"
  audioSamplingRate : Nat := 16000
  audioPadding : Bool := true
  audioReturnAttentionMask : Bool := true
  deriving Repr, Inhabited

/-- Lightweight Lean processor config/state. -/
structure Qwen3ASRProcessor where
  audioToken : String := "<|audio_pad|>"
  audioBosToken : String := "<|audio_start|>"
  audioEosToken : String := "<|audio_end|>"
  imStartToken : String := "<|im_start|>"
  imEndToken : String := "<|im_end|>"
  kwargs : ProcessorKwargs := {}
  deriving Repr, Inhabited

namespace Qwen3ASRProcessor

private def repeatToken (tok : String) (n : Nat) : String :=
  Id.run do
    let mut out := ""
    for _ in [:n] do
      out := out ++ tok
    out

/-- Port of processor-side helper.
    Converts raw feature lengths to audio-token lengths after conv stack. -/
def featExtractOutputLengths (inputLens : Array Nat) : Array Nat :=
  AudioEncoderConfig.featExtractOutputLengths inputLens

private def replaceInSample
    (audioToken : String)
    (sample : String)
    (audioLengths : Array Nat)
    (startIdx : Nat)
    : Except String (String × Nat) := do
  let parts := (sample.splitOn audioToken).toArray
  if parts.size ≤ 1 then
    return (sample, startIdx)

  let mut idx := startIdx
  let mut out := parts[0]!
  for i in [1:parts.size] do
    if idx < audioLengths.size then
      let aLen := audioLengths[idx]!
      out := out ++ repeatToken audioToken aLen ++ parts[i]!
      idx := idx + 1
    else
      throw s!"Not enough audio lengths while replacing `{audioToken}` placeholders"
  return (out, idx)

/-- Port of `replace_multimodal_special_tokens` for audio placeholders.
    Each `audioToken` occurrence is replaced by repeating `audioToken`
    `audio_length` times for the corresponding audio in stream order. -/
def replaceMultimodalSpecialTokens
    (p : Qwen3ASRProcessor)
    (text : Array String)
    (audioLengths : Array Nat)
    : Except String (Array String) := do
  let mut out : Array String := #[]
  let mut idx : Nat := 0
  for sample in text do
    let (sample', idx') ← replaceInSample p.audioToken sample audioLengths idx
    out := out.push sample'
    idx := idx'
  return out

/-- Encode one prompt while expanding `audioToken` placeholders directly to
    exact `audioTokenId` runs.
    This avoids tokenizer greedy-special matching from collapsing boundary
    placeholders and desynchronizing audio/text lengths. -/
def encodeWithExpandedAudioTokenIds
    (p : Qwen3ASRProcessor)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (text : String)
    (audioLengths : Array Nat)
    (audioTokenId : UInt32)
    : Except String (Array UInt32) := do
  let parts := (text.splitOn p.audioToken).toArray
  if parts.size ≤ 1 then
    return tokenizer.qwen3.encodeText tok text

  let needed := parts.size - 1
  if audioLengths.size < needed then
    throw s!"Not enough audio lengths while encoding `{p.audioToken}` placeholders"

  let mut out : Array UInt32 := #[]
  let mut idx : Nat := 0
  for i in [:parts.size] do
    out := out ++ tokenizer.qwen3.encodeText tok parts[i]!
    if i + 1 < parts.size then
      let aLen := audioLengths[idx]!
      for _ in [:aLen] do
        out := out.push audioTokenId
      idx := idx + 1
  return out

/-- Port of processor chunking helper.
    Splits monotonically increasing token indices into value-range chunks. -/
def getChunkedIndex (tokenIndices : Array Nat) (tokensPerChunk : Nat) : Array (Nat × Nat) :=
  if tokensPerChunk == 0 then
    #[(0, tokenIndices.size)]
  else
    Id.run do
      let mut out : Array (Nat × Nat) := #[]
      let mut i : Nat := 0
      let mut startIdx : Nat := 0
      let mut currentChunk : Nat := 1
      while i < tokenIndices.size do
        if tokenIndices[i]! >= currentChunk * tokensPerChunk then
          out := out.push (startIdx, i)
          startIdx := i
          currentChunk := currentChunk + 1
        i := i + 1
      out.push (startIdx, tokenIndices.size)

/-- Equivalent of processor `model_input_names` merge.
    We expose the canonical multimodal names used by the Lean port. -/
def modelInputNames (_p : Qwen3ASRProcessor) : Array String :=
  #["input_ids", "attention_mask", "input_features", "feature_attention_mask"]

private def mkRoleBlock (p : Qwen3ASRProcessor) (role content : String) : String :=
  p.imStartToken ++ role ++ "\n" ++ content ++ p.imEndToken ++ "\n"

/-- Processor-side chat-template rendering used by ASR prompt construction.
    This keeps prompt assembly aligned with processor role/marker semantics. -/
def applyChatTemplate
    (p : Qwen3ASRProcessor)
    (systemContent : String)
    (userContent : String)
    (addGenerationPrompt : Bool := true)
    : String :=
  let base :=
    mkRoleBlock p "system" systemContent ++
    mkRoleBlock p "user" userContent
  if addGenerationPrompt then
    base ++ p.imStartToken ++ "assistant\n"
  else
    base

/-- Build user audio placeholder span in processor token space. -/
def audioUserPlaceholder (p : Qwen3ASRProcessor) : String :=
  p.audioBosToken ++ p.audioToken ++ p.audioEosToken

/-- Build ASR text prompt from processor chat-template semantics. -/
def buildAsrPrompt
    (p : Qwen3ASRProcessor)
    (context : String := "")
    (forceLanguage : Option String := none)
    : String :=
  let base := p.applyChatTemplate context p.audioUserPlaceholder true
  match forceLanguage with
  | some l => base ++ s!"language {l}<asr_text>"
  | none => base

end Qwen3ASRProcessor

end torch.qwen3asr
