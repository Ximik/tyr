/-  Tyr/Model/Qwen3TTS/Streaming.lean

  Reusable Lean-native Qwen3-TTS streaming interface:
  - stream codec-frame generation from prepared talker inputs
  - optional in-process speech-tokenizer streaming decode
  - callback hooks for per-frame codes and per-chunk audio
-/
import Tyr.Model.Qwen3TTS.Model
import Tyr.Model.Qwen3TTS.SpeechTokenizer

namespace torch.qwen3tts

variable {device : Device} {dtypes : DTypeLayout}

namespace Qwen3TTSForConditionalGeneration

/-- Runtime options for streaming generation from prepared talker inputs. -/
structure StreamingOptions (cfg : Qwen3TTSConfig) (dtype : RealFloatingDType := float32DType) where
  maxFrames : Nat := 256
  minNewTokens : Nat := 2
  temperature : Float := 0.9
  topK : Nat := 50
  topP : Float := 1.0
  subtalkerTemperature : Float := 0.9
  subtalkerTopK : Nat := 50
  subtalkerTopP : Float := 1.0
  repetitionPenalty : Float := 1.05
  suppressTail : Nat := 1024
  trailingTextHidden : Option (Sigma fun trailingSeq => Tensor { shape := #[1, trailingSeq, cfg.talkerConfig.hiddenSize], dtype := dtype.val, device := device }) := none
  ttsPadEmbed : Option (Tensor { shape := #[1, 1, cfg.talkerConfig.hiddenSize], dtype := dtype.val, device := device }) := none
  subtalkerTemperaturesByGroup : Option (Array Float) := none
  subtalkerTopKsByGroup : Option (Array Nat) := none
  subtalkerTopPsByGroup : Option (Array Float) := none
  decodeChunkSize : Nat := 1
  decodeLeftContext : Nat := 25
  emitEosFrame : Bool := false

/-- Streaming callbacks for generated codec rows and decoded audio chunks. -/
structure StreamingCallbacks where
  onCodeFrame : Nat → Array UInt64 → IO Unit := fun _ _ => pure ()
  onAudioChunk : (Sigma fun samples => SomeDTypeTensor #[1, 1, samples] device) → IO Unit := fun _ => pure ()

/-- Final metadata for one streaming run. -/
structure StreamingResult where
  codeRows : Array (Array UInt64)
  lengths : Array Nat
  deriving Repr, Inhabited

/-- Stream from prepared talker inputs (`[1, seq, hidden]`) with optional
    in-process streaming decode.

    When `speechDecoder?` is provided, code rows are decoded incrementally and
    emitted through `callbacks.onAudioChunk` if the talker uses 16 code groups
    (upstream 12Hz tokenizer). -/
def streamFromTalkerInputs {seq : Nat}
    (m : Qwen3TTSForConditionalGeneration (device := device) cfg dtypes)
    (talkerInputs : Tensor { shape := #[1, seq, cfg.talkerConfig.hiddenSize], dtype := dtypes.talker.val, device := device })
    (opts : StreamingOptions (device := device) cfg dtypes.talker := {})
    (speechDecoder? : Option (Σ codecDTypes : SpeechTokenizerDTypes, SpeechTokenizer12HzDecoder (device := device) codecDTypes) := none)
    (callbacks : (StreamingCallbacks (device := device)) := {})
    : IO StreamingResult := do
  let canStreamDecode : Bool :=
    speechDecoder?.isSome && cfg.talkerConfig.numCodeGroups == 16

  let codeRowsRef ← IO.mkRef (#[] : Array (Array UInt64))
  let decodeStateRef? : Option (IO.Ref (SpeechTokenizer12HzDecoder.DecodeStreamState (device := device) 1)) ←
    if canStreamDecode then
      match speechDecoder? with
      | some ⟨_, dec⟩ =>
        let st : SpeechTokenizer12HzDecoder.DecodeStreamState (device := device) 1 :=
          SpeechTokenizer12HzDecoder.initDecodeStreamState
            opts.decodeChunkSize opts.decodeLeftContext dec.preConvWeight.device
        let stRef ← IO.mkRef st
        pure (some stRef)
      | none =>
        pure none
    else
      pure none

  let onFrame : Nat → Tensor { shape := #[1, cfg.talkerConfig.numCodeGroups], dtype := .Int64, device := device } → IO Unit := fun step frame => do
    let frameRow : Tensor { shape := #[cfg.talkerConfig.numCodeGroups], dtype := .Int64, device := device } := Tensor.reshape frame #[cfg.talkerConfig.numCodeGroups]
    let rowVals ← Tensor.toUInt64Array frameRow
    let firstTok := rowVals.getD 0 cfg.talkerConfig.codecEosTokenId
    let keepRow := opts.emitEosFrame || firstTok != cfg.talkerConfig.codecEosTokenId
    if keepRow then
      codeRowsRef.modify (fun rows => rows.push rowVals)
      callbacks.onCodeFrame step rowVals
      match speechDecoder?, decodeStateRef? with
      | some ⟨codecDTypes, dec⟩, some stRef =>
          let frame16a : Tensor { shape := #[1, 16, 1], dtype := .Int64, device := device } := Tensor.reshapeChecked frame #[1, 16, 1]
          let frame16 : Tensor { shape := #[1, 16, 1], dtype := .Int64, device := device } :=
            frame16a.to dec.preConvWeight.device
          let st ← stRef.get
          let (st', chunks) := dec.pushDecodeStream st frame16
          stRef.set st'
          for chunk in chunks do
            callbacks.onAudioChunk ⟨chunk.1, ⟨codecDTypes.compute.val, chunk.2⟩⟩
      | _, _ =>
          pure ()

  let lengths ← TalkerForConditionalGeneration.streamCodes
    cfg.talkerConfig m.talker talkerInputs onFrame
    opts.maxFrames opts.minNewTokens
    opts.temperature opts.topK opts.topP
    opts.subtalkerTemperature opts.subtalkerTopK opts.subtalkerTopP
    opts.repetitionPenalty opts.suppressTail
    opts.trailingTextHidden opts.ttsPadEmbed
    opts.subtalkerTemperaturesByGroup opts.subtalkerTopKsByGroup opts.subtalkerTopPsByGroup

  match speechDecoder?, decodeStateRef? with
  | some ⟨codecDTypes, dec⟩, some stRef =>
      let st ← stRef.get
      let (_stFinal, chunks) := dec.flushDecodeStream st
      for chunk in chunks do
        callbacks.onAudioChunk ⟨chunk.1, ⟨codecDTypes.compute.val, chunk.2⟩⟩
  | _, _ =>
      pure ()

  let codeRows ← codeRowsRef.get
  pure { codeRows, lengths }

end Qwen3TTSForConditionalGeneration

end torch.qwen3tts
