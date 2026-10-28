import Tests.Assertions
import Tyr
import Tyr.Model.Qwen3TTS
import LeanTest

open torch
open torch.qwen3tts

private def tinyCfg : Qwen3TTSConfig :=
  { talkerConfig := {
      codePredictorConfig := {
        vocabSize := 64
        hiddenSize := 32
        intermediateSize := 64
        numHiddenLayers := 1
        numAttentionHeads := 4
        numKeyValueHeads := 2
        headDim := 8
      }
      vocabSize := 128
      hiddenSize := 32
      intermediateSize := 64
      numHiddenLayers := 2
      numAttentionHeads := 4
      numKeyValueHeads := 2
      headDim := 8
      numCodeGroups := 4
      textHiddenSize := 32
      textVocabSize := 128
      codecPadId := 0
      codecBosId := 3
      codecEosTokenId := 7
      codecThinkId := 8
      codecNoThinkId := 9
      codecThinkBosId := 10
      codecThinkEosId := 11
      spkId := #[("speaker_a", 1), ("speaker_b", 2)]
      codecLanguageId := #[("en", 1), ("dialect_test", 2), ("fr", 3)]
    }
    speakerEncoderConfig := {
      melDim := 16
      encDim := 32
      sampleRate := 24000
    }
    ttsModelType := "base"
  }

@[test]
def testQwen3TTSInitAndCapabilities : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3TTSForConditionalGeneration.init cfg
  let langs := model.getSupportedLanguages
  let speakers := model.getSupportedSpeakers

  LeanTest.assertTrue (langs.contains "auto") "supported languages should include auto"
  LeanTest.assertTrue (langs.contains "en") "supported languages should include configured language tags"
  LeanTest.assertTrue (langs.contains "fr") "supported languages should include configured language tags"
  LeanTest.assertFalse (langs.contains "dialect_test") "dialect-tagged language entries should be filtered"

  LeanTest.assertEqual speakers.size 2 "supported speakers should include configured speaker IDs"
  LeanTest.assertTrue (speakers.contains "speaker_a") "supported speakers should include speaker_a"
  LeanTest.assertTrue (speakers.contains "speaker_b") "supported speakers should include speaker_b"

@[test]
def testQwen3TTSRMSNormModelDType : IO Unit := do
  let input : Tensor { shape := #[2, 3, 4], dtype := .Float32, device := .CPU } := Tensor.reshape (Tensor.fromFloatArray
    #[0.17, -0.41, 0.83, 1.27, 0.9, 0.5, -0.7, 1.1,
      -0.6, 1.3, 0.2, -0.9, 1.2, 0.4, -0.8, 0.6,
      0.3, -1.1, 0.7, 0.2, -0.5, 0.8, 1.4, -0.2]) #[2, 3, 4]
  let weight : Tensor { shape := #[4], dtype := .Float32, device := .CPU } := Tensor.reshape (Tensor.fromFloatArray #[0.73, 1.19, -0.67, 1.53]) #[4]
  for bf16 in #[false, true] do
    let selected : RealFloatingDType :=
      if bf16 then ⟨.BFloat16, by decide⟩ else ⟨.Float32, by decide⟩
    let dtype := selected.val
    let x := input.castDType dtype
    let w := weight.castDType dtype
    let norm : RMSNorm 4 dtype := { weight := w, eps := ⟨0.01⟩ }
    let normalized := Tensor.rmsNormWeighted x.toFloat32 (Tensor.ones #[4]) 0.01
    let expected : Tensor { shape := #[2, 3, 4], dtype, device := .CPU } :=
      Tensor.mul
        (normalized.castDType dtype)
        (w.expand #[2, 3, 4])
    let actual := norm.forward3d x
    LeanTest.assertEqual actual.actualSpec.dtype x.actualSpec.dtype "RMSNorm preserves matching FP32/BF16 model dtype"
    LeanTest.assertEqual (Tensor.item (Tensor.maxAllChecked (Tensor.abs (Tensor.sub actual.toFloat32 expected.toFloat32)))) 0.0
      "RMSNorm restores activation dtype before applying non-unit weights"
    for output in #[(norm.forward2d (Tensor.reshape x #[6, 4])).reshape #[2, 3, 4],
        (norm.forward4d (Tensor.reshape x #[1, 2, 3, 4])).reshape #[2, 3, 4],
        (norm.forward5d (Tensor.reshape x #[1, 2, 3, 4])).reshape #[2, 3, 4]] do
      LeanTest.assertEqual output.actualSpec.dtype x.actualSpec.dtype "All RMSNorm module layouts preserve model dtype"
      LeanTest.assertEqual (Tensor.item (Tensor.maxAllChecked (Tensor.abs
        (Tensor.sub (Tensor.reshape output #[2, 3, 4]).toFloat32 expected.toFloat32)))) 0.0
        "RMSNorm layout wrappers agree"
    LeanTest.assertEqual (if bf16 then (Tensor.rmsNormWeighted x.toBFloat16 w.toBFloat16 0.01).actualSpec.dtype
      else (Tensor.rmsNormWeighted x.toFloat32 w.toFloat32 0.01).actualSpec.dtype) DType.Float32
      "Functional RMSNorm keeps its documented Float32 contract"

@[test]
def testQwen3TTSBFloat16IncrementalCache : IO Unit := do
  torch.manualSeed 7341
  let inputs : Tensor { shape := #[1, 5, 8], dtype := .Float32, device := .CPU } := Tensor.reshape
    (Tensor.mulScalar (Tensor.sin (Tensor.toFloat32 (Tensor.arange 40))) 0.5) #[1, 5, 8]
  let (cos, sin) := Tensor.rotaryFreqs 5 4 10000
  for bf16 in #[false, true] do
    let selected : RealFloatingDType :=
      if bf16 then ⟨.BFloat16, by decide⟩ else ⟨.Float32, by decide⟩
    let dtype := selected.val
    let support ← dtype.acquireConstantSupport .CPU
    let _ := support.down
    let raw ← qwen.QwenLayer.init 8 2 1 4 16 (dtype := dtype)
    let raw := TensorStruct.detach raw
    let layer : qwen.QwenLayer (device := .CPU) 8 2 1 4 16 dtype := { raw with
      self_attn := TensorStruct.scale raw.self_attn 0.25
      mlp := TensorStruct.scale raw.mlp 0.25 }
    let x := inputs.castDType dtype
    let full := layer.forward x cos sin
    LeanTest.assertEqual full.actualSpec.dtype x.actualSpec.dtype "Full Qwen layer retains model dtype"
    let fresh : qwen.QwenAttention.KVCache 1 1 4 dtype := qwen.QwenAttention.initKVCache 5 (dtype := dtype)
    let mut cache := fresh
    -- Three prompt tokens followed by two decode tokens exercise the Talker cache path.
    let mut outputs : Array (Sigma (fun shape => Tensor { shape := shape, dtype, device := .CPU })) := #[]
    for pos in [:3] do
      let (out, next) := layer.forwardStep (Tensor.sliceChecked x 1 pos 1 #[1, 1, 8])
        (Tensor.sliceChecked cos 0 pos 1 #[1, 2]) (Tensor.sliceChecked sin 0 pos 1 #[1, 2]) cache
      LeanTest.assertEqual out.actualSpec.dtype x.actualSpec.dtype "Prefill step retains model dtype"
      cache := next
      outputs := outputs.push out.packShape
    let prefixCache := cache
    let prefixK ← Tensor.toFloatArray prefixCache.kStoreDyn.2
    let prefixV ← Tensor.toFloatArray prefixCache.vStoreDyn.2
    for pos in [3:5] do
      let (out, next) := layer.forwardStep (Tensor.sliceChecked x 1 pos 1 #[1, 1, 8])
        (Tensor.sliceChecked cos 0 pos 1 #[1, 2]) (Tensor.sliceChecked sin 0 pos 1 #[1, 2]) cache
      LeanTest.assertEqual out.actualSpec.dtype x.actualSpec.dtype "Decode step retains model dtype"
      cache := next
      outputs := outputs.push out.packShape
    LeanTest.assertEqual cache.seq 5 "All prompt and decode tokens enter the cache"
    LeanTest.assertEqual cache.kStoreDyn.2.actualSpec.dtype x.actualSpec.dtype "K cache follows projected key dtype"
    LeanTest.assertEqual cache.vStoreDyn.2.actualSpec.dtype x.actualSpec.dtype "V cache follows projected value dtype"
    LeanTest.assertEqual (← Tensor.toFloatArray prefixCache.kStoreDyn.2) prefixK
      "Later decode leaves the functional K snapshot unchanged"
    LeanTest.assertEqual (← Tensor.toFloatArray prefixCache.vStoreDyn.2) prefixV
      "Later decode leaves the functional V snapshot unchanged"
    LeanTest.assertEqual (Tensor.item (Tensor.sumAll fresh.kStoreDyn.2)) 0.0 "Fresh K snapshot remains empty"
    LeanTest.assertEqual (Tensor.item (Tensor.sumAll fresh.vStoreDyn.2)) 0.0 "Fresh V snapshot remains empty"
    let stepped : Tensor { shape := #[1, 5, 8], dtype, device := .CPU } := Tensor.catManyChecked outputs 1 #[1, 5, 8]
    let error := Tensor.item (Tensor.maxAll (Tensor.abs (Tensor.sub (Tensor.toFloat32 stepped) (Tensor.toFloat32 full))))
    let tolerance := if bf16 then 0.0078125 else 0.00001
    LeanTest.assertTrue (Float.isFinite error && error <= tolerance)
      s!"Full and cached Qwen outputs agree at model precision: max error {error}, bound {tolerance}"

@[test]
def testQwen3TTSBFloat16Generation : IO Unit := do
  let cfg := tinyCfg.talkerConfig
  let talker ← TalkerForConditionalGeneration.init cfg (dtype := ⟨.BFloat16, by decide⟩)
  let ids : Tensor { shape := #[1, 3], dtype := .Int64, device := .CPU } := Tensor.fullInt #[1, 3] 1
  let prompt := TalkerModel.buildInputsFromText cfg talker.model ids
  LeanTest.assertEqual prompt.actualSpec.dtype DType.BFloat16 "Text conditioning follows BF16 weights"
  -- Both main Talker and the residual-code predictor decode repeatedly. Omitting
  -- ttsPadEmbed exercises the default padding after each generated codec frame.
  let out ← TalkerForConditionalGeneration.generateCodesWithLengths cfg talker prompt
    3 3 1.0 1 1.0 1.0 1 1.0 1.0 0
  LeanTest.assertEqual out.codes.runtimeShape #[1, 3, cfg.numCodeGroups]
    "BF16 generation completes the requested frames and code groups"
  LeanTest.assertEqual out.lengths #[3] "EOS is suppressed for all requested frames"
  let codes ← Tensor.toUInt64Array out.codes
  LeanTest.assertEqual codes.size (3 * cfg.numCodeGroups) "Every generated code materializes"
  for i in [:codes.size] do
    let upper := if i % cfg.numCodeGroups == 0 then cfg.vocabSize else cfg.codePredictorConfig.vocabSize
    LeanTest.assertTrue (codes[i]!.toNat < upper) "Generated codes stay within their vocabulary"

@[test]
def testQwen3TTSGenerateFromText : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3TTSForConditionalGeneration.init cfg

  let batch : Nat := 2
  let textSeq : Nat := 6
  let maxFrames : Nat := 5

  let textIds ← Tensor.randint 0 cfg.talkerConfig.textVocabSize.toInt64 #[batch, textSeq]
  let out ← model.generateFromText textIds maxFrames

  LeanTest.assertEqual out.lengths.size batch "lengths should have one value per batch row"

  -- Verify padded output shape by materializing first codebook and flattening.
  let firstBook3 : Tensor { shape := #[batch, maxFrames, 1], dtype := .Int64, device := .CPU } := Tensor.sliceChecked out.codes 2 0 1 #[batch, maxFrames, 1]
  let firstBook2 : Tensor { shape := #[batch, maxFrames], dtype := .Int64, device := .CPU } := Tensor.reshape firstBook3 #[batch, maxFrames]
  let flat : Tensor { shape := #[batch * maxFrames], dtype := .Int64, device := .CPU } := Tensor.reshape firstBook2 #[batch * maxFrames]
  let toks ← Tensor.toUInt64Array flat
  LeanTest.assertEqual toks.size (batch * maxFrames) "flattened token count should match batch*frames"

  for len in out.lengths do
    LeanTest.assertTrue (len >= 1 && len <= maxFrames) "each generated length must be within [1, maxFrames]"

@[test]
def testQwen3TTSGenerateFromInstructText : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3TTSForConditionalGeneration.init cfg

  let batch : Nat := 1
  let instructSeq : Nat := 3
  let textSeq : Nat := 4
  let maxFrames : Nat := 4

  let instructIds ← Tensor.randint 0 cfg.talkerConfig.textVocabSize.toInt64 #[batch, instructSeq]
  let textIds ← Tensor.randint 0 cfg.talkerConfig.textVocabSize.toInt64 #[batch, textSeq]
  let out ← model.generateFromInstructText instructIds textIds maxFrames

  LeanTest.assertEqual out.lengths.size batch "lengths should have one value per batch row"

  let firstBook3 : Tensor { shape := #[batch, maxFrames, 1], dtype := .Int64, device := .CPU } := Tensor.sliceChecked out.codes 2 0 1 #[batch, maxFrames, 1]
  let firstBook2 : Tensor { shape := #[batch, maxFrames], dtype := .Int64, device := .CPU } := Tensor.reshape firstBook3 #[batch, maxFrames]
  let flat : Tensor { shape := #[batch * maxFrames], dtype := .Int64, device := .CPU } := Tensor.reshape firstBook2 #[batch * maxFrames]
  let toks ← Tensor.toUInt64Array flat
  LeanTest.assertEqual toks.size (batch * maxFrames) "flattened token count should match batch*frames"

  for len in out.lengths do
    LeanTest.assertTrue (len >= 1 && len <= maxFrames) "each generated length must be within [1, maxFrames]"

@[test]
def testQwen3TTSGenerateFromTextZeroFrames : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3TTSForConditionalGeneration.init cfg

  let batch : Nat := 2
  let textSeq : Nat := 5
  let textIds ← Tensor.randint 0 cfg.talkerConfig.textVocabSize.toInt64 #[batch, textSeq]
  let out ← model.generateFromText textIds 0

  LeanTest.assertEqual out.lengths.size batch "lengths should have one value per batch row"
  for len in out.lengths do
    LeanTest.assertEqual len 0 "maxFrames=0 should produce zero generated length"

  LeanTest.assertEqual out.codes.runtimeShape #[batch, 0, cfg.talkerConfig.numCodeGroups]
    "maxFrames=0 should produce an empty frame dimension"

@[test]
def testQwen3TTSRejectsNonPositiveTemperature : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3TTSForConditionalGeneration.init cfg
  let textIds : Tensor { shape := #[1, 4], dtype := .Int64, device := .CPU } ← Tensor.randint 0 cfg.talkerConfig.textVocabSize.toInt64 #[1, 4]

  Tests.assertThrows
    (do
      let _ ← model.generateFromText textIds 2 1 0.0
      pure ())
    (some "temperature > 0")

@[test]
def testQwen3TTSSpeakerEmbedding : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3TTSForConditionalGeneration.init cfg
  let batch : Nat := 2
  let frames : Nat := 12
  let mel ← Tensor.randn #[batch, frames, cfg.speakerEncoderConfig.melDim]
  let emb ← model.extractSpeakerEmbedding mel
  let s := Tensor.item (Tensor.sumAll emb)
  LeanTest.assertTrue (Float.isFinite s) "speaker embedding sum should be finite"

@[test]
def testQwen3TTSSpeakerEmbeddingAlignsFeatureDType : IO Unit := do
  -- Small ECAPA blocks exercise the complete speaker path with real convolutions.
  let cfg : Qwen3TTSConfig := { tinyCfg with speakerEncoderConfig := {
    tinyCfg.speakerEncoderConfig with
    encChannels := #[8, 8, 8, 8, 24]
    encAttentionChannels := 4
    encRes2NetScale := 2
    encSeChannels := 4 } }
  let features : Tensor { shape := #[2, 12, cfg.speakerEncoderConfig.melDim], dtype := .Float32, device := .CPU } ←
    Tensor.randn #[2, 12, cfg.speakerEncoderConfig.melDim]
  for bf16Weights in #[false, true] do
    let speaker : RealFloatingDType := if bf16Weights then ⟨.BFloat16, by decide⟩ else float32DType
    let model ← Qwen3TTSForConditionalGeneration.init cfg (dtypes := { speaker })
    let some enc := model.speakerEncoder
      | throw <| IO.userError "Missing test speaker encoder"
    for bf16Features in #[false, true] do
      let mel := features.castDType (if bf16Features then .BFloat16 else .Float32)
      let originalDType := mel.actualSpec.dtype
      let explicit := mel.castDType speaker.val
      let expected := enc.forward explicit
      let actual ← model.extractSpeakerEmbedding mel
      LeanTest.assertEqual actual.runtimeShape #[2, cfg.speakerEncoderConfig.encDim]
        "Speaker output shape follows the model configuration"
      LeanTest.assertEqual actual.actualSpec.dtype enc.tdnn0.weight.actualSpec.dtype
        "Speaker output uses the loaded weight dtype"
      LeanTest.assertTrue (actual.device == enc.tdnn0.weight.device)
        "Speaker output uses the loaded weight device"
      LeanTest.assertEqual mel.actualSpec.dtype originalDType "Feature alignment does not mutate caller dtype"
      let error := Tensor.item (Tensor.maxAll (Tensor.abs (Tensor.sub (Tensor.toFloat32 actual) (Tensor.toFloat32 expected))))
      LeanTest.assertEqual error 0.0
        "Automatic feature alignment exactly matches an explicit cast for FP32/BF16 weights and inputs"

@[test]
def testQwen3TTSSpeakerEmbeddingUnavailableForNonBaseModel : IO Unit := do
  let cfg : Qwen3TTSConfig := {
    tinyCfg with
    ttsModelType := "custom_voice"
  }
  let model ← Qwen3TTSForConditionalGeneration.init cfg
  let mel : Tensor { shape := #[1, 8, cfg.speakerEncoderConfig.melDim], dtype := .Float32, device := .CPU } ← Tensor.randn #[1, 8, cfg.speakerEncoderConfig.melDim]

  Tests.assertThrows
    (do
      let _ ← model.extractSpeakerEmbedding mel
      pure ())
    (some "Speaker encoder is unavailable")

@[test]
def testQwen3TTSStreamingApi : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3TTSForConditionalGeneration.init cfg
  let textIds : Tensor { shape := #[1, 5], dtype := .Int64, device := .CPU } ← Tensor.randint 0 cfg.talkerConfig.textVocabSize.toInt64 #[1, 5]
  let talkerInputs : Tensor { shape := #[1, 6, cfg.talkerConfig.hiddenSize], dtype := .Float32, device := .CPU } := model.buildTalkerInputsFromText textIds

  let callbacksSeenRef ← IO.mkRef (0 : Nat)
  let callbacks : Qwen3TTSForConditionalGeneration.StreamingCallbacks := {
    onCodeFrame := fun _ row => do
      LeanTest.assertEqual row.size cfg.talkerConfig.numCodeGroups "stream row width should equal numCodeGroups"
      callbacksSeenRef.modify (fun n => n + 1)
  }
  let opts : Qwen3TTSForConditionalGeneration.StreamingOptions cfg := {
    maxFrames := 4
    minNewTokens := 1
    emitEosFrame := true
  }
  let out ← model.streamFromTalkerInputs talkerInputs opts none callbacks
  LeanTest.assertEqual out.lengths.size 1 "stream lengths should have one value for batch=1"
  LeanTest.assertTrue (out.codeRows.size <= opts.maxFrames) "streamed rows should not exceed maxFrames"
  let callbacksSeen ← callbacksSeenRef.get
  LeanTest.assertEqual callbacksSeen out.codeRows.size "callback should fire once per emitted stream row"

@[test] def testQwen3TTSEmptyGenerationKeepsIntegerCodes : IO Unit := do
  let cfg := tinyCfg.talkerConfig
  let model ← TalkerForConditionalGeneration.init cfg
  let prompt := TalkerModel.buildInputsFromText cfg model.model (Tensor.fullInt #[1, 2] 1)
  let empty ← TalkerForConditionalGeneration.generateCodesWithLengths cfg model prompt 0
  LeanTest.assertEqual empty.codes.runtimeShape #[1, 0, cfg.numCodeGroups]
  LeanTest.assertEqual empty.codes.actualSpec.dtype DType.Int64
  LeanTest.assertEqual empty.lengths #[0]
  let noGroups := { cfg with numCodeGroups := 0 }
  let model ← TalkerForConditionalGeneration.init noGroups
  let prompt := TalkerModel.buildInputsFromText noGroups model.model (Tensor.fullInt #[1, 2] 1)
  let frame ← TalkerForConditionalGeneration.generateFrame noGroups model prompt
  LeanTest.assertEqual frame.codes.runtimeShape #[1, 0]
  LeanTest.assertEqual frame.codes.actualSpec.dtype DType.Int64
