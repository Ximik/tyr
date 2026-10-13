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
  let input : T #[2, 3, 4] := reshape (data.fromFloatArray
    #[0.17, -0.41, 0.83, 1.27, 0.9, 0.5, -0.7, 1.1,
      -0.6, 1.3, 0.2, -0.9, 1.2, 0.4, -0.8, 0.6,
      0.3, -1.1, 0.7, 0.2, -0.5, 0.8, 1.4, -0.2]) #[2, 3, 4]
  let weight : T #[4] := reshape (data.fromFloatArray #[0.73, 1.19, -0.67, 1.53]) #[4]
  for bf16 in #[false, true] do
    let x := if bf16 then toBFloat16' input else input
    let w := if bf16 then toBFloat16' weight else weight
    let norm : RMSNorm 4 := { weight := w, eps := ⟨0.01⟩ }
    let normalized := nn.rmsNorm x 0.01
    let expected : T #[2, 3, 4] :=
      (if bf16 then toBFloat16' normalized else normalized) * w
    let actual := norm.forward3d x
    LeanTest.assertEqual actual.dtype x.dtype "RMSNorm preserves matching FP32/BF16 model dtype"
    LeanTest.assertEqual (nn.item (nn.maxAll (nn.abs (sub actual expected)))) 0.0
      "RMSNorm restores activation dtype before applying non-unit weights"
    for output in #[nn.eraseShape (norm.forward2d (reshape x #[6, 4])),
        nn.eraseShape (norm.forward4d (reshape x #[1, 2, 3, 4])),
        nn.eraseShape (norm.forward5d (reshape x #[1, 2, 3, 4]))] do
      LeanTest.assertEqual output.dtype x.dtype "All RMSNorm module layouts preserve model dtype"
      LeanTest.assertEqual (nn.item (nn.maxAll (nn.abs
        (sub (reshape output #[2, 3, 4]) expected)))) 0.0
        "RMSNorm layout wrappers agree"
    LeanTest.assertEqual (nn.rmsNormWeighted x w 0.01).dtype DType.Float32
      "Functional RMSNorm keeps its documented Float32 contract"

@[test]
def testQwen3TTSBFloat16IncrementalCache : IO Unit := autograd.no_grad do
  torch.manualSeed 7341
  let raw ← qwen.QwenLayer.init 8 2 1 4 16
  let base := { raw with
    self_attn := TensorStruct.scale raw.self_attn 0.25
    mlp := TensorStruct.scale raw.mlp 0.25 }
  let inputs : T #[1, 5, 8] := reshape
    (mul_scalar (nn.sin (toFloat' (torch.arange 0 40 1))) 0.5) #[1, 5, 8]
  let (cos, sin) := rotary.computeFreqsPure 5 4 10000
  for bf16 in #[false, true] do
    let layer := if bf16 then TensorStruct.map (fun t => t.checkedReplacement (RawTensor.toBFloat16 t.shape t.raw)) base else base
    let x := if bf16 then toBFloat16' inputs else inputs
    let full := layer.forward x cos sin
    LeanTest.assertEqual full.dtype x.dtype "Full Qwen layer retains model dtype"
    let fresh : qwen.QwenAttention.KVCache 1 1 4 := qwen.QwenAttention.initKVCache 5
    let mut cache := fresh
    -- Three prompt tokens followed by two decode tokens exercise the Talker cache path.
    let mut outputs : Array (T #[]) := #[]
    for pos in [:3] do
      let (out, next) := layer.forwardStep (data.slice x 1 pos.toUInt64 1)
        (data.slice cos 0 pos.toUInt64 1) (data.slice sin 0 pos.toUInt64 1) cache
      LeanTest.assertEqual out.dtype x.dtype "Prefill step retains model dtype"
      cache := next
      outputs := outputs.push (nn.eraseShape out)
    let prefixCache := cache
    let prefixK ← data.tensorToFloatArray' prefixCache.kStoreDyn
    let prefixV ← data.tensorToFloatArray' prefixCache.vStoreDyn
    for pos in [3:5] do
      let (out, next) := layer.forwardStep (data.slice x 1 pos.toUInt64 1)
        (data.slice cos 0 pos.toUInt64 1) (data.slice sin 0 pos.toUInt64 1) cache
      LeanTest.assertEqual out.dtype x.dtype "Decode step retains model dtype"
      cache := next
      outputs := outputs.push (nn.eraseShape out)
    LeanTest.assertEqual cache.seq 5 "All prompt and decode tokens enter the cache"
    LeanTest.assertEqual cache.kStoreDyn.dtype x.dtype "K cache follows projected key dtype"
    LeanTest.assertEqual cache.vStoreDyn.dtype x.dtype "V cache follows projected value dtype"
    LeanTest.assertEqual (← data.tensorToFloatArray' prefixCache.kStoreDyn) prefixK
      "Later decode leaves the functional K snapshot unchanged"
    LeanTest.assertEqual (← data.tensorToFloatArray' prefixCache.vStoreDyn) prefixV
      "Later decode leaves the functional V snapshot unchanged"
    LeanTest.assertEqual (nn.item (nn.sumAll fresh.kStoreDyn)) 0.0 "Fresh K snapshot remains empty"
    LeanTest.assertEqual (nn.item (nn.sumAll fresh.vStoreDyn)) 0.0 "Fresh V snapshot remains empty"
    let stepped : T #[1, 5, 8] := reshape (nn.cat_dyn outputs 1) #[1, 5, 8]
    let error := nn.item (nn.maxAll (nn.abs (sub (toFloat' stepped) (toFloat' full))))
    let tolerance := if bf16 then 0.0078125 else 0.00001
    LeanTest.assertTrue (Float.isFinite error && error <= tolerance)
      s!"Full and cached Qwen outputs agree at model precision: max error {error}, bound {tolerance}"

@[test]
def testQwen3TTSBFloat16Generation : IO Unit := do
  let cfg := tinyCfg.talkerConfig
  let original ← TalkerForConditionalGeneration.init cfg
  let talker := TensorStruct.map (fun t => t.checkedReplacement (RawTensor.toBFloat16 t.shape t.raw)) original
  let ids : T #[1, 3] := torch.full_int #[1, 3] 1
  let prompt := TalkerModel.buildInputsFromText cfg talker.model ids
  LeanTest.assertEqual prompt.dtype DType.BFloat16 "Text conditioning follows BF16 weights"
  -- Both main Talker and the residual-code predictor decode repeatedly. Omitting
  -- ttsPadEmbed exercises the default padding after each generated codec frame.
  let out ← TalkerForConditionalGeneration.generateCodesWithLengths cfg talker prompt
    3 3 1.0 1 1.0 1.0 1 1.0 1.0 0
  LeanTest.assertEqual out.codes.runtimeShape #[1, 3, cfg.numCodeGroups]
    "BF16 generation completes the requested frames and code groups"
  LeanTest.assertEqual out.lengths #[3] "EOS is suppressed for all requested frames"
  let codes ← data.tensorToUInt64Array' (nn.eraseShape out.codes)
  LeanTest.assertEqual codes.size (3 * cfg.numCodeGroups.toNat) "Every generated code materializes"
  for i in [:codes.size] do
    let upper := if i % cfg.numCodeGroups.toNat == 0 then cfg.vocabSize else cfg.codePredictorConfig.vocabSize
    LeanTest.assertTrue (codes[i]! < upper) "Generated codes stay within their vocabulary"

@[test]
def testQwen3TTSGenerateFromText : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3TTSForConditionalGeneration.init cfg

  let batch : UInt64 := 2
  let textSeq : UInt64 := 6
  let maxFrames : UInt64 := 5

  let textIds ← randint 0 cfg.talkerConfig.textVocabSize.toInt64 #[batch, textSeq]
  let out ← model.generateFromText textIds maxFrames

  LeanTest.assertEqual out.lengths.size batch.toNat "lengths should have one value per batch row"

  -- Verify padded output shape by materializing first codebook and flattening.
  let firstBook3 : T #[batch, maxFrames, 1] := data.slice out.codes 2 0 1
  let firstBook2 : T #[batch, maxFrames] := reshape firstBook3 #[batch, maxFrames]
  let flat : T #[batch * maxFrames] := reshape firstBook2 #[batch * maxFrames]
  let toks ← data.tensorToUInt64Array flat
  LeanTest.assertEqual toks.size (batch * maxFrames).toNat "flattened token count should match batch*frames"

  for len in out.lengths do
    LeanTest.assertTrue (len >= 1 && len <= maxFrames) "each generated length must be within [1, maxFrames]"

@[test]
def testQwen3TTSGenerateFromInstructText : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3TTSForConditionalGeneration.init cfg

  let batch : UInt64 := 1
  let instructSeq : UInt64 := 3
  let textSeq : UInt64 := 4
  let maxFrames : UInt64 := 4

  let instructIds ← randint 0 cfg.talkerConfig.textVocabSize.toInt64 #[batch, instructSeq]
  let textIds ← randint 0 cfg.talkerConfig.textVocabSize.toInt64 #[batch, textSeq]
  let out ← model.generateFromInstructText instructIds textIds maxFrames

  LeanTest.assertEqual out.lengths.size batch.toNat "lengths should have one value per batch row"

  let firstBook3 : T #[batch, maxFrames, 1] := data.slice out.codes 2 0 1
  let firstBook2 : T #[batch, maxFrames] := reshape firstBook3 #[batch, maxFrames]
  let flat : T #[batch * maxFrames] := reshape firstBook2 #[batch * maxFrames]
  let toks ← data.tensorToUInt64Array flat
  LeanTest.assertEqual toks.size (batch * maxFrames).toNat "flattened token count should match batch*frames"

  for len in out.lengths do
    LeanTest.assertTrue (len >= 1 && len <= maxFrames) "each generated length must be within [1, maxFrames]"

@[test]
def testQwen3TTSGenerateFromTextZeroFrames : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3TTSForConditionalGeneration.init cfg

  let batch : UInt64 := 2
  let textSeq : UInt64 := 5
  let textIds ← randint 0 cfg.talkerConfig.textVocabSize.toInt64 #[batch, textSeq]
  let out ← model.generateFromText textIds 0

  LeanTest.assertEqual out.lengths.size batch.toNat "lengths should have one value per batch row"
  for len in out.lengths do
    LeanTest.assertEqual len 0 "maxFrames=0 should produce zero generated length"

  LeanTest.assertEqual out.codes.runtimeShape #[batch, 0, cfg.talkerConfig.numCodeGroups]
    "maxFrames=0 should produce an empty frame dimension"

@[test]
def testQwen3TTSRejectsNonPositiveTemperature : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3TTSForConditionalGeneration.init cfg
  let textIds : T #[1, 4] ← randint 0 cfg.talkerConfig.textVocabSize.toInt64 #[1, 4]

  LeanTest.assertThrows
    (do
      let _ ← model.generateFromText textIds 2 1 0.0
      pure ())
    (some "temperature > 0")

@[test]
def testQwen3TTSSpeakerEmbedding : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3TTSForConditionalGeneration.init cfg
  let batch : UInt64 := 2
  let frames : UInt64 := 12
  let mel ← randn #[batch, frames, cfg.speakerEncoderConfig.melDim]
  let emb ← model.extractSpeakerEmbedding mel
  let s := nn.item (nn.sumAll emb)
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
  let original ← Qwen3TTSForConditionalGeneration.init cfg
  let some baseEncoder := original.speakerEncoder
    | throw <| IO.userError "Missing test speaker encoder"
  let features : T #[2, 12, cfg.speakerEncoderConfig.melDim] ←
    randn #[2, 12, cfg.speakerEncoderConfig.melDim]
  for bf16Weights in #[false, true] do
    let enc := if bf16Weights then TensorStruct.map (fun t => t.checkedReplacement (RawTensor.toBFloat16 t.shape t.raw)) baseEncoder else baseEncoder
    let model := { original with speakerEncoder := some enc }
    for bf16Features in #[false, true] do
      let mel := if bf16Features then toBFloat16' features else features
      let originalDType := mel.dtype
      let explicit := if bf16Weights then toBFloat16' mel else toFloat' mel
      let expected := enc.forward explicit
      let actual ← model.extractSpeakerEmbedding mel
      LeanTest.assertEqual actual.runtimeShape #[2, cfg.speakerEncoderConfig.encDim]
        "Speaker output shape follows the model configuration"
      LeanTest.assertEqual actual.dtype enc.tdnn0.weight.dtype
        "Speaker output uses the loaded weight dtype"
      LeanTest.assertTrue (actual.device == enc.tdnn0.weight.device)
        "Speaker output uses the loaded weight device"
      LeanTest.assertEqual mel.dtype originalDType "Feature alignment does not mutate caller dtype"
      let error := nn.item (nn.maxAll (nn.abs (sub (toFloat' actual) (toFloat' expected))))
      LeanTest.assertEqual error 0.0
        "Automatic feature alignment exactly matches an explicit cast for FP32/BF16 weights and inputs"

@[test]
def testQwen3TTSSpeakerEmbeddingUnavailableForNonBaseModel : IO Unit := do
  let cfg : Qwen3TTSConfig := {
    tinyCfg with
    ttsModelType := "custom_voice"
  }
  let model ← Qwen3TTSForConditionalGeneration.init cfg
  let mel : T #[1, 8, cfg.speakerEncoderConfig.melDim] ← randn #[1, 8, cfg.speakerEncoderConfig.melDim]

  LeanTest.assertThrows
    (do
      let _ ← model.extractSpeakerEmbedding mel
      pure ())
    (some "Speaker encoder is unavailable")

@[test]
def testQwen3TTSStreamingApi : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3TTSForConditionalGeneration.init cfg
  let textIds : T #[1, 5] ← randint 0 cfg.talkerConfig.textVocabSize.toInt64 #[1, 5]
  let talkerInputs : T #[1, 6, cfg.talkerConfig.hiddenSize] := model.buildTalkerInputsFromText textIds

  let callbacksSeenRef ← IO.mkRef (0 : Nat)
  let callbacks : Qwen3TTSForConditionalGeneration.StreamingCallbacks := {
    onCodeFrame := fun _ row => do
      LeanTest.assertEqual row.size cfg.talkerConfig.numCodeGroups.toNat "stream row width should equal numCodeGroups"
      callbacksSeenRef.modify (fun n => n + 1)
  }
  let opts : Qwen3TTSForConditionalGeneration.StreamingOptions cfg := {
    maxFrames := 4
    minNewTokens := 1
    emitEosFrame := true
  }
  let out ← model.streamFromTalkerInputs talkerInputs opts none callbacks
  LeanTest.assertEqual out.lengths.size 1 "stream lengths should have one value for batch=1"
  LeanTest.assertTrue (out.codeRows.size <= opts.maxFrames.toNat) "streamed rows should not exceed maxFrames"
  let callbacksSeen ← callbacksSeenRef.get
  LeanTest.assertEqual callbacksSeen out.codeRows.size "callback should fire once per emitted stream row"
