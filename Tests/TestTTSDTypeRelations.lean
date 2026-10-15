import Tyr.Model.Qwen3TTS.Model
import LeanTest

namespace Tests.TTSDTypeRelations
open torch torch.qwen3tts

-- Before exact shared indices, erasing this Float64 tensor allowed an invalid
-- codec embedding to be installed in an otherwise Float32 decoder.
example (cfg : TalkerConfig) (_model : TalkerModel (device := .CPU) cfg)
    (_foreign : Tensor ⟨#[cfg.vocabSize, cfg.hiddenSize], .Float64, .CPU⟩) : True := by
  fail_if_success
    have _ : TalkerModel (device := .CPU) cfg := { _model with codecEmbedding := _foreign }
  trivial


private def bf16 : ExactDType := ⟨.BFloat16, by decide⟩

example (cfg : TalkerConfig)
    (_model : TalkerForConditionalGeneration (device := .CPU) cfg bf16)
    (_head : Tensor ⟨#[cfg.vocabSize, cfg.hiddenSize], .Float32, .CPU⟩)
    (_predictor : TalkerCodePredictor (device := .CPU) cfg float32DType) : True := by
  fail_if_success
    have _ : TalkerForConditionalGeneration (device := .CPU) cfg bf16 := { _model with codecHead := _head }
  fail_if_success
    have _ : TalkerForConditionalGeneration (device := .CPU) cfg bf16 := { _model with codePredictor := _predictor }
  fail_if_success
    have _ : TalkerForConditionalGeneration (device := .CPU) cfg float32DType := _model
  trivial

example (_cache : qwen.QwenAttention.KVCache (device := .CPU) 1 1 4 .Float32)
    (_dtype : ExactDType) : True := by
  fail_if_success
    have _ : qwen.QwenAttention.KVCache (device := .CPU) 1 1 4 .BFloat16 := _cache
  fail_if_success
    have _ : ExactDType := ⟨.Unknown "dynamic", by decide⟩
  trivial

private def tinyConfig : Qwen3TTSConfig := {
  talkerConfig := {
    codePredictorConfig := {
      vocabSize := 16, hiddenSize := 8, intermediateSize := 16,
      numHiddenLayers := 1, numAttentionHeads := 2, numKeyValueHeads := 1, headDim := 4 }
    vocabSize := 32, hiddenSize := 8, intermediateSize := 16,
    numHiddenLayers := 1, numAttentionHeads := 2, numKeyValueHeads := 1, headDim := 4,
    numCodeGroups := 3, textHiddenSize := 8, textVocabSize := 32,
    codecPadId := 0, codecBosId := 3, codecEosTokenId := 7 }
  speakerEncoderConfig := {
    melDim := 4, encDim := 8, encChannels := #[4, 4, 4, 4, 12],
    encAttentionChannels := 2, encRes2NetScale := 2, encSeChannels := 2 }
  ttsModelType := "base" }

@[test] def testMixedSpeakerAndTalkerDTypeRelationships : IO Unit := do
  let cfg := tinyConfig
  let layout : DTypeLayout := { talker := bf16, speaker := float32DType }
  let model ← Qwen3TTSForConditionalGeneration.init cfg (dtypes := layout)
  let modelTalker := TensorStruct.detach model.talker
  let prompt : Tensor ⟨#[1, 3, cfg.talkerConfig.hiddenSize], .BFloat16, .CPU⟩ :=
    TalkerModel.buildInputsFromText cfg.talkerConfig modelTalker.model (Tensor.fullInt #[1, 2] 1)
  LeanTest.assertEqual prompt.raw.dtype .BFloat16
  let hidden : Tensor ⟨#[1, 3, cfg.talkerConfig.hiddenSize], .BFloat16, .CPU⟩ :=
    TalkerModel.forwardEmbeds cfg.talkerConfig modelTalker.model prompt
  LeanTest.assertEqual hidden.raw.dtype .BFloat16
  let projected : Tensor ⟨#[1, 3, cfg.talkerConfig.codePredictorConfig.hiddenSize], .BFloat16, .CPU⟩ :=
    TalkerCodePredictor.projectInputs modelTalker.codePredictor hidden
  LeanTest.assertEqual projected.raw.dtype .BFloat16
  let codes ← TalkerForConditionalGeneration.generateCodesWithLengths cfg.talkerConfig modelTalker prompt
    2 2 1.0 1 1.0 1.0 1 1.0 1.0 0
  LeanTest.assertEqual codes.codes.raw.dtype .Int64
  LeanTest.assertEqual codes.codes.runtimeShape #[1, 2, cfg.talkerConfig.numCodeGroups]
  LeanTest.assertEqual codes.lengths #[2]
  let some encoder := model.speakerEncoder
    | throw (IO.userError "Expected the independent speaker encoder")
  LeanTest.assertTrue (TensorStruct.fold (fun {σ} (t : Tensor σ) valid =>
    valid && t.raw.dtype == .Float32) true encoder)
    "Loading/initialization must not homogenize the Float32 speaker into the BF16 talker"
  let features ← Tensor.randn #[1, 8, cfg.speakerEncoderConfig.melDim]
  for inputDType in #[DType.Float32, .BFloat16] do
    let mel ← IO.ofExcept (features.castDType? inputDType)
    let embedding : Tensor ⟨#[1, cfg.speakerEncoderConfig.encDim], .Float32, .CPU⟩ ←
      model.extractSpeakerEmbedding mel
    LeanTest.assertEqual embedding.raw.dtype .Float32
    LeanTest.assertEqual mel.raw.dtype inputDType "Feature conversion preserves the caller tensor"
    let aligned ← IO.ofExcept (mel.castDType? .Float32)
    let expected := encoder.forward aligned
    LeanTest.assertEqual (← embedding.toFloatArray) (← expected.toFloatArray)
      "The explicit preprocessing cast is the only cross-role dtype conversion"

@[test] def testExactTalkerCacheStoresModelDType : IO Unit := do
  let cfg := tinyConfig.talkerConfig
  let model ← TalkerModel.init cfg (dtype := bf16)
  let model := TensorStruct.detach model
  let input := TalkerModel.embedCodec model (Tensor.fullInt #[1, 1] 1)
  let (cos, sin) := Tensor.rotaryFreqs 1 cfg.headDim cfg.ropeTheta
  let cache : qwen.QwenAttention.KVCache (device := .CPU) 1 cfg.numKeyValueHeads cfg.headDim .BFloat16 :=
    qwen.QwenAttention.initKVCache 3 (dtype := .BFloat16)
  let some layer := model.layers[0]? | throw (IO.userError "Expected one test layer")
  let (output, updated) := layer.forwardStep input cos sin cache
  LeanTest.assertEqual output.raw.dtype .BFloat16
  LeanTest.assertEqual updated.kStoreDyn.2.raw.dtype .BFloat16
  LeanTest.assertEqual updated.vStoreDyn.2.raw.dtype .BFloat16
  LeanTest.assertEqual updated.seq 1
  LeanTest.assertEqual cache.seq 0 "The original cache value remains unchanged"
  LeanTest.assertEqual (← cache.kStoreDyn.2.toFloatArray) (Array.replicate (3 * cfg.numKeyValueHeads * cfg.headDim).toNat 0.0)

end Tests.TTSDTypeRelations
