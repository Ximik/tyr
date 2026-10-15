import Tyr.Model.Qwen3TTS
import LeanTest
import Tests.Assertions

namespace Tests.TTSTypedLoading
open torch torch.qwen3tts Lean

private def cfg : Qwen3TTSConfig := {
  ttsModelType := "custom_voice"
  talkerConfig := {
    vocabSize := 8, textVocabSize := 8, textHiddenSize := 8,
    hiddenSize := 8, intermediateSize := 16, numHiddenLayers := 1,
    numAttentionHeads := 2, numKeyValueHeads := 1, headDim := 4,
    numCodeGroups := 2, codecBosId := 1, codecPadId := 0, codecEosTokenId := 7,
    codePredictorConfig := {
      vocabSize := 8, hiddenSize := 8, intermediateSize := 16,
      numHiddenLayers := 1, numAttentionHeads := 2, numKeyValueHeads := 1, headDim := 4
    }
  }
}

private def layerEntries (namePrefix : String) : Array (String × Array Nat) :=
  #[ ("input_layernorm.weight", #[8]), ("post_attention_layernorm.weight", #[8]),
     ("self_attn.q_proj.weight", #[8, 8]), ("self_attn.k_proj.weight", #[4, 8]),
     ("self_attn.v_proj.weight", #[4, 8]), ("self_attn.o_proj.weight", #[8, 8]),
     ("self_attn.q_norm.weight", #[4]), ("self_attn.k_norm.weight", #[4]),
     ("mlp.gate_proj.weight", #[16, 8]), ("mlp.up_proj.weight", #[16, 8]),
     ("mlp.down_proj.weight", #[8, 16]) ].map fun (name, shape) => (namePrefix ++ name, shape)

private structure FixtureOptions where
  speaker : Bool := false
  speakerBF16 : Bool := false
  dtypeOverrides : Array (String × Bool) := #[]
  shapeOverrides : Array (String × Array Nat) := #[]
  omitted : Array String := #[]

private def tdnnEntries (namePrefix : String) (input output kernel : Nat) : Array (String × Array Nat) :=
  #[(namePrefix ++ ".conv.weight", #[output, input, kernel]),
    (namePrefix ++ ".conv.bias", #[output])]

private def speakerEntries : Array (String × Array Nat) := Id.run do
  let mut entries := tdnnEntries "speaker_encoder.blocks.0" 4 4 1
  for i in [1:4] do
    let p := s!"speaker_encoder.blocks.{i}"
    entries := entries ++ tdnnEntries (p ++ ".tdnn1") 4 4 1 ++
      tdnnEntries (p ++ ".res2net_block.blocks.0") 2 2 1 ++
      tdnnEntries (p ++ ".tdnn2") 4 4 1 ++ #[
        (p ++ ".se_block.conv1.weight", #[2, 4, 1]),
        (p ++ ".se_block.conv1.bias", #[2]),
        (p ++ ".se_block.conv2.weight", #[4, 2, 1]),
        (p ++ ".se_block.conv2.bias", #[4])]
  return entries ++ tdnnEntries "speaker_encoder.mfa" 12 12 1 ++
    tdnnEntries "speaker_encoder.asp.tdnn" 36 2 1 ++ #[
      ("speaker_encoder.asp.conv.weight", #[12, 2, 1]),
      ("speaker_encoder.asp.conv.bias", #[12]),
      ("speaker_encoder.fc.weight", #[8, 24, 1]),
      ("speaker_encoder.fc.bias", #[8])]

private def baseCfg : Qwen3TTSConfig := { cfg with
  ttsModelType := "base"
  speakerEncoderConfig := {
    melDim := 4, encDim := 8, encChannels := #[4, 4, 4, 4, 12],
    encKernelSizes := #[1, 1, 1, 1, 1], encDilations := #[1, 1, 1, 1, 1],
    encRes2NetScale := 2, encSeChannels := 2, encAttentionChannels := 2 }
}

private def writeCheckpoint (dir : System.FilePath) (bf16 projection sharded : Bool)
    (options : FixtureOptions := {}) : IO Unit := do
  let entries : Array (String × Array Nat) := #[
    ("talker.model.codec_embedding.weight", #[8, 8]),
    ("talker.model.text_embedding.weight", #[8, 8]),
    ("talker.text_projection.linear_fc1.weight", #[8, 8]),
    ("talker.text_projection.linear_fc1.bias", #[8]),
    ("talker.text_projection.linear_fc2.weight", #[8, 8]),
    ("talker.text_projection.linear_fc2.bias", #[8]),
    ("talker.model.norm.weight", #[8]), ("talker.codec_head.weight", #[8, 8]),
    ("talker.code_predictor.model.codec_embedding.0.weight", #[8, 8]),
    ("talker.code_predictor.model.norm.weight", #[8]),
    ("talker.code_predictor.lm_head.0.weight", #[8, 8])
  ] ++ layerEntries "talker.model.layers.0." ++ layerEntries "talker.code_predictor.model.layers.0." ++
    (if projection then #[
      ("talker.code_predictor.small_to_mtp_projection.weight", #[8, 8]),
      ("talker.code_predictor.small_to_mtp_projection.bias", #[8]) ] else #[])
  let mut header : List (String × Json) := []
  let mut index : List (String × Json) := []
  let mut data := ByteArray.empty
  for (name, originalShape) in entries ++ (if options.speaker then speakerEntries else #[]) do
    if options.omitted.contains name then continue
    let shape := (options.shapeOverrides.findSome? fun (key, shape) =>
      if key == name then some shape else none).getD originalShape
    let entryBF16 := (options.dtypeOverrides.findSome? fun (key, bf16) =>
      if key == name then some bf16 else none).getD
        (if name.startsWith "speaker_encoder." then options.speakerBF16 else bf16)
    let start := data.size
    for _ in [:shape.foldl (· * ·) 1] do
      data := data ++ (if entryBF16 then ByteArray.mk #[128, 63] else ByteArray.mk #[0, 0, 128, 63])
    header := header ++ [(name, Json.mkObj [
      ("dtype", toJson (if entryBF16 then "BF16" else "F32")),
      ("shape", toJson shape), ("data_offsets", toJson #[start, data.size])])]
    index := index ++ [(name, toJson "model.safetensors")]
  let bytes := (Json.mkObj header).compress.toUTF8
  let mut namePrefix := ByteArray.empty
  for i in [:8] do
    namePrefix := namePrefix.push ((bytes.size.toUInt64 >>> (i * 8).toUInt64).toUInt8)
  IO.FS.writeBinFile (dir / "model.safetensors") (namePrefix ++ bytes ++ data)
  if sharded then
    IO.FS.writeFile (dir / "model.safetensors.index.json")
      (Json.mkObj [("weight_map", Json.mkObj index)]).compress

@[test] def testTTSTypedCheckpointDTypesAndOptionalProjection : IO Unit := do
  for bf16 in [false, true] do
    for projection in [false, true] do
      for sharded in [false, true] do
        IO.FS.withTempDir fun dir => do
          writeCheckpoint dir bf16 projection sharded
          let ⟨dtypes, model⟩ ← Qwen3TTSForConditionalGeneration.loadSharded dir.toString cfg
          let model := { model with
            talker := TensorStruct.detach model.talker
            speakerEncoder := model.speakerEncoder.map TensorStruct.detach }
          let dtype := if bf16 then DType.BFloat16 else .Float32
          LeanTest.assertEqual dtypes.talker.val dtype
          LeanTest.assertEqual model.talker.model.codecEmbedding.dtype dtype
          LeanTest.assertEqual model.talker.model.codecEmbedding.actualSpec.dtype dtype
          LeanTest.assertEqual model.talker.codePredictor.inputProjection.actualSpec.dtype dtype
          LeanTest.assertEqual model.talker.codePredictor.inputProjectionBias.actualSpec.dtype dtype
          LeanTest.assertTrue model.speakerEncoder.isNone
          let ids := Tensor.fullInt #[1, 2] 1
          let prompt := model.buildTalkerInputsFromText ids
          let logits := TalkerForConditionalGeneration.forward cfg.talkerConfig model.talker prompt
          LeanTest.assertEqual logits.actualSpec.dtype dtype
          LeanTest.assertTrue ((← logits.toFloatArray).all Float.isFinite)
          if !projection then
            let identity := (Tensor.eye 8).toFloatArray
            LeanTest.assertEqual (← model.talker.codePredictor.inputProjection.toFloatArray) (← identity)
          let generated ← model.generateCodesWithLengths prompt 2 2 1.0 1 1.0
          LeanTest.assertEqual generated.codes.actualSpec.dtype DType.Int64
          LeanTest.assertEqual generated.codes.runtimeShape (#[1, 2, 2] : Shape)
          LeanTest.assertEqual generated.lengths (#[2] : Array UInt64)

@[test] def testTTSLoadingRejectsInconsistentTalkerDType : IO Unit := do
  for name in #["talker.model.text_embedding.weight",
      "talker.model.layers.0.self_attn.q_proj.weight",
      "talker.model.layers.0.self_attn.k_norm.weight",
      "talker.model.layers.0.input_layernorm.weight",
      "talker.code_predictor.model.codec_embedding.0.weight",
      "talker.code_predictor.small_to_mtp_projection.bias",
      "talker.code_predictor.lm_head.0.weight", "talker.codec_head.weight"] do
    IO.FS.withTempDir fun dir => do
      writeCheckpoint dir true true true { dtypeOverrides := #[(name, false)] }
      Tests.assertThrows (Qwen3TTSForConditionalGeneration.loadSharded dir.toString cfg)
        (some "Expected dtype")

@[test] def testTTSLoadingRejectsMalformedOptionalEntries : IO Unit := do
  for (name, shape) in #[
      ("talker.model.layers.0.self_attn.q_norm.weight", #[2, 2]),
      ("talker.code_predictor.small_to_mtp_projection.weight", #[4, 16]),
      ("talker.code_predictor.small_to_mtp_projection.bias", #[4, 2])] do
    IO.FS.withTempDir fun dir => do
      writeCheckpoint dir true true false { shapeOverrides := #[(name, shape)] }
      Tests.assertThrows (Qwen3TTSForConditionalGeneration.loadSharded dir.toString cfg)
        (some "Expected shape")

@[test] def testTTSLoadingRejectsPartiallyMissingProjection : IO Unit := do
  for missing in #["talker.code_predictor.small_to_mtp_projection.weight",
      "talker.code_predictor.small_to_mtp_projection.bias"] do
    IO.FS.withTempDir fun dir => do
      writeCheckpoint dir true true false { omitted := #[missing] }
      Tests.assertThrows (Qwen3TTSForConditionalGeneration.loadSharded dir.toString cfg)
        (some "projection")

-- A model carries one shared talker index all the way from its embeddings to
-- its head; the optional speaker role has an independent exact index.
example (cfg : Qwen3TTSConfig) (dtypes : DTypeLayout)
    (model : Qwen3TTSForConditionalGeneration (device := .CPU) cfg dtypes) :
    Tensor ⟨#[cfg.talkerConfig.vocabSize, cfg.talkerConfig.hiddenSize], dtypes.talker.val, .CPU⟩ :=
  model.talker.codecHead

@[test] def testTTSLoadingRetainsIndependentSpeakerDType : IO Unit := do
  for talkerBF16 in [false, true] do
    IO.FS.withTempDir fun dir => do
      writeCheckpoint dir talkerBF16 false true {
        speaker := true, speakerBF16 := !talkerBF16 }
      let ⟨dtypes, model⟩ ← Qwen3TTSForConditionalGeneration.loadSharded dir.toString baseCfg
      let talkerDType := if talkerBF16 then DType.BFloat16 else .Float32
      let speakerDType := if talkerBF16 then DType.Float32 else .BFloat16
      LeanTest.assertEqual dtypes.talker.val talkerDType
      LeanTest.assertEqual dtypes.speaker.val speakerDType
      let some encoder := model.speakerEncoder | throw (IO.userError "expected speaker encoder")
      LeanTest.assertEqual encoder.tdnn0.weight.dtype speakerDType
      LeanTest.assertEqual encoder.fcWeight.raw.dtype speakerDType
      LeanTest.assertEqual model.talker.codecHead.raw.dtype talkerDType
      let mel := Tensor.ones #[1, 3, 4]
      let embedding ← model.extractSpeakerEmbedding mel
      LeanTest.assertEqual embedding.dtype speakerDType
      LeanTest.assertTrue ((← embedding.toFloatArray).all Float.isFinite)

@[test] def testTTSLoadingRejectsInconsistentSpeakerDType : IO Unit := do
  IO.FS.withTempDir fun dir => do
    writeCheckpoint dir true true true {
      speaker := true, speakerBF16 := false,
      dtypeOverrides := #[("speaker_encoder.fc.weight", true)] }
    Tests.assertThrows (Qwen3TTSForConditionalGeneration.loadSharded dir.toString baseCfg)
      (some "Expected dtype")

@[test] def testTTSLoadingAllowsAbsentQKNorms : IO Unit := do
  IO.FS.withTempDir fun dir => do
    writeCheckpoint dir true false false { omitted := #[
      "talker.model.layers.0.self_attn.q_norm.weight",
      "talker.model.layers.0.self_attn.k_norm.weight"] }
    let ⟨_, model⟩ ← Qwen3TTSForConditionalGeneration.loadSharded dir.toString cfg
    let some layer := model.talker.model.layers[0]? | throw (IO.userError "expected talker layer")
    LeanTest.assertTrue layer.self_attn.q_norm.isNone
    LeanTest.assertTrue layer.self_attn.k_norm.isNone

end Tests.TTSTypedLoading
