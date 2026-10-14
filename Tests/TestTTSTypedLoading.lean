import Tyr.Model.Qwen3TTS
import LeanTest

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

private def writeCheckpoint (dir : System.FilePath) (bf16 projection sharded : Bool) : IO Unit := do
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
  for (name, shape) in entries do
    let start := data.size
    for _ in [:shape.foldl (· * ·) 1] do
      data := data ++ (if bf16 then ByteArray.mk #[128, 63] else ByteArray.mk #[0, 0, 128, 63])
    header := header ++ [(name, Json.mkObj [
      ("dtype", toJson (if bf16 then "BF16" else "F32")),
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
          let model ← Qwen3TTSForConditionalGeneration.loadSharded dir.toString cfg
          let model := { model with
            talker := TensorStruct.detach model.talker
            speakerEncoder := model.speakerEncoder.map TensorStruct.detach }
          let dtype := if bf16 then DType.BFloat16 else .Float32
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

end Tests.TTSTypedLoading
