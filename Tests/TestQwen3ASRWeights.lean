import Tyr.Model.Qwen3ASR.Weights
import Tests.Assertions
import LeanTest

open torch torch.qwen3asr

namespace Tests.Qwen3ASRWeights

private def cfg : Qwen3ASRConfig := {
  thinkerConfig := {
    audioConfig := {
      numMelBins := 8, dModel := 8, outputDim := 8,
      downsampleHiddenSize := 2, encoderLayers := 1, encoderAttentionHeads := 2,
      encoderFfnDim := 12 }
    textConfig := {
      vocabSize := 16, hiddenSize := 8, intermediateSize := 12,
      numHiddenLayers := 1, numAttentionHeads := 2, numKeyValueHeads := 1,
      headDim := 4 }
    audioTokenId := 3 } }

private def entries : Array (String × Array Nat) := Id.run do
  let a := "thinker.audio_tower"
  let t := "thinker.model.layers.0"
  let mut result := #[
    (s!"{a}.conv2d1.weight", #[2, 1, 3, 3]), (s!"{a}.conv2d1.bias", #[2]),
    (s!"{a}.conv2d2.weight", #[2, 2, 3, 3]), (s!"{a}.conv2d2.bias", #[2]),
    (s!"{a}.conv2d3.weight", #[2, 2, 3, 3]), (s!"{a}.conv2d3.bias", #[2]),
    (s!"{a}.conv_out.weight", #[8, 2]),
    (s!"{a}.proj1.weight", #[8, 8]), (s!"{a}.proj1.bias", #[8]),
    (s!"{a}.proj2.weight", #[8, 8]), (s!"{a}.proj2.bias", #[8]),
    (s!"{a}.layers.0.fc1.weight", #[12, 8]), (s!"{a}.layers.0.fc1.bias", #[12]),
    (s!"{a}.layers.0.fc2.weight", #[8, 12]), (s!"{a}.layers.0.fc2.bias", #[8]),
    ("thinker.model.embed_tokens.weight", #[16, 8]), ("thinker.model.norm.weight", #[8]),
    (s!"{t}.input_layernorm.weight", #[8]), (s!"{t}.post_attention_layernorm.weight", #[8]),
    (s!"{t}.self_attn.q_proj.weight", #[8, 8]), (s!"{t}.self_attn.k_proj.weight", #[4, 8]),
    (s!"{t}.self_attn.v_proj.weight", #[4, 8]), (s!"{t}.self_attn.o_proj.weight", #[8, 8]),
    (s!"{t}.mlp.gate_proj.weight", #[12, 8]), (s!"{t}.mlp.up_proj.weight", #[12, 8]),
    (s!"{t}.mlp.down_proj.weight", #[8, 12]), ("thinker.lm_head.weight", #[16, 8])]
  for norm in #["ln_post", "layers.0.self_attn_layer_norm", "layers.0.final_layer_norm"] do
    result := result ++ #[(s!"{a}.{norm}.weight", #[8]), (s!"{a}.{norm}.bias", #[8])]
  for proj in #["q_proj", "k_proj", "v_proj", "out_proj"] do
    result := result ++ #[(s!"{a}.layers.0.self_attn.{proj}.weight", #[8, 8]),
      (s!"{a}.layers.0.self_attn.{proj}.bias", #[8])]
  return result

private def writeCheckpoint (dir : System.FilePath) (bf16 : Bool)
    (mismatchedDType : String := "") (badOptionalShape : Bool := false) : IO Unit := do
  let weights := if badOptionalShape then
    entries.push ("thinker.model.layers.0.self_attn.q_norm.weight", #[2, 2])
    else entries
  let mut fields : List (String × Lean.Json) := []
  let mut data := ByteArray.empty
  for (name, shape) in weights do
    let entryBf16 := if name == mismatchedDType then !bf16 else bf16
    let start := data.size
    for _ in [:shape.foldl (· * ·) 1] do
      data := data ++ (if entryBf16 then ByteArray.mk #[0, 62] else ByteArray.mk #[0, 0, 0, 62])
    fields := fields ++ [(name, Lean.Json.mkObj [
      ("dtype", Lean.toJson (if entryBf16 then "BF16" else "F32")),
      ("shape", Lean.toJson shape), ("data_offsets", Lean.toJson #[start, data.size])])]
  let header := (Lean.Json.mkObj fields).compress.toUTF8
  let mut headerSize := ByteArray.empty
  for i in [:8] do headerSize := headerSize.push ((header.size.toUInt64 >>> (i * 8).toUInt64).toUInt8)
  IO.FS.writeBinFile (dir / "model.safetensors") (headerSize ++ header ++ data)

@[test] def testExactASRCheckpointDTypesAndOptionalNorms : IO Unit := do
  for bf16 in #[false, true] do
    IO.FS.withTempDir fun dir => do
      writeCheckpoint dir bf16
      let ⟨dtype, model⟩ ← Qwen3ASRForConditionalGeneration.loadShardedAt dir.toString cfg (device := .CPU)
      LeanTest.assertEqual dtype.val (if bf16 then .BFloat16 else .Float32)
      LeanTest.assertEqual model.thinker.audioProjectionWeight.actualSpec.dtype dtype.val
      LeanTest.assertEqual model.thinker.audioProjectionBias.actualSpec.dtype dtype.val
      let some layer := model.thinker.textModel.layers[0]? | throw (IO.userError "missing text layer")
      LeanTest.assertTrue layer.self_attn.q_norm.isNone "missing optional norm stays absent"
      let features := Tensor.ones #[1, 8, 16]
      let ids := (Tensor.fromInt64Array #[1, 2]).reshape #[1, 2]
      let logits := model.forwardFromMel features (by decide) ids
      LeanTest.assertEqual logits.actualSpec.dtype dtype.val
      for value in (← logits.toFloatArray) do
        LeanTest.assertTrue value.isFinite "loaded exact model produces finite logits"

@[test] def testExactASRRejectsInconsistentStoredDTypes : IO Unit := do
  for name in #["thinker.audio_tower.proj2.weight", "thinker.model.embed_tokens.weight",
      "thinker.lm_head.weight"] do
    IO.FS.withTempDir fun dir => do
      writeCheckpoint dir true (mismatchedDType := name)
      Tests.assertThrows (do
        let _ ← Qwen3ASRForConditionalGeneration.loadShardedAt dir.toString cfg (device := .CPU)
        pure ())

@[test] def testExactASRRejectsMalformedOptionalNorm : IO Unit :=
  IO.FS.withTempDir fun dir => do
    writeCheckpoint dir false (badOptionalShape := true)
    Tests.assertThrows (do
      let _ ← Qwen3ASRForConditionalGeneration.loadShardedAt dir.toString cfg (device := .CPU)
      pure ())

end Tests.Qwen3ASRWeights
