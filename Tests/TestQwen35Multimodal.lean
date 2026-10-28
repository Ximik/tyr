import Tests.Assertions
import Tyr.Model.Qwen35.Media
import Tyr.Model.Qwen35.Multimodal
import Tyr.Model.Qwen35.VLConfigIO
import LeanTest

open torch

private def randnDynamic (shape : Shape) : IO (Tensor { shape := shape, device := .CPU, dtype := .Float32 }) := do
  return (← Tensor.randn shape)
open torch.Model
open torch.qwen35

private def tinyTextCfg : Config := {
  vocab_size := 64
  hidden_size := 16
  intermediate_size := 32
  num_hidden_layers := 2
  num_attention_heads := 4
  num_key_value_heads := 2
  head_dim := 4
  rope_theta := 10000.0
  partial_rotary_factor := 1.0
  rms_norm_eps := 1e-6
  max_position_embeddings := 128
  attention_bias := false
  attention_dropout := 0.0
  hidden_act := "silu"
  linear_conv_kernel_dim := 4
  linear_key_head_dim := 4
  linear_value_head_dim := 4
  linear_num_key_heads := 2
  linear_num_value_heads := 4
  layer_types := #[.fullAttention, .linearAttention]
  full_attention_interval := 2
  moe_intermediate_size := 8
  shared_expert_intermediate_size := 8
  num_experts_per_tok := 2
  num_experts := 0
  use_cache := true
  tie_word_embeddings := false
  pad_token_id := some 0
  bos_token_id := some 1
  eos_token_id := some 2
}

private def tinyVisionCfg : VisionConfig := {
  depth := 2
  hidden_size := 16
  hidden_act := "gelu_pytorch_tanh"
  intermediate_size := 32
  num_heads := 4
  in_channels := 3
  patch_size := 2
  spatial_merge_size := 1
  temporal_patch_size := 1
  out_hidden_size := 16
  num_position_embeddings := 32
  initializer_range := 0.02
}

private def tinyVLCfg : VLConfig := VLConfig.normalize {
  text_config := tinyTextCfg
  vision_config := tinyVisionCfg
  image_token_id := 50
  video_token_id := 51
  vision_start_token_id := 52
  vision_end_token_id := 53
  tie_word_embeddings := false
}

@[test]
def testQwen35MultimodalConfigLoad : IO Unit := do
  let path := "/tmp/qwen35_vl_config_test.json"
  let json :=
    "{\"image_token_id\":250001,\"video_token_id\":250002,\"vision_start_token_id\":250003," ++
    "\"vision_end_token_id\":250004,\"text_config\":{" ++
    "\"vocab_size\":128,\"hidden_size\":48,\"intermediate_size\":96,\"num_hidden_layers\":4," ++
    "\"num_attention_heads\":6,\"num_key_value_heads\":2,\"head_dim\":8," ++
    "\"attn_output_gate\":true," ++
    "\"rope_parameters\":{\"rope_theta\":500000.0,\"partial_rotary_factor\":0.5," ++
    "\"mrope_interleaved\":true,\"mrope_section\":[2,2,2]}}," ++
    "\"vision_config\":{" ++
    "\"depth\":3,\"hidden_size\":24,\"intermediate_size\":48,\"num_heads\":4," ++
    "\"in_channels\":3,\"patch_size\":2,\"temporal_patch_size\":1,\"spatial_merge_size\":1," ++
    "\"out_hidden_size\":48,\"num_position_embeddings\":64}}"
  IO.FS.writeFile path json

  let cfg ← VLConfig.loadFromFile path
  LeanTest.assertEqual cfg.image_token_id 250001 "image token id should load"
  LeanTest.assertEqual cfg.video_token_id 250002 "video token id should load"
  LeanTest.assertEqual cfg.text_config.hidden_size 48 "text hidden size should load"
  LeanTest.assertEqual cfg.text_config.attn_output_gate true "nested text attention gate should load"
  LeanTest.assertEqual cfg.text_config.mrope_interleaved true "nested text mrope flag should load"
  LeanTest.assertEqual cfg.text_config.mrope_section #[2, 2, 2] "nested text mrope section should load"
  LeanTest.assertEqual cfg.vision_config.depth 3 "vision depth should load"
  LeanTest.assertEqual cfg.vision_config.out_hidden_size 48 "vision out_hidden_size should load"

@[test]
def testQwen35MultimodalForwardWithImagePatches : IO Unit := do
  let model ← Qwen35ForConditionalGeneration.init tinyVLCfg

  let ids : Tensor { shape := #[1, 6], dtype := .Int64, device := .CPU } := Tensor.reshape (Tensor.fromInt64Array #[2, 50, 50, 50, 50, 3]) #[1, 6]
  let patches ← randnDynamic #[4, VisionConfig.patchDim tinyVLCfg.vision_config]
  let logits ← model.forwardWithImagePatches tinyVLCfg ids patches none

  LeanTest.assertEqual logits.runtimeShape #[1, 6, tinyVLCfg.text_config.vocab_size]
    "multimodal forward should return [batch, seq, vocab]"

@[test]
def testQwen35MultimodalPlaceholderCountMismatch : IO Unit := do
  let model ← Qwen35ForConditionalGeneration.init tinyVLCfg

  -- only 3 image placeholders in ids but 4 visual tokens from patches
  let ids : Tensor { shape := #[1, 6], dtype := .Int64, device := .CPU } := Tensor.reshape (Tensor.fromInt64Array #[2, 50, 50, 50, 4, 3]) #[1, 6]
  let patches ← randnDynamic #[4, VisionConfig.patchDim tinyVLCfg.vision_config]

  let didThrow ←
    try
      let _ ← model.forwardWithImagePatches tinyVLCfg ids patches none
      pure false
    catch _ =>
      pure true

  LeanTest.assertTrue didThrow
    "forward should fail when placeholder token count and visual feature count differ"

@[test]
def testQwen35MultimodalGenerateStreamBatched : IO Unit := do
  let model ← Qwen35ForConditionalGeneration.init tinyVLCfg
  let ids : Tensor { shape := #[2, 4], dtype := .Int64, device := .CPU } :=
    Tensor.reshape (Tensor.fromInt64Array #[2, 5, 6, 7, 2, 8, 9, 10]) #[2, 4]

  let callbacksRef ← IO.mkRef (0 : Nat)
  let onStep : StreamCallback 2 := fun _ nextTok => do
    let flat : Tensor { shape := #[2], dtype := .Int64, device := .CPU } := Tensor.reshape (id nextTok) #[2]
    let vals ← Tensor.toUInt64Array flat
    if vals.size == 2 then
      callbacksRef.modify (fun n => n + 1)

  let ⟨outSeq, outIds⟩ ←
    model.generateStream
      tinyVLCfg
      ids
      onStep
      3
      .greedy
      #[]
      none
      none

  LeanTest.assertEqual outSeq 7
    "batched streaming generation should append maxNewTokens"
  LeanTest.assertEqual outIds.runtimeShape #[2, outSeq]
    "batched streaming output should have shape [batch, seq]"

  let nCallbacks ← callbacksRef.get
  LeanTest.assertEqual nCallbacks 3
    "stream callback should run once per decoding step"

@[test]
def testQwen35MultimodalGenerateStreamWithImageAndVideoFeatures : IO Unit := do
  let model ← Qwen35ForConditionalGeneration.init tinyVLCfg
  let ids : Tensor { shape := #[1, 8], dtype := .Int64, device := .CPU } :=
    Tensor.reshape (Tensor.fromInt64Array #[2, 50, 50, 51, 51, 51, 9, 10]) #[1, 8]

  let imageFeat : Tensor { shape := #[2, tinyVLCfg.vision_config.out_hidden_size], device := .CPU, dtype := .Float32 } ←
    randnDynamic #[2, tinyVLCfg.vision_config.out_hidden_size]
  let videoFeat : Tensor { shape := #[3, tinyVLCfg.vision_config.out_hidden_size], device := .CPU, dtype := .Float32 } ←
    randnDynamic #[3, tinyVLCfg.vision_config.out_hidden_size]

  let callbacksRef ← IO.mkRef (0 : Nat)
  let onStep : StreamCallback 1 := fun _ nextTok => do
    let flat : Tensor { shape := #[1], dtype := .Int64, device := .CPU } := Tensor.reshape (id nextTok) #[1]
    let vals ← Tensor.toUInt64Array flat
    if vals.size == 1 then
      callbacksRef.modify (fun n => n + 1)

  let ⟨seqStream, outStream⟩ ←
    model.generateStream
      tinyVLCfg
      ids
      onStep
      3
      .greedy
      #[]
      (some ⟨2, imageFeat⟩)
      (some ⟨3, videoFeat⟩)

  let ⟨seqPlain, outPlain⟩ ←
    model.generate
      tinyVLCfg
      ids
      3
      .greedy
      #[]
      (some ⟨2, imageFeat⟩)
      (some ⟨3, videoFeat⟩)

  LeanTest.assertEqual seqStream seqPlain
    "multimodal streaming and non-streaming generation should have equal output length"

  let streamFlat : Tensor { shape := #[seqStream], dtype := .Int64, device := .CPU } := Tensor.reshape (id outStream) #[seqStream]
  let plainFlat : Tensor { shape := #[seqPlain], dtype := .Int64, device := .CPU } := Tensor.reshape (id outPlain) #[seqPlain]
  let streamIds ← Tensor.toUInt64Array streamFlat
  let plainIds ← Tensor.toUInt64Array plainFlat
  LeanTest.assertEqual streamIds plainIds
    "multimodal streaming and non-streaming generation should produce identical token ids"

  let nCallbacks ← callbacksRef.get
  LeanTest.assertEqual nCallbacks 3
    "stream callback should run once per generated step with multimodal features"

@[test] def testQwen35MediaTypedPatchBoundary : IO Unit := IO.FS.withTempDir fun dir => do
  let path := dir / "pixels.png"
  let bytes := ByteArray.mk #[137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 2, 0, 0, 0, 2, 8, 2, 0, 0, 0, 253, 212, 154, 115, 0, 0, 0, 16, 73, 68, 65, 84, 120, 156, 99, 112, 104, 56, 0, 68, 12, 16, 10, 0, 40, 14, 6, 1, 67, 58, 99, 218, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130]
  IO.FS.writeBinFile path bytes
  if System.Platform.isOSX then
    let ⟨count, patches⟩ ← media.loadImagePatches tinyVLCfg path.toString
    LeanTest.assertEqual count 1
    LeanTest.assertEqual patches.runtimeShape #[1, 12]
    LeanTest.assertEqual patches.actualSpec.dtype DType.Float32
    LeanTest.assertTrue ((← patches.toFloatArray).all Float.isFinite)
  else
    Tests.assertThrows (media.loadImagePatches tinyVLCfg path.toString) (some "requires macOS")
  Tests.assertThrows (media.loadVideoPatches tinyVLCfg (dir / "missing.mov").toString)
