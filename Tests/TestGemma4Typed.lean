import Tyr.Model.Gemma4.Multimodal
import Tyr.Model.Gemma4.Media
import Tyr.Model.Gemma4.Weights
import Tyr.Model.Gemma4.VLWeights
import Tyr.Checkpoint
import LeanTest

open torch torch.gemma4

namespace Tests.Gemma4Typed

private def cfg : Config := {
  vocab_size := 16, hidden_size := 8, intermediate_size := 12
  num_hidden_layers := 4, num_attention_heads := 2, num_key_value_heads := 1
  head_dim := 4, global_head_dim := 8, num_global_key_value_heads := 1
  sliding_window := 3, full_partial_rotary_factor := 1.0
  num_kv_shared_layers := 2
  layer_types := #[.fullAttention, .slidingAttention, .slidingAttention, .fullAttention]
  hidden_size_per_layer_input := 2, vocab_size_per_layer_input := 16
}

private def weight (s : Shape) (offset : Float := 0.0) : STensor s :=
  let n := s.foldl (· * ·) 1
  let vals := Array.ofFn (n := n.toNat) fun i => Float.sin (i.val.toFloat * 0.71 + offset) * 0.2
  (Tensor.fromFloatArray vals).reshapeChecked s |>.forgetDType

private def norm (n : UInt64) : Gemma4RMSNorm n := { weight := (Tensor.ones #[n]).forgetDType }

private def textModel (c : Config) : Gemma4ForCausalLM c := Id.run do
  let mut layers := #[]
  for i in [:c.num_hidden_layers.toNat] do
    let layer : Gemma4Layer c := {
      layerIdx := i.toUInt64, layerType := Config.layerTypeAt c i.toUInt64
      self_attn := {
        q_proj := weight #[Config.maxAttentionDim c, c.hidden_size] 0.1
        k_proj := weight #[Config.maxKVProjDim c, c.hidden_size] 0.3
        v_proj := weight #[Config.maxKVProjDim c, c.hidden_size] 0.5
        o_proj := weight #[c.hidden_size, Config.maxAttentionDim c] 0.7
        q_norm := norm (Config.maxHeadDim c), k_norm := norm (Config.maxHeadDim c) }
      mlp := {
        gate_proj := weight #[Config.maxIntermediateSize c, c.hidden_size] 0.2
        up_proj := weight #[Config.maxIntermediateSize c, c.hidden_size] 0.4
        down_proj := weight #[c.hidden_size, Config.maxIntermediateSize c] 0.6 }
      input_layernorm := norm c.hidden_size, post_attention_layernorm := norm c.hidden_size
      pre_feedforward_layernorm := norm c.hidden_size, post_feedforward_layernorm := norm c.hidden_size
      layer_scalar := (Tensor.ones #[1]).forgetDType
      per_layer_input := some {
        per_layer_input_gate := weight #[c.hidden_size_per_layer_input, c.hidden_size] 0.8
        per_layer_projection := weight #[c.hidden_size, c.hidden_size_per_layer_input] 0.9
        post_per_layer_input_norm := norm c.hidden_size } }
    layers := layers.push layer
  return {
    model := {
      embed_tokens := weight #[c.vocab_size, c.hidden_size] 1.0
      embed_tokens_per_layer := some (weight #[c.vocab_size_per_layer_input, c.num_hidden_layers * c.hidden_size_per_layer_input] 1.2)
      per_layer_model_projection := some (weight #[c.num_hidden_layers * c.hidden_size_per_layer_input, c.hidden_size] 1.4)
      per_layer_projection_norm := some (norm c.hidden_size_per_layer_input)
      layers, norm := norm c.hidden_size }
    lmHead := weight #[c.vocab_size, c.hidden_size] 1.6 }

private def ids : DTensor #[1, 7] .Int64 :=
  (Tensor.fromInt64Array #[2, 4, 7, 3, 9, 5, 11]).reshapeChecked #[1, 7]

private def close {σ τ : StaticSpec} (x : Tensor σ) (y : Tensor τ) (tol : Float := 1e-4) : IO Unit := do
  LeanTest.assertEqual x.runtimeShape y.runtimeShape
  let a ← x.toFloatArray
  let b ← y.toFloatArray
  for i in [:a.size] do
    LeanTest.assertTrue (a[i]!.isFinite && b[i]!.isFinite && (a[i]! - b[i]!).abs <= tol)
      s!"tensor mismatch at {i}: {a[i]!} vs {b[i]!}, tolerance {tol}"

@[test] def testGemma4SharedCacheAndPerLayerInputs : IO Unit := do
  let m := textModel cfg
  LeanTest.assertEqual (Config.sharedSourceLayer? cfg 2) (some 1)
  LeanTest.assertEqual (Config.sharedSourceLayer? cfg 3) (some 0)
  let expected := m.model.forward cfg ids
  let perLayer := m.model.computePerLayerInputs cfg ids (m.embedTokens cfg ids)
  LeanTest.assertTrue perLayer.isSome
  let (sc, ss) := Tensor.rotaryFreqs 7 cfg.head_dim cfg.sliding_rope_theta
  let (fc, fs) := Tensor.rotaryFreqs 7 (Config.fullHeadDim cfg) cfg.full_rope_theta
  let cache := m.model.initCache cfg 7 .CPU
  let (prefill, initial) := m.model.forwardWithCache cfg (ids.sliceChecked 1 0 3 #[1, 3]) cache sc ss fc fs
  close prefill (expected.sliceChecked 1 0 3 #[1, 3, cfg.hidden_size])
  let mut cache := initial
  for pos in [3:7] do
    let (out, next) := m.model.forwardStep cfg (ids.sliceChecked 1 pos.toUInt64 1 #[1, 1]) pos.toUInt64 cache sc ss fc fs
    close out (expected.sliceChecked 1 pos.toUInt64 1 #[1, 1, cfg.hidden_size])
    cache := next

@[test] def testGemma4GenerationDTypesAndStreaming : IO Unit := do
  for bf16 in #[false, true] do
    let m := textModel cfg
    let m := if bf16 then TensorStruct.map (fun t => t.checkedReplacement t.toBFloat16.raw) m else m
    let prediction := m.forward cfg ids
    LeanTest.assertEqual prediction.actualSpec.dtype (if bf16 then .BFloat16 else .Float32)
    for v in (← prediction.toFloatArray) do LeanTest.assertTrue v.isFinite
    let plain ← m.generate cfg ids 3 .greedy #[]
    let steps ← IO.mkRef (#[] : Array UInt64)
    let streamed ← m.generateStream cfg ids (fun _ tok => do
      LeanTest.assertEqual tok.actualSpec.dtype .Int64
      steps.modify (· ++ (← tok.toUInt64Array))) 3 .greedy #[]
    LeanTest.assertEqual plain.1 10
    LeanTest.assertEqual (← plain.2.toUInt64Array) (← streamed.2.toUInt64Array)
    LeanTest.assertEqual (← steps.get).size 3

@[noinline, nospecialize]
private def roundTrip {α : Type} [TensorStruct α] (params : α) (dir : String) : IO α := do
  checkpoint.saveParams params dir
  checkpoint.loadParams params dir

@[test] def testGemma4IndexedCheckpoint : IO Unit := IO.FS.withTempDir fun dir => do
  let m := textModel cfg
  let restored ← roundTrip m dir.toString
  close (m.forward cfg ids) (restored.forward cfg ids) 0.0

@[test] def testGemma4BFloat16Router : IO Unit := do
  let c := { cfg with num_experts := 3, top_k_experts := 2 }
  let router : Gemma4TextRouter c := {
    proj := weight #[3, 8] 0.3
    scale := weight #[8] 0.5
    per_expert_scale := (Tensor.ones #[3]).forgetDType }
  let router := TensorStruct.map (fun t => t.checkedReplacement t.toBFloat16.raw) router
  let x := (weight #[2, 8] 0.7).toBFloat16.forgetDType
  let (probs, weights, indices) := router.forward c x
  LeanTest.assertEqual probs.actualSpec.dtype .Float32
  LeanTest.assertEqual indices.actualSpec.dtype .Int64
  for v in (← weights.toFloatArray) do LeanTest.assertTrue v.isFinite
  let rows ← probs.toFloatArray
  for row in [:2] do
    let total := rows[row * 3]! + rows[row * 3 + 1]! + rows[row * 3 + 2]!
    LeanTest.assertTrue ((total - 1.0).abs < 1e-6)

@[test] def testGemma4RoutedExpertsReference : IO Unit := do
  let x := weight #[2, 4] 0.4
  let gate := weight #[3, 12, 4] 0.6
  let down := weight #[3, 4, 6] 0.8
  let indices := (Tensor.fromInt64Array #[2, 0, 1, 2]).reshapeChecked #[2, 2]
  let weights := (Tensor.fromFloatArray #[0.7, 0.3, 0.2, 0.8]).reshapeChecked #[2, 2] |>.forgetDType
  let actual := routedTextExpertsForward x gate down weights indices
  let mut expected := qwen35.zerosDynamic #[2, 4]
  let expertIds ← indices.toUInt64Array
  let scales ← weights.toFloatArray
  for token in [:2] do
    let row := x.sliceChecked 0 token.toUInt64 1 #[1, 4]
    let mut sum := qwen35.zerosDynamic #[1, 4]
    for slot in [:2] do
      let e := expertIds[token * 2 + slot]!
      let gu := (gate.sliceChecked 0 e 1 #[1, 12, 4]).reshapeChecked #[12, 4]
      let g := Tensor.linear row (gu.sliceChecked 0 0 6 #[6, 4])
      let u := Tensor.linear row (gu.sliceChecked 0 6 6 #[6, 4])
      let dw := (down.sliceChecked 0 e 1 #[1, 4, 6]).reshapeChecked #[4, 6]
      sum := sum + Tensor.Preserving.mulScalar (Tensor.linear (Tensor.Preserving.gelu g * u) dw) scales[token * 2 + slot]!
    expected := Tensor.sliceScatter expected 0 token.toUInt64 sum
  close actual expected 1e-6

private def vlCfg : VLConfig := {
  text_config := { cfg with use_bidirectional_attention := "vision" }
  vision_config := {
    hidden_size := 8, intermediate_size := 12, num_hidden_layers := 1
    num_attention_heads := 2, num_key_value_heads := 2, head_dim := 4
    patch_size := 2, pooling_kernel_size := 2, position_embedding_size := 8 }
  image_processor := { patch_size := 2, pooling_kernel_size := 2, max_soft_tokens := 4 }
  image_token_id := 15 }

private def visionLinear (outDim inDim : UInt64) (phase : Float) : Gemma4VisionLinear outDim inDim :=
  { weight := weight #[outDim, inDim] phase }

private def multimodalModel : Gemma4ForConditionalGeneration vlCfg := {
  language_model := textModel vlCfg.text_config
  embed_vision := { embedding_projection := weight #[8, 8] 0.4 }
  vision_tower := {
    patch_embedder := {
      input_proj := visionLinear 8 12 0.2
      position_embedding_table := weight #[2, 8, 8] 0.3 }
    blocks := #[{
      input_layernorm := norm 8, post_attention_layernorm := norm 8
      pre_feedforward_layernorm := norm 8, post_feedforward_layernorm := norm 8
      self_attn := {
        q_proj := visionLinear 8 8 0.5, k_proj := visionLinear 8 8 0.6
        v_proj := visionLinear 8 8 0.7, o_proj := visionLinear 8 8 0.8
        q_norm := norm 4, k_norm := norm 4 }
      mlp := {
        gate_proj := visionLinear 12 8 0.9, up_proj := visionLinear 12 8 1.0
        down_proj := visionLinear 8 12 1.1 } }]
    std_bias := some (weight #[8] 1.2)
    std_scale := some ((Tensor.ones #[8]).forgetDType) } }

@[test] def testGemma4VisionClippingAndFeatureBatching : IO Unit := do
  let clipped : Gemma4VisionLinear 2 2 := {
    weight := (Tensor.eye 2).forgetDType
    input_min := some (-1.0), input_max := some 1.0
    output_min := some (-0.5), output_max := some 0.5 }
  let input := (Tensor.fromFloatArray #[-2.0, 3.0]).reshapeChecked #[1, 2] |>.forgetDType
  LeanTest.assertEqual (← (clipped.forward2d input).toFloatArray) #[-0.5, 0.5]
  let a := weight #[4, 4, 12] 0.1
  let b := weight #[2, 4, 12] 0.2
  let some ⟨n, features⟩ ← multimodalModel.getImageFeaturesMany vlCfg #[⟨4, ⟨4, a⟩⟩, ⟨2, ⟨4, b⟩⟩]
    | throw <| IO.userError "missing image features"
  LeanTest.assertEqual n 6
  let ⟨total, repeated⟩ := Gemma4ForConditionalGeneration.repeatFeaturesForBatch (batch := 3) vlCfg features
  LeanTest.assertEqual total 18
  LeanTest.assertEqual repeated.runtimeShape #[18, 8]
  let values ← features.toFloatArray
  LeanTest.assertEqual (← repeated.toFloatArray) (values ++ values ++ values)
  LeanTest.assertTrue (values.all Float.isFinite)

@[test] def testGemma4MultimodalDTypesAndPlaceholderChecks : IO Unit := do
  for bf16 in #[false, true] do
    let m := if bf16 then TensorStruct.map (fun t => t.checkedReplacement t.toBFloat16.raw) multimodalModel else multimodalModel
    let features := m.getImageFeatures vlCfg (weight #[4, 4, 12] 0.1)
    LeanTest.assertEqual features.actualSpec.dtype (if bf16 then .BFloat16 else .Float32)
    let prompt := (Tensor.fromInt64Array #[2, 15, 15, 15, 15, 3, 4]).reshapeChecked #[1, 7]
    let out ← m.forwardWithImageFeatures vlCfg prompt features
    LeanTest.assertEqual out.runtimeShape #[1, 7, 16]
    for value in (← out.toFloatArray) do LeanTest.assertTrue value.isFinite
    let wrong := features.sliceChecked 0 0 3 #[3, 8]
    LeanTest.assertThrows (m.forwardWithImageFeatures vlCfg prompt wrong) (some "placeholder token count mismatch")
    let generated ← m.generate vlCfg prompt 2 .greedy #[] (some ⟨4, features⟩)
    LeanTest.assertEqual generated.1 9
    LeanTest.assertEqual generated.2.actualSpec.dtype .Int64

@[test] def testGemma4MediaIndexedPatchGrid : IO Unit := IO.FS.withTempDir fun dir => do
  let path := dir / "pixels.png"
  let bytes := ByteArray.mk #[137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 2, 0, 0, 0, 2, 8, 2, 0, 0, 0, 253, 212, 154, 115, 0, 0, 0, 16, 73, 68, 65, 84, 120, 156, 99, 112, 104, 56, 0, 68, 12, 16, 10, 0, 40, 14, 6, 1, 67, 58, 99, 218, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130]
  IO.FS.writeBinFile path bytes
  if System.Platform.isOSX then
    let ⟨rows, ⟨cols, grid⟩⟩ ← media.loadImagePatchGrid vlCfg path.toString
    LeanTest.assertTrue (rows > 0 && cols > 0 && rows % 2 == 0 && cols % 2 == 0)
    LeanTest.assertEqual grid.runtimeShape #[rows, cols, 12]
    LeanTest.assertEqual grid.actualSpec.dtype .Float32
    LeanTest.assertTrue ((← grid.toFloatArray).all Float.isFinite)
  else
    LeanTest.assertThrows (media.loadImagePatchGrid vlCfg path.toString) (some "requires macOS")

end Tests.Gemma4Typed
