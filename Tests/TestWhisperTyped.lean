import Tyr.Model.Whisper
import Tyr.Checkpoint
import LeanTest

open torch torch.whisper
namespace Tests.WhisperTyped

private def cfg : WhisperConfig := {
  numMelBins := 4, vocabSize := 16, dModel := 8,
  encoderLayers := 1, encoderAttentionHeads := 2, encoderFfnDim := 12,
  decoderLayers := 2, decoderAttentionHeads := 2, decoderFfnDim := 12,
  maxSourcePositions := 4, maxTargetPositions := 8,
  padTokenId := 0, bosTokenId := 1, eosTokenId := 2, decoderStartTokenId := 1 }

private def features : Tensor { shape := #[1, 4, 5], dtype := .Float32, device := .CPU } :=
  (Tensor.fromFloatArray (Array.ofFn (n := 20) fun i => Float.sin (i.val.toFloat * 0.3))).reshape #[1, 4, 5]
private def tokens : Tensor { shape := #[1, 4], dtype := .Int64, device := .CPU } := (Tensor.fromInt64Array #[1, 3, 5, 7]).reshape #[1, 4]

private def close {σ τ : StaticSpec} (x : Tensor σ) (y : Tensor τ) (tol : Float := 1e-4) : IO Unit := do
  LeanTest.assertEqual x.runtimeShape y.runtimeShape
  let xs ← x.toFloatArray
  let ys ← y.toFloatArray
  for i in [:xs.size] do
    LeanTest.assertTrue (xs[i]!.isFinite && (xs[i]! - ys[i]!).abs <= tol)
      s!"Whisper mismatch at {i}: {xs[i]!} vs {ys[i]!}, tolerance {tol}"

@[test] def testWhisperCachedDecodeDTypesAndParity : IO Unit := do
  for bf16 in #[false, true] do
    let selected : RealFloatingDType :=
      if bf16 then ⟨.BFloat16, by decide⟩ else ⟨.Float32, by decide⟩
    let dtype := selected.val
    let support ← dtype.acquireConstantSupport .CPU
    let _ := support.down
    let model ← WhisperForConditionalGeneration.init cfg (dtype := dtype)
    let model := TensorStruct.detach model
    let encoded ← model.encode features (by decide)
    LeanTest.assertEqual encoded.runtimeShape #[1, 3, 8]
    LeanTest.assertEqual encoded.actualSpec.dtype (if bf16 then .BFloat16 else .Float32)
    let expected ← model.decode tokens encoded
    let cross ← model.precomputeCrossCaches encoded
    let initial := model.initLayerKVCaches (batch := 1) 4 .CPU
    let mut caches := initial
    for pos in [:4] do
      let tok := tokens.sliceChecked 1 pos 1 #[1, 1]
      let (actual, next) ← model.decodeStepWithCache (encSeq := 3) tok pos cross caches
      close actual ((expected.sliceChecked 1 pos 1 #[1, 1, 16]).reshape #[1, 16]) (if bf16 then 0.08 else 1e-4)
      LeanTest.assertEqual actual.actualSpec.dtype expected.actualSpec.dtype
      caches := next
    let some untouched := initial[0]? | throw (IO.userError "missing initial cache")
    LeanTest.assertEqual untouched.seq 0
    let stored ← untouched.kStore.toFloatArray
    LeanTest.assertTrue (stored.all (· == 0)) "incremental updates must preserve the input cache snapshot"
    let (again, _) ← model.decodeStepWithCache (encSeq := 3) (tokens.sliceChecked 1 0 1 #[1, 1]) 0 cross initial
    close again ((expected.sliceChecked 1 0 1 #[1, 1, 16]).reshape #[1, 16]) (if bf16 then 0.08 else 1e-4)

@[test] def testWhisperCacheCapacityAndPositionErrors : IO Unit := do
  let model ← WhisperForConditionalGeneration.init cfg
  let encoded ← model.encode features (by decide)
  let cross ← model.precomputeCrossCaches encoded
  let caches := model.initLayerKVCaches (batch := 1) 1 .CPU
  let tok := tokens.sliceChecked 1 0 1 #[1, 1]
  let (_, filled) ← model.decodeStepWithCache (encSeq := 3) tok 0 cross caches
  let fullRejected ← try
    let _ ← model.decodeStepWithCache (encSeq := 3) tok 1 cross filled
    pure false
  catch e => pure ((toString e).contains "capacity")
  LeanTest.assertTrue fullRejected
  let positionRejected ← try
    let _ ← model.decodeStepWithCache (encSeq := 3) tok 1 cross caches
    pure false
  catch e => pure ((toString e).contains "position")
  LeanTest.assertTrue positionRejected
  let empty := model.initLayerKVCaches (batch := 1) 0 .CPU
  let emptyRejected ← try
    let _ ← model.decodeStepWithCache (encSeq := 3) tok 0 cross empty
    pure false
  catch e => pure ((toString e).contains "capacity")
  LeanTest.assertTrue emptyRejected

@[test] def testWhisperLengthAndHeadValidation : IO Unit := do
  let model ← WhisperForConditionalGeneration.init cfg
  let tooLong ← try
    let _ ← model.encode ((Tensor.zeros #[1, 4, 10])) (by decide)
    pure false
  catch e => pure ((toString e).contains "max_source_positions")
  LeanTest.assertTrue tooLong
  let encoded ← model.encode features (by decide)
  let decoderTooLong ← try
    let _ ← model.decode (Tensor.fullInt #[1, 9] 1) encoded
    pure false
  catch e => pure ((toString e).contains "max_target_positions")
  LeanTest.assertTrue decoderTooLong

@[noinline, nospecialize]
private def roundTrip {α : Type} [TensorStruct α] (value : α) (dir : String) : IO α := do
  checkpoint.saveParams value dir
  checkpoint.loadParams value dir

@[test] def testWhisperIndexedCheckpoint : IO Unit := IO.FS.withTempDir fun dir => do
  let model ← WhisperForConditionalGeneration.init cfg
  let restored ← roundTrip model dir.toString
  let encoded ← model.encode features (by decide)
  let other ← restored.encode features (by decide)
  close encoded other 0
  close (← model.decode tokens encoded) (← restored.decode tokens other) 0

@[test] def testWhisperWaveformGreedyBeamAndSampling : IO Unit := do
  let audioCfg := { cfg with maxSourcePositions := 16, maxTargetPositions := 16, eosTokenId := 15 }
  let model ← WhisperForConditionalGeneration.init audioCfg
  let model := TensorStruct.detach model
  let norm : LayerNorm 8 .Float32 := {
    weight := zerosFor #[8], bias := (Tensor.ones #[8]) }
  let projection := Tensor.sliceScatter (zerosFor #[16, 8]) 0 0 ((Tensor.ones #[1, 8]))
  let model := { model with model := { model.model with decoderLayerNorm := norm }, projOut := projection }
  let tok : tokenizer.qwen3.QwenTokenizer := { (default : tokenizer.qwen3.QwenTokenizer) with
    vocabSize := 16, idToToken := Array.replicate 16 "a" }
  let pre : qwen3asr.PreprocessorConfig := {
    featureSize := 4, samplingRate := 16000, hopLength := 64, nFft := 64,
    nSamples := 2048, nbMaxFrames := 32, chunkLength := 0 }
  let waveform := Array.ofFn (n := 1024) fun i => 0.1 * Float.sin (i.val.toFloat * 0.2)
  for (beam, temperature) in #[(1, 0.0), (2, 0.0), (1, 0.7)] do
    let result ← transcribeWaveform16k model tok pre rfl waveform "english" 2 true {
      beamSize := beam, temperature, noFallback := true, conditionOnPreviousText := false,
      chunkOverlapSeconds := 0, noSpeechThreshold := 1.0 }
    LeanTest.assertEqual result.language "en"
    LeanTest.assertTrue (result.tokenIds.size <= 2)
    LeanTest.assertTrue (result.tokenIds.all (· < 16))
    if beam == 1 && temperature == 0 then
      LeanTest.assertEqual result.tokenIds #[0, 0]
      LeanTest.assertEqual result.text "aa"

private def writeWhisperCheckpoint (dir : System.FilePath) (bf16 : Bool) (badBias : Bool := false)
    (mismatchedDType : String := "") : IO Unit := do
  let mut entries : Array (String × Array Nat) := #[
    ("model.encoder.conv1.weight", #[8, 4, 3]), ("model.encoder.conv1.bias", #[8]),
    ("model.encoder.conv2.weight", #[8, 8, 3]), ("model.encoder.conv2.bias", #[8]),
    ("model.encoder.embed_positions.weight", #[4, 8]),
    ("model.decoder.embed_tokens.weight", #[16, 8]),
    ("model.decoder.embed_positions.weight", #[8, 8]) ]
  for part in #["encoder", "decoder"] do
    entries := entries ++ #[(s!"model.{part}.layer_norm.weight", #[8]), (s!"model.{part}.layer_norm.bias", #[8])]
    let layers := if part == "encoder" then 1 else 2
    for i in [:layers] do
      let stem := s!"model.{part}.layers.{i}"
      for norm in #["self_attn_layer_norm", "final_layer_norm"] ++ (if part == "decoder" then #["encoder_attn_layer_norm"] else #[]) do
        entries := entries ++ #[(s!"{stem}.{norm}.weight", #[8]), (s!"{stem}.{norm}.bias", #[8])]
      entries := entries ++ #[(s!"{stem}.fc1.weight", #[12, 8]), (s!"{stem}.fc1.bias", #[12]),
        (s!"{stem}.fc2.weight", #[8, 12]), (s!"{stem}.fc2.bias", #[8])]
      for attn in #["self_attn"] ++ (if part == "decoder" then #["encoder_attn"] else #[]) do
        for projection in #["q_proj", "k_proj", "v_proj", "out_proj"] do
          entries := entries.push (s!"{stem}.{attn}.{projection}.weight", #[8, 8])
          if projection != "k_proj" then entries := entries.push (s!"{stem}.{attn}.{projection}.bias", #[8])
  if badBias then
    entries := entries.push ("model.decoder.layers.0.self_attn.k_proj.bias", #[2, 4])
  let mut fields : List (String × Lean.Json) := []
  let mut index : List (String × Lean.Json) := []
  let mut data := ByteArray.empty
  for (name, shape) in entries do
    let entryBf16 := if name == mismatchedDType then !bf16 else bf16
    let start := data.size
    for _ in [:shape.foldl (· * ·) 1] do
      data := data ++ (if entryBf16 then ByteArray.mk #[0, 62] else ByteArray.mk #[0, 0, 0, 62])
    fields := fields ++ [(name, Lean.Json.mkObj [
      ("dtype", Lean.toJson (if entryBf16 then "BF16" else "F32")),
      ("shape", Lean.toJson shape), ("data_offsets", Lean.toJson #[start, data.size])])]
    index := index ++ [(name, Lean.toJson "model.safetensors")]
  let header := (Lean.Json.mkObj fields).compress.toUTF8
  let mut headerSize := ByteArray.empty
  for i in [:8] do headerSize := headerSize.push ((header.size >>> (i * 8)).toUInt8)
  IO.FS.writeBinFile (dir / "model.safetensors") (headerSize ++ header ++ data)
  IO.FS.writeFile (dir / "model.safetensors.index.json") (Lean.Json.mkObj [("weight_map", Lean.Json.mkObj index)]).compress

@[test] def testWhisperShardedCheckpointDTypesAndOptionalWeights : IO Unit := do
  for bf16 in #[false, true] do
    IO.FS.withTempDir fun dir => do
      writeWhisperCheckpoint dir bf16
      let ⟨storedDType, model⟩ ← WhisperForConditionalGeneration.loadShardedAt dir.toString cfg (device := .CPU)
      let dtype := if bf16 then DType.BFloat16 else .Float32
      LeanTest.assertEqual storedDType.val dtype
      LeanTest.assertEqual model.projOut.actualSpec.dtype dtype
      close model.projOut model.model.decoderTokenEmbedding 0
      let some layer := model.model.decoderLayers[0]? | throw (IO.userError "missing decoder layer")
      LeanTest.assertEqual layer.selfAttn.kProjBias.actualSpec.dtype dtype
      close layer.selfAttn.kProjBias (Tensor.zeros #[8]) 0
      let encoded ← model.encode features (by decide)
      let logits ← model.decode tokens encoded
      LeanTest.assertEqual logits.actualSpec.dtype dtype
      for v in (← logits.toFloatArray) do LeanTest.assertTrue v.isFinite

  IO.FS.withTempDir fun dir => do
    writeWhisperCheckpoint dir false true
    let rejected ← try
      let _ ← WhisperForConditionalGeneration.loadSharded dir.toString cfg
      pure false
    catch _ => pure true
    LeanTest.assertTrue rejected "malformed optional bias must not be treated as absent"

@[test] def testWhisperRejectsInconsistentStoredDTypes : IO Unit := do
  for name in #["model.decoder.embed_tokens.weight", "model.encoder.layer_norm.weight",
      "model.decoder.layers.0.self_attn.q_proj.bias"] do
    IO.FS.withTempDir fun dir => do
      writeWhisperCheckpoint dir true (mismatchedDType := name)
      let rejected ← try
        let _ ← WhisperForConditionalGeneration.loadShardedAt dir.toString cfg (device := .CPU)
        pure false
      catch _ => pure true
      LeanTest.assertTrue rejected s!"inconsistent stored dtype must be rejected: {name}"

end Tests.WhisperTyped
