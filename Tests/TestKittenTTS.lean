import Tyr.Model.KittenTTS.Model
import Tyr.Checkpoint
import LeanTest

open torch
open torch.kittentts

private def tinyCfg : KittenTTSConfig :=
  {
    hiddenDim := 8
    maxConvDim := 8
    maxDur := 4
    nLayer := 1
    nMels := 8
    nToken := 32
    styleDim := 4
    textEncoderKernelSize := 3
    asrResDim := 4
    decoderOutDim := 8
    plbert := {
      numHiddenLayers := 1
      numAttentionHeads := 2
      hiddenSize := 8
      intermediateSize := 16
      maxPositionEmbeddings := 32
      embeddingSize := 4
      innerGroupNum := 1
      numHiddenGroups := 1
      typeVocabSize := 2
    }
    generator := {
      resblockKernelSizes := #[3]
      upsampleRates := #[2]
      upsampleInitialChannel := 8
      resblockDilationSizes := #[#[1, 3, 5]]
      upsampleKernelSizes := #[4]
      genIstftNFft := 8
      genIstftHopSize := 2
      harmonicCount := 2
    }
  }

@[test]
def testKittenTTSInitAndForward : IO Unit := do
  let cfg := tinyCfg
  let model ← Model.init cfg
  let inputIds : DTensor #[1, 5] .Int64 ← Tensor.randint 0 cfg.nToken.toInt64 #[1, 5]
  let refStyle : DTensor #[1, KittenTTSConfig.fullStyleDim cfg] .Float32 ← Tensor.randn #[1, KittenTTSConfig.fullStyleDim cfg]
  torch.manualSeed 12345
  let out ← model.synthesizeIds inputIds refStyle
  torch.manualSeed 12345
  let debugOut ← model.debugSynthesizeIds inputIds refStyle

  LeanTest.assertEqual out.predDurations.size 5 "predicted durations should align with the input token count"
  LeanTest.assertTrue (out.alignmentFrames >= 5) "each token should contribute at least one frame"
  LeanTest.assertEqual debugOut.predDurations out.predDurations
    "debug synthesis should preserve duration predictions"
  LeanTest.assertEqual debugOut.alignmentFrames out.alignmentFrames
    "debug synthesis should preserve alignment length"
  let audioShape := out.audio.2.runtimeShape
  LeanTest.assertEqual (audioShape.getD 0 0) 1 "audio batch dimension should be 1"
  LeanTest.assertEqual (audioShape.getD 1 0) 1 "audio channel dimension should be 1"
  LeanTest.assertEqual
    (audioShape.getD 2 0)
    (GeneratorConfig.waveSamples cfg.generator (2 * out.alignmentFrames))
    "audio sample length should match the full iSTFT vocoder expansion"

  let audioVals ← Tensor.toFloatArray out.audio.2
  let debugAudioVals ← Tensor.toFloatArray debugOut.audio.2
  LeanTest.assertTrue (!audioVals.isEmpty) "audio output should contain samples"
  LeanTest.assertEqual debugAudioVals.size audioVals.size
    "debug synthesis audio should match the regular forward path sample count"
  for v in audioVals do
    LeanTest.assertTrue (Float.isFinite v) "audio values should be finite"
  let mut maxDiff := 0.0
  let mut sumDiff := 0.0
  let mut maxIdx := 0
  for i in [:audioVals.size] do
    let dv := debugAudioVals[i]!
    let v := audioVals[i]!
    let diff := Float.abs (dv - v)
    sumDiff := sumDiff + diff
    if diff > maxDiff then
      maxDiff := diff
      maxIdx := i
  let avgDiff := sumDiff / audioVals.size.toFloat
  LeanTest.assertTrue
    (maxDiff <= 1e-6)
    s!"debug synthesis audio should stay numerically aligned with the normal forward path (maxDiff={maxDiff}, avgDiff={avgDiff}, maxIdx={maxIdx})"

-- Keep this generic to avoid expanding the entire model traversal into test C code.
@[noinline, nospecialize]
private def roundTripParams {α : Type} [TensorStruct α] (model template : α) (dir : String) : IO α := do
  checkpoint.saveParams model dir
  checkpoint.loadParams template dir

@[test]
def testKittenTTSIndexedCheckpointRoundTrip : IO Unit := IO.FS.withTempDir fun dir => do
  let model ← Model.init tinyCfg
  let template ← Model.init tinyCfg
  let loaded ← roundTripParams model template dir.toString
  let inputIds := (Tensor.fromInt64Array #[1, 2, 3]).reshapeChecked #[1, 3]
  let style : DTensor #[1, KittenTTSConfig.fullStyleDim tinyCfg] .Float32 := Tensor.zeros _
  torch.manualSeed 42
  let expected ← model.synthesizeIds inputIds style
  torch.manualSeed 42
  let actual ← loaded.synthesizeIds inputIds style
  LeanTest.assertEqual actual.predDurations expected.predDurations
  LeanTest.assertEqual actual.audio.1 expected.audio.1
  LeanTest.assertEqual (← actual.audio.2.toFloatArray) (← expected.audio.2.toFloatArray)
