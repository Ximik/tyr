import LeanTest
import Tyr.Model.Qwen3ASR.Model
import Tyr.Torch

namespace Tests.ASRAttentionEffects

open torch torch.qwen3asr

private def cfg : AudioEncoderConfig :=
  { numMelBins := 8, encoderLayers := 1, encoderAttentionHeads := 1,
    encoderFfnDim := 4, dModel := 2, attentionDropout := 1.0,
    outputDim := 2, downsampleHiddenSize := 2, nWindow := 8, nWindowInfer := 8 }

private def attention : AudioAttention (device := .CPU) cfg :=
  { qProjWeight := (Tensor.zeros #[2, 2])
    qProjBias := (Tensor.zeros #[2])
    kProjWeight := (Tensor.zeros #[2, 2])
    kProjBias := (Tensor.zeros #[2])
    vProjWeight := (Tensor.eye 2)
    vProjBias := (Tensor.zeros #[2])
    outProjWeight := (Tensor.eye 2)
    outProjBias := (Tensor.fromFloatArray #[0.25, -0.5]) }

private def input : Tensor { dtype := .Float32, shape := #[1, 2, 2], device := .CPU } :=
  ((Tensor.fromFloatArray #[2, 4, 6, 8]).reshape #[1, 2, 2])

private def assertClose (actual expected : Array Float) : IO Unit := do
  LeanTest.assertEqual actual.size expected.size
  for i in [:actual.size] do
    LeanTest.assertTrue (Float.abs (actual[i]! - expected[i]!) < 1e-5)
      s!"entry {i}: expected {expected[i]!}, got {actual[i]!}"

@[test] def testASRAttentionInferenceDisablesConfiguredDropout : IO Unit := do
  -- Uniform attention averages the two value rows; p=1 must not affect inference.
  assertClose (← (attention.forward input).toFloatArray) #[4.25, 5.5, 4.25, 5.5]

@[test] def testASRMaskedAttentionInferenceDisablesConfiguredDropout : IO Unit := do
  let mask := ((Tensor.fromInt64Array #[1, 1]).reshape #[1, 2])
  assertClose (← (attention.forward input (some mask)).toFloatArray) #[4.25, 5.5, 4.25, 5.5]

@[test] def testASRAttentionTrainingHonorsConfiguredDropout : IO Unit := do
  -- p=1 drops all attention probabilities, leaving only the output bias.
  assertClose (← (← attention.forwardIO input true).toFloatArray) #[0.25, -0.5, 0.25, -0.5]
  let mask := ((Tensor.fromInt64Array #[1, 1]).reshape #[1, 2])
  assertClose (← (← attention.forwardIO input true (some mask)).toFloatArray)
    #[0.25, -0.5, 0.25, -0.5]

@[test] def testASRAttentionEvaluationMatchesInference : IO Unit := do
  assertClose (← (← attention.forwardIO input false).toFloatArray)
    (← (attention.forwardInference input).toFloatArray)
  let mask := ((Tensor.fromInt64Array #[1, 1]).reshape #[1, 2])
  assertClose (← (← attention.forwardIO input false (some mask)).toFloatArray)
    (← (attention.forwardInference input (some mask)).toFloatArray)

@[test] def testASRAttentionTrainingPreservesParameterGradient : IO Unit := do
  let bias ← Tensor.parameter attention.outProjBias
  let model := { attention with outProjBias := bias }
  let out ← model.forwardIO input true
  let loss ← IO.ofExcept (Tensor.sumAll? out)
  Tensor.backwardLoss loss
  assertClose (← (← bias.gradOf).toFloatArray) #[2, 2]

@[test] def testASREncoderTrainingAndChunkedTraining : IO Unit := do
  let model ← AudioEncoder.init cfg
  -- Zeroing the output projection removes the attention contribution just as p=1 does.
  let reference := { model with layers := model.layers.map fun (layer : AudioEncoderLayer (device := .CPU) cfg) =>
    { layer with selfAttn := { layer.selfAttn with
        outProjWeight := Tensor.zerosLike layer.selfAttn.outProjWeight } } }
  let features := (Tensor.ones #[1, 8, 32])
  assertClose (← (← model.forwardIO features true).toFloatArray)
    (← (reference.forwardInference features).toFloatArray)
  assertClose (← (← model.forwardVarLenIO features #[24] true).toFloatArray)
    (← (reference.forwardVarLenInference features #[24]).toFloatArray)
  assertClose (← (← model.forwardIO features false).toFloatArray)
    (← (model.forwardInference features).toFloatArray)
  assertClose (← (← model.forwardVarLenIO features #[24] false).toFloatArray)
    (← (model.forwardVarLenInference features #[24]).toFloatArray)

private def parentCfg (dropout : Float) : Qwen3ASRConfig :=
  { thinkerConfig := {
      audioConfig := { cfg with attentionDropout := dropout }
      textConfig := { vocabSize := 8, hiddenSize := 2, intermediateSize := 4, numHiddenLayers := 0, numAttentionHeads := 1, numKeyValueHeads := 1, headDim := 2 }
      audioTokenId := 7 } }

private def expectDropoutError (action : IO α) : IO Unit := do
  let message ← try
    let _ ← action
    pure none
  catch e => pure (some e.toString)
  match message with
  | none => throw <| IO.userError "expected invalid training dropout to fail"
  | some message => LeanTest.assertTrue ((message.splitOn "dropout").length > 1) message

@[test] def testASRParentFeatureTrainingPropagatesDropout : IO Unit := do
  -- Invalid configuration must reach an IO attention boundary instead of pure inference.
  let model ← Qwen3ASRForConditionalGeneration.init (parentCfg 2.0)
  let features := (Tensor.ones #[1, 8, 32])
  let textIds := (Tensor.fromInt64Array #[1]).reshape #[1, 1]
  let placeholders := (Tensor.fromInt64Array #[7, 7, 7, 7]).reshape #[1, 4]
  let featureMask := (Tensor.fullInt #[1, 32] 1)
  expectDropoutError (model.thinker.encodeAudioIO features true)
  expectDropoutError (model.thinker.encodeAudioVarLenIO features #[32] true)
  expectDropoutError (model.forwardFromMelIO features textIds true)
  expectDropoutError (model.forward placeholders (some features) (training := true))
  expectDropoutError (model.forwardWithAux placeholders (some features) (some featureMask)
    (training := true))
  -- Effectful evaluation still validates the configured probability before disabling dropout.
  expectDropoutError (model.thinker.encodeAudioIO features false)
  LeanTest.assertEqual (← (model.thinker.encodeAudio features).toFloatArray).size 8

@[test] def testASRParentTrainingFlagControlsRNG : IO Unit := do
  let model ← Qwen3ASRForConditionalGeneration.init (parentCfg 0.5)
  let features := (Tensor.ones #[1, 8, 32])
  let textIds := (Tensor.fromInt64Array #[1]).reshape #[1, 1]
  let placeholders := (Tensor.fromInt64Array #[7, 7, 7, 7]).reshape #[1, 4]
  let featureMask := Tensor.fullInt #[1, 32] 1
  -- Valid p distinguishes true from false; invalid p is rejected in both modes.
  -- Model initialization finishes before seeding, and these forwards have no
  -- random operation other than attention dropout.
  let entrypoints : Array (String × (Bool → IO Unit)) := #[
    ("encodeAudioIO", fun training =>
      (model.thinker.encodeAudioIO features training).map (fun _ => ())),
    ("encodeAudioVarLenIO", fun training =>
      (model.thinker.encodeAudioVarLenIO features #[32] training).map (fun _ => ())),
    ("thinker.forwardFromMelIO", fun training =>
      (model.thinker.forwardFromMelIO features textIds training).map (fun _ => ())),
    ("forwardFromMelIO", fun training =>
      (model.forwardFromMelIO features textIds training).map (fun _ => ())),
    ("forward", fun training =>
      (model.forward placeholders (some features) (training := training)).map (fun _ => ())),
    ("forward varlen", fun training =>
      (model.forward placeholders (some features) (some featureMask)
        (training := training)).map (fun _ => ())),
    ("forwardWithAux", fun training =>
      (model.forwardWithAux placeholders (some features)
        (training := training)).map (fun _ => ())),
    ("forwardWithAux varlen", fun training =>
      (model.forwardWithAux placeholders (some features) (some featureMask)
        (training := training)).map (fun _ => ())) ]
  for (name, forward) in entrypoints do
    manualSeed 314159
    let expectedNext ← Tensor.randn #[8]
    let expectedValues ← expectedNext.toFloatArray
    manualSeed 314159
    forward false
    let afterEvaluation ← Tensor.randn #[8]
    LeanTest.assertEqual (← afterEvaluation.toFloatArray) expectedValues
      (some s!"{name}: evaluation must not consume attention RNG")
    manualSeed 314159
    forward true
    let afterTraining ← Tensor.randn #[8]
    LeanTest.assertTrue ((← afterTraining.toFloatArray) != expectedValues)
      s!"{name}: training must execute attention dropout"

end Tests.ASRAttentionEffects
