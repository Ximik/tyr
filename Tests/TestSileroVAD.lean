import Tyr.Typed
import Tyr.Model.SileroVAD
import LeanTest

open torch
open torch.silerovad

@[test]
def testSileroVADReflectPadRight : IO Unit := do
  let xs : Array Float := #[1.0, 2.0, 3.0, 4.0]
  let got := reflectPadRight xs 3
  let expected : Array Float := #[1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]
  LeanTest.assertEqual got expected "reflectPadRight should match PyTorch right-reflect behavior"

@[test]
def testSileroVADTimestampSegmentation : IO Unit := do
  let probs : Array Float := #[0.1, 0.8, 0.9, 0.8, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1]
  let cfg : TimestampConfig := {
    threshold := 0.5
    samplingRate := 16000
    minSpeechDurationMs := 0
    minSilenceDurationMs := 0
    speechPadMs := 0
  }
  let tss ← timestampsFromProbabilities probs (audioLengthSamples := 5120) cfg
  let expected : Array SpeechTimestamp := #[
    { start := 512, endPos := 2048 },
    { start := 3072, endPos := 4096 }
  ]
  LeanTest.assertTrue (tss == expected) "timestampsFromProbabilities should produce expected speech segments"

@[test]
def testSileroVADRuntimeStepAndAudioForward : IO Unit := do
  let model ← SileroVAD.init
  let rt0 := SileroVADRuntime.init model

  let chunk := Array.replicate 512 0.0
  let (p0, rt1) ← rt0.step chunk
  LeanTest.assertTrue (Float.isFinite p0) "runtime step probability should be finite"
  LeanTest.assertTrue (p0 >= 0.0 && p0 <= 1.0) "runtime step probability should be in [0,1]"

  let audio := Array.replicate 1000 0.01
  let (probs, _rt2) ← rt1.audioForward audio
  LeanTest.assertEqual probs.size 2 "audioForward should return ceil(len/512) chunk probabilities"

@[test]
def testSileroVADLoadPretrainedIfAvailable : IO Unit := do
  let path := "../silero-vad/src/silero_vad/data/silero_vad_16k.safetensors"
  let hasWeights ← data.fileExists path
  if hasWeights then
    let model ← SileroVAD.load path
    let rt0 := SileroVADRuntime.init model
    let chunk := Array.replicate 512 0.0
    let (p, _rt1) ← rt0.step chunk
    LeanTest.assertTrue (Float.isFinite p) "pretrained SileroVAD inference should produce finite output"
    LeanTest.assertTrue (p >= 0.0 && p <= 1.0) "pretrained SileroVAD output should be a probability"
  else
    pure ()
