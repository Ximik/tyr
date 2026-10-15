import Tyr.Typed
import Tyr.Model.SileroVAD
import Tyr.SafeTensors.Exact
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

private def expectSileroLoadError (path detail : String) : IO Unit := do
  let failure ← try
    let _ ← SileroVAD.load path
    pure none
  catch error => pure (some error.toString)
  let some message := failure | throw (IO.userError "Expected SileroVAD loading to fail")
  for fragment in #["SileroVAD", "stft_conv.weight", detail] do
    LeanTest.assertTrue (message.containsSubstr fragment)
      s!"Loading error must retain '{fragment}', got: {message}"

@[test]
def testSileroVADLoadingPreservesFileErrors : IO Unit :=
    IO.FS.withTempDir fun dir => do
  expectSileroLoadError (dir / "missing.safetensors").toString "Failed to open"
  let path := (dir / "missing-weight.safetensors").toString
  (Tensor.zeros #[1]).saveSafeTensor path "other.weight"
  expectSileroLoadError path "Tensor not found"

@[test]
def testSileroVADLoadingPreservesLayoutErrors : IO Unit :=
    IO.FS.withTempDir fun dir => do
  let path := (dir / "wrong-layout.safetensors").toString
  -- Same element count as [258, 1, 256]: loading must not silently reshape it.
  (Tensor.zeros #[258, 256, 1]).saveSafeTensor path "stft_conv.weight"
  expectSileroLoadError path "Expected shape"

/-- Small deterministic full-model checkpoint, using the same in-test
SafeTensors byte fixture pattern as the other model loading regressions. -/
private def writeSileroCheckpoint (path : System.FilePath) (bf16 : Bool) : IO Unit := do
  let entries : Array (String × Array Nat) := #[
    ("stft_conv.weight", #[258, 1, 256]),
    ("conv1.weight", #[128, 129, 3]), ("conv1.bias", #[128]),
    ("conv2.weight", #[64, 128, 3]), ("conv2.bias", #[64]),
    ("conv3.weight", #[64, 64, 3]), ("conv3.bias", #[64]),
    ("conv4.weight", #[128, 64, 3]), ("conv4.bias", #[128]),
    ("lstm_cell.weight_ih", #[512, 128]), ("lstm_cell.weight_hh", #[512, 128]),
    ("lstm_cell.bias_ih", #[512]), ("lstm_cell.bias_hh", #[512]),
    ("final_conv.weight", #[1, 128, 1]), ("final_conv.bias", #[1])]
  let mut fields : List (String × Lean.Json) := []
  let mut data := ByteArray.empty
  for (name, shape) in entries do
    let start := data.size
    -- Exact BF16/Float32 bit patterns: conv1 bias = 0.25, final bias = 0.5;
    -- other weights are zero, so inference has the known sigmoid(0.5) result.
    let high : UInt8 := if name == "conv1.bias" then 62 else if name == "final_conv.bias" then 63 else 0
    let low : UInt8 := if name == "conv1.bias" then 128 else 0
    let value := if bf16 then ByteArray.mk #[low, high] else ByteArray.mk #[0, 0, low, high]
    for _ in [:shape.foldl (· * ·) 1] do data := data ++ value
    fields := fields ++ [(name, Lean.Json.mkObj [
      ("dtype", Lean.toJson (if bf16 then "BF16" else "F32")),
      ("shape", Lean.toJson shape), ("data_offsets", Lean.toJson #[start, data.size])])]
  let header := (Lean.Json.mkObj fields).compress.toUTF8
  let mut headerSize := ByteArray.empty
  for i in [:8] do headerSize := headerSize.push ((header.size.toUInt64 >>> (i * 8).toUInt64).toUInt8)
  IO.FS.writeBinFile path (headerSize ++ header ++ data)

@[test]
def testSileroVADExactLoadingRetainsFloat32InferencePolicy : IO Unit := do
  for bf16 in #[false, true] do
    IO.FS.withTempDir fun dir => do
      let path := dir / "model.safetensors"
      writeSileroCheckpoint path bf16
      let stored ← safetensors.loadTensorExact path.toString "conv1.bias" #[128]
      LeanTest.assertEqual stored.1.val (if bf16 then .BFloat16 else .Float32)
      LeanTest.assertEqual (← stored.2.toFloatArray) (Array.replicate 128 0.25)
      let model ← SileroVAD.load path.toString
      let valid := TensorStruct.fold (fun {σ} (tensor : Tensor σ) valid =>
        valid && tensor.raw.dtype == .Float32 && tensor.runtimeDevice == .CPU &&
          tensor.runtimeShape == σ.shape && !tensor.requiresGrad) true model
      LeanTest.assertTrue valid "All loaded model fields must be exact Float32 CPU inference tensors"
      LeanTest.assertEqual (← model.conv1Bias.toFloatArray) (Array.replicate 128 0.25)
      LeanTest.assertEqual (← model.finalConvBias.toFloatArray) #[0.5]
      LeanTest.assertEqual stored.2.raw.dtype (if bf16 then .BFloat16 else .Float32)
      let (probability, _) ← (SileroVADRuntime.init model).step (Array.replicate 512 0.0)
      let expected := 1.0 / (1.0 + Float.exp (-0.5))
      LeanTest.assertTrue ((probability - expected).abs < 1e-6)
        "Explicit Float32 conversion must preserve the deterministic inference result"
