import Tyr.Typed.Neural
import Tyr.Module.RMSNorm
import LeanTest

open torch

@[test] def testTypedAttentionValuesAndMetadata : IO Unit := do
  let σ : StaticSpec := ⟨#[1, 1, 2, 2], .Float64, .CPU⟩
  let q := Tensor.zerosFor σ
  let values := ((Tensor.fromFloatArray #[2, 4, 6, 8]).reshape #[1, 1, 2, 2]).castDType .Float64
  let v : Tensor σ ← match Tensor.ofRaw values.raw with
    | .ok t => pure t
    | .error message => throw <| IO.userError message
  let full := Tensor.attention q q v false
  let causal := Tensor.attention q q v true
  for output in #[full, causal] do
    match output.validate with
    | .ok () => pure ()
    | .error message => LeanTest.fail message
  LeanTest.assertEqual (← full.toFloatArray) #[4, 6, 4, 6]
  LeanTest.assertEqual (← causal.toFloatArray) #[2, 4, 4, 6]

@[test] def testTypedNeuralGatherAndMasks : IO Unit := do
  let x := (Tensor.fromFloatArray #[1, 2, 3, 4, 5, 6]).reshapeChecked #[2, 3]
  let indices := (Tensor.fromInt64Array #[2, 0, 1, 1]).reshapeChecked #[2, 2]
  let selected := Tensor.gather x (-1) indices
  LeanTest.assertEqual (← selected.toFloatArray) #[3, 1, 5, 5]
    "Signed gather dimension preserves native negative-axis semantics"
  let mask := (Tensor.fromInt64Array #[1, 0, 1, 0, 0, 1]).reshapeChecked #[2, 3] |>.eqScalar 1
  let packed := Tensor.maskedSelect x mask
  LeanTest.assertEqual packed.1 3
  LeanTest.assertEqual (← packed.2.toFloatArray) #[1, 3, 6]
  let replaced := Tensor.maskedScatter x mask (Tensor.fromFloatArray #[10, 30, 60])
  LeanTest.assertEqual (← replaced.toFloatArray) #[10, 2, 30, 4, 5, 60]
  let empty := Tensor.maskedSelect x (Tensor.fullInt #[2, 3] 0 |>.eqScalar 1)
  LeanTest.assertEqual empty.1 0
  LeanTest.assertEqual empty.2.runtimeShape #[0]

@[test] def testTypedNeuralConvAndExistentialConcat : IO Unit := do
  let x := (Tensor.fromFloatArray #[1, 2, 3, 4, 5, 6, 7, 8, 9]).reshapeChecked #[1, 1, 3, 3]
  let kernel := Tensor.ones #[1, 1, 2, 2]
  let result := Tensor.conv2dBias x kernel (Tensor.full #[1] 0.5)
  LeanTest.assertEqual result.runtimeShape #[1, 1, 2, 2]
  LeanTest.assertEqual (← result.toFloatArray) #[12.5, 16.5, 24.5, 28.5]
  let joined := Tensor.catManyChecked
    #[(Tensor.fromFloatArray #[1, 2]).packShape, (Tensor.fromFloatArray #[3]).packShape] 0 #[3]
  LeanTest.assertEqual (← joined.toFloatArray) #[1, 2, 3]

@[test] def testTypedRandintSignedBounds : IO Unit := do
  let sampled ← Tensor.randint (-7) (-6) #[16]
  LeanTest.assertEqual sampled.actualSpec.dtype DType.Int64
  LeanTest.assertEqual (← sampled.toFloatArray) (Array.replicate 16 (-7.0))
    "Lean Int64 bounds cross the C ABI with their signed bit pattern intact"

@[test] def testTypedFlashAttentionCheckedErrors : IO Unit := do
  let q := Tensor.zeros #[1, 1, 2, 2]
  let v := (Tensor.fromFloatArray #[2, 4, 6, 8]).reshape #[1, 1, 2, 2]
  let result ← match Tensor.flashAttention? q q v with
    | .ok result => pure result
    | .error message => LeanTest.fail message
  LeanTest.assertEqual (← result.toFloatArray) #[4, 6, 4, 6]
  LeanTest.assertTrue (result.actualSpec == q.actualSpec)
  for dropout in #[-0.5, 1.5] do
    let rejected ← try
      let _ ← Tensor.flashAttentionIO q q v dropout true
      pure false
    catch error =>
      LeanTest.assertTrue (!error.toString.isEmpty)
      pure true
    LeanTest.assertTrue rejected "Invalid FlashAttention dropout must return IO.error"
  let ints := q.castDType .Int64
  match Tensor.flashAttention? ints ints ints with
  | .ok _ => LeanTest.fail "Integer FlashAttention must return an error"
  | .error message => LeanTest.assertTrue (!message.isEmpty)

@[test] def testTypedRandintRecoverableErrors : IO Unit := do
  for (low, high, device) in #[(3, 3, Device.CPU), (4, 3, Device.CPU)] do
    let rejected ← try
        let _ ← Tensor.randint low high #[2] device
        pure false
      catch _ => pure true
    LeanTest.assertTrue rejected "Invalid randint arguments must return IO.error"

@[test] def testTypedTransposeRecoverableErrors : IO Unit := do
  let matrix := Tensor.ones #[2, 3]
  for (dim0, dim1) in #[(0, 9), (9, 0), (0, 18446744073709551615)] do
    match matrix.transpose? dim0 dim1 with
    | .ok _ => LeanTest.fail "Invalid transpose axes must return Except.error"
    | .error message => LeanTest.assertTrue (!message.isEmpty)
  match (Tensor.full #[] 5).transpose? 0 0 with
  | .ok scalar => LeanTest.assertEqual (← scalar.toFloatArray) #[5]
  | .error message => LeanTest.fail message

@[test] def testTypedRMSNormBFloat16Rounding : IO Unit := do
  let values := (Tensor.fromFloatArray
    #[0.17, -0.41, 0.83, 1.27, 0.9, 0.5, -0.7, 1.1]).reshapeChecked #[1, 2, 4]
  let weights := Tensor.fromFloatArray #[0.73, 1.19, -0.67, 1.53]
  for dtype in #[DType.Float32, .BFloat16] do
    let x := values.castDType dtype
    let w := weights.castDType dtype
    let norm : RMSNorm 4 dtype := { weight := w, eps := ⟨0.01⟩ }
    let normalized := Tensor.rmsNormWeighted x.toFloat32 (Tensor.ones #[4]) 0.01
    let expected := Tensor.Preserving.mul (Tensor.castLike x normalized) (w.expand #[1, 2, 4])
    let outputs := #[norm.forward3d x,
      (norm.forward2d (x.reshapeChecked #[2, 4])).reshapeChecked #[1, 2, 4],
      (norm.forward4d (x.reshapeChecked #[1, 1, 2, 4])).reshapeChecked #[1, 2, 4],
      (norm.forward5d (x.reshapeChecked #[1, 1, 2, 4])).reshapeChecked #[1, 2, 4]]
    for actual in outputs do
      LeanTest.assertEqual actual.actualSpec.dtype x.actualSpec.dtype
      LeanTest.assertEqual (← actual.toFloatArray) (← expected.toFloatArray)
        "RMSNorm restores activation dtype before multiplying non-unit weights"

@[test] def testTypedGroupedAndTransposeConvolution : IO Unit := do
  let x := (Tensor.fromFloatArray #[1, 2, 3, 4, 5, 6]).reshapeChecked #[1, 2, 3]
  let w := (Tensor.fromFloatArray #[2, 3]).reshapeChecked #[2, 1, 1]
  let grouped := Tensor.conv1dGroups x w (Tensor.zeros #[2]) 1 0 1 2
  LeanTest.assertEqual (← grouped.toFloatArray) #[2, 4, 6, 12, 15, 18]
  let z := (Tensor.fromFloatArray #[1, 2]).reshapeChecked #[1, 1, 2]
  let kernel := (Tensor.fromFloatArray #[1, 2]).reshapeChecked #[1, 1, 2]
  let expanded := Tensor.convTranspose1d z kernel (Tensor.zeros #[1]) 2
  LeanTest.assertEqual expanded.runtimeShape #[1, 1, 4]
  LeanTest.assertEqual (← expanded.toFloatArray) #[1, 2, 2, 4]

@[test] def testTypedStreamingWav : IO Unit := IO.FS.withTempDir fun dir => do
  let path := (dir / "stream.wav").toString
  Tensor.wavBegin path 16000
  let emit (chunk : Sigma fun n => SomeDTypeTensor #[1, 1, n] .CPU) : IO Unit :=
    Tensor.wavAppend chunk.2.2 path
  emit ⟨3, ((Tensor.fromFloatArray #[-1, 0, 1]).reshapeChecked #[1, 1, 3]).packDType (by decide)⟩
  emit ⟨0, (Tensor.zeros #[1, 1, 0]).packDType (by decide)⟩
  emit ⟨2, ((Tensor.fromFloatArray #[0.5, -0.5]).reshapeChecked #[1, 1, 2]).toBFloat16.packDType (by decide)⟩
  Tensor.wavFinalize path
  let bytes ← IO.FS.readBinFile path
  LeanTest.assertEqual bytes.size 54 "Five mono PCM16 samples plus the WAV header"
  let read32 (offset : Nat) : Nat :=
    (bytes[offset]!).toNat + 256 * (bytes[offset + 1]!).toNat +
      65536 * (bytes[offset + 2]!).toNat + 16777216 * (bytes[offset + 3]!).toNat
  LeanTest.assertEqual (read32 4) 46 "RIFF size reflects both chunks"
  LeanTest.assertEqual (read32 24) 16000 "Requested sample rate survives the native ABI"
  LeanTest.assertEqual (read32 40) 10 "Data size excludes the header"
  LeanTest.assertEqual (bytes.extract 44 54).data #[1, 128, 0, 0, 255, 127, 0, 64, 0, 192]
    "FP32 and BF16 chunks are converted to signed PCM16 in order"
