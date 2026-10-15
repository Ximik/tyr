import Tyr.Model.Qwen3TTS.SpeechTokenizer
import LeanTest
import Tests.Assertions

namespace Tests.SpeechTokenizerDTypes
open torch torch.qwen3tts

example (_rvq : RVQDecode (device := .CPU))
    (_weight : Tensor ⟨#[512, 256, 1], .Float64, .CPU⟩) : True := by
  fail_if_success
    have _ : RVQDecode (device := .CPU) := { _rvq with outputProj := _weight }
  trivial

private abbrev bf16 : RealFloatingDType := ⟨.BFloat16, by decide⟩

example (_layout : SpeechTokenizerDTypes)
    (_codebook : QuantizerCodebook (device := .CPU) _layout)
    (_projection : Tensor ⟨#[512, 256, 1], .Float32, .CPU⟩) : True := by
  fail_if_success
    have _ : RVQDecode (device := .CPU) _layout :=
      { outputProj := _projection, codebooks := #[] }
  fail_if_success
    have _ : QuantizerCodebook (device := .CPU) float32SpeechTokenizerDTypes := _codebook
  trivial

example : True := by
  fail_if_success
    have _ : SpeechTokenizerDTypes := {
      compute := bf16, usage := float32DType.toExactDType, sum := bf16.toExactDType,
      promoted := .Float32, promotion := ⟨rfl⟩, resultEq := rfl }
  trivial

@[test] def testSpeechCodebookDivisionPreservesStoredDTypes : IO Unit := do
  -- The result is F32 for either mixed storage order. Homogeneous BF16 stays
  -- BF16; neither source is converted to make matrix multiplication typecheck.
  for (compute, usage, sum) in #[(float32DType, float32DType, bf16),
      (float32DType, bf16, float32DType), (bf16, bf16, bf16)] do
    let layout ← IO.ofExcept (SpeechTokenizerDTypes.ofDTypes? compute.toExactDType usage.toExactDType sum.toExactDType)
    let sums ← IO.ofExcept ((Tensor.fromFloatArray #[6, 8]).castDType? layout.sum.val)
    let counts ← IO.ofExcept ((Tensor.fromFloatArray #[2, 4]).castDType? layout.usage.val)
    let result : Tensor ⟨#[2], layout.compute.val, .CPU⟩ := layout.divideCodebook sums counts
    LeanTest.assertEqual result.raw.dtype compute.val
    LeanTest.assertEqual (← result.toFloatArray) #[3, 2]
    LeanTest.assertEqual sums.raw.dtype sum.val
    LeanTest.assertEqual counts.raw.dtype usage.val
    LeanTest.assertEqual (← sums.toFloatArray) #[6, 8]
    LeanTest.assertEqual (← counts.toFloatArray) #[2, 4]

@[test] def testSpeechCodebookRejectsWrongProjectionPrecision : IO Unit := do
  let invalid := SpeechTokenizerDTypes.ofDTypes? bf16.toExactDType float32DType.toExactDType bf16.toExactDType
  match invalid with
  | .ok _ => throw (IO.userError "Mixed BF16/F32 codebook cannot feed a BF16 projection")
  | .error message =>
    LeanTest.assertTrue (message.contains "Codebook division produces")

-- A three-entry checkpoint is enough to exercise the loader's precision
-- anchors without allocating the full released speech-tokenizer model.
private def writeAnchors (path : System.FilePath) (computeShape : Array Nat)
    (computeBF16 usageBF16 sumBF16 : Bool) (usageShape : Array Nat := #[2048]) : IO Unit := do
  let entries := #[
    ("compute", computeShape, computeBF16),
    ("usage", usageShape, usageBF16),
    ("sum", #[2048, 256], sumBF16)]
  let mut header : List (String × Lean.Json) := []
  let mut data := ByteArray.empty
  for (name, shape, isBF16) in entries do
    let start := data.size
    let byteCount := shape.foldl (· * ·) 1 * (if isBF16 then 2 else 4)
    data := data ++ ByteArray.mk (Array.replicate byteCount 0)
    header := header ++ [(name, Lean.Json.mkObj [
      ("dtype", Lean.toJson (if isBF16 then "BF16" else "F32")),
      ("shape", Lean.toJson shape), ("data_offsets", Lean.toJson #[start, data.size])])]
  let json := (Lean.Json.mkObj header).compress.toUTF8
  let mut headerPrefix := ByteArray.empty
  for i in [:8] do
    headerPrefix := headerPrefix.push ((json.size.toUInt64 >>> (i * 8).toUInt64).toUInt8)
  IO.FS.writeBinFile path (headerPrefix ++ json ++ data)

@[test] def testSpeechCodebookLoaderRetainsMixedStatistics : IO Unit :=
    IO.FS.withTempDir fun dir => do
  let path := dir / "mixed.safetensors"
  writeAnchors path #[2] false false true
  let layout ← SpeechTokenizerDTypes.load path.toString "compute" #[2] "usage" "sum" .CPU
  LeanTest.assertEqual layout.compute.val .Float32
  LeanTest.assertEqual layout.usage.val .Float32
  LeanTest.assertEqual layout.sum.val .BFloat16
  let sums ← safetensors.loadTensorExact path.toString "sum" #[2048, 256]
  let checked ← IO.ofExcept (sums.atDType? layout.sum)
  LeanTest.assertEqual checked.raw.dtype .BFloat16
  LeanTest.assertEqual checked.runtimeShape #[2048, 256]

@[test] def testSpeechCodebookLoaderRejectsShapeAndPrecisionMismatch : IO Unit :=
    IO.FS.withTempDir fun dir => do
  let path := dir / "invalid.safetensors"
  writeAnchors path #[2] true false true
  Tests.assertThrows
    (SpeechTokenizerDTypes.load path.toString "compute" #[2] "usage" "sum" .CPU)
    (some "Speech tokenizer codebook 'sum': Codebook division produces")
  let shapePath := dir / "shape.safetensors"
  writeAnchors shapePath #[2] false false true #[1]
  Tests.assertThrows
    (SpeechTokenizerDTypes.load shapePath.toString "compute" #[2] "usage" "sum" .CPU)
    (some "Expected shape")

end Tests.SpeechTokenizerDTypes
