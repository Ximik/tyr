import LeanTest
import Tyr.Typed.Neural
import Tyr.RawTensor.Qwen35
import Tyr.RawTensor.Gemma4

open torch

namespace Tests.TestFFIBoundary

private def values {σ : StaticSpec} (t : Tensor σ) : IO (Array Float) :=
  t.toFloatArray

/-- Exercise boxed Lean Int/Nat arguments through the real native bridge. -/
@[test]
def testBoxedIndexAndUnbind : IO Unit := do
  let x := (Tensor.arangeFrom 0 6).reshapeChecked #[2, 3]
  LeanTest.assertEqual (← values (x.getOp 0)) #[0.0, 1.0, 2.0]
  LeanTest.assertEqual (← values (x.getOp 1)) #[3.0, 4.0, 5.0]
  LeanTest.assertEqual (← values (x.getOp (-1))) #[3.0, 4.0, 5.0]
  let rows := Tensor.unbind x 0
  LeanTest.assertEqual rows.size 2
  LeanTest.assertEqual rows[0]!.runtimeShape #[3]
  LeanTest.assertEqual (← values rows[1]!) #[3.0, 4.0, 5.0]
  let columns := Tensor.unbind x 1
  LeanTest.assertEqual columns.size 3
  LeanTest.assertEqual columns[0]!.runtimeShape #[2]
  LeanTest.assertEqual (← values columns[2]!) #[2.0, 5.0]

@[test]
def testArangeLengthAndScalarWidths : IO Unit := do
  let cases : Array (UInt64 × UInt64 × UInt64) :=
    #[(0, 5, 2), (0, 6, 2), (2, 2, 1), (0, 1, 4), (4294967296, 4294967299, 1)]
  for (start, stop, step) in cases do
    let x := Tensor.arangeFrom start stop step
    LeanTest.assertEqual x.runtimeShape #[Tensor.rangeLength start stop step]
  LeanTest.assertEqual (← values (Tensor.arangeFrom 0 5 2)) #[0.0, 2.0, 4.0]
  -- Host arrays use float32 conversion; item preserves these int64 values.
  let wide := Tensor.arangeFrom 4294967296 4294967299
  LeanTest.assertEqual (Tensor.item ((wide.getOp 0).reshapeChecked #[])) 4294967296.0
  LeanTest.assertEqual (Tensor.item ((wide.getOp 1).reshapeChecked #[])) 4294967297.0
  LeanTest.assertEqual (Tensor.item ((wide.getOp 2).reshapeChecked #[])) 4294967298.0
  LeanTest.assertEqual (Tensor.eye 2).runtimeShape #[2, 2]
  LeanTest.assertEqual (← values (Tensor.linspace 0.0 1.0 3)) #[0.0, 0.5, 1.0]
  LeanTest.assertEqual (← values (Tensor.logspace 0.0 2.0 3)) #[1.0, 10.0, 100.0]

@[test]
def testExactScalarReshape : IO Unit := do
  let x := Tensor.ones #[1]
  let scalar : Tensor { shape := #[], dtype := .Float32, device := .CPU } := x.reshape #[]
  LeanTest.assertEqual scalar.runtimeShape #[]
  LeanTest.assertEqual scalar.item 1.0

@[test]
def testExactLoadRejectsSameNumelWrongShape : IO Unit :=
  IO.FS.withTempDir fun dir => do
    let path := (dir / "tensor.pt").toString
    Tensor.saveTensor (Tensor.ones #[2, 3]) path
    let exact ← Tensor.loadTensorExact {shape := #[2, 3], dtype := .Float32, device := .CPU} path
    LeanTest.assertEqual exact.runtimeShape #[2, 3]
    try
      let _ ← Tensor.loadTensorExact {shape := #[3, 2], dtype := .Float32, device := .CPU} path
      LeanTest.fail "exact loading must reject a shape mismatch"
    catch error =>
      LeanTest.assertTrue (error.toString.containsSubstr "Tensor shape mismatch")

/-- Invalid STFT parameters must stay recoverable at the advertised IO boundary. -/
@[test]
def testStftNativeFailuresAreRecoverable : IO Unit := do
  let input := Tensor.ones #[8]
  let window := Tensor.hannWindow 4
  let invalid : Array (UInt64 × UInt64) := #[(0, 1), (4, 0), (2, 1), (16, 1)]
  for (nFft, hopLength) in invalid do
    let caught ← try
      let _ ← Tensor.stft input nFft hopLength window (center := false)
      pure false
    catch error =>
      LeanTest.assertTrue (error.toString.containsSubstr "stft")
      pure true
    LeanTest.assertTrue caught
  -- A failed call must leave borrowed inputs valid for a subsequent transform.
  let result ← Tensor.stft input 4 2 window
  LeanTest.assertEqual result.1 #[3, 5, 2]
  LeanTest.assertEqual result.2.runtimeShape #[3, 5, 2]
  LeanTest.assertTrue (result.2.actualSpec.dtype == .Float32)
  LeanTest.assertTrue (result.2.raw.device == .CPU)

/-- Media IO catches native size errors without allocating an enormous buffer. -/
@[test]
def testMediaNativeFailuresAreRecoverable : IO Unit := IO.FS.withTempDir fun dir => do
  if !System.Platform.isOSX then return
  let path := dir / "pixel.png"
  let bytes := ByteArray.mk #[137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
    0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0, 144, 119, 83, 222, 0, 0, 0, 12,
    73, 68, 65, 84, 120, 156, 99, 96, 96, 96, 0, 0, 0, 4, 0, 1, 246, 23, 56, 85,
    0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130]
  IO.FS.writeBinFile path bytes
  -- More than vector<float>::max_size(): throws before attempting allocation.
  LeanTest.assertThrows
    (RawTensor.loadImagePatchified path.toString 3 1 2305843009213693952)
    (some "loadImagePatchified")
  LeanTest.assertThrows
    (RawTensor.loadVideoPatchified (dir / "missing.mov").toString 3 1 1 1 1)
    (some "loadVideoPatchified")
  LeanTest.assertThrows (RawTensor.gemma4LoadImage path.toString 0 1 1 (1 / 255))
    (some "loadGemma4ImagePatchGrid")
  let image ← RawTensor.loadImagePatchified path.toString 3 1 1
  LeanTest.assertEqual image.runtimeShape #[1, 3]
  let grid ← RawTensor.gemma4LoadImage path.toString 1 1 1 (1 / 255)
  LeanTest.assertEqual grid.runtimeShape #[1, 1, 3]

end Tests.TestFFIBoundary
