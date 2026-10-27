import LeanTest
import Tests.Assertions
import Tyr.SafeTensors.Exact
import Tyr.Model.Qwen.Weights

namespace Tests.ExactSafeTensors

open torch

-- Stored dtype is a witnessed index, not a shape-only tensor or implicit cast.
example (path name : String) (shape : Shape) (device : Device) :
    IO (SomeDTypeTensor shape device) :=
  safetensors.loadTensorExact path name shape device

example (_packed : SomeDTypeTensor #[2] .CPU) : True := by
  fail_if_success
    have _ : Tensor ⟨#[2], .Float32, .CPU⟩ := _packed.2
  fail_if_success
    have _ : Tensor ⟨#[2], .Float32, .CPU⟩ := _packed
  trivial

@[test] def testExactSafeTensorsStoredDTypeAndExplicitConversion : IO Unit :=
    IO.FS.withTempDir fun dir => do
  let source := Tensor.fromFloatArray #[1.5, -2.0]
  for dtype in #[DType.BFloat16, .Float32] do
    let path := (dir / s!"{dtype.canonicalName}.safetensors").toString
    let stored ← IO.ofExcept (source.castDType? dtype)
    stored.saveSafeTensor path "weight"
    let packed ← safetensors.loadTensorExact path "weight" #[2]
    LeanTest.assertEqual packed.1 dtype
    LeanTest.assertEqual packed.2.dtype dtype
    LeanTest.assertEqual packed.2.raw.dtype dtype
    LeanTest.assertTrue packed.2.validate.toOption.isSome
    let converted : Tensor ⟨#[2], .Float32, .CPU⟩ ←
      IO.ofExcept (packed.2.castDType? .Float32)
    LeanTest.assertEqual (← converted.toFloatArray) #[1.5, -2.0]
    LeanTest.assertEqual packed.2.raw.dtype dtype

@[test] def testExactSafeTensorsRejectsStoredLayoutMismatch : IO Unit :=
    IO.FS.withTempDir fun dir => do
  let path := (dir / "layout.safetensors").toString
  let source := (Tensor.fromFloatArray #[1, 2, 3, 4, 5, 6]).reshapeChecked #[2, 3]
  source.saveSafeTensor path "weight"
  Tests.assertThrows (safetensors.loadTensorExact path "weight" #[3, 2])
    (some "Expected shape")
  Tests.assertThrows (safetensors.loadOptionalTensorExact path "weight" #[3, 2])
    (some "Expected shape")
  Tests.assertThrows (safetensors.loadTensorShardedExact dir.toString "weight" #[6])
    (some "Expected shape")
  Tests.assertThrows (safetensors.loadOptionalTensorShardedExact dir.toString "weight" #[6])
    (some "Expected shape")

@[test] def testExactSafeTensorsOptionalMissingAndReadErrors : IO Unit :=
    IO.FS.withTempDir fun dir => do
  let path := (dir / "weights.safetensors").toString
  (Tensor.fromFloatArray #[2, 3]).saveSafeTensor path "weight"
  LeanTest.assertTrue (← safetensors.loadOptionalTensorExact path "absent" #[2]).isNone
  LeanTest.assertTrue
    (← safetensors.loadOptionalTensorShardedExact dir.toString "absent" #[2]).isNone
  let present ← safetensors.loadOptionalTensorShardedExact dir.toString "weight" #[2]
  let some packed := present | throw (IO.userError "expected stored weight")
  LeanTest.assertEqual packed.1 .Float32
  LeanTest.assertEqual (← packed.2.toFloatArray) #[2, 3]
  Tests.assertThrows
    (safetensors.loadOptionalTensorExact (dir / "absent.safetensors").toString "absent" #[2])
    (some "no such file or directory")
  Tests.assertThrows
    (safetensors.loadOptionalTensorShardedExact (dir / "absent").toString "absent" #[2])

@[test] def testQwenExactSafeTensorsRetainsExplicitFloat32Policy : IO Unit :=
    IO.FS.withTempDir fun dir => do
  let path := (dir / "model.safetensors").toString
  let source := (Tensor.fromFloatArray #[1.5, -2.0]).castDType .BFloat16
  source.saveSafeTensor path "model.norm.weight"
  let norm ← qwen.loadRMSNorm path "model.norm" 2
  let sharded ← qwen.loadRMSNormSharded dir.toString "model.norm" 2
  LeanTest.assertEqual norm.weight.raw.dtype .Float32
  LeanTest.assertEqual sharded.weight.raw.dtype .Float32
  LeanTest.assertEqual (← norm.weight.toFloatArray) #[1.5, -2.0]
  LeanTest.assertEqual (← sharded.weight.toFloatArray) #[1.5, -2.0]

@[test] def testExactDTypeFP8DequantizationKeepsIndependentScaleDType : IO Unit := do
  let weight := (Tensor.ones #[128, 128]).castDType .Float8E4M3FN
  let scale := (Tensor.fromFloatArray #[2]).reshapeChecked #[1, 1]
  let result : Tensor ⟨#[128, 128], .Float32, .CPU⟩ := Model.dequantizeFP8 weight scale
  LeanTest.assertEqual (Tensor.sumAll result).item 32768
  LeanTest.assertEqual weight.raw.dtype .Float8E4M3FN
  LeanTest.assertEqual scale.raw.dtype .Float32

end Tests.ExactSafeTensors
