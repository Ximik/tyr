import LeanTest
import Tyr.TensorStruct.Cast
import Tyr.Typed.Neural
import Tyr.SafeTensors.Schema
import Tyr.SafeTensors.Source

namespace Tests.DTypeConversions

open torch

@[test] def testComplexConversionRoundtrip : IO Unit := do
  let original := Tensor.fromFloatArray #[1.25, -2.5]
  let complex64 := original.castDType .Complex64
  let complex128 := complex64.castDType .Complex128
  let restored := complex128.castDType .Float32
  LeanTest.assertEqual complex64.raw.dtype .Complex64
  LeanTest.assertEqual complex128.raw.dtype .Complex128
  LeanTest.assertEqual (← restored.toFloatArray) (← original.toFloatArray)
  LeanTest.assertEqual original.raw.dtype .Float32
  LeanTest.assertTrue (complex128.raw.device == original.raw.device)

@[test] def testTreeComplexConversionRoundtrip : IO Unit := do
  let original := Tensor.fromFloatArray #[1.25, -2.5]
  let complex ← TensorStruct.castTo
    (Tensor { shape := #[2], dtype := .Complex128, device := .CPU }) original
  let restored ← TensorStruct.castTo
    (Tensor { shape := #[2], dtype := .Float32, device := .CPU }) complex
  LeanTest.assertEqual complex.raw.dtype .Complex128
  LeanTest.assertEqual (← restored.toFloatArray) (← original.toFloatArray)

@[test] def testComplexIdentityFactory : IO Unit := do
  let identity : Tensor { shape := #[2, 2], dtype := .Complex64, device := .CPU } :=
    Tensor.eyeFor (n := 2) .Complex64 (.CPU)
  LeanTest.assertEqual identity.raw.dtype .Complex64
  LeanTest.assertTrue (identity.raw.device == .CPU)
  let restored := identity.castDType .Float32
  LeanTest.assertEqual (← restored.toFloatArray) #[1, 0, 0, 1]

@[test] def testUnsupportedTreeConversionsReportIOErrors : IO Unit := do
  let original := Tensor.ones #[2]
  for dtype in #[DType.QInt8, DType.Bits1x8] do
    let rejected ← try
        let _ ← TensorStruct.castTo (Tensor { shape := #[2], dtype, device := .CPU }) original
        pure false
      catch error =>
        LeanTest.assertTrue (error.toString.containsSubstr "castTo at <root>")
        pure true
    LeanTest.assertTrue rejected "Tree conversion must report native rejection through IO."
  LeanTest.assertEqual (← original.toFloatArray) #[1, 1]

example (_left _right : Tensor ⟨#[2], .Bool, .CPU⟩) (_floats : Tensor ⟨#[2], .Float32, .CPU⟩) : True := by
  have _ := Tensor.sub _floats _floats
  fail_if_success have _ := Tensor.sub _left _right
  fail_if_success have _ := Tensor.sub _left _floats
  fail_if_success have _ := Tensor.subScalar _left 1
  trivial

example (_float8 : Tensor ⟨#[1], .Float8E4M3FN, .CPU⟩) (_floats : Tensor ⟨#[1], .Float32, .CPU⟩) : True := by
  have _ := Tensor.add _floats _floats
  fail_if_success have _ := Tensor.add _float8 _floats
  fail_if_success have _ := Tensor.mul _floats _float8
  trivial

@[test] def testFloatingArithmeticRetainsResultDTypeAndValues : IO Unit := do
  let lhs := (Tensor.fromFloatArray #[6, 12])
  let rhs := (Tensor.fromFloatArray #[2, 3])
  let cases : Array (Array Float × Tensor { shape := #[2], dtype := .Float32, device := .CPU }) := #[
    (#[8, 15], Tensor.add lhs rhs),
    (#[4, 9], Tensor.sub lhs rhs),
    (#[12, 36], Tensor.mul lhs rhs),
    (#[3, 4], Tensor.div lhs rhs),
    (#[8, 14], Tensor.addScalar lhs 2),
    (#[12, 24], Tensor.mulScalar lhs 2),
    (#[3, 6], Tensor.divScalar lhs 2)]
  for (expected, output) in cases do
    LeanTest.assertEqual output.raw.dtype .Float32
    LeanTest.assertEqual (← output.toFloatArray) expected
  LeanTest.assertEqual (← lhs.toFloatArray) #[6, 12]
  LeanTest.assertEqual (← rhs.toFloatArray) #[2, 3]

/-- Each mixed-dtype result is admitted by `ofRaw!`, which aborts if libtorch's
promotion differs from the Lean result index; the values show the arithmetic ran
in the promoted dtype. -/
@[test] def testMixedPromotionMatchesNative : IO Unit := do
  let bf16 := (Tensor.fromFloatArray #[6, 12]).castDType .BFloat16
  let f32 := Tensor.fromFloatArray #[2, 3]
  let f64 := (Tensor.fromFloatArray #[1, 1]).castDType .Float64
  let u8 := (Tensor.fromFloatArray #[250, 1]).castDType .UInt8
  let i8 := (Tensor.fromFloatArray #[-1, 2]).castDType .Int8
  let i32 := (Tensor.fromFloatArray #[6, 12]).castDType .Int32
  let i64 := (Tensor.fromFloatArray #[2, 3]).castDType .Int64
  let cases : Array (String × Array Float × Array Float) := #[
    ("BFloat16 + Float32", ← (Tensor.add bf16 f32 : Tensor ⟨#[2], .Float32, .CPU⟩).toFloatArray, #[8, 15]),
    ("BFloat16 - Float32", ← (Tensor.sub bf16 f32 : Tensor ⟨#[2], .Float32, .CPU⟩).toFloatArray, #[4, 9]),
    ("BFloat16 * Float32", ← (Tensor.mul bf16 f32 : Tensor ⟨#[2], .Float32, .CPU⟩).toFloatArray, #[12, 36]),
    ("BFloat16 / Float32", ← (Tensor.div bf16 f32 : Tensor ⟨#[2], .Float32, .CPU⟩).toFloatArray, #[3, 4]),
    ("Float32 + Float64", ← (Tensor.add f32 f64 : Tensor ⟨#[2], .Float64, .CPU⟩).toFloatArray, #[3, 4]),
    ("UInt8 + Int8", ← (Tensor.add u8 i8 : Tensor ⟨#[2], .Int16, .CPU⟩).toFloatArray, #[249, 3]),
    ("Int32 / Int64", ← (Tensor.div i32 i64 : Tensor ⟨#[2], .Float32, .CPU⟩).toFloatArray, #[3, 4]),
    ("Int64 / Int64", ← (Tensor.div i64 i64 : Tensor ⟨#[2], .Float32, .CPU⟩).toFloatArray, #[1, 1]),
    ("Int32 + 0.5", ← (Tensor.addScalar i32 0.5 : Tensor ⟨#[2], .Float32, .CPU⟩).toFloatArray, #[6.5, 12.5]),
    ("Int32 - 0.5", ← (Tensor.subScalar i32 0.5 : Tensor ⟨#[2], .Float32, .CPU⟩).toFloatArray, #[5.5, 11.5]),
    ("Int32 * 0.5", ← (Tensor.mulScalar i32 0.5 : Tensor ⟨#[2], .Float32, .CPU⟩).toFloatArray, #[3, 6]),
    ("Int32 / 0.5", ← (Tensor.divScalar i32 0.5 : Tensor ⟨#[2], .Float32, .CPU⟩).toFloatArray, #[12, 24])]
  for (label, actual, expected) in cases do
    LeanTest.assertEqual actual expected label

end Tests.DTypeConversions
