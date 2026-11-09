import LeanTest
import Tyr.TensorStruct.Cast
import Tyr.Typed.Neural
import Tyr.SafeTensors.Schema
import Tyr.SafeTensors.Source

namespace Tests.DTypeConversions

open torch

private def checked {α : Type} (result : Except String α) : IO α :=
  match result with
  | .ok value => pure value
  | .error message => throw (IO.userError message)

@[test] def testCheckedComplexConversionRoundtrip : IO Unit := do
  let original := Tensor.fromFloatArray #[1.25, -2.5]
  let complex64 ← checked (original.castDType? .Complex64)
  let complex128 ← checked (complex64.castDType? .Complex128)
  let restored ← checked (complex128.castDType? .Float32)
  LeanTest.assertEqual complex64.raw.dtype .Complex64
  LeanTest.assertEqual complex128.raw.dtype .Complex128
  LeanTest.assertEqual (← restored.toFloatArray) (← original.toFloatArray)
  LeanTest.assertEqual original.raw.dtype .Float32
  LeanTest.assertTrue (complex128.raw.device == original.raw.device)
  LeanTest.assertTrue complex128.validate.toOption.isSome
  -- The ordinary convenience uses the same checked native conversion.
  LeanTest.assertEqual (original.castDType .Complex64).raw.dtype .Complex64

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
  LeanTest.assertTrue identity.validate.toOption.isSome
  let restored ← checked (identity.castDType? .Float32)
  LeanTest.assertEqual (← restored.toFloatArray) #[1, 0, 0, 1]

@[test] def testUnsupportedKnownConversionsAreRecoverable : IO Unit := do
  let original := Tensor.ones #[2]
  for dtype in #[DType.QInt8, DType.Bits1x8] do
    match original.castDType? dtype with
    | .ok _ => LeanTest.fail s!"Expected unsupported native conversion to {dtype}"
    | .error message => LeanTest.assertTrue (message.containsSubstr "castDType:")
    let rejected ← try
        let _ ← TensorStruct.castTo (Tensor { shape := #[2], dtype, device := .CPU }) original
        pure false
      catch error =>
        LeanTest.assertTrue (error.toString.containsSubstr "castTo at <root>")
        pure true
    LeanTest.assertTrue rejected "Tree conversion must report native rejection through IO."
  LeanTest.assertEqual (← original.toFloatArray) #[1, 1]

@[test] def testInt64TrueDivisionReturnsFloat32 : IO Unit := do
  let numerator := Tensor.fromInt64Array #[6]
  let denominator := Tensor.fromInt64Array #[2]
  let result : Tensor ⟨#[1], .Float32, .CPU⟩ := Tensor.div numerator denominator
  LeanTest.assertEqual result.raw.dtype .Float32
  LeanTest.assertEqual (← result.toFloatArray) #[3]

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
  let cases : Array (Array Float × Except String (Tensor { shape := #[2], dtype := .Float32, device := .CPU })) := #[
    (#[8, 15], Tensor.add? lhs rhs),
    (#[4, 9], Tensor.sub? lhs rhs),
    (#[12, 36], Tensor.mul? lhs rhs),
    (#[3, 4], Tensor.div? lhs rhs),
    (#[8, 14], Tensor.addScalar? lhs 2),
    (#[12, 24], Tensor.mulScalar? lhs 2),
    (#[3, 6], Tensor.divScalar? lhs 2)]
  for (expected, result) in cases do
    let output ← checked result
    LeanTest.assertEqual output.raw.dtype .Float32
    LeanTest.assertTrue output.validate.toOption.isSome
    LeanTest.assertEqual (← output.toFloatArray) expected
  LeanTest.assertEqual (← lhs.toFloatArray) #[6, 12]
  LeanTest.assertEqual (← rhs.toFloatArray) #[2, 3]

@[test] def testMixedFloatPromotion : IO Unit := do
  let left := ((Tensor.fromFloatArray #[6, 12]).castDType .BFloat16)
  let right := (Tensor.fromFloatArray #[2, 3])
  let sum : Tensor ⟨#[2], .Float32, .CPU⟩ := Tensor.add left right
  let difference : Tensor ⟨#[2], .Float32, .CPU⟩ := Tensor.sub left right
  let product : Tensor ⟨#[2], .Float32, .CPU⟩ := Tensor.mul left right
  let quotient : Tensor ⟨#[2], .Float32, .CPU⟩ := Tensor.div left right
  for (expected, result) in #[(#[8, 15], sum), (#[4, 9], difference), (#[12, 36], product),
      (#[3, 4], quotient)] do
    LeanTest.assertEqual result.raw.dtype .Float32
    LeanTest.assertEqual (← result.toFloatArray) expected
  LeanTest.assertEqual left.raw.dtype .BFloat16
  LeanTest.assertEqual (← left.toFloatArray) #[6, 12]
  LeanTest.assertEqual (← right.toFloatArray) #[2, 3]

@[test] def testMixedIntegerPromotion : IO Unit := do
  let left := ((Tensor.fromFloatArray #[250, 1]).castDType .UInt8)
  let right := ((Tensor.fromFloatArray #[-1, 2]).castDType .Int8)
  let result : Tensor ⟨#[2], .Int16, .CPU⟩ := Tensor.add left right
  LeanTest.assertEqual result.raw.dtype .Int16
  LeanTest.assertEqual (← result.toFloatArray) #[249, 3]

@[test] def testIntegerDivisionPromotion : IO Unit := do
  let left := ((Tensor.fromFloatArray #[6, 12]).castDType .Int32)
  let right := ((Tensor.fromFloatArray #[2, 3]).castDType .Int64)
  let result : Tensor ⟨#[2], .Float32, .CPU⟩ := Tensor.div left right
  LeanTest.assertEqual result.raw.dtype .Float32
  LeanTest.assertEqual (← result.toFloatArray) #[3, 4]

@[test] def testHostFloatScalarPromotion : IO Unit := do
  let input := ((Tensor.fromFloatArray #[6, 12]).castDType .Int32)
  let sum : Tensor ⟨#[2], .Float32, .CPU⟩ := Tensor.addScalar input 0.5
  let difference : Tensor ⟨#[2], .Float32, .CPU⟩ := Tensor.subScalar input 0.5
  let product : Tensor ⟨#[2], .Float32, .CPU⟩ := Tensor.mulScalar input 0.5
  let quotient : Tensor ⟨#[2], .Float32, .CPU⟩ := Tensor.divScalar input 0.5
  for (expected, result) in #[(#[6.5, 12.5], sum), (#[5.5, 11.5], difference),
      (#[3, 6], product), (#[12, 24], quotient)] do
    LeanTest.assertEqual result.raw.dtype .Float32
    LeanTest.assertEqual (← result.toFloatArray) expected
  LeanTest.assertEqual input.raw.dtype .Int32
  LeanTest.assertEqual (← input.toFloatArray) #[6, 12]

end Tests.DTypeConversions
