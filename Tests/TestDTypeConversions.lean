import LeanTest
import Tyr.TensorStruct.Cast
import Tyr.Typed.Neural
import Tyr.SafeTensors.Schema

namespace Tests.DTypeConversions

open torch

private def checked {α : Type} (result : Except String α) : IO α :=
  match result with
  | .ok value => pure value
  | .error message => throw (IO.userError message)

private def expectError {α : Type} (result : Except String α)
    (expected : String) : IO Unit :=
  match result with
  | .ok _ => LeanTest.fail s!"Expected dtype-operation rejection: {expected}"
  | .error message => LeanTest.assertTrue (message.containsSubstr expected) message

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

@[test] def testErasedUnsupportedPromotionIsRejectedBeforeNativeArithmetic : IO Unit := do
  let path := "Tests/fixtures/typed/dtype_scalars.safetensors"
  let floating := (Tensor.ones #[1])
  for dtype in #[DType.Float8E4M3FN, DType.Float8E5M2] do
    let stored ← safetensors.loadTensorWithSpec path dtype.canonicalName ⟨#[1], dtype⟩
    let exact := stored
    LeanTest.assertEqual exact.raw.dtype dtype
    -- These pairs have no native promotion. Reaching a raw arithmetic call
    -- would throw across its pure FFI boundary, so returning Except is essential.
    expectError (Tensor.Promoting.add? exact floating) "Unsupported dtype promotion"
    expectError (Tensor.Promoting.sub? exact floating) "Unsupported dtype promotion"
    expectError (Tensor.Promoting.mul? exact floating) "Unsupported dtype promotion"
    expectError (Tensor.Promoting.div? exact floating) "Unsupported dtype promotion"
    expectError (Tensor.Promoting.add? floating exact) "Unsupported dtype promotion"
    expectError (Tensor.Promoting.sub? floating exact) "Unsupported dtype promotion"
    expectError (Tensor.Promoting.mul? floating exact) "Unsupported dtype promotion"
    expectError (Tensor.Promoting.div? floating exact) "Unsupported dtype promotion"

@[test] def testErasedIntegerPromotionDoesNotPassPreservingChecks : IO Unit := do
  for dtype in #[DType.Int32, DType.Int64] do
    let integers := ((Tensor.fromFloatArray #[6, 12]).castDType dtype)
    let expected := s!"Expected dtype {dtype}, got Float32"
    expectError (Tensor.Preserving.div? integers integers) expected
    expectError (Tensor.Preserving.addScalar? integers 2) expected
    expectError (Tensor.Preserving.mulScalar? integers 2) expected
    expectError (Tensor.Preserving.divScalar? integers 2) expected
    LeanTest.assertEqual integers.raw.dtype dtype
    LeanTest.assertEqual (← integers.toFloatArray) #[6, 12]

@[test] def testErasedBoolSubtractionIsRejected : IO Unit := do
  let floating := (Tensor.ones #[2])
  let booleans := ((Tensor.ones #[2]).castDType .Bool)
  expectError (Tensor.Preserving.sub? booleans booleans) "Bool operands"
  expectError (Tensor.Promoting.sub? booleans floating) "Bool operands"
  expectError (Tensor.Promoting.sub? floating booleans) "Bool operands"

@[test] def testErasedFloatPreservingArithmeticValues : IO Unit := do
  let lhs := (Tensor.fromFloatArray #[6, 12])
  let rhs := (Tensor.fromFloatArray #[2, 3])
  let cases : Array (Array Float × Except String (Tensor { shape := #[2], dtype := .Float32, device := .CPU })) := #[
    (#[8, 15], Tensor.Preserving.add? lhs rhs),
    (#[4, 9], Tensor.Preserving.sub? lhs rhs),
    (#[12, 36], Tensor.Preserving.mul? lhs rhs),
    (#[3, 4], Tensor.Preserving.div? lhs rhs),
    (#[8, 14], Tensor.Preserving.addScalar? lhs 2),
    (#[12, 24], Tensor.Preserving.mulScalar? lhs 2),
    (#[3, 6], Tensor.Preserving.divScalar? lhs 2)]
  for (expected, result) in cases do
    let output ← checked result
    LeanTest.assertEqual output.raw.dtype .Float32
    LeanTest.assertTrue output.validate.toOption.isSome
    LeanTest.assertEqual (← output.toFloatArray) expected
  LeanTest.assertEqual (← lhs.toFloatArray) #[6, 12]
  LeanTest.assertEqual (← rhs.toFloatArray) #[2, 3]

-- The package's proof mentions the supplied tensors' actual metadata. It can
-- also be supplied as an erased typeclass witness to generic typed code.
example {s : Shape} {l r : DType} {dev : Device}
    (left : Tensor ⟨s, l, dev⟩) (right : Tensor ⟨s, r, dev⟩)
    (result : Tensor.Promoting.Promoted left.actualSpec.dtype right.actualSpec.dtype s dev) :
    DType.promote? left.actualSpec.dtype right.actualSpec.dtype = some result.dtype := by
  letI := result.witness
  exact DType.Promotes.valid

example {σ : StaticSpec} (input : Tensor σ)
    (result : Tensor.Promoting.ScalarPromoted input.actualSpec.dtype σ.shape σ.device) :
    DType.promoteScalar? input.actualSpec.dtype .Float32 = some result.dtype :=
  result.witness.valid

@[test] def testRuntimeMixedFloatPromotion : IO Unit := do
  let left := ((Tensor.fromFloatArray #[6, 12]).castDType .BFloat16)
  let right := (Tensor.fromFloatArray #[2, 3])
  for (expected, pending) in #[(#[8, 15], Tensor.Promoting.add? left right),
      (#[4, 9], Tensor.Promoting.sub? left right),
      (#[12, 36], Tensor.Promoting.mul? left right)] do
    let result ← checked pending
    LeanTest.assertEqual result.dtype .Float32
    LeanTest.assertEqual result.tensor.raw.dtype .Float32
    LeanTest.assertTrue result.tensor.validate.toOption.isSome
    LeanTest.assertEqual (← result.tensor.toFloatArray) expected
  let quotient ← checked (Tensor.Promoting.div? left right)
  LeanTest.assertEqual quotient.dtype .Float32
  LeanTest.assertEqual quotient.tensor.raw.dtype .Float32
  LeanTest.assertEqual (← quotient.tensor.toFloatArray) #[3, 4]
  let exact : Tensor ⟨#[2], .Float32, .CPU⟩ := Tensor.add left right
  LeanTest.assertEqual exact.raw.dtype .Float32
  LeanTest.assertEqual (← exact.toFloatArray) #[8, 15]
  LeanTest.assertEqual left.raw.dtype .BFloat16
  LeanTest.assertEqual (← left.toFloatArray) #[6, 12]
  LeanTest.assertEqual (← right.toFloatArray) #[2, 3]

@[test] def testRuntimeMixedIntegerPromotion : IO Unit := do
  let left := ((Tensor.fromFloatArray #[250, 1]).castDType .UInt8)
  let right := ((Tensor.fromFloatArray #[-1, 2]).castDType .Int8)
  let result ← checked (Tensor.Promoting.add? left right)
  LeanTest.assertEqual result.dtype .Int16
  LeanTest.assertEqual result.tensor.raw.dtype .Int16
  LeanTest.assertTrue result.tensor.validate.toOption.isSome
  LeanTest.assertEqual (← result.tensor.toFloatArray) #[249, 3]
  LeanTest.assertEqual left.raw.dtype .UInt8
  LeanTest.assertEqual right.raw.dtype .Int8

@[test] def testRuntimeIntegerDivisionPromotion : IO Unit := do
  let left := ((Tensor.fromFloatArray #[6, 12]).castDType .Int32)
  let right := ((Tensor.fromFloatArray #[2, 3]).castDType .Int64)
  let result ← checked (Tensor.Promoting.div? left right)
  -- Ordinary promotion is Int64, but true division produces Float32.
  LeanTest.assertEqual result.dtype .Int64
  LeanTest.assertEqual result.tensor.raw.dtype .Float32
  LeanTest.assertTrue result.tensor.validate.toOption.isSome
  LeanTest.assertEqual (← result.tensor.toFloatArray) #[3, 4]
  LeanTest.assertEqual (Tensor.div left right).raw.dtype .Float32
  LeanTest.assertEqual left.raw.dtype .Int32
  LeanTest.assertEqual right.raw.dtype .Int64

@[test] def testRuntimeHostFloatScalarPromotion : IO Unit := do
  let input := ((Tensor.fromFloatArray #[6, 12]).castDType .Int32)
  for (expected, pending) in #[(#[6.5, 12.5], Tensor.Promoting.addScalar? input 0.5),
      (#[5.5, 11.5], Tensor.Promoting.subScalar? input 0.5),
      (#[3, 6], Tensor.Promoting.mulScalar? input 0.5),
      (#[12, 24], Tensor.Promoting.divScalar? input 0.5)] do
    let result ← checked pending
    LeanTest.assertEqual result.dtype .Float32
    LeanTest.assertEqual result.tensor.raw.dtype .Float32
    LeanTest.assertTrue result.tensor.validate.toOption.isSome
    LeanTest.assertEqual (← result.tensor.toFloatArray) expected
  LeanTest.assertEqual (Tensor.addScalar input 0.5).raw.dtype .Float32
  LeanTest.assertEqual input.raw.dtype .Int32
  LeanTest.assertEqual (← input.toFloatArray) #[6, 12]

@[test] def testRuntimeUndefinedPromotionIsRejected : IO Unit := do
  let path := "Tests/fixtures/typed/dtype_scalars.safetensors"
  let floating := (Tensor.ones #[1])
  for dtype in #[DType.Float8E4M3FN, DType.Float8E5M2] do
    let stored ← safetensors.loadTensorWithSpec path dtype.canonicalName ⟨#[1], dtype⟩
    let exact := stored
    expectError (Tensor.Promoting.add? exact floating) "Unsupported dtype promotion"
    expectError (Tensor.Promoting.sub? exact floating) "Unsupported dtype promotion"
    expectError (Tensor.Promoting.mul? exact floating) "Unsupported dtype promotion"
    expectError (Tensor.Promoting.div? exact floating) "Unsupported dtype promotion"

@[test] def testRuntimeBoolSubtractionIsRejected : IO Unit := do
  let floating := (Tensor.ones #[2])
  let booleans := ((Tensor.ones #[2]).castDType .Bool)
  expectError (Tensor.Promoting.sub? booleans booleans) "Bool operands"
  expectError (Tensor.Promoting.sub? booleans floating) "Bool operands"
  expectError (Tensor.Promoting.sub? floating booleans) "Bool operands"
  expectError (Tensor.Promoting.subScalar? booleans 1) "Bool operands"

end Tests.DTypeConversions
