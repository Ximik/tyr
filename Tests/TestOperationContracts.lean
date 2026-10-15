import LeanTest
import Tyr.Typed.Neural
import Tyr.Model.BranchingFlows.TypedOps

namespace Tests.OperationContracts

open torch

private def checked {α : Type} (result : Except String α) : IO α :=
  match result with
  | .ok value => pure value
  | .error message => throw (IO.userError message)

private def expectError {α : Type} (result : Except String α) (fragment : String) : IO Unit :=
  match result with
  | .ok _ => LeanTest.fail s!"Expected rejection containing {fragment}"
  | .error message => LeanTest.assertTrue (message.containsSubstr fragment) message

-- Invalid domains must be rejected without evaluating a native operation.
example (_complex : Tensor ⟨#[2], .Complex64, .any⟩)
    (_empty : Tensor ⟨#[0], .Float32, .any⟩)
    (_vector : Tensor ⟨#[2], .Float32, .any⟩)
    (_integer : Tensor ⟨#[2], .Int64, .any⟩) : True := by
  fail_if_success have _ := Tensor.relu _complex
  fail_if_success have _ := Tensor.maxAll _empty
  fail_if_success have _ := Tensor.sumDim _vector 1
  fail_if_success have _ := Tensor.meanAll _integer
  trivial

example (x : Tensor ⟨#[2], .Int64, .any⟩) (y : Tensor ⟨#[2], .Float32, .any⟩)
    (condition : Tensor ⟨#[2], .Bool, .any⟩) : Tensor ⟨#[2], .Float32, .any⟩ :=
  Tensor.whereSelect condition x y

example (x : Tensor ⟨#[2], .Int64, .any⟩) (scalar : Tensor ⟨#[], .Float32, .any⟩) :
    Tensor ⟨#[2], .Float32, .any⟩ := branching.mulTensorScalar x scalar

@[test] def testWherePromotesErasedOperands : IO Unit := do
  let x := (Tensor.fromInt64Array #[1, 2]).forgetDType
  let y := (Tensor.fromFloatArray #[9.5, 8.5]).forgetDType
  let condition := Tensor.eqScalar (Tensor.fromInt64Array #[0, 0]) 1
  let result := Tensor.Promoting.whereSelect condition x y
  LeanTest.assertEqual result.raw.dtype .Float32
  LeanTest.assertEqual (← result.toFloatArray) #[9.5, 8.5]

@[test] def testTensorScalarPromotesValues : IO Unit := do
  let x := Tensor.fromInt64Array #[2, 4]
  let scalar := (Tensor.fromFloatArray #[0.5]).reshapeChecked #[]
  let result := branching.mulTensorScalar x scalar
  LeanTest.assertEqual result.raw.dtype .Float32
  LeanTest.assertEqual (← result.toFloatArray) #[1, 2]

@[test] def testCheckedShapeErrorsAreRecoverable : IO Unit := do
  let x := Tensor.ones #[2]
  expectError (Tensor.expand? x #[3]) "expand"
  expectError (Tensor.catMany? #[x.packShape, x.packShape] 9 #[4]) "Dimension"
  expectError (Tensor.catMany? (d := .Float32) (dev := .any) #[] 0 #[0]) "at least one"
  expectError (Tensor.sumDim? x 1) "axis"
  expectError (Tensor.Preserving.sumDim? x 1) "axis"
  expectError (Tensor.maxAll? (Tensor.zeros #[0])) "nonempty"
  let expanded ← checked (Tensor.expand? (Tensor.ones #[1]) #[3])
  LeanTest.assertEqual (← expanded.toFloatArray) #[1, 1, 1]
  let concatenated ← checked (Tensor.catMany? #[x.packShape, x.packShape] 0 #[4])
  LeanTest.assertEqual (← concatenated.toFloatArray) #[1, 1, 1, 1]
  let summed ← checked (Tensor.sumDim? x 0)
  LeanTest.assertEqual (← summed.toFloatArray) #[2]

@[test] def testCheckedOperatorDomains : IO Unit := do
  let complex := (Tensor.fromFloatArray #[2, 4]).castDType .Complex64
  expectError (Tensor.relu? complex) "real numeric"
  expectError (Tensor.maxAll? complex) "real or Bool"
  let integer := (Tensor.fromInt64Array #[2, 4]).forgetDType
  expectError (Tensor.Preserving.meanDim? integer 0) "floating or complex"
  expectError (Tensor.meanAll? integer) "floating or complex"
  let mean := Tensor.meanAll complex
  LeanTest.assertEqual mean.raw.dtype .Complex64
  LeanTest.assertEqual (← mean.toFloat32.toFloatArray) #[3]
  let maximum ← checked (Tensor.maxAll? (Tensor.fromInt64Array #[2, 4]))
  LeanTest.assertEqual maximum.itemInt 4

@[test] def testMatrixActualDTypeEquality : IO Unit := do
  let left := (Tensor.ones #[2, 2]).toBFloat16.forgetDType
  let right := (Tensor.ones #[2, 2]).forgetDType
  expectError (Tensor.mm? left right) "Expected dtype BFloat16, got Float32"
  expectError (Tensor.matmul? left right) "Expected dtype BFloat16, got Float32"
  expectError (Tensor.linear? left right) "Expected dtype BFloat16, got Float32"
  let product ← checked (Tensor.mm? left left)
  LeanTest.assertEqual product.raw.dtype .BFloat16
  LeanTest.assertEqual (← product.toFloatArray) #[2, 2, 2, 2]

@[test] def testMissingKernelIsRecoverable : IO Unit := do
  let x := (Tensor.ones #[2]).castDType .Complex32
  -- Promotion exists, but the pinned CPU backend lacks ComplexHalf division.
  expectError (Tensor.Promoting.div? x x) "div"
  expectError (Tensor.Preserving.div? x x) "div"

@[test] def testRuntimeBroadcastPromotion : IO Unit := do
  let x := Tensor.fromInt64Array #[2, 4]
  let scalar := (Tensor.fromFloatArray #[0.5]).reshapeChecked #[]
  let result ← checked (Tensor.Promoting.mulB? x scalar #[2])
  LeanTest.assertEqual result.dtype .Float32
  LeanTest.assertEqual (← result.tensor.toFloatArray) #[1, 2]
  expectError (Tensor.Promoting.mulB? x scalar #[3]) "broadcast"
  expectError (branching.mulTensorScalarPreserving? x scalar) "Expected dtype Int64"

end Tests.OperationContracts
