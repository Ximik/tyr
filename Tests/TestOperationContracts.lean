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
example (_complex : Tensor ⟨#[2], .Complex64, .CPU⟩)
    (_empty : Tensor ⟨#[0], .Float32, .CPU⟩)
    (_vector : Tensor ⟨#[2], .Float32, .CPU⟩)
    (_integer : Tensor ⟨#[2], .Int64, .CPU⟩) : True := by
  fail_if_success have _ := Tensor.relu _complex
  fail_if_success have _ := Tensor.maxAll _empty
  fail_if_success have _ := Tensor.sumDim _vector 1
  fail_if_success have _ := Tensor.meanAll _integer
  trivial

example (x : Tensor ⟨#[2], .Int64, .CPU⟩) (y : Tensor ⟨#[2], .Float32, .CPU⟩)
    (condition : Tensor ⟨#[2], .Bool, .CPU⟩) : Tensor ⟨#[2], .Float32, .CPU⟩ :=
  Tensor.whereSelect condition x y

example (x : Tensor ⟨#[2], .Int64, .CPU⟩) (scalar : Tensor ⟨#[], .Float32, .CPU⟩) :
    Tensor ⟨#[2], .Float32, .CPU⟩ := branching.mulTensorScalar x scalar

@[test] def testWherePromotesErasedOperands : IO Unit := do
  let x := (Tensor.fromInt64Array #[1, 2])
  let y := (Tensor.fromFloatArray #[9.5, 8.5])
  let condition := Tensor.eqScalar (Tensor.fromInt64Array #[0, 0]) 1
  let promoted ← checked (Tensor.Promoting.whereSelect? condition x y)
  let result := promoted.tensor
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
  expectError (Tensor.catMany? (d := .Float32) (dev := .CPU) #[] 0 #[0]) "at least one"
  expectError (Tensor.sumDim? x 1) "axis"
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
  let integer := (Tensor.fromInt64Array #[2, 4])
  expectError (Tensor.meanDim? integer 0) "floating or complex"
  expectError (Tensor.meanAll? integer) "floating or complex"
  let mean := Tensor.meanAll complex
  LeanTest.assertEqual mean.raw.dtype .Complex64
  LeanTest.assertEqual (← mean.toFloat32.toFloatArray) #[3]
  let maximum ← checked (Tensor.maxAll? (Tensor.fromInt64Array #[2, 4]))
  LeanTest.assertEqual maximum.itemInt 4

example (_left : Tensor ⟨#[2, 2], .BFloat16, .CPU⟩)
    (_right : Tensor ⟨#[2, 2], .Float32, .CPU⟩)
    (_input : Tensor ⟨#[1, 2, 2], .Float32, .CPU⟩)
    (_bias : Tensor ⟨#[2], .Float32, .CPU⟩)
    (_otherBias : Tensor ⟨#[2], .Float64, .CPU⟩) : True := by
  fail_if_success have _ := Tensor.mm? _left _right
  fail_if_success have _ := Tensor.matmul? _left _right
  fail_if_success have _ := Tensor.linear? _left _right
  fail_if_success have _ := Tensor.affine3d? _input _left _bias
  fail_if_success have _ := Tensor.affine3d? _input _right _otherBias
  trivial

@[test] def testMatrixActualDTypeEquality : IO Unit := do
  let left := (Tensor.ones #[2, 2]).toBFloat16
  let product ← checked (Tensor.mm? left left)
  LeanTest.assertEqual product.raw.dtype .BFloat16
  LeanTest.assertEqual (← product.toFloatArray) #[2, 2, 2, 2]

@[test] def testAffineActualDTypeEquality : IO Unit := do
  let input := (Tensor.full #[1, 2, 2] 3)
  let weight := (Tensor.full #[2, 2] 2)
  let bias := (Tensor.ones #[2])
  let result ← checked (Tensor.affine3d? input weight bias)
  LeanTest.assertEqual result.raw.dtype .Float32
  LeanTest.assertEqual (← result.toFloatArray) #[13, 13, 13, 13]

@[test] def testAffineMissingKernelIsRecoverable : IO Unit := do
  let input := (Tensor.ones #[1, 2, 2]).castDType .Bool
  let weight := (Tensor.ones #[2, 2]).castDType .Bool
  let bias := (Tensor.ones #[2]).castDType .Bool
  expectError (Tensor.affine3d? input weight bias) "affine3d"

example (_input : Tensor ⟨#[2, 2], .Int64, .CPU⟩) : True := by
  fail_if_success have _ := Tensor.spectralNorm _input
  fail_if_success have _ := Tensor.nuclearNorm _input
  trivial

example {d : DType} [DType.FloatingOrComplex d]
    (input : Tensor ⟨#[2, 2], d, .CPU⟩) : Float :=
  Tensor.spectralNorm input + Tensor.nuclearNorm input

@[test] def testSingularValueNormDomains : IO Unit := do
  let real := (Tensor.fromFloatArray #[3, 0, 0, 4]).reshapeChecked #[2, 2]
  for dtype in #[DType.Float32, .Float64, .Complex64, .Complex128] do
    let input := real.castDType dtype
    LeanTest.assertEqual (← checked (Tensor.spectralNorm? input)) 4
    LeanTest.assertEqual (← checked (Tensor.nuclearNorm? input)) 7
  let integers := (real.castDType .Int64)
  expectError (Tensor.spectralNorm? integers) "floating or complex"
  expectError (Tensor.nuclearNorm? integers) "floating or complex"
  let halfComplex := real.castDType .Complex32
  expectError (Tensor.spectralNorm? halfComplex) "spectralNorm"
  expectError (Tensor.nuclearNorm? halfComplex) "nuclearNorm"
  expectError (Tensor.spectralNorm? (Tensor.zeros #[0, 2])) "nonempty"
  LeanTest.assertEqual (← checked (Tensor.nuclearNorm? (Tensor.zeros #[0, 2]))) 0

@[test] def testMissingKernelIsRecoverable : IO Unit := do
  let x := (Tensor.ones #[2]).castDType .Complex32
  -- Promotion exists, but the pinned CPU backend lacks ComplexHalf division.
  expectError (Tensor.Promoting.div? x x) "div"
  expectError (Tensor.div? x x) "div"

@[test] def testRuntimeBroadcastPromotion : IO Unit := do
  let x := Tensor.fromInt64Array #[2, 4]
  let scalar := (Tensor.fromFloatArray #[0.5]).reshapeChecked #[]
  let result ← checked (Tensor.Promoting.mulB? x scalar #[2])
  LeanTest.assertEqual result.dtype .Float32
  LeanTest.assertEqual (← result.tensor.toFloatArray) #[1, 2]
  expectError (Tensor.Promoting.mulB? x scalar #[3]) "broadcast"
  let native : Tensor ⟨#[2], .Float32, .CPU⟩ := branching.mulTensorScalar x scalar
  LeanTest.assertEqual native.raw.dtype .Float32
  LeanTest.assertEqual (← native.toFloatArray) #[1, 2]

end Tests.OperationContracts
