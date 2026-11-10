import LeanTest
import Tyr.Typed.Neural
import Tyr.Model.BranchingFlows.TypedOps

namespace Tests.OperationContracts

open torch

private def expectOk {α : Type} (result : Except String α) : IO α :=
  match result with
  | .ok value => pure value
  | .error message => throw (IO.userError message)

private def expectError {α : Type} (result : Except String α) (fragment : String) : IO Unit :=
  match result with
  | .ok _ => LeanTest.fail s!"Expected rejection containing {fragment}"
  | .error message => LeanTest.assertTrue (message.containsSubstr fragment) message

-- Domain and axis requirements, one block per operation family.
example (_values : Tensor ⟨#[2], .Float32, .CPU⟩) (_complex : Tensor ⟨#[2], .Complex64, .CPU⟩) : True := by
  have _ := Tensor.relu _values
  fail_if_success have _ := Tensor.relu _complex
  trivial

example (_values : Tensor ⟨#[2], .Float32, .CPU⟩) : True := by
  have _ := Tensor.sumDim _values 0
  fail_if_success have _ := Tensor.sumDim _values 1
  trivial

example (_values : Tensor ⟨#[2], .Float32, .CPU⟩) (_ints : Tensor ⟨#[2], .Int64, .CPU⟩) : True := by
  have _ := Tensor.meanAll _values
  fail_if_success have _ := Tensor.meanAll _ints
  have _ := Tensor.meanDim _values 0
  fail_if_success have _ := Tensor.meanDim _values 1
  fail_if_success have _ := Tensor.meanDim _ints 0
  trivial

example (x : Tensor ⟨#[2], .Int64, .CPU⟩) (y : Tensor ⟨#[2], .Float32, .CPU⟩)
    (condition : Tensor ⟨#[2], .Bool, .CPU⟩) : Tensor ⟨#[2], .Float32, .CPU⟩ :=
  Tensor.whereSelect condition x y

example (x : Tensor ⟨#[2], .Int64, .CPU⟩) (scalar : Tensor ⟨#[], .Float32, .CPU⟩) :
    Tensor ⟨#[2], .Float32, .CPU⟩ := branching.mulTensorScalar x scalar

@[test] def testWherePromotesOperands : IO Unit := do
  let x := (Tensor.fromInt64Array #[1, 2])
  let y := (Tensor.fromFloatArray #[9.5, 8.5])
  let condition := Tensor.eqScalar (Tensor.fromInt64Array #[0, 0]) 1
  let result : Tensor ⟨#[2], .Float32, .CPU⟩ := Tensor.whereSelect condition x y
  LeanTest.assertEqual result.raw.dtype .Float32
  LeanTest.assertEqual (← result.toFloatArray) #[9.5, 8.5]

@[test] def testTensorScalarPromotesValues : IO Unit := do
  let x := Tensor.fromInt64Array #[2, 4]
  let scalar := (Tensor.fromFloatArray #[0.5]).reshape #[]
  let result := branching.mulTensorScalar x scalar
  LeanTest.assertEqual result.raw.dtype .Float32
  LeanTest.assertEqual (← result.toFloatArray) #[1, 2]

example (_empty : Tensor ⟨#[0], .Float32, .CPU⟩) (_complex : Tensor ⟨#[2], .Complex64, .CPU⟩)
    (_values : Tensor ⟨#[2], .Float32, .CPU⟩) : True := by
  have _ := Tensor.maxAll _values
  fail_if_success have _ := Tensor.maxAll _empty
  fail_if_success have _ := Tensor.maxAll _complex
  trivial

@[test] def testShapeErrorsAreRecoverable : IO Unit := do
  let x := Tensor.ones #[2]
  expectError (Tensor.catMany? #[x.packShape, x.packShape] 9 #[4]) "axis"
  expectError (Tensor.catMany? (d := .Float32) (dev := .CPU) #[] 0 #[0]) "at least one"
  let concatenated ← expectOk (Tensor.catMany? #[x.packShape, x.packShape] 0 #[4])
  LeanTest.assertEqual (← concatenated.toFloatArray) #[1, 1, 1, 1]
  let summed := Tensor.sumDim x 0
  LeanTest.assertEqual (← summed.toFloatArray) #[2]

@[test] def testOperatorDomains : IO Unit := do
  let complex := (Tensor.fromFloatArray #[2, 4]).castDType .Complex64
  let mean := Tensor.meanAll complex
  LeanTest.assertEqual mean.raw.dtype .Complex64
  LeanTest.assertEqual (← mean.toFloat32.toFloatArray) #[3]
  let maximum := Tensor.maxAll (Tensor.fromInt64Array #[2, 4])
  LeanTest.assertEqual maximum.itemInt 4

example (_left : Tensor ⟨#[2, 2], .BFloat16, .CPU⟩)
    (_right : Tensor ⟨#[2, 2], .Float32, .CPU⟩)
    (_input : Tensor ⟨#[1, 2, 2], .Float32, .CPU⟩)
    (_bias : Tensor ⟨#[2], .Float32, .CPU⟩)
    (_otherBias : Tensor ⟨#[2], .Float64, .CPU⟩) : True := by
  have _ := Tensor.mm _right _right
  fail_if_success have _ := Tensor.mm _left _right
  fail_if_success have _ := Tensor.linear _left _right
  have _ := Tensor.affine3d _input _right _bias
  fail_if_success have _ := Tensor.affine3d _input _left _bias
  fail_if_success have _ := Tensor.affine3d _input _right _otherBias
  trivial

@[test] def testMatrixActualDTypeEquality : IO Unit := do
  let left := (Tensor.ones #[2, 2]).toBFloat16
  let product := Tensor.mm left left
  LeanTest.assertEqual product.raw.dtype .BFloat16
  LeanTest.assertEqual (← product.toFloatArray) #[2, 2, 2, 2]

@[test] def testAffineActualDTypeEquality : IO Unit := do
  let input := (Tensor.full #[1, 2, 2] 3)
  let weight := (Tensor.full #[2, 2] 2)
  let bias := (Tensor.ones #[2])
  let result := Tensor.affine3d input weight bias
  LeanTest.assertEqual result.raw.dtype .Float32
  LeanTest.assertEqual (← result.toFloatArray) #[13, 13, 13, 13]

example (_values : Tensor ⟨#[2, 2], .Float32, .CPU⟩) (_ints : Tensor ⟨#[2, 2], .Int64, .CPU⟩)
    (_empty : Tensor ⟨#[0, 2], .Float32, .CPU⟩) : True := by
  have _ := Tensor.spectralNorm _values
  fail_if_success have _ := Tensor.spectralNorm _ints
  fail_if_success have _ := Tensor.spectralNorm _empty
  have _ := Tensor.nuclearNorm _values
  fail_if_success have _ := Tensor.nuclearNorm _ints
  have _ := Tensor.nuclearNorm _empty
  trivial

example {d : DType} [DType.FloatingOrComplex d]
    (input : Tensor ⟨#[2, 2], d, .CPU⟩) : Float :=
  Tensor.spectralNorm input + Tensor.nuclearNorm input

@[test] def testSingularValueNormDomains : IO Unit := do
  let real := (Tensor.fromFloatArray #[3, 0, 0, 4]).reshape #[2, 2]
  for dtype in #[DType.Float32, .Float64, .Complex64, .Complex128] do
    let input := real.castDType dtype
    LeanTest.assertEqual (Tensor.spectralNorm! input) 4
    LeanTest.assertEqual (Tensor.nuclearNorm! input) 7
  LeanTest.assertEqual (Tensor.nuclearNorm (Tensor.zeros #[0, 2])) 0

end Tests.OperationContracts
