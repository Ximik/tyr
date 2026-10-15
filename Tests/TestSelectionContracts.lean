import LeanTest
import Tyr.Model.Generation
import Tyr.Typed.Neural

namespace Tests.SelectionContracts

open torch

private def checked {α : Type} (result : Except String α) : IO α :=
  IO.ofExcept result

private def expectError {α : Type} (result : Except String α) (fragment : String) : IO Unit :=
  match result with
  | .ok _ => LeanTest.fail s!"Expected selection error containing {fragment}"
  | .error message => LeanTest.assertTrue (message.containsSubstr fragment) message

example (_complex : Tensor ⟨#[1, 1, 2], .Complex64, .CPU⟩)
    (_empty : Tensor ⟨#[1, 1, 0], .Float32, .CPU⟩)
    (_rows : Tensor ⟨#[1, 2], .Float32, .CPU⟩) : True := by
  fail_if_success have _ := Tensor.argmax _complex 2
  fail_if_success have _ := Tensor.argmax _rows 2
  fail_if_success have _ := Tensor.maxLast3d _complex
  fail_if_success have _ := Tensor.maxLast3d _empty
  fail_if_success have _ := Tensor.topKLast2d _rows 3
  trivial

@[test] def testSelectionDomainsAndRangesAreRecoverable : IO Unit := do
  let complex := (Tensor.ones #[1, 1, 2]).castDType .Complex64
  expectError (Tensor.argmax? complex 2) "real or Bool"
  expectError (Tensor.maxLast3d? complex) "real or Bool"
  expectError (Tensor.topKLast2d? (complex.reshapeChecked #[1, 2]) 1) "real or Bool"
  expectError (Tensor.argmax? (Tensor.ones #[1, 2]) 2) "axis"
  expectError (Tensor.argmax? (Tensor.ones #[1, 2]) 18446744073709551615) "axis"
  expectError (Tensor.argmax? (Tensor.zeros #[1, 0]) 1) "nonempty"
  expectError (Tensor.maxLast3d? (Tensor.zeros #[1, 1, 0])) "nonempty"
  expectError (Tensor.topKLast2d? (Tensor.ones #[1, 2]) 3) "exceeds"
  expectError (Tensor.topKLast2d? (Tensor.ones #[1, 2]) 18446744073709551615) "exceeds"

@[test] def testSelectionValuesAndDTypes : IO Unit := do
  let x := (Tensor.fromFloatArray #[1, 4, 2, 5, 3, 0]).reshapeChecked #[2, 3]
  let indices ← checked (Tensor.argmax? x 1)
  LeanTest.assertEqual (← indices.toFloatArray) #[1, 0]
  LeanTest.assertEqual indices.raw.dtype .Int64
  let (maximum, maximumIndices) ← checked (Tensor.maxLast3d? (x.reshapeChecked #[1, 2, 3]))
  LeanTest.assertEqual (← maximum.toFloatArray) #[4, 5]
  LeanTest.assertEqual (← maximumIndices.toFloatArray) #[1, 0]
  let (values, topIndices) ← checked (Tensor.topKLast2d? x 2)
  LeanTest.assertEqual (← values.toFloatArray) #[4, 2, 5, 3]
  LeanTest.assertEqual (← topIndices.toFloatArray) #[1, 2, 0, 1]
  let dynamic := (x.castDType .Float64).forgetDType
  let (dynamicValues, _) ← checked (Tensor.topKLast2d? dynamic 2)
  LeanTest.assertEqual dynamicValues.raw.dtype .Float64
  let (empty, emptyIndices) ← checked (Tensor.topKLast2d? (Tensor.zeros #[2, 0]) 0)
  LeanTest.assertEqual empty.runtimeShape #[2, 0]
  LeanTest.assertEqual emptyIndices.raw.dtype .Int64
  let scalar ← checked (Tensor.argmax? (Tensor.ones #[]) 0)
  LeanTest.assertEqual (← scalar.toFloatArray) #[0]

@[test] def testGreedySamplingEmptyVocabularyIsRecoverable : IO Unit := do
  let failed ← try
    let _ ← Model.sampleFromLogits (Tensor.zeros #[1, 0]) .greedy
    pure false
  catch error =>
    LeanTest.assertTrue (error.toString.containsSubstr "nonempty") error.toString
    pure true
  LeanTest.assertTrue failed "Empty-vocabulary sampling must return an IO error"
  let tokens ← Model.sampleFromLogits
    ((Tensor.fromFloatArray #[1, 4, 2]).reshapeChecked #[1, 3]) .greedy
  LeanTest.assertEqual (← tokens.toFloatArray) #[1]

end Tests.SelectionContracts
