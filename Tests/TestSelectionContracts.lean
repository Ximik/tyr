import LeanTest
import Tyr.Model.Generation
import Tyr.Typed.Neural

namespace Tests.SelectionContracts

open torch

example (_complex : Tensor ⟨#[1, 1, 2], .Complex64, .CPU⟩)
    (_empty : Tensor ⟨#[1, 1, 0], .Float32, .CPU⟩)
    (_rows : Tensor ⟨#[1, 2], .Float32, .CPU⟩) : True := by
  fail_if_success have _ := Tensor.argmax _complex 2
  fail_if_success have _ := Tensor.argmax _rows 2
  fail_if_success have _ := Tensor.argmax _rows 18446744073709551615
  fail_if_success have _ := Tensor.maxLast3d _complex
  fail_if_success have _ := Tensor.maxLast3d _empty
  fail_if_success have _ := Tensor.topKLast2d _rows 3
  fail_if_success have _ := Tensor.topKLast2d _rows 18446744073709551615
  trivial

@[test] def testSelectionValuesAndDTypes : IO Unit := do
  let x := (Tensor.fromFloatArray #[1, 4, 2, 5, 3, 0]).reshape #[2, 3]
  let indices := Tensor.argmax x 1
  LeanTest.assertEqual (← indices.toFloatArray) #[1, 0]
  LeanTest.assertEqual indices.raw.dtype .Int64
  let (maximum, maximumIndices) := Tensor.maxLast3d (x.reshape #[1, 2, 3])
  LeanTest.assertEqual (← maximum.toFloatArray) #[4, 5]
  LeanTest.assertEqual (← maximumIndices.toFloatArray) #[1, 0]
  let (values, topIndices) := Tensor.topKLast2d x 2
  LeanTest.assertEqual (← values.toFloatArray) #[4, 2, 5, 3]
  LeanTest.assertEqual (← topIndices.toFloatArray) #[1, 2, 0, 1]
  let dynamic := (x.castDType .Float64)
  let (dynamicValues, _) := Tensor.topKLast2d dynamic 2
  LeanTest.assertEqual dynamicValues.raw.dtype .Float64
  let (empty, emptyIndices) := Tensor.topKLast2d (Tensor.zeros #[2, 0]) 0
  LeanTest.assertEqual empty.runtimeShape #[2, 0]
  LeanTest.assertEqual emptyIndices.raw.dtype .Int64
  let scalar := Tensor.argmax (Tensor.ones #[]) 0
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
    ((Tensor.fromFloatArray #[1, 4, 2]).reshape #[1, 3]) .greedy
  LeanTest.assertEqual (← tokens.toFloatArray) #[1]

end Tests.SelectionContracts
