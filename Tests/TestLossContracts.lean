import LeanTest
import Tyr.Typed.Neural
import Tyr.Train.ChatSFT

open torch

namespace Tests.LossContracts

private def checked {α : Type} (result : Except String α) : IO α :=
  IO.ofExcept result

private def expectError {α : Type} (result : Except String α) : IO Unit :=
  match result with
  | .error _ => pure ()
  | .ok _ => LeanTest.fail "Invalid cross entropy input was accepted"

example (_logits : Tensor ⟨#[2, 3], .Int64, .CPU⟩)
    (_targets : Tensor ⟨#[2], .Int64, .CPU⟩) : True := by
  fail_if_success have _ := Tensor.crossEntropy _logits _targets
  fail_if_success have _ := Tensor.crossEntropyNone _logits _targets
  trivial

example (_logits : Tensor ⟨#[1, 2, 2], .Float32, .CPU⟩)
    (_targets : Tensor ⟨#[1, 2], .Int64, .CPU⟩)
    (_mask : Tensor ⟨#[1, 2], .Complex64, .CPU⟩) : True := by
  fail_if_success have _ := Train.ChatSFT.maskedCrossEntropy _logits _targets _mask
  trivial

@[test] def testCrossEntropyDomainsAndTargetBounds : IO Unit := do
  let logits := Tensor.zeros #[2, 2]
  let targets := Tensor.fromInt64Array #[0, 1]
  for dtype in #[DType.Int64, .Bool, .Complex64] do
    let invalid := (logits.castDType dtype)
    expectError (Tensor.crossEntropy? invalid targets)
    expectError (Tensor.crossEntropyNone? invalid targets)
  for labels in #[#[0, 2], #[-1, 1]] do
    let invalid := Tensor.fromInt64Array labels
    -- Re-admission retains the common batch index for runtime-provided labels.
    let invalid ← checked (Tensor.ofRaw (σ := ⟨#[2], .Int64, .CPU⟩) invalid.raw)
    expectError (Tensor.crossEntropy? logits invalid)
    expectError (Tensor.crossEntropyNone? logits invalid)
  expectError (Tensor.crossEntropy? (Tensor.zeros #[2, 0]) targets)
  for dtype in #[DType.Float32, .Float64] do
    let input := logits.castDType dtype
    let mean ← checked (Tensor.crossEntropy? input targets)
    let losses ← checked (Tensor.crossEntropyNone? input targets)
    LeanTest.assertEqual mean.raw.dtype dtype
    LeanTest.assertEqual losses.raw.dtype dtype
    LeanTest.assertTrue (Float.abs (mean.item - Float.log 2) < 1e-6)
    let values ← losses.toFloatArray
    LeanTest.assertEqual values.size 2
    for value in values do
      LeanTest.assertTrue (Float.abs (value - Float.log 2) < 1e-6)

@[test] def testCrossEntropyIgnoreIndexAndGradient : IO Unit := do
  let logits ← (Tensor.zeros #[2, 2]).parameter
  let targets := Tensor.fromInt64Array #[-100, 1]
  let losses ← checked (Tensor.crossEntropyNone? logits targets)
  let values ← losses.toFloatArray
  LeanTest.assertEqual values[0]! 0
  let mean ← checked (Tensor.crossEntropy? logits targets)
  LeanTest.assertTrue (Float.abs (mean.item - Float.log 2) < 1e-6)
  Tensor.backwardLoss mean
  LeanTest.assertEqual (← (← logits.gradOf).toFloatArray) #[0, 0, 0.5, -0.5]

@[test] def testMaskedLossRejectsInvalidMasksAndLabels : IO Unit := do
  let logits := Tensor.zeros #[1, 2, 2]
  let targets := (Tensor.fromInt64Array #[0, 1]).reshapeChecked #[1, 2]
  let mask := Tensor.ones #[1, 2]
  expectError (Train.ChatSFT.maskedCrossEntropy? logits targets (mask.castDType .Complex64))
  let invalid := (Tensor.fromInt64Array #[0, 2]).reshapeChecked #[1, 2]
  expectError (Train.ChatSFT.maskedCrossEntropy? logits invalid mask)
  let loss ← checked (Train.ChatSFT.maskedCrossEntropy? logits targets mask)
  LeanTest.assertTrue (Float.abs (loss.item - Float.log 2) < 1e-6)
  let boolMask := mask.castDType .Bool
  let boolLoss ← checked (Train.ChatSFT.maskedCrossEntropy? logits targets boolMask)
  LeanTest.assertTrue (Float.abs (boolLoss.item - Float.log 2) < 1e-6)
  let empty := Tensor.zeros #[1, 0, 2]
  let emptyTargets := (Tensor.fromInt64Array #[]).reshapeChecked #[1, 0]
  let emptyLoss ← checked (Train.ChatSFT.maskedCrossEntropy? empty emptyTargets (Tensor.zeros #[1, 0]))
  LeanTest.assertEqual emptyLoss.item 0

end Tests.LossContracts
