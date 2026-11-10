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

example (_logits : Tensor ⟨#[1, 2, 2], .Float32, .CPU⟩)
    (_targets : Tensor ⟨#[1, 2], .Int64, .CPU⟩)
    (_mask : Tensor ⟨#[1, 2], .Complex64, .CPU⟩) : True := by
  fail_if_success have _ := Train.ChatSFT.maskedCrossEntropy _logits _targets _mask
  trivial

@[test] def testCrossEntropyFloatingDTypes : IO Unit := do
  let logits := Tensor.zeros #[2, 2]
  let targets := Tensor.fromInt64Array #[0, 1]
  for dtype in #[DType.Float32, .Float64] do
    let input := logits.castDType dtype
    let mean := Tensor.crossEntropy! input targets
    let losses := Tensor.crossEntropyNone! input targets
    LeanTest.assertEqual mean.raw.dtype dtype
    LeanTest.assertEqual losses.raw.dtype dtype
    LeanTest.assertTrue (Float.abs (mean.item! - Float.log 2) < 1e-6)
    let values ← losses.toFloatArray
    LeanTest.assertEqual values.size 2
    for value in values do
      LeanTest.assertTrue (Float.abs (value - Float.log 2) < 1e-6)

@[test] def testCrossEntropyIgnoreIndexAndGradient : IO Unit := do
  let logits ← (Tensor.zeros #[2, 2]).parameter
  let targets := Tensor.fromInt64Array #[-100, 1]
  let losses := Tensor.crossEntropyNone! logits targets
  let values ← losses.toFloatArray
  LeanTest.assertEqual values[0]! 0
  let mean := Tensor.crossEntropy! logits targets
  LeanTest.assertTrue (Float.abs (mean.item - Float.log 2) < 1e-6)
  Tensor.backwardLoss mean
  LeanTest.assertEqual (← (← logits.gradOf).toFloatArray) #[0, 0, 0.5, -0.5]

@[test] def testMaskedLossRejectsInvalidMasks : IO Unit := do
  let logits := Tensor.zeros #[1, 2, 2]
  let targets := (Tensor.fromInt64Array #[0, 1]).reshape #[1, 2]
  let mask := Tensor.ones #[1, 2]
  expectError (Train.ChatSFT.maskedCrossEntropy? logits targets (mask.castDType .Complex64))
  let loss ← checked (Train.ChatSFT.maskedCrossEntropy? logits targets mask)
  LeanTest.assertTrue (Float.abs (loss.item - Float.log 2) < 1e-6)
  let boolMask := mask.castDType .Bool
  let boolLoss ← checked (Train.ChatSFT.maskedCrossEntropy? logits targets boolMask)
  LeanTest.assertTrue (Float.abs (boolLoss.item - Float.log 2) < 1e-6)
  let empty := Tensor.zeros #[1, 0, 2]
  let emptyTargets := (Tensor.fromInt64Array #[]).reshape #[1, 0]
  let emptyLoss ← checked (Train.ChatSFT.maskedCrossEntropy? empty emptyTargets (Tensor.zeros #[1, 0]))
  LeanTest.assertEqual emptyLoss.item 0

@[test] def testSftGradAccumStepsRequireAFullPass : IO Unit := do
  let cfg : torch.Train.ChatSFT.SFTConfig := { deviceBatchSize := 4, targetExamplesPerStep := 32 }
  match cfg.gradAccumSteps 2 with
  | .ok steps => LeanTest.assertEqual steps.val 4 "32 examples over 4 per device on 2 ranks"
  | .error e => throw <| IO.userError s!"A full step should be accepted: {e}"
  match cfg.gradAccumSteps 16 with
  | .ok steps => throw <| IO.userError s!"64 examples per pass exceed a 32-example step, got {steps.val}"
  | .error e => LeanTest.assertTrue (e.contains "targetExamplesPerStep") e

@[test] def testSftValidationLossIsNoneWithoutValidTokens : IO Unit := do
  let batch (mask : Tensor ⟨#[1, 2], .Float32, .CPU⟩) (valid : Nat) :
      torch.Train.ChatSFT.SFTBatch (device := .CPU) :=
    { batchSize := 1, seqLen := 2, inputs := Tensor.fullInt #[1, 2] 0,
      targets := Tensor.fullInt #[1, 2] 0, mask, numValidTokens := valid }
  let params : Tensor ⟨#[1], .Float32, .CPU⟩ := Tensor.zeros #[1]
  let uniform := fun (_ : Tensor ⟨#[1], .Float32, .CPU⟩) {b s : Nat} (_ : Tensor ⟨#[b, s], .Int64, .CPU⟩) =>
    (pure ⟨3, Tensor.zeros #[b, s, 3]⟩ : IO (Σ v, Tensor ⟨#[b, s, v], .Float32, .CPU⟩))
  let empty ← torch.Train.ChatSFT.evalValidationLoss (device := .CPU) params
    (pure (batch (Tensor.zeros #[1, 2]) 0)) uniform 2
  LeanTest.assertTrue empty.isNone "a pass without valid tokens has no loss"
  let some loss ← torch.Train.ChatSFT.evalValidationLoss (device := .CPU) params
      (pure (batch (Tensor.ones #[1, 2]) 2)) uniform 2
    | LeanTest.fail "a pass with valid tokens has a loss"
  LeanTest.assertTrue ((loss - Float.log 3.0).abs < 1e-5) s!"uniform logits over 3 classes give log 3, got {loss}"

end Tests.LossContracts
