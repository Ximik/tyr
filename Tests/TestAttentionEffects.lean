import LeanTest
import Tests.Assertions
import Tyr.Typed.Neural
import Tyr.Torch
import Tyr.GPU.Ops.FlashAttn

namespace Tests.AttentionEffects

open torch

private abbrev AttentionTensor := Tensor ⟨#[1, 1, 16, 4], .Float32, .CPU⟩

@[inline] private def draw (q v : AttentionTensor) : IO AttentionTensor :=
  Tensor.attentionIO q q v 0.5 true false

@[test] def testAttentionDrawsSequenceAndReplay : IO Unit := do
  let q : AttentionTensor := Tensor.zeros #[1, 1, 16, 4]
  let v : AttentionTensor := Tensor.ones #[1, 1, 16, 4]
  manualSeed 1729
  let first ← draw q v
  let second ← draw q v
  let firstValues ← first.toFloatArray
  let secondValues ← second.toFloatArray
  LeanTest.assertTrue (firstValues != secondValues)
    "Identical IO attention calls must draw separate dropout masks"
  let after ← Tensor.randn #[8]
  manualSeed 1729
  let replayFirst ← draw q v
  let replaySecond ← draw q v
  let replayAfter ← Tensor.randn #[8]
  LeanTest.assertEqual (← replayFirst.toFloatArray) firstValues
  LeanTest.assertEqual (← replaySecond.toFloatArray) secondValues
  LeanTest.assertEqual (← replayAfter.toFloatArray) (← after.toFloatArray)

@[test] def testReusableAttentionActionDrawsFreshMasks : IO Unit := do
  let q : AttentionTensor := Tensor.zeros #[1, 1, 16, 4]
  let v : AttentionTensor := Tensor.ones #[1, 1, 16, 4]
  let action := draw q v
  manualSeed 4104
  let first ← action
  let second ← action
  LeanTest.assertTrue ((← first.toFloatArray) != (← second.toFloatArray))
    "Reusing an IO action must execute the native draw again"

@[test] def testInferenceAttentionDoesNotConsumeRNG : IO Unit := do
  let q : AttentionTensor := Tensor.zeros #[1, 1, 16, 4]
  let v : AttentionTensor := Tensor.ones #[1, 1, 16, 4]
  let mask := (Tensor.ones #[1, 16]).castDType .Bool
  manualSeed 42
  let expectedNext ← Tensor.randn #[8]
  manualSeed 42
  let pureOutput ← IO.ofExcept (Tensor.attention? q q v false)
  let outputs ← #[
    Tensor.attentionIO q q v 1 false false,
    Tensor.attentionGQAIO q q v 1 false false,
    Tensor.attentionGQAMaskIO q q v mask 1 false false,
    Tensor.attentionGQAQKVIO q q v 1 false false,
    Tensor.attentionWindowIO q q v 1 false true false 8,
    Tensor.flashAttentionIO q q v 1 false].mapM id
  for output in outputs do
    LeanTest.assertEqual (← output.toFloatArray) (← pureOutput.toFloatArray)
  let actualNext ← Tensor.randn #[8]
  LeanTest.assertEqual (← actualNext.toFloatArray) (← expectedNext.toFloatArray)

@[test] def testAttentionDropoutGraphRemainsReusable : IO Unit := do
  let q : AttentionTensor := Tensor.zeros #[1, 1, 16, 4]
  let v ← (Tensor.ones #[1, 1, 16, 4]).parameter
  manualSeed 1729
  let output ← draw q v
  let loss := Tensor.sumAll output
  manualSeed 42
  let expectedNext ← Tensor.randn #[8]
  manualSeed 42
  Tensor.backwardLoss loss (retainGraph := some true)
  let firstGradient ← v.gradOf
  let firstValues ← firstGradient.toFloatArray
  LeanTest.assertTrue (firstValues.any (· > 0))
  v.zeroGrad
  Tensor.backwardLoss loss (retainGraph := some true)
  LeanTest.assertEqual (← (← v.gradOf).toFloatArray) firstValues
  let actualNext ← Tensor.randn #[8]
  LeanTest.assertEqual (← actualNext.toFloatArray) (← expectedNext.toFloatArray)
    "Backward must reuse the saved mask without sampling"
  let newer ← draw q v
  v.zeroGrad
  Tensor.backwardLoss (Tensor.sumAll newer)
  LeanTest.assertTrue ((← (← v.gradOf).toFloatArray) != firstValues)
  v.zeroGrad
  Tensor.backwardLoss loss
  LeanTest.assertEqual (← (← v.gradOf).toFloatArray) firstValues
    "A later forward must preserve a surviving graph's sampled mask"
  LeanTest.assertEqual (← firstGradient.toFloatArray) firstValues
  Tests.assertThrows (Tensor.backwardLoss loss) "backward through the graph a second time"

@[test] def testAttentionProbabilityErrorsAreRecoverable : IO Unit := do
  let q : AttentionTensor := Tensor.zeros #[1, 1, 16, 4]
  let v : AttentionTensor := Tensor.ones #[1, 1, 16, 4]
  let mask := (Tensor.ones #[1, 16]).castDType .Bool
  for training in #[false, true] do
    for p in #[-0.5, 1.5, (0.0 / 0.0), (1.0 / 0.0), (-1.0 / 0.0)] do
      LeanTest.assertTrue (match Tensor.attentionDropout? p training with
        | .error _ => true
        | .ok _ => false)
        "Invalid configured probabilities must be rejected before selecting evaluation mode"
      let actions : Array (IO AttentionTensor) := #[
        Tensor.attentionIO q q v p training false,
        Tensor.attentionGQAIO q q v p training false,
        Tensor.attentionGQAMaskIO q q v mask p training false,
        Tensor.attentionGQAQKVIO q q v p training false,
        Tensor.attentionWindowIO q q v p training true false 8,
        Tensor.flashAttentionIO q q v p training]
      for action in actions do
        let rejected ← try
          let _ ← action
          pure false
        catch error =>
          pure (error.toString.containsSubstr "dropout")
        LeanTest.assertTrue rejected
          "Every attention IO adapter must reject invalid dropout in both modes"
  LeanTest.assertEqual (← IO.ofExcept (Tensor.attentionDropout? 0.5 false)) 0.0
  LeanTest.assertEqual (← IO.ofExcept (Tensor.attentionDropout? 0.5 true)) 0.5
  let dropped ← Tensor.attentionIO q q v 1 true false
  LeanTest.assertEqual (← dropped.toFloatArray) (Array.replicate 64 0)
  let valid ← Tensor.attentionIO q q v 0 true false
  LeanTest.assertEqual (← valid.toFloatArray) (Array.replicate 64 1)

@[test] def testFlashAdaptersHonorTrainingMode : IO Unit := do
  let q : AttentionTensor := Tensor.zeros #[1, 1, 16, 4]
  let v : AttentionTensor := Tensor.ones #[1, 1, 16, 4]
  let (pureRoute, pureOutput) := Tyr.GPU.Ops.FlashAttn.flashAttnWithRoute q q v
  let (evalRoute, evalOutput) ← Tyr.GPU.Ops.FlashAttn.flashAttnWithRouteIO q q v 1 false
  LeanTest.assertTrue (evalRoute == pureRoute)
  LeanTest.assertEqual (← evalOutput.toFloatArray) (← pureOutput.toFloatArray)
  let (_, trainingOutput) ← Tyr.GPU.Ops.FlashAttn.flashAttnWithRouteIO q q v 1 true
  LeanTest.assertEqual (← trainingOutput.toFloatArray) (Array.replicate 64 0)
  let dynEval ← Tyr.GPU.Ops.FlashAttn.flashAttnDyn q.raw q.raw v.raw 1 false
  LeanTest.assertEqual (← dynEval.2.2.2.2.2.toFloatArray) (← pureOutput.toFloatArray)
  let dynTraining ← Tyr.GPU.Ops.FlashAttn.flashAttnDyn q.raw q.raw v.raw 1 true
  LeanTest.assertEqual (← dynTraining.2.2.2.2.2.toFloatArray) (Array.replicate 64 0)

end Tests.AttentionEffects
