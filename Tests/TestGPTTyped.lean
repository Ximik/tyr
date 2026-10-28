import LeanTest
import Examples.GPT.GPT
import Examples.GPT.Train
import Examples.GPT.Pretraining

open torch torch.gpt

namespace Tests.TestGPTTyped

set_option linter.unusedVariables false

-- Exact regression: the off-by-one projection must not fit the model field.
example (n : Nat) (params : BlockParams (device := .CPU) n) (scale : Float) : True := by
  fail_if_success
    have _bad : IO (BlockParams (device := .CPU) n) := do
      let projection ← Tensor.randn #[n, 4*n + 1]
      return { params with mlp_proj := ← makeLeafParam (projection * scale) }
  trivial

example (n : Nat) (params : BlockParams (device := .CPU) n) (scale : Float) : IO (BlockParams (device := .CPU) n) := do
  let projection ← Tensor.randn #[n, 4*n]
  return { params with mlp_proj := ← makeLeafParam (projection * scale) }

example (params : GPTParams (device := .CPU) Config.tiny_shakespeare)
    (wrongTokens : Tokens (device := .CPU) #[1, 3]) (wrongDtype : Activation (device := .CPU) #[1, 2]) : True := by
  fail_if_success
    have _ := forward (batch := 1) (seq := 2) params wrongTokens
  fail_if_success
    have _ := forward params wrongDtype
  trivial

private def config : Config :=
  { vocab_size := 8, block_size := 4, n_embd := 4, n_head := 2, n_layer := 1 }

@[test] def typedGPTTraining : IO Unit := do
  let params ← GPTParams.init config
  let tokens := Tensor.fullInt #[1, 2] 1
  let logits ← forward params tokens false
  unless logits.raw.runtimeShape == #[1, 2, 8] do
    throw (IO.userError "incorrect GPT output shape")
  let loss ← loss params tokens tokens false
  let lossValue := loss.item
  if lossValue.isNaN || lossValue.isInf then
    throw (IO.userError "nonfinite GPT loss")
  Tensor.backwardLoss loss
  let opt := Optim.adamw (α := GPTParams (device := .CPU) config) (lr := 0.001)
  let (updated, _) ← Optim.step opt params (← TensorStruct.grads params) (opt.init params)
  let some block := updated.blocks[0]? | throw (IO.userError "missing GPT block")
  unless block.mlp_proj.raw.runtimeShape == #[4, 16] do
    throw (IO.userError "optimizer lost GPT projection shape")
  let change := (Tensor.sumAll (Tensor.abs (Tensor.sub updated.wte params.wte))).item
  unless change > 0 do throw (IO.userError "GPT optimizer did not update weights")
  let next ← forward updated tokens false
  unless next.raw.runtimeShape == #[1, 2, 8] do
    throw (IO.userError "updated GPT cannot run")

@[test] def typedGPTGenerationAndInputValidation : IO Unit := do
  let params ← GPTParams.init config
  let generated ← generate params #[1] 1
  unless generated.size == 2 do throw (IO.userError "GPT generation failed")
  let wrong := Tensor.fullInt #[1, 3] 1
  let rejected ← try
    let _ ← TensorOps.checked (device := .CPU) (s := #[1, 2]) .Int64 wrong.raw
    pure false
  catch _ => pure true
  unless rejected do throw (IO.userError "raw shape mismatch was accepted")
  let wrongDtype := Tensor.zeros #[1, 2]
  let rejected ← try
    let _ ← TensorOps.checked (device := .CPU) (s := #[1, 2]) .Int64 wrongDtype.raw
    pure false
  catch _ => pure true
  unless rejected do throw (IO.userError "raw dtype mismatch was accepted")

@[test] def typedGPTBatchAndAccumulatedTraining : IO Unit := do
  let stream := Tensor.fromInt64Array #[0, 1, 2, 3, 4, 5]
  let (x, y) ← train.getBatch stream 2 2
  LeanTest.assertEqual (← x.toUInt64Array) #[0, 1, 3, 4]
  LeanTest.assertEqual (← y.toUInt64Array) #[1, 2, 4, 5]
  for (batch, seq) in #[(0, 2), (2, 0), (3, 2)] do
    let rejected ← try
      let _ ← train.getBatch stream batch seq
      pure false
    catch _ => pure true
    LeanTest.assertTrue rejected "Invalid batches fail before native sampling/slicing"
  let params ← GPTParams.init config
  let opt := Optim.adamw (α := GPTParams (device := .CPU) config) (lr := 0.001)
  let trainCfg : train.TrainConfig := { gradAccumSteps := 2, gradClip := 1.0 }
  let (updated, _, value) ← train.trainStepWithAccum trainCfg params (opt.init params)
    (pure (x, y)) 0.001
  LeanTest.assertTrue value.isFinite
  LeanTest.assertTrue ((updated.wte - params.wte).abs.sumAll.item > 0)
    "Accumulated typed losses update GPT parameters"
  let eval ← train.evalLoss updated stream 2 2 1
  LeanTest.assertTrue eval.isFinite

@[test] def typedGPTBinaryTokens : IO Unit := IO.FS.withTempDir fun dir => do
  let path := (dir / "tokens.bin").toString
  IO.FS.writeBinFile path (ByteArray.mk #[0, 0, 1, 0, 255, 255])
  let tokens ← Tensor.loadU16Bin 3 path
  LeanTest.assertEqual tokens.actualSpec.dtype DType.Int64
  LeanTest.assertEqual (← tokens.toUInt64Array) #[0, 1, 65535]
  let rejected ← try
    let _ ← Tensor.loadU16Bin 4 path
    pure false
  catch _ => pure true
  LeanTest.assertTrue rejected "Short token files return IO errors"

@[test] def typedGPTGenerationRejectsInvalidInputs : IO Unit := do
  let params ← GPTParams.init config
  for (context, temperature) in #[(#[], 1.0), (#[1], 0.0), (#[1], -1.0)] do
    let rejected ← try
      let _ ← generate params context 1 temperature
      pure false
    catch _ => pure true
    LeanTest.assertTrue rejected
  LeanTest.assertEqual (← generate params #[] 0) #[]

@[test] def typedGPTPretrainingExactFit : IO Unit := do
  let cfg : Examples.GPT.Pretraining.PretrainingConfig := {
    modelConfig := config, deviceBatchSize := 2, seqLen := 2, totalSteps := 1
  }
  let stream := Tensor.fromInt64Array #[0, 1, 2, 3, 4]
  let trained ← Examples.GPT.Pretraining.runPretraining cfg 5 stream none
  let logits ← forward trained (Tensor.fullInt #[1, 2] 1) false
  LeanTest.assertTrue ((← logits.toFloatArray).all Float.isFinite)

end Tests.TestGPTTyped
