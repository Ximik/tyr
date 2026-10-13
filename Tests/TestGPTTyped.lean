import LeanTest
import Examples.GPT.GPT

open torch torch.gpt

namespace Tests.TestGPTTyped

set_option linter.unusedVariables false

-- Exact regression: the off-by-one projection must not fit the model field.
example (n : UInt64) (params : BlockParams n) (scale : Float) : True := by
  fail_if_success
    have _bad : IO (BlockParams n) := do
      let projection ← Tensor.randn #[n, 4*n + 1]
      return { params with mlp_proj := makeLeafParam (projection * scale) }
  trivial

example (n : UInt64) (params : BlockParams n) (scale : Float) : IO (BlockParams n) := do
  let projection ← Tensor.randn #[n, 4*n]
  return { params with mlp_proj := makeLeafParam (projection * scale) }

example (params : GPTParams Config.tiny_shakespeare)
    (wrongTokens : Tokens #[1, 3]) (wrongDtype : Activation #[1, 2]) : True := by
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
  let lossValue := nn.item (s := #[]) loss.raw
  if lossValue.isNaN || lossValue.isInf then
    throw (IO.userError "nonfinite GPT loss")
  Tensor.backwardLoss loss
  let opt := Optim.adamw (α := GPTParams config) (lr := 0.001)
  let (updated, _) := Optim.step opt params (TensorStruct.grads params) (opt.init params)
  let some block := updated.blocks[0]? | throw (IO.userError "missing GPT block")
  unless block.mlp_proj.raw.runtimeShape == #[4, 16] do
    throw (IO.userError "optimizer lost GPT projection shape")
  let change := nn.item (s := #[]) (Tensor.sumAll (Tensor.abs (Tensor.sub updated.wte params.wte))).raw
  unless change > 0 do throw (IO.userError "GPT optimizer did not update weights")
  let next ← forward updated tokens false
  unless next.raw.runtimeShape == #[1, 2, 8] do
    throw (IO.userError "updated GPT cannot run")

@[test] def typedGPTRawBoundary : IO Unit := do
  let params ← GPTParams.init config
  let generated ← generate params #[1] 1
  unless generated.size == 2 do throw (IO.userError "GPT generation failed")
  let wrong := Tensor.fullInt #[1, 3] 1
  let rejected ← try
    let _ ← forwardRaw (batch := 1) (seq := 2) params wrong.raw false
    pure false
  catch _ => pure true
  unless rejected do throw (IO.userError "raw shape mismatch was accepted")
  let wrongDtype := Tensor.zeros #[1, 2]
  let rejected ← try
    let _ ← forwardRaw (batch := 1) (seq := 2) params wrongDtype.raw false
    pure false
  catch _ => pure true
  unless rejected do throw (IO.userError "raw dtype mismatch was accepted")

end Tests.TestGPTTyped
