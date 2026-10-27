import Examples.GPT.GPT
import LeanTest

open torch

namespace Tests.GPTAttentionEffects

private def attentionOnlyBlock : gpt.BlockParams (device := .CPU) 2 := {
  ln1_weight := Tensor.ones #[2]
  ln1_bias := Tensor.zeros #[2]
  q_proj := Tensor.zeros #[2, 2]
  k_proj := Tensor.zeros #[2, 2]
  v_proj := Tensor.eyeFor .Float32
  c_proj := Tensor.eyeFor .Float32
  c_proj_bias := Tensor.zeros #[2]
  ln2_weight := Tensor.ones #[2]
  ln2_bias := Tensor.zeros #[2]
  mlp_fc := Tensor.zeros #[8, 2]
  mlp_fc_bias := Tensor.zeros #[8]
  mlp_proj := Tensor.zeros #[2, 8]
  mlp_proj_bias := Tensor.zeros #[2]
}

@[test] def testGPTEvaluationDisablesConfiguredAttentionDropout : IO Unit := do
  let input := (Tensor.fromFloatArray #[1, 3, 3, 1]).reshapeChecked #[1, 2, 2]
  let reference ← gpt.blockForward (n_head := 1) attentionOnlyBlock input 0 false
  let evaluation ← gpt.blockForward (n_head := 1) attentionOnlyBlock input 1 false
  LeanTest.assertTrue (Tensor.allclose evaluation reference)
    "GPT evaluation must ignore configured dropout in attention as well as residual paths"
  LeanTest.assertTrue (!(Tensor.allclose reference input))
    "The fixture must have a nonzero attention contribution"
  let training ← gpt.blockForward (n_head := 1) attentionOnlyBlock input 1 true
  LeanTest.assertTrue (Tensor.allclose training input)
    "Training retains configured dropout on residual paths"

@[test] def testGPTAttentionRetainsTrainingDropout : IO Unit := do
  let q := Tensor.zeros #[1, 1, 2, 2]
  let v := (Tensor.fromFloatArray #[2, 4, 6, 8]).reshapeChecked #[1, 1, 2, 2]
  let training ← gpt.TensorOps.attention q q v 1 true
  LeanTest.assertEqual (← training.toFloatArray) #[0, 0, 0, 0]
  let evaluation ← gpt.TensorOps.attention q q v 1 false
  LeanTest.assertEqual (← evaluation.toFloatArray) #[2, 4, 4, 6]

end Tests.GPTAttentionEffects
