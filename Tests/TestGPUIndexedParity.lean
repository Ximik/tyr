import Examples.GPU.IndexedParity
import Examples.GPU.RunRKCombine
import Examples.GPU.RunRKFusedSolve
import Examples.GPU.RunBrownianDescent
import Examples.GPU.RunEulerMaruyamaFused
import LeanTest

open torch Examples.GPU.IndexedParity

set_option linter.unusedVariables false in
example (a : DTensor #[2] .Float32) (b : DTensor #[3] .Float32) : True := by
  fail_if_success
    have _ := compareTensors "bad shape" a b 0 0
  trivial

@[test] def testGPUIndexedParityMetrics : IO Unit := do
  let expected := Tensor.fromFloatArray #[1, 2, 3]
  let actual := (Tensor.fromFloatArray #[1, 2.5, 2.5]).toBFloat16
  let check := compareTensors "mixed floating dtypes" expected actual 0 0.5
  LeanTest.assertTrue check.ok
  LeanTest.assertTrue ((check.mae - 1.0 / 3.0).abs < 1e-6)
  LeanTest.assertEqual check.maxErr 0.5
  let rejected := compareTensors "strict tolerance" expected actual 0 0.1
  LeanTest.assertTrue (!rejected.ok)
  let nan := Tensor.fromFloatArray #[0.0 / 0.0]
  LeanTest.assertTrue (!(compareTensors "NaN is not parity" nan nan 0 0).ok)

@[test] def testGPUOdeRunnersRequireCuda : IO Unit := do
  if ← torch.cuda_is_available then
    return
  for run in #[Examples.GPU.RunRKCombine.runOnce, Examples.GPU.RunRKFusedSolve.runOnce,
      Examples.GPU.RunBrownianDescent.runOnce, Examples.GPU.RunEulerMaruyamaFused.runOnce] do
    LeanTest.assertTrue (!(← run)) "CPU hosts cannot claim a successful GPU parity run"
