import Examples.GPU.IndexedParity
import Examples.GPU.LossReference
import Examples.GPU.RunRMSNorm
import Examples.GPU.RunLayerNorm
import Examples.GPU.RunRotary
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

@[test] def testGPUIndexedRunnersRequireCuda : IO Unit := do
  if ← torch.cuda_is_available then
    return
  for run in #[Examples.GPU.RunRKCombine.runOnce, Examples.GPU.RunRKFusedSolve.runOnce,
      Examples.GPU.RunBrownianDescent.runOnce, Examples.GPU.RunEulerMaruyamaFused.runOnce,
      Examples.GPU.RunRotary.runOnce, Examples.GPU.RunLayerNorm.runOnce,
      Examples.GPU.RunLayerNorm.runBFloat16Once, Examples.GPU.RunRMSNorm.runOnce,
      Examples.GPU.RunRMSNorm.runBFloat16Once] do
    LeanTest.assertTrue (!(← run)) "CPU hosts cannot claim a successful GPU parity run"

@[test] def testGPURotaryIndexedTrainingReference : IO Unit := do
  let cos := Tensor.zeros #[2, 32]
  let sin := Tensor.ones #[2, 32]
  let signs := Tensor.fromFloatArray ((Array.replicate 32 (-1.0)) ++ Array.replicate 32 1.0)
    |>.reshapeChecked #[1, 1, 1, 64]
  for dtype in #[DType.Float32, DType.BFloat16] do
    let q := (Tensor.full #[1, 2, 2, 64] 1.0).castDType dtype
    let k := (Tensor.full #[1, 2, 1, 64] 2.0).castDType dtype
    let dq := (Tensor.full #[1, 2, 2, 64] 3.0).castDType dtype
    let dk := (Tensor.full #[1, 2, 1, 64] 4.0).castDType dtype
    let (qOut, kOut, dqOut, dkOut) := Examples.GPU.RunRotary.referenceTraining q k dq dk cos sin
    LeanTest.assertEqual qOut.actualSpec.dtype dtype
    LeanTest.assertEqual kOut.actualSpec.dtype dtype
    LeanTest.assertEqual dqOut.actualSpec.dtype dtype
    LeanTest.assertEqual dkOut.actualSpec.dtype dtype
    LeanTest.assertTrue (Tensor.allclose qOut.toFloat32 (signs.expand #[1, 2, 2, 64]) 0 0)
    LeanTest.assertTrue (Tensor.allclose kOut.toFloat32 (Tensor.Preserving.mulScalar (signs.expand #[1, 2, 1, 64]) 2.0) 0 0)
    LeanTest.assertTrue (Tensor.allclose dqOut.toFloat32 (Tensor.Preserving.mulScalar (signs.expand #[1, 2, 2, 64]) (-3.0)) 0 0)
    LeanTest.assertTrue (Tensor.allclose dkOut.toFloat32 (Tensor.Preserving.mulScalar (signs.expand #[1, 2, 1, 64]) (-4.0)) 0 0)

set_option linter.unusedVariables false in
example (inputs : Examples.GPU.RunLayerNorm.LayerNormInputs .Float32) : True := by
  fail_if_success
    have _ : Examples.GPU.RunLayerNorm.LayerNormInputs .Float32 :=
      { inputs with bias := inputs.bias.toBFloat16 }
  trivial

@[test] def testGPULayerNormIndexedReference : IO Unit := do
  let x := Tensor.fromFloatArray ((Array.replicate 512 (-1.0)) ++ Array.replicate 512 1.0)
    |>.reshapeChecked #[1, 1, 1024] |>.expand #[1, 64, 1024]
  let inputs : Examples.GPU.RunLayerNorm.LayerNormInputs .Float32 := {
    x, residual := Tensor.zeros #[1, 64, 1024],
    weight := Tensor.full #[1024] 2.0, bias := Tensor.full #[1024] 0.5
  }
  let (output, residual) := Examples.GPU.RunLayerNorm.layerNormReference inputs
  LeanTest.assertTrue (Tensor.allclose residual x 0 0)
  let values ← output.toFloatArray
  let scale := 2.0 / Float.sqrt (1.0 + 1e-5)
  LeanTest.assertTrue ((values[0]! - (0.5 - scale)).abs < 1e-5)
  LeanTest.assertTrue ((values[1023]! - (0.5 + scale)).abs < 1e-5)
  let inputs16 := Examples.GPU.RunLayerNorm.asBFloat16Inputs inputs
  let (output16, residual16) := Examples.GPU.RunLayerNorm.layerNormReferenceBFloat16 inputs16
  LeanTest.assertEqual output16.actualSpec.dtype DType.BFloat16
  LeanTest.assertEqual residual16.actualSpec.dtype DType.BFloat16
  LeanTest.assertTrue (Tensor.allclose residual16.toFloat32 x 0 0)
  let values16 ← output16.toFloatArray
  LeanTest.assertEqual values16[0]! (-1.5)
  LeanTest.assertEqual values16[1023]! 2.5

@[test] def testGPURMSNormIndexedTrainingReference : IO Unit := do
  let x := (Tensor.full #[2, 1024] 0.75).toBFloat16
  let residual := (Tensor.full #[2, 1024] 0.25).toBFloat16
  let weight := (Tensor.full #[1024] 2.0).toBFloat16
  let gradResidual := (Tensor.full #[2, 1024] 4.0).toBFloat16
  for alternating in #[false, true] do
    let gradValues := if alternating then
      Array.replicate 512 3.0 ++ Array.replicate 512 (-3.0)
      else Array.replicate 1024 3.0
    let grad := (Tensor.fromFloatArray gradValues).reshapeChecked #[1, 1024]
      |>.expand #[2, 1024] |>.toBFloat16
    let (output, resid, inv, dx, dw) :=
      Examples.GPU.RunRMSNorm.referenceTraining x residual weight grad gradResidual
    LeanTest.assertEqual output.actualSpec.dtype DType.BFloat16
    LeanTest.assertEqual resid.actualSpec.dtype DType.BFloat16
    LeanTest.assertEqual dx.actualSpec.dtype DType.BFloat16
    LeanTest.assertEqual inv.actualSpec.dtype DType.Float32
    LeanTest.assertEqual dw.actualSpec.dtype DType.Float32
    LeanTest.assertTrue (Tensor.allclose output.toFloat32 (Tensor.full #[2, 1024] 2.0) 0 0)
    LeanTest.assertTrue (Tensor.allclose resid.toFloat32 (Tensor.ones #[2, 1024]) 0 0)
    let inverse := 1.0 / Float.sqrt (1.0 + 1e-6)
    LeanTest.assertTrue (Tensor.allclose inv (Tensor.full #[2] inverse) 0 1e-6)
    let expectedDx := if alternating then
      (Tensor.fromFloatArray (Array.replicate 512 10.0 ++ Array.replicate 512 (-2.0))).reshapeChecked #[1, 1024]
        |>.expand #[2, 1024]
      else Tensor.full #[2, 1024] 4.0
    LeanTest.assertTrue (Tensor.allclose dx.toFloat32 expectedDx 0 0)
    let expectedDw := Tensor.fromFloatArray (gradValues.map (fun v => 2.0 * v * inverse))
      |>.reshapeChecked #[1024]
    LeanTest.assertTrue (Tensor.allclose dw expectedDw 0 1e-5)

@[test] def testGPULossIndexedTrainingReference : IO Unit := do
  let logits := (Tensor.zeros #[2, 3072]).toBFloat16
  let targets := (Tensor.fromInt64Array #[0, 3071]).reshapeChecked #[2]
  let (losses, mean, gradient) := Examples.GPU.RunLoss.referenceTraining logits targets
  LeanTest.assertEqual losses.actualSpec.dtype DType.Float32
  LeanTest.assertEqual mean.actualSpec.dtype DType.Float32
  LeanTest.assertEqual gradient.actualSpec.dtype DType.BFloat16
  LeanTest.assertTrue (Tensor.allclose losses (Tensor.full #[2] (Float.log 3072.0)) 0 1e-5)
  LeanTest.assertTrue ((mean.item - Float.log 3072.0).abs < 1e-5)
  let expected := Tensor.fromFloatArray ((Array.range 6144).map fun i =>
    ((1.0 / 3072.0) - (if i == 0 || i == 6143 then 1.0 else 0.0)) / 2.0)
    |>.reshapeChecked #[2, 3072] |>.toBFloat16
  LeanTest.assertTrue (Tensor.allclose gradient.toFloat32 expected.toFloat32 0 0)
