import Examples.GPU.IndexedParity
import Examples.GPU.RunCopy
import Examples.GPU.RunBrownianSample
import Examples.GPU.FlashBenchReference
import Examples.GPU.DecodeReference
import Examples.GPU.MhaTrainingReference
import Examples.GPU.GemmReference
import Examples.GPU.AttentionReference
import Examples.GPU.LossReference
import Examples.GPU.OptimizerReference
import Examples.GPU.RunRMSNorm
import Examples.GPU.RunLayerNorm
import Examples.GPU.RunMhaGB10
import Examples.GPU.RunMhaH100
import Examples.GPU.RunFlashAttn
import Examples.GPU.RunRotary
import Examples.GPU.RunRKCombine
import Examples.GPU.RunRKFusedSolve
import Examples.GPU.RunBrownianDescent
import Examples.GPU.RunEulerMaruyamaFused
import LeanTest

open torch Examples.GPU.IndexedParity

set_option linter.unusedVariables false in
example (a : Tensor { shape := #[2], dtype := .Float32, device := .CPU }) (b : Tensor { shape := #[3], dtype := .Float32, device := .CPU }) : True := by
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
      Examples.GPU.RunRMSNorm.runBFloat16Once, Examples.GPU.RunFlashAttn.runOnce, Examples.GPU.RunMhaH100.runOnce false, Examples.GPU.RunMhaGB10.runOnce, Examples.GPU.RunCopy.runOnce,
      Examples.GPU.RunBrownianSample.runOnce] do
    LeanTest.assertTrue (!(← run)) "CPU hosts cannot claim a successful GPU parity run"

@[test] def testGPURotaryIndexedTrainingReference : IO Unit := do
  let cos := Tensor.zeros #[2, 32]
  let sin := Tensor.ones #[2, 32]
  let signs := Tensor.fromFloatArray ((Array.replicate 32 (-1.0)) ++ Array.replicate 32 1.0)
    |>.reshape #[1, 1, 1, 64]
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
    let expectedK := Tensor.mulScalar (signs.expand #[1, 2, 1, 64]) 2.0
    let expectedDQ := Tensor.mulScalar (signs.expand #[1, 2, 2, 64]) (-3.0)
    let expectedDK := Tensor.mulScalar (signs.expand #[1, 2, 1, 64]) (-4.0)
    LeanTest.assertTrue (Tensor.allclose kOut.toFloat32 expectedK 0 0)
    LeanTest.assertTrue (Tensor.allclose dqOut.toFloat32 expectedDQ 0 0)
    LeanTest.assertTrue (Tensor.allclose dkOut.toFloat32 expectedDK 0 0)

set_option linter.unusedVariables false in
example (inputs : Examples.GPU.RunLayerNorm.LayerNormInputs .Float32) : True := by
  fail_if_success
    have _ : Examples.GPU.RunLayerNorm.LayerNormInputs .Float32 :=
      { inputs with bias := inputs.bias.toBFloat16 }
  trivial

@[test] def testGPULayerNormIndexedReference : IO Unit := do
  let x := Tensor.fromFloatArray ((Array.replicate 512 (-1.0)) ++ Array.replicate 512 1.0)
    |>.reshape #[1, 1, 1024] (by simp) |>.expand #[1, 64, 1024]
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
    let grad := (Tensor.fromFloatArray gradValues).reshape #[1, 1024] (by simp [gradValues]; split <;> simp)
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
      (Tensor.fromFloatArray (Array.replicate 512 10.0 ++ Array.replicate 512 (-2.0))).reshape #[1, 1024] (by simp)
        |>.expand #[2, 1024]
      else Tensor.full #[2, 1024] 4.0
    LeanTest.assertTrue (Tensor.allclose dx.toFloat32 expectedDx 0 0)
    let expectedDw := Tensor.fromFloatArray (gradValues.map (fun v => 2.0 * v * inverse))
      |>.reshape #[1024] (by simp [gradValues]; split <;> simp)
    LeanTest.assertTrue (Tensor.allclose dw expectedDw 0 1e-5)

@[test] def testGPULossIndexedTrainingReference : IO Unit := do
  let logits := (Tensor.zeros #[2, 3072]).toBFloat16
  let targets := (Tensor.fromInt64Array #[0, 3071]).reshape #[2]
  let (losses, mean, gradient) ← Examples.GPU.RunLoss.referenceTraining logits targets
  LeanTest.assertEqual losses.actualSpec.dtype DType.Float32
  LeanTest.assertEqual mean.actualSpec.dtype DType.Float32
  LeanTest.assertEqual gradient.actualSpec.dtype DType.BFloat16
  LeanTest.assertTrue (Tensor.allclose losses (Tensor.full #[2] (Float.log 3072.0)) 0 1e-5)
  LeanTest.assertTrue ((mean.item - Float.log 3072.0).abs < 1e-5)
  let expected := Tensor.fromFloatArray ((Array.range 6144).map fun i =>
    ((1.0 / 3072.0) - (if i == 0 || i == 6143 then 1.0 else 0.0)) / 2.0)
    |>.reshape #[2, 3072] |>.toBFloat16
  LeanTest.assertTrue (Tensor.allclose gradient.toFloat32 expected.toFloat32 0 0)

@[test] def testGPUOptimizerIndexedReference : IO Unit := do
  let master := Tensor.fromFloatArray #[1.0, -2.0, 3.0]
  let grad := (Tensor.fromFloatArray #[2.0, -2.0, 0.0]).toBFloat16
  for prior in #[false, true] do
    let m0 := if prior then 0.25 else 0.0
    let v0 := if prior then 0.5 else 0.0
    let m := Tensor.full #[3] m0
    let v := Tensor.full #[3] v0
    let (updated, model, m1, v1) := Examples.GPU.RunOptimizer.referenceUpdate
      master grad m v 0.01 0.5 0.75 1e-8 0.1 2.0 4.0
    let gradients := #[2.0, -2.0, 0.0]
    let masters := #[1.0, -2.0, 3.0]
    let expectedM := gradients.map (fun g => 0.5 * m0 + 0.5 * g)
    let expectedV := gradients.map (fun g => 0.75 * v0 + 0.25 * g * g)
    let expected := Tensor.fromFloatArray ((Array.range 3).map fun i =>
      masters[i]! - 0.01 * (2.0 * expectedM[i]! / (Float.sqrt (4.0 * expectedV[i]!) + 1e-8)
        + 0.1 * masters[i]!)) |>.reshape #[3]
    LeanTest.assertTrue (Tensor.allclose updated expected 0 1e-6)
    LeanTest.assertTrue (Tensor.allclose model.toFloat32 expected.toBFloat16.toFloat32 0 0)
    LeanTest.assertTrue (Tensor.allclose m1 ((Tensor.fromFloatArray expectedM).reshape #[3]) 0 1e-6)
    LeanTest.assertTrue (Tensor.allclose v1 ((Tensor.fromFloatArray expectedV).reshape #[3]) 0 1e-6)
  let dst ← Tensor.uniformLike master 0 0
  let dst ← Tensor.copyInto dst master
  LeanTest.assertTrue (Tensor.allclose dst master 0 0)
  let dst ← Tensor.copyInto dst (Tensor.zerosLike master)
  LeanTest.assertTrue (Tensor.allclose master (Tensor.fromFloatArray #[1.0, -2.0, 3.0]) 0 0)
  LeanTest.assertTrue (Tensor.allclose dst (Tensor.zerosLike master) 0 0)

set_option linter.unusedVariables false in
example (a : Tensor { shape := #[2], dtype := .Float32, device := .CPU }) (b : Tensor { shape := #[2], dtype := .BFloat16, device := .CPU })
    (c : Tensor { shape := #[3], dtype := .Float32, device := .CPU }) : True := by
  fail_if_success
    have _ := Tensor.copyInto a b
  fail_if_success
    have _ := Tensor.copyInto a c
  trivial

@[test] def testGPUGemmIndexedReference : IO Unit := do
  let a := (Tensor.fromFloatArray #[1, 2, -1, 0.5]).reshape #[2, 2] |>.toBFloat16
  let b := (Tensor.fromFloatArray #[3, 4, 0, 2, -2, 1]).reshape #[3, 2] |>.toBFloat16
  let output := Examples.GPU.gemmReference a b
  let expected := (Tensor.fromFloatArray #[11, 4, 0, -1, 1, 2.5]).reshape #[2, 3]
  LeanTest.assertEqual output.actualSpec.dtype DType.Float32
  LeanTest.assertTrue (Tensor.allclose output expected 0 0)
  let cancellation := (Tensor.fromFloatArray #[256, 1, -256]).reshape #[1, 3] |>.toBFloat16
  let ones := (Tensor.ones #[1, 3]).toBFloat16
  LeanTest.assertTrue (Tensor.allclose (Examples.GPU.gemmReference cancellation ones)
    (Tensor.ones #[1, 1]) 0 0)

@[test] def testGPUAttentionIndexedReference : IO Unit := do
  let q := (Tensor.fromFloatArray (Array.replicate 64 0.0 ++ Array.replicate 64 1.0))
    |>.reshape #[1, 1, 2, 64] (by simp) |>.toBFloat16
  let k := (Tensor.fromFloatArray
    (Array.replicate 64 0.0 ++ Array.replicate 64 1.0 ++ Array.replicate 64 0.0))
    |>.reshape #[1, 1, 3, 64] (by simp) |>.toBFloat16
  let v := (Tensor.fromFloatArray
    (Array.replicate 64 1.0 ++ Array.replicate 64 2.0 ++ Array.replicate 64 6.0))
    |>.reshape #[1, 1, 3, 64] (by simp) |>.toBFloat16
  let (output, lse) := Examples.GPU.attentionReference q k v
  LeanTest.assertEqual output.actualSpec.dtype DType.BFloat16
  LeanTest.assertEqual lse.actualSpec.dtype DType.Float32
  let expected := (Tensor.fromFloatArray
    (Array.replicate 64 3.0 ++ Array.replicate 64 ((7.0 + 2.0 * Float.exp 8.0) / (2.0 + Float.exp 8.0))))
    |>.reshape #[1, 1, 2, 64] (by simp) |>.toBFloat16
  LeanTest.assertTrue (Tensor.allclose output.toFloat32 expected.toFloat32 0 0)
  let expectedLse := (Tensor.fromFloatArray #[Float.log 3.0, Float.log (2.0 + Float.exp 8.0)])
    |>.reshape #[1, 1, 2]
  LeanTest.assertTrue (Tensor.allclose lse expectedLse 0 1e-6)

set_option linter.unusedVariables false in
example (a : Tensor { shape := #[2, 3], dtype := .BFloat16, device := .CPU }) (b : Tensor { shape := #[4, 5], dtype := .BFloat16, device := .CPU }) : True := by
  fail_if_success
    have _ := Examples.GPU.gemmReference a b
  trivial

@[test] def testGPUAttentionIndexedTrainingReference : IO Unit := do
  let q := (Tensor.zeros #[1, 1, 2, 64]).toBFloat16
  let k := (Tensor.fromFloatArray (Array.replicate 64 0.0 ++ Array.replicate 64 1.0))
    |>.reshape #[1, 1, 2, 64] (by simp) |>.toBFloat16
  let v := (Tensor.fromFloatArray (Array.replicate 64 1.0 ++ Array.replicate 64 3.0))
    |>.reshape #[1, 1, 2, 64] (by simp) |>.toBFloat16
  let cotangent := (Tensor.ones #[1, 1, 2, 64]).toBFloat16
  for _ in [:2] do
    let (output, dq, dk, dv) ← Examples.GPU.attentionTrainingReference q k v cotangent
    LeanTest.assertTrue (Tensor.allclose output.toFloat32 (Tensor.full #[1, 1, 2, 64] 2.0) 0 0)
    LeanTest.assertTrue (Tensor.allclose dq (Tensor.full #[1, 1, 2, 64] 4.0) 0 0)
    LeanTest.assertTrue (Tensor.allclose dk (Tensor.zeros #[1, 1, 2, 64]) 0 0)
    LeanTest.assertTrue (Tensor.allclose dv (Tensor.ones #[1, 1, 2, 64]) 0 0)

@[test] def testGPUMhaIndexedTrainingLossAndStep : IO Unit := do
  let out := (Tensor.full #[1, 1, 128, 64] 3.0).toBFloat16
  let target := (Tensor.ones #[1, 1, 128, 64]).toBFloat16
  let (loss, cotangent) := Examples.GPU.lossAndGradOut out target
  LeanTest.assertEqual loss.item 2.0
  LeanTest.assertTrue (Tensor.allclose cotangent.toFloat32
    (Tensor.full #[1, 1, 128, 64] (2.0 / 8192.0)) 0 0)
  let q := Tensor.zeros #[1, 1, 128, 64]
  let k := Tensor.zeros #[1, 1, 128, 64]
  let v := Tensor.full #[1, 1, 128, 64] 3.0
  for _ in [:2] do
    let (qn, kn, vn, stepLoss) ← Examples.GPU.oneTorchTrainStep q k v target.toFloat32 1.0
    LeanTest.assertTrue ((stepLoss.item - 2.0).abs < 1e-6)
    LeanTest.assertTrue (Tensor.allclose qn q 0 0)
    LeanTest.assertTrue (Tensor.allclose kn k 0 0)
    LeanTest.assertTrue (Tensor.allclose vn
      (Tensor.full #[1, 1, 128, 64] (3.0 - 2.0 / 8192.0)) 0 1e-6)
    LeanTest.assertTrue (Tensor.allclose v (Tensor.full #[1, 1, 128, 64] 3.0) 0 0)

@[test] def testGPUDecodeIndexedCacheReference : IO Unit := do
  torch.manualSeed 42
  LeanTest.assertTrue (← Examples.GPU.runDecodeCacheParity .CPU)

@[test] def testGPUAttentionIndexedBatchedHeads : IO Unit := do
  let q := (Tensor.zeros #[2, 3, 2, 64]).toBFloat16
  let k := (Tensor.fromFloatArray (Array.replicate 64 0.0 ++ Array.replicate 64 1.0))
    |>.reshape #[1, 1, 2, 64] (by simp) |>.expand #[2, 3, 2, 64] |>.toBFloat16
  let v := (Tensor.fromFloatArray (Array.replicate 64 1.0 ++ Array.replicate 64 3.0))
    |>.reshape #[1, 1, 2, 64] (by simp) |>.expand #[2, 3, 2, 64] |>.toBFloat16
  let cotangent := (Tensor.ones #[2, 3, 2, 64]).toBFloat16
  let (out, lse) := Examples.GPU.attentionReference q k v
  LeanTest.assertTrue (Tensor.allclose out.toFloat32 (Tensor.full #[2, 3, 2, 64] 2.0) 0 0)
  LeanTest.assertTrue (Tensor.allclose lse (Tensor.full #[2, 3, 2] (Float.log 2.0)) 0 1e-6)
  let (_, dq, dk, dv) ← Examples.GPU.attentionTrainingReference q k v cotangent
  LeanTest.assertTrue (Tensor.allclose dq (Tensor.full #[2, 3, 2, 64] 4.0) 0 0)
  LeanTest.assertTrue (Tensor.allclose dk (Tensor.zeros #[2, 3, 2, 64]) 0 0)
  LeanTest.assertTrue (Tensor.allclose dv (Tensor.ones #[2, 3, 2, 64]) 0 0)

@[test] def testGPUFlashBenchIndexedMetricsAndCotangent : IO Unit := do
  let ref := Tensor.fromFloatArray #[3.0, 4.0]
  let actual := Tensor.fromFloatArray #[6.0, 8.0]
  LeanTest.assertEqual (Examples.GPU.FlashBenchReference.relL2 actual ref) 1.0
  LeanTest.assertEqual (Examples.GPU.FlashBenchReference.relL2 ref (Tensor.zeros #[2])) 5.0
  for _ in [:2] do
    let leaf ← Examples.GPU.FlashBenchReference.makeLeaf ref.toBFloat16
    let out := leaf * leaf
    let cotangent := (Tensor.fromFloatArray #[2.0, -1.0]).toBFloat16
    Examples.GPU.FlashBenchReference.backwardWithCotangent out cotangent
    LeanTest.assertTrue (Tensor.allclose (← Tensor.gradOf leaf).toFloat32
      (Tensor.fromFloatArray #[12.0, -8.0]) 0 0)
