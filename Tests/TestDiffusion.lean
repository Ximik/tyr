/-
  Tests for Diffusion Model Implementation
-/
import Tyr
import Examples.Diffusion.Diffusion
import Examples.Diffusion.DiffusionSchedule
import Examples.Diffusion.DiffusionTrain
import LeanTest

namespace Tests.Diffusion

open torch
open torch.diffusion



@[test]
def testMaskSchedule : IO Unit := do
  let schedule := MaskedDiffusionSchedule.init 64 0 16
  -- Verify mask_probs goes from 1/64 to ~1.0
  let firstProb := (schedule.mask_probs.sliceChecked 0 0 1 #[1] |>.reshapeChecked #[]).item
  let lastProb := (schedule.mask_probs.sliceChecked 0 63 1 #[1] |>.reshapeChecked #[]).item
  LeanTest.assertTrue (firstProb > 0.01) "First prob > 0.01"
  LeanTest.assertTrue (firstProb < 0.05) "First prob < 0.05"
  LeanTest.assertTrue (lastProb > 0.95) "Last prob > 0.95"

@[test]
def testMaskPreservesContext : IO Unit := do
  let schedule := MaskedDiffusionSchedule.init 64 0 16  -- context_len=16
  -- Create random tokens (avoiding 0 which is mask)
  let x_0 ← Tensor.randint 1 127 #[(2 : Nat), 128]
  -- High timestep = lots of masking
  let t := Tensor.fullInt #[(2 : Nat)] 63
  let x_t ← schedule.addMasks x_0 t
  -- Count total masks and check some were added
  let numMasked := Tensor.item (Tensor.sumAll ((Tensor.eqScalar x_t 0).toFloat32))
  -- With context_len=16 and seq=128, we have 112 non-context positions per sample
  -- At timestep 63 with linear schedule, mask_prob ~= 1.0, so most should be masked
  LeanTest.assertTrue (numMasked > 50) "Many positions are masked"
  let originalContext := x_0.sliceChecked 1 0 16 #[2, 16]
  let maskedContext := x_t.sliceChecked 1 0 16 #[2, 16]
  LeanTest.assertTrue (Tensor.allclose originalContext maskedContext 0.0 0.0)
    "Every context token is preserved"

@[test]
def testMaskedLoss : IO Unit := do
  -- Create dummy logits and targets
  let logits ← Tensor.randn #[(2 : Nat), 32, 128] false
  let targets ← Tensor.randint 0 127 #[(2 : Nat), 32]
  let x_t := Tensor.fullInt #[(2 : Nat), 32] 0  -- All masked
  let loss := maskedCrossEntropyLoss logits targets x_t 0
  let lossVal := Tensor.item loss
  LeanTest.assertTrue (not lossVal.isNaN) "Loss not NaN"
  LeanTest.assertTrue (lossVal > 0) "Loss positive"
  -- Test with no masked positions
  let x_t_none ← Tensor.randint 1 127 #[(2 : Nat), 32]  -- No masks
  let loss_none := maskedCrossEntropyLoss logits targets x_t_none 0
  let lossValNone := Tensor.item loss_none
  -- When no positions are masked, loss should be ~0 (masked loss / small epsilon)
  LeanTest.assertTrue (lossValNone < 1.0) "Loss with no masks is small"

@[test]
def testDiffusionForward : IO Unit := do
  let cfg := Config.tiny
  let params ← DiffusionParams.init cfg
  let rotaryCache ← RotaryCache.init cfg.seq_len cfg.headDim
  -- Create input tensors
  let x_t ← Tensor.randint 0 cfg.vocab_size.toInt64 #[(2 : Nat), cfg.seq_len]
  let t ← Tensor.randint 0 cfg.diffusion_steps.toInt64 #[(2 : Nat)]
  let logits := forward params x_t t rotaryCache
  -- Verify output is finite
  let sum := Tensor.item (Tensor.sumAll logits)
  LeanTest.assertTrue (not sum.isNaN) "Output not NaN"
  LeanTest.assertTrue (not sum.isInf) "Output not Inf"

@[test]
def testSampleConfidence : IO Unit := do
  let cfg := Config.tiny
  let params ← DiffusionParams.init cfg
  let rotaryCache ← RotaryCache.init cfg.seq_len cfg.headDim
  let result ← sampleConfidence (batch := 1) params rotaryCache (confidenceThreshold := 0.3) (maxSteps := 50)
  -- Check if any positions still masked
  let _anyMasked := Tensor.any (Tensor.eqScalar result cfg.mask_token_id.toInt64)
  -- Note: with random weights, sampling may not fully decode
  -- Just verify it doesn't crash and produces valid output
  let sum := Tensor.item (Tensor.sumAll (result.toFloat32))
  LeanTest.assertTrue (not sum.isNaN) "Output not NaN"

@[test]
def testSampleTopK : IO Unit := do
  let cfg := Config.tiny
  let params ← DiffusionParams.init cfg
  let rotaryCache ← RotaryCache.init cfg.seq_len cfg.headDim
  let result ← sampleTopK (batch := 1) params rotaryCache (k := 8) (maxSteps := 50)
  -- Verify output is valid
  let sum := Tensor.item (Tensor.sumAll (result.toFloat32))
  LeanTest.assertTrue (not sum.isNaN) "Output not NaN"

@[test]
def testTrainStep : IO Unit := do
  manualSeed 20260908
  let modelCfg := Config.tiny
  let trainCfg : diffusion.train.TrainConfig := { maxIters := 1, batchSize := 2 }
  let params ← DiffusionParams.init modelCfg
  let rotaryCache ← RotaryCache.init modelCfg.seq_len modelCfg.headDim
  let schedule := MaskedDiffusionSchedule.init modelCfg.diffusion_steps modelCfg.mask_token_id modelCfg.context_len
  -- Initialize optimizer
  let opt := Optim.adamw (lr := 1e-4)
  let optState := opt.init params
  -- Create dummy batch
  let x_0 ← Tensor.randint 1 127 #[trainCfg.batchSize, modelCfg.seq_len]
  let (params', _optState', loss) ← diffusion.train.trainStep trainCfg params optState schedule x_0 rotaryCache 1e-4
  let params' : DiffusionParams modelCfg := params'
  LeanTest.assertTrue (not (Float.isNaN loss)) "Loss not NaN"
  LeanTest.assertTrue (loss > 0) "Loss positive"
  -- Verify parameters changed
  let diffTensor := params.token_emb - params'.token_emb
  let diffSum := Tensor.item (Tensor.sumAll (Tensor.abs diffTensor))
  
  -- Parameters should change after training step
  LeanTest.assertTrue (diffSum > 1e-6) s!"Parameters updated (diff={diffSum})"
  -- The zero-initialized output head receives the first step's learning signal;
  -- embedding changes alone can be explained by decoupled weight decay.
  let headDiff := Tensor.item (Tensor.sumAll (Tensor.abs (params.output_head - params'.output_head)))
  LeanTest.assertTrue (headDiff > 1e-6) s!"Output head learned (diff={headDiff})"

set_option linter.unusedVariables false in
example (n : Nat) (params : MLPParams n) : True := by
  fail_if_success
    have _ : MLPParams n := { params with c_proj := Tensor.zeros #[n, 4*n + 1] }
  trivial

set_option linter.unusedVariables false in
example (cfg : Config) (params : DiffusionParams cfg) (x : Activation #[1, cfg.seq_len])
    (t : Tokens #[1]) (rotary : RotaryCache cfg.seq_len cfg.headDim) : True := by
  fail_if_success
    have _ := forward params x t rotary
  trivial

private def checkpointConfig : Config := {
  seq_len := 4, vocab_size := 8, n_layer := 1, n_head := 1, n_embd := 4,
  diffusion_steps := 2, context_len := 2
}

@[test] def testDiffusionCheckpointResumeAndShape : IO Unit := IO.FS.withTempDir fun dir => do
  let cfg := checkpointConfig
  let params ← DiffusionParams.init cfg
  let rotary ← RotaryCache.init cfg.seq_len cfg.headDim
  let schedule := MaskedDiffusionSchedule.init 1 0 2
  let x := (Tensor.fromInt64Array #[1, 2, 3, 4]).reshapeChecked #[1, 4]
  let opt := Optim.adamw (α := DiffusionParams cfg) (lr := 0.001)
  let trainCfg : diffusion.train.TrainConfig := { gradClip := 0 }
  let (trained, state, loss) ← diffusion.train.trainStep trainCfg params (opt.init params) schedule x rotary 0.001
  checkpoint.saveFullCheckpoint trained state 1 loss loss dir.toString
  let (loaded, loadedState, ckptMeta) ← checkpoint.loadFullCheckpoint cfg dir.toString
  LeanTest.assertEqual ckptMeta.iteration 1
  LeanTest.assertEqual loadedState.fst.count state.fst.count
  LeanTest.assertTrue (Tensor.allclose loaded.output_head trained.output_head 0 0)
  LeanTest.assertTrue (Tensor.allclose loadedState.fst.mu.output_head state.fst.mu.output_head 0 0)
  let (expected, _, _) ← diffusion.train.trainStep trainCfg trained state schedule x rotary 0.001
  let (resumed, _, _) ← diffusion.train.trainStep trainCfg loaded loadedState schedule x rotary 0.001
  LeanTest.assertTrue (Tensor.allclose expected.output_head resumed.output_head 0 0)
    "Resuming restores the optimizer update exactly"
  Tensor.saveTensor (Tensor.zeros #[4, 8]) (dir / "output_head.pt").toString
  let rejected ← try
    let _ ← checkpoint.loadDiffusionParams cfg dir.toString
    pure false
  catch _ => pure true
  LeanTest.assertTrue rejected "Same-numel checkpoint tensors cannot silently change layout"

@[test] def testDiffusionSamplingContextAndBounds : IO Unit := do
  let cfg := { checkpointConfig with n_layer := 0 }
  let params ← DiffusionParams.init cfg
  let head := Tensor.fromFloatArray #[0,0,0,0, 1,1,1,1, 0,0,0,0, 0,0,0,0,
    0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0] |>.reshapeChecked #[8,4]
  let params := { params with token_emb := Tensor.ones #[8,4], time_emb := Tensor.ones #[2,4], output_head := head }
  let rotary ← RotaryCache.init cfg.seq_len cfg.headDim
  let contextIds := Tensor.fromInt64Array #[0,3] |>.reshapeChecked #[1,2]
  let topk ← sampleTopK params rotary 2 1.0 4 (some contextIds)
  let confidence ← sampleConfidence params rotary 0.99 1.0 4 (some contextIds)
  LeanTest.assertEqual (← topk.toUInt64Array) #[0,3,1,1]
  LeanTest.assertEqual (← confidence.toUInt64Array) #[0,3,1,1]
    "Even mask-valued context tokens survive confidence fallback decoding"
  for k in #[0,5] do
    let rejected ← try
      let _ ← sampleTopK (batch := 1) params rotary k
      pure false
    catch _ => pure true
    LeanTest.assertTrue rejected
  let stream := Tensor.fromInt64Array #[1,2,3,4]
  let batch ← diffusion.train.getBatch stream 2 2
  LeanTest.assertEqual (← batch.toUInt64Array) #[1,2,3,4]

end Tests.Diffusion
