/-
  TestModdedGPT.lean

  Unit tests for modded-nanogpt port to Tyr.

  Test categories:
  1. C++ FFI smoke tests
  2. Pure function tests (schedules, shapes)
  3. Optimizer unit tests
  4. Sharding tests
  5. Model forward tests
  6. Training step tests

  Usage:
    ninja test-modded
    export DYLD_LIBRARY_PATH=external/wheels/torch/lib:/opt/homebrew/opt/libomp/lib
    out/exe/TestModdedGPT
-/
import Tyr
import Tyr.Distributed
import Tyr.Sharding
import Examples.NanoChat.ModdedGPT
import Tyr.DataLoader
import Examples.NanoChat.ModdedTrain
import Tyr.Optim.PolarExpress
import Tyr.Optim.NorMuon
import Tyr.Optim.DistAdam
import LeanTest

open torch
open torch.dist
open torch.Sharding
open torch.moddedGpt
open torch.DataLoader
open torch.ModdedTrain
open torch.Optim

/-! ## Test Utilities -/

/-- Check that a Float is finite and non-NaN -/
def isFiniteFloat (x : Float) : Bool :=
  !x.isNaN && !x.isInf

/-! ## 1. C++ FFI Smoke Tests -/

@[test]
def testDistributedFFI : IO Unit := do
  let _isInit ← dist.isInitialized
  LeanTest.assertTrue true

@[test]
def testPolarExpressFFI : IO Unit := do
  -- Test XXT: A @ A.T
  let A ← randn #[32, 32] false
  let _xxt ← dist.xxt A

  -- Test polar express orthogonalization
  let G ← randn #[64, 64] false
  let orthG ← dist.polarExpress G 5

  -- Verify result is not all zeros
  let norm := nn.item (nn.meanAll (nn.abs orthG))
  LeanTest.assertTrue (norm > 0.0) "Orthogonalized gradient should be non-zero"

/-! ## 2. Pure Function Tests (Schedules) -/

@[test]
def testSchedules : IO Unit := do
  -- Batch size schedule
  LeanTest.assertEqual (getBatchSize 0) 8
  LeanTest.assertEqual (getBatchSize 199) 8
  LeanTest.assertEqual (getBatchSize 200) 16
  LeanTest.assertEqual (getBatchSize 999) 16
  LeanTest.assertEqual (getBatchSize 1000) 24

  -- Window size schedule
  let (ws0_s, ws0_l) := getWindowSizes 0
  LeanTest.assertTrue (ws0_s == 3 && ws0_l == 3) "Window sizes at step 0 should be (3, 3)"
  let (ws500_s, ws500_l) := getWindowSizes 500
  LeanTest.assertTrue (ws500_s == 3 && ws500_l == 7) "Window sizes at step 500 should be (3, 7)"
  let (ws1500_s, ws1500_l) := getWindowSizes 1500
  LeanTest.assertTrue (ws1500_s == 3 && ws1500_l == 11) "Window sizes at step 1500 should be (3, 11)"

  -- Learning rate schedule
  let hp : Hyperparameters := {
    Hyperparameters.default with
    warmupFrac := 0.2
    cooldownFrac := 0.4
    finalLrFrac := 0.0
  }
  let lr0 := getLearningRate 0 hp
  let lr300 := getLearningRate 300 hp  -- After warmup
  let lr2000 := getLearningRate 2000 hp  -- During cooldown
  LeanTest.assertTrue (lr0 < lr300) "LR should increase during warmup"
  LeanTest.assertTrue (lr2000 < lr300) "LR should decrease during cooldown"

  -- Momentum schedule
  let mom0 := getMuonMomentum 0 hp
  let mom300 := getMuonMomentum 300 hp
  LeanTest.assertTrue (mom0 < mom300) "Momentum should increase during warmup"

/-! ## 3. Optimizer Unit Tests -/

@[test]
def testNorMuon : IO Unit := do
  -- Create a parameter and gradient
  let param ← randn #[64, 64] false
  let param := autograd.set_requires_grad param true
  let grad ← randn #[64, 64] false

  -- Initialize state
  let state := NorMuon.initParamState param
  let cfg : NorMuon.Config := default

  -- Single step (using Adam-like update since orthogonalization may be slow)
  let (newParam, newState) ← NorMuon.stepAdamLike param grad state cfg 1.0 1.0

  -- Verify param changed
  let diff := nn.item (nn.meanAll (nn.abs (autograd.detach newParam - autograd.detach param)))
  LeanTest.assertTrue (diff > 0) "Parameter should have changed after optimizer step"

  -- Verify state updated
  LeanTest.assertEqual newState.step 1

@[test]
def testDistAdam : IO Unit := do
  -- Create a parameter and gradient
  let param ← randn #[128, 64] false
  let param := autograd.set_requires_grad param true
  let grad ← randn #[128, 64] false

  -- Initialize state
  let state := DistAdam.initParamState param
  let cfg : DistAdam.Config := default

  -- Single step
  let (newParam, newState) ← DistAdam.stepSingle param grad state cfg 1.0 1.0

  -- Verify param changed
  let diff := nn.item (nn.meanAll (nn.abs (autograd.detach newParam - autograd.detach param)))
  LeanTest.assertTrue (diff > 0) "Parameter should have changed after optimizer step"

  -- Verify state updated
  LeanTest.assertEqual newState.step 1

/-! ## 4. Sharding Tests -/

@[test]
def testShardingShapes : IO Unit := do
  -- Test shard size computation
  let fullSize : UInt64 := 100
  let worldSize : UInt64 := 4

  let size0 := shardSize fullSize worldSize 0
  let size1 := shardSize fullSize worldSize 1
  let size2 := shardSize fullSize worldSize 2
  let size3 := shardSize fullSize worldSize 3

  -- Sum should equal full size
  let totalSize := size0 + size1 + size2 + size3
  LeanTest.assertEqual totalSize fullSize

  -- Test offset computation
  let offset0 := shardOffset fullSize worldSize 0
  let _offset1 := shardOffset fullSize worldSize 1
  let _offset2 := shardOffset fullSize worldSize 2
  let _offset3 := shardOffset fullSize worldSize 3

  LeanTest.assertEqual offset0 0

/-! ## 5. Model Tests -/

@[test]
def testModdedGPTInit : IO Unit := do
  -- Use tiny config for fast testing
  let cfg : moddedGpt.Config := {
    vocabSize := 1000
    nLayer := 2
    nHead := 2
    headDim := 32
    modelDim := 64
    maxSeqLen := 128
    blockSize := 32
    numValueEmbeds := 2
  }

  let params ← ModdedGPTParams.init cfg

  -- Check that arrays have correct sizes
  LeanTest.assertEqual params.blocks.size cfg.nLayer.toNat
  LeanTest.assertEqual params.valueEmbeds.size cfg.numValueEmbeds

@[test]
def testYarnRotary : IO Unit := do
  let headDim : UInt64 := 64
  let maxSeqLen : UInt64 := 128

  let yarn ← YarnRotary.init headDim maxSeqLen

  -- Check cos/sin are non-zero
  let cosNorm := nn.item (nn.meanAll (nn.abs yarn.cos))

  LeanTest.assertTrue (cosNorm > 0) "Cos should be non-zero"

@[test]
def testModdedGPTForward : IO Unit := do
  -- Tiny config
  let cfg : moddedGpt.Config := {
    vocabSize := 1000
    nLayer := 2
    nHead := 2
    headDim := 32
    modelDim := 64
    maxSeqLen := 128
    blockSize := 32
    numValueEmbeds := 2
    softcapValue := 30.0
  }

  let params ← ModdedGPTParams.init cfg
  let yarn ← YarnRotary.init cfg.headDim cfg.maxSeqLen

  -- Create input
  let batchSize : UInt64 := 2
  let seqLen : UInt64 := 32
  let input ← randint 0 cfg.vocabSize.toInt64 #[batchSize, seqLen]

  -- Forward pass (window pattern is in cfg)
  let logits ← forward params yarn input

  -- Check output
  let logitsMean := nn.item (nn.meanAll logits)
  let logitsMax := nn.item (nn.maxAll (nn.abs logits))

  -- Verify softcapping (values should be bounded)
  LeanTest.assertTrue (logitsMax < 35.0) s!"Logits should be softcapped below 35, got {logitsMax}"
  LeanTest.assertTrue (isFiniteFloat logitsMean) "Logits mean should be finite"

/-! ## 6. Loss and Training Tests -/

@[test]
def testLoss : IO Unit := do
  -- Tiny config
  let cfg : moddedGpt.Config := {
    vocabSize := 1000
    nLayer := 2
    nHead := 2
    headDim := 32
    modelDim := 64
    maxSeqLen := 128
    blockSize := 32
    numValueEmbeds := 2
  }

  let params ← ModdedGPTParams.init cfg
  let yarn ← YarnRotary.init cfg.headDim cfg.maxSeqLen

  -- Create input and targets
  let batchSize : UInt64 := 2
  let seqLen : UInt64 := 32
  let input ← randint 0 cfg.vocabSize.toInt64 #[batchSize, seqLen]
  let targets ← randint 0 cfg.vocabSize.toInt64 #[batchSize, seqLen]

  -- Compute loss (window pattern is now in cfg)
  let lossT ← moddedGpt.loss params yarn input targets
  let lossVal := nn.item lossT

  LeanTest.assertTrue (isFiniteFloat lossVal) "Loss should be finite"
  LeanTest.assertTrue (lossVal > 0) "Loss should be positive"

/-! ## 7. Data Loader Tests -/

@[test]
def testDataLoader : IO Unit := do
  -- Test hyperparams for step
  let (bs0, _wsS0, _wsL0) := getHyperparamsForStep 0
  let (_bs500, _wsS500, _wsL500) := getHyperparamsForStep 500
  let (bs1500, _wsS1500, _wsL1500) := getHyperparamsForStep 1500

  LeanTest.assertEqual bs0 8
  LeanTest.assertEqual bs1500 24

  -- Test tokens per step
  let tps := tokensPerStep 8 2048 1
  LeanTest.assertEqual tps (8 * 2048)

