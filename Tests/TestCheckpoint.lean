/-
  Tests/TestCheckpoint.lean

  Tests for checkpointing functionality and TensorStruct integration.
-/
import Tyr.Typed.Neural
import Tyr.Checkpoint
import Examples.GPT.GPT
import Examples.NanoChat.ModdedGPT
import Examples.NanoChat.ModdedTrain
import Tyr.Optim.DistAdam
import Tyr.Optim.NorMuon
import LeanTest

open torch
open torch.ModdedTrain
open torch.Optim
open torch.checkpoint

@[test]
def testCheckpointMetaParsesFloatLosses : IO Unit := do
  let ts ← IO.monoMsNow
  let path := s!"/tmp/tyr_checkpoint_meta_{ts}.txt"
  let checkpointMeta : CheckpointMeta := {
    iteration := 17
    bestValLoss := 0.625
    trainLoss := 1.75
    optimCount := 3
  }
  saveCheckpointMeta checkpointMeta path
  let loaded ← loadCheckpointMeta path
  LeanTest.assertEqual loaded.iteration checkpointMeta.iteration
  LeanTest.assertEqual loaded.optimCount checkpointMeta.optimCount
  LeanTest.assertTrue (Float.abs (loaded.bestValLoss - checkpointMeta.bestValLoss) < 1e-6)
    s!"bestValLoss should round-trip: {loaded.bestValLoss} vs {checkpointMeta.bestValLoss}"
  LeanTest.assertTrue (Float.abs (loaded.trainLoss - checkpointMeta.trainLoss) < 1e-6)
    s!"trainLoss should round-trip: {loaded.trainLoss} vs {checkpointMeta.trainLoss}"

@[test]
def testCheckpointCreation : IO Unit := do
  -- Create tiny config
  let cfg : moddedGpt.Config := {
    vocabSize := 128
    nLayer := 2
    nHead := 2
    headDim := 16
    modelDim := 32
    maxSeqLen := 64
    blockSize := 32
    numValueEmbeds := 1
  }

  -- Init params
  let params ← moddedGpt.ModdedGPTParams.init cfg

  -- Init optimizer state
  let optState := OptimizerState.init cfg params

  -- Create checkpoint
  let ckpt : Checkpoint cfg := {
    params := params
    optState := optState
    step := 10
    bestValLoss := 0.5
  }

  -- Test saveCheckpoint (functionality check)
  -- Writes the checkpoint tree to a temp dir (never the repo root) and
  -- removes it afterwards; we also verify TensorStruct statistics below.
  let ts ← IO.monoMsNow
  let ckptDir := s!"/tmp/tyr_checkpoint_test_{ts}.ckpt"
  saveCheckpoint ckpt ckptDir
  LeanTest.assertTrue (← System.FilePath.pathExists ckptDir)
    s!"Checkpoint directory should be created at {ckptDir}"
  IO.FS.removeDirAll ckptDir

  -- Verify TensorStruct on params directly
  let numParams := TensorStruct.fold (fun {s} _t acc => acc + s.shape.foldl (· * ·) 1) 0 params
  LeanTest.assertTrue (numParams > 0) "Parameter count should be positive"
  
  -- Rough calculation: 
  -- embed: 128*32 = 4096
  -- valueEmbeds: 1 * 128*32 = 4096
  -- blocks (2):
  --   attn qkvo: 4*32 * 2*16 = 128 * 32 = 4096
  --   attn gate: 12 * 2 = 24
  --   mlp fc: 4*32 * 32 = 4096
  --   mlp proj: 4*32 * 32 = 4096
  -- lmHead: 32 * 128 = 4096
  -- scalars: ...
  -- Total should be > 20000
  LeanTest.assertTrue (numParams > 20000) s!"Parameter count {numParams} too low"

@[test]
def testDistAdamCheckpointing : IO Unit := do
  -- Test that DistAdam.ParamState implements TensorStruct correctly
  let shape : Shape := #[10, 10]
  let param := Tensor.zeros shape
  let state := DistAdam.initParamState param
  
  -- Verify structure traversal
  let numTensors := TensorStruct.fold (fun {_s} _t acc => acc + 1) 0 state
  LeanTest.assertEqual numTensors 2 "DistAdam state should have 2 tensors (expAvg, expAvgSq)"
  
  let totalElements := TensorStruct.fold (fun {s} _t acc => acc + s.shape.foldl (· * ·) 1) 0 state
  LeanTest.assertEqual totalElements 200 "DistAdam state should have 200 elements"

  -- Verify map works
  let stateOnes := FloatingTensorStruct.map (fun t => Tensor.addScalar t 1.0) state
  let sumOnes := TensorStruct.fold (fun {_s} t acc => acc + Tensor.item (Tensor.sumAll t)) 0.0 stateOnes
  LeanTest.assertEqual sumOnes 200.0 "Mapped state should sum to 200"

@[test]
def testNorMuonCheckpointing : IO Unit := do
  -- Test that NorMuon.ParamState implements TensorStruct correctly
  let shape : Shape := #[10, 10]
  let param := Tensor.zeros shape
  let state := NorMuon.initParamState param
  
  -- Initial state has no tensors (Options are none)
  let numTensors := TensorStruct.fold (fun {_s} _t acc => acc + 1) 0 state
  LeanTest.assertEqual numTensors 0 "Initial NorMuon state should have 0 tensors"
  
  -- Simulate state update
  let momentum := Tensor.ones shape
  let secondMoment := Tensor.ones #[]
  let stateWithData : NorMuon.ParamState ⟨shape, .Float32, .CPU⟩ := {
    momentumBuffer := some momentum
    secondMoment := some secondMoment
    step := 1
  }
  
  -- Verify structure traversal with data
  let numTensorsFilled := TensorStruct.fold (fun {_s} _t acc => acc + 1) 0 stateWithData
  LeanTest.assertEqual numTensorsFilled 2 "Filled NorMuon state should have 2 tensors"
  
  let totalElements := TensorStruct.fold (fun {s} _t acc => acc + s.shape.foldl (· * ·) 1) 0 stateWithData
  LeanTest.assertEqual totalElements 101 "Filled NorMuon state should have 101 elements (100 + 1)"

@[test]
def testLoadCheckpointMock : IO Unit := do
  -- Test loadCheckpoint stub
  let cfg : moddedGpt.Config := moddedGpt.Config.default
  let ckpt ← loadCheckpoint cfg "nonexistent.ckpt"
  match ckpt with
  | none => LeanTest.assertTrue true
  | some _ => LeanTest.fail "Should return none for stub implementation"
