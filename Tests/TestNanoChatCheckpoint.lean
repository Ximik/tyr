import Tests.Assertions
import Examples.NanoChat.ModdedTrain
import LeanTest

namespace Tests.NanoChatCheckpoint

open torch torch.ModdedTrain LeanTest

private def tinyConfig : moddedGpt.Config := {
  vocabSize := 16, nLayer := 7, nHead := 1, headDim := 12, modelDim := 12,
  maxSeqLen := 8, blockSize := 8, numValueEmbeds := 1
}

private def assertTreeEqual [TensorStruct α] (actual expected : α) : IO Unit := do
  let describe := fun (value : α) =>
    TensorStruct.fold (fun {_s} t acc => acc.push (t.runtimeShape, t.raw.dtype)) #[] value
  assertTrue (describe actual == describe expected) "checkpoint tensor traversal metadata"
  let errors := TensorStruct.zipWith (fun a b => Tensor.Preserving.abs (Tensor.Preserving.sub a b)) actual expected
  let error := TensorStruct.fold (fun {_s} t acc => acc + Tensor.item (Tensor.sumAll t)) 0.0 errors
  assertTrue (error.isFinite && error < 1e-6) s!"checkpoint tensor difference {error}"

private def markedMuon {s : Shape} (step : Nat) : Optim.NorMuon.ParamState ⟨s, .Float32, .CPU⟩ := {
  momentumBuffer := some (Tensor.full s (step.toFloat / 128.0))
  -- The second moment is a genuine rank-zero tensor.
  secondMoment := some (Tensor.full #[] (step.toFloat / 64.0))
  step
}

private def trainedCheckpoint : IO (Checkpoint (device := .CPU) tinyConfig) := do
  let initialParams ← moddedGpt.ModdedGPTParams.init tinyConfig
  let params := TensorStruct.map (fun t => Tensor.Preserving.addScalar (Tensor.zerosLike t) 0.25) initialParams
  let initialOpt := OptimizerState.init tinyConfig params 0.023 0.017
  let grads := TensorStruct.map (fun t => Tensor.Preserving.addScalar (Tensor.zerosLike t) 0.125) params
  let opt := Optim.adamw (α := moddedGpt.ModdedGPTParams (device := .CPU) tinyConfig) (lr := 0.023) (weight_decay := 0.017)
  let (params, adamState) ← Optim.step opt params grads initialOpt.adamState
  let (_, embedState) ← Optim.DistAdam.stepSingle params.embed grads.embed
    initialOpt.dualState.embed { lr := 0.023 }
  let dualState := { initialOpt.dualState with
    embed := embedState
    lmHead := { initialOpt.dualState.lmHead with step := 3 }
    scalars := { initialOpt.dualState.scalars with step := 4 }
    valueEmbeds := initialOpt.dualState.valueEmbeds.map (fun state => { state with step := 5 })
    smearGate := markedMuon 6
    blocks := initialOpt.dualState.blocks.mapIdx fun i block => {
      attn := block.attn.map fun _ => {
        wQ := markedMuon (10 * i + 7), wK := markedMuon (10 * i + 8),
        wV := markedMuon (10 * i + 9), wO := markedMuon (10 * i + 10) }
      cFc := markedMuon (10 * i + 11)
      cProj := {
        momentumBuffer := some (Tensor.full _ ((10 * i + 12).toFloat / 128.0))
        secondMoment := none
        step := 10 * i + 12
      }
    }
  }
  let optState := { initialOpt with adamState := adamState, dualState := dualState, step := 13 }
  pure { params := params, optState := optState, step := 14, bestValLoss := 1.0e-8 }

private def metadata (tokens : UInt64) : CheckpointMetadata := {
  modelConfigRepr := s!"{repr tinyConfig}"
  modelConfig? := some tinyConfig
  hyperparameters := { numIterations := 20 }
  totalTokens := tokens
  dataCursor? := some { trainPathIdx := 2, globalStep := 7, epoch := 1, batchCount := 3, bosCurrentPos := 11 }
}

@[test] def testNanoChatSnapshotRestoresOptimizerUpdate : IO Unit := IO.FS.withTempDir fun dir => do
  let original ← trainedCheckpoint
  torch.ModdedTrain.saveCheckpoint original dir.toString (some (metadata 1234))
  assertTrue (← torch.ModdedTrain.checkpointExists dir.toString) "published NanoChat checkpoint exists"
  let some (loaded, some savedMeta) ← loadCheckpointForResume tinyConfig dir.toString true
    | fail "NanoChat snapshot should restore tensors and training metadata"
  assertTreeEqual loaded.params original.params
  assertTreeEqual loaded.optState.adamState original.optState.adamState
  assertTreeEqual loaded.optState.dualState original.optState.dualState
  assertEqual loaded.step original.step
  assertEqual loaded.optState.step original.optState.step
  assertEqual loaded.optState.adamState.fst.count original.optState.adamState.fst.count
  assertEqual loaded.bestValLoss.toBits original.bestValLoss.toBits
  assertEqual loaded.optState.baseLr.toBits original.optState.baseLr.toBits
  assertEqual loaded.optState.weightDecay.toBits original.optState.weightDecay.toBits
  assertEqual savedMeta.totalTokens 1234
  assertTrue (savedMeta.dataCursor?.isSome) "data-loader cursor should be restored"
  let dual := loaded.optState.dualState
  assertEqual dual.embed.step original.optState.dualState.embed.step
  assertEqual dual.lmHead.step 3
  assertEqual dual.scalars.step 4
  let some valueEmbed := dual.valueEmbeds[0]? | fail "test value embedding should exist"
  assertEqual valueEmbed.step 5
  assertEqual dual.smearGate.step 6
  let some block := dual.blocks[0]? | fail "test block should exist"
  let some attn := block.attn | fail "test block should contain attention state"
  assertEqual #[attn.wQ.step, attn.wK.step, attn.wV.step, attn.wO.step, block.cFc.step, block.cProj.step]
    #[7, 8, 9, 10, 11, 12]
  assertTrue (block.cProj.secondMoment.isNone) "absent optimizer buffer stays absent"
  let some skipped := dual.blocks[5]? | fail "skipped-attention test block should exist"
  assertTrue skipped.attn.isNone "skipped attention must not consume optimizer layout entries"
  assertEqual skipped.cFc.step 61
  let some afterSkipped := dual.blocks[6]? | fail "post-skip test block should exist"
  assertEqual afterSkipped.cFc.step 71
  let some moment := dual.smearGate.secondMoment | fail "second moment should be restored"
  assertEqual moment.runtimeShape #[]
  let detached := TensorStruct.fold (fun {_s} t ok => ok && !t.requiresGrad) true loaded.optState.dualState
  assertTrue detached "optimizer buffers should be detached"

  -- Compare the next update, including bias correction and Muon momentum.
  let grads := TensorStruct.map (fun t => Tensor.Preserving.addScalar (Tensor.zerosLike t) 0.125) original.params
  let opt := Optim.adamw (α := moddedGpt.ModdedGPTParams (device := .CPU) tinyConfig)
    (lr := original.optState.baseLr) (weight_decay := original.optState.weightDecay)
  let (expectedParams, expectedAdam) ← Optim.step opt original.params grads original.optState.adamState
  let (actualParams, actualAdam) ← Optim.step opt loaded.params grads loaded.optState.adamState
  assertTreeEqual actualParams expectedParams
  assertTreeEqual actualAdam expectedAdam
  assertEqual actualAdam.fst.count expectedAdam.fst.count
  let adamCfg : Optim.DistAdam.Config := { lr := original.optState.baseLr }
  let (expectedEmbed, expectedEmbedState) ← Optim.DistAdam.stepSingle original.params.embed grads.embed
    original.optState.dualState.embed adamCfg
  let (actualEmbed, actualEmbedState) ← Optim.DistAdam.stepSingle loaded.params.embed grads.embed dual.embed adamCfg
  assertTreeEqual actualEmbed expectedEmbed
  assertTreeEqual actualEmbedState expectedEmbedState
  assertEqual actualEmbedState.step expectedEmbedState.step
  let muonCfg : Optim.NorMuon.Config := { lr := original.optState.baseLr, numIters := 1 }
  let (expectedGate, expectedGateState) ← Optim.NorMuon.stepSingle original.params.smearGate.weight
    grads.smearGate.weight original.optState.dualState.smearGate muonCfg 1.0 1.0
  let (actualGate, actualGateState) ← Optim.NorMuon.stepSingle loaded.params.smearGate.weight
    grads.smearGate.weight dual.smearGate muonCfg 1.0 1.0
  assertTreeEqual actualGate expectedGate
  assertTreeEqual actualGateState expectedGateState
  assertEqual actualGateState.step expectedGateState.step

@[test] def testNanoChatSnapshotReplacementAndCorruption : IO Unit := IO.FS.withTempDir fun dir => do
  let original ← trainedCheckpoint
  torch.ModdedTrain.saveCheckpoint original dir.toString (some (metadata 1234))
    (writeExtra? := some fun snapshot =>
      IO.FS.writeFile (s!"{snapshot}/stream_cursor_rank0.json" : System.FilePath) "old cursor")
  let pinned ← torch.checkpoint.resolveSnapshot dir.toString
  torch.ModdedTrain.saveCheckpoint { original with step := 99, bestValLoss := 0.5 }
    dir.toString (some (metadata 9999))
    (writeExtra? := some fun snapshot =>
      IO.FS.writeFile (s!"{snapshot}/stream_cursor_rank0.json" : System.FilePath) "new cursor")
  let some (old, some oldMeta) ← loadCheckpointForResume tinyConfig pinned true
    | fail "previous snapshot should remain readable after replacement"
  assertEqual old.step 14
  assertEqual oldMeta.totalTokens 1234
  let some (latest, some latestMeta) ← loadCheckpointForResume tinyConfig dir.toString true
    | fail "latest snapshot should be readable"
  assertEqual latest.step 99
  assertEqual latestMeta.totalTokens 9999
  let current ← torch.checkpoint.resolveSnapshot dir.toString
  assertEqual (← IO.FS.readFile (s!"{pinned}/stream_cursor_rank0.json" : System.FilePath)) "old cursor"
  assertEqual (← IO.FS.readFile (s!"{current}/stream_cursor_rank0.json" : System.FilePath)) "new cursor"
  Tests.assertThrows (torch.ModdedTrain.saveCheckpoint { original with step := 100 }
    dir.toString (some (metadata 10000))
    (writeExtra? := some fun _ => throw <| IO.userError "injected cursor save failure"))
    (some "injected cursor save failure")
  assertEqual (← torch.checkpoint.resolveSnapshot dir.toString) current
  let metaPath : System.FilePath := s!"{current}/training_meta.json"
  let metaContents ← IO.FS.readFile metaPath
  IO.FS.removeFile metaPath
  assertTrue (← loadCheckpointForResume tinyConfig dir.toString true).isNone
    "missing promised training metadata must not reset hyperparameters or data position"
  IO.FS.writeFile metaPath metaContents
  IO.FS.removeFile (s!"{current}/optim_dual_0.pt" : System.FilePath)
  assertTrue (← torch.ModdedTrain.loadCheckpoint (device := .CPU) tinyConfig dir.toString true).isNone
    "a corrupt new dual-state snapshot must not silently reset optimizer state"
  IO.FS.removeFile (s!"{current}/state_meta.json" : System.FilePath)
  assertTrue (← loadCheckpointForResume tinyConfig dir.toString true).isNone
    "a new snapshot must require optimizer layout metadata"
  IO.FS.removeFile (s!"{current}/step.pt" : System.FilePath)
  assertTrue (← torch.ModdedTrain.checkpointExists dir.toString)
    "a damaged published snapshot must not trigger a fresh auto-resume run"

@[test] def testNanoChatLegacyFlatCheckpoint : IO Unit := IO.FS.withTempDir fun dir => do
  let params ← moddedGpt.ModdedGPTParams.init tinyConfig
  let original : Checkpoint (device := .CPU) tinyConfig := {
    params, optState := OptimizerState.init tinyConfig params 0.25 0.125,
    step := 7, bestValLoss := 0.5
  }
  let root := (dir / "new").toString
  torch.ModdedTrain.saveCheckpoint original root (some (metadata 33))
  let snapshot ← torch.checkpoint.resolveSnapshot root
  let legacy := dir / "legacy"
  IO.FS.rename snapshot legacy
  -- Old flat checkpoints contain scalar .pt files but neither state layout nor
  -- the new per-prefix tensor counts.
  for name in #["state_meta.json", "param_count.txt", "optim_adam_count.txt", "optim_dual_count.txt"] do
    IO.FS.removeFile (legacy / name)
  let some (loaded, some savedMeta) ← loadCheckpointForResume tinyConfig legacy.toString true
    | fail "legacy flat checkpoint should remain loadable"
  assertTreeEqual loaded.params original.params
  assertTreeEqual loaded.optState.adamState original.optState.adamState
  assertEqual loaded.step 7
  assertEqual loaded.bestValLoss.toBits (0.5 : Float).toBits
  assertEqual savedMeta.totalTokens 33

@[test] def testNanoChatIndexedTrainingStep : IO Unit := do
  let params ← moddedGpt.ModdedGPTParams.init tinyConfig
  let yarn ← moddedGpt.YarnRotary.init tinyConfig.headDim tinyConfig.maxSeqLen
  let input := (Tensor.fromInt64Array #[0, 1, 2, 3]).reshapeChecked #[1, 4]
  let target := (Tensor.fromInt64Array #[1, 2, 3, 4]).reshapeChecked #[1, 4]
  let opt := OptimizerState.init tinyConfig params
  let (updated, nextOpt, result) ← trainStep params yarn input target opt
    { numIterations := 10 } 1.0
  assertTrue result.loss.isFinite "Indexed NanoChat forward/backward produces a finite loss"
  assertEqual result.tokensProcessed 4
  assertEqual nextOpt.step 1
  let change := Tensor.item (Tensor.sumAll (Tensor.abs (updated.lmHead.weight - params.lmHead.weight)))
  assertTrue (change.isFinite && change > 0) "Optimizer updates the output head"


def run : IO Unit := do
  testNanoChatSnapshotRestoresOptimizerUpdate
  testNanoChatSnapshotReplacementAndCorruption
  testNanoChatLegacyFlatCheckpoint

end Tests.NanoChatCheckpoint
