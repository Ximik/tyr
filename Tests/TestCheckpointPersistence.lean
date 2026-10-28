import Tests.Assertions
import Tyr.Checkpoint
import Tyr.Tokenizer.TokenBytes
import Tyr.Optim
import Tyr.Module.Derive
import LeanTest

namespace Tests.CheckpointPersistence

open torch torch.checkpoint LeanTest

@[test]
def testMetadataRoundTrip : IO Unit := IO.FS.withTempDir fun dir => do
  let path := (dir / "metadata.txt").toString
  for value in #[1.0e-8, -0.0, 1.2345678901234567, (1.0 / 0.0 : Float), Float.ofBits 1] do
    let m : CheckpointMeta := { iteration := 17, bestValLoss := value, trainLoss := value, optimCount := 3 }
    saveCheckpointMeta m path
    let loaded ← loadCheckpointMeta path
    assertEqual loaded.iteration m.iteration
    assertEqual loaded.optimCount m.optimCount
    assertEqual loaded.bestValLoss.toBits value.toBits
    assertEqual loaded.trainLoss.toBits value.toBits

@[test]
def testMetadataRejectsIncompleteAndUnknownVersions : IO Unit := IO.FS.withTempDir fun dir => do
  let path := (dir / "metadata.txt").toString
  for content in #["", "iteration=2", "iteration=2\nbestValLoss=1\ntrainLoss=2\niteration=3",
      "{\"version\":3,\"iteration\":0,\"optimCount\":0,\"bestValLossBits\":0,\"trainLossBits\":0}",
      "{\"version\":2}"] do
    IO.FS.writeFile path content
    Tests.assertThrows (loadCheckpointMeta path)
  IO.FS.writeFile path "iteration=7\nbestValLoss=1e-8\ntrainLoss=1.75"
  let legacy ← loadCheckpointMeta path
  assertEqual legacy.iteration 7
  assertTrue (Float.abs (legacy.bestValLoss - 1e-8) < 1e-20)
  assertEqual legacy.optimCount 0

private def assertTensorEqual {σ : StaticSpec} [DType.RealFloating σ.dtype] (a b : Tensor σ) : IO Unit := do
  assertTrue (a.runtimeShape == b.runtimeShape) "checkpoint shape"
  assertTrue (a.dtype == b.dtype) "checkpoint dtype"
  let error := Tensor.item (Tensor.sumAll (Tensor.abs (Tensor.sub a b)))
  assertTrue (!error.isNaN && !error.isInf && error < 1e-7) s!"checkpoint tensor error {error}"

@[test]
def testPlacementLoaderRetainsStoredShapeAndDType : IO Unit := IO.FS.withTempDir fun dir => do
  let path := (dir / "tensor.pt").toString
  let stored := Tensor.full #[2] 3.5
  Tensor.saveTensor stored path
  let loaded ← Tensor.loadTensorOn ⟨#[2], .Float32, .CPU⟩ path
  assertTensorEqual stored loaded
  assertTrue (loaded.runtimeDevice == .CPU) "placement loader returns the requested device"
  Tests.assertThrows (Tensor.loadTensorOn ⟨#[1, 2], .Float32, .CPU⟩ path) (some "shape")
  Tests.assertThrows (Tensor.loadTensorOn ⟨#[2], .Float64, .CPU⟩ path) (some "dtype")

@[test]
def testTrainingResumeRoundTrip : IO Unit := IO.FS.withTempDir fun dir => do
  let params : Tensor { shape := #[2], dtype := .Float32, device := .CPU } := Tensor.full #[2] 0.25
  let grads : Tensor { shape := #[2], dtype := .Float32, device := .CPU } := Tensor.full #[2] 0.125
  let adam := Optim.scale_by_adam (α := Tensor { shape := #[2], dtype := .Float32, device := .CPU })
  let (_, state) := adam.update params grads (adam.init params)
  let m : CheckpointMeta := { iteration := 1, bestValLoss := 1e-8, trainLoss := 0.5, optimCount := state.count }
  saveTrainingCheckpoint params state.mu state.nu m dir.toString
  let (loaded, mu, nu, metadata) ← loadTrainingCheckpoint params dir.toString
  assertTensorEqual params loaded
  assertTensorEqual state.mu mu
  assertTensorEqual state.nu nu
  assertTrue (!mu.requiresGrad && !nu.requiresGrad) "optimizer moments must remain detached"
  assertEqual metadata.optimCount state.count
  let (expectedUpdates, expectedState) := adam.update params grads state
  let (actualUpdates, actualState) := adam.update loaded grads { count := metadata.optimCount, mu, nu }
  assertTensorEqual expectedUpdates actualUpdates
  assertTensorEqual expectedState.mu actualState.mu
  assertTensorEqual expectedState.nu actualState.nu
  assertEqual actualState.count expectedState.count
  assertTensorEqual (params - expectedUpdates) (loaded - actualUpdates)

@[test]
def testInterruptedSavePreservesCommittedSnapshot : IO Unit := IO.FS.withTempDir fun dir => do
  let original : Tensor { shape := #[2], dtype := .Float32, device := .CPU } := Tensor.ones #[2]
  let m : CheckpointMeta := { iteration := 1, bestValLoss := 0.5, trainLoss := 0.75 }
  saveTrainingCheckpoint original original original m dir.toString
  let before ← IO.FS.readFile (dir / "CURRENT")
  let calls ← IO.mkRef (0 : Nat)
  let log : Log.Handlers := { onInfo := fun _ => do
    calls.modify (· + 1)
    if (← calls.get) == 2 then throw <| IO.userError "injected interrupted write" }
  Tests.assertThrows (saveTrainingCheckpoint (Tensor.zeros #[2]) original original
    { m with iteration := 2 } dir.toString log) (some "injected interrupted write")
  assertEqual (← IO.FS.readFile (dir / "CURRENT")) before
  let (loaded, _, _, metadata) ← loadTrainingCheckpoint original dir.toString
  assertTensorEqual loaded original
  assertEqual metadata.iteration 1
  assertEqual (← (dir / ".snapshots").readDir).size 1

@[test]
def testModelAndOptimizerStandaloneSnapshots : IO Unit := IO.FS.withTempDir fun dir => do
  let template : Tensor { shape := #[2], dtype := .Float32, device := .CPU } := Tensor.zeros #[2]
  saveCheckpoint (Tensor.ones #[2]) 7 0.25 0.5 dir.toString
  saveOptimizerState (Tensor.full #[2] 0.1) (Tensor.full #[2] 0.2) 9 dir.toString
  assertTrue (← checkpointExists dir.toString)
  assertTrue (← optimStateExists dir.toString)
  let (params, metadata) ← loadCheckpoint template dir.toString
  let (mu, nu, count) ← loadOptimizerState template dir.toString
  assertTensorEqual params (Tensor.ones #[2])
  assertEqual metadata.iteration 7
  assertTensorEqual mu (Tensor.full #[2] 0.1)
  assertTensorEqual nu (Tensor.full #[2] 0.2)
  assertEqual count 9
  Tests.assertThrows (loadCheckpoint (Tensor.zeros #[3]) dir.toString) (some "shape mismatch")

@[test]
def testStoredDTypeMustMatchTypedTemplate : IO Unit := IO.FS.withTempDir fun dir => do
  let stored := Tensor.toBFloat16 (Tensor.ones #[2])
  saveCheckpoint stored 1 0.0 0.0 dir.toString
  Tests.assertThrows (loadCheckpoint (Tensor.zeros #[2]) dir.toString) (some "dtype")
  let (loaded, _) ← loadCheckpoint (Tensor.toBFloat16 (Tensor.zeros #[2])) dir.toString
  assertTrue (loaded.dtype == .BFloat16) "matching typed templates preserve stored dtype"
  assertTensorEqual loaded stored

@[test]
def testTensorCountMismatchRejected : IO Unit := IO.FS.withTempDir fun dir => do
  let params : Array (Tensor { shape := #[2], dtype := .Float32, device := .CPU }) := #[Tensor.ones #[2], Tensor.zeros #[2]]
  saveCheckpoint params 1 0.0 0.0 dir.toString
  Tests.assertThrows (loadCheckpoint (#[Tensor.zeros #[2]] : Array (Tensor { shape := #[2], dtype := .Float32, device := .CPU })) dir.toString)
    (some "tensor count mismatch")
  Tests.assertThrows (loadCheckpoint (#[] : Array (Tensor { shape := #[2], dtype := .Float32, device := .CPU })) dir.toString)
    (some "tensor count mismatch")
  Tests.assertThrows (loadCheckpoint (#[Tensor.zeros #[2], Tensor.zeros #[2], Tensor.zeros #[2]] : Array (Tensor { shape := #[2], dtype := .Float32, device := .CPU })) dir.toString)
    (some "tensor count mismatch")

@[test]
def testReaderPinsOneSnapshotAcrossReplacement : IO Unit := IO.FS.withTempDir fun dir => do
  let old : Tensor { shape := #[2], dtype := .Float32, device := .CPU } := Tensor.ones #[2]
  let fresh : Tensor { shape := #[2], dtype := .Float32, device := .CPU } := Tensor.full #[2] 2.0
  let m : CheckpointMeta := { iteration := 1, bestValLoss := 0.0, trainLoss := 0.0 }
  saveTrainingCheckpoint old old old m dir.toString
  let replaced ← IO.mkRef false
  let log : Log.Handlers := { onInfo := fun _ => do
    if !(← replaced.get) then
      replaced.set true
      saveTrainingCheckpoint fresh fresh fresh { m with iteration := 2 } dir.toString }
  let (params, mu, nu, metadata) ← loadTrainingCheckpoint old dir.toString log
  assertTensorEqual params old
  assertTensorEqual mu old
  assertTensorEqual nu old
  assertEqual metadata.iteration 1
  let (next, _, _, nextMetadata) ← loadTrainingCheckpoint old dir.toString
  assertTensorEqual next fresh
  assertEqual nextMetadata.iteration 2

@[test]
def testTokenBytesExactPersistence : IO Unit := IO.FS.withTempDir fun dir => do
  let path := (dir / "token-bytes.pt").toString
  let mapping := tokenizer.TokenBytes.dummy 4
  mapping.save path
  let loaded ← tokenizer.TokenBytes.load path 4
  LeanTest.assertEqual (← loaded.bytes.toUInt64Array) #[2, 2, 2, 2]
  Tensor.saveTensor (mapping.bytes.reshape #[2, 2]) path
  Tests.assertThrows (tokenizer.TokenBytes.load path 4) (some "shape")
  Tensor.saveTensor (Tensor.ones #[4]) path
  Tests.assertThrows (tokenizer.TokenBytes.load path 4) (some "dtype")

end Tests.CheckpointPersistence
