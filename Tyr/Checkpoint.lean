import Tyr.TensorStruct
import Tyr.Torch
import Tyr.Log
import Lean.Data.Json
import Lean.Data.Json.FromToJson

/-!
# Tyr.Checkpoint

`Tyr.Checkpoint` provides model-agnostic checkpoint persistence based on `TensorStruct`.
It enables saving and restoring parameter trees (and mirrored optimizer trees) without
model-specific serialization code.

## Major Components

- `CheckpointMeta`: iteration/loss metadata persisted with checkpoints.
- Generic tensor-tree save/load via `saveParams` and `loadParams`.
- Full checkpoint helpers (`saveCheckpoint`, `loadCheckpoint`).
- Optimizer-state variants that reuse the same TensorStruct traversal patterns.

## Scope

This module targets straightforward local checkpoint persistence for training workflows.
It prioritizes generic structure traversal and reproducible load/save behavior over
custom binary formats or distributed snapshot orchestration.
-/

namespace torch.checkpoint

open torch
open torch.Log

/-- Checkpoint metadata. Version 2 persists the exact IEEE-754 loss bits. -/
structure CheckpointMeta where
  iteration : Nat
  bestValLoss : Float
  trainLoss : Float
  optimCount : Nat := 0
  deriving Repr, Inhabited

private structure MetadataV2 where
  version : Nat
  iteration : Nat
  bestValLossBits : UInt64
  trainLossBits : UInt64
  optimCount : Nat
  deriving Lean.ToJson, Lean.FromJson

private def newPrivateDir (parent : System.FilePath) : IO System.FilePath := do
  IO.FS.createDirAll parent
  let nonce ← IO.rand 0 18446744073709551615
  let path := parent / s!"snapshot-{← IO.monoMsNow}-{nonce}"
  -- mkdir is exclusive: a collision fails without touching existing data.
  IO.FS.createDir path
  pure path

private def writeFileAtomic (path : System.FilePath) (content : String) : IO Unit := do
  let scratch ← newPrivateDir (path.parent.getD ".")
  try
    let temporary := scratch / "pending"
    IO.FS.writeFile temporary content
    IO.FS.rename temporary path
  finally
    -- Once rename succeeds, cleanup must not report a failed publication:
    -- callers could otherwise remove the snapshot now named by CURRENT.
    try IO.FS.removeDirAll scratch catch _ => pure ()

/-- Write versioned, lossless metadata by atomically replacing the file. -/
def saveCheckpointMeta (m : CheckpointMeta) (path : String) : IO Unit := do
  let encoded : MetadataV2 := {
    version := 2, iteration := m.iteration, optimCount := m.optimCount
    bestValLossBits := m.bestValLoss.toBits, trainLossBits := m.trainLoss.toBits
  }
  writeFileAtomic path (Lean.toJson encoded).pretty

private def parseLegacyMeta (content : String) : Except String CheckpointMeta := do
  let mut fields : List (String × String) := []
  for raw in content.splitOn "\n" do
    let line := raw.trimAscii.toString
    if line.isEmpty then continue
    let [key, value] := line.splitOn "="
      | throw s!"Malformed checkpoint metadata line: {line}"
    if fields.any (fun entry => entry.1 == key) then
      throw s!"Duplicate checkpoint metadata field: {key}"
    fields := (key, value.trimAscii.toString) :: fields
  let required := fun key => match fields.lookup key with
    | some value => Except.ok value
    | none => Except.error s!"Missing checkpoint metadata field: {key}"
  let parseNat := fun key (value : String) => match value.toNat? with
    | some value => Except.ok value
    | none => Except.error s!"Invalid Nat value for {key}: {value}"
  let parseFloat := fun key value => do
    let json ← Lean.Json.parse value
    (Lean.fromJson? json : Except String Float).mapError
      (fun err => s!"Invalid Float value for {key}: {err}")
  let iteration ← parseNat "iteration" (← required "iteration")
  let bestValLoss ← parseFloat "bestValLoss" (← required "bestValLoss")
  let trainLoss ← parseFloat "trainLoss" (← required "trainLoss")
  let optimCount ← parseNat "optimCount" ((fields.lookup "optimCount").getD "0")
  pure { iteration, bestValLoss, trainLoss, optimCount }

/-- Read version 2 metadata or validated legacy key=value metadata. -/
def loadCheckpointMeta (path : String) : IO CheckpointMeta := do
  let content ← IO.FS.readFile path
  let parsed : Except String CheckpointMeta := do
    if content.trimAscii.toString.startsWith "{" then
      let json ← Lean.Json.parse content
      let m ← (Lean.fromJson? json : Except String MetadataV2)
      if m.version != 2 then throw s!"Unsupported checkpoint metadata version: {m.version}"
      pure {
        iteration := m.iteration
        optimCount := m.optimCount
        bestValLoss := Float.ofBits m.bestValLossBits
        trainLoss := Float.ofBits m.trainLossBits
      }
    else
      parseLegacyMeta content
  match parsed with
  | .ok m => pure m
  | .error err => throw <| IO.userError err

/-- Pin one committed snapshot directory for a multi-file read. Flat legacy
directories are returned unchanged. Use the returned directory for every file
in the read rather than resolving the pointer again between files. -/
def resolveSnapshot (dir : String) (pointer : String := "CURRENT") : IO String := do
  let manifest : System.FilePath := (System.FilePath.mk dir) / pointer
  if !(← manifest.pathExists) then return dir
  let name := (← IO.FS.readFile manifest).trimAscii.toString
  if !name.startsWith "snapshot-" || name.contains '/' || name.contains '\\' || name.contains '.' then
    throw <| IO.userError s!"Invalid checkpoint snapshot pointer: {manifest}"
  let snapshot := (System.FilePath.mk dir) / ".snapshots" / name
  if !(← snapshot.isDir) then
    throw <| IO.userError s!"Missing checkpoint snapshot: {snapshot}"
  pure snapshot.toString

/-- Publish an immutable snapshot only after its writer completes. Previous
snapshots remain readable by concurrent loaders. This gives atomic visibility,
not a power-loss durability guarantee; old snapshots can be pruned offline. -/
def publishSnapshot (dir : String) (pointer : String)
    (write : String → IO Unit) : IO Unit := do
  let snapshot ← newPrivateDir ((System.FilePath.mk dir) / ".snapshots")
  try
    write snapshot.toString
    writeFileAtomic ((System.FilePath.mk dir) / pointer) (snapshot.fileName.getD "")
  catch e =>
    IO.FS.removeDirAll snapshot
    throw e

/-- Check whether a committed checkpoint has metadata. -/
def checkpointExists (dir : String) : IO Bool := do
  let snapshot ← resolveSnapshot dir
  data.fileExists (snapshot ++ "/meta.txt")

/-! ## TensorStruct-based Save/Load

These functions use the TensorStruct typeclass to generically save and load
any model structure containing tensors. Tensors are saved with auto-generated
sequential names based on traversal order.
-/

/-- State for tracking tensor index during save -/
structure SaveState where
  index : IO.Ref Nat

/-- Save all tensors in a TensorStruct to a directory with a namePrefix -/
def saveParams [TensorStruct α]
    (params : α)
    (dir : String)
    (namePrefix : String := "param")
    (log : Handlers := {})
    : IO Unit := do
  IO.FS.createDirAll dir
  let indexRef ← IO.mkRef 0
  let _ ← TensorStruct.fold (fun {σ} (t : Tensor σ) (acc : IO Unit) => do
    acc
    let idx ← indexRef.get
    let path := s!"{dir}/{namePrefix}_{idx}.pt"
    data.saveTensor (s := σ.shape) t.raw path
    indexRef.set (idx + 1)
  ) (pure ()) params
  let finalIdx ← indexRef.get
  IO.FS.writeFile s!"{dir}/{namePrefix}_count.txt" (toString finalIdx)
  log.onInfo s!"Saved {finalIdx} tensors to {dir}"

/-- Load all tensors in a TensorStruct from a directory with a namePrefix.
    Requires a template structure to know the shapes. -/
def loadParams [TensorStruct α]
    (template : α)
    (dir : String)
    (namePrefix : String := "param")
    (log : Handlers := {})
    (asParameters : Bool := true)
    : IO α := do
  let countPath : System.FilePath := s!"{dir}/{namePrefix}_count.txt"
  if ← countPath.pathExists then
    let countText ← IO.FS.readFile countPath
    let some storedCount := countText.trimAscii.toString.toNat?
      | throw <| IO.userError s!"Invalid checkpoint tensor count: {countPath}"
    let expectedCount := TensorStruct.fold (fun {_s} _ (n : Nat) => n + 1) 0 template
    if storedCount != expectedCount then
      throw <| IO.userError s!"Checkpoint tensor count mismatch: expected {expectedCount}, stored {storedCount}"
  let indexRef ← IO.mkRef 0
  let result ← TensorStruct.mapM (fun {σ} (_expected : Tensor σ) => do
    let idx ← indexRef.get
    let path := s!"{dir}/{namePrefix}_{idx}.pt"
    let raw ← data.loadTensorExact σ.shape path
    let t ← match Tensor.ofRaw (σ := σ) raw with
      | .ok tensor => pure tensor
      | .error message => throw <| IO.userError s!"Invalid checkpoint tensor {path}: {message}"
    indexRef.set (idx + 1)
    pure t
  ) template
  let finalIdx ← indexRef.get
  log.onInfo s!"Loaded {finalIdx} tensors from {dir}"
  return if asParameters then TensorStruct.makeLeafParams result else TensorStruct.detach result

/-- Save full checkpoint (params + metadata) -/
def saveCheckpoint [TensorStruct α]
    (params : α)
    (iteration : Nat)
    (bestValLoss : Float)
    (trainLoss : Float)
    (dir : String)
    (namePrefix : String := "param")
    (log : Handlers := {})
    : IO Unit := do
  publishSnapshot dir "CURRENT" fun snapshot => do
    saveParams params snapshot namePrefix log
    saveCheckpointMeta { iteration, bestValLoss, trainLoss } (snapshot ++ "/meta.txt")
  log.onInfo s!"Checkpoint saved at iteration {iteration}"

/-- Load checkpoint (params + metadata) -/
def loadCheckpoint [TensorStruct α]
    (template : α)
    (dir : String)
    (namePrefix : String := "param")
    (log : Handlers := {})
    : IO (α × CheckpointMeta) := do
  let snapshot ← resolveSnapshot dir
  let m ← loadCheckpointMeta (snapshot ++ "/meta.txt")
  let params ← loadParams template snapshot namePrefix log
  log.onInfo s!"Checkpoint loaded from iteration {m.iteration}"
  return (params, m)

/-! ## Optimizer State Checkpointing

For optimizer states that mirror the model structure (like Adam's mu/nu),
use saveParams/loadParams with different namePrefixes.
-/

/-- Save optimizer state (for optimizers with matching structure like Adam) -/
def saveOptimizerState [TensorStruct α]
    (mu nu : α)
    (count : Nat)
    (dir : String)
    (log : Handlers := {})
    : IO Unit := do
  publishSnapshot dir "OPTIMIZER_CURRENT" fun snapshot => do
    saveParams mu snapshot "optim_mu" log
    saveParams nu snapshot "optim_nu" log
    IO.FS.writeFile (snapshot ++ "/optim_count.txt") (toString count)
  log.onInfo s!"Optimizer state saved to {dir}"

/-- Load optimizer state -/
def loadOptimizerState [TensorStruct α]
    (template : α)
    (dir : String)
    (log : Handlers := {})
    : IO (α × α × Nat) := do
  let snapshot ← resolveSnapshot dir "OPTIMIZER_CURRENT"
  let mu ← loadParams template snapshot "optim_mu" log false
  let nu ← loadParams template snapshot "optim_nu" log false
  let countStr ← IO.FS.readFile (snapshot ++ "/optim_count.txt")
  let some count := countStr.trimAscii.toString.toNat?
    | throw <| IO.userError "Invalid optimizer step count"
  log.onInfo s!"Optimizer state loaded from {dir} (count={count})"
  return (mu, nu, count)

/-- Check if optimizer state exists -/
def optimStateExists (dir : String) : IO Bool := do
  let snapshot ← resolveSnapshot dir "OPTIMIZER_CURRENT"
  data.fileExists (snapshot ++ "/optim_count.txt")

/-- Atomically save model parameters, Adam moments, count, and metadata together.
Use this API when a training resume must observe one consistent optimizer step. -/
def saveTrainingCheckpoint [TensorStruct α] (params mu nu : α)
    (metadata : CheckpointMeta) (dir : String) (log : Handlers := {}) : IO Unit :=
  publishSnapshot dir "CURRENT" fun snapshot => do
    saveParams params snapshot "param" log
    saveParams mu snapshot "optim_mu" log
    saveParams nu snapshot "optim_nu" log
    saveCheckpointMeta metadata (snapshot ++ "/meta.txt")

/-- Resolve the committed snapshot once before restoring all training state. -/
def loadTrainingCheckpoint [TensorStruct α] (template : α) (dir : String)
    (log : Handlers := {}) : IO (α × α × α × CheckpointMeta) := do
  let snapshot ← resolveSnapshot dir
  let metadata ← loadCheckpointMeta (snapshot ++ "/meta.txt")
  let params ← loadParams template snapshot "param" log
  let mu ← loadParams template snapshot "optim_mu" log false
  let nu ← loadParams template snapshot "optim_nu" log false
  pure (params, mu, nu, metadata)

end torch.checkpoint
