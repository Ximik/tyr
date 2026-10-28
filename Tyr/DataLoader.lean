import Tyr.Torch
import Tyr.Typed
import Tyr.Typed.Neural
import Tyr.Distributed
import Lean.Data.Json.FromToJson

/-!
# Tyr.DataLoader

`Tyr.DataLoader` implements Tyr's low-level token/shard loading pipeline.
It handles shard discovery, BOS-aware batching, and distributed rank-aware iteration
for language-model style training corpora.

## Major Components

- Loader configuration (`Config`) and shard-kind routing.
- Path resolution helpers for file, directory, and prefix-based shard specs.
- BOS-aware batch extraction (`BOSFinder`) for document boundary handling.
- Iteration and sharding utilities for distributed training setups.

## Scope

This module focuses on data movement and batch construction mechanics.
Higher-level training/evaluation orchestration belongs in pipeline/task modules.
-/

namespace torch.DataLoader

open torch

variable {device : Device}

/-! ## Configuration -/

structure Config where
  dataPath : String := "data"
  valPath : Option String := none
  seqLen : Nat := 2048
  bosToken : UInt64 := 50256
  /-- Compatibility field only: this loader performs synchronous reads. -/
  numWorkers : Nat := 0
  /-- Compatibility field only: this loader does not prefetch shards. -/
  bufferSize : Nat := 0
  seed : UInt64 := 42
  /-- Shuffle documents within each rank-local file partition each epoch. -/
  shuffle : Bool := true
  deriving Repr, Inhabited

/-- Validate the batch token count before it reaches native code.
    `extra` is used for the additional GPT target token in each row. -/
def checkedBatchTokens (batchSize seqLen : Nat) (extra : Nat := 0) :
    IO { count : Nat // count = batchSize * (seqLen + extra) } := do
  if batchSize == 0 || seqLen == 0 then
    throw <| IO.userError "Batch size and sequence length must be positive"
  let count := batchSize * (seqLen + extra)
  if count > 9223372036854775807 then
    throw <| IO.userError "Batch token count exceeds the native signed 64-bit range"
  return ⟨count, rfl⟩

/-! ## Path Resolution -/

inductive ShardKind where
  | train
  | val

private def shardPrefix : ShardKind → String
  | .train => "fineweb_train_"
  | .val => "fineweb_val_"

private def sortPaths (paths : Array String) : Array String :=
  paths.qsort (· < ·)

private def listShardFilesInDir (dir : System.FilePath) (kind : ShardKind)
    : IO (Array String) := do
  let entries ← dir.readDir
  let mut preferred : Array String := #[]
  let mut anyBin : Array String := #[]
  for e in entries do
    if e.fileName.endsWith ".bin" then
      let path := e.path.toString
      anyBin := anyBin.push path
      if e.fileName.startsWith (shardPrefix kind) then
        preferred := preferred.push path
  let chosen := if preferred.isEmpty then anyBin else preferred
  return sortPaths chosen

/--
Resolve a shard path specification into concrete `.bin` files.

Supports:
1. Exact file path.
2. Directory path (prefers `fineweb_train_*.bin` / `fineweb_val_*.bin`).
3. Prefix path whose parent exists (e.g. `data/fineweb_val` -> `data/fineweb_val_*.bin`).
-/
def resolveShardPaths (pathSpec : String) (kind : ShardKind) : IO (Array String) := do
  let p : System.FilePath := ⟨pathSpec⟩
  if ← p.pathExists then
    if ← p.isDir then
      let files ← listShardFilesInDir p kind
      if files.isEmpty then
        throw <| IO.userError s!"No .bin files found under directory: {pathSpec}"
      return files
    else
      return #[pathSpec]

  let cwd : System.FilePath := ⟨"."⟩
  let parent := p.parent.getD cwd
  let stem := p.fileName.getD pathSpec
  if (← parent.pathExists) && (← parent.isDir) then
    let entries ← parent.readDir
    let mut prefixed : Array String := #[]
    for e in entries do
      if e.fileName.startsWith stem && e.fileName.endsWith ".bin" then
        prefixed := prefixed.push e.path.toString
    let files := sortPaths prefixed
    if !files.isEmpty then
      return files

  throw <| IO.userError s!"Could not resolve shard path: {pathSpec}"

/-! ## BOS Finder -/

structure BOSFinder where
  bosToken : UInt64
  bosPositions : Array Nat
  currentPos : Nat
  dataLen : Nat
  /-- Source document starts in stream order; empty means source order. -/
  orderedStarts : Array Nat := #[]
  /-- Exclusive cumulative document ends in the reordered stream. -/
  orderedEnds : Array Nat := #[]
  deriving Repr

def BOSFinder.init {n : Nat} (tokens : Tensor { shape := #[n], dtype := .Int64, device := device }) (bosToken : UInt64)
    : IO BOSFinder := do
  let dataLen := n
  let ⟨_, positionsTensor⟩ ← Tensor.findTokenPositions tokens bosToken.toInt64
  let positions := (← positionsTensor.toUInt64Array).map (·.toNat)
  -- Preserve a prefix before the first BOS as a document fragment. Rank
  -- partitions may begin inside a document; no tokens are discarded.
  let positions := if positions[0]? == some 0 then positions else #[0] ++ positions
  return {
    bosToken := bosToken
    bosPositions := positions
    currentPos := 0
    dataLen := dataLen
  }

def BOSFinder.findNextValidStart (finder : BOSFinder) (after : Nat) : Option Nat :=
  finder.bosPositions.find? (· >= after)

/-- Consume up to `count` real tokens in document order. Documents are packed
    across batch boundaries, never padded or repeated inside a shard.
    The result holds exactly `min count (remaining)` Int64 tokens. -/
def BOSFinder.take (finder : BOSFinder) {n : Nat} (tokens : Tensor { shape := #[n], dtype := .Int64, device := device })
    (count : Nat)
    : IO (Tensor { shape := #[min count (finder.dataLen - finder.currentPos)], dtype := .Int64, device := device } × BOSFinder) := do
  if finder.currentPos > finder.dataLen then
    throw <| IO.userError "Data cursor position exceeds shard length"
  let count := min count (finder.dataLen - finder.currentPos)
  if count == 0 then
    return (tokens.sliceChecked 0 0 0 #[count], finder)
  let endPos := finder.currentPos + count
  if finder.orderedStarts.isEmpty then
    return (tokens.sliceChecked 0 finder.currentPos count #[count],
      { finder with currentPos := endPos })
  -- Binary search avoids rescanning every earlier document for each batch.
  let mut lo := 0
  let mut hi := finder.orderedEnds.size
  while lo < hi do
    let mid := (lo + hi) / 2
    if finder.orderedEnds[mid]! <= finder.currentPos then lo := mid + 1 else hi := mid
  let mut doc := lo
  let mut pos := finder.currentPos
  let mut chunks : Array (Sigma fun shape => Tensor { shape := shape, dtype := .Int64, device := device }) := #[]
  while pos < endPos do
    let docEnd := finder.orderedEnds[doc]!
    let docStart := if doc == 0 then 0 else finder.orderedEnds[doc - 1]!
    let sourceStart := finder.orderedStarts[doc]! + (pos - docStart)
    let len := min (endPos - pos) (docEnd - pos)
    chunks := chunks.push (Tensor.packShape (tokens.sliceChecked 0 sourceStart len #[len]))
    pos := pos + len
    doc := doc + 1
  -- Concatenating the document slices yields exactly `count` tokens.
  let chunk ← match Tensor.catMany? chunks 0 #[count] with
    | .ok tensor => pure tensor
    | .error message => throw <| IO.userError s!"BOS document batch: {message}"
  return (chunk, { finder with currentPos := endPos })

def BOSFinder.getBatch (finder : BOSFinder) {n : Nat} (tokens : Tensor { shape := #[n], dtype := .Int64, device := device })
    (batchSize seqLen : Nat)
    : IO (Option (Tensor { shape := #[batchSize, seqLen], dtype := .Int64, device := device }) × BOSFinder) := do
  let ⟨requiredLen, hRequired⟩ ← checkedBatchTokens batchSize seqLen
  if finder.currentPos > finder.dataLen then
    throw <| IO.userError "Data cursor position exceeds shard length"
  if hFits : requiredLen ≤ finder.dataLen - finder.currentPos then
    let (batch, finder) ← finder.take tokens requiredLen
    return (some (batch.reshape #[batchSize, seqLen] (by rw [Nat.min_eq_left hFits, hRequired]; simp)), finder)
  else
    return (none, finder)

def BOSFinder.reset (finder : BOSFinder) : BOSFinder :=
  { finder with currentPos := 0 }

/-! ## Randomization Utilities -/

private def lcgNext (state : UInt64) : UInt64 :=
  state * 6364136223846793005 + 1442695040888963407

private def fisherYatesShuffle (arr : Array Nat) (seed : UInt64) : Array Nat := Id.run do
  if arr.size <= 1 then return arr
  let mut result := arr
  let mut state := seed
  for i in [:(arr.size - 1)] do
    state := lcgNext state
    let range := arr.size - i
    let j := i + (state % range.toUInt64).toNat
    let tmp := result[i]!
    result := result.set! i result[j]!
    result := result.set! j tmp
  return result

def BOSFinder.shuffle (finder : BOSFinder) (seed : UInt64) : BOSFinder :=
  Id.run do
    let order := fisherYatesShuffle (List.range finder.bosPositions.size).toArray seed
    let mut starts := #[]
    let mut ends := #[]
    let mut total := 0
    for idx in order do
      let start := finder.bosPositions[idx]!
      let stop := finder.bosPositions.getD (idx + 1) finder.dataLen
      starts := starts.push start
      total := total + (stop - start)
      ends := ends.push total
    return { finder with currentPos := 0, orderedStarts := starts, orderedEnds := ends }

/-! ## Data Shard -/

def defaultShardSize : Nat := 1000000

structure DataShard (n : Nat := defaultShardSize) where
  tokens : Tensor { shape := #[n], dtype := .Int64, device := .CPU }
  bosFinder : BOSFinder
  shardIdx : Nat
  numShards : Nat
  deriving Repr

/-- fineweb/modded-nanogpt binary format: 256 int32 header words. -/
def finewebHeaderI32Words : Nat := 256

/-- Header size in uint16 words (= 1024 bytes = 512 uint16 entries). -/
def finewebHeaderU16Words : Nat := finewebHeaderI32Words * 2

/-- Magic/version used by modded-nanogpt fineweb shards. -/
def finewebMagic : UInt64 := 20240520
def finewebVersion : UInt64 := 1

/-- Decode one little-endian u32 value from two u16 words. -/
private def decodeLEU32FromU16Words (lo hi : UInt64) : UInt64 :=
  (lo &&& (0xFFFF : UInt64)) ||| ((hi &&& (0xFFFF : UInt64)) <<< 16)

structure FinewebHeader where
  magic : UInt64
  version : UInt64
  tokenCount : UInt64
  deriving Repr

/-- Parse the leading modded-nanogpt header values (magic, version, tokenCount). -/
def parseFinewebHeader? {n : Nat} (tokens : Tensor { shape := #[n], dtype := .Int64, device := device })
    : IO (Option FinewebHeader) := do
  if n < 6 then
    return none

  let hdr6 : Tensor { shape := #[6], dtype := .Int64, device := device } := tokens.sliceChecked 0 0 6 #[6]
  let vals ← hdr6.toUInt64Array
  if vals.size < 6 then
    return none

  let magic := decodeLEU32FromU16Words vals[0]! vals[1]!
  let version := decodeLEU32FromU16Words vals[2]! vals[3]!
  let tokenCount := decodeLEU32FromU16Words vals[4]! vals[5]!
  return some { magic, version, tokenCount }

/-- Split a tensor loaded from .bin into payload tokens, parsing fineweb header when present. -/
def splitFinewebPayload {n : Nat} (tokens : Tensor { shape := #[n], dtype := .Int64, device := device })
    : IO (Σ m, Tensor { shape := #[m], dtype := .Int64, device := device }) := do
  if n < finewebHeaderU16Words then
    return ⟨n, tokens⟩

  let some header ← parseFinewebHeader? tokens
    | return ⟨n, tokens⟩

  if header.magic != finewebMagic then
    return ⟨n, tokens⟩

  if header.version != finewebVersion then
    throw <| IO.userError s!"Unsupported fineweb shard version {header.version} in header (expected {finewebVersion})"

  let payloadWords := n - finewebHeaderU16Words
  -- In practice some shard sets keep a nominal/global tokenCount in the header
  -- while each file stores only a local payload. Treat tokenCount as advisory
  -- and clamp to available payload to avoid noisy false-positive warnings.
  let payloadCount :=
    if header.tokenCount == 0 then
      payloadWords
    else
      min header.tokenCount.toNat payloadWords

  let payload := tokens.sliceChecked 0 finewebHeaderU16Words payloadCount #[payloadCount]
  return ⟨payloadCount, payload⟩

def DataShard.loadFromFile (path : String) (shardIdx numShards : Nat)
    (bosToken : UInt64) : IO (Σ n, DataShard n) := do
  if numShards == 0 || shardIdx >= numShards then
    throw <| IO.userError "Invalid data-loader rank/world size"
  let rawTokensCount := (← data.binFileTokenCount path).toNat
  let allTokens ← Tensor.loadU16Bin rawTokensCount path
  let ⟨totalTokens, allTokens⟩ ← splitFinewebPayload allTokens
  let tokensPerShard := totalTokens / numShards
  let startToken := tokensPerShard * shardIdx
  let endToken := if shardIdx == numShards - 1 then totalTokens else startToken + tokensPerShard
  let shardSize := endToken - startToken
  let shardTokens := allTokens.sliceChecked 0 startToken shardSize #[shardSize]
  let bosFinder ← BOSFinder.init shardTokens bosToken
  return ⟨shardSize, { tokens := shardTokens, bosFinder, shardIdx, numShards }⟩

def DataShard.load (path : String) (shardIdx numShards : Nat)
    (bosToken : UInt64) : IO (Σ n, DataShard n) := do
  let fileExists ← data.fileExists path
  if fileExists then
    DataShard.loadFromFile path shardIdx numShards bosToken
  else
    throw <| IO.userError s!"Data shard not found: {path}"

/-! ## Batch Iterator -/

structure BatchIterator where
  numTokens : Nat
  shard : DataShard numTokens
  batchSize : Nat
  seqLen : Nat
  batchCount : Nat
  epoch : Nat
  shuffleSeed : Option UInt64 := none
  deriving Repr

def BatchIterator.new {n : Nat} (shard : DataShard n) (batchSize seqLen : Nat)
    (shuffleSeed : Option UInt64 := none) (epoch : Nat := 0) : BatchIterator :=
  let finder := match shuffleSeed with
    | none => shard.bosFinder.reset
    | some seed => shard.bosFinder.shuffle (lcgNext (seed + epoch.toUInt64))
  { numTokens := n, shard := { shard with bosFinder := finder }, batchSize, seqLen,
    batchCount := 0, epoch, shuffleSeed }

def BatchIterator.next (iter : BatchIterator)
    : IO (Option (Tensor { shape := #[iter.batchSize, iter.seqLen], dtype := .Int64, device := .CPU }) × BatchIterator) := do
  let (maybeBatch, newBosFinder) ← iter.shard.bosFinder.getBatch
    iter.shard.tokens iter.batchSize iter.seqLen
  match maybeBatch with
  | none =>
    let newIter := BatchIterator.new { iter.shard with bosFinder := newBosFinder }
      iter.batchSize iter.seqLen iter.shuffleSeed (iter.epoch + 1)
    return (none, newIter)
  | some batch =>
    let newShard := { iter.shard with bosFinder := newBosFinder }
    let newIter := { iter with shard := newShard, batchCount := iter.batchCount + 1 }
    return (some batch, newIter)

def BatchIterator.updateParams (iter : BatchIterator)
    (batchSize seqLen : Nat) : IO BatchIterator := do
  let _ ← checkedBatchTokens batchSize seqLen
  return { iter with batchSize, seqLen }

/-! ## Distributed Data Generator -/

structure DistributedDataGenerator where
  iterator : BatchIterator
  config : Config
  globalStep : Nat
  rank : Nat
  worldSize : Nat
  trainPaths : Array String
  trainPathIdx : Nat
  deriving Repr

/-- Stream identity and logical offset needed to reproduce document order. -/
structure StreamCursor where
  version : UInt64 := 1
  seed : UInt64
  shuffle : Bool
  bosToken : UInt64
  rank : Nat
  worldSize : Nat
  trainPaths : Array String
  trainPathIdx : Nat
  shardLength : Nat
  position : Nat
  epoch : Nat
  globalStep : Nat
  batchCount : Nat
  deriving Repr, Inhabited, Lean.ToJson, Lean.FromJson

private def streamSeed (config : Config) (rank : Nat) (pathIdx : Nat) : Option UInt64 :=
  if config.shuffle then some (lcgNext (config.seed + rank.toUInt64) + pathIdx.toUInt64) else none

/-- Explicit rank entry point, also usable without initializing a process group. -/
def DistributedDataGenerator.initForRank (config : Config) (batchSize seqLen rank worldSize : Nat)
    : IO DistributedDataGenerator := do
  let _ ← checkedBatchTokens batchSize seqLen
  let trainPaths ← resolveShardPaths config.dataPath .train
  let trainPathIdx := 0
  let ⟨_, shard⟩ ← DataShard.load trainPaths[trainPathIdx]! rank worldSize config.bosToken
  let iterator := BatchIterator.new shard batchSize seqLen (streamSeed config rank trainPathIdx)
  return { iterator, config, globalStep := 0, rank, worldSize, trainPaths, trainPathIdx }

/-- Initialize a synchronous loader using the active distributed rank. -/
def DistributedDataGenerator.init (config : Config) (batchSize seqLen : Nat)
    : IO DistributedDataGenerator := do
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then dist.getRankAndWorldSize else pure (0, 1)
  let (rank, worldSize) := (rank.toNat, worldSize.toNat)
  initForRank config batchSize seqLen rank worldSize

def DistributedDataGenerator.cursor (gen : DistributedDataGenerator) : StreamCursor := {
  seed := gen.config.seed, shuffle := gen.config.shuffle, bosToken := gen.config.bosToken
  rank := gen.rank, worldSize := gen.worldSize, trainPaths := gen.trainPaths
  trainPathIdx := gen.trainPathIdx, shardLength := gen.iterator.numTokens
  position := gen.iterator.shard.bosFinder.currentPos, epoch := gen.iterator.epoch
  globalStep := gen.globalStep, batchCount := gen.iterator.batchCount }

/-- Restore only an identical stream configuration. The ordered document
    mapping is reconstructed from seed, epoch, rank and file index. -/
def DistributedDataGenerator.restoreCursor (gen : DistributedDataGenerator) (cursor : StreamCursor)
    : IO DistributedDataGenerator := do
  if cursor.version != 1 || cursor.seed != gen.config.seed || cursor.shuffle != gen.config.shuffle ||
      cursor.bosToken != gen.config.bosToken || cursor.rank != gen.rank ||
      cursor.worldSize != gen.worldSize || cursor.trainPaths != gen.trainPaths then
    throw <| IO.userError "Data cursor stream mismatch (version, seed, rank, world size, BOS policy or files)"
  if cursor.trainPathIdx >= gen.trainPaths.size then
    throw <| IO.userError "Data cursor shard index out of bounds"
  let ⟨n, shard⟩ ← DataShard.load gen.trainPaths[cursor.trainPathIdx]! gen.rank gen.worldSize gen.config.bosToken
  if n != cursor.shardLength || cursor.position > n then
    throw <| IO.userError "Data cursor shard length or position mismatch"
  let iterator := BatchIterator.new shard gen.iterator.batchSize gen.iterator.seqLen
    (streamSeed gen.config gen.rank cursor.trainPathIdx) cursor.epoch
  let finder := { iterator.shard.bosFinder with currentPos := cursor.position }
  let iterator := { iterator with
    shard := { iterator.shard with bosFinder := finder }
    batchCount := cursor.batchCount }
  return { gen with iterator, trainPathIdx := cursor.trainPathIdx, globalStep := cursor.globalStep }

/-- Read a continuous training stream, preserving incomplete document/file
    tails across batches. Crossing the final file starts a new seeded epoch;
    it does not pad individual files to an artificial working-shard size.
    The result holds exactly `count` Int64 tokens. -/
def DistributedDataGenerator.nextTokens (gen : DistributedDataGenerator) (count : Nat)
    : IO (Tensor { shape := #[count], dtype := .Int64, device := .CPU } × DistributedDataGenerator) := do
  if count == 0 then throw <| IO.userError "Batch token count must be positive"
  let mut gen := gen
  let mut remaining := count
  let mut chunks : Array (Sigma fun shape => Tensor { shape := shape, dtype := .Int64, device := .CPU }) := #[]
  let mut emptyFiles := 0
  while remaining > 0 do
    let iter := gen.iterator
    let available := iter.shard.bosFinder.dataLen - iter.shard.bosFinder.currentPos
    let len := min remaining available
    if len > 0 then
      let (chunk, finder) ← iter.shard.bosFinder.take iter.shard.tokens len
      chunks := chunks.push (Tensor.packShape chunk)
      gen := { gen with iterator := { iter with shard := { iter.shard with bosFinder := finder } } }
      remaining := remaining - len
      emptyFiles := 0
    if remaining == 0 then break
    if available == 0 then emptyFiles := emptyFiles + 1
    if emptyFiles > gen.trainPaths.size then
      throw <| IO.userError "Training corpus has no tokens for this rank"
    let nextIdx := (gen.trainPathIdx + 1) % gen.trainPaths.size
    let nextPath := gen.trainPaths[nextIdx]!
    let epoch := gen.iterator.epoch + (if nextIdx == 0 then 1 else 0)
    let ⟨_, shard⟩ ← DataShard.load nextPath gen.rank gen.worldSize gen.config.bosToken
    let iterator := BatchIterator.new shard iter.batchSize iter.seqLen
      (streamSeed gen.config gen.rank nextIdx) epoch
    gen := { gen with iterator, trainPathIdx := nextIdx }
  gen := { gen with
    globalStep := gen.globalStep + 1
    iterator := { gen.iterator with batchCount := gen.iterator.batchCount + 1 } }
  -- The chunks consumed exactly `count` tokens in total.
  let tokens ← match Tensor.catMany? chunks 0 #[count] with
    | .ok tensor => pure tensor
    | .error message => throw <| IO.userError s!"Training stream batch: {message}"
  return (tokens, gen)

def DistributedDataGenerator.nextBatch (gen : DistributedDataGenerator)
    : IO (Option (Tensor { shape := #[gen.iterator.batchSize, gen.iterator.seqLen], dtype := .Int64, device := .CPU }) ×
          DistributedDataGenerator) := do
  let ⟨count, hCount⟩ ← checkedBatchTokens gen.iterator.batchSize gen.iterator.seqLen
  let (tokens, gen') ← gen.nextTokens count
  return (some (tokens.reshape #[gen.iterator.batchSize, gen.iterator.seqLen] (by simp [hCount])), gen')

def DistributedDataGenerator.batchSize (gen : DistributedDataGenerator) : Nat :=
  gen.iterator.batchSize

def DistributedDataGenerator.seqLen (gen : DistributedDataGenerator) : Nat :=
  gen.iterator.seqLen

/-! ## Validation and Utilities -/

def loadValidationData (path : String) (_seqLen : Nat) (bosToken : UInt64)
    : IO (Σ n, DataShard n) := do
  let valPaths ← resolveShardPaths path .val
  DataShard.load valPaths[0]! 0 1 bosToken

def estimateRemainingTime (currentStep totalSteps : Nat)
    (msPerStep : Float) : Float :=
  (totalSteps - currentStep).toFloat * msPerStep / 1000.0 / 60.0

end torch.DataLoader
