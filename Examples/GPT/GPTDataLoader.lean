/-
  Examples/GPT/GPTDataLoader.lean

  GPT-specific data loading extensions for Tyr.
  
  Includes:
  - Language modeling batching (input/target shifting)
  - Modded-nanogpt training schedules
  - Simple sequential loading for character-level data (Shakespeare)
-/
import Tyr.DataLoader
import Examples.GPT.GPT

namespace torch.DataLoader

open torch

/-! ## GPT Batching -/

/-- Get the next batch of (input, target) pairs with indexed shapes and integral dtype.
    For causal language modeling, target is input shifted by 1.
-/
def BatchIterator.nextGPT (iter : BatchIterator)
    : IO (Option (Tensor { shape := #[iter.batchSize, iter.seqLen], dtype := .Int64 } × Tensor { shape := #[iter.batchSize, iter.seqLen], dtype := .Int64 }) × BatchIterator) := do
  let _ ← checkedBatchTokens iter.batchSize iter.seqLen 1
  -- Get batch from BOS finder
  -- We need seqLen + 1 to split into input and target
  let (maybeBatch, newBosFinder) ← iter.shard.bosFinder.getBatch
    iter.shard.tokens iter.batchSize (iter.seqLen + 1)

  match maybeBatch with
  | none =>
    -- Epoch complete, reset and increment epoch
    let newIter := BatchIterator.new { iter.shard with bosFinder := newBosFinder }
      iter.batchSize iter.seqLen iter.shuffleSeed (iter.epoch + 1)
    return (none, newIter)
  | some batch =>
    -- Split into input and target (target is shifted by 1)
    -- batch is [batchSize, seqLen+1]
    let input := batch.sliceChecked 1 0 iter.seqLen #[iter.batchSize, iter.seqLen]
    let target := batch.sliceChecked 1 1 iter.seqLen #[iter.batchSize, iter.seqLen]
    let newShard := { iter.shard with bosFinder := newBosFinder }
    let newIter := { iter with
      shard := newShard
      batchCount := iter.batchCount + 1
    }
    return (some (input, target), newIter)

/-- Get the next batch for GPT training. -/
def DistributedDataGenerator.nextBatchGPT (gen : DistributedDataGenerator)
    : IO (Option (Tensor { shape := #[gen.iterator.batchSize, gen.iterator.seqLen], dtype := .Int64 } × Tensor { shape := #[gen.iterator.batchSize, gen.iterator.seqLen], dtype := .Int64 }) × DistributedDataGenerator) := do
  let b := gen.iterator.batchSize
  let s := gen.iterator.seqLen
  let count ← checkedBatchTokens b s 1
  let (tokens, gen') ← gen.nextTokens count
  let batch := tokens.reshapeChecked #[b, s + 1]
  let input := batch.sliceChecked 1 0 s #[b, s]
  let target := batch.sliceChecked 1 1 s #[b, s]
  return (some (input, target), gen')

/-- Update batch and sequence parameters based on training step for modded-nanogpt. -/
def DistributedDataGenerator.updateForStepGPT (gen : DistributedDataGenerator)
    (step : UInt64) (blockSize : UInt64 := 128) : IO DistributedDataGenerator := do
  let (batchSize, windowBlocks) :=
    if step < 200 then (8, 3)
    else if step < 1000 then (16, 7)
    else (24, 11)
  let seqLen ← checkedBatchTokens windowBlocks blockSize
  let newIterator ← gen.iterator.updateParams batchSize seqLen
  return { gen with iterator := newIterator }

/-! ## Utility Functions -/

/-- Get batch and window sizes for a given step based on modded-nanogpt. -/
def getHyperparamsForStep (step : UInt64) (_blockSize : UInt64 := 128)
    : (UInt64 × UInt64 × UInt64) :=  -- (batchSize, wsShort, wsLong)
  if step < 200 then
    (8, 3, 3)  -- All short windows initially
  else if step < 1000 then
    (16, 3, 7)  -- Mixed windows
  else
    (24, 3, 11)  -- Full windows

/-- Compute effective tokens per step -/
def tokensPerStep (batchSize seqLen worldSize : UInt64) : UInt64 :=
  batchSize * seqLen * worldSize

/-! ## Sequential DataLoader (Shakespeare style) -/

/-- Sequential data loader for simple datasets like Shakespeare. -/
structure SequentialLoader (n : UInt64) where
  tokens : Tensor { shape := #[n], dtype := .Int64 }
  currentPos : UInt64
  deriving Repr

def SequentialLoader.fromFile (path : String) : IO (Σ n, SequentialLoader n) := do
  let n ← data.binFileTokenCount path
  let tokens ← Tensor.loadU16Bin n path
  return ⟨n, { tokens := tokens, currentPos := 0 }⟩

def SequentialLoader.reset {n : UInt64} (loader : SequentialLoader n) : SequentialLoader n :=
  { loader with currentPos := 0 }

def SequentialLoader.getBatch {n : UInt64} (loader : SequentialLoader n)
    (batchSize seqLen : UInt64) : Option (Tensor { shape := #[batchSize, seqLen + 1], dtype := .Int64 } × SequentialLoader n) :=
  let required := batchSize.toNat * (seqLen.toNat + 1)
  if batchSize == 0 || seqLen == 0 || loader.currentPos.toNat + required > n.toNat then
    none
  else
    let requiredLen := required.toUInt64
    let batch := (loader.tokens.sliceChecked 0 loader.currentPos requiredLen #[requiredLen]).reshapeChecked
      #[batchSize, seqLen + 1]
    let newLoader := { loader with currentPos := loader.currentPos + requiredLen }
    some (batch, newLoader)

def SequentialLoader.sampleRandomBatch {n : UInt64} (loader : SequentialLoader n)
    (batchSize blockSize : UInt64) : IO (Tensor { shape := #[batchSize, blockSize], dtype := .Int64 } × Tensor { shape := #[batchSize, blockSize], dtype := .Int64 }) := do
  let requiredLen ← checkedBatchTokens batchSize blockSize 1
  if requiredLen > n then
    throw <| IO.userError "SequentialLoader: batch exceeds available tokens"
  let choices := n.toNat - requiredLen.toNat + 1
  if choices > 9223372036854775807 then
    throw <| IO.userError "SequentialLoader: random range exceeds Int64 capacity"
  let startIdx ← Tensor.randint 0 (Int64.ofNat choices) #[]
  let startPos := startIdx.itemInt.toInt.toNat.toUInt64
  let sliced := loader.tokens.sliceChecked 0 startPos requiredLen #[requiredLen]
  let batch := sliced.reshapeChecked #[batchSize, blockSize + 1]
  let input := batch.sliceChecked 1 0 blockSize #[batchSize, blockSize]
  let target := batch.sliceChecked 1 1 blockSize #[batchSize, blockSize]
  return (input, target)

structure SequentialBatchIterator (n b s : UInt64) where
  loader : SequentialLoader n
  epoch : UInt64
  batchCount : UInt64
  deriving Repr

def SequentialBatchIterator.new {n : UInt64} (loader : SequentialLoader n)
    (batchSize seqLen : UInt64) : SequentialBatchIterator n batchSize seqLen := {
  loader := loader
  epoch := 0
  batchCount := 0
}

def SequentialBatchIterator.next {n b s : UInt64} (iter : SequentialBatchIterator n b s)
    : Option (Tensor { shape := #[b, s], dtype := .Int64 } × Tensor { shape := #[b, s], dtype := .Int64 }) × SequentialBatchIterator n b s :=
  match iter.loader.getBatch b s with
  | none =>
    let newLoader := iter.loader.reset
    let newIter := { iter with
      loader := newLoader
      epoch := iter.epoch + 1
      batchCount := 0
    }
    (none, newIter)
  | some (batch, newLoader) =>
    let input := batch.sliceChecked 1 0 s #[b, s]
    let target := batch.sliceChecked 1 1 s #[b, s]
    let newIter := { iter with
      loader := newLoader
      batchCount := iter.batchCount + 1
    }
    (some (input, target), newIter)

def loadShakespeareData (trainPath valPath : String) (batchSize seqLen : UInt64)
    : IO (Σ n, SequentialBatchIterator n batchSize seqLen × Option (Σ m, SequentialLoader m)) := do
  let ⟨nTrain, trainLoader⟩ ← SequentialLoader.fromFile trainPath
  let trainIter := SequentialBatchIterator.new trainLoader batchSize seqLen
  let valExists ← data.fileExists valPath
  let valLoader ← if valExists then do
    let ⟨nVal, loader⟩ ← SequentialLoader.fromFile valPath
    pure (some ⟨nVal, loader⟩)
  else
    pure none
  return ⟨nTrain, trainIter, valLoader⟩

end torch.DataLoader
