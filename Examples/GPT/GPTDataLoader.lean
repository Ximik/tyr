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
def BatchIterator.nextGPT (iter : BatchIterator) (batchSize seqLen : Nat)
    : IO (Option (Tensor { shape := #[batchSize, seqLen], dtype := .Int64, device := .CPU } × Tensor { shape := #[batchSize, seqLen], dtype := .Int64, device := .CPU }) × BatchIterator) := do
  let _ ← batchTokenCount batchSize seqLen 1
  -- We need seqLen + 1 to split into input and target
  let (maybeBatch, newBosFinder) ← iter.shard.bosFinder.getBatch
    iter.shard.tokens batchSize (seqLen + 1)

  match maybeBatch with
  | none =>
    -- Epoch complete, reset and increment epoch
    let newIter := BatchIterator.new { iter.shard with bosFinder := newBosFinder }
      iter.shuffleSeed (iter.epoch + 1)
    return (none, newIter)
  | some batch =>
    let input := batch.slice 1 0 seqLen
    let target := batch.slice 1 1 seqLen
    let newShard := { iter.shard with bosFinder := newBosFinder }
    let newIter := { iter with
      shard := newShard
      batchCount := iter.batchCount + 1
    }
    return (some (input, target), newIter)

/-- Get the next batch for GPT training. -/
def DistributedDataGenerator.nextBatchGPT (gen : DistributedDataGenerator) (b s : Nat)
    : IO (Option (Tensor { shape := #[b, s], dtype := .Int64, device := .CPU } × Tensor { shape := #[b, s], dtype := .Int64, device := .CPU }) × DistributedDataGenerator) := do
  let ⟨count, hCount⟩ ← batchTokenCount b s 1
  let (tokens, gen') ← gen.nextTokens count
  let batch := tokens.reshape #[b, s + 1] (by simp [hCount])
  let input := batch.slice 1 0 s
  let target := batch.slice 1 1 s
  return (some (input, target), gen')

/-! ## Utility Functions -/

/-- Get batch and window sizes for a given step based on modded-nanogpt. -/
def getHyperparamsForStep (step : Nat) (_blockSize : Nat := 128)
    : (Nat × Nat × Nat) :=  -- (batchSize, wsShort, wsLong)
  if step < 200 then
    (8, 3, 3)  -- All short windows initially
  else if step < 1000 then
    (16, 3, 7)  -- Mixed windows
  else
    (24, 3, 11)  -- Full windows

/-- Compute effective tokens per step -/
def tokensPerStep (batchSize seqLen worldSize : Nat) : Nat :=
  batchSize * seqLen * worldSize

/-! ## Sequential DataLoader (Shakespeare style) -/

/-- Sequential data loader for simple datasets like Shakespeare. -/
structure SequentialLoader (n : Nat) where
  tokens : Tensor { shape := #[n], dtype := .Int64, device := .CPU }
  currentPos : Nat
  deriving Repr

def SequentialLoader.fromFile (path : String) : IO (Σ n, SequentialLoader n) := do
  let n ← data.binFileTokenCount path
  let tokens ← Tensor.loadU16Bin n path
  return ⟨n, { tokens := tokens, currentPos := 0 }⟩

def SequentialLoader.reset {n : Nat} (loader : SequentialLoader n) : SequentialLoader n :=
  { loader with currentPos := 0 }

def SequentialLoader.getBatch {n : Nat} (loader : SequentialLoader n)
    (batchSize seqLen : Nat) : Option (Tensor { shape := #[batchSize, seqLen + 1], dtype := .Int64, device := .CPU } × SequentialLoader n) :=
  let required := batchSize * (seqLen + 1)
  if h : batchSize == 0 || seqLen == 0 || loader.currentPos + required > n then
    none
  else
    let requiredLen := required
    let batch := (loader.tokens.slice 0 loader.currentPos requiredLen (_bounds := by simp at h; simp [requiredLen]; omega)).reshape
      #[batchSize, seqLen + 1]
    let newLoader := { loader with currentPos := loader.currentPos + requiredLen }
    some (batch, newLoader)

def SequentialLoader.sampleRandomBatch {n : Nat} (loader : SequentialLoader n)
    (batchSize blockSize : Nat) : IO (Tensor { shape := #[batchSize, blockSize], dtype := .Int64, device := .CPU } × Tensor { shape := #[batchSize, blockSize], dtype := .Int64, device := .CPU }) := do
  let ⟨requiredLen, hRequired⟩ ← batchTokenCount batchSize blockSize 1
  if requiredLen > n then
    throw <| IO.userError "SequentialLoader: batch exceeds available tokens"
  let choices := n - requiredLen + 1
  if choices > 9223372036854775807 then
    throw <| IO.userError "SequentialLoader: random range exceeds Int64 capacity"
  let startIdx ← Tensor.randint 0 (Int64.ofNat choices) #[]
  let startPos := startIdx.itemInt.toInt.toNat
  let ⟨hStart⟩ ← if h : startPos + requiredLen ≤ n then pure (PLift.up h) else
    throw <| IO.userError "SequentialLoader: random start exceeds available tokens"
  let sliced := loader.tokens.slice 0 startPos requiredLen
  let batch := sliced.reshape #[batchSize, blockSize + 1] (by simp [hRequired])
  let input := batch.slice 1 0 blockSize
  let target := batch.slice 1 1 blockSize
  return (input, target)

structure SequentialBatchIterator (n b s : Nat) where
  loader : SequentialLoader n
  epoch : Nat
  batchCount : Nat
  deriving Repr

def SequentialBatchIterator.new {n : Nat} (loader : SequentialLoader n)
    (batchSize seqLen : Nat) : SequentialBatchIterator n batchSize seqLen := {
  loader := loader
  epoch := 0
  batchCount := 0
}

def SequentialBatchIterator.next {n b s : Nat} (iter : SequentialBatchIterator n b s)
    : Option (Tensor { shape := #[b, s], dtype := .Int64, device := .CPU } × Tensor { shape := #[b, s], dtype := .Int64, device := .CPU }) × SequentialBatchIterator n b s :=
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
    let input := batch.slice 1 0 s
    let target := batch.slice 1 1 s
    let newIter := { iter with
      loader := newLoader
      batchCount := iter.batchCount + 1
    }
    (some (input, target), newIter)

def loadShakespeareData (trainPath valPath : String) (batchSize seqLen : Nat)
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
