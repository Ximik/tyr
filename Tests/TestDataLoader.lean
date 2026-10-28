/- Generated-fixture regression tests for token coverage, document order and resume. -/
import Tyr.DataLoader
import Examples.GPT.GPTDataLoader
import LeanTest

open torch
open torch.DataLoader

private def writeTokens (path : System.FilePath) (tokens : Array UInt64) : IO Unit := do
  let mut bytes := ByteArray.empty
  for token in tokens do
    bytes := bytes.push token.toUInt8
    bytes := bytes.push (token >>> 8).toUInt8
  IO.FS.writeBinFile path bytes

private def readTokens {s : Shape} (tokens : Tensor { shape := s, dtype := .Int64, device := .CPU }) : IO (Array UInt64) :=
  tokens.toUInt64Array

private def withSmallFixture (action : String → IO Unit) : IO Unit :=
  IO.FS.withTempDir fun dir => do
    let path := dir / "tokens.bin"
    writeTokens path ((List.range 8192).toArray.map fun n => (n % 127 + 1).toUInt64)
    action path.toString

private def assertRejected (action : IO Unit) (label : String) : IO Unit := do
  let rejected ← try action; pure false catch _ => pure true
  LeanTest.assertTrue rejected label

@[test] def testSequentialLoader : IO Unit := withSmallFixture fun path => do
  let ⟨n, _⟩ ← SequentialLoader.fromFile path
  LeanTest.assertEqual n 8192

@[test] def testRandomBatchSampling : IO Unit := withSmallFixture fun path => do
  let ⟨_, loader⟩ ← SequentialLoader.fromFile path
  let (input, target) ← loader.sampleRandomBatch 4 32
  LeanTest.assertTrue (Tensor.itemInt (Tensor.sumAll input) > 0) "Nonempty input"
  LeanTest.assertEqual (← readTokens (input.slice 1 1 31))
    (← readTokens (target.slice 1 0 31)) "Targets are shifted inputs"

@[test] def testSequentialBatchIterator : IO Unit := withSmallFixture fun path => do
  let ⟨_, loader⟩ ← SequentialLoader.fromFile path
  let mut iter := SequentialBatchIterator.new loader 4 32
  for _ in [:5] do
    let (batch, next) := iter.next
    LeanTest.assertTrue batch.isSome "Generated fixture has five batches"
    iter := next

@[test] def testEpochReset : IO Unit := withSmallFixture fun path => do
  let ⟨_, loader⟩ ← SequentialLoader.fromFile path
  let mut iter := SequentialBatchIterator.new loader 4 32
  let mut ended := false
  for _ in [:100] do
    let (batch, next) := iter.next
    iter := next
    if batch.isNone then
      ended := true
      break
  LeanTest.assertTrue ended "Finite generated fixture reaches epoch end"
  LeanTest.assertEqual iter.epoch 1

private def documentTokens : Array UInt64 :=
  #[91, 92, 0, 101, 102, 0, 201, 202, 203, 0, 301, 302, 0, 401, 402, 0, 501, 502]

@[test] def testBosFinderInit : IO Unit := do
  let tokens : Tensor { shape := #[documentTokens.size], dtype := .Int64, device := .CPU } := (Tensor.fromInt64Array (documentTokens.map UInt64.toInt64)).reshape #[documentTokens.size]
  let finder ← BOSFinder.init tokens 0
  LeanTest.assertEqual finder.bosPositions #[0, 2, 5, 9, 12, 15]
    "BOS boundaries include the initial partition fragment"

@[test] def testDocumentAwareLoader : IO Unit := do
  let tokens : Tensor { shape := #[documentTokens.size], dtype := .Int64, device := .CPU } := (Tensor.fromInt64Array (documentTokens.map UInt64.toInt64)).reshape #[documentTokens.size]
  let finder := (← BOSFinder.init tokens 0).shuffle 42
  let (batch, next) ← finder.getBatch tokens 1 documentTokens.size
  let some batch := batch | throw <| IO.userError "Missing document batch"
  let values ← readTokens batch
  LeanTest.assertEqual (values.qsort (· < ·)) (documentTokens.qsort (· < ·))
    "Every real token appears exactly once"
  for pair in #[(91, 92), (101, 102), (201, 202), (202, 203), (301, 302), (401, 402), (501, 502)] do
    let i := values.toList.idxOf pair.1
    LeanTest.assertEqual (values[i + 1]?) (some pair.2) "Document-internal order is preserved"
  LeanTest.assertEqual next.currentPos documentTokens.size

@[test] def testShuffleDeterminism : IO Unit := do
  let tokens : Tensor { shape := #[documentTokens.size], dtype := .Int64, device := .CPU } := (Tensor.fromInt64Array (documentTokens.map UInt64.toInt64)).reshape #[documentTokens.size]
  let finder ← BOSFinder.init tokens 0
  let (a, _) ← (finder.shuffle 42).take tokens finder.dataLen
  let (b, _) ← (finder.shuffle 42).take tokens finder.dataLen
  let (c, _) ← (finder.shuffle 43).take tokens finder.dataLen
  LeanTest.assertEqual (← readTokens a) (← readTokens b) "Same seed reproduces actual batches"
  LeanTest.assertTrue ((← readTokens a) != (← readTokens c)) "Different seeds change actual token order"

@[test] def testResolveShardPathsDirectoryAndPrefix : IO Unit := IO.FS.withTempDir fun dir => do
  writeTokens (dir / "fineweb_train_0.bin") #[1, 2]
  writeTokens (dir / "fineweb_train_1.bin") #[3, 4]
  writeTokens (dir / "fineweb_val_0.bin") #[5, 6]
  let train ← resolveShardPaths dir.toString .train
  let val ← resolveShardPaths dir.toString .val
  LeanTest.assertEqual train.size 2
  LeanTest.assertEqual val.size 1
  LeanTest.assertEqual (← resolveShardPaths (dir / "fineweb_train").toString .train) train

@[test] def testDistributedGeneratorRotatesAcrossTrainShards : IO Unit := IO.FS.withTempDir fun dir => do
  writeTokens (dir / "fineweb_train_0.bin") #[1, 2, 3]
  writeTokens (dir / "fineweb_train_1.bin") #[4, 5, 6, 7, 8, 9, 10]
  let cfg : Config := { dataPath := dir.toString, bosToken := 65535, shuffle := false }
  let mut gen ← DistributedDataGenerator.initForRank cfg 1 4 0 1
  LeanTest.assertEqual gen.iterator.numTokens 3 "Small files keep their real length"
  for expected in #[#[1, 2, 3, 4], #[5, 6, 7, 8], #[9, 10, 1, 2]] do
    let (batch, next) ← gen.nextBatch
    let some batch := batch | throw <| IO.userError "Missing stream batch"
    LeanTest.assertEqual (← readTokens batch) expected "File tails are packed before epoch rollover"
    gen := next
  LeanTest.assertEqual gen.iterator.epoch 1 "Only a complete file cycle advances the epoch"
  LeanTest.assertEqual gen.globalStep 3

@[test] def testDataLoaderRetainsLargeRankPartitionTails : IO Unit := IO.FS.withTempDir fun dir => do
  let path := dir / "fineweb_train_0.bin"
  let partition : Nat := 1000006
  let mut bytes := ByteArray.empty
  for i in [:partition * 2] do
    let token : UInt8 := if i % partition < 1000000 then 7 else if i < partition then 101 else 201
    bytes := (bytes.push token).push 0
  IO.FS.writeBinFile path bytes
  let cfg : Config := { dataPath := path.toString, bosToken := 65535, shuffle := false }
  for rank in #[0, 1] do
    let gen ← DistributedDataGenerator.initForRank cfg 1 1 rank 2
    LeanTest.assertEqual gen.iterator.numTokens partition "Full rank partition survives loading"
    let (_, gen) ← gen.nextTokens 1000000
    let (tail, gen) ← gen.nextTokens 6
    LeanTest.assertEqual (← readTokens tail) (Array.replicate 6 (if rank == 0 then 101 else 201))
      "Tokens beyond the former one-million limit are consumed"
    LeanTest.assertEqual gen.iterator.shard.bosFinder.currentPos partition

@[test] def testDataLoaderRankPartitionsCoverUnequalFiles : IO Unit := IO.FS.withTempDir fun dir => do
  let path := dir / "tokens.bin"
  writeTokens path #[1, 2, 3, 4, 5, 6, 7]
  let mut joined : Array UInt64 := #[]
  for rank in #[0, 1, 2] do
    let ⟨_, shard⟩ ← DataShard.load path.toString rank 3 65535
    joined := joined ++ (← readTokens shard.tokens)
  LeanTest.assertEqual joined #[1, 2, 3, 4, 5, 6, 7] "Rank partitions cover every token without overlap"
  assertRejected (do let _ ← DataShard.load path.toString 0 0 0; pure ()) "Zero world size is rejected"
  assertRejected (do let _ ← DataShard.load path.toString 3 3 0; pure ()) "Invalid rank is rejected"

@[test] def testDataLoaderSeededGPTResume : IO Unit := IO.FS.withTempDir fun dir => do
  writeTokens (dir / "fineweb_train_0.bin") documentTokens
  writeTokens (dir / "fineweb_train_1.bin") #[0, 601, 602, 0, 701, 702, 703]
  let cfg : Config := { dataPath := dir.toString, bosToken := 0, seed := 42 }
  let initial ← DistributedDataGenerator.initForRank cfg 1 4 0 1
  let (_, advanced) ← initial.nextTokens 7
  let saved := advanced.cursor
  let json := Lean.toJson saved
  let decoded ← match Lean.fromJson? (α := StreamCursor) json with
    | .ok cursor => pure cursor
    | .error msg => throw <| IO.userError msg
  let mut continued := advanced
  let mut restored ← initial.restoreCursor decoded
  for _ in [:8] do
    let (a, nextA) ← continued.nextBatchGPT
    let (b, nextB) ← restored.nextBatchGPT
    let some (ai, targetA) := a | throw <| IO.userError "Missing continued GPT batch"
    let some (bi, bt) := b | throw <| IO.userError "Missing restored GPT batch"
    LeanTest.assertEqual (← readTokens ai) (← readTokens bi) "Resume reproduces GPT inputs through file/epoch changes"
    LeanTest.assertEqual (← readTokens targetA) (← readTokens bt) "Resume reproduces shifted GPT targets"
    continued := nextA
    restored := nextB
  LeanTest.assertEqual continued.cursor.position restored.cursor.position
  LeanTest.assertEqual continued.cursor.epoch restored.cursor.epoch
  LeanTest.assertTrue (continued.iterator.epoch > 0) "Resume test traverses a shuffled epoch boundary"
  assertRejected (do let _ ← initial.restoreCursor { saved with seed := 43 }; pure ()) "Seed mismatch is rejected"
  assertRejected (do let _ ← initial.restoreCursor { saved with worldSize := 2 }; pure ()) "World-size mismatch is rejected"
  assertRejected (do let _ ← initial.restoreCursor { saved with position := 999 }; pure ()) "Invalid position is rejected"
  assertRejected (do let _ ← initial.restoreCursor { saved with trainPathIdx := 99 }; pure ()) "Invalid file index is rejected"

@[test] def testDataLoaderEmptyRankFailsWithoutLooping : IO Unit := IO.FS.withTempDir fun dir => do
  let path := dir / "tokens.bin"
  writeTokens path #[1]
  let cfg : Config := { dataPath := path.toString }
  let gen ← DistributedDataGenerator.initForRank cfg 1 1 0 2
  assertRejected (do let _ ← gen.nextBatch; pure ()) "An empty rank reports no tokens instead of looping"

@[test] def testDataLoaderRejectsInvalidBatchSizes : IO Unit := IO.FS.withTempDir fun dir => do
  let path := dir / "tokens.bin"
  writeTokens path #[1, 2, 3, 4]
  let cfg : Config := { dataPath := path.toString, shuffle := false }
  let gen ← DistributedDataGenerator.initForRank cfg 1 1 0 1
  let huge : Nat := 18446744073709551615
  for dims in #[(0, 1), (1, 0), (huge, 2)] do
    assertRejected (do let _ ← gen.iterator.updateParams dims.1 dims.2; pure ())
      "Parameter updates reject zero sizes and overflowing products"
    let invalid := { gen with iterator := { gen.iterator with batchSize := dims.1, seqLen := dims.2 } }
    assertRejected (do let _ ← invalid.nextBatch; pure ()) "Direct record updates cannot bypass size validation"
    assertRejected (do let _ ← invalid.nextBatchGPT; pure ()) "GPT batches validate dimensions before allocating"
  let extraOverflow := { gen with iterator := { gen.iterator with seqLen := huge } }
  assertRejected (do let _ ← extraOverflow.nextBatchGPT; pure ()) "GPT extra target token cannot wrap sequence length"
  assertRejected (do let _ ← extraOverflow.iterator.nextGPT; pure ()) "Finite GPT iterators reject target-token overflow"

@[test] def testRandomBatchExactFitAndOversized : IO Unit := IO.FS.withTempDir fun dir => do
  let path := dir / "tokens.bin"
  writeTokens path #[1, 2, 3, 4, 5, 6]
  let ⟨_, loader⟩ ← SequentialLoader.fromFile path.toString
  let (input, target) ← loader.sampleRandomBatch 2 2
  LeanTest.assertEqual (← readTokens input) #[1, 2, 4, 5]
  LeanTest.assertEqual (← readTokens target) #[2, 3, 5, 6]
  assertRejected (do let _ ← loader.sampleRandomBatch 3 2; pure ())
    "Random batches validate the whole batch before choosing an offset"
