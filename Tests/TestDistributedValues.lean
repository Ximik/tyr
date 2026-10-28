import LeanTest
import Tests.Assertions
import Tyr.Module.Derive
import Tyr.Distributed

namespace Tests.DistributedValues

open torch

-- Pending collectives cannot be used as completed tensors or tensor trees.
example {σ : StaticSpec} (_input : Tensor σ) : True := by
  fail_if_success
    have _ : IO (Tensor σ) := dist.allReduceAsync _input
  fail_if_success
    have _ : IO (Tensor σ × Tensor σ) := dist.allReduceGradsAsync (_input, _input)
  trivial

@[test]
def testUninitializedCollectivesReturnErrors : IO Unit := do
  if ← dist.isInitialized then
    return
  let input := Tensor.fromFloatArray #[2.0, 3.0]
  let rejected ← try
    let _ ← dist.allReduce input
    pure false
  catch error =>
    LeanTest.assertTrue (error.toString.containsSubstr "not initialized")
    pure true
  LeanTest.assertTrue rejected "An uninitialized collective must return an IO error"
  let rejectedAsync ← try
    let _ ← dist.allReduceAsync input
    pure false
  catch error =>
    LeanTest.assertTrue (error.toString.containsSubstr "not initialized")
    pure true
  LeanTest.assertTrue rejectedAsync "An async launch failure must return an IO error"
  LeanTest.assertEqual (← input.toFloatArray) #[2.0, 3.0]

@[test]
def testInvalidCollectiveOutputExtentReturnsError : IO Unit := do
  let input := Tensor.fromFloatArray #[1.0]
  Tests.assertThrows (dist.allGather 9223372036854775808 input)
    (some "Tensor extent")
  LeanTest.assertEqual (← input.toFloatArray) #[1.0]

@[test]
def testInvalidDistributedDeviceIndicesReturnErrors : IO Unit := do
  for index in #[128, 256, 18446744073709551615] do
    Tests.assertThrows (dist.setCudaDevice index) (some "device")

@[test]
def testInvalidDistributedInitMetadataReturnsErrors : IO Unit := do
  LeanTest.assertFalse (← dist.isInitialized)
  -- Every case is rejected before opening a rendezvous socket, including when
  -- the selected backend is unavailable in the native build.
  let cases : Array (String × Nat × Nat × Nat × String) := #[
    ("gloo", 65536, 0, 1, "master_port"),
    ("gloo", 0, 0, 0, "world_size"),
    ("gloo", 0, 0, 2147483648, "world_size"),
    ("gloo", 0, 1, 1, "rank"),
    ("gloo", 0, 18446744073709551615, 1, "rank"),
    ("unsupported", 0, 0, 1, "backend")]
  for (backend, port, rank, worldSize, detail) in cases do
    Tests.assertThrows (dist.initProcessGroup backend "127.0.0.1" port rank worldSize)
      (some detail)
    LeanTest.assertFalse (← dist.isInitialized)
    dist.destroyProcessGroup
  LeanTest.assertFalse (← dist.isInitialized)

/-- Run on each of exactly two ranks in an initialized CPU process group.
    Retained inputs and earlier results detect writes into shared storage. -/
def checkTwoRankValues (rank : Nat) : IO Unit := do
  let rankValue := rank.toFloat + 1.0
  let input ← (Tensor.fromFloatArray #[rankValue]).setRequiresGrad true
  let reduced ← dist.allReduce input .sum
  LeanTest.assertEqual (← input.toFloatArray) #[rankValue]
  LeanTest.assertEqual (← reduced.toFloatArray) #[3.0]
  LeanTest.assertTrue (input.requiresGrad && reduced.requiresGrad && reduced.isLeaf)
    "Completed reductions preserve trainability without changing the input"
  let broadcast ← dist.broadcast input 0
  LeanTest.assertEqual (← input.toFloatArray) #[rankValue]
  LeanTest.assertEqual (← broadcast.toFloatArray) #[1.0]
  LeanTest.assertTrue (broadcast.requiresGrad && broadcast.isLeaf)
    "Broadcast results retain the input's trainability"

  let template ← (Tensor.zeros #[1]).setRequiresGrad true
  let gathered ← dist.allGatherList #[template, template] input
  LeanTest.assertEqual (← template.toFloatArray) #[0.0]
  LeanTest.assertEqual (← gathered[0]!.toFloatArray) #[1.0]
  LeanTest.assertEqual (← gathered[1]!.toFloatArray) #[2.0]
  LeanTest.assertTrue (gathered.all Tensor.requiresGrad)
  let next ← dist.allGatherList #[template, template] (Tensor.fromFloatArray #[rankValue + 10.0])
  LeanTest.assertEqual (← next[0]!.toFloatArray) #[11.0]
  LeanTest.assertEqual (← next[1]!.toFloatArray) #[12.0]
  LeanTest.assertEqual (← gathered[0]!.toFloatArray) #[1.0]
  LeanTest.assertEqual (← gathered[1]!.toFloatArray) #[2.0]

  let scattered ← dist.reduceScatterList template
    #[input, Tensor.fromFloatArray #[rankValue + 10.0]] .sum
  LeanTest.assertEqual (← template.toFloatArray) #[0.0]
  LeanTest.assertEqual (← scattered.toFloatArray) #[if rank == 0 then 3.0 else 23.0]
  LeanTest.assertTrue scattered.requiresGrad

  -- A dynamic static dtype must not turn native integer collectives into FP32.
  let integerInput := (Tensor.fromInt64Array #[(rank + 1).toInt64])
  let integerGathered ← dist.allGather 2 integerInput
  LeanTest.assertEqual integerGathered.raw.dtype .Int64
  LeanTest.assertEqual (← integerGathered.toFloatArray) #[1.0, 2.0]
  let integerChunks := (Tensor.fromInt64Array
    #[(rank + 1).toInt64, (rank + 11).toInt64])
  let integerScattered ← dist.reduceScatter 2 integerChunks .sum
  LeanTest.assertEqual integerScattered.raw.dtype .Int64
  LeanTest.assertEqual (← integerScattered.toFloatArray) #[if rank == 0 then 3.0 else 23.0]

  let awaiting ← dist.allReduceAsync input .sum
  LeanTest.assertEqual (← input.toFloatArray) #[rankValue]
  let completed ← awaiting
  LeanTest.assertEqual (← completed.toFloatArray) #[3.0]
  LeanTest.assertTrue (completed.requiresGrad && completed.isLeaf)
  let awaitingTree ← dist.allReduceGradsAsync (input, input) .sum
  LeanTest.assertEqual (← input.toFloatArray) #[rankValue]
  let completedTree ← awaitingTree
  LeanTest.assertEqual (← completedTree.1.toFloatArray) #[3.0]
  LeanTest.assertEqual (← completedTree.2.toFloatArray) #[3.0]
  LeanTest.assertTrue (completedTree.1.requiresGrad && completedTree.2.requiresGrad)

/-- Exercise normal cleanup, reinitialization, and cleanup after an action error
    on a real two-rank Gloo group. This is run by the bounded separate launcher. -/
def checkTwoRankSession (rank port restartPort : Nat) : IO Unit := do
  let result ← dist.withDistributed "gloo" "127.0.0.1" port rank 2 do
    checkTwoRankValues rank
    dist.barrier
    pure (rank + 7)
  LeanTest.assertEqual result (rank + 7)
  LeanTest.assertFalse (← dist.isInitialized)
    "A successful distributed action must release its process group"

  -- Cleanup is local: the first barrier does not guarantee that rank 0 has
  -- closed its TCPStore before rank 1 starts rendezvous again.
  let failure ← try
    dist.withDistributed "gloo" "127.0.0.1" restartPort rank 2 do
      dist.barrier
      throw (IO.userError "intentional distributed action failure")
    pure none
  catch error => pure (some error.toString)
  let retained ← dist.isInitialized
  -- Also clean up the unfixed baseline, so a failing regression leaves no peer.
  if retained then dist.destroyProcessGroup
  LeanTest.assertEqual failure (some "intentional distributed action failure")
  LeanTest.assertFalse retained
    "A failing distributed action must release its process group"

end Tests.DistributedValues
