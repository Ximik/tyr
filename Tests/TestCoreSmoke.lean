/-
  Tests/TestCoreSmoke.lean

  Core smoke tests: deterministic PRNG streams, silent-by-default log
  handlers, and basic TensorStruct traversal.
-/
import LeanTest
import Tyr

open torch

namespace Tests.CoreSmoke

/-! ## PRNG smoke tests -/

@[test]
def testPRNGKeyFromSeedRepeatable : IO Unit := do
  let k := PRNGKey.fromUInt64 42
  let kAgain := PRNGKey.fromUInt64 42
  LeanTest.assertTrue (k == kAgain) "the same seed should produce the same key"

@[test]
def testPRNGSplitDistinctAndRepeatable : IO Unit := do
  let k := PRNGKey.fromUInt64 42
  let (k1, k2) := PRNGKey.split k
  let (k1b, k2b) := PRNGKey.split k
  LeanTest.assertTrue (k1 == k1b && k2 == k2b) "split should be deterministic"
  LeanTest.assertFalse (k1 == k2) "split should produce distinct child keys"
  LeanTest.assertFalse (k1 == k) "first child key should differ from the parent"
  LeanTest.assertFalse (k2 == k) "second child key should differ from the parent"

@[test]
def testPRNGNormal01SaneAndDistinct : IO Unit := do
  let k := PRNGKey.fromUInt64 7
  let (k1, k2) := PRNGKey.split k
  -- Box-Muller with u1 clamped to >= 1e-12 bounds |x| <= sqrt(-2 log 1e-12) < 7.5.
  for tag in ([0, 1, 2, 3] : List UInt32) do
    let x := PRNGKey.normal01 k1 tag
    LeanTest.assertTrue (x == x) "normal01 should not be NaN"
    LeanTest.assertTrue (Float.abs x < 10.0) s!"normal01 should stay in a sane range, got {x}"
  LeanTest.assertTrue (PRNGKey.normal01 k1 0 == PRNGKey.normal01 k1 0)
    "normal01 should be repeatable for the same key and tag"
  for tag in ([0, 1, 2, 3] : List UInt32) do
    let a := PRNGKey.normal01 k1 tag
    let b := PRNGKey.normal01 k2 tag
    LeanTest.assertFalse (a == b) s!"distinct keys should give distinct streams (tag {tag})"

/-! ## Log handler smoke tests -/

private def recordingHandlers (ref : IO.Ref (Array String)) : torch.Log.Handlers := {
  onInfo := fun msg => ref.modify (·.push msg)
  onWarn := fun msg => ref.modify (·.push msg)
  onError := fun msg => ref.modify (·.push msg)
}

@[test]
def testDefaultLogHandlersAreSilentNoOps : IO Unit := do
  let h : torch.Log.Handlers := {}
  -- Defaults are `fun _ => pure ()`: invoking them must not fail or record anywhere.
  h.onInfo "info"
  h.onWarn "warn"
  h.onError "error"
  LeanTest.assertTrue true "default handlers should be silent no-ops"

@[test]
def testLogHandlersCombineReachesBothSinks : IO Unit := do
  let ref1 ← IO.mkRef (#[] : Array String)
  let ref2 ← IO.mkRef (#[] : Array String)
  let combined := torch.Log.Handlers.combine (recordingHandlers ref1) (recordingHandlers ref2)
  combined.onInfo "hello-info"
  combined.onWarn "hello-warn"
  combined.onError "hello-error"
  LeanTest.assertEqual (← ref1.get) #["hello-info", "hello-warn", "hello-error"]
    "first sink should receive every message"
  LeanTest.assertEqual (← ref2.get) #["hello-info", "hello-warn", "hello-error"]
    "second sink should receive every message"

/-! ## TensorStruct smoke tests -/

structure SmokeParams where
  weight : DTensor #[2, 3] .Float32
  bias : DTensor #[2] .Float32
  tag : Static String
  deriving TensorStruct

private def smokeParams : SmokeParams :=
  { weight := Tensor.zeros #[2, 3]
    bias := Tensor.zeros #[2]
    tag := "smoke" }

@[test]
def testTensorStructCount : IO Unit := do
  LeanTest.assertEqual (TensorStruct.count smokeParams) 2
    "Static fields should be skipped; only tensor leaves are counted"

@[test]
def testTensorStructMapReachesEveryLeaf : IO Unit := do
  let shifted := TensorStruct.map (fun t => Tensor.Preserving.addScalar t 1.0) smokeParams
  let total := TensorStruct.fold (fun {_s} t acc => acc + nn.item (s := #[]) (Tensor.sumAll t).raw) 0.0 shifted
  LeanTest.assertEqual total 8.0
    s!"all-ones leaves (2x3 + 2 elements) should sum to 8, got {total}"
  LeanTest.assertEqual shifted.tag.val "smoke"
    "Static fields should pass through map unchanged"

@[test]
def testTensorStructFoldCountsScalars : IO Unit := do
  let numScalars := TensorStruct.fold (fun {s} _t acc => acc + s.shape.foldl (· * ·) 1) 0 smokeParams
  LeanTest.assertEqual numScalars 8 "2x3 weight plus 2 bias entries give 8 scalars"

end Tests.CoreSmoke
