import Tyr.Inference.KVCache
import LeanTest

namespace Tests.InferenceCache

open torch
open torch.Generator.KVCache

private def values {σ : StaticSpec} (t : Tensor σ) : IO (Array UInt64) :=
  Tensor.toUInt64Array t.toInt64

private def layer {n b len h d : UInt64} (cache : Cache n b len h d) (i : Nat)
    : IO (LayerCache b len h d) :=
  match cache.layers[i]? with
  | some value => pure value
  | none => throw <| IO.userError "Missing test cache layer"

private def token (x : Float) : DTensor #[1, 1, 1, 2] .BFloat16 :=
  (Tensor.full #[1, 1, 1, 2] x).toBFloat16

private def rejected (action : IO α) (label : String) : IO Unit := do
  let failed ← try let _ ← action; pure false catch _ => pure true
  LeanTest.assertTrue failed label

@[test] def testInferenceCacheAllocationsAreIndependent : IO Unit := do
  let initial ← Cache.init 2 1 4 1 2
  let untouched ← Cache.init 2 1 4 1 2
  let cache ← initial.appendLayer 0 (token 3) (token 7)
  LeanTest.assertEqual (← values (← layer cache 0).keys) #[3, 3, 0, 0, 0, 0, 0, 0]
    "Key writes must survive a different value write"
  LeanTest.assertEqual (← values (← layer cache 0).values) #[7, 7, 0, 0, 0, 0, 0, 0]
    "K/V use independent storage"
  LeanTest.assertEqual (← values (← layer cache 1).keys) (Array.replicate 8 0)
    "Identically shaped layers are independently allocated"
  LeanTest.assertEqual (← values (← layer cache 1).values) (Array.replicate 8 0)
    "Value storage is independent across layers"
  LeanTest.assertEqual (← values (← layer untouched 0).keys) (Array.replicate 8 0)
    "Repeated Cache.init calls do not share storage"
  let cache ← cache.appendLayer 1 (token 5) (token 9)
  let cache ← cache.incrementSeqLens
  let cache ← cache.appendLayer 0 (token 11) (token 13)
  LeanTest.assertEqual (← values (← layer cache 0).keys) #[3, 3, 11, 11, 0, 0, 0, 0]
    "Ordered appends preserve preceding key slots"
  LeanTest.assertEqual (← values (← layer cache 0).values) #[7, 7, 13, 13, 0, 0, 0, 0]
    "Ordered appends preserve preceding value slots"
  LeanTest.assertEqual (← values (← layer cache 1).keys) #[5, 5, 0, 0, 0, 0, 0, 0]
    "Later writes do not change another layer"

@[test] def testInferenceCacheRejectsInvalidWritesBeforeMutation : IO Unit := do
  rejected (Cache.init 1 1 0 1 2) "Zero capacity is rejected"
  let cache ← Cache.init 1 1 1 1 2
  let wrongShape : DTensor #[1, 1, 1, 2] .BFloat16 := Tensor.assumeSpec ((Tensor.full #[1, 1, 1, 3] 9).toBFloat16.raw)
  rejected (cache.appendLayer 0 (token 3) wrongShape) "Invalid V shape is rejected before writing K"
  LeanTest.assertEqual (← values (← layer cache 0).keys) #[0, 0] "Rejected pair leaves K unchanged"
  rejected (cache.attendLayer 0 wrongShape (token 3) (token 7))
    "Invalid query metadata is rejected before either write"
  LeanTest.assertEqual (← values (← layer cache 0).keys) #[0, 0]
    "Rejected query leaves K unchanged"
  LeanTest.assertEqual (← values (← layer cache 0).values) #[0, 0]
    "Rejected query leaves V unchanged"
  rejected (cache.appendLayer 1 (token 3) (token 7)) "Invalid layer is rejected"
  rejected ((← layer cache 0).attentionView 2) "Views cannot exceed capacity"
  let cache ← cache.appendLayer 0 (token 3) (token 7)
  let fullCache ← cache.incrementSeqLens
  rejected (fullCache.appendLayer 0 (token 5) (token 9)) "Full cache rejects append"
  rejected fullCache.incrementSeqLens "Sequence counters cannot exceed capacity"
  LeanTest.assertEqual (← values (← layer fullCache 0).keys) #[3, 3] "Rejected overflow preserves K"
  let invalid : Cache 1 1 1 1 2 := { fullCache with seqLens := #[18446744073709551615] }
  rejected invalid.incrementSeqLens "Sequence counters cannot wrap"

@[test] def testInferenceCacheOrderedAttentionHasNoGraph : IO Unit := do
  let cache ← Cache.init 1 1 2 1 2
  let q := Tensor.setRequiresGrad (token 0) true
  let enabled ← autograd.is_grad_enabled
  let (cache, first) ← cache.attendLayer 0 q (token 3) (token 7)
  LeanTest.assertEqual (← values first) #[7, 7] "One-token attention reads the value after both writes"
  LeanTest.assertTrue (!autograd.has_grad_fn (s := first.shape) first.raw) "Inference attention has no autograd graph"
  LeanTest.assertEqual (← autograd.is_grad_enabled) enabled "Inference restores caller grad mode"
  let cache ← cache.incrementSeqLens
  let (_, second) ← cache.attendLayer 0 q (token 5) (token 11)
  LeanTest.assertEqual (← values second) #[9, 9] "Zero query averages the two separately stored values"

end Tests.InferenceCache
