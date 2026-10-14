import Tyr.Typed.Neural
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

private def token (x : Float) : Tensor { shape := #[1, 1, 1, 2], dtype := .BFloat16 } :=
  (Tensor.full #[1, 1, 1, 2] x).toBFloat16

private def admit {σ : StaticSpec} (raw : RawTensor) : IO (Tensor σ) :=
  match Tensor.ofRaw raw with
  | .ok tensor => pure tensor
  | .error message => throw (IO.userError message)

private def rejected (action : IO α) (label : String) : IO Unit := do
  let failed ← try let _ ← action; pure false catch _ => pure true
  LeanTest.assertTrue failed label

@[test] def testInferenceCloneChecksMetadataAndOwnsStorage : IO Unit := do
  let src := Tensor.full #[2] 3
  let copy ← data.cloneInferenceIO src
  let copy ← Tensor.copyInto copy (Tensor.full #[2] 7)
  LeanTest.assertEqual (← src.toFloatArray) #[3, 3]
    "Writing an inference clone must not mutate its source"
  LeanTest.assertEqual (← copy.toFloatArray) #[7, 7]
  LeanTest.assertTrue ((Tensor.ofRaw (σ := ⟨#[3], .Float32, .any⟩) src.raw).toOption.isNone)
    "A false shape claim is rejected before inference clone admission"
  LeanTest.assertTrue ((Tensor.ofRaw (σ := ⟨#[2], .Int64, .any⟩) src.raw).toOption.isNone)
    "A false dtype claim is rejected before inference clone admission"
  LeanTest.assertTrue ((Tensor.ofRaw (σ := ⟨#[2], .Float32, .exact .MPS⟩) src.raw).toOption.isNone)
    "A false device claim is rejected before inference clone admission"

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
  let wrongShape := (Tensor.full #[1, 1, 1, 3] 9).toBFloat16.raw
  let rejectedPair := do
    let value ← admit (σ := { shape := #[1, 1, 1, 2], dtype := .BFloat16 }) wrongShape
    cache.appendLayer 0 (token 3) value
  rejected rejectedPair "Invalid V shape is rejected before writing K"
  LeanTest.assertEqual (← values (← layer cache 0).keys) #[0, 0] "Rejected pair leaves K unchanged"
  let rejectedQuery := do
    let query ← admit (σ := { shape := #[1, 1, 1, 2], dtype := .BFloat16 }) wrongShape
    cache.attendLayer 0 query (token 3) (token 7)
  rejected rejectedQuery "Invalid query metadata is rejected before either write"
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
  let q ← Tensor.setRequiresGrad (token 0) true
  let (cache, first) ← cache.attendLayer 0 q (token 3) (token 7)
  LeanTest.assertEqual (← values first) #[7, 7] "One-token attention reads the value after both writes"
  LeanTest.assertTrue (!Tensor.hasGradFn first) "Inference attention has no autograd graph"
  LeanTest.assertTrue q.requiresGrad "Inference leaves the caller query trainable"
  let cache ← cache.incrementSeqLens
  let (_, second) ← cache.attendLayer 0 q (token 5) (token 11)
  LeanTest.assertEqual (← values second) #[9, 9] "Zero query averages the two separately stored values"

@[test] def testInferenceAttentionUsesReturnedCacheAndPreservesEarlierViews : IO Unit := do
  let original ← Cache.init 1 1 2 1 2
  let (_, originalView) ← (← layer original 0).attentionView 1
  let (updated, output) ← original.attendLayer 0 (token 0) (token 3) (token 7)
  LeanTest.assertEqual (← values output) #[7, 7]
    "Attention must read the returned cache, not the retained input layer"
  LeanTest.assertEqual (← values (← layer updated 0).values) #[7, 7, 0, 0]
  LeanTest.assertEqual (← values (← layer original 0).values) #[0, 0, 0, 0]
    "Appending must preserve an earlier cache value"
  LeanTest.assertEqual (← values originalView) #[0, 0]
    "Appending must preserve previously exposed cache views"

@[test] def testInferenceCacheMiddleLayerUpdatePreservesOrderAndAliases : IO Unit := do
  let cache ← Cache.init 3 1 2 1 2
  let cache ← cache.appendLayer 0 (token 3) (token 5)
  let before ← cache.appendLayer 2 (token 7) (token 9)
  let after ← before.appendLayer 1 (token 11) (token 13)
  LeanTest.assertEqual (← values (← layer after 0).values) #[5, 5, 0, 0]
    "Updating the middle layer must preserve the first layer's position"
  LeanTest.assertEqual (← values (← layer after 1).values) #[13, 13, 0, 0]
  LeanTest.assertEqual (← values (← layer after 2).values) #[9, 9, 0, 0]
    "Updating the middle layer must restore the last layer's position"
  LeanTest.assertEqual (← values (← layer before 1).values) #[0, 0, 0, 0]
    "The retained earlier cache must preserve its middle layer"

end Tests.InferenceCache
