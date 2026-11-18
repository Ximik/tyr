/-
  Tests/TestCoreSmoke.lean

  Core smoke tests: silent-by-default log handlers and basic TensorStruct
  traversal.
-/
import LeanTest
import Tyr

open torch

namespace Tests.CoreSmoke

/-! ## Log handler smoke tests -/

private def recordingHandlers (ref : IO.Ref (Array String)) : torch.Log.Handlers := {
  onInfo := fun msg => ref.modify (·.push msg)
  onWarn := fun msg => ref.modify (·.push msg)
  onError := fun msg => ref.modify (·.push msg)
}

@[test]
def testDefaultLogHandlersAreSilentNoOps : IO Unit := do
  let h : torch.Log.Handlers := {}
  let (output, _) ← IO.FS.withIsolatedStreams do
    h.onInfo "info"
    h.onWarn "warn"
    h.onError "error"
  LeanTest.assertEqual output "" (some "default handlers must write nothing to stdout or stderr")

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
  weight : Tensor { shape := #[2, 3], dtype := .Float32, device := .CPU }
  bias : Tensor { shape := #[2], dtype := .Float32, device := .CPU }
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
  let shifted := FloatingTensorStruct.map (fun t => Tensor.addScalar t 1.0) smokeParams
  let total := TensorStruct.fold (fun {_s} t acc => acc + Tensor.item! (Tensor.sumAll t)) 0.0 shifted
  LeanTest.assertEqual total 8.0
    s!"all-ones leaves (2x3 + 2 elements) should sum to 8, got {total}"
  LeanTest.assertEqual shifted.tag.val "smoke"
    "Static fields should pass through map unchanged"

@[test]
def testTensorStructFoldCountsScalars : IO Unit := do
  let numScalars := TensorStruct.fold (fun {s} _t acc => acc + s.shape.foldl (· * ·) 1) 0 smokeParams
  LeanTest.assertEqual numScalars 8 "2x3 weight plus 2 bias entries give 8 scalars"

end Tests.CoreSmoke
