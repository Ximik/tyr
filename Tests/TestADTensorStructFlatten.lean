import Tyr.Module.Derive
import Tyr.Typed.Ops
import LeanTest

namespace Tests.ADTensorStructFlatten

open torch
open LeanTest

structure Inner where
  bias : Tensor { shape := #[4], dtype := .Float32 }
  tag : Static String
  cache : Frozen { shape := #[4], dtype := .Float32 }
  deriving Model

structure Outer where
  weight : Tensor { shape := #[2, 4], dtype := .Float32 }
  inner : Inner
  gates : torch.Vector 2 (Frozen { shape := #[1], dtype := .Float32 })
  optScale : Option (Tensor { shape := #[3], dtype := .Float32 })
  flag : Bool
  deriving Model

set_option linter.unusedVariables false in
example (wrongShape : Tensor { shape := #[3], dtype := .Float32 }) (wrongDType : Tensor { shape := #[4], dtype := .Int64 }) : True := by
  fail_if_success
    have _ : Frozen { shape := #[4], dtype := .Float32 } := ⟨wrongShape⟩
  fail_if_success
    have _ : Frozen { shape := #[4], dtype := .Float32 } := ⟨wrongDType⟩
  trivial

set_option linter.unusedVariables false in
example (params : Tensor { shape := #[4], dtype := .Float32 }) : True := by
  fail_if_success
    have _ := TensorStruct.map (fun _ => Tensor.zeros #[3]) params
  fail_if_success
    have _ := TensorStruct.map (fun tensor => Tensor.toInt64 tensor) params
  trivial

private def templateOuter : Outer :=
  { weight := Tensor.zeros #[2, 4]
    inner := {
      bias := Tensor.zeros #[4]
      tag := "template-tag"
      cache := (Tensor.full #[4] 7.0 : Frozen { shape := #[4], dtype := .Float32 })
    }
    gates := ⟨#[
      (Tensor.full #[1] 10.0 : Frozen { shape := #[1], dtype := .Float32 }),
      (Tensor.full #[1] 20.0 : Frozen { shape := #[1], dtype := .Float32 })
    ], rfl⟩
    optScale := some (Tensor.zeros #[3])
    flag := true }

private def replacementOuter : Outer :=
  { weight := Tensor.full #[2, 4] 3.0
    inner := {
      bias := Tensor.full #[4] 4.0
      tag := "replacement-tag"
      cache := (Tensor.full #[4] 8.0 : Frozen { shape := #[4], dtype := .Float32 })
    }
    gates := ⟨#[
      (Tensor.full #[1] 30.0 : Frozen { shape := #[1], dtype := .Float32 }),
      (Tensor.full #[1] 40.0 : Frozen { shape := #[1], dtype := .Float32 })
    ], rfl⟩
    optScale := some (Tensor.full #[3] 5.0)
    flag := false }

private def expectedDiffOnlyRebuild : Outer :=
  { weight := replacementOuter.weight
    inner := {
      bias := replacementOuter.inner.bias
      tag := templateOuter.inner.tag
      cache := templateOuter.inner.cache
    }
    gates := templateOuter.gates
    optScale := replacementOuter.optScale
    flag := templateOuter.flag }

private def expectedDiffAndFrozenRebuild : Outer :=
  { weight := replacementOuter.weight
    inner := {
      bias := replacementOuter.inner.bias
      tag := templateOuter.inner.tag
      cache := replacementOuter.inner.cache
    }
    gates := replacementOuter.gates
    optScale := replacementOuter.optScale
    flag := templateOuter.flag }

private def leafRoles (leaves : Array TensorLeafValue) : Array TensorLeafRole :=
  leaves.map (·.role)

private def leafShapes (leaves : Array TensorLeafValue) : Array (Array Nat) :=
  leaves.map (·.shape)

private def leavesAllclose (lhs rhs : Array TensorLeafValue) : Bool := Id.run do
  if lhs.size != rhs.size then
    return false
  for h : i in [:lhs.size] do
    let l := lhs[i]
    let some r := rhs[i]? | return false
    if l.role != r.role then
      return false
    match l.payload, r.payload with
    | ⟨ls, lt⟩, ⟨rs, rt⟩ =>
        if ls == rs then
          if !(RawTensor.allclose lt.raw rt.raw 1e-05 1e-08) then
            return false
        else
          return false
  return true

private def unwrapOrFail {α : Type} (result : Except String α) : IO α :=
  match result with
  | .ok value => pure value
  | .error err => LeanTest.fail err

@[test]
def testTensorStructFlattenDiffOnlyUsesDifferentiableLeafOrder : IO Unit := do
  let leaves := TensorStructFlatten.flatten templateOuter .diffOnly
  LeanTest.assertEqual leaves.size 3
    "diffOnly flattening should keep only differentiable tensor leaves."
  let expectedRoles : Array TensorLeafRole := #[.diff, .diff, .diff]
  LeanTest.assertTrue (leafRoles leaves == expectedRoles)
    s!"diffOnly roles should be purely differentiable, got {reprStr (leafRoles leaves)}"
  let expectedShapes : Array (Array Nat) := #[#[2, 4], #[4], #[3]]
  LeanTest.assertTrue (leafShapes leaves == expectedShapes)
    s!"diffOnly flattening should preserve declaration-order tensor shapes, got {reprStr (leafShapes leaves)}"

@[test]
def testTensorStructFlattenDiffAndFrozenIncludesForwardOnlyLeaves : IO Unit := do
  let leaves := TensorStructFlatten.flatten templateOuter .diffAndFrozen
  LeanTest.assertEqual leaves.size 6
    "diffAndFrozen flattening should include forward-only frozen tensor leaves."
  let expectedRoles : Array TensorLeafRole := #[
    .diff,
    .diff,
    .frozen,
    .frozen,
    .frozen,
    .diff
  ]
  LeanTest.assertTrue (leafRoles leaves == expectedRoles)
    s!"diffAndFrozen roles should retain differentiable/frozen ordering, got {reprStr (leafRoles leaves)}"
  let expectedShapes : Array (Array Nat) := #[
    #[2, 4],
    #[4],
    #[4],
    #[1],
    #[1],
    #[3]
  ]
  LeanTest.assertTrue (leafShapes leaves == expectedShapes)
    s!"diffAndFrozen flattening should preserve all selected tensor shapes, got {reprStr (leafShapes leaves)}"

@[test]
def testTensorStructRebuildFromDiffOnlyReplacesOnlyDiffLeaves : IO Unit := do
  let rebuilt ← unwrapOrFail <|
    TensorStructFlatten.rebuildFrom templateOuter
      (TensorStructFlatten.flatten replacementOuter .diffOnly)
      .diffOnly
  LeanTest.assertTrue (rebuilt.flag == templateOuter.flag)
    "Rebuilding from diffOnly leaves should preserve static fields from the template."
  LeanTest.assertTrue (rebuilt.inner.tag == templateOuter.inner.tag)
    "Rebuilding from diffOnly leaves should preserve nested static fields from the template."
  LeanTest.assertTrue
    (leavesAllclose
      (TensorStructFlatten.flatten rebuilt .diffAndFrozen)
      (TensorStructFlatten.flatten expectedDiffOnlyRebuild .diffAndFrozen))
    "Rebuilding from diffOnly leaves should update only differentiable tensors."

@[test]
def testTensorStructRebuildFromDiffAndFrozenReplacesAllSelectedLeaves : IO Unit := do
  let rebuilt ← unwrapOrFail <|
    TensorStructFlatten.rebuildFrom templateOuter
      (TensorStructFlatten.flatten replacementOuter .diffAndFrozen)
      .diffAndFrozen
  LeanTest.assertTrue (rebuilt.flag == templateOuter.flag)
    "Rebuilding from diffAndFrozen leaves should still preserve static fields from the template."
  LeanTest.assertTrue (rebuilt.inner.tag == templateOuter.inner.tag)
    "Rebuilding from diffAndFrozen leaves should still preserve nested static fields from the template."
  LeanTest.assertTrue
    (leavesAllclose
      (TensorStructFlatten.flatten rebuilt .diffAndFrozen)
      (TensorStructFlatten.flatten expectedDiffAndFrozenRebuild .diffAndFrozen))
    "Rebuilding from diffAndFrozen leaves should update differentiable and frozen tensors."

@[test]
def testTensorStructRebuildRejectsShapeMismatch : IO Unit := do
  let diffLeaves := TensorStructFlatten.flatten replacementOuter .diffOnly
  let badLeaves := diffLeaves.set! 0 (TensorLeafValue.ofTensor .diff (Tensor.full #[8] 1.0))
  match TensorStructFlatten.rebuildFrom templateOuter badLeaves .diffOnly with
  | .ok _ =>
      LeanTest.fail "Rebuild should reject tensor leaves whose shapes do not match the template."
  | .error err =>
      LeanTest.assertTrue (err.containsSubstr "weight")
        s!"Shape mismatch should point at the failing path, got: {err}"

@[test]
def testTensorStructRebuildRejectsMissingAndExtraLeaves : IO Unit := do
  let diffLeaves := TensorStructFlatten.flatten replacementOuter .diffOnly
  let missingLeaves := diffLeaves.extract 0 2
  match TensorStructFlatten.rebuildFrom templateOuter missingLeaves .diffOnly with
  | .ok _ =>
      LeanTest.fail "Rebuild should reject missing tensor leaves."
  | .error err =>
      LeanTest.assertTrue (err.containsSubstr "optScale")
        s!"Missing-leaf diagnostics should point at the first missing path, got: {err}"

  let extraLeaves := diffLeaves.push (TensorLeafValue.ofTensor .diff (Tensor.full #[1] 0.0))
  match TensorStructFlatten.rebuildFrom templateOuter extraLeaves .diffOnly with
  | .ok _ =>
      LeanTest.fail "Rebuild should reject extra tensor leaves after consuming the template shape."
  | .error err =>
      LeanTest.assertTrue (err.containsSubstr "Unexpected extra tensor leaves")
        s!"Extra-leaf diagnostics should explain the unused suffix, got: {err}"

structure MixedParams where
  weight : Tensor { shape := #[2], dtype := .Float32, device := .exact .CPU }
  indices : Tensor { shape := #[2], dtype := .Int64 }
  deriving Model

private def mixedParams : MixedParams := {
  weight := (Tensor.onesFor { shape := #[2], dtype := .Float32, device := .exact .CPU }).mulScalar 3.0
  indices := Tensor.fullInt #[2] 7
}

@[test]
def testTypedModelTraversalAndRebuildPreserveFullSpec : IO Unit := do
  let original := mixedParams
  let mapped ← TensorStruct.mapM (fun tensor => pure tensor) original
  let zipped := TensorStruct.zipWith (fun left _ => left) mapped original
  LeanTest.assertEqual (TensorStruct.count zipped) 2
    "Generic typed leaves should participate in the derived Model traversal."
  let leaves := TensorStructFlatten.flatten zipped
  LeanTest.assertTrue ((leaves.map (·.dtype)) == #[.Float32, .Int64])
    "Flattening must retain each leaf's dtype."
  let some first := leaves[0]? | LeanTest.fail "Missing first typed parameter leaf."
  LeanTest.assertTrue (first.payload.1.device == .exact .CPU)
    "Flattening must retain the pinned device policy."
  let rebuilt ← unwrapOrFail (TensorStructFlatten.rebuildFrom original leaves)
  let _ ← unwrapOrFail rebuilt.weight.validate
  let _ ← unwrapOrFail rebuilt.indices.validate
  LeanTest.assertTrue (leavesAllclose leaves (TensorStructFlatten.flatten rebuilt))
    "Typed heterogeneous parameters must round-trip through flatten/rebuild."

private def expectWeightRebuildError (leaf : TensorLeafValue) (detail : String) : IO Unit := do
  let leaves := (TensorStructFlatten.flatten mixedParams).set! 0 leaf
  match TensorStructFlatten.rebuildFrom mixedParams leaves with
  | .ok _ => LeanTest.fail s!"Rebuild accepted invalid {detail}."
  | .error message =>
      LeanTest.assertTrue (message.containsSubstr "weight" && message.containsSubstr detail)
        s!"Rebuild diagnostic should identify the field and {detail}: {message}"

@[test]
def testFrozenRebuildRejectsDTypeMismatch : IO Unit := do
  let leaves := (TensorStructFlatten.flatten templateOuter).set! 2
    (TensorLeafValue.ofTensor .frozen (Tensor.fullInt #[4] 1))
  match TensorStructFlatten.rebuildFrom templateOuter leaves with
  | .ok _ => LeanTest.fail "Frozen tensor reconstruction accepted a different dtype."
  | .error message =>
      LeanTest.assertTrue (message.containsSubstr "inner.cache" && message.containsSubstr "dtype")
        s!"Frozen mismatch diagnostic should identify the field and dtype: {message}"

@[test]
def testTypedRebuildRejectsShapeDTypeAndRole : IO Unit := do
  expectWeightRebuildError
    (TensorLeafValue.ofTensor .diff (Tensor.fullInt #[2] 1)) "dtype"
  expectWeightRebuildError
    (TensorLeafValue.ofTensor .frozen mixedParams.weight) "role"
  expectWeightRebuildError (TensorLeafValue.ofTensor .diff (Tensor.zeros #[3])) "shape"

end Tests.ADTensorStructFlatten
