import Tyr.Module.Derive
import Tyr.Typed.Ops
import LeanTest

namespace Tests.ADTensorStructSchema

open torch
open LeanTest

structure Inner where
  bias : Tensor { shape := #[4], dtype := .Float32, device := .CPU }
  tag : Static String
  cache : Frozen { shape := #[4], dtype := .Float32, device := .CPU }
  deriving TensorStruct, ToTensorStructSchema

structure Outer where
  weight : Tensor { shape := #[2, 4], dtype := .Float32, device := .CPU }
  inner : Inner
  gates : torch.Vector 2 (Frozen { shape := #[1], dtype := .Float32, device := .CPU })
  optScale : Option (Tensor { shape := #[3], dtype := .Float32, device := .CPU })
  flag : Bool
  deriving TensorStruct, ToTensorStructSchema

structure IndexedContainer where
  blocks : Array (Tensor { shape := #[2], dtype := .Float32, device := .CPU })
  labels : List (Static String)
  maybeFrozen : Option (Frozen { shape := #[2], dtype := .Float32, device := .CPU })
  deriving TensorStruct, ToTensorStructSchema

private def sampleOuter : Outer :=
  { weight := Tensor.zeros #[2, 4]
    inner := {
      bias := Tensor.zeros #[4]
      tag := "inner-tag"
      cache := (Tensor.zeros #[4] : Frozen { shape := #[4], dtype := .Float32, device := .CPU })
    }
    gates := torch.Vector.replicate 2 ((Tensor.zeros #[1]) : Frozen { shape := #[1], dtype := .Float32, device := .CPU })
    optScale := some (Tensor.zeros #[3])
    flag := true }

private def sampleIndexed : IndexedContainer :=
  { blocks := #[Tensor.zeros #[2], Tensor.zeros #[2]]
    labels := ["a", "b"]
    maybeFrozen := none }

@[test]
def testDerivedTensorStructSchemaForNestedStructure : IO Unit := do
  let schema := ToTensorStructSchema.schema sampleOuter
  LeanTest.assertEqual schema.typeName ``Outer
    "Derived schema should retain the owning structure name."
  LeanTest.assertEqual schema.renderedLeafPaths
    #[
      "weight",
      "inner.bias",
      "inner.tag",
      "inner.cache",
      "gates[0]",
      "gates[1]",
      "optScale",
      "flag"
    ]
    "Derived schema should preserve declaration-order leaf paths."
  let expectedRoles : Array TensorLeafRole := #[
    .diff,
    .diff,
    .static,
    .frozen,
    .frozen,
    .frozen,
    .diff,
    .static
  ]
  LeanTest.assertTrue ((schema.leaves.map (·.role)) == expectedRoles)
    s!"Leaf roles should track differentiable/static/frozen participation, got {reprStr (schema.leaves.map (·.role))}"
  let expectedShapes : Array (Option (Array Nat)) := #[
    some #[2, 4],
    some #[4],
    none,
    some #[4],
    some #[1],
    some #[1],
    some #[3],
    none
  ]
  LeanTest.assertTrue ((schema.leaves.map (·.shape)) == expectedShapes)
    s!"Tensor/frozen leaf shapes should be preserved, got {reprStr (schema.leaves.map (·.shape))}"

@[test]
def testTensorStructSchemaIndexesArraysListsAndSkipsNone : IO Unit := do
  let schema := ToTensorStructSchema.schema sampleIndexed
  LeanTest.assertEqual schema.renderedLeafPaths
    #[
      "blocks[0]",
      "blocks[1]",
      "labels[0]",
      "labels[1]"
    ]
    "Indexed containers should contribute stable bracketed path segments."
  let expectedRoles : Array TensorLeafRole := #[
    .diff,
    .diff,
    .static,
    .static
  ]
  LeanTest.assertTrue ((schema.leaves.map (·.role)) == expectedRoles)
    s!"Container roles should classify tensor vs static leaves, got {reprStr (schema.leaves.map (·.role))}"
  let expectedShapes : Array (Option (Array Nat)) := #[
    some #[2],
    some #[2],
    none,
    none
  ]
  LeanTest.assertTrue ((schema.leaves.map (·.shape)) == expectedShapes)
    s!"Container element shapes should be preserved, got {reprStr (schema.leaves.map (·.shape))}"

end Tests.ADTensorStructSchema
