import LeanTest
import Tyr.TensorStruct.DeriveCast

namespace Tests.TensorTreeCast

open torch

structure SourceModel where
  weight : Tensor { shape := #[2, 2], dtype := .Float32, device := .CPU }
  norm : Tensor { shape := #[2], dtype := .Float64, device := .CPU }
  indices : Tensor { shape := #[2], dtype := .Int64, device := .CPU }
  config : Static String

structure TargetModel where
  weight : Tensor { shape := #[2, 2], dtype := .BFloat16, device := .CPU }
  norm : Tensor { shape := #[2], dtype := .Float32, device := .CPU }
  indices : Tensor { shape := #[2], dtype := .Int64, device := .CPU }
  config : Static String

derive_tensor_tree_cast SourceModel => TargetModel

structure GenericCastModel (size : Nat) (dtype : DType) where
  weight : Tensor { shape := #[size], dtype, device := .CPU }
  label : Static String

abbrev FloatCastAlias := GenericCastModel 3 .Float32
abbrev BFloatCastAlias := GenericCastModel 3 .BFloat16

derive_tensor_tree_cast FloatCastAlias => BFloatCastAlias

-- Derivation unfolds aliases and retains the instantiated target parameters.
example (input : FloatCastAlias) : IO BFloatCastAlias :=
  TensorStruct.castTo BFloatCastAlias input

example (_output : BFloatCastAlias) : True := by
  fail_if_success
    have _ : Tensor { shape := #[3], dtype := .Float32, device := .CPU } := _output.weight
  trivial

structure RenamedCastTarget where
  kernel : Tensor { shape := #[3], dtype := .BFloat16, device := .CPU }
  label : Static String

structure ReorderedCastTarget where
  label : Static String
  weight : Tensor { shape := #[3], dtype := .BFloat16, device := .CPU }

/-- error: Tensor tree cast requires the same field names in the same order;
source fields are [weight, label], target fields are [kernel, label]. -/
#guard_msgs (whitespace := lax) in
derive_tensor_tree_cast FloatCastAlias => RenamedCastTarget

/-- error: Tensor tree cast requires the same field names in the same order;
source fields are [weight, label], target fields are [label, weight]. -/
#guard_msgs (whitespace := lax) in
derive_tensor_tree_cast FloatCastAlias => ReorderedCastTarget

private def source : SourceModel := {
  weight := Tensor.onesFor { shape := #[2, 2], dtype := .Float32, device := .CPU }
  norm := Tensor.onesFor { shape := #[2], dtype := .Float64, device := .CPU }
  indices := Tensor.fromInt64Array #[4, 9]
  config := ⟨"mixed target"⟩
}

-- The result carries the target leaf types; it cannot retain the source dtype.
example (_model : TargetModel) : True := by
  fail_if_success
    have _ : Tensor { shape := #[2, 2], dtype := .Float32, device := .CPU } := _model.weight
  trivial

@[test] def testHeterogeneousTargetRecord : IO Unit := do
  let input := source
  let output ← TensorStruct.castTo TargetModel input
  LeanTest.assertEqual output.weight.raw.dtype .BFloat16
  LeanTest.assertEqual output.norm.raw.dtype .Float32
  LeanTest.assertEqual output.indices.raw.dtype .Int64
  LeanTest.assertEqual (← output.weight.toFloatArray) #[1, 1, 1, 1]
  LeanTest.assertEqual (← output.indices.toUInt64Array) #[4, 9]
  LeanTest.assertEqual output.config.val "mixed target"
  LeanTest.assertTrue (output.weight.staticSpec.device == .CPU)
  LeanTest.assertEqual input.weight.raw.dtype .Float32
  LeanTest.assertEqual input.norm.raw.dtype .Float64

abbrev SourceLeaves := Array (Option (Tensor { shape := #[2], dtype := .Float32, device := .CPU }))
abbrev TargetLeaves := Array (Option (Tensor { shape := #[2], dtype := .Float64, device := .CPU }))

@[test] def testNestedTargetContainers : IO Unit := do
  let input : SourceLeaves := #[some (Tensor.fromFloatArray #[1.25, 2.5]), none,
    some (Tensor.fromFloatArray #[3, 4])]
  let output ← TensorStruct.castTo TargetLeaves input
  LeanTest.assertEqual output.size 3
  LeanTest.assertTrue output[1]!.isNone
  let some first := output[0]! | LeanTest.fail "Missing converted first leaf"
  LeanTest.assertEqual first.raw.dtype .Float64
  LeanTest.assertTrue (first.device == .CPU)
  LeanTest.assertEqual (← first.toFloatArray) #[1.25, 2.5]
  let pair ← TensorStruct.castTo
    (Frozen { shape := #[2], dtype := .BFloat16, device := .CPU } × List (Tensor { shape := #[2], dtype := .Float64, device := .CPU }))
    ((Frozen.mk (Tensor.ones #[2])), [Tensor.ones #[2], Tensor.full #[2] 3])
  LeanTest.assertEqual pair.1.tensor.raw.dtype .BFloat16
  LeanTest.assertEqual pair.2.length 2
  LeanTest.assertEqual pair.2.head!.raw.dtype .Float64
  let vector : torch.Vector 2 (Tensor { shape := #[2], dtype := .Float32, device := .CPU }) :=
    ⟨#[Tensor.ones #[2], Tensor.full #[2] 3], rfl⟩
  let converted ← TensorStruct.castTo
    (torch.Vector 2 (Tensor { shape := #[2], dtype := .Float64, device := .CPU })) vector
  LeanTest.assertEqual converted.data.size 2
  LeanTest.assertEqual converted.data[1]!.raw.dtype .Float64

example : True := by
  fail_if_success
    have _ : torch.TensorTreeCast
      (torch.Vector 2 (Tensor { shape := #[2], dtype := .Float32, device := .CPU }))
      (torch.Vector 3 (Tensor { shape := #[2], dtype := .Float64, device := .CPU })) := inferInstance
  fail_if_success
    have _ : torch.TensorTreeCast
      (Frozen { shape := #[2], dtype := .Float32, device := .CPU })
      (Tensor { shape := #[2], dtype := .Float64, device := .CPU }) := inferInstance
  trivial

structure WrongShapeModel where
  weight : Tensor { shape := #[4], dtype := .Float32, device := .CPU }
  norm : Tensor { shape := #[2], dtype := .Float32, device := .CPU }
  indices : Tensor { shape := #[2], dtype := .Int64, device := .CPU }
  config : Static String

derive_tensor_tree_cast SourceModel => WrongShapeModel

@[test] def testTreeCastRejectsShapeAndUnsupportedDType : IO Unit := do
  let shapeRejected ← try
    let _ ← TensorStruct.castTo WrongShapeModel source
    pure false
  catch error =>
    LeanTest.assertTrue (error.toString.containsSubstr "weight" && error.toString.containsSubstr "shape")
      "A shape mismatch must identify its field path"
    pure true
  LeanTest.assertTrue shapeRejected "A type-tree cast must not implicitly reshape equal-size tensors"
  let dtypeRejected ← try
    let _ ← TensorStruct.castTo (Array (Tensor { shape := #[2], dtype := .QInt8, device := .CPU }))
      #[Tensor.ones #[2]]
    pure false
  catch error =>
    LeanTest.assertTrue (error.toString.containsSubstr "[0]")
      "An unsupported dtype must identify its array index"
    pure true
  LeanTest.assertTrue dtypeRejected "Unsupported target dtype must fail"

@[test] def testExistentialTargetRetainsActualDType : IO Unit := do
  let input := (Tensor.ones #[2]).toBFloat16
  let output ← TensorStruct.castTo (SomeDTypeTensor #[2] .CPU) input
  LeanTest.assertEqual output.1 .BFloat16
  LeanTest.assertEqual output.2.raw.dtype .BFloat16
  LeanTest.assertEqual output.2.runtimeShape #[2]
  LeanTest.assertTrue (output.2.device == .CPU)

@[test] def testTreeCastReportsNativeError : IO Unit := do
  let input := #[Tensor.ones #[2]]
  let rejected ← try
    let _ ← TensorStruct.castTo
      (Array (Tensor {
        shape := #[2]
        dtype := .QInt8
        device := .CPU
      })) input
    pure false
  catch error =>
    LeanTest.assertTrue (error.toString.containsSubstr "[0]" &&
      error.toString.containsSubstr "cast")
      "Native conversion errors must retain the tensor's structural path"
    pure true
  LeanTest.assertTrue rejected "An unsupported quantized conversion must return an IO error"
  LeanTest.assertEqual input[0]!.raw.dtype .Float32
  LeanTest.assertEqual (← input[0]!.toFloatArray) #[1, 1]

end Tests.TensorTreeCast
