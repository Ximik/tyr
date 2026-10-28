import LeanTest
import Tyr.Module.Derive
import Tyr.TensorStruct.Cast

namespace Tests.TensorStructDeriveCast

open torch

structure Precision where
  compute : DType
  norm : DType

structure Block (size : Nat) (precision : Precision) {device : Device} where
  weight : Tensor ⟨#[size], precision.compute, device⟩
  norm : Tensor ⟨#[size], precision.norm, device⟩
  indices : Tensor ⟨#[size], .Int64, device⟩
  label : Static String
  deriving TensorStruct

structure Model (size : Nat) (precision : Precision) (device : Device) where
  blocks : Array (Option (Block size precision (device := device)))
  step : Nat
  deriving TensorStruct

abbrev Source := Model 2 ⟨.Float32, .Float64⟩ .CPU
abbrev Target := Model 2 ⟨.BFloat16, .Float32⟩ .CPU

-- Ordinary derivation must provide the conversion, including nested records.
example (input : Source) : IO Target := TensorStruct.castTo Target input

-- Source and destination parameters can be runtime values.
example {size : Nat} {source target : Precision} {sourceDevice targetDevice : Device}
    (input : Model size source sourceDevice) : IO (Model size target targetDevice) :=
  TensorStruct.castTo (Model size target targetDevice) input

private def input : Source := {
  blocks := #[some {
    weight := Tensor.fromFloatArray #[1.25, 2.5]
    norm := (Tensor.fromFloatArray #[3, 4]).castDType .Float64
    indices := Tensor.fromInt64Array #[7, 11]
    label := ⟨"retained metadata"⟩
  }, none]
  step := 9
}

@[test] def testDerivedMixedPrecisionTreeCast : IO Unit := do
  let result ← TensorStruct.castTo Target input
  LeanTest.assertEqual result.blocks.size 2
  LeanTest.assertTrue result.blocks[1]!.isNone
  let some block := result.blocks[0]! | LeanTest.fail "Missing converted block"
  LeanTest.assertEqual block.weight.raw.dtype .BFloat16
  LeanTest.assertEqual block.norm.raw.dtype .Float32
  LeanTest.assertEqual block.indices.raw.dtype .Int64
  LeanTest.assertEqual (← block.weight.toFloatArray) #[1.25, 2.5]
  LeanTest.assertEqual (← block.indices.toUInt64Array) #[7, 11]
  LeanTest.assertEqual block.label.val "retained metadata"
  LeanTest.assertEqual result.step 9

@[test] def testDerivedCastChecksNestedShape : IO Unit := do
  let rejected ← try
    let _ ← TensorStruct.castTo (Model 3 ⟨.BFloat16, .Float32⟩ .CPU) input
    pure false
  catch error =>
    LeanTest.assertTrue (error.toString.containsSubstr "blocks[0].weight")
      "Shape errors must identify the nested field"
    pure true
  LeanTest.assertTrue rejected "Derivation must never turn a cast into a reshape"

structure UnifiedModel (dtype : DType) where
  weight : Tensor ⟨#[2], dtype, .CPU⟩
  deriving torch.Model

example (source : UnifiedModel .Float32) : IO (UnifiedModel .BFloat16) :=
  TensorStruct.castTo (UnifiedModel .BFloat16) source

structure Metadata (α : Type) [Inhabited α] where
  value : Static α
  deriving TensorStruct

example (input : Metadata Nat) : IO (Metadata Nat) :=
  TensorStruct.castTo (Metadata Nat) input

example : True := by
  fail_if_success
    have _ : TensorTreeCast (Metadata Nat) (Metadata String) := inferInstance
  trivial

structure DependentParameters (size : Nat) (selected : Fin size) (dtype : DType) where
  weight : Tensor ⟨#[selected.val], dtype, .CPU⟩
  deriving TensorStruct

example {n m : Nat} {i : Fin n} {j : Fin m} {source target : DType}
    (input : DependentParameters n i source) : IO (DependentParameters m j target) :=
  TensorStruct.castTo (DependentParameters m j target) input

structure DependentFields where
  size : Nat
  weight : Tensor ⟨#[size], .Float32, .CPU⟩

/-- error: Automatic TensorTreeCast derivation does not support fields whose types depend on earlier field values; use a manual TensorTreeCast instance. -/
#guard_msgs in
run_cmd torch.deriveTensorTreeCastFor ``DependentFields

end Tests.TensorStructDeriveCast
