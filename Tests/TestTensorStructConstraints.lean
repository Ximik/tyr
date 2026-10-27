import LeanTest
import Tyr.Module.Derive
import Tyr.Optim
import Tyr.DiffEq.Typed

namespace Tests.TensorStructConstraints

open torch LeanTest

structure Block (dtype : DType) {device : Device} where
  weight : Tensor ⟨#[2], dtype, device⟩
  frozen : Frozen ⟨#[1], dtype, device⟩
  label : Static String
  deriving TensorStruct

structure Parameters (dtype : DType) {device : Device} where
  blocks : Array (Option (Block dtype (device := device)))
  step : Nat
  deriving TensorStruct

structure MixedParameters where
  weights : Parameters .Float32 (device := .CPU)
  mask : Tensor ⟨#[2], .Bool, .CPU⟩
  deriving TensorStruct

-- Arbitrary tensors remain traversable independently of arithmetic support.
example : TensorStruct (Parameters .Int64 (device := .CPU)) := inferInstance
example : TensorStruct MixedParameters := inferInstance

-- Ordinary deriving supplies evidence recursively through records and containers.
example : FloatingTensorStruct (Parameters .Float32 (device := .CPU)) := inferInstance
example : FloatingTensorStruct (Parameters .Float64 (device := .CPU)) := inferInstance
example {dtype : DType} {device : Device} [DType.RealFloating dtype] :
    FloatingTensorStruct (Parameters dtype (device := device)) := inferInstance

example : Optim.GradientTransformation (Parameters .Float32 (device := .CPU)) Optim.EmptyState :=
  Optim.sgd 0.1
example : Optim.GradientTransformation (Parameters .Float64 (device := .CPU))
    (Optim.AdamWState (Parameters .Float64 (device := .CPU))) := Optim.adamw
example : torch.DiffEq.DiffEqSpace (Parameters .Float32 (device := .CPU)) := inferInstance

-- A structural traversal is insufficient to justify floating arithmetic.
example : True := by
  fail_if_success
    have _ := inferInstanceAs (FloatingTensorStruct (Parameters .Int64 (device := .CPU)))
  fail_if_success
    have _ := inferInstanceAs (FloatingTensorStruct MixedParameters)
  fail_if_success
    have _ := inferInstanceAs (FloatingTensorStruct (SomeDTypeTensor #[2] .CPU))
  fail_if_success
    have _ := fun (parameters : Parameters .Int64 (device := .CPU)) =>
      TensorStruct.scale parameters 0.5
  fail_if_success
    have _ := Optim.sgd (α := Parameters .Int64 (device := .CPU)) 0.1
  fail_if_success
    have _ := Optim.adamw (α := Parameters .Int64 (device := .CPU))
  fail_if_success
    have _ := inferInstanceAs (torch.DiffEq.DiffEqSpace (Parameters .Int64 (device := .CPU)))
  trivial

example {dtype : DType} {device : Device} [DType.RealFloating dtype] :
    FloatingTensorStruct (ShapePackedTensor dtype device) := inferInstance

private def OnCPU (σ : StaticSpec) : Prop := σ.device = .CPU

instance {shape : Shape} {dtype : DType} :
    TensorStructFor.Fact (OnCPU ⟨shape, dtype, .CPU⟩) := ⟨rfl⟩

-- Predicates are independent of floating arithmetic: the Bool leaf is valid here.
example : TensorStructFor OnCPU MixedParameters := inferInstance

structure CustomTraversal where
  tensor : Tensor ⟨#[2], .Float64, .CPU⟩

instance customTraversal : TensorStruct CustomTraversal where
  map f value := ⟨f value.tensor⟩
  mapM f value := do pure ⟨← f value.tensor⟩
  zipWith f lhs rhs := ⟨f lhs.tensor rhs.tensor⟩
  fold f initial value := f value.tensor initial

derive_tensor_struct_for CustomTraversal

example : FloatingTensorStruct CustomTraversal := inferInstance
example : (inferInstance : TensorStruct CustomTraversal) = customTraversal := rfl

private def parameters : Parameters .Float32 (device := .CPU) := {
  blocks := #[some {
    weight := Tensor.fromFloatArray #[1, 2]
    frozen := ⟨Tensor.fromFloatArray #[3]⟩
    label := ⟨"retained metadata"⟩
  }, none]
  step := 7
}

@[test] def testConstrainedTraversalPreservesStructureAndLeafEvidence : IO Unit := do
  let visits ← IO.mkRef (#[] : Array Shape)
  let result ← TensorStructFor.mapM
    (P := fun σ => DType.RealFloating σ.dtype)
    (fun {σ} proof tensor => do
      let _ : σ.dtype.isRealFloating = true := proof.valid
      visits.modify (·.push σ.shape)
      pure (Tensor.detach tensor)) parameters
  assertEqual (← visits.get) #[#[2], #[1]]
    "Constrained traversal must visit ordinary and Frozen leaves in field order"
  assertEqual result.step 7
  assertEqual result.blocks.size 2
  assertTrue result.blocks[1]!.isNone
  let some block := result.blocks[0]! | fail "Missing traversed block"
  assertEqual block.label.val "retained metadata"
  assertEqual (← block.weight.toFloatArray) #[1, 2]
  assertEqual (← block.frozen.tensor.toFloatArray) #[3]
  let count := TensorStructFor.fold
    (P := fun σ => DType.RealFloating σ.dtype)
    (fun _ _ count => count + 1) 0 result
  assertEqual count 2

@[test] def testFloatingTreeArithmeticUsesLeafCapabilities : IO Unit := do
  let scaled := TensorStruct.scale parameters 3
  let combined := TensorStruct.sub (TensorStruct.add scaled parameters) parameters
  let some block := combined.blocks[0]! | fail "Missing transformed block"
  assertEqual (← block.weight.toFloatArray) #[3, 6]
  assertEqual (← block.frozen.tensor.toFloatArray) #[9]
  assertEqual block.label.val "retained metadata"
  let double : CustomTraversal := ⟨(Tensor.fromFloatArray #[2, 4]).castDType .Float64⟩
  let result := FloatingTensorStruct.map (fun tensor => Tensor.mulScalar tensor 0.5) double
  assertEqual result.tensor.raw.dtype .Float64
  assertEqual (← result.tensor.toFloatArray) #[1, 2]

end Tests.TensorStructConstraints
