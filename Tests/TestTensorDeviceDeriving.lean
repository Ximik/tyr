import LeanTest
import Tyr.Core.TensorStruct.Derive
import Tyr.Core.TensorStruct.DeriveCast
import Tests.SafeTensorsRecords.Single

namespace Tests.TensorDeviceDeriving

open torch

-- A runtime-selected device is shared by a generated record and its loader.
example (src : safetensors.Source) (device : Device) : IO (SingleSafe.Weights device) :=
  SingleSafe.Weights.load src "" device

example (device : Device) (weights : SingleSafe.Weights device) :
    Tensor { shape := #[2, 3], dtype := .Float32, device } :=
  weights.linear.weight

variable {device : Device}

-- Each handler must apply both the implicit device and explicit size correctly.
structure MappedRecord (size : Nat) where
  value : Tensor { shape := #[size], dtype := .Float32, device := device }
  deriving TensorStruct

structure SchemaRecord (size : Nat) where
  value : Tensor { shape := #[size], dtype := .Float32, device := device }
  deriving ToTensorStructSchema

structure FlattenRecord (size : Nat) where
  value : Tensor { shape := #[size], dtype := .Float32, device := device }
  deriving TensorStructFlatten

structure DeviceBlock (size : Nat) (dtype : DType) where
  weight : Tensor { shape := #[size], dtype, device := device }
  label : Static String
  deriving Model

structure DeviceModel (size : Nat) (dtype : DType) where
  blocks : Array (DeviceBlock (device := device) size dtype)
  deriving Model

derive_tensor_tree_cast {source target : Device} {size : Nat} :
  DeviceBlock (device := source) size .Float32 =>
  DeviceBlock (device := target) size .Float64

derive_tensor_tree_cast {source target : Device} {size : Nat} :
  DeviceModel (device := source) size .Float32 =>
  DeviceModel (device := target) size .Float64

example {size : Nat} : TensorStruct (MappedRecord (device := device) size) :=
  inferInstance

example {size : Nat} : ToTensorStructSchema (SchemaRecord (device := device) size) :=
  inferInstance

example {size : Nat} : TensorStructFlatten (FlattenRecord (device := device) size) :=
  inferInstance

-- Both source and destination placement remain in the generated instance type.
example {source target : Device} {size : Nat}
    (input : DeviceModel (device := source) size .Float32) :
    IO (DeviceModel (device := target) size .Float64) :=
  TensorStruct.castTo (DeviceModel (device := target) size .Float64) input

example {otherDevice : Device} (_output : DeviceBlock (device := otherDevice) 3 .Float64) : True := by
  fail_if_success
    have _ : Tensor { shape := #[3], dtype := .Float64, device := .CPU } :=
      _output.weight
  fail_if_success
    have _ : Tensor { shape := #[3], dtype := .Float32, device := otherDevice } :=
      _output.weight
  trivial

private def cpuModel (value : Float) : DeviceModel (device := .CPU) 3 .Float32 :=
  let weight := (Tensor.onesFor
    { shape := #[3], dtype := .Float32, device := .CPU }).mulScalar value
  { blocks := #[{ weight, label := ⟨"parameters"⟩ }] }

@[test] def testImplicitDeviceSchemaAndRebuild : IO Unit := do
  let original := cpuModel 2
  let mapped := TensorStruct.map (fun tensor => tensor.detach) original
  let schema := ToTensorStructSchema.schema mapped
  LeanTest.assertEqual schema.renderedLeafPaths #["blocks[0].weight", "blocks[0].label"]
  let replacement := cpuModel 7
  let leaves := TensorStructFlatten.flatten replacement
  LeanTest.assertEqual leaves.size 1
  let rebuilt ← match TensorStructFlatten.rebuildFrom mapped leaves with
    | .ok model => pure model
    | .error message => LeanTest.fail message
  let some rebuiltBlock := rebuilt.blocks[0]? | LeanTest.fail "Missing rebuilt block"
  let some originalBlock := original.blocks[0]? | LeanTest.fail "Missing source block"
  LeanTest.assertEqual (← rebuiltBlock.weight.toFloatArray) #[7, 7, 7]
  LeanTest.assertEqual rebuiltBlock.label.val "parameters"
  LeanTest.assertTrue (rebuiltBlock.weight.device == .CPU)
  LeanTest.assertEqual (← originalBlock.weight.toFloatArray) #[2, 2, 2]

@[test] def testParameterizedDeviceTreeCast : IO Unit := do
  let original := cpuModel 2
  let converted ← TensorStruct.castTo (DeviceModel (device := .CPU) 3 .Float64) original
  LeanTest.assertEqual converted.blocks.size 1
  let some convertedBlock := converted.blocks[0]? | LeanTest.fail "Missing converted block"
  let some originalBlock := original.blocks[0]? | LeanTest.fail "Missing source block"
  LeanTest.assertEqual convertedBlock.weight.raw.dtype .Float64
  LeanTest.assertTrue (convertedBlock.weight.device == .CPU)
  LeanTest.assertEqual (← convertedBlock.weight.toFloatArray) #[2, 2, 2]
  LeanTest.assertEqual convertedBlock.label.val "parameters"
  LeanTest.assertEqual originalBlock.weight.raw.dtype .Float32

@[test] def testGeneratedCheckpointDevice : IO Unit := do
  let src ← safetensors.Source.open "Tests/fixtures/safetensors/single.safetensors"
  let weights ← SingleSafe.Weights.load src "" .CPU
  LeanTest.assertTrue (weights.linear.weight.device == .CPU)
  LeanTest.assertEqual (← weights.linear.weight.toFloatArray) #[1, 2, 3, 4, 5, 6]

end Tests.TensorDeviceDeriving
