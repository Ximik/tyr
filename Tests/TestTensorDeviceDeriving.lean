import LeanTest
import Tyr.Module.Derive
import Tyr.TensorStruct.DeriveCast
import Tyr.SafeTensors.TypeProvider

namespace Tests.TensorDeviceDeriving

open torch

safetensors_type_provider "Tests/fixtures/safetensors/single.safetensors" as DeviceCheckpoint

-- A runtime-selected device is shared by generated records and every loader.
example (device : Device) : IO (DeviceCheckpoint.Weights device) :=
  DeviceCheckpoint.loadAll (device := device)

example (device : Device) :
    IO (Tensor { shape := #[2, 3], dtype := .Float32, device }) :=
  DeviceCheckpoint.load_linear_weightTypedOnDevice (device := device)

example (device : Device) (weights : DeviceCheckpoint.Weights device) :
    Tensor { shape := #[2, 3], dtype := .Float32, device } :=
  weights.linear.weight

variable {device : Device}

-- Each handler must apply both the implicit device and explicit size correctly.
structure MappedRecord (size : UInt64) where
  value : Tensor { shape := #[size], dtype := .Float32, device := device }
  deriving TensorStruct

structure SchemaRecord (size : UInt64) where
  value : Tensor { shape := #[size], dtype := .Float32, device := device }
  deriving ToTensorStructSchema

structure FlattenRecord (size : UInt64) where
  value : Tensor { shape := #[size], dtype := .Float32, device := device }
  deriving TensorStructFlatten

structure DeviceBlock (size : UInt64) (dtype : DType) where
  weight : Tensor { shape := #[size], dtype, device := device }
  label : Static String
  deriving Model

structure DeviceModel (size : UInt64) (dtype : DType) where
  blocks : Array (DeviceBlock (device := device) size dtype)
  deriving Model

derive_tensor_tree_cast {source target : Device} {size : UInt64} :
  DeviceBlock (device := source) size .Float32 =>
  DeviceBlock (device := target) size .Float64

derive_tensor_tree_cast {source target : Device} {size : UInt64} :
  DeviceModel (device := source) size .Float32 =>
  DeviceModel (device := target) size .Float64

example {size : UInt64} : TensorStruct (MappedRecord (device := device) size) :=
  inferInstance

example {size : UInt64} : ToTensorStructSchema (SchemaRecord (device := device) size) :=
  inferInstance

example {size : UInt64} : TensorStructFlatten (FlattenRecord (device := device) size) :=
  inferInstance

-- Both source and destination placement remain in the generated instance type.
example {source target : Device} {size : UInt64}
    (input : DeviceModel (device := source) size .Float32) :
    IO (DeviceModel (device := target) size .Float64) :=
  TensorStruct.castTo (DeviceModel (device := target) size .Float64) input

example (_output : DeviceBlock (device := .CUDA 1) 3 .Float64) : True := by
  fail_if_success
    have _ : Tensor { shape := #[3], dtype := .Float64, device := .CPU } :=
      _output.weight
  fail_if_success
    have _ : Tensor { shape := #[3], dtype := .Float32, device := (.CUDA 1) } :=
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
  let weights ← DeviceCheckpoint.loadAll (device := .CPU)
  LeanTest.assertTrue (weights.linear.weight.device == .CPU)
  LeanTest.assertEqual (← weights.linear.weight.toFloatArray) #[1, 2, 3, 4, 5, 6]

end Tests.TensorDeviceDeriving
