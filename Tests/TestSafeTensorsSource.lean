import LeanTest
import Tests.Assertions
import Tyr.SafeTensors.Source
import Tyr.Model.Qwen.Weights

namespace Tests.SafeTensorsSource

open torch
open torch.safetensors (Source)

-- Stored dtype is a witnessed index, not a shape-only tensor or implicit cast.
example (src : Source) (name : String) (shape : Shape) (device : Device) :
    IO (SomeDTypeTensor shape device) :=
  src.load name shape device

example (_packed : SomeDTypeTensor #[2] .CPU) : True := by
  fail_if_success
    have _ : Tensor ⟨#[2], .Float32, .CPU⟩ := _packed.2
  fail_if_success
    have _ : Tensor ⟨#[2], .Float32, .CPU⟩ := _packed
  trivial

private def writeIndex (dir : System.FilePath) (weightMap : List (String × String)) : IO Unit :=
  IO.FS.writeFile (dir / "model.safetensors.index.json")
    (Lean.Json.mkObj [("weight_map",
      Lean.Json.mkObj (weightMap.map fun (name, file) => (name, Lean.Json.str file)))]).compress

@[test] def testSourceStoredDTypeAndExplicitConversion : IO Unit :=
    IO.FS.withTempDir fun dir => do
  let source := Tensor.fromFloatArray #[1.5, -2.0]
  for dtype in #[DType.BFloat16, .Float32] do
    let path := (dir / s!"{dtype.canonicalName}.safetensors").toString
    let stored := source.castDType dtype
    stored.saveSafeTensor path "weight"
    let packed ← (← Source.open path).load "weight" #[2]
    LeanTest.assertEqual packed.1 dtype
    LeanTest.assertEqual packed.2.dtype dtype
    LeanTest.assertEqual packed.2.raw.dtype dtype
    let converted : Tensor ⟨#[2], .Float32, .CPU⟩ ←
      pure (packed.2.castDType .Float32)
    LeanTest.assertEqual (← converted.toFloatArray) #[1.5, -2.0]
    LeanTest.assertEqual packed.2.raw.dtype dtype

@[test] def testSourceRejectsStoredLayoutMismatch : IO Unit :=
    IO.FS.withTempDir fun dir => do
  let path := (dir / "layout.safetensors").toString
  let source := (Tensor.fromFloatArray #[1, 2, 3, 4, 5, 6]).reshape #[2, 3]
  source.saveSafeTensor path "weight"
  writeIndex dir [("weight", "layout.safetensors")]
  for src in #[← Source.open path, ← Source.open dir.toString] do
    Tests.assertThrows (src.load "weight" #[3, 2]) (some "Expected shape")
    Tests.assertThrows (src.load? "weight" #[3, 2]) (some "Expected shape")
    Tests.assertThrows (src.load "weight" #[6]) (some "Expected shape")
    Tests.assertThrows (src.loadAs? "weight" ⟨#[6], .Float32⟩) (some "Expected shape")

@[test] def testSourceRejectsStoredDTypeMismatch : IO Unit :=
    IO.FS.withTempDir fun dir => do
  let path := (dir / "weights.safetensors").toString
  (Tensor.fromFloatArray #[2, 3]).saveSafeTensor path "weight"
  let src ← Source.open path
  Tests.assertThrows (src.loadAs "weight" ⟨#[2], .BFloat16⟩) (some "Expected dtype")
  Tests.assertThrows (src.loadAs? "weight" ⟨#[2], .BFloat16⟩) (some "Expected dtype")
  let loaded ← src.loadAs "weight" ⟨#[2], .Float32⟩
  LeanTest.assertEqual (← loaded.toFloatArray) #[2, 3]

/-- Scalars are stored with shape `[]` (e.g. NVFP4 global scales) and load as such. -/
@[test] def testSourceLoadsScalars : IO Unit :=
    IO.FS.withTempDir fun dir => do
  let path := (dir / "scalar.safetensors").toString
  (Tensor.fromFloatArray #[0.5]).reshape #[] |>.saveSafeTensor path "global_scale"
  let src ← Source.open path
  LeanTest.assertTrue (src.spec? "global_scale" == some ⟨#[], .Float32⟩)
    "a scalar is stored with shape []"
  let scalar ← src.loadAs "global_scale" ⟨#[], .Float32⟩
  LeanTest.assertEqual (← (scalar.reshape #[1]).toFloatArray) #[0.5]
  Tests.assertThrows (src.load "global_scale" #[1]) (some "Expected shape")

@[test] def testSourceOptionalMissingAndReadErrors : IO Unit :=
    IO.FS.withTempDir fun dir => do
  let path := (dir / "weights.safetensors").toString
  (Tensor.fromFloatArray #[2, 3]).saveSafeTensor path "weight"
  writeIndex dir [("weight", "weights.safetensors")]
  LeanTest.assertTrue (← (← Source.open path).load? "absent" #[2]).isNone
  LeanTest.assertTrue (← (← Source.open path).loadAs? "absent" ⟨#[2], .Float32⟩).isNone
  let sharded ← Source.open dir.toString
  LeanTest.assertTrue (← sharded.load? "absent" #[2]).isNone
  let some packed ← sharded.load? "weight" #[2] | throw (IO.userError "expected stored weight")
  LeanTest.assertEqual packed.1 .Float32
  LeanTest.assertEqual (← packed.2.toFloatArray) #[2, 3]
  Tests.assertThrows (sharded.load "absent" #[2]) (some "not found")
  Tests.assertThrows (Source.open (dir / "absent.safetensors").toString)
    (some "no such file or directory")
  Tests.assertThrows (Source.open (dir / "absent").toString)

/-- A directory opens the shards its index lists, or else its `model.safetensors`;
other `.safetensors` files in it are never read. -/
@[test] def testSourceDirectoryLayouts : IO Unit := do
  IO.FS.withTempDir fun dir => do
    (Tensor.fromFloatArray #[1]).saveSafeTensor (dir / "model-1.safetensors").toString "a"
    (Tensor.fromFloatArray #[2]).saveSafeTensor (dir / "model-2.safetensors").toString "b"
    (Tensor.fromFloatArray #[3]).saveSafeTensor (dir / "reference.safetensors").toString "c"
    Tests.assertThrows (Source.open dir.toString) (some "No model.safetensors.index.json")
    writeIndex dir [("a", "model-1.safetensors"), ("b", "model-2.safetensors")]
    let src ← Source.open dir.toString
    LeanTest.assertEqual (← (← src.loadAs "a" ⟨#[1], .Float32⟩).toFloatArray) #[1]
    LeanTest.assertEqual (← (← src.loadAs "b" ⟨#[1], .Float32⟩).toFloatArray) #[2]
    LeanTest.assertTrue (!src.contains "c") "files missing from the index are not shards"
  IO.FS.withTempDir fun dir => do
    (Tensor.fromFloatArray #[1]).saveSafeTensor (dir / "model.safetensors").toString "a"
    (Tensor.fromFloatArray #[3]).saveSafeTensor (dir / "reference.safetensors").toString "c"
    let src ← Source.open dir.toString
    LeanTest.assertTrue (src.contains "a" && !src.contains "c")
      "without an index a directory is its model.safetensors"
  IO.FS.withTempDir fun dir => do
    (Tensor.fromFloatArray #[1]).saveSafeTensor (dir / "model-1.safetensors").toString "a"
    writeIndex dir [("b", "model-1.safetensors")]
    Tests.assertThrows (Source.open dir.toString) (some "not found in shard")

@[test] def testSourceCandidateNames : IO Unit :=
    IO.FS.withTempDir fun dir => do
  let path := (dir / "weights.safetensors").toString
  (Tensor.fromFloatArray #[2, 3]).saveSafeTensor path "model.weight"
  let src ← Source.open path
  LeanTest.assertEqual (src.find? #["weight", "model.weight"]) (some "model.weight")
  LeanTest.assertEqual (src.find? #["weight", "other"]) none
  LeanTest.assertEqual (← src.resolve #["weight", "model.weight"]) "model.weight"
  Tests.assertThrows (src.resolve #["weight", "other"]) (some "none of")

@[test] def testQwenSourceRetainsExplicitFloat32Policy : IO Unit :=
    IO.FS.withTempDir fun dir => do
  let path := (dir / "model.safetensors").toString
  let source := (Tensor.fromFloatArray #[1.5, -2.0]).castDType .BFloat16
  source.saveSafeTensor path "model.norm.weight"
  for src in #[← Source.open path, ← Source.open dir.toString] do
    let norm ← qwen.loadRMSNorm src "model.norm" 2
    LeanTest.assertEqual norm.weight.raw.dtype .Float32
    LeanTest.assertEqual (← norm.weight.toFloatArray) #[1.5, -2.0]

@[test] def testExactDTypeFP8DequantizationKeepsIndependentScaleDType : IO Unit := do
  let weight := (Tensor.ones #[128, 128]).castDType .Float8E4M3FN
  let scale := (Tensor.fromFloatArray #[2]).reshape #[1, 1]
  let result : Tensor ⟨#[128, 128], .Float32, .CPU⟩ := Model.dequantizeFP8 weight scale (by decide)
  LeanTest.assertEqual (Tensor.sumAll result).item 32768
  LeanTest.assertEqual weight.raw.dtype .Float8E4M3FN
  LeanTest.assertEqual scale.raw.dtype .Float32

end Tests.SafeTensorsSource
