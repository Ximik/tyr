/-
  Tests/TestSafeTensorsTypeProvider.lean

  Coverage for:
  - runtime safetensors schema introspection
  - elaboration-time `safetensors_type_provider`
-/
import Tyr.SafeTensors
import LeanTest

open torch

safetensors_type_provider "Tests/fixtures/safetensors/single.safetensors" as SingleSafe
safetensors_type_provider "Tests/fixtures/safetensors/sharded" as ShardedSafe
safetensors_type_provider "Tests/fixtures/safetensors/indexed.safetensors" as IndexedSafe
safetensors_type_provider "Tests/fixtures/safetensors/indexed_dir" as IndexedDirSafe
safetensors_type_provider "Tests/fixtures/safetensors/provider_errors/nonuniform.safetensors" as NonUniformSafe

-- Default loaders and aggregate records must preserve the generated indices.
example : IO (DTensor #[2, 3] .Float32) := SingleSafe.load_linear_weight
example (_weights : SingleSafe.Weights) : True := by
  fail_if_success
    have _badShape : DTensor #[2, 4] .Float32 := _weights.linear.weight
  fail_if_success
    have _badDtype : DTensor #[2, 3] .Int64 := _weights.linear.weight
  trivial

private def writeBinCopy (src dst : String) : IO Unit := do
  let bytes ← IO.FS.readBinFile src
  IO.FS.writeBinFile ⟨dst⟩ bytes

private def writeIndexDir (name indexJson : String) : IO String := do
  let dir := s!"/tmp/tyr_safetensors_{name}"
  IO.FS.createDirAll ⟨dir⟩
  IO.FS.writeFile s!"{dir}/model.safetensors.index.json" indexJson
  pure dir

private def tensorValues (t : RawTensor) : Array Float :=
  t.getValues 1024 |>.toList.toArray

private def headerPrefix (size : UInt64) : ByteArray := Id.run do
  let mut result := ByteArray.empty
  for i in [:8] do
    result := result.push ((size >>> (i * 8).toUInt64).toUInt8)
  pure result

@[test]
def testSafeTensorsHeaderReadBounds : IO Unit := IO.FS.withTempDir fun dir => do
  let path := (dir / "bounded.safetensors").toString
  IO.FS.writeBinFile path (ByteArray.mk #[1, 2, 3])
  LeanTest.assertThrows (safetensors.introspect path) (some "missing 8-byte header size")
  IO.FS.writeBinFile path (headerPrefix 4096)
  LeanTest.assertThrows (safetensors.introspect path) (some "header exceeds file size")
  IO.FS.writeBinFile path (headerPrefix (safetensors.maxHeaderBytes + 1).toUInt64)
  LeanTest.assertThrows (safetensors.introspect path) (some "byte limit")
  let header := "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}".toUTF8
  -- Payload bytes are deliberately not UTF-8 and must not enter JSON parsing.
  IO.FS.writeBinFile path (headerPrefix header.size.toUInt64 ++ header ++ ByteArray.mk #[255, 254, 253, 252])
  let schema ← safetensors.introspect path
  LeanTest.assertEqual schema.tensors.size 1
  LeanTest.assertTrue ((schema.find? "x").isSome)

@[test]
def testSafeTensorsIntrospectionSingle : IO Unit := do
  let schema ← safetensors.introspect "Tests/fixtures/safetensors/single.safetensors"
  LeanTest.assertEqual schema.sourceIsDirectory false "single source should not be a directory"
  LeanTest.assertEqual schema.tensors.size 1 "single fixture should have one tensor"

  let some tensor := schema.find? "linear.weight"
    | throw <| IO.userError "expected tensor 'linear.weight'"
  LeanTest.assertEqual tensor.dtype DType.Float32 "dtype should be parsed into core DType"
  LeanTest.assertTrue (tensor.shape == #[2, 3]) "shape should match safetensors header"
  LeanTest.assertEqual tensor.sourceFile "" "single-file introspection should use empty sourceFile"

@[test]
def testSafeTensorsIntrospectionSharded : IO Unit := do
  let schema ← safetensors.introspect "Tests/fixtures/safetensors/sharded"
  LeanTest.assertEqual schema.sourceIsDirectory true "sharded source should be a directory"
  LeanTest.assertEqual schema.tensors.size 2 "sharded fixture should have two tensors"

  let some embed := schema.find? "embed.weight"
    | throw <| IO.userError "expected tensor 'embed.weight'"
  LeanTest.assertEqual embed.sourceFile "part1.safetensors" "embed tensor should map to part1 shard"
  LeanTest.assertTrue (embed.shape == #[2, 2]) "embed tensor shape should match header"

  let some bias := schema.find? "proj.bias"
    | throw <| IO.userError "expected tensor 'proj.bias'"
  LeanTest.assertEqual bias.sourceFile "part2.safetensors" "bias tensor should map to part2 shard"
  LeanTest.assertTrue (bias.shape == #[3]) "bias tensor shape should match header"

@[test]
def testSafeTensorsIntrospectionShardedIndexJson : IO Unit := do
  let schema ← safetensors.introspect "Tests/fixtures/safetensors/indexed_dir"
  LeanTest.assertEqual schema.sourceIsDirectory true "indexed sharded source should be a directory"
  LeanTest.assertEqual schema.tensors.size 2 "weight_map should define exactly two tensors"

  let some embed := schema.find? "embed.weight"
    | throw <| IO.userError "expected tensor 'embed.weight'"
  LeanTest.assertEqual embed.sourceFile "part1.safetensors" "embed tensor should map through index json"

  let some bias := schema.find? "proj.bias"
    | throw <| IO.userError "expected tensor 'proj.bias'"
  LeanTest.assertEqual bias.sourceFile "part2.safetensors" "proj.bias tensor should map through index json"

  let unmapped := schema.find? "linear.weight"
  LeanTest.assertTrue unmapped.isNone "tensors from unmapped shard files should not be exposed"

@[test]
def testSafeTensorsIntrospectionIndexValidationErrors : IO Unit := do
  let missingWeightMapDir ← writeIndexDir
    "index_missing_weight_map"
    "{\"metadata\":{}}"
  LeanTest.assertThrows
    (safetensors.introspect missingWeightMapDir)
    (some "missing 'weight_map'")

  let nonObjectWeightMapDir ← writeIndexDir
    "index_non_object_weight_map"
    "{\"weight_map\":[]}"
  LeanTest.assertThrows
    (safetensors.introspect nonObjectWeightMapDir)
    (some "'weight_map' must be a JSON object")

  let nonStringWeightMapDir ← writeIndexDir
    "index_non_string_weight_map"
    "{\"weight_map\":{\"embed.weight\":123}}"
  LeanTest.assertThrows
    (safetensors.introspect nonStringWeightMapDir)
    (some "has non-string shard filename")

  let missingShardDir ← writeIndexDir
    "index_missing_shard"
    "{\"weight_map\":{\"embed.weight\":\"missing-shard-does-not-exist.safetensors\"}}"
  LeanTest.assertThrows
    (safetensors.introspect missingShardDir)
    (some "referenced shard does not exist")

  let missingTensorDir ← writeIndexDir
    "index_missing_tensor"
    "{\"weight_map\":{\"proj.bias\":\"part1.safetensors\"}}"
  writeBinCopy
    "Tests/fixtures/safetensors/sharded/part1.safetensors"
    s!"{missingTensorDir}/part1.safetensors"
  LeanTest.assertThrows
    (safetensors.introspect missingTensorDir)
    (some "tensor 'proj.bias' not found in shard 'part1.safetensors'")

@[test]
def testSafeTensorsIntrospectionIndexPathSafety : IO Unit := do
  let traversalDir ← writeIndexDir
    "index_unsafe_traversal"
    "{\"weight_map\":{\"embed.weight\":\"../part1.safetensors\"}}"
  LeanTest.assertThrows
    (safetensors.introspect traversalDir)
    (some "unsafe shard path")

  let absoluteDir ← writeIndexDir
    "index_unsafe_absolute"
    "{\"weight_map\":{\"embed.weight\":\"/tmp/evil.safetensors\"}}"
  LeanTest.assertThrows
    (safetensors.introspect absoluteDir)
    (some "unsafe shard path")

@[test]
def testSafeTensorsTypeProviderSingle : IO Unit := do
  LeanTest.assertEqual SingleSafe.sourceIsDirectory false "provider should detect single-file source"
  LeanTest.assertEqual SingleSafe.tensorCount 1 "provider tensor count should match fixture"
  LeanTest.assertTrue (SingleSafe.hasTensor "linear.weight") "provider should include linear.weight"
  LeanTest.assertEqual SingleSafe.fieldToTensorName [("linear_weight", "linear.weight")]
    "field map should expose generated field names for aggregate record"

  let t ← SingleSafe.load_linear_weight
  LeanTest.assertTrue (t.runtimeShape == #[2, 3]) "typed loader should enforce generated shape"
  LeanTest.assertEqual SingleSafe.linear_weightSpec.dtype DType.Float32
    "generated spec should retain typed core DType"
  LeanTest.assertTrue
    (SingleSafe.linear_weightTensorSpec == { shape := #[2, 3], dtype := DType.Float32 })
    "generated TensorSpec should expose shape and dtype through the shared spec layer"
  let typed ← SingleSafe.load_linear_weightTyped
  LeanTest.assertTrue (Tensor.actualSpec typed == SingleSafe.linear_weightTensorSpec)
    "checked typed loader should validate runtime metadata before returning DTensor"
  LeanTest.assertEqual (Tensor.dtype typed) DType.Float32
    "checked typed loader should carry dtype in the return type"
  let handle ← safetensors.openHandle "Tests/fixtures/safetensors/single.safetensors"
  let tFromHandle ← SingleSafe.load_linear_weightFromHandle handle
  LeanTest.assertTrue (tFromHandle.runtimeShape == #[2, 3])
    "single-file provider should expose a per-tensor from-handle loader"
  let typedFromHandle ← SingleSafe.load_linear_weightTypedFromHandle handle
  LeanTest.assertTrue (Tensor.actualSpec typedFromHandle == SingleSafe.linear_weightTensorSpec)
    "single-file provider should expose a checked per-tensor from-handle loader"

  let wrongDTypeSpec : TensorSpec := { SingleSafe.linear_weightTensorSpec with dtype := DType.BFloat16 }
  LeanTest.assertThrows
    (safetensors.loadTensorWithSpec
      "Tests/fixtures/safetensors/single.safetensors"
      "linear.weight"
      wrongDTypeSpec)
    (some "Expected dtype")
  let wrongDeviceContract : TensorContract := {
    spec := SingleSafe.linear_weightTensorSpec
    role := .parameter
    devicePolicy := .exact Device.MPS
  }
  LeanTest.assertThrows
    (safetensors.loadTensorWithContract
      "Tests/fixtures/safetensors/single.safetensors"
      "linear.weight"
      wrongDeviceContract
      Device.CPU)
    (some "Expected device")

  let weights ← SingleSafe.loadAll
  LeanTest.assertTrue (weights.linear.weight.runtimeShape == #[2, 3])
    "hierarchical aggregate typed record should expose nested typed tensor fields"
  let weightsFromHandle ← SingleSafe.loadAllFromHandle handle
  LeanTest.assertTrue (weightsFromHandle.linear.weight.runtimeShape == #[2, 3])
    "single-file provider should expose a handle-reusing loadAll"
  let linear ← SingleSafe.linear.load
  LeanTest.assertTrue (linear.weight.runtimeShape == #[2, 3])
    "hierarchical namespace loader should expose subtree load"
  let linearFromHandle ← SingleSafe.linear.loadFromHandle handle
  LeanTest.assertTrue (linearFromHandle.weight.runtimeShape == #[2, 3])
    "single-file provider should expose handle-reusing subtree loads"

@[test]
def testSafeTensorsTypeProviderSharded : IO Unit := do
  LeanTest.assertEqual ShardedSafe.sourceIsDirectory true "provider should detect sharded directory source"
  LeanTest.assertEqual ShardedSafe.tensorCount 2 "provider tensor count should match sharded fixture"
  LeanTest.assertTrue (ShardedSafe.hasTensor "embed.weight") "provider should include embed.weight"
  LeanTest.assertTrue (ShardedSafe.hasTensor "proj.bias") "provider should include proj.bias"

  let e ← ShardedSafe.load_embed_weight
  LeanTest.assertTrue (e.runtimeShape == #[2, 2]) "embed loader should return generated typed shape"
  let eTyped ← ShardedSafe.load_embed_weightTyped
  LeanTest.assertTrue (Tensor.actualSpec eTyped == ShardedSafe.embed_weightTensorSpec)
    "sharded provider should expose checked typed loaders"

  let b ← ShardedSafe.load_proj_bias
  LeanTest.assertTrue (b.runtimeShape == #[3]) "bias loader should return generated typed shape"
  LeanTest.assertEqual ShardedSafe.proj_biasSpec.sourceFile "part2.safetensors"
    "generated tensor spec should expose shard source file"

  let weights ← ShardedSafe.loadAll
  LeanTest.assertTrue (weights.embed.weight.runtimeShape == #[2, 2])
    "hierarchical aggregate typed record should expose nested embed tensor"
  LeanTest.assertTrue (weights.proj.bias.runtimeShape == #[3])
    "hierarchical aggregate typed record should expose nested bias tensor"
  let embed ← ShardedSafe.embed.load
  LeanTest.assertTrue (embed.weight.runtimeShape == #[2, 2])
    "hierarchical namespace loader should work for sharded subtree"

@[test]
def testSafeTensorsTypeProviderIndexedHierarchy : IO Unit := do
  LeanTest.assertEqual IndexedSafe.tensorCount 2 "indexed fixture should have two tensors"
  let direct0 ← IndexedSafe.load_layers_0_weight
  let direct1 ← IndexedSafe.load_layers_1_weight
  let directVals0 := tensorValues direct0.raw
  let directVals1 := tensorValues direct1.raw
  LeanTest.assertTrue (directVals0 != directVals1)
    "indexed fixture should use distinct tensor values across numeric siblings"
  let weights ← IndexedSafe.loadAll
  LeanTest.assertEqual weights.layers.size 2 "numeric path segments should produce an indexed collection"
  let some layer0 := weights.layers[0]?
    | throw <| IO.userError "expected first indexed layer"
  let some layer1 := weights.layers[1]?
    | throw <| IO.userError "expected second indexed layer"
  LeanTest.assertTrue (layer0.weight.runtimeShape == #[2])
    "first indexed subtree should expose typed nested tensor"
  LeanTest.assertTrue (layer1.weight.runtimeShape == #[2])
    "second indexed subtree should expose typed nested tensor"
  LeanTest.assertTrue (tensorValues layer0.weight.raw == directVals0)
    "loadAll should preserve the first indexed tensor values"
  LeanTest.assertTrue (tensorValues layer1.weight.raw == directVals1)
    "loadAll should preserve later indexed tensor values instead of repeating the first"
  let handle ← safetensors.openHandle "Tests/fixtures/safetensors/indexed.safetensors"
  let weightsFromHandle ← IndexedSafe.loadAllFromHandle handle
  let some handleLayer0 := weightsFromHandle.layers[0]?
    | throw <| IO.userError "expected first indexed layer from loadAllFromHandle"
  let some handleLayer1 := weightsFromHandle.layers[1]?
    | throw <| IO.userError "expected second indexed layer from loadAllFromHandle"
  LeanTest.assertTrue (tensorValues handleLayer0.weight.raw == directVals0)
    "loadAllFromHandle should preserve the first indexed tensor values"
  LeanTest.assertTrue (tensorValues handleLayer1.weight.raw == directVals1)
    "loadAllFromHandle should preserve later indexed tensor values"
  let layers ← IndexedSafe.layers.load
  LeanTest.assertEqual layers.size 2 "hierarchical namespace loader should load array subtree"
  let some nsLayer0 := layers[0]?
    | throw <| IO.userError "expected first indexed layer from namespace loader"
  let some nsLayer1 := layers[1]?
    | throw <| IO.userError "expected second indexed layer from namespace loader"
  LeanTest.assertTrue (tensorValues nsLayer0.weight.raw == directVals0)
    "namespace array loader should preserve the first indexed tensor values"
  LeanTest.assertTrue (tensorValues nsLayer1.weight.raw == directVals1)
    "namespace array loader should preserve later indexed tensor values"
  let layersFromHandle ← IndexedSafe.layers.loadFromHandle handle
  let some nsHandleLayer0 := layersFromHandle[0]?
    | throw <| IO.userError "expected first indexed layer from namespace handle loader"
  let some nsHandleLayer1 := layersFromHandle[1]?
    | throw <| IO.userError "expected second indexed layer from namespace handle loader"
  LeanTest.assertTrue (tensorValues nsHandleLayer0.weight.raw == directVals0)
    "namespace handle loader should preserve the first indexed tensor values"
  LeanTest.assertTrue (tensorValues nsHandleLayer1.weight.raw == directVals1)
    "namespace handle loader should preserve later indexed tensor values"

@[test]
def testSafeTensorsTypeProviderShardedIndexJson : IO Unit := do
  LeanTest.assertEqual IndexedDirSafe.sourceIsDirectory true "provider should detect sharded index directory"
  LeanTest.assertEqual IndexedDirSafe.tensorCount 2 "provider should expose only index-mapped tensors"
  LeanTest.assertTrue (IndexedDirSafe.hasTensor "embed.weight")
    "index-mapped tensor should exist"
  LeanTest.assertTrue (!(IndexedDirSafe.hasTensor "linear.weight"))
    "tensor from unmapped shard should not be generated"

  let weights ← IndexedDirSafe.loadAll
  LeanTest.assertTrue (weights.embed.weight.runtimeShape == #[2, 2])
    "index-backed loadAll should load embed tensor from mapped shard"
  LeanTest.assertTrue (weights.proj.bias.runtimeShape == #[3])
    "index-backed loadAll should load proj bias tensor from mapped shard"

@[test]
def testSafeTensorsTypeProviderNonUniformIndexedHierarchy : IO Unit := do
  LeanTest.assertEqual NonUniformSafe.tensorCount 2 "non-uniform fixture should have two tensors"
  let weights ← NonUniformSafe.loadAll
  LeanTest.assertTrue (weights.layers.i0.weight.runtimeShape == #[1])
    "non-uniform indexed subtree should fall back to named index fields"
  LeanTest.assertTrue (weights.layers.i1.weight.runtimeShape == #[2])
    "non-uniform indexed subtree should retain later numeric siblings"
  let layers ← NonUniformSafe.layers.load
  LeanTest.assertTrue (layers.i0.weight.runtimeShape == #[1])
    "hierarchical namespace loader should work for non-uniform subtree"

/-- The default generated loader enforces dtype, not just its header shape. -/
@[test]
def testSafeTensorsDefaultLoaderRejectsDtype : IO Unit := IO.FS.withTempDir fun dir => do
  let path := (dir / "wrong-dtype.safetensors").toString
  let header := "{\"linear.weight\":{\"dtype\":\"I32\",\"shape\":[2,3],\"data_offsets\":[0,24]}}".toUTF8
  IO.FS.writeBinFile path
    (headerPrefix header.size.toUInt64 ++ header ++ ByteArray.mk (Array.replicate 24 0))
  LeanTest.assertThrows (SingleSafe.load_linear_weight path) (some "dtype")
