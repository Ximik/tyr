import Tyr.SafeTensors.Schema
import Tyr.Model.Qwen35
import LeanTest

open torch
open torch.qwen35

private def ensureDir (path : String) : IO Unit :=
  IO.FS.createDirAll ⟨path⟩

@[test]
def testQwen35HubShardFilesFromIndex : IO Unit := do
  let dir := "/tmp/qwen35_hub_index_test"
  ensureDir dir
  let indexPath := s!"{dir}/model.safetensors.index.json"
  let json :=
    "{\"metadata\":{},\"weight_map\":{" ++
    "\"model.embed_tokens.weight\":\"model-00001-of-00002.safetensors\"," ++
    "\"model.layers.0.self_attn.q_proj.weight\":\"model-00002-of-00002.safetensors\"," ++
    "\"model.layers.0.self_attn.k_proj.weight\":\"model-00001-of-00002.safetensors\"}}"
  IO.FS.writeFile indexPath json

  let shards ← hub.shardFilesFromIndexFile indexPath
  LeanTest.assertEqual shards.size 2 "index parser should return unique shard names"
  LeanTest.assertTrue (shards.contains "model-00001-of-00002.safetensors")
    "first shard should be present"
  LeanTest.assertTrue (shards.contains "model-00002-of-00002.safetensors")
    "second shard should be present"

@[test]
def testQwen35HubDetectWeightLayout : IO Unit := do
  let shardedDir := "/tmp/qwen35_hub_layout_sharded"
  ensureDir shardedDir
  IO.FS.writeFile s!"{shardedDir}/model.safetensors.index.json" "{\"metadata\":{},\"weight_map\":{}}"
  let isSharded ← hub.detectWeightLayout shardedDir
  LeanTest.assertEqual isSharded true "index file should detect sharded layout"

  let singleDir := "/tmp/qwen35_hub_layout_single"
  ensureDir singleDir
  IO.FS.writeFile s!"{singleDir}/model.safetensors" ""
  let isShardedSingle ← hub.detectWeightLayout singleDir
  LeanTest.assertEqual isShardedSingle false "single safetensors file should detect single layout"

@[test]
def testQwen35HubResolveLocalPretrainedDir : IO Unit := do
  let localDir := "/tmp/qwen35_hub_local_dir"
  ensureDir localDir
  IO.FS.writeFile s!"{localDir}/config.json" "{}"
  IO.FS.writeFile s!"{localDir}/model.safetensors" ""

  let resolved ← hub.resolvePretrainedDir localDir
  LeanTest.assertEqual resolved localDir "existing local model directory should resolve without network"

@[test]
def testQwen35CollectionListCoverage : IO Unit := do
  LeanTest.assertTrue (hub.isQwen35CollectionRepoId "Qwen/Qwen3.5-0.8B")
    "collection list should include 0.8B instruct variant"
  LeanTest.assertTrue (hub.isQwen35CollectionRepoId "Qwen/Qwen3.5-0.8B-Base")
    "collection list should include 0.8B base variant"
  LeanTest.assertTrue (hub.isQwen35CollectionRepoId "Qwen/Qwen3.5-397B-A17B")
    "collection list should include 397B"
  LeanTest.assertTrue (hub.isQwen35CollectionRepoId "Qwen/Qwen3.5-35B-A3B-Base")
    "collection list should include 35B base variant"
  LeanTest.assertTrue (hub.isQwen35CollectionRepoId "Qwen/Qwen3.5-27B-FP8")
    "collection list should include 27B FP8 variant"
  LeanTest.assertTrue (hub.isQwen35CollectionRepoId "Qwen/Qwen3.6-35B-A3B")
    "shared Qwen3.5 resolver should include the supported Qwen3.6-35B-A3B checkpoint"

@[test]
def testSafeTensorsLoadShardedHfSingleShardPattern : IO Unit := do
  let dir := "/tmp/qwen35_hf_single_shard_pattern"
  ensureDir dir
  let shardPath := s!"{dir}/model.safetensors-00001-of-00001.safetensors"
  let input : DTensor #[2, 2] .Int64 := (Tensor.fromInt64Array #[1, 2, 3, 4]).reshapeChecked #[2, 2]
  Tensor.saveSafeTensor input shardPath "model.language_model.embed_tokens.weight"
  let loaded : DTensor #[2, 2] .Int64 ←
    safetensors.loadTensorShardedWithSpec dir "model.language_model.embed_tokens.weight" ⟨#[2, 2], .Int64⟩
  let vals ← loaded.toUInt64Array
  LeanTest.assertEqual vals #[1, 2, 3, 4]
    "sharded loader should discover HuggingFace-style single-shard filenames"
