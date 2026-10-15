# Weights and serialization

## Purpose and when to use

Tyr has three cooperating serialization subsystems, each aimed at a different job:

- `Tyr/SafeTensors/` reads HuggingFace `.safetensors` checkpoints — single files,
  sharded directories, and `model.safetensors.index.json` layouts — and can generate
  shape-typed Lean loaders from a checkpoint at elaboration time. Use this for
  pretrained weight loading.
- `Tyr/Hub.lean` resolves a local directory or a HuggingFace `repo_id` to a local
  model directory, downloading and caching as needed. Use this as the front door
  for `loadFromPretrained`-style APIs.
- `Tyr/Checkpoint.lean` persists training state — any `TensorStruct` parameter tree
  plus mirrored Adam-style optimizer state — for resume. Use this for Tyr's own
  training checkpoints, not for HF interop.

Everything below lives in three namespaces: `torch.safetensors`, `torch.Hub`, and
`torch.checkpoint`.

## Architecture and main abstractions

### Native boundary and indexed loaders

`Tyr.RawTensor.SafeTensors` contains native single-file, open-handle, and sharded
loading adapters. They preserve stored dimensions rather than reshaping to the
caller's expected layout. `Tyr.SafeTensors.Schema` then checks shape, dtype, and
placement before returning an indexed tensor.

`torch.safetensors.openHandle` returns a `SafeTensorsHandle` for repeated reads.
Use `loadFromHandleWithSpec`, `loadTensorWithSpec`, or
`loadTensorShardedWithSpec` with a `TensorSpec`; each returns
`IO (Tensor { shape := spec.shape, dtype := spec.dtype, device := device })`. The result is indexed by the requested device, which defaults to CPU.
Contract variants also carry a role and optional expected-device metadata.

The native header `cc/src/tyr_safetensors_abi.h` and generated-C audit cover all
four loading entry points. `Tensor.saveSafeTensor tensor path name` writes one
tensor under a named key. Training snapshots use the separate checkpoint APIs.

### Runtime schema introspection

`Tyr/SafeTensors/Schema.lean` builds a metadata surface on top of the FFI. The two
central types (`Tyr/SafeTensors/Schema.lean:27-56`):

```lean
structure TensorSchema where
  name : String
  dtype : DType
  shape : Shape
  /-- For directory sources this is the shard filename; for single-file sources it is empty. -/
  sourceFile : String := ""

structure Schema where
  source : String
  sourceIsDirectory : Bool
  tensors : Array TensorSchema
```

`introspect (source : String) : IO Schema` (`Tyr/SafeTensors/Schema.lean:413`) is
the central entry point. It accepts:

- a single `.safetensors` file — parses the 8-byte little-endian header size and
  the JSON header itself;
- a directory of `.safetensors` shards — parses every shard header;
- a directory with `model.safetensors.index.json` — follows the `weight_map`,
  rejects unsafe shard paths (absolute paths, `\` prefixes, Windows drive letters,
  `.`/`..` segments; `Tyr/SafeTensors/Schema.lean:345-350`), and cross-checks every
  index entry against the actual shard headers. Only index-mapped tensors are
  exposed;
- a `.schema.json` snapshot — a previously saved `Schema`, loaded via
  `loadSnapshot` without touching any weight files.

Results are sorted by name and duplicate tensor names are rejected.
`saveSnapshot`/`loadSnapshot` persist a `Schema` as JSON; this is how
`Tyr/Model/KittenTTS/kokoro_v1.schema.json` lets the type provider run at build
time without a multi-GB checkpoint on disk.

The schema layer also provides contract-checked typed loaders that return a
dtype-indexed `Tensor` (see [Typed tensors](core/typed.md)):

```lean
def loadTensorWithContract (path : String) (name : String)
    (contract : TensorContract) (device : Device := Device.CPU)
    : IO (Tensor { shape := contract.spec.shape, dtype := contract.spec.dtype, device := device })
```

plus `loadFromHandleWithContract`, `loadTensorShardedWithContract`, `...WithSpec`
variants (which fix `expectedDevice := some device`), and `...WithSchema` variants
that take a discovered `TensorSchema`. Checking happens after the load via
`TensorContract.check` against the runtime shape/dtype/device
(`Tyr/SafeTensors/Schema.lean:62-82`).

### Try-load helpers

`Tyr/SafeTensors/Load.lean` is a thin error-handling layer over the FFI, used by
the per-model `Weights` modules (`Tyr/Model/Qwen35/Weights.lean`,
`Tyr/Model/Gemma4/Weights.lean`, etc.). The two flavors matter:

- `tryLoadTensor` / `tryLoadTensorSharded` return `none` on **any** failure —
  missing tensor, shape mismatch, unreadable file.
- `tryLoadOptionalTensor` / `tryLoadOptionalTensorSharded` return `none` only when
  the tensor is absent (detected by the `isMissingTensorError` message heuristic,
  `Tyr/SafeTensors/Load.lean:39-41`); other errors are rethrown.

On top of these sit candidate-name fallbacks — `tryLoadTensorCandidates`,
`tryLoadTensorShardedCandidates`, and the throwing `loadTensorCandidates` /
`loadTensorShardedCandidates` — which try an array of alternative names (model
families use `pushUnique` to build prefix-variant lists, e.g. with and without a
`language_model.` prefix).

### The `safetensors_type_provider` command

`Tyr/SafeTensors/TypeProvider.lean` defines a command that introspects a
checkpoint **at elaboration time** and generates typed loaders from it:

```lean
safetensors_type_provider "path/to/model.safetensors" as MyWeights
safetensors_type_provider "path/to/sharded_dir" as MyWeights
safetensors_type_provider "path/to/model.schema.json" as MyWeights  -- snapshot
```

For each tensor, inside `namespace MyWeights`, it emits (tensor `linear.weight`
shown; the declaration base name is sanitized and lowercased to `linear_weight`):

- `abbrev linear_weightShape : Shape` — the static shape;
- `def linear_weightSpec : TensorSchema` and `def linear_weightTensorSpec : TensorSpec`;
- `def load_linear_weight (source : String := defaultSource) (device : Device := .CPU) : IO (Tensor { shape := linear_weightShape, dtype := linear_weightTensorSpec.dtype, device := device })`;
- explicitly named checked variants `load_linear_weightTyped` (CPU) and
  `...TypedOnDevice` (the requested device);
- for single-file sources additionally `load_linear_weightFromHandle` and
  `load_linear_weightTypedFromHandle[OnDevice]` taking a `SafeTensorsHandle`.

Tensor names split on `.` form a hierarchy. Named segments become fields of
generated `structure Weights<Segment>` records; a uniform run of numeric segments
(`layers.0.*`, `layers.1.*`, ...) becomes an `Array` field; a non-uniform run
falls back to `i0`, `i1`, ... fields; non-contiguous numeric indices are a
compile-time error (`validateIndexedContiguous`, `Tyr/SafeTensors/TypeProvider.lean:265`).
The top level of the generated namespace contains:

- `abbrev Weights (device : Device := .CPU)` — the full hierarchical record;
- `def loadAll (source : String := defaultSource) (device : Device := .CPU) : IO (Weights device)`;
- `def loadAllFromHandle (handle : SafeTensorsHandle) (device : Device := .CPU) : IO (Weights device)` (single-file sources);
- subtree namespaces with their own `load` / `loadFromHandle` — e.g.
  `MyWeights.model.layers.load`, with the same requested-device parameter;
- metadata: `defaultSource`, `sourceIsDirectory`, `tensorCount`, `schema`,
  `hasTensor`, `find?`, `fieldToTensorName`.

Generated code is rendered as strings and re-elaborated via
`Parser.runParserCategory` (`Tyr/SafeTensors/TypeProvider.lean:116-124`), with all
references `_root_.`-qualified. KittenTTS is the in-tree consumer: the provider
runs against a checked-in schema snapshot (`Tyr/Model/KittenTTS/Checkpoint.lean:14`)
and `Tyr/Model/KittenTTS/Weights.lean:784-792` loads subtrees from one open handle.

### HuggingFace Hub resolution

`Tyr/Hub.lean` (namespace `torch.Hub`) is deliberately small and depends only on
`Lean.Data.Json` and an external `curl` binary — no `Tyr.Torch` import. The flow:

```lean
structure DownloadOptions where
  revision : String := "main"
  cacheDir : String := defaultCacheDir  -- "~/.cache/huggingface/tyr-models"
  includeTokenizer : Bool := true

def resolvePretrainedDir (source : String) (opts : DownloadOptions := {})
    (tokenizerFiles : Array String := #[]) : IO String
```

`resolvePretrainedDir` resolves in order (`Tyr/Hub.lean:230-244`):

1. an existing local directory is returned as-is;
2. the standard HF hub cache is searched (`findCachedSnapshot?` scans
   `$HF_HOME/hub` or `~/.cache/huggingface/hub` for
   `models--<org>--<repo>` snapshots containing `config.json` plus weights);
3. otherwise files are downloaded into the Tyr cache
   `<cacheDir>/<org>__<repo>/<revision>`: `config.json` (required), weights via
   `ensureModelWeights` (prefers `model.safetensors.index.json` + its shards,
   falls back to a single `model.safetensors`), tokenizer files best-effort.

Downloads shell out to `curl -fL --retry 3` with `.tmp`-file resume and an
optional `HF_TOKEN` bearer header (`Tyr/Hub.lean:89-118`). The generic entry point
used by every `Tyr/Model/*/Pretrained.lean` is:

```lean
def loadModelFromPretrained
    {Cfg : Type} {Model : Cfg → Type}
    (source : String) (revision : String) (cacheDir : String)
    (tokenizerFiles : Array String)
    (loadConfig : String → Cfg → IO Cfg) (defaults : Cfg)
    (loadSharded : String → (cfg : Cfg) → IO (Model cfg))
    (loadSingle : String → (cfg : Cfg) → IO (Model cfg))
    : IO (Sigma (fun cfg => Model cfg))
```

It resolves the directory, loads the config, picks sharded vs single-file via
`detectWeightLayout`, and returns the config and model as a dependent pair.

### Training checkpoints

`Tyr/Checkpoint.lean` (namespace `torch.checkpoint`) is model-agnostic
persistence over the `TensorStruct` class (see
[TensorStruct](core/tensorstruct.md)):

```lean
structure CheckpointMeta where
  iteration : Nat
  bestValLoss : Float
  trainLoss : Float
  optimCount : Nat := 0

def saveParams [TensorStruct α] (params : α) (dir : String)
    (namePrefix : String := "param") (log : Handlers := {}) : IO Unit
def loadParams [TensorStruct α] (template : α) (dir : String)
    (namePrefix : String := "param") (log : Handlers := {})
    (asParameters : Bool := true) : IO α
```

Checkpoint saves publish immutable directories under `.snapshots/` by atomically
replacing a small `CURRENT` pointer after every tensor and metadata file has been
written. Readers resolve that pointer once, so an interrupted overwrite cannot
mix the previous and next checkpoint. Existing flat checkpoint directories remain
readable. Old snapshots are retained for concurrent readers and may be pruned
offline; this guarantees atomic visibility, not durability after power loss.

Within each snapshot:

- `{namePrefix}_{i}.pt` contains one tensor in `TensorStruct` traversal order.
  Tensor shapes and dtypes are stored by LibTorch; shapes are checked against the
  template and stored dtypes are preserved (including bf16 resumes with a freshly
  initialized fp32 template). Parameter names are positional: changing traversal order can still
  invalidate an old checkpoint.
- `{namePrefix}_count.txt` records the tensor count for new saves; loading into a
  shorter, empty, or longer parameter tree fails instead of silently dropping state.
- `meta.txt` contains version 2 JSON, with required iteration and optimizer count
  fields and exact IEEE-754 bit encodings for losses. This preserves tiny values,
  signed zero, and infinity. Valid legacy `key=value` files remain readable;
  missing or duplicate required fields are rejected.
- `saveTrainingCheckpoint` stores parameters, Adam moments, count, and metadata in
  one snapshot. `loadTrainingCheckpoint` restores them from a single resolved
  snapshot, suitable for an exact training resume.
- The standalone `saveOptimizerState` helper publishes its own
  `OPTIMIZER_CURRENT` snapshot. Use the combined training API when model and
  optimizer state must become visible together.

Training formats with heterogeneous optimizer state can use `publishSnapshot`
to write all artifacts into one private directory, then call `resolveSnapshot`
once when loading. These callbacks use the same publication protocol as the
combined Adam helper.

NanoChat's `ModdedTrain.saveCheckpoint` uses this protocol for its model, legacy
Adam state, dual optimizer state, counters, and optional training metadata.
`state_meta.json` stores exact scalar bits, per-parameter optimizer steps, and
optional Muon-buffer layout so resumed updates retain their original state.
New snapshots reject missing or corrupted state metadata; legacy flat directories
retain their previous best-effort optimizer loading behavior.
For streaming training, every rank writes its cursor to unpublished staging first.
The master includes all rank cursor files in the model snapshot before publication;
resume uses that same pinned directory and requires the saved world size to match.
Distributed training assumes one writer per checkpoint directory and no concurrent
publisher during distributed resume. Each rank pins its own read directory;
rank cursor I/O failures are exchanged before publication so peers fail together.

This is a Tyr-native format, not interchangeable with HF weights — note it is
libtorch `.pt` persistence. `Tensor.saveSafeTensor` is available for single-tensor
SafeTensors export.

## Key APIs

### SafeTensors schema (`torch.safetensors`, `Tyr/SafeTensors/Schema.lean`)

| API | Signature | Notes |
| --- | --- | --- |
| `introspect` | `String → IO Schema` | file, shard dir, index dir, or `.schema.json` snapshot |
| `Schema.find?` | `Schema → String → Option TensorSchema` | exact-name lookup |
| `saveSnapshot` / `loadSnapshot` | `String → Schema → IO Unit` / `String → IO Schema` | persist/restore a schema |
| `TensorSchema.toSpec` | `TensorSchema → TensorSpec` | shape + dtype |
| `TensorSchema.contract` | `(role := .parameter) → (expectedDevice := none) → TensorContract` | boundary contract |
| `loadTensor[Sharded]WithContract` | `... → IO (Tensor { shape := contract.spec.shape, dtype := contract.spec.dtype, device := device })` | checked load from file/dir |
| `loadFromHandleWithContract` | `SafeTensorsHandle → ... → IO (Tensor { shape := contract.spec.shape, dtype := contract.spec.dtype, device := device })` | checked load from open handle |
| `...WithSpec`, `...WithSchema` | same shapes | spec/schema conveniences |

### Try-load helpers (`Tyr/SafeTensors/Load.lean`)

| API | Returns `none` when |
| --- | --- |
| `tryLoadTensor` / `tryLoadTensorSharded` | any error occurs |
| `tryLoadOptionalTensor` / `tryLoadOptionalTensorSharded` | tensor is missing (other errors rethrown) |
| `tryLoadTensorCandidates` / `tryLoadTensorShardedCandidates` | no candidate name loads |
| `loadTensorCandidates` / `loadTensorShardedCandidates` | throwing variants of the above |

All take `(path|dir) (name[s]) (s : Shape) (device : Device := Device.CPU)` and
return `IO (Option (Tensor { shape := s, device := device }))` resp. `IO (Tensor { shape := s, device := device })`; dtype follows the file.

### Hub (`torch.Hub`, `Tyr/Hub.lean`)

| API | Signature | Notes |
| --- | --- | --- |
| `defaultCacheDir` | `String` | `~/.cache/huggingface/tyr-models` |
| `resolvePretrainedDir` | `String → DownloadOptions → Array String → IO String` | local dir, HF cache, or download |
| `findCachedSnapshot?` | `String → (revision : String := "main") → IO (Option String)` | standard HF hub layout |
| `detectWeightLayout` | `String → IO Bool` | `true` = sharded |
| `shardFilesFromIndexFile` | `String → IO (Array String)` | unique shards from `weight_map` |
| `ensureModelWeights` / `ensureTokenizerFiles` | `... → IO Unit` | explicit download steps |
| `loadModelFromPretrained` | see signature above | generic `from_pretrained` |

### Checkpoint (`torch.checkpoint`, `Tyr/Checkpoint.lean`)

| API | Signature | Notes |
| --- | --- | --- |
| `saveParams` / `loadParams` | see above | positional `{prefix}_{i}.pt` tree |
| `saveCheckpoint` | `(params : α) → (iteration : Nat) → (bestValLoss trainLoss : Float) → (dir : String) → ... → IO Unit` | params + `meta.txt` |
| `loadCheckpoint` | `(template : α) → (dir : String) → ... → IO (α × CheckpointMeta)` | |
| `saveOptimizerState` | `(mu nu : α) → (count : Nat) → (dir : String) → ... → IO Unit` | mirrored Adam trees |
| `loadOptimizerState` | `(template : α) → (dir : String) → ... → IO (α × α × Nat)` | |
| `saveTrainingCheckpoint` | `(params mu nu : α) → CheckpointMeta → String → ... → IO Unit` | one model + Adam snapshot |
| `loadTrainingCheckpoint` | `(template : α) → String → ... → IO (α × α × α × CheckpointMeta)` | one pinned training state |
| `publishSnapshot` / `resolveSnapshot` | `String → String → (String → IO Unit) → IO Unit` / `String → ... → IO String` | custom multi-file training formats |
| `checkpointExists` / `optimStateExists` | `String → IO Bool` | probe for `meta.txt` / `optim_count.txt` |
| `saveCheckpointMeta` / `loadCheckpointMeta` | `CheckpointMeta → String → IO Unit` / `String → IO CheckpointMeta` | `meta.txt` only |

## Usage examples

Reconstructed example (from `Tests/TestSafeTensorsTypeProvider.lean:13-219` and
`launch/demos/safetensors_schema.lean`):

```lean
import Tyr.SafeTensors

open torch

-- Elaboration-time introspection; the source must exist at build time.
safetensors_type_provider "Tests/fixtures/safetensors/single.safetensors" as SingleSafe
safetensors_type_provider "Tests/fixtures/safetensors/sharded" as ShardedSafe

def demo : IO Unit := do
  -- Per-tensor loaders; shape is checked against the generated Shape abbrev.
  let w ← SingleSafe.load_linear_weight            -- Tensor { shape := #[2, 3], dtype := .Float32, device := .CPU }
  IO.println s!"weight shape: {w.runtimeShape}"
  -- Explicitly named dtype-indexed checked loader.
  let wTyped ← SingleSafe.load_linear_weightTyped  -- Tensor { shape := #[2, 3], dtype := .Float32, device := .CPU }
  IO.println s!"weight dtype: {wTyped.actualSpec.dtype}"
  -- Full hierarchical record: fields follow tensor names split on '.'.
  let weights ← ShardedSafe.loadAll
  IO.println s!"{weights.embed.weight.runtimeShape}"   -- #[2, 2]
  -- Subtree namespace loader.
  let embed ← ShardedSafe.embed.load
  IO.println s!"embedding shape: {embed.weight.runtimeShape}"
  -- Metadata.
  IO.println s!"{ShardedSafe.tensorCount} tensors"     -- 2
  IO.println s!"{ShardedSafe.proj_biasSpec.sourceFile}" -- "part2.safetensors"
```

Reconstructed example (from `Tyr/Model/Qwen35/Pretrained.lean:54-69`):

```lean
/-- Load a text Qwen3.5 checkpoint from local dir or HF repo id. -/
def Qwen35ForCausalLM.loadFromPretrained
    (source : String)
    (defaults : Config := Config.qwen35_9B)
    (revision : String := "main")
    (cacheDir : String := Hub.defaultCacheDir)
    (device : Device := Device.CPU)
    : IO (Sigma (fun cfg => Qwen35ForCausalLM (device := device) cfg)) := do
  Hub.loadModelFromPretrained
    source revision cacheDir hub.tokenizerFiles
    (fun modelDir cfg => Config.loadFromPretrainedDir modelDir cfg)
    defaults
    (fun modelDir cfg => Qwen35ForCausalLM.loadSharded modelDir cfg device)
    (fun path cfg => Qwen35ForCausalLM.load path cfg device)

-- `loadFromPretrained "Qwen/Qwen3.5-0.8B"` downloads on first use;
-- `loadFromPretrained "./my-local-qwen-dir"` uses the directory directly.
```

Combined Adam checkpoint example:

```lean
open torch.checkpoint

-- Publish the model and optimizer at the same training step.
saveTrainingCheckpoint finalParams optState.mu optState.nu {
  iteration := trainCfg.maxIters
  bestValLoss := bestValLoss
  trainLoss := 0.0
  optimCount := optState.count
} "checkpoints/gpt"

-- Resume: the template (freshly initialized params) supplies the shapes.
def resume (initParams : α) [TensorStruct α] :
    IO (α × α × α × CheckpointMeta) := do
  loadTrainingCheckpoint initParams "checkpoints/gpt"
```

The molecule BranchingFlows executable publishes model parameters, metadata,
Muon momentum leaves (`optim_muon_momentum_*`), and `optim_muon_count.txt`
together. Resume pins that snapshot and checks the optimizer step against the
metadata. Its generation launcher also pins the snapshot before hashing and
loading it.

NanoChat publishes its heterogeneous optimizer state with the same protocol.
Its versioned `state_meta.json` preserves exact scalar values, per-parameter
optimizer counters, and optional buffer layout. New snapshots reject missing or
invalid state; legacy flat directories retain their compatibility loader.

## Caveats

- The type provider introspects at elaboration time: the checkpoint (or a
  `.schema.json` snapshot) must exist on the build machine, and introspection
  failures surface as a generic `safetensors_type_provider failed while
  introspecting source ...` error (`Tyr/SafeTensors/TypeProvider.lean:628-632`).
  `introspect` reads only the eight-byte length prefix and JSON header of each
  shard. Headers are limited to 100 MiB and checked against the file size before
  allocation; tensor payloads are not read during schema discovery.
- Generated `defaultSource` is the elaboration-time path string. Every generated
  loader takes `source := defaultSource`, so pass the runtime path explicitly
  when the weights live elsewhere (as KittenTTS does with `openHandle` +
  `loadFromHandle`).
- Tyr checkpoint trees are positional and template-driven. Shapes are checked,
  but reordering same-shaped parameters requires an explicit migration.

## Related guides

- [Core tensors](core/tensors.md) — `Tensor σ`, `Shape`, `DType`, `Device`
- [Typed tensors](core/typed.md) — `TensorSpec`, `TensorContract`, `Tensor`
- [TensorStruct](core/tensorstruct.md) — the traversal class behind checkpointing
- [Optimization](optimization.md) — the Adam state trees being persisted
- [LLM models](models/llms.md), [Audio and speech models](models/audio-speech.md) —
  per-family `Weights`/`Pretrained` modules built on these APIs
- [FFI and build](ffi-and-build.md) — the `lean_torch_safetensors_*` C++ side

Exhaustive per-symbol documentation is generated by doc-gen4 (see `docbuild/`);
this chapter is a guide, not a symbol dump.
