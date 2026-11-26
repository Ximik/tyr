# Data pipeline

## Purpose & when to use

This chapter covers everything between raw bytes on disk and token/mask tensors
ready for training or inference: a pure-Lean BPE tokenizer stack (training,
encode/decode, binary persistence), Hugging Face-compatible tokenizer codecs for
the Qwen and Gemma families, binary shard loaders for pretraining corpora with
distributed rank-aware iteration, parquet dataset access over an Arrow FFI,
nanochat-style task/conversation machinery with loss masking for SFT and
midtraining, and LR scheduling math for multi-stage training. Reach for these
modules when you need to feed a model; training loops and model code live
elsewhere. The design is heavily modeled on Karpathy's nanochat.

## Architecture & main abstractions

### Namespaces and imports

The data layer predates the `Tyr.*` naming convention, so it lives under three
unrelated roots:

- `tokenizer` (global, no `Tyr` prefix) — the BPE stack in `Tyr/Lib/Tokenizer/`.
  Umbrella import: `import Tyr.Lib.Tokenizer` (re-exports all 13 submodules).
- `torch.DataLoader` (`Tyr/Lib/Data/Loader.lean`) and `torch.Data.*`
  (`Tyr/Lib/Data/{Task,TaskClass,Tasks,Download,HuggingFace}.lean`).
  Umbrella `import Tyr.Lib.Data` re-exports all of them.
- `Tyr.Text` (`Tyr/Text/`) — streaming-ASR utilities; no umbrella module and
  not re-exported by `import Tyr`.

Token tensors use `Tensor { shape := shape, dtype := .Int64, device := .CPU }`; training masks use `Tensor { shape := shape, dtype := .Float32, device := .CPU }` (see [core/tensors.md](core/tensors.md)).

### BPE tokenizer stack

The core type is `tokenizer.BPETokenizer` (`Tyr/Lib/Tokenizer/Types.lean`), a
purely functional structure of arrays and `Std.HashMap`s:

```lean
structure BPETokenizer where
  vocabSize : UInt32
  idToBytes : Array ByteArray
  bytesToId : Std.HashMap ByteArray UInt32
  merges : Array MergeRule
  mergeLookup : Std.HashMap (UInt32 × UInt32) UInt32
  mergePriority : Std.HashMap (UInt32 × UInt32) Nat
  specialTokens : Std.HashMap String UInt32
  idToSpecial : Std.HashMap UInt32 String
```

Encoding pipeline (`Tyr/Lib/Tokenizer/Encode.lean`): text is split by the
hand-rolled GPT-style pretokenizer (`pretokenizeFull`,
`Tyr/Lib/Tokenizer/Pretokenize.lean`), each chunk is mapped to one token per
byte, then merge rules are applied greedily by priority (`encodeWord`,
`Encode.lean`). Special tokens are split out first and matched
longest-first (`encodeWithSpecials`, `Encode.lean`). Decoding concatenates
token bytes with a lossy UTF-8 fallback (`decode`,
`Tyr/Lib/Tokenizer/Decode.lean`).

Lifecycle: `trainBPE docs cfg` (`Tyr/Lib/Tokenizer/Training.lean`) runs a
pair-counting merge loop and returns a `TrainResult` (tokenizer, `TokenBytes`
table for bits-per-byte evaluation, statistics). `save`/`load`
(`Tyr/Lib/Tokenizer/IO.lean`) persist to a little-endian binary format
with magic `TYR_BPE1`, version 1.

`Tyr/Lib/Tokenizer/SpecialTokens.lean` provides `addSpecialTokens`; `Training.lean`
defines `defaultChatSpecialTokens`, the nanochat chat set.

### Hugging Face tokenizer codecs

Separate structures load HF tokenizers without round-tripping through
`BPETokenizer`; they reuse only `MergeRule` and duplicate the merge
loop internally:

- `tokenizer.ByteLevelBPETokenizer` (`Tyr/Model/Qwen/Tokenizer.lean`) — byte-level
  BPE loaded from `tokenizer.json` or `vocab.json` + `merges.txt`
  (`loadTokenizer`, `Qwen3.lean`), plus chat templates and TTS text
  wrappers.
- `tokenizer.qwen35` / `tokenizer.qwen36` — `abbrev` aliases of the Qwen3
  codec with family-specific chat templates (`chatTemplateThinking`,
  `userPrefix`, `assistantGenerationSuffix`; `Tyr/Model/Qwen35/Tokenizer.lean`,
  `Qwen36.lean`).
- `tokenizer.gemma4.GemmaTokenizer` (`Tyr/Model/Gemma4/Tokenizer.lean`) —
  SentencePiece-style BPE with `▁` whitespace marker and `<0x..>` byte
  fallback, loaded from `tokenizer.json` (`loadTokenizer`, `Gemma4.lean`).

### Binary shard loading

`torch.DataLoader` (`Tyr/Lib/Data/Loader.lean`) streams fixed-shape windows out of
uint16 `.bin` shards in the fineweb/modded-nanogpt format (256 int32 header
words, magic `20240520`, version 1; `splitFinewebPayload`,
`DataLoader.lean`). The chain:

1. `resolveShardPaths spec kind` (`DataLoader.lean`) accepts a file,
   directory, or path prefix and returns sorted `.bin` paths (preferring
   `fineweb_train_*` / `fineweb_val_*`).
2. `DataShard.loadFromFile` (`DataLoader.lean`) reads a shard via the torch
   FFI, strips the header, slices the per-rank token range
   `[rank·⌊n/world⌋, ...)`, and builds a `BOSFinder`.
3. `DataShard.load` retains the actual partition length and returns
   `Σ n, DataShard n`, just like `loadFromFile`; it neither truncates large
   partitions nor tiles small ones.
4. `BOSFinder` packs document fragments in a deterministic order. A seeded
   shuffle changes document order while preserving token order inside each
   document, including the partition fragment before its first BOS. Sequences
   may span documents; BOS tokens remain in the stream, and attention masking
   across document boundaries remains the model's responsibility.
5. `DistributedDataGenerator` preserves incomplete file tails across batches.
   After every file has been consumed it advances the epoch and reshuffles;
   a fixed-shape batch can span that epoch boundary. Rank and world size
   are auto-detected via `Tyr.Lib.Distributed` in `DistributedDataGenerator.init`.

`Config.seed` controls document order (`shuffle := false` selects source
order). `numWorkers` and `bufferSize` are retained only for source compatibility:
this loader performs synchronous reads and has no worker pool or prefetch queue.
It loads one complete file before selecting the rank partition; it is not yet a
bounded-memory reader for arbitrarily large files. `BatchIterator` used directly
is a finite single-shard iterator and returns `none` for an incomplete final
batch; use the distributed generator to preserve tails across files.
Batch dimensions are arguments of each `next…` call, not iterator state, so a
batch's shape follows from the call. They and GPT's additional target token are
checked before fixed-width arithmetic, and invalid sizes report errors through `IO`.

`DistributedDataGenerator.cursor` records the file index, logical token
position, epoch, seed, BOS policy, rank/world size, paths and partition length.
`restoreCursor` reconstructs document order and rejects incompatible metadata;
the underlying shard contents must remain unchanged. ModdedTrain embeds this
versioned cursor in checkpoint metadata. Legacy cursor-only checkpoints are
rejected for exact data resume because their offsets addressed the old truncated,
unshuffled stream. ModdedTrain also rejects multi-rank exact data resume until
its checkpoints collect every rank's cursor instead of only rank 0's.

Batch shapes remain indexed even when their dimensions come from runtime state.
`BatchIterator.next` returns an optional Int64 tensor with shape
`#[iter.batchSize, iter.seqLen]`. Updating the iterator's configuration changes
the dependent result type on the next call; no shape erasure is needed.

`SequentialBatchIterator (n b s : UInt64)` also carries dimensions in its type
and returns optional pairs of `Tensor { shape := #[b, s], dtype := .Int64, device := .CPU }`, shifted by one token.
It lives in `Examples.GPT.GPTDataLoader`, alongside `BatchIterator.nextGPT`
and the distributed GPT batching and scheduling helpers. These GPT helpers
retain both the input and target dimensions and their Int64 dtype.

### Parquet

`torch.Data.Parquet` (`Tyr/Core/FFI/Parquet.lean`) binds Apache Arrow
through `@[extern]` declarations implemented in `cc/src/tyr_parquet.cpp`: `listParquetFiles`, `getParquetMetadata`,
`readRowGroup`, `readParquetAsJson`, `readRowGroupAsJson`.

`torch.Data.Download` shells out to `curl`/`unzip` with retry/backoff
(`downloadWithRetry`, `Tyr/IO/Download.lean`) and resolves HF parquet
URLs (`ensureHFParquet`). `torch.Data.HuggingFace` turns parquet into
`Lean.Json` rows and ships per-dataset loaders (`loadARC`, `loadGSM8K`,
`loadMMLU`, `loadSmolTalk`; `Tyr/IO/HuggingFace.lean`).

### Tasks, conversations, and mixtures

`torch.Data.Task` (`Tyr/Lib/Data/Task.lean`) models chat data:
`Conversation = Array Message` with `Role` (system/user/assistant/tool) and
optional structured `ContentPart`s (text/code/toolCall/toolResult).
`renderConversation` (`Task.lean`) is the loss-mask heart:

```lean
def renderConversation
    (conv : Conversation)
    (tokens : ChatTokens)
    (encode : String → Array UInt64)
    : TokenizedConversation
```

It emits BOS plus role-delimited tokens; the mask is 1 on assistant content
and end markers, 0 elsewhere (tool outputs are masked out), and a leading
system message is merged into the first user message. `collate`
pads tokenized conversations into a `ConversationBatch`: its `batchSize` and
`seqLen` fields index Int64 tokens and Float32 masks.

Mixtures come in three forms, all projecting to the function-closure
`ConversationMixture { size, seed, getAtSeed }` (`Task.lean`):

- `TaskMixture` (`Task.lean`) — closed, weighted, deterministically
  shuffled mixture over `LoadedTask`.
- `EvalTask` typeclass + `BoxedTask` existential + `GenericTaskMixture`
  (`Tyr/Lib/Data/TaskClass.lean`) — the open extension point; add your
  own task type with an `EvalTask` instance, box it with `boxTask`, combine
  with `entry`/`GenericTaskMixture.create`.

Consumers: `TaskIterator` (`Task.lean`) yields padded
`[batch, maxLen]` batches with masks; `TaskTokenStream` (`Task.lean`)
streams rank-strided rendered conversations into a token buffer and emits
contiguous `batchSize * seqLen + 1` windows, split into `(inputs, targets)` by
one-token shift — the nanochat midtraining data flow.

### Streaming text utilities

`Tyr.Text` (`Tyr/Text/`) serves streaming ASR, not training:

- `updateWithSignals` (`Tyr/Lib/Inference/StreamingConsensus.lean`) stabilizes
  streaming hypotheses into an append-only stable prefix plus a mutable tail,
  combining prefix agreement with the previous hypothesis, k-window history
  consensus, freeze and VAD-boundary signals.
- `SileroProvider` (`Tyr/Model/SileroVAD/VADProvider.lean`) wraps the Silero VAD
  model (`Tyr.Model.SileroVAD`), consuming 16 kHz audio in 512-sample chunks
  and emitting the `VADSignal`s (`speechActive`, `boundary`) that feed
  `updateWithSignals`.

## Key APIs

### Tokenizer core — `namespace tokenizer`

| API | Signature | Location |
| --- | --- | --- |
| `trainBPE` | `(docs : Array String) (config : TrainConfig) (log : Handlers := {}) : IO TrainResult` | `Tyr/Lib/Tokenizer/Training.lean` |
| `save` / `load` | `(tok : BPETokenizer) (path : String) : IO Unit` / `(path : String) : IO BPETokenizer` | `Tyr/Lib/Tokenizer/IO.lean` |
| `encode` / `encodeWithSpecials` | `(tok : BPETokenizer) (text : String) : Array UInt32` | `Tyr/Lib/Tokenizer/Encode.lean` |
| `decode` / `decodeToBytes` | `(tok : BPETokenizer) (ids : Array UInt32) : String` / `: ByteArray` | `Tyr/Lib/Tokenizer/Decode.lean` |
| `countTokens` | `(tok : BPETokenizer) (text : String) : Nat` | `Tyr/Lib/Tokenizer/Encode.lean` |
| `TrainConfig` | `{ vocabSize := 32768, maxChars, docCap, splitPattern, specialTokens, seed }` | `Tyr/Lib/Tokenizer/Training.lean` |
| `defaultChatSpecialTokens` | `Array String` (nanochat chat token set) | `Tyr/Lib/Tokenizer/Training.lean` |
| `TokenBytes.fromTokenizer` / `.totalBytes` | `BPETokenizer → TokenBytes` / `TokenBytes → Array UInt32 → IO UInt64` | `Tyr/Lib/Tokenizer/TokenBytes.lean` |

### HF codecs — `tokenizer.qwen3` / `qwen35` / `qwen36` / `gemma4`

| API | Signature | Location |
| --- | --- | --- |
| `tokenizer.ByteLevelBPETokenizer.load` | `(dir : String) : IO ByteLevelBPETokenizer` | `Tyr/Lib/Tokenizer/ByteLevelBPE.lean` |
| `tokenizer.ByteLevelBPETokenizer.encodeText` / `decodeText` | `(tok) (text : String) : Array UInt32` / `(tok) (ids) : String` | `Tyr/Lib/Tokenizer/ByteLevelBPE.lean` |
| `qwen3.chatTemplate` | `(prompt : String) : String` | `Tyr/Model/Qwen/Tokenizer.lean` |
| `qwen3.encodePrompt` | `(tok) (prompt) (maxLen : Nat := 512) : Array UInt32 × Array UInt32` (tokens + attention mask) | `Tyr/Model/Qwen/Tokenizer.lean` |
| `qwen3.ttsAssistantText` | `String → String` | `Qwen3.lean` |
| `qwen35.chatTemplateThinking`, `userPrefix`, `assistantGenerationSuffix` | `String → String` / `String` | `Tyr/Model/Qwen35/Tokenizer.lean` |
| `gemma4.loadTokenizer` | `(dir : String) : IO GemmaTokenizer` | `Tyr/Model/Gemma4/Tokenizer.lean` |
| `gemma4.encodeText` / `decodeText` / `chatTemplate` / `chatTemplateThinking` | as Qwen3 | `Gemma4.lean` |

`qwen35` re-exports the qwen3 codec as `abbrev`s (`loadTokenizer`,
`encodeText`, ...); `qwen36` re-exports `qwen35`.

### Shard loading — `namespace torch.DataLoader`

| API | Signature | Location |
| --- | --- | --- |
| `Config` | `{ dataPath, valPath, seqLen, bosToken, seed, shuffle }`; synchronous compatibility fields `numWorkers`, `bufferSize` | `Tyr/Lib/Data/Loader.lean` |
| `resolveShardPaths` | `(pathSpec : String) (kind : ShardKind) : IO (Array String)` | `Tyr/Lib/Data/Loader.lean` |
| `DataShard.loadFromFile` | `(path) (shardIdx numShards bosToken : UInt64) : IO (Σ n, DataShard n)` | `Tyr/Lib/Data/Loader.lean` |
| `DataShard.load` / `loadValidationData` | Return `IO (Σ n, DataShard n)` with the real partition length | `Tyr/Lib/Data/Loader.lean` |
| `DistributedDataGenerator.cursor` / `.restoreCursor` | Versioned `StreamCursor` capture and validated restore | `Tyr/Lib/Data/Loader.lean` |
| `BatchIterator.new` / `.next` | `(shard) : BatchIterator` / `(iter) (batchSize seqLen : Nat) : IO (Option (Tensor { shape := #[batchSize, seqLen], dtype := .Int64, device := .CPU }) × BatchIterator)` | `Tyr/Lib/Data/Loader.lean` |
| `DistributedDataGenerator.init` | `(config : Config) : IO DistributedDataGenerator` | `Tyr/Lib/Data/Loader.lean` |
| `DistributedDataGenerator.nextBatch` | `(gen) (batchSize seqLen : Nat) : IO (Option (Tensor { shape := #[batchSize, seqLen], dtype := .Int64, device := .CPU }) × DistributedDataGenerator)` | `Tyr/Lib/Data/Loader.lean` |
| `BatchIterator.nextGPT` | `(iter) (batchSize seqLen : Nat) : IO (Option (Tensor { shape := #[batchSize, seqLen], dtype := .Int64, device := .CPU } × Tensor { shape := #[batchSize, seqLen], dtype := .Int64, device := .CPU }) × BatchIterator)` (input/target shift) | `Examples/GPT/GPTDataLoader.lean` |
| `DistributedDataGenerator.nextBatchGPT` | `(gen) (batchSize seqLen : Nat)`; GPT input/target batch | `Examples/GPT/GPTDataLoader.lean` |
| `SequentialLoader.fromFile` / `SequentialBatchIterator.new` / `.next` | fixed-dim Shakespeare-style loading, `Tensor { shape := #[b, s], dtype := .Int64, device := .CPU }` batches | `Examples/GPT/GPTDataLoader.lean` |

### Parquet / downloads — `torch.Data.Parquet`, `torch.Data.Download`, `torch.Data.HuggingFace`

| API | Signature | Location |
| --- | --- | --- |
| `listParquetFiles`, `getParquetMetadata`, `readRowGroup`, `readParquetAsJson`, `readRowGroupAsJson` | `@[extern]` opaque IO functions | `Tyr/Core/FFI/Parquet.lean` |
| `downloadWithRetry` | `(url dest : String) (maxRetries := 5) (initialBackoffMs := 1000) (log := {}) : IO Bool` | `Tyr/IO/Download.lean` |
| `ensureHFParquet` | `(repoId subset split : String) (cacheDir ..) : IO String` | `Download.lean` |
| `loadARC` / `loadGSM8K` / `loadMMLU` / `loadSmolTalk` | `... : IO (Array Lean.Json)` | `Tyr/IO/HuggingFace.lean` |
| `loadJsonlFromFile` | `... : IO (Array Lean.Json)` | `HuggingFace.lean` |

### Tasks — `torch.Data.Task`, `torch.Data.TaskClass`

| API | Signature | Location |
| --- | --- | --- |
| `Message.user` / `.assistant` / `.system` | `(content : String) : Message` | `Tyr/Lib/Data/Task.lean` |
| `ChatTokens` | special-token id record for rendering | `Task.lean` |
| `renderConversation` | `(conv) (tokens : ChatTokens) (encode : String → Array UInt64) : TokenizedConversation` | `Task.lean` |
| `collate` | `(convs) (maxLen : Nat) (padToken := 0) : IO ConversationBatch` | `Task.lean` |
| `TaskIterator.new` / `.nextBatch` | padded masked batches from a `ConversationMixture` | `Task.lean` |
| `TaskTokenStream.new` / `.nextGPTBatch` | rank-strided token windows; indexed Int64 input/target pairs with the stream dimensions | `Task.lean` |
| `TaskMixture.create` / `.toConversationMixture` | `(entries : Array MixtureEntry) (seed := 42)` | `Task.lean` |
| `EvalTask` class / `boxTask` / `entry` / `GenericTaskMixture.create` | open task typeclass + existential mixture | `Tyr/Lib/Data/TaskClass.lean` |

### Streaming text — `namespace Tyr.Text`

| API | Signature | Location |
| --- | --- | --- |
| `updateWithSignals` | `(st : ConsensusState) (ids : Array UInt32) (speechActive := true) (boundary := false) (decode : Array UInt32 → String) : ConsensusState × TextDelta` | `Tyr/Lib/Inference/StreamingConsensus.lean` |
| `update` | signal-free wrapper | `StreamingConsensus.lean` |
| `initSileroProvider` | `(weightsPath : String) (threshold := 0.5) (minSilenceDurationMs := 100) (speechPadMs := 30) : IO SileroProvider` | `Tyr/Model/SileroVAD/VADProvider.lean` |
| `stepSileroProvider` | `(p) (pcm16k : Array Float) : IO (SileroProvider × VADSignal)` | `VADProvider.lean` |

## Usage examples

Train, persist, and use a BPE tokenizer — reconstructed example (from
`Examples/NanoChat/Pipeline.lean`):

```lean
import Tyr.Lib.Tokenizer

-- texts : Array String, e.g. loaded from parquet shards
let trainConfig : tokenizer.TrainConfig := {
  vocabSize := 32768
  specialTokens := tokenizer.defaultChatSpecialTokens
}
let result ← tokenizer.trainBPE texts trainConfig
tokenizer.save result.tokenizer "tokenizer.bin"

let tok ← tokenizer.load "tokenizer.bin"
let ids := tokenizer.encodeWithSpecials tok "<|user_start|>hello<|user_end|>"
let text := tokenizer.decode tok ids
let n := tokenizer.countTokens tok someText
```

Pretraining batches from fineweb shards — reconstructed example (from
`Tests/TestDataLoader.lean`, `Examples/NanoChat/ModdedTrain.lean`):

```lean
import Tyr.Lib.Data.Loader
open torch torch.DataLoader

def nextTokenCount (gen : DistributedDataGenerator) : IO Nat := do
  let (batch?, _) ← gen.nextBatch 8 1024
  match batch? with
  | none => pure 0
  | some batch =>
    let indexed : Tensor { shape := #[8, 1024], dtype := .Int64, device := .CPU } := batch
    pure (← indexed.toUInt64Array).size
```

Create a generator with `DistributedDataGenerator.init cfg`.
With `import Examples.GPT.GPTDataLoader`, `gen.nextBatchGPT batchSize seqLen`
returns indexed input/target pairs separated by one token.


Fixed-dim Shakespeare batches — reconstructed example (from
`Tests/TestDataLoader.lean`, definitions in
`Examples/GPT/GPTDataLoader.lean`):

```lean
import Examples.GPT.GPTDataLoader
open torch.DataLoader

let ⟨n, loader⟩ ← SequentialLoader.fromFile "data/shakespeare_char/train.bin"
let iter : SequentialBatchIterator n 4 32 := SequentialBatchIterator.new loader 4 32
let (pair?, iter') := iter.next    -- pair? : Option (Tensor { shape := #[4, 32], dtype := .Int64, device := .CPU } × Tensor { shape := #[4, 32], dtype := .Int64, device := .CPU })
```

Midtraining/SFT token stream over a task mixture — reconstructed example
(from `Examples/NanoChat/Pipeline.lean`):

```lean
import Tyr.Lib.Tokenizer
import Tyr.Lib.Data
open torch.Data.Task

let tok ← tokenizer.load tokenizerFile
let chatTokens ← buildChatTokensFromTokenizer tok   -- helper: look up special ids in `tok`
let encode := fun (text : String) => (tokenizer.encodeWithSpecials tok text).map (·.toUInt64)
let mixture := (← createMidtrainTaskMixture cfg).toConversationMixture
let stream := TaskTokenStream.new mixture batchSize seqLen chatTokens encode rank worldSize
let (pair?, stream') ← stream.nextGPTBatch          -- optional indexed Int64 input/target pair
```

HF tokenizer for inference — reconstructed example (from
`Examples/Qwen35/RunHF.lean`):

```lean
import Tyr.Lib.Tokenizer

let tok ← tokenizer.qwen35.loadTokenizer modelDir
let text := tokenizer.qwen35.chatTemplate prompt    -- or chatTemplateThinking
let ids := (tokenizer.qwen35.encodeText tok text).map (·.toUInt64)
let out := tokenizer.qwen35.decodeText tok generatedIds
```

## Caveats

Behavior verified against source that the module docs do not advertise:

- `trainBPE` never pretokenizes and ignores `TrainConfig.{maxChars, docCap,
  splitPattern, seed}` (`Tyr/Lib/Tokenizer/Training.lean`).

## Related guides

- [getting-started.md](getting-started.md) — install and first model
- [core/tensors.md](core/tensors.md) — indexed `Tensor` specifications used throughout
- [core/typed.md](core/typed.md) — dependent typing patterns behind fixed-dim batches
- [serialization.md](serialization.md) — checkpoints and weights (tokenizer save/load is separate, above)
- [distributed.md](distributed.md) — rank/world-size detection used by the loaders
- [models/llms.md](models/llms.md) — models consuming the Qwen/Gemma codecs
- [audio.md](audio.md) — ASR stack behind `Tyr.Text`
- [examples-and-testing.md](examples-and-testing.md) — NanoChat/GPT examples and data tests

For exhaustive symbol-level documentation of these definitions, see the API reference generated by doc-gen4 (built from `docbuild/`).
