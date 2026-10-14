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

- `tokenizer` (global, no `Tyr` prefix) — the BPE stack in `Tyr/Tokenizer/`.
  Umbrella import: `import Tyr.Tokenizer` (re-exports all 13 submodules).
- `torch.DataLoader` (`Tyr/DataLoader.lean`) and `torch.Data.*`
  (`Tyr/Data/{Task,TaskClass,Tasks,Pipeline,Pretraining,Download,HuggingFace}.lean`).
  Umbrella `import Tyr.Data` re-exports only Task/TaskClass/Tasks/Pipeline/
  Pretraining, not Download or HuggingFace.
- `Tyr.Text` (`Tyr/Text/`) — streaming-ASR utilities; no umbrella module and
  not re-exported by `import Tyr`.

Token tensors use `DTensor shape .Int64`; training masks use `DTensor shape .Float32` (see [core/tensors.md](core/tensors.md)).

### BPE tokenizer stack

The core type is `tokenizer.BPETokenizer` (`Tyr/Tokenizer/Types.lean:21`), a
purely functional structure of arrays and `Std.HashMap`s:

```lean
structure BPETokenizer where
  vocabSize : UInt32
  idToBytes : Array ByteArray
  bytesToId : Std.HashMap ByteArray TokenId
  merges : Array MergeRule
  mergeLookup : Std.HashMap (TokenId × TokenId) TokenId
  mergePriority : Std.HashMap (TokenId × TokenId) Nat
  specialTokens : Std.HashMap String TokenId
  idToSpecial : Std.HashMap TokenId String
```

Encoding pipeline (`Tyr/Tokenizer/Encode.lean`): text is split by the
hand-rolled GPT-style pretokenizer (`pretokenizeFull`,
`Tyr/Tokenizer/Pretokenize.lean:119`), each chunk is mapped to one token per
byte, then merge rules are applied greedily by priority (`encodeWord`,
`Encode.lean:119`). Special tokens are split out first and matched
longest-first (`encodeWithSpecials`, `Encode.lean:156`). Decoding concatenates
token bytes with a lossy UTF-8 fallback (`decode`,
`Tyr/Tokenizer/Decode.lean:29`).

Lifecycle: `trainBPE docs cfg` (`Tyr/Tokenizer/Training.lean:211`) runs a
pair-counting merge loop and returns a `TrainResult` (tokenizer, `TokenBytes`
table for bits-per-byte evaluation, statistics). `save`/`load`
(`Tyr/Tokenizer/IO.lean:190,197`) persist to a little-endian binary format
with magic `TYR_BPE1`, version 1.

`Tyr/Tokenizer/SpecialTokens.lean` provides legacy "NanoProof" token sets
(chat, tool-use, math, Lean keywords) and `addSpecialTokens`; `Training.lean:28`
defines `defaultChatSpecialTokens`, the nanochat chat set.

### Hugging Face tokenizer codecs

Separate structures load HF tokenizers without round-tripping through
`BPETokenizer`; they reuse only `MergeRule`/`TokenId` and duplicate the merge
loop internally:

- `tokenizer.qwen3.QwenTokenizer` (`Tyr/Tokenizer/Qwen3.lean:50`) — byte-level
  BPE loaded from `tokenizer.json` or `vocab.json` + `merges.txt`
  (`loadTokenizer`, `Qwen3.lean:82`), plus chat templates and TTS text
  wrappers.
- `tokenizer.qwen35` / `tokenizer.qwen36` — `abbrev` aliases of the Qwen3
  codec with family-specific chat templates (`chatTemplateThinking`,
  `userPrefix`, `assistantGenerationSuffix`; `Tyr/Tokenizer/Qwen35.lean:20-38`,
  `Qwen36.lean`).
- `tokenizer.gemma4.GemmaTokenizer` (`Tyr/Tokenizer/Gemma4.lean:48`) —
  SentencePiece-style BPE with `▁` whitespace marker and `<0x..>` byte
  fallback, loaded from `tokenizer.json` (`loadTokenizer`, `Gemma4.lean:85`).

### Binary shard loading

`torch.DataLoader` (`Tyr/DataLoader.lean`) streams fixed-shape windows out of
uint16 `.bin` shards in the fineweb/modded-nanogpt format (256 int32 header
words, magic `20240520`, version 1; `splitFinewebPayload`,
`DataLoader.lean:210`). The chain:

1. `resolveShardPaths spec kind` (`DataLoader.lean:75`) accepts a file,
   directory, or path prefix and returns sorted `.bin` paths (preferring
   `fineweb_train_*` / `fineweb_val_*`).
2. `DataShard.loadFromFile` (`DataLoader.lean:239`) reads a shard via the torch
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
   are auto-detected via `Tyr.Distributed` in `DistributedDataGenerator.init`.

`Config.seed` controls document order (`shuffle := false` selects source
order). `numWorkers` and `bufferSize` are retained only for source compatibility:
this loader performs synchronous reads and has no worker pool or prefetch queue.
It loads one complete file before selecting the rank partition; it is not yet a
bounded-memory reader for arbitrarily large files. `BatchIterator` used directly
is a finite single-shard iterator and returns `none` for an incomplete final
batch; use the distributed generator to preserve tails across files.
Batch dimensions and GPT's additional target token are checked before fixed-width
arithmetic. `BatchIterator.updateParams` and `updateForStepGPT` now return `IO`
so invalid parameter updates report errors before consumption.

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
and returns optional pairs of `DTensor #[b, s] .Int64`, shifted by one token.
It lives in `Examples.GPT.GPTDataLoader`, alongside `BatchIterator.nextGPT`
and the distributed GPT batching and scheduling helpers. These GPT helpers
retain both the input and target dimensions and their Int64 dtype.

### Parquet pretraining loaders

`torch.Data.Pretraining` (`Tyr/Data/Pretraining.lean`) binds Apache Arrow
through `@[extern]` declarations implemented in `cc/src/tyr_parquet.cpp` (with
stubs when Arrow is absent): `listParquetFiles`, `getParquetMetadata`,
`readRowGroup`, `readParquetAsJson`, `readRowGroupAsJson`
(`Pretraining.lean:48-74`). On top sit `RowGroupIterator` (rank-strided row
groups, `:189`), a `TokenBuffer` ring, and `StreamingLoaderState` (`:405`),
which tokenizes documents into windows with approximate resume via `LoaderState`
checkpoints. `lossToBPB` (`:368`) converts validation loss to bits-per-byte
using a `TokenBytes` table.

`torch.Data.Download` shells out to `curl`/`unzip` with retry/backoff
(`downloadWithRetry`, `Tyr/Data/Download.lean:60`) and resolves HF parquet
URLs (`ensureHFParquet`, `:187`). `torch.Data.HuggingFace` turns parquet into
`Lean.Json` rows and ships per-dataset loaders (`loadARC`, `loadGSM8K`,
`loadMMLU`, `loadSmolTalk`; `Tyr/Data/HuggingFace.lean:59-167`).

### Tasks, conversations, and mixtures

`torch.Data.Task` (`Tyr/Data/Task.lean`) models chat data:
`Conversation = Array Message` with `Role` (system/user/assistant/tool) and
optional structured `ContentPart`s (text/code/toolCall/toolResult).
`renderConversation` (`Task.lean:236`) is the loss-mask heart:

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
`ConversationMixture { size, seed, getAtSeed }` (`Task.lean:179`):

- `TaskMixture` (`Task.lean:125`) — closed, weighted, deterministically
  shuffled mixture over `LoadedTask`.
- `EvalTask` typeclass + `BoxedTask` existential + `GenericTaskMixture`
  (`Tyr/Data/TaskClass.lean:57,111,165`) — the open extension point; add your
  own task type with an `EvalTask` instance, box it with `boxTask`, combine
  with `entry`/`GenericTaskMixture.create`.
- Built-in toy tasks in `Tyr/Data/Tasks.lean` (identity, math, spelling,
  multiple choice) and mixture presets `createMidtrainingMixture` /
  `createSFTMixture` (`Tasks.lean:188,197`).

Consumers: `TaskIterator` (`Task.lean:391`) yields padded
`[batch, maxLen]` batches with masks; `TaskTokenStream` (`Task.lean:454`)
streams rank-strided rendered conversations into a token buffer and emits
contiguous `batchSize * seqLen + 1` windows, split into `(inputs, targets)` by
one-token shift — the nanochat midtraining data flow.

### Multi-stage pipeline scheduling

`torch.Data.Pipeline` (`Tyr/Data/Pipeline.lean`) is pure configuration and
scheduling math — no data movement. `Stage` is
`pretraining | midtraining | sft | rl` (`:30`); each stage has a
`StageLRConfig` (per-parameter-group base LRs, warmup/warmdown ratios) and a
`StageConfig` (steps/epochs, batch size, intervals, weight decay) with nanochat
defaults (`pretrainingConfig`, `sftConfig`, ...). `getStageLRMultiplier`
(`:187`) computes the warmup/plateau/warmdown multiplier,
`computeStepLRs` (`:216`) applies it with square-root batch-size scaling, and
`calculateIterations` (`:246`) derives step counts from a `TrainingDuration`
(iterations/FLOPs/param-data ratio/epochs). `Pipeline.standard` (`:317`)
assembles a pretraining → midtraining → SFT sequence.

### Streaming text utilities

`Tyr.Text` (`Tyr/Text/`) serves streaming ASR, not training:

- `updateWithSignals` (`Tyr/Text/StreamingConsensus.lean:72`) stabilizes
  streaming hypotheses into an append-only stable prefix plus a mutable tail,
  combining prefix agreement with the previous hypothesis, k-window history
  consensus, freeze and VAD-boundary signals.
- `SileroProvider` (`Tyr/Text/VADProvider.lean:10`) wraps the Silero VAD
  model (`Tyr.Model.SileroVAD`), consuming 16 kHz audio in 512-sample chunks
  and emitting the `VADSignal`s (`speechActive`, `boundary`) that feed
  `updateWithSignals`.

## Key APIs

### Tokenizer core — `namespace tokenizer`

| API | Signature | Location |
| --- | --- | --- |
| `trainBPE` | `(docs : Array String) (config : TrainConfig) (log : Handlers := {}) : IO TrainResult` | `Tyr/Tokenizer/Training.lean:211` |
| `save` / `load` | `(tok : BPETokenizer) (path : String) : IO Unit` / `(path : String) : IO BPETokenizer` | `Tyr/Tokenizer/IO.lean:190,197` |
| `encode` / `encodeWithSpecials` | `(tok : BPETokenizer) (text : String) : Array TokenId` | `Tyr/Tokenizer/Encode.lean:143,156` |
| `decode` / `decodeToBytes` | `(tok : BPETokenizer) (ids : Array TokenId) : String` / `: ByteArray` | `Tyr/Tokenizer/Decode.lean:29,36` |
| `countTokens` | `(tok : BPETokenizer) (text : String) : Nat` | `Tyr/Tokenizer/Encode.lean:176` |
| `TrainConfig` | `{ vocabSize := 32768, maxChars, docCap, splitPattern, specialTokens, seed }` | `Tyr/Tokenizer/Training.lean:41` |
| `defaultChatSpecialTokens` | `Array String` (nanochat chat token set) | `Tyr/Tokenizer/Training.lean:28` |
| `createBase` | `BPETokenizer` (256 byte tokens + all special tokens) | `Tyr/Tokenizer/IO.lean:204` |
| `TokenBytes.fromTokenizer` / `.totalBytes` | `BPETokenizer → TokenBytes` / `TokenBytes → Array UInt32 → IO UInt64` | `Tyr/Tokenizer/TokenBytes.lean:32,50` |

### HF codecs — `tokenizer.qwen3` / `qwen35` / `qwen36` / `gemma4`

| API | Signature | Location |
| --- | --- | --- |
| `qwen3.loadTokenizer` | `(dir : String) : IO QwenTokenizer` | `Tyr/Tokenizer/Qwen3.lean:82` |
| `qwen3.encodeText` / `decodeText` | `(tok) (text : String) : Array TokenId` / `(tok) (ids) : String` | `Qwen3.lean:530,541` |
| `qwen3.chatTemplate` | `(prompt : String) : String` | `Qwen3.lean:260` |
| `qwen3.encodePrompt` | `(tok) (prompt) (maxLen : Nat := 512) : Array TokenId × Array TokenId` (tokens + attention mask) | `Qwen3.lean:566` |
| `qwen3.ttsAssistantText` / `ttsRefText` / `ttsInstructText` | `String → String` | `Qwen3.lean:264-272` |
| `qwen35.chatTemplateThinking`, `userPrefix`, `assistantGenerationSuffix` | `String → String` / `String` | `Tyr/Tokenizer/Qwen35.lean:24-38` |
| `gemma4.loadTokenizer` | `(dir : String) : IO GemmaTokenizer` | `Tyr/Tokenizer/Gemma4.lean:85` |
| `gemma4.encodeText` / `decodeText` / `chatTemplate` / `chatTemplateThinking` | as Qwen3 | `Gemma4.lean:362,389,235,238` |

`qwen35` re-exports the qwen3 codec as `abbrev`s (`loadTokenizer`,
`encodeText`, ...); `qwen36` re-exports `qwen35`.

### Shard loading — `namespace torch.DataLoader`

| API | Signature | Location |
| --- | --- | --- |
| `Config` | `{ dataPath, valPath, seqLen, bosToken, seed, shuffle }`; synchronous compatibility fields `numWorkers`, `bufferSize` | `Tyr/DataLoader.lean` |
| `resolveShardPaths` | `(pathSpec : String) (kind : ShardKind) : IO (Array String)` | `Tyr/DataLoader.lean:75` |
| `DataShard.loadFromFile` | `(path) (shardIdx numShards bosToken : UInt64) : IO (Σ n, DataShard n)` | `Tyr/DataLoader.lean:239` |
| `DataShard.load` / `loadValidationData` | Return `IO (Σ n, DataShard n)` with the real partition length | `Tyr/DataLoader.lean` |
| `DistributedDataGenerator.cursor` / `.restoreCursor` | Versioned `StreamCursor` capture and validated restore | `Tyr/DataLoader.lean` |
| `BatchIterator.new` / `.next` | `(shard) (batchSize seqLen : UInt64) : BatchIterator` / `IO (Option (DTensor #[iter.batchSize, iter.seqLen] .Int64) × BatchIterator)` | `Tyr/DataLoader.lean:288,291` |
| `DistributedDataGenerator.init` | `(config : Config) (batchSize seqLen : UInt64) : IO DistributedDataGenerator` | `Tyr/DataLoader.lean:323` |
| `DistributedDataGenerator.nextBatch` | `IO (Option (DTensor #[gen.iterator.batchSize, gen.iterator.seqLen] .Int64) × DistributedDataGenerator)` | `Tyr/DataLoader.lean:333` |
| `BatchIterator.nextGPT` | `IO (Option (DTensor #[iter.batchSize, iter.seqLen] .Int64 × DTensor #[iter.batchSize, iter.seqLen] .Int64) × BatchIterator)` (input/target shift) | `Examples/GPT/GPTDataLoader.lean:23` |
| `DistributedDataGenerator.nextBatchGPT` / `updateForStepGPT` | GPT batch + modded-nanogpt schedule | `Examples/GPT/GPTDataLoader.lean:60,84` |
| `SequentialLoader.fromFile` / `SequentialBatchIterator.new` / `.next` | fixed-dim Shakespeare-style loading, `DTensor #[b, s] .Int64` batches | `Examples/GPT/GPTDataLoader.lean:118,159,166` |
| `loadShakespeareData` | `(trainPath valPath) (batchSize seqLen) : IO (Σ n, SequentialBatchIterator n .. × Option (Σ m, SequentialLoader m))` — one-shot Shakespeare setup | `Examples/GPT/GPTDataLoader.lean:188` |

### Parquet / downloads — `torch.Data.Pretraining`, `torch.Data.Download`, `torch.Data.HuggingFace`

| API | Signature | Location |
| --- | --- | --- |
| `listParquetFiles`, `getParquetMetadata`, `readRowGroup`, `readParquetAsJson`, `readRowGroupAsJson` | `@[extern]` opaque IO functions | `Tyr/Data/Pretraining.lean:48-74` |
| `RowGroupIterator.init` / `.next` | `(dataPath) (rank worldSize : UInt64) (log := {})` / `IO (Option RowGroupData × _)` | `Pretraining.lean:189,229` |
| `StreamingLoaderState.init` / `.nextBatch` / `.getCheckpoint` / `.resume` | streaming windows with resume | `Pretraining.lean:419,496,534,538` |
| `lossToBPB` | `(loss : Float) (numTokens numBytes : Nat) : Float` | `Pretraining.lean:368` |
| `downloadWithRetry` | `(url dest : String) (maxRetries := 5) (initialBackoffMs := 1000) (log := {}) : IO Bool` | `Tyr/Data/Download.lean:60` |
| `ensureHFParquet` | `(repoId subset split : String) (cacheDir ..) : IO String` | `Download.lean:187` |
| `loadARC` / `loadGSM8K` / `loadMMLU` / `loadSmolTalk` | `... : IO (Array Lean.Json)` | `Tyr/Data/HuggingFace.lean:59,67,75,167` |
| `loadJsonlFromUrl` / `loadJsonlFromFile` | `... : IO (Array Lean.Json)` | `HuggingFace.lean:179,195` |

### Tasks and scheduling — `torch.Data.Task`, `torch.Data.TaskClass`, `torch.Data.Tasks`, `torch.Data.Pipeline`

| API | Signature | Location |
| --- | --- | --- |
| `Message.user` / `.assistant` / `.system` | `(content : String) : Message` | `Tyr/Data/Task.lean:59-61` |
| `ChatTokens` | special-token id record for rendering | `Task.lean:204` |
| `renderConversation` | `(conv) (tokens : ChatTokens) (encode : String → Array UInt64) : TokenizedConversation` | `Task.lean:236` |
| `collate` | `(convs) (maxLen : Nat) (padToken := 0) : IO ConversationBatch` | `Task.lean:352` |
| `TaskIterator.new` / `.nextBatch` | padded masked batches from a `ConversationMixture` | `Task.lean:400,409` |
| `TaskTokenStream.new` / `.nextGPTBatch` | rank-strided token windows; indexed Int64 input/target pairs with the stream dimensions | `Task.lean:470,521` |
| `TaskMixture.create` / `.toConversationMixture` | `(entries : Array MixtureEntry) (seed := 42)` | `Task.lean:153,193` |
| `EvalTask` class / `boxTask` / `entry` / `GenericTaskMixture.create` | open task typeclass + existential mixture | `Tyr/Data/TaskClass.lean:57,119,237,172` |
| `createMidtrainingMixture` / `createSFTMixture` | `(modelName : String) : TaskMixture` | `Tyr/Data/Tasks.lean:188,197` |
| `getStageLRMultiplier` / `computeStepLRs` | warmup/plateau/warmdown LR math | `Tyr/Data/Pipeline.lean:187,216` |
| `calculateIterations` / `Pipeline.standard` | duration → steps; 3-stage pipeline | `Pipeline.lean:246,317` |

### Streaming text — `namespace Tyr.Text`

| API | Signature | Location |
| --- | --- | --- |
| `updateWithSignals` | `(st : ConsensusState) (ids : Array UInt32) (speechActive := true) (boundary := false) (decode : Array UInt32 → String) : ConsensusState × TextDelta` | `Tyr/Text/StreamingConsensus.lean:72` |
| `update` | signal-free wrapper | `StreamingConsensus.lean:138` |
| `initSileroProvider` | `(weightsPath : String) (threshold := 0.5) (minSilenceDurationMs := 100) (speechPadMs := 30) : IO SileroProvider` | `Tyr/Text/VADProvider.lean:29` |
| `stepSileroProvider` | `(p) (pcm16k : Array Float) : IO (SileroProvider × VADSignal)` | `VADProvider.lean:40` |

## Usage examples

Train, persist, and use a BPE tokenizer — reconstructed example (from
`Examples/NanoChat/Pipeline.lean:668-714`):

```lean
import Tyr.Tokenizer

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
`Tests/TestDataLoader.lean:199-214`, `Examples/NanoChat/ModdedTrain.lean:755`):

```lean
import Tyr.DataLoader
open torch torch.DataLoader

def nextTokenCount (gen : DistributedDataGenerator) : IO Nat := do
  let (batch?, _) ← gen.nextBatch
  match batch? with
  | none => pure 0
  | some batch =>
    let indexed : DTensor #[gen.iterator.batchSize, gen.iterator.seqLen] .Int64 := batch
    pure (← indexed.toUInt64Array).size
```

Create a generator with `DistributedDataGenerator.init cfg batchSize seqLen`.
With `import Examples.GPT.GPTDataLoader`, `gen.nextBatchGPT` returns indexed
input/target pairs separated by one token.


Fixed-dim Shakespeare batches — reconstructed example (from
`Tests/TestDataLoader.lean:92-126`, definitions in
`Examples/GPT/GPTDataLoader.lean:153-198`):

```lean
import Examples.GPT.GPTDataLoader
open torch.DataLoader

let ⟨n, loader⟩ ← SequentialLoader.fromFile "data/shakespeare_char/train.bin"
let iter : SequentialBatchIterator n 4 32 := SequentialBatchIterator.new loader 4 32
let (pair?, iter') := iter.next    -- pair? : Option (DTensor #[4, 32] .Int64 × DTensor #[4, 32] .Int64)
```

Midtraining/SFT token stream over a task mixture — reconstructed example
(from `Examples/NanoChat/Pipeline.lean:1408-1424`):

```lean
import Tyr.Tokenizer
import Tyr.Data
open torch.Data.Task

let tok ← tokenizer.load tokenizerFile
let chatTokens ← buildChatTokensFromTokenizer tok   -- helper: look up special ids in `tok`
let encode := fun (text : String) => (tokenizer.encodeWithSpecials tok text).map (·.toUInt64)
let mixture := (← createMidtrainTaskMixture cfg).toConversationMixture
let stream := TaskTokenStream.new mixture batchSize seqLen chatTokens encode rank worldSize
let (pair?, stream') ← stream.nextGPTBatch          -- optional indexed Int64 input/target pair
```

HF tokenizer for inference — reconstructed example (from
`Examples/Qwen35/RunHF.lean:97-105`):

```lean
import Tyr.Tokenizer

let tok ← tokenizer.qwen35.loadTokenizer modelDir
let text := tokenizer.qwen35.chatTemplate prompt    -- or chatTemplateThinking
let ids := (tokenizer.qwen35.encodeText tok text).map (·.toUInt64)
let out := tokenizer.qwen35.decodeText tok generatedIds
```

## Caveats

Behavior verified against source that the module docs do not advertise:

- `trainBPE` never pretokenizes and ignores `TrainConfig.{maxChars, docCap,
  splitPattern, seed}` (`Tyr/Tokenizer/Training.lean:211-278`).
- `PretrainingLoader.nextBatch` / `ValidationLoader.nextBatch` are incomplete
  (the refill path returns `none`), and `tokenizeDocuments` is a byte-level
  placeholder (`Tyr/Data/Pretraining.lean:298-356,445`); the NanoChat pipeline
  drives the parquet FFI declarations directly.
- `Tasks.loadJsonlTask` / `loadHFDataset` are stubs returning empty tasks
  (`Tyr/Data/Tasks.lean:208,215`).
- `allSpecialTokens` mixes control tokens with ordinary words ("theorem",
  "def", Greek letters); via `encodeWithSpecials` these match atomically
  anywhere in text (`Tyr/Tokenizer/SpecialTokens.lean:53-99`).

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
