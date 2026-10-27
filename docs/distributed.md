# Distributed training and orchestration

Tyr's distributed stack covers four layers: raw c10d collectives (`Tyr.Distributed`),
type-level shard arithmetic (`Tyr.Sharding`), staged pipeline orchestration with
checkpoint/resume (`Tyr.Pipeline`), and reusable training-loop recipes
(`Tyr.Train.ChatSFT`, `Tyr.RL.GRPO`) that write to a JSONL run ledger
(`Tyr.Train.RunLedger`). Use it when one GPU is not enough — data-parallel
training with averaged gradients is the well-trodden path — or when you want a
resumable, multi-stage training run with structured artifacts. All of it is
exercised end-to-end by the NanoChat example (`Examples/NanoChat/Pipeline.lean`).

## Architecture and main abstractions

```
Tyr.RL.GRPO        Tyr.Train.ChatSFT            -- recipe training loops
        \          /
         Tyr.Train.RunLedger                    -- config.json / metrics.jsonl / checkpoints.jsonl
         Tyr.Pipeline                           -- PipelineM: stages, retry, resume, report
Tyr.Sharding       Tyr.Distributed              -- ShardedTensor over collectives
                   cc/src/tyr_distributed.cpp   -- c10d: NCCL / Gloo / TCPStore
```

The layers compose bottom-up; each one is usable independently. All Lean
modules are re-exported from the umbrella `Tyr.lean` (`Tyr.lean:30-54`) and
live in `torch.*` namespaces. The C++ side is a single global
`c10d::ProcessGroup` plus a `TCPStore` in `cc/src/tyr_distributed.cpp`; NCCL
is compile-time gated (`USE_C10D_NCCL`) and Gloo is always built. `deps/fetch.sh`
fetches the Gloo headers at the exact submodule revision used by the pinned
PyTorch wheel, and the native build stops if they are missing. Custom LibTorch
builds must also provide a compatible Gloo implementation. Async
work items live in a global map keyed by `UInt64` work ids.

### Collectives: `torch.dist` (`Tyr/Distributed.lean`)

Process-group lifecycle is explicit:

```lean
opaque initProcessGroup (backend masterAddr : @& String) (masterPort rank worldSize : UInt64) : IO Unit
opaque setCudaDevice (device : UInt64) : IO Unit
opaque getRank : IO UInt64
opaque getWorldSize : IO UInt64
opaque isInitialized : IO Bool
opaque destroyProcessGroup : IO Unit
opaque barrier : IO Unit
```

Collectives preserve input values and return independent results. Private native
collectives write only into fresh effectfully allocated buffers. `ReduceOp` is
`sum | avg | product | min | max`. An asynchronous reduction returns an awaiting
action; execute that action once to wait and obtain the completed tensor. A tensor
whose storage is still being written is never exposed to the caller.

```lean
def allReduce {σ : StaticSpec} (tensor : Tensor σ) (op : ReduceOp := .sum) : IO (Tensor σ)
def allReduceAsync {σ : StaticSpec} (tensor : Tensor σ) (op : ReduceOp := .sum) : IO (IO (Tensor σ))
def broadcast {σ : StaticSpec} (tensor : Tensor σ) (srcRank : UInt64) : IO (Tensor σ)
def reduceScatter {σ : StaticSpec} (worldSize : UInt64) (input : Tensor σ)
    (op : ReduceOp := .sum)
    : IO (Tensor (σ.withShape (σ.shape.set! 0 (σ.shape.getD 0 0 / worldSize))))
def allGather {σ : StaticSpec} (worldSize : UInt64) (input : Tensor σ)
    : IO (Tensor (σ.withShape (σ.shape.set! 0 (σ.shape.getD 0 0 * worldSize))))

structure WorkHandle where id : UInt64
def wait (handle : WorkHandle) : IO Unit
def WorkHandle.isCompleted (handle : WorkHandle) : IO Bool
def waitAll (handles : Array WorkHandle) : IO Unit
```

For explicit work handles, `isCompleted` only observes successful completion. It
returns false while pending and reports failed work or an unknown handle as an
error. Polling retains the handle, including after an error, so `wait` or teardown
can drain it. Call `wait` after a successful poll to establish backend stream
synchronization and release the handle. Waiting consumes a handle even when the
operation fails or is aborted; a second wait is an error. `waitAll` attempts every
handle before reporting the first error.

For whole models, TensorStruct-aware helpers map the collective over every
leaf (`Tyr/Distributed.lean:280-311`) — the functions most loops call:

```lean
def broadcastParams [TensorStruct α] (params : α) : IO α          -- from rank 0
def allReduceGrads [TensorStruct α] (grads : α) (op := .avg) : IO α
def allReduceGradsAsync [TensorStruct α] (grads : α) (op := .avg) : IO (IO α)
```

The standard data-parallel recipe: `broadcastParams` once after init, local
forward/backward per rank, `allReduceGrads .avg` after backward, one
identical optimizer step on every rank using the returned gradient tree (see
`Examples/NanoChat/ModdedTrain.lean:1167-1196`).

`Tests/RunDistributedValues.lean` checks retained inputs and earlier outputs,
repeated list collectives, actual integer dtypes behind dynamic specifications,
trainability flags, and async completion on two CPU ranks:

```sh
deps/fetch.sh cpu
lake build --wfail RunDistributedValues
lake env python3 scripts/test_distributed_values.py
```

Use `deps/fetch.sh cuda` instead when the checkout has a CUDA configuration.
The runner selects a localhost rendezvous port, starts both ranks, and terminates
and reaps its remaining processes if either rank fails or the 60-second deadline
expires. Per-rank logs and `report.json` go to `output/ci/distributed`; `--log-dir`,
`--timeout`, and `--port` can override these defaults. CPU CI runs this numerical
check on Linux and macOS. It qualifies Gloo CPU collectives, not multi-GPU NCCL
behavior.

`dist.withDistributed` brackets init/action/teardown for a single process group.
After successful initialization it runs teardown even if the action fails,
preserving the action's error when cleanup also fails. Teardown drains every
pending work item and resets process-group state before reporting its first
failure. Invalid initialization arguments fail before rendezvous, and process-group
state is published only after native construction succeeds. A
`DistributedSampler` (`Tyr/Distributed.lean:318-423`) shards a dataset
deterministically per rank with an LCG-seeded Fisher–Yates shuffle; the
examples currently shard data ad hoc instead.

`Tyr/Distributed.lean` also hosts the Polar Express / Muon FFI kernels
(`polarExpress`, `muonOrthogonalize`, `newtonSchulzStep`, `cautiousUpdate`,
native declarations in `Tyr/RawTensor/Distributed.lean`) — orthogonalization math, not communication;
see [Optimization](optimization.md).

### Sharding: `torch.Sharding` (`Tyr/Sharding.lean`)

Sharding is expressed at the type level: `shardedShape` projects a compile-time
`Shape` to this rank's shard shape, so a `ShardedTensor`'s local data has a
statically known size.

```lean
inductive ShardDim where first | last | dim (n : Nat)
structure ShardSpec where shardDim : ShardDim := .first; numShards shardIdx : UInt64

def shardSize   (fullSize numShards shardIdx : UInt64) : UInt64   -- first `remainder` shards get +1
def shardOffset (fullSize numShards shardIdx : UInt64) : UInt64
def shardedShape (s : Shape) (spec : ShardSpec) : Shape

abbrev ValidShard (rank worldSize : UInt64) := rank < worldSize

structure ShardedTensor (σ : StaticSpec) (rank worldSize : UInt64)
    (shardDim : ShardDim := .first) where
  shard : Tensor (σ.withShape (shardedShape σ.shape ⟨shardDim, worldSize, rank⟩))
  cachedFull : Option (Tensor σ) := none
  cacheIsStale : Bool := true
```

Operations retain the full tensor's dtype and concrete device:

- `ShardedTensor.fromFull full proof shardDim` slices a `Tensor σ` and checks
  the local shard metadata. The proof establishes `rank < worldSize`.
- `sharded.gather` returns `IO (Tensor σ × ShardedTensor σ rank worldSize sd)`
  and caches the gathered value. A fresh collective requires equal-sized shards.
- `sharded.scatterGrad fullGrad` reduces/scatters a `Tensor σ` and checks the
  result against `σ.withShape (shardedShape σ.shape ⟨sd, worldSize, rank⟩)`.
- `sharded.updateShard newShard` accepts that same indexed local specification
  and marks the cache stale.

This is the ZeRO pattern: keep one shard of each parameter (and optimizer
state) per rank, `gather` lazily for the forward pass, `scatterGrad` the
gradients back after backward. `MaybeSharded` (`Tyr/Sharding.lean:192`) models
a parameter that is either sharded or replicated, with `updateWithGrad`
choosing all-reduce vs reduce-scatter accordingly. The production consumer is
`torch.Optim.DistAdam` (`Tyr/Optim/DistAdam.lean`); `ShardedEmbedding` and
`ShardedAdamState` here are sketches.

Caveats: `fromFull` and `gather` slice and reassemble along dimension 0
regardless of `shardDim` (`Tyr/Sharding.lean:122-130`), so non-`.first`
shardings are not honored at runtime, and `shardedShape` silently returns the
unsharded shape for cases it does not handle (`Tyr/Sharding.lean:81`).

### Orchestration: `torch.Pipeline` (`Tyr/Pipeline.lean`)

The pipeline monad sequences named stages with status tracking, optional retry, and disk checkpointing:

```lean
abbrev PipelineM := StateT PipelineState IO

structure PipelineConfig where
  baseDir : String := "~/.cache/tyr"
  numGpus : Nat := 1
  resumeFromCheckpoint : Bool := true
  retryPolicy : RetryPolicy := {}
  checkpointAfterStage : Bool := true
  failFast : Bool := true
  runId : String := ""
  -- plus wandbEnabled / wandbRun, currently unused
```

Stage API:

```lean
def stage (name : String) (action : PipelineM Unit) : PipelineM Unit
def stageWithRetry (name : String) (policy : RetryPolicy) (action : PipelineM Unit) : PipelineM Unit
def recordMetrics (metrics : List (String × String)) : PipelineM Unit
def log / logError (msg : String) : PipelineM Unit        -- master rank only
def masterOnly (action : PipelineM α) (default : α) : PipelineM α
def syncBarrier : PipelineM Unit
def checkAllRanksHealthy : PipelineM Bool
```

Each `stage` records a `StageInfo` (status `pending | running | completed |
failed error`, timestamps, metrics, retry count) and, on master, appends a
ledger event under scope `stage/<name>` (`Tyr/Pipeline.lean:660-675`). With
`checkpointAfterStage`, stage results are written to
`<baseDir>/.pipeline_checkpoint.json` after every stage; on startup, completed
stages are skipped. Resume is refused when the saved `repr config` does not
match the current config (`Tyr/Pipeline.lean:707-718`), so
`resumeFromCheckpoint := true` is safe against config drift. On clean
completion the checkpoint file is removed.

Entry points:

```lean
def runPipeline (config : PipelineConfig) (action : PipelineM α) : IO α
def runPipelineWithHandlers (config) (logHandlers : LogHandlers := {}) (action) : IO α
def withDistributed (config : PipelineConfig) (action : PipelineM α) : IO α
def withDistributedWithHandlers (config) (logHandlers := {}) (action) : IO α
```

`Pipeline.withDistributed` (not to be confused with `dist.withDistributed`)
reads the torchrun environment — `WORLD_SIZE`, `RANK`, `LOCAL_RANK`,
`MASTER_ADDR`, `MASTER_PORT` — and initializes NCCL only when `WORLD_SIZE > 1`
(`Tyr/Pipeline.lean:804-823`). `resolveTrainingDevice`
(`Tyr/Pipeline.lean:306`) honors `TYR_DEVICE=cpu|mps|cuda|auto` and maps
`LOCAL_RANK` to the CUDA ordinal. Logging is silent unless you pass
`LogHandlers` (`onInfo`/`onError` callbacks, `Tyr/Pipeline.lean:373`).

There is a background-task API (`background`, `await`, `backgroundTracked`,
`awaitTrackedOrThrow`, `Tyr/Pipeline.lean:580-625`). `spawnBackground`
(`Tyr/Pipeline.lean:154-163`) starts the action immediately via `IO.asTask`
and defers the wait to `await`, so these tasks run concurrently. At the end,
`finalizePipeline` writes `report.md` via the `Report` accumulator
(`Tyr/Pipeline.lean:216-231`).

### Run ledger: `torch.Train.RunLedger` (`Tyr/Train/RunLedger.lean`)

A Tinker-style artifact layout shared by the recipe loops —
`<baseDir>/{config.json, metrics.jsonl, checkpoints.jsonl, checkpoints/, report.md}`:

```lean
structure RunArtifacts where baseDir; configPath; metricsPath; checkpointsPath; checkpointsDir; reportPath
def RunArtifacts.ofBaseDir (baseDir : String) : RunArtifacts

inductive ExistingRunPolicy where resume | overwrite | failIfExists
def prepare [Lean.ToJson α] (artifacts : RunArtifacts) (config : α) (policy := .resume) : IO Unit

def appendMetricEvent (artifacts) (event : MetricEvent) : IO Unit        -- one JSON line
def appendCheckpointEvent (artifacts) (event : CheckpointEvent) : IO Unit
```

`MetricEvent` carries a `scope`, optional `step`, and `MetricFields` built
with `metricStr` / `metricFloat` / `metricNat` / `metricUInt64` / `metricBool`;
a zero `timestampMs` is filled with `IO.monoMsNow`. Evaluator scheduling:

```lean
structure EvalSchedule where every : Nat := 0; runAtStart : Bool := false; runAtEnd : Bool := true
structure Evaluator (α : Type) where name : String; schedule : EvalSchedule; run : α → IO MetricFields
def runDueEvaluators (evaluators : Array (Evaluator α)) (ctx : α) (step : Nat) (isLastStep := false)
    : IO MetricFields   -- metrics prefixed with "<name>/"
```

### Chat SFT: `torch.Train.ChatSFT` (`Tyr/Train/ChatSFT.lean`)

A nanochat-style supervised fine-tuning loop: masked cross-entropy over
assistant tokens only, gradient accumulation, linear LR decay, one AdamW
step per iteration. Each batch carries its runtime dimensions as fields, and
its tensor types depend on those fields:

```lean
variable {device : Device}

structure SFTBatch where
  batchSize seqLen : UInt64
  inputs targets : Tensor { shape := #[batchSize, seqLen], dtype := .Int64, device := device }
  mask : Tensor { shape := #[batchSize, seqLen], dtype := .Float32, device := device }
  numValidTokens : Nat

def maskedCrossEntropy {batch seq vocab : UInt64} {d m result : DType} {dev : Device}
    [DType.Promotes d m result]
    (logits : Tensor ⟨#[batch, seq, vocab], d, dev⟩)
    (targets : Tensor ⟨#[batch, seq], .Int64, dev⟩)
    (mask : Tensor ⟨#[batch, seq], m, dev⟩)
    : Tensor ⟨#[], (DType.promote d m).sumResult, dev⟩
```

The main entry is generic over the parameter tree:

```lean
def trainLoop {device : Device} [TensorStruct P]
    (cfg : SFTConfig) (params : P) (optState : Optim.AdamWState P)
    (forwardFn : P → {b s : UInt64} → Tensor { shape := #[b, s], dtype := .Int64, device := device } →
      IO (Σ v : UInt64, Tensor { shape := #[b, s, v], dtype := .Float32, device := device }))
    (lossFn : {b s v : UInt64} → Tensor { shape := #[b, s, v], dtype := .Float32, device := device } →
      Tensor { shape := #[b, s], dtype := .Int64, device := device } → Tensor { shape := #[b, s], dtype := .Float32, device := device } → Tensor { shape := #[], dtype := .Float32, device := device })
    (trainDataFn : IO (SFTBatch (device := .CPU)))
    (valDataFn : Option (IO (SFTBatch (device := .CPU))) := none)
    (numTrainExamples : Nat := 10000) (callbacks : Callbacks := {})
    : IO (P × Optim.AdamWState P × SFTState)
```

`SFTConfig` (`Tyr/Train/ChatSFT.lean:30`) holds epochs/iterations, batch
geometry (`deviceBatchSize`, `targetExamplesPerStep` — `gradAccumSteps` is
derived per world size), LRs, `maxSeqLen`, eval cadences, `device`, and
`logInterval`. The loop checks that `cfg.device` matches the forward function’s
concrete device, then moves CPU data batches to that device. Inside the loop (`Tyr/Train/ChatSFT.lean:404-473`):
`gradAccumSteps` micro-batches accumulate gradients, averaged via
`dist.allReduceGrads .avg` when distributed, then one `Optim.adamw` step at
`cfg.embeddingLr * cfg.initLrFrac * linearLRMultiplier ...`. Note the loop
uses one AdamW over the whole tree — the `matrixLr`, `unembeddingLr`,
`weightDecay`, and `gradClip` fields are not consumed by `trainLoop`, despite
the file header advertising a dual Muon/Adam optimizer.

Data plumbing: `makeTaskDataGenerator (iter : TaskIterator) (maxLen)
(padTokenId) : IO (IO (SFTBatch (device := .CPU)))` adapts the task-mixture iterator from
`Tyr.Data.Task` (see [Data](data.md)); `prepareSFTBatch` builds the shifted
input/target/mask triple directly. `Callbacks` (`onTrainStep`, `onValidation`)
compose via `Callbacks.combine`; `artifactCallbacks artifacts "sft"` logs to
the ledger under `sft/train` and `sft/eval`.

### GRPO: `torch.RL.GRPO` (`Tyr/RL/GRPO.lean`)

A simplified, REINFORCE-like GRPO ported from nanochat's `chat_rl.py`: no
trust region, no importance weights, no clipping, advantage
`A = r - mean(r)` applied uniformly to all tokens of a sequence.

```lean
def computeAdvantages (rewards : Array Float) : Array Float
def computePGLoss {device : Device} (logProbs : Tensor { shape := #[b, t], dtype := .Float32, device := device })
    (advantages : Tensor { shape := #[b], dtype := .Float32, device := device }) (mask : Tensor { shape := #[b, t], dtype := .Float32, device := device })
    (numPasses : UInt64 := 1) (examplesPerRank : UInt64 := 1) : Tensor { shape := #[], dtype := .Float32, device := device }
```

Rewards are pluggable; `mathReward` parses the GSM8K `#### <answer>` marker,
`exactMatchReward` is a string compare. Rollout generation is callback-driven
— you supply `generateOneFn : Array UInt64 → IO UInt64` (context → next
token); `createRolloutBatch` samples `numSamples` completions per prompt and
scores them. `GRPOConfig.temperature`/`topK` are plumbed but never applied —
sampling is entirely your callback's business.

The update-capable training loop:

```lean
def trainOnPromptsWithUpdates (b t : UInt64) [TensorStruct P]
    (prompts : Array (Array UInt64 × String × String))   -- (tokens, text, answer)
    (generateOneFn : P → Array UInt64 → IO UInt64)
    (forwardFn : P → Tensor { shape := #[b, t], dtype := .Int64, device := .CPU } → Tensor { shape := #[b, t], dtype := .Int64, device := .CPU } → IO (Tensor { shape := #[b, t], dtype := .Float32, device := .CPU }))  -- (params, input, target) → per-token NLL
    (params : P) (optState : Optim.AdamWState P)
    (decodeFn : Array UInt64 → String) (rewardFn : String → String → Float)
    (config : GRPOConfig) (numEpochs : Nat := 1) (callbacks : Callbacks := {})
    : IO (P × Optim.AdamWState P × TrainResult)
```

Each step generates rollouts, accumulates the per-sample PG losses into one
backward, all-reduces gradients when distributed, and applies one AdamW step
at `config.matrixLr * getLrMultiplier ...` (`Tyr/RL/GRPO.lean:766-879`).
`grpoStepWithModelUpdate` currently requires `b = 1`. Evaluation is pass@k via
`computePassK`; `Callbacks` + `artifactCallbacks` mirror the SFT design.

## Key APIs

| Task | API | Where |
|---|---|---|
| Init/teardown a process group | `dist.initProcessGroup`, `dist.destroyProcessGroup`, `dist.setCudaDevice`, `dist.barrier` | `Tyr/Distributed.lean:61-86` |
| Rank/world queries | `dist.getRankAndWorldSize`, `dist.isMaster`, `dist.isInitialized` | `Tyr/Distributed.lean:269-277` |
| Sync parameters across ranks | `dist.broadcastParams` | `Tyr/Distributed.lean:280` |
| Average gradients across ranks | `dist.allReduceGrads grads .avg` | `Tyr/Distributed.lean:292` |
| Raw collectives | `dist.allReduce`, `dist.broadcast`, `dist.reduceScatter`, `dist.allGather` (+ `*Async`, `WorkHandle`) | `Tyr/Distributed.lean:103-188` |
| ZeRO-style sharding | `Sharding.ShardedTensor` (`fromFull`/`gather`/`scatterGrad`/`updateShard`), `shardedShape`, `ValidShard` | `Tyr/Sharding.lean:73-183` |
| Run a staged pipeline | `Pipeline.withDistributedWithHandlers`, `stage`, `recordMetrics`, `log`, `PipelineConfig.resumeFromCheckpoint` | `Tyr/Pipeline.lean:542,660,804` |
| Run artifacts | `RunLedger.RunArtifacts.ofBaseDir`, `prepare`, `appendMetricEvent`, `appendCheckpointEvent` | `Tyr/Train/RunLedger.lean:42-152` |
| Chat SFT | `ChatSFT.trainLoop`, `SFTConfig`, `makeTaskDataGenerator`, `artifactCallbacks` | `Tyr/Train/ChatSFT.lean:30-526` |
| RL fine-tuning | `GRPO.trainOnPromptsWithUpdates`, `GRPOConfig`, `mathReward`, `artifactCallbacks` | `Tyr/RL/GRPO.lean:39-991` |

## Usage example

Reconstructed example (from `Examples/NanoChat/Pipeline.lean:1660-1940,
2117`). The real entry point builds its config from environment variables;
this sketch keeps the essential shape — torchrun env drives distributed
setup, each stage runs on every rank, only rank 0 writes checkpoints:

```lean
import Tyr
open torch torch.Pipeline torch.dist

def main : IO Unit :=
  withDistributedWithHandlers pipelineCfg consolePipelineLogHandlers do
    stage "sft" do
      -- rank-sharded task mixture → batch generator
      let trainDataFn ← ChatSFT.makeTaskDataGenerator taskIterator sftCfg.maxSeqLen chatTokens.assistantEnd
      let callbacks := consoleSFTCallbacks.combine (ChatSFT.artifactCallbacks runArtifacts "sft")
      let optState := (Optim.adamw (lr := sftCfg.embeddingLr)).init params
      -- forwardFn/lossFn preserve batch dimensions; grads all-reduced inside when distributed
      let (params', optState', sftState) ← ChatSFT.trainLoop sftCfg
        params optState forwardFn lossFn trainDataFn none numTrainExamples callbacks
      if (← get).isMaster then
        recordMetrics [("total_tokens", toString sftState.totalTokens)]

    stage "rl" do
      -- per-rank prompt sharding: keep idx % worldSize == rank
      let rewardFn := fun expected response => (GRPO.mathReward expected response).reward
      let (params'', _, result) ← GRPO.trainOnPromptsWithUpdates 1 256
        gsm8kPrompts generateOneFn forwardNLLFn params' optState'
        decodeFn rewardFn grpoCfg numEpochs (GRPO.artifactCallbacks runArtifacts "grpo")
      log s!"best pass@1 = {result.bestPass1}"
```

Without the pipeline layer, the raw data-parallel pattern (from
`Examples/NanoChat/TrainNanoChat.lean:147-161`, `Examples/NanoChat/ModdedTrain.lean:1167-1186`):

```lean
if worldSize > 1 then
  if ← cuda_is_available then dist.setCudaDevice localRank
  dist.initProcessGroup "nccl" masterAddr masterPort rank worldSize
let params ← dist.broadcastParams params        -- identical starting point
-- per step: local forward/backward, then
let grads ← TensorStruct.grads params
let grads ← dist.allReduceGrads grads .avg      -- use the returned reduced values
-- ... identical optimizer step on every rank; teardown at the end:
if isDistributed then dist.barrier; dist.destroyProcessGroup
```

A lighter-weight path for tests — no distributed init, no report — is
`initPipeline cfg` then `(do stage "a" ...; stage "b" ...).run initial` (`Tests/TestPipeline.lean:120-142`).

## Running under torchrun

The binaries read torchrun's environment directly; launch them with
`torchrun --no_python` so env vars are set per process. The wrappers in
`scripts/nanochat/` handle the module stack:

```bash
./scripts/nanochat/run_train_torchrun.sh        # TrainNanoChat, NPROC_PER_NODE=2 default
NPROC_PER_NODE=4 ./scripts/nanochat/run_pipeline_torchrun.sh   # NanoChatPipeline
torchrun --standalone --nnodes=1 --nproc_per_node=8 --no_python \
  .lake/build/bin/TrainNanoChat --data data/fineweb10B --val data/fineweb_val
```

Script knobs (`TORCHRUN_BIN`, `NPROC_PER_NODE`, `SKIP_BUILD`, `TYR_DEVICE`,
`PIPELINE_EXE`, resume smoke test): see
[Getting started](getting-started.md#distributed-nanochat-scripts) and
`scripts/nanochat/ENV_INVENTORY.md`.

## Related guides

- [Getting started](getting-started.md) — torchrun wrappers, `TYR_DEVICE`, build prerequisites.
- [Tensors](core/tensors.md) — `Tensor σ`, `Shape`, `Device`, the ops used by the collectives.
- [TensorStruct](core/tensorstruct.md) — the traversal that powers `broadcastParams` / `allReduceGrads`.
- [Optimization](optimization.md) — `Optim.adamw` / `AdamWState` used by both recipe loops, and the Polar Express kernels hosted in `Tyr/Distributed.lean`.
- [Data](data.md) — `TaskIterator` and conversation batches behind `makeTaskDataGenerator`.
- [Serialization](serialization.md) — checkpoint files referenced by `checkpoints.jsonl`.
- [Examples and testing](examples-and-testing.md) — the NanoChat pipeline and `Tests/TestPipeline.lean` / `Tests/TestRunLedger.lean`.

For exhaustive, per-symbol documentation of every definition in these modules,
see the doc-gen4 API reference generated from `docbuild/`.
