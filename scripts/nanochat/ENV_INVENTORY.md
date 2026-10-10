# NanoChat Environment Variable Inventory

This inventory covers the current NanoChat launcher and executables:

- `scripts/nanochat/run_pipeline_torchrun.sh`
- `Examples/NanoChat/Pipeline.lean`
- `Examples/NanoChat/TrainNanoChat.lean`
- `Examples/NanoChat/RunChat.lean`

## 1) Launcher and Orchestration

Source: `scripts/nanochat/run_pipeline_torchrun.sh:14`

- Pager and shell behavior:
  - `LMOD_PAGER`
  - `MODULES_PAGER`
  - `PAGER`
- Build and binary selection:
  - `TORCHRUN_BIN`
  - `PIPELINE_EXE`
  - `SKIP_BUILD`
- Torchrun process topology:
  - `NPROC_PER_NODE`
- Mode toggles:
  - `QUICK_MODE_FLAG` (maps to `QUICK_MODE`)
  - `ENABLE_RL_FLAG` (maps to `ENABLE_RL`)
- Run directory and naming:
  - `NANOCHAT_USE_RUN_ID_DIR`
  - `NANOCHAT_RUN_ID`
  - `NANOCHAT_DIR`
- Device selection:
  - `TYR_DEVICE`
- Cache/symlink behavior:
  - `AUTO_REUSE_CACHE`
- Pipeline checkpoint file reset:
  - `CLEAR_PIPELINE_CHECKPOINT`

## 2) Core Pipeline Config (Global)

Sources:

- Launcher defaults: `scripts/nanochat/run_pipeline_torchrun.sh:73`
- Read by pipeline main: `Examples/NanoChat/Pipeline.lean:1795`

- Model and scaling:
  - `MODEL_DEPTH`
  - `MODEL_ASPECT_RATIO`
  - `MODEL_HEAD_DIM`
  - `MAX_SEQ_LEN`
  - `ROPE_BASE`
  - `PARAM_DATA_RATIO`
  - `VOCAB_SIZE`
- Data shaping:
  - `DATA_PATH`
  - `NUM_SHARDS`
  - `INITIAL_DATA_SHARDS`
  - `TOKENIZER_MAX_CHARS`
  - `TOKENIZER_DOC_CAP`
- Pipeline stages:
  - `SFT_EPOCHS`
  - `GRPO_NUM_SAMPLES`
  - `GRPO_MAX_NEW_TOKENS`
- Feature toggles:
  - `QUICK_MODE`
  - `ENABLE_RL`
- Logging:
  - `WANDB_RUN`
  - `WANDB_ENABLED`
- Distributed world size:
  - `WORLD_SIZE`

## 3) Data Path Resolution and Format Guards

Source: `scripts/nanochat/run_pipeline_torchrun.sh:122`

- Token format requirement and path inference:
  - `REQUIRE_BIN_DATA_PATHS`
  - `PRETRAIN_DATA_PATH`
  - `MIDTRAIN_DATA_PATH`

## 4) Pretraining

Sources:

- Launcher defaults: `scripts/nanochat/run_pipeline_torchrun.sh:94`
- Pipeline consumption: `Examples/NanoChat/Pipeline.lean:747`

- Schedules and batching:
  - `PRETRAIN_ITERS`
  - `PRETRAIN_EXTENSION_ITERS`
  - `PRETRAIN_DEVICE_BATCH_SIZE`
  - `PRETRAIN_TOTAL_BATCH_SIZE`
  - `PRETRAIN_VAL_INTERVAL`
  - `PRETRAIN_LOG_INTERVAL`
  - `PRETRAIN_CHECKPOINT_INTERVAL`
- Data/tokenizer controls:
  - `PRETRAIN_TEXT_COLUMN`
  - `PRETRAIN_TOKENIZER_BATCH_SIZE`
- Eval token budget:
  - `PRETRAIN_EVAL_TOKENS`

## 5) Midtraining

Sources:

- Launcher defaults: `scripts/nanochat/run_pipeline_torchrun.sh:105`
- Pipeline consumption: `Examples/NanoChat/Pipeline.lean:1083`

- Schedules and batching:
  - `MIDTRAIN_ITERS`
  - `MIDTRAIN_EXTENSION_ITERS`
  - `MIDTRAIN_DEVICE_BATCH_SIZE`
  - `MIDTRAIN_TOTAL_BATCH_SIZE`
  - `MIDTRAIN_VAL_INTERVAL`
  - `MIDTRAIN_LOG_INTERVAL`
  - `MIDTRAIN_CHECKPOINT_INTERVAL`
- Eval token budget:
  - `MIDTRAIN_EVAL_TOKENS`

## 6) SFT

Sources:

- Launcher defaults: `scripts/nanochat/run_pipeline_torchrun.sh:114`
- Pipeline consumption: `Examples/NanoChat/Pipeline.lean:1311`

- `SFT_EPOCHS`
- `SFT_DEVICE_BATCH_SIZE`
- `SFT_TARGET_EXAMPLES_PER_STEP`
- `SFT_MAX_EXAMPLES`

## 7) RL / GRPO

Sources:

- Launcher defaults: `scripts/nanochat/run_pipeline_torchrun.sh:119`
- Pipeline consumption: `Examples/NanoChat/Pipeline.lean:1506`

- Core GRPO config:
  - `GRPO_NUM_SAMPLES`
  - `GRPO_MAX_NEW_TOKENS`
  - `GRPO_EXAMPLES_PER_STEP`
- Runtime/eval throttles:
  - `GRPO_EVAL_EVERY`
  - `GRPO_LOG_EVERY`
  - `GRPO_EPOCHS`
  - `GRPO_MAX_PROMPTS`

## 8) Evaluation and Task-Data Overrides

Source: `Examples/NanoChat/Pipeline.lean:311`

- `CORE_MAX_EXAMPLES`
- `IDENTITY_CONVERSATIONS_PATH`

## 9) Distributed Runtime (Standalone Trainers)

Sources:

- `Examples/NanoChat/TrainNanoChat.lean:84`
- `Examples/NanoChat/Pipeline.lean:198`

- `LOCAL_RANK`
- `RANK`
- `WORLD_SIZE`
- `MASTER_ADDR`
- `MASTER_PORT`
- `TYR_DEVICE`

## 10) Chat Runtime

Source: `Examples/NanoChat/RunChat.lean:99`

- `NANOCHAT_DIR`
- `MODEL_DEPTH`
- `VOCAB_SIZE`
- `TYR_DEVICE`

## 11) Path Expansion

Sources:

- `Examples/NanoChat/Pipeline.lean:128`
- `Examples/NanoChat/RunChat.lean:133`

- `HOME`

---

## Proposed Organization for Collection

### A) Use one canonical, typed environment snapshot

Create a single resolved structure at runtime, grouped by domain:

- `runtime`
- `distributed`
- `model`
- `data`
- `pretrain`
- `midtrain`
- `sft`
- `rl`
- `eval`
- `logging`

This can be materialized as JSON and persisted per run.

### B) Directory layout

Use a layered config directory with explicit precedence:

- `config/nanochat/defaults.env`
- `config/nanochat/profiles/<profile>.env`
- `config/nanochat/local.env` (gitignored)
- `config/nanochat/runs/<run_id>/resolved_env.json`
- `config/nanochat/runs/<run_id>/resolved_env.env`

### C) Precedence and resume rules

For fresh runs:

1. defaults
2. profile
3. local override
4. process env override

For resume runs:

1. load checkpoint metadata (model config, hyperparameters, tokens, dataloader cursor)
2. load requested env stack
3. merge with explicit policy:
   - strict-match fields: model shape and other compatibility-critical fields
   - allowed override fields: iteration horizon fields such as extension iterations
4. write merged result to `resolved_env.json` and log it

This matches existing resume behavior in `Examples/NanoChat/ModdedTrain.lean:592` and
`Examples/NanoChat/ModdedTrain.lean:1503` and makes effective config auditable.

### D) Collector command

Add a small collector script and always run it before launch:

- `uv run scripts/nanochat/collect_env.py --profile us_euv --run-id <id>`

Outputs:

- fully resolved, typed, grouped config
- source attribution per value (`default`, `profile`, `local`, `env`, `checkpoint`)
- validation report for missing or conflicting values

