# Runpod Workflow

This workflow treats spot pods as disposable and keeps the durable state
outside the machine:

- local machine or GitHub is the source of truth for the repo
- `.runpod-state/` stores only local non-secret metadata such as pod/volume ids
- a Runpod network volume keeps remote caches and benchmark logs alive across
  preemption

Secrets are intentionally out of scope:

- do not commit `RUNPOD_API_KEY`
- do not put `RUNPOD_API_KEY` into tracked config files
- authenticate with `runpodctl doctor` or export the key in your shell only

## Files

- `scripts/runpod/create_or_resume.sh`: create a fresh pod when no managed pod is running
- `scripts/runpod/sync_repo.sh`: rsync the repo to the remote workspace without syncing secrets or local build junk
- `scripts/runpod/bootstrap.sh`: remote bootstrap for Lean/CUDA prerequisites
- `scripts/runpod/run_bench.sh`: full create/sync/bootstrap/benchmark wrapper
- `scripts/runpod/config.env.example`: non-secret configuration template

## Recommended Setup

1. Copy the example config into a gitignored location, for example:

```bash
mkdir -p .runpod-state
cp scripts/runpod/config.env.example .runpod-state/h100.env
```

2. Edit `.runpod-state/h100.env` and set either `TYR_RUNPOD_TEMPLATE_ID` or
   `TYR_RUNPOD_IMAGE`.

Template is preferred. The template or image should already contain:

- CUDA with `nvcc`
- `git`, `curl`, `rsync`, `make`, and a C++ compiler

3. Authenticate locally:

```bash
runpodctl doctor
```

4. Run the benchmark:

```bash
TYR_RUNPOD_CONFIG=.runpod-state/h100.env \
  ./scripts/runpod/run_bench.sh
```

You can forward extra arguments to
[`scripts/gpu/bench_mha_h100_train.sh`](../gpu/bench_mha_h100_train.sh),
for example:

```bash
TYR_RUNPOD_CONFIG=.runpod-state/h100.env \
  ./scripts/runpod/run_bench.sh --warmup 10 --bench-iters 200
```

## Preemption Model

If the spot pod is preempted:

1. rerun `run_bench.sh`
2. it will create a fresh pod under the configured price cap
3. it will attach the same network volume when configured
4. it will resync the local repo into the new pod
5. it will rerun the benchmark

The workflow does not try to resurrect a dead pod id by default. Fresh pod +
same durable storage is more reliable than depending on spot resume behavior.
