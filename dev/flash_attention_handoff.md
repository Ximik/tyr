# Flash Attention Handoff

Date: 2026-04-23
Branch: `feat_flash_attn_runtime_bridge`

## Current State

- Temporary worktree: `/tmp/tyr_fused_dq_wip`
- Current long-running validation build in the main worktree:

```bash
source ./load_modules.sh && LEAN_CC=$PWD/scripts/lean_cc_wrapper.sh LEAN_CC_FAST=1 lake -R build Tyr.GPU.Kernels.MhaH100 GenerateGpuKernels
```

- That build is not hung in Lean. It is in the known heavy Lake link path and is currently linking `GenerateGpuKernels` with libtorch/CUDA/Arrow/Parquet.

## Current Goal

1. Get Tyr's ThunderKittens-backed H100 flash attention to a state that is:
   - correct end-to-end for forward and backward
   - benchmarked on one H100
   - structurally closer to real ThunderKittens, not just "works"
2. Then use that path for real training/inference integration in Qwen/Gemma.
3. Keep docs/dev log updated as checkpoints land.

## What Is Working

- Native runtime bridge exists and dispatches exact H100 shapes.
- Current native supported shapes are effectively:
  - `seq=128, head_dim=64`
  - `seq=768, head_dim=64`
  - BF16, non-causal, self-attention
- Forward native kernels now use async TMA loads and WGMMA in the forward path.
- Backward `dK`/`dV` sweep kernels now use WGMMA for the KV contractions.
- Current runtime backward path is:
  - prep kernel
  - separate `Dq` kernel
  - separate KV sweep kernel
- Current native parity against PyTorch SDPA is green for the benchmarked exact rows.

## Measured State

Latest useful benchmark file:

- `benchmarks/results/flash_attn_cpp_native_h100_bwd_kv_wgmma_reload_q.jsonl`

Latest measured numbers:

- `native_dense_128x64`
  - Tyr runtime: `0.196867 ms`
  - Torch SDPA: `0.186461 ms`
  - correctness: green
- `native_dense_768x64`
  - Tyr runtime: `0.471824 ms`
  - Torch SDPA: `0.188912 ms`
  - correctness: green

So:

- correctness for the current exact-route runtime path is in place
- performance is still materially behind SDPA on `768x64`

## What Was Already Landed

Recent commits on the branch:

- `cedc6bc feat(gpu): add async tma primitives for h100 mha`
- `a0dfcf4 feat(gpu): add warpgroup mma for h100 attention forward`
- `49abfdf feat(gpu): add shared wgmma subtile primitives`
- `9ac4a84 perf(gpu): remove unused h100 forward registers`
- `7bd59f4 perf(gpu): use wgmma in h100 kv sweep`

These are already on `origin/feat_flash_attn_runtime_bridge`.

## Key Structural Gap Vs ThunderKittens

Tyr is still not close enough to vendored TK in backward structure.

Current Tyr generated/runtime shape:

- forward: partially TK-like now
- backward: still split into separate `dQ` pass and KV sweep
- lots of simplification relative to TK

Vendored TK H100 backward does:

- one pipelined backward kernel
- producer/consumer warpgroup structure
- semaphore/phased TMA pipeline
- fused ownership: KV tile owners sweep all Q tiles
- `dQ` store-add happens inside the same sweep
- 16-row warpgroup tiles, not naive 64x64 everywhere

Main disparities still remaining:

- Tyr still has separate `Dq` kernel
- Tyr still recomputes backward work across two kernels
- Tyr does not yet model TK's fused backward sweep ownership
- Tyr is not yet using TK-like 16-row subtiled accumulation structure in the full way
- no TK-like producer/consumer semaphore pipeline in backward
- no full seq/head-dim generality story yet
- no Qwen/Gemma integration yet

## Most Important Next Step

The next change to implement in `/tmp/tyr_fused_dq_wip` is:

- modify `tkMhaH100Bwd2BlockKvSweep` and `tkMhaH100Bwd12BlockKvSweep` in `Tyr/GPU/Kernels/MhaH100.lean`
  - make them also compute per-`(q,kv)` `dQ` contributions
  - `storeGlobalAdd` those contributions to `dQ_ptr`
- then remove the separate `tkMhaH100Bwd*BlockDq` launch from:
  - `cc/src/tyr_ops.cpp`
  - `Tyr/GPU/Ops/MhaH100.lean`

Why:

- it is the most direct step toward TK's fused backward ownership
- it should remove one entire backward kernel launch
- it should reduce duplicate recomputation
- it is a tractable intermediate step before a full TK-style pipelined backward kernel

## Important Code Locations

- Native runtime bridge:
  - `cc/src/tyr_ops.cpp`
- High-level exact-shape GPU wrappers:
  - `Tyr/GPU/Ops/MhaH100.lean`
- Kernel definitions:
  - `Tyr/GPU/Kernels/MhaH100.lean`
- WGMMA IR/emission/primitives:
  - `Tyr/GPU/Codegen/IR.lean`
  - `Tyr/GPU/Codegen/EmitNew.lean`
  - `Tyr/GPU/Codegen/Primitives.lean`
- Benchmark harness:
  - `Examples/GPU/RunFlashAttnBench.lean`
  - `scripts/gpu/bench_flash_attn_matrix.sh`
- Status docs:
  - `docs/gpu/thunderkittens-porting-status.md`
  - `dev/thunderkittens_porting_tracker.md`

## Docs Already Updated

Both docs already include the recent checkpoints:

- async TMA
- forward WGMMA
- failed shared-Q full-WGMMA attempt
- forward register cleanup
- KV sweep WGMMA checkpoint

## Build/Infra Reality

The clean Lake-native GPU build path already exists via `extern_lib libtyr`, but it is too slow because package-level link args drag libtorch/CUDA/Arrow/Parquet into things like `GenerateGpuKernels`.

That is why the current `lake -R build ... GenerateGpuKernels` takes so long.

So there are really two tracks:

1. kernel correctness/performance
2. build-path cleanup

Track 1 is the priority.

## Untracked Files To Ignore

Do not stage these by accident:

- `.codex`
- `Tyr/GPU/Ops/Activations.lean`
- `Tyr/GPU/Ops/FFTConv.lean`
- `Tyr/GPU/Ops/Flux.lean`
- `Tyr/GPU/Ops/Gemm.lean`
- `Tyr/GPU/Ops/LinearAttn.lean`
- `Tyr/GPU/Ops/Mamba.lean`
- `Tyr/GPU/Ops/MoE.lean`
- `Tyr/GPU/Ops/Normalization.lean`
- `Tyr/GPU/Ops/Rotary.lean`
- `benchmarks/results/flash_attn_cpp_native_h100_store_add_accum.jsonl`
- `dev/_fftconv_build.sh`
- `dev/nanogpt_gpu_kernel_plan.md`
- `make`

## Handoff Recommendation

Tell the next agent to do this in order:

1. Wait for or leave alone the current main-worktree Lake build; it is in heavy link, not obviously dead.
2. Work in `/tmp/tyr_fused_dq_wip` first.
3. Implement fused `dQ` accumulation inside KV sweep kernels.
4. Remove separate `Dq` launch from runtime/op wrapper.
5. Rebuild with the fast native path, not full Lake first:
   - direct Lean compile/generator
   - `make -C cc bench-flash-attn TYR_GPU_CODEGEN_MODULE=Tyr.GPU.Kernels.MhaH100`
6. Re-run one-H100 benchmarks and parity.
7. If parity is green and performance improves, update the two docs and create an intermediate commit.
8. Only then return to the build-system simplification issue.

## Short Version

- Correct exact-route H100 runtime path exists and is benchmarked.
- Performance is still behind SDPA, especially at `768x64`.
- Main next step is to fuse `dQ` into the KV sweep kernel and drop the separate backward `Dq` launch.
- Current long-running Lake build is blocked by the known oversized link path, not by kernel codegen correctness.
