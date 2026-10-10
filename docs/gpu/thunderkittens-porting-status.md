# ThunderKittens Porting Status

This note tracks the public ThunderKittens parity surface inside
`Tyr.GPU.Kernels`.

The GPU catalog is grouped into logical family entrypoints:

- `Tyr.GPU.Kernels.Attention`
- `Tyr.GPU.Kernels.StateSpace`
- `Tyr.GPU.Kernels.Parallel`
- `Tyr.GPU.Kernels.Gemm`
- `Tyr.GPU.Kernels.Normalization`
- `Tyr.GPU.Kernels.Experimental`

## Coverage

Every vendored ThunderKittens `.cu` source under
[external/git/ThunderKittens/kernels](../../external/git/ThunderKittens/kernels)
now has a built Lean counterpart in the catalog.

The important distinction now is not coverage vs missing families. It is:

- source-backed kernels that are fully represented in the Lean DSL today, and
- source-backed kernels whose source structure is still compressed into typed
  tiled shells because the DSL does not yet model every TMEM / cluster /
  packed-scale construct as a first-class operation.

The working exhaustive source-to-Lean matrix lives in
[dev/thunderkittens_porting_tracker.md](../../dev/thunderkittens_porting_tracker.md).

## Canonical Public Surface

| Tyr module | Vendored ThunderKittens source | Notes |
| --- | --- | --- |
| `Tyr/GPU/Kernels/FusedLayerNorm.lean` (`tkFusedLayerNormResidual1024`) | `external/git/ThunderKittens/kernels/layernorm/layernorm.cu` | Canonical fused residual + layernorm port. |
| `Tyr/GPU/Kernels/MhaH100.lean` | `external/git/ThunderKittens/kernels/attention/mha_h100/mha_h100.cu` | Canonical Hopper MHA surface. |
| `Tyr/GPU/Kernels/MhaH100LCF.lean` (`tkMhaH100LCFFwd64`, `tkMhaH100LCFFwd128`) | `external/git/ThunderKittens/kernels/attention/mha_h100_lcf/mha_h100_lcf.cu` | Dedicated LCF load-compute-finish counterparts, now fully typed as stationary-Q / streamed-KV shells. |
| `Tyr/GPU/Kernels/Based.lean` (`tkBasedLinearAttnFwd`) | `external/git/ThunderKittens/kernels/based/linear_attn.cu` | Source-backed forward owns the local polynomial/state contract. |
| `Tyr/GPU/Kernels/LinearAttn.lean` (`tkLinearAttnFwd`) | `external/git/ThunderKittens/kernels/linear_attention/linear_attention.cu` | Canonical decayed recurrent/local forward surface. |
| `Tyr/GPU/Kernels/FFTConv.lean` (`tkFFTConvPC1024`, `tkFFTConvNonPC64`) | `external/git/ThunderKittens/kernels/fftconv/*.cu` | Persistent and non-persistent FFTConv counterparts. |
| `Tyr/GPU/Kernels/Hedgehog.lean` (`tkHedgehogFwd`) | `external/git/ThunderKittens/kernels/hedgehog/hedgehog.cu` | Canonical chunk/state surface. |
| `Tyr/GPU/Kernels/Mamba2.lean` (`mamba2Fwd`) | `external/git/ThunderKittens/kernels/mamba2/mamba2.cu` | Dedicated typed chunk/state counterpart with runtime-bounded chunk loops and recurrent state updates. |
| `Tyr/GPU/Kernels/Flux.lean` (`tkFluxMatmulGeluFwd`, `tkFluxMatmulGateFwd`) | `external/git/ThunderKittens/kernels/flux/flux_*.cu` | Dedicated source-facing flux surfaces. |
| `Tyr/GPU/Kernels/Bf16Gemm.lean` (`tkH100Bf16GemmFwd`, `tkB200Bf16GemmFwd`) | `external/git/ThunderKittens/kernels/gemm/bf16_*/*.cu` | Hopper and Blackwell BF16 GEMM counterparts. |
| `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkH100Fp8E4M3GemmFwd`, `tkH100Fp8ScaledGemmFwd`, `tkB200Fp8E4M3Gemm1CtaFwd`, `tkB200Fp8E4M3Gemm2CtaFwd`, `tkB200MxFp8GemmFwd`) | `external/git/ThunderKittens/kernels/gemm/fp8_*`, `external/git/ThunderKittens/kernels/gemm/mxfp8_b200/*` | H100 and Blackwell FP8/MXFP8 GEMM surfaces. |
| `Tyr/GPU/Kernels/NvFp4Gemm.lean` (`tkB200NvFp4GemmFwd`) | `external/git/ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_b200_gemm.cu` | Dedicated NVFP4 Blackwell GEMM counterpart. |
| `Tyr/GPU/Kernels/Distributed.lean` (`allGatherFwd`, `allReduceFwd`, `allReduceEducationalFwd`, `reduceScatterFwd`, `agGemmFwd`, `agGemmB200Fwd`, `agGemmFp8B200Fwd`, `gemmArFwd`, `gemmArH100LcscFwd`, `gemmRsFwd`, `gemmRsB200Fwd`, `gemmRsFp8B200Fwd`) | `external/git/ThunderKittens/kernels/parallel/*` | Collective and communication+compute counterparts for the distributed family. |
| `Tyr/GPU/Kernels/RingAttn.lean` (`ringAttnPartial`, `ringAttnComm`, `ringAttnReduce`) | `external/git/ThunderKittens/kernels/parallel/ring_attn/ring_attn_h100.cu` | Forward ring-attention phases are represented directly, with the partial phase now using a typed runtime-bounded KV-shard loop. |
| `Tyr/GPU/Kernels/UlyssesAttn.lean` (`allToAllFwd`, `ulyssesQkvAllToAll`, `ulyssesAttnFwd`) | `external/git/ThunderKittens/kernels/parallel/ulysses_attn/ulysses_attn.cu` | Ulysses transport/orchestration family built on the typed shared all-to-all surface. |
| `Tyr/GPU/Kernels/MOE.lean` (`tkMoeDispatchGemm`) | `external/git/ThunderKittens/kernels/parallel/moe_dispatch_gemm/moe_dispatch_gemm_h100.cu` | Canonical fused dispatch/grouped-GEMM surface. |
| `Tyr/GPU/Kernels/Rotary.lean` | `external/git/ThunderKittens/kernels/rotary/rotary.cu` | Canonical rotary position kernel. |

## Derived Tyr Kernels

These modules stay in the catalog, but they are Tyr-native extensions rather
than vendored ThunderKittens parity surfaces:

- `Tyr/GPU/Kernels/LinearAttnBwd.lean`
- `Tyr/GPU/Kernels/RingAttnBwd.lean`
- `Tyr/GPU/Kernels/UlyssesAttnBwd.lean`

## Follow-Ups

The remaining work is now mostly DSL expressiveness work:

1. Add first-class TMEM, cluster, and packed-scale constructs so the Blackwell
   GEMM family can model the source structure directly rather than through typed
   compressed shells.
2. Tighten exact CTA worker packing for some attention/state-space families
   where the source structure is represented, but the runtime packing is still
   compressed.

## Training Bring-Up Roadmap (H100-First)

### 2026-04-23 update

Current performance checkpoint:

- [x] Rebuilt the vendored ThunderKittens H100 MHA PyTorch extension for the
  active Python 3.12 ABI.
  - `PyTorch/2.7.1-foss-2024a-CUDA-12.6.0` alone fails to compile the vendored
    source because CUDA 12.6 headers do not expose
    `cudaLaunchAttributePreferredClusterDimension`.
  - Loading `CUDA/12.9.1` after the PyTorch module builds the extension without
    a source compatibility shim.
- [x] Added a direct vendored-TK benchmark harness:
  - [benchmarks/bench_tk_mha_h100.py](../../benchmarks/bench_tk_mha_h100.py)
  - It records TK rows as `unsupported` when the vendored source cannot provide
    a valid apples-to-apples case.
- [x] Switched the native C++ flash-attention benchmark to CUDA-event timing so
  Tyr/SDPA rows use the same timing style as the ThunderKittens benchmark.
- [x] Captured one-H100 CUDA-event benchmark outputs:
  - [benchmarks/results/flash_attn_cpp_native_h100_cuda_event.jsonl](../../benchmarks/results/flash_attn_cpp_native_h100_cuda_event.jsonl)
  - [benchmarks/results/thunderkittens_mha_h100_cuda_event.jsonl](../../benchmarks/results/thunderkittens_mha_h100_cuda_event.jsonl)
- [x] Reached generated-vs-TK parity on the direct performance target:
  - `native_dense_768x64`, fwd+bwd, BF16, non-causal, `B=1,H=1`
  - generated Tyr: `0.135669 ms` / `0.152279 ms` across opposite backend
    orders,
  - vendored TK: `0.137159 ms` / `0.153591 ms`,
  - PyTorch SDPA: `0.170481 ms` / `0.187031 ms`.
- [x] Extended the generated DSL/codegen surface for TK-style producer/store
  warp specialization:
  - added current-warp TMA load/store-add emission instead of always forcing
    async issue from CTA warp `0`,
  - added explicit `warpid` / `laneid` runtime values to the DSL,
  - this removes the immediate codegen blocker for modeling TK's backward
    producer warp split where one producer warp loads the next Q/O/L/D tiles
    and another producer warp performs `qg` `store_add_async`.
- [x] Marked `native_dense_128x64` as not directly comparable to vendored TK:
  - the vendored H100 MHA launch grid is zero for `N=128`,
  - an exploratory `N=256` run launched but failed parity,
  - the vendored benchmark itself starts at `N=768`.

Performance gap list:

- [x] Correctness checkpoint for the generated Tyr bridge:
  - current `dQ` plus K/V sweep backward is deterministic and gradient-correct
    on `128x64` and `768x64`.
- [x] Benchmark comparability:
  - the native C++ bench can run SDPA, generated Tyr, and vendored TK in one
    process with CUDA-event timing.
- [x] Forward structure:
  - generated H100 forward now mirrors TK's shared staging, typed TMA coords,
    `warpgroup::laneid()` compute-done arrive, and final LSE sequence.
- [x] Dynamic shared-memory support:
  - generated Hopper kernels use TK-style dynamic shared allocation and launch
    wrappers set the effective dynamic shared-memory attribute.
- [x] D=64 generated backward parity:
  - `dQ` / `dK` errors now match the vendored TK scale on `native_dense_768x64`.
- [~] Remaining structural delta:
  - generated backward uses a generated prep plus K/V sweep path, while TK uses
    the original monolithic templated kernel shape.
- [ ] Broader supported rows:
  - extend generated specializations to TK-compatible `seq >= 768`,
    `seq % 256 == 0`, `headDim in {64,128}` first,
  - only then broaden to Qwen/Gemma-oriented GQA/MQA and causal rows.
- [~] Unified reporting:
  - native C++ now has a TK backend; remaining work is to persist the joined
    benchmark report from generated Tyr, vendored TK, and SDPA rows.
    harness as the official vendored baseline and teach the report step to join
    Tyr and TK JSONL files into one table.

### 2026-04-22 update

Current one-H100 flash-attention bring-up status:

- [x] Native H100 attention backward now has an explicit stacked-partials
  reduction contract in the runtime bridge:
  - `Tyr.GPU.Ops.MhaH100` reduces `dK` / `dV` from
    `[qBlocks, kvBlocks, 64, 64]` to `[seq, 64]` explicitly instead of relying
    on the accidental old reshape contract.
- [x] The direct example runners can now dump raw backward partial tiles for
  `dK` / `dV` before reduction:
  - `Examples/GPU/RunMhaH100.lean`
  - `Examples/GPU/RunMhaH100Seq768.lean`
  - output files:
    - `data/gpu_fixtures/mha_h100_128x64/diag_dK_tiles.pt`
    - `data/gpu_fixtures/mha_h100_128x64/diag_dV_tiles.pt`
    - `data/gpu_fixtures/mha_h100_768x64/diag_dK_tiles.pt`
    - `data/gpu_fixtures/mha_h100_768x64/diag_dV_tiles.pt`
- [x] A raw-partial comparison helper now exists for localizing gradient
  mismatches to tile math vs reduction/layout:
  - `scripts/gpu/compare_mha_partial_tile.py`
- [x] The native backward shared-memory overrun is fixed in source:
  - `Tyr/GPU/Codegen/GlobalLayout.lean` now exposes direct RV global loads for
    row vectors,
  - `Tyr/GPU/Kernels/MhaH100.lean` and
    `Tyr/GPU/Kernels/AttentionFactory.lean` now load `l` / `d` directly into
    RV registers instead of staging them through an extra shared
    `SV<float, 64>`,
  - this removes the extra `0x100` shared-memory allocation that had pushed
    the generated `tkMhaH100Bwd*Partials` kernels over the H100 limit.
- [x] A fresh native manual `RunMhaH100` binary has been rebuilt and run:
  - the relevant Lean artifacts were refreshed directly,
  - `GenerateGpuKernels` was rerun,
  - `make -C cc` rebuilt the generated object and native archive,
  - a trace-based manual relink produced `/tmp/RunMhaH100.manual`,
  - that binary now runs the current kernels and confirms the remaining native
    mismatch is narrower than before.
- [x] The PyTorch raw-partial comparator now accepts TorchScript-wrapped
  fixture payloads:
  - `scripts/gpu/compare_mha_partial_tile.py` unwraps the single-tensor
    `RecursiveScriptModule` format emitted by the fixture dumps,
  - this makes the dumped `diag_dK_tiles.pt` / `diag_dV_tiles.pt` files
    directly comparable without repacking them by hand.
- [x] The normal Lake build path is closer to usable:
  - `extern_lib libtyr` now builds `GenerateGpuKernels`, marks it executable,
    fingerprints the GPU-codegen environment, and runs the generator via
    `lake -R env ...`
  - this removes the earlier failure mode where the nested generator step died
    with `compiled configuration is invalid`
- [x] A host-specific executable-output workaround is now in place:
  - some `lean_exe` outputs under `.lake/build/bin` are being emitted as sparse
    zero files on this filesystem,
  - when that happens, `lakefile.lean` now relinks the executable from its
    `.trace` into `/tmp/tyr_relinked/<ExeName>` and runs the repaired copy
    instead
- [x] The one-H100 benchmark scaffold now has three backend classes:
  - `tyr_runtime`
  - `torch_sdpa`
  - `flash_attention`
  with the `flash_attention` slot wired narrowly to the repo-local FA3 kernel
  for the exact `1x1x256x64` forward-only row.
- [~] The remaining native correctness blocker is now narrower:
  - raw partial dumps and the PyTorch comparator both show the mismatch is
    already present in per-tile `dK`,
  - `dV` tiles compare correctly, so reduction is no longer the leading issue,
  - the next kernel-side task is to align the `dK` contraction/layout path
    with the ThunderKittens reference before reduction.
- [~] Broad Lake rebuild fanout is still a build-system issue:
  - `lake -R build RunMhaH100` still replays a much larger graph than this
    kernel loop should require on this machine,
  - the trace-based `/tmp/RunMhaH100.manual` relink is the current reliable
    way to validate the freshest native changes without waiting for a full
    graph rebuild.
- [ ] General native kernel coverage is still narrow:
  - current native route is still centered on `headDim=64` and known fixed
    families,
  - broader sequence/head-dimension support still needs the registry-backed
    specialization plan described below.

### 2026-04-21 update

- Native Hopper MHA backward now uses the ThunderKittens-style contract for
  `dK` / `dV`:
  - the Lean kernels write final gradients directly with `storeGlobalAdd`
    instead of writing `[seq, seq]` scratch partials and reducing them on the
    host side,
  - `Tyr/GPU/Ops/MhaH100.lean`,
    `Examples/GPU/RunMhaH100*.lean`, and
    `cc/src/tyr_ops.cpp`
    were simplified accordingly.
- The remaining exact-route mismatch was traced to a backend/codegen gap rather
  than attention math:
  - ThunderKittens `warp::tma::store_add_async` requires the destination
    `gl<...>` to carry the matching shared-tile descriptor type,
  - Tyr was still emitting bare `gl<T, ...>` params for those outputs.
- A first codegen fix landed in:
  - `Tyr/GPU/Codegen/EmitNew.lean`
  - `Tyr/GPU/Codegen/Attribute.lean`
  - `Tyr/GPU/Codegen/FFI.lean`
  It infers per-parameter TMA descriptor requirements from kernel IR and uses
  them when rendering `gl<...>` types.
- The clean end-to-end generator route is still not fully confirmed:
  - the current `GenerateGpuKernels` path is still emitting bare `gl<...>` for
    `MhaH100` backward outputs,
  - a narrow local patch to
    `cc/src/generated/Tyr_GPU_Kernels_MhaH100.cu`
    adding `st<float, 64, 64>` descriptors to the two backward accumulation
    outputs unblocked the native CUDA compile,
  - `make -C cc -B build/generated/Tyr_GPU_Kernels_MhaH100.o build/libTyrC.so`
    now succeeds again on this host.
- Current blocker before declaring the runtime bridge training-ready:
  - rebuild the higher-level
    `Tyr.GPU.Ops.FlashAttn` / `Examples.GPU.RunFlashAttnOp`
    path cleanly through Lake and rerun the parity smoke with the new native
    backward route.

Kernel source coverage is complete, but end-to-end training integration is still
in progress. The execution order is:

1. Single-GPU bring-up on one H100.
2. FlashAttention performance/correctness benchmark milestone.
3. Model integration (NanoChat attention path).
4. Multi-GPU scaling to 4xH100.

### Stage 1: One-H100 baseline and kernel validation

- Load the expected toolchain before GPU runs:
  - `source ./load_modules.sh`
  - Override module selections when needed with `TYR_CUDA_MODULE=...`,
    `TYR_NCCL_MODULE=...`, and `TYR_ARROW_MODULE=...`.
  - The default one-H100 path now uses `CUDA/12.9.1` and skips NCCL unless
    `TYR_NCCL_MODULE` is explicitly set.
  - `TYR_NCCL_MODULE=` cleanly skips NCCL loading, which is useful for the
    one-H100 direct path and keeps the default module stack smaller.
  - For non-login automation shells, `load_modules.sh` now bootstraps the
    `module` function from `/etc/profile.d/modules.sh` or the Lmod init script
    before loading the EasyBuild stack.
- Hopper note:
  - This host exposes newer CUDA modules including `CUDA/12.9.1` and
    `CUDA/13.1.0`, plus matching NCCL builds such as
    `NCCL/2.27.7-GCCcore-14.3.0-CUDA-12.9.1`.
  - A direct `nvcc` compile of the generated
    [Tyr_GPU_Kernels_MhaH100.cu](../../cc/src/generated/Tyr_GPU_Kernels_MhaH100.cu)
    object succeeded under `CUDA/12.9.1` without the old
    `cudaLaunchAttributePreferredClusterDimension` compatibility define, so the
    shim has now been removed from [cc/Makefile](../../cc/Makefile).
- Preferred direct build/run path:
  - `lake -R build GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 RunFlashAttn`
  - `lake -R env ./.lake/build/bin/GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 --out-dir cc/src/generated`
  - `make -C cc -j"$(nproc)"`
  - `lake -R env ./.lake/build/bin/RunFlashAttn`
  - The `GenerateGpuKernels` binary still needs `lake -R env`; invoking
    `./.lake/build/bin/GenerateGpuKernels` directly fails package lookup with
    `unknown module prefix 'Tyr'`.
  - Keep the older `scripts/gpu/*.sh` helpers as convenience wrappers only; the
    direct `lake` + `make` sequence is the path to stabilize.
  - For iteration speed, avoid the broad
    `lake -R build GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 RunFlashAttn RunMhaH100Train`
    command during proof/debug cycles. The faster loop is:
    - `lake -R build <module> +<module>:dynlib` for the failing Lean module
    - then rebuild only the runner you actually need
    - then rerun `GenerateGpuKernels` + `make -C cc`
  - If a replayed broad build dies late in `Tyr.GPU.Codegen.Primitives`, rebuild
    `Tyr.GPU.Codegen.Primitives` and `+Tyr.GPU.Codegen.Primitives:dynlib`
    directly. The concrete failure mode seen on this host was generic
    tensor-core complex wrappers not propagating the `M % 16 = 0`,
    `K % 16 = 0`, and `N % 16 = 0` shape proofs to `mma` / `mmaT`.
- Lean native-plugin note:
  - After changing import wiring in `Tyr.GPU.Codegen` / `Tyr.GPU.Kernels`,
    `lake build <module>` may refresh `.olean` / generated C without refreshing
    the corresponding native plugin. When debugging symbol mismatches in
    `.lake/build/lib/lean/*.so`, rebuild the affected module's `:dynlib`
    facet explicitly, for example:
    `lake build +Tyr.GPU.Codegen.Attribute:dynlib +Tyr.GPU.Kernels.Prelude:dynlib`.
- Run generated-kernel fixture checks on one visible GPU:
  - `lake -R build GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 RunFlashAttn`
  - `lake -R env ./.lake/build/bin/GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 --out-dir cc/src/generated`
  - `make -C cc -j"$(nproc)"`
  - `CUDA_VISIBLE_DEVICES=0 lake -R env ./.lake/build/bin/RunFlashAttn --regen`
  - `scripts/gpu/test_flashattn_e2e.sh` remains a convenience wrapper around
    the same flow.
- Run one-GPU kernel-vs-portable training benchmark:
  - `lake -R build GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 RunMhaH100Train`
  - `lake -R env ./.lake/build/bin/GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 --out-dir cc/src/generated`
  - `make -C cc -j"$(nproc)"`
  - `CUDA_VISIBLE_DEVICES=0 lake -R env ./.lake/build/bin/RunMhaH100Train --benchmark --warmup 20 --bench-iters 500 --lr 200.0 --noise 0.5`
  - `scripts/gpu/bench_mha_h100_train.sh` remains a convenience wrapper around
    the same direct commands.
- Acceptance:
  - fixture parity pass (forward and backward paths),
  - stable benchmark output from `RunMhaH100Train`,
  - no NaN/Inf regressions.

### Stage 2: Intermediate performance goal (SoTA FlashAttention vs Tyr TK)

- Add a reproducible benchmark harness that compares:
  - Tyr ThunderKittens-backed FlashAttention route,
  - PyTorch SDPA baseline,
  - SoTA FlashAttention backend available on the host stack.
- Record for each tested shape:
  - `ms/step`, `steps/s`, speedup ratio,
  - max/mean forward error and gradient error vs reference.
- Start with one H100 only before any multi-GPU rollout.

#### Benchmark Families

- `Family A: forward parity + throughput`
  - purpose: compare pure FlashAttention forward performance and numerical
    parity on identical inputs.
  - immediate implementation target: a dedicated one-command runner that
    launches the current direct TK kernels, PyTorch SDPA, and the strongest
    host FlashAttention backend available for the same shape.
- `Family B: train-step benchmark`
  - purpose: measure the practical end-to-end value of the H100 path using the
    existing forward + backward + SGD step loop.
  - immediate implementation target: extend the existing
    [RunMhaH100Train.lean](../../Examples/GPU/RunMhaH100Train.lean)
    path as the training benchmark reference until a generalized op-wrapper
    route exists.
- `Family C: kernel-generation generality sweep`
  - purpose: measure how broad the native kernel-generation surface actually is,
    not just how fast one supported benchmark shape runs.
  - immediate implementation target: a shape sweep that records whether a shape
    runs natively, falls back portably, or fails generation/build, together
    with the failure reason.

#### Benchmark Phases

- `Phase 0: lock the current exact-supported baseline`
  - benchmark only shapes already known to run through the direct path on one
    H100.
  - this is the release gate for the current bring-up work.
- `Phase 1: SoTA comparison on exact-supported shapes`
  - use the same input tensors and same dtype/layout for:
    - Tyr TK direct kernel path,
    - PyTorch SDPA default dispatch,
    - strongest host backend available for the same shape.
  - if the SoTA backend does not support the exact shape, report
    `unsupported`, not a silent fallback.
- `Phase 2: broaden along the near-term kernel family`
  - add benchmark rows only after a native kernel surface exists for them, for
    example through `MhaH100LCF` or `AttentionFactory`.
  - this is where `headDim = 128` and additional KV-tile counts should enter.
- `Phase 3: generality sweep`
  - run a broader matrix whose goal is coverage reporting:
    - native kernel hit,
    - portable fallback hit,
    - generation/build failure.
  - this is how to measure the generality problem systematically instead of
    discussing it abstractly.

#### Initial Matrix

- `Tier 0: must-run now`
  - `batch=1`, `heads=1`, `dtype=bf16`, `causal=false`
  - `(seq=128, headDim=64)` using the current 2-block path
  - `(seq=768, headDim=64)` once the 12-block direct runner is wired into the
    same benchmark harness
- `Tier 1: near-term native expansion`
  - `batch in {1, 4}`
  - `heads in {1, 8}`
  - `seq in {128, 768}`
  - `headDim in {64, 128}`
  - add rows only when the native path is actually present; otherwise report
    them in the generality sweep, not in the runtime leaderboard.
- `Tier 2: generality sweep only`
  - `seq in {64, 128, 256, 512, 768, 1024, 2048}`
  - `headDim in {64, 128}`
  - `causal in {false, true}`
  - `batch in {1, 4}`
  - `heads in {1, 8}`
  - for each row, log:
    - selected backend,
    - native/fallback/fail status,
    - failure reason if not native.

#### Metrics

- `Correctness`
  - forward: `out_mae`, `out_max`, `lse_mae`, `lse_max`
  - train-step: `dQ`, `dK`, `dV` max/mean error once exposed by the harness
  - convergence sanity: initial loss, final loss, relative improvement over a
    fixed small-step run
- `Performance`
  - median `ms/iter`
  - `iters/s`
  - speedup vs PyTorch SDPA default
  - speedup vs strongest SoTA backend available
  - estimated `tokens/s`
  - estimated effective `TFLOP/s` once the harness includes a FLOP model
- `Stability`
  - min/median/max over repeated runs
  - pass/fail/oom/nan status
  - backend-selected vs backend-requested, so silent fallbacks are visible
- `Generality`
  - generation success/failure
  - `lake` build time
  - `GenerateGpuKernels` time
  - `make -C cc` time for the generated kernel set

### Runtime Bridge Status (2026-04-21)

The high-level runtime bridge for FlashAttention now exists.

- `Tyr/Torch.lean`
  - `torch.nn.tyrFlashAttn4d` is no longer a temporary Lean fallback.
  - it now binds the real C++ entrypoint
    `lean_torch_tyr_flash_attn_4d`.
- `cc/src/tyr_ops.cpp`
  - now registers `tyr::flash_attn` through `TORCH_LIBRARY`.
  - current dispatch policy is:
    - native ThunderKittens-backed H100 route for the validated exact shapes
      (`bf16`, `batch=1`, `q_heads=kv_heads=1`, `headDim=64`,
      non-causal dense self-attention, `seq in {128, 768}`),
    - portable fallback to PyTorch SDPA for everything else.
  - the portable fallback now also covers:
    - causal attention,
    - padding-mask expansion,
    - grouped-query attention by repeating KV heads.
- `cc/Makefile`
  - now compiles `cc/src/tyr_ops.cpp` as part of `libTyrC`.

### Runtime Validation Snapshot

There is now a dedicated smoke test at
`Examples/GPU/RunFlashAttnOp.lean`.

It checks three cases:

1. native exact-shape H100 route,
2. portable causal fallback,
3. portable GQA fallback.

Current result on one H100:

- native route selection is correct: `route=tkKernel`
- native forward matches SDPA closely:
  - `out_mae=0.000057`
  - `out_max=0.001953`
- portable fallback paths are exact for the current checked cases:
  - causal fallback: all outputs and gradients matched exactly
  - GQA fallback: all outputs and gradients matched exactly
- native backward is not yet training-ready through the new high-level op:
  - `dQ` is close
  - `dK` and `dV` still miss the current tolerance window versus SDPA
  - after forcing a fresh native rebuild of the exact-route artifacts:
    - `lake -R build Tyr.Torch Tyr.GPU.Kernels.MhaH100 Tyr.GPU.Ops.MhaH100 Tyr.GPU.Ops.FlashAttn`
    - `lake -R env ./.lake/build/bin/GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 --out-dir cc/src/generated`
    - `make -C cc -B build/generated/Tyr_GPU_Kernels_MhaH100.o build/libTyrC.so -j"$(nproc)"`
  - the mismatch remains with the same localized outlier pattern:
    - `dk_mae=0.002040`, `dk_max=0.363281`
    - `dv_mae=0.002166`, `dv_max=0.550781`

Interpretation:

- the new runtime surface is now real and usable for forward benchmarking,
- portable fallback behavior is correct for non-native shapes and GQA,
- the remaining `dK` / `dV` mismatch is not explained by stale generated code,
  missing launcher symbols, or route-selection bugs,
- the remaining blocker for end-to-end training through `tyr::flash_attn` is
  the native backward mismatch on the exact H100 path.

### Build-System Note

While re-running the bridge parity check from fresh build artifacts, a concrete
integration gap showed up:

- `cc/build/libTyrC.so` can be rebuilt without native `MhaH100` launchers if
  `cc/src/generated/Tyr_GPU_Kernels_MhaH100.cu` has not been regenerated first
- in that state, `nm -D cc/build/libTyrC.so` showed the bridge-side launcher
  symbols as undefined:
  - `lean_launch_Tyr_GPU_Kernels_tkMhaH100Fwd2Block`
  - `lean_launch_Tyr_GPU_Kernels_tkMhaH100Fwd12Block`
  - `lean_launch_Tyr_GPU_Kernels_tkMhaH100BwdPrep2Block`
  - `lean_launch_Tyr_GPU_Kernels_tkMhaH100Bwd2BlockPartials`
  - `lean_launch_Tyr_GPU_Kernels_tkMhaH100Bwd12BlockPartials`
- regenerating `cc/src/generated/Tyr_GPU_Kernels_MhaH100.cu`, rebuilding
  `cc/build/generated/Tyr_GPU_Kernels_MhaH100.o`, and relinking
  `cc/build/libTyrC.so` restored those symbols
- after that real rebuild, the bridge parity result did not change, so the
  remaining failure is in the native exact-route backward path itself rather
  than in bridge linkage

While bringing up the runtime smoke test, a broader `lake -R build RunFlashAttnOp`
path exposed an unrelated package-native-plugin issue:

- `.lake/build/lib/lean/tyr_Tyr_GPU_Codegen_FFI.so` was observed as `file too short`
- later, `tyr_Tyr_GPU_Codegen_TileIR_Frontend.so` failed to materialize

This is not specific to the attention bridge itself, but it does affect the
fully packaged executable path. The narrower interpreter-based path with
explicit dynlib loading was sufficient to validate the attention bridge while
that broader package-native issue remains open.

#### Model-Driven Specialization Priorities

The current direct H100 path proves that Tyr can run generated TK-style
attention kernels on one H100, but it is still a narrow benchmark surface:

- `AttentionFactory` is already parameterized by `seq`, `headDim`, `kvBlocks`,
  `tileSize`, `dtype`, `arch`, `isCausal`, and `enableGqa`.
- The published wrapper/op layer is not yet parameterized in the same way:
  [MhaH100.lean](../../Tyr/GPU/Ops/MhaH100.lean)
  still hard-codes `[1, 1, seq, 64]` with manual dispatch for `seq=128` and
  `seq=768`, and
  [FlashAttn.lean](../../Tyr/GPU/Ops/FlashAttn.lean)
  still advertises TK coverage only for the same `headDim=64` / non-causal
  shapes.

The next backend milestone is therefore not "add one more benchmark shape". It
is to expose a stable runtime attention API over a registry of compiled
specializations, following the same pattern ThunderKittens uses from PyTorch:

- stable public call signature,
- runtime problem descriptor carrying tensor/layout/semantics metadata,
- dispatch to a compiled specialization keyed by concrete shape/mode,
- portable fallback when no specialization exists,
- benchmarking on top of that same surface so performance and generality are
  measured together.

This also matches how larger systems behave in practice:

- PyTorch exposes stable operator entrypoints and selects specialized kernels at
  runtime.
- JAX lowers a stable high-level operation and caches specialized executables
  per concrete signature.

#### Native Target Matrix Beyond the Current Demo Shapes

The next specialization work should be driven by the model code already in the
tree, not by the current `headDim=64` benchmark alone.

- `Qwen`
  - [Tyr/Model/Qwen/Attention.lean](../../Tyr/Model/Qwen/Attention.lean)
    uses grouped-query attention directly.
  - [Tyr/Model/Qwen35/Config.lean](../../Tyr/Model/Qwen35/Config.lean)
    defaults to `head_dim=256`, with common ratios such as
    `num_attention_heads=16`, `num_key_value_heads=4` and some variants with
    `num_key_value_heads=2`.
  - Qwen also needs decode-time KV-cache paths (`q_len=1`, growing `kv_len`)
    for incremental generation.
- `Gemma4`
  - [Tyr/Model/Gemma4/Config.lean](../../Tyr/Model/Gemma4/Config.lean)
    uses `head_dim=256`, sometimes `global_head_dim=512`, plus sliding-window
    attention with periodic full-attention layers.
  - Gemma variants also use nontrivial KV-head ratios and global/sliding mixes,
    so backend support cannot stop at plain non-causal dense MHA.

That implies the near-term native target matrix should be:

1. Existing H100 baseline:
   - `headDim=64`, `seq in {128, 768}`, non-causal
2. First model-relevant training specialization:
   - `headDim=256`, dense full-sequence attention on one H100
3. First grouped-query specialization:
   - `num_heads != num_kv_heads`
4. First decode specialization:
   - `q_len=1`, KV cache append/read path
5. First Gemma-style locality specialization:
   - sliding-window/full-attention split

#### Concrete Backend Refactor Sequence

1. Define a runtime `AttentionProblem` descriptor that includes:
   - `batch`
   - `num_q_heads`
   - `num_kv_heads`
   - `q_seq`
   - `kv_seq`
   - `head_dim`
   - `dtype`
   - `arch`
   - `is_causal`
   - attention mode flags such as dense/sliding/decode/GQA
2. Replace manual wrapper predicates with a specialization registry keyed by
   concrete problem classes, for example:
   - `(arch, dtype, q_seq, kv_seq, head_dim, causal, mode, gqa_class)`
3. Seed the registry with the already validated one-H100 variants:
   - `(128, 64, non-causal)`
   - `(768, 64, non-causal)`
   - the existing causal-64 family once wired through the same path
4. Move the benchmark harness onto that runtime surface so benchmark rows report
   both timing and dispatch outcome (`native`, `fallback`, `unsupported`).
5. Expand specializations in model order, not arbitrary shape order:
   - `headDim=256` full-sequence first
   - then GQA
   - then decode/KV-cache
   - then sliding-window/full-attention mixtures

Implementation status on 2026-04-20:

- The first operator-facing `AttentionProblem` layer is now in tree at
  [Tyr/GPU/Ops/AttentionProblem.lean](../../Tyr/GPU/Ops/AttentionProblem.lean).
- It records runtime attention metadata and centralizes the current
  specialization decision instead of leaving fixed-shape predicates duplicated
  across wrappers.
- The current selector is intentionally conservative and only exposes the
  already validated one-H100 native paths:
  - `tkMhaH1002Block`
  - `tkMhaH10012Block`
  - otherwise `portable`
- [MhaH100.lean](../../Tyr/GPU/Ops/MhaH100.lean)
  now dispatches through `AttentionProblem.currentSpecialization` and stores the
  selected specialization in its forward context.
- [FlashAttn.lean](../../Tyr/GPU/Ops/FlashAttn.lean)
  now constructs the same `AttentionProblem` descriptor instead of maintaining a
  separate hand-written shape predicate.
- Because the dedicated C++ op registration still does not exist, the current
  `torch.nn.tyrFlashAttn4d` symbol is temporarily implemented as a Lean-side
  fallback wrapper in [Tyr/Torch.lean](../../Tyr/Torch.lean).
  That keeps the wrapper layer buildable and makes the remaining missing piece
  explicit: the next step is still the real runtime bridge in `cc/src`.

#### Why This Feels Harder Than The ThunderKittens Demos

ThunderKittens itself does have a fairly clean integration story, but it is a
different story from "fully generic operator compiler".

- The ThunderKittens demos and PyTorch entrypoints expose a stable user-facing
  API and then launch a compiled specialization behind that API.
- The hard part is not the raw kernel call. The hard part is choosing the right
  specialization, handling fallback, and matching model semantics such as GQA,
  decode-time KV cache, and sliding-window attention.
- Tyr is currently paying extra complexity because the public wrapper layer
  still exposes fixed-shape typed variants directly instead of hiding them
  behind one runtime attention operator with specialization dispatch.

So the practical conclusion is:

- ThunderKittens is not the main blocker.
- The blocker is that Tyr still needs the same thin runtime operator layer that
  ThunderKittens-style integrations normally place above the kernels.
- Once that layer exists, adding or benchmarking new compiled variants should
  become much more mechanical.

#### How To Choose The "Right" Fixed Specializations

The right question is not "which exact fixed shapes should become the public
API?" The right question is "which problem axes materially change kernel
structure and therefore deserve specialization?"

For attention, those axes are usually:

- architecture: `SM90` vs future targets,
- dtype / accumulator dtype,
- attention mode:
  - dense prefill,
  - decode (`q_len = 1`),
  - sliding-window,
  - distributed/ring variants later,
- head-dimension class:
  - e.g. `64`, `128`, `256`, `512`,
- causal vs non-causal,
- grouped-query structure:
  - whether `num_q_heads == num_kv_heads`,
  - if not, the ratio class such as `2:1`, `4:1`, `8:1`,
- tile/block geometry:
  - tile sizes, warp layout, staging depth, shared-memory strategy.

By contrast, these should generally stay runtime values whenever possible:

- total `q_seq`,
- total `kv_seq`,
- batch size,
- exact number of heads.

That is especially important for large context lengths. If Tyr specializes on
exact full sequence length, the specialization matrix will explode and the
runtime surface will become brittle. For long-context support, the kernel should
usually specialize on:

- tile geometry,
- head-dimension class,
- mode,
- causal/GQA class,

while iterating over sequence tiles at runtime with tail masking for the final
partial tile.

In other words:

- good specialization: `SM90 + bf16 + dense_prefill + headDim=256 + causal +
  gqa_ratio=4 + tile=64`
- bad specialization: `SM90 + bf16 + qSeq=32768 + kvSeq=32768 + heads=16`

Exact-sequence specializations still have a place, but only when the algorithm
really depends on them:

- current bring-up/demo kernels with fixed block counts,
- tiny fixed-shape kernels used for validation,
- cases where full unrolling or staging decisions genuinely depend on exact
  small problem size.

They should not be the default strategy for scaling to large context lengths.

#### Practical Selection Rule For Tyr

Until there are production traces, Tyr should choose its first native
specializations from the model code already in tree and keep the matrix small.

1. Choose specialization families from model semantics:
   - `Qwen`: dense causal prefill, GQA, decode/KV-cache
   - `Gemma4`: sliding-window + periodic full attention
2. Choose head-dimension classes from model configs:
   - `256` first
   - `512` only when global-attention kernels are in scope
3. Choose GQA classes from model ratios:
   - equality case
   - `q_heads / kv_heads = 4`
   - `q_heads / kv_heads = 8` if needed by a concrete target model
4. Keep `q_seq` / `kv_seq` runtime-driven unless a kernel truly requires fixed
   loop counts.
5. Admit a new specialization only if it is both:
   - required by a target model path, and
   - measurably better than the portable or less-specialized path.

That gives a cleaner first internal matrix:

- `dense_prefill`, `headDim=256`, causal, equality/GQA
- `decode`, `headDim=256`, causal, KV-cache
- `sliding_window`, `headDim=256`, causal, window `{512,1024}` if Gemma paths
  need both

The current `headDim=64`, `seq in {128,768}` kernels remain valuable as:

- bring-up references,
- benchmarking scaffolding,
- proof that the runtime operator path works end-to-end,

but they should not define the long-term specialization policy.

#### Measurement Protocol

- Use one visible GPU only:
  - `CUDA_VISIBLE_DEVICES=0`
- Fix the module/toolchain stack in the result metadata:
  - `load_modules.sh` defaults
  - `nvcc --version`
  - `nvidia-smi` GPU/driver snapshot
- Fix input randomness:
  - one seed per benchmark row
  - reuse the same generated tensors across all compared backends
- For every reported runtime number:
  - 1 untimed cold start
  - 20 warmup iterations for stable runs
  - 100 to 500 timed iterations depending on runtime
  - 3 repeated benchmark runs
  - report the median of medians as the headline number
- Synchronize explicitly before reading timers.
- Keep correctness and performance modes separate:
  - correctness mode may use stricter references and fewer iterations
  - performance mode should avoid extra host-side checks inside the timed loop

#### Harness Design

- Add a dedicated runner rather than overloading the generalized op wrapper too
  early.
- Immediate preferred shape:
  - `Examples/GPU/RunFlashAttnBench.lean`
- Required backend modes:
  - `tyr_tk_direct`
  - `torch_sdpa`
  - `host_best_flash`
- Required output formats:
  - human-readable summary table on stdout
  - machine-readable `jsonl` or `csv` row per `(backend, shape, repeat)`
- Required metadata per row:
  - git commit
  - hostname
  - GPU model
  - driver version
  - CUDA toolkit version
  - module stack
  - shape, dtype, causal flag
  - backend requested
  - backend actually executed

#### Backend-Generalization Plan

- The next architectural priority is not just "more benchmark rows"; it is
  making the TK/codegen backend accept varying `seq` and `headDim` through one
  stable calling convention.
- Current state in-tree:
  - [AttentionFactory.lean](../../Tyr/GPU/Kernels/AttentionFactory.lean)
    already carries `seq`, `headDim`, `kvBlocks`, `tileSize`, `dtype`,
    `accDtype`, `scale`, and `isCausal` in `FAVariantConfig`,
  - but [MhaH100.lean](../../Tyr/GPU/Ops/MhaH100.lean)
    still publishes only a small fixed set of runtime variants (`seq=128` and
    `seq=768`, `headDim=64`),
  - and [FlashAttn.lean](../../Tyr/GPU/Ops/FlashAttn.lean)
    still classifies coverage with a hard-coded predicate before calling a
    torch-registered runtime op.
- Target direction, taking inspiration from ThunderKittens' PyTorch-side call
  pattern:
  - expose one stable runtime entrypoint that accepts runtime tensor metadata
    plus semantic flags (`causal`, `scale`, optional mask, maybe GQA mode),
  - lower that to a `globals`-style runtime descriptor,
  - dispatch to the best compiled specialization from a registry of
    `(seq, headDim, causal, dtype, arch)` variants,
  - make fallback selection explicit when no native specialization exists.
- This also matches how the larger frameworks tend to work:
  - PyTorch exposes stable high-level ops and then selects among specialized
    kernels/backends at runtime; `torch.compile` adds graph specialization and
    recompilation/guarding when needed,
  - JAX traces and lowers once per program shape/signature family, then
    compiles specialized executables per concrete shape/sharding; custom kernel
    routes such as Pallas still fit that specialization model rather than a
    single universally-generic kernel.
- Concretely, that means the benchmark harness should be built on top of the
  same future runtime surface we want NanoChat and the generalized ops to use,
  not on a permanently separate one-off interface.
- Practical implementation order:
  - first keep the direct one-H100 path as the correctness/perf baseline,
  - then build a runtime specialization registry over the existing factory +
    hand-written kernels,
  - then route the benchmark harness through that registry,
  - only after that widen the benchmark matrix to claim broader shape support.

#### Acceptance Criteria

- `Phase 0 gate`
  - all Tier-0 rows pass correctness
  - no NaN/Inf
  - training step reduces loss on the fixed 10-step sanity run
- `Phase 1 gate`
  - one command produces comparable rows for Tyr TK, PyTorch SDPA, and the
    strongest supported host backend on the same exact shape
  - unsupported backends are reported explicitly, not skipped silently
- `Phase 2 gate`
  - each newly-added native shape lands with both parity and runtime rows
- `Phase 3 gate`
  - the harness can report native coverage vs fallback coverage vs failure
    coverage for the broader shape matrix

## Current Bring-Up Notes

- The temporary `Tyr.GPU.Codegen.Ops` compatibility shim was removed again once
  the source tree no longer imported it.
- The concrete Lean-native blocker on 2026-04-20 was not a generic stale-cache
  issue; it was a package-prefix mismatch in generated native initializer
  symbols (`initialize_Tyr_GPU_Codegen_Monad` vs
  `initialize_tyr_Tyr_GPU_Codegen_Monad`) exposed by redundant direct
  `Monad` imports in:
  - `Tyr.GPU.Codegen.Attribute`
  - `Tyr.GPU.Kernels.Prelude`
- The current workaround is to route those modules through
  `Tyr.GPU.Codegen.Primitives` and rebuild the affected `:dynlib` facets.
- The remaining build-system blockers on this host turned out to be runtime
  linkage, not GPU codegen:
  - the same Lake `moreLinkArgs` are used for both
    `.lake/build/lib/lean/*.so` and `cc/build/libTyrC.so`,
  - a single relative LibTorch RPATH cannot be correct for both output
    locations,
  - the Linux link args now use an absolute repo-local LibTorch RPATH instead.
- The current cluster-specific compiler/linker constraint is:
  - Lean's bundled `clang` cannot relink native shared libraries on this host
    because of a glibc mismatch,
  - plain GCC cannot satisfy Lean's static `libc++` / `libc++abi` / `libgmp` /
    `libuv` link expectations by itself,
  - this was later resolved by linking `libTyrC` as a shared library, so no
    `LEAN_CC` wrapper is needed (see `docs/ffi-and-build.md`).
- The current build-time bottleneck is not Lean elaboration anymore; it is the
  final link step for large executables such as `GenerateGpuKernels`,
  `RunFlashAttn`, and `RunMhaH100Train` once the broad graph has already been
  replayed.
- The default one-H100 direct path now works end-to-end under:
  - `CUDA/12.9.1`
  - no NCCL module by default
  - `lake -R env ./.lake/build/bin/GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 --out-dir cc/src/generated`
  - `make -C cc -B build/generated/Tyr_GPU_Kernels_MhaH100.o build/libTyrC.a build/libTyrC.so -j"$(nproc)"`
- The `cc` build needed one additional separation between one-GPU and
  distributed paths:
  - `TORCH_CUDA_LINK_FLAGS` stay enabled whenever LibTorch ships `torch_cuda`
    and `c10_cuda`,
  - `USE_C10D_NCCL` is now enabled only when an NCCL module is actually loaded
    (`EBROOTNCCL` present), which prevents one-GPU builds from failing on
    missing `nccl.h`.
- Validated one-H100 results after the CUDA/NCCL cleanup:
  - `RunFlashAttn --regen`:
    `fwd_ok=true`, `fwd_lse_ok=true`, `lse_ok=true`,
    `out_mae=0.000151`, `out_max=0.003906`
  - `RunMhaH100Train --steps 10 --log-every 5 --lr 200.0 --noise 0.5`:
    `init_loss=0.010400`, `final_loss=0.006443`,
    `rel_improvement=0.380499`, `ok=true`
  - `RunMhaH100Train --benchmark --warmup 5 --bench-iters 20 --lr 200.0 --noise 0.5`:
    `kernel_ms_per_step=0.157652`,
    `torch_ms_per_step=0.220110`,
    `kernel_speedup_vs_torch=1.396180`

### Stage 3: NanoChat integration

- Route compatible attention calls in `Examples/NanoChat/ModdedGPT.lean`
  through the GPU op wrapper path with explicit fallback.
- Keep unsupported shapes/features on the portable SDPA route.
- Add one-run fallback reason reporting and parity tests at block level.

### Stage 4: Scale out to 4xH100

- After one-H100 correctness/perf gates pass, run distributed training:
  - `NPROC_PER_NODE=4 ./scripts/nanochat/run_train_torchrun.sh ...`
- Compare 1/2/4 GPU throughput and convergence behavior.
- Keep the one-H100 benchmark as the release gate for kernel regressions.

## Cleanup Result

The old sketch-only kernel modules that duplicated the canonical ports were
removed from the core catalog:

- `Tyr/GPU/Kernels/LayerNorm.lean`
- `Tyr/GPU/Kernels/LayerNormBwd.lean`
- `Tyr/GPU/Kernels/LayerNormResidual.lean`
- `Tyr/GPU/Kernels/Mamba.lean`
- `Tyr/GPU/Kernels/MambaBwd.lean`

That leaves the build/docs surface centered on the actual source-backed
ThunderKittens counterparts instead of parallel educational shims.

## Current Runtime-Bridge State (2026-04-21)

- The one-H100 runtime bridge for `tyr::flash_attn` is now materially closer to
  training-ready:
  - `cc/build/libTyrC.so` now exports the real TK/Mha launcher symbols instead
    of resolving to weak stubs from `cc/src/generated/tyr_gpu_kernel_stubs.cpp`.
  - The `cc` shared library now links CUDA explicitly (`-lcudart -lcuda`), which
    fixes the Hopper TMA runtime symbol gap (`cuTensorMapEncodeTiled`).
  - `lakefile.lean` no longer deletes previously generated `.cu` files when
    `TYR_SKIP_GPU_CODEGEN=1`; that deletion was silently forcing later rebuilds
    back onto stub launchers.
- The generated-code path is still not fully clean:
  - the current codegen still fails to infer TMA descriptor-bearing `gl<...>`
    parameter types for the `dK_part_ptr` / `dV_part_ptr` outputs of the
    `MhaH100` backward kernels,
  - so the current working state is still using local generated-CUDA edits for
    the `MhaH100` translation unit while the permanent codegen fix is finished.
- The more important runtime result is that the bridge no longer dies on kernel
  launch:
  - the native dense route now completes end-to-end,
  - `forward`, `dQ`, and `dK` are within tolerance against PyTorch SDPA,
  - the remaining parity failure is isolated to `dV`.
- Current `RunFlashAttnOp` status on one H100 (`CUDA_LAUNCH_BLOCKING=1`):
  - `route=tkKernel`
  - `out_ok=true`
  - `dq_ok=true`
  - `dk_ok=true`
  - `dv_ok=false`
  - representative error snapshot:
    - `out_mae=0.000057`
    - `dq_mae=0.000032`
    - `dk_mae=0.000046`
    - `dv_mae=0.003351`
    - `dv_max=0.451660`
- The current backward workaround is:
  - avoid the crashing TMA `store_add_async` direct-accumulation path for
    `dK`/`dV`,
  - stack per-query-block partials in plain global memory,
  - reduce them in `native_backward`.
- This proves the remaining blocker is no longer build integration or bridge
  wiring. The next concrete task is to fix the per-tile `dK`
  contraction/layout mismatch in `tkMhaH100Bwd*Partials`, then port that fix
  back into the Lean kernel surface so the generated CUDA no longer needs local
  edits.

### Runtime Bridge Update (Later 2026-04-21)

- The bridge parity blocker is now cleared for the current one-H100 route:
  - `RunFlashAttnOp` now reports
    - `route=tkKernel`
    - `out_ok=true`
    - `dq_ok=true`
    - `dk_ok=true`
    - `dv_ok=true`
  - representative native-dense error snapshot:
    - `out_mae=0.000057`
    - `dq_mae=0.000032`
    - `dk_mae=0.000046`
    - `dv_mae=0.000172`
    - `dv_max=0.003906`
- The source-of-truth kernel path no longer depends on hand-edited generated
  CUDA for the non-crashing accumulation workaround:
  - `Tyr/GPU/Kernels/MhaH100.lean` now emits q-block-major stacked
    `dK`/`dV` partial writes via plain `storeGlobal`,
  - `Tyr/GPU/Ops/MhaH100.lean` and the `RunMhaH100*` examples were updated to
    reduce `[1, 1, kvBlocks * seq, 64]` stacks back to `[1, 1, seq, 64]`.
- The remaining split is now explicit:
  - raw kernel forward + `dQ` + `dK` are coming from the TK/Mha path,
  - runtime-op `dV` is currently computed exactly in `cc/src/tyr_ops.cpp`
    by recomputing `P = softmax(Q K^T / sqrt(d))` in FP32 and forming
    `dV = P^T dO`.
- This is an intentional correctness bridge, not the final performance state.
  It makes the high-level `tyr::flash_attn` op training-correct on the current
  supported shape while the standalone kernel-side `dV` mismatch remains
  isolated to the raw `tkMhaH100Bwd*Partials` path.
- Operational note:
  - raw launcher examples such as `RunMhaH100` must be exercised via their
    compiled `lean_exe` targets (`lake build RunMhaH100` then run the binary),
    not `lean --run`, because the interpreter cannot resolve the generated
    launcher externs.

### Writeback Staging Checkpoint (2026-04-22)

- Referencing the ThunderKittens `mha_h100` backward path clarified an important distinction:
  - TK writes `dK` and `dV` through separate shared-memory staging regions (`kg_smem`, `vg_smem`) and fences async TMA stores with `group<8>::sync`, `group<4>::sync`, `wait(bar, toc)`, and `warp::tma::store_async_wait()`.
  - Tyr plain `storeGlobal` does not lower to TMA. It lowers to `warp::store(...)`, which is a blocking shared-to-global copy in the generated CUDA.
- That means the present `MhaH100` experiment is not add-a-wait-after-`storeGlobal`.
  - The closer reference-oriented change is to keep `dK` and `dV` staging disjoint.
  - `Tyr/GPU/Kernels/MhaH100.lean` now uses separate `dKShared` and `dVShared` buffers for backward writeback.
  - `Tyr/GPU/Kernels/AttentionFactory.lean` was updated the same way so future regeneration preserves the same direction.
- A separate generality limitation also became explicit during this review:
  - the stacked-partials reduction currently only behaves correctly for the supported fixed shapes because `qBlocks == kvBlocks` for `seq=128` and `seq=768`.
  - this is acceptable for the current bring-up target, but it is not the general reduction contract needed for arbitrary sequence families.
- A clean `GenerateGpuKernels -> make -C cc -> RunMhaH100{,Seq768}` validation loop is in progress against this writeback-staging patch.
  - The first long rebuild attempt also exposed a build-orchestration problem:
    overlapping Lake work can race on `GenerateGpuKernels`, fail with a
    missing `.lake/build/bin/GenerateGpuKernels`, and leave
    `cc/src/generated/Tyr_GPU_Kernels_MhaH100.cu` stale.
  - The intended validation flow should therefore stay serial:
    generate kernels, rebuild the native side, then build/run the raw examples.
- The Lake-side raw-example workflow is also narrower than before:
  - `lakefile.lean` now supports `TYR_BUILD_TYRC_DYLIB=0` during the narrow raw-example build path.
  - `buildMhaH100Examples`, `runMhaH100Exe`, `runMhaH100Seq768Exe`, and `validateMhaH100Examples` provide a standard compiled-executable loop without relying on the interpreter.
  - `buildNamedExecutables` now tries plain `lake build` first and only falls
    back to `lake -R build` when the failure text looks like a reconfigure
    issue, which should remove an avoidable full graph replay from normal
    iteration.
  - `extern_lib libtyr` now also narrows its GPU IR invalidation set for the
    `Tyr.GPU.Kernels.MhaH100` loop, which should reduce rebuild churn from
    unrelated GPU IR exports.
  - `extern_lib libtyr` now also fingerprints the active codegen environment
    (`TYR_GPU_CODEGEN_MODULE`, `TYR_SKIP_GPU_CODEGEN`, `TYR_BUILD_TYRC_DYLIB`)
    into its dependency set, so those build-shaping inputs can no longer leave
    a stale generated CUDA file behind without invalidating `libtyr`.
  - This does not make the build cheap, but it removes avoidable churn and makes the intended validation path explicit.
- ThunderKittens-to-Tyr TODO state:
  - [x] Split `dK` and `dV` writeback staging so the backward path is closer to the TK separation instead of reusing one shared tile.
  - [x] Add narrower Lake-side raw-example helpers so the intended compiled validation path is explicit.
  - [~] Validate the writeback-staging patch through a clean `GenerateGpuKernels -> make -C cc -> RunMhaH100{,Seq768}` loop.
  - [ ] Generalize the stacked-partials reduction so it remains correct when `qBlocks != kvBlocks`.
  - [ ] Add the `head_dim=128` specialization that exists in TK `mha_h100` and is relevant to the in-tree classic `Qwen` path.
  - [ ] Decide how to cover the in-tree `Qwen35` / `Gemma4` style `head_dim=256` attention path, which is outside the current TK `mha_h100` `64` / `128` family.
  - [ ] Add the head-ratio (`hr`) path needed for GQA/MQA-style head sharing.
  - [ ] Add causal variants in the `MhaH100` family instead of only the current non-causal fixed-shape route.
  - [ ] Decide how much of the TK async writeback pipeline (`qg_ready`, async K/V store ordering, explicit store waits) needs to be mirrored for performance rather than just correctness.

### Manual Native Rebuild And Tile-Math Checkpoint (2026-04-22)

- The native backward compile blocker is now fixed at the source level:
  - the `MhaH100` backward partial kernels had been allocating one extra shared
    `SV<float, 64>` scratch vector for `l` / `d`,
  - that pushed the generated CUDA to `0xc100` shared bytes instead of the
    Hopper limit `0xc000`,
  - the current fix is to load those vectors directly into RV registers from
    global memory.
- A fresh native validation run now exists even though the normal Lake route is
  still too broad:
  - the relevant `.olean` files were refreshed directly,
  - `cc/src/generated/Tyr_GPU_Kernels_MhaH100.cu` was regenerated from those
    refreshed Lean artifacts,
  - `make -C cc` rebuilt the generated object and native archive,
  - a trace-based manual relink produced `/tmp/RunMhaH100.manual`,
  - that binary successfully ran with `--dump-partials`.
- The rebuilt native run materially narrows the bug:
  - forward output, `l`, `dQ`, and `dV` remain on the expected route,
  - the remaining mismatch is `dK`,
  - the raw-partial comparator shows the `dK` error is already present at the
    per-tile level before any q-block reduction is applied.
- The PyTorch-side diagnostic path is now usable end to end:
  - dumped fixture payloads load even when saved as TorchScript-wrapped single
    tensors,
  - `diag_dV_tiles.pt` compares correctly across inspected tiles,
  - `diag_dK_tiles.pt` is wrong across inspected tiles, which points at the raw
    partial contraction/layout path instead of the reduction step.
- Updated TODO state:
  - [x] Fix the shared-memory overrun in `tkMhaH100Bwd*Partials` by removing the extra staged `l` / `d` shared vector.
  - [x] Rebuild and run a fresh native manual `RunMhaH100` binary against the regenerated CUDA.
  - [x] Confirm that the raw partial dump path works from the rebuilt native binary.
  - [x] Make the PyTorch comparator accept TorchScript-wrapped fixture payloads.
  - [x] Identify the true root cause of the apparent `dK` failure on the 128x64 example path.
  - [~] Reduce the broad Lake rebuild fanout so the standard build path is usable for rapid native iteration.
  - [ ] Fold the reliable native rebuild path into a simpler standard build flow that does not require trace-based manual relinking.
  - [ ] Resume one-H100 benchmarking now that native 128x64 parity is closed.

### Output Buffer Aliasing Root Cause (2026-04-22)

- The earlier 128x64 `dK` mismatch was not a kernel-math failure.
- Actual root cause:
  - Lean merged identical pure `torch.zeros ...` expressions for mutable output
    tensors in generated callers,
  - the compiled `RunMhaH100` backward launch passed the same stack tensor for
    both `dK_part_ptr` and `dV_part_ptr`,
  - the kernel then wrote `dK` first and `dV` second into the same backing
    memory, which made `dk_ref_ok=false` and `dv_ref_ok=true` look like a raw
    `dK` bug.
- The fix was applied in the caller layer, not the kernel math:
  - `Examples/GPU/RunMhaH100.lean`
  - `Examples/GPU/RunMhaH100Train.lean`
  - `Examples/GPU/RunMhaH100Seq768.lean`
  - `Tyr/GPU/Ops/MhaH100.lean`
  - each now derives `dK` and `dV` partial buffers from distinct
    `torch.mul_scalar` expressions on a shared zero seed so the generated code
    cannot alias them.
- After restoring the intended backward kernel path and relinking the compiled
  example binary directly from the trace link line, the raw 128x64 example is
  fully green:
  - `overall_ok=true`
  - `dq_ref_ok=true`
  - `dk_ref_ok=true`
  - `dv_ref_ok=true`
  - `dk_mae=0.000166`
  - `dv_mae=0.000153`
- The high-level runtime bridge is also green on one H100 after relinking the
  compiled `RunFlashAttnOp` binary against the updated native archive:
  - `flash_attn_native_dense route=tkKernel route_ok=true out_ok=true dq_ok=true dk_ok=true dv_ok=true`
  - representative native-dense errors:
    - `out_mae=0.000057`
    - `dq_mae=0.000032`
    - `dk_mae=0.000046`
    - `dv_mae=0.000172`
- This supersedes the earlier 128x64 `dK`-forensics conclusion for the raw
  example path.
- Current implication:
  - correctness is no longer the blocker for the one-H100 128x64 TK route,
  - the next work items are benchmarking, shape coverage, and build-flow
    simplification.

### One-H100 Benchmark Matrix

- Current headline rows must stay inside the native runtime-op surface:
  - `B=1, H=1, KV=1, seq=128, headDim=64, bf16, dense_prefill, non-causal`
  - `B=1, H=1, KV=1, seq=768, headDim=64, bf16, dense_prefill, non-causal`
- These are the only rows that can currently support a Tyr-vs-SDPA or
  Tyr-vs-Flash performance claim, because they are the only cases expected to
  route to the TK-backed kernel path today.
- Immediate control rows should still be recorded, but only as fallback/general
  correctness rows:
  - `seq=96, headDim=64, non-causal` -> expected portable fallback
  - `seq=128, headDim=64, causal=true` -> expected portable fallback
  - `q_heads=4, kv_heads=2, seq=96, headDim=64, enable_gqa=true` -> expected portable fallback
- Per-row reporting should always include:
  - requested backend and executed backend
  - route (`tkKernel` vs `portable`)
  - `out`, `dQ`, `dK`, `dV` error metrics
  - p50/p10/p90 latency
  - speedup vs SDPA only for rows that actually execute the native path and
    pass correctness

### Model-Driven Wrapper Order

- In-tree model pressure does not stop at the current `head_dim=64` bring-up.
- Near-term dense wrapper priority inferred from `Qwen`, `Qwen35`, and
  `Gemma4` is:
  - `dense_gqa_hd128_r{4,5}` for classic `Qwen`
  - `dense_gqa_hd256_r{4,8}` for `Qwen35` full-attention layers
  - `dense_gqa_hd512_r{4,8}` for `Gemma4` full-attention/global-head layers
- `Gemma4` also needs a distinct windowed family after the dense path:
  - `window_gqa_hd256_r{2,4,8}` with sliding-window prefill/decode semantics
- Consequence:
  - the current `d=64` kernel is a bring-up and benchmarking target
  - `d=128` is the first realistic text-model specialization
  - `d=256` and then windowed `d=256` are what move the backend toward real
    `Qwen35` / `Gemma4` coverage

### Next Implementation Order

- The engineering sequence is now fixed at a higher level:
  - refactor runtime routing from exact fixed wrappers to `family + key`
    without changing current `d=64` behavior,
  - land `hd128` dense GQA forward first for real `Qwen` shapes,
  - then complete `hd128` decode/mask/backward,
  - then repeat for `hd256` dense GQA for `Qwen35`,
  - then land the first `windowed hd256` family for `Gemma4` sliding decode.
- This order matters because it keeps one benchmark/control baseline alive
  while expanding one model-relevant family at a time.

### Benchmark Scaffold Status

- `Examples/GPU/RunFlashAttnBench.lean` now provides a structured one-H100
  benchmark surface with:
  - case selection,
  - backend selection,
  - JSONL output,
  - strict-mode failure semantics.
- The static CLI paths are already useful:
  - `--list-cases`
  - `--list-backends`
- The remaining blocker is not the benchmark schema but the compiled executable
  path:
  - interpreter mode is enough for static CLI inspection,
  - native benchmark execution still depends on a successful compiled
    `RunFlashAttnBench` build.

### Clean Minimal Benchmark Path

- The cleanest one-H100 benchmark path in-tree is still the native Lean runner:
  - [Examples/GPU/RunFlashAttnBench.lean](../../Examples/GPU/RunFlashAttnBench.lean)
  - [scripts/gpu/bench_flash_attn_matrix.sh](../../scripts/gpu/bench_flash_attn_matrix.sh)
- Reason:
  - it keeps Tyr, ThunderKittens-backed runtime dispatch, and the PyTorch
    reference inside one process and one timing harness,
  - it avoids a second Python-side benchmark implementation,
  - it avoids introducing `uv` or pip state into the default path.
- Current recommended commands:
  - build the generic benchmark binary once:
    - `source ./load_modules.sh && scripts/gpu/bench_flash_attn_matrix.sh --build-only`
  - benchmark Tyr runtime vs PyTorch SDPA on the current native-now rows:
    - `source ./load_modules.sh && scripts/gpu/bench_flash_attn_matrix.sh --skip-build -- --case native_now --backend tyr_runtime,torch_sdpa --warmup 20 --iters 200 --repeats 3 --jsonl-stdout`
  - benchmark the exact repo-local FA3 smoke row:
    - `source ./load_modules.sh && TYR_GPU_CODEGEN_MODULE=Tyr.GPU.Kernels.FlashAttn3 scripts/gpu/bench_flash_attn_matrix.sh --ensure-native -- --case future_flash_256x64 --backend flash_attention,torch_sdpa --warmup 20 --iters 200 --repeats 3 --jsonl-stdout`
- Backend interpretation:
  - `torch_sdpa` is the minimal PyTorch baseline and requires no Python
    packaging beyond the linked libtorch path already used by Tyr,
  - `flash_attention` currently means the repo-local FA3 kernel only for the
    exact `1x1x256x64` forward-only row,
  - there is no in-tree external `flash_attn` wheel dependency today.
- Host-stack inspection on this machine:
  - the site PyTorch module (`PyTorch/2.7.1-foss-2024a-CUDA-12.6.0`) reports
    CUDA Flash SDPA support,
  - it does **not** provide a separate `flash_attn` Python package.
- Practical implication:
  - do **not** introduce a `uv`-managed Python benchmark as the default route,
  - use `uv` only if we later decide to benchmark an external Python wheel in
    addition to the in-tree native runner.
- The next minimal improvement, if we want a cleaner PyTorch-vs-Tyr claim
  without external dependencies, is:
  - add a `torch_flash` backend to `RunFlashAttnBench` that forces PyTorch's
    CUDA flash SDPA path inside the existing native harness instead of creating
    a separate Python benchmark script.

### Raw Partial Observability

- The raw MHA example runners now have a diagnostic mode for kernel-side
  partials:
  - `RunMhaH100 --dump-partials`
  - `RunMhaH100Seq768 --dump-partials`
- In that mode they dump `dKStack` and `dVStack` before reduction as explicit
  tile tensors:
  - `128x64` case: `[2, 2, 64, 64]`
  - `768x64` case: `[12, 12, 64, 64]`
- This is the intended next diagnostic if parity still fails after the current
  rebuild:
  - wrong raw partials => kernel/store path bug
  - correct raw partials but wrong reduced `dK`/`dV` => reduction/layout bug

### Latest 128x64 Correctness Check

- The current compiled raw `RunMhaH100` path was revalidated after the concern
  that the mismatch may have moved from `dV` to `dK`.
- Trusted command:
  - `source ./load_modules.sh && lake -R run runMhaH100Exe --dump-partials`
- Result:
  - `overall_ok=true`
  - `kernel_ref_ok=true`
  - `dq_ref_ok=true`
  - `dk_ref_ok=true`
  - `dv_ref_ok=true`
  - `dq_mae=0.000168`
  - `dk_mae=0.000166`
  - `dv_mae=0.000153`
- Practical conclusion:
  - the raw 128x64 native kernel is not currently showing a `dK` regression,
  - stale overlapping Lake/linker processes and the earlier output-buffer
    aliasing bug were the misleading factors.

### Latest 768x64 Correctness Check

- The apparent 768x64 `dK` / `dV` failure was caused by running an old
  `.lake/build/bin/RunMhaH100Seq768` executable.
- A fresh compiled-object relink and the hardened compiled-run helper now
  validate the 12-block path:
  - command:
    - `source ./load_modules.sh && CUDA_VISIBLE_DEVICES=0 lake -R run runMhaH100Seq768Exe --dump-partials`
  - result:
    - `overall_ok=true`
    - `kernel_ref_ok=true`
    - `dq_ref_ok=true`
    - `dk_ref_ok=true`
    - `dv_ref_ok=true`
    - `dq_mae=0.000078`
    - `dk_mae=0.000077`
    - `dv_mae=0.000070`
- Build-path fix:
  - `runBuiltExecutable` now checks whether
    `.lake/build/ir/**/<Exe>.c.o.export` is newer than the executable,
  - if so, it relinks through the existing `/tmp/tyr_relinked` path even when
    the old executable is a valid ELF,
  - this preserves compiled execution and prevents stale binaries from being
    misdiagnosed as gradient regressions.

### Runtime Bridge Benchmark

- Added a compiled C++ benchmark path for the current native runtime rows:
  - [cc/tools/bench_flash_attn.cpp](../../cc/tools/bench_flash_attn.cpp)
  - build:
    - `source ./load_modules.sh && make -C cc bench-flash-attn TYR_GPU_CODEGEN_MODULE=Tyr.GPU.Kernels.MhaH100`
- The benchmark calls `tyr_ops::flash_attn_dispatch` directly and compares
  fwd+bwd against LibTorch SDPA.
- The runtime bridge now returns native reduced `dVStack` in backward instead
  of recomputing `dV` through PyTorch matmul/softmax.
- One-H100 result:
  - command:
    - `source ./load_modules.sh && CUDA_VISIBLE_DEVICES=0 cc/build/tools/bench_flash_attn --case native_now --backend torch_sdpa,tyr_runtime --warmup 5 --iters 20 --repeats 3 --jsonl-out benchmarks/results/flash_attn_cpp_native_h100_native_dv.jsonl --jsonl-stdout`
  - `native_dense_128x64`:
    - `torch_sdpa p50_ms=0.186584`
    - `tyr_runtime p50_ms=0.197651`
    - `correctnessOk=true`
    - `speedupVsSdpaP50=0.944007`
  - `native_dense_768x64`:
    - `torch_sdpa p50_ms=0.188121`
    - `tyr_runtime p50_ms=0.562412`
    - `correctnessOk=true`
    - `speedupVsSdpaP50=0.334489`
- Current interpretation:
  - the native runtime route is training-correct for the current supported
    128x64 and 768x64 rows,
  - it does not yet demonstrate a speed advantage over PyTorch SDPA,
  - the main performance gap is still kernel fusion/generalization: the current
    runtime backward uses separate prep/partial/reduction work, while SDPA is a
    fused production path.

### Build Status Note

- On this host, direct `lake -R build RunMhaH100` can still stall while linking
  `GenerateGpuKernels` into `.lake/build/bin`.
- The repo's compiled-run helper path is currently more reliable:
  - it detects invalid `.lake/build/bin` executables,
  - relinks from the Lake trace into `/tmp`,
  - then executes the compiled binary.
- Tested but rejected linker accelerators:
  - Lean-toolchain `ld.lld` fails with `GLIBC_2.29`,
  - module-stack `ld.gold` fails against the older
    `/cm/local/apps/gcc/9.2.0/lib64/libstdc++.so.6` because it needs
    `GLIBCXX_3.4.29`,
  - even after forcing the GCCcore runtime library path, `gold` fails with
    hidden-symbol errors (`_ZdlPvm`).

### 2026-04-22 Store-Add Accumulation Pass

- New direction after comparing Tyr's generated CUDA with
  `external/git/ThunderKittens/kernels/attention/mha_h100/mha_h100.cu`:
  - ThunderKittens accumulates `kg_reg` / `vg_reg` across query tiles inside
    the backward kernel and emits final KV gradients with
    `warp::tma::store_add_async`,
  - Tyr's previous runtime path wrote q-block-major `dK` / `dV` partial stacks
    and reduced those stacks after the kernel,
  - that external partial-reduction contract was correct for the fixed rows but
    both slower and less general than the ThunderKittens contract.
- Current implementation changes in flight:
  - `Tyr/GPU/Kernels/MhaH100.lean` now stores `dK` / `dV` contributions to
    final `[1, 1, seq, 64]` buffers with `storeGlobalAdd`,
  - the kernels issue `warp::tma::store_async_wait()` after the async
    store-adds, matching the important completion wait in TK `kv_store`,
  - `cc/src/tyr_ops.cpp` now allocates final zeroed `dK` / `dV` tensors and no
    longer calls a host-side stacked-partial reduction,
  - raw validation runners and the training demo now consume final gradients
    directly under `contract=store_add_accum`.
- Validation status:
  - [x] Identify that TK's production path does not use Tyr's external partial
    stack reduction contract.
  - [x] Move the native runtime bridge to final dK/dV buffers.
  - [x] Update raw example and training callers to the final-gradient contract.
  - [~] Rebuild the generated `Tyr_GPU_Kernels_MhaH100.cu` and confirm it emits
    `store_add_async` plus `store_async_wait`.
  - [ ] Re-run raw 128x64 and 768x64 parity on one H100.
  - [ ] Re-run the compiled C++ bridge benchmark and compare against the
    previous `native_dv` JSONL.
- Gap list:
  - [x] Remove the largest obvious performance tax: PyTorch-side reduction of
    q-major `dK` / `dV` partial stacks.
  - [~] Close the semantic gap to TK's final-gradient store-add contract.
  - [ ] Replace per-q-block CTAs that store-add every KV tile with a more
    TK-like KV-centric sweep that accumulates across query tiles in registers.
  - [ ] Revisit generic `sync` placement after the store-add path is correct;
    TK's syncs are tied to pipeline handoffs and async store completion, not
    blanket barriers after every logical operation.
  - [ ] Add head-dim 128 coverage and decide how Qwen/Gemma head-dim 256 should
    route.

### 2026-04-23 Store-Add Fault Resolution

- The illegal-memory-access blocker in the store-add backward route was not a
  math-gradient issue.
- Root cause:
  - ThunderKittens requires TMA descriptor-bearing global-layout arguments to
    be grid-constant kernel parameters,
  - ThunderKittens also allocates TMA-swizzled shared tiles through
    `tma_swizzle_allocator`, giving them 1024-byte alignment,
  - Tyr's generated CUDA passed `gl<..., st<...>>` descriptor objects by value
    as ordinary kernel parameters and emitted plain static `__shared__ st<>`
    objects.
- Codegen fix:
  - pointer parameters with inferred TMA descriptors now emit as
    `const __grid_constant__ gl<..., st<...>>`,
  - generated shared tiles now emit `__shared__ KITTENS_ALIGN_AS(1024) st<...>`,
  - the H100 MHA backward store-add sections use `group<4>::sync(4)`, one
    issuing warp, `warp::tma::store_add_async`, and
    `warp::tma::store_async_wait()`.
- Validation:
  - blocking one-H100 runtime check:
    - `CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 cc/build/tools/bench_flash_attn --case native_dense_128x64 --backend tyr_runtime --warmup 0 --iters 1 --repeats 1 --jsonl-stdout`
    - result: `correctnessOk=true`
    - `dkMae=4.58404e-05`
    - `dvMae=0`
  - compiled one-H100 bridge benchmark:
    - `source ./load_modules.sh && CUDA_VISIBLE_DEVICES=0 cc/build/tools/bench_flash_attn --case native_now --backend all --warmup 5 --iters 20 --repeats 3 --jsonl-out benchmarks/results/flash_attn_cpp_native_h100_store_add_gridconst.jsonl --jsonl-stdout`
  - `native_dense_128x64`:
    - `torch_sdpa p50_ms=0.147628`
    - `tyr_runtime p50_ms=0.160957`
    - `correctnessOk=true`
    - `speedupVsSdpaP50=0.91719`
  - `native_dense_768x64`:
    - `torch_sdpa p50_ms=0.177279`
    - `tyr_runtime p50_ms=0.522411`
    - `correctnessOk=true`
    - `speedupVsSdpaP50=0.339348`
- Updated gap list:
  - [x] Diagnose and fix the TMA store-add illegal memory access.
  - [x] Verify native dK and dV parity through the compiled C++ runtime bridge.
  - [x] Remove the external q-major partial-stack reduction from the runtime
    bridge path.
  - [~] Generalize the codegen contract for TMA: current descriptor and shared
    alignment requirements are encoded, but issuer policy is still hardcoded for
    the single-warpgroup MHA kernels.
  - [ ] Close the performance gap with a TK-like KV-centric backward sweep that
    accumulates across query tiles in registers before one final store-add.
  - [ ] Add head-dim 128 and GQA/MQA routes needed by Qwen/Gemma-style model
    shapes.

### 2026-04-23 KV-Sweep Backward Pass

- Compared Tyr generated CUDA against ThunderKittens `mha_h100.cu` again, with
  focus on `compute_bwd_loop`, the `qg` store path, and `kv_store`.
- Implemented a TK-like K/V backward sweep:
  - new `tkMhaH100Bwd2BlockKvSweep` and `tkMhaH100Bwd12BlockKvSweep` kernels
    make each CTA own one KV tile,
  - each CTA sweeps all query tiles and accumulates `dK` / `dV` in registers,
  - each CTA emits one final `dK` store-add and one final `dV` store-add,
    instead of the previous per-`(qBlock, kvBlock)` K/V store-add pattern.
- Also added direct q-centric `dQ` kernels:
  - `tkMhaH100Bwd2BlockDq`,
  - `tkMhaH100Bwd12BlockDq`.
- Reason for the split:
  - a fully fused KV-sweep that store-added `dQ` across KV CTAs was buildable
    after reducing static shared memory, but repeated diagnostics showed
    intermittent large gradient errors on either `dQ` or `dK`, depending on the
    run,
  - keeping `dQ` on the known-correct q-centric direct-store path removed that
    nondeterminism,
  - restoring separate `dK` / `dV` shared staging in the KV sweep keeps the K/V
    writeback closer to TK and avoids reusing a TMA source tile before the final
    async store path has fully completed.
- Generated-CUDA sync comparison:
  - [x] K/V sweep now has the TK ownership shape: KV CTA, query sweep,
    register accumulation, final K/V store-add.
  - [x] Descriptor-bearing K/V store-add globals are emitted as
    `const __grid_constant__ gl<..., st<float,64,64>>`.
  - [x] TMA source shared tiles are emitted with `KITTENS_ALIGN_AS(1024)`.
  - [x] Separate K/V FP32 staging tiles are restored in the stable KV sweep.
  - [~] `dQ` is not yet the TK-style q-gradient store-add path; it is a
    correctness-preserving direct q-centric pass.
  - [~] Sync policy is still explicit raw `group<4>::sync(4)` around staging
    and store-add issue points. This is intentionally conservative for the
    current warp-level generated kernels; a general backend should model TK's
    pipeline handoff semaphores instead of hardcoding barriers.
  - [ ] Fuse the stable direct `dQ` path back into the KV sweep once the TMA
    q-gradient store-add path is deterministic under repeated runs.
- Validation:
  - [x] Direct Lean compile of `Tyr/GPU/Kernels/MhaH100.lean` succeeds.
  - [x] Generated `cc/src/generated/Tyr_GPU_Kernels_MhaH100.cu` contains the
    new `Bwd*Dq` and `Bwd*KvSweep` launchers.
  - [x] `make -C cc bench-flash-attn TYR_GPU_CODEGEN_MODULE=Tyr.GPU.Kernels.MhaH100`
    succeeds.
  - [x] `CUDA_LAUNCH_BLOCKING=1` 128x64 runtime smoke passes with
    `correctnessOk=true`.
  - [x] `CUDA_LAUNCH_BLOCKING=1` 768x64 runtime smoke passes with
    `correctnessOk=true`.
  - [x] Five repeated C++ diagnostics for 128x64 showed stable
    `out/dQ/dK/dV` parity after the split.
- Benchmark result:
  - `benchmarks/results/flash_attn_cpp_native_h100_dq_direct_kv_sweep.jsonl`
  - command:
    - `source ./load_modules.sh && CUDA_VISIBLE_DEVICES=0 cc/build/tools/bench_flash_attn --case native_now --backend all --warmup 5 --iters 20 --repeats 3 --jsonl-out benchmarks/results/flash_attn_cpp_native_h100_dq_direct_kv_sweep.jsonl --jsonl-stdout`
  - `native_dense_128x64`:
    - `torch_sdpa p50_ms=0.231418`
    - `tyr_runtime p50_ms=0.329764`
    - `correctnessOk=true`
    - `speedupVsSdpaP50=0.701769`
  - `native_dense_768x64`:
    - `torch_sdpa p50_ms=0.348799`
    - `tyr_runtime p50_ms=1.33397`
    - `correctnessOk=true`
    - `speedupVsSdpaP50=0.261476`
- Current gap list:
  - [x] Build and validate a TK-like K/V sweep for fixed 128/768 x 64 rows.
  - [x] Remove the unstable fused q-gradient store-add path from the training
    bridge.
  - [~] Preserve training correctness through a direct q-centric `dQ` pass plus
    KV-centric K/V pass.
  - [ ] Recover performance by making the TK-style `dQ` store-add path
    deterministic and fusing it back into the KV sweep.
  - [ ] Replace raw barrier snippets with first-class backend concepts for
    async TMA producer/consumer handoffs.
  - [ ] Add head-dim 128, causal, and GQA/MQA routes for Qwen/Gemma-style
    model coverage.

### 2026-04-23 Async TMA DSL Parity Checkpoint

- Implemented first-class DSL support for the TK synchronization pieces that were
  previously missing from generated H100 MHA code:
  - [x] `init_semaphore(sem, threadCount, transactionCount)` with one-thread init.
  - [x] `warp::tma::expect_bytes` issued by one loader warp.
  - [x] `warp::tma::load_async` for Q/K/V/O staging.
  - [x] `wait(sem, phase)` phase-aware waits.
  - [x] `group<N>::sync(barrier)` as a typed primitive.
  - [x] `warp::tma::store_commit_group()` and `store_async_wait()` as typed primitives.
  - [x] launcher-side dynamic shared-memory attribute setting.
- Generated-CUDA structural counts for `Tyr_GPU_Kernels_MhaH100.cu` after this pass:
  - `tma::load_async`: 44
  - `init_semaphore`: 31
  - `expect_bytes`: 31
  - `wait(`: 39
  - `warp::sync`: 0
  - `group<4>::sync`: 24
  - `store_commit_group`: 8
  - `store_async_wait`: 8
  - `warp::mma`: 36
  - `warpgroup::mma/mm`: 0
- Build issue found and fixed:
  - [x] The initial async semaphore pass pushed the backward kernels from exactly
    `0xc000` static shared bytes to `0xc018`, which ptxas rejects.
  - [x] Backward dK/dV writeback now reuses one FP32 staging tile and waits for
    the dK async store before overwriting that tile for dV. This preserves the
    TK lifetime rule: do not reuse a TMA store source before commit/wait.
- Validation:
  - [x] Direct Lean compile of `Tyr/GPU/Kernels/MhaH100.lean` succeeds.
  - [x] `GenerateGpuKernels Tyr.GPU.Kernels.MhaH100` succeeds.
  - [x] `make -C cc bench-flash-attn TYR_GPU_CODEGEN_MODULE=Tyr.GPU.Kernels.MhaH100` succeeds.
  - [x] One-H100 C++ runtime benchmark passes fwd+bwd correctness for 128x64 and 768x64.
- Benchmark result:
  - `benchmarks/results/flash_attn_cpp_native_h100_async_tma_dsl_parity.jsonl`
  - command:
    - `source ./load_modules.sh && CUDA_VISIBLE_DEVICES=0 cc/build/tools/bench_flash_attn --case native_now --backend tyr_runtime,torch_sdpa --warmup 5 --iters 20 --repeats 3 --jsonl-out benchmarks/results/flash_attn_cpp_native_h100_async_tma_dsl_parity.jsonl --jsonl-stdout`
  - `native_dense_128x64`: Tyr `0.152307 ms`, SDPA `0.143101 ms`, `correctnessOk=true`, `speedupVsSdpaP50=0.939554`.
  - `native_dense_768x64`: Tyr `0.496581 ms`, SDPA `0.172443 ms`, `correctnessOk=true`, `speedupVsSdpaP50=0.347261`.
- Current gap list:
  - [x] Remove generated `warp::sync` from H100 MHA.
  - [x] Add generated TMA load/semaphore/store-commit/store-wait structure.
  - [x] Keep training-route gradients correct after the async TMA change.
  - [~] Async load/store lifetime is now modeled, but still with conservative
    CTA-wide init syncs rather than TK's full producer/consumer phase pipeline.
  - [ ] Replace warp-level `warp::mma_*` attention math with TK-style
    `warpgroup::mm/mma_*` over shared/register tiles.
  - [ ] Move from the current split dQ plus KV sweep toward TK's fused backward
    kernel with loader/store warps and consumer warpgroups.
  - [ ] Add the 512-thread forward and 384-thread backward launch shapes needed
    for direct `mha_h100.cu` schedule parity.

### 2026-04-23 WGMMA Forward Parity Attempt

- Current implementation direction:
  - [x] Add first-class IR nodes for generated `warpgroup::mm_*` and
    `warpgroup::mma_*` calls.
  - [x] Add typed DSL primitives for the TK forward contraction forms:
    `warpgroup::mm_ABt(scores, q, k_smem)` and
    `warpgroup::mma_AB(out, probs, v_smem)`.
  - [x] Convert generated H100 MHA forward loops from `load + warp::mma` to
    WGMMA over shared K/V tiles.
  - [x] Rebuild the compiled generator after the IR constructor change. This
    exposed a build-speed problem: `GenerateGpuKernels` is still linked with
    `libTyrC.a`, libtorch, and CUDA even though codegen only emits text.
  - [x] Regenerate `Tyr_GPU_Kernels_MhaH100.cu` and verify the emitted forward
    math matches the TK call shape.
  - [x] Build `cc/build/tools/bench_flash_attn` with the regenerated CUDA.
  - [x] Benchmark `native_dense_128x64` and `native_dense_768x64` against SDPA.
- Generated-CUDA structural counts after this pass:
  - `warpgroup::mm_ABt`: 6
  - `warpgroup::mma_AB`: 6
  - `warpgroup::mma_async_wait`: 12
  - `warp::mma`: 24
  - `warp::sync`: 0
  - `group<4>::sync`: 18
  - `tma::load_async`: 44
  - `store_commit_group`: 8
  - `store_async_wait`: 8
- Benchmark result:
  - `benchmarks/results/flash_attn_cpp_native_h100_wgmma_forward_parity.jsonl`
  - command:
    - `source ./load_modules.sh && CUDA_VISIBLE_DEVICES=0 cc/build/tools/bench_flash_attn --case native_now --backend tyr_runtime,torch_sdpa --warmup 3 --iters 10 --repeats 2 --jsonl-out benchmarks/results/flash_attn_cpp_native_h100_wgmma_forward_parity.jsonl --jsonl-stdout`
  - `native_dense_128x64`: Tyr `0.216733 ms`, SDPA `0.202832 ms`,
    `correctnessOk=true`, `speedupVsSdpaP50=0.935862`.
  - `native_dense_768x64`: Tyr `0.498877 ms`, SDPA `0.215072 ms`,
    `correctnessOk=true`, `speedupVsSdpaP50=0.431112`.
- Explicit ThunderKittens reference checked:
  - TK forward uses `warpgroup::mm_ABt(att_block, q_smem[warpgroupid],
    k_smem[stage])`, then `warpgroup::mma_AB(o_reg, att_block_mma,
    v_smem[stage])`, with `warpgroup::mma_async_wait()` around each group.
  - TK backward uses `warpgroup::mma_ABt` / `warpgroup::mm_ABt` for score and
    dP, `warpgroup::mma_AB` for dV and dK, and a KV-owned final store.
- Remaining parity gaps after this attempt, even if it builds:
  - [~] Generated forward math is WGMMA-like at the call level, but performance
    did not improve. ptxas reports WGMMA serialization from insufficient
    register resources, which is consistent with missing TK warpgroup role and
    register allocation structure.
  - [~] Tyr still uses register-resident Q for the forward score matmul; TK uses
    shared `q_smem[warpgroupid]` with three consumer warpgroups.
  - [ ] Backward math still needs WGMMA conversion in the KV sweep.
  - [ ] Full schedule parity still needs loader/store warp roles, phase-flipped
    semaphores, and fused dQ store-add.

### 2026-04-23 Shared-Q WGMMA Finding

- Attempted the next mechanical TK alignment: feed the forward score matmul from
  shared Q/K tiles instead of register Q plus shared K.
- Result:
  - [x] The generated call changed to TK's shared/shared shape,
    `warpgroup::mm_ABt(scores, q_smem, k_smem)`.
  - [x] `nvcc` rejected the current 64x64 destination tile with TK's
    `static_assert(D::height == 1)`.
  - [x] This matches TK source: the shared/shared overload consumes a 64-row
    shared tile but writes one 16-row register tile per warp/consumer group.
  - [x] Added typed shared/shared primitive names with that constraint:
    `warpgroupMmSharedT64x16`, `warpgroupMmaSharedT64x16`, and AB variants.
- Updated gap list:
  - [~] The DSL now has a typed representation of TK's shared/shared 64-to-16
    WGMMA primitive, but the MHA forward loop still needs 16-row score/output
    subtiles and a way to assemble/store the four row groups.
  - [ ] Add explicit warpgroup role/subtile APIs before trying shared-Q forward
    again.
  - [ ] Once 16-row subtiles exist, re-run the shared-Q forward test and check
    whether ptxas WGMMA serialization disappears.

### 2026-04-23 WGMMA Register Cleanup

- Cleaned up the validated register-Q WGMMA forward route:
  - [x] Removed unused forward K/V register tile allocations left over from the
    pre-WGMMA load path.
  - [x] Regenerated `Tyr_GPU_Kernels_MhaH100.cu`.
  - [x] Rebuilt `cc/build/tools/bench_flash_attn`.
  - [x] Confirmed the unused-register CUDA warnings disappeared.
  - [~] ptxas still reports WGMMA serialization from insufficient register
    resources.
- Benchmark result:
  - `benchmarks/results/flash_attn_cpp_native_h100_wgmma_forward_clean_regs.jsonl`
  - command:
    - `source ./load_modules.sh && CUDA_VISIBLE_DEVICES=0 cc/build/tools/bench_flash_attn --case native_now --backend tyr_runtime,torch_sdpa --warmup 3 --iters 10 --repeats 2 --jsonl-out benchmarks/results/flash_attn_cpp_native_h100_wgmma_forward_clean_regs.jsonl --jsonl-stdout`
  - `native_dense_128x64`: Tyr `0.208570 ms`, SDPA `0.191270 ms`,
    `correctnessOk=true`, `speedupVsSdpaP50=0.917058`.
  - `native_dense_768x64`: Tyr `0.493654 ms`, SDPA `0.196349 ms`,
    `correctnessOk=true`, `speedupVsSdpaP50=0.397745`.
- Interpretation:
  - [x] Removing dead forward RT allocations helps slightly and makes ptxas
    output easier to interpret.
  - [ ] This does not close the performance gap; the next required step remains
    TK-style 16-row WGMMA subtiles and warpgroup role structure.

### 2026-04-23 K/V Sweep WGMMA Checkpoint

- Implemented a conservative WGMMA conversion for the KV-owned backward sweep:
  - [x] `dV += P @ dO` now uses register/shared `warpgroup::mma_AB`.
  - [x] `dK += dS @ Q` now uses register/shared `warpgroup::mma_AB`.
  - [x] The implementation reuses the existing BF16 shared staging tile to stay
    under the 48 KB static shared-memory limit.
  - [~] Q is reloaded before the `dK` WGMMA because the current DSL does not yet
    model TK's double-buffered Q/O shared-memory lifetime.
- Generated-CUDA structural counts after this pass:
  - `warpgroup::mm_ABt`: 6
  - `warpgroup::mma_AB`: 10
  - `warpgroup::mma_async_wait`: 16
  - `warp::mma`: 20
  - `warp::sync`: 0
  - `group<4>::sync`: 18
  - `tma::load_async`: 46
  - `store_commit_group`: 8
  - `store_async_wait`: 8
- Benchmark result:
  - `benchmarks/results/flash_attn_cpp_native_h100_bwd_kv_wgmma_reload_q.jsonl`
  - command:
    - `source ./load_modules.sh && CUDA_VISIBLE_DEVICES=0 cc/build/tools/bench_flash_attn --case native_now --backend tyr_runtime,torch_sdpa --warmup 3 --iters 10 --repeats 2 --jsonl-out benchmarks/results/flash_attn_cpp_native_h100_bwd_kv_wgmma_reload_q.jsonl --jsonl-stdout`
  - `native_dense_128x64`: Tyr `0.196867 ms`, SDPA `0.186461 ms`,
    `correctnessOk=true`, `speedupVsSdpaP50=0.947140`.
  - `native_dense_768x64`: Tyr `0.471824 ms`, SDPA `0.188912 ms`,
    `correctnessOk=true`, `speedupVsSdpaP50=0.400387`.
- Remaining gaps:
  - [~] This reduces warp-level MMA count and improves latency versus the
    cleaned forward-only WGMMA checkpoint.
  - [ ] ptxas still reports WGMMA serialization for forward and KV sweep kernels.
  - [ ] Full TK parity still needs 16-row consumer subtiles, role-specific
    register allocation, and fused dQ store-add.
