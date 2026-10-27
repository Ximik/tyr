# Tyr Code Review & Technical Debt

This document tracks architectural, performance, and correctness issues identified during deep semantic review.

## Launch research preview

Launch preparation notes, incomplete experiment results, and publication gates
are tracked under [`dev/launch/`](launch/README.md). In particular, the
BranchingFlows sampler audit and learned constellation/QM9 work live in
[`dev/launch/branchingflows.md`](launch/branchingflows.md), rather than in the
public launch article.

## 🚀 Performance Bottlenecks

### P01: Recursive KV-Cache Prefill
- **Issue**: `prefillCachesFromEmbeds` in `Qwen35/Model.lean` was performing $N$ separate forward passes for a prompt of length $N$.
- **Status**: Partially addressed via Batch Prefill refactor. Needs verification for other models.
- **Impact**: Drastic first-token latency reduction.

### P02: Sequential SafeTensors Loading
- **Issue**: Sharded model loading is fully sequential. Each tensor load involves re-parsing metadata and re-opening file handles.
- **Status**: Standard handle caching implemented.
- **Recommendation**: Implement `mmap`-backed loading in `cc/src/tyr.cpp` to eliminate copies and allow zero-copy weight access. Parallelize the Lean `Array.mapM` calls during layer loading.

### P03: FFI Boundary Overhead
- **Issue**: High-level Lean code calls Torch for every atomic op (e.g. `add`, `mul`, `relu`). In a 32-layer GPT, this incurs thousands of FFI context switches.
- **Recommendation**: Implement **Kernel Fusion** for hot blocks (e.g., Fused Attention, Fused MLP) where a single C++ call handles the entire block.

### P04: AD IR-Level Overhead
- **Issue**: `Tyr/AutoGrad.lean` uses heavy Lean IR rewriting. Tuple packing/unpacking for multiple gradients creates excessive allocations in the generated C code.
- **Recommendation**: Optimize `unpackTupleValues` and `mkTupleReturn` to avoid intermediate list/array allocations where possible.

## 🏛️ Architectural Refactoring

### A01: Monolithic `Tyr/Torch.lean`
- **Issue**: `Torch.lean` is 1450+ lines, mixing low-level FFI, high-level wrappers, and domain-specific submodules (`rotary`, `nn`, `linalg`).
- **Recommendation**: Split into:
  - `Tyr/FFI.lean`: Raw `@extern` declarations.
  - `Tyr/Ops.lean`: Safe Lean wrappers with shape logic.
  - `Tyr/NN/*.lean`: Move functional primitives to dedicated files.

### A02: TensorStruct Higher-Kinded Data (HKD)
- **Issue**: `TensorStruct` traverses indexed `Tensor σ` leaves. This makes generic `tree_transpose` (e.g., `Model (Array α) -> Array (Model α)`) nearly impossible to express cleanly.
- **Recommendation**: Refactor model structures to be parameterized by a leaf provider: `Linear (f : Shape -> Type)`.

### A03: Redundant Linear/Affine Wrappers
- **Issue**: `lean_torch_linear`, `lean_torch_linear3d`, `lean_torch_affine`, and `lean_torch_affine3d` all map to the same `torch::linear` call in C++.
- **Recommendation**: Consolidate into a single polymorphic `torch.linear` in Lean that handles rank-2 and rank-3 via dependent types.

## ⚖️ Correctness & Type Safety

### C01: Shape Inference Runtime Checks
- **Issue**: Complex shapes (e.g. `matmulShape`) are computed at runtime and then "cast" via `reshape`. This deferment hides shape errors until execution.
- **Recommendation**: Move more shape arithmetic to the type-level using Lean 4's macro system and typeclasses.

### C02: Device Consistency
- **Issue**: Some FFI calls (like `rand`, `full`) take an optional `device`, but most ops assume all inputs are on the same device. Mismatches lead to hard Torch crashes.
- **Recommendation**: Add a `Device` index to the `T` type: `T (s : Shape) (d : Device)`. This would provide compile-time guarantees for device-locality.

## ⚡ Decode kernel — open issues

V1 of the TK-style H100 decode kernel landed (`tkMhaH100DecodeFwd[64]` in
`Tyr/GPU/Kernels/MhaH100Decode.lean`). Tracker for the remaining work.
Longer-form rationale per item in `dev/decode_kernel_v2_plan.md`.

### D01: head_dim=256 (unblock Qwen 3.6 / Gemma-2 27B) — **done**
- Landed in `feat(gpu-kernels): head_dim=256 decode kernel for Qwen
  3.5/3.6 family`. `tkMhaH100DecodeFwd256` calls `decodeFwdBodyImpl`
  with `hdim=256` and `scoreScaleLog2e = 1/sqrt(256) * log2(e)`.
- Verified: `qwen36_35B` (B=1, qHeads=16, kvHeads=2, kvSeq=2048,
  head_dim=256) parity-passes vs SDPA at mae≈4.4e-5, max_abs≈4.9e-4
  — same precision as head_dim=128 cases.
- Eligibility test in `TestGPUKernels.lean` updated: head_dim=256
  selects `tkMhaH100Decode`; head_dim=192 (and other unsupported
  dims) still route to portable.

### D01b: refactor decode kernel to 16-row warp tiles — **done**
- Landed in `perf(gpu-kernels): D01b — 16-row per-warp tiles for
  decode kernel`. o/s/p register tiles are now per-warp 16 rows (TK
  reference structure); shared tiles stay 64 rows for warpgroup
  WGMMA. Q stays in shared (consumed via `warpgroupMmSharedT64x16`).
  Output goes through `warpgroupStore` then a single TMA store.
- **Result** (vs V1): 4.6×–15× faster on Tyr-internal across all
  shapes; the `qwen36_35B_kv8k` 0.48× SDPA regression is gone (now
  1.00×). All 6 parity shapes + cache parity still pass at the
  prior tolerances.

### D02: Benchmark V1 vs SDPA — **done**
- Landed in `test(gpu): forward-only decode benchmark vs PyTorch SDPA`.
  `Examples/GPU/RunDecodeBench.lean` runs Tyr decode + SDPA on the
  decode fixtures and emits JSONL.
- **First numbers** (H100 NVL,
  `benchmarks/results/decode_v1_bench.jsonl`): B=4 d=128 kv=2048 is
  **5.17×** faster than SDPA; B=1 cases at d∈{64,128,256} kv=2048 tie
  at 1.00×; **`qwen36_35B_kv8k` is 0.48×** (2× slower than SDPA) —
  register-spill on d=256 hurts most at long kv_seq where WGMMA
  pipelining actually matters. Motivates D01b and D03.

### D03: GQA-group Q-packing
- **Issue**: At Llama-3 batch=1, grid is `batch * q_heads = 32` CTAs
  (~25% util on 132-SM H100); each GQA group's R sibling CTAs
  redundantly load the same K/V tiles.
- **Plan**: Pack the R = `q_heads/kv_heads` query heads of each group
  into the first R rows of one 64-row Q tile. Grid drops to
  `batch * kv_heads`, KV bandwidth drops R×, output write becomes a
  per-row scatter. ~2 days. Detail in `decode_kernel_v2_plan.md` §1.
- **Expected**: ~3× on Llama-3-8B batch=1.

### D04: Producer/consumer + 2-stage KV ring buffer
- **Issue**: V1 is single-buffer; load and compute are only one stage
  deep, so long-`kv_seq` decode spends measurable time waiting on
  KV transfers.
- **Plan**: 1 producer + 1 consumer warpgroup, `STArray BFloat16 64
  hdim 2`, per-stage `SemaphoreArray 2`. All required primitives are
  already in `Tyr/GPU/Codegen/Primitives.lean`; main effort is
  semaphore phase-bit ergonomics. ~3 days. Detail in §2.
- **Expected**: 1.5–2× for `kv_seq ≥ 1k`.

### D05: Split-KV reduction
- **Issue**: At batch=1, kv_seq=16k, even with GQA packing each CTA
  processes 256 KV blocks sequentially while ~124 SMs sit idle.
- **Plan**: Split kv_seq across 8–16 CTAs writing partial
  `(max, sum, acc)` to scratch; tiny combine kernel applies the
  streaming softmax-rescale recurrence. ~3 days. Detail in §3.
- **Expected**: ~10× at batch=1, kv_seq=16k.

### D06 (deferred): Paged KV cache
- vLLM-style block-table indirection. Detail in §4. Defer until a
  concrete serving user lands.

### D07 (deferred): FP8 K/V cache
- Halves KV bandwidth on Hopper at the cost of one scale-factor
  multiply per block. Detail in §6. Defer until a target model demands
  it (Llama-3 / Qwen 3.6 still in BF16 today).

### D08 (cross-cutting): Decode backward
- Decode forward only. Decode is typically inference-time so backward
  isn't needed yet, but the kernel doc-comment should call it out
  explicitly.
