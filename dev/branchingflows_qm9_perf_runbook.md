# BranchingFlows QM9 — Tyr overhead runbook (spark-e626)

Goal: verify that Tyr's per-step training overhead for the QM9 paper profile is
**not massive** — i.e. step throughput on GPU is within the same ballpark as a
tuned loop for this model size (order 2× of expectation, not 10×). If overhead
is large, find and fix the top causes, then re-measure.

This document is both the procedure and the progress log. Newest log entries
at the bottom.

## Target workload

- Profile: `paper-qm9-main` (`scripts/branchingflows/qm9_paper.env.example`)
- Full architecture: hidden 384, 12 heads × 64, MLP 1536, RFF 64, 12 layers,
  6 coord-update layers (~25–40M params), batch 128, seq ≤ ~32
- Step: `trainStepMoleculeMuon` (Julia-compatible Muon path), four losses
  (coord MSE, label CE, split Poisson, deletion BCE) with flow time-weighting
- Expected per-step FLOPs ≈ 1 TFLOP → on GB10 the step is overhead-bound, so
  per-step wall time is the metric that matters

## Environment

- Host: `spark-e626` (SSH alias, BatchMode works from this machine)
- Arch: aarch64 (Grace CPU) + NVIDIA GB10 (Blackwell, SM120-class)
- Repo state on spark-e626: see log entries below
- Local machine: macOS (Apple Silicon), used for commits; smoke tests CPU-only

## Procedure

1. Verify + commit the in-flight Julia-parity Muon work locally (build
   `BranchingFlowsMoleculeTrainGenerate` + `test_runner_experimental`, run
   `--filter BranchingFlows`, run `--profile smoke` on CPU).
2. Sync the branch to spark-e626; build there (GB10 target; check
   `TYR_GPU_*` env and vendored CUDA libtorch on aarch64).
3. Baseline: time N steps of the paper profile on GPU (batch 128, full arch).
   Report steps/sec and ms/step; compare with the ~1 TFLOP/step expectation
   (10–100 ms/step is sane for an overhead-bound small model; > 1 s/step is
   "massive overhead" territory).
4. If overhead is massive: profile the step (device placement of every op,
   CPU↔GPU transfers, `nn.item`/host reads per step, Newton–Schulz cost,
   loss reduction on host vs device, allocator churn). Fix the top cause,
   re-measure, iterate. One change per iteration, numbers logged below.
5. Write the conclusion: measured steps/sec → projected 800k-run wall time
   and cost on this host.

## Baseline invocation (spark-e626)

```bash
cd ~/dev/tyr-qm9-perf
.lake/build/bin/BranchingFlowsMoleculeTrainGenerate \
  --profile paper-qm9-main --device cuda \
  --steps 300 --total-steps 300 --no-generate --no-checkpoint \
  --out-prefix /tmp/qm9_perf/base
time  # wall time / 300 = ms per step
```

Synthetic fixture data is fine for timing (shapes match the profile); only
absolute loss values are meaningless.

## Overhead suspects to audit (from code reading)

1. **Per-leaf host sync in Muon**: `moleculeMuonOrthogonalize`
   (`Tyr/Model/BranchingFlows/MoleculeTrain.lean`) does
   `nn.item (nn.maxAll (nn.abs flat))` per parameter leaf per step — one
   GPU→CPU sync each; ~100+ leaves means 100+ syncs/step. Prime suspect.
2. **Per-step loss readback**: `BranchingMoleculeLossReport` carries `Float`s,
   so losses are `nn.item`-ed every step even though they're only printed
   every 50. One sync, minor vs (1).
3. **Mask/flow-scale recomputation per step**: `moleculeMask`,
   `moleculeFlowLossScale` — check they're built on-device, not on CPU then
   uploaded.
4. **Newton–Schulz iterations** (`Optim.PolarExpress.muonOrthogonalize`):
   real GPU compute; fine — but check it's not running on CPU tensors for
   leaves that never got moved to the device.
5. **Batch packing**: `packBranchingMolecule` runs the bridge sampler on CPU
   per batch and uploads — expected; only a problem if it dominates step time.

## Code-audit findings (pre-baseline)

- **CONFIRMED — per-leaf sync is pure overhead**: the C++ Newton–Schulz
  (`cc/src/tyr_polar.cpp:297-299`) normalizes with an epsilon guard
  (`X / (norm + 1e-7)`), so a zero gradient orthogonalizes safely to zero.
  The Lean-side `nn.item (nn.maxAll …)` zero-check in
  `moleculeMuonOrthogonalize` forces a GPU→CPU sync **per parameter leaf per
  step** (~150 leaves at full arch) with zero behavioral benefit. Iteration 1
  candidate: drop the host check, always call the extern.
- `clipGrads` only runs when `cfg.gradClip > 0` (check the paper default);
  the per-leaf `nn.clip_grad_norm_` path is inert if 0.
- `packBranchingMolecule` on CPU + one `toDevice` upload per step — by design.
- Per-step loss `Float` readback for the report: ~4 syncs/step — second-order.

## Measurements (GB10, batch 128, full arch, maxLen 64, synthetic fixture)

| run | commit | steps | wall | ms/step | notes |
|---|---|---|---|---|---|
| baseline | `c8152d3` | 300 | 8m38.6s | **1730** | sanity 5-step OK; loss 32.3→2.03 trains fine |
| iter 1 | `a4bd3cf` | 300 | 8m24.9s | **1680** | sync removal: ~0 gain — hypothesis wrong |

### Iteration 1 result — syncs were NOT the bottleneck

300 steps in 8m24.9s (vs 8m38.6s baseline, inside noise). Loss curve
bit-identical (32.272911 → 2.027680), so semantics preserved; keep the change
(it's still ~150 fewer FFI calls/step) but the dominant cost is elsewhere.
Key signal: `user` time is ~87% of wall (7m17s/8m25s) — the process is
CPU-bound in user space, not sync-stalled. sys time 1m6s is also high
(allocation/IO churn?).

### Attribution matrix (running)

100-step timings varying one axis from base: `layers=2` (compact) vs
`batch=8` vs `full batch=128`, plus `nvidia-smi` utilization sampling.
Interpretation: layers-dominant → forward/backward/optimizer; batch-dominant
→ CPU bridge sampling + packing; low GPU util throughout → CPU-bound
confirmed.

### Attribution result — CPU sampling/packing per molecule dominates

| run (100 steps) | wall | ms/step | user/step |
|---|---|---|---|
| layers=2, batch=128 | 1m54.8s | 1148 | 1123 |
| layers=12, batch=8 | 0m22.9s | 229 | 211 |
| layers=12, batch=128 | 2m50.2s | 1702 | 1478 |

- Batch is the dominant axis: +120 molecules adds ~1.5 s/step → **~12 ms of
  CPU work per molecule per step** (bridge sampling + packing in pure Lean).
- Layers are near-irrelevant (6× layers = 1.5× time at fixed batch 128), so
  the GPU model + optimizer is NOT the bottleneck (rough split at batch 128:
  ~1.4–1.5 s CPU sampling/packing vs ~0.2 s model/optimizer).
- `nvidia-smi --query-gpu` syntax rejected by the GB10 driver build; util
  sampling skipped — the user-time evidence is conclusive enough.
- Conclusion: attack the sampling path (`sampleMoleculeBridgeBatch` →
  `branchingBridge` per molecule), not the GPU side.

### Phase split (`c0fe733`, `--time-phases` instrumentation)

100 steps, batch 128, full arch: **sample 988 ms/step vs train 672 ms/step**
(total 1660, matches the 1702 measurement). The sampling phase is the pure
Lean `forestBridge` path — embarrassingly parallel across molecules; Grace
has 20 cores.

### Iteration 2 — parallel sampling (`45d1f06`)

Change: `sampleMoleculeBridgeBatch` gains `parallelism : Nat`; when > 1 the
batch is chunked across `IO.asTask` tasks (decorrelated per-chunk LCG
streams via a golden-ratio jump; results concatenated field-wise; RNG
semantics vs the sequential path documented in the code). Exe gains
`--parallel-sampling n`. Expectation: sampling ≈ 1000 ms → ~100–200 ms at 16
chunks, bounded by memory bandwidth and the sequential merge.

### Iteration 2 result — no speedup, and why

`--parallel-sampling 16`: sample 920 ms vs 988 baseline — nothing. Root
cause: `IO.asTask (pure (f x))` evaluates `f x` **eagerly at the call site**
(Lean is strict for constructor arguments), so every chunk was computed
sequentially at spawn time and the tasks only wrapped finished values.

Fix (iteration 2b): `Task.spawn (fun () => f x)` — takes a genuine thunk, so
the bridge computation runs on the worker thread. Local tests 44/44 green
throughout (sequential path untouched; default `parallelism := 1`).

### Iteration 2b result — 8.8× on sampling, 2.2× overall (`3e66dff`)

`--parallel-sampling 16`, 100 steps, batch 128, full arch:

- **sample 988 → 112 ms/step**; train 629 ms/step (unchanged)
- **total 1702 → 763 ms/step** (wall 1m16s vs 2m50s for 100 steps)
- true parallelism confirmed: `user` 2m46s > `real` 1m16s
- loss curve identical to the sequential-chunk run (33.81 → 16.76)

The train step is now 82% of the step. Next: sub-phase instrumentation of
`trainStepMoleculeMuon` (pack / forward+loss / backward / Muon) to locate the
remaining 629 ms; candidates afterwards: batched/fused Muon ops, SDPA fusion
in the molecule transformer, and sampling/training pipelining (overlap next
batch's sampling with the current train step, worth ~110 ms/step).

### Iteration 3 — train sub-phase attribution (`1c04063`)

`trainStepMoleculeMuon` instrumented behind `--time-phases` with
`cuda_synchronize`-bracketed phases. Step-50 split: **pack 4, fwd 212,
bwd 340, muon 71** — the model (fwd+bwd = 552 ms) is 87% of the train step;
Muon is a non-issue.

### Iterations 4–5 — SDPA fusion: no win on GB10, plus a real bug found

Swapped both molecule attention blocks (compact slope-bias, full RFF-bias)
from the manual matmul→bias→masked_fill→softmax→matmul pipeline to a new
`lean_torch_sdpa_4d_bias` fused binding (`71681ae`). Two findings:

- **Bug (mine)**: the mask term was built from `torch.zeros`, a CPU tensor —
  `masked_fill` on CUDA hard-aborts (device mismatch; gdb backtrace:
  `masked_fill__cuda` ← `Full...spatialAttention`). Fixed by computing the
  mask term arithmetically from the on-device `keyMask0` (`b8fd11a`).
- **No speedup**: with the fixed mask, fused SDPA measures identically
  (fwd 212, bwd 330). GB10 is bandwidth-bound (~273 GB/s unified); the L×L
  work is traffic, not launches. A mid-flight attempt to force the math
  backend was reverted (`d9944c5`) — the original crash was the mask bug,
  not kernel availability.

Probe that guided the real fix: `--max-len 32` (quarters L² work) cut
fwd 212→72, bwd 340→112 — L²-proportional traffic dominates.

### Iteration 6 — hoist shared RFF pair features (`93dee88`)

`pairAttentionBias` recomputed the full RFF pipeline (pairwise distances +
shared `pairRffW` embedding + sin/cos/cat over 524k×128 = 268 MB tensors)
**per layer**, although it only depends on the running coordinates. Now
computed once and recomputed only after coordinate updates (7× instead of
12× per forward at layers=12, coordUpdateLayers=6). Result (step-50):
**fwd 212→172, bwd 336→286; total step 763→674 ms.**

### Iteration 7 — overlap sampling with training (`811f412`)

The train loop now pre-spawns the next batch's sampling as a `Task`
(`sampleMoleculeBridgeBatchPure` added; rng chaining keeps the draw order
identical to the sequential loop). Result: **697→675 ms/step (~3%)** —
small, because the sampler already runs in 16 chunk-tasks and the Grace
cores are nearly saturated by the training thread + sampler pool (the
`.get` wait ≈ full sampling duration ⇒ CPU contention, not scheduling
overhead). Loss at step 50 is bit-identical to iteration 6 (16.452284),
verifying the pipelined RNG order.

## Conclusion

| stage | ms/step (batch 128, full arch, maxLen 64, GB10) |
|---|---|
| baseline (`c8152d3`) | 1730 |
| + parallel sampling (`3e66dff`) | 763 |
| + RFF hoisting (`93dee88`) | 674 |
| + sampling/train overlap (`811f412`) | **675** |

**2.6× overall.** Step-50 breakdown at the end: fwd 172, bwd 288, muon 70,
sample ~110 (partially overlapped), pack ~3.

Verdict on "is Tyr's overhead massive": it *was* — and it was not one thing:
(a) pure-Lean per-molecule sampling at ~1 s/step (fixed with task
parallelism), (b) an accidental 12× recomputation of the shared RFF pair
pipeline (fixed by hoisting), (c) minor: per-leaf host syncs (removed),
mask built on the wrong device (fixed). What remains is intrinsic model
cost: forward+backward ≈ 460 ms/step is dominated by L²-scale tensor
traffic on GB10's 273 GB/s unified memory — not framework overhead.

Cost projection for the 800k paper run:

- **On spark-e626 (GB10)**: 800k × 0.675 s ≈ 150 h ≈ 6.3 days. Feasible but
  slow; fine as a free on-prem run.
- **On a datacenter GPU (H100, ~12× the memory bandwidth)**: the
  bandwidth-bound fwd+bwd should collapse (est. 80–120 ms/step) →
  **18–27 h, ≈ $50–80** at current rates — consistent with the original
  $25–75 estimate. Recommended: benchmark 500 steps on one pod first
  (the `--time-phases` flag gives the breakdown directly).

Remaining ideas (not done — diminishing returns / parity risk):

- Fused C++ kernel for dist→RFF→sin/cos→bias (collapses ~10 ops × 7
  recomputes in fwd+bwd; est. −100–150 ms/step on GB10).
- bf16 compute path: halves bandwidth but needs a numerics-parity check
  against the Julia reference (its precision is unverified).
- Muon leaf batching (~30 ms/step).
- The fused-SDPA binding (`lean_torch_sdpa_4d_bias`) is kept: no win on
  GB10, but on H100 the fused kernel avoids materializing the scores chain
  and should outperform the manual pipeline.

Reproduction commands live in "Baseline invocation" above; every iteration
was verified with `Tests/TestBranchingFlows` (44/44) and loss-curve
comparison (bit-identical where semantics required it).

## Roofline check (from the logged config)

Is 675 ms/step at the GB10's roofline? **No — ~7–15% of both ceilings;
latency-bound, not hardware-bound.**

- **Compute**: fwd ≈ 500 GFLOP, fwd+bwd ≈ 1.5 TFLOP/step (41 GFLOP/layer:
  QKV+O ≈ 19, MLP ≈ 19, attention bmms ≈ 1.6, RFF bias affine ≈ 1.6). At
  460 ms fwd+bwd → ~3.3 TFLOPS achieved vs ~20 TFLOPS fp32 peak → ~15%.
- **Bandwidth**: ~9–12 GB/step (RFF pipeline 4.7 GB after hoisting, scores
  chain ~1.2 GB fwd, bias affines 0.3 GB, backward ~2×) at 460 ms →
  ~20–25 GB/s vs 273 GB/s spec → ~8%.
- **Actual bound**: per-op latency. Calibration from the muon phase: ~1200
  small ops in 70 ms ≈ 60 µs/op; fwd+bwd carries ~1.5–2k ops at similar
  effective cost ≈ the measured 460 ms. Grace/aarch64 eager dispatch is
  expensive per op; tiny tiles also quantize badly on the GPU.

Levers in order of leverage: (1) op-count collapse (hand-fused RFF kernel,
CUDA graphs) — attacks the real bound; (2) bf16 (halves traffic, needs
Julia-precision parity check); (3) H100-class hardware (12× bandwidth +
faster dispatch CPU) — helps, but even there the stack stays below
roofline, just at acceptable absolute cost.

## Roofline push (phase 2)

Constraints: no problem-specific C++; generic FFI infrastructure and Lean-
or DSL-level changes are in scope. Specialised-kernel exploration goes
through Tyr's own GPU DSL (`@[gpu_kernel]` codegen — `nvcc` confirmed at
`/usr/local/cuda/bin/nvcc` on spark-e626, though off the default PATH).

Decomposition being attacked (per step, GB10, fp32):

| component | ms | nature |
|---|---|---|
| fwd+bwd | ~460 | mix: big-op traffic (near-roofline) + small-op latency |
| muon | ~70 | ~1200 small ops, latency-bound |
| sample | ~110 | partially overlapped; CPU-bound |
| pack | ~3 | negligible |

Planned levers, in order:

1. **bf16 compute path** (`--dtype bf16`): halves big-op traffic and doubles
   tensor-core rates. Lean-only via `toBFloat16'` + `castLike` (rope freqs).
   Optimizer also runs bf16 in this experiment — momentum-precision risk is
   acceptable for a measurement; loss-curve drift vs fp32 is the gate.
2. **Muon leaf batching**: group same-shaped leaves through
   `PolarExpress.applyBatch`-style stacking; targets the ~70 ms muon phase.
3. **CUDA-graph capture** (generic FFI infra): kill per-launch latency for
   the static-shape step. Requires static buffer discipline (copy-in/out of
   the packed batch + params), since the functional param-update style
   reallocates leaves every step.
4. **GPU DSL fused RFF kernel** (dist→rff→sin/cos→bias): forward-only, so it
   fits the *generation* path (10k×1000 sampling); training needs backward
   through the kernel, which requires autograd registration — noted as the
   boundary of the no-problem-specific-C++ constraint.

### Phase-2 measurements

| run | commit | config | ms/step | notes |
|---|---|---|---|---|
| bf16 | `d67edff` | bf16 params+activations | **435** | fwd 172→96, bwd 288→151, muon 66, train 321; loss drift ~4.6% at 50 steps (33.89 vs 33.81 init) — acceptable for the experiment, parity decision pending for the paper run |

bf16 is the single biggest generic lever so far (4.0× total vs baseline).
Remaining at 435 ms/step: sample ~110 (partially overlapped), train 321 —
of which the small-op latency share is now the main attack surface
(CUDA-graph capture), with muon leaf batching as a minor follow-up.

## Progress log

### 2026-07-17 — setup

- spark-e626 reachable via SSH: aarch64, 1× NVIDIA GB10. Existing
  `~/dev/tyr` checkout belongs to other GPU work (dirty, different branch) —
  left untouched; using dedicated worktree `~/dev/tyr-qm9-perf` tracking
  `origin/ranvier-labs/event-skeleton`, `external/wheels/torch` symlinked from the
  main checkout (CUDA build), `~/dev/lean-urdf-typeprovider` present.

### 2026-07-17 — in-flight Muon work verified and committed

- Local verification of the Julia-parity Muon diff: build green (1055 jobs,
  `BranchingFlowsMoleculeTrainGenerate` + `test_runner_experimental`),
  `--filter BranchingFlows` 44/44 pass, CPU `--profile smoke` end-to-end OK
  (total loss 7.51 → 2.70 over 260 steps, `optimizer=muon`,
  coord_weight=10, label_weight=1/3, checkpoint/trajectory/.xyz export fine).
- Committed as `e775b4f` (Muon path), `f5b3174` (demo wiring), `6918d43`
  (tests), `442db58` (paper scripts), `55f7b15` (docs); pushed to origin
  through `c8152d3`.
- First full build of the worktree on spark-e626 started at base commit
  `dad87f8` (warms `.lake` for shared deps; BranchingFlows commits will be an
  incremental rebuild on top).
