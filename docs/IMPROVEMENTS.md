# Improvement suggestions

Consolidated findings collected while writing the `docs/` guides, plus their
resolution status after a dedicated improvement pass. Every item was verified
against source. Tags: **[bug-risk]** likely wrong behavior ·
**[correctness]** misleading or incomplete semantics · **[docs]** stale/missing
documentation · **[testing]** missing coverage · **[cleanup]** dead code /
duplication · **[performance]** avoidable cost.

## Resolved

Items fixed in the improvement pass, with their commits.

### Core

- **[correctness]** Generic `Tensor.matmul` accepted inner-dim mismatches at
  compile time (junk `#[]` from the total `matmulShape`). Fixed: `Tensor.matmul`
  now requires a static proof via `TensorSpec.matmulShape?`
  (`26d78ec fix(typed)`).
- **[docs]** `Tyr/Torch.lean` module docstring referenced a nonexistent
  `torch.nanoproof` submodule; `focal_loss` docstring pointed at nonexistent
  `nn.cross_entropy_loss`. Both fixed (`177e7e2 docs(core)`).
- **[docs]** `TensorStruct.lean` docstring now states that
  `deriving TensorStruct` and the `T s` leaf instance require
  `import Tyr.Module.Derive` (`06bbf9c docs(core)`).

### Modules and modular

- **[bug-risk]** `Conv2d.outShape` returned the *input* channel count. Fixed to
  `out_channels` (`ddcf7b1 fix(module)`; no call sites existed).
- **[cleanup]** `Tyr/Module.lean` umbrella now re-exports `RMSNorm`
  (`5271b85 fix(umbrella)`).

### Optimization

- **[correctness]** NorMuon silently ignored `weightDecay`/`wdMul`. Decoupled
  AdamW-style decay implemented in `stepSingle` (`3372a0f fix(optim)`).
  Note: default `weightDecay := 1.2` now has real effect; set it to `0.0` for
  the old behavior.
- **[bug-risk]** `stepMatrixSingle` silently reset state — the actual hazard
  was a state/backend *constructor* mismatch (shape mismatch is
  unrepresentable). Now throws `IO.userError` (`351ddb4 fix(optim)`).
- **[docs]** Grassmann `distance` docstring said geodesic, implementation is
  chordal — doc fixed (`d8cd06d docs(optim)`), together with a note that
  `stepAdamLike` uses coupled L2 decay while `stepSingle` uses decoupled.

### Distributed and pipeline

- **[bug-risk]** `Pipeline.spawnBackground` awaited the task inside the spawn,
  making "background" tasks synchronous. The wait is now deferred to `await`
  (`10ac633 fix(pipeline)`).
- **[correctness]** `Sharding.fromFull`/`gather` ignored `shardDim` and
  `shardedShape` silently fell back to the unsharded shape. All three now
  handle arbitrary shard axes (gather permutes through dim 0 and throws loudly
  on indivisible axes) (`57d3f78 fix(sharding)`).
- **[docs]** `ChatSFT` dead config fields and GRPO's unapplied sampling
  parameters / batch-1 restriction are now documented in the module headers
  (`709e6c9 docs(train)`).

### Models

- **[bug-risk]** `sampleForest` index normalization shadowed `i`, collapsing
  reversed pairs to a self-merge that silently deleted a node. Fixed with
  `lo`/`hi` computed from the original pair (`4a65826 fix(branching)`).
- **[docs]** `Tyr/Model/Qwen.lean` header rewritten — it is the Qwen3 /
  Qwen2.5-Omni causal-LM substrate, not a Flux-only embedder
  (`07b3bca docs(data)`).

### Data

- **[docs]** `Pretraining.lean` comment claiming unimplemented parquet FFI
  corrected — `cc/src/tyr_parquet.cpp` implements it (`07b3bca docs(data)`).
- **[cleanup]** `Tyr/Data.lean` umbrella now includes `Download`/`HuggingFace`;
  new `Tyr.Text` umbrella is re-exported from `Tyr.lean`
  (`5271b85 fix(umbrella)`, `e7c7366 feat(text)`).

### Autodiff

- **[correctness]** The placeholder/hybrid rule packs in
  `Tyr/AD/JaxprLike/RulePackKStmt.lean` (numerically meaningless) now carry a
  prominent experimental-scaffolding notice (`6cf6968 docs(ad)`).
- **[cleanup]** Documented that the GPU VJP registry has zero registrations
  and `Tyr.GPU.AD.init` has no call sites (`d0afdce docs(gpu)`).

### GPU

- **[cleanup]** Removed the unimplemented `#print_gpu_kernel` and
  `#print_kernel_ffi` syntax declarations and the dead `scaleMatchesKernel`
  (`4e4d730 chore(gpu)`).
- **[docs]** `Kernel.family` doc comment corrected (no `GB10` constructor);
  `Readme.md`'s GPU parity section synced with the actual
  `test_parity_suite.sh` contents (`d0afdce docs(gpu)`, `215de6e docs(readme)`).

### DiffEq

- **[correctness]** Marker-trait audit result: **all** Itô/Stratonovich
  markers match current diffrax — including `SRA1`, which diffrax classifies
  as a Stratonovich solver (the original list item was wrong). Added the
  missing `ReversibleSolver ReversibleHeun` instance (`440ff34 fix(diffeq)`).
- **[cleanup]** Stale `Adjoint/Core.lean` header updated: Stratonovich SDE
  backsolve adjoints now exist (`cd6a428 docs(diffeq)`).

### Event-skeleton

- **[cleanup]** Dead move kinds `belNoise`/`itoPQ`/`dropSmallTimingTerm`
  removed (`68964f5 chore(event-skeleton)`); `Tyr.lean` now re-exports
  `Tyr.EventSkeleton` and `Tyr.MctxDag` (`5271b85 fix(umbrella)`).

### MCTS

- **[bug-risk]** `MctxDag.expandEdge` evaluated `recurrentFn` before the
  visited-edge check and overwrote existing edge reward/discount. Now returns
  early on visited edges (`7cefdee fix(mctx)`).
- **[docs]** Module headers added to all 14 `Tyr/Mctx`/`Tyr/MctxDag`
  submodules plus `Mctx/Math.lean` (`a857dea docs(mctx)`, `3c1f9bc docs(mctx)`).

### Audio and inference

- **[performance]** KV-cache append cloned the full buffer per token. New
  `sliceScatterInplace` FFI op + adoption in `KVCache.lean`; a latent
  cross-layer aliasing bug in `Cache.init` (one shared buffer replicated
  across layers) was fixed first (`8c067b6 fix(cc)`, `c0cb3f9 fix(inference)`).
- **[bug-risk]** Out-of-range `layerIdx` silently returned `newQ`; now panics
  with a descriptive message (`47607f8 fix(inference)`).
- **[bug-risk]** Off-macOS audio stubs failed inconsistently (`start` raised,
  `read`/`stop` silently no-op). All stubs now raise (`8c067b6 fix(cc)`).

### FFI and build

- **[cleanup]** Dead `cc/lxla.cpp`, `cc/plugin_init.cpp`,
  `lean_float_buffer_inhabited`, empty `HDRS :=`, and undefined
  `XLA_INC_FLAGS` removed (`4451a8b chore(cc)`).
- **[docs]** `Readme.md` now documents the required
  `../lean-urdf-typeprovider` sibling checkout (`215de6e docs(readme)`).

### Testing

- **[testing]** The 14 orphaned test files (12 DiffEq parity suites,
  `TestModularBudget`, `TestNanoGPTCopy`) are wired into `Tests.lean` and run
  in the main suite; five stray root-`main`s removed. `Tests/TestCheckpoint.lean`
  writes to a temp dir and cleans up instead of polluting the repo root.
  New `Tests/TestCoreSmoke.lean` covers PRNG determinism, `Log.Handlers`, and
  `TensorStruct` traversal. One wired test encoded a stale expectation
  (empty `SubSaveAt` leaf rejected with `internalError`); the library already
  rejects with the semantically correct `Result.invalidInput`, and the test
  was corrected.

## September 8 correctness pass

- **[correctness]** Audited tensor FFI declarations now agree with generated Lean
  C, including boxed indexing and unbinding, 64-bit factory arguments, and exact
  reshape/load operations. A shared header and CI ABI check prevent drift.
- **[correctness]** Checked typed constructors validate actual shapes. Typed
  reshape checks element counts and supports real scalars. Integral promotion
  handles UInt8/Int8 correctly and float8 metadata uses eight bits. Public
  examples distinguish the raw shape-erased alias from the checked typed facade.
- **[correctness]** Reverse dense interpolation and physical knot-side semantics
  are direction-aware. The former Newton/VeryChord heuristics are now accurately
  named NormRatioFixedPoint/AdaptiveFixedPoint, with their limitations documented.
- **[testing]** Numerical comparisons reject nonfinite values and unequal lengths;
  saltation boundaries validate dimensions and finite inputs/results.
- **[testing]** Required CPU suites are enumerated in `scripts/test_suites.json`,
  including restored Diffusion tests, TileIR specialization, and Laguna CPU
  runners. Strict GPU CI requires actual execution and fails skipped cases.
- **[correctness]** Native builds track compiler header dependencies, effective
  configuration, and generated launcher inventories, with isolated incremental
  and clean-build regression fixtures.
- **[correctness]** Checkpoints use versioned exact loss metadata and atomically
  published immutable snapshots. Combined training snapshots restore model,
  optimizer moments, and step together; regressions cover interruption, concurrent
  replacement, legacy metadata, bf16 templates, and tensor-count mismatches.
- **[performance]** SafeTensors schema discovery reads bounded headers only,
  without loading shard payloads; malformed and oversized headers are rejected.

## Remaining

Open items, roughly by area. These were deliberately not addressed in the
pass (scope, semantics risk, or hardware requirements).

### Core

- **[correctness]** The broadcast `HMul (T s1) (T s2) (T s2)` instance captures
  bare numeric literals (`w * 0.02` fails with `OfScientific (T …)`);
  documented in `docs/core/tensors.md` — an `OfScientific` instance or
  dedicated literal path would remove the papercut.
- **[cleanup]** `class differentiable` was *not* removed: it has a real
  instance in `Tyr/Module/Affine.lean:10` (imported via `Tyr/Module.lean`).
  The correct cleanup deletes/rewrites legacy `Affine.lean` together with the
  class.

### Serialization

- **[docs]** `safetensors_type_provider` failure is a generic message that
  swallows the underlying error (`TypeProvider.lean:628-632`). → Surface it.
- **[testing]** Hub coverage remains absent. The repo-root `dummy_path.ckpt/` directory left behind by earlier test runs
  can be deleted manually (the test no longer writes it).

### Modules and modular

- **[cleanup]** The `⊛` infix has zero call sites; `Affine`/`Conv2d` are
  legacy; no `NormedModule` instance exists for `RMSNorm`.

### Optimization

- **[correctness]** `NorMuon.stepAdamLike` uses coupled L2 decay while
  `stepSingle` uses decoupled — documented, but the paths remain numerically
  inconsistent.
- **[cleanup]** Gradient clipping is still a TODO (`Tyr/Optim.lean:299`).

### Distributed and pipeline

- **[correctness]** `ChatSFT.trainLoop`'s unused fields
  (`matrixLr`/`unembeddingLr`/`weightDecay`/`gradClip`) and GRPO's unapplied
  `temperature`/`topK`/`topP` are documented; implementing them is feature
  work, not a fix.

### Models

- **[cleanup]** VAE hardcodes `[1,128,16,16] → [1,3,256,256]` shapes and
  duplicates `forward`/`forward1` bodies.
- **[testing]** No Gemma4, Flux, or VAE tests (only demo executables).

### GPU

- **[cleanup]** `mhaFwdDispatch` maps the `.tkMhaH100Decode` variant to
  portable SDPA (`Ops/MhaH100.lean:176-177`) — dispatch asymmetry now
  documented in `docs/gpu/kernels.md`; aligning it is a kernel-routing
  decision.
- **[docs]** SM100 capability constants are marked "estimated" — needs
  confirmation against hardware documentation.

### DiffEq

- **[cleanup]** `Solution.ensureOkay` is a no-op. → Implement or remove.

### MCTS

- **Resolved:** Tree, batched, and DAG policies share seeded gamma-based
  Dirichlet and Gumbel sampling, honor `dirichletAlpha`, and sample actions
  from temperature-adjusted visits. Regression tests cover distribution
  moments, replay, masks, and all policy backends; see `docs/mctx.md`.

### Testing and CI

- **[testing]** 6 GPU example harnesses remain unreferenced (they need GPU
  hardware); `Tests/` still has no coverage for `Tyr.Audio`, `Hub`, `Widget`,
  `Inference` (hardware/editor dependencies); `parseArgs` is copy-pasted
  across ~11 test runners. → Factor out; add smoke tests where feasible.
- **[follow-up]** The in-place KV-cache path was verified by typecheck only —
  run `RunMhaH100Decode` on a CUDA/Hopper host to confirm numerical parity
  end-to-end.
