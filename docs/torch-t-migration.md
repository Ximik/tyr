# Removing the legacy tensor alias

The intended public tensor type is `torch.Tensor σ`. The legacy
`torch.T shape` is a transparent alias for a single native handle type: different
shapes are definitionally equal. It must not be used to claim shape safety.
This migration is in progress; the alias has not yet been eliminated.

## What actually needs a raw handle

`cc/src/tyr.cpp` implements `borrowTensor` and `giveTensor` using Lean external
objects containing `c10::TensorImpl` pointers. Native extern declarations need
that representation. CUDA codegen launchers also consume native handles while
their tile IR tracks separate shape/layout information. `torch.RawTensor`
names that representation without attaching a misleading shape index.

Neither boundary needs the legacy `T shape` alias. Model parameters, activations,
optimizer leaves, and checkpoint records should use `Tensor`; heterogeneous
collections can carry existential specifications. Keep raw handles inside
explicit FFI/interoperability boundaries and validate metadata when accepting
raw data into an indexed tensor.

## Inventory

A lexical audit on 2026-10-12 found 288 Lean files with 6,432 lines mentioning
the exact identifier `T` across `Tyr`, `Examples`, `Tests`, `launch`, and `dev`
after the first migration slices. These are search candidates, including
comments and possibly unrelated generic variables, not a semantic count of
legacy tensor applications. Search generated source strings and quoted Lean
names too: changing only ordinary annotations misses generators and reflection.

| Area | Files with candidates | Lines | Required migration |
| --- | ---: | ---: | --- |
| `Tyr/Model` | 68 | 4,196 | Typed weights, activations, loading, cache operations |
| `Examples` | 55 | 584 | Typed model/training APIs and explicit loader boundaries |
| `Tests` | 61 | 482 | Typed fixtures, negative compilation tests, raw FFI probes |
| Top-level `Tyr/*.lean` | 10 | 473 | FFI, TensorStruct, autograd and aggregate APIs |
| `Tyr/Optim` | 7 | 168 | Typed optimizer state and traversal |
| `Tyr/GPU` | 28 | 151 | Raw launcher ABI vs indexed high-level operations |
| `Tyr/Manifolds` | 14 | 101 | Indexed manifold carriers and transforms |
| `Tyr/Modular` | 6 | 94 | Typed model/metric interfaces |
| Remaining directories | 39 | 183 | Data, AD, providers, simulation, training and dev code |

Reproduce the initial candidate search with
`rg -n '\bT\b' Tyr Examples Tests launch dev --glob '*.lean'`.
It is an inventory aid, not the completion criterion.

## Audited dependencies and current changes

- GPT parameters, activations and token inputs now use `Tensor`. The exact
  off-by-one projection initializer is rejected by Lean. Legacy data loaders
  enter through checked `forwardRaw`/`lossRaw` adapters.
- SafeTensors' default generated loaders and hierarchical records now return
  `DTensor shape dtype`, just like its explicit `Typed` loaders. The previous
  defaults still returned `T` and bypassed dtype validation. Single-file,
  sharded, indexed and nonuniform hierarchy fixtures cover the generator.
- KittenTTS still consumes legacy model structures. Its checkpoint loader has
  a local conversion from typed weights to that interface; this is a migration
  boundary to remove with the KittenTTS model/layer migration, not a public
  coercion back to `T`.
- CUDA codegen's untyped launcher alias now explicitly uses `RawTensor`.
- The indexed `Tensor` carrier, `StaticSpec` and device policy now live in
  `Tyr.Basic`. The legacy alias and its metadata adapters are isolated in
  `Tyr.Legacy.Tensor`, re-exported by `Tyr.Torch` for remaining callers.
  Importing `Tyr.Typed` alone does not expose `torch.T`, enforced by a standalone
  compilation guard. Checked tensor conversions have no legacy dependency.
  `Tyr/Typed/Ops.lean` calls the native adapter in `Tyr/RawTensor.lean` directly.
  Shape and dimension arguments required by the existing C ABI are passed
  explicitly from the static specification; they are not implicit raw-type
  indices. Broadcasting no longer uses dummy shape-erasure reshapes.
- `cc/src/tyr_raw_tensor_abi.h` checks all 75 adapter declarations against the
  C++ definitions, and CI compares that header with Lean's generated C. This
  exposed pre-existing C++ signature mismatches (dimension parameters,
  Boolean widths, and signed integer bit representations), now corrected.
- `Tyr/AD/Frontend/Elab.lean` now recognizes `Tensor` (including `DTensor`
  aliases) and extracts shape and known dtype from `StaticSpec`. Its primitive-derivation
  tests now use indexed tensors. Legacy recognition remains until its callers
  migrate; frozen participation now reads the indexed `Frozen σ` specification.
- `Frozen σ` now stores `Tensor σ`; its map/get helpers preserve the complete
  specification. Typed models derive structural schema and flatten/rebuild
  instances, and flattened leaves now carry `Sigma Tensor`. Rebuild validates
  source metadata and destination shape, dtype and device policy, with field
  paths in diagnostics. Mixed-dtype round trips and malformed shape/dtype/device
  rejection run on the native CPU path.
- AD structural schemas retain known dtypes, which flow into frontend signature
  bindings. Frontend validation checks native leaf metadata before evaluation,
  including handles whose claimed shape disagrees with their runtime shape.
  A focused native run passes all 28 tests across `TestADTensorStructFlatten`,
  `TestADTensorStructSchema`, `TestADFrontendSignature`, `TestADFrontendElab`, and
  `TestGPTTyped`; the dynamic loader confirms the rebuilt native runtime was used.
- `TensorStruct` callbacks now quantify over `StaticSpec` and preserve
  `Tensor σ` directly. The traversal core no longer imports `Tyr.Torch` or
  mentions `T` in its executable API. Compile-negative tests reject callbacks
  that change shape or dtype. `Frozen` and container traversal use this contract.
- AdamW, SGD, schedules, gradients, detach and tree arithmetic now use typed
  operations. Native scalar operations in generic parameter trees validate
  that the result retains its input specification; integer-to-float promotion
  cannot silently change an indexed parameter's dtype.
- Checkpoint traversal now loads against the static template specification and
  validates shape, dtype and device. Typed BF16/Float32 mismatch rejection,
  BF16 round-trip, interrupted publication, snapshot consistency, optimizer
  resume and GPT training passed in a 33-test native run alongside AD schema tests.
- The model serialization fixture now stores `Sigma Tensor` and rebuilds through
  checked metadata. Its unchecked cast and incorrect claim that legacy tensor
  shapes were distinct Lean types have been removed; a wrong-shape state entry
  is explicitly rejected.
- The temporary `T` leaf instance is confined to `Tyr.Module.Derive`. It reads
  actual runtime shapes (legacy `T #[]` fields can contain arbitrary buffers)
  and leaves dtype dynamic. Legacy model fields, optimizer states and their
  schema instances still require migration before that bridge can be removed.
- Adjoint and Riemannian tree reductions now use native sums instead of
  reconstructing a sum from a mean and an unchecked `T.shape` element count.
  Riemannian cotangent reshaping also reads the actual output shape. Both
  native linear-ODE adjoint gradient tests pass.
- Typed defaults now allocate zeros matching the complete specification,
  including concrete dtype and pinned device. Device moves check the static
  placement policy. Native CPU tests cover defaults for all 12 concrete dtypes.
- Sphere and Poincare-ball points and tangent vectors now use indexed tensors,
  parameterized by dtype and device policy. Projection, tangent projection,
  retraction and exponential-map operations preserve those indices. The sphere's
  zero-vector fallback and manifold zero tangents inherit the input's actual
  dtype and placement instead of allocating default CPU Float32 values.
  All nine additional-manifold tests pass, including Float64/pinned-CPU
  projection and nonzero tangent updates; compilation rejects mismatched shape,
  dtype and device indices. Neither migrated module uses a raw conversion.
- Positive matrices, oblique matrices, SPD matrices, PSD factors and elliptope
  factors now carry indexed tensors in their points and tangents. Their matrix
  operations use typed native adapters, and oblique/elliptope tangent projection
  uses axis reductions instead of temporary default-device tensors of ones.
  Generic preserving reductions validate that accumulation retained the dtype;
  the general typed reduction exposes the native accumulation dtype explicitly.
  The 13-test additional-manifold suite passes, covering Float64/pinned-CPU
  column and row tangents, metric maps, retractions, SPD inverse/log/exp, and
  PSD Gram matrices before and after updates. Negative compilation checks reject
  incompatible manifold shapes and dtypes. GPU execution remains unverified.
- Fixed-rank matrices and tangents now use indexed tensors. Exact slicing
  requires a valid axis and natural-number bounds proofs, preventing silently
  truncated slices and wrapped `start + len`. Fixed-rank points retain
  `k <= m` and `k <= n` proofs for later retractions. The existing algorithm is
  documented accurately as a factor-product approximation, not truncated SVD.
  Native tests cover zero rank, nonzero slice offsets, exact end boundaries and
  retractions; compilation rejects invalid axes, oversized ranks and overflowing
  slice ranges.
- The generic embedded-tensor manifold instance now accepts `Tensor σ`, and its
  Euclidean ambient-step fixture uses typed tensors. The manifold core no longer
  imports the legacy tensor API; remaining legacy manifolds import it explicitly.
- Modular embedding tables now store indexed weights, preserve dtype and device
  policy through traversal and normalization, and require Int64 token inputs.
  GPT shares the typed batched/vector embedding adapters instead of wrapping
  the native handles independently. Other atomic modules still use legacy
  Linear/LayerNorm representations and need their own migrations. Six
  embedding-related tests pass, including repeated-token gradient accumulation
  and Float64/pinned-CPU normalization, as do both GPT training/raw-boundary
  regressions using the shared lookup operations.
- `Affine` now stores indexed weights/biases and implements typed `Module`,
  `TensorStruct` and gradient extraction. Its old forward call passed weight,
  bias and input to native `linear` in the wrong order; it now computes
  `input @ weight + bias` with the declared `[input, output]` weight convention.
  The lowercase `differentiable` class lives in `Tyr.Differentiable`, and its
  tensor instance now preserves `Tensor σ` rather than relying on `T`.
- Module-interface examples use typed tensors and are checked in
  `Tests.TestModuleTyped`. Typed dropout is shared by those interfaces and GPT.
  Both module tests pass, including non-square Float64/pinned-CPU affine forward
  and weight/bias gradients, as do both GPT regressions after sharing dropout.
- `Conv2d` stores indexed weight and bias fields dependent on its runtime
  configuration. Initialization validates positive/divisible channel groups and
  kernel dimensions, calculates fan-in without UInt64 multiplication overflow,
  and uses a typed uniform initializer retaining the reference dtype/device.
  Sampling occurs in the reference dtype on CPU before device transfer, matching
  the existing random-factory strategy for GB10. Unsupported dtypes and invalid
  bounds return IO errors; Float64 samples are not generated via Float32 casts.
  All five typed-module tests pass, including grouped/non-square kernel shapes,
  unchanged template and bias values, uniform bounds across the four supported
  floating dtypes, and invalid configuration/bounds/dtype rejection.
- Widget property helpers and `#tensor` examples now take indexed tensors.
  `ToModuleDisplay` supports indexed model fields directly, including optional
  fields, and native statistics no longer require a legacy tensor alias.
  The one remaining widget compatibility instance reads actual native shape
  and dtype for old model fields; it must disappear with those models.
  `Tests.TestWidgetTyped` covers Float64/pinned-CPU values and gradients plus
  derived display of an optional Int64 field.
- `Tyr.DiffEq.Typed` now implements its operations directly on indexed tensors;
  its integration and timing fixtures also use indexed tensors.
- Grassmann, Stiefel, Orthogonal/SkewSymmetric and Hyperbolic manifolds now
  carry indexed tensors, parameterized by dtype and device policy. Stiefel and
  Grassmann retain a `p ≤ n` proof witness (used by the new proof-carrying
  decomposition ops). Hyperbolic computes its Minkowski inner product and
  projections with mask vectors instead of the legacy private `cat2` extern,
  whose declared return type did not match its runtime shape. The ambient
  (`EmbeddedManifold`) and dual-map (`DualMapGeometry`) adapters are typed with
  them. `Tensor.qrReduced`/`svdReduced` carry the tall-matrix precondition as
  an auto-param proof: wide inputs do not typecheck at concrete dimensions.
  The native `lean_torch_svd` triple is now nested `Prod`-shaped — it
  previously returned a flat 3-field ctor that Lean read as nested pairs, so
  the legacy `linalg.svd` (and `Grassmann.log`) read garbage at runtime.
- The distributed collectives and polar-iteration externs no longer take a
  shape argument the native side never read; both the Lean declarations and
  the C++ definitions dropped it. `dist.allGather`/`dist.reduceScatter` are
  return-allocated with result specs computed from the input spec and world
  size, and the list collectives share one spec across inputs and outputs, so
  no independent output shape can be claimed. `Tyr.Sharding` re-validates
  gathered/scattered handles against its static indices through `Tensor.ofRaw`,
  throwing on indivisible shards. `Sharding`, `DistAdam`, `NorMuon`,
  `ManifoldMuon`, `DualOptimizer`, `PolarExpress`, `RiemannianSGD` and
  `RiemannianTreeSGD` are fully typed; per-parameter optimizer states are
  generic over `σ : StaticSpec`. Muon's Stiefel retraction handles wide
  matrices via the transposed projection instead of the legacy mislabeled
  reduced QR. New typed ops with audited ABI entries: `eye`/`eyeFor`, `qr`,
  `qrReduced`, `svdReduced`, `svdvals`, `diagflat`, `spectralNorm`,
  `nuclearNorm`, `Preserving.atan`, `maxAll`, `stack1d`, `grad` (VJP),
  `fromFloatArray`, `multinomial`, `scatter2d`, `crossEntropy`, `layerNorm`,
  `onesFor`, `reshapeChecked`/`sliceChecked` (runtime-validated). The
  `max_all`/`min_all`/`multinomial` externs dropped their never-read shape
  (and phantom IO world) arguments in both Lean and C++.
- `torch.Linear`, `torch.LayerNorm` and `torch.RMSNorm` now store indexed
  weights and biases with dtype/device parameters; `RMSNorm`'s Float32
  accumulation is visible in the result type (`promote Float32 dtype`), and
  its `Module` instances are pinned to Float32 so the `isFloating` evidence is
  `rfl`. `Tyr/Modular/{Manifold,RiemannianModule,MetricFactor,Atomic}` are
  typed with them: manifold carriers assert real shape constraints
  (`p ≤ n` for Stiefel/Grassmann, square for Orthogonal), and the Jacobian
  row-materialization paths validate runtime slices against their claimed
  specs instead of reshaping through the erasure alias.
- The shared-library build (`lake build Tyr`) completed successfully across
  887 jobs. The full `lake build Tests` target passed across 1,385 jobs, and
  `lake test -- --fail-fast` passed all 1,406 main CPU tests with no failures
  or skips. Separate experimental, model qualification and GPU suites remain
  outside this result.

## 2026-10-13 continuation checkpoint

This is a validated migration slice, **not completion of the migration**. The
older full-suite pass above describes an earlier checkpoint; it is not the
current status after the intervening model and shared-API changes.

- SileroVAD weights, recurrent state, convolution, and streaming input use
  indexed Float32 tensors. Its existing pretrained checkpoint test executed
  against the locally available Silero checkpoint.
- Standalone Qwen3 carries Int64 token IDs, Boolean masks, and Float32 model
  activations. Its cached/uncached and batched EOS regressions pass. Qwen3 and
  Qwen2.5-Omni thinker loaders explicitly convert checkpoint storage to the
  Float32 specification already required by the shared Qwen blocks.
- Generated SafeTensors fixtures exercise 16 combinations: Qwen3/Omni,
  Float32/BFloat16 storage, single-file/indexed-shard loading, and tied/separate
  LM heads. Each loaded model validates and executes a finite-logit forward.
  This is loader coverage, not large-model numerical or GPU qualification.
- Typed convolution computes its spatial index using natural arithmetic and
  validates native result metadata. Numerical tests cover padding, stride,
  and dilation; negative compilation tests reject channel and dtype mismatches.
- The native convolution argument widths and host-array-copy signatures now
  agree with Lean's generated C. Raw-tensor ABI validation passes for 110
  declarations; the existing general ABI audit passes for 11 declarations.
- Typed cache tests exposed partial mutation on malformed value handles:
  keys were written before the value shape was rejected. Append now validates
  both inputs and both buffers before either write. Attention validates query
  metadata before append as well. Rejection tests verify unchanged buffers.
- AD, gradient/widget, cache, token-stream, Brownian, and ODE/SDE adjoint
  fixtures now consume typed APIs. Adjoint tree reduction explicitly weakens
  the placement policy of scalar leaves while retaining shape and dtype.

Reproducible focused acceptance:

```sh
lake build test_typed_models
lake env .lake/build/bin/test_typed_models
python3 scripts/check_ffi_abi.py --header cc/src/tyr_raw_tensor_abi.h --generated .lake/build/ir/Tyr/RawTensor.c
python3 scripts/check_ffi_abi.py
```

The native runner passes **69 tests, zero failures or skips** on the local CPU.
`lake build Tests Examples` still fails in these ten prerequisite targets:

- `Tyr.Model.FLM`
- `Tyr.Model.Qwen3TTS.SpeechTokenizer25HzEncoder`
- `Tyr.Model.Qwen3TTS.SpeakerEncoder`
- `Tyr.Model.Qwen3TTS.Talker`
- `Tyr.Model.Qwen3TTS.SpeechTokenizer`
- `Tyr.Model.Qwen3ASR.AudioEncoder`
- `Tyr.Model.Qwen35.Model`
- `Tyr.Model.KittenTTS.Model`
- `Examples.GPT.GPTDataLoader`
- `Examples.NanoChat.Eval.CORE`

These are the currently exposed build failures, not an exhaustive inventory of
unmigrated modules. Fixing them can expose downstream callers. `torch.T` and
legacy bridges remain; full CPU/experimental and GPU qualification are still
outstanding. GitHub CLI reports successful CI for older commit
`05304ff22a1cf5f17df15eef372fc2a4bdf9f74c`; that run does not cover this branch.

## 2026-10-13 ASR and training-data continuation

- FLM states, losses, token inputs and generated IDs are indexed. GPT sequential
  loaders and batch iterators retain their batch/sequence indices. Random batches
  validate the full requested window and sample an integral offset, including
  the exact-fit case, instead of routing the index through floating point.
- Qwen3-ASR audio encoding, thinker, checkpoint loading, forced alignment,
  streaming caches and offline transcription use indexed tensors. Variable
  shape collections carry existential indices; frontend features are Float32,
  masks/IDs Int64, and checkpoint-dependent activations retain dynamic dtype
  claims rather than being forcibly converted to Float32.
- Shared Qwen blocks now parameterize dtype so ASR BF16 checkpoints remain
  usable. RMSNorm restores activation dtype before multiplying learned weights,
  preserving HF inference rounding. KV stores align with the projected key/value
  dtype before append. Rotary and attention operations use typed adapters over
  explicit raw FFI declarations rather than legacy `T` wrappers.
- `Tyr.Typed.Neural` provides checked convolution, heterogeneous concatenation,
  gather, masked selection/scatter, attention, rotary, and audio operations.
  Its 19 native declarations have a compiled header and generated-C ABI audit,
  now included in CPU CI and GPU qualification. This caught incorrect native
  dimension widths in rotary/FlashAttention and the obsolete WAV IO-world
  argument. The core raw-tensor audit now covers 111 declarations.
- The native runner passes **134 tests, zero failures or skips** on local CPU.
  This includes BF16 ASR feature conversion and variable-length generation,
  streaming reuse, exact BF16 RMSNorm rounding, FLM, token loaders, signed
  randint/gather, masked selection/scatter, and convolution/concatenation.
  Four formerly skipped WAV regressions now run, including offline transcription
  and generation. Their platform skips have been removed.

```sh
lake build test_typed_models
lake env .lake/build/bin/test_typed_models
python3 scripts/check_ffi_abi.py --header cc/src/tyr_neural_abi.h --generated .lake/build/ir/Tyr/RawTensor/Neural.c
python3 scripts/check_ffi_abi.py --header cc/src/tyr_raw_tensor_abi.h --generated .lake/build/ir/Tyr/RawTensor.c
python3 scripts/check_ffi_abi.py
```

This is a verified continuation checkpoint, not full migration acceptance.
The preceding ten-target failure list is historical: FLM, ASR, and the GPT
loader now build. TTS and other model families, remaining callers and legacy
infrastructure still require migration, followed by the full CPU/experimental,
example, checkpoint, AD, and hardware GPU gates below.

## 2026-10-13 TTS continuation

- Qwen3-TTS talker, residual code predictor, speaker encoder, 12 Hz encoder and
  decoder, 25 Hz encoder, checkpoint loading, bridge and streaming interfaces
  now use indexed tensors. Codec IDs are Int64; floating parameters retain
  checkpoint-dependent dtype. Dynamic waveform chunks carry sample counts and
  decoder state retains its total-frame tensor index. No legacy `T` applications
  remain in `Tyr/Model/Qwen3TTS`.
- Typed grouped/transposed convolutions, sliding-window attention, comparison,
  masking, ELU and SafeTensors saving extend the audited native adapters.
  The neural ABI audit now covers 29 declarations, including correction of the
  obsolete single-tensor SafeTensors IO-world parameter.
- Native acceptance passes **154 tests, zero failures or skips**. Added coverage
  includes BF16 talker generation, cached/full Qwen parity and immutable cache
  snapshots, speaker dtype alignment, streaming callbacks, grouped/transposed
  convolution, and Int64 empty-generation outputs.
- Eight generated SafeTensors fixtures cover TTS Float32/BFloat16 storage,
  single-file/indexed-shard loading, and explicit/missing optional code-predictor
  projections. Loaded models retain their dtype, execute finite-logit forwards,
  and generate codec frames. Missing projections use a dtype-matched identity.
  This does not qualify full pretrained speech-tokenizer numerical parity or GPU
  execution; those remain part of final acceptance.

The last full `lake build Tests Examples` exposed Qwen35, KittenTTS, NanoChat
CORE/ModdedTrain and TTS prerequisites. The TTS family now builds, but downstream
callers and other model families still require work. The full migration goal
remains active.

## 2026-10-13 NanoChat continuation

- NanoChat's ModdedGPT, trainer, pipeline, CORE evaluator and sampling engine
  now use indexed Float32 activations and Int64 tokens. Streaming batches carry
  their batch and sequence dimensions; SFT callbacks retain them and carry the
  vocabulary size explicitly. CORE no longer assumes a 65,536-token vocabulary.
- Optimizer checkpoint state uses Float32 specifications and genuine scalar
  second moments. Loading rejects malformed nonscalar Muon moment metadata.
  Tensor-file loading validates both stored shape and dtype.
- Native Muon orthogonalization returns BF16 regardless of input dtype. Its
  interface now states that fact; NorMuon scales the result in BF16 before
  converting to the parameter dtype for subtraction. A regression checks the
  exact rounding order. A model dimension below 12 is rejected before slicing
  the smear-gate input.
- All **180 native CPU tests pass with zero failures, skips or runtime panics**,
  including forward/backward with clipping and optimizer updates, checkpoint
  replacement/corruption, legacy checkpoint loading, and a seven-token CORE
  model and the remaining serialization fixtures. The NanoChat pipeline and generator engine build. The neural ABI audit
  covers 32 declarations; gradient clipping and tensor saving no longer carry
  obsolete native IO-world arguments.
- GitHub CI remains green at `05304ff`, not at these local migration commits.
  Full aggregate tests/examples and hardware qualification remain outstanding.

## Remaining acceptance criteria

Remove the `T` declaration and all model, generator, reflection, documentation,
and test dependencies on it. A rename of `T` to another unchecked public tensor
alias does not satisfy this requirement. The raw carrier is allowed only where
native interoperability actually requires it.

Validate each migrated subsystem at its real acceptance path. Final validation
must include the full CPU test suites, example typechecking, native FFI ABI
checks, GPU qualification on actual hardware, generated checkpoint loading,
and AD metadata/derivation tests. Focused passes during the migration do not
prove these broader gates. No changes from this migration are to be pushed to
Ximik's PR branch.
