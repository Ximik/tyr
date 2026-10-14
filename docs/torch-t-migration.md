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
- KittenTTS uses indexed Float32 weights/activations and Int64 tokens. Its
  checkpoint loader checks runtime configuration dimensions and reports
  incompatible weights through IO errors; the local legacy coercion is gone.
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

## 2026-10-13 Qwen3.5 / Qwen3.6 continuation

- Qwen3.5 text, vision, media and checkpoint interfaces now use indexed tensors;
  Qwen3.6's shared interfaces build against them. Floating weights preserve
  checkpoint dtype, token/routing indices are Int64, finish masks are Bool,
  rotary tables are Float32, and GatedDeltaNet recurrent state is Float32 even
  for BF16 activations. No legacy `T` applications remain in either family.
- Explicit native adapters validate einsum outputs, top-k indices, recurrent
  kernel outputs and media patch shape/dtype. Their seven declarations are
  audited in CI. The audit corrected UInt64 dimension parameters on the recurrent
  kernels and obsolete IO-world arguments on Apple media loaders and stubs.
- **207 native CPU tests pass with zero failures, skips or runtime panics**.
  Coverage includes dense/MoE forwards, attention gate layout, cached/uncached
  and streaming parity, per-row EOS, multimodal fusion and a real PNG decoded to
  indexed Float32 patches on macOS. BF16 hybrid-cache tests verify Float32
  recurrent state and cached/uncached token parity.
- Sixteen generated checkpoint combinations cover dense/MoE, Float32/BF16,
  single-file/indexed-shard loading, and present/absent optional LM-head and
  convolution-bias weights. Loaded forwards and generation run successfully;
  tied embeddings and fallback bias retain the checkpoint dtype.
- Full pretrained-model numerical qualification, video decode success, and GPU
  execution remain unverified. KittenTTS and other legacy callers are still
  part of the active full-migration goal.

## 2026-10-13 KittenTTS continuation

- The complete KittenTTS model, vocoder, pretrained interface and checkpoint
  loader use indexed tensors. Dynamic vocoder weights and audio outputs carry
  their real shapes in `ShapePackedTensor`; they no longer pretend to be scalar
  tensors. TensorStruct traversal and AD rebuild preserve and validate these
  shapes, including rejection of a different shape with the same element count.
- Native adapter declarations are checked against a dedicated six-function ABI
  header in CPU CI and hardware qualification. The cumulative-sum argument now
  matches Lean's unsigned ABI representation of Int64 and explicitly recovers
  the signed axis; a negative-axis regression covers it.
- **212 focused native CPU tests pass with no failures, skips or runtime
  panics**. Tiny-model synthesis agrees with debug synthesis, and saving and
  reloading all indexed/shape-packed parameters reproduces exact durations
  and audio with a fixed seed. This is not full pretrained Kokoro numerical
  qualification.
- The aggregate `lake build Tests Examples`, KittenTTS family and all four
  standalone KittenTTS examples typecheck. Full migration is still active:
  optional model families, reflection/code generation and other legacy callers
  remain, and real GPU qualification is outstanding.
- Broad validation is not green: the experimental executable cannot build
  because `BranchingFlows/MoleculeTrain` still passes legacy tensors to indexed
  optimizer/traversal interfaces. The rebuilt main CPU runner terminates with
  SIGBUS/SIGSEGV during GPT optimizer validation after hundreds of tests; that same
  GPT test passes alone and in the focused suite. This requires isolation of
  preceding-test state or native memory corruption before acceptance.

## 2026-10-13 native stack ownership correction

- The main-suite native crash was caused by a borrowed/owned mismatch in
  `RawTensor.stack1d`: C++ consumes its tensor array, but the indexed declaration
  marked it borrowed, causing Lean to release the temporary array again.
  The declaration now transfers ownership, matching the existing native and
  legacy contracts.
- A repeated-stack regression reproduced SIGSEGV before the correction and
  passes afterward. All four RiemannianSGD tests, including the isolated
  Jacobian reproducer, also pass. Temporary diagnostic runner changes were
  removed. The 111-declaration raw ABI audit remains green; C type signatures
  alone cannot detect this ownership mismatch.
- The rebuilt main CPU suite passes **1,436 tests with zero failures, skips or
  runtime panics**. The focused 212-test suite also passes. `Tests`, `Examples`
  and both native test executables build. The experimental BranchingFlows
  build errors and the remaining legacy callers are still outstanding.

## 2026-10-13 BranchingFlows continuation

- Discrete/continuous/molecule batches, both molecule transformers, their
  training and prediction interfaces, and all BranchingFlows examples use
  indexed tensors. Labels and molecule masks are Int64. Floating tensors keep
  their shape indices while preserving Float32/BF16 runtime dtype. Public
  forwards match floating inputs to the model weights, including BF16 inference.
- Model/config maximum-length mismatches now return IO errors. The indexed
  interfaces exposed reversed affine weight dimensions in the discrete and
  mixed toy demos; their declarations and initializers now agree with the native
  output-by-input convention.
- Muon preserves native BF16 scaling before restoring the parameter dtype.
  Scalar and tensor learning-rate paths retain their respective rounding order.
  Tests check both against native BF16 directions rather than assuming that
  the two multiplication orders always round identically.
- Static molecule buffers and the mutable learning-rate scalar allocate through
  effectful clones. A regression exposed shared pure zero allocations: copying
  masks could overwrite labels. Each buffer now has independent storage.
- All **49 experimental CPU tests** and **1,436 main CPU tests** pass with zero
  failures, skips or runtime panics. Coverage includes Float32/BF16 forwards,
  backward/Muon updates, rotary parity, parameter checkpoint round trips,
  buffer independence, and rejection of incompatible static dtype conversions.
  All six BranchingFlows native declarations pass the dedicated ABI audit;
  additive-attention dimensions and the touch IO signature were corrected.
- All training examples build and run. Continuous overfit loss decreased from
  1.484375 to 0.000481; molecule loss from 3.579090 to 0.032313; transformer loss
  from 4.179966 to 0.000094. The mixed demo writes its four plotting tensors.
  The discrete and mixed demos now have native Lake executable targets.
- The full-architecture BF16 dataset CLI completed 80 CPU steps, checkpointed
  parameters and Muon state, and generated finite XYZ/JSONL artifacts. Resuming
  from step 80 loaded both states, completed two further steps, and generated
  again. This small fixture run is not QM9 replication or CUDA-graph
  qualification; actual GPU capture/replay remains unverified.

## 2026-10-13 Laguna continuation

Laguna's model, MoE router and expert banks, rotary tables, NVFP4 dequantizer,
checkpoint loader, generation APIs, standalone example, and six numerical
runners now use indexed tensors. The shared `Examples.ModelRunner` batching,
decoding, and streaming helpers carry Int64 token IDs. Floating weights retain
their checkpoint dtype through `STensor`; routing indices are Int64, finished
masks are Bool, rotary tables are Float32, and dequantized weights are BF16.
Runtime table and cache dimensions are existential indices rather than scalar
shape claims. Dense-checkpoint placeholder banks use broadcast views with true
bank shapes, so unused placeholders do not allocate full packed-bank storage.

The native Laguna boundary has four audited declarations. The fused kernel's
obsolete IO-world parameter was removed from both CPU and CUDA definitions.
Scatter-add now uses Lean's generated UInt64 C representation for Int64 and
converts it back to a signed dimension at the libtorch call; a negative-axis
regression checks the resulting values. NVFP4 validation rejects equal-numel
packed layouts with the wrong shape and non-scalar global-scale layouts.

Validation completed:

- All nine required CPU suites build under `--wfail` (1,668 build jobs) and
  pass via `lake env python3 scripts/test_inventory.py --run`. The main runner
  passes 1,436 tests and the experimental runner passes 49, each with zero
  failures/skips. All seven standalone Laguna CPU suites exit successfully.
- Rotary tables and partial/full rotary applications pass the existing Python
  reference comparisons, including decoding at position five.
- NVFP4 matrix/bank dequantization and the MoE fixture output match exactly on CPU.
- Full model tests pass FP32/BF16 cache checks, erased-history negative controls,
  sliding-window invariance, generation, ring wraparound, independent forks,
  snapshot preservation, capacity rejection, and failed-session invalidation.
- HuggingFace fixture parity passes all top-five sets and all 16 exact generated
  IDs. Final generation logits differ by about `1e-6`. BF16-versus-FP32 outputs
  remain diagnostic under the existing policy; no tolerance was relaxed.
- `LagunaFusedTest`, `LagunaRunHF`, and aggregate `Examples` build under `--wfail`;
  the native example's `--help` runs. CUDA-only execution is skipped on this CPU
  host, so neither fused CUDA numerics nor GPU performance is qualified here.
- The four-declaration ABI audit and qualification shell syntax check pass.

The full migration remains active: other model families, legacy reflection and
GPU generators, remaining examples/tests/docs, and actual GPU qualification
still need their final acceptance checks before removing `torch.T`.

## 2026-10-13 Gemma4 continuation

Gemma4's text and image models, checkpoint loaders, image preprocessing boundary,
cache and generation APIs, and CLI now use indexed tensors. Floating parameters
retain checkpoint dtype; token IDs and routing indices are Int64, finished masks
are Bool, and rotary tables are Float32. Image grids and variable feature counts
carry existential dimensions. Five native declarations have explicit audited
prototypes, including the corrected image-loader IO signature on both platforms.

The new cache parity test exposed an existing discrepancy: uncached evaluation
ignored configured KV sharing. Full-sequence evaluation now retains source-layer
K/V and supplies it to shared layers, matching prefill and incremental decoding.
BF16 cache storage and padding preserve model dtype; vision projections convert
fresh Float32 image patches to their weight dtype. Router projections explicitly
use Float32 alongside the existing Float32 normalization scale.

Validation completed:

- The focused indexed-model runner passes 220 tests, with zero failures or skips.
  Eight Gemma4 tests cover shared-cache parity and per-layer inputs, FP32/BF16
  generation and streaming, checkpoint round trips, expert reference numerics,
  BF16 routing, vision clipping and feature repetition, multimodal dtype and
  placeholder checks, and real native image patch loading.
- All nine required CPU suites build under `--wfail` (1,695 build jobs) and
  pass. The main runner passes 1,444 tests and the experimental runner passes
  49, both with zero failures/skips; all seven standalone Laguna CPU suites
  also succeed. CUDA-only cases remain skipped on this CPU host.
- Broad validation exposed millisecond timer truncation in the existing
  ManifoldMuon benchmark. Its separate fix measures nanoseconds and returns
  milliseconds, preserving the positive-duration assertion.
- `Gemma4RunHF` and the focused runner build under `--wfail`; the CLI's native
  `--help` invocation succeeds.
- The five-declaration Gemma4 ABI audit, core and raw-tensor ABI audits, and
  qualification shell syntax check pass.

These synthetic model tests do not qualify a downloaded Gemma4 checkpoint or
GPU execution. The full migration remains active under the criteria below.

## 2026-10-13 VAE continuation

VAE parameters, residual and attention blocks, up/downsampling, checkpoint
loaders, latent unpacking, and decoder outputs now carry indexed shapes. Floating
checkpoint dtype remains dynamic. Convolution output dimensions use the actual
stride/padding formula, including odd downsample sizes. The decoder accepts
indexed spatial dimensions; its existing batch-one entry point delegates to that
implementation. Group normalization has an audited raw boundary; nearest-neighbor
upsampling reuses the indexed interpolation boundary.

A negative checkpoint fixture exposed implicit reshaping in the legacy native
SafeTensors loader. Shared indexed loading now requests the stored layout before
validating the requested shape, rejecting equal-numel layout mismatches for both
single-file and sharded checkpoints. Explicit model reshapes must be visible in
model code rather than hidden inside checkpoint loading.

Validation completed:

- All 226 focused indexed-model tests pass, with zero failures/skips. Six VAE
  tests cover analytical normalization gradients, FP32/BF16 blocks, exact latent
  unpacking, odd downsample dimensions, complete decoder execution at 2x3 latent
  resolution, and single-file/sharded checkpoint layout rejection.
- All nine required CPU suites build under `--wfail` (1,720 jobs) and pass.
  Main: 1,450 tests; experimental: 49; both have zero failures/skips. Standalone
  Laguna checkpoint parity also passes with strict stored-layout loading.
- The group-normalization ABI audit and qualification shell syntax check pass.
  CUDA-only portions of standalone suites remain skipped on this CPU host.

Flux and its VAE demo callers still need migration; this subsystem work does not
complete the full migration or qualify actual GPU execution.

## 2026-10-13 Flux continuation

Flux weights, modulation, double/single attention blocks, rotary embeddings,
sampling, position IDs, latent packing, checkpoint loading, and both Flux/VAE
CLI demos now use indexed tensors. Position IDs are Int64; timesteps and rotary
matrices are Float32. Image and text rotary tables have separate sequence
indices. Axis dimensions are checked for an even four-axis split summing to the
head dimension. Rotary arithmetic runs in Float32 and restores the activation
dtype before attention, allowing BF16 checkpoints to run end to end.

The two-declaration native boundary covers timestep embedding and PPM output.
PPM's obsolete IO-world parameter was removed, and the typed image-output helper
moves pixels to CPU before native file writing. Demo token helpers return indexed
IDs/masks, check text embedding width, and explicitly reshape the known VAE
layout. Both CLIs provide a native `--help` path.

Validation completed:

- The focused indexed-model runner passes 234 tests, with zero failures/skips.
  Eight Flux tests cover exact position IDs, rotary and timestep references,
  FP32/BF16 forward and denoising, latent pack/unpack ordering, PPM bytes,
  generic checkpoint round trips, and complete generated Flux-format checkpoint
  loading in FP32/BF16 with forward execution.
- Both Flux demos and aggregate `Examples` build under `--wfail`, and both
  native CLI `--help` paths run.
- All nine required CPU suites build under `--wfail` (1,757 jobs) and pass:
  1,458 main tests and 49 experimental tests, each with zero failures/skips,
  plus all seven standalone Laguna CPU suites. CUDA-only cases remain skipped.
- The timestep/PPM ABI audit passes.

Downloaded Klein checkpoints, full-resolution image quality, and actual GPU
execution remain unqualified. Other legacy models, reflection/generators, and
remaining examples/tests/docs still need migration before removing `torch.T`.

## 2026-10-13 Flowfusion continuation

`MaskedState` and `Guide` now carry separate `StaticSpec` parameters for their
conditioning and loss masks. Constructors/getters preserve each mask's shape,
dtype, and device policy. Masks remain static during tensor-tree traversal;
BranchingFlows anchor merging preserves the first state's mask specifications
and values while combining the state.

The focused indexed runner builds under `--wfail` and passes 237 tests, including
three new checks for map/mapM/zip/fold behavior, optional guide masks, and weighted
anchor merging with differently shaped Float32 and Bool masks. All 49
experimental BranchingFlows tests also pass with zero failures/skips. This
change does not claim a fresh full CPU or GPU qualification run.

## 2026-10-13 Whisper continuation

Whisper attention, encoder/decoder parameters and activations, KV and cross
caches, checkpoint loading, and transcription now use indexed tensors. Token IDs
and sampled indices are Int64; sampling probabilities are Float32; floating
checkpoint parameters retain their dtype. Dynamic cache storage carries actual
shape witnesses. Fresh frontend features are cast to the encoder weight dtype.

Cached decode rejects exhausted capacity or a cache position that disagrees
with the requested token position. Empty encoder inputs fail with an IO error.
Optional checkpoint weights now distinguish absence from malformed shape: a
missing key bias gets a correctly typed zero tensor and a missing output head
ties to token embeddings, while a malformed stored tensor is rejected.

Validation completed:

- The focused indexed-model runner builds under `--wfail` and passes 243 tests,
  with zero failures/skips. Six Whisper tests cover FP32/BF16 cached/full decode
  parity, preserved cache snapshots, capacity and position errors, head/length
  validation, generic checkpoint round trips, generated sharded FP32/BF16
  checkpoints with optional weights, and waveform-to-text greedy/beam/sampling
  execution through the native audio frontend.
- All nine required CPU suites build under `--wfail` (1,781 jobs) and pass:
  1,467 main tests and 49 experimental tests with zero failures/skips, plus all
  seven standalone Laguna CPU suites. CUDA-only portions remain skipped.
- Aggregate `Examples` and all three Whisper executables build under `--wfail`.
  Transcribe and VoiceMode print native help and exit successfully; their
  existing empty-error help path was corrected in a separate CLI fix.
- The core FFI ABI audit passes.

Downloaded Whisper checkpoint accuracy, live microphone interaction, and actual
GPU execution remain unqualified. Executable `torch.T` uses are gone from model
families; legacy infrastructure, generators, remaining examples/tests, and final
hardware acceptance are still outstanding.

## Qwen generation CLI migration

`Examples/Qwen35/RunHF.lean` and `Examples/Qwen25Omni/RunHF.lean` now
use indexed Int64 token tensors. Qwen3.5 image/video feature values retain
shape witnesses, and Float32 media inputs explicitly enter the model's
dynamic-dtype API. Neither executable depends on the legacy `T` alias.

Both executables build under `--wfail` (956 jobs), print native `--help`, and
exit successfully. This validates compilation and startup; downloaded-model
inference and actual GPU execution remain unqualified. GitHub CLI reports
green upstream main CI at `05304ff`, which does not validate these local edits.
The subsequent TTS CLI migration is described below.

## Qwen3TTS end-to-end CLI migration

The TTS CLI now uses indexed Int64 token/code tensors and shape-indexed floating
embeddings. Variable prompt lengths carry a `Sigma` witness through optional
speaker concatenation. Speaker embedding files must have their exact expected
stored shape. Streaming audio callbacks unpack their sample-count witness before
appending through the typed WAV API.

The WAV begin/append/finalize declarations are audited native boundaries. Their
obsolete IO-world arguments were removed, and append releases its owned shape
argument. A native test verifies RIFF/data sizes, sample rate, empty chunks, and
exact PCM16 samples across Float32 and BF16 input chunks.

`scripts/check_qwen3tts_cli.py` builds tiny sharded FP32/BF16 checkpoint fixtures
without downloads or Python ML packages. It exercises native help and twenty
codec-generation cases covering every language/think-mode branch and ordinary
versus true-streaming generation. This does not qualify real speech quality,
external speaker embeddings, a downloaded speech decoder, or actual GPU use.

The final strict build covers the TTS executable, focused runner, all required
CPU executables, and aggregate examples (1,801 jobs). The rebuilt focused runner
passes 244 tests with no failures or skips; the twenty native CLI cases pass.
The neural and core ABI audits pass 35 and 11 declarations respectively.
All nine required CPU suites pass: 1,468 main tests and 49 experimental tests
with no failures/skips in those runners, plus the seven standalone Laguna CPU
suites. CUDA-only portions remain skipped. Upstream CI is green at `05304ff`,
which does not validate these local commits.

## GPT training-example migration

The GPT training, pretraining, NanoGPT copy, Riemannian NanoGPT, and TrainGPT
paths now keep Int64 token shapes through batching, loss computation, and
updates. `forwardRaw` and `lossRaw` are removed. Generation uses indexed
positions, logits, probabilities, and sampled indices throughout. Gradient
clipping and scalar loss operations use the typed API.

Batch sampling accepts an exactly sized dataset and includes the last valid
starting offset. Empty dimensions and undersized datasets fail before native
sampling. Generation rejects empty active contexts and nonpositive/nonfinite
temperatures. Tests cover these boundaries, shifted token values, accumulated
updates, evaluation, and one-step pretraining on an exactly sized stream.

The audited UInt16 file loader returns indexed Int64 tensors and replaces the
legacy adapter in both GPT's sequential loader and the shared data-shard loader.
Its obsolete native IO-world argument is removed. A generated binary fixture
checks token values and rejection of short reads.

The strict build passes across aggregate examples, TrainGPT,
RunRiemannianNanoGPT, the focused runner, and all required CPU executables
(1,805 jobs). The focused runner passes 248 tests with no failures/skips. Both
ABI audits pass: 36 neural declarations and 11 core declarations. Actual GPU
training remains unqualified.
All nine required CPU suites pass: 1,472 main tests and 49 experimental tests
with no failures/skips in those runners, plus all seven standalone Laguna CPU
suites. CUDA-only portions remain skipped. GitHub reports green upstream main
CI at `05304ff`; those runs do not validate this local migration.

## Discrete diffusion example migration

The diffusion parameters, rotary cache, schedule, forward/loss paths, training,
checkpoint/optimizer persistence, and terminal sampler now use indexed tensors.
Float32 activations, Int64 tokens/indices, and Boolean masks are distinct. The
new native max/top-k adapters validate both value and index metadata. All
non-GPU examples are now free of executable legacy `T` uses.

Checkpoint loading requires exact stored shapes and Float32 dtype instead of
silently reshaping equal-sized tensors. Exact-fit training batches work, while
invalid sampling dimensions, temperatures, and top-k values return IO errors.
The core samplers preserve every supplied context token, including the mask
value, and confidence fallback cannot overwrite already-decoded positions.

Nine native diffusion tests pass, covering forward/training/masked loss, both
samplers, exact optimizer checkpoint resume, wrong-layout rejection, and
mask-valued context preservation. Compile-time regressions reject wrong matrix
shapes and floating token IDs. The TrainDiffusion executable, TestDiffusion,
aggregate examples, and aggregate tests build under `--wfail` (1,513 jobs).
Native TrainDiffusion help exits successfully. The neural/core ABI audits pass
39/11 declarations. Real trained-checkpoint text quality, terminal-animation
interaction, and GPU execution remain unqualified.
All nine required CPU suites build under `--wfail` (1,783 jobs) and pass:
1,474 main tests and 49 experimental tests with no failures/skips in those
runners, plus all seven standalone Laguna CPU suites. CUDA-only portions remain
skipped. Upstream main CI is green at `05304ff`; it does not validate these local
commits.

## GPU ODE/SDE runner migration

The RK combination, fused Dopri5 solve, Brownian descent, and fused
Euler-Maruyama parity runners now use indexed Float32 states and typed CPU
references. Raw handles are extracted only for the generated RK launch ABI.
RK combination scratch outputs are allocated through IO, avoiding accidental
sharing of pure allocation expressions under compiler common-subexpression
elimination.

`Examples/GPU/IndexedParity.lean` compares equal indexed shapes across storage
dtypes with Float32 metrics. CPU tests check mixed Float32/BF16 metrics,
tolerance failures, NaN rejection, compile-time shape rejection, and all four
runners' explicit failure result when CUDA is unavailable. The older parity
helper remains for GPU runners still awaiting migration.

The four modules and main test executable build under `--wfail` (1,712 jobs).
This is not GPU execution evidence. The latest upstream CUDA smoke run
`29984516285` failed while building its GB10 test runner, before parity ran;
it cannot qualify this local migration. Upstream main CI remains green at
`05304ff`, also unrelated to these local edits.
The current `TestGPUE2E` and `TestGPUGB10E2E` targets also build under
`--wfail` (1,072 jobs). The main CPU runner passes 1,476 tests with no
failures/skips, including the new indexed metric and CUDA-unavailable checks.
The test inventory still validates all nine required and four optional suites.
No CUDA numerical run was performed on this host.

## Rotary and LayerNorm GPU runner migration

The rotary runner now keeps Float32 fixture tables and explicit BF16 training
inputs through forward/adjoint references, metrics, and launch boundaries.
Query and key head counts remain separate shape indices. Its fixture loader
requires the exact stored Float32 shapes.

LayerNorm inputs are parameterized by one dtype shared by activation, residual,
weight, and bias. Mixed-dtype construction is rejected at compile time; fixture
loading validates exact shapes and dtype agreement. Float32 and BF16 dispatch
branches use corresponding typed references. Mutable output buffers are
allocated through IO rather than pure factories.

Both runners use indexed parity metrics and expose raw handles only at their
native launch calls. CPU regressions exercise analytic normalization values,
BF16 residual/output metadata, and forward/adjoint quarter-turn rotations with
unequal query/key head counts. CUDA-unavailable checks cover these runners too.
The main runner and both GPU test targets build under `--wfail` (1,749 jobs).
Actual CUDA kernel parity and benchmark timings remain unqualified. The GitHub
CLI still reports green upstream main CI at `05304ff`, not these local changes.
The main CPU runner passes all 1,478 tests with zero failures/skips. The test
inventory continues to validate nine required and four optional suites.

## RMSNorm and loss GPU runner migration

RMSNorm now uses dtype-indexed inputs, exact fixture metadata checks, and
indexed forward/backward references. BF16 residuals, outputs, and input
gradients remain distinct from Float32 saved inverse RMS and weight gradients.
The loss benchmark uses Int64 targets, Float32 cross-entropy accumulation,
and a BF16 logits cotangent. Both runners allocate writable buffers through IO
and expose raw handles only at native kernel launches.

CPU regressions cover RMSNorm gradient cancellation, an orthogonal gradient,
residual-gradient addition, and the analytic uniform-logit cross-entropy loss
and gradient. RMSNorm also participates in the CUDA-unavailable runner check.
The loss reference lives in an importable module separate from the executable
entry point. Actual CUDA parity and performance remain unqualified.
The main runner, both GPU test targets, and the loss executable build under
`--wfail` (1,754 jobs). All 1,480 main CPU tests pass without failures or skips.
The inventory validates nine required and four optional suites. The loss
executable returns failure on this CUDA-unavailable host as expected. GitHub
CLI reports green upstream main CI; this does not validate the local commits.

## Optimizer GPU runner migration

The AdamW benchmark now carries Float32 master weights and moments separately
from BF16 model weights and gradients. Its reference update is importable
without the executable entry point, and parity checks require indexed shapes.
The IO copy adapter requires matching shape/dtype indices for state resets;
its native signature now matches Lean's generated three-argument declaration.
The neural ABI audit covers all 40 declarations, including this copy boundary.

CPU regression coverage includes positive, negative, and zero gradients,
nonzero prior moments, weight decay, BF16 rounding, copy storage independence,
and compile-time rejection of shape/dtype mismatches. Actual optimizer CUDA
parity and benchmark timings remain unqualified.
The main runner and both GPU targets build under `--wfail` (1,753 jobs),
and all 1,481 main CPU tests pass with zero failures/skips. The native build
retains its existing unused-function warning in `tyr_ops.cpp`. The optimizer
executable correctly fails on this CUDA-unavailable host. Test inventory
validation still finds nine required and four optional suites. Upstream main
CI remains green, independently of these local changes.

## BF16 GEMM and FlashAttention fixture runner migration

The B200/GB10 BF16 GEMM fixture and benchmark runner now uses indexed operands,
Float32 reference accumulation, BF16 expected outputs, and exact fixture
shape/dtype loading. Both FlashAttention fixture runners use a shared indexed
head-dimension-64 reference with separate query and key/value sequence lengths.
BF16 outputs and Float32 log-sum-exp values remain distinct. Mutable kernel
outputs are allocated through IO; raw handles appear only at launch boundaries.
FlashAttention3's executable also supplies the suite name required by the
current fixture-runner API.

CPU regressions cover nonsquare GEMM orientation, Float32 accumulation across
cancellation, and analytic attention output/log-sum-exp with unequal sequence
lengths and nonuniform attention weights. Compile-time coverage rejects
incompatible GEMM reduction dimensions. The FlashAttention runner participates
in the CUDA-unavailable check. All three executables terminate with a
CUDA-unavailable error before generating fixtures on this host; no actual
CUDA parity or performance is qualified by these checks.
The main runner, both GPU test targets, and all three migrated executables
build under `--wfail` (1,766 jobs). All 1,483 main CPU tests pass without
failures or skips, and the test inventory validates nine required and four
optional suites. GitHub CLI reports green upstream main CI and a failed older
CUDA smoke run; neither provides qualification for the local migration.

## H100 attention forward/backward fixture migration

The reduced 128-token H100 runner now uses BF16 forward activations and
Float32 log-normalizer and gradient buffers. Fixture loading checks exact
shape and dtype, diagnostic gradient dumps remain indexed, and all writable
kernel outputs receive independent IO allocations. Raw handles are confined
to kernel launches.

The shared training reference computes a scalar cotangent contraction through
BF16 SDPA and returns Float32 gradients. Detached reference leaves prevent
accumulation across calls. An analytic two-token regression checks output,
query/key/value derivatives, and repeated-call independence. The runner also
participates in the CUDA-unavailable check. Actual H100 kernel parity and
gradient accumulation on hardware remain unqualified.
The main runner, both GPU test targets, and H100 executable build under
`--wfail` (1,760 jobs). All 1,484 main CPU tests pass with zero failures/skips.
The inventory validates nine required and four optional suites.

## H100 768-token and training runner migration

The 768-token fixture runner now shares indexed attention references, requires
exact BF16 activation and Float32 gradient/normalizer fixtures, and allocates
independent writable buffers through IO. The training demo distinguishes BF16
kernel activations from Float32 optimizer state and scalar losses. Its loss,
cotangent, SGD, and PyTorch baseline step are importable without an executable
entry point. Initial parameter tensors can be shared because each SGD update
produces new storage; a CPU regression verifies initial-value preservation and
repeated-call independence, alongside analytic loss and update values.

Both executables and the GPU test targets build locally. Each executable
returns failure without CUDA. Hardware training convergence, multi-block
backward parity, and timings remain unqualified.
The combined build passes under `--wfail` (1,765 jobs). All 1,485 main CPU
tests pass with zero failures/skips. Inventory validation still covers nine
required and four optional suites. GitHub CLI reports green upstream main CI,
which is separate from these local commits.

## Decode runner and benchmark migration

The decode fixture runner and benchmark now use indexed BF16 Q/K/V, Float32
reference attention, and exact BF16 fixture loading. Production dispatch uses
the typed FlashAttention adapter. Benchmark sinks are independent IO
allocations and use the audited typed copy operation.

The token-by-token cache parity path is shared with a CPU regression. It uses
32 query heads, eight KV heads, head dimension 128, and ten appended tokens,
comparing every cached output with the matching full causal attention slice.
This exercises the real indexed KV-cache API, checked slices, sequence-length
updates, and GQA dispatch. CPU parity does not qualify CUDA decode kernels,
tail masking on hardware, or benchmark timings.
The main runner, both GPU targets, and both decode executables build under
`--wfail` (1,767 jobs). All 1,486 main CPU tests pass with zero failures/skips;
the cache check reports maximum absolute error zero over all ten steps. Both
executables correctly return failure without CUDA. The inventory validates
nine required and four optional suites. Upstream main CI remains green and
is separate from this local migration.

## GB10 attention runner migration

The GB10 fixture and benchmark paths now carry indexed BF16 activations,
Float32 saved normalization state, and the kernel-specific gradient dtype:
Float32 for the reduced direct backward kernels and BF16 for the specialized
768-token backward kernels. Fixture loading requires exact shape/dtype.
Writable outputs use independent IO allocations, replacing syntactically
different pure expressions previously used to avoid allocation aliasing.

Shared attention references now support indexed batch and head counts. A
CPU regression checks two batches and three heads against analytic forward,
log-sum-exp, and Q/K/V derivatives. The GB10 runner also participates in the
CUDA-unavailable regression. Actual GB10 numerical parity and performance
remain unqualified without hardware execution.
The main runner, both GPU test targets, and GB10 executable build under
`--wfail` (1,764 jobs). All 1,487 main CPU tests pass with zero failures/skips.
The benchmark correctly fails on this CUDA-unavailable host. Test inventory
validation still covers nine required and four optional suites. GitHub CLI
reports green upstream main CI, independently of the local changes.

## Remaining GPU example migration and parity consolidation

The FlashAttention benchmark now distinguishes BF16 outputs from Float32
backward references and uses indexed dispatch APIs throughout. Its numerical
helpers have CPU coverage for relative L2 metrics and repeated cotangent
backward calls. That regression exposed reuse of a freed autograd graph when
leaf creation was pure. Benchmark leaves now allocate independent storage
through the audited IO clone operation before enabling gradients. Timing
includes this fresh leaf allocation and must be remeasured on hardware.

A scan of legacy operation calls found inferred tensor dependencies missed by
searches for explicit `T` annotations. The copy runner, FlashAttention smoke
runner, and Brownian sampling runner are now indexed too. Copy/output buffers
are allocated through IO, and Brownian readback uses the indexed host adapter.
The shared parity helper now requires indexed shapes and computes metrics in
Float32; `IndexedParity` reexports it instead of maintaining a second version.

No direct legacy tensor constructors, operations, or `T` annotations remain
in `Examples/GPU` in the source scan. This is a source-migration milestone,
not CUDA qualification or removal of the repository's legacy API. Actual GPU
parity, routes, convergence, and benchmark timings still require hardware.
The main runner, both GPU targets, FlashAttention benchmark/smoke executables,
and copy executable build under `--wfail` (1,778 jobs). All 1,488 main CPU
tests pass with zero failures/skips after the fresh-leaf fix. The three
executables correctly fail without CUDA. The test inventory validates nine
required and four optional suites; upstream main CI is green independently.

## Remove legacy structural tensor support

Module derivation, AD schema/flattening, and widget display no longer provide
instances for the shape-erasing `T` carrier. The AD frontend no longer treats
`torch.T` as a tensor parameter; indexed `Tensor` and `Frozen` remain supported.
The remaining structured AD fixture now uses explicit Float32 tensor fields,
indexed constructors/comparisons, and matching dtype metadata in its frontend
graph. Missing dtype metadata is rejected against those indexed boundaries.

All nine required CPU suites build with `--wfail` and pass through
`lake env python3 scripts/test_inventory.py --run`; the main suite passes all
1,488 tests. CUDA-only checks remain skipped on this machine. The legacy
`T` declaration and operations in `Tyr.Torch` still exist and require separate
removal after their remaining consumers are migrated.

## Migrate remaining constructor and FFI boundary tests

AD frontend placeholders and modular norm tests now construct indexed tensors.
Typed constructor rejection tests use native handles directly at the tested
raw-to-indexed boundary. All 1,488 main tests passed after these changes.

FFI boundary tests and the manual crash probe now use indexed tensors. Checked
adapters cover start/stop/step ranges, boxed integer indexing, unbind, and
logspace. Scalar reshape is tested as an actual scalar; exact loading still
rejects equal-element-count tensors with the wrong shape. Legacy shape erasure
and permissive loading are no longer asserted as supported public behavior.

`lake build test_runner ffi_crash_probe --wfail` passes, all four migrated FFI
boundary tests pass, and the neural ABI audit passes all 43 declarations.
The intentional crash probe was compiled but not executed. Actual GPU
qualification remains outstanding.

## Remove legacy linear algebra, rotary, and tensor autograd APIs

`Tyr.Torch` no longer declares legacy linear algebra, rotary, or tensor autograd
operations. Global gradient-mode controls remain available. Indexed
`Tensor.backward` accepts an explicit cotangent with the output specification,
and `Tensor.hasGradFn` inspects graph ownership without exposing a raw handle.
MHA portable backward and inference-cache graph checks use these adapters.
The native backward function now matches Lean's three-argument generated ABI;
the old unused world argument has been removed. All 45 neural ABI declarations
match generated C output.

NanoChat chat sampling now retains indexed Int64 tokens and Float32 logits
through final-position slicing, filtering, softmax, multinomial sampling, and
scalar extraction. ChatSFT uses the existing indexed unreduced cross entropy;
checkpoint and module smoke tests use indexed scalar extraction.

The strict build passes for `test_runner`, `NanoChatChat`, `TestGPUE2E`, and
`TestGPUGB10E2E`. The CPU explicit-cotangent test passes: differentiating
`[2, 3]²` with cotangent `[4, 5]` produces `[16, 30]`, and detach removes the
gradient function. GPU targets have only been compiled on this machine.

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
