# Examples and testing

## Purpose & when to use

`Examples/` is the collection of runnable end-to-end demonstrations: training
scripts (char-GPT, NanoChat, discrete diffusion, branching flows), pretrained-model
demos (Flux, Whisper, Qwen, Gemma, KittenTTS), and GPU kernel parity/benchmark
harnesses. `Tests/` is the LeanTest-based test suite — unit tests, parity tests
against reference implementations, and GPU end-to-end suites. Read this chapter
when you want to run something rather than build it, or when you are adding a
test or example of your own.

## Architecture & main abstractions

### Build wiring

Everything is declared in `lakefile.lean`:

- `lean_lib Tests` (roots `Tests`, no precompile),
  `lean_lib TestsExperimental` (`:586`), `lean_lib Examples` (`:591`).
- Test-side `lean_exe` targets root under `Tests/`, example
  executables root under `Examples/`.
- `@[test_driver] lean_exe test_runner` (`:598`) is Lake's test driver, so
  `lake test` builds and runs it.
- The test framework is an external dependency
  (`lakefile.lean`): `require LeanTest from git "https://github.com/cpehle/lean_test.git"`.

### The LeanTest framework

Tests are plain `IO Unit` definitions tagged with attributes
(`.lake/packages/LeanTest/LeanTest/Attr.lean`):

```lean
@[test] def testSomething : IO Unit := do ...        -- discovered and run
@[test_ignore] def testSlow : IO Unit := do ...      -- skipped unless --ignored
@[test_should_error] def testFails : IO Unit := do ... -- passes iff it throws
```

Assertions live in `LeanTest/Assert.lean` and throw on failure:

| Function | Purpose |
|---|---|
| `LeanTest.assertTrue / assertFalse` | boolean conditions |
| `LeanTest.assertEqual / assertNotEqual` | `[BEq α] [ToString α]` equality |
| `LeanTest.assertSome / assertNone` | `Option` shape |
| `LeanTest.assertThrows / assertNoThrow` | error behavior of an `IO` action |
| `LeanTest.assertSatisfies` | predicate with a message |
| `LeanTest.fail` | unconditional failure |

Discovery is environment-based. A runner imports modules with
`Lean.importModules` and hands the resulting `Environment` to the runner,
which scans it for tagged declarations
(`.lake/packages/LeanTest/LeanTest/Runner.lean:145`):

```lean
unsafe def LeanTest.runTestsAndExit
    (env : Environment) (opts : Options) (config : RunConfig := {}) : IO UInt32

structure LeanTest.RunConfig where
  includeIgnored : Bool := false
  failFast       : Bool := false
  filter         : Option String := none
```

**A test file that no runner imports never executes.**
`python3 scripts/test_inventory.py` checks that every LeanTest module is
reachable from a required suite or an explicitly optional suite. It also checks
standalone test executables for a `main` and an assigned suite.

### Test tiers

Tests are organized in three tiers:

1. **Main suite** — `lake test`. `test_runner`'s root is
   `Tests/RunTests.lean`, which imports the umbrella module `Tests.lean`
   plus `Tests.TestAdditionalManifolds` (special-cased there, not
   in `Tests.lean`). It parses `--filter PATTERN`, `--ignored`, `--fail-fast`,
   `--help` into a `LeanTest.RunConfig`.
2. **Experimental suite** — `lake exe test_runner_experimental`, root
   `Tests/RunTestsExperimental.lean`, which imports `TestsExperimental.lean`.
   That umbrella currently contains exactly one module,
   `Tests.TestBranchingFlows`, parked there until
   `Tyr/Model/BranchingFlows.lean` stabilizes (note at `Tests.lean:118`).
3. **Standalone executables** — focused or heavy suites built and run
   individually: `TestDiffEq`, `TestDiffEqAdjoint`, `TestDiffEqAdjointCore`,
   `TestDataLoader`, `TestGPUDSL`, `TestGPUKernels`, `TestGPUE2E`,
   `TestGPUGB10E2E`, `TestGPUTileIR`, `TestTileIRGenerateMain`, `TestDiffusion`,
   and `RunRiemannianNanoGPTTests`. Each is a thin runner module
   (`Tests/RunTest*.lean`, `Tests/Run*.lean` — the prefix split is historical)
   with a `parseArgs` → `runTestsAndExit` body in most runners. `TestDiffusion`
   has a dedicated `Tests/RunTestDiffusion.lean` driver. The SDE/underdamped order-parity runners
   (`Tests/RunDiffEqSDEOrderParity.lean`, `RunDiffEqUnderdampedOrderParity.lean`)
   which just call a `run` function.

Hosted CI (`.github/workflows/cpu.yml`) builds and runs the manifest in
`scripts/test_suites.json`: the main and experimental runners plus Laguna's
config, tokenizer, NVFP4, MoE, model, reference parity, and rotary suites.
These Laguna suites run CPU checks against tracked fixtures; their CUDA branches
remain conditional. Diffusion and TileIR export-driver tests are in the main
suite. Add new standalone CPU assertion suites to the same manifest.

CI also builds `BranchingFlowsMoleculeTrainGenerate` and runs
`scripts/test_branching_checkpoint_resume.py` on its embedded CPU dataset. That
regression checks save/resume counts, retained snapshots, and rejection of
missing or mismatched optimizer state. NanoChat checkpoint tests in the main
suite compare the next optimizer update after restoring a snapshot.

GPU suites run on self-hosted hardware via `.github/workflows/gpu.yml`.
That workflow sets `TYR_GPU_TEST_STRICT=1`: the linked LibTorch must have CUDA,
the configured family must match, and the selected suite must execute tests
with no skips. Blackwell uses `TestGPUGB10E2E`; Hopper runs `TestGPUE2E` filtered
to `TorchParity`, then `RunMhaH100Decode` including cache parity. GPU test and
example changes are included in the workflow path filters. Local GPU runs
without strict mode retain optional skips and print executed/skipped counts.

There is also `lean_exe ffi_crash_probe` (root
`Tests/FfiCrashProbe.lean`): a manual probe that intentionally triggers a
libtorch CUDA error to check the process dies with an intelligible `c10::Error`
message instead of a bare SIGABRT. It is deliberately not part of any suite;
the Linux CI job runs it and fails unless it aborts (SIGABRT) with the
libtorch error message.

### Fixtures

- `Tests/fixtures/safetensors/` — `single.safetensors`, `sharded/`,
  `indexed.safetensors`, `indexed_dir/`, `provider_errors/` for the SafeTensors
  loader and type-provider tests (see [serialization.md](serialization.md)).
- `Tests/MctxData/` — independent upstream MuZero/Gumbel references checked
  against complete search statistics. The original large dumps are preserved
  as historical data; see the [fixture guide](../Tests/MctxData/README.md).
- GPU kernel examples compare against `.pt` fixtures produced by libtorch
  reference implementations; see `Examples/GPU/FixtureRunner.lean` below.

### Example organization

Examples pair library modules with thin executable wrappers. The convention is
a 4-line entrypoint module, e.g. `Examples/GPU/RunRMSNormExe.lean`:

```lean
import Examples.GPU.RunRMSNorm

/-- Lake executable entrypoint for the fused residual + RMSNorm demo. -/
def main (args : List String) : IO UInt32 :=
  Examples.GPU.RunRMSNorm.main args
```

The root module `Examples.lean` imports only library modules (GPT, NanoChat,
AlphaGradPort) — executables are excluded by design because they define global
`main`.

Shared infrastructure:

- `Examples/ModelRunner.lean` — CLI/device/streaming helpers for the
  pretrained-model demos: `resolveDevice : String → IO (Device × Option String)`
  (parses `auto`/`cpu`/`mps`/`cuda[:n]`, warns and falls back to CPU),
  `parseNatArg`, `loadPrompts`.
- `Examples/GPU/Parity.lean` — parity-check helpers: `gpuTarget`/`gpuFamily`
  (read `TYR_GPU_TARGET`/`TYR_GPU_FAMILY`), `isBlackwellFamily`, `requireCuda`,
  `seedFixtures`, and `TensorCheck` (label/ok/mae/maxErr/rtol/atol) with
  `compareTensors`.
- `Examples/GPU/FixtureRunner.lean` — `FixtureSpec` (fixture dir + tensor
  names), `CommonArgs`, `parseCommonArgs` (`--regen`, `--gen-only`, `--trials`,
  `--seed`, `--regen-every-trial`), and `runWithFixtures`, which regenerates
  missing `.pt` fixtures from libtorch references and then compares kernel
  outputs against them.
- `Examples/GPT/GPT.lean` — the canonical dependently-typed model demo:
  `torch.gpt.Config` with presets `Config.gpt2_small`, `gpt2_mini`,
  `gpt2_micro`, `tiny_shakespeare`, `nanogpt_cpu_shakespeare`,
  `nanogpt_gpu_shakespeare`, and shape-indexed parameters such as
  `BlockParams (n_embd : UInt64)` whose fields are `T #[n_embd]`
  (`Examples/GPT/GPT.lean:16`).

Run any executable with `lake exe <Exe> [args...]` (builds it if needed) or,
without rebuilding, `lake env .lake/build/bin/<Exe> [args...]`. GPU-backed
targets build with the kernel modules chosen at configure time, e.g.
`lake -R -Kcuda=/usr/local/cuda -Kgpu=H100 -Kkernels=Tyr.GPU.Kernels.MhaH100 build RunMhaH100 RunMhaH100Seq768`.

## Key APIs: the example tour

Per-directory summary of what each example demonstrates, with its executable
targets. `Examples/README.md` has per-example data/weights setup and CLI flags
and is the authoritative source for running them.

| Example | Demonstrates | Executables |
|---|---|---|
| `Examples/GPT/` | Char-level GPT-2 on Shakespeare (nanoGPT configs), shape-indexed model code, Riemannian variant | `TrainGPT`, `RunRiemannianNanoGPT` |
| `Examples/Diffusion/` | Masked discrete diffusion on ASCII text, checkpoint save/load, terminal animation | `TrainDiffusion` |
| `Examples/NanoChat/` | Modded-nanogpt port: NorMuon + DistAdam training, SFT (`ChatSFT`), GRPO, tool-augmented generation, eval tasks | `TrainNanoChat` (torchrun-capable), `NanoChatPipeline`, `NanoChatChat` |
| `Examples/BranchingFlows/` | Branching-flow generative models: toy demos, QM9 molecule training/generation (`qm9_xyz/` data) | `BranchingFlows{Continuous,Molecule,MoleculeTransformer}Train`, `BranchingFlowsMoleculeGenerate`, `BranchingFlowsMoleculeTrainGenerate` |
| `Examples/AlphaGradPort/` | PDE flux tasks with MCTS-driven search (see [mctx.md](mctx.md)) | `AlphaGradRoeFlux1dA0`, `AlphaGradPortSweep`, `AlphaGradPolicyTrain`, `AlphaGradPolicySweep` |
| `Examples/GPU/` | ThunderKittens-style kernel parity vs libtorch fixtures, H100 MHA, decode benchmarks, B200 GEMM | `RunCopy`, `RunRotary`, `RunLayerNorm`, `RunRMSNorm`, `RunFlashAttn{,3,Op,Bench}`, `RunMhaH100{,Decode,Train,Seq768}`, `RunDecodeBench`, `RunB200Bf16Gemm` |
| `Examples/EventSkeleton/` | URDF-backed hybrid contact simulation | `RunUrdfContactExample` |
| `Examples/Flux/` | Flux Klein 4B image generation (text encode → diffusion → VAE decode) | `FluxDemo`, `FluxDebug` |
| `Examples/Qwen35/`, `Qwen25Omni/`, `Gemma4/` | Pretrained HF-hub inference with a shared CLI (`--source`, `--prompt[-file]`, `--image`, `--video`, `--batch-size`, `--max-new-tokens`, `--stream`, `--device`, ...) | `Qwen35RunHF`, `Qwen25OmniRunHF`, `Gemma4RunHF` |
| `Examples/Qwen3ASR/` | Speech transcription, live microphone streaming | `Qwen3ASRTranscribe`, `Qwen3ASRLiveMic`, `Qwen3ASRLiveMicTrueStream` |
| `Examples/Qwen3TTS/` | Text-to-speech end to end | `Qwen3TTSEndToEnd` |
| `Examples/Whisper/` | Whisper transcription (file and in-memory), voice-mode loop | `WhisperTranscribe`, `WhisperTranscribeInMem`, `WhisperVoiceMode` |
| `Examples/` (root) | KittenTTS pretrained synthesis and duration/debug harnesses; `TestGuard.lean` is a Widget display demo despite the name | `KittenTTSPretrained`, `KittenTTSDurations`, `KittenTTSDebug`, `KittenTTSCompare` |

Test-side executables: `test_runner`, `test_runner_experimental`,
`RunRiemannianNanoGPTTests`, `ffi_crash_probe`, and the standalone suites
listed under "Test tiers" above. Two codegen tools (`GenerateGpuKernels`,
`GenerateTileIRKernels`) are covered in [gpu/dsl-codegen.md](gpu/dsl-codegen.md).

## Usage example

Reconstructed example (from `Tests/TestSafeTensorsTypeProvider.lean`):

```lean
import Tyr.SafeTensors
import LeanTest

open torch

-- Elaboration-time type provider over a checked-in fixture:
safetensors_type_provider "Tests/fixtures/safetensors/single.safetensors" as SingleSafe

@[test]
def testSafeTensorsIntrospectionSingle : IO Unit := do
  let schema ← safetensors.introspect "Tests/fixtures/safetensors/single.safetensors"
  LeanTest.assertEqual schema.sourceIsDirectory false "single source should not be a directory"
  LeanTest.assertEqual schema.tensors.size 1 "single fixture should have one tensor"
```

To make a new test file actually run, add `import Tests.MyNewTest` to
`Tests.lean` — the runner discovers tests by scanning the imported environment.
Run `python3 scripts/test_inventory.py` before committing. GPU-only or
intentional-crash executables belong in the manifest's `optional` map with a
reason; focused runners that import already-routed LeanTest modules are checked
automatically.

Running tests and examples:

```bash
lake test                                  # main suite (test_runner)
lake test -- --filter diffeq --fail-fast   # subset, stop on first failure
lake exe test_runner_experimental          # experimental suite
lake exe TestGPUE2E                        # standalone GPU suite (needs CUDA + fixtures)
lake exe ffi_crash_probe                   # manual FFI failure-mode probe
python3 scripts/test_inventory.py --build # build every required CPU suite
lake env python3 scripts/test_inventory.py --run # run that same suite list
TYR_GPU_TEST_STRICT=1 lake exe TestGPUGB10E2E --fail-fast

lake exe TrainGPT                          # char-GPT on Shakespeare
lake exe Qwen35RunHF --source Qwen/Qwen3.5-0.8B --prompt "Summarize dependent types." --stream
lake exe BranchingFlowsMoleculeTrainGenerate
lake exe RunMhaH100 && lake exe RunMhaH100Seq768   # H100 MHA fixture checks
```

## What is covered vs uncovered

The main suite (`Tests.lean`) is deepest in: autodiff
(`TestAD*`, `TestAutoGrad`), differential equations (`TestDiffEq*` —
`TestDiffEq.lean` alone is 3880 lines), event-skeleton physics (~40
`TestEventSkeleton*` files), Mctx (`TestMctx*`), GPU DSL/kernels/TileIR
(`TestGPUDSL`, `TestGPUKernels`, `TestGPUTileIR` — these also run standalone),
model configs and pretrained-weight loading (Qwen3/3.5/3.6, Qwen2.5-Omni,
Gemma-era FLM, KittenTTS, Qwen3TTS, Qwen3ASR, SileroVAD), the SafeTensors type
provider, checkpoints, manifolds/optimizers, and NanoChat tokenizer/tasks.

No test file imports these modules directly (verified by grep): `Tyr.Audio`,
`Tyr.Hub`, `Tyr.PRNG`, `Tyr.RL`, `Tyr.Widget`, `Tyr.Log`, `Tyr.Text`,
`Tyr.Inference`. Coverage is thin (one or two files) for `Tyr.Distributed`,
`Tyr.Sharding`, and `Tyr.Data`. Multi-process collective behavior is only
smoke-tested through the FFI surface in `Tests/TestModdedGPT.lean`.

Caveats to be aware of when extending the suite:

- GPU parity harnesses under `Examples/GPU/Run{BrownianSample,BrownianDescent,EulerMaruyamaFused,RKCombine,RKFusedSolve,MhaGB10}.lean`
  are exercised through `TestGPUGB10E2E`; hosted CPU success does not validate
  their numerical CUDA behavior.
- `Tests/Test.lean` defines helpers (`encode`, `decode`, `charToInt`, ...) at
  global scope and imports `Examples.GPT.*`, so the test library depends on the
  examples library and those generic names leak into downstream test modules.

## Related guides

- [getting-started.md](getting-started.md) — environment setup required before any example
- [core/tensors.md](core/tensors.md) — the `T #[...]` shape-tracked tensors used throughout
- [serialization.md](serialization.md) — SafeTensors provider and fixtures used by the tests
- [data.md](data.md) — datasets and tokenizers behind TrainGPT/NanoChat
- [autodiff.md](autodiff.md), [modules.md](modules.md), [optimization.md](optimization.md) — subsystems under test
- [distributed.md](distributed.md) — torchrun-based NanoChat training
- [models/llms.md](models/llms.md), [models/generative.md](models/generative.md), [models/audio-speech.md](models/audio-speech.md) — models exercised by the RunHF/ASR/TTS demos
- [gpu/dsl-codegen.md](gpu/dsl-codegen.md), [gpu/kernels.md](gpu/kernels.md) — the GPU suites and kernel examples
- [diffeq.md](diffeq.md), [event-skeleton.md](event-skeleton.md), [mctx.md](mctx.md) — heavily tested subsystems
- [ffi-and-build.md](ffi-and-build.md) — the C++ FFI layer probed by `ffi_crash_probe`

This chapter is a guide, not a symbol dump: exhaustive API documentation is
generated separately by doc-gen4 (see `docbuild/`).
