# Examples and testing

## Purpose & when to use

`Examples/` is the collection of runnable end-to-end demonstrations: training
scripts (char-GPT, NanoChat, discrete diffusion) and
pretrained-model demos (Flux, Whisper, Qwen, Gemma, KittenTTS). `Tests/` is the
LeanTest-based test suite. Read this chapter when you want to run something rather than
build it, or when you are adding a test or example of your own.

## Architecture & main abstractions

### Build wiring

Everything is declared in `lakefile.lean`:

- `lean_lib Tests` (roots `Tests`, no precompile),
  `lean_lib Examples`.
- Test-side `lean_exe` targets root under `Tests/`, example
  executables root under `Examples/`.
- `@[test_driver] lean_exe test_runner` is Lake's test driver, so
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
(`.lake/packages/LeanTest/LeanTest/Runner.lean`):

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

Tests are organized in two tiers:

1. **Main suite** — `lake test`. `test_runner`'s root is
   `Tests/RunTests.lean`, which imports the umbrella module `Tests.lean`
   plus `Tests.TestAdditionalManifolds` (special-cased there, not
   in `Tests.lean`). It parses `--filter PATTERN`, `--ignored`, `--fail-fast`,
   `--help` into a `LeanTest.RunConfig`.
2. **Standalone executables** — focused or heavy suites built and run
   individually: `TestDataLoader`, `TestDiffusion`, and `RunRiemannianNanoGPTTests`. Each is
   a thin runner module
   (`Tests/RunTest*.lean`, `Tests/Run*.lean` — the prefix split is historical)
   with a `parseArgs` → `runTestsAndExit` body in most runners. `TestDiffusion`
   has a dedicated `Tests/RunTestDiffusion.lean` driver.

Hosted CI (`.github/workflows/cpu.yml`) builds and runs the manifest in
`scripts/test_suites.json`: the main runner plus Laguna's
NVFP4, MoE, model, and rotary suites.
These Laguna suites run CPU checks against tracked fixtures; their CUDA branches
remain conditional. Diffusion tests are in the main suite. Add new standalone
CPU assertion suites to the same manifest.

NanoChat checkpoint tests in the main suite compare the next optimizer update
after restoring a snapshot.

Model outputs are compared with the upstream Python implementations on the
published weights by the parity tests in `parity/` (see "Parity Tests" in the
root `Readme.md`),
run by hand on CPU and on a GPU.

There is also `lean_exe ffi_crash_probe` (root
`Tests/FfiCrashProbe.lean`), a probe of Tyr's fatal failure paths selected by
its argument: `libtorch` triggers a libtorch error in an unguarded native call,
`admission` passes `Tensor.ofRaw!` a handle that contradicts its spec, and
`admission-ok` admits a matching one. It is deliberately not part of any suite;
`.github/scripts/check_ffi_crash_probe.sh`, run by the Linux CI job, requires the
first two to abort (SIGABRT) with Tyr's report and the third to exit normally.

### Fixtures

- `Tests/fixtures/safetensors/` — `single.safetensors`, `sharded/`,
  `indexed.safetensors`, `indexed_dir/`, `provider_errors/` for the SafeTensors
  loader and type-provider tests (see [serialization.md](serialization.md)).

### Example organization

The root module `Examples.lean` imports only library modules (GPT, NanoChat) — executables are excluded by design because they define global
`main`.

Shared infrastructure:

- `Examples/ModelRunner.lean` — CLI/streaming helpers for the
  pretrained-model demos: `deviceToString`, `parseNatArg`, `loadPrompts`.
  Every example takes its device from `--device`, acquired with
  `Device.acquireNamed`.
- `Examples/GPT/GPT.lean` — the canonical dependently-typed model demo:
  `torch.gpt.Config` with presets `tiny_shakespeare` and
  `nanogpt_cpu_shakespeare`, and shape-indexed parameters such as
  `BlockParams (n_embd : UInt64)` whose fields use `torch.Tensor` through
  `Activation shape` (Float32). `Tokens shape` carries Int64 inputs. The typed
  `forward` and `loss` APIs preserve these indices throughout the model;
  `forwardRaw` and `lossRaw` validate legacy loader shapes/dtypes on entry.
  `Examples/GPT/TensorOps.lean` contains FFI adapters and checked optimizer
  traversal. Dynamic head reshapes validate element counts; the embedding width
  must be divisible by a nonzero head count. `Tests/TestGPTTyped.lean` covers
  compile-time rejection of malformed projection weights, CPU training and
  generation, and rejection of invalid raw inputs.

Run any executable with `lake exe <Exe> [args...]` (builds it if needed) or,
without rebuilding, `lake env .lake/build/bin/<Exe> [args...]`.

## Key APIs: the example tour

Per-directory summary of what each example demonstrates, with its executable
targets. `Examples/README.md` has per-example data/weights setup and CLI flags
and is the authoritative source for running them.

| Example | Demonstrates | Executables |
|---|---|---|
| `Examples/GPT/` | Char-level GPT-2 on Shakespeare (nanoGPT configs), shape-indexed model code, Riemannian variant | `TrainGPT`, `RunRiemannianNanoGPT` |
| `Examples/Diffusion/` | Masked discrete diffusion on ASCII text, checkpoint save/load, terminal animation | `TrainDiffusion` |
| `Examples/NanoChat/` | Modded-nanogpt port: NorMuon + DistAdam training, SFT (`ChatSFT`), GRPO, tool-augmented generation, eval tasks | `TrainNanoChat` (torchrun-capable), `NanoChatPipeline`, `NanoChatChat` |
| `Examples/Flux/` | Flux Klein 4B image generation (text encode → diffusion → VAE decode) | `FluxDemo` |
| `Examples/Qwen35/`, `Qwen25Omni/`, `Gemma4/` | Pretrained HF-hub inference with a shared CLI (`--source`, `--prompt[-file]`, `--image`, `--video`, `--batch-size`, `--max-new-tokens`, `--stream`, `--device`, ...) | `Qwen35RunHF`, `Qwen25OmniRunHF`, `Gemma4RunHF` |
| `Examples/Qwen3ASR/` | Speech transcription, live microphone streaming | `Qwen3ASRTranscribe`, `Qwen3ASRLiveMic`, `Qwen3ASRLiveMicTrueStream` |
| `Examples/Qwen3TTS/` | Text-to-speech end to end | `Qwen3TTSEndToEnd` |
| `Examples/Whisper/` | Whisper transcription (file and in-memory), voice-mode loop | `WhisperTranscribe`, `WhisperTranscribeInMem`, `WhisperVoiceMode` |
| `Examples/` (root) | KittenTTS pretrained synthesis; `TestGuard.lean` is a Widget display demo despite the name | `KittenTTSPretrained` |

Test-side executables: `test_runner`,
`RunRiemannianNanoGPTTests`, `ffi_crash_probe`, and the standalone suites
listed under "Test tiers" above. `GenerateFfiHeader`, which writes `libTyrC`'s
FFI header, is covered in [ffi-and-build.md](ffi-and-build.md).

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
lake test -- --filter qwen --fail-fast     # subset, stop on first failure
lake exe ffi_crash_probe                   # manual FFI failure-mode probe
python3 scripts/test_inventory.py --build # build every required CPU suite
lake env python3 scripts/test_inventory.py --run # run that same suite list

lake exe TrainGPT --device cpu             # char-GPT on Shakespeare
lake exe Qwen35RunHF --device cpu --source Qwen/Qwen3.5-0.8B --prompt "Summarize dependent types." --stream
```

## What is covered vs uncovered

The main suite (`Tests.lean`) is deepest in:
model configs and pretrained-weight loading (Qwen3/3.5/3.6, Qwen2.5-Omni,
KittenTTS, Qwen3TTS, Qwen3ASR, SileroVAD), the SafeTensors type
provider, checkpoints, manifolds/optimizers, and NanoChat tokenizer/tasks.

No test file imports these modules directly (verified by grep): `Tyr.Audio`,
`Tyr.Hub`, `Tyr.RL`, `Tyr.Widget`, `Tyr.Log`, `Tyr.Text`,
`Tyr.Inference`. Coverage is thin (one or two files) for `Tyr.Distributed`,
`Tyr.Sharding`, and `Tyr.Data`. Multi-process collective behavior is only
smoke-tested through the FFI surface in `Tests/TestModdedGPT.lean`.

Caveats to be aware of when extending the suite:

- `Tests/Test.lean` defines helpers (`encode`, `decode`, `charToInt`, ...) at
  global scope and imports `Examples.GPT.*`, so the test library depends on the
  examples library and those generic names leak into downstream test modules.

## Related guides

- [getting-started.md](getting-started.md) — environment setup required before any example
- [core/tensors.md](core/tensors.md) — the indexed `Tensor` specifications used throughout
- [serialization.md](serialization.md) — SafeTensors provider and fixtures used by the tests
- [data.md](data.md) — datasets and tokenizers behind TrainGPT/NanoChat
- [autodiff.md](autodiff.md), [modules.md](modules.md), [optimization.md](optimization.md) — subsystems under test
- [distributed.md](distributed.md) — torchrun-based NanoChat training
- [models/llms.md](models/llms.md), [models/generative.md](models/generative.md), [models/audio-speech.md](models/audio-speech.md) — models exercised by the RunHF/ASR/TTS demos
- [ffi-and-build.md](ffi-and-build.md) — the C++ FFI layer probed by `ffi_crash_probe`

This chapter is a guide, not a symbol dump: exhaustive API documentation is
generated separately by doc-gen4 (see `docbuild/`).
