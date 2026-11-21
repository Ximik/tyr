import Lake
open Lake DSL
open System (FilePath)

/-- Return `none` for blank strings after trimming whitespace. -/
def nonEmptyTrimmed? (s : String) : Option String :=
  let trimmed := s.trimAscii.toString
  if trimmed.isEmpty then none else some trimmed

/-- CUDA toolkit chosen when configuring: `lake -R -Kcuda=/usr/local/cuda` builds
    with CUDA, plain `lake -R` builds for CPU. Lake keeps the choice until the
    next `lake -R`. The `libtyr` target checks it against the fetched wheels. -/
def cudaHome? : Option String := get_config? cuda

/-- GPU to compile CUDA sources for, with `-Kcuda`: `-Kgpu=H100` (or `A100`,
    `B200`, `B300`, `GB10`). cc/Makefile maps it to the architecture flags. -/
def gpu? : Option String := get_config? gpu

/-- Runtime search path for `cc/build/libTyrC`, relative to the loading binary
    so the checkout can move. Binaries sit two to four levels below the repo
    root: executables in `.lake/build/bin`, module libraries in
    `.lake/build/lib/lean`. libtorch and the other vendored libraries are found
    through `libTyrC`'s own run path. -/
def tyrCRPathArgs : Array String :=
  let origin := if System.Platform.isOSX then "@loader_path" else "$ORIGIN"
  #["../..", "../../..", "../../../.."].map fun up => s!"-Wl,-rpath,{origin}/{up}/cc/build"

/-- Keep the Lean runtime's private copies of shared libraries out of the
    executable's exported symbols, where they would override the copies that
    libtorch uses (Lean links with `-rdynamic`):

    * Lake appends `-lunwind` (LLVM's static unwinder) to the bundled clang's
      link. libtorch and `libTyrC` unwind with libgcc's, and a C++ exception
      thrown from libtorch crashes in LLVM's. Linking `libgcc_s` first
      satisfies the `_Unwind_*` symbols, so no libunwind member is pulled in.
    * Lean's static libuv would replace the libuv bundled in `libtorch_cpu`
      (used by its TCPStore) symbol by symbol. -/
def linuxRuntimeIsolationLinkArgs : Array String :=
  #["-l:libgcc_s.so.1", "-Wl,--exclude-libs,libuv.a"]

def packageLinkArgs : Array String :=
  if System.Platform.isOSX then
    tyrCRPathArgs
  else
    linuxRuntimeIsolationLinkArgs ++ tyrCRPathArgs

package tyr where
  srcDir := "."
  buildDir := ".lake/build"
  moreServerArgs := #["-Dpp.unicode.fun=true"]
  moreLinkArgs := packageLinkArgs

require LeanTest from git "https://github.com/cpehle/lean_test.git" @ "b42cd3d78716e5a2de5b640ac82d7fe3f05f2a4c"

/-! ## C++ Library Build -/

/-- Shared library holding the C++ bindings, built by `cc/Makefile` with the
    system compiler. It carries libtorch, Arrow, soxr and the CUDA libraries as
    its own dependencies, so Lean code links against it with Lake's bundled
    toolchain (see `moreLinkLibs` on the package). -/
target libtyr pkg : Dynlib := do
  -- The configured CUDA choice (`-Kcuda`) must match what `deps/fetch.sh`
  -- put in external/; Make then builds exactly what it is told.
  let fetchedCuda ← (pkg.dir / "external" / "wheels" / "torch" / "lib" / "libtorch_cuda.so").pathExists
  match cudaHome?, fetchedCuda with
  | some cuda, false =>
    error s!"configured for CUDA (-Kcuda={cuda}) but external/ has the CPU libtorch; run: deps/fetch.sh cuda"
  | none, true =>
    error "configured for CPU but external/ has the CUDA libtorch; run: deps/fetch.sh cpu"
  | some cuda, true =>
    unless ← (FilePath.mk cuda / "bin" / "nvcc").pathExists do
      error s!"-Kcuda={cuda} has no bin/nvcc"
  | none, false => pure ()
  match cudaHome?, gpu? with
  | some cuda, none => error s!"-Kcuda={cuda} needs -Kgpu (H100, A100, B200, B300 or GB10)"
  | none, some gpu => error s!"-Kgpu={gpu} needs -Kcuda=<toolkit>"
  | _, _ => pure ()

  let tyrCLib := pkg.dir / "cc" / "build" / nameToSharedLib "TyrC"
  let sysroot ← getLeanSysroot
  let generatorEnv ← getAugmentedEnv
  let generatedDir := pkg.dir / "cc" / "src" / "generated"

  -- 1. C prototypes of every `TyrFFI` extern, which `libTyrC`'s sources include.
  --    The generator leaves an unchanged header untouched, so Make recompiles
  --    only when the prototypes change.
  let some ffiGenerator ← findLeanExe? `GenerateFfiHeader
    | error "lakefile.lean: no GenerateFfiHeader executable"
  let some ffiLib ← findLeanLib? `TyrFFI
    | error "lakefile.lean: no TyrFFI library"
  let ffiModules ← ffiLib.getModuleArray
  let ffiHeader := generatedDir / "tyr_ffi.h"
  let mut ffiInputs := (← ffiGenerator.exe.fetch).map fun _ => ()
  for mod in ffiModules do
    ffiInputs := ffiInputs.mix (← mod.olean.fetch)
  let ffiJob ← ffiInputs.mapM fun _ => do
    buildUnlessUpToDate ffiHeader (← getTrace) (pkg.buildDir / "codegen" / "ffi-header.trace") do
      proc {
        cmd := ffiGenerator.file.toString
        args := #[ffiHeader.toString] ++ ffiModules.map (·.name.toString)
        env := generatorEnv
      }

  -- 2. make, on every build: Make decides what to recompile. The library's
  --    content is its trace, so an unchanged library rebuilds nothing else.
  let libJob ← ffiJob.mapM fun _ => do
    let jobs ← match (← IO.getEnv "TYR_MAKE_JOBS").bind nonEmptyTrimmed? with
      | some jobs => pure jobs
      | none => captureProc { cmd := "getconf", args := #["_NPROCESSORS_ONLN"] }
    proc {
      cmd := "make"
      args := #[s!"-j{jobs}", "-C", (pkg.dir / "cc").toString, "dylib"]
      env := #[
        ("LEAN_HOME", some sysroot.toString),
        -- Unset for a CPU build, so a `CUDA_HOME` in the caller's shell is ignored.
        ("CUDA_HOME", cudaHome?),
        ("GPU", gpu?)
      ]
    }
    let libTrace ← computeTrace tyrCLib
    setTrace libTrace
    return libTrace

  -- 3. Every `lean_*` symbol the library exports must be a `TyrFFI` extern;
  --    the generated header only checks definitions Lean declares. The
  --    library keeps its own trace.
  let checkJob ← (libJob.mix ffiInputs).mapM fun _ => do
    buildUnlessUpToDate tyrCLib (← getTrace) (pkg.buildDir / "codegen" / "ffi-exports.trace") do
      proc {
        cmd := ffiGenerator.file.toString
        args := #["--check-exports", tyrCLib.toString] ++ ffiModules.map (·.name.toString)
        env := generatorEnv
      }
  let doneJob ← (libJob.zipWith (fun libTrace _ => libTrace) checkJob).mapM fun libTrace => do
    setTrace libTrace
    return tyrCLib
  return doneJob.map fun path => { path, name := "TyrC" }

/-! ## Lean Library -/

/-- Main Lean library: every `Tyr.*` module not claimed by `TyrFFI`. -/
@[default_target]
lean_lib Tyr where
  roots := #[`Tyr]
  precompileModules := true
  -- Linked into every executable that imports a Tyr module and into the
  -- precompiled module libraries.
  moreLinkLibs := #[`@/libtyr]

/-- Every `@[extern]` declaration of `Tyr`, with the types they use. Declared
    after `Tyr`: Lake gives a module to the last declared library whose roots
    match it, and `Tyr` matches every `Tyr.*`. Not precompiled, so
    its `.olean` files build without `libTyrC`: the `libtyr` target generates
    `libTyrC`'s FFI header from them. The declared functions are in `libTyrC`,
    so linking needs it. -/
lean_lib TyrFFI where
  roots := #[`Tyr.Basic, `Tyr.ConstantSupport, `Tyr.RawTensor, `Tyr.Audio.AppleInput,
    `Tyr.Audio.AppleOutput]
  globs := #[.one `Tyr.Basic, .one `Tyr.Basic.Types, .andSubmodules `Tyr.ConstantSupport, .andSubmodules `Tyr.RawTensor,
    .one `Tyr.Audio.FloatBuffer, .one `Tyr.Audio.AppleInput, .one `Tyr.Audio.AppleOutput]
  precompileModules := false
  moreLinkLibs := #[`@/libtyr]

/-- Test library containing all tests -/
lean_lib Tests where
  roots := #[`Tests]
  precompileModules := false

/-- Examples library -/
lean_lib Examples where
  roots := #[`Examples]
  precompileModules := false

/-! ## Executables -/

/-- Main test runner using LeanTest -/
@[test_driver]
lean_exe test_runner where
  root := `Tests.RunTests
  supportInterpreter := true

/-- Two-rank CPU collective value regressions; launched by scripts/test_distributed_values.py. -/
lean_exe RunDistributedValues where
  root := `Tests.RunDistributedValues
  supportInterpreter := true

/-- Write the C prototypes of `TyrFFI`'s extern declarations. Imports only
    `Lean`, so it does not link `libTyrC`. -/
lean_exe GenerateFfiHeader where
  root := `Tyr.GenerateFfiHeader

/-- Manual FFI failure-mode probe (intentionally crashes; not in the suite).
    Run it to check that an uncaught libtorch exception terminates with an
    intelligible message instead of a bare SIGABRT/SIGSEGV. -/
lean_exe ffi_crash_probe where
  root := `Tests.FfiCrashProbe
  supportInterpreter := true

/-- Focused LeanTest runner for the Riemannian nanoGPT tests. -/
lean_exe RunRiemannianNanoGPTTests where
  root := `Tests.RunRiemannianNanoGPTTests
  supportInterpreter := true

/-- GPT training executable -/
lean_exe TrainGPT where
  root := `Examples.TrainGPT
  supportInterpreter := true

/-- Exact-VJP Riemannian nanoGPT prototype runner. -/
lean_exe RunRiemannianNanoGPT where
  root := `Examples.GPT.RunRiemannianNanoGPT
  supportInterpreter := true

/-- Diffusion training executable -/
lean_exe TrainDiffusion where
  root := `Examples.TrainDiffusion
  supportInterpreter := true

/-- NanoChat training executable (modded GPT + distributed) -/
lean_exe TrainNanoChat where
  root := `Examples.NanoChat.TrainNanoChat
  supportInterpreter := true

/-- NanoChat multi-stage pipeline executable. -/
lean_exe NanoChatPipeline where
  root := `Examples.NanoChat.Pipeline
  supportInterpreter := true

/-- NanoChat checkpoint-backed chat/inference executable. -/
lean_exe NanoChatChat where
  root := `Examples.NanoChat.RunChat
  supportInterpreter := true

/-- Live microphone streaming Qwen3-ASR demo (macOS AudioToolbox input). -/
lean_exe Qwen3ASRLiveMic where
  root := `Examples.Qwen3ASR.LiveMic
  supportInterpreter := true

/-- Separate streaming-native ASR session executable (parallel path). -/
lean_exe Qwen3ASRLiveMicTrueStream where
  root := `Examples.Qwen3ASR.LiveMicTrueStream
  supportInterpreter := true

/-- Diffusion tests executable -/
lean_exe TestDiffusion where
  root := `Tests.RunTestDiffusion
  supportInterpreter := true

/-- DataLoader test executable -/
lean_exe TestDataLoader where
  root := `Tests.RunTestDataLoader
  supportInterpreter := true

/-- Laguna NVFP4 dequantization tests. -/
lean_exe LagunaNvFp4Test where
  root := `Tests.RunLagunaNvFp4
  supportInterpreter := true

/-- Laguna MoE block (router + packed experts) tests. -/
lean_exe LagunaMoeTest where
  root := `Tests.RunLagunaMoe
  supportInterpreter := true

/-- Laguna attention/model forward tests. -/
lean_exe LagunaModelTest where
  root := `Tests.RunLagunaModel
  supportInterpreter := true

/-- Laguna fused NVFP4 MoE kernel tests. -/
lean_exe LagunaFusedTest where
  root := `Tests.RunLagunaFused
  supportInterpreter := true

/-- Laguna rotary (YaRN + plain) table tests. -/
lean_exe LagunaRopeTest where
  root := `Tests.RunLagunaRope
  supportInterpreter := true

/-- Laguna-S-2.1 model loader/generation demo with HF repo-id resolution. -/
lean_exe LagunaRunHF where
  root := `Examples.Laguna.RunHF
  supportInterpreter := true

/-- Flux image generation demo -/
lean_exe FluxDemo where
  root := `Examples.Flux.FluxDemo
  supportInterpreter := true

/-- End-to-end Qwen3-TTS demo (Lean talker + Python speech-tokenizer decode). -/
lean_exe Qwen3TTSEndToEnd where
  root := `Examples.Qwen3TTS.EndToEnd
  supportInterpreter := true

/-- Offline KittenTTS / Kokoro synthesis demo using converted safetensors checkpoints. -/
lean_exe KittenTTSPretrained where
  root := `Examples.KittenTTSPretrained
  supportInterpreter := true

/-- Offline Qwen3-ASR transcription demo (fully Lean pipeline). -/
lean_exe Qwen3ASRTranscribe where
  root := `Examples.Qwen3ASR.Transcribe
  supportInterpreter := true

/-- Offline Whisper transcription demo (native Tyr encoder-decoder implementation). -/
lean_exe WhisperTranscribe where
  root := `Examples.Whisper.Transcribe
  supportInterpreter := true

/-- Interactive Whisper voice mode with microphone input and silence detection. -/
lean_exe WhisperVoiceMode where
  root := `Examples.Whisper.VoiceMode
  supportInterpreter := true

/-- Isolated test: in-memory Whisper transcription (no WAV round-trip). -/
lean_exe WhisperTranscribeInMem where
  root := `Examples.Whisper.TranscribeInMem
  supportInterpreter := true

/-- Qwen3.5 model loader/generation demo with HF repo-id resolution. -/
lean_exe Qwen35RunHF where
  root := `Examples.Qwen35.RunHF
  supportInterpreter := true

/-- Qwen2.5-Omni thinker text loader/generation demo (3B/7B). -/
lean_exe Qwen25OmniRunHF where
  root := `Examples.Qwen25Omni.RunHF
  supportInterpreter := true

/-- Gemma 4 text loader/generation demo with HF repo-id resolution. -/
lean_exe Gemma4RunHF where
  root := `Examples.Gemma4.RunHF
  supportInterpreter := true

/-! Parity probes run by `parity/run.py`: `Parity<Name>` is `Examples/Parity/<Name>.lean`. -/

lean_exe ParityQwen3 where
  root := `Examples.Parity.Qwen3
  supportInterpreter := true

lean_exe ParityQwen35 where
  root := `Examples.Parity.Qwen35
  supportInterpreter := true

lean_exe ParityQwen25Omni where
  root := `Examples.Parity.Qwen25Omni
  supportInterpreter := true

lean_exe ParityGemma4 where
  root := `Examples.Parity.Gemma4
  supportInterpreter := true

lean_exe ParityWhisper where
  root := `Examples.Parity.Whisper
  supportInterpreter := true

lean_exe ParityLaguna where
  root := `Examples.Parity.Laguna
  supportInterpreter := true

lean_exe ParityQwen3ASR where
  root := `Examples.Parity.Qwen3ASR
  supportInterpreter := true

lean_exe ParityQwen3ASRAligner where
  root := `Examples.Parity.Qwen3ASRAligner
  supportInterpreter := true

lean_exe ParitySileroVAD where
  root := `Examples.Parity.SileroVAD
  supportInterpreter := true

lean_exe ParityQwen3TTSEncode where
  root := `Examples.Parity.Qwen3TTSEncode
  supportInterpreter := true

lean_exe ParityQwen3TTSSpeaker where
  root := `Examples.Parity.Qwen3TTSSpeaker
  supportInterpreter := true

lean_exe ParityQwen3TTSDecode where
  root := `Examples.Parity.Qwen3TTSDecode
  supportInterpreter := true

lean_exe ParityQwen3TTSTalker where
  root := `Examples.Parity.Qwen3TTSTalker
  supportInterpreter := true

lean_exe ParityKittenTTS where
  root := `Examples.Parity.KittenTTS
  supportInterpreter := true

/-- Focused native acceptance path while model tensor migration is in progress. -/
lean_exe test_typed_models where
  root := `Tests.RunTypedModels
  supportInterpreter := true

lean_exe ParityFluxText where
  root := `Examples.Parity.FluxText
  supportInterpreter := true

lean_exe ParityFluxStep where
  root := `Examples.Parity.FluxStep
  supportInterpreter := true

lean_exe ParityFluxSample where
  root := `Examples.Parity.FluxSample
  supportInterpreter := true
