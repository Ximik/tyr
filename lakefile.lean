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

/-- GPU to build CUDA kernels for, with `-Kcuda`: `-Kgpu=H100` (or `A100`,
    `B200`, `B300`, `GB10`). cc/Makefile maps it to the architecture flags. -/
def gpu? : Option String := get_config? gpu

/-- Kernel modules to generate CUDA for, space-separated:
    `-Kkernels="Tyr.GPU.Kernels.MhaH100 Tyr.GPU.Kernels.MhaH100Decode"`. -/
def gpuKernels : String := (get_config? kernels).getD "Tyr.GPU.Kernels.MhaH100"

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
  -- Linked into every executable and precompiled module library.
  moreLinkLibs := #[`@/libtyr]

require LeanTest from git "https://github.com/cpehle/lean_test.git" @ "b42cd3d78716e5a2de5b640ac82d7fe3f05f2a4c"
require LeanBenchmark from git "https://github.com/cpehle/lean-benchmark.git" @
  "9ab68a2e976aef3791b5b5630be8f5f1e8f79fe9"
require LeanUrdfTypeProvider from git
  "https://github.com/ranvier-labs/lean-urdf-typeprovider.git" @
  "5712c1fcdf4462d1e7a216f12159651381410149"

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
  let gpuIrRoot := pkg.buildDir / "ir" / "Tyr" / "GPU"
  let generatedCudaDir := pkg.dir / "cc" / "src" / "generated"
  let gpuCodegenConfigPath := pkg.buildDir / "libtyr_gpu_codegen.env"
  let gpuCodegenModule := gpuKernels
  let gpuCodegenModules : Array String :=
    (gpuCodegenModule.splitOn " ").toArray.filterMap (fun s => nonEmptyTrimmed? s)
  let skipGpuCodegenValue := (← IO.getEnv "TYR_SKIP_GPU_CODEGEN").getD ""
  let gpuCodegenConfig :=
    s!"TYR_GPU_CODEGEN_MODULE={gpuCodegenModule}\nTYR_SKIP_GPU_CODEGEN={skipGpuCodegenValue}\n"
  let shouldWriteConfig ← do
    if ← gpuCodegenConfigPath.pathExists then
      pure ((← IO.FS.readFile gpuCodegenConfigPath) != gpuCodegenConfig)
    else
      pure true
  if shouldWriteConfig then
    -- On a fresh checkout (e.g. CI runner) `pkg.buildDir` may not yet
    -- exist; `writeFile` won't create it.
    IO.FS.createDirAll pkg.buildDir
    IO.FS.writeFile gpuCodegenConfigPath gpuCodegenConfig

  let sysroot ← getLeanSysroot
  let nativeEnv := #[
    ("LEAN_HOME", some sysroot.toString),
    -- Unset for a CPU build, so a `CUDA_HOME` in the caller's shell is ignored.
    ("CUDA_HOME", cudaHome?),
    ("GPU", gpu?),
    ("TYR_GPU_CODEGEN_MODULE", some gpuCodegenModule)
  ]
  -- Refresh content-stable manifests before Lake checks its native trace. Make
  -- owns effective compiler/GPU detection; the stub inventory also notices new
  -- kernel declarations without invalidating every native object on body edits.
  let nativeConfigOut ← IO.Process.output {
    cmd := "make"
    args := #["-s", "-C", (pkg.dir / "cc").toString, "native-config", "gpu-stubs"]
    env := nativeEnv
  }
  if nativeConfigOut.exitCode != 0 then
    error s!"Failed to refresh native build inputs:\n{nativeConfigOut.stderr}"

  -- Track Makefile plus C/CUDA sources/headers so Lake reruns `make` when FFI changes.
  let makefileJob ← inputTextFile <| pkg.dir / "cc" / "Makefile"
  let gpuCodegenConfigJob ← inputTextFile gpuCodegenConfigPath
  let nativeConfigJob ← inputTextFile <| pkg.dir / "cc" / "build" / "native-build.json"
  let nativeDependenciesPath := pkg.dir / "cc" / "build" / "native-dependencies.txt"
  let nativeDependenciesManifestJob ← inputTextFile nativeDependenciesPath
  let mut nativeDependenciesJob := Job.mixArray #[nativeDependenciesManifestJob]
  -- Consume compiler-discovered dependencies too (including vendor headers).
  -- Make exports absolute, existing paths; removed headers change this manifest.
  for path in (← IO.FS.readFile nativeDependenciesPath).splitOn "\n" do
    if !path.isEmpty then
      let header ← inputTextFile (FilePath.mk path)
      nativeDependenciesJob := nativeDependenciesJob.mix header
  let srcJob ← inputDir (pkg.dir / "cc" / "src") (text := true) fun p =>
    p.toString.endsWith ".cpp" || p.toString.endsWith ".mm" ||
      p.toString.endsWith ".cu" || p.toString.endsWith ".h" || p.toString.endsWith ".hpp"
  let headerJob ← inputDir (pkg.dir / "cc" / "include") (text := true) fun p =>
    p.toString.endsWith ".h" || p.toString.endsWith ".hpp"
  let toolJob ← inputDir (pkg.dir / "cc" / "tools") (text := true) fun p =>
    p.toString.endsWith ".py"
  -- Note: we deliberately do NOT watch the kernel `.lean`
  -- source tree). Changes to a kernel `.lean` file flow through Lean
  -- compilation to its `.c.o.export`, which `gpuIrJob` already watches with
  -- the right scope (only the active codegen module's IR triggers a rebuild).
  -- Watching the whole `Tyr/GPU/Kernels/` directory caused every kernel-edit
  -- in the workspace to invalidate the libtyr build cascade.
  -- Fresh checkouts do not have the generated GPU IR tree yet.
  -- Create it so the optional IR scan can track later `.c.o.export` files instead of failing early.
  IO.FS.createDirAll gpuIrRoot
  let gpuIrJob ←
    if gpuCodegenModule == "Tyr.GPU.Kernels.MhaH100" then
      let mhaH100IrSuffixes : Array String := #[
        "Kernels/MhaH100.c.o.export",
        "Kernels/Prelude.c.o.export",
        "Types.c.o.export",
        "Codegen/Macros.c.o.export",
        "Codegen/Var.c.o.export",
        "Codegen/TileTypes.c.o.export",
        "Codegen/IR.c.o.export",
        "Codegen/Monad.c.o.export",
        "Codegen/AST.c.o.export",
        "Codegen/Primitives.c.o.export",
        "Codegen/Loop.c.o.export",
        "Codegen/GlobalLayout.c.o.export",
        "Codegen/EmitNew.c.o.export",
        "Codegen/Attribute.c.o.export",
        "Codegen/FFI.c.o.export",
        "Codegen/GenerateMain.c.o.export",
        "Codegen/Arch/Level.c.o.export"
      ]
      inputDir gpuIrRoot (text := false) fun p =>
        mhaH100IrSuffixes.any fun suffix => p.toString.endsWith suffix
    else
      inputDir gpuIrRoot (text := false) fun p =>
        p.toString.endsWith ".c.o.export"
  let depJob := makefileJob.mix gpuCodegenConfigJob |>.mix nativeConfigJob |>.mix nativeDependenciesJob
    |>.mix srcJob |>.mix headerJob |>.mix toolJob |>.mix gpuIrJob

  let libJob ← buildFileAfterDep tyrCLib depJob fun _ => do
    -- TYR_SKIP_GPU_CODEGEN: "1" skips, "0" forces; unset skips when make found
    -- no nvcc, since the Makefile then drops generated .cu files and links the
    -- weak launcher stubs (refreshed by `gpu-stubs` above) instead.
    let hasNvcc := (← IO.FS.readFile (pkg.dir / "cc" / "build" / "native-build.json")).contains
      "\"HAS_NVCC\": \"1\""
    let skipGpuCodegen :=
      match (← IO.getEnv "TYR_SKIP_GPU_CODEGEN").bind nonEmptyTrimmed? with
      | some "1" => true
      | some "0" => false
      | _ => !hasNvcc
    if !skipGpuCodegen then
      let generatorExe := pkg.dir / ".lake" / "build" / "bin" / "GenerateGpuKernels"
      proc {
        cmd := "lake"
        args := #["build", "GenerateGpuKernels"]
        cwd := pkg.dir
        -- The nested Lake reads the same stored -K options; it only has to
        -- skip codegen to break the cycle (see `TYR_SKIP_GPU_CODEGEN` above).
        env := #[
          ("LEAN_HOME", some sysroot.toString),
          ("TYR_SKIP_GPU_CODEGEN", some "1")
        ]
      }
      proc {
        cmd := "lake"
        args := #["env", generatorExe.toString]
                  ++ gpuCodegenModules
                  ++ #["--out-dir", generatedCudaDir.toString]
        cwd := pkg.dir
        env := #[
          ("LEAN_HOME", some sysroot.toString),
          ("TYR_SKIP_GPU_CODEGEN", some "1")
        ]
      }
    -- Parallel jobs for make: TYR_MAKE_JOBS, or the CPU count.
    let jobs ← match (← IO.getEnv "TYR_MAKE_JOBS").bind nonEmptyTrimmed? with
      | some jobs => pure jobs
      | none => captureProc { cmd := "getconf", args := #["_NPROCESSORS_ONLN"] }
    proc {
      cmd := "make"
      args := #[s!"-j{jobs}", "-C", (pkg.dir / "cc").toString, "dylib"]
      env := nativeEnv
    }
  return libJob.map fun path => { path, name := "TyrC" }

/-! ## Lean Library -/

/-- Codegen-only sub-library.

    Owns `Tyr.GPU.Codegen.*` plus the small set of `Tyr.GPU.*` modules they
    depend on (Types, Capabilities, Tile) and `Tyr.Basic`. These modules are
    pure Lean — no FFI, no libtorch — so we explicitly disable
    `precompileModules`. That way `lean_exe GenerateGpuKernels` (whose root
    is `Tyr.GPU.Codegen.GenerateMain`) doesn't trigger the per-module `.so`
    cascade across all of `Tyr.*` whenever a kernel `.lean` file is touched. -/
lean_lib TyrCodegen where
  roots := #[
    `Tyr.GPU.Codegen,
    `Tyr.GPU.Types,
    `Tyr.GPU.Capabilities,
    `Tyr.GPU.Tile,
    `Tyr.Basic
  ]
  precompileModules := false

/-- Main Lean library containing all Tyr modules.

    `roots := #[\`Tyr]` claims everything under `Tyr.*` that isn't already
    owned by `TyrCodegen` (a more specific match for `Tyr.GPU.Codegen.*`
    etc. wins per Lake's lib resolution). -/
@[default_target]
lean_lib Tyr where
  roots := #[`Tyr]
  precompileModules := true

/-- Test library containing all tests -/
lean_lib Tests where
  roots := #[`Tests]
  precompileModules := false

/-- Experimental tests that track in-progress modules. -/
lean_lib TestsExperimental where
  roots := #[`TestsExperimental]
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

/-- Native tensor-free MCTS search and allocation microbenchmarks. -/
lean_exe mctx_bench where
  root := `benchmarks.Mctx

/-- Generate CUDA translation units from registered @[gpu_kernel] declarations.

    Its root `Tyr.GPU.Codegen.GenerateMain` lives in the codegen-only sub-lib
    `TyrCodegen` (above) so the per-module `.so` cascade across `Tyr.*` is
    skipped. -/
lean_exe GenerateGpuKernels where
  root := `Tyr.GPU.Codegen.GenerateMain
  supportInterpreter := true

/-- Compile registered @[tileir_kernel] declarations through NVIDIA TileIR tooling. -/
lean_exe GenerateTileIRKernels where
  root := `Tyr.GPU.Codegen.TileIR.GenerateMain
  supportInterpreter := true

/-- Experimental test runner for unstable/in-progress modules. -/
lean_exe test_runner_experimental where
  root := `Tests.RunTestsExperimental
  supportInterpreter := true

/-- BranchingFlows continuous overfit training check. -/
lean_exe BranchingFlowsContinuousTrain where
  root := `Examples.BranchingFlows.ContinuousTrainDemo
  supportInterpreter := true

/-- BranchingFlows molecule overfit training check. -/
lean_exe BranchingFlowsMoleculeTrain where
  root := `Examples.BranchingFlows.MoleculeTrainDemo
  supportInterpreter := true

/-- Molecule-shaped oracle generation with branch-event trajectory export. -/
lean_exe BranchingFlowsMoleculeGenerate where
  root := `Examples.BranchingFlows.MoleculeGenerationDemo
  supportInterpreter := true

/-- BranchingFlows molecule transformer overfit training check. -/
lean_exe BranchingFlowsMoleculeTransformerTrain where
  root := `Examples.BranchingFlows.MoleculeTransformerTrainDemo
  supportInterpreter := true

/-- Dataset-backed BranchingFlows molecule transformer training and generation. -/
lean_exe BranchingFlowsMoleculeTrainGenerate where
  root := `Examples.BranchingFlows.MoleculeTrainGenerate
  supportInterpreter := true

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

/-- URDF-backed hybrid contact event-skeleton simulation demo. -/
lean_exe RunUrdfContactExample where
  root := `Examples.EventSkeleton.RunUrdfContactExample
  supportInterpreter := true

/-- AlphaGrad-style RoeFlux_1d elimination planning port demo. -/
lean_exe AlphaGradRoeFlux1dA0 where
  root := `Examples.AlphaGradPort.RoeFlux1dA0
  supportInterpreter := true

/-- AlphaGrad port task sweep runner (targets tasks one-by-one). -/
lean_exe AlphaGradPortSweep where
  root := `Examples.AlphaGradPort.TaskSweep
  supportInterpreter := true

/-- AlphaGrad policy-training runner with real parameter updates. -/
lean_exe AlphaGradPolicyTrain where
  root := `Examples.AlphaGradPort.PolicyTrainMain
  supportInterpreter := true

/-- AlphaGrad policy-training sweep runner across tasks and training modes. -/
lean_exe AlphaGradPolicySweep where
  root := `Examples.AlphaGradPort.PolicySweepMain
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

/-- Differential equation baseline test executable. -/
lean_exe TestDiffEq where
  root := `Tests.RunTestDiffEq
  supportInterpreter := true

/-- Adjoint differential equation test executable. -/
lean_exe TestDiffEqAdjoint where
  root := `Tests.RunTestDiffEqAdjoint
  supportInterpreter := true

/-- Core adjoint differential equation test executable. -/
lean_exe TestDiffEqAdjointCore where
  root := `Tests.RunTestDiffEqAdjointCore
  supportInterpreter := true

/-- GPU DSL regression test executable. -/
lean_exe TestGPUDSL where
  root := `Tests.RunTestGPUDSL
  supportInterpreter := true

/-- GPU kernel fixture test executable. -/
lean_exe TestGPUKernels where
  root := `Tests.RunTestGPUKernels
  supportInterpreter := true

/-- End-to-end GPU parity tests (Tyr vs PyTorch, with optional vendored references). -/
lean_exe TestGPUE2E where
  root := `Tests.RunGPUE2E
  supportInterpreter := true

/-- GB10/Blackwell-specific end-to-end GPU parity tests. -/
lean_exe TestGPUGB10E2E where
  root := `Tests.RunGPUGB10E2E
  supportInterpreter := true

/-- Laguna config.json parsing tests. -/
lean_exe LagunaConfigTest where
  root := `Tests.RunLagunaConfig
  supportInterpreter := true

/-- Laguna tokenizer encode/decode tests. -/
lean_exe LagunaTokenizerTest where
  root := `Tests.RunLagunaTokenizer
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

/-- Laguna end-to-end parity test vs the HF reference (tiny fixture). -/
lean_exe LagunaParityTest where
  root := `Tests.RunLagunaParity
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

/-- NVIDIA TileIR rendering and toolchain driver tests. -/
lean_exe TestGPUTileIR where
  root := `Tests.RunTestGPUTileIR
  supportInterpreter := true

/-- TileIR export driver regression tests. -/
lean_exe TestTileIRGenerateMain where
  root := `Tests.RunTestTileIRGenerateMain
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

lean_exe KittenTTSDurations where
  root := `Examples.KittenTTSDurations
  supportInterpreter := true

lean_exe KittenTTSDebug where
  root := `Examples.KittenTTSDebug
  supportInterpreter := true

lean_exe KittenTTSCompare where
  root := `Examples.KittenTTSCompare
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

/-- Flux debug harness (saves intermediate tensors) -/
lean_exe FluxDebug where
  root := `Examples.Flux.FluxDebug
  supportInterpreter := true

/-- End-to-end demo for a minimal ThunderKittens-style copy kernel. -/
lean_exe RunCopy where
  root := `Examples.GPU.RunCopyExe
  supportInterpreter := true

/-- End-to-end rotary fixture validation using a ThunderKittens-style kernel. -/
lean_exe RunRotary where
  root := `Examples.GPU.RunRotaryExe
  supportInterpreter := true

/-- End-to-end ThunderKittens layernorm fixture validation. -/
lean_exe RunLayerNorm where
  root := `Examples.GPU.RunLayerNormExe
  supportInterpreter := true

/-- End-to-end fused residual + RMSNorm fixture validation. -/
lean_exe RunRMSNorm where
  root := `Examples.GPU.RunRMSNormExe
  supportInterpreter := true

/-- Fused BF16 cross-entropy training benchmark. -/
lean_exe RunLoss where
  root := `Examples.GPU.RunLoss
  supportInterpreter := true

/-- Fused mixed-precision AdamW training benchmark. -/
lean_exe RunOptimizer where
  root := `Examples.GPU.RunOptimizer
  supportInterpreter := true

/-- End-to-end ThunderKittens flash attention fixture validation. -/
lean_exe RunFlashAttn where
  root := `Examples.GPU.RunFlashAttnExe
  supportInterpreter := true

/-- End-to-end FlashAttention3 validation. -/
lean_exe RunFlashAttn3 where
  root := `Examples.GPU.RunFlashAttn3
  supportInterpreter := true

/-- Runtime validation for the high-level `tyr::flash_attn` bridge. -/
lean_exe RunFlashAttnOp where
  root := `Examples.GPU.RunFlashAttnOp
  supportInterpreter := true

/-- One-H100 benchmark scaffold for the `tyr::flash_attn` bring-up. -/
lean_exe RunFlashAttnBench where
  root := `Examples.GPU.RunFlashAttnBench
  supportInterpreter := true

/-- Numerical correctness harness for the TK-style decode kernel.

    Generates random Q/K/V plus a torch SDPA reference for several decode
    shapes (Llama-3 head_dim=128, Qwen3-4B head_dim=64, tail-mask case,
    single-block case, batch>1) and asserts kernel parity within BF16
    tolerance. Also runs a cache-vs-no-cache parity test against
    `Cache.attendLayer`.

    Usage: `lake exe RunMhaH100Decode [--regen|--gen-only]`. -/
lean_exe RunMhaH100Decode where
  root := `Examples.GPU.RunMhaH100Decode
  supportInterpreter := true

/-- Decode-specific perf benchmark: timed forward over the same shape
    matrix as `RunMhaH100Decode`, reporting p50 latency vs PyTorch SDPA
    plus a couple of long-context (kv_seq=8k) rows. Forward-only
    (decode is inference, no backward).

    Usage: `lake exe RunDecodeBench [--case all|<id>] [--backend all|tyr|sdpa]
                                    [--warmup N] [--iters N] [--repeats N]
                                    [--jsonl-out path]`. -/
lean_exe RunDecodeBench where
  root := `Examples.GPU.RunDecodeBench
  supportInterpreter := true

/-- End-to-end ThunderKittens `mha_h100` forward/backward fixture validation. -/
lean_exe RunMhaH100 where
  root := `Examples.GPU.RunMhaH100Exe
  supportInterpreter := true

/-- End-to-end `mha_h100` training/benchmark demo (kernel + optional torch baseline). -/
lean_exe RunMhaH100Train where
  root := `Examples.GPU.RunMhaH100Train
  supportInterpreter := true

/-- End-to-end GB10 MHA validation and synchronized benchmark. -/
lean_exe RunMhaGB10 where
  root := `Examples.GPU.RunMhaGB10Exe
  supportInterpreter := true

/-- End-to-end multi-block `mha_h100` validation (`seq=768`, `d=64`). -/
lean_exe RunMhaH100Seq768 where
  root := `Examples.GPU.RunMhaH100Seq768
  supportInterpreter := true

/-- End-to-end Blackwell/B200 BF16 GEMM validation. -/
lean_exe RunB200Bf16Gemm where
  root := `Examples.GPU.RunB200Bf16Gemm
  supportInterpreter := true
