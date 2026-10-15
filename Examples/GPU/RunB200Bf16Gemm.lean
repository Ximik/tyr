/- End-to-end Blackwell/B200 BF16 GEMM validation. -/
import Tyr.Torch
import Tyr.Typed.Neural
import Tyr.GPU.Kernels.Bf16Gemm
import Examples.GPU.IndexedParity
import Examples.GPU.GemmReference
import Examples.GPU.FixtureRunner
import Examples.GPU.Benchmark

namespace Examples.GPU.RunB200Bf16Gemm

open torch

variable {device : Device}
open Tyr.GPU.Kernels.Bf16Gemm

def suiteName : String := "b200_bf16_gemm"

def fixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/b200_bf16_gemm_256x256x64"⟩
  names := #["a", "b", "expected_c"]
}

def fixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fixtureSpec name

def generateFixtures : IO Unit := do
  if !(← requireCuda suiteName) then
    throw <| IO.userError "CUDA is not available; cannot generate b200 bf16 GEMM fixtures."

  IO.FS.createDirAll fixtureSpec.dir
  let device ← Device.cuda 0

  let aFloat ← Tensor.randn #[256, 64] false device
  let bFloat ← Tensor.randn #[256, 64] false device
  let a := aFloat.toBFloat16
  let b : Tensor { shape := #[256, 64], dtype := .BFloat16, device := device } := bFloat.toBFloat16
  let expectedFloat : Tensor { shape := #[256, 256], dtype := .Float32, device := device } := gemmReference a b
  let expected := expectedFloat.toBFloat16

  Tensor.saveTensor a (fixtureFile "a").toString
  Tensor.saveTensor b (fixtureFile "b").toString
  Tensor.saveTensor expected (fixtureFile "expected_c").toString

  let aMean := Tensor.item (Tensor.meanAll aFloat)
  let cMean := Tensor.item (Tensor.meanAll expectedFloat)
  IO.println s!"Generated b200 bf16 GEMM fixtures in {fixtureSpec.dir} aMean={aMean} expectedMean={cMean}"

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  let device ← Device.cuda 0
  if !(← isBlackwellFamily) then
    IO.println s!"[skip] {suiteName}: requires TYR_GPU_FAMILY=BLACKWELL"
    return true

  if !(← fixturesPresent fixtureSpec) then
    generateFixtures

  let a ← Tensor.loadTensorExact { device := device, shape := #[256, 64], dtype := .BFloat16 } (fixtureFile "a").toString
  let b ← Tensor.loadTensorExact { device := device, shape := #[256, 64], dtype := .BFloat16 } (fixtureFile "b").toString
  let expected ← Tensor.loadTensorExact { device := device, shape := #[256, 256], dtype := .BFloat16 } (fixtureFile "expected_c").toString

  let output ← Tensor.uniformLike expected 0 0
  let stream ← torch.cuda_current_stream

  -- One CTA covers the full 256x256 tile for this focused parity case.
  tkGB10Bf16GemmFwd.launch a.raw b.raw output.raw 256 256 64 4 4 1 32 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let check := IndexedParity.compareTensors "b200_bf16_gemm.output" expected output 5e-2 5e-2
  logTensorCheck check
  pure check.ok

private def parseNatArg (args : List String) (flag : String) (default : Nat) : Nat := Id.run do
  let rec loop (xs : List String) : Nat :=
    match xs with
    | key :: value :: rest =>
        if key == flag then value.toNat?.getD default else loop (value :: rest)
    | _ => default
  loop args

private structure GemmShape where
  m : UInt64
  n : UInt64
  k : UInt64

private structure TrainingBatchPoint where
  label : String
  tokens : UInt64
  scaleTags : Array String

private def trainingBatchPoints : Array TrainingBatchPoint := #[
  { label := "b1", tokens := 768, scaleTags := #["latency"] },
  { label := "b2", tokens := 1536, scaleTags := #["throughput"] },
  { label := "b4", tokens := 3072, scaleTags := #["throughput", "primary"] },
  { label := "b8", tokens := 6144, scaleTags := #["throughput", "saturation"] }
]

private def qwen3TtsTalkerCases : Array (LeanBenchmark.MatrixCase GemmShape) :=
  trainingBatchPoints.flatMap fun point =>
    let modelProfiles :=
      if point.scaleTags.contains "primary" then
        #["qwen3tts-talker", "qwen3tts-talker-primary"]
      else
        #["qwen3tts-talker"]
    let projectionTags := #["training", "projection"] ++ point.scaleTags
    let mlpTags := #["training", "mlp"] ++ point.scaleTags
    #[
      { id := s!"square_h1024_{point.label}_s768_fwd_dx",
        payload := ⟨point.tokens, 1024, 1024⟩,
        profiles := #["gb10-realistic", "model-shapes", "batch-sweep",
          "training-triplets", "projection-triplets"] ++ modelProfiles,
        tags := projectionTags ++ #["q-output", "forward", "activation-gradient"] },
      { id := s!"square_h1024_{point.label}_s768_dw",
        payload := ⟨1024, 1024, point.tokens⟩,
        profiles := #["model-shapes", "training-triplets", "weight-grad-sweep",
          "projection-triplets"] ++ modelProfiles,
        tags := projectionTags ++ #["q-output", "weight-gradient"] },
      { id := s!"qwen3tts_talker_kv_{point.label}_s768_fwd",
        payload := ⟨point.tokens, 128, 1024⟩,
        profiles := #["model-shapes", "projection-triplets"] ++ modelProfiles,
        tags := projectionTags ++ #["kv", "forward"] },
      { id := s!"qwen3tts_talker_kv_{point.label}_s768_dx",
        payload := ⟨point.tokens, 1024, 128⟩,
        profiles := #["model-shapes", "projection-triplets"] ++ modelProfiles,
        tags := projectionTags ++ #["kv", "activation-gradient"] },
      { id := s!"qwen3tts_talker_kv_{point.label}_s768_dw",
        payload := ⟨128, 1024, point.tokens⟩,
        profiles := #["model-shapes", "projection-triplets"] ++ modelProfiles,
        tags := projectionTags ++ #["kv", "weight-gradient"] },
      { id := s!"qwen3tts_talker_mlp_up_{point.label}_s768_fwd_down_dx",
        payload := ⟨point.tokens, 2048, 1024⟩,
        profiles := #["model-shapes", "mlp-triplets"] ++ modelProfiles,
        tags := mlpTags ++ #["up-gate-forward", "down-activation-gradient"] },
      { id := s!"qwen3tts_talker_mlp_down_{point.label}_s768_fwd_up_dx",
        payload := ⟨point.tokens, 1024, 2048⟩,
        profiles := #["model-shapes", "mlp-triplets"] ++ modelProfiles,
        tags := mlpTags ++ #["down-forward", "up-gate-activation-gradient"] },
      { id := s!"qwen3tts_talker_mlp_up_{point.label}_s768_dw",
        payload := ⟨2048, 1024, point.tokens⟩,
        profiles := #["model-shapes", "mlp-triplets"] ++ modelProfiles,
        tags := mlpTags ++ #["up-gate-weight-gradient"] },
      { id := s!"qwen3tts_talker_mlp_down_{point.label}_s768_dw",
        payload := ⟨1024, 2048, point.tokens⟩,
        profiles := #["model-shapes", "mlp-triplets"] ++ modelProfiles,
        tags := mlpTags ++ #["down-weight-gradient"] }
    ]

private def gemmMatrix : LeanBenchmark.Matrix GemmShape := {
  defaultProfile := "gb10-realistic"
  cases := #[
    { id := "tiny_m256_n256_k64", payload := ⟨256, 256, 64⟩,
      profiles := #["quick", "micro"], tags := #["launch-bound", "forward"] }
  ] ++ qwen3TtsTalkerCases
}

private def customShapeRequested (args : List String) : Bool :=
  args.contains "--m" || args.contains "--n" || args.contains "--k"

private def benchmarkShape (cfg : Benchmark.Config) (caseLabel : String) (m n k : UInt64) : IO (String × Bool) := do
  let device ← Device.cuda 0
  let a : Tensor { shape := #[m, k], dtype := .BFloat16, device := device } := (← Tensor.randn #[m, k] false device).toBFloat16
  let b : Tensor { shape := #[n, k], dtype := .BFloat16, device := device } := (← Tensor.randn #[n, k] false device).toBFloat16
  let expectedFloat : Tensor { shape := #[m, n], dtype := .Float32, device := device } :=
    gemmReference a b
  let expected : Tensor { shape := #[m, n], dtype := .BFloat16, device := device } := expectedFloat.toBFloat16
  let output ← Tensor.uniformLike expected 0 0
  let stream ← torch.cuda_current_stream
  let atol := if k >= 3072 then 1.0 else if k >= 1536 then 0.5 else 5e-2
  let gridX := n / 64
  let gridY := m / 64
  let route := if k == 64 then
    "generated_gb10_warp32_64x64_k64"
  else if k == 1024 then
    "generated_gb10_warp4_rows16_64x64_k1024_specialized_tma2"
  else
    s!"generated_gb10_warp4_rows16_64x64_runtime_k{k}_tma2"
  let launch : IO Unit :=
    if k == 64 then
      tkGB10Bf16GemmFwd.launch a.raw b.raw output.raw m n k gridX gridY 1 32 1 1 0 stream
    else if k == 1024 then
      tkGB10Bf16GemmK1024Fwd.launch a.raw b.raw output.raw m n k gridX gridY 1 128 1 1 0 stream
    else
      tkGB10Bf16GemmKRuntimeFwd.launch a.raw b.raw output.raw m n k gridX gridY 1 128 1 1 0 stream
  launch
  torch.cuda_synchronize
  let check := IndexedParity.compareTensors s!"gb10_bf16_gemm.bench.{m}x{n}x{k}" expected output 5e-2 atol
  logTensorCheck check
  let samples ← Benchmark.timeCudaEvents cfg stream launch
  let postTimingCheck := IndexedParity.compareTensors s!"gb10_bf16_gemm.bench.post_timing.{m}x{n}x{k}" expected output 5e-2 atol
  logTensorCheck postTimingCheck
  let correct := check.ok && postTimingCheck.ok
  let caseId := s!"bf16_gemm_{caseLabel}_{m}x{n}x{k}"
  let flopCount := 2.0 * m.toFloat * n.toFloat * k.toFloat
  pure (Benchmark.summaryJson cfg caseId "tyr" route samples correct
    "kernel_only" true (some flopCount) (some "FLOP"), correct)

private def runBenchmark (args : List String) : IO UInt32 := do
  if !(← requireCuda suiteName) then return 1
  if !(← isBlackwellFamily) then
    IO.eprintln "[gb10_bf16_gemm] benchmark requires TYR_GPU_FAMILY=BLACKWELL"
    return 1
  let cfg ← Benchmark.parseConfig args "gb10_bf16_gemm_bench"
  torch.manualSeed 0
  let selected : Array (String × GemmShape) ← if customShapeRequested args then
    let m := parseNatArg args "--m" 256
    let n := parseNatArg args "--n" 256
    let k := parseNatArg args "--k" 64
    pure #[("custom", ⟨UInt64.ofNat m, UInt64.ofNat n, UInt64.ofNat k⟩)]
  else
    let selection := LeanBenchmark.MatrixSelection.parse args gemmMatrix.defaultProfile
    match gemmMatrix.select selection with
    | .ok cases => pure (cases.map fun c => (c.id, c.payload))
    | .error msg => throw <| IO.userError msg
  if selected.any (fun c => c.2.m == 0 || c.2.n == 0 || c.2.m % 64 != 0 || c.2.n % 64 != 0) then
    IO.eprintln "[gb10_bf16_gemm] m and n must be positive multiples of 64"
    return 2
  if selected.any (fun c => c.2.k != 64 && (c.2.k < 128 || c.2.k % 64 != 0)) then
    IO.eprintln "[gb10_bf16_gemm] supported k values are 64 or a multiple of 64 greater than or equal to 128"
    return 2
  let mut lines : Array String := #[]
  let mut allOk := true
  for matrixCase in selected do
    let caseLabel := matrixCase.1
    let shape := matrixCase.2
    let (line, ok) ← benchmarkShape cfg caseLabel shape.m shape.n shape.k
    IO.println line
    lines := lines.push line
    allOk := allOk && ok
  match cfg.jsonlOut? with
  | some path => Benchmark.writeJsonl path lines
  | none => pure ()
  pure (if allOk then 0 else 1)

def main (args : List String) : IO UInt32 := do
  if args.contains "--benchmark" then runBenchmark args else
    runWithFixtures args suiteName fixtureSpec generateFixtures runOnce

end Examples.GPU.RunB200Bf16Gemm

def main (args : List String) : IO UInt32 :=
  Examples.GPU.RunB200Bf16Gemm.main args
