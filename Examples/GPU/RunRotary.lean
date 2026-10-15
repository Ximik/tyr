/- End-to-end rotary validation:
   generate deterministic input/reference tensors, launch the kernel, compare outputs. -/
import Tyr.Torch
import Tyr.GPU.Kernels.Rotary
import Examples.GPU.IndexedParity
import Tyr.Typed.Neural
import Examples.GPU.FixtureRunner
import Examples.GPU.Benchmark

namespace Examples.GPU.RunRotary

open torch
open Tyr.GPU.Kernels.Rotary

def suiteName : String := "rotary"

def fixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/rotary64"⟩
  names := #["x", "sin", "cos", "expected"]
}

def fixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fixtureSpec name

def generateFixtures : IO Unit := do
  if !(← requireCuda suiteName) then
    throw <| IO.userError "CUDA is not available; cannot generate rotary fixtures."

  IO.FS.createDirAll fixtureSpec.dir

  let device := Device.CUDA 0
  let x ← Tensor.uniformLike (Tensor.zeros #[64, 64] false device)

  let (cosCpu, sinCpu) := Tensor.rotaryFreqs (64 : UInt64) (64 : UInt64) 10000.0
  let cos : Tensor { shape := #[64, 32], dtype := .Float32, device := (.CUDA 0) } := cosCpu.to device
  let sin : Tensor { shape := #[64, 32], dtype := .Float32, device := (.CUDA 0) } := sinCpu.to device

  let x4 : Tensor { shape := #[1, 64, 1, 64], dtype := .Float32, device := (.CUDA 0) } := Tensor.reshapeChecked x #[1, 64, 1, 64]
  let expected4 := Tensor.applyRotary x4 cos sin
  let expected : Tensor { shape := #[64, 64], dtype := .Float32, device := (.CUDA 0) } := Tensor.reshapeChecked expected4 #[64, 64]

  Tensor.saveTensor x (fixtureFile "x").toString
  Tensor.saveTensor sin (fixtureFile "sin").toString
  Tensor.saveTensor cos (fixtureFile "cos").toString
  Tensor.saveTensor expected (fixtureFile "expected").toString

  let xMean := Tensor.item (Tensor.meanAll x)
  let eMean := Tensor.item (Tensor.meanAll expected)
  IO.println s!"Generated rotary fixtures in {fixtureSpec.dir} xMean={xMean} expectedMean={eMean}"

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  if !(← fixturesPresent fixtureSpec) then
    generateFixtures

  let x ← Tensor.loadTensorExact { device := .CUDA 0, shape := #[64, 64], dtype := .Float32 } (fixtureFile "x").toString
  let sin ← Tensor.loadTensorExact { device := .CUDA 0, shape := #[64, 32], dtype := .Float32 } (fixtureFile "sin").toString
  let cos ← Tensor.loadTensorExact { device := .CUDA 0, shape := #[64, 32], dtype := .Float32 } (fixtureFile "cos").toString
  let expected ← Tensor.loadTensorExact { device := .CUDA 0, shape := #[64, 64], dtype := .Float32 } (fixtureFile "expected").toString

  let output := Tensor.zerosLike x
  let stream ← torch.cuda_current_stream

  rotaryFwd64x64Direct.launch x.raw sin.raw cos.raw output.raw 1 1 1 512 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let check := IndexedParity.compareTensors "rotary.output" expected output 1e-4 1e-4
  let outMean := Tensor.item (Tensor.meanAll output)
  let expMean := Tensor.item (Tensor.meanAll expected)
  logTensorCheck check
  IO.println s!"rotary output_mean={outMean} expected_mean={expMean}"
  pure check.ok

private structure TrainingShape where
  batch : UInt64
  seqLen : UInt64
  qHeads : UInt64
  kvHeads : UInt64

private def trainingMatrix : LeanBenchmark.Matrix TrainingShape := {
  defaultProfile := "training"
  cases := #[
    { id := "b1_h16_kv2_s768", payload := ⟨1, 768, 16, 2⟩,
      profiles := #["training", "batch-sweep", "gqa", "qwen3tts-talker"],
      tags := #["training", "gqa", "latency"] },
    { id := "b2_h16_kv2_s768", payload := ⟨2, 768, 16, 2⟩,
      profiles := #["training", "batch-sweep", "gqa", "qwen3tts-talker"],
      tags := #["training", "gqa", "throughput"] },
    { id := "b4_h16_kv2_s768", payload := ⟨4, 768, 16, 2⟩,
      profiles := #["training", "batch-sweep", "gqa", "qwen3tts-talker"],
      tags := #["training", "gqa", "throughput", "primary"] },
    { id := "b8_h16_kv2_s768", payload := ⟨8, 768, 16, 2⟩,
      profiles := #["training", "batch-sweep", "gqa", "qwen3tts-talker"],
      tags := #["training", "gqa", "throughput", "saturation"] },
    { id := "b1_h16_kv16_s768", payload := ⟨1, 768, 16, 16⟩,
      profiles := #["equal-head", "mha-compat"], tags := #["training", "latency"] },
    { id := "b2_h16_kv16_s768", payload := ⟨2, 768, 16, 16⟩,
      profiles := #["equal-head", "mha-compat"], tags := #["training", "throughput"] },
    { id := "b4_h16_kv16_s768", payload := ⟨4, 768, 16, 16⟩,
      profiles := #["equal-head", "mha-compat"],
      tags := #["training", "throughput", "primary"] },
    { id := "b8_h16_kv16_s768", payload := ⟨8, 768, 16, 16⟩,
      profiles := #["equal-head", "mha-compat"],
      tags := #["training", "throughput", "saturation"] }
  ]
}

/-- Reference forward and adjoint rotations for separate query and key head counts. -/
def referenceTraining {device : Device} {batch seqLen qHeads kvHeads : UInt64} {dtype : DType}
    (q : Tensor { shape := #[batch, seqLen, qHeads, 64], dtype := dtype, device := device })
    (k : Tensor { shape := #[batch, seqLen, kvHeads, 64], dtype := dtype, device := device })
    (gradQ : Tensor { shape := #[batch, seqLen, qHeads, 64], dtype := dtype, device := device })
    (gradK : Tensor { shape := #[batch, seqLen, kvHeads, 64], dtype := dtype, device := device })
    (cos sin : Tensor { shape := #[seqLen, 32], dtype := .Float32, device := device }) :=
  let negSin := Tensor.Preserving.mulScalar sin (-1.0)
  (Tensor.applyRotary q cos sin, Tensor.applyRotary k cos sin,
    Tensor.applyRotary gradQ cos negSin, Tensor.applyRotary gradK cos negSin)

private def benchmarkTrainingShape (cfg : Benchmark.Config) (caseLabel : String)
    (shape : TrainingShape) : IO (Array String × Bool) := do
  let { batch, seqLen, qHeads, kvHeads } := shape
  let device := Device.CUDA 0
  let qf ← Tensor.randn #[batch, seqLen, qHeads, 64] false device
  let kf ← Tensor.randn #[batch, seqLen, kvHeads, 64] false device
  let gradQf ← Tensor.randn #[batch, seqLen, qHeads, 64] false device
  let gradKf ← Tensor.randn #[batch, seqLen, kvHeads, 64] false device
  let q : Tensor { shape := #[batch, seqLen, qHeads, 64], dtype := .BFloat16, device := (.CUDA 0) } := qf.toBFloat16
  let k : Tensor { shape := #[batch, seqLen, kvHeads, 64], dtype := .BFloat16, device := (.CUDA 0) } := kf.toBFloat16
  let gradQ : Tensor { shape := #[batch, seqLen, qHeads, 64], dtype := .BFloat16, device := (.CUDA 0) } := gradQf.toBFloat16
  let gradK : Tensor { shape := #[batch, seqLen, kvHeads, 64], dtype := .BFloat16, device := (.CUDA 0) } := gradKf.toBFloat16
  let (cos, sin) := Tensor.rotaryFreqs seqLen 64 10000.0 device
  let (expectedQ, expectedK, expectedGradQ, expectedGradK) :=
    referenceTraining q k gradQ gradK cos sin
  let qOut := Tensor.Preserving.mulScalar q 0.0
  let kOut := Tensor.Preserving.mulScalar k 0.0
  let gradQOut := Tensor.Preserving.mulScalar gradQ 0.0
  let gradKOut := Tensor.Preserving.mulScalar gradK 0.0
  let qRows := batch * seqLen * qHeads
  let kRows := batch * seqLen * kvHeads
  let totalPairs := (if qRows > kRows then qRows else kRows) * 32
  let gridX := (totalPairs + 1023) / 1024
  let stream ← torch.cuda_current_stream
  let useTalkerSpecialization := qHeads == 16 && kvHeads == 2 && seqLen == 768
  let launchForward : IO Unit :=
    if useTalkerSpecialization then
      rotaryFwdQwen3TtsTalkerD64Bf16Direct.launch q.raw k.raw sin.raw cos.raw qOut.raw kOut.raw
        qRows kRows gridX 1 1 256 1 1 0 stream
    else
      rotaryFwdQKD64Bf16Direct.launch q.raw k.raw sin.raw cos.raw qOut.raw kOut.raw
        qRows kRows qHeads kvHeads seqLen gridX 1 1 256 1 1 0 stream
  let launchBackward : IO Unit :=
    if useTalkerSpecialization then
      rotaryBwdQwen3TtsTalkerD64Bf16Direct.launch gradQ.raw gradK.raw sin.raw cos.raw
        gradQOut.raw gradKOut.raw qRows kRows gridX 1 1 256 1 1 0 stream
    else
      rotaryBwdQKD64Bf16Direct.launch gradQ.raw gradK.raw sin.raw cos.raw gradQOut.raw gradKOut.raw
        qRows kRows qHeads kvHeads seqLen gridX 1 1 256 1 1 0 stream
  launchForward
  launchBackward
  torch.cuda_synchronize
  let qCheck := IndexedParity.compareTensors s!"rotary.{caseLabel}.q" expectedQ qOut 3e-2 3e-2
  let kCheck := IndexedParity.compareTensors s!"rotary.{caseLabel}.k" expectedK kOut 3e-2 3e-2
  let gradQCheck := IndexedParity.compareTensors s!"rotary.{caseLabel}.dq" expectedGradQ gradQOut 3e-2 3e-2
  let gradKCheck := IndexedParity.compareTensors s!"rotary.{caseLabel}.dk" expectedGradK gradKOut 3e-2 3e-2
  for check in #[qCheck, kCheck, gradQCheck, gradKCheck] do logTensorCheck check
  let fwdSamples ← Benchmark.timeCudaEvents cfg stream launchForward
  let bwdSamples ← Benchmark.timeCudaEvents cfg stream launchBackward
  let trainingSamples ← Benchmark.timeCudaEvents cfg stream do
    launchForward
    launchBackward
  let postQ := IndexedParity.compareTensors s!"rotary.{caseLabel}.q.post" expectedQ qOut 3e-2 3e-2
  let postK := IndexedParity.compareTensors s!"rotary.{caseLabel}.k.post" expectedK kOut 3e-2 3e-2
  let postGradQ := IndexedParity.compareTensors s!"rotary.{caseLabel}.dq.post" expectedGradQ gradQOut 3e-2 3e-2
  let postGradK := IndexedParity.compareTensors s!"rotary.{caseLabel}.dk.post" expectedGradK gradKOut 3e-2 3e-2
  for check in #[postQ, postK, postGradQ, postGradK] do logTensorCheck check
  let correct := qCheck.ok && kCheck.ok && gradQCheck.ok && gradKCheck.ok &&
    postQ.ok && postK.ok && postGradQ.ok && postGradK.ok
  let route :=
    if useTalkerSpecialization then
      "generated_bf16_d64_qh16_kvh2_s768_u32_gridstride4"
    else
      "generated_bf16_d64_qk_pair_fused_gridstride4"
  let work := (qRows + kRows).toFloat * 32.0
  let fwdLine := Benchmark.summaryJson cfg s!"rotary_qk_fwd_{caseLabel}" "tyr" route
    fwdSamples correct "kernel_only" true (some work) (some "pairs")
  let bwdLine := Benchmark.summaryJson cfg s!"rotary_dqdk_bwd_{caseLabel}" "tyr" route
    bwdSamples correct "kernel_only" true (some work) (some "pairs")
  let trainingLine := Benchmark.summaryJson cfg s!"rotary_training_{caseLabel}" "tyr"
    s!"{route}_fwd_plus_bwd" trainingSamples correct "full_training_operation" true
    (some (2.0 * work)) (some "pairs")
  pure (#[fwdLine, bwdLine, trainingLine], correct)

private def runBenchmark (args : List String) : IO UInt32 := do
  if !(← requireCuda suiteName) then return 1
  let cfg ← Benchmark.parseConfig args "rotary_bench"
  torch.manualSeed 0
  let selection := LeanBenchmark.MatrixSelection.parse args trainingMatrix.defaultProfile
  let selected ← match trainingMatrix.select selection with
    | .ok cases => pure cases
    | .error msg => throw <| IO.userError msg
  let mut lines : Array String := #[]
  let mut allOk := true
  for matrixCase in selected do
    let (caseLines, ok) ← benchmarkTrainingShape cfg matrixCase.id matrixCase.payload
    for line in caseLines do
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

end Examples.GPU.RunRotary
