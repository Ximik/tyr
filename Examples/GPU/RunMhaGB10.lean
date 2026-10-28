/- End-to-end GB10/Blackwell MHA validation using the reduced 2-block path. -/
import Tyr.Torch
import Tyr.GPU.Kernels.MhaGB10
import Examples.GPU.IndexedParity
import Examples.GPU.AttentionReference
import Examples.GPU.FixtureRunner
import Examples.GPU.Benchmark

namespace Examples.GPU.RunMhaGB10

open torch

variable {device : Device}
open Tyr.GPU.Kernels

def suiteName : String := "mha_gb10"

def fixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/mha_gb10_128x64"⟩
  names := #[
    "q", "k", "v", "dO",
    "expected_o", "expected_l",
    "expected_dq", "expected_dk", "expected_dv"
  ]
}

def fixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fixtureSpec name

def generateFixtures : IO Unit := do
  if !(← requireCuda suiteName) then
    throw <| IO.userError "CUDA is not available; cannot generate mha_gb10 fixtures."

  IO.FS.createDirAll fixtureSpec.dir
  let device ← Device.cuda 0

  let qf ← Tensor.randn #[1, 1, 128, 64] false device
  let kf ← Tensor.randn #[1, 1, 128, 64] false device
  let vf ← Tensor.randn #[1, 1, 128, 64] false device
  let dOf ← Tensor.randn #[1, 1, 128, 64] false device

  let q := qf.toBFloat16
  let k := kf.toBFloat16
  let v := vf.toBFloat16
  let dO := dOf.toBFloat16

  let expectedOut := Tensor.attentionGQA q k v false

  let (_, lse) := attentionReference q k v
  let expectedL := (Tensor.mulScalar lse (-8.0)).reshapeChecked #[2, 64]
  let (_, expectedDQ, expectedDK, expectedDV) ← attentionTrainingReference q k v dO

  Tensor.saveTensor q (fixtureFile "q").toString
  Tensor.saveTensor k (fixtureFile "k").toString
  Tensor.saveTensor v (fixtureFile "v").toString
  Tensor.saveTensor dO (fixtureFile "dO").toString
  Tensor.saveTensor expectedOut (fixtureFile "expected_o").toString
  Tensor.saveTensor expectedL (fixtureFile "expected_l").toString
  Tensor.saveTensor expectedDQ (fixtureFile "expected_dq").toString
  Tensor.saveTensor expectedDK (fixtureFile "expected_dk").toString
  Tensor.saveTensor expectedDV (fixtureFile "expected_dv").toString

  let outMean := Tensor.item (Tensor.meanAll expectedOut)
  let lMean := Tensor.item (Tensor.meanAll expectedL)
  let dqMean := Tensor.item (Tensor.meanAll expectedDQ)
  IO.println s!"Generated mha_gb10 fixtures in {fixtureSpec.dir} outMean={outMean} lMean={lMean} dqMean={dqMean}"

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  let device ← Device.cuda 0
  if !(← isBlackwellFamily) then
    IO.println s!"[skip] {suiteName}: requires TYR_GPU_FAMILY=BLACKWELL"
    return true

  if !(← fixturesPresent fixtureSpec) then
    generateFixtures

  let stream ← torch.cuda_current_stream

  let q ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "q").toString
  let k ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "k").toString
  let v ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "v").toString
  let dO ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "dO").toString
  let expectedOut ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "expected_o").toString
  let expectedL ← Tensor.loadTensorExact { device := device, shape := #[2, 64], dtype := .Float32 } (fixtureFile "expected_l").toString
  let expectedDQ ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .Float32 } (fixtureFile "expected_dq").toString
  let expectedDK ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .Float32 } (fixtureFile "expected_dk").toString
  let expectedDV ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .Float32 } (fixtureFile "expected_dv").toString

  let outLse ← Tensor.uniformLike q 0 0
  let lseOut ← Tensor.uniformLike (Tensor.zeros #[2, 64] false device) 0 0
  tkFlashAttnGb10Fwd2BlockLse.launch q.raw k.raw v.raw outLse.raw lseOut.raw 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize
  let lFromLse : Tensor { shape := #[2, 64], dtype := .Float32, device := device } := Tensor.mulScalar lseOut (-8.0)

  let out ← Tensor.uniformLike q 0 0
  let lOut : Tensor { shape := #[2, 64], dtype := .Float32, device := device } ← Tensor.uniformLike lseOut 0 0
  tkMhaGb10Fwd2Block.launch q.raw k.raw v.raw out.raw lOut.raw 128 64 1 8 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let dVec ← Tensor.uniformLike lOut 0 0
  tkMhaGb10BwdPrep2Block.launch dO.raw out.raw dVec.raw 128 64 1 8 1 32 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let dQ ← Tensor.uniformLike (Tensor.zeros #[1, 1, 128, 64] false device) 0 0
  let dK ← Tensor.uniformLike dQ 0 0
  let dV ← Tensor.uniformLike dQ 0 0
  tkMhaGb10BwdDQDirect.launch q.raw k.raw v.raw dO.raw lOut.raw dVec.raw dQ.raw 128 64 1 8 1 32 1 1 0 stream
  tkMhaGb10BwdDKDVDirect.launch q.raw k.raw v.raw dO.raw lOut.raw dVec.raw dK.raw dV.raw 128 64 1 8 1 32 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let outCheck := IndexedParity.compareTensors "mha_gb10.forward" expectedOut out 3e-2 3e-2
  let lCheck := IndexedParity.compareTensors "mha_gb10.l" expectedL lOut 3e-2 3e-2
  let lKernelCheck := IndexedParity.compareTensors "mha_gb10.l_vs_lse_kernel" lFromLse lOut 3e-2 3e-2
  let lFixtureCheck := IndexedParity.compareTensors "mha_gb10.lse_kernel_vs_fixture" expectedL lFromLse 3e-2 3e-2
  let dqCheck := IndexedParity.compareTensors "mha_gb10.dq" expectedDQ dQ 3e-2 3e-2
  let dkCheck := IndexedParity.compareTensors "mha_gb10.dk" expectedDK dK 3e-2 3e-2
  let dvCheck := IndexedParity.compareTensors "mha_gb10.dv" expectedDV dV 3e-2 3e-2
  for check in #[outCheck, lCheck, lKernelCheck, lFixtureCheck, dqCheck, dkCheck, dvCheck] do
    logTensorCheck check

  pure (outCheck.ok && lCheck.ok && dqCheck.ok && dkCheck.ok && dvCheck.ok)

private structure ForwardShape where
  batch : Nat
  qHeads : Nat
  seqLen : Nat

private def forwardMatrix : LeanBenchmark.Matrix ForwardShape := {
  defaultProfile := "gb10-realistic"
  cases := #[
    { id := "s64", payload := ⟨1, 1, 64⟩, profiles := #["sequence-sweep"], tags := #["edge", "launch-bound"] },
    { id := "s128", payload := ⟨1, 1, 128⟩, profiles := #["quick", "sequence-sweep"], tags := #["small"] },
    { id := "s256", payload := ⟨1, 1, 256⟩, profiles := #["sequence-sweep"], tags := #["medium"] },
    { id := "s512", payload := ⟨1, 1, 512⟩, profiles := #["sequence-sweep"], tags := #["medium"] },
    { id := "s768", payload := ⟨1, 1, 768⟩, profiles := #["sequence-sweep"], tags := #["training", "micro"] },
    { id := "b1_h16_s768", payload := ⟨1, 16, 768⟩, profiles := #["gb10-realistic", "model-shapes", "batch-sweep"], tags := #["training", "specialized", "multi-head", "latency"] },
    { id := "b2_h16_s768", payload := ⟨2, 16, 768⟩, profiles := #["model-shapes", "batch-sweep"], tags := #["training", "specialized", "multi-head", "throughput"] },
    { id := "b4_h16_s768", payload := ⟨4, 16, 768⟩, profiles := #["model-shapes", "batch-sweep"], tags := #["training", "specialized", "multi-head", "throughput"] },
    { id := "b8_h16_s768", payload := ⟨8, 16, 768⟩, profiles := #["model-shapes", "batch-sweep"], tags := #["training", "specialized", "multi-head", "throughput"] },
    { id := "s1024", payload := ⟨1, 1, 1024⟩, profiles := #["sequence-sweep"], tags := #["large"] },
    { id := "s2048", payload := ⟨1, 1, 2048⟩, profiles := #["sequence-sweep"], tags := #["large"] }
  ]
}

private def benchmarkForwardSeq (cfg : Benchmark.Config) (stream : UInt64)
    (shape : ForwardShape) : IO (String × Bool) := do
  let { batch, qHeads, seqLen } := shape
  let device ← Device.cuda 0
  let qf ← Tensor.randn #[batch, qHeads, seqLen, 64] false device
  let kf ← Tensor.randn #[batch, qHeads, seqLen, 64] false device
  let vf ← Tensor.randn #[batch, qHeads, seqLen, 64] false device
  let q := qf.toBFloat16
  let k := kf.toBFloat16
  let v := vf.toBFloat16
  let expected := Tensor.attentionGQA q k v false
  let (_, lse) := attentionReference q k v
  let expectedL3 := Tensor.mulScalar lse (-8.0)
  let out ← Tensor.uniformLike q 0 0
  let useS768Specialization := seqLen == 768
  let useLongSchedule := seqLen >= 2048
  let rowTileSize : Nat := if useS768Specialization then 32 else if useLongSchedule then 32 else 16
  let queryBlocks := (seqLen + rowTileSize - 1) / rowTileSize
  let launchQueryBlocks := if useS768Specialization then seqLen / (4 * rowTileSize) else queryBlocks
  let lBlocks := if useS768Specialization then batch * qHeads * queryBlocks else queryBlocks
  let lOut : Tensor { shape := #[lBlocks, rowTileSize], dtype := .Float32, device := device } ← Tensor.uniformLike (Tensor.zeros #[lBlocks, rowTileSize] false device) 0 0
  let expectedL : Tensor { shape := #[lBlocks, rowTileSize], dtype := .Float32, device := device } := Tensor.reshapeChecked expectedL3 #[lBlocks, rowTileSize]
  let seqLenArg ← Tyr.GPU.Codegen.launchArg seqLen "seqLen"
  let qHeadsArg ← Tyr.GPU.Codegen.launchArg qHeads "qHeads"
  let launch : IO Unit := if useS768Specialization then
    tkMhaGb10FwdS768D64Warp4.launch q.raw k.raw v.raw out.raw lOut.raw seqLenArg 64 qHeadsArg
      launchQueryBlocks qHeads batch 128 1 1 0 stream
  else if useLongSchedule then
    tkMhaGb10FwdLong.launch q.raw k.raw v.raw out.raw lOut.raw seqLenArg 64 1 queryBlocks 1 128 1 1 0 stream
  else
    tkMhaGb10Fwd2Block.launch q.raw k.raw v.raw out.raw lOut.raw seqLenArg 64 1 queryBlocks 1 128 1 1 0 stream
  launch
  torch.cuda_synchronize
  let check := IndexedParity.compareTensors s!"mha_gb10.bench.forward.s{seqLen}" expected out 3e-2 3e-2
  let lCheck := IndexedParity.compareTensors s!"mha_gb10.bench.l.s{seqLen}" expectedL lOut 3e-2 3e-2
  logTensorCheck check
  logTensorCheck lCheck
  let samples ← Benchmark.timeCudaEvents cfg stream launch
  let line := Benchmark.summaryJson cfg s!"mha_forward_b{batch}_h{qHeads}_s{seqLen}_d64" "tyr"
    (if useS768Specialization then "generated_gb10_specialized_s768_d64_warp4_rows32_splitkv_tma1_exp2"
     else if useLongSchedule then "generated_gb10_specialized_rows32_d64"
     else "generated_gb10_specialized_rows16_d64") samples (check.ok && lCheck.ok)
    "kernel_only" true (some (batch * seqLen).toFloat) (some "tokens")
  pure (line, check.ok && lCheck.ok)

private def benchmarkBackwardS768 (cfg : Benchmark.Config) (stream : UInt64) (batch : Nat)
    : IO (Array String × Bool) := do
  let device ← Device.cuda 0
  let shape := #[batch, 16, 768, 64]
  let qf ← Tensor.randn shape false device
  let kf ← Tensor.randn shape false device
  let vf ← Tensor.randn shape false device
  let dOf ← Tensor.randn shape false device
  let q := qf.toBFloat16
  let k := kf.toBFloat16
  let v := vf.toBFloat16
  let dO := dOf.toBFloat16
  let (_, expectedDQ, expectedDK, expectedDV) ← attentionTrainingReference q k v dO
  let out ← Tensor.uniformLike q 0 0
  let lOut32 : Tensor { shape := #[batch * 384, 32], dtype := .Float32, device := device } ← Tensor.uniformLike (Tensor.zeros #[batch * 384, 32] false device) 0 0
  let launchForward := tkMhaGb10FwdS768D64Warp4.launch q.raw k.raw v.raw out.raw lOut32.raw 768 64 16
    6 16 batch 128 1 1 0 stream
  launchForward
  let lOut : Tensor { shape := #[batch * 768, 16], dtype := .Float32, device := device } := Tensor.reshapeChecked lOut32 #[batch * 768, 16]
  torch.cuda_synchronize
  let dVec ← Tensor.uniformLike lOut 0 0
  let dO32 := dO.toFloat32
  let out32 := out.toFloat32
  let expectedD3 : Tensor { shape := #[batch, 16, 768], dtype := .Float32, device := device } := (Tensor.sumDim (dO32 * out32) 3 false).reshapeChecked #[batch, 16, 768]
  let expectedD : Tensor { shape := #[batch * 768, 16], dtype := .Float32, device := device } := Tensor.reshapeChecked expectedD3 #[batch * 768, 16]
  let dQ ← Tensor.uniformLike q 0 0
  let dK ← Tensor.uniformLike k 0 0
  let dV ← Tensor.uniformLike v 0 0
  let launchDQ := tkMhaGb10BwdDQS768D64.launch q.raw k.raw v.raw dO.raw out.raw lOut.raw dVec.raw dQ.raw 768 64 16
    12 16 batch 128 1 1 0 stream
  let launchDKDV := tkMhaGb10BwdDKDVS768D64.launch q.raw k.raw v.raw dO.raw lOut.raw dVec.raw dK.raw dV.raw 768 64 16
    12 16 batch 128 1 1 0 stream
  let launch := do
    launchDQ
    launchDKDV
  let launchTraining := do
    launchForward
    launch
  launch
  torch.cuda_synchronize
  let dCheck := IndexedParity.compareTensors "mha_gb10.bench.s768.d" expectedD dVec 3e-2 3e-2
  let dqCheck := IndexedParity.compareTensors "mha_gb10.bench.s768.dq" expectedDQ (dQ.toFloat32) 3e-2 3e-2
  let dkCheck := IndexedParity.compareTensors "mha_gb10.bench.s768.dk" expectedDK (dK.toFloat32) 3e-2 3e-2
  let dvCheck := IndexedParity.compareTensors "mha_gb10.bench.s768.dv" expectedDV (dV.toFloat32) 3e-2 3e-2
  for check in #[dCheck, dqCheck, dkCheck, dvCheck] do logTensorCheck check
  let ok := dCheck.ok && dqCheck.ok && dkCheck.ok && dvCheck.ok
  let fullSamples ← Benchmark.timeCudaEvents cfg stream launch
  let trainingSamples ← Benchmark.timeCudaEvents cfg stream launchTraining
  let dqSamples ← Benchmark.timeCudaEvents cfg stream launchDQ
  let dkdvSamples ← Benchmark.timeCudaEvents cfg stream launchDKDV
  let workItems := some (batch * 768).toFloat
  let workUnit := some "tokens"
  let dqLine := Benchmark.summaryJson cfg s!"mha_backward_dq_b{batch}_h16_s768_d64" "tyr"
    "generated_gb10_s768_dq_fusedprep_rvshuffle_frag16_kv32x2_bf16grad" dqSamples ok "kernel_only" true workItems workUnit
  let dkdvLine := Benchmark.summaryJson cfg s!"mha_backward_dkdv_b{batch}_h16_s768_d64" "tyr"
    "generated_gb10_s768_dkdv_warp4_tma_vrow_q48stream_qdo3_bf16grad" dkdvSamples ok
    "kernel_only" true workItems workUnit
  let fullLine := Benchmark.summaryJson cfg s!"mha_backward_b{batch}_h16_s768_d64" "tyr"
    "generated_gb10_s768_dq_fusedprep_rvshuffle_frag16_kv32x2_dkdv_vrow_q48stream_qdo3_bf16grad" fullSamples ok
    "full_generated_kernel_sequence" true workItems workUnit
  let trainingLine := Benchmark.summaryJson cfg s!"mha_training_step_b{batch}_h16_s768_d64" "tyr"
    "generated_gb10_s768_fwd_saved_lse_dq_fusedprep_rvshuffle_frag16_kv32x2_dkdv_q48stream_bf16grad" trainingSamples ok
    "forward_saved_state_plus_backward_kernel_sequence" true workItems workUnit
  pure (#[dqLine, dkdvLine, fullLine, trainingLine], ok)

private def runBenchmark (args : List String) : IO UInt32 := do
  if !(← requireCuda suiteName) then return 1
  let device ← Device.cuda 0
  if !(← isBlackwellFamily) then
    IO.eprintln "[mha_gb10] benchmark requires TYR_GPU_FAMILY=BLACKWELL"
    return 1
  if !(← fixturesPresent fixtureSpec) then generateFixtures
  -- Keep correctness inputs stable across candidate/control benchmark runs.
  torch.manualSeed 0
  let cfg ← Benchmark.parseConfig args "mha_gb10_bench"
  let stream ← torch.cuda_current_stream
  let selection := LeanBenchmark.MatrixSelection.parse args forwardMatrix.defaultProfile
  let selected ← match forwardMatrix.select selection with
    | .ok cases => pure cases
    | .error msg => throw <| IO.userError msg
  let mut forwardLines : Array String := #[]
  let mut forwardOk := true
  for matrixCase in selected do
    let (line, ok) ← benchmarkForwardSeq cfg stream matrixCase.payload
    forwardLines := forwardLines.push line
    forwardOk := forwardOk && ok
  let mut s768BackwardLines : Array String := #[]
  let mut s768BackwardOk := true
  for matrixCase in selected do
    if matrixCase.payload.seqLen == 768 && matrixCase.payload.qHeads == 16 then
      let (lines, ok) ← benchmarkBackwardS768 cfg stream matrixCase.payload.batch
      s768BackwardLines := s768BackwardLines ++ lines
      s768BackwardOk := s768BackwardOk && ok
  let q ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "q").toString
  let k ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "k").toString
  let v ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "v").toString
  let expectedOut ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "expected_o").toString
  let dO ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "dO").toString
  let expectedDQ ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .Float32 } (fixtureFile "expected_dq").toString
  let expectedDK ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .Float32 } (fixtureFile "expected_dk").toString
  let expectedDV ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .Float32 } (fixtureFile "expected_dv").toString
  let out ← Tensor.uniformLike q 0 0
  let lOut : Tensor { shape := #[2, 64], dtype := .Float32, device := device } ← Tensor.uniformLike (Tensor.zeros #[2, 64] false device) 0 0
  let launch := tkMhaGb10Fwd2Block.launch q.raw k.raw v.raw out.raw lOut.raw
    128 64 1 8 1 128 1 1 0 stream
  launch
  torch.cuda_synchronize
  let check := IndexedParity.compareTensors "mha_gb10.bench.forward" expectedOut out 3e-2 3e-2
  logTensorCheck check
  let dVec ← Tensor.uniformLike lOut 0 0
  let dO32 := dO.toFloat32
  let out32 := out.toFloat32
  let expectedD3 : Tensor { shape := #[1, 1, 128], dtype := .Float32, device := device } := (Tensor.sumDim (dO32 * out32) 3 false).reshapeChecked #[1, 1, 128]
  let expectedD : Tensor { shape := #[2, 64], dtype := .Float32, device := device } := Tensor.reshapeChecked expectedD3 #[2, 64]
  let dQ ← Tensor.uniformLike (Tensor.zeros #[1, 1, 128, 64] false device) 0 0
  -- Each mutable gradient buffer is independently allocated through IO.
  let dK ← Tensor.uniformLike dQ 0 0
  let dV ← Tensor.uniformLike dQ 0 0
  let launchBackward := do
    tkMhaGb10BwdPrep2Block.launch dO.raw out.raw dVec.raw 128 64 1 8 1 32 1 1 0 stream
    tkMhaGb10BwdDQDirect.launch q.raw k.raw v.raw dO.raw lOut.raw dVec.raw dQ.raw
      128 64 1 8 1 32 1 1 0 stream
    tkMhaGb10BwdDKDVDirect.launch q.raw k.raw v.raw dO.raw lOut.raw dVec.raw dK.raw dV.raw
      128 64 1 8 1 32 1 1 0 stream
  launchBackward
  torch.cuda_synchronize
  let dCheck := IndexedParity.compareTensors "mha_gb10.bench.d" expectedD dVec 3e-2 3e-2
  let dqCheck := IndexedParity.compareTensors "mha_gb10.bench.dq" expectedDQ dQ 3e-2 3e-2
  let dkCheck := IndexedParity.compareTensors "mha_gb10.bench.dk" expectedDK dK 3e-2 3e-2
  let dvCheck := IndexedParity.compareTensors "mha_gb10.bench.dv" expectedDV dV 3e-2 3e-2
  for gradCheck in #[dCheck, dqCheck, dkCheck, dvCheck] do logTensorCheck gradCheck
  let backwardOk := dCheck.ok && dqCheck.ok && dkCheck.ok && dvCheck.ok
  let backwardSamples ← Benchmark.timeCudaEvents cfg stream launchBackward
  let backwardLine := Benchmark.summaryJson cfg "mha_backward_b1_h1_s128_d64" "tyr"
    "generated_gb10_prep_direct_dq_dkdv" backwardSamples backwardOk
    "full_generated_kernel_sequence" true
  for line in forwardLines do IO.println line
  IO.println backwardLine
  for line in s768BackwardLines do IO.println line
  let allLines := (forwardLines.push backwardLine) ++ s768BackwardLines
  match cfg.jsonlOut? with
  | some path => Benchmark.writeJsonl path allLines
  | none => pure ()
  pure (if forwardOk && check.ok && backwardOk && s768BackwardOk then 0 else 1)

def main (args : List String) : IO UInt32 := do
  if args.contains "--benchmark" then runBenchmark args else
    runWithFixtures args suiteName fixtureSpec generateFixtures runOnce

end Examples.GPU.RunMhaGB10
