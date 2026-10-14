/- End-to-end training + benchmarking demo for ThunderKittens-style `mha_h100`.
   It optimizes q/k/v to match a fixed target attention output and can benchmark
   kernel-based training steps against a PyTorch SDPA autograd baseline. -/
import Tyr.Torch
import Examples.GPU.MhaTrainingReference
import Tyr.GPU.Kernels.MhaH100

namespace Examples.GPU

open torch
open Tyr.GPU.Kernels

private def contractLabel : String := "dq_direct_kv_sweep_store_add"
private def seqLen : Nat := 128
private def headDim : Nat := 64
private def kvTiles : Nat := 2

structure TrainSetup where
  stream : UInt64
  targetOut : MhaTensor
  targetOut32 : MasterTensor
  qInit : MasterTensor
  kInit : MasterTensor
  vInit : MasterTensor

def forwardMha (q k v : MhaTensor) (stream : UInt64) : IO (MhaTensor × LTensor) := do
  let out ← Tensor.uniformLike q 0 0
  let lOut : LTensor ← Tensor.uniformLike (Tensor.zeros #[2, 64] false (Device.CUDA 0)) 0 0
  tkMhaH100Fwd2Block.launch q.raw k.raw v.raw out.raw lOut.raw 128 64 1 2 1 128 1 1 0 stream
  pure (out, lOut)

def backwardMha
    (q k v dO : MhaTensor)
    (lOut : LTensor)
    (out : MhaTensor)
    (stream : UInt64)
    : IO (MasterTensor × MasterTensor × MasterTensor) := do
  let dVec ← Tensor.uniformLike lOut 0 0
  tkMhaH100BwdPrep2Block.launch dO.raw out.raw dVec.raw 128 64 1 2 1 128 1 1 0 stream

  let dQ : MasterTensor ← Tensor.uniformLike (Tensor.zeros #[1, 1, 128, 64] false (Device.CUDA 0)) 0 0
  let dK : MasterTensor ← Tensor.uniformLike (Tensor.zeros #[1, 1, 128, 64] false (Device.CUDA 0)) 0 0
  let dV : MasterTensor ← Tensor.uniformLike (Tensor.zeros #[1, 1, 128, 64] false (Device.CUDA 0)) 0 0
  tkMhaH100Bwd2BlockDq.launch q.raw k.raw v.raw dO.raw lOut.raw dVec.raw dQ.raw 128 64 1 2 1 128 1 1 0 stream
  tkMhaH100Bwd2BlockKvSweep.launch q.raw k.raw v.raw dO.raw lOut.raw dVec.raw dQ.raw dK.raw dV.raw 128 64 1 2 1 128 1 1 0 stream
  pure (dQ, dK, dV)

def mkNoisyInit (targetF : MasterTensor) (noiseScale : Float) (device : Device) : IO MasterTensor := do
  let noise : MasterTensor ← Tensor.randn #[1, 1, 128, 64] false device
  pure (targetF + (noise * noiseScale))

def oneKernelTrainStep
    (qM kM vM : MasterTensor)
    (targetOut : MhaTensor)
    (stream : UInt64)
    (lr : Float)
    : IO (MasterTensor × MasterTensor × MasterTensor × Tensor { shape := #[], dtype := .Float32 }) := do
  let q : MhaTensor := qM.toBFloat16
  let k : MhaTensor := kM.toBFloat16
  let v : MhaTensor := vM.toBFloat16
  let (out, lOut) ← forwardMha q k v stream
  let (lossT, dO) := lossAndGradOut out targetOut
  let (dQ, dK, dV) ← backwardMha q k v dO lOut out stream
  let qM' := sgdUpdate qM dQ lr
  let kM' := sgdUpdate kM dK lr
  let vM' := sgdUpdate vM dV lr
  pure (qM', kM', vM', lossT)

def makeSetup (noiseScale : Float) : IO (Option TrainSetup) := do
  if !(← torch.cuda_is_available) then
    return none

  let device := Device.CUDA 0
  let stream ← torch.cuda_current_stream

  -- Ground-truth tensors defining the training target.
  let qTargetF : MasterTensor ← Tensor.randn #[1, 1, 128, 64] false device
  let kTargetF : MasterTensor ← Tensor.randn #[1, 1, 128, 64] false device
  let vTargetF : MasterTensor ← Tensor.randn #[1, 1, 128, 64] false device
  let qTarget : MhaTensor := qTargetF.toBFloat16
  let kTarget : MhaTensor := kTargetF.toBFloat16
  let vTarget : MhaTensor := vTargetF.toBFloat16
  let (targetOut, _) ← forwardMha qTarget kTarget vTarget stream

  let qInit ← mkNoisyInit qTargetF noiseScale device
  let kInit ← mkNoisyInit kTargetF noiseScale device
  let vInit ← mkNoisyInit vTargetF noiseScale device
  let targetOut32 : MasterTensor := targetOut.toFloat32
  let _ ← torch.cuda_synchronize

  return some {
    stream, targetOut, targetOut32, qInit, kInit, vInit
  }

private def parseArgNat (args : List String) (flag : String) (default : Nat) : Nat := Id.run do
  let rec loop (acc : Nat) (xs : List String) : Nat :=
    match xs with
    | k :: v :: rest =>
        if k == flag then
          let acc :=
            match v.toNat? with
            | some n => n
            | none => acc
          loop acc rest
        else
          loop acc (v :: rest)
    | _ => acc
  loop default args

private def parseFloatLit? (s : String) : Option Float :=
  match s.splitOn "." with
  | [whole] =>
      whole.toNat?.map (·.toFloat)
  | [whole, frac] =>
      match whole.toNat?, frac.toNat? with
      | some w, some f =>
          let denom : Float := (Nat.pow 10 frac.length).toFloat
          some (w.toFloat + f.toFloat / denom)
      | _, _ => none
  | _ => none

private def parseArgFloat (args : List String) (flag : String) (default : Float) : Float := Id.run do
  let rec loop (acc : Float) (xs : List String) : Float :=
    match xs with
    | k :: v :: rest =>
        if k == flag then
          let acc :=
            match parseFloatLit? v with
            | some x => x
            | none => acc
          loop acc rest
        else
          loop acc (v :: rest)
    | _ => acc
  loop default args

def runTrainKernel (steps : Nat) (lr : Float) (noiseScale : Float) (logEvery : Nat) : IO Bool := do
  if steps == 0 then
    IO.eprintln "--steps must be > 0"
    return false

  let some setup ← makeSetup noiseScale
    | IO.eprintln "CUDA is not available on this host."; return false

  let mut qM := setup.qInit
  let mut kM := setup.kInit
  let mut vM := setup.vInit

  let mut firstLoss : Float := 0.0
  let mut lastLoss : Float := 0.0

  for step in [:steps] do
    let (qN, kN, vN, lossT) ← oneKernelTrainStep qM kM vM setup.targetOut setup.stream lr
    qM := qN; kM := kN; vM := vN

    let mustLog := step == 0 || step + 1 == steps || (logEvery > 0 && step % logEvery == 0)
    if mustLog then
      let _ ← torch.cuda_synchronize
      let loss := Tensor.item lossT
      if step == 0 then
        firstLoss := loss
      if step + 1 == steps then
        lastLoss := loss
      IO.println s!"mha_h100_train contract={contractLabel} seq={seqLen} head_dim={headDim} kv_tiles={kvTiles} step={step+1}/{steps} loss={loss}"

  let relImprovement :=
    if firstLoss > 0.0 then (firstLoss - lastLoss) / firstLoss else 0.0
  let ok : Bool := if lastLoss < firstLoss then true else false
  IO.println s!"mha_h100_train contract={contractLabel} seq={seqLen} head_dim={headDim} kv_tiles={kvTiles} init_loss={firstLoss} final_loss={lastLoss} rel_improvement={relImprovement} ok={ok}"
  pure ok

def benchKernel (setup : TrainSetup) (warmup benchIters : Nat) (lr : Float) : IO Float := do
  let mut qM := setup.qInit
  let mut kM := setup.kInit
  let mut vM := setup.vInit

  for _ in [:warmup] do
    let (qN, kN, vN, _) ← oneKernelTrainStep qM kM vM setup.targetOut setup.stream lr
    qM := qN; kM := kN; vM := vN
  let _ ← torch.cuda_synchronize

  -- Reinitialize to avoid timing warmup-mutated state.
  qM := setup.qInit
  kM := setup.kInit
  vM := setup.vInit

  let t0 ← IO.monoNanosNow
  for _ in [:benchIters] do
    let (qN, kN, vN, _) ← oneKernelTrainStep qM kM vM setup.targetOut setup.stream lr
    qM := qN; kM := kN; vM := vN
  let _ ← torch.cuda_synchronize
  let t1 ← IO.monoNanosNow

  let elapsedMs := (t1 - t0).toFloat / 1000000.0
  pure (elapsedMs / benchIters.toFloat)

def benchTorch (setup : TrainSetup) (warmup benchIters : Nat) (lr : Float) : IO Float := do
  let mut qM := setup.qInit
  let mut kM := setup.kInit
  let mut vM := setup.vInit

  for _ in [:warmup] do
    let (qN, kN, vN, _) ← oneTorchTrainStep qM kM vM setup.targetOut32 lr
    qM := qN; kM := kN; vM := vN
  let _ ← torch.cuda_synchronize

  qM := setup.qInit
  kM := setup.kInit
  vM := setup.vInit

  let t0 ← IO.monoNanosNow
  for _ in [:benchIters] do
    let (qN, kN, vN, _) ← oneTorchTrainStep qM kM vM setup.targetOut32 lr
    qM := qN; kM := kN; vM := vN
  let _ ← torch.cuda_synchronize
  let t1 ← IO.monoNanosNow

  let elapsedMs := (t1 - t0).toFloat / 1000000.0
  pure (elapsedMs / benchIters.toFloat)

def runBenchmark
    (warmup benchIters : Nat)
    (lr noiseScale : Float)
    (compareTorch : Bool)
    : IO Bool := do
  if benchIters == 0 then
    IO.eprintln "--bench-iters must be > 0"
    return false

  let some setup ← makeSetup noiseScale
    | IO.eprintln "CUDA is not available on this host."; return false

  let kernelMs ← benchKernel setup warmup benchIters lr
  let kernelStepsPerSec := 1000.0 / kernelMs
  IO.println s!"mha_h100_bench contract={contractLabel} seq={seqLen} head_dim={headDim} kv_tiles={kvTiles} kernel_ms_per_step={kernelMs} kernel_steps_per_sec={kernelStepsPerSec} warmup={warmup} bench_iters={benchIters}"

  if compareTorch then
    let torchMs ← benchTorch setup warmup benchIters lr
    let torchStepsPerSec := 1000.0 / torchMs
    let kernelSpeedupVsTorch := torchMs / kernelMs
    IO.println s!"mha_h100_bench contract={contractLabel} seq={seqLen} head_dim={headDim} kv_tiles={kvTiles} torch_ms_per_step={torchMs} torch_steps_per_sec={torchStepsPerSec} kernel_speedup_vs_torch={kernelSpeedupVsTorch}"

  pure true

def main (args : List String) : IO UInt32 := do
  -- Keep compatibility with scripts/gpu/run_e2e_kernel.sh flow.
  if args.contains "--gen-only" then
    return 0

  let steps := parseArgNat args "--steps" 30
  let lr := parseArgFloat args "--lr" 200.0
  let noiseScale := parseArgFloat args "--noise" 0.5
  let logEvery := parseArgNat args "--log-every" 5
  let warmup := parseArgNat args "--warmup" 10
  let benchIters := parseArgNat args "--bench-iters" 200
  let benchmark := args.contains "--benchmark"
  let compareTorch := !args.contains "--no-compare"

  let ok ←
    if benchmark then
      runBenchmark warmup benchIters lr noiseScale compareTorch
    else
      runTrainKernel steps lr noiseScale logEvery
  pure (if ok then 0 else 1)

end Examples.GPU

def main : List String → IO UInt32 := Examples.GPU.main
