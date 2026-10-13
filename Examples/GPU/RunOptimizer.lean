import Tyr.Torch
import Tyr.GPU.Kernels.Optimizer
import Examples.GPU.IndexedParity
import Tyr.Typed.Neural
import Examples.GPU.OptimizerReference
import Examples.GPU.Benchmark

namespace Examples.GPU.RunOptimizer

open torch
open Tyr.GPU.Kernels.Optimizer

private structure OptimizerShape where
  elements : UInt64

private def optimizerMatrix : LeanBenchmark.Matrix OptimizerShape := {
  defaultProfile := "qwen3tts-talker-primary"
  cases := #[
    { id := "qkv_weight_n1048576", payload := ⟨1048576⟩,
      profiles := #["qwen3tts-talker-primary", "training"],
      tags := #["attention", "matrix", "primary"] },
    { id := "kv_weight_n131072", payload := ⟨131072⟩,
      profiles := #["qwen3tts-talker-primary", "training"],
      tags := #["attention", "gqa", "small"] },
    { id := "mlp_weight_n2097152", payload := ⟨2097152⟩,
      profiles := #["qwen3tts-talker-primary", "training"],
      tags := #["mlp", "matrix", "primary"] },
    { id := "embedding_n3145728", payload := ⟨3145728⟩,
      profiles := #["qwen3tts-talker-primary", "training"],
      tags := #["embedding", "adam", "large"] },
    { id := "micro_n4096", payload := ⟨4096⟩,
      profiles := #["micro"], tags := #["regression"] }
  ]
}

private def benchmarkShape (cfg : Benchmark.Config) (stream : UInt64)
    (caseLabel : String) (elements : UInt64) : IO (String × Bool) := do
  let device := Device.CUDA 0
  let master : DTensor #[elements] .Float32 ← Tensor.randn #[elements] false device
  let masterOriginal : DTensor #[elements] .Float32 ← Tensor.randn #[elements] false device
  Tensor.copyInto masterOriginal master
  let gradF : DTensor #[elements] .Float32 ← Tensor.randn #[elements] false device
  let grad : DTensor #[elements] .BFloat16 := gradF.toBFloat16
  let gradBackupF : DTensor #[elements] .Float32 ← Tensor.randn #[elements] false device
  let gradBackup : DTensor #[elements] .BFloat16 := gradBackupF.toBFloat16
  Tensor.copyInto gradBackup grad
  let model : DTensor #[elements] .BFloat16 := master.toBFloat16
  let moment1 : DTensor #[elements] .Float32 ← Tensor.randn #[elements] false device
  let moment2 : DTensor #[elements] .Float32 ← Tensor.randn #[elements] false device
  let zeroMoment : DTensor #[elements] .Float32 := Tensor.zerosLike master
  Tensor.copyInto moment1 zeroMoment
  Tensor.copyInto moment2 zeroMoment
  let learningRate : Float := 3.0e-4
  let beta1 : Float := 0.9
  let beta2 : Float := 0.95
  let epsilon : Float := 1.0e-8
  let weightDecay : Float := 0.1
  let invBias1 : Float := 1.0 / (1.0 - beta1)
  let invBias2 : Float := 1.0 / (1.0 - beta2)
  let (expectedMaster, expectedModel, expectedM, expectedV) :=
    referenceUpdate masterOriginal grad moment1 moment2 learningRate beta1 beta2 epsilon
      weightDecay invBias1 invBias2
  let blocks := (elements + 255) / 256
  let launch : IO Unit :=
    adamWMasterBf16.launch master.raw model.raw grad.raw moment1.raw moment2.raw elements
      learningRate.toFloat32 beta1.toFloat32 beta2.toFloat32 epsilon.toFloat32
      weightDecay.toFloat32 invBias1.toFloat32 invBias2.toFloat32
      blocks 1 1 256 1 1 0 stream

  launch
  torch.cuda_synchronize
  let preMaster := IndexedParity.compareTensors s!"optimizer.{caseLabel}.master" expectedMaster master 2e-5 2e-5
  let preModel := IndexedParity.compareTensors s!"optimizer.{caseLabel}.model" expectedModel model 2e-3 2e-3
  let preM := IndexedParity.compareTensors s!"optimizer.{caseLabel}.moment1" expectedM moment1 2e-5 2e-5
  let preV := IndexedParity.compareTensors s!"optimizer.{caseLabel}.moment2" expectedV moment2 2e-5 2e-5
  for check in #[preMaster, preModel, preM, preV] do logTensorCheck check

  let zeroMaster : DTensor #[elements] .Float32 ← Tensor.randn #[elements] false device
  Tensor.copyInto master zeroMaster
  Tensor.copyInto model (zeroMaster.toBFloat16)
  Tensor.copyInto grad (Tensor.zerosLike grad)
  Tensor.copyInto moment1 zeroMoment
  Tensor.copyInto moment2 zeroMoment
  launch
  torch.cuda_synchronize
  let zeroExpectedMaster : DTensor #[elements] .Float32 :=
    zeroMaster * (1.0 - learningRate * weightDecay)
  let zeroExpectedModel := zeroExpectedMaster.toBFloat16
  let zeroMasterCheck :=
    IndexedParity.compareTensors s!"optimizer.{caseLabel}.zero_grad.master" zeroExpectedMaster master 2e-5 2e-5
  let zeroModelCheck :=
    IndexedParity.compareTensors s!"optimizer.{caseLabel}.zero_grad.model" zeroExpectedModel model 2e-3 2e-3
  let zeroMCheck :=
    IndexedParity.compareTensors s!"optimizer.{caseLabel}.zero_grad.moment1" zeroMoment moment1 0 0
  let zeroVCheck :=
    IndexedParity.compareTensors s!"optimizer.{caseLabel}.zero_grad.moment2" zeroMoment moment2 0 0
  for check in #[zeroMasterCheck, zeroModelCheck, zeroMCheck, zeroVCheck] do
    logTensorCheck check

  let resetMaster : DTensor #[elements] .Float32 ← Tensor.randn #[elements] false device
  Tensor.copyInto master resetMaster
  Tensor.copyInto model (resetMaster.toBFloat16)
  Tensor.copyInto grad gradBackup
  Tensor.copyInto moment1 zeroMoment
  Tensor.copyInto moment2 zeroMoment
  let samples ← Benchmark.timeCudaEvents cfg stream launch
  let finiteMaster := IndexedParity.compareTensors s!"optimizer.{caseLabel}.master.self" master master 0 0
  let finiteModel := IndexedParity.compareTensors s!"optimizer.{caseLabel}.model.self" model model 0 0
  let correct := preMaster.ok && preModel.ok && preM.ok && preV.ok &&
    zeroMasterCheck.ok && zeroModelCheck.ok && zeroMCheck.ok && zeroVCheck.ok &&
    finiteMaster.ok && finiteModel.ok
  let line := Benchmark.summaryJson cfg s!"adamw_training_{caseLabel}" "tyr"
    "generated_fused_fp32_master_moments_bf16_grad_model" samples correct
    "in_place_full_optimizer_update" true (some elements.toFloat) (some "parameters")
  pure (line, correct)

private def runBenchmark (args : List String) : IO UInt32 := do
  if !(← requireCuda "optimizer") then return 1
  let cfg ← Benchmark.parseConfig args "optimizer_bench"
  torch.manualSeed 0
  let selection := LeanBenchmark.MatrixSelection.parse args optimizerMatrix.defaultProfile
  let selected ← match optimizerMatrix.select selection with
    | .ok cases => pure cases
    | .error msg => throw <| IO.userError msg
  let stream ← torch.cuda_current_stream
  let mut lines : Array String := #[]
  let mut allOk := true
  for matrixCase in selected do
    let (line, ok) ← benchmarkShape cfg stream matrixCase.id matrixCase.payload.elements
    lines := lines.push line
    allOk := allOk && ok
  for line in lines do IO.println line
  match cfg.jsonlOut? with
  | some path => Benchmark.writeJsonl path lines
  | none => pure ()
  pure (if allOk then 0 else 1)

def main (args : List String) : IO UInt32 := runBenchmark args

end Examples.GPU.RunOptimizer

def main (args : List String) : IO UInt32 :=
  Examples.GPU.RunOptimizer.main args
