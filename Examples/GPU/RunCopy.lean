/-
  Examples/GPU/RunCopy.lean

  End-to-end GPU demo and copy benchmark.
-/
import Examples.GPU.Benchmark
import Tyr.Torch
import Tyr.Typed.Neural
import Tyr.GPU.Kernels.Copy
import Examples.GPU.Parity

namespace Examples.GPU.RunCopy

open torch

variable {device : Device}
open Tyr.GPU.Kernels
open Examples.GPU.Benchmark

def suiteName : String := "copy"

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  seedFixtures suiteName 1001
  let device ← Device.cuda 0
  let input ← Tensor.uniformLike (Tensor.zeros #[1, 1, 64, 64] false device)
  let output ← Tensor.uniformLike input 0 0
  let stream ← torch.cuda_current_stream

  copy64x64Float4.launch input.raw output.raw 1 1 1 512 1 1 0 stream
  torch.cuda_synchronize

  reportTensorComparison "copy.output" input output 1e-7 1e-7

private def runBenchmark (args : List String) : IO UInt32 := do
  if !(← requireCuda suiteName) then return 1
  let device ← Device.cuda 0
  let cfg ← parseConfig args "copy_bench"

  seedFixtures suiteName 1001
  let input ← Tensor.uniformLike (Tensor.zeros #[1, 1, 64, 64] false device)
  let tyrOut ← Tensor.uniformLike input 0 0
  let torchOut ← Tensor.uniformLike input 0 0
  let stream ← torch.cuda_current_stream

  copy64x64Float4.launch input.raw tyrOut.raw 1 1 1 512 1 1 0 stream
  let torchOut ← Tensor.copyInto torchOut input
  torch.cuda_synchronize
  let tyrCheck ← reportTensorComparison "copy.bench.tyr" input tyrOut 1e-7 1e-7
  let torchCheck ← reportTensorComparison "copy.bench.torch" input torchOut 1e-7 1e-7

  let tyrSamples ← timeCudaEvents cfg stream do
    copy64x64Float4.launch input.raw tyrOut.raw 1 1 1 512 1 1 0 stream
  let torchBuffer ← IO.mkRef (some torchOut)
  let torchSamples ← timeCudaEvents cfg stream do
    let some destination ← torchBuffer.swap none
      | throw (IO.userError "Copy benchmark destination is unavailable")
    let updated ← Tensor.copyInto destination input
    torchBuffer.set (some updated)

  let tyrLine := summaryJson cfg "copy_f32_64x64" "tyr" "generated_float4_512x2" tyrSamples tyrCheck
  let torchLine := summaryJson cfg "copy_f32_64x64" "torch_eager" "libtorch_copy_" torchSamples torchCheck
    "copy_on_write_update" false
  IO.println tyrLine
  IO.println torchLine
  match cfg.jsonlOut? with
  | some path => writeJsonl path #[tyrLine, torchLine]
  | none => pure ()
  pure (if tyrCheck && torchCheck then 0 else 1)

def main (args : List String) : IO UInt32 := do
  if args.contains "--benchmark" then runBenchmark args else
    let ok ← runOnce
    pure (if ok then 0 else 1)

end Examples.GPU.RunCopy
