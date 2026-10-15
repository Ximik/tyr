/- End-to-end ThunderKittens-style layernorm validation. -/
import Tyr.Torch
import Tyr.GPU.Kernels.FusedLayerNorm
import Examples.GPU.IndexedParity
import Tyr.Typed.Neural
import Examples.GPU.FixtureRunner
import Examples.GPU.Benchmark

namespace Examples.GPU.RunLayerNorm

open torch

variable {device : Device}
open Tyr.GPU.Kernels

private def launchBlockX : UInt64 := 32

def suiteName : String := "layernorm"

def fixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/layernorm64x1024"⟩
  names := #["x", "residual", "weight", "bias", "expected_out", "expected_resid"]
}

def fixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fixtureSpec name

abbrev LayerNormInput (dtype : DType) (device : Device := .CPU) := Tensor { shape := #[1, 64, 1024], dtype := dtype, device := device }
abbrev LayerNormWeight (dtype : DType) (device : Device := .CPU) := Tensor { shape := #[1024], dtype := dtype, device := device }

structure LayerNormInputs (dtype : DType) (device : Device := .CPU) where
  x : LayerNormInput dtype device
  residual : LayerNormInput dtype device
  weight : LayerNormWeight dtype device
  bias : LayerNormWeight dtype device

def layerNormReference
    (inputs : LayerNormInputs .Float32 device)
    : LayerNormInput .Float32 device × LayerNormInput .Float32 device :=
  let eps : Float := 1.0e-5
  let expectedResid := inputs.x + inputs.residual
  let expectedOut := Tensor.layerNorm expectedResid inputs.weight inputs.bias eps
  (expectedOut, expectedResid)

def layerNormReferenceBFloat16
    (inputs : LayerNormInputs .BFloat16 device)
    : LayerNormInput .BFloat16 device × LayerNormInput .BFloat16 device :=
  let eps : Float := 1.0e-5
  let expectedResid := inputs.x + inputs.residual
  let expectedOut32 : LayerNormInput .Float32 device :=
    Tensor.layerNorm
      (expectedResid.toFloat32)
      (inputs.weight.toFloat32)
      (inputs.bias.toFloat32)
      eps
  let expectedOut := expectedOut32.toBFloat16
  (expectedOut, expectedResid)

def asBFloat16Inputs (inputs : LayerNormInputs .Float32 device) : LayerNormInputs .BFloat16 device := {
  x := inputs.x.toBFloat16
  residual := inputs.residual.toBFloat16
  weight := inputs.weight.toBFloat16
  bias := inputs.bias.toBFloat16
}

private def logInputDTypes {dtype : DType} (label : String) (inputs : LayerNormInputs dtype device) : IO Unit := do
  IO.println <|
    s!"[{suiteName}:{label}] input_dtypes x={inputs.x.dtype} residual={inputs.residual.dtype} " ++
    s!"weight={inputs.weight.dtype} bias={inputs.bias.dtype}"

private def compareResults {dtype : DType}
    (mode : String)
    (expectedOut : LayerNormInput dtype device)
    (expectedResid : LayerNormInput dtype device)
    (out : LayerNormInput dtype device)
    (outResid : LayerNormInput dtype device)
    (rtol atol : Float)
    : IO Bool := do
  let outCheck := IndexedParity.compareTensors s!"layernorm.{mode}.output" (expectedOut.toFloat32) (out.toFloat32) rtol atol
  let residCheck := IndexedParity.compareTensors s!"layernorm.{mode}.residual" (expectedResid.toFloat32) (outResid.toFloat32) rtol atol
  logTensorCheck outCheck
  logTensorCheck residCheck
  pure (outCheck.ok && residCheck.ok)

private def runInputs {dtype : DType} (label : String) (inputs : LayerNormInputs dtype (.CUDA 0)) : IO Bool := do
  logInputDTypes label inputs
  let stream ← torch.cuda_current_stream
  let blackwell ← isBlackwellFamily
  match dtype, inputs with
  | .Float32, inputs => do
      let (expectedOut, expectedResid) := layerNormReference inputs
      let out ← Tensor.uniformLike inputs.x
      let outResid ← Tensor.uniformLike inputs.residual
      fusedLayerNormResidual64x1024F32Direct.launch
        inputs.x.raw inputs.residual.raw inputs.weight.raw inputs.bias.raw out.raw outResid.raw
        64 1 1 256 1 1 0 stream
      torch.cuda_synchronize
      compareResults (if blackwell then "blackwell.f32" else "f32") expectedOut expectedResid out outResid 5e-3 5e-3
  | .BFloat16, inputs => do
      let (expectedOut, expectedResid) := layerNormReferenceBFloat16 inputs
      let out ← Tensor.uniformLike inputs.x
      let outResid ← Tensor.uniformLike inputs.residual
      fusedLayerNormResidual64x1024Bf16Direct.launch
        inputs.x.raw inputs.residual.raw inputs.weight.raw inputs.bias.raw out.raw outResid.raw
        64 1 1 256 1 1 0 stream
      torch.cuda_synchronize
      compareResults (if blackwell then "blackwell.bf16" else "bf16") expectedOut expectedResid out outResid 2e-2 2e-2
  | dtype, _ =>
      IO.eprintln s!"[{suiteName}:{label}] unsupported_dtype dtype={dtype}"
      pure false

private def randomInputs (dtype : DType) : IO (LayerNormInputs dtype (.CUDA 0)) := do
  let device := Device.CUDA 0
  let base : LayerNormInputs .Float32 (.CUDA 0) := {
    x := ← Tensor.uniformLike (Tensor.zeros #[1, 64, 1024] false device)
    residual := ← Tensor.uniformLike (Tensor.zeros #[1, 64, 1024] false device)
    weight := ← Tensor.uniformLike (Tensor.zeros #[1024] false device)
    bias := ← Tensor.uniformLike (Tensor.zeros #[1024] false device)
  }
  match dtype with
  | .Float32 => pure base
  | .BFloat16 => pure (asBFloat16Inputs base)
  | _ => throw <| IO.userError s!"unsupported random layernorm dtype {dtype}"

def generateFixtures : IO Unit := do
  if !(← requireCuda suiteName) then
    throw <| IO.userError "CUDA is not available; cannot generate layernorm fixtures."

  IO.FS.createDirAll fixtureSpec.dir
  let inputs ← randomInputs .Float32
  let vendoredInputs := asBFloat16Inputs inputs
  let (expectedOut, expectedResid) := layerNormReferenceBFloat16 vendoredInputs

  Tensor.saveTensor inputs.x (fixtureFile "x").toString
  Tensor.saveTensor inputs.residual (fixtureFile "residual").toString
  Tensor.saveTensor inputs.weight (fixtureFile "weight").toString
  Tensor.saveTensor inputs.bias (fixtureFile "bias").toString
  Tensor.saveTensor expectedOut (fixtureFile "expected_out").toString
  Tensor.saveTensor expectedResid (fixtureFile "expected_resid").toString

  let outMean := Tensor.item (Tensor.meanAll expectedOut)
  let residMean := Tensor.item (Tensor.meanAll expectedResid)
  IO.println <|
    s!"Generated layernorm fixtures in {fixtureSpec.dir} " ++
    s!"input_dtype={inputs.x.dtype} vendored_ref_dtype={vendoredInputs.x.dtype} " ++
    s!"outMean={outMean} residMean={residMean}"

def runFloat32Once : IO Bool := do
  if !(← requireCuda suiteName) then
    return false
  seedFixtures s!"{suiteName}.f32" 0
  runInputs "random_f32" (← randomInputs .Float32)

def runBFloat16Once : IO Bool := do
  if !(← requireCuda suiteName) then
    return false
  seedFixtures s!"{suiteName}.bf16" 0
  runInputs "random_bf16" (← randomInputs .BFloat16)

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  if !(← fixturesPresent fixtureSpec) then
    generateFixtures

  let x ← Tensor.loadTensorExact { device := .CUDA 0, shape := #[1, 64, 1024], dtype := .Unknown "dynamic" }
    (fixtureFile "x").toString
  let dtype := x.actualSpec.dtype
  let inputs : LayerNormInputs dtype (.CUDA 0) := {
    x := x.castDType dtype
    residual := ← Tensor.loadTensorExact { device := .CUDA 0, shape := #[1, 64, 1024], dtype } (fixtureFile "residual").toString
    weight := ← Tensor.loadTensorExact { device := .CUDA 0, shape := #[1024], dtype } (fixtureFile "weight").toString
    bias := ← Tensor.loadTensorExact { device := .CUDA 0, shape := #[1024], dtype } (fixtureFile "bias").toString
  }
  runInputs "fixtures" inputs


private def benchmarkFloat32 (cfg : Benchmark.Config) (stream : UInt64)
    (_blackwell : Bool) : IO (String × Bool) := do
  let inputs ← randomInputs .Float32
  let (expectedOut, expectedResid) := layerNormReference inputs
  let out ← Tensor.uniformLike inputs.x
  let outResid ← Tensor.uniformLike inputs.residual
  let launch := do
    fusedLayerNormResidual64x1024F32Direct.launch
      inputs.x.raw inputs.residual.raw inputs.weight.raw inputs.bias.raw out.raw outResid.raw
      64 1 1 256 1 1 0 stream
  launch
  torch.cuda_synchronize
  let correct ← compareResults "bench.f32" expectedOut expectedResid out outResid 5e-3 5e-3
  let samples ← Benchmark.timeCudaEvents cfg stream launch
  let route := "generated_direct_row_cta_256x4"
  pure (Benchmark.summaryJson cfg "layernorm_residual_f32_64x1024" "tyr" route samples correct, correct)

private def benchmarkBFloat16 (cfg : Benchmark.Config) (stream : UInt64)
    (_blackwell : Bool) : IO (String × Bool) := do
  let inputs ← randomInputs .BFloat16
  let (expectedOut, expectedResid) := layerNormReferenceBFloat16 inputs
  let out ← Tensor.uniformLike inputs.x
  let outResid ← Tensor.uniformLike inputs.residual
  let launch := do
    fusedLayerNormResidual64x1024Bf16Direct.launch
      inputs.x.raw inputs.residual.raw inputs.weight.raw inputs.bias.raw out.raw outResid.raw
      64 1 1 256 1 1 0 stream
  launch
  torch.cuda_synchronize
  let correct ← compareResults "bench.bf16" expectedOut expectedResid out outResid 2e-2 2e-2
  let samples ← Benchmark.timeCudaEvents cfg stream launch
  let route := "generated_direct_row_cta_256x4"
  pure (Benchmark.summaryJson cfg "layernorm_residual_bf16_64x1024" "tyr" route samples correct, correct)

private def runBenchmark (args : List String) : IO UInt32 := do
  if !(← requireCuda suiteName) then return 1
  let cfg ← Benchmark.parseConfig args "layernorm_bench"
  seedFixtures s!"{suiteName}.benchmark" 0
  let stream ← torch.cuda_current_stream
  let blackwell ← isBlackwellFamily
  let (f32Line, f32Ok) ← benchmarkFloat32 cfg stream blackwell
  let (bf16Line, bf16Ok) ← benchmarkBFloat16 cfg stream blackwell
  IO.println f32Line
  IO.println bf16Line
  match cfg.jsonlOut? with
  | some path => Benchmark.writeJsonl path #[f32Line, bf16Line]
  | none => pure ()
  pure (if f32Ok && bf16Ok then 0 else 1)

def main (args : List String) : IO UInt32 := do
  if args.contains "--benchmark" then runBenchmark args else
    runWithFixtures args suiteName fixtureSpec generateFixtures runOnce

end Examples.GPU.RunLayerNorm
