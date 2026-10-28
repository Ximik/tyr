/- End-to-end fused residual + RMSNorm validation. -/
import Tyr.Torch
import Tyr.GPU.Kernels.FusedRMSNorm
import Examples.GPU.IndexedParity
import Tyr.Typed.Neural
import Examples.GPU.FixtureRunner
import Examples.GPU.Benchmark

namespace Examples.GPU.RunRMSNorm

open torch

variable {device : Device}
open Tyr.GPU.Kernels

private def launchBlockX : Nat := 32
private def seqLen : Nat := 64
private def hiddenDim : Nat := 1024
private def rmsNormEps : Float := 1.0e-6

def suiteName : String := "rmsnorm"

def fixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/rmsnorm64x1024"⟩
  names := #["x", "residual", "weight", "expected_out", "expected_resid"]
}

def fixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fixtureSpec name

abbrev RMSNormInput (dtype : DType) (device : Device := .CPU) := Tensor { shape := #[1, 64, 1024], dtype := dtype, device := device }
abbrev RMSNormWeight (dtype : DType) (device : Device := .CPU) := Tensor { shape := #[1024], dtype := dtype, device := device }

structure RMSNormInputs (dtype : DType) (device : Device := .CPU) where
  x : RMSNormInput dtype device
  residual : RMSNormInput dtype device
  weight : RMSNormWeight dtype device

def rmsNormReference
    (inputs : RMSNormInputs .Float32 device)
    : RMSNormInput .Float32 device × RMSNormInput .Float32 device :=
  let eps := rmsNormEps
  let expectedResid := inputs.x + inputs.residual
  let expectedOut := Tensor.rmsNormWeighted expectedResid inputs.weight eps
  (expectedOut, expectedResid)

def rmsNormReferenceBFloat16
    (inputs : RMSNormInputs .BFloat16 device)
    : RMSNormInput .BFloat16 device × RMSNormInput .BFloat16 device :=
  let eps := rmsNormEps
  let expectedResid := inputs.x + inputs.residual
  let expectedOut32 : RMSNormInput .Float32 device :=
    Tensor.rmsNormWeighted
      (expectedResid.toFloat32)
      (inputs.weight.toFloat32)
      eps
  let expectedOut := expectedOut32.toBFloat16
  (expectedOut, expectedResid)

def asBFloat16Inputs (inputs : RMSNormInputs .Float32 device) : RMSNormInputs .BFloat16 device := {
  x := inputs.x.toBFloat16
  residual := inputs.residual.toBFloat16
  weight := inputs.weight.toBFloat16
}

private def logInputDTypes {dtype : DType} (label : String) (inputs : RMSNormInputs dtype device) : IO Unit := do
  IO.println <|
    s!"[{suiteName}:{label}] input_dtypes x={inputs.x.dtype} residual={inputs.residual.dtype} " ++
    s!"weight={inputs.weight.dtype}"

private def compareResults {dtype : DType}
    (mode : String)
    (expectedOut : RMSNormInput dtype device)
    (expectedResid : RMSNormInput dtype device)
    (out : RMSNormInput dtype device)
    (outResid : RMSNormInput dtype device)
    (rtol atol : Float)
    : IO Bool := do
  let outCheck := IndexedParity.compareTensors s!"rmsnorm.{mode}.output" (expectedOut.toFloat32) (out.toFloat32) rtol atol
  let residCheck := IndexedParity.compareTensors s!"rmsnorm.{mode}.residual" (expectedResid.toFloat32) (outResid.toFloat32) rtol atol
  logTensorCheck outCheck
  logTensorCheck residCheck
  pure (outCheck.ok && residCheck.ok)

private def runInputs {dtype : DType} (label : String) (inputs : RMSNormInputs dtype device) : IO Bool := do
  logInputDTypes label inputs
  let stream ← torch.cuda_current_stream
  let blackwell ← isBlackwellFamily
  match dtype, inputs with
  | .Float32, inputs => do
      let (expectedOut, expectedResid) := rmsNormReference inputs
      let out ← Tensor.uniformLike inputs.x 0 0
      let outResid ← Tensor.uniformLike inputs.residual 0 0
      fusedRMSNormResidual64x1024F32Direct.launch
        inputs.x.raw inputs.residual.raw inputs.weight.raw out.raw outResid.raw
        64 1 1 256 1 1 0 stream
      torch.cuda_synchronize
      compareResults (if blackwell then "blackwell.f32" else "f32") expectedOut expectedResid out outResid 5e-3 5e-3
  | .BFloat16, inputs => do
      let (expectedOut, expectedResid) := rmsNormReferenceBFloat16 inputs
      let out ← Tensor.uniformLike inputs.x 0 0
      let outResid ← Tensor.uniformLike inputs.residual 0 0
      fusedRMSNormResidual64x1024Bf16Direct.launch
        inputs.x.raw inputs.residual.raw inputs.weight.raw out.raw outResid.raw
        64 1 1 256 1 1 0 stream
      torch.cuda_synchronize
      compareResults (if blackwell then "blackwell.bf16" else "bf16") expectedOut expectedResid out outResid 2e-2 2e-2
  | dtype, _ =>
      IO.eprintln s!"[{suiteName}:{label}] unsupported_dtype dtype={dtype}"
      pure false

private def randomInputs (dtype : DType) (device : Device) : IO (RMSNormInputs dtype device) := do
  let base : RMSNormInputs .Float32 device := {
    x := ← Tensor.uniformLike (Tensor.zeros #[1, 64, 1024] false device)
    residual := ← Tensor.uniformLike (Tensor.zeros #[1, 64, 1024] false device)
    weight := ← Tensor.uniformLike (Tensor.zeros #[1024] false device)
  }
  match dtype with
  | .Float32 => pure base
  | .BFloat16 => pure (asBFloat16Inputs base)
  | _ => throw <| IO.userError s!"unsupported random rmsnorm dtype {dtype}"

def generateFixtures : IO Unit := do
  if !(← requireCuda suiteName) then
    throw <| IO.userError "CUDA is not available; cannot generate rmsnorm fixtures."

  IO.FS.createDirAll fixtureSpec.dir
  let inputs ← randomInputs .Float32 (← Device.cuda 0)
  let vendoredInputs := asBFloat16Inputs inputs
  let (expectedOut, expectedResid) := rmsNormReferenceBFloat16 vendoredInputs

  Tensor.saveTensor inputs.x (fixtureFile "x").toString
  Tensor.saveTensor inputs.residual (fixtureFile "residual").toString
  Tensor.saveTensor inputs.weight (fixtureFile "weight").toString
  Tensor.saveTensor expectedOut (fixtureFile "expected_out").toString
  Tensor.saveTensor expectedResid (fixtureFile "expected_resid").toString

  let outMean := Tensor.item (Tensor.meanAll expectedOut)
  let residMean := Tensor.item (Tensor.meanAll expectedResid)
  IO.println <|
    s!"Generated rmsnorm fixtures in {fixtureSpec.dir} " ++
    s!"input_dtype={inputs.x.dtype} ref_dtype={vendoredInputs.x.dtype} " ++
    s!"outMean={outMean} residMean={residMean}"

def runFloat32Once : IO Bool := do
  if !(← requireCuda suiteName) then
    return false
  seedFixtures s!"{suiteName}.f32" 0
  runInputs "random_f32" (← randomInputs .Float32 (← Device.cuda 0))

def runBFloat16Once : IO Bool := do
  if !(← requireCuda suiteName) then
    return false
  seedFixtures s!"{suiteName}.bf16" 0
  runInputs "random_bf16" (← randomInputs .BFloat16 (← Device.cuda 0))

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  let device ← Device.cuda 0
  if !(← fixturesPresent fixtureSpec) then
    generateFixtures

  let x ← SomeDTypeTensor.loadTensorExact #[1, 64, 1024] device
    (fixtureFile "x").toString
  let dtype := x.1
  let inputs : RMSNormInputs dtype device := {
    x := x.2
    residual := ← Tensor.loadTensorExact { device := device, shape := #[1, 64, 1024], dtype } (fixtureFile "residual").toString
    weight := ← Tensor.loadTensorExact { device := device, shape := #[1024], dtype } (fixtureFile "weight").toString
  }
  runInputs "fixtures" inputs


private def benchmarkFloat32 (cfg : Benchmark.Config) (stream : UInt64)
    (_blackwell : Bool) : IO (String × Bool) := do
  let inputs ← randomInputs .Float32 (← Device.cuda 0)
  let (expectedOut, expectedResid) := rmsNormReference inputs
  let out ← Tensor.uniformLike inputs.x 0 0
  let outResid ← Tensor.uniformLike inputs.residual 0 0
  let launch := do
    fusedRMSNormResidual64x1024F32Direct.launch
      inputs.x.raw inputs.residual.raw inputs.weight.raw out.raw outResid.raw
      64 1 1 256 1 1 0 stream
  launch
  torch.cuda_synchronize
  let correct ← compareResults "bench.f32" expectedOut expectedResid out outResid 5e-3 5e-3
  let samples ← Benchmark.timeCudaEvents cfg stream launch
  let route := "generated_direct_row_cta_256x4"
  pure (Benchmark.summaryJson cfg "rmsnorm_residual_f32_64x1024" "tyr" route samples correct, correct)

private def benchmarkBFloat16 (cfg : Benchmark.Config) (stream : UInt64)
    (_blackwell : Bool) : IO (String × Bool) := do
  let inputs ← randomInputs .BFloat16 (← Device.cuda 0)
  let (expectedOut, expectedResid) := rmsNormReferenceBFloat16 inputs
  let out ← Tensor.uniformLike inputs.x 0 0
  let outResid ← Tensor.uniformLike inputs.residual 0 0
  let launch := do
    fusedRMSNormResidual64x1024Bf16Direct.launch
      inputs.x.raw inputs.residual.raw inputs.weight.raw out.raw outResid.raw
      64 1 1 256 1 1 0 stream
  launch
  torch.cuda_synchronize
  let correct ← compareResults "bench.bf16" expectedOut expectedResid out outResid 2e-2 2e-2
  let samples ← Benchmark.timeCudaEvents cfg stream launch
  let route := "generated_direct_row_cta_256x4"
  pure (Benchmark.summaryJson cfg "rmsnorm_residual_bf16_64x1024" "tyr" route samples correct, correct)


private inductive RMSNormBenchPayload where
  | micro
  | training (rows : Nat)
  deriving Repr

private def rmsNormMatrix : LeanBenchmark.Matrix RMSNormBenchPayload := {
  defaultProfile := "training"
  cases := #[
    { id := "micro_64", payload := .training 64,
      profiles := #["micro"], tags := #["forward", "regression"] },
    { id := "b1_s768", payload := .training 768,
      profiles := #["training", "batch-sweep", "qwen3tts-talker"],
      tags := #["training", "latency"] },
    { id := "b2_s768", payload := .training 1536,
      profiles := #["training", "batch-sweep", "qwen3tts-talker"],
      tags := #["training", "throughput"] },
    { id := "b4_s768", payload := .training 3072,
      profiles := #["training", "batch-sweep", "qwen3tts-talker"],
      tags := #["training", "throughput", "primary"] },
    { id := "b8_s768", payload := .training 6144,
      profiles := #["training", "batch-sweep", "qwen3tts-talker"],
      tags := #["training", "throughput", "saturation"] }
  ]
}

/-- BF16 residual/output/input-gradient reference with Float32 saved inverse RMS and weight gradient. -/
def referenceTraining {rows : Nat}
    (x residual : Tensor { shape := #[rows, hiddenDim], dtype := .BFloat16, device := device })
    (weight : Tensor { shape := #[hiddenDim], dtype := .BFloat16, device := device })
    (gradOut gradOutResid : Tensor { shape := #[rows, hiddenDim], dtype := .BFloat16, device := device }) := Id.run do
  let expectedResid : Tensor { shape := #[rows, hiddenDim], dtype := .BFloat16, device := device } := x + residual
  let residF : Tensor { shape := #[rows, hiddenDim], dtype := .Float32, device := device } := expectedResid.toFloat32
  let weightF : Tensor { shape := #[hiddenDim], dtype := .Float32, device := device } := weight.toFloat32
  let gradOutF : Tensor { shape := #[rows, hiddenDim], dtype := .Float32, device := device } := gradOut.toFloat32
  let gradOutResidF : Tensor { shape := #[rows, hiddenDim], dtype := .Float32, device := device } := gradOutResid.toFloat32
  let inv2d : Tensor { shape := #[rows, 1], dtype := .Float32, device := device } :=
    Tensor.rsqrt (Tensor.meanDim (residF * residF) 1 true + rmsNormEps)
  let expectedInv : Tensor { shape := #[rows], dtype := .Float32, device := device } := Tensor.reshapeChecked inv2d #[rows]
  let invFull : Tensor { shape := #[rows, hiddenDim], dtype := .Float32, device := device } := Tensor.expand inv2d #[rows, hiddenDim]
  let weight2d : Tensor { shape := #[1, hiddenDim], dtype := .Float32, device := device } := Tensor.reshapeChecked weightF #[1, hiddenDim]
  let weightFull : Tensor { shape := #[rows, hiddenDim], dtype := .Float32, device := device } :=
    Tensor.expand weight2d #[rows, hiddenDim]
  let normalized : Tensor { shape := #[rows, hiddenDim], dtype := .Float32, device := device } := residF * invFull
  let expectedOut : Tensor { shape := #[rows, hiddenDim], dtype := .BFloat16, device := device } :=
    (normalized * weightFull).toBFloat16
  let weightedGrad : Tensor { shape := #[rows, hiddenDim], dtype := .Float32, device := device } := gradOutF * weightFull
  let dot : Tensor { shape := #[rows, 1], dtype := .Float32, device := device } :=
    Tensor.sumDim (weightedGrad * residF) 1 true
  let correction : Tensor { shape := #[rows, 1], dtype := .Float32, device := device } :=
    Tensor.mulScalar (dot * inv2d * inv2d * inv2d) 0.0009765625
  let correctionFull : Tensor { shape := #[rows, hiddenDim], dtype := .Float32, device := device } :=
    Tensor.expand correction #[rows, hiddenDim]
  let expectedGradInput : Tensor { shape := #[rows, hiddenDim], dtype := .BFloat16, device := device } :=
    (weightedGrad * invFull - residF * correctionFull + gradOutResidF).toBFloat16
  let expectedGradWeight : Tensor { shape := #[hiddenDim], dtype := .Float32, device := device } :=
    (Tensor.sumDim (gradOutF * normalized) 0 false).reshapeChecked #[hiddenDim]

  return (expectedOut, expectedResid, expectedInv, expectedGradInput, expectedGradWeight)

private def benchmarkTrainingShape (cfg : Benchmark.Config) (stream : UInt64)
    (caseLabel : String) (rows : Nat) : IO (Array String × Bool) := do
  let device ← Device.cuda 0
  let x : Tensor { shape := #[rows, hiddenDim], dtype := .BFloat16, device := device } :=
    (← Tensor.randn #[rows, hiddenDim] false device).toBFloat16
  let residual : Tensor { shape := #[rows, hiddenDim], dtype := .BFloat16, device := device } :=
    (← Tensor.randn #[rows, hiddenDim] false device).toBFloat16
  let weight : Tensor { shape := #[hiddenDim], dtype := .BFloat16, device := device } :=
    (← Tensor.randn #[hiddenDim] false device).toBFloat16
  let gradOut : Tensor { shape := #[rows, hiddenDim], dtype := .BFloat16, device := device } :=
    (← Tensor.randn #[rows, hiddenDim] false device).toBFloat16
  let gradOutResid : Tensor { shape := #[rows, hiddenDim], dtype := .BFloat16, device := device } :=
    (← Tensor.randn #[rows, hiddenDim] false device).toBFloat16

  let (expectedOut, expectedResid, expectedInv, expectedGradInput, expectedGradWeight) :=
    referenceTraining x residual weight gradOut gradOutResid

  let out ← Tensor.uniformLike x 0 0
  let outResid ← Tensor.uniformLike residual 0 0
  let invRms ← Tensor.uniformLike (Tensor.zeros #[rows] false device) 0 0
  let gradInput ← Tensor.uniformLike x 0 0
  let gradWeight ← Tensor.uniformLike (Tensor.zeros #[hiddenDim] false device) 0 0

  let rowsArg ← Tyr.GPU.Codegen.launchArg rows "rows"
  let launchForward : IO Unit :=
    fusedRMSNormResidualRows1024Bf16TrainFwd.launch
      x.raw residual.raw weight.raw out.raw outResid.raw invRms.raw rowsArg
      rows 1 1 256 1 1 0 stream
  let launchBwdInput : IO Unit :=
    fusedRMSNormResidualRows1024Bf16BwdInput.launch
      gradOut.raw gradOutResid.raw outResid.raw weight.raw invRms.raw gradInput.raw rowsArg
      rows 1 1 256 1 1 0 stream
  let launchBwdWeight : IO Unit :=
    fusedRMSNormResidualRows1024Bf16BwdWeight.launch
      gradOut.raw outResid.raw invRms.raw gradWeight.raw rowsArg
      4 1 1 256 1 1 0 stream
  let launchTraining : IO Unit := do
    launchForward
    launchBwdInput
    launchBwdWeight

  launchTraining
  torch.cuda_synchronize
  let outCheck := IndexedParity.compareTensors s!"rmsnorm.{caseLabel}.out" expectedOut out 3e-2 3e-2
  let residCheck := IndexedParity.compareTensors s!"rmsnorm.{caseLabel}.resid" expectedResid outResid 0.0 0.0
  let invCheck := IndexedParity.compareTensors s!"rmsnorm.{caseLabel}.inv" expectedInv invRms 2e-4 2e-4
  let dxCheck := IndexedParity.compareTensors s!"rmsnorm.{caseLabel}.dx" expectedGradInput gradInput 3e-2 3e-2
  let dwCheck := IndexedParity.compareTensors s!"rmsnorm.{caseLabel}.dw" expectedGradWeight gradWeight 3e-3 0.25
  for check in #[outCheck, residCheck, invCheck, dxCheck, dwCheck] do
    logTensorCheck check

  let fwdSamples ← Benchmark.timeCudaEvents cfg stream launchForward
  let dxSamples ← Benchmark.timeCudaEvents cfg stream launchBwdInput
  let dwSamples ← Benchmark.timeCudaEvents cfg stream launchBwdWeight
  let trainingSamples ← Benchmark.timeCudaEvents cfg stream launchTraining

  let postOut := IndexedParity.compareTensors s!"rmsnorm.{caseLabel}.out.post" expectedOut out 3e-2 3e-2
  let postResid := IndexedParity.compareTensors s!"rmsnorm.{caseLabel}.resid.post" expectedResid outResid 0.0 0.0
  let postInv := IndexedParity.compareTensors s!"rmsnorm.{caseLabel}.inv.post" expectedInv invRms 2e-4 2e-4
  let postDx := IndexedParity.compareTensors s!"rmsnorm.{caseLabel}.dx.post" expectedGradInput gradInput 3e-2 3e-2
  let postDw := IndexedParity.compareTensors s!"rmsnorm.{caseLabel}.dw.post" expectedGradWeight gradWeight 3e-3 0.25
  for check in #[postOut, postResid, postInv, postDx, postDw] do
    logTensorCheck check

  let correct := outCheck.ok && residCheck.ok && invCheck.ok && dxCheck.ok && dwCheck.ok &&
    postOut.ok && postResid.ok && postInv.ok && postDx.ok && postDw.ok
  let elements := rows.toFloat * hiddenDim.toFloat
  let fwdRoute := "generated_bf16_rows1024_saved_inv_rowcta256"
  let dxRoute := "generated_bf16_rows1024_vjp_rowcta256"
  let dwRoute := "generated_bf16_rows1024_dw_colcta256"
  let lines := #[
    Benchmark.summaryJson cfg s!"rmsnorm_training_fwd_{caseLabel}" "tyr"
      fwdRoute fwdSamples correct "saved_state_forward" true
      (some elements) (some "elements"),
    Benchmark.summaryJson cfg s!"rmsnorm_training_bwd_input_{caseLabel}" "tyr"
      dxRoute dxSamples correct "input_and_residual_vjp" true
      (some elements) (some "elements"),
    Benchmark.summaryJson cfg s!"rmsnorm_training_bwd_weight_{caseLabel}" "tyr"
      dwRoute dwSamples correct "weight_gradient" true
      (some elements) (some "elements"),
    Benchmark.summaryJson cfg s!"rmsnorm_training_{caseLabel}" "tyr"
      s!"{fwdRoute}_plus_{dxRoute}_plus_{dwRoute}" trainingSamples correct
      "full_training_operation" true (some (3.0 * elements)) (some "elements")
  ]
  pure (lines, correct)

private def runBenchmark (args : List String) : IO UInt32 := do
  if !(← requireCuda suiteName) then return 1
  let cfg ← Benchmark.parseConfig args "rmsnorm_bench"
  seedFixtures s!"{suiteName}.benchmark" 0
  let stream ← torch.cuda_current_stream
  let blackwell ← isBlackwellFamily
  let selection := LeanBenchmark.MatrixSelection.parse args rmsNormMatrix.defaultProfile
  let selected ← match rmsNormMatrix.select selection with
    | .ok cases => pure cases
    | .error msg => throw <| IO.userError msg
  let mut lines : Array String := #[]
  let mut allOk := true
  for matrixCase in selected do
    match matrixCase.payload with
    | .micro => do
      let (f32Line, f32Ok) ← benchmarkFloat32 cfg stream blackwell
      let (bf16Line, bf16Ok) ← benchmarkBFloat16 cfg stream blackwell
      lines := lines.push f32Line |>.push bf16Line
      allOk := allOk && f32Ok && bf16Ok
    | .training rows => do
      let (caseLines, ok) ← benchmarkTrainingShape cfg stream matrixCase.id rows
      for line in caseLines do lines := lines.push line
      allOk := allOk && ok
  for line in lines do IO.println line
  match cfg.jsonlOut? with
  | some path => Benchmark.writeJsonl path lines
  | none => pure ()
  pure (if allOk then 0 else 1)

def main (args : List String) : IO UInt32 := do
  if args.contains "--benchmark" then runBenchmark args else
    runWithFixtures args suiteName fixtureSpec generateFixtures runOnce

end Examples.GPU.RunRMSNorm
