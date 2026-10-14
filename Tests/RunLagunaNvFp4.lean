/-
  Tests/RunLagunaNvFp4.lean

  Validates NVFP4 dequantization (Tyr.Model.Laguna.NvFp4) against PyTorch
  ground truth in Tests/fixtures/laguna/nvfp4.safetensors, on CPU and CUDA.

  This also exercises U8 and F8_E4M3 SafeTensors FFI loading, which the real
  Laguna-S-2.1-NVFP4 checkpoint depends on.

  Ground truth generator: Tests/fixtures/laguna/gen_nvfp4_fixtures.py
-/
import Tyr.Torch
import Tyr.SafeTensors.Load
import Tyr.Model.Laguna.NvFp4

open torch
open torch.laguna

private def check (cond : Bool) (msg : String) : IO Unit := do
  if cond then
    IO.println s!"PASS: {msg}"
  else
    throw (IO.userError s!"FAIL: {msg}")

/-- Max abs error and max rel error (denominator clamped by 1e-6) between
    two tensors, computed in FP32. -/
private def maxErrors {s : Shape} {d e : DType}
    (got : Tensor { shape := s, dtype := d }) (expected : Tensor { shape := s, dtype := e }) : IO (Float × Float) := do
  let a ← Tensor.toFloatArray got.toFloat32
  let b ← Tensor.toFloatArray expected.toFloat32
  let mut maxAbs := 0.0
  let mut maxRel := 0.0
  for i in [:a.size] do
    let diff := (a[i]! - b[i]!).abs
    if !a[i]!.isFinite || !b[i]!.isFinite || !diff.isFinite then
      throw <| IO.userError "non-finite tensor comparison"
    maxAbs := max maxAbs diff
    maxRel := max maxRel (diff / (b[i]!.abs + 1e-6))
  pure (maxAbs, maxRel)

private def deviceLabel : Device → String
  | .CPU => "cpu"
  | .CUDA i => s!"cuda:{i}"
  | .MPS => "mps"

private def findFixture : IO String := do
  let candidates := #[
    "Tests/fixtures/laguna/nvfp4.safetensors",
    "../Tests/fixtures/laguna/nvfp4.safetensors"
  ]
  for p in candidates do
    if ← System.FilePath.pathExists p then
      return p
  throw (IO.userError "nvfp4.safetensors fixture not found (run from repo root)")

/-- Load the four tensors of one case, run dequant on `device`, and report. -/
private def runMatrixCase (path : String) (device : Device) : IO Unit := do
  let packed ← torch.safetensors.loadTensorCandidates path #["m_packed"] #[64, 64]
  let scales ← torch.safetensors.loadTensorCandidates path #["m_scales"] #[64, 8]
  let globalS ← torch.safetensors.loadTensorCandidates path #["m_global"] #[1]
  let expected ← torch.safetensors.loadTensorCandidates path #["m_expected"] #[64, 128]
  if device == Device.CPU then
    check (packed.actualSpec.dtype == .UInt8) s!"m_packed loads as UInt8 (got {packed.actualSpec.dtype})"
    check (globalS.actualSpec.dtype == .Float32) s!"m_global loads as Float32 (got {globalS.actualSpec.dtype})"
    check (expected.actualSpec.dtype == .BFloat16) s!"m_expected loads as BFloat16 (got {expected.actualSpec.dtype})"
    -- NOTE: `lean_torch_get_dtype` has no float8 case, so an F8_E4M3 tensor
    -- reports `Unknown(Unknown)` via `Tensor.actualSpec.dtype`. The numerical comparison below
    -- is the actual validation that F8_E4M3 loading works.
    IO.println s!"INFO: m_scales dtypeStr = {scales.raw.dtypeStr} (F8_E4M3 expected; FFI has no float8 dtype string)"
  let got ← nvfp4.dequantMatrix (packed.to device) (scales.to device) (globalS.to device) 64 128
  check (got.actualSpec.dtype == .BFloat16) "dequantMatrix preserves BF16 output dtype"
  if device == .CPU then
    let wrongShapeRejected ← try
      let _ ← nvfp4.dequantMatrix (packed.reshapeChecked #[32, 128]) scales globalS 64 128
      pure false
    catch _ => pure true
    check wrongShapeRejected "dequant rejects equal-numel packed shape mismatch"
    let wrongGlobalRejected ← try
      let _ ← nvfp4.dequantMatrix packed scales (globalS.reshapeChecked #[1, 1]) 64 128
      pure false
    catch _ => pure true
    check wrongGlobalRejected "dequant rejects non-scalar global-scale layout"
  let (maxAbs, maxRel) ← maxErrors got (expected.to device)
  IO.println s!"  dequantMatrix [{deviceLabel device}]: maxAbs={maxAbs} maxRel={maxRel}"
  check (maxAbs <= 1e-3) s!"dequantMatrix [{deviceLabel device}] maxAbs={maxAbs} <= 1e-3"

private def runBankCase (path : String) (device : Device) : IO Unit := do
  let packed ← torch.safetensors.loadTensorCandidates path #["b_packed"] #[4, 32, 32]
  let scales ← torch.safetensors.loadTensorCandidates path #["b_scales"] #[4, 32, 4]
  let globalS ← torch.safetensors.loadTensorCandidates path #["b_global"] #[4]
  let expected ← torch.safetensors.loadTensorCandidates path #["b_expected"] #[4, 32, 64]
  let got ← nvfp4.dequantBank (packed.to device) (scales.to device) (globalS.to device) 4 32 64
  let (maxAbs, maxRel) ← maxErrors got (expected.to device)
  IO.println s!"  dequantBank [{deviceLabel device}]: maxAbs={maxAbs} maxRel={maxRel}"
  check (maxAbs <= 1e-3) s!"dequantBank [{deviceLabel device}] maxAbs={maxAbs} <= 1e-3"

def main : IO Unit := do
  let path ← findFixture
  IO.println s!"Using fixture: {path}"

  IO.println "-- dequantMatrix / dequantBank on CPU"
  runMatrixCase path Device.CPU
  runBankCase path Device.CPU

  if ← torch.cuda_is_available then
    IO.println "-- dequantMatrix / dequantBank on CUDA"
    let device := Device.CUDA 0
    runMatrixCase path device
    runBankCase path device
    torch.cuda_synchronize
  else
    IO.println "CUDA not available; skipped CUDA cases."

  IO.println "All Laguna NVFP4 dequantization tests passed."
