/-
  Tests/RunLagunaMoe.lean

  Validates the Laguna MoE block (Tyr.Model.Laguna.MoE: sigmoid top-k router,
  NVFP4-packed experts, shared expert) against PyTorch ground truth in
  Tests/fixtures/laguna/moe.safetensors, on CPU and CUDA.

  Small config: hidden=64, 8 experts, top-2, moeInt=32, sharedInt=32,
  scaling=2.5, norm_topk_prob=true, with a nonzero e_score_correction_bias.

  Ground truth generator: Tests/fixtures/laguna/gen_moe_fixtures.py
-/
import Tyr.SafeTensors.Load
import Tyr.Model.Laguna.Config
import Tyr.Model.Laguna.MoE

open torch

variable {device : Device}

private def pack {σ : StaticSpec} (x : Tensor σ) : IO (SomeDTypeTensor σ.shape σ.device) :=
  IO.ofExcept x.reifyDType

private def requireBF16 {shape : Shape} (x : SomeDTypeTensor shape device) :
    IO (Tensor ⟨shape, .BFloat16, device⟩) :=
  IO.ofExcept (x.atDType? ⟨.BFloat16, by decide⟩)

private def moveStored {shape : Shape} {source : Device} (x : SomeDTypeTensor shape source)
    (target : Device) : SomeDTypeTensor shape target := ⟨x.1, x.2.to target⟩
open torch.laguna

private def check (cond : Bool) (msg : String) : IO Unit := do
  if cond then
    IO.println s!"PASS: {msg}"
  else
    throw (IO.userError s!"FAIL: {msg}")

/-- Small test config mirroring the fixture generator. -/
private def testCfg : Config := { LagunaConfig.laguna_s_2_1 with
  hidden_size := 64
  num_experts := 8
  num_experts_per_tok := 2
  moe_intermediate_size := 32
  shared_expert_intermediate_size := 32
  norm_topk_prob := true
  moe_routed_scaling_factor := 2.5
  moe_router_logit_softcapping := 0.0 }

private def findFixture : IO String := do
  let candidates := #[
    "Tests/fixtures/laguna/moe.safetensors",
    "../Tests/fixtures/laguna/moe.safetensors"
  ]
  for p in candidates do
    if ← System.FilePath.pathExists p then
      return p
  throw (IO.userError "moe.safetensors fixture not found (run from repo root)")

/-- Build the MoE block from the fixture, moving all tensors to `device`. -/
private def loadBlock (path : String) (device : Device) : IO (LagunaSparseMoeBlock (device := device) testCfg .BFloat16) := do
  let routerW ← torch.safetensors.loadTensorCandidates path #["router_weight"] #[8, 64]
  let routerB ← torch.safetensors.loadTensorCandidates path #["router_bias"] #[8]
  let router : LagunaTopKRouter (device := device) testCfg := {
    weight := moveStored routerW device
    eScoreCorrectionBias := some (moveStored routerB device) }
  let gatePacked ← torch.safetensors.loadTensorCandidates path #["gate_packed"] #[8, 32, 32]
  let gateScale ← torch.safetensors.loadTensorCandidates path #["gate_scales"] #[8, 32, 4]
  let gateGlobal ← torch.safetensors.loadTensorCandidates path #["gate_global"] #[8]
  let upPacked ← torch.safetensors.loadTensorCandidates path #["up_packed"] #[8, 32, 32]
  let upScale ← torch.safetensors.loadTensorCandidates path #["up_scales"] #[8, 32, 4]
  let upGlobal ← torch.safetensors.loadTensorCandidates path #["up_global"] #[8]
  let downPacked ← torch.safetensors.loadTensorCandidates path #["down_packed"] #[8, 64, 16]
  let downScale ← torch.safetensors.loadTensorCandidates path #["down_scales"] #[8, 64, 2]
  let downGlobal ← torch.safetensors.loadTensorCandidates path #["down_global"] #[8]
  let experts : LagunaPackedExperts (device := device) testCfg := {
    gatePacked := moveStored gatePacked device
    gateScale := moveStored gateScale device
    gateGlobal := moveStored gateGlobal device
    upPacked := moveStored upPacked device
    upScale := moveStored upScale device
    upGlobal := moveStored upGlobal device
    downPacked := moveStored downPacked device
    downScale := moveStored downScale device
    downGlobal := moveStored downGlobal device }
  let sharedGate ← torch.safetensors.loadTensorCandidates path #["shared_gate"] #[32, 64]
  let sharedUp ← torch.safetensors.loadTensorCandidates path #["shared_up"] #[32, 64]
  let sharedDown ← torch.safetensors.loadTensorCandidates path #["shared_down"] #[64, 32]
  pure {
    router := router
    experts := experts
    sharedGateProj := (← requireBF16 sharedGate).to device
    sharedUpProj := (← requireBF16 sharedUp).to device
    sharedDownProj := (← requireBF16 sharedDown).to device }

/-- Max abs / max rel (denominator clamped by 1e-3) error in FP32. -/
private def maxErrors {s : Shape} {d e : DType}
    (got : Tensor { shape := s, dtype := d, device := device }) (expected : Tensor { shape := s, dtype := e, device := device }) : IO (Float × Float) := do
  let a ← Tensor.toFloatArray got.toFloat32
  let b ← Tensor.toFloatArray expected.toFloat32
  let mut maxAbs := 0.0
  let mut maxRel := 0.0
  for i in [:a.size] do
    let diff := (a[i]! - b[i]!).abs
    if !a[i]!.isFinite || !b[i]!.isFinite || !diff.isFinite then
      throw <| IO.userError "non-finite tensor comparison"
    maxAbs := max maxAbs diff
    maxRel := max maxRel (diff / (b[i]!.abs + 1e-3))
  pure (maxAbs, maxRel)

private def countViolations {s : Shape} {d e : DType}
    (got : Tensor { shape := s, dtype := d, device := device }) (expected : Tensor { shape := s, dtype := e, device := device }) (rtol atol : Float) : IO Float := do
  let a ← Tensor.toFloatArray got.toFloat32
  let b ← Tensor.toFloatArray expected.toFloat32
  let mut violations := 0.0
  for i in [:a.size] do
    if !(a[i]! - b[i]!).abs.isFinite || (a[i]! - b[i]!).abs > atol + rtol * b[i]!.abs then
      violations := violations + 1.0
  pure violations

private def deviceLabel (device : Device) : String :=
  match device.kind with
  | .CPU => "cpu"
  | .CUDA i => s!"cuda:{i}"
  | .MPS => "mps"

private def runCase (path : String) (device : Device) : IO Unit := do
  let lbl := deviceLabel device
  let block ← loadBlock path device
  let x ← torch.safetensors.loadTensorCandidates path #["x"] #[5, 64]
  let xD : Tensor { shape := #[5, 64], dtype := .BFloat16, device := device } := (← requireBF16 x).to device
  let yExp ← torch.safetensors.loadTensorCandidates path #["y_expected"] #[5, 64]
  let idxExp ← torch.safetensors.loadTensorCandidates path #["top_idx_expected"] #[5, 2]
  let wExp ← torch.safetensors.loadTensorCandidates path #["weights_expected"] #[5, 2]

  -- Router unit check: selected expert SETS match (order-insensitive via
  -- descending sort), routing-weight multisets match within tolerance.
  let (idx, w) ← block.router.route testCfg xD
  let (sortedGot, _) := qwen35.topK2d idx 2 1
  let (sortedExp, _) := qwen35.topK2d (idxExp.2.to device) 2 1
  let gotIds ← Tensor.toUInt64Array sortedGot.toInt64
  let expectedIds ← Tensor.toUInt64Array sortedExp.toInt64
  let idxMismatch := (gotIds.zip expectedIds).foldl (fun n (a, b) => n + if a == b then 0 else 1) (0 : Nat)
  check (idxMismatch == 0) s!"router [{lbl}] selected expert sets match (mismatches={idxMismatch})"
  let (wGotSorted, _) := qwen35.topK2d (qwen35.floatDynamic w) 2 1
  let (wExpSorted, _) := qwen35.topK2d (wExp.2.to device) 2 1
  let (wAbs, wRel) ← maxErrors wGotSorted wExpSorted
  IO.println s!"  router [{lbl}]: weights maxAbs={wAbs} maxRel={wRel}"
  check (wAbs <= 0.02) s!"router [{lbl}] weights maxAbs={wAbs} <= 0.02"

  -- Full MoE block forward: eager path (strict vs fixture, same op order),
  let y ← block.forward2d testCfg xD false
  let (maxAbs, maxRel) ← maxErrors y (yExp.2.to device)
  let viol ← countViolations y (yExp.2.to device) 0.02 0.05
  IO.println s!"  forward2d(eager) [{lbl}]: maxAbs={maxAbs} maxRel={maxRel} violations={viol}"
  check (viol == 0.0) s!"forward2d(eager) [{lbl}] within rtol=2% atol=0.05 (violations={viol})"
  -- ... and the fused kernel path (exact-FP4-in-FP32 vs the BF16 reference
  -- rounding; kernel-validated tolerance, see Tests/RunLagunaFused.lean).
  let onCuda := match device.kind with | .CUDA _ => true | _ => false
  if onCuda then do
    let yF ← block.forward2d testCfg xD
    let (fAbs, _) ← maxErrors yF (yExp.2.to device)
    let violF ← countViolations yF (yExp.2.to device) 0.02 0.5
    IO.println s!"  forward2d(fused) [{lbl}]: maxAbs={fAbs} violations={violF}"
    check (violF == 0.0) s!"forward2d(fused) [{lbl}] within rtol=2% atol=0.5 (violations={violF})"

private def checkSignedScatterDimension : IO Unit := do
  let input := qwen35.zerosDynamic #[2, 3]
  let indices := (Tensor.fromInt64Array #[2, 0]).reshapeChecked #[2, 1]
  let values := (Tensor.fromFloatArray #[5.0, 7.0]).reshapeChecked #[2, 1]
  let out := scatterAdd input (-1) indices values
  check ((← Tensor.toFloatArray out) == #[0.0, 0.0, 5.0, 7.0, 0.0, 0.0])
    "scatter-add preserves signed negative dimension across the native ABI"

def main : IO Unit := do
  checkSignedScatterDimension
  let path ← findFixture
  IO.println s!"Using fixture: {path}"
  IO.println "-- router + MoE block on CPU"
  runCase path Device.CPU
  if ← torch.cuda_is_available then
    let device ← Device.cuda 0
    IO.println "-- router + MoE block on CUDA"
    runCase path device
    torch.cuda_synchronize
  else
    IO.println "CUDA not available; skipped CUDA cases."
  IO.println "All Laguna MoE tests passed."
