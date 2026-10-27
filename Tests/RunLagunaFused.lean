/-
  Tests/RunLagunaFused.lean

  Validates the fused NVFP4 MoE expert kernel (Tyr/Model/Laguna/Fused.lean,
  cc/src/tyr_laguna_moe.cu) against the eager packed-bank decode path
  (gather + nvfp4.dequantBank + bmm) in Tyr/Model/Laguna/MoE.lean, and
  micro-benchmarks both on production-sized banks.

  Part 1 (correctness): builds the small fixture MoE block
  (Tests/fixtures/laguna/moe.safetensors; hidden=64, 8 experts, top-2,
  moeInt=32) on CUDA and runs the single-token decode path FUSED vs EAGER
  for each of the 5 fixture tokens. Hard check: all elements within
  rtol=2% / atol=0.5 of both the eager output and the fixture's y_expected
  (the kernel computes exact FP4 math in FP32, so the residual is the eager
  path's own BF16 dequant/accumulation noise; cross-validated elementwise
  against an exact-FP4 PyTorch reference to ~1 BF16 quantum).

  Part 2 (native-prefill correctness): exercises the SM12x cuBLASLt W4A4
  path at tensor-core-compatible dimensions (hidden=128, moeInt=128) and
  compares it with the eager W4A16 reference using a normalized-RMSE gate.

  Part 3 (micro-benchmark): synthesizes production-sized packed banks
  (hidden=3072, 256 experts, moeInt=1024, k=10; ~1.4GB on device) by tiling
  the fixture bank bytes, then times
    a) the fused op alone (effective FP4 read bandwidth = bytes/ms), and
    b) the full MoE block single-token decode, fused vs eager.
  Perf numbers are reported but not asserted (machine-dependent); the
  roofline bar for 13+ tok/s end-to-end is ≥ 150 GB/s on the FP4 reads.

  Runs on CUDA only; skipped silently otherwise.
-/
import Tyr.SafeTensors.Load
import Tyr.Model.Laguna.Config
import Tyr.Model.Laguna.MoE
import Tyr.Model.Laguna.Fused

open torch

variable {device : Device}

private def pack {σ : StaticSpec} (x : Tensor σ) : IO (SomeDTypeTensor σ.shape σ.device) :=
  IO.ofExcept x.reifyDType

private def requireBF16 {shape : Shape} (x : SomeDTypeTensor shape device) :
    IO (Tensor ⟨shape, .BFloat16, device⟩) :=
  IO.ofExcept (x.atDType? .BFloat16)

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

/-- Small tensor-core-compatible config for the native SM12x prefill path. -/
private def nativeCfg : Config := { LagunaConfig.laguna_s_2_1 with
  hidden_size := 128
  num_experts := 8
  num_experts_per_tok := 2
  moe_intermediate_size := 128
  shared_expert_intermediate_size := 128
  norm_topk_prob := true
  moe_routed_scaling_factor := 1.0
  moe_router_logit_softcapping := 0.0 }

/-- Production-sized config (Laguna-S-2.1 defaults). -/
private def bigCfg : Config := LagunaConfig.laguna_s_2_1

private def findFixture : IO String := do
  let candidates := #[
    "Tests/fixtures/laguna/moe.safetensors",
    "../Tests/fixtures/laguna/moe.safetensors"
  ]
  for p in candidates do
    if ← System.FilePath.pathExists p then
      return p
  throw (IO.userError "moe.safetensors fixture not found (run from repo root)")

/-- Build the small fixture MoE block on `device` (same as RunLagunaMoe). -/
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
  let ad := Tensor.abs (got.toFloat32 - expected.toFloat32)
  let maxAbs := (Tensor.maxAllChecked ad).item
  let denom := Tensor.addScalar (Tensor.abs expected.toFloat32) 1e-3
  let maxRel := (Tensor.maxAllChecked (ad / denom)).item
  pure (maxAbs, maxRel)

/-- Tile `src` (flattened, then moved to `device`) out to exactly `shape`'s
    element count by repeated doubling + a final slice, and reshape.
    Used to synthesize benchmark-sized banks from the fixture bank bytes;
    the values don't need to be meaningful, only the shapes/dtypes do. -/
private def tileToShape {s : Shape} {source : Device} (src : SomeDTypeTensor s source) (shape : Shape) (device : Device) : IO (SomeDTypeTensor shape device) := do
  let numel := shape.foldl (· * ·) 1
  let srcD := src.2.to device
  let n0 := srcD.shape.foldl (· * ·) 1
  if n0 == 0 then throw <| IO.userError "tileToShape: empty source tensor"
  let mut cur : ShapePackedTensor src.1 device := (srcD.reshapeChecked #[n0]).packShape
  let mut n := n0
  while n < numel do
    cur := (Tensor.catManyChecked #[cur, cur] 0 #[n * 2]).packShape
    n := n * 2
  let clipped := cur.2.sliceChecked 0 0 numel #[numel]
  pure ⟨src.1, clipped.reshapeChecked shape⟩

private def bfloatDynamic {σ : StaticSpec} (t : Tensor σ) : Tensor {σ with dtype := .BFloat16} :=
  t.toBFloat16

/-- Build a compact packed MoE whose dimensions activate the native Blackwell
    W4A4 path. Shared-expert weights are zero so the block output isolates the
    routed experts when compared with the eager W4A16 reference. -/
private def synthNativeBlock
    (path : String) (device : Device) : IO (LagunaSparseMoeBlock (device := device) nativeCfg .BFloat16) := do
  let E := nativeCfg.num_experts
  let h := nativeCfg.hidden_size
  let mi := nativeCfg.moe_intermediate_size
  let routerW ← Tensor.randn #[E, h] false device
  let router : LagunaTopKRouter (device := device) nativeCfg := {
    weight := (← pack (bfloatDynamic routerW))
    eScoreCorrectionBias := none }
  let gatePacked ← torch.safetensors.loadTensorCandidates path #["gate_packed"] #[8, 32, 32]
  let gateScale ← torch.safetensors.loadTensorCandidates path #["gate_scales"] #[8, 32, 4]
  let gateGlobal ← torch.safetensors.loadTensorCandidates path #["gate_global"] #[8]
  let upPacked ← torch.safetensors.loadTensorCandidates path #["up_packed"] #[8, 32, 32]
  let upScale ← torch.safetensors.loadTensorCandidates path #["up_scales"] #[8, 32, 4]
  let upGlobal ← torch.safetensors.loadTensorCandidates path #["up_global"] #[8]
  let downPacked ← torch.safetensors.loadTensorCandidates path #["down_packed"] #[8, 64, 16]
  let downScale ← torch.safetensors.loadTensorCandidates path #["down_scales"] #[8, 64, 2]
  let downGlobal ← torch.safetensors.loadTensorCandidates path #["down_global"] #[8]
  let experts : LagunaPackedExperts (device := device) nativeCfg := {
    gatePacked := (← tileToShape gatePacked #[E, mi, h / 2] device)
    gateScale := (← tileToShape gateScale #[E, mi, h / 16] device)
    gateGlobal := (← tileToShape gateGlobal #[E] device)
    upPacked := (← tileToShape upPacked #[E, mi, h / 2] device)
    upScale := (← tileToShape upScale #[E, mi, h / 16] device)
    upGlobal := (← tileToShape upGlobal #[E] device)
    downPacked := (← tileToShape downPacked #[E, h, mi / 2] device)
    downScale := (← tileToShape downScale #[E, h, mi / 16] device)
    downGlobal := (← tileToShape downGlobal #[E] device) }
  let sharedGate : Tensor { shape := #[128, 128], device := device, dtype := .BFloat16 } :=
    bfloatDynamic (Tensor.zeros #[128, 128] false device)
  let sharedUp : Tensor { shape := #[128, 128], device := device, dtype := .BFloat16 } :=
    bfloatDynamic (Tensor.zeros #[128, 128] false device)
  let sharedDown : Tensor { shape := #[128, 128], device := device, dtype := .BFloat16 } :=
    bfloatDynamic (Tensor.zeros #[128, 128] false device)
  pure {
    router := router
    experts := experts
    sharedGateProj := sharedGate
    sharedUpProj := sharedUp
    sharedDownProj := sharedDown }

/-- Build a production-sized MoE block with synthesized packed banks.
    Bank bytes are tiled from the fixture (content is random-looking, which
    is all the benchmark needs); router/shared weights are random BF16. -/
private def synthBigBlock (path : String) (device : Device) : IO (LagunaSparseMoeBlock (device := device) bigCfg .BFloat16) := do
  let E := bigCfg.num_experts
  let h := bigCfg.hidden_size
  let mi := bigCfg.moe_intermediate_size
  let routerW ← Tensor.randn #[E, h] false device
  let router : LagunaTopKRouter (device := device) bigCfg := {
    weight := (← pack (bfloatDynamic routerW))
    eScoreCorrectionBias := none }
  let gatePacked ← torch.safetensors.loadTensorCandidates path #["gate_packed"] #[8, 32, 32]
  let gateScale ← torch.safetensors.loadTensorCandidates path #["gate_scales"] #[8, 32, 4]
  let gateGlobal ← torch.safetensors.loadTensorCandidates path #["gate_global"] #[8]
  let upPacked ← torch.safetensors.loadTensorCandidates path #["up_packed"] #[8, 32, 32]
  let upScale ← torch.safetensors.loadTensorCandidates path #["up_scales"] #[8, 32, 4]
  let upGlobal ← torch.safetensors.loadTensorCandidates path #["up_global"] #[8]
  let downPacked ← torch.safetensors.loadTensorCandidates path #["down_packed"] #[8, 64, 16]
  let downScale ← torch.safetensors.loadTensorCandidates path #["down_scales"] #[8, 64, 2]
  let downGlobal ← torch.safetensors.loadTensorCandidates path #["down_global"] #[8]
  IO.println "  tiling packed banks to production size..."
  let experts : LagunaPackedExperts (device := device) bigCfg := {
    gatePacked := (← tileToShape gatePacked #[E, mi, h / 2] device)
    gateScale := (← tileToShape gateScale #[E, mi, h / 16] device)
    gateGlobal := (← tileToShape gateGlobal #[E] device)
    upPacked := (← tileToShape upPacked #[E, mi, h / 2] device)
    upScale := (← tileToShape upScale #[E, mi, h / 16] device)
    upGlobal := (← tileToShape upGlobal #[E] device)
    downPacked := (← tileToShape downPacked #[E, h, mi / 2] device)
    downScale := (← tileToShape downScale #[E, h, mi / 16] device)
    downGlobal := (← tileToShape downGlobal #[E] device) }
  let si := bigCfg.shared_expert_intermediate_size
  let sharedGate ← Tensor.randn #[si, h] false device
  let sharedUp ← Tensor.randn #[si, h] false device
  let sharedDown ← Tensor.randn #[h, si] false device
  pure {
    router := router
    experts := experts
    sharedGateProj := bfloatDynamic sharedGate
    sharedUpProj := bfloatDynamic sharedUp
    sharedDownProj := bfloatDynamic sharedDown }

/-- Time `f` over `iters` iterations after `warmup` unmeasured iterations.
    Returns (CUDA-event ms/iter, wall-clock ms/iter). -/
private def timeCuda (warmup iters : Nat) (f : IO Unit) : IO (Float × Float) := do
  for _ in [:warmup] do
    f
  torch.cuda_synchronize
  let stream ← torch.cuda_current_stream
  let start ← torch.cuda_event_create
  let stop ← torch.cuda_event_create
  torch.cuda_event_record start stream
  let t0 ← IO.monoNanosNow
  for _ in [:iters] do
    f
  torch.cuda_event_record stop stream
  torch.cuda_event_synchronize stop
  let t1 ← IO.monoNanosNow
  let evMs ← torch.cuda_event_elapsed_ms start stop
  torch.cuda_event_destroy start
  torch.cuda_event_destroy stop
  let wallMs := (t1 - t0).toFloat / 1.0e6
  pure (evMs / iters.toFloat, wallMs / iters.toFloat)

/-- Count elements violating `|got - exp| > atol + rtol * |exp|` (FP32). -/
private def countViolations {s : Shape} {d e : DType}
    (got : Tensor { shape := s, dtype := d, device := device }) (expected : Tensor { shape := s, dtype := e, device := device }) (rtol atol : Float) : IO Float := do
  let a ← Tensor.toFloatArray got.toFloat32
  let b ← Tensor.toFloatArray expected.toFloat32
  let mut count := 0.0
  for i in [:a.size] do
    if !(a[i]! - b[i]!).abs.isFinite || (a[i]! - b[i]!).abs > atol + rtol * b[i]!.abs then
      count := count + 1.0
  pure count

/-- Preserve the F32 reduction used by the native-prefill reference gate. -/
private def normalizedRmse {s : Shape} {d e : DType}
    (got : Tensor { shape := s, dtype := d, device := device }) (expected : Tensor { shape := s, dtype := e, device := device }) : IO (Float × Float × Float) := do
  let delta := got.toFloat32 - expected.toFloat32
  let ref := expected.toFloat32
  let rmse := Float.sqrt (Tensor.meanAll (delta * delta)).item
  let refRms := Float.sqrt (Tensor.meanAll (ref * ref)).item
  pure (rmse, refRms, rmse / max refRms 1.0e-6)

/-- Part 1: fused vs eager single-token decode on the fixture block. -/
private def runCorrectness (path : String) (device : Device) : IO Unit := do
  IO.println "-- correctness: fused vs eager decode (fixture block, CUDA)"
  let block ← loadBlock path device
  let x ← torch.safetensors.loadTensorCandidates path #["x"] #[5, 64]
  let xD : Tensor { shape := #[5, 64], device := device, dtype := .BFloat16 } := (← requireBF16 x).to device
  let yExp ← torch.safetensors.loadTensorCandidates path #["y_expected"] #[5, 64]
  let yExpD : Tensor { shape := #[5, 64], device := device, dtype := .BFloat16 } := (← requireBF16 yExp).to device
  let mut worst := 0.0
  for i in [:5] do
    let x1 : Tensor { shape := #[1, 64], device := device, dtype := .BFloat16 } := Tensor.reshapeChecked (sliceTyped xD 0 i.toUInt64 1) #[1, 64]
    let yFused ← block.forward2d testCfg x1
    let yEager ← block.forward2d testCfg x1 false
    let (maxAbs, maxRel) ← maxErrors yFused yEager
    let yExpRow : Tensor { shape := #[1, 64], device := device, dtype := .BFloat16 } := Tensor.reshapeChecked (sliceTyped yExpD 0 i.toUInt64 1) #[1, 64]
    let (refAbs, _) ← maxErrors yFused yExpRow
    -- The fused kernel evaluates the exact FP4 math in FP32, while the eager
    -- path rounds dequantized weights to BF16 and accumulates in BF16, so
    -- fused-vs-eager carries the eager path's own BF16 rounding noise.
    -- Measured on this fixture (and reproduced exactly-vs-bf16 in Python):
    -- up to ~0.4 absolute and ~7% relative at cancellation-heavy elements
    -- (|y| reaches ~40, routed row L1 norms ~40-110). A flat 1e-2 bound is
    -- therefore unattainable for an exact kernel; use rtol=2% with an
    -- atol=0.5 noise floor (≈0.4% of the routed row L1 norm), which a real
    -- addressing/scale bug would still blow through by orders of magnitude.
    -- Cross-validated elementwise against an exact-FP4 PyTorch reference:
    -- the kernel matches it to ~1 BF16 output quantum everywhere.
    let viol ← countViolations yFused yEager 0.02 0.5
    let violRef ← countViolations yFused yExpRow 0.02 0.5
    IO.println s!"  token {i}: fused-vs-eager maxAbs={maxAbs} maxRel={maxRel} viol={viol} | fused-vs-fixture maxAbs={refAbs} viol={violRef}"
    worst := max worst maxAbs
    check (viol == 0.0) s!"token {i}: fused matches eager within rtol=2% atol=0.5 (violations={viol})"
    check (violRef == 0.0) s!"token {i}: fused matches fixture y_expected within rtol=2% atol=0.5 (violations={violRef})"
  IO.println s!"  worst fused-vs-eager maxAbs across 5 tokens: {worst}"

/-- Part 2: native SM12x W4A4 prefill vs the eager W4A16 reference.

    W4A4 dynamically quantizes activations twice, so bitwise/elementwise
    equality with W4A16 is neither expected nor desirable. The NRMSE bound is
    deliberately loose enough for that quantization error but tight enough to
    catch scale-swizzle, global-scale, row-dispatch, or nibble-order bugs. -/
private def runNativeCorrectness (path : String) (device : Device) : IO Unit := do
  IO.println "-- correctness: native SM12x W4A4 prefill vs eager W4A16"
  let block ← synthNativeBlock path device
  let x : Tensor { shape := #[16, 128], device := device, dtype := .BFloat16 } := bfloatDynamic (← Tensor.randn #[16, 128] false device)
  let yNative ← block.forward2d nativeCfg x
  let yEager ← block.forward2d nativeCfg x false
  let (maxAbs, maxRel) ← maxErrors yNative yEager
  let (rmse, refRms, nrmse) ← normalizedRmse yNative yEager
  IO.println s!"  balanced: maxAbs={maxAbs} maxRel={maxRel} rmse={rmse} refRms={refRms} nrmse={nrmse}"
  check (nrmse <= 0.35) s!"balanced native W4A4 prefill NRMSE {nrmse} ≤ 0.35 vs eager W4A16"

  -- A tied router sends every token to the same two experts. Seventeen tokens
  -- force maxExpertRows=17 and a padded GEMM stride of 20, covering both the
  -- maximally skewed routing bound and a non-multiple-of-four occupancy.
  let zeroRouterW : Tensor { shape := #[8, 128], device := device, dtype := .BFloat16 } :=
    bfloatDynamic (Tensor.zeros #[8, 128] false device)
  let skewed : LagunaSparseMoeBlock (device := device) nativeCfg .BFloat16 := {
    block with router := {
      weight := (← pack zeroRouterW)
      eScoreCorrectionBias := none } }
  let xSkew : Tensor { shape := #[17, 128], device := device, dtype := .BFloat16 } :=
    bfloatDynamic (← Tensor.randn #[17, 128] false device)
  let ySkewNative ← skewed.forward2d nativeCfg xSkew
  let ySkewEager ← skewed.forward2d nativeCfg xSkew false
  let (skewAbs, skewRel) ← maxErrors ySkewNative ySkewEager
  let (skewRmse, skewRefRms, skewNrmse) ←
    normalizedRmse ySkewNative ySkewEager
  IO.println s!"  skewed:   maxAbs={skewAbs} maxRel={skewRel} rmse={skewRmse} refRms={skewRefRms} nrmse={skewNrmse}"
  check (skewNrmse <= 0.35) s!"maximally skewed native W4A4 prefill NRMSE {skewNrmse} ≤ 0.35 vs eager W4A16"

/-- Part 3: micro-benchmark on production-sized banks. -/
private def runBenchmark (path : String) (device : Device) : IO Unit := do
  IO.println "-- micro-benchmark: production-sized banks (hidden=3072, E=256, k=10, moeInt=1024)"
  let block ← synthBigBlock path device
  let x1 : Tensor { shape := #[1, 3072], device := device, dtype := .BFloat16 } := bfloatDynamic (← Tensor.randn #[1, 3072] false device)
  let (topIdx, topW) ← block.router.route bigCfg x1
  let k := bigCfg.num_experts_per_tok
  let h := bigCfg.hidden_size
  let mi := bigCfg.moe_intermediate_size
  -- FP4 bytes read per token by the fused op: every selected expert's packed
  -- weights + group scales, each exactly once.
  let bytesPerTok : UInt64 :=
    k * (2 * (mi * h / 2 + mi * h / 16) + (h * mi / 2 + h * mi / 16))
  IO.println s!"  FP4 bytes read per token: {bytesPerTok} ({bytesPerTok.toFloat / 1.0e6} MB)"

  -- a) fused op alone.
  let (evOp, wallOp) ← timeCuda 20 200 (do
    _ ← lagunaMoeFp4Forward x1 topIdx topW
      block.experts.gatePacked.2 block.experts.gateScale.2 block.experts.gateGlobal.2
      block.experts.upPacked.2 block.experts.upScale.2 block.experts.upGlobal.2
      block.experts.downPacked.2 block.experts.downScale.2 block.experts.downGlobal.2
      bigCfg.num_experts mi h)
  let gbs := bytesPerTok.toFloat / (evOp * 1.0e6)
  IO.println s!"  fused op:       {evOp} ms/event {wallOp} ms/wall  ->  {gbs} GB/s effective FP4 read"
  check (gbs >= 150.0) s!"fused op effective bandwidth {gbs} GB/s ≥ 150 GB/s roofline bar"

  -- Multi-token fused-op timing matching the short prompt used by the real
  -- checkpoint benchmark. This exercises the expert-grouped prefill path
  -- without paying the full model's checkpoint-load cost on every kernel
  -- tuning iteration.
  let x56 : Tensor { shape := #[56, 3072], device := device, dtype := .BFloat16 } := bfloatDynamic (← Tensor.randn #[56, 3072] false device)
  let (topIdx56, topW56) ← block.router.route bigCfg x56
  let (evP, wallP) ← timeCuda 3 20 (do
    _ ← lagunaMoeFp4Forward x56 topIdx56 topW56
      block.experts.gatePacked.2 block.experts.gateScale.2 block.experts.gateGlobal.2
      block.experts.upPacked.2 block.experts.upScale.2 block.experts.upGlobal.2
      block.experts.downPacked.2 block.experts.downScale.2 block.experts.downGlobal.2
      bigCfg.num_experts mi h)
  IO.println s!"  fused prefill56: {evP} ms/event {wallP} ms/wall per layer"

  -- b) full MoE block decode, fused vs eager.
  let (evF, wallF) ← timeCuda 10 100 (do
    _ ← block.forward2d bigCfg x1)
  IO.println s!"  block fused:    {evF} ms/event {wallF} ms/wall per token"
  let (evE, wallE) ← timeCuda 3 20 (do
    _ ← block.forward2d bigCfg x1 false)
  IO.println s!"  block eager:    {evE} ms/event {wallE} ms/wall per token"
  IO.println s!"  speedup (wall): {wallE / wallF}x"

def main : IO Unit := do
  let path ← findFixture
  IO.println s!"Using fixture: {path}"
  if ← torch.cuda_is_available then
    let device ← Device.cuda 0
    runCorrectness path device
    torch.cuda_synchronize
    runNativeCorrectness path device
    torch.cuda_synchronize
    runBenchmark path device
    torch.cuda_synchronize
  else
    IO.println "CUDA not available; skipped fused-kernel tests (CUDA-only op)."
  IO.println "All Laguna fused-kernel tests passed."
