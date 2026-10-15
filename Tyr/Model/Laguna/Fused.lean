/-
  Tyr/Model/Laguna/Fused.lean

  Fused NVFP4 MoE expert forward (CUDA-only), bound to the
  `lean_torch_laguna_moe_fp4_forward` CUDA kernel in
  `cc/src/tyr_laguna_moe.cu`.

  The op computes the routed expert sum for a batch of (token, slot) pairs
  directly from the NVFP4-packed expert banks, without materializing BF16
  weights:

    routed[t] = Σ_p w[t,p] · expert_{idx[t,p]}(x[t])
    expert_e(x) = down( silu(gate x) * up x )

  It returns ONLY the routed sum (`[tokens, hidden]`, BF16); the
  `moe_routed_scaling_factor` multiplication and the shared expert stay in
  Lean (see `LagunaSparseMoeBlock.forwardDecode1` in
  `Tyr/Model/Laguna/MoE.lean`).

  Bank layouts (same as `LagunaPackedExperts`):
  - gate/up: packed U8 `[E, moeInt, hidden/2]`, scales F8_E4M3
    `[E, moeInt, hidden/16]`, globals F32 `[E]`
  - down: packed U8 `[E, hidden, moeInt/2]`, scales F8_E4M3
    `[E, hidden, moeInt/16]`, globals F32 `[E]`

  Backend selection is internal to the C++ operation. Decode, small prefills,
  older CUDA toolkits, and non-SM12x GPUs use the portable weight-streaming
  implementation. Eligible SM12x prefills use native block-scaled W4A4
  cuBLASLt GEMMs with exact compact expert-row batching. Set
  `TYR_LAGUNA_DISABLE_NATIVE_FP4=1` to force the portable path. The Lean API is
  intentionally architecture-neutral so server-Blackwell acceleration can be
  added as another backend without changing model code.

  Preconditions (validated by `lagunaMoeFp4Forward` before crossing the FFI;
  the C++ side re-checks everything and reports IO errors instead of raising
  exceptions that cannot unwind across the boundary):
  - `hidden % 32 == 0` and `moeInt % 32 == 0`
  - `x`/`topW` BF16, `topIdx` Int64, all on CUDA, all contiguous
  - `1 ≤ tokens * k ≤ 65535`
-/
import Tyr.Typed.Neural
import Tyr.RawTensor.Laguna

namespace torch.laguna

variable {device : Device}
private def lagunaMoeFp4Fail {α : Type} (msg : String) : IO α :=
  throw (IO.userError s!"lagunaMoeFp4Forward: {msg}")

private def lagunaMoeFp4CheckShape
    {σ : StaticSpec} (name : String) (t : Tensor σ) (expected : Array UInt64) : IO Unit := do
  if t.runtimeShape != expected then
    lagunaMoeFp4Fail s!"{name} must have shape {expected}, got {t.runtimeShape}"

/-- Fused NVFP4 MoE routed-expert sum on CUDA.
    - `x`: BF16 `[tokens, hidden]`
    - `topIdx`: Int64 `[tokens, k]` (expert ids from the router)
    - `topW`: BF16 `[tokens, k]` (routing weights)
    - the nine packed expert bank tensors (`LagunaPackedExperts` fields)
    Returns BF16 `[tokens, hidden]` with `Σ_p w[t,p] · expert_{idx[t,p]}(x[t])`. -/
def lagunaMoeFp4Forward
    {xs ids ws gp gs gg up us ug dp ds dg : Shape}
    {xd wd gpd gsd ggd upd usd ugd dpd dsd dgd : DType}
    (x : Tensor { device := device, shape := xs, dtype := xd }) (topIdx : Tensor { device := device, shape := ids, dtype := .Int64 }) (topW : Tensor { device := device, shape := ws, dtype := wd })
    (gatePacked : Tensor { device := device, shape := gp, dtype := gpd }) (gateScale : Tensor { device := device, shape := gs, dtype := gsd }) (gateGlobal : Tensor { device := device, shape := gg, dtype := ggd })
    (upPacked : Tensor { device := device, shape := up, dtype := upd }) (upScale : Tensor { device := device, shape := us, dtype := usd }) (upGlobal : Tensor { device := device, shape := ug, dtype := ugd })
    (downPacked : Tensor { device := device, shape := dp, dtype := dpd }) (downScale : Tensor { device := device, shape := ds, dtype := dsd }) (downGlobal : Tensor { device := device, shape := dg, dtype := dgd })
    (numExperts moeInt hidden : UInt64) : IO (Tensor { device := device, shape := xs, dtype := .BFloat16 }) := do
  if hidden == 0 || hidden % 32 != 0 then
    lagunaMoeFp4Fail s!"hidden={hidden} must be a positive multiple of 32"
  if moeInt == 0 || moeInt % 32 != 0 then
    lagunaMoeFp4Fail s!"moeInt={moeInt} must be a positive multiple of 32"
  if numExperts == 0 then
    lagunaMoeFp4Fail "numExperts must be positive"
  match x.device.kind with
  | .CUDA _ => pure ()
  | _ => lagunaMoeFp4Fail "x must be on a CUDA device"
  if x.actualSpec.dtype != .BFloat16 then lagunaMoeFp4Fail "x must be BF16"
  if topIdx.actualSpec.dtype != .Int64 then lagunaMoeFp4Fail "topIdx must be Int64"
  if topW.actualSpec.dtype != .BFloat16 then lagunaMoeFp4Fail "topW must be BF16"
  if gatePacked.actualSpec.dtype != .UInt8 || upPacked.actualSpec.dtype != .UInt8 || downPacked.actualSpec.dtype != .UInt8 then
    lagunaMoeFp4Fail "packed banks must be UInt8"
  if gateGlobal.actualSpec.dtype != .Float32 || upGlobal.actualSpec.dtype != .Float32 || downGlobal.actualSpec.dtype != .Float32 then
    lagunaMoeFp4Fail "global scales must be Float32"
  if gateScale.actualSpec.dtype != .Float8E4M3FN || upScale.actualSpec.dtype != .Float8E4M3FN || downScale.actualSpec.dtype != .Float8E4M3FN then
    lagunaMoeFp4Fail "block scales must be Float8E4M3FN"
  let xs := x.runtimeShape
  if xs.size != 2 || xs.getD 1 0 != hidden then
    lagunaMoeFp4Fail s!"x must be [tokens, hidden={hidden}], got {xs}"
  let tokens := xs.getD 0 0
  let is := topIdx.runtimeShape
  if is.size != 2 || is.getD 0 0 != tokens then
    lagunaMoeFp4Fail s!"topIdx must be [tokens={tokens}, k], got {is}"
  let k := is.getD 1 0
  if topW.runtimeShape != is then
    lagunaMoeFp4Fail s!"topW must match topIdx shape {is}, got {topW.runtimeShape}"
  if tokens == 0 || k == 0 || tokens * k > 65535 then
    lagunaMoeFp4Fail s!"require 1 ≤ tokens*k ≤ 65535, got tokens={tokens} k={k}"
  lagunaMoeFp4CheckShape "gatePacked" gatePacked #[numExperts, moeInt, hidden / 2]
  lagunaMoeFp4CheckShape "gateScale" gateScale #[numExperts, moeInt, hidden / 16]
  lagunaMoeFp4CheckShape "gateGlobal" gateGlobal #[numExperts]
  lagunaMoeFp4CheckShape "upPacked" upPacked #[numExperts, moeInt, hidden / 2]
  lagunaMoeFp4CheckShape "upScale" upScale #[numExperts, moeInt, hidden / 16]
  lagunaMoeFp4CheckShape "upGlobal" upGlobal #[numExperts]
  lagunaMoeFp4CheckShape "downPacked" downPacked #[numExperts, hidden, moeInt / 2]
  lagunaMoeFp4CheckShape "downScale" downScale #[numExperts, hidden, moeInt / 16]
  lagunaMoeFp4CheckShape "downGlobal" downGlobal #[numExperts]
  let raw ← RawTensor.lagunaMoeFp4Forward x.raw topIdx.raw topW.raw
    gatePacked.raw gateScale.raw gateGlobal.raw
    upPacked.raw upScale.raw upGlobal.raw
    downPacked.raw downScale.raw downGlobal.raw
    numExperts moeInt hidden
  match Tensor.ofRaw raw with
  | .ok result => pure result
  | .error message => lagunaMoeFp4Fail message

end torch.laguna
