/-
  Tyr/Model/Laguna/NvFp4.lean

  NVFP4 dequantization for the poolside Laguna-S-2.1-NVFP4 checkpoint
  (compressed-tensors `nvfp4-pack-quantized` format).

  Format (verified against the real checkpoint):
  - `weight_packed`: U8 `[out, in/2]`, two E2M1 values per byte along dim -1;
    LOW nibble = element `2i`, HIGH nibble = element `2i+1`.
  - `weight_scale`: F8_E4M3 `[out, in/16]`, one scale per 16-element group
    along the input dim.
  - `weight_global_scale`: F32 scalar (one per expert for stacked banks).
  - Dequant: `W[i,j] = e2m1(nibble) * (scale[i, j/16].to(float32) / globalScale)`.
    E2M1: bits 0-2 index into magnitudes `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`,
    bit 3 is the sign.

  Implementation notes:
  - Nibble unpacking uses a byte-level lookup table (`e2m1LutLo`/`e2m1LutHi`,
    256 F32 entries each) plus an embedding gather, avoiding per-element bit
    math (`index_select_1d` would be the natural choice but its FFI binding is
    ABI-mismatched and segfaults; see `dequantCore`).
  - F8_E4M3 scales are cast to F32 with the libtorch `.to(kFloat32)` cast
    (`toFloat'`), which handles float8 dtypes in this vendored build.
  - The arithmetic is done in F32 in the same op order as the reference
    (`e2m1 * (scale / global)`), then rounded once to BF16, so results are
    bitwise-identical to a PyTorch F32 dequant followed by `.to(bfloat16)`.
-/
import Tyr.Typed.Neural

namespace torch.laguna.nvfp4

variable {device : Device}
/-- E2M1 magnitude table indexed by the low 3 bits of a nibble. -/
private def e2m1Magnitudes : Array Float :=
  #[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

/-- Decode one E2M1 nibble (bits 0-2 magnitude index, bit 3 sign). -/
private def e2m1Value (nibble : Nat) : Float :=
  let mag := e2m1Magnitudes.getD (nibble % 8) 0.0
  if nibble >= 8 then -mag else mag

/-- F32 lookup table for the LOW nibble (element `2i`) of each byte. -/
private def e2m1LutLo : Tensor { device := .CPU, shape := #[256], dtype := .Float32 } :=
  (Tensor.fromFloatArray (Array.ofFn (n := 256) fun b => e2m1Value (b.val % 16))).reshapeChecked #[256]

/-- F32 lookup table for the HIGH nibble (element `2i+1`) of each byte. -/
private def e2m1LutHi : Tensor { device := .CPU, shape := #[256], dtype := .Float32 } :=
  (Tensor.fromFloatArray (Array.ofFn (n := 256) fun b => e2m1Value (b.val / 16))).reshapeChecked #[256]

/-- Shared dequant kernel.

    - `packed`: U8 tensor, runtime shape `[lead, out, in/2]`
    - `scales`: F8_E4M3 tensor, runtime shape `[lead, out, in/16]`
    - `globalB`: F32 tensor broadcastable against `[lead, 1, 1, 1]`
      (i.e. one global scale per leading slice)

    Returns BF16 tensor of runtime shape `[lead, out, in]`. -/
private def dequantCore {pd sd : DType}
    (lead outFeatures inFeatures : UInt64)
    (packed : Tensor { device := device, shape := #[lead, outFeatures, inFeatures / 2], dtype := pd })
    (scales : Tensor { device := device, shape := #[lead, outFeatures, inFeatures / 16], dtype := sd })
    (globalB : Tensor { device := device, shape := #[lead, 1, 1, 1], dtype := .Float32 })
    : IO (Tensor { device := device, shape := #[lead, outFeatures, inFeatures], dtype := .BFloat16 }) := do
  if inFeatures % 16 != 0 then
    throw <| IO.userError
      s!"nvfp4.dequantCore: inFeatures={inFeatures} not divisible by 16"
  let numGroups := inFeatures / 16
  let numPacked := lead * outFeatures * (inFeatures / 2)
  let device := packed.device
  let lutLo := (e2m1LutLo.to device).reshapeChecked #[256, 1]
  let lutHi := (e2m1LutHi.to device).reshapeChecked #[256, 1]
  let idx := (packed.reshapeChecked #[numPacked]).toInt64
  let lo := (Tensor.embedding1d idx lutLo).reshapeChecked #[numPacked, 1]
  let hi := (Tensor.embedding1d idx lutHi).reshapeChecked #[numPacked, 1]
  let pairs := Tensor.catManyChecked #[lo.packShape, hi.packShape] 1 #[numPacked, 2]
  let vals4 := pairs.reshapeChecked #[lead, outFeatures, numGroups, 16]
  let scalesF := scales.toFloat32.reshapeChecked #[lead, outFeatures, numGroups, 1]
  -- Preserve reference rounding: divide in F32, multiply in F32, then cast once.
  let globalExpanded := Tensor.expand globalB #[lead, outFeatures, numGroups, 1]
  let scaleDiv := scalesF / globalExpanded
  let prod4 := (Tensor.expand scaleDiv #[lead, outFeatures, numGroups, 16]) * vals4
  pure (prod4.reshapeChecked #[lead, outFeatures, inFeatures]).toBFloat16

/-- Validate the caller's indexed input shape before reshaping a packed bank. -/
private def checkShape {s : Shape} {d : DType}
    (name : String) (_tensor : Tensor { device := device, shape := s, dtype := d }) (expected : Shape) : IO Unit := do
  if s != expected then
    throw <| IO.userError s!"nvfp4: {name} must have shape {expected}, got {s}"

/-- Dequantize an NVFP4 matrix into a shape- and BF16-indexed result.
Packed bytes and float8 scales retain their actual checkpoint dtype; their
shapes are validated before any reshape. Global scale must contain one value. -/
def dequantMatrix {ps ss gs : Shape} {pd sd gd : DType}
    (packed : Tensor { device := device, shape := ps, dtype := pd }) (scales : Tensor { device := device, shape := ss, dtype := sd }) (globalScale : Tensor { device := device, shape := gs, dtype := gd })
    (outFeatures inFeatures : UInt64)
    : IO (Tensor { device := device, shape := #[outFeatures, inFeatures], dtype := .BFloat16 }) := do
  checkShape "packed" packed #[outFeatures, inFeatures / 2]
  checkShape "scales" scales #[outFeatures, inFeatures / 16]
  if gs != #[] && gs != #[1] then
    throw <| IO.userError s!"nvfp4: global scale must be scalar or [1], got {gs}"
  let globalB := globalScale.toFloat32.reshapeChecked #[1, 1, 1, 1]
  let packed3 := packed.reshapeChecked #[1, outFeatures, inFeatures / 2]
  let scales3 := scales.reshapeChecked #[1, outFeatures, inFeatures / 16]
  let out3 ← dequantCore 1 outFeatures inFeatures packed3 scales3 globalB
  pure (out3.reshapeChecked #[outFeatures, inFeatures])

/-- Dequantize an NVFP4 expert bank, preserving expert and matrix dimensions. -/
def dequantBank {ps ss gs : Shape} {pd sd gd : DType}
    (packed : Tensor { device := device, shape := ps, dtype := pd }) (scales : Tensor { device := device, shape := ss, dtype := sd }) (globalScale : Tensor { device := device, shape := gs, dtype := gd })
    (numExperts outFeatures inFeatures : UInt64)
    : IO (Tensor { device := device, shape := #[numExperts, outFeatures, inFeatures], dtype := .BFloat16 }) := do
  checkShape "packed" packed #[numExperts, outFeatures, inFeatures / 2]
  checkShape "scales" scales #[numExperts, outFeatures, inFeatures / 16]
  checkShape "global scale" globalScale #[numExperts]
  let globalB := globalScale.toFloat32.reshapeChecked #[numExperts, 1, 1, 1]
  dequantCore numExperts outFeatures inFeatures
    (packed.reshapeChecked #[numExperts, outFeatures, inFeatures / 2])
    (scales.reshapeChecked #[numExperts, outFeatures, inFeatures / 16]) globalB

end torch.laguna.nvfp4
