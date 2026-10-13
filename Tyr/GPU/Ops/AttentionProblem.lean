/- Runtime attention problem description and current specialization selection.

   This is the thin operator-facing layer that should sit above concrete kernel
   families. Public wrappers should construct an `AttentionProblem`, let this
   module choose the best available specialization, and only then dispatch to a
   generated kernel or a portable fallback.

   The current specialization matrix is intentionally small:

   - `tkMhaH100Decode`   : one-H100, BF16, decode (`qSeq=1`), runtime heads/KV length/head dim
   - `tkMhaH1002Block`   : one-H100, BF16, dense prefill, `seq=128`, `headDim=64`
   - `tkMhaH10012Block`  : one-H100, BF16, dense prefill, `seq=768`, `headDim=64`
   - `portable`          : everything else

   The key design point is that the public API is now centered on a runtime
   problem descriptor rather than a handful of ad hoc fixed-shape predicates.
-/
import Tyr.Typed.Ops
import Tyr.GPU.Types

namespace Tyr.GPU.Ops

open torch
open Tyr.GPU

/-- High-level attention execution mode. -/
inductive AttentionMode where
  | densePrefill
  | decode
  | slidingWindow
  deriving Repr, Inhabited, BEq

/-- Mask representation class exposed at the runtime surface. -/
inductive AttentionMaskKind where
  | none
  | padding
  | explicitQK
  deriving Repr, Inhabited, BEq

/-- Coarse head-dimension bucket used for specialization planning. -/
inductive HeadDimClass where
  | d64
  | d128
  | d256
  | d512
  | other (dim : UInt64)
  deriving Repr, Inhabited, BEq

/-- Grouped-query structure class. `grouped r` means `numQHeads / numKVHeads = r`. -/
inductive GqaClass where
  | equal
  | grouped (ratio : UInt64)
  | invalid
  deriving Repr, Inhabited, BEq

/-- Coarse scaling-policy bucket used for future family-based routing. -/
inductive ScaleClass where
  | implicitDefault
  | explicitDefault
  | custom
  deriving Repr, Inhabited, BEq

/-- Incremental family-routing metadata view derived from `AttentionProblem`.

    This is intentionally additive. Existing selectors can keep using the raw
    `AttentionProblem` fields while newer family-based dispatch code can consume
    a stable classification surface. -/
structure AttentionRoutingMetadata where
  headDimClass : HeadDimClass
  gqaRatio : Option UInt64
  execMode : AttentionMode
  scaleClass : ScaleClass
  qTiles : UInt64
  kvTiles : UInt64
  windowTiles : Option UInt64
  deriving Repr, Inhabited, BEq

/-- Current native specialization set known to the runtime operator layer. -/
inductive AttentionSpecialization where
  | portable
  | tkMhaH100Decode
  | tkMhaH1002Block
  | tkMhaH10012Block
  deriving Repr, Inhabited, BEq

namespace AttentionSpecialization

/-- Whether the selection corresponds to a native generated-kernel path. -/
def isNative : AttentionSpecialization → Bool
  | .portable => false
  | _ => true

/-- Tile-block count for the current H100 MHA specializations, when applicable. -/
def kvBlocks? : AttentionSpecialization → Option UInt64
  | .portable => none
  | .tkMhaH100Decode => none
  | .tkMhaH1002Block => some 2
  | .tkMhaH10012Block => some 12

end AttentionSpecialization

/-- Runtime description of an attention call.

    Fields are intentionally broader than the current native kernel set so this
    type can survive future backend growth without changing the public surface. -/
structure AttentionProblem where
  batch : UInt64
  numQHeads : UInt64
  numKVHeads : UInt64
  qSeq : UInt64
  kvSeq : UInt64
  headDim : UInt64
  dtype : DType
  device : Device
  arch : GpuArch := .SM90
  mode : AttentionMode := .densePrefill
  maskKind : AttentionMaskKind := .none
  dropoutP : Float := 0.0
  isCausal : Bool := false
  scale : Option Float := none
  enableGqa : Bool := false
  windowSize : Option UInt64 := none
  deriving Repr, Inhabited

namespace AttentionProblem

/-- Ceiling division for positive tile sizes. Returns `0` when `d = 0`. -/
private def ceilDiv (n d : UInt64) : UInt64 :=
  if d == 0 then
    0
  else
    (n + d - 1) / d

/-- Infer the broad attention mode from sequence lengths and optional windowing. -/
def inferMode (qSeq kvSeq : UInt64) (windowSize : Option UInt64 := none) : AttentionMode :=
  match windowSize with
  | some _ => .slidingWindow
  | none =>
      if qSeq == 1 && kvSeq >= qSeq then
        .decode
      else
        .densePrefill

/-- Construct a runtime problem descriptor from typed Q/K/V tensors.
    The Q/K/V dtype is shared statically; the descriptor still records the
    runtime dtype/device of the query handle, as before. -/
def ofQKV
    {batch nHead nKvHead qSeq kvSeq headDim : UInt64} {d : DType}
    (q : DTensor #[batch, nHead, qSeq, headDim] d)
    (k : DTensor #[batch, nKvHead, kvSeq, headDim] d)
    (v : DTensor #[batch, nKvHead, kvSeq, headDim] d)
    (attnMask : Option (STensor #[batch, kvSeq]) := none)
    (dropoutP : Float := 0.0)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    (windowSize : Option UInt64 := none)
    (arch : GpuArch := .SM90)
    : AttentionProblem :=
    {
      batch := batch
      numQHeads := nHead
      numKVHeads := nKvHead
      qSeq := qSeq
      kvSeq := kvSeq
      headDim := headDim
      dtype := q.raw.dtype
      device := q.raw.device
      arch := arch
      mode := inferMode qSeq kvSeq windowSize
      maskKind := if attnMask.isSome then .padding else .none
      dropoutP := dropoutP
      isCausal := isCausal
      scale := scale
      enableGqa := enableGqa
      windowSize := windowSize
    }

/-- Shape-only helper for fixed-layout self-attention surfaces. -/
def selfAttention
    (seq : UInt64)
    (headDim : UInt64 := 64)
    (device : Device := .CUDA 0)
    (dtype : DType := .BFloat16)
    (isCausal : Bool := false)
    (arch : GpuArch := .SM90)
    : AttentionProblem := {
      batch := 1
      numQHeads := 1
      numKVHeads := 1
      qSeq := seq
      kvSeq := seq
      headDim := headDim
      dtype := dtype
      device := device
      arch := arch
      mode := inferMode seq seq none
      isCausal := isCausal
    }

/-- Whether the problem is placed on a CUDA device. -/
def isCuda (problem : AttentionProblem) : Bool :=
  match problem.device with
  | .CUDA _ => true
  | _ => false

/-- Whether query and KV sequence lengths match. -/
def isSelfAttention (problem : AttentionProblem) : Bool :=
  problem.qSeq == problem.kvSeq

/-- Whether `qSeq` and `kvSeq` are aligned to the current tile size. -/
def tileAligned (problem : AttentionProblem) (tile : UInt64 := 64) : Bool :=
  problem.qSeq % tile == 0 && problem.kvSeq % tile == 0

/-- Whether the provided scale matches the standard `1 / sqrt(headDim)` default. -/
def scaleMatchesDefault (problem : AttentionProblem) : Bool :=
  match problem.scale with
  | none => true
  | some s =>
      let expected := 1.0 / Float.sqrt problem.headDim.toFloat
      Float.abs (s - expected) <= 1.0e-6

/-- Coarse head-dimension bucket used by specialization planning. -/
def headDimClass (problem : AttentionProblem) : HeadDimClass :=
  match problem.headDim with
  | 64 => .d64
  | 128 => .d128
  | 256 => .d256
  | 512 => .d512
  | d => .other d

/-- Grouped-query ratio class derived from the runtime head counts. -/
def gqaClass (problem : AttentionProblem) : GqaClass :=
  if problem.numKVHeads == 0 then
    .invalid
  else if problem.numQHeads == problem.numKVHeads then
    .equal
  else if problem.numQHeads % problem.numKVHeads == 0 then
    .grouped (problem.numQHeads / problem.numKVHeads)
  else
    .invalid

/-- Exact grouped-query ratio when the runtime head counts are well-formed. -/
def gqaRatio (problem : AttentionProblem) : Option UInt64 :=
  match problem.gqaClass with
  | .equal => some 1
  | .grouped ratio => some ratio
  | .invalid => none

/-- Execution-mode accessor retained separately from the raw `mode` field so
    future routing code can talk in terms of a stable classification surface. -/
def execMode (problem : AttentionProblem) : AttentionMode :=
  problem.mode

/-- Coarse scaling-policy bucket for family-level dispatch decisions. -/
def scaleClass (problem : AttentionProblem) : ScaleClass :=
  match problem.scale with
  | none => .implicitDefault
  | some _ =>
      if problem.scaleMatchesDefault then
        .explicitDefault
      else
        .custom

/-- Query tile count at the current planning granularity. -/
def qTiles (problem : AttentionProblem) (tile : UInt64 := 64) : UInt64 :=
  ceilDiv problem.qSeq tile

/-- KV tile count at the current planning granularity. -/
def kvTiles (problem : AttentionProblem) (tile : UInt64 := 64) : UInt64 :=
  ceilDiv problem.kvSeq tile

/-- Optional windowed-attention tile count at the current planning granularity. -/
def windowTiles (problem : AttentionProblem) (tile : UInt64 := 64) : Option UInt64 :=
  problem.windowSize.map (fun window => ceilDiv window tile)

/-- Stable family-routing metadata view derived from the full runtime problem. -/
def routingMetadata (problem : AttentionProblem) (tile : UInt64 := 64) : AttentionRoutingMetadata := {
  headDimClass := problem.headDimClass
  gqaRatio := problem.gqaRatio
  execMode := problem.execMode
  scaleClass := problem.scaleClass
  qTiles := problem.qTiles tile
  kvTiles := problem.kvTiles tile
  windowTiles := problem.windowTiles tile
}

/-- Whether the current one-H100 TK-backed selector may consider this problem. -/
def currentTkBaseEligible (problem : AttentionProblem) : Bool :=
  let deviceOk := problem.isCuda
  let dtypeOk := problem.dtype == .BFloat16
  let modeOk := problem.mode == .densePrefill
  let semanticsOk :=
    problem.maskKind == .none &&
    problem.dropoutP == 0.0 &&
    !problem.enableGqa &&
    problem.scaleMatchesDefault
  let shapeOk :=
    problem.batch == 1 &&
    problem.numQHeads == 1 &&
    problem.numKVHeads == 1 &&
    problem.isSelfAttention &&
    problem.headDim == 64 &&
    !problem.isCausal
  deviceOk && dtypeOk && modeOk && semanticsOk && shapeOk

/-- Whether the current one-H100 decode selector may consider this problem.

    V1 of the TK-style decode kernel supports any positive `kvSeq` (the kernel
    iterates `ceil(kvSeq / 64)` blocks and applies a runtime tail mask via
    ThunderKittens' `right_fill`). `headDim` is one of {64, 128, 256} — the
    C++ launcher dispatches to `tkMhaH100DecodeFwd`, `tkMhaH100DecodeFwd64`,
    or `tkMhaH100DecodeFwd256` based on the tensor's head dim. head_dim=256
    targets the Qwen 3.5/3.6 family and Gemma-2 27B; other head dims fall
    back to portable SDPA. -/
def currentTkDecodeEligible (problem : AttentionProblem) : Bool :=
  let deviceOk := problem.isCuda && problem.arch == .SM90
  let dtypeOk := problem.dtype == .BFloat16
  let modeOk := problem.mode == .decode && problem.qSeq == 1
  let gqaOk :=
    match problem.gqaClass with
    | .equal => true
    | .grouped _ => problem.enableGqa
    | .invalid => false
  let semanticsOk :=
    problem.maskKind == .none &&
    problem.dropoutP == 0.0 &&
    !problem.isCausal &&
    problem.scaleMatchesDefault
  let shapeOk :=
    problem.batch > 0 &&
    problem.numQHeads > 0 &&
    problem.numKVHeads > 0 &&
    problem.kvSeq > 0 &&
    (problem.headDim == 128 || problem.headDim == 64 || problem.headDim == 256) &&
    gqaOk
  deviceOk && dtypeOk && modeOk && semanticsOk && shapeOk

/-- Current specialization selector.

    This is intentionally conservative. It centralizes the existing native
    coverage and portable fallback decision in one place so the wrappers do not
    each re-encode the same fixed-shape logic. -/
def currentSpecialization (problem : AttentionProblem) : AttentionSpecialization :=
  if currentTkDecodeEligible problem then
    .tkMhaH100Decode
  else if !currentTkBaseEligible problem then
    .portable
  else if problem.qSeq == 128 then
    .tkMhaH1002Block
  else if problem.qSeq == 768 then
    .tkMhaH10012Block
  else
    .portable

/-- Whether the current runtime layer has a native specialization for `problem`. -/
def hasNativeKernel (problem : AttentionProblem) : Bool :=
  (currentSpecialization problem).isNative

end AttentionProblem

end Tyr.GPU.Ops
