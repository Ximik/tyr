/- High-level FlashAttention/SDPA wrappers with typed dispatch and dynamic adapters. -/
import Tyr.Torch
import Tyr.Typed.Neural
import Tyr.GPU.Ops.AttentionProblem
import Tyr.GPU.Ops.MhaH100

namespace Tyr.GPU.Ops.FlashAttn

open torch

variable {device : Device}
open Tyr.GPU.Ops

/-- Query (device := device) layout for the typed FlashAttention surface: `[batch, nHead, qSeq, headDim]`
    with the dtype tracked statically. -/
abbrev Query (batch nHead qSeq headDim : UInt64) (d : DType) : Type :=
  Tensor { shape := #[batch, nHead, qSeq, headDim], dtype := d, device := device }
/-- Key/value layout: `[batch, nKvHead, kvSeq, headDim]`, sharing the query dtype. -/
abbrev KeyValue (batch nKvHead kvSeq headDim : UInt64) (d : DType) : Type :=
  Tensor { shape := #[batch, nKvHead, kvSeq, headDim], dtype := d, device := device }
/-- Padding-mask layout with an existential exact dtype. -/
abbrev PaddingMask (batch kvSeq : UInt64) : Type := SomeDTypeTensor #[batch, kvSeq] device

inductive DispatchRoute where
  | tkKernel
  | portable
  deriving Repr, Inhabited, BEq

/-- Reusable runtime problem descriptor for the typed FlashAttention surface. -/
def attentionProblem
    {batch nHead nKvHead qSeq kvSeq headDim : UInt64} {d : DType}
    (q : Query (device := device) batch nHead qSeq headDim d)
    (k : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (v : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (attnMask : Option (PaddingMask (device := device) batch kvSeq) := none)
    (dropoutP : Float := 0.0)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    : AttentionProblem :=
  AttentionProblem.ofQKV q k v attnMask dropoutP isCausal scale enableGqa none .SM90

/-- Current native specialization eligibility for the typed FlashAttention surface. -/
def currentSpecialization
    {batch nHead nKvHead qSeq kvSeq headDim : UInt64} {d : DType}
    (q : Query (device := device) batch nHead qSeq headDim d)
    (k : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (v : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (attnMask : Option (PaddingMask (device := device) batch kvSeq) := none)
    (dropoutP : Float := 0.0)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    : AttentionSpecialization :=
  AttentionProblem.currentSpecialization <|
    attentionProblem q k v attnMask dropoutP isCausal scale enableGqa

/-- Typed route classifier for current TK-backed coverage. -/
def supportsTkMhaKernel
    {batch nHead nKvHead qSeq kvSeq headDim : UInt64} {d : DType}
    (q : Query (device := device) batch nHead qSeq headDim d)
    (k : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (v : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (attnMask : Option (PaddingMask (device := device) batch kvSeq) := none)
    (dropoutP : Float := 0.0)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    : Bool :=
  (currentSpecialization q k v attnMask dropoutP isCausal scale enableGqa).isNative

/-- Typed dispatch route (kernel vs portable) for observability/debugging. -/
def dispatchRoute
    {batch nHead nKvHead qSeq kvSeq headDim : UInt64} {d : DType}
    (q : Query (device := device) batch nHead qSeq headDim d)
    (k : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (v : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (attnMask : Option (PaddingMask (device := device) batch kvSeq) := none)
    (dropoutP : Float := 0.0)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    : DispatchRoute :=
  match currentSpecialization q k v attnMask dropoutP isCausal scale enableGqa with
  | .portable => .portable
  | _ => .tkKernel

/-- Pure FlashAttention/SDPA with zero dropout.
    This routes through a torch-registered op (`tyr::flash_attn`) so autograd remains native.
    The indexed adapter checks that native output metadata matches the query. -/
@[inline] def flashAttn
    {batch nHead nKvHead qSeq kvSeq headDim : UInt64} {d : DType}
    (query : Query (device := device) batch nHead qSeq headDim d)
    (key : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (value : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (attnMask : Option (PaddingMask (device := device) batch kvSeq) := none)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    : Query (device := device) batch nHead qSeq headDim d :=
  Tensor.flashAttention query key value enableGqa attnMask isCausal scale

/-- Effectful FlashAttention/SDPA. Training uses the configured dropout;
evaluation passes zero dropout to the native IO operation. -/
@[inline] def flashAttnIO
    {batch nHead nKvHead qSeq kvSeq headDim : UInt64} {d : DType}
    (query : Query (device := device) batch nHead qSeq headDim d)
    (key : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (value : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (dropoutP : Float) (training : Bool)
    (attnMask : Option (PaddingMask (device := device) batch kvSeq) := none)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    : IO (Query (device := device) batch nHead qSeq headDim d) :=
  Tensor.flashAttentionIO query key value dropoutP training enableGqa attnMask isCausal scale

/-- Alias matching PyTorch naming. -/
@[inline] def scaledDotProductAttention
    {batch nHead nKvHead qSeq kvSeq headDim : UInt64} {d : DType}
    (query : Query (device := device) batch nHead qSeq headDim d)
    (key : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (value : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (attnMask : Option (PaddingMask (device := device) batch kvSeq) := none)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    : Query (device := device) batch nHead qSeq headDim d :=
  flashAttn query key value attnMask isCausal scale enableGqa

/-- Effectful alias matching PyTorch naming, with explicit training mode. -/
@[inline] def scaledDotProductAttentionIO
    {batch nHead nKvHead qSeq kvSeq headDim : UInt64} {d : DType}
    (query : Query (device := device) batch nHead qSeq headDim d)
    (key : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (value : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (dropoutP : Float) (training : Bool)
    (attnMask : Option (PaddingMask (device := device) batch kvSeq) := none)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    : IO (Query (device := device) batch nHead qSeq headDim d) :=
  flashAttnIO query key value dropoutP training attnMask isCausal scale enableGqa

/-- Typed API returning both the dispatch route and the output. -/
def flashAttnWithRoute
    {batch nHead nKvHead qSeq kvSeq headDim : UInt64} {d : DType}
    (query : Query (device := device) batch nHead qSeq headDim d)
    (key : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (value : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (attnMask : Option (PaddingMask (device := device) batch kvSeq) := none)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    : DispatchRoute × Query (device := device) batch nHead qSeq headDim d :=
  let route := dispatchRoute query key value attnMask 0.0 isCausal scale enableGqa
  let out := flashAttn query key value attnMask isCausal scale enableGqa
  (route, out)

/-- Effectful attention and its route. The descriptor uses the same effective
dropout as execution, including zero dropout during evaluation. -/
def flashAttnWithRouteIO
    {batch nHead nKvHead qSeq kvSeq headDim : UInt64} {d : DType}
    (query : Query (device := device) batch nHead qSeq headDim d)
    (key : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (value : KeyValue (device := device) batch nKvHead kvSeq headDim d)
    (dropoutP : Float) (training : Bool)
    (attnMask : Option (PaddingMask (device := device) batch kvSeq) := none)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    : IO (DispatchRoute × Query (device := device) batch nHead qSeq headDim d) := do
  let effectiveDropout := if training then dropoutP else 0.0
  let route := dispatchRoute query key value attnMask effectiveDropout isCausal scale enableGqa
  let out ← flashAttnIO query key value dropoutP training attnMask isCausal scale enableGqa
  pure (route, out)

structure PackedQKV where
  batch : UInt64
  nHead : UInt64
  nKvHead : UInt64
  qSeq : UInt64
  kvSeq : UInt64
  headDim : UInt64
  dtype : ExactDType
  q : Tensor { shape := #[batch, nHead, qSeq, headDim], dtype := dtype.val, device := device }
  k : Tensor { shape := #[batch, nKvHead, kvSeq, headDim], dtype := dtype.val, device := device }
  v : Tensor { shape := #[batch, nKvHead, kvSeq, headDim], dtype := dtype.val, device := device }

abbrev DynOut :=
  Sigma (fun batch =>
    Sigma (fun nHead =>
      Sigma (fun qSeq =>
        Sigma (fun headDim =>
          SomeDTypeTensor #[batch, nHead, qSeq, headDim] device))))

private def checkedRaw {σ : StaticSpec} (label : String) (raw : RawTensor) : IO (Tensor σ) :=
  match Tensor.ofRaw raw with
  | .ok tensor => pure tensor
  | .error message => throw <| IO.userError s!"flashAttnDyn: {label}: {message}"

private def parseQKVDyn (qDyn kDyn vDyn : RawTensor) : IO (PackedQKV (device := qDyn.device)) := do
  let device := qDyn.device
  let qShape := qDyn.runtimeShape
  if qShape.size != 4 then
    throw <| IO.userError s!"flashAttnDyn: query must be rank-4, got shape={qShape}"
  let kShape := kDyn.runtimeShape
  if kShape.size != 4 then
    throw <| IO.userError s!"flashAttnDyn: key must be rank-4, got shape={kShape}"
  let vShape := vDyn.runtimeShape
  if vShape.size != 4 then
    throw <| IO.userError s!"flashAttnDyn: value must be rank-4, got shape={vShape}"

  let batch := qShape.getD 0 0
  let nHead := qShape.getD 1 0
  let qSeq := qShape.getD 2 0
  let headDim := qShape.getD 3 0
  let kBatch := kShape.getD 0 0
  let nKvHead := kShape.getD 1 0
  let kvSeq := kShape.getD 2 0
  let kHeadDim := kShape.getD 3 0
  let vBatch := vShape.getD 0 0
  let vKvHead := vShape.getD 1 0
  let vKvSeq := vShape.getD 2 0
  let vHeadDim := vShape.getD 3 0

  if !(kBatch == batch && vBatch == batch) then
    throw <| IO.userError s!"flashAttnDyn: batch mismatch q={batch} k={kBatch} v={vBatch}"
  if !(vKvHead == nKvHead) then
    throw <| IO.userError s!"flashAttnDyn: K/V head mismatch k={nKvHead} v={vKvHead}"
  if !(vKvSeq == kvSeq) then
    throw <| IO.userError s!"flashAttnDyn: K/V sequence mismatch k={kvSeq} v={vKvSeq}"
  if !(kHeadDim == headDim && vHeadDim == headDim) then
    throw <| IO.userError s!"flashAttnDyn: head_dim mismatch q={headDim} k={kHeadDim} v={vHeadDim}"

  let dtype ← IO.ofExcept (ExactDType.ofDType? qDyn.dtype)
  let q : Tensor { shape := #[batch, nHead, qSeq, headDim], dtype := dtype.val, device := device } ← checkedRaw "query" qDyn
  let k : Tensor { shape := #[batch, nKvHead, kvSeq, headDim], dtype := dtype.val, device := device } ← checkedRaw "key" kDyn
  let v : Tensor { shape := #[batch, nKvHead, kvSeq, headDim], dtype := dtype.val, device := device } ← checkedRaw "value" vDyn
  pure {
    batch := batch
    nHead := nHead
    nKvHead := nKvHead
    qSeq := qSeq
    kvSeq := kvSeq
    headDim := headDim
    dtype := dtype
    q := q
    k := k
    v := v
  }

private def parseMaskDyn
    (batch kvSeq : UInt64)
    (attnMask : Option RawTensor)
    : IO (Option (PaddingMask (device := device) batch kvSeq)) := do
  match attnMask with
  | none => pure none
  | some maskDyn =>
      let maskShape := maskDyn.runtimeShape
      if maskShape.size != 2 then
        throw <| IO.userError s!"flashAttnDyn: attn_mask must be rank-2 [batch, kv_seq], got shape={maskShape}"
      let gotBatch := maskShape.getD 0 0
      let gotKvSeq := maskShape.getD 1 0
      if !(gotBatch == batch && gotKvSeq == kvSeq) then
        throw <| IO.userError
          s!"flashAttnDyn: attn_mask shape mismatch expected=[{batch},{kvSeq}] got=[{gotBatch},{gotKvSeq}]"
      let mask : PaddingMask (device := device) batch kvSeq ← IO.ofExcept (SomeDTypeTensor.ofRaw maskDyn)
      pure (some mask)

/-- Fully general existential adapter:
    accepts shape-erased 4D raw tensors, validates runtime shapes, dispatches through the
    typed API, and returns a shape-indexed existential output. -/
def flashAttnDyn
    (qDyn kDyn vDyn : RawTensor)
    (dropoutP : Float) (training : Bool)
    (attnMask : Option RawTensor := none)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    : IO (DynOut (device := qDyn.device)) := do
  let device := qDyn.device
  let packed ← parseQKVDyn qDyn kDyn vDyn
  let typedMask ← parseMaskDyn (device := device) packed.batch packed.kvSeq attnMask
  let out : Tensor { shape := #[packed.batch, packed.nHead, packed.qSeq, packed.headDim], dtype := packed.dtype.val, device := device } ←
    flashAttnIO packed.q packed.k packed.v dropoutP training typedMask isCausal scale enableGqa
  pure ⟨packed.batch, ⟨packed.nHead, ⟨packed.qSeq, ⟨packed.headDim, ⟨packed.dtype, out⟩⟩⟩⟩⟩

end Tyr.GPU.Ops.FlashAttn
