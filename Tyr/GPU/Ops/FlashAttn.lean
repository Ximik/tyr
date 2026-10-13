/- High-level FlashAttention/SDPA wrappers with typed dispatch and dynamic adapters. -/
import Tyr.Torch
import Tyr.Typed.Ops
import Tyr.GPU.Ops.AttentionProblem
import Tyr.GPU.Ops.MhaH100

namespace Tyr.GPU.Ops.FlashAttn

open torch
open Tyr.GPU.Ops

/-- Query layout for the typed FlashAttention surface: `[batch, nHead, qSeq, headDim]`
    with the dtype tracked statically. -/
abbrev Query (batch nHead qSeq headDim : UInt64) (d : DType) : Type :=
  DTensor #[batch, nHead, qSeq, headDim] d
/-- Key/value layout: `[batch, nKvHead, kvSeq, headDim]`, sharing the query dtype. -/
abbrev KeyValue (batch nKvHead kvSeq headDim : UInt64) (d : DType) : Type :=
  DTensor #[batch, nKvHead, kvSeq, headDim] d
/-- Padding-mask layout: `[batch, kvSeq]`; the mask dtype stays dynamic. -/
abbrev PaddingMask (batch kvSeq : UInt64) : Type := STensor #[batch, kvSeq]

inductive DispatchRoute where
  | tkKernel
  | portable
  deriving Repr, Inhabited, BEq

/-- Reusable runtime problem descriptor for the typed FlashAttention surface. -/
def attentionProblem
    {batch nHead nKvHead qSeq kvSeq headDim : UInt64} {d : DType}
    (q : Query batch nHead qSeq headDim d)
    (k : KeyValue batch nKvHead kvSeq headDim d)
    (v : KeyValue batch nKvHead kvSeq headDim d)
    (attnMask : Option (PaddingMask batch kvSeq) := none)
    (dropoutP : Float := 0.0)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    : AttentionProblem :=
  AttentionProblem.ofQKV q k v attnMask dropoutP isCausal scale enableGqa none .SM90

/-- Current native specialization eligibility for the typed FlashAttention surface. -/
def currentSpecialization
    {batch nHead nKvHead qSeq kvSeq headDim : UInt64} {d : DType}
    (q : Query batch nHead qSeq headDim d)
    (k : KeyValue batch nKvHead kvSeq headDim d)
    (v : KeyValue batch nKvHead kvSeq headDim d)
    (attnMask : Option (PaddingMask batch kvSeq) := none)
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
    (q : Query batch nHead qSeq headDim d)
    (k : KeyValue batch nKvHead kvSeq headDim d)
    (v : KeyValue batch nKvHead kvSeq headDim d)
    (attnMask : Option (PaddingMask batch kvSeq) := none)
    (dropoutP : Float := 0.0)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    : Bool :=
  (currentSpecialization q k v attnMask dropoutP isCausal scale enableGqa).isNative

/-- Typed dispatch route (kernel vs portable) for observability/debugging. -/
def dispatchRoute
    {batch nHead nKvHead qSeq kvSeq headDim : UInt64} {d : DType}
    (q : Query batch nHead qSeq headDim d)
    (k : KeyValue batch nKvHead kvSeq headDim d)
    (v : KeyValue batch nKvHead kvSeq headDim d)
    (attnMask : Option (PaddingMask batch kvSeq) := none)
    (dropoutP : Float := 0.0)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    : DispatchRoute :=
  match currentSpecialization q k v attnMask dropoutP isCausal scale enableGqa with
  | .portable => .portable
  | _ => .tkKernel

/-- Main typed FlashAttention/SDPA API.
    This routes through a torch-registered op (`tyr::flash_attn`) so autograd remains native.
    There is no typed wrapper for that registered op, so the raw handles cross at
    this FFI edge; the result is re-graded at the statically known spec. -/
@[inline] def flashAttn
    {batch nHead nKvHead qSeq kvSeq headDim : UInt64} {d : DType}
    (query : Query batch nHead qSeq headDim d)
    (key : KeyValue batch nKvHead kvSeq headDim d)
    (value : KeyValue batch nKvHead kvSeq headDim d)
    (attnMask : Option (PaddingMask batch kvSeq) := none)
    (dropoutP : Float := 0.0)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    : Query batch nHead qSeq headDim d :=
  Tensor.assumeSpec <|
    torch.nn.tyrFlashAttn4d
      (batch := batch) (n_head := nHead) (n_kv_head := nKvHead)
      (q_seq := qSeq) (kv_seq := kvSeq) (head_dim := headDim)
      query.raw key.raw value.raw (attnMask.map Tensor.raw) dropoutP isCausal scale enableGqa

/-- Alias matching PyTorch naming. -/
@[inline] def scaledDotProductAttention
    {batch nHead nKvHead qSeq kvSeq headDim : UInt64} {d : DType}
    (query : Query batch nHead qSeq headDim d)
    (key : KeyValue batch nKvHead kvSeq headDim d)
    (value : KeyValue batch nKvHead kvSeq headDim d)
    (attnMask : Option (PaddingMask batch kvSeq) := none)
    (dropoutP : Float := 0.0)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    : Query batch nHead qSeq headDim d :=
  flashAttn query key value attnMask dropoutP isCausal scale enableGqa

/-- Typed API returning both the dispatch route and the output. -/
def flashAttnWithRoute
    {batch nHead nKvHead qSeq kvSeq headDim : UInt64} {d : DType}
    (query : Query batch nHead qSeq headDim d)
    (key : KeyValue batch nKvHead kvSeq headDim d)
    (value : KeyValue batch nKvHead kvSeq headDim d)
    (attnMask : Option (PaddingMask batch kvSeq) := none)
    (dropoutP : Float := 0.0)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    : DispatchRoute × Query batch nHead qSeq headDim d :=
  let route := dispatchRoute query key value attnMask dropoutP isCausal scale enableGqa
  let out := flashAttn query key value attnMask dropoutP isCausal scale enableGqa
  (route, out)

structure PackedQKV where
  batch : UInt64
  nHead : UInt64
  nKvHead : UInt64
  qSeq : UInt64
  kvSeq : UInt64
  headDim : UInt64
  q : STensor #[batch, nHead, qSeq, headDim]
  k : STensor #[batch, nKvHead, kvSeq, headDim]
  v : STensor #[batch, nKvHead, kvSeq, headDim]

abbrev DynOut :=
  Sigma (fun batch =>
    Sigma (fun nHead =>
      Sigma (fun qSeq =>
        Sigma (fun headDim =>
          STensor #[batch, nHead, qSeq, headDim]))))

private def parseQKVDyn (qDyn kDyn vDyn : RawTensor) : IO PackedQKV := do
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

  -- The runtime shapes were just validated above, so the erased handles carry
  -- exactly the shapes the specs claim.
  let q : STensor #[batch, nHead, qSeq, headDim] := Tensor.assumeSpec qDyn
  let k : STensor #[batch, nKvHead, kvSeq, headDim] := Tensor.assumeSpec kDyn
  let v : STensor #[batch, nKvHead, kvSeq, headDim] := Tensor.assumeSpec vDyn
  pure {
    batch := batch
    nHead := nHead
    nKvHead := nKvHead
    qSeq := qSeq
    kvSeq := kvSeq
    headDim := headDim
    q := q
    k := k
    v := v
  }

private def parseMaskDyn
    (batch kvSeq : UInt64)
    (attnMask : Option RawTensor)
    : IO (Option (PaddingMask batch kvSeq)) := do
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
      let mask : PaddingMask batch kvSeq := Tensor.assumeSpec maskDyn
      pure (some mask)

/-- Fully general existential adapter:
    accepts shape-erased 4D raw tensors, validates runtime shapes, dispatches through the
    typed API, and returns a shape-indexed existential output. -/
def flashAttnDyn
    (qDyn kDyn vDyn : RawTensor)
    (attnMask : Option RawTensor := none)
    (dropoutP : Float := 0.0)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    : IO DynOut := do
  let packed ← parseQKVDyn qDyn kDyn vDyn
  let typedMask ← parseMaskDyn packed.batch packed.kvSeq attnMask
  let out : STensor #[packed.batch, packed.nHead, packed.qSeq, packed.headDim] :=
    flashAttn packed.q packed.k packed.v typedMask dropoutP isCausal scale enableGqa
  pure ⟨packed.batch, ⟨packed.nHead, ⟨packed.qSeq, ⟨packed.headDim, out⟩⟩⟩⟩

end Tyr.GPU.Ops.FlashAttn
