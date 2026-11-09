import Tyr.Typed.Ops
import Tyr.RawTensor.Neural
import Tyr.SafeTensors.File

namespace torch.Tensor

/-- Validate a native neural result before assigning its static specification. -/
private def checkedNeuralResult {σ : StaticSpec} (operation : String)
    (raw : RawTensor) : Tensor σ :=
  checkedResult operation (Tensor.ofRaw raw)

/-- Return a copy of the source in the destination's placement, reusing the
destination storage only when it has no observable aliases.
Dynamic specifications do not authorize implicit dtype conversion; copying
between actual devices remains supported by the native transfer. -/
def copyInto {s : Shape} {d : DType} {dstDevice srcDevice : Device}
    (dst : Tensor ⟨s, d, dstDevice⟩) (src : Tensor ⟨s, d, srcDevice⟩) :
    IO (Tensor ⟨s, d, dstDevice⟩) := do
  match TensorSpec.checkCompatible dst.actualSpec src.actualSpec with
  | .error message => throw (IO.userError s!"Tensor.copyInto: {message}")
  | .ok () => pure ()
  let dtype := dst.raw.dtype
  let device := dst.raw.device
  let raw ← RawTensor.copyInto dst.raw src.raw
  match Tensor.checkedUpdate? dtype device raw with
  | .ok result => pure result
  | .error message => throw (IO.userError s!"Tensor.copyInto: {message}")

/-- Checked dtype conversion. Unsupported native conversions return errors;
recognizing a dtype does not promise that a backend implements its conversion. -/
def castDType? {σ : StaticSpec} (t : Tensor σ) (dtype : DType) :
    Except String (Tensor (σ.withDType dtype)) := do
  let raw ← RawTensor.castDTypeChecked σ.shape t.raw dtype
  Device.check t.raw.device raw.device
  Tensor.ofRaw (σ := σ.withDType dtype) raw

/-- Cast to a concrete dtype; use `castDType?` to handle conversion failures. -/
def castDType {σ : StaticSpec} (t : Tensor σ) (dtype : DType) : Tensor (σ.withDType dtype) :=
  match t.castDType? dtype with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"Tensor.castDType: {message}"

/-- Match the reference's actual dtype and preserve input shape and placement. -/
def castLike? {σ τ : StaticSpec} (reference : Tensor τ) (t : Tensor σ) :
    Except String (Tensor (σ.withDType τ.dtype)) := do
  let dtype := reference.raw.dtype
  let raw ← if t.raw.dtype == dtype then pure t.raw else
    RawTensor.castDTypeChecked σ.shape t.raw dtype
  TensorSpec.checkDType dtype raw.dtype
  Device.check t.raw.device raw.device
  Tensor.ofRaw (σ := σ.withDType τ.dtype) raw

/-- Match the reference's native dtype; use `castLike?` for recoverable errors. -/
def castLike {σ τ : StaticSpec} (reference : Tensor τ) (t : Tensor σ) :
    Tensor (σ.withDType τ.dtype) :=
  match castLike? reference t with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"Tensor.castLike: {message}"

/-- RMS normalization accumulated in Float32, with the output dtype indexed. -/
def rmsNorm {σ : StaticSpec} (x : Tensor σ) (eps : Float := 1e-6) : Tensor (σ.withDType .Float32) :=
  let xf := RawTensor.toFloat32 x.raw
  let variance := RawTensor.mean! (RawTensor.mul! xf xf)
    (some #[(σ.shape.size - 1)]) true
  match Tensor.ofRaw
      (RawTensor.mul! xf (RawTensor.rsqrt! (RawTensor.addScalar! variance eps))) with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"Tensor.rmsNorm: {message}"

/-- RMS normalization with Float32 accumulation and explicit restoration of the input dtype. -/
def rmsNormPreserving {σ : StaticSpec} {w : Shape}
    (x : Tensor σ) (weight : Tensor (σ.withShape w)) (eps : Float := 1e-6) : Tensor σ :=
  -- HF checkpoints round the normalized activation before multiplying weights.
  let restored := Tensor.castLike x (rmsNorm x eps)
  x.checkedReplacement (RawTensor.mul! restored.raw weight.raw)

/-- Package a tensor's shape existentially; this does not erase its shape claim. -/
def packShape {s : Shape} {d : DType} {dev : Device} (t : Tensor ⟨s, d, dev⟩) :
    Sigma (fun shape => Tensor ⟨shape, d, dev⟩) := ⟨s, t⟩

/-- Concatenate tensors with runtime-varying shapes, checking the requested result.
Inputs retain their individual shape indices rather than pretending to be scalars. -/
def catMany? {d : DType} {dev : Device}
    (tensors : Array (Sigma (fun shape => Tensor ⟨shape, d, dev⟩)))
    (dim : Nat) (shape : Shape) : Except String (Tensor ⟨shape, d, dev⟩) := do
  let some first := tensors[0]? | throw "catMany requires at least one tensor"
  for t in tensors do
    Tensor.checkSameDevice first.2 t.2
  Tensor.ofRaw (← RawTensor.catManyChecked (tensors.map fun t => t.2.raw) dim)

def catManyChecked {d : DType} {dev : Device}
    (tensors : Array (Sigma (fun shape => Tensor ⟨shape, d, dev⟩)))
    (dim : Nat) (shape : Shape) : Tensor ⟨shape, d, dev⟩ :=
  match catMany? tensors dim shape with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"Tensor.catManyChecked: {message}"

def arange (stop : Nat) : Tensor { shape := #[stop], dtype := .Int64, device := .CPU } :=
  checkedNeuralResult "arange" (RawTensor.arange 0 stop 1)

def sin! {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  checkedNeuralResult "sin" (RawTensor.sin! t.raw)

def sin {σ : StaticSpec} (t : Tensor σ)
    (_domain : σ.dtype.isInexact = true := by first | rfl | exact DType.isInexact_of_realFloating | decide) : Tensor σ :=
  sin! t

def cos! {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  checkedNeuralResult "cos" (RawTensor.cos! t.raw)

def cos {σ : StaticSpec} (t : Tensor σ)
    (_domain : σ.dtype.isInexact = true := by first | rfl | exact DType.isInexact_of_realFloating | decide) : Tensor σ :=
  cos! t

def elu! {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  checkedNeuralResult "elu" (RawTensor.elu! t.raw)

def elu {σ : StaticSpec} (t : Tensor σ)
    (_domain : σ.dtype.isRealFloating = true := by first | rfl | exact DType.isRealFloating_of_realFloating | decide) : Tensor σ :=
  elu! t


/-- Mean for floating or complex checkpoint tensors; invalid axes, integral
inputs and missing kernels are recoverable errors. -/
def meanDim! {σ : StaticSpec} (t : Tensor σ) (dim : Nat) (keepdim : Bool := false) :
    Tensor (σ.withShape (reduceShape σ.shape dim keepdim)) :=
  checkedNeuralResult "meanDim" (RawTensor.mean! t.raw (some #[dim]) keepdim)

def meanDim {σ : StaticSpec} [DType.Mean σ.dtype] (t : Tensor σ) (dim : Nat) (keepdim : Bool := false)
    (_axis : dim < max 1 σ.shape.size := by first | exact of_decide_eq_true rfl | (simp <;> fail "axis out of range")) :
    Tensor (σ.withShape (reduceShape σ.shape dim keepdim)) :=
  meanDim! t dim keepdim

/-- Spatial extent computed without fixed-width arithmetic wrapping. -/
def convExtent (input kernel stride padding dilation : Nat) : Nat :=
  ((input + 2 * padding - (dilation * (kernel - 1) + 1)) /
    stride + 1)

/-- A stride-1 convolution padded by half its odd kernel keeps any nonempty extent. -/
theorem convExtent_same {n kernel padding : Nat} (hn : 0 < n) (hk : 2 * padding + 1 = kernel) :
    convExtent n kernel 1 padding 1 = n := by
  unfold convExtent; omega

/-- Project a spatial parameter, repeating singleton arrays as native conv2d does.
Only lengths one and two are valid; the native operation still receives the
original array and rejects invalid lengths. The zero for a missing entry only
makes the result index total and is never supplied as a native default. -/
@[simp] def conv2dParam (values : Array Nat) (axis : Nat) : Nat :=
  values.getD (if values.size == 1 then 0 else axis) 0

private def checkConvolutionOperands {σ τ υ : StaticSpec}
    (input : Tensor σ) (weight : Tensor τ) (bias : Tensor υ) : Except String Unit := do
  DType.expectedEq input.raw.dtype weight.raw.dtype
  DType.expectedEq input.raw.dtype bias.raw.dtype
  Tensor.checkSameDevice input weight
  Tensor.checkSameDevice input bias

/-- Checked convolution, including actual dtype equality for dynamic operands.
Native stride, padding and dilation errors remain recoverable. -/
def conv2dBias? {batch cin cout height width kh kw : Nat} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, cin, height, width], d, dev⟩)
    (weight : Tensor ⟨#[cout, cin, kh, kw], d, dev⟩)
    (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : Array Nat := #[1, 1]) (padding : Array Nat := #[0, 0])
    (dilation : Array Nat := #[1, 1]) :
    Except String (Tensor ⟨#[batch, cout,
      convExtent height kh (conv2dParam stride 0) (conv2dParam padding 0) (conv2dParam dilation 0),
      convExtent width kw (conv2dParam stride 1) (conv2dParam padding 1) (conv2dParam dilation 1)], d, dev⟩) := do
  checkConvolutionOperands input weight bias
  let raw ← RawTensor.conv2dBiasChecked
    input.raw weight.raw bias.raw stride padding dilation
  DType.expectedEq input.raw.dtype raw.dtype
  Tensor.ofRaw raw

/-- Biased two-dimensional convolution; `conv2dBias?` returns argument errors. -/
def conv2dBias {batch cin cout height width kh kw : Nat} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, cin, height, width], d, dev⟩)
    (weight : Tensor ⟨#[cout, cin, kh, kw], d, dev⟩)
    (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : Array Nat := #[1, 1]) (padding : Array Nat := #[0, 0])
    (dilation : Array Nat := #[1, 1]) :
    Tensor ⟨#[batch, cout,
      convExtent height kh (conv2dParam stride 0) (conv2dParam padding 0) (conv2dParam dilation 0),
      convExtent width kw (conv2dParam stride 1) (conv2dParam padding 1) (conv2dParam dilation 1)], d, dev⟩ :=
  match conv2dBias? input weight bias stride padding dilation with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"Tensor.conv2dBias: {message}"

def splitHeads {batch seq heads dim : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[batch, seq, heads, dim], d, dev⟩) : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  checkedNeuralResult "splitHeads" (RawTensor.splitHeads t.raw)

def joinHeads {batch heads seq dim : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩) : Tensor ⟨#[batch, seq, heads, dim], d, dev⟩ :=
  checkedNeuralResult "joinHeads" (RawTensor.joinHeads t.raw)

/-- Dynamic dtype indices still require the actual Q/K/V dtypes to agree. -/
private def checkAttentionOperands {sq sk sv : StaticSpec}
    (q : Tensor sq) (k : Tensor sk) (v : Tensor sv) : Except String Unit := do
  DType.expectedEq q.raw.dtype k.raw.dtype
  DType.expectedEq q.raw.dtype v.raw.dtype
  Device.check q.raw.device k.raw.device
  Device.check q.raw.device v.raw.device

private def attentionCheckIO {α : Type} (operation : String)
    (result : Except String α) : IO α :=
  match result with
  | .ok value => pure value
  | .error message => throw (IO.userError s!"Tensor.{operation}: {message}")

/-- Validate the configured dropout probability before selecting training or
evaluation behavior. Invalid configurations remain errors during evaluation. -/
def attentionDropout? (dropout : Float) (training : Bool) : Except String Float := do
  if dropout.isNaN || dropout.isInf || dropout < 0.0 || dropout > 1.0 then
    throw "attention dropout must be finite and in [0, 1]"
  pure (if training then dropout else 0.0)

def attention! {batch heads seq dim : Nat} {d : DType} {dev : Device}
    (q k v : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (causal : Bool := true) : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  q.checkedReplacement (RawTensor.attention! q.raw k.raw v.raw causal)

/-- Zero-dropout attention. -/
def attention {batch heads seq dim : Nat} {d : DType} {dev : Device}
    (q k v : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (causal : Bool := true) : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  attention! q k v causal

/-- Attention with explicit training-mode dropout, sequenced through native IO. -/
def attentionIO {batch heads seq dim : Nat} {d : DType} {dev : Device}
    (q k v : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (dropout : Float) (training : Bool) (causal : Bool := true) : IO (Tensor ⟨#[batch, heads, seq, dim], d, dev⟩) := do
  let effective ← attentionCheckIO "attentionIO" (attentionDropout? dropout training)
  attentionCheckIO "attentionIO" (checkAttentionOperands q k v)
  let raw ← RawTensor.attentionIO
    q.raw k.raw v.raw effective causal
  attentionCheckIO "attentionIO" (q.checkedReplacement? raw)

def attentionGQA! {batch heads kvHeads seq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (causal : Bool := true) (gqa : Bool := false) : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  q.checkedReplacement (RawTensor.attentionGQA! q.raw k.raw v.raw causal gqa)

/-- Zero-dropout grouped-query attention. -/
def attentionGQA {batch heads kvHeads seq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (causal : Bool := true) (gqa : Bool := false) :
    Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  attentionGQA! q k v causal gqa

/-- Attention with explicit training-mode dropout, sequenced through native IO. -/
def attentionGQAIO {batch heads kvHeads seq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (dropout : Float) (training : Bool) (causal : Bool := true) (gqa : Bool := false) : IO (Tensor ⟨#[batch, heads, seq, dim], d, dev⟩) := do
  let effective ← attentionCheckIO "attentionGQAIO" (attentionDropout? dropout training)
  attentionCheckIO "attentionGQAIO" (checkAttentionOperands q k v)
  let raw ← RawTensor.attentionGQAIO
    q.raw k.raw v.raw effective causal gqa
  attentionCheckIO "attentionGQAIO" (q.checkedReplacement? raw)

def attentionGQAMask! {batch heads kvHeads seq dim : Nat} {d dm : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (mask : Tensor ⟨#[batch, seq], dm, dev⟩)
    (causal : Bool := true) (gqa : Bool := false) : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  q.checkedReplacement (RawTensor.attentionGQAMask! q.raw k.raw v.raw mask.raw causal gqa)

/-- Zero-dropout grouped-query attention with a key padding mask. -/
def attentionGQAMask {batch heads kvHeads seq dim : Nat} {d dm : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (mask : Tensor ⟨#[batch, seq], dm, dev⟩)
    (causal : Bool := true) (gqa : Bool := false) :
    Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  attentionGQAMask! q k v mask causal gqa

/-- Attention with explicit training-mode dropout, sequenced through native IO. -/
def attentionGQAMaskIO {batch heads kvHeads seq dim : Nat} {d dm : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (mask : Tensor ⟨#[batch, seq], dm, dev⟩)
    (dropout : Float) (training : Bool) (causal : Bool := true) (gqa : Bool := false) : IO (Tensor ⟨#[batch, heads, seq, dim], d, dev⟩) := do
  let effective ← attentionCheckIO "attentionGQAMaskIO" (attentionDropout? dropout training)
  attentionCheckIO "attentionGQAMaskIO" (checkAttentionOperands q k v)
  let raw ← RawTensor.attentionGQAMaskIO
    q.raw k.raw v.raw mask.raw effective causal gqa
  attentionCheckIO "attentionGQAMaskIO" (q.checkedReplacement? raw)

/-- Gather has the index tensor's shape, and the source tensor's dtype. -/
def gather {s indicesShape : Shape} {d : DType} {dev : Device}
    (input : Tensor ⟨s, d, dev⟩) (dim : Nat)
    (indices : Tensor ⟨indicesShape, .Int64, dev⟩)
    (_h : TensorSpec.gatherCompatible s indicesShape dim = true := by
      first | exact of_decide_eq_true rfl | (simp <;> fail "incompatible gather shapes")) :
    Tensor ⟨indicesShape, d, dev⟩ :=
  match Tensor.ofRaw (RawTensor.gather input.raw dim indices.raw) with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"Tensor.gather: {message}"

/-- Masked selection returns an existential vector length, checked against the native result. -/
def maskedSelect {s : Shape} {d : DType} {dev : Device}
    (input : Tensor ⟨s, d, dev⟩) (mask : Tensor ⟨s, .Bool, dev⟩) :
    Sigma (fun n => Tensor ⟨#[n], d, dev⟩) :=
  let raw := RawTensor.maskedSelect input.raw mask.raw
  let n := raw.runtimeShape.getD 0 0
  match Tensor.ofRaw (σ := ⟨#[n], d, dev⟩) raw with
  | .ok result => ⟨n, result⟩
  | .error message => ⟨n, nativeInvariantFailure s!"Tensor.maskedSelect: {message}"⟩

def maskedScatter {s sourceShape : Shape} {d : DType} {dev : Device}
    (input : Tensor ⟨s, d, dev⟩) (mask : Tensor ⟨s, .Bool, dev⟩)
    (source : Tensor ⟨sourceShape, d, dev⟩) : Tensor ⟨s, d, dev⟩ :=
  input.checkedReplacement (RawTensor.maskedScatter input.raw mask.raw source.raw)

def attentionGQAQKV! {batch heads kvHeads qSeq kvSeq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, kvSeq, dim], d, dev⟩)
    (causal : Bool := false) (gqa : Bool := false) : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩ :=
  q.checkedReplacement (RawTensor.attentionGQAQKV! q.raw k.raw v.raw causal gqa)

/-- Zero-dropout grouped-query attention with separate query and key/value lengths. -/
def attentionGQAQKV {batch heads kvHeads qSeq kvSeq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, kvSeq, dim], d, dev⟩)
    (causal : Bool := false) (gqa : Bool := false) :
    Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩ :=
  attentionGQAQKV! q k v causal gqa

/-- Attention with explicit training-mode dropout, sequenced through native IO. -/
def attentionGQAQKVIO {batch heads kvHeads qSeq kvSeq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, kvSeq, dim], d, dev⟩)
    (dropout : Float) (training : Bool) (causal : Bool := false) (gqa : Bool := false) : IO (Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩) := do
  let effective ← attentionCheckIO "attentionGQAQKVIO" (attentionDropout? dropout training)
  attentionCheckIO "attentionGQAQKVIO" (checkAttentionOperands q k v)
  let raw ← RawTensor.attentionGQAQKVIO
    q.raw k.raw v.raw effective causal gqa
  attentionCheckIO "attentionGQAQKVIO" (q.checkedReplacement? raw)

def flashAttention! {batch heads kvHeads qSeq kvSeq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, kvSeq, dim], d, dev⟩) (gqa : Bool := false)
    (mask : Option (SomeDTypeTensor #[batch, kvSeq] dev) := none)
    (causal : Bool := false) (scale : Option Float := none) :
    Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩ :=
  q.checkedReplacement (RawTensor.flashAttention!
    q.raw k.raw v.raw (mask.map (fun packed => packed.2.raw)) causal scale gqa)

/-- Zero-dropout FlashAttention. -/
def flashAttention {batch heads kvHeads qSeq kvSeq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, kvSeq, dim], d, dev⟩) (gqa : Bool := false)
    (mask : Option (SomeDTypeTensor #[batch, kvSeq] dev) := none)
    (causal : Bool := false) (scale : Option Float := none) :
    Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩ :=
  flashAttention! q k v gqa mask causal scale

/-- FlashAttention with explicit training-mode dropout through native IO.
Evaluation uses zero dropout, independently of the configured probability. -/
def flashAttentionIO {batch heads kvHeads qSeq kvSeq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, kvSeq, dim], d, dev⟩)
    (dropout : Float) (training : Bool) (gqa : Bool := false)
    (mask : Option (SomeDTypeTensor #[batch, kvSeq] dev) := none)
    (causal : Bool := false) (scale : Option Float := none) :
    IO (Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩) := do
  let effective ← attentionCheckIO "flashAttentionIO" (attentionDropout? dropout training)
  attentionCheckIO "flashAttentionIO" (checkAttentionOperands q k v)
  let raw ← RawTensor.flashAttentionIO
    q.raw k.raw v.raw (mask.map (fun packed => packed.2.raw)) effective causal scale gqa
  attentionCheckIO "flashAttentionIO" (q.checkedReplacement? raw)

def sliceScatter {s sourceShape : Shape} {d : DType} {dev : Device}
    (input : Tensor ⟨s, d, dev⟩) (dim start : Nat)
    (source : Tensor ⟨sourceShape, d, dev⟩)
    (_h : TensorSpec.sliceScatterCompatible s sourceShape dim = true := by
      first | exact of_decide_eq_true rfl | (simp <;> fail "incompatible slice scatter shapes"))
    (_bound : start + sourceShape.getD dim 0 ≤ s.getD dim 0 := by
      first | exact of_decide_eq_true rfl | omega | (simp <;> omega)) :
    Tensor ⟨s, d, dev⟩ :=
  let dtype := input.raw.dtype
  let device := input.raw.device
  Tensor.checkedUpdate dtype device
    (RawTensor.sliceScatter input.raw dim start source.raw)

def applyRotary {batch seq heads dim : Nat} {d dc : DType} {dev : Device}
    (input : Tensor ⟨#[batch, seq, heads, dim], d, dev⟩)
    (cos sin : Tensor ⟨#[seq, dim / 2], dc, dev⟩) :=
  input.checkedReplacement (RawTensor.applyRotary input.raw cos.raw sin.raw)

def rotaryFreqs (seq dim : Nat) (base : Float := 10000.0) (device : Device := .CPU) :
    Tensor { shape := #[seq, dim / 2], dtype := .Float32, device } × Tensor { shape := #[seq, dim / 2], dtype := .Float32, device } :=
  let (cos, sin) := RawTensor.rotaryFreqs seq dim base device
  match Tensor.ofRaw cos, Tensor.ofRaw sin with
  | .ok c, .ok s => (c, s)
  | .error message, _ | _, .error message => nativeInvariantFailure s!"Tensor.rotaryFreqs: {message}"

def hannWindow (n : Nat) : Tensor { shape := #[n], dtype := .Float32, device := .CPU } :=
  match Tensor.ofRaw (RawTensor.hannWindow n) with
  | .ok t => t
  | .error message => nativeInvariantFailure s!"Tensor.hannWindow: {message}"

/-- Frames of `stft` over `n` samples; `center` reflect-pads `nFft / 2` on each side. -/
def stftFrames (n nFft hopLength : Nat) (center : Bool) : Nat :=
  ((if center then n + 2 * (nFft / 2) else n) - nFft) / hopLength + 1

/-- One-sided STFT with real and imaginary parts packed in the last axis. -/
def stft {n winLength : Nat} {dev : Device} (input : Tensor { shape := #[n], dtype := .Float32, device := dev }) (nFft hopLength : Nat)
    (window : Tensor { shape := #[winLength], dtype := .Float32, device := dev }) (center : Bool := true) (normalized : Bool := false) :
    IO (Tensor { shape := #[nFft / 2 + 1, stftFrames n nFft hopLength center, 2], dtype := .Float32, device := dev }) := do
  let raw ← match RawTensor.stftChecked input.raw nFft hopLength winLength window.raw center normalized with
    | .ok raw => pure raw
    | .error message => throw <| IO.userError s!"Tensor.stft: {message}"
  match Tensor.ofRaw raw with
  | .ok t => pure t
  | .error message => throw <| IO.userError s!"Tensor.stft: {message}"

def log10! {σ : StaticSpec} (input : Tensor σ) : Tensor σ :=
  checkedNeuralResult "log10" (RawTensor.log10! input.raw)

def log10 {σ : StaticSpec} (input : Tensor σ)
    (_inexact : σ.dtype.isInexact = true := by first | rfl | exact DType.isInexact_of_realFloating | decide) : Tensor σ :=
  log10! input


def saveWav {σ : StaticSpec} (input : Tensor σ) (path : String) (sampleRate : Nat := 24000) : IO Unit :=
  RawTensor.saveWav input.raw path sampleRate

def lt! {s : Shape} {d : DType} {dev : Device} (a b : Tensor ⟨s, d, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  checkedNeuralResult "lt" (RawTensor.lt! a.raw b.raw)

def lt {s : Shape} {d : DType} {dev : Device} [DType.Ordered d] (a b : Tensor ⟨s, d, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  lt! a b

def ge! {s : Shape} {d : DType} {dev : Device} (a b : Tensor ⟨s, d, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  checkedNeuralResult "ge" (RawTensor.ge! a.raw b.raw)

def ge {s : Shape} {d : DType} {dev : Device} [DType.Ordered d] (a b : Tensor ⟨s, d, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  ge! a b
def ltScalar {s : Shape} {d : DType} {dev : Device} (input : Tensor ⟨s, d, dev⟩) (value : Float) :
    Tensor ⟨s, .Bool, dev⟩ := checkedNeuralResult "ltScalar" (RawTensor.ltScalar input.raw value)
def maskedFill {s : Shape} {d : DType} {dev : Device}
    (input : Tensor ⟨s, d, dev⟩) (mask : Tensor ⟨s, .Bool, dev⟩) (value : Float) : Tensor ⟨s, d, dev⟩ :=
  input.checkedReplacement (RawTensor.maskedFill input.raw mask.raw value)

/-- Grouped convolution checks channel grouping at the native boundary. -/
def conv1dGroups? {batch cin cout cinPerGroup length kernel : Nat} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, cin, length], d, dev⟩)
    (weight : Tensor ⟨#[cout, cinPerGroup, kernel], d, dev⟩) (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : Nat := 1) (padding : Nat := 0) (dilation : Nat := 1) (groups : Nat := 1) :
    Except String (Tensor ⟨#[batch, cout, convExtent length kernel stride padding dilation], d, dev⟩) := do
  checkConvolutionOperands input weight bias
  let raw ← RawTensor.conv1dGroupsChecked
    input.raw weight.raw bias.raw stride padding dilation groups
  DType.expectedEq input.raw.dtype raw.dtype
  Tensor.ofRaw raw

/-- Grouped convolution; use `conv1dGroups?` for recoverable argument errors. -/
def conv1dGroups {batch cin cout cinPerGroup length kernel : Nat} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, cin, length], d, dev⟩)
    (weight : Tensor ⟨#[cout, cinPerGroup, kernel], d, dev⟩) (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : Nat := 1) (padding : Nat := 0) (dilation : Nat := 1) (groups : Nat := 1) :
    Tensor ⟨#[batch, cout, convExtent length kernel stride padding dilation], d, dev⟩ :=
  match conv1dGroups? input weight bias stride padding dilation groups with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"Tensor.conv1dGroups: {message}"

/-- Checked transposed convolution, including native output-padding validation. -/
def convTranspose1d? {batch cin cout length kernel : Nat} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, cin, length], d, dev⟩)
    (weight : Tensor ⟨#[cin, cout, kernel], d, dev⟩) (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : Nat := 1) (padding : Nat := 0) (outputPadding : Nat := 0) (dilation : Nat := 1) :
    Except String (Tensor ⟨#[batch, cout, ((length - 1) * stride + dilation * (kernel - 1)
      + outputPadding + 1 - 2 * padding)], d, dev⟩) := do
  checkConvolutionOperands input weight bias
  let raw ← RawTensor.convTranspose1dChecked
    input.raw weight.raw bias.raw stride padding outputPadding dilation
  DType.expectedEq input.raw.dtype raw.dtype
  Tensor.ofRaw raw

/-- Transposed convolution; `convTranspose1d?` returns argument errors. -/
def convTranspose1d {batch cin cout length kernel : Nat} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, cin, length], d, dev⟩)
    (weight : Tensor ⟨#[cin, cout, kernel], d, dev⟩) (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : Nat := 1) (padding : Nat := 0) (outputPadding : Nat := 0) (dilation : Nat := 1) :
    Tensor ⟨#[batch, cout, ((length - 1) * stride + dilation * (kernel - 1)
      + outputPadding + 1 - 2 * padding)], d, dev⟩ :=
  match convTranspose1d? input weight bias stride padding outputPadding dilation with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"Tensor.convTranspose1d: {message}"

def attentionWindow! {batch heads kvHeads seq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (causal : Bool := true) (gqa : Bool := false) (window : Nat := 0) : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  q.checkedReplacement (RawTensor.attentionWindow! q.raw k.raw v.raw causal gqa window)

/-- Zero-dropout sliding-window attention. -/
def attentionWindow {batch heads kvHeads seq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (causal : Bool := true) (gqa : Bool := false) (window : Nat := 0) : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  attentionWindow! q k v causal gqa window

/-- Attention with explicit training-mode dropout, sequenced through native IO. -/
def attentionWindowIO {batch heads kvHeads seq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (dropout : Float) (training : Bool) (causal : Bool := true) (gqa : Bool := false) (window : Nat := 0) : IO (Tensor ⟨#[batch, heads, seq, dim], d, dev⟩) := do
  let effective ← attentionCheckIO "attentionWindowIO" (attentionDropout? dropout training)
  attentionCheckIO "attentionWindowIO" (checkAttentionOperands q k v)
  let raw ← RawTensor.attentionWindowIO
    q.raw k.raw v.raw effective causal gqa window
  attentionCheckIO "attentionWindowIO" (q.checkedReplacement? raw)

def log! {σ : StaticSpec} (input : Tensor σ) : Tensor σ :=
  checkedNeuralResult "log" (RawTensor.log! input.raw)

def log {σ : StaticSpec} (input : Tensor σ)
    (_inexact : σ.dtype.isInexact = true := by first | rfl | exact DType.isInexact_of_realFloating | decide) : Tensor σ :=
  log! input


def saveSafeTensor {σ : StaticSpec} (input : Tensor σ) (path name : String) : IO Unit :=
  RawTensor.saveSafeTensor path name input.raw

/-- Clip the parameter's accumulated gradient in place, returning its pre-clip norm. -/
def clipGradNorm {σ : StaticSpec} (input : Tensor σ) (maxNorm : Float) : IO Float :=
  RawTensor.clipGradNorm input.raw maxNorm

def saveTensor {σ : StaticSpec} (input : Tensor σ) (path : String) : IO Unit :=
  RawTensor.saveTensor input.raw path

/-- Load a tensor with exact stored shape and validate its dtype and placement. -/
def loadTensorExact (σ : StaticSpec) (path : String) : IO (Tensor σ) := do
  let raw ← RawTensor.loadTensorExact σ.shape path
  match Tensor.ofRaw raw with
  | .ok tensor => pure tensor
  | .error message => throw (IO.userError s!"Tensor.loadTensorExact: {message}")

/-- Load the exact stored shape and dtype, moving only its placement to `σ.device`.
The stored dtype is checked before conversion; loading never silently casts it. -/
def loadTensorOn (σ : StaticSpec) (path : String) : IO (Tensor σ) := do
  let raw ← RawTensor.loadTensorExact σ.shape path
  let stored ← match Tensor.ofRaw (σ := { σ with device := raw.device }) raw with
    | .ok tensor => pure tensor
    | .error message => throw (IO.userError s!"Tensor.loadTensorOn: {message}")
  let moved ← RawTensor.convertIO σ.shape stored.raw σ.device none
  match Tensor.ofRaw moved with
  | .ok tensor => pure tensor
  | .error message => throw (IO.userError s!"Tensor.loadTensorOn: {message}")

/-- Start a mono streaming WAV file with a placeholder header. -/
def wavBegin (path : String) (sampleRate : Nat := 24000) : IO Unit :=
  RawTensor.wavBegin path sampleRate

/-- Append samples, converting to CPU Float32 PCM at the native file boundary. -/
def wavAppend {σ : StaticSpec} (input : Tensor σ) (path : String) : IO Unit :=
  RawTensor.wavAppend input.raw path

/-- Patch the streaming WAV header to match the final sample count. -/
def wavFinalize (path : String) : IO Unit := RawTensor.wavFinalize path

/-- Load UInt16 token data as an indexed Int64 tensor, validating native metadata. -/
def loadU16Bin (count : Nat) (path : String) : IO (Tensor { shape := #[count], dtype := .Int64, device := .CPU }) := do
  let raw ← RawTensor.loadU16Bin count path
  match Tensor.ofRaw raw with
  | .ok tensor => pure tensor
  | .error message => throw <| IO.userError s!"Tensor.loadU16Bin: {message}"

def linspace (start stop : Float) (count : Nat) : Tensor { shape := #[count], dtype := .Float32, device := .CPU } :=
  checkedNeuralResult "linspace" (RawTensor.linspace start stop count false)

/-- Reduce the last axis, preserving the actual input dtype, without precondition
checks; libtorch rejects an unordered dtype or an empty axis and the process aborts. -/
def maxLast3d! {batch seq vocab : Nat} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, seq, vocab], d, dev⟩) :
    Tensor ⟨#[batch, seq], d, dev⟩ × Tensor ⟨#[batch, seq], .Int64, dev⟩ :=
  let (values, indices) := RawTensor.maxLast3d! input.raw 2
  checkedResult "maxLast3d" do
    return (← Tensor.checkedUpdate? input.raw.dtype input.raw.device values, ← Tensor.ofRaw indices)

/-- Reduce a nonempty last axis of an ordered tensor. -/
def maxLast3d {batch seq vocab : Nat} {d : DType} {dev : Device} [DType.Ordered d]
    (input : Tensor ⟨#[batch, seq, vocab], d, dev⟩)
    (_nonempty : 0 < vocab := by first | exact of_decide_eq_true rfl | omega | (simp <;> omega)) :
    Tensor ⟨#[batch, seq], d, dev⟩ × Tensor ⟨#[batch, seq], .Int64, dev⟩ :=
  maxLast3d! input

/-- Select the largest `k` entries per row, preserving the actual input dtype,
without precondition checks; libtorch rejects an unordered dtype or `k` above the
row width and the process aborts. -/
def topKLast2d! {rows cols : Nat} {d : DType} {dev : Device}
    (input : Tensor ⟨#[rows, cols], d, dev⟩) (k : Nat) :
    Tensor ⟨#[rows, k], d, dev⟩ × Tensor ⟨#[rows, k], .Int64, dev⟩ :=
  let (values, indices) := RawTensor.topKLast2d! input.raw k 1
  checkedResult "topKLast2d" do
    return (← Tensor.checkedUpdate? input.raw.dtype input.raw.device values, ← Tensor.ofRaw indices)

/-- Select the largest `k` entries of ordered rows with sufficient width. -/
def topKLast2d {rows cols : Nat} {d : DType} {dev : Device} [DType.Ordered d]
    (input : Tensor ⟨#[rows, cols], d, dev⟩) (k : Nat)
    (_range : k ≤ cols := by first | exact of_decide_eq_true rfl | omega | (simp <;> omega)) :
    Tensor ⟨#[rows, k], d, dev⟩ × Tensor ⟨#[rows, k], .Int64, dev⟩ :=
  topKLast2d! input k

/-- End-exclusive integer range, with overflow-safe length computation. -/
def rangeLength (start stop step : Nat) : Nat :=
  if step == 0 || stop <= start then 0
  else ((stop - start + step - 1) / step)

def arangeFrom (start stop : Nat) (step : Nat := 1) :
    Tensor { shape := #[rangeLength start stop step], dtype := .Int64, device := .CPU } :=
  match Tensor.ofRaw (RawTensor.arange start stop step) with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"Tensor.arangeFrom: {message}"

/-- Select entry `idx` of the leading axis; a scalar has no leading axis. -/
def getOp {σ : StaticSpec} (input : Tensor σ) (idx : Nat)
    (_bound : idx < σ.shape[0]?.getD 0 := by
      first | exact of_decide_eq_true rfl | (simp <;> fail "index out of range")) :
    Tensor (σ.withShape (σ.shape.extract 1 σ.shape.size)) :=
  match Tensor.ofRaw (RawTensor.getOp input.raw idx) with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"Tensor.getOp: {message}"

def unbind {σ : StaticSpec} (input : Tensor σ) (dim : Nat := 0)
    (h : dim < σ.shape.size := by first | exact of_decide_eq_true rfl | (simp <;> fail "axis out of range")) :
    Array (Tensor (σ.withShape (unbindShape σ.shape dim h))) :=
  (RawTensor.unbind input.raw dim).map fun raw =>
    match Tensor.ofRaw raw with
    | .ok result => result
    | .error message => nativeInvariantFailure s!"Tensor.unbind: {message}"

def logspace (start stop : Float) (steps : Nat) (base : Float := 10.0) :
    Tensor { shape := #[steps], dtype := .Float32, device := .CPU } :=
  match Tensor.ofRaw (RawTensor.logspace start stop steps base false) with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"Tensor.logspace: {message}"

/-- Backpropagate an explicit cotangent. Saved intermediates are released by
default; omitted retention follows `createGraph`, as in PyTorch. -/
def backward {σ : StaticSpec} (output cotangent : Tensor σ)
    (retainGraph : Option Bool := none) (createGraph : Bool := false) : IO Unit :=
  RawTensor.backward output.raw cotangent.raw (retainGraph.getD createGraph) createGraph

def hasGradFn {σ : StaticSpec} (input : Tensor σ) : Bool :=
  RawTensor.hasGradFn input.raw

/-- Find matching token positions, retaining the runtime-dependent vector length. -/
def findTokenPositions {n : Nat} {dev : Device}
    (tokens : Tensor ⟨#[n], .Int64, dev⟩) (token : Int64) :
    IO (Sigma fun count => Tensor ⟨#[count], .Int64, dev⟩) := do
  let raw ← RawTensor.findTokenPositions tokens.raw token
  let count := raw.runtimeShape.getD 0 0
  match Tensor.ofRaw (σ := ⟨#[count], .Int64, dev⟩) raw with
  | .ok result => pure ⟨count, result⟩
  | .error message => throw (IO.userError s!"Tensor.findTokenPositions: {message}")

end torch.Tensor

namespace torch.SomeDTypeTensor

/-- Load the exact stored shape and placement while retaining the stored dtype
as an existential index. Loading does not cast or reshape the tensor. -/
def loadTensorExact (shape : Shape) (device : Device) (path : String) :
    IO (SomeDTypeTensor shape device) := do
  let raw ← RawTensor.loadTensorExact shape path
  match SomeDTypeTensor.ofRaw raw with
  | .ok tensor => pure tensor
  | .error message => throw (IO.userError s!"SomeDTypeTensor.loadTensorExact: {message}")

end torch.SomeDTypeTensor
