import Tyr.Typed.Ops
import Tyr.RawTensor.Neural

namespace torch.Tensor

/-- Validate a native neural result before assigning its static specification. -/
private def checkedNeuralResult {σ : StaticSpec} (operation : String)
    (raw : RawTensor) : Tensor σ :=
  match Tensor.ofRaw raw with
  | .ok result => result
  | .error message => panic! s!"Tensor.{operation}: {message}"

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
  let raw ← RawTensor.copyInto s dst.raw src.raw
  match Tensor.checkedUpdate? dtype device raw with
  | .ok result => pure result
  | .error message => throw (IO.userError s!"Tensor.copyInto: {message}")

/-- Forget only the dtype claim. Shape and placement remain indexed. -/
def forgetDType {σ : StaticSpec} (t : Tensor σ) : Tensor { σ with dtype := .Unknown "dynamic" } :=
  checkedNeuralResult "forgetDType" t.raw

/-- Checked dtype conversion. Only the reserved `dynamic` dtype weakens the
claim without casting. Unsupported tokens and native conversions return errors;
recognizing a dtype does not promise that a backend implements its conversion. -/
def castDType? {σ : StaticSpec} (t : Tensor σ) (dtype : DType) :
    Except String (Tensor (σ.withDType dtype)) := do
  let converted : Except String RawTensor := match dtype with
    | .Unknown _ =>
        if dtype == .Unknown "dynamic" then pure t.raw
        else throw s!"Unsupported target dtype {dtype}"
    | _ => RawTensor.castDTypeChecked σ.shape t.raw dtype.canonicalName
  let raw ← converted
  Device.check t.raw.device raw.device
  Tensor.ofRaw (σ := σ.withDType dtype) raw

/-- Cast to a concrete dtype; use `castDType?` to handle conversion failures. -/
def castDType {σ : StaticSpec} (t : Tensor σ) (dtype : DType) : Tensor (σ.withDType dtype) :=
  match t.castDType? dtype with
  | .ok result => result
  | .error message => panic! s!"Tensor.castDType: {message}"

/-- Match the reference's actual dtype and preserve input shape and placement.
An unsupported native dtype may only be retained when both inputs already agree. -/
def castLike? {σ τ : StaticSpec} (reference : Tensor τ) (t : Tensor σ) :
    Except String (Tensor (σ.withDType τ.dtype)) := do
  let dtype := reference.raw.dtype
  let converted : Except String RawTensor := if t.raw.dtype == dtype then pure t.raw else
    match dtype with
    | .Unknown _ => throw s!"Unsupported reference dtype {dtype}"
    | _ => RawTensor.castDTypeChecked σ.shape t.raw dtype.canonicalName
  let raw ← converted
  TensorSpec.checkDType dtype raw.dtype
  Device.check t.raw.device raw.device
  Tensor.ofRaw (σ := σ.withDType τ.dtype) raw

/-- Match the reference's native dtype; use `castLike?` for recoverable errors. -/
def castLike {σ τ : StaticSpec} (reference : Tensor τ) (t : Tensor σ) :
    Tensor (σ.withDType τ.dtype) :=
  match castLike? reference t with
  | .ok result => result
  | .error message => panic! s!"Tensor.castLike: {message}"

/-- RMS normalization accumulated in Float32, with the output dtype indexed. -/
def rmsNorm {σ : StaticSpec} (x : Tensor σ) (eps : Float := 1e-6) : Tensor (σ.withDType .Float32) :=
  let xf := RawTensor.toFloat32 σ.shape x.raw
  let variance := RawTensor.mean σ.shape (RawTensor.mul σ.shape xf xf)
    (some #[(σ.shape.size - 1).toUInt64]) true
  let reduced := reduceShape σ.shape (σ.shape.size - 1) true
  match Tensor.ofRaw
      (RawTensor.mul σ.shape xf (RawTensor.rsqrt reduced (RawTensor.addScalar reduced variance eps))) with
  | .ok result => result
  | .error message => panic! s!"Tensor.rmsNorm: {message}"

/-- RMS normalization with Float32 accumulation and explicit restoration of the input dtype. -/
def rmsNormPreserving {σ : StaticSpec} {w : Shape}
    (x : Tensor σ) (weight : Tensor (σ.withShape w)) (eps : Float := 1e-6) : Tensor σ :=
  -- HF checkpoints round the normalized activation before multiplying weights.
  let restored := Tensor.castLike x (rmsNorm x eps)
  let result := do
    Tensor.checkPointwise restored weight
    x.checkedReplacement? (← RawTensor.mulChecked σ.shape restored.raw weight.raw)
  match result with
  | .ok tensor => tensor
  | .error message => panic! s!"Tensor.rmsNormPreserving: {message}"

/-- Package a tensor's shape existentially; this does not erase its shape claim. -/
def packShape {s : Shape} {d : DType} {dev : Device} (t : Tensor ⟨s, d, dev⟩) :
    Sigma (fun shape => Tensor ⟨shape, d, dev⟩) := ⟨s, t⟩

/-- Concatenate tensors with runtime-varying shapes, checking the requested result.
Inputs retain their individual shape indices rather than pretending to be scalars. -/
def catMany? {d : DType} {dev : Device}
    (tensors : Array (Sigma (fun shape => Tensor ⟨shape, d, dev⟩)))
    (dim : Int) (shape : Shape) : Except String (Tensor ⟨shape, d, dev⟩) := do
  let some first := tensors[0]? | throw "catMany requires at least one tensor"
  for t in tensors do
    Tensor.checkSameDevice first.2 t.2
  Tensor.ofRaw (← RawTensor.catManyChecked (tensors.map fun t => t.2.raw) dim)

def catManyChecked {d : DType} {dev : Device}
    (tensors : Array (Sigma (fun shape => Tensor ⟨shape, d, dev⟩)))
    (dim : Int) (shape : Shape) : Tensor ⟨shape, d, dev⟩ :=
  match catMany? tensors dim shape with
  | .ok result => result
  | .error message => panic! s!"Tensor.catManyChecked: {message}"

def arange (stop : UInt64) : Tensor { shape := #[stop], dtype := .Int64, device := .CPU } :=
  checkedNeuralResult "arange" (RawTensor.arange 0 stop 1)

private def checkedNeuralUnary? {σ : StaticSpec} (operation domain : String)
    (accepts : DType → Bool) (native : Shape → RawTensor → Except String RawTensor)
    (t : Tensor σ) : Except String (Tensor σ) := do
  if !accepts t.raw.dtype then
    throw s!"{operation} requires {domain}, got {t.raw.dtype}"
  t.checkedReplacement? (← native σ.shape t.raw)

private def neuralUnaryResult {σ : StaticSpec} (operation : String) (t : Tensor σ)
    (result : Except String (Tensor σ)) : Tensor σ :=
  match result with
  | .ok result => result
  | .error message =>
    letI : Inhabited (Tensor σ) := ⟨t⟩
    panic! s!"Tensor.{operation}: {message}"

def sin? {σ : StaticSpec} (t : Tensor σ) : Except String (Tensor σ) :=
  checkedNeuralUnary? "sin" "a floating or complex dtype" DType.isInexact RawTensor.sinChecked t

def sin {σ : StaticSpec} (t : Tensor σ)
    (_domain : σ.dtype.isInexact = true := by first | rfl | decide) : Tensor σ :=
  neuralUnaryResult "sin" t (sin? t)

def cos? {σ : StaticSpec} (t : Tensor σ) : Except String (Tensor σ) :=
  checkedNeuralUnary? "cos" "a floating or complex dtype" DType.isInexact RawTensor.cosChecked t

def cos {σ : StaticSpec} (t : Tensor σ)
    (_domain : σ.dtype.isInexact = true := by first | rfl | decide) : Tensor σ :=
  neuralUnaryResult "cos" t (cos? t)

def elu? {σ : StaticSpec} (t : Tensor σ) : Except String (Tensor σ) :=
  checkedNeuralUnary? "elu" "a standard real floating dtype" DType.isRealFloating RawTensor.eluChecked t

def elu {σ : StaticSpec} (t : Tensor σ)
    (_domain : σ.dtype.isRealFloating = true := by first | rfl | decide) : Tensor σ :=
  neuralUnaryResult "elu" t (elu? t)

namespace Preserving
/-- Runtime-domain and native-error checks for checkpoint activations. -/
def gelu? {σ : StaticSpec} (t : Tensor σ) : Except String (Tensor σ) :=
  Tensor.gelu? t

def gelu {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  neuralUnaryResult "Preserving.gelu" t (gelu? t)

def silu? {σ : StaticSpec} (t : Tensor σ) : Except String (Tensor σ) :=
  Tensor.silu? t

def silu {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  neuralUnaryResult "Preserving.silu" t (silu? t)

def sigmoid? {σ : StaticSpec} (t : Tensor σ) : Except String (Tensor σ) :=
  Tensor.sigmoid? t

def sigmoid {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  neuralUnaryResult "Preserving.sigmoid" t (sigmoid? t)

def tanh? {σ : StaticSpec} (t : Tensor σ) : Except String (Tensor σ) :=
  Tensor.tanh? t

def tanh {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  neuralUnaryResult "Preserving.tanh" t (tanh? t)

def softmax? {σ : StaticSpec} (t : Tensor σ) (dim : Int32 := -1) : Except String (Tensor σ) :=
  Tensor.softmax? t dim

def softmax {σ : StaticSpec} (t : Tensor σ) (dim : Int32 := -1) : Tensor σ :=
  neuralUnaryResult "Preserving.softmax" t (softmax? t dim)

/-- Mean for floating or complex checkpoint tensors; invalid axes, integral
inputs and missing kernels are recoverable errors. -/
def meanDim? {σ : StaticSpec} (t : Tensor σ) (dim : Nat) (keepdim : Bool := false) :
    Except String (Tensor (σ.withShape (reduceShape σ.shape dim keepdim))) := do
  DType.checkMean t.raw.dtype
  if dim >= max 1 σ.shape.size then
    throw s!"Reduction axis {dim} is out of range for rank {σ.shape.size}"
  let raw ← RawTensor.meanChecked σ.shape t.raw (some #[dim.toUInt64]) keepdim
  TensorSpec.checkDType t.raw.dtype raw.dtype
  Device.check t.raw.device raw.device
  Tensor.ofRaw raw

def meanDim {σ : StaticSpec} (t : Tensor σ) (dim : Nat) (keepdim : Bool := false) :
    Tensor (σ.withShape (reduceShape σ.shape dim keepdim)) :=
  match meanDim? t dim keepdim with
  | .ok result => result
  | .error message => panic! s!"Tensor.meanDim: {message}"
def exp? {σ : StaticSpec} (t : Tensor σ) : Except String (Tensor σ) := Tensor.exp? t

def exp {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  neuralUnaryResult "Preserving.exp" t (exp? t)

def sin? {σ : StaticSpec} (t : Tensor σ) : Except String (Tensor σ) := Tensor.sin? t

def sin {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  neuralUnaryResult "Preserving.sin" t (sin? t)

def cos? {σ : StaticSpec} (t : Tensor σ) : Except String (Tensor σ) := Tensor.cos? t

def cos {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  neuralUnaryResult "Preserving.cos" t (cos? t)

def elu? {σ : StaticSpec} (t : Tensor σ) : Except String (Tensor σ) := Tensor.elu? t

def elu {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  neuralUnaryResult "Preserving.elu" t (elu? t)
end Preserving

/-- Spatial extent computed without fixed-width arithmetic wrapping. -/
def convExtent (input kernel stride padding dilation : UInt64) : UInt64 :=
  ((input.toNat + 2 * padding.toNat - (dilation.toNat * (kernel.toNat - 1) + 1)) /
    stride.toNat + 1).toUInt64

/-- Project a spatial parameter, repeating singleton arrays as native conv2d does.
Only lengths one and two are valid; the native operation still receives the
original array and rejects invalid lengths. The zero for a missing entry only
makes the result index total and is never supplied as a native default. -/
def conv2dParam (values : Array UInt64) (axis : Nat) : UInt64 :=
  values.getD (if values.size == 1 then 0 else axis) 0

private def checkConvolutionOperands {σ τ υ : StaticSpec}
    (input : Tensor σ) (weight : Tensor τ) (bias : Tensor υ) : Except String Unit := do
  DType.expectedEq input.raw.dtype weight.raw.dtype
  DType.expectedEq input.raw.dtype bias.raw.dtype
  Tensor.checkSameDevice input weight
  Tensor.checkSameDevice input bias

/-- Checked convolution, including actual dtype equality for dynamic operands.
Native stride, padding and dilation errors remain recoverable. -/
def conv2dBias? {batch cin cout height width kh kw : UInt64} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, cin, height, width], d, dev⟩)
    (weight : Tensor ⟨#[cout, cin, kh, kw], d, dev⟩)
    (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : Array UInt64 := #[1, 1]) (padding : Array UInt64 := #[0, 0])
    (dilation : Array UInt64 := #[1, 1]) :
    Except String (Tensor ⟨#[batch, cout,
      convExtent height kh (conv2dParam stride 0) (conv2dParam padding 0) (conv2dParam dilation 0),
      convExtent width kw (conv2dParam stride 1) (conv2dParam padding 1) (conv2dParam dilation 1)], d, dev⟩) := do
  checkConvolutionOperands input weight bias
  let raw ← RawTensor.conv2dBiasChecked input.shape weight.shape bias.shape
    input.raw weight.raw bias.raw stride padding dilation
  DType.expectedEq input.raw.dtype raw.dtype
  Tensor.ofRaw raw

/-- Biased two-dimensional convolution; `conv2dBias?` returns argument errors. -/
def conv2dBias {batch cin cout height width kh kw : UInt64} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, cin, height, width], d, dev⟩)
    (weight : Tensor ⟨#[cout, cin, kh, kw], d, dev⟩)
    (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : Array UInt64 := #[1, 1]) (padding : Array UInt64 := #[0, 0])
    (dilation : Array UInt64 := #[1, 1]) :
    Tensor ⟨#[batch, cout,
      convExtent height kh (conv2dParam stride 0) (conv2dParam padding 0) (conv2dParam dilation 0),
      convExtent width kw (conv2dParam stride 1) (conv2dParam padding 1) (conv2dParam dilation 1)], d, dev⟩ :=
  match conv2dBias? input weight bias stride padding dilation with
  | .ok result => result
  | .error message => panic! s!"Tensor.conv2dBias: {message}"

def splitHeads {batch seq heads dim : UInt64} {d : DType} {dev : Device}
    (t : Tensor ⟨#[batch, seq, heads, dim], d, dev⟩) : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  checkedNeuralResult "splitHeads" (RawTensor.splitHeads batch seq heads dim t.raw)

def joinHeads {batch heads seq dim : UInt64} {d : DType} {dev : Device}
    (t : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩) : Tensor ⟨#[batch, seq, heads, dim], d, dev⟩ :=
  checkedNeuralResult "joinHeads" (RawTensor.joinHeads batch heads seq dim t.raw)

/-- Scaled dot-product attention with a checked shape, dtype and placement result. -/
def attention {batch heads seq dim : UInt64} {d : DType} {dev : Device}
    (q k v : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (dropout : Float := 0.0) (causal : Bool := true) :
    Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  q.checkedReplacement (RawTensor.attention batch heads seq dim q.raw k.raw v.raw dropout causal)

def attentionGQA {batch heads kvHeads seq dim : UInt64} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (dropout : Float := 0.0) (causal : Bool := true) (gqa : Bool := false) :
    Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  q.checkedReplacement (RawTensor.attentionGQA batch heads kvHeads seq dim q.raw k.raw v.raw dropout causal gqa)

def attentionGQAMask {batch heads kvHeads seq dim : UInt64} {d dm : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (mask : Tensor ⟨#[batch, seq], dm, dev⟩)
    (dropout : Float := 0.0) (causal : Bool := true) (gqa : Bool := false) :
    Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  q.checkedReplacement (RawTensor.attentionGQAMask batch heads kvHeads seq dim q.raw k.raw v.raw mask.raw dropout causal gqa)

/-- Gather has the index tensor's shape, and the source tensor's dtype. -/
def gather {s indicesShape : Shape} {d : DType} {dev : Device}
    (input : Tensor ⟨s, d, dev⟩) (dim : Int64)
    (indices : Tensor ⟨indicesShape, .Int64, dev⟩) : Tensor ⟨indicesShape, d, dev⟩ :=
  match Tensor.ofRaw (RawTensor.gather s indicesShape input.raw dim indices.raw) with
  | .ok result => result
  | .error message => panic! s!"Tensor.gather: {message}"

/-- Masked selection returns an existential vector length, checked against the native result. -/
def maskedSelect {s : Shape} {d : DType} {dev : Device}
    (input : Tensor ⟨s, d, dev⟩) (mask : Tensor ⟨s, .Bool, dev⟩) :
    Sigma (fun n => Tensor ⟨#[n], d, dev⟩) :=
  letI : Inhabited (Sigma (fun n => Tensor ⟨#[n], d, dev⟩)) := ⟨⟨0, Tensor.zerosFor _⟩⟩
  let raw := RawTensor.maskedSelect s input.raw mask.raw
  let n := raw.runtimeShape.getD 0 0
  match Tensor.ofRaw (σ := ⟨#[n], d, dev⟩) raw with
  | .ok result => ⟨n, result⟩
  | .error message => panic! s!"Tensor.maskedSelect: {message}"

def maskedScatter {s sourceShape : Shape} {d : DType} {dev : Device}
    (input : Tensor ⟨s, d, dev⟩) (mask : Tensor ⟨s, .Bool, dev⟩)
    (source : Tensor ⟨sourceShape, d, dev⟩) : Tensor ⟨s, d, dev⟩ :=
  input.checkedReplacement (RawTensor.maskedScatter s sourceShape input.raw mask.raw source.raw)

/-- Attention over different query and KV lengths, as used in cached decoding. -/
def attentionGQAQKV {batch heads kvHeads qSeq kvSeq dim : UInt64} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, kvSeq, dim], d, dev⟩)
    (dropout : Float := 0.0) (causal : Bool := false) (gqa : Bool := false) :=
  q.checkedReplacement (RawTensor.attentionGQAQKV batch heads kvHeads qSeq kvSeq dim
    q.raw k.raw v.raw dropout causal gqa)

/-- Native FlashAttention dispatch with recoverable native errors and a
metadata-checked result. -/
def flashAttention? {batch heads kvHeads qSeq kvSeq dim : UInt64} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, kvSeq, dim], d, dev⟩) (gqa : Bool := false)
    (mask : Option (Tensor { shape := #[batch, kvSeq], device := dev }) := none) (dropout : Float := 0.0)
    (causal : Bool := false) (scale : Option Float := none) :
    Except String (Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩) := do
  let raw ← RawTensor.flashAttention batch heads kvHeads qSeq kvSeq dim
    q.raw k.raw v.raw (mask.map Tensor.raw) dropout causal scale gqa
  q.checkedReplacement? raw

/-- FlashAttention convenience wrapper. Use `flashAttention?` to handle native
validation or execution failures explicitly. -/
def flashAttention {batch heads kvHeads qSeq kvSeq dim : UInt64} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, kvSeq, dim], d, dev⟩) (gqa : Bool := false)
    (mask : Option (Tensor { shape := #[batch, kvSeq], device := dev }) := none) (dropout : Float := 0.0)
    (causal : Bool := false) (scale : Option Float := none) :
    Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩ :=
  letI : Inhabited (Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩) := ⟨q⟩
  match flashAttention? q k v gqa mask dropout causal scale with
  | .ok result => result
  | .error message => panic! s!"Tensor.flashAttention: {message}"

def sliceScatter {s sourceShape : Shape} {d : DType} {dev : Device}
    (input : Tensor ⟨s, d, dev⟩) (dim start : UInt64)
    (source : Tensor ⟨sourceShape, d, dev⟩) : Tensor ⟨s, d, dev⟩ :=
  let dtype := input.raw.dtype
  let device := input.raw.device
  Tensor.checkedUpdate dtype device
    (RawTensor.sliceScatter s sourceShape input.raw dim start source.raw)

def applyRotary {batch seq heads dim : UInt64} {d dc : DType} {dev : Device}
    (input : Tensor ⟨#[batch, seq, heads, dim], d, dev⟩)
    (cos sin : Tensor ⟨#[seq, dim / 2], dc, dev⟩) :=
  input.checkedReplacement (RawTensor.applyRotary batch seq heads dim input.raw cos.raw sin.raw)

def rotaryFreqs (seq dim : UInt64) (base : Float := 10000.0) (device : Device := .CPU) :
    Tensor { shape := #[seq, dim / 2], dtype := .Float32, device } × Tensor { shape := #[seq, dim / 2], dtype := .Float32, device } :=
  let (cos, sin) := RawTensor.rotaryFreqs seq dim base device
  match Tensor.ofRaw cos, Tensor.ofRaw sin with
  | .ok c, .ok s => (c, s)
  | .error message, _ | _, .error message => panic! s!"Tensor.rotaryFreqs: {message}"

def hannWindow (n : UInt64) : Tensor { shape := #[n], dtype := .Float32, device := .CPU } :=
  match Tensor.ofRaw (RawTensor.hannWindow n) with
  | .ok t => t
  | .error message => panic! s!"Tensor.hannWindow: {message}"

/-- The native STFT frame count is returned as a checked existential shape. -/
def stft {n winLength : UInt64} {dev : Device} (input : Tensor { shape := #[n], dtype := .Float32, device := dev }) (nFft hopLength : UInt64)
    (window : Tensor { shape := #[winLength], dtype := .Float32, device := dev }) (center : Bool := true) (normalized : Bool := false) :
    IO (Sigma fun shape : Shape => Tensor { shape := shape, dtype := .Float32, device := dev }) := do
  let raw ← match RawTensor.stftChecked n input.raw nFft hopLength winLength window.raw center normalized with
    | .ok raw => pure raw
    | .error message => throw <| IO.userError s!"Tensor.stft: {message}"
  let shape := raw.runtimeShape
  match Tensor.ofRaw (σ := ⟨shape, .Float32, dev⟩) raw with
  | .ok t => pure ⟨shape, t⟩
  | .error message => throw <| IO.userError s!"Tensor.stft: {message}"

def log10? {σ : StaticSpec} (input : Tensor σ) : Except String (Tensor σ) :=
  checkedNeuralUnary? "log10" "a floating or complex dtype" DType.isInexact RawTensor.log10Checked input

def log10 {σ : StaticSpec} (input : Tensor σ)
    (_inexact : σ.dtype.isInexact = true := by first | rfl | decide) : Tensor σ :=
  neuralUnaryResult "log10" input (log10? input)

namespace Preserving

def log10? {σ : StaticSpec} (input : Tensor σ) : Except String (Tensor σ) := Tensor.log10? input

def log10 {σ : StaticSpec} (input : Tensor σ) : Tensor σ :=
  neuralUnaryResult "Preserving.log10" input (log10? input)

end Preserving

def saveWav {σ : StaticSpec} (input : Tensor σ) (path : String) (sampleRate : UInt64 := 24000) : IO Unit :=
  RawTensor.saveWav σ.shape input.raw path sampleRate

def lt? {s : Shape} {d : DType} {dev : Device} (a b : Tensor ⟨s, d, dev⟩) :
    Except String (Tensor ⟨s, .Bool, dev⟩) := do
  DType.checkOrdered a.raw.dtype
  DType.checkOrdered b.raw.dtype
  Tensor.checkPointwise a b
  Tensor.ofRaw (← RawTensor.ltChecked s a.raw b.raw)

def lt {s : Shape} {d : DType} {dev : Device} (a b : Tensor ⟨s, d, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  match lt? a b with
  | .ok result => result
  | .error message => panic! s!"Tensor.lt: {message}"

def ge? {s : Shape} {d : DType} {dev : Device} (a b : Tensor ⟨s, d, dev⟩) :
    Except String (Tensor ⟨s, .Bool, dev⟩) := do
  DType.checkOrdered a.raw.dtype
  DType.checkOrdered b.raw.dtype
  Tensor.checkPointwise a b
  Tensor.ofRaw (← RawTensor.geChecked s a.raw b.raw)

def ge {s : Shape} {d : DType} {dev : Device} (a b : Tensor ⟨s, d, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  match ge? a b with
  | .ok result => result
  | .error message => panic! s!"Tensor.ge: {message}"
def ltScalar {s : Shape} {d : DType} {dev : Device} (input : Tensor ⟨s, d, dev⟩) (value : Float) :
    Tensor ⟨s, .Bool, dev⟩ := checkedNeuralResult "ltScalar" (RawTensor.ltScalar s input.raw value)
def maskedFill {s : Shape} {d : DType} {dev : Device}
    (input : Tensor ⟨s, d, dev⟩) (mask : Tensor ⟨s, .Bool, dev⟩) (value : Float) : Tensor ⟨s, d, dev⟩ :=
  input.checkedReplacement (RawTensor.maskedFill s input.raw mask.raw value)

/-- Grouped convolution checks channel grouping at the native boundary. -/
def conv1dGroups? {batch cin cout cinPerGroup length kernel : UInt64} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, cin, length], d, dev⟩)
    (weight : Tensor ⟨#[cout, cinPerGroup, kernel], d, dev⟩) (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : UInt64 := 1) (padding : UInt64 := 0) (dilation : UInt64 := 1) (groups : UInt64 := 1) :
    Except String (Tensor ⟨#[batch, cout, convExtent length kernel stride padding dilation], d, dev⟩) := do
  checkConvolutionOperands input weight bias
  let raw ← RawTensor.conv1dGroupsChecked input.shape weight.shape bias.shape
    input.raw weight.raw bias.raw stride padding dilation groups
  DType.expectedEq input.raw.dtype raw.dtype
  Tensor.ofRaw raw

/-- Grouped convolution; use `conv1dGroups?` for recoverable argument errors. -/
def conv1dGroups {batch cin cout cinPerGroup length kernel : UInt64} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, cin, length], d, dev⟩)
    (weight : Tensor ⟨#[cout, cinPerGroup, kernel], d, dev⟩) (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : UInt64 := 1) (padding : UInt64 := 0) (dilation : UInt64 := 1) (groups : UInt64 := 1) :
    Tensor ⟨#[batch, cout, convExtent length kernel stride padding dilation], d, dev⟩ :=
  match conv1dGroups? input weight bias stride padding dilation groups with
  | .ok result => result
  | .error message => panic! s!"Tensor.conv1dGroups: {message}"

/-- Checked transposed convolution, including native output-padding validation. -/
def convTranspose1d? {batch cin cout length kernel : UInt64} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, cin, length], d, dev⟩)
    (weight : Tensor ⟨#[cin, cout, kernel], d, dev⟩) (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : UInt64 := 1) (padding : UInt64 := 0) (outputPadding : UInt64 := 0) (dilation : UInt64 := 1) :
    Except String (Tensor ⟨#[batch, cout, ((length.toNat - 1) * stride.toNat + dilation.toNat * (kernel.toNat - 1)
      + outputPadding.toNat + 1 - 2 * padding.toNat).toUInt64], d, dev⟩) := do
  checkConvolutionOperands input weight bias
  let raw ← RawTensor.convTranspose1dChecked input.shape weight.shape bias.shape
    input.raw weight.raw bias.raw stride padding outputPadding dilation
  DType.expectedEq input.raw.dtype raw.dtype
  Tensor.ofRaw raw

/-- Transposed convolution; `convTranspose1d?` returns argument errors. -/
def convTranspose1d {batch cin cout length kernel : UInt64} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, cin, length], d, dev⟩)
    (weight : Tensor ⟨#[cin, cout, kernel], d, dev⟩) (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : UInt64 := 1) (padding : UInt64 := 0) (outputPadding : UInt64 := 0) (dilation : UInt64 := 1) :
    Tensor ⟨#[batch, cout, ((length.toNat - 1) * stride.toNat + dilation.toNat * (kernel.toNat - 1)
      + outputPadding.toNat + 1 - 2 * padding.toNat).toUInt64], d, dev⟩ :=
  match convTranspose1d? input weight bias stride padding outputPadding dilation with
  | .ok result => result
  | .error message => panic! s!"Tensor.convTranspose1d: {message}"

def attentionWindow {batch heads kvHeads seq dim : UInt64} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (dropout : Float := 0.0) (causal : Bool := true) (gqa : Bool := false) (window : UInt64 := 0) :=
  q.checkedReplacement (RawTensor.attentionWindow batch heads kvHeads seq dim q.raw k.raw v.raw dropout causal gqa window)

def log? {σ : StaticSpec} (input : Tensor σ) : Except String (Tensor σ) :=
  checkedNeuralUnary? "log" "a floating or complex dtype" DType.isInexact RawTensor.logChecked input

def log {σ : StaticSpec} (input : Tensor σ)
    (_inexact : σ.dtype.isInexact = true := by first | rfl | decide) : Tensor σ :=
  neuralUnaryResult "log" input (log? input)

namespace Preserving

def log? {σ : StaticSpec} (input : Tensor σ) : Except String (Tensor σ) := Tensor.log? input

def log {σ : StaticSpec} (input : Tensor σ) : Tensor σ :=
  neuralUnaryResult "Preserving.log" input (log? input)

end Preserving

def saveSafeTensor {σ : StaticSpec} (input : Tensor σ) (path name : String) : IO Unit :=
  RawTensor.saveSafeTensor σ.shape path name input.raw

/-- Clip the parameter's accumulated gradient in place, returning its pre-clip norm. -/
def clipGradNorm {σ : StaticSpec} (input : Tensor σ) (maxNorm : Float) : IO Float :=
  RawTensor.clipGradNorm σ.shape input.raw maxNorm

def saveTensor {σ : StaticSpec} (input : Tensor σ) (path : String) : IO Unit :=
  RawTensor.saveTensor σ.shape input.raw path

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
def wavBegin (path : String) (sampleRate : UInt64 := 24000) : IO Unit :=
  RawTensor.wavBegin path sampleRate

/-- Append samples, converting to CPU Float32 PCM at the native file boundary. -/
def wavAppend {σ : StaticSpec} (input : Tensor σ) (path : String) : IO Unit :=
  RawTensor.wavAppend σ.shape input.raw path

/-- Patch the streaming WAV header to match the final sample count. -/
def wavFinalize (path : String) : IO Unit := RawTensor.wavFinalize path

/-- Load UInt16 token data as an indexed Int64 tensor, validating native metadata. -/
def loadU16Bin (count : UInt64) (path : String) : IO (Tensor { shape := #[count], dtype := .Int64, device := .CPU }) := do
  let raw ← RawTensor.loadU16Bin count path
  match Tensor.ofRaw raw with
  | .ok tensor => pure tensor
  | .error message => throw <| IO.userError s!"Tensor.loadU16Bin: {message}"

def linspace (start stop : Float) (count : UInt64) : Tensor { shape := #[count], dtype := .Float32, device := .CPU } :=
  checkedNeuralResult "linspace" (RawTensor.linspace start stop count false)

/-- Reduce the last axis with recoverable ordered-domain, empty-axis and
native kernel failures. Values preserve the actual input dtype. -/
def maxLast3d? {batch seq vocab : UInt64} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, seq, vocab], d, dev⟩) :
    Except String (Tensor ⟨#[batch, seq], d, dev⟩ × Tensor ⟨#[batch, seq], .Int64, dev⟩) := do
  DType.checkOrdered input.raw.dtype
  if vocab == 0 then throw "maxLast3d requires a nonempty reduction axis"
  let (values, indices) ← RawTensor.maxLast3dChecked batch seq vocab input.raw 2
  return (← Tensor.checkedUpdate? input.raw.dtype input.raw.device values, ← Tensor.ofRaw indices)

/-- Runtime-checked pure convenience; use `maxLast3d?` to retain errors. -/
def maxLast3dChecked {batch seq vocab : UInt64} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, seq, vocab], d, dev⟩) :
    Tensor ⟨#[batch, seq], d, dev⟩ × Tensor ⟨#[batch, seq], .Int64, dev⟩ :=
  match maxLast3d? input with
  | .ok result => result
  | .error message => panic! s!"Tensor.maxLast3d: {message}"

/-- Reduce a nonempty last axis of an ordered tensor. -/
def maxLast3d {batch seq vocab : UInt64} {d : DType} {dev : Device} [DType.Ordered d]
    (input : Tensor ⟨#[batch, seq, vocab], d, dev⟩)
    (_nonempty : 0 < vocab.toNat := by exact of_decide_eq_true rfl) :
    Tensor ⟨#[batch, seq], d, dev⟩ × Tensor ⟨#[batch, seq], .Int64, dev⟩ :=
  maxLast3dChecked input

/-- Select the largest `k` entries per row, preserving the actual input dtype.
Runtime dtype/range and native kernel failures are recoverable. -/
def topKLast2d? {rows cols : UInt64} {d : DType} {dev : Device}
    (input : Tensor ⟨#[rows, cols], d, dev⟩) (k : UInt64) :
    Except String (Tensor ⟨#[rows, k], d, dev⟩ × Tensor ⟨#[rows, k], .Int64, dev⟩) := do
  DType.checkOrdered input.raw.dtype
  if k > cols then throw s!"topKLast2d k={k} exceeds the row width {cols}"
  let (values, indices) ← RawTensor.topKLast2dChecked rows cols input.raw k 1
  return (← Tensor.checkedUpdate? input.raw.dtype input.raw.device values, ← Tensor.ofRaw indices)

/-- Runtime-checked pure convenience; use `topKLast2d?` to retain errors. -/
def topKLast2dChecked {rows cols : UInt64} {d : DType} {dev : Device}
    (input : Tensor ⟨#[rows, cols], d, dev⟩) (k : UInt64) :
    Tensor ⟨#[rows, k], d, dev⟩ × Tensor ⟨#[rows, k], .Int64, dev⟩ :=
  match topKLast2d? input k with
  | .ok result => result
  | .error message => panic! s!"Tensor.topKLast2d: {message}"

/-- Select the largest `k` entries of ordered rows with sufficient width. -/
def topKLast2d {rows cols : UInt64} {d : DType} {dev : Device} [DType.Ordered d]
    (input : Tensor ⟨#[rows, cols], d, dev⟩) (k : UInt64)
    (_range : k.toNat ≤ cols.toNat := by exact of_decide_eq_true rfl) :
    Tensor ⟨#[rows, k], d, dev⟩ × Tensor ⟨#[rows, k], .Int64, dev⟩ :=
  topKLast2dChecked input k

/-- End-exclusive integer range, with overflow-safe length computation. -/
def rangeLength (start stop step : UInt64) : UInt64 :=
  if step == 0 || stop <= start then 0
  else ((stop.toNat - start.toNat + step.toNat - 1) / step.toNat).toUInt64

def arangeFrom (start stop : UInt64) (step : UInt64 := 1) :
    Tensor { shape := #[rangeLength start stop step], dtype := .Int64, device := .CPU } :=
  match Tensor.ofRaw (RawTensor.arange start stop step) with
  | .ok result => result
  | .error message => panic! s!"Tensor.arangeFrom: {message}"

def getOp {σ : StaticSpec} (input : Tensor σ) (idx : Int) :
    Tensor (σ.withShape σ.shape[1:].toArray) :=
  match Tensor.ofRaw (RawTensor.getOp σ.shape input.raw idx) with
  | .ok result => result
  | .error message => panic! s!"Tensor.getOp: {message}"

def unbind {σ : StaticSpec} (input : Tensor σ) (dim : Nat := 0) :
    Array (Tensor (σ.withShape (unbindShape σ.shape dim))) :=
  (RawTensor.unbind σ.shape input.raw dim).map fun raw =>
    match Tensor.ofRaw raw with
    | .ok result => result
    | .error message => panic! s!"Tensor.unbind: {message}"

def logspace (start stop : Float) (steps : UInt64) (base : Float := 10.0) :
    Tensor { shape := #[steps], dtype := .Float32, device := .CPU } :=
  match Tensor.ofRaw (RawTensor.logspace start stop steps base false) with
  | .ok result => result
  | .error message => panic! s!"Tensor.logspace: {message}"

/-- Backpropagate an explicit cotangent through the output graph. -/
def backward {σ : StaticSpec} (output cotangent : Tensor σ) : IO Unit :=
  RawTensor.backward σ.shape output.raw cotangent.raw

def hasGradFn {σ : StaticSpec} (input : Tensor σ) : Bool :=
  RawTensor.hasGradFn σ.shape input.raw

/-- Find matching token positions, retaining the runtime-dependent vector length. -/
def findTokenPositions {n : UInt64} {dev : Device}
    (tokens : Tensor ⟨#[n], .Int64, dev⟩) (token : Int64) :
    IO (Sigma fun count => Tensor ⟨#[count], .Int64, dev⟩) := do
  let raw ← RawTensor.findTokenPositions n tokens.raw token
  let count := raw.runtimeShape.getD 0 0
  match Tensor.ofRaw (σ := ⟨#[count], .Int64, dev⟩) raw with
  | .ok result => pure ⟨count, result⟩
  | .error message => throw (IO.userError s!"Tensor.findTokenPositions: {message}")

namespace Promoting

private def unaryNeural? {σ : StaticSpec}
    (native : Shape → RawTensor → Except String RawTensor) (input : Tensor σ) :
    Except String (Tensor (σ.withDType input.actualSpec.dtype.atLeastFloat)) := do
  Tensor.checkedUpdate? input.actualSpec.dtype.atLeastFloat input.raw.device
    (← native σ.shape input.raw)

def log? {σ : StaticSpec} (input : Tensor σ) :
    Except String (Tensor (σ.withDType input.actualSpec.dtype.atLeastFloat)) :=
  unaryNeural? RawTensor.logChecked input

def log {σ : StaticSpec} (input : Tensor σ) :
    Tensor (σ.withDType (.Unknown "dynamic")) :=
  let result : Except String (Tensor (σ.withDType (.Unknown "dynamic"))) := do
    Tensor.ofRaw (← log? input).raw
  match result with
  | .ok result => result
  | .error message => panic! s!"Tensor.Promoting.log: {message}"

def log10? {σ : StaticSpec} (input : Tensor σ) :
    Except String (Tensor (σ.withDType input.actualSpec.dtype.atLeastFloat)) :=
  unaryNeural? RawTensor.log10Checked input

def log10 {σ : StaticSpec} (input : Tensor σ) :
    Tensor (σ.withDType (.Unknown "dynamic")) :=
  let result : Except String (Tensor (σ.withDType (.Unknown "dynamic"))) := do
    Tensor.ofRaw (← log10? input).raw
  match result with
  | .ok result => result
  | .error message => panic! s!"Tensor.Promoting.log10: {message}"

def sin? {σ : StaticSpec} (input : Tensor σ) :
    Except String (Tensor (σ.withDType input.actualSpec.dtype.atLeastFloat)) :=
  unaryNeural? RawTensor.sinChecked input

def sin {σ : StaticSpec} (input : Tensor σ) :
    Tensor (σ.withDType (.Unknown "dynamic")) :=
  let result : Except String (Tensor (σ.withDType (.Unknown "dynamic"))) := do
    Tensor.ofRaw (← sin? input).raw
  match result with
  | .ok result => result
  | .error message => panic! s!"Tensor.Promoting.sin: {message}"

def cos? {σ : StaticSpec} (input : Tensor σ) :
    Except String (Tensor (σ.withDType input.actualSpec.dtype.atLeastFloat)) :=
  unaryNeural? RawTensor.cosChecked input

def cos {σ : StaticSpec} (input : Tensor σ) :
    Tensor (σ.withDType (.Unknown "dynamic")) :=
  let result : Except String (Tensor (σ.withDType (.Unknown "dynamic"))) := do
    Tensor.ofRaw (← cos? input).raw
  match result with
  | .ok result => result
  | .error message => panic! s!"Tensor.Promoting.cos: {message}"

end Promoting

end torch.Tensor
