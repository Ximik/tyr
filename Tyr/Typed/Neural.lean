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
def copyInto {σ : StaticSpec} (dst src : Tensor σ) : IO (Tensor σ) := do
  match TensorSpec.checkCompatible dst.actualSpec src.actualSpec with
  | .error message => throw (IO.userError s!"Tensor.copyInto: {message}")
  | .ok () => pure ()
  let dtype := dst.raw.dtype
  let device := dst.raw.device
  let raw ← RawTensor.copyInto σ.shape dst.raw src.raw
  match Tensor.checkedUpdate? dtype device raw with
  | .ok result => pure result
  | .error message => throw (IO.userError s!"Tensor.copyInto: {message}")

/-- Forget only the dtype claim. Shape and placement remain indexed. -/
def forgetDType {σ : StaticSpec} (t : Tensor σ) : Tensor { σ with dtype := .Unknown "dynamic" } :=
  checkedNeuralResult "forgetDType" t.raw

/-- Checked dtype conversion. Only the reserved `dynamic` dtype weakens the
claim without casting; unsupported target tokens return an error. -/
def castDType? {σ : StaticSpec} (t : Tensor σ) (dtype : DType) :
    Except String (Tensor (σ.withDType dtype)) := do
  let raw ← match dtype.safeTensorTag? with
    | some tag => pure (RawTensor.castDType σ.shape t.raw tag)
    | none =>
        if dtype == .Unknown "dynamic" then pure t.raw
        else throw s!"Unsupported target dtype {dtype}"
  DevicePolicy.check (.exact t.raw.device) raw.device
  Tensor.ofRaw raw

/-- Cast to a concrete dtype; use `castDType?` to handle an unsupported token. -/
def castDType {σ : StaticSpec} (t : Tensor σ) (dtype : DType) : Tensor (σ.withDType dtype) :=
  match t.castDType? dtype with
  | .ok result => result
  | .error message => panic! s!"Tensor.castDType: {message}"

/-- Match the reference's actual dtype and preserve input shape and placement.
An unsupported native dtype may only be retained when both inputs already agree. -/
def castLike? {σ τ : StaticSpec} (reference : Tensor τ) (t : Tensor σ) :
    Except String (Tensor (σ.withDType τ.dtype)) := do
  let dtype := reference.raw.dtype
  let raw ← if t.raw.dtype == dtype then pure t.raw else
    match dtype.safeTensorTag? with
    | some tag => pure (RawTensor.castDType σ.shape t.raw tag)
    | none => throw s!"Unsupported reference dtype {dtype}"
  TensorSpec.checkDType dtype raw.dtype
  DevicePolicy.check (.exact t.raw.device) raw.device
  Tensor.ofRaw raw

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
  x.checkedReplacement (RawTensor.mul σ.shape restored.raw weight.raw)

/-- Package a tensor's shape existentially; this does not erase its shape claim. -/
def packShape {s : Shape} {d : DType} {dev : DevicePolicy} (t : Tensor ⟨s, d, dev⟩) :
    Sigma (fun shape => Tensor ⟨shape, d, dev⟩) := ⟨s, t⟩

/-- Concatenate tensors with runtime-varying shapes, checking the requested result.
Inputs retain their individual shape indices rather than pretending to be scalars. -/
def catMany? {d : DType} {dev : DevicePolicy}
    (tensors : Array (Sigma (fun shape => Tensor ⟨shape, d, dev⟩)))
    (dim : Int) (shape : Shape) : Except String (Tensor ⟨shape, d, dev⟩) :=
  Tensor.ofRaw (RawTensor.catMany (tensors.map fun t => t.2.raw) dim)

def catManyChecked {d : DType} {dev : DevicePolicy}
    (tensors : Array (Sigma (fun shape => Tensor ⟨shape, d, dev⟩)))
    (dim : Int) (shape : Shape) : Tensor ⟨shape, d, dev⟩ :=
  match catMany? tensors dim shape with
  | .ok result => result
  | .error message => panic! s!"Tensor.catManyChecked: {message}"

def arange (stop : UInt64) : Tensor { shape := #[stop], dtype := .Int64 } :=
  checkedNeuralResult "arange" (RawTensor.arange 0 stop 1)

def sin {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) : Tensor σ :=
  t.checkedReplacement (RawTensor.sin σ.shape t.raw)

def cos {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) : Tensor σ :=
  t.checkedReplacement (RawTensor.cos σ.shape t.raw)

namespace Preserving
/-- Metadata-checked activation for models whose checkpoint dtype is dynamic. -/
def gelu {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  t.checkedReplacement (RawTensor.gelu σ.shape t.raw)
def silu {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  t.checkedReplacement (RawTensor.silu σ.shape t.raw)
def sigmoid {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  t.checkedReplacement (RawTensor.sigmoid σ.shape t.raw)
def tanh {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  t.checkedReplacement (RawTensor.tanh σ.shape t.raw)
def softmax {σ : StaticSpec} (t : Tensor σ) (dim : Int32 := -1) : Tensor σ :=
  t.checkedReplacement (RawTensor.softmax σ.shape t.raw dim)
/-- Mean for floating checkpoint tensors with a possibly dynamic dtype index. -/
def meanDim {σ : StaticSpec} (t : Tensor σ) (dim : Nat) (keepdim : Bool := false) :
    Tensor (σ.withShape (reduceShape σ.shape dim keepdim)) :=
  let raw := RawTensor.mean σ.shape t.raw (some #[dim.toUInt64]) keepdim
  let result := do
    TensorSpec.checkDType t.raw.dtype raw.dtype
    DevicePolicy.check (.exact t.raw.device) raw.device
    Tensor.ofRaw raw
  match result with
  | .ok result => result
  | .error message => panic! s!"Tensor.meanDim: {message}"
def exp {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  t.checkedReplacement (RawTensor.exp σ.shape t.raw)
def sin {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  t.checkedReplacement (RawTensor.sin σ.shape t.raw)
def elu {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  t.checkedReplacement (RawTensor.elu σ.shape t.raw)
end Preserving

/-- Spatial extent computed without fixed-width arithmetic wrapping. -/
def convExtent (input kernel stride padding dilation : UInt64) : UInt64 :=
  ((input.toNat + 2 * padding.toNat - (dilation.toNat * (kernel.toNat - 1) + 1)) /
    stride.toNat + 1).toUInt64

def conv2dBias {batch cin cout height width kh kw : UInt64} {d : DType} {dev : DevicePolicy}
    (input : Tensor ⟨#[batch, cin, height, width], d, dev⟩)
    (weight : Tensor ⟨#[cout, cin, kh, kw], d, dev⟩)
    (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : Array UInt64 := #[1, 1]) (padding : Array UInt64 := #[0, 0])
    (dilation : Array UInt64 := #[1, 1]) :
    Tensor ⟨#[batch, cout,
      convExtent height kh (stride.getD 0 1) (padding.getD 0 0) (dilation.getD 0 1),
      convExtent width kw (stride.getD 1 1) (padding.getD 1 0) (dilation.getD 1 1)], d, dev⟩ :=
  match Tensor.ofRaw (RawTensor.conv2dBias input.shape weight.shape bias.shape
      input.raw weight.raw bias.raw stride padding dilation) with
  | .ok result => result
  | .error message => panic! s!"Tensor.conv2dBias: {message}"

def splitHeads {batch seq heads dim : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[batch, seq, heads, dim], d, dev⟩) : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  checkedNeuralResult "splitHeads" (RawTensor.splitHeads batch seq heads dim t.raw)

def joinHeads {batch heads seq dim : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩) : Tensor ⟨#[batch, seq, heads, dim], d, dev⟩ :=
  checkedNeuralResult "joinHeads" (RawTensor.joinHeads batch heads seq dim t.raw)

/-- Scaled dot-product attention with a checked shape, dtype and placement result. -/
def attention {batch heads seq dim : UInt64} {d : DType} {dev : DevicePolicy}
    (q k v : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (dropout : Float := 0.0) (causal : Bool := true) :
    Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  q.checkedReplacement (RawTensor.attention batch heads seq dim q.raw k.raw v.raw dropout causal)

def attentionGQA {batch heads kvHeads seq dim : UInt64} {d : DType} {dev : DevicePolicy}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (dropout : Float := 0.0) (causal : Bool := true) (gqa : Bool := false) :
    Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  q.checkedReplacement (RawTensor.attentionGQA batch heads kvHeads seq dim q.raw k.raw v.raw dropout causal gqa)

def attentionGQAMask {batch heads kvHeads seq dim : UInt64} {d dm : DType} {dev : DevicePolicy}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (mask : Tensor ⟨#[batch, seq], dm, dev⟩)
    (dropout : Float := 0.0) (causal : Bool := true) (gqa : Bool := false) :
    Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  q.checkedReplacement (RawTensor.attentionGQAMask batch heads kvHeads seq dim q.raw k.raw v.raw mask.raw dropout causal gqa)

/-- Gather has the index tensor's shape, and the source tensor's dtype. -/
def gather {s indicesShape : Shape} {d : DType} {dev : DevicePolicy}
    (input : Tensor ⟨s, d, dev⟩) (dim : Int64)
    (indices : Tensor ⟨indicesShape, .Int64, dev⟩) : Tensor ⟨indicesShape, d, dev⟩ :=
  match Tensor.ofRaw (RawTensor.gather s indicesShape input.raw dim indices.raw) with
  | .ok result => result
  | .error message => panic! s!"Tensor.gather: {message}"

/-- Masked selection returns an existential vector length, checked against the native result. -/
def maskedSelect {s : Shape} {d : DType} {dev : DevicePolicy}
    (input : Tensor ⟨s, d, dev⟩) (mask : Tensor ⟨s, .Bool, dev⟩) :
    Sigma (fun n => Tensor ⟨#[n], d, dev⟩) :=
  letI : Inhabited (Sigma (fun n => Tensor ⟨#[n], d, dev⟩)) := ⟨⟨0, Tensor.zerosFor _⟩⟩
  let raw := RawTensor.maskedSelect s input.raw mask.raw
  let n := raw.runtimeShape.getD 0 0
  match Tensor.ofRaw (σ := ⟨#[n], d, dev⟩) raw with
  | .ok result => ⟨n, result⟩
  | .error message => panic! s!"Tensor.maskedSelect: {message}"

def maskedScatter {s sourceShape : Shape} {d : DType} {dev : DevicePolicy}
    (input : Tensor ⟨s, d, dev⟩) (mask : Tensor ⟨s, .Bool, dev⟩)
    (source : Tensor ⟨sourceShape, d, dev⟩) : Tensor ⟨s, d, dev⟩ :=
  input.checkedReplacement (RawTensor.maskedScatter s sourceShape input.raw mask.raw source.raw)

/-- Attention over different query and KV lengths, as used in cached decoding. -/
def attentionGQAQKV {batch heads kvHeads qSeq kvSeq dim : UInt64} {d : DType} {dev : DevicePolicy}
    (q : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, kvSeq, dim], d, dev⟩)
    (dropout : Float := 0.0) (causal : Bool := false) (gqa : Bool := false) :=
  q.checkedReplacement (RawTensor.attentionGQAQKV batch heads kvHeads qSeq kvSeq dim
    q.raw k.raw v.raw dropout causal gqa)

/-- Native FlashAttention dispatch with recoverable native errors and a
metadata-checked result. -/
def flashAttention? {batch heads kvHeads qSeq kvSeq dim : UInt64} {d : DType} {dev : DevicePolicy}
    (q : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, kvSeq, dim], d, dev⟩) (gqa : Bool := false)
    (mask : Option (Tensor { shape := #[batch, kvSeq] }) := none) (dropout : Float := 0.0)
    (causal : Bool := false) (scale : Option Float := none) :
    Except String (Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩) := do
  let raw ← RawTensor.flashAttention batch heads kvHeads qSeq kvSeq dim
    q.raw k.raw v.raw (mask.map Tensor.raw) dropout causal scale gqa
  q.checkedReplacement? raw

/-- FlashAttention convenience wrapper. Use `flashAttention?` to handle native
validation or execution failures explicitly. -/
def flashAttention {batch heads kvHeads qSeq kvSeq dim : UInt64} {d : DType} {dev : DevicePolicy}
    (q : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, kvSeq, dim], d, dev⟩) (gqa : Bool := false)
    (mask : Option (Tensor { shape := #[batch, kvSeq] }) := none) (dropout : Float := 0.0)
    (causal : Bool := false) (scale : Option Float := none) :
    Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩ :=
  letI : Inhabited (Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩) := ⟨q⟩
  match flashAttention? q k v gqa mask dropout causal scale with
  | .ok result => result
  | .error message => panic! s!"Tensor.flashAttention: {message}"

def sliceScatter {s sourceShape : Shape} {d : DType} {dev : DevicePolicy}
    (input : Tensor ⟨s, d, dev⟩) (dim start : UInt64)
    (source : Tensor ⟨sourceShape, d, dev⟩) : Tensor ⟨s, d, dev⟩ :=
  let dtype := input.raw.dtype
  let device := input.raw.device
  Tensor.checkedUpdate dtype device
    (RawTensor.sliceScatter s sourceShape input.raw dim start source.raw)

def applyRotary {batch seq heads dim : UInt64} {d dc : DType} {dev : DevicePolicy}
    (input : Tensor ⟨#[batch, seq, heads, dim], d, dev⟩)
    (cos sin : Tensor ⟨#[seq, dim / 2], dc, dev⟩) :=
  input.checkedReplacement (RawTensor.applyRotary batch seq heads dim input.raw cos.raw sin.raw)

def rotaryFreqs (seq dim : UInt64) (base : Float := 10000.0) (device : Device := .CPU) :
    Tensor { shape := #[seq, dim / 2], dtype := .Float32 } × Tensor { shape := #[seq, dim / 2], dtype := .Float32 } :=
  let (cos, sin) := RawTensor.rotaryFreqs seq dim base device
  match Tensor.ofRaw cos, Tensor.ofRaw sin with
  | .ok c, .ok s => (c, s)
  | .error message, _ | _, .error message => panic! s!"Tensor.rotaryFreqs: {message}"

def hannWindow (n : UInt64) : Tensor { shape := #[n], dtype := .Float32 } :=
  match Tensor.ofRaw (RawTensor.hannWindow n) with
  | .ok t => t
  | .error message => panic! s!"Tensor.hannWindow: {message}"

/-- The native STFT frame count is returned as a checked existential shape. -/
def stft {n winLength : UInt64} (input : Tensor { shape := #[n], dtype := .Float32 }) (nFft hopLength : UInt64)
    (window : Tensor { shape := #[winLength], dtype := .Float32 }) (center : Bool := true) (normalized : Bool := false) :
    IO (Sigma fun shape : Shape => Tensor { shape := shape, dtype := .Float32 }) := do
  let raw := RawTensor.stft n input.raw nFft hopLength winLength window.raw center normalized
  let shape := raw.runtimeShape
  match Tensor.ofRaw (σ := ⟨shape, .Float32, .any⟩) raw with
  | .ok t => pure ⟨shape, t⟩
  | .error message => throw <| IO.userError s!"Tensor.stft: {message}"

def log10 {σ : StaticSpec} (input : Tensor σ)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) : Tensor σ :=
  input.checkedReplacement (RawTensor.log10 σ.shape input.raw)

def saveWav {σ : StaticSpec} (input : Tensor σ) (path : String) (sampleRate : UInt64 := 24000) : IO Unit :=
  RawTensor.saveWav σ.shape input.raw path sampleRate

def lt {s : Shape} {d : DType} {dev : DevicePolicy} (a b : Tensor ⟨s, d, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ := checkedNeuralResult "lt" (RawTensor.lt s a.raw b.raw)
def ge {s : Shape} {d : DType} {dev : DevicePolicy} (a b : Tensor ⟨s, d, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ := checkedNeuralResult "ge" (RawTensor.ge s a.raw b.raw)
def ltScalar {s : Shape} {d : DType} {dev : DevicePolicy} (input : Tensor ⟨s, d, dev⟩) (value : Float) :
    Tensor ⟨s, .Bool, dev⟩ := checkedNeuralResult "ltScalar" (RawTensor.ltScalar s input.raw value)
def maskedFill {s : Shape} {d : DType} {dev : DevicePolicy}
    (input : Tensor ⟨s, d, dev⟩) (mask : Tensor ⟨s, .Bool, dev⟩) (value : Float) : Tensor ⟨s, d, dev⟩ :=
  input.checkedReplacement (RawTensor.maskedFill s input.raw mask.raw value)

/-- Grouped convolution checks channel grouping at the native boundary. -/
def conv1dGroups {batch cin cout cinPerGroup length kernel : UInt64} {d : DType} {dev : DevicePolicy}
    (input : Tensor ⟨#[batch, cin, length], d, dev⟩)
    (weight : Tensor ⟨#[cout, cinPerGroup, kernel], d, dev⟩) (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : UInt64 := 1) (padding : UInt64 := 0) (dilation : UInt64 := 1) (groups : UInt64 := 1) :
    Tensor ⟨#[batch, cout, convExtent length kernel stride padding dilation], d, dev⟩ :=
  match Tensor.ofRaw (RawTensor.conv1dGroups input.shape weight.shape bias.shape
      input.raw weight.raw bias.raw stride padding dilation groups) with
  | .ok result => result
  | .error message => panic! s!"Tensor.conv1dGroups: {message}"

def convTranspose1d {batch cin cout length kernel : UInt64} {d : DType} {dev : DevicePolicy}
    (input : Tensor ⟨#[batch, cin, length], d, dev⟩)
    (weight : Tensor ⟨#[cin, cout, kernel], d, dev⟩) (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : UInt64 := 1) (padding : UInt64 := 0) (outputPadding : UInt64 := 0) (dilation : UInt64 := 1) :
    Tensor ⟨#[batch, cout, ((length.toNat - 1) * stride.toNat + dilation.toNat * (kernel.toNat - 1)
      + outputPadding.toNat + 1 - 2 * padding.toNat).toUInt64], d, dev⟩ :=
  match Tensor.ofRaw (RawTensor.convTranspose1d input.shape weight.shape bias.shape
      input.raw weight.raw bias.raw stride padding outputPadding dilation) with
  | .ok result => result
  | .error message => panic! s!"Tensor.convTranspose1d: {message}"

def attentionWindow {batch heads kvHeads seq dim : UInt64} {d : DType} {dev : DevicePolicy}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (dropout : Float := 0.0) (causal : Bool := true) (gqa : Bool := false) (window : UInt64 := 0) :=
  q.checkedReplacement (RawTensor.attentionWindow batch heads kvHeads seq dim q.raw k.raw v.raw dropout causal gqa window)

def log {σ : StaticSpec} (input : Tensor σ)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) : Tensor σ :=
  input.checkedReplacement (RawTensor.log σ.shape input.raw)

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

/-- Start a mono streaming WAV file with a placeholder header. -/
def wavBegin (path : String) (sampleRate : UInt64 := 24000) : IO Unit :=
  RawTensor.wavBegin path sampleRate

/-- Append samples, converting to CPU Float32 PCM at the native file boundary. -/
def wavAppend {σ : StaticSpec} (input : Tensor σ) (path : String) : IO Unit :=
  RawTensor.wavAppend σ.shape input.raw path

/-- Patch the streaming WAV header to match the final sample count. -/
def wavFinalize (path : String) : IO Unit := RawTensor.wavFinalize path

/-- Load UInt16 token data as an indexed Int64 tensor, validating native metadata. -/
def loadU16Bin (count : UInt64) (path : String) : IO (Tensor { shape := #[count], dtype := .Int64 }) := do
  let raw ← RawTensor.loadU16Bin count path
  match Tensor.ofRaw raw with
  | .ok tensor => pure tensor
  | .error message => throw <| IO.userError s!"Tensor.loadU16Bin: {message}"

def linspace (start stop : Float) (count : UInt64) : Tensor { shape := #[count], dtype := .Float32 } :=
  checkedNeuralResult "linspace" (RawTensor.linspace start stop count false)

/-- Reduce the last axis, preserving floating values and Int64 index metadata. -/
def maxLast3d {batch seq vocab : UInt64} {d : DType} {dev : DevicePolicy}
    (input : Tensor ⟨#[batch, seq, vocab], d, dev⟩) :
    Tensor ⟨#[batch, seq], d, dev⟩ × Tensor ⟨#[batch, seq], .Int64, dev⟩ :=
  let (values, indices) := RawTensor.maxLast3d batch seq vocab input.raw 2
  match Tensor.ofRaw values, Tensor.ofRaw indices with
  | .ok values, .ok indices => (values, indices)
  | .error message, _ | _, .error message => panic! s!"Tensor.maxLast3d: {message}"

/-- Select the largest `k` entries in each row. -/
def topKLast2d {rows cols : UInt64} {d : DType} {dev : DevicePolicy}
    (input : Tensor ⟨#[rows, cols], d, dev⟩) (k : UInt64) :
    Tensor ⟨#[rows, k], d, dev⟩ × Tensor ⟨#[rows, k], .Int64, dev⟩ :=
  let (values, indices) := RawTensor.topKLast2d rows cols input.raw k 1
  match Tensor.ofRaw values, Tensor.ofRaw indices with
  | .ok values, .ok indices => (values, indices)
  | .error message, _ | _, .error message => panic! s!"Tensor.topKLast2d: {message}"

/-- End-exclusive integer range, with overflow-safe length computation. -/
def rangeLength (start stop step : UInt64) : UInt64 :=
  if step == 0 || stop <= start then 0
  else ((stop.toNat - start.toNat + step.toNat - 1) / step.toNat).toUInt64

def arangeFrom (start stop : UInt64) (step : UInt64 := 1) :
    Tensor { shape := #[rangeLength start stop step], dtype := .Int64 } :=
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
    Tensor { shape := #[steps], dtype := .Float32 } :=
  match Tensor.ofRaw (RawTensor.logspace start stop steps base false) with
  | .ok result => result
  | .error message => panic! s!"Tensor.logspace: {message}"

/-- Backpropagate an explicit cotangent through the output graph. -/
def backward {σ : StaticSpec} (output cotangent : Tensor σ) : IO Unit :=
  RawTensor.backward σ.shape output.raw cotangent.raw

def hasGradFn {σ : StaticSpec} (input : Tensor σ) : Bool :=
  RawTensor.hasGradFn σ.shape input.raw

/-- Find matching token positions, retaining the runtime-dependent vector length. -/
def findTokenPositions {n : UInt64} {dev : DevicePolicy}
    (tokens : Tensor ⟨#[n], .Int64, dev⟩) (token : Int64) :
    IO (Sigma fun count => Tensor ⟨#[count], .Int64, dev⟩) := do
  let raw ← RawTensor.findTokenPositions n tokens.raw token
  let count := raw.runtimeShape.getD 0 0
  match Tensor.ofRaw (σ := ⟨#[count], .Int64, dev⟩) raw with
  | .ok result => pure ⟨count, result⟩
  | .error message => throw (IO.userError s!"Tensor.findTokenPositions: {message}")

end torch.Tensor
