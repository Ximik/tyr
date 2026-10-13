import Tyr.Typed.Ops
import Tyr.RawTensor.Neural

namespace torch.Tensor

/-- Forget only the dtype claim. Shape and placement remain indexed. -/
def forgetDType {σ : StaticSpec} (t : Tensor σ) : Tensor { σ with dtype := .Unknown "dynamic" } :=
  .assumeSpec t.raw

/-- Cast to a concrete dtype; a dynamic dtype explicitly makes no dtype claim. -/
def castDType {σ : StaticSpec} (t : Tensor σ) (dtype : DType) : Tensor (σ.withDType dtype) :=
  let raw := match dtype.safeTensorTag? with
    | some tag => RawTensor.castDType σ.shape t.raw tag
    | none => t.raw
  match Tensor.ofRaw raw with
  | .ok result => result
  | .error message => panic! s!"Tensor.castDType: {message}"

/-- Match the reference's native dtype while preserving the input's shape and placement. -/
def castLike {σ τ : StaticSpec} (reference : Tensor τ) (t : Tensor σ) :
    Tensor (σ.withDType τ.dtype) :=
  let raw := match reference.raw.dtype.safeTensorTag? with
    | some tag => RawTensor.castDType σ.shape t.raw tag
    | none => t.raw
  match Tensor.ofRaw raw with
  | .ok result => result
  | .error message => panic! s!"Tensor.castLike: {message}"

/-- RMS normalization with Float32 accumulation and explicit restoration of the input dtype. -/
def rmsNormPreserving {σ : StaticSpec} {w : Shape}
    (x : Tensor σ) (weight : Tensor (σ.withShape w)) (eps : Float := 1e-6) : Tensor σ :=
  let xf := RawTensor.toFloat32 σ.shape x.raw
  let variance := RawTensor.mean σ.shape (RawTensor.mul σ.shape xf xf)
    (some #[(σ.shape.size - 1).toUInt64]) true
  let reduced := reduceShape σ.shape (σ.shape.size - 1) true
  let normalized : STensor σ.shape := .assumeSpec
    (RawTensor.mul σ.shape xf (RawTensor.rsqrt reduced (RawTensor.addScalar reduced variance eps)))
  -- HF checkpoints round the normalized activation before multiplying weights.
  let restored := Tensor.castLike x normalized
  x.checkedReplacement (RawTensor.mul σ.shape restored.raw weight.raw)

/-- Package a tensor's shape existentially; this does not erase its shape claim. -/
def packShape {s : Shape} {d : DType} {dev : DevicePolicy} (t : Tensor ⟨s, d, dev⟩) :
    Sigma (fun shape => Tensor ⟨shape, d, dev⟩) := ⟨s, t⟩

/-- Concatenate tensors with runtime-varying shapes, checking the requested result.
Inputs retain their individual shape indices rather than pretending to be scalars. -/
def catManyChecked {d : DType} {dev : DevicePolicy}
    (tensors : Array (Sigma (fun shape => Tensor ⟨shape, d, dev⟩)))
    (dim : Int) (shape : Shape) : Tensor ⟨shape, d, dev⟩ :=
  match Tensor.ofRaw (RawTensor.catMany (tensors.map fun t => t.2.raw) dim) with
  | .ok result => result
  | .error message => panic! s!"Tensor.catManyChecked: {message}"

def arange (stop : UInt64) : DTensor #[stop] .Int64 :=
  .assumeSpec (RawTensor.arange 0 stop 1)

def sin {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) : Tensor σ :=
  .assumeSpec (RawTensor.sin σ.shape t.raw)

def cos {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) : Tensor σ :=
  .assumeSpec (RawTensor.cos σ.shape t.raw)

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
  match Tensor.ofRaw (RawTensor.mean σ.shape t.raw (some #[dim.toUInt64]) keepdim) with
  | .ok result => result
  | .error message => panic! s!"Tensor.meanDim: {message}"
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
  .assumeSpec (RawTensor.splitHeads batch seq heads dim t.raw)

def joinHeads {batch heads seq dim : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩) : Tensor ⟨#[batch, seq, heads, dim], d, dev⟩ :=
  .assumeSpec (RawTensor.joinHeads batch heads seq dim t.raw)

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

/-- Experimental native FlashAttention dispatch, with a metadata-checked result. -/
def flashAttention {batch heads kvHeads qSeq kvSeq dim : UInt64} {d : DType} {dev : DevicePolicy}
    (q : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, kvSeq, dim], d, dev⟩) (gqa : Bool := false) :=
  q.checkedReplacement (RawTensor.flashAttention batch heads kvHeads qSeq kvSeq dim
    q.raw k.raw v.raw none 0.0 false none gqa)

def sliceScatter {s sourceShape : Shape} {d : DType} {dev : DevicePolicy}
    (input : Tensor ⟨s, d, dev⟩) (dim start : UInt64)
    (source : Tensor ⟨sourceShape, d, dev⟩) : Tensor ⟨s, d, dev⟩ :=
  input.checkedReplacement (RawTensor.sliceScatter s sourceShape input.raw dim start source.raw)

def applyRotary {batch seq heads dim : UInt64} {d dc : DType} {dev : DevicePolicy}
    (input : Tensor ⟨#[batch, seq, heads, dim], d, dev⟩)
    (cos sin : Tensor ⟨#[seq, dim / 2], dc, dev⟩) :=
  input.checkedReplacement (RawTensor.applyRotary batch seq heads dim input.raw cos.raw sin.raw)

def rotaryFreqs (seq dim : UInt64) (base : Float := 10000.0) (device : Device := .CPU) :
    DTensor #[seq, dim / 2] .Float32 × DTensor #[seq, dim / 2] .Float32 :=
  let (cos, sin) := RawTensor.rotaryFreqs seq dim base device
  match Tensor.ofRaw cos, Tensor.ofRaw sin with
  | .ok c, .ok s => (c, s)
  | .error message, _ | _, .error message => panic! s!"Tensor.rotaryFreqs: {message}"

def hannWindow (n : UInt64) : DTensor #[n] .Float32 :=
  match Tensor.ofRaw (RawTensor.hannWindow n) with
  | .ok t => t
  | .error message => panic! s!"Tensor.hannWindow: {message}"

/-- The native STFT frame count is returned as a checked existential shape. -/
def stft {n winLength : UInt64} (input : DTensor #[n] .Float32) (nFft hopLength : UInt64)
    (window : DTensor #[winLength] .Float32) (center : Bool := true) (normalized : Bool := false) :
    IO (Sigma fun shape : Shape => DTensor shape .Float32) := do
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

end torch.Tensor
