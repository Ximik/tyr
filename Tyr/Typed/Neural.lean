import Tyr.Typed.Ops
import Tyr.RawTensor.Neural
import Tyr.SafeTensors.File

namespace torch.Tensor

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
  let raw ← RawTensor.copyInto dst.raw src.raw
  pure (Tensor.ofRaw! raw)

def castDType! {σ : StaticSpec} (t : Tensor σ) (dtype : DType) : Tensor (σ.withDType dtype) :=
  Tensor.ofRaw! (RawTensor.castDType t.raw dtype)

/-- Cast to `dtype`. A conversion the backend does not implement aborts; no value
can make a supported conversion fail. -/
def castDType {σ : StaticSpec} (t : Tensor σ) (dtype : DType) : Tensor (σ.withDType dtype) :=
  castDType! t dtype

def castLike! {σ τ : StaticSpec} (_reference : Tensor τ) (t : Tensor σ) :
    Tensor (σ.withDType τ.dtype) :=
  castDType! t τ.dtype

/-- Cast `t` to the reference's dtype, preserving its shape and placement. -/
def castLike {σ τ : StaticSpec} (reference : Tensor τ) (t : Tensor σ) :
    Tensor (σ.withDType τ.dtype) :=
  castLike! reference t

/-- RMS normalization accumulated in Float32, with the output dtype indexed. -/
def rmsNorm {σ : StaticSpec} (x : Tensor σ) (eps : Float := 1e-6) : Tensor (σ.withDType .Float32) :=
  let xf := RawTensor.toFloat32 x.raw
  let variance := RawTensor.mean! (RawTensor.mul! xf xf)
    (some #[(σ.shape.size - 1)]) true
  (Tensor.ofRaw! (RawTensor.mul! xf (RawTensor.rsqrt! (RawTensor.addScalar! variance eps))))

/-- RMS normalization with Float32 accumulation and explicit restoration of the input dtype. -/
def rmsNormPreserving {σ : StaticSpec} {w : Shape}
    (x : Tensor σ) (weight : Tensor (σ.withShape w)) (eps : Float := 1e-6) : Tensor σ :=
  -- HF checkpoints round the normalized activation before multiplying weights.
  let restored := Tensor.castLike x (rmsNorm x eps)
  Tensor.ofRaw! (RawTensor.mul! restored.raw weight.raw)

/-- Package a tensor's shape existentially; this does not erase its shape claim. -/
def packShape {s : Shape} {d : DType} {dev : Device} (t : Tensor ⟨s, d, dev⟩) :
    Sigma (fun shape => Tensor ⟨shape, d, dev⟩) := ⟨s, t⟩

/-- Concatenate tensors whose shapes are known only at runtime. An empty array,
an axis out of range, or sizes that differ off the axis are returned as errors;
the result is admitted at the requested `shape`. -/
def catMany? {d : DType} {dev : Device}
    (tensors : Array (Sigma (fun shape => Tensor ⟨shape, d, dev⟩)))
    (dim : Nat) (shape : Shape) : Except String (Tensor ⟨shape, d, dev⟩) := do
  pure (Tensor.ofRaw! (← RawTensor.catMany? (tensors.map fun t => t.2.raw) dim))

/-- `catMany?` without its checks; invalid inputs abort natively. -/
def catMany! {d : DType} {dev : Device}
    (tensors : Array (Sigma (fun shape => Tensor ⟨shape, d, dev⟩)))
    (dim : Nat) (shape : Shape) : Tensor ⟨shape, d, dev⟩ :=
  Tensor.ofRaw! (RawTensor.catMany! (tensors.map fun t => t.2.raw) dim)

def arange (stop : Nat) : Tensor { shape := #[stop], dtype := .Int64, device := .CPU } :=
  Tensor.ofRaw! (RawTensor.arange 0 stop 1)

def sin! {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  Tensor.ofRaw! (RawTensor.sin! t.raw)

def sin {σ : StaticSpec} (t : Tensor σ)
    (_domain : σ.dtype.isInexact = true := by first | rfl | exact DType.isInexact_of_realFloating | decide) : Tensor σ :=
  sin! t

def cos! {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  Tensor.ofRaw! (RawTensor.cos! t.raw)

def cos {σ : StaticSpec} (t : Tensor σ)
    (_domain : σ.dtype.isInexact = true := by first | rfl | exact DType.isInexact_of_realFloating | decide) : Tensor σ :=
  cos! t

def elu! {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  Tensor.ofRaw! (RawTensor.elu! t.raw)

def elu {σ : StaticSpec} (t : Tensor σ)
    (_domain : σ.dtype.isRealFloating = true := by first | rfl | exact DType.isRealFloating_of_realFloating | decide) : Tensor σ :=
  elu! t


/-- Mean for floating or complex checkpoint tensors; invalid axes, integral
inputs and missing kernels are recoverable errors. -/
def meanDim! {σ : StaticSpec} (t : Tensor σ) (dim : Nat) (keepdim : Bool := false) :
    Tensor (σ.withShape (reduceShape σ.shape dim keepdim)) :=
  Tensor.ofRaw! (RawTensor.mean! t.raw (some #[dim]) keepdim)

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

def conv2dBias! {batch cin cout height width kh kw : Nat} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, cin, height, width], d, dev⟩)
    (weight : Tensor ⟨#[cout, cin, kh, kw], d, dev⟩)
    (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : Nat × Nat := (1, 1)) (padding : Nat × Nat := (0, 0))
    (dilation : Nat × Nat := (1, 1)) :
    Tensor ⟨#[batch, cout,
      convExtent height kh stride.1 padding.1 dilation.1,
      convExtent width kw stride.2 padding.2 dilation.2], d, dev⟩ :=
  Tensor.ofRaw! (RawTensor.conv2dBias! input.raw weight.raw bias.raw
    #[stride.1, stride.2] #[padding.1, padding.2] #[dilation.1, dilation.2])

/-- Biased two-dimensional convolution; parameters are (height, width) pairs. -/
def conv2dBias {batch cin cout height width kh kw : Nat} {d : DType} {dev : Device}
    [DType.FloatingOrComplex d]
    (input : Tensor ⟨#[batch, cin, height, width], d, dev⟩)
    (weight : Tensor ⟨#[cout, cin, kh, kw], d, dev⟩)
    (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : Nat × Nat := (1, 1)) (padding : Nat × Nat := (0, 0))
    (dilation : Nat × Nat := (1, 1))
    (_height : ConvValid height kh stride.1 padding.1 dilation.1 := by conv_valid)
    (_width : ConvValid width kw stride.2 padding.2 dilation.2 := by conv_valid) :
    Tensor ⟨#[batch, cout,
      convExtent height kh stride.1 padding.1 dilation.1,
      convExtent width kw stride.2 padding.2 dilation.2], d, dev⟩ :=
  conv2dBias! input weight bias stride padding dilation

def splitHeads {batch seq heads dim : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[batch, seq, heads, dim], d, dev⟩) : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  Tensor.ofRaw! (RawTensor.splitHeads t.raw)

def joinHeads {batch heads seq dim : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩) : Tensor ⟨#[batch, seq, heads, dim], d, dev⟩ :=
  Tensor.ofRaw! (RawTensor.joinHeads t.raw)

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

/-- Query heads either equal the K/V heads, or form whole groups of them under GQA. -/
def AttentionHeads (heads kvHeads : Nat) (gqa : Bool) : Prop :=
  kvHeads = heads ∨ (gqa = true ∧ kvHeads ∣ heads)

instance (heads kvHeads : Nat) (gqa : Bool) : Decidable (AttentionHeads heads kvHeads gqa) := by
  unfold AttentionHeads; infer_instance

theorem AttentionHeads.ofDvd {heads kvHeads : Nat} (h : kvHeads ∣ heads) :
    AttentionHeads heads kvHeads true :=
  Or.inr ⟨rfl, h⟩

/-- Discharges `AttentionHeads` for equal heads, literal heads, or a divisibility fact in scope. -/
macro "attention_heads" : tactic => `(tactic| first
  | exact Or.inl rfl
  | exact of_decide_eq_true rfl
  | exact Or.inr ⟨rfl, by assumption⟩)

def attention! {batch heads seq dim : Nat} {d : DType} {dev : Device}
    (q k v : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (causal : Bool := true) : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  Tensor.ofRaw! (RawTensor.attention! q.raw k.raw v.raw causal)

/-- Zero-dropout attention. -/
def attention {batch heads seq dim : Nat} {d : DType} {dev : Device} [DType.RealFloating d]
    (q k v : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (causal : Bool := true) : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  attention! q k v causal

/-- Attention with explicit training-mode dropout, sequenced through native IO. -/
def attentionIO {batch heads seq dim : Nat} {d : DType} {dev : Device}
    (q k v : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (dropout : Float) (training : Bool) (causal : Bool := true) : IO (Tensor ⟨#[batch, heads, seq, dim], d, dev⟩) := do
  let effective ← attentionCheckIO "attentionIO" (attentionDropout? dropout training)
  let raw ← RawTensor.attentionIO
    q.raw k.raw v.raw effective causal
  pure (Tensor.ofRaw! raw)

def attentionGQA! {batch heads kvHeads seq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (causal : Bool := true) (gqa : Bool := false) : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  Tensor.ofRaw! (RawTensor.attentionGQA! q.raw k.raw v.raw causal gqa)

/-- Zero-dropout grouped-query attention. -/
def attentionGQA {batch heads kvHeads seq dim : Nat} {d : DType} {dev : Device} [DType.RealFloating d]
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (causal : Bool := true) (gqa : Bool := false)
    (_heads : AttentionHeads heads kvHeads gqa := by attention_heads) :
    Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  attentionGQA! q k v causal gqa

/-- Attention with explicit training-mode dropout, sequenced through native IO. -/
def attentionGQAIO {batch heads kvHeads seq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (dropout : Float) (training : Bool) (causal : Bool := true) (gqa : Bool := false) : IO (Tensor ⟨#[batch, heads, seq, dim], d, dev⟩) := do
  let effective ← attentionCheckIO "attentionGQAIO" (attentionDropout? dropout training)
  let raw ← RawTensor.attentionGQAIO
    q.raw k.raw v.raw effective causal gqa
  pure (Tensor.ofRaw! raw)

def attentionGQAMask! {batch heads kvHeads seq dim : Nat} {d dm : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (mask : Tensor ⟨#[batch, seq], dm, dev⟩)
    (causal : Bool := true) (gqa : Bool := false) : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  Tensor.ofRaw! (RawTensor.attentionGQAMask! q.raw k.raw v.raw mask.raw causal gqa)

/-- Zero-dropout grouped-query attention with a key padding mask. -/
def attentionGQAMask {batch heads kvHeads seq dim : Nat} {d dm : DType} {dev : Device} [DType.RealFloating d]
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (mask : Tensor ⟨#[batch, seq], dm, dev⟩)
    (causal : Bool := true) (gqa : Bool := false)
    (_heads : AttentionHeads heads kvHeads gqa := by attention_heads) :
    Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  attentionGQAMask! q k v mask causal gqa

/-- Attention with explicit training-mode dropout, sequenced through native IO. -/
def attentionGQAMaskIO {batch heads kvHeads seq dim : Nat} {d dm : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (mask : Tensor ⟨#[batch, seq], dm, dev⟩)
    (dropout : Float) (training : Bool) (causal : Bool := true) (gqa : Bool := false) : IO (Tensor ⟨#[batch, heads, seq, dim], d, dev⟩) := do
  let effective ← attentionCheckIO "attentionGQAMaskIO" (attentionDropout? dropout training)
  let raw ← RawTensor.attentionGQAMaskIO
    q.raw k.raw v.raw mask.raw effective causal gqa
  pure (Tensor.ofRaw! raw)

/-- Gather has the index tensor's shape, and the source tensor's dtype. -/
def gather {s indicesShape : Shape} {d : DType} {dev : Device}
    (input : Tensor ⟨s, d, dev⟩) (dim : Nat)
    (indices : Tensor ⟨indicesShape, .Int64, dev⟩)
    (_h : TensorSpec.gatherCompatible s indicesShape dim = true := by
      first | exact of_decide_eq_true rfl | (simp <;> fail "incompatible gather shapes")) :
    Tensor ⟨indicesShape, d, dev⟩ :=
  (Tensor.ofRaw! (RawTensor.gather input.raw dim indices.raw))

/-- Masked selection returns an existential vector length, checked against the native result. -/
def maskedSelect {s : Shape} {d : DType} {dev : Device}
    (input : Tensor ⟨s, d, dev⟩) (mask : Tensor ⟨s, .Bool, dev⟩) :
    Sigma (fun n => Tensor ⟨#[n], d, dev⟩) :=
  let raw := RawTensor.maskedSelect input.raw mask.raw
  let n := raw.runtimeShape.getD 0 0
  let result := Tensor.ofRaw! (σ := ⟨#[n], d, dev⟩) raw
  ⟨n, result⟩

def maskedScatter {s sourceShape : Shape} {d : DType} {dev : Device}
    (input : Tensor ⟨s, d, dev⟩) (mask : Tensor ⟨s, .Bool, dev⟩)
    (source : Tensor ⟨sourceShape, d, dev⟩) : Tensor ⟨s, d, dev⟩ :=
  Tensor.ofRaw! (RawTensor.maskedScatter input.raw mask.raw source.raw)

def attentionGQAQKV! {batch heads kvHeads qSeq kvSeq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, kvSeq, dim], d, dev⟩)
    (causal : Bool := false) (gqa : Bool := false) : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩ :=
  Tensor.ofRaw! (RawTensor.attentionGQAQKV! q.raw k.raw v.raw causal gqa)

/-- Zero-dropout grouped-query attention with separate query and key/value lengths. -/
def attentionGQAQKV {batch heads kvHeads qSeq kvSeq dim : Nat} {d : DType} {dev : Device} [DType.RealFloating d]
    (q : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, kvSeq, dim], d, dev⟩)
    (causal : Bool := false) (gqa : Bool := false)
    (_heads : AttentionHeads heads kvHeads gqa := by attention_heads) :
    Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩ :=
  attentionGQAQKV! q k v causal gqa

/-- Attention with explicit training-mode dropout, sequenced through native IO. -/
def attentionGQAQKVIO {batch heads kvHeads qSeq kvSeq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, kvSeq, dim], d, dev⟩)
    (dropout : Float) (training : Bool) (causal : Bool := false) (gqa : Bool := false) : IO (Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩) := do
  let effective ← attentionCheckIO "attentionGQAQKVIO" (attentionDropout? dropout training)
  let raw ← RawTensor.attentionGQAQKVIO
    q.raw k.raw v.raw effective causal gqa
  pure (Tensor.ofRaw! raw)

def flashAttention! {batch heads kvHeads qSeq kvSeq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, kvSeq, dim], d, dev⟩) (gqa : Bool := false)
    (mask : Option (SomeDTypeTensor #[batch, kvSeq] dev) := none)
    (causal : Bool := false) (scale : Option Float := none) :
    Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩ :=
  Tensor.ofRaw! (RawTensor.flashAttention!
    q.raw k.raw v.raw (mask.map (fun packed => packed.2.raw)) causal scale gqa)

/-- Zero-dropout FlashAttention. -/
def flashAttention {batch heads kvHeads qSeq kvSeq dim : Nat} {d : DType} {dev : Device} [DType.RealFloating d]
    (q : Tensor ⟨#[batch, heads, qSeq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, kvSeq, dim], d, dev⟩) (gqa : Bool := false)
    (mask : Option (SomeDTypeTensor #[batch, kvSeq] dev) := none)
    (causal : Bool := false) (scale : Option Float := none)
    (_heads : AttentionHeads heads kvHeads gqa := by attention_heads) :
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
  let raw ← RawTensor.flashAttentionIO
    q.raw k.raw v.raw (mask.map (fun packed => packed.2.raw)) effective causal scale gqa
  pure (Tensor.ofRaw! raw)

def sliceScatter {s sourceShape : Shape} {d : DType} {dev : Device}
    (input : Tensor ⟨s, d, dev⟩) (dim start : Nat)
    (source : Tensor ⟨sourceShape, d, dev⟩)
    (_h : TensorSpec.sliceScatterCompatible s sourceShape dim = true := by
      first | exact of_decide_eq_true rfl | (simp <;> fail "incompatible slice scatter shapes"))
    (_bound : start + sourceShape.getD dim 0 ≤ s.getD dim 0 := by
      first | exact of_decide_eq_true rfl | omega | (simp <;> omega)) :
    Tensor ⟨s, d, dev⟩ :=
  Tensor.ofRaw! (RawTensor.sliceScatter input.raw dim start source.raw)

/-- Rotate-half RoPE: the two halves of the last axis are rotated by `cos`/`sin`. -/
def applyRotary {batch seq heads half : Nat} {d dc : DType} {dev : Device}
    (input : Tensor ⟨#[batch, seq, heads, 2 * half], d, dev⟩)
    (cos sin : Tensor ⟨#[seq, half], dc, dev⟩) : Tensor ⟨#[batch, seq, heads, 2 * half], d, dev⟩ :=
  Tensor.ofRaw! (RawTensor.applyRotary input.raw cos.raw sin.raw)

/-- Rotary cos/sin tables `[seq, half]` for a rotated width of `2 * half`. -/
def rotaryFreqs (seq half : Nat) (base : Float := 10000.0) (device : Device := .CPU) :
    Tensor { shape := #[seq, half], dtype := .Float32, device } × Tensor { shape := #[seq, half], dtype := .Float32, device } :=
  let (cos, sin) := RawTensor.rotaryFreqs seq (2 * half) base device
  (Tensor.ofRaw! cos, Tensor.ofRaw! sin)

def hannWindow (n : Nat) : Tensor { shape := #[n], dtype := .Float32, device := .CPU } :=
  (Tensor.ofRaw! (RawTensor.hannWindow n))

/-- Frames of `stft` over `n` samples; `center` reflect-pads `nFft / 2` on each side. -/
def stftFrames (n nFft hopLength : Nat) (center : Bool) : Nat :=
  ((if center then n + 2 * (nFft / 2) else n) - nFft) / hopLength + 1

/-- Parameters `torch.stft` accepts for `n` samples: a positive hop, a positive
window no longer than the FFT, and enough samples for the reflect padding when
`center` holds, or for one frame otherwise. -/
def StftValid (n nFft hopLength winLength : Nat) (center : Bool) : Prop :=
  0 < hopLength ∧ 0 < winLength ∧ winLength ≤ nFft ∧ (if center then nFft / 2 < n else nFft ≤ n)

instance (n nFft hopLength winLength : Nat) (center : Bool) :
    Decidable (StftValid n nFft hopLength winLength center) := by
  unfold StftValid; infer_instance

def stft! {n winLength : Nat} {dev : Device} (input : Tensor { shape := #[n], dtype := .Float32, device := dev }) (nFft hopLength : Nat)
    (window : Tensor { shape := #[winLength], dtype := .Float32, device := dev }) (center : Bool := true) (normalized : Bool := false) :
    Tensor { shape := #[nFft / 2 + 1, stftFrames n nFft hopLength center, 2], dtype := .Float32, device := dev } :=
  Tensor.ofRaw! (RawTensor.stft input.raw nFft hopLength winLength window.raw center normalized)

/-- One-sided STFT with real and imaginary parts packed in the last axis. -/
def stft {n winLength : Nat} {dev : Device} (input : Tensor { shape := #[n], dtype := .Float32, device := dev }) (nFft hopLength : Nat)
    (window : Tensor { shape := #[winLength], dtype := .Float32, device := dev }) (center : Bool := true) (normalized : Bool := false)
    (_valid : StftValid n nFft hopLength winLength center := by
      first | exact of_decide_eq_true rfl | (simp only [StftValid] <;> omega)) :
    Tensor { shape := #[nFft / 2 + 1, stftFrames n nFft hopLength center, 2], dtype := .Float32, device := dev } :=
  stft! input nFft hopLength window center normalized

def log10! {σ : StaticSpec} (input : Tensor σ) : Tensor σ :=
  Tensor.ofRaw! (RawTensor.log10! input.raw)

def log10 {σ : StaticSpec} (input : Tensor σ)
    (_inexact : σ.dtype.isInexact = true := by first | rfl | exact DType.isInexact_of_realFloating | decide) : Tensor σ :=
  log10! input


def saveWav {σ : StaticSpec} (input : Tensor σ) (path : String) (sampleRate : Nat := 24000) : IO Unit :=
  RawTensor.saveWav input.raw path sampleRate

def lt! {s : Shape} {d : DType} {dev : Device} (a b : Tensor ⟨s, d, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  Tensor.ofRaw! (RawTensor.lt! a.raw b.raw)

def lt {s : Shape} {d : DType} {dev : Device} [DType.Ordered d] (a b : Tensor ⟨s, d, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  lt! a b

def ge! {s : Shape} {d : DType} {dev : Device} (a b : Tensor ⟨s, d, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  Tensor.ofRaw! (RawTensor.ge! a.raw b.raw)

def ge {s : Shape} {d : DType} {dev : Device} [DType.Ordered d] (a b : Tensor ⟨s, d, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  ge! a b
def ltScalar {s : Shape} {d : DType} {dev : Device} (input : Tensor ⟨s, d, dev⟩) (value : Float) :
    Tensor ⟨s, .Bool, dev⟩ := Tensor.ofRaw! (RawTensor.ltScalar input.raw value)
def maskedFill {s : Shape} {d : DType} {dev : Device}
    (input : Tensor ⟨s, d, dev⟩) (mask : Tensor ⟨s, .Bool, dev⟩) (value : Float) : Tensor ⟨s, d, dev⟩ :=
  Tensor.ofRaw! (RawTensor.maskedFill input.raw mask.raw value)

/-- Input channels split into `groups` equal groups, and output channels divide among them. -/
def ConvGroups (cin cinPerGroup cout groups : Nat) : Prop :=
  0 < groups ∧ cin = cinPerGroup * groups ∧ groups ∣ cout

instance (cin cinPerGroup cout groups : Nat) : Decidable (ConvGroups cin cinPerGroup cout groups) := by
  unfold ConvGroups; infer_instance

/-- Discharges `ConvGroups` for literal parameters, or from facts in scope. -/
macro "conv_groups" : tactic => `(tactic| first
  | exact of_decide_eq_true rfl
  | (refine ⟨?_, ?_, ?_⟩ <;> first | omega | assumption))

def conv1dGroups! {batch cin cout cinPerGroup length kernel : Nat} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, cin, length], d, dev⟩)
    (weight : Tensor ⟨#[cout, cinPerGroup, kernel], d, dev⟩) (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : Nat := 1) (padding : Nat := 0) (dilation : Nat := 1) (groups : Nat := 1) :
    Tensor ⟨#[batch, cout, convExtent length kernel stride padding dilation], d, dev⟩ :=
  Tensor.ofRaw! (RawTensor.conv1dGroups! input.raw weight.raw bias.raw
    stride padding dilation groups)

/-- Grouped biased one-dimensional convolution. -/
def conv1dGroups {batch cin cout cinPerGroup length kernel : Nat} {d : DType} {dev : Device}
    [DType.FloatingOrComplex d]
    (input : Tensor ⟨#[batch, cin, length], d, dev⟩)
    (weight : Tensor ⟨#[cout, cinPerGroup, kernel], d, dev⟩) (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : Nat := 1) (padding : Nat := 0) (dilation : Nat := 1) (groups : Nat := 1)
    (_valid : ConvValid length kernel stride padding dilation := by conv_valid)
    (_groups : ConvGroups cin cinPerGroup cout groups := by conv_groups) :
    Tensor ⟨#[batch, cout, convExtent length kernel stride padding dilation], d, dev⟩ :=
  conv1dGroups! input weight bias stride padding dilation groups

/-- Positive stride, dilation, kernel and input length; output padding below the
stride or dilation; and padding that leaves a nonempty output. -/
def ConvTransposeValid (length kernel stride padding outputPadding dilation : Nat) : Prop :=
  0 < stride ∧ 0 < dilation ∧ 0 < kernel ∧ 0 < length ∧
  (outputPadding < stride ∨ outputPadding < dilation) ∧
  2 * padding ≤ (length - 1) * stride + dilation * (kernel - 1) + outputPadding

instance (length kernel stride padding outputPadding dilation : Nat) :
    Decidable (ConvTransposeValid length kernel stride padding outputPadding dilation) := by
  unfold ConvTransposeValid; infer_instance

def convTranspose1d! {batch cin cout length kernel : Nat} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, cin, length], d, dev⟩)
    (weight : Tensor ⟨#[cin, cout, kernel], d, dev⟩) (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : Nat := 1) (padding : Nat := 0) (outputPadding : Nat := 0) (dilation : Nat := 1) :
    Tensor ⟨#[batch, cout, ((length - 1) * stride + dilation * (kernel - 1)
      + outputPadding + 1 - 2 * padding)], d, dev⟩ :=
  Tensor.ofRaw! (RawTensor.convTranspose1d! input.raw weight.raw bias.raw
    stride padding outputPadding dilation)

/-- Biased one-dimensional transposed convolution. -/
def convTranspose1d {batch cin cout length kernel : Nat} {d : DType} {dev : Device}
    [DType.FloatingOrComplex d]
    (input : Tensor ⟨#[batch, cin, length], d, dev⟩)
    (weight : Tensor ⟨#[cin, cout, kernel], d, dev⟩) (bias : Tensor ⟨#[cout], d, dev⟩)
    (stride : Nat := 1) (padding : Nat := 0) (outputPadding : Nat := 0) (dilation : Nat := 1)
    (_valid : ConvTransposeValid length kernel stride padding outputPadding dilation := by
      exact of_decide_eq_true rfl) :
    Tensor ⟨#[batch, cout, ((length - 1) * stride + dilation * (kernel - 1)
      + outputPadding + 1 - 2 * padding)], d, dev⟩ :=
  convTranspose1d! input weight bias stride padding outputPadding dilation

def attentionWindow! {batch heads kvHeads seq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (causal : Bool := true) (gqa : Bool := false) (window : Nat := 0) : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  Tensor.ofRaw! (RawTensor.attentionWindow! q.raw k.raw v.raw causal gqa window)

/-- Zero-dropout sliding-window attention. -/
def attentionWindow {batch heads kvHeads seq dim : Nat} {d : DType} {dev : Device} [DType.RealFloating d]
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (causal : Bool := true) (gqa : Bool := false) (window : Nat := 0)
    (_heads : AttentionHeads heads kvHeads gqa := by attention_heads) :
    Tensor ⟨#[batch, heads, seq, dim], d, dev⟩ :=
  attentionWindow! q k v causal gqa window

/-- Attention with explicit training-mode dropout, sequenced through native IO. -/
def attentionWindowIO {batch heads kvHeads seq dim : Nat} {d : DType} {dev : Device}
    (q : Tensor ⟨#[batch, heads, seq, dim], d, dev⟩)
    (k v : Tensor ⟨#[batch, kvHeads, seq, dim], d, dev⟩)
    (dropout : Float) (training : Bool) (causal : Bool := true) (gqa : Bool := false) (window : Nat := 0) : IO (Tensor ⟨#[batch, heads, seq, dim], d, dev⟩) := do
  let effective ← attentionCheckIO "attentionWindowIO" (attentionDropout? dropout training)
  let raw ← RawTensor.attentionWindowIO
    q.raw k.raw v.raw effective causal gqa window
  pure (Tensor.ofRaw! raw)

def log! {σ : StaticSpec} (input : Tensor σ) : Tensor σ :=
  Tensor.ofRaw! (RawTensor.log! input.raw)

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
  if raw.dtype != σ.dtype then
    throw (IO.userError s!"Tensor.loadTensorExact {path}: stored dtype {raw.dtype}, expected {σ.dtype}")
  if raw.device != σ.device then
    throw (IO.userError s!"Tensor.loadTensorExact {path}: stored on another device than requested")
  pure (Tensor.ofRaw! raw)

/-- Load the exact stored shape and dtype, moving only its placement to `σ.device`.
The stored dtype is checked before conversion; loading never silently casts it. -/
def loadTensorOn (σ : StaticSpec) (path : String) : IO (Tensor σ) := do
  let raw ← RawTensor.loadTensorExact σ.shape path
  if raw.dtype != σ.dtype then
    throw (IO.userError s!"Tensor.loadTensorOn {path}: stored dtype {raw.dtype}, expected {σ.dtype}")
  pure (Tensor.ofRaw! (← RawTensor.convertIO σ.shape raw σ.device none))

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
  pure (Tensor.ofRaw! raw)

def linspace (start stop : Float) (count : Nat) : Tensor { shape := #[count], dtype := .Float32, device := .CPU } :=
  Tensor.ofRaw! (RawTensor.linspace start stop count false)

/-- Reduce the last axis, preserving the actual input dtype, without precondition
checks; libtorch rejects an unordered dtype or an empty axis and the process aborts. -/
def maxLast3d! {batch seq vocab : Nat} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, seq, vocab], d, dev⟩) :
    Tensor ⟨#[batch, seq], d, dev⟩ × Tensor ⟨#[batch, seq], .Int64, dev⟩ :=
  let (values, indices) := RawTensor.maxLast3d! input.raw 2
  (Tensor.ofRaw! values, Tensor.ofRaw! indices)

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
  (Tensor.ofRaw! values, Tensor.ofRaw! indices)

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
  (Tensor.ofRaw! (RawTensor.arange start stop step))

/-- Select entry `idx` of the leading axis; a scalar has no leading axis. -/
def getOp {σ : StaticSpec} (input : Tensor σ) (idx : Nat)
    (_bound : idx < σ.shape[0]?.getD 0 := by
      first | exact of_decide_eq_true rfl | (simp <;> fail "index out of range")) :
    Tensor (σ.withShape (σ.shape.extract 1 σ.shape.size)) :=
  (Tensor.ofRaw! (RawTensor.getOp input.raw idx))

def unbind {σ : StaticSpec} (input : Tensor σ) (dim : Nat := 0)
    (h : dim < σ.shape.size := by first | exact of_decide_eq_true rfl | (simp <;> fail "axis out of range")) :
    Array (Tensor (σ.withShape (unbindShape σ.shape dim h))) :=
  (RawTensor.unbind input.raw dim).map fun raw =>
    (Tensor.ofRaw! raw)

def logspace (start stop : Float) (steps : Nat) (base : Float := 10.0) :
    Tensor { shape := #[steps], dtype := .Float32, device := .CPU } :=
  (Tensor.ofRaw! (RawTensor.logspace start stop steps base false))

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
  let result := Tensor.ofRaw! (σ := ⟨#[count], .Int64, dev⟩) raw
  pure ⟨count, result⟩

end torch.Tensor

namespace torch.SomeDTypeTensor

/-- Load the exact stored shape and placement while retaining the stored dtype
as an existential index. Loading does not cast or reshape the tensor. -/
def loadTensorExact (shape : Shape) (device : Device) (path : String) :
    IO (SomeDTypeTensor shape device) := do
  let raw ← RawTensor.loadTensorExact shape path
  if raw.device != device then
    throw (IO.userError s!"SomeDTypeTensor.loadTensorExact {path}: stored on another device than requested")
  pure (SomeDTypeTensor.ofRaw! raw)

end torch.SomeDTypeTensor
