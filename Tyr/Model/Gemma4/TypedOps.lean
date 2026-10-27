import Tyr.Model.Qwen35.TypedOps
import Tyr.RawTensor.Gemma4

namespace torch.gemma4

variable {device : Device} {dtype : DType}

private def checked {σ : StaticSpec} (raw : RawTensor) : Tensor σ :=
  match Tensor.ofRaw raw with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"Gemma4 tensor boundary: {message}"

def packExactTensor {σ : StaticSpec} (tensor : Tensor σ) : SomeDTypeTensor σ.shape σ.device :=
  match tensor.reifyDType with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"Gemma4 exact tensor: {message}"

def promotedSub {shape : Shape} {leftDType rightDType : DType}
    (left : Tensor ⟨shape, leftDType, device⟩)
    (right : Tensor ⟨shape, rightDType, device⟩) : SomeDTypeTensor shape device :=
  match Tensor.Promoting.sub? left right with
  | .ok result => packExactTensor result.tensor
  | .error message => nativeInvariantFailure s!"Gemma4 mixed subtraction: {message}"

def sliceTyped {s : Shape} {d : DType} (x : Tensor { device := device, shape := s, dtype := d })
    (dim : Nat) (start len : UInt64) : Tensor { device := device, shape := (s.set! dim len), dtype := d } :=
  x.sliceChecked dim start len (s.set! dim len)

def transposeAttention {b s h d : UInt64} {dt : DType}
    (x : Tensor { device := device, shape := #[b, s, h, d], dtype := dt }) : Tensor { device := device, shape := #[b, h, s, d], dtype := dt } := x.transpose 1 2

def softcap {σ : StaticSpec} (x : Tensor σ) (cap : Float) : Tensor σ :=
  x.checkedReplacement (RawTensor.gemma4Softcap x.raw cap)

def routedTextExpertsForward {tokens numExperts topK inter hidden : UInt64} {routingDType : DType}
    (x : Tensor { device := device, shape := #[tokens, hidden], dtype := dtype }) (gateUp : Tensor { device := device, shape := #[numExperts, 2 * inter, hidden], dtype := dtype })
    (down : Tensor { device := device, shape := #[numExperts, hidden, inter], dtype := dtype }) (weights : Tensor { device := device, shape := #[tokens, topK], dtype := routingDType })
    (indices : Tensor { device := device, shape := #[tokens, topK], dtype := .Int64 }) : Tensor { device := device, shape := #[tokens, hidden], dtype := dtype } :=
  x.checkedReplacement (RawTensor.gemma4TextExperts tokens numExperts topK inter hidden
    x.raw gateUp.raw down.raw weights.raw indices.raw)

def maskedAttention? {b h kh qs ks d : UInt64} {dm : DType}
    (q : Tensor { device := device, shape := #[b, h, qs, d], dtype := dtype }) (k v : Tensor { device := device, shape := #[b, kh, ks, d], dtype := dtype })
    (mask : Tensor { device := device, shape := #[b, qs, ks], dtype := dm }) (gqa : Bool := false) :
    Except String (Tensor { device := device, shape := #[b, h, qs, d], dtype := dtype }) := do
  q.checkedReplacement? (← RawTensor.gemma4MaskedAttention q.raw k.raw v.raw mask.raw gqa)

/-- Deterministic inference attention; dropout is available only through IO. -/
def maskedAttention {b h kh qs ks d : UInt64} {dm : DType}
    (q : Tensor { device := device, shape := #[b, h, qs, d], dtype := dtype }) (k v : Tensor { device := device, shape := #[b, kh, ks, d], dtype := dtype })
    (mask : Tensor { device := device, shape := #[b, qs, ks], dtype := dm }) (gqa : Bool := false) :
    Tensor { device := device, shape := #[b, h, qs, d], dtype := dtype } :=
  match maskedAttention? q k v mask gqa with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"Gemma4.maskedAttention: {message}"

def maskedAttentionIO {b h kh qs ks d : UInt64} {dm : DType}
    (q : Tensor { device := device, shape := #[b, h, qs, d], dtype := dtype }) (k v : Tensor { device := device, shape := #[b, kh, ks, d], dtype := dtype })
    (mask : Tensor { device := device, shape := #[b, qs, ks], dtype := dm })
    (dropout : Float) (training : Bool) (gqa : Bool := false) :
    IO (Tensor { device := device, shape := #[b, h, qs, d], dtype := dtype }) := do
  let effective ← IO.ofExcept (Tensor.attentionDropout? dropout training)
  let raw ← RawTensor.gemma4MaskedAttentionIO q.raw k.raw v.raw mask.raw
    effective gqa
  IO.ofExcept (q.checkedReplacement? raw)

def zerosDynamicOn {s : Shape} (device : Device) (dtype : DType := .Float32)
    [DType.ConstantSupported dtype device] : Tensor { device := device, shape := s, dtype := dtype } :=
  Tensor.zerosFor { shape := s, dtype, device }

def onesFloatOn {s : Shape} (device : Device) : Tensor { device := device, shape := s, dtype := .Float32 } :=
  (Tensor.ones s).to device

def zerosFloatOn {s : Shape} (device : Device) : Tensor { device := device, shape := s, dtype := .Float32 } :=
  Tensor.zeros s false device

def avgPool2d {b c h w : UInt64} (x : Tensor { device := device, shape := #[b, c, h, w], dtype := dtype }) (pool : UInt64) :
    Tensor { device := device, shape := #[b, c, h / pool, w / pool], dtype := dtype } :=
  checked (RawTensor.gemma4AvgPool2d x.raw #[pool, pool] #[pool, pool] #[0, 0])

end torch.gemma4
