import Tyr.Model.Qwen35.TypedOps
import Tyr.RawTensor.Gemma4

namespace torch.gemma4

variable {device : Device} {dtype : DType}

def transposeAttention {b s h d : Nat} {dt : DType}
    (x : Tensor { device := device, shape := #[b, s, h, d], dtype := dt }) : Tensor { device := device, shape := #[b, h, s, d], dtype := dt } := x.transpose 1 2

def softcap {σ : StaticSpec} (x : Tensor σ) (cap : Float) : Tensor σ :=
  Tensor.ofRaw! (RawTensor.gemma4Softcap x.raw cap)

def routedTextExpertsForward {tokens numExperts topK inter hidden : Nat} {routingDType : DType}
    (x : Tensor { device := device, shape := #[tokens, hidden], dtype := dtype }) (gateUp : Tensor { device := device, shape := #[numExperts, 2 * inter, hidden], dtype := dtype })
    (down : Tensor { device := device, shape := #[numExperts, hidden, inter], dtype := dtype }) (weights : Tensor { device := device, shape := #[tokens, topK], dtype := routingDType })
    (indices : Tensor { device := device, shape := #[tokens, topK], dtype := .Int64 }) : Tensor { device := device, shape := #[tokens, hidden], dtype := dtype } :=
  Tensor.ofRaw! (RawTensor.gemma4TextExperts tokens numExperts topK inter hidden
    x.raw gateUp.raw down.raw weights.raw indices.raw)

def maskedAttention! {b h kh qs ks d : Nat} {dm : DType}
    (q : Tensor { device := device, shape := #[b, h, qs, d], dtype := dtype }) (k v : Tensor { device := device, shape := #[b, kh, ks, d], dtype := dtype })
    (mask : Tensor { device := device, shape := #[b, qs, ks], dtype := dm }) (gqa : Bool := false) :
    Tensor { device := device, shape := #[b, h, qs, d], dtype := dtype } :=
  Tensor.ofRaw! (RawTensor.gemma4MaskedAttention! q.raw k.raw v.raw mask.raw gqa)

/-- Deterministic inference attention; dropout is available only through IO. -/
def maskedAttention {b h kh qs ks d : Nat} {dm : DType} [DType.RealFloating dtype]
    (q : Tensor { device := device, shape := #[b, h, qs, d], dtype := dtype }) (k v : Tensor { device := device, shape := #[b, kh, ks, d], dtype := dtype })
    (mask : Tensor { device := device, shape := #[b, qs, ks], dtype := dm }) (gqa : Bool := false)
    (_heads : Tensor.AttentionHeads h kh gqa := by attention_heads) :
    Tensor { device := device, shape := #[b, h, qs, d], dtype := dtype } :=
  maskedAttention! q k v mask gqa

def maskedAttentionIO {b h kh qs ks d : Nat} {dm : DType}
    (q : Tensor { device := device, shape := #[b, h, qs, d], dtype := dtype }) (k v : Tensor { device := device, shape := #[b, kh, ks, d], dtype := dtype })
    (mask : Tensor { device := device, shape := #[b, qs, ks], dtype := dm })
    (dropout : Float) (training : Bool) (gqa : Bool := false) :
    IO (Tensor { device := device, shape := #[b, h, qs, d], dtype := dtype }) := do
  let effective ← IO.ofExcept (Tensor.attentionDropout? dropout training)
  let raw ← RawTensor.gemma4MaskedAttentionIO q.raw k.raw v.raw mask.raw
    effective gqa
  pure (Tensor.ofRaw! raw)

def zerosDynamicOn {s : Shape} (device : Device) (dtype : DType := .Float32)
    [DType.ConstantSupported dtype device] : Tensor { device := device, shape := s, dtype := dtype } :=
  Tensor.zerosFor { shape := s, dtype, device }

def onesFloatOn {s : Shape} (device : Device) : Tensor { device := device, shape := s, dtype := .Float32 } :=
  (Tensor.ones s).to device

def zerosFloatOn {s : Shape} (device : Device) : Tensor { device := device, shape := s, dtype := .Float32 } :=
  Tensor.zeros s false device

def avgPool2d {b c h w : Nat} (x : Tensor { device := device, shape := #[b, c, h, w], dtype := dtype }) (pool : Nat) :
    Tensor { device := device, shape := #[b, c, h / pool, w / pool], dtype := dtype } :=
  Tensor.ofRaw! (RawTensor.gemma4AvgPool2d x.raw #[pool, pool] #[pool, pool] #[0, 0])

end torch.gemma4
