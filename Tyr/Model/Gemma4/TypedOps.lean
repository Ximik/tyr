import Tyr.Model.Qwen35.TypedOps
import Tyr.RawTensor.Gemma4

namespace torch.gemma4

variable {device : Device}

private def checked {σ : StaticSpec} (raw : RawTensor) : Tensor σ :=
  match Tensor.ofRaw raw with
  | .ok result => result
  | .error message => panic! s!"Gemma4 tensor boundary: {message}"

def sliceTyped {s : Shape} {d : DType} (x : Tensor { device := device, shape := s, dtype := d })
    (dim : Nat) (start len : UInt64) : Tensor { device := device, shape := (s.set! dim len), dtype := d } :=
  x.sliceChecked dim start len (s.set! dim len)

def transposeAttention {b s h d : UInt64} {dt : DType}
    (x : Tensor { device := device, shape := #[b, s, h, d], dtype := dt }) : Tensor { device := device, shape := #[b, h, s, d], dtype := dt } := x.transpose 1 2

def softcap {σ : StaticSpec} (x : Tensor σ) (cap : Float) : Tensor σ :=
  x.checkedReplacement (RawTensor.gemma4Softcap σ.shape x.raw cap)

def routedTextExpertsForward {tokens numExperts topK inter hidden : UInt64}
    (x : Tensor { device := device, shape := #[tokens, hidden] }) (gateUp : Tensor { device := device, shape := #[numExperts, 2 * inter, hidden] })
    (down : Tensor { device := device, shape := #[numExperts, hidden, inter] }) (weights : Tensor { device := device, shape := #[tokens, topK] })
    (indices : Tensor { device := device, shape := #[tokens, topK], dtype := .Int64 }) : Tensor { device := device, shape := #[tokens, hidden] } :=
  x.checkedReplacement (RawTensor.gemma4TextExperts tokens numExperts topK inter hidden
    x.raw gateUp.raw down.raw weights.raw indices.raw)

def maskedAttention {b h kh qs ks d : UInt64} {dm : DType}
    (q : Tensor { device := device, shape := #[b, h, qs, d] }) (k v : Tensor { device := device, shape := #[b, kh, ks, d] })
    (mask : Tensor { device := device, shape := #[b, qs, ks], dtype := dm }) (dropout : Float := 0.0) (gqa : Bool := false) :
    Tensor { device := device, shape := #[b, h, qs, d] } :=
  checked (RawTensor.gemma4MaskedAttention b h kh qs ks d q.raw k.raw v.raw mask.raw dropout gqa)

def zerosDynamicOn {s : Shape} (device : Device) : Tensor { device := device, shape := s } :=
  (Tensor.zeros s false device).forgetDType

def onesFloatOn {s : Shape} (device : Device) : Tensor { device := device, shape := s, dtype := .Float32 } :=
  (Tensor.ones s).to device

def zerosFloatOn {s : Shape} (device : Device) : Tensor { device := device, shape := s, dtype := .Float32 } :=
  Tensor.zeros s false device

def avgPool2d {b c h w : UInt64} (x : Tensor { device := device, shape := #[b, c, h, w] }) (pool : UInt64) :
    Tensor { device := device, shape := #[b, c, h / pool, w / pool] } :=
  checked (RawTensor.gemma4AvgPool2d x.shape x.raw #[pool, pool] #[pool, pool] #[0, 0])

end torch.gemma4
