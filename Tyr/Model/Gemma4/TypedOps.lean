import Tyr.Model.Qwen35.TypedOps
import Tyr.RawTensor.Gemma4

namespace torch.gemma4

private def checked {σ : StaticSpec} (raw : RawTensor) : Tensor σ :=
  match Tensor.ofRaw raw with
  | .ok result => result
  | .error message => panic! s!"Gemma4 tensor boundary: {message}"

def sliceTyped {s : Shape} {d : DType} (x : DTensor s d)
    (dim : Nat) (start len : UInt64) : DTensor (s.set! dim len) d :=
  x.sliceChecked dim start len (s.set! dim len)

def transposeAttention {b s h d : UInt64} {dt : DType}
    (x : DTensor #[b, s, h, d] dt) : DTensor #[b, h, s, d] dt := x.transpose 1 2

def softcap {σ : StaticSpec} (x : Tensor σ) (cap : Float) : Tensor σ :=
  x.checkedReplacement (RawTensor.gemma4Softcap σ.shape x.raw cap)

def routedTextExpertsForward {tokens numExperts topK inter hidden : UInt64}
    (x : STensor #[tokens, hidden]) (gateUp : STensor #[numExperts, 2 * inter, hidden])
    (down : STensor #[numExperts, hidden, inter]) (weights : STensor #[tokens, topK])
    (indices : DTensor #[tokens, topK] .Int64) : STensor #[tokens, hidden] :=
  x.checkedReplacement (RawTensor.gemma4TextExperts tokens numExperts topK inter hidden
    x.raw gateUp.raw down.raw weights.raw indices.raw)

def maskedAttention {b h kh qs ks d : UInt64} {dm : DType}
    (q : STensor #[b, h, qs, d]) (k v : STensor #[b, kh, ks, d])
    (mask : DTensor #[b, qs, ks] dm) (dropout : Float := 0.0) (gqa : Bool := false) :
    STensor #[b, h, qs, d] :=
  checked (RawTensor.gemma4MaskedAttention b h kh qs ks d q.raw k.raw v.raw mask.raw dropout gqa)

def zerosDynamicOn {s : Shape} (device : Device) : STensor s :=
  (Tensor.zeros s false device).forgetDType

def onesFloatOn {s : Shape} (device : Device) : DTensor s .Float32 :=
  (Tensor.ones s).to device

def zerosFloatOn {s : Shape} (device : Device) : DTensor s .Float32 :=
  Tensor.zeros s false device

def avgPool2d {b c h w : UInt64} (x : STensor #[b, c, h, w]) (pool : UInt64) :
    STensor #[b, c, h / pool, w / pool] :=
  checked (RawTensor.gemma4AvgPool2d x.shape x.raw #[pool, pool] #[pool, pool] #[0, 0])

end torch.gemma4
