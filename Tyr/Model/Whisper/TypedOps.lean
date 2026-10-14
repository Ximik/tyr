import Tyr.Model.Qwen35.TypedOps
import Tyr.Model.Utils
import Tyr.Module.LayerNorm

namespace torch.whisper

def initWeightDynamic (s : Shape) (fanIn : UInt64) : IO (Tensor { shape := s }) :=
  Model.initWeight s fanIn (.Unknown "dynamic")
def initBiasDynamic (s : Shape) : Tensor { shape := s } := Model.initBias s (.Unknown "dynamic")
def zerosDynamic (s : Shape) (grad : Bool := false) (device : Device := .CPU) : Tensor { shape := s } :=
  (Tensor.zeros s grad device).forgetDType

def initLayerNorm (dim : UInt64) (eps : Float := 1e-5) : LayerNorm dim (.Unknown "dynamic") :=
  { weight := (Tensor.ones #[dim] true).forgetDType,
    bias := (Tensor.zeros #[dim] true).forgetDType, eps := ⟨eps⟩ }

def sliceTyped {s : Shape} {d : DType} (x : Tensor { shape := s, dtype := d }) (dim : Nat) (start len : UInt64) : Tensor { shape := (s.set! dim len), dtype := d } :=
  x.sliceChecked dim start len (s.set! dim len)

def transposeAttention {b s h d : UInt64} (x : Tensor { shape := #[b, s, h, d] }) : Tensor { shape := #[b, h, s, d] } := x.transpose 1 2

def fromAttention {b h s d : UInt64} (x : Tensor { shape := #[b, h, s, d] }) : Tensor { shape := #[b, s, h, d] } := x.transpose 1 2

end torch.whisper
