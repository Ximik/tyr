import Tyr.Model.Qwen35.TypedOps
import Tyr.Model.Utils
import Tyr.Module.LayerNorm

namespace torch.whisper

variable {device : Device}
def initWeightDynamic (s : Shape) (fanIn : UInt64) (device : Device := .CPU) : IO (Tensor { device := device, shape := s }) :=
  Model.initWeight s fanIn (.Unknown "dynamic") device
def initBiasDynamic (s : Shape) (device : Device := .CPU) : IO (Tensor { device := device, shape := s }) := Model.initBias s (.Unknown "dynamic") device
def zerosDynamic (s : Shape) (grad : Bool := false) (device : Device := .CPU)
    (_noGrad : grad = false := by decide) : Tensor { device := device, shape := s } :=
  (Tensor.zeros s false device).forgetDType

def initLayerNorm (dim : UInt64) (eps : Float := 1e-5) (device : Device := .CPU) : IO (LayerNorm (device := device) dim (.Unknown "dynamic")) := do
  pure {
    weight := (← (Tensor.ones (device := device) #[dim]).forgetDType.parameter)
    bias := (← (Tensor.zeros (device := device) #[dim]).forgetDType.parameter)
    eps := ⟨eps⟩ }

def sliceTyped {s : Shape} {d : DType} (x : Tensor { device := device, shape := s, dtype := d }) (dim : Nat) (start len : UInt64) : Tensor { device := device, shape := (s.set! dim len), dtype := d } :=
  x.sliceChecked dim start len (s.set! dim len)

def transposeAttention {b s h d : UInt64} (x : Tensor { device := device, shape := #[b, s, h, d] }) : Tensor { device := device, shape := #[b, h, s, d] } := x.transpose 1 2

def fromAttention {b h s d : UInt64} (x : Tensor { device := device, shape := #[b, h, s, d] }) : Tensor { device := device, shape := #[b, s, h, d] } := x.transpose 1 2

end torch.whisper
