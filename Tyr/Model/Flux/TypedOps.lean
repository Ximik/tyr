import Tyr.Model.Qwen35.TypedOps
import Tyr.RawTensor.Flux
import Tyr.Module.LayerNorm

namespace torch.flux

variable {device : Device}
def randomWeight (s : Shape) (device : Device := .CPU) : IO (Tensor { device := device, shape := s }) := do
  return (← Tensor.randn (device := device) s).forgetDType

def zerosWeight (s : Shape) (device : Device := .CPU) : Tensor { device := device, shape := s } := (Tensor.zeros (device := device) s).forgetDType

def initLayerNorm (dim : UInt64) (eps : Float := 1e-5) (train : Bool := true) (device : Device := .CPU) :
    IO (LayerNorm (device := device) dim (.Unknown "dynamic")) := do
  let weight := (← Tensor.onesIO #[dim] device).forgetDType
  let bias := (← Tensor.zerosIO #[dim] device).forgetDType
  if train then
    pure { weight := (← weight.parameter), bias := (← bias.parameter), eps := ⟨eps⟩ }
  else
    pure { weight, bias, eps := ⟨eps⟩ }

def sliceTyped {s : Shape} {d : DType} (x : Tensor { device := device, shape := s, dtype := d }) (dim : Nat) (start len : UInt64) : Tensor { device := device, shape := (s.set! dim len), dtype := d } :=
  x.sliceChecked dim start len (s.set! dim len)

def transposeAttention {b s h d : UInt64} {dt : DType} (x : Tensor { device := device, shape := #[b, s, h, d], dtype := dt }) : Tensor { device := device, shape := #[b, h, s, d], dtype := dt } := x.transpose 1 2

def fromAttention {b h s d : UInt64} {dt : DType} (x : Tensor { device := device, shape := #[b, h, s, d], dtype := dt }) : Tensor { device := device, shape := #[b, s, h, d], dtype := dt } := x.transpose 1 2

def saveImagePPM {b h w : UInt64} (image : Tensor { device := device, shape := #[b, 3, h, w] }) (path : String) : IO Unit :=
  RawTensor.savePPM image.shape (image.to .CPU).raw path

end torch.flux
