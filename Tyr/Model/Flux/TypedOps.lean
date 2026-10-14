import Tyr.Model.Qwen35.TypedOps
import Tyr.RawTensor.Flux
import Tyr.Module.LayerNorm

namespace torch.flux

def randomWeight (s : Shape) : IO (Tensor { shape := s }) := do
  return (← Tensor.randn s).forgetDType

def zerosWeight (s : Shape) : Tensor { shape := s } := (Tensor.zeros s).forgetDType

def initLayerNorm (dim : UInt64) (eps : Float := 1e-5) (train : Bool := true) :
    IO (LayerNorm dim (.Unknown "dynamic")) := do
  let weight := (Tensor.ones #[dim]).forgetDType
  let bias := (Tensor.zeros #[dim]).forgetDType
  if train then
    pure { weight := (← weight.parameter), bias := (← bias.parameter), eps := ⟨eps⟩ }
  else
    pure { weight, bias, eps := ⟨eps⟩ }

def sliceTyped {s : Shape} {d : DType} (x : Tensor { shape := s, dtype := d }) (dim : Nat) (start len : UInt64) : Tensor { shape := (s.set! dim len), dtype := d } :=
  x.sliceChecked dim start len (s.set! dim len)

def transposeAttention {b s h d : UInt64} {dt : DType} (x : Tensor { shape := #[b, s, h, d], dtype := dt }) : Tensor { shape := #[b, h, s, d], dtype := dt } := x.transpose 1 2

def fromAttention {b h s d : UInt64} {dt : DType} (x : Tensor { shape := #[b, h, s, d], dtype := dt }) : Tensor { shape := #[b, s, h, d], dtype := dt } := x.transpose 1 2

def saveImagePPM {b h w : UInt64} (image : Tensor { shape := #[b, 3, h, w] }) (path : String) : IO Unit :=
  RawTensor.savePPM image.shape (image.to .CPU).raw path

end torch.flux
