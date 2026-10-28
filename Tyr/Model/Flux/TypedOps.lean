import Tyr.Model.Qwen35.TypedOps
import Tyr.RawTensor.Flux
import Tyr.Module.LayerNorm
import Tyr.TensorStruct.DeriveCast

namespace torch.flux

variable {device : Device} {dtype : DType}
def randomWeight (s : Shape) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Tensor { dtype := dtype, device := device, shape := s }) := do
  IO.ofExcept ((← Tensor.randn (device := device) s).castDType? dtype)

def zerosWeight (s : Shape) (device : Device := .CPU) (dtype : DType := .Float32)
    [DType.ConstantSupported dtype device] : Tensor { dtype := dtype, device := device, shape := s } :=
  Tensor.zerosFor { shape := s, dtype, device }

def initLayerNorm (dim : Nat) (eps : Float := 1e-5) (train : Bool := true) (device : Device := .CPU) (dtype : DType := .Float32) :
    IO (LayerNorm (device := device) dim dtype) := do
  IO.ofExcept (Shape.validateAllocation #[dim])
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  let weight := Tensor.onesFor { shape := #[dim], dtype, device }
  let bias := Tensor.zerosFor { shape := #[dim], dtype, device }
  if train then
    pure { weight := (← weight.parameter), bias := (← bias.parameter), eps := ⟨eps⟩ }
  else
    pure { weight, bias, eps := ⟨eps⟩ }

def sliceTyped {s : Shape} {d : DType} (x : Tensor { device := device, shape := s, dtype := d }) (dim : Nat) (start len : Nat) : Tensor { device := device, shape := (s.set! dim len), dtype := d } :=
  x.sliceChecked dim start len (s.set! dim len)

def transposeAttention {b s h d : Nat} {dt : DType} (x : Tensor { device := device, shape := #[b, s, h, d], dtype := dt }) : Tensor { device := device, shape := #[b, h, s, d], dtype := dt } := x.transpose 1 2

def fromAttention {b h s d : Nat} {dt : DType} (x : Tensor { device := device, shape := #[b, h, s, d], dtype := dt }) : Tensor { device := device, shape := #[b, s, h, d], dtype := dt } := x.transpose 1 2

def saveImagePPM {b h w : Nat} (image : Tensor { dtype := dtype, device := device, shape := #[b, 3, h, w] }) (path : String) : IO Unit :=
  RawTensor.savePPM (image.to .CPU).raw path

end torch.flux
