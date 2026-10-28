import Tyr.Model.Utils
import Tyr.Module.LayerNorm

namespace torch.whisper

variable {device : Device} {dtype : DType}
def initWeightFor (s : Shape) (fanIn : Nat) (device : Device := .CPU)
    (dtype : DType := .Float32) : IO (Tensor { device := device, shape := s, dtype }) :=
  Model.initWeight s fanIn dtype device
def initBiasFor (s : Shape) (device : Device := .CPU)
    (dtype : DType := .Float32) : IO (Tensor { device := device, shape := s, dtype }) :=
  Model.initBias s dtype device
def zerosFor (s : Shape) (grad : Bool := false) (device : Device := .CPU)
    (_noGrad : grad = false := by decide) (dtype : DType := .Float32)
    [DType.ConstantSupported dtype device] :
    Tensor { device := device, shape := s, dtype } :=
  Tensor.zerosFor { shape := s, dtype, device }

def initLayerNorm (dim : Nat) (eps : Float := 1e-5) (device : Device := .CPU)
    (dtype : DType := .Float32) : IO (LayerNorm (device := device) dim dtype) := do
  IO.ofExcept (Shape.validateAllocation #[dim])
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  pure {
    weight := (← (Tensor.onesFor { shape := #[dim], dtype, device }).parameter)
    bias := (← (Tensor.zerosFor { shape := #[dim], dtype, device }).parameter)
    eps := ⟨eps⟩ }

def sliceTyped {s : Shape} {d : DType} (x : Tensor { device := device, shape := s, dtype := d }) (dim : Nat) (start len : Nat) : Tensor { device := device, shape := (s.set! dim len), dtype := d } :=
  x.sliceChecked dim start len (s.set! dim len)

def transposeAttention {b s h d : Nat} (x : Tensor { device := device, shape := #[b, s, h, d], dtype }) : Tensor { device := device, shape := #[b, h, s, d], dtype } := x.transpose 1 2

def fromAttention {b h s d : Nat} (x : Tensor { device := device, shape := #[b, h, s, d], dtype }) : Tensor { device := device, shape := #[b, s, h, d], dtype } := x.transpose 1 2

end torch.whisper
