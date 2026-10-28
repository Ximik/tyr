import Tyr.Typed.Neural
import Tyr.RawTensor.KittenTTS
import Tyr.TensorStruct

namespace torch.kittentts

variable {device : Device}

private def checked {s : Shape} (raw : RawTensor) : Tensor { shape := s, dtype := .Float32, device := device } :=
  match Tensor.ofRaw raw with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"KittenTTS tensor boundary: {message}"

def catTo {s t : Shape} {d : DType} (x : Tensor { shape := s, dtype := d, device := device }) (y : Tensor { shape := t, dtype := d, device := device }) (dim : Nat) :
    Tensor { shape := (TensorSpec.catShape s t dim), dtype := d, device := device } := Tensor.cat x y dim

def affine2d {batch input output : Nat}
    (x : Tensor { shape := #[batch, input], dtype := .Float32, device := device }) (w : Tensor { shape := #[output, input], dtype := .Float32, device := device })
    (bias : Tensor { shape := #[output], dtype := .Float32, device := device }) : Tensor { shape := #[batch, output], dtype := .Float32, device := device } :=
  Tensor.linear x w + (bias.reshape #[1, output]).expand #[batch, output]

def bmm {batch rows inner cols : Nat} (a : Tensor { shape := #[batch, rows, inner], dtype := .Float32, device := device })
    (b : Tensor { shape := #[batch, inner, cols], dtype := .Float32, device := device }) : Tensor { shape := #[batch, rows, cols], dtype := .Float32, device := device } :=
  (Tensor.bmm4d (a.reshape #[batch, 1, rows, inner])
    (b.reshape #[batch, 1, inner, cols])).reshape #[batch, rows, cols]

def uniformTensor (shape : Shape) (lo hi : Float) : IO (Tensor { shape := shape, dtype := .Float32, device := device }) := do
  IO.ofExcept (Shape.validateAllocation shape)
  Tensor.uniformLike (Tensor.zeros shape (device := device)) lo hi

def randTensor (shape : Shape) (requiresGrad : Bool := false) (device : Device := .CPU) :
    IO (Tensor { shape := shape, dtype := .Float32, device := device }) := do
  IO.ofExcept (Shape.validateAllocation shape)
  let sample ← Tensor.uniformLike (Tensor.zeros shape (device := device))
  if requiresGrad then sample.parameter else pure sample

def leakyRelu {s : Shape} (x : Tensor { shape := s, dtype := .Float32, device := device }) (slope : Float := 0.01) : Tensor { shape := s, dtype := .Float32, device := device } :=
  checked (RawTensor.leakyRelu x.raw slope)
def floorTensor {s : Shape} (x : Tensor { shape := s, dtype := .Float32, device := device }) : Tensor { shape := s, dtype := .Float32, device := device } := checked (RawTensor.floor x.raw)
def atan2Tensor {s : Shape} (y x : Tensor { shape := s, dtype := .Float32, device := device }) : Tensor { shape := s, dtype := .Float32, device := device } := checked (RawTensor.atan2 y.raw x.raw)
def cumsumTensor {s : Shape} (x : Tensor { shape := s, dtype := .Float32, device := device }) (dim : Int64) : Tensor { shape := s, dtype := .Float32, device := device } := checked (RawTensor.cumsum x.raw dim)

def resizeTensor {batch channels frames : Nat} (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device })
    (outFrames : Nat) (mode : String) : Tensor { shape := #[batch, channels, outFrames], dtype := .Float32, device := device } :=
  checked (RawTensor.interpolate x.raw #[outFrames] mode false)

def istftTensor {s : Shape} (input : Tensor { shape := s, dtype := .Float32, device := device }) (nFft hopLength winLength : Nat)
    (window : Tensor { shape := #[winLength], dtype := .Float32, device := device }) (center normalized : Bool) (length : Nat) :
    Tensor { shape := #[length], dtype := .Float32, device := device } :=
  checked (RawTensor.istft input.raw nFft hopLength winLength window.raw center normalized length)

def sliceTensor {s : Shape} {d : DType} (x : Tensor { shape := s, dtype := d, device := device }) (dim : Nat) (start len : Nat) :
    Tensor { shape := (s.set! dim len), dtype := d, device := device } := x.sliceChecked dim start len (s.set! dim len)

def stftTensor {n winLength : Nat} {out : Shape} (input : Tensor { shape := #[n], dtype := .Float32, device := device })
    (nFft hopLength : Nat) (window : Tensor { shape := #[winLength], dtype := .Float32, device := device })
    (center normalized : Bool) : Tensor { shape := out, dtype := .Float32, device := device } :=
  checked (RawTensor.stft input.raw nFft hopLength winLength window.raw center normalized)

end torch.kittentts
