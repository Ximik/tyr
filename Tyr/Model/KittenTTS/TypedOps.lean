import Tyr.Typed.Neural
import Tyr.RawTensor.KittenTTS
import Tyr.TensorStruct

namespace torch.kittentts

private def checked {s : Shape} (raw : RawTensor) : Tensor { shape := s, dtype := .Float32 } :=
  match Tensor.ofRaw raw with
  | .ok result => result
  | .error message => panic! s!"KittenTTS tensor boundary: {message}"

def catTo {s t : Shape} {d : DType} (x : Tensor { shape := s, dtype := d }) (y : Tensor { shape := t, dtype := d }) (dim : Nat) :
    Tensor { shape := (TensorSpec.catShape s t dim), dtype := d } := Tensor.cat x y dim

def affine2d {batch input output : UInt64}
    (x : Tensor { shape := #[batch, input], dtype := .Float32 }) (w : Tensor { shape := #[output, input], dtype := .Float32 })
    (bias : Tensor { shape := #[output], dtype := .Float32 }) : Tensor { shape := #[batch, output], dtype := .Float32 } :=
  Tensor.linear x w + (bias.reshapeChecked #[1, output]).expand #[batch, output]

def bmm {batch rows inner cols : UInt64} (a : Tensor { shape := #[batch, rows, inner], dtype := .Float32 })
    (b : Tensor { shape := #[batch, inner, cols], dtype := .Float32 }) : Tensor { shape := #[batch, rows, cols], dtype := .Float32 } :=
  (Tensor.bmm4d (a.reshapeChecked #[batch, 1, rows, inner])
    (b.reshapeChecked #[batch, 1, inner, cols])).reshapeChecked #[batch, rows, cols]

def uniformTensor (shape : Shape) (lo hi : Float) : IO (Tensor { shape := shape, dtype := .Float32 }) :=
  Tensor.uniformLike (Tensor.zeros shape) lo hi

def randTensor (shape : Shape) (requiresGrad : Bool := false) (device : Device := .CPU) :
    IO (Tensor { shape := shape, dtype := .Float32 }) :=
  Tensor.uniformLike (Tensor.zeros shape requiresGrad device)

def leakyRelu {s : Shape} (x : Tensor { shape := s, dtype := .Float32 }) (slope : Float := 0.01) : Tensor { shape := s, dtype := .Float32 } :=
  checked (RawTensor.leakyRelu s x.raw slope)
def floorTensor {s : Shape} (x : Tensor { shape := s, dtype := .Float32 }) : Tensor { shape := s, dtype := .Float32 } := checked (RawTensor.floor s x.raw)
def atan2Tensor {s : Shape} (y x : Tensor { shape := s, dtype := .Float32 }) : Tensor { shape := s, dtype := .Float32 } := checked (RawTensor.atan2 s y.raw x.raw)
def cumsumTensor {s : Shape} (x : Tensor { shape := s, dtype := .Float32 }) (dim : Int64) : Tensor { shape := s, dtype := .Float32 } := checked (RawTensor.cumsum s x.raw dim)

def resizeTensor {batch channels frames : UInt64} (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32 })
    (outFrames : UInt64) (mode : String) : Tensor { shape := #[batch, channels, outFrames], dtype := .Float32 } :=
  checked (RawTensor.interpolate x.shape x.raw #[outFrames] mode false)

def istftTensor {s : Shape} (input : Tensor { shape := s, dtype := .Float32 }) (nFft hopLength winLength : UInt64)
    (window : Tensor { shape := #[winLength], dtype := .Float32 }) (center normalized : Bool) (length : UInt64) :
    Tensor { shape := #[length], dtype := .Float32 } :=
  checked (RawTensor.istft input.raw nFft hopLength winLength window.raw center normalized length)

def sliceTensor {s : Shape} {d : DType} (x : Tensor { shape := s, dtype := d }) (dim : Nat) (start len : UInt64) :
    Tensor { shape := (s.set! dim len), dtype := d } := x.sliceChecked dim start len (s.set! dim len)

def stftTensor {n winLength : UInt64} {out : Shape} (input : Tensor { shape := #[n], dtype := .Float32 })
    (nFft hopLength : UInt64) (window : Tensor { shape := #[winLength], dtype := .Float32 })
    (center normalized : Bool) : Tensor { shape := out, dtype := .Float32 } :=
  checked (RawTensor.stft n input.raw nFft hopLength winLength window.raw center normalized)

end torch.kittentts
