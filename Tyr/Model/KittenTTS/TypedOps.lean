import Tyr.Typed.Neural
import Tyr.RawTensor.KittenTTS
import Tyr.TensorStruct

namespace torch.kittentts

private def checked {s : Shape} (raw : RawTensor) : DTensor s .Float32 :=
  match Tensor.ofRaw raw with
  | .ok result => result
  | .error message => panic! s!"KittenTTS tensor boundary: {message}"

def catTo {s t : Shape} {d : DType} (x : DTensor s d) (y : DTensor t d) (dim : Nat) :
    DTensor (TensorSpec.catShape s t dim) d := Tensor.cat x y dim

def affine2d {batch input output : UInt64}
    (x : DTensor #[batch, input] .Float32) (w : DTensor #[output, input] .Float32)
    (bias : DTensor #[output] .Float32) : DTensor #[batch, output] .Float32 :=
  Tensor.linear x w + (bias.reshapeChecked #[1, output]).expand #[batch, output]

def bmm {batch rows inner cols : UInt64} (a : DTensor #[batch, rows, inner] .Float32)
    (b : DTensor #[batch, inner, cols] .Float32) : DTensor #[batch, rows, cols] .Float32 :=
  (Tensor.bmm4d (a.reshapeChecked #[batch, 1, rows, inner])
    (b.reshapeChecked #[batch, 1, inner, cols])).reshapeChecked #[batch, rows, cols]

def uniformTensor (shape : Shape) (lo hi : Float) : IO (DTensor shape .Float32) :=
  Tensor.uniformLike (Tensor.zeros shape) lo hi

def randTensor (shape : Shape) (requiresGrad : Bool := false) (device : Device := .CPU) :
    IO (DTensor shape .Float32) :=
  Tensor.uniformLike (Tensor.zeros shape requiresGrad device)

def leakyRelu {s : Shape} (x : DTensor s .Float32) (slope : Float := 0.01) : DTensor s .Float32 :=
  checked (RawTensor.leakyRelu s x.raw slope)
def floorTensor {s : Shape} (x : DTensor s .Float32) : DTensor s .Float32 := checked (RawTensor.floor s x.raw)
def atan2Tensor {s : Shape} (y x : DTensor s .Float32) : DTensor s .Float32 := checked (RawTensor.atan2 s y.raw x.raw)
def cumsumTensor {s : Shape} (x : DTensor s .Float32) (dim : Int64) : DTensor s .Float32 := checked (RawTensor.cumsum s x.raw dim)

def resizeTensor {batch channels frames : UInt64} (x : DTensor #[batch, channels, frames] .Float32)
    (outFrames : UInt64) (mode : String) : DTensor #[batch, channels, outFrames] .Float32 :=
  checked (RawTensor.interpolate x.shape x.raw #[outFrames] mode false)

def istftTensor {s : Shape} (input : DTensor s .Float32) (nFft hopLength winLength : UInt64)
    (window : DTensor #[winLength] .Float32) (center normalized : Bool) (length : UInt64) :
    DTensor #[length] .Float32 :=
  checked (RawTensor.istft input.raw nFft hopLength winLength window.raw center normalized length)

def sliceTensor {s : Shape} {d : DType} (x : DTensor s d) (dim : Nat) (start len : UInt64) :
    DTensor (s.set! dim len) d := x.sliceChecked dim start len (s.set! dim len)

def stftTensor {n winLength : UInt64} {out : Shape} (input : DTensor #[n] .Float32)
    (nFft hopLength : UInt64) (window : DTensor #[winLength] .Float32)
    (center normalized : Bool) : DTensor out .Float32 :=
  checked (RawTensor.stft n input.raw nFft hopLength winLength window.raw center normalized)

end torch.kittentts
