import Tyr.Typed.Neural
import Tyr.RawTensor.Qwen35
import Tyr.Model.Utils

namespace torch.qwen35

variable {device : Device}

private def checked {σ : StaticSpec} (raw : RawTensor) : Tensor σ :=
  match Tensor.ofRaw raw with
  | .ok result => result
  | .error message => panic! s!"Qwen35 tensor boundary: {message}"

def typedSlice {s out : Shape} {d : DType} (x : Tensor { device := device, shape := s, dtype := d }) (dim : Nat) (start len : UInt64) : Tensor { device := device, shape := out, dtype := d } :=
  x.sliceChecked dim start len out

def typedCat {s t out : Shape} {d : DType} (x : Tensor { device := device, shape := s, dtype := d }) (y : Tensor { device := device, shape := t, dtype := d }) (dim : Nat) : Tensor { device := device, shape := out, dtype := d } :=
  Tensor.catManyChecked #[x.packShape, y.packShape] (Int.ofNat dim) out

def transposeSeq {batch seq dim : UInt64} {d : DType} (x : Tensor { device := device, shape := #[batch, seq, dim], dtype := d }) :
    Tensor { device := device, shape := #[batch, dim, seq], dtype := d } := x.transpose 1 2

def einsumChecked {s t out : Shape} (equation : String) (x : Tensor { device := device, shape := s }) (y : Tensor { device := device, shape := t }) : Tensor { device := device, shape := out } :=
  checked (RawTensor.einsum2 s t equation x.raw y.raw)

def topK2d {rows cols : UInt64} (x : Tensor { device := device, shape := #[rows, cols] }) (k dim : UInt64) :
    Tensor { device := device, shape := #[rows, k] } × Tensor { device := device, shape := #[rows, k], dtype := .Int64 } :=
  let (values, indices) := RawTensor.topK2d rows cols x.raw k dim
  (checked values, checked indices)

def rsqrtDynamic {s : Shape} (x : Tensor { device := device, shape := s }) : Tensor { device := device, shape := s } :=
  Tensor.Promoting.rsqrt x
def logDynamic {s : Shape} (x : Tensor { device := device, shape := s }) : Tensor { device := device, shape := s } :=
  Tensor.Promoting.log x
def softplusDynamic {s : Shape} (x : Tensor { device := device, shape := s }) : Tensor { device := device, shape := s } :=
  checked (RawTensor.softplus s x.raw)
def floatDynamic {s : Shape} {d : DType} (x : Tensor { device := device, shape := s, dtype := d }) : Tensor { device := device, shape := s } := x.toFloat32.forgetDType

def zerosDynamic (shape : Shape) (requiresGrad : Bool := false) (device : Device := .CPU)
    (_noGrad : requiresGrad = false := by decide) : Tensor { device := device, shape := shape } :=
  (Tensor.zeros shape false device).forgetDType
def onesDynamic (shape : Shape) (device : Device := .CPU) : Tensor { device, shape } :=
  (Tensor.ones shape (device := device)).forgetDType
def zerosDynamicOn {shape : Shape} (device : Device) : Tensor { device := device, shape := shape } := zerosDynamic shape false device

def randDynamic (shape : Shape) (device : Device := .CPU) : IO (Tensor { device, shape }) := do
  return (← Tensor.uniformLike (← Tensor.zerosIO shape device)).forgetDType

def initDynamicWeight (shape : Shape) (fanIn : UInt64) (device : Device := .CPU) :
    IO (Tensor { device, shape }) :=
  Model.initWeight shape fanIn (.Unknown "dynamic") device
def initDynamicBias (shape : Shape) (device : Device := .CPU) : IO (Tensor { device, shape }) :=
  Model.initBias shape (.Unknown "dynamic") device

def convGroups {batch cin cout cinGroup length kernel : UInt64} {out : Shape}
    (input : Tensor { device := device, shape := #[batch, cin, length] }) (weight : Tensor { device := device, shape := #[cout, cinGroup, kernel] })
    (bias : Tensor { device := device, shape := #[cout] }) (stride padding dilation groups : UInt64) : Tensor { device := device, shape := out } :=
  checked (RawTensor.conv1dGroups input.shape weight.shape bias.shape input.raw weight.raw bias.raw
    stride padding dilation groups)

def falseMaskTyped {n : UInt64} (device : Device) : Tensor { device := device, shape := #[n], dtype := .Bool } :=
  Tensor.eqScalar ((Tensor.fullInt #[n] 0).to device) 1

def chunkGatedDeltaRule {batch seq n_head kdim vdim : UInt64}
    (query key : Tensor { device := device, shape := #[batch, seq, n_head, kdim] })
    (value : Tensor { device := device, shape := #[batch, seq, n_head, vdim] })
    (g beta : Tensor { device := device, shape := #[batch, seq, n_head] }) :
    Tensor { device := device, shape := #[batch, seq, n_head, vdim] } × Tensor { device := device, shape := #[batch, n_head, kdim, vdim], dtype := .Float32 } :=
  let (out, state) := RawTensor.qwen35Chunk batch seq n_head kdim vdim query.raw key.raw value.raw g.raw beta.raw
  (checked out, checked state)

def recurrentGatedDeltaRule {batch seq n_head kdim vdim : UInt64}
    (query key : Tensor { device := device, shape := #[batch, seq, n_head, kdim] })
    (value : Tensor { device := device, shape := #[batch, seq, n_head, vdim] })
    (g beta : Tensor { device := device, shape := #[batch, seq, n_head] })
    (initialState : Tensor { device := device, shape := #[batch, n_head, kdim, vdim], dtype := .Float32 }) :
    Tensor { device := device, shape := #[batch, seq, n_head, vdim] } × Tensor { device := device, shape := #[batch, n_head, kdim, vdim], dtype := .Float32 } :=
  let (out, state) := RawTensor.qwen35Recurrent batch seq n_head kdim vdim
    query.raw key.raw value.raw g.raw beta.raw initialState.raw
  (checked out, checked state)

end torch.qwen35
