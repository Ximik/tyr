import Tyr.Typed.Neural
import Tyr.RawTensor.Qwen35
import Tyr.Model.Utils

namespace torch.qwen35

private def checked {σ : StaticSpec} (raw : RawTensor) : Tensor σ :=
  match Tensor.ofRaw raw with
  | .ok result => result
  | .error message => panic! s!"Qwen35 tensor boundary: {message}"

def typedSlice {s out : Shape} {d : DType} (x : Tensor { shape := s, dtype := d }) (dim : Nat) (start len : UInt64) : Tensor { shape := out, dtype := d } :=
  x.sliceChecked dim start len out

def typedCat {s t out : Shape} {d : DType} (x : Tensor { shape := s, dtype := d }) (y : Tensor { shape := t, dtype := d }) (dim : Nat) : Tensor { shape := out, dtype := d } :=
  Tensor.catManyChecked #[x.packShape, y.packShape] (Int.ofNat dim) out

def transposeSeq {batch seq dim : UInt64} {d : DType} (x : Tensor { shape := #[batch, seq, dim], dtype := d }) :
    Tensor { shape := #[batch, dim, seq], dtype := d } := x.transpose 1 2

def einsumChecked {s t out : Shape} (equation : String) (x : Tensor { shape := s }) (y : Tensor { shape := t }) : Tensor { shape := out } :=
  checked (RawTensor.einsum2 s t equation x.raw y.raw)

def topK2d {rows cols : UInt64} (x : Tensor { shape := #[rows, cols] }) (k dim : UInt64) :
    Tensor { shape := #[rows, k] } × Tensor { shape := #[rows, k], dtype := .Int64 } :=
  let (values, indices) := RawTensor.topK2d rows cols x.raw k dim
  (checked values, checked indices)

def rsqrtDynamic {s : Shape} (x : Tensor { shape := s }) : Tensor { shape := s } :=
  Tensor.Promoting.rsqrt x
def logDynamic {s : Shape} (x : Tensor { shape := s }) : Tensor { shape := s } :=
  Tensor.Promoting.log x
def softplusDynamic {s : Shape} (x : Tensor { shape := s }) : Tensor { shape := s } :=
  checked (RawTensor.softplus s x.raw)
def floatDynamic {s : Shape} {d : DType} (x : Tensor { shape := s, dtype := d }) : Tensor { shape := s } := x.toFloat32.forgetDType

def zerosDynamic (shape : Shape) (requiresGrad : Bool := false) (device : Device := .CPU)
    (_noGrad : requiresGrad = false := by decide) : Tensor { shape := shape } :=
  (Tensor.zeros shape false device).forgetDType
def onesDynamic (shape : Shape) : Tensor { shape := shape } := (Tensor.ones shape).forgetDType
def zerosDynamicOn {shape : Shape} (device : Device) : Tensor { shape := shape } := zerosDynamic shape false device

def randDynamic (shape : Shape) : IO (Tensor { shape := shape }) := do
  return (← Tensor.uniformLike (Tensor.zeros shape)).forgetDType

def initDynamicWeight (shape : Shape) (fanIn : UInt64) : IO (Tensor { shape := shape }) :=
  Model.initWeight shape fanIn (.Unknown "dynamic")
def initDynamicBias (shape : Shape) : IO (Tensor { shape := shape }) := Model.initBias shape (.Unknown "dynamic")

def convGroups {batch cin cout cinGroup length kernel : UInt64} {out : Shape}
    (input : Tensor { shape := #[batch, cin, length] }) (weight : Tensor { shape := #[cout, cinGroup, kernel] })
    (bias : Tensor { shape := #[cout] }) (stride padding dilation groups : UInt64) : Tensor { shape := out } :=
  checked (RawTensor.conv1dGroups input.shape weight.shape bias.shape input.raw weight.raw bias.raw
    stride padding dilation groups)

def falseMaskTyped {n : UInt64} (device : Device) : Tensor { shape := #[n], dtype := .Bool } :=
  Tensor.eqScalar ((Tensor.fullInt #[n] 0).to device) 1

def chunkGatedDeltaRule {batch seq n_head kdim vdim : UInt64}
    (query key : Tensor { shape := #[batch, seq, n_head, kdim] })
    (value : Tensor { shape := #[batch, seq, n_head, vdim] })
    (g beta : Tensor { shape := #[batch, seq, n_head] }) :
    Tensor { shape := #[batch, seq, n_head, vdim] } × Tensor { shape := #[batch, n_head, kdim, vdim], dtype := .Float32 } :=
  let (out, state) := RawTensor.qwen35Chunk batch seq n_head kdim vdim query.raw key.raw value.raw g.raw beta.raw
  (checked out, checked state)

def recurrentGatedDeltaRule {batch seq n_head kdim vdim : UInt64}
    (query key : Tensor { shape := #[batch, seq, n_head, kdim] })
    (value : Tensor { shape := #[batch, seq, n_head, vdim] })
    (g beta : Tensor { shape := #[batch, seq, n_head] })
    (initialState : Tensor { shape := #[batch, n_head, kdim, vdim], dtype := .Float32 }) :
    Tensor { shape := #[batch, seq, n_head, vdim] } × Tensor { shape := #[batch, n_head, kdim, vdim], dtype := .Float32 } :=
  let (out, state) := RawTensor.qwen35Recurrent batch seq n_head kdim vdim
    query.raw key.raw value.raw g.raw beta.raw initialState.raw
  (checked out, checked state)

end torch.qwen35
