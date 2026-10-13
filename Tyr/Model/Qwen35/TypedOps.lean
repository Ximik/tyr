import Tyr.Typed.Neural
import Tyr.RawTensor.Qwen35
import Tyr.Model.Utils

namespace torch.qwen35

private def checked {σ : StaticSpec} (raw : RawTensor) : Tensor σ :=
  match Tensor.ofRaw raw with
  | .ok result => result
  | .error message => panic! s!"Qwen35 tensor boundary: {message}"

def typedSlice {s out : Shape} {d : DType} (x : DTensor s d) (dim : Nat) (start len : UInt64) : DTensor out d :=
  x.sliceChecked dim start len out

def typedCat {s t out : Shape} {d : DType} (x : DTensor s d) (y : DTensor t d) (dim : Nat) : DTensor out d :=
  Tensor.catManyChecked #[x.packShape, y.packShape] (Int.ofNat dim) out

def transposeSeq {batch seq dim : UInt64} {d : DType} (x : DTensor #[batch, seq, dim] d) :
    DTensor #[batch, dim, seq] d := x.transpose 1 2

def einsumChecked {s t out : Shape} (equation : String) (x : STensor s) (y : STensor t) : STensor out :=
  checked (RawTensor.einsum2 s t equation x.raw y.raw)

def topK2d {rows cols : UInt64} (x : STensor #[rows, cols]) (k dim : UInt64) :
    STensor #[rows, k] × DTensor #[rows, k] .Int64 :=
  let (values, indices) := RawTensor.topK2d rows cols x.raw k dim
  (checked values, checked indices)

def rsqrtDynamic {s : Shape} (x : STensor s) : STensor s :=
  checked (RawTensor.rsqrt s x.raw)
def logDynamic {s : Shape} (x : STensor s) : STensor s :=
  checked (RawTensor.log s x.raw)
def softplusDynamic {s : Shape} (x : STensor s) : STensor s :=
  checked (RawTensor.softplus s x.raw)
def floatDynamic {s : Shape} {d : DType} (x : DTensor s d) : STensor s := x.toFloat32.forgetDType

def zerosDynamic (shape : Shape) (requiresGrad : Bool := false) (device : Device := .CPU) : STensor shape :=
  (Tensor.zeros shape requiresGrad device).forgetDType
def onesDynamic (shape : Shape) : STensor shape := (Tensor.ones shape).forgetDType
def zerosDynamicOn {shape : Shape} (device : Device) : STensor shape := zerosDynamic shape false device

def randDynamic (shape : Shape) : IO (STensor shape) := do
  return (← Tensor.uniformLike (Tensor.zeros shape)).forgetDType

def initDynamicWeight (shape : Shape) (fanIn : UInt64) : IO (STensor shape) :=
  Model.initWeight shape fanIn (.Unknown "dynamic")
def initDynamicBias (shape : Shape) : STensor shape := Model.initBias shape (.Unknown "dynamic")

def convGroups {batch cin cout cinGroup length kernel : UInt64} {out : Shape}
    (input : STensor #[batch, cin, length]) (weight : STensor #[cout, cinGroup, kernel])
    (bias : STensor #[cout]) (stride padding dilation groups : UInt64) : STensor out :=
  checked (RawTensor.conv1dGroups input.shape weight.shape bias.shape input.raw weight.raw bias.raw
    stride padding dilation groups)

def falseMaskTyped {n : UInt64} (device : Device) : DTensor #[n] .Bool :=
  Tensor.eqScalar ((Tensor.fullInt #[n] 0).to device) 1

def chunkGatedDeltaRule {batch seq n_head kdim vdim : UInt64}
    (query key : STensor #[batch, seq, n_head, kdim])
    (value : STensor #[batch, seq, n_head, vdim])
    (g beta : STensor #[batch, seq, n_head]) :
    STensor #[batch, seq, n_head, vdim] × DTensor #[batch, n_head, kdim, vdim] .Float32 :=
  let (out, state) := RawTensor.qwen35Chunk batch seq n_head kdim vdim query.raw key.raw value.raw g.raw beta.raw
  (checked out, checked state)

def recurrentGatedDeltaRule {batch seq n_head kdim vdim : UInt64}
    (query key : STensor #[batch, seq, n_head, kdim])
    (value : STensor #[batch, seq, n_head, vdim])
    (g beta : STensor #[batch, seq, n_head])
    (initialState : DTensor #[batch, n_head, kdim, vdim] .Float32) :
    STensor #[batch, seq, n_head, vdim] × DTensor #[batch, n_head, kdim, vdim] .Float32 :=
  let (out, state) := RawTensor.qwen35Recurrent batch seq n_head kdim vdim
    query.raw key.raw value.raw g.raw beta.raw initialState.raw
  (checked out, checked state)

end torch.qwen35
