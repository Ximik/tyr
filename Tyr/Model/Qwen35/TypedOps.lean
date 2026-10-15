import Tyr.Typed.Neural
import Tyr.RawTensor.Qwen35
import Tyr.Model.Utils

namespace torch.qwen35

variable {device : Device} {dtype : DType}

private def checked {σ : StaticSpec} (raw : RawTensor) : Tensor σ :=
  match Tensor.ofRaw raw with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"Qwen35 tensor boundary: {message}"

/-- Retain the selected dtype after mixed-precision arithmetic. In particular,
expert routing multiplies before adding to its Float32 accumulator. -/
def promotedAdd {shape : Shape} {leftDType rightDType : DType}
    (left : Tensor ⟨shape, leftDType, device⟩)
    (right : Tensor ⟨shape, rightDType, device⟩) : SomeDTypeTensor shape device :=
  match Tensor.Promoting.add? left right >>= fun result => result.tensor.reifyDType with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"Qwen35 mixed addition: {message}"

def promotedMul {shape : Shape} {leftDType rightDType : DType}
    (left : Tensor ⟨shape, leftDType, device⟩)
    (right : Tensor ⟨shape, rightDType, device⟩) : SomeDTypeTensor shape device :=
  match Tensor.Promoting.mul? left right >>= fun result => result.tensor.reifyDType with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"Qwen35 mixed multiplication: {message}"

def typedSlice {s out : Shape} {d : DType} (x : Tensor { device := device, shape := s, dtype := d }) (dim : Nat) (start len : UInt64) : Tensor { device := device, shape := out, dtype := d } :=
  x.sliceChecked dim start len out

def typedCat {s t out : Shape} {d : DType} (x : Tensor { device := device, shape := s, dtype := d }) (y : Tensor { device := device, shape := t, dtype := d }) (dim : Nat) : Tensor { device := device, shape := out, dtype := d } :=
  Tensor.catManyChecked #[x.packShape, y.packShape] (Int.ofNat dim) out

def transposeSeq {batch seq dim : UInt64} {d : DType} (x : Tensor { device := device, shape := #[batch, seq, dim], dtype := d }) :
    Tensor { device := device, shape := #[batch, dim, seq], dtype := d } := x.transpose 1 2

def einsumChecked {s t out : Shape} (equation : String) (x : Tensor { device := device, shape := s, dtype := dtype }) (y : Tensor { device := device, shape := t, dtype := dtype }) : Tensor { device := device, shape := out, dtype := dtype } :=
  checked (RawTensor.einsum2 s t equation x.raw y.raw)

def topK2d {rows cols : UInt64} (x : Tensor { device := device, shape := #[rows, cols], dtype := dtype }) (k dim : UInt64) :
    Tensor { device := device, shape := #[rows, k], dtype := dtype } × Tensor { device := device, shape := #[rows, k], dtype := .Int64 } :=
  let (values, indices) := RawTensor.topK2d rows cols x.raw k dim
  (checked values, checked indices)

def rsqrtDynamic {s : Shape} (x : Tensor { device := device, shape := s, dtype := dtype }) : Tensor { device := device, shape := s, dtype := dtype } :=
  Tensor.Preserving.rsqrt x
def logDynamic {s : Shape} (x : Tensor { device := device, shape := s, dtype := dtype }) : Tensor { device := device, shape := s, dtype := dtype } :=
  Tensor.Preserving.log x
def softplusDynamic {s : Shape} (x : Tensor { device := device, shape := s, dtype := dtype }) : Tensor { device := device, shape := s, dtype := dtype } :=
  checked (RawTensor.softplus s x.raw)
def floatDynamic {s : Shape} {d : DType} (x : Tensor { device := device, shape := s, dtype := d }) : Tensor { device := device, shape := s, dtype := .Float32 } := x.toFloat32

/-- Explicit-dtype constants for shape-parametric model code. -/
def zerosDynamic (shape : Shape) (requiresGrad : Bool := false) (device : Device := .CPU)
    (_noGrad : requiresGrad = false := by decide) (dtype : DType := .Float32)
    [DType.ConstantSupported dtype device] :
    Tensor { device, shape, dtype } :=
  Tensor.zerosFor { device, shape, dtype }
def onesDynamic (shape : Shape) (device : Device := .CPU) (dtype : DType := .Float32)
    [DType.ConstantSupported dtype device] :
    Tensor { device, shape, dtype } :=
  Tensor.onesFor { device, shape, dtype }
def zerosDynamicOn {shape : Shape} (device : Device) (dtype : DType := .Float32)
    [DType.ConstantSupported dtype device] :
    Tensor { device, shape, dtype } := zerosDynamic shape false device (dtype := dtype)

def randDynamic (shape : Shape) (device : Device := .CPU) (dtype : DType := .Float32) :
    IO (Tensor { device, shape, dtype }) := do
  IO.ofExcept (Shape.validateAllocation shape)
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  Tensor.uniformLike (Tensor.zerosFor { shape, dtype, device })

def initDynamicWeight (shape : Shape) (fanIn : UInt64) (device : Device := .CPU)
    (dtype : DType := .Float32) : IO (Tensor { device, shape, dtype }) :=
  Model.initWeight shape fanIn dtype device
def initDynamicBias (shape : Shape) (device : Device := .CPU) (dtype : DType := .Float32) :
    IO (Tensor { device, shape, dtype }) := Model.initBias shape dtype device

def convGroups {batch cin cout cinGroup length kernel : UInt64} {out : Shape}
    (input : Tensor { device := device, shape := #[batch, cin, length], dtype := dtype }) (weight : Tensor { device := device, shape := #[cout, cinGroup, kernel], dtype := dtype })
    (bias : Tensor { device := device, shape := #[cout], dtype := dtype }) (stride padding dilation groups : UInt64) : Tensor { device := device, shape := out, dtype := dtype } :=
  checked (RawTensor.conv1dGroups input.shape weight.shape bias.shape input.raw weight.raw bias.raw
    stride padding dilation groups)

def falseMaskTyped {n : UInt64} (device : Device) : Tensor { device := device, shape := #[n], dtype := .Bool } :=
  Tensor.eqScalar ((Tensor.fullInt #[n] 0).to device) 1

def chunkGatedDeltaRule {batch seq n_head kdim vdim : UInt64}
    (query key : Tensor { device := device, shape := #[batch, seq, n_head, kdim], dtype := dtype })
    (value : Tensor { device := device, shape := #[batch, seq, n_head, vdim], dtype := dtype })
    (g : Tensor { device := device, shape := #[batch, seq, n_head], dtype := .Float32 })
    (beta : Tensor { device := device, shape := #[batch, seq, n_head], dtype := dtype }) :
    Tensor { device := device, shape := #[batch, seq, n_head, vdim], dtype := dtype } × Tensor { device := device, shape := #[batch, n_head, kdim, vdim], dtype := .Float32 } :=
  let (out, state) := RawTensor.qwen35Chunk batch seq n_head kdim vdim query.raw key.raw value.raw g.raw beta.raw
  (checked out, checked state)

def recurrentGatedDeltaRule {batch seq n_head kdim vdim : UInt64}
    (query key : Tensor { device := device, shape := #[batch, seq, n_head, kdim], dtype := dtype })
    (value : Tensor { device := device, shape := #[batch, seq, n_head, vdim], dtype := dtype })
    (g : Tensor { device := device, shape := #[batch, seq, n_head], dtype := .Float32 })
    (beta : Tensor { device := device, shape := #[batch, seq, n_head], dtype := dtype })
    (initialState : Tensor { device := device, shape := #[batch, n_head, kdim, vdim], dtype := .Float32 }) :
    Tensor { device := device, shape := #[batch, seq, n_head, vdim], dtype := dtype } × Tensor { device := device, shape := #[batch, n_head, kdim, vdim], dtype := .Float32 } :=
  let (out, state) := RawTensor.qwen35Recurrent batch seq n_head kdim vdim
    query.raw key.raw value.raw g.raw beta.raw initialState.raw
  (checked out, checked state)

end torch.qwen35
