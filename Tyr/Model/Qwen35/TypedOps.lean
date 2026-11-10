import Tyr.Typed.Neural
import Tyr.RawTensor.Qwen35
import Tyr.Model.Utils

namespace torch.qwen35

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]

def transposeSeq {batch seq dim : Nat} {d : DType} (x : Tensor { device := device, shape := #[batch, seq, dim], dtype := d }) :
    Tensor { device := device, shape := #[batch, dim, seq], dtype := d } := x.transpose 1 2

def rsqrtDynamic {s : Shape} (x : Tensor { device := device, shape := s, dtype := dtype }) : Tensor { device := device, shape := s, dtype := dtype } :=
  Tensor.rsqrt x
def logDynamic {s : Shape} (x : Tensor { device := device, shape := s, dtype := dtype }) : Tensor { device := device, shape := s, dtype := dtype } :=
  Tensor.log x
def softplusDynamic {s : Shape} (x : Tensor { device := device, shape := s, dtype := dtype }) : Tensor { device := device, shape := s, dtype := dtype } :=
  Tensor.ofRaw! (RawTensor.softplus x.raw)
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

def initDynamicWeight (shape : Shape) (fanIn : Nat) (device : Device := .CPU)
    (dtype : DType := .Float32) : IO (Tensor { device, shape, dtype }) :=
  Model.initWeight shape fanIn dtype device
def initDynamicBias (shape : Shape) (device : Device := .CPU) (dtype : DType := .Float32) :
    IO (Tensor { device, shape, dtype }) := Model.initBias shape dtype device

def falseMaskTyped {n : Nat} (device : Device) : Tensor { device := device, shape := #[n], dtype := .Bool } :=
  Tensor.eqScalar ((Tensor.fullInt #[n] 0).to device) 1

def chunkGatedDeltaRule {batch seq n_head kdim vdim : Nat}
    (query key : Tensor { device := device, shape := #[batch, seq, n_head, kdim], dtype := dtype })
    (value : Tensor { device := device, shape := #[batch, seq, n_head, vdim], dtype := dtype })
    (g : Tensor { device := device, shape := #[batch, seq, n_head], dtype := .Float32 })
    (beta : Tensor { device := device, shape := #[batch, seq, n_head], dtype := dtype }) :
    Tensor { device := device, shape := #[batch, seq, n_head, vdim], dtype := dtype } × Tensor { device := device, shape := #[batch, n_head, kdim, vdim], dtype := .Float32 } :=
  let (out, state) := RawTensor.qwen35Chunk query.raw key.raw value.raw g.raw beta.raw
  (Tensor.ofRaw! out, Tensor.ofRaw! state)

def recurrentGatedDeltaRule {batch seq n_head kdim vdim : Nat}
    (query key : Tensor { device := device, shape := #[batch, seq, n_head, kdim], dtype := dtype })
    (value : Tensor { device := device, shape := #[batch, seq, n_head, vdim], dtype := dtype })
    (g : Tensor { device := device, shape := #[batch, seq, n_head], dtype := .Float32 })
    (beta : Tensor { device := device, shape := #[batch, seq, n_head], dtype := dtype })
    (initialState : Tensor { device := device, shape := #[batch, n_head, kdim, vdim], dtype := .Float32 }) :
    Tensor { device := device, shape := #[batch, seq, n_head, vdim], dtype := dtype } × Tensor { device := device, shape := #[batch, n_head, kdim, vdim], dtype := .Float32 } :=
  let (out, state) := RawTensor.qwen35Recurrent
    query.raw key.raw value.raw g.raw beta.raw initialState.raw
  (Tensor.ofRaw! out, Tensor.ofRaw! state)

end torch.qwen35
