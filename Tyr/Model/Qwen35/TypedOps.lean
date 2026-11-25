import Tyr.Typed.Neural
import Tyr.FFI.Qwen35
import Tyr.Model.Utils

namespace torch.qwen35

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]

def transposeSeq {batch seq dim : Nat} {d : DType} (x : Tensor { device := device, shape := #[batch, seq, dim], dtype := d }) :
    Tensor { device := device, shape := #[batch, dim, seq], dtype := d } := x.transpose 1 2

def softplusDynamic {s : Shape} (x : Tensor { device := device, shape := s, dtype := dtype }) : Tensor { device := device, shape := s, dtype := dtype } :=
  Tensor.ofRaw! (RawTensor.softplus x.raw)

def randDynamic (shape : Shape) (device : Device := .CPU) (dtype : DType := .Float32) :
    IO (Tensor { device, shape, dtype }) := do
  IO.ofExcept (Shape.validateAllocation shape)
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  Tensor.uniformLike (Tensor.zerosFor { shape, dtype, device })

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
