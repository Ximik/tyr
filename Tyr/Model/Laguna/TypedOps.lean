import Tyr.Model.Qwen35.TypedOps
import Tyr.RawTensor.Laguna

namespace torch.laguna

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]
private def checked {σ : StaticSpec} (raw : RawTensor) : Tensor σ :=
  match Tensor.ofRaw raw with
  | .ok tensor => tensor
  | .error message => nativeInvariantFailure s!"Laguna tensor boundary: {message}"

def mulBroadcast {s t out : Shape} (a : Tensor { dtype := dtype, device := device, shape := s }) (b : Tensor { dtype := dtype, device := device, shape := t }) : Tensor { dtype := dtype, device := device, shape := out } :=
  match Tensor.Promoting.mulB? a b out with
  | .ok result => checked result.tensor.raw
  | .error message => nativeInvariantFailure s!"Laguna.mulBroadcast: {message}"

def bmm {b m n k : UInt64} (x : Tensor { dtype := dtype, device := device, shape := #[b, m, k] }) (y : Tensor { dtype := dtype, device := device, shape := #[b, k, n] }) :
    Tensor { dtype := dtype, device := device, shape := #[b, m, n] } :=
  checked (RawTensor.lagunaBmm x.raw y.raw)

def scatterAdd {s t : Shape} (input : Tensor { dtype := dtype, device := device, shape := s }) (dim : Int64)
    (indices : Tensor { device := device, shape := t, dtype := .Int64 }) (src : Tensor { dtype := dtype, device := device, shape := t }) : Tensor { dtype := dtype, device := device, shape := s } :=
  input.checkedReplacement (RawTensor.lagunaScatterAdd input.raw dim indices.raw src.raw)

def transposeMatrixBatch {b m n : UInt64} {d : DType}
    (x : Tensor { device := device, shape := #[b, m, n], dtype := d }) : Tensor { device := device, shape := #[b, n, m], dtype := d } := x.transpose 1 2

def mulSame {s : Shape} (x y : Tensor { dtype := dtype, device := device, shape := s }) : Tensor { dtype := dtype, device := device, shape := s } := Tensor.mul x y
def addSame {s : Shape} (x y : Tensor { dtype := dtype, device := device, shape := s }) : Tensor { dtype := dtype, device := device, shape := s } := Tensor.add x y

def transposeAttention {b s h d : UInt64} {dt : DType}
    (x : Tensor { device := device, shape := #[b, s, h, d], dtype := dt }) : Tensor { device := device, shape := #[b, h, s, d], dtype := dt } := x.transpose 1 2

def sliceTyped {s : Shape} {d : DType} (x : Tensor { device := device, shape := s, dtype := d })
    (dim : Nat) (start len : UInt64) : Tensor { device := device, shape := (s.set! dim len), dtype := d } :=
  x.sliceChecked dim start len (s.set! dim len)

def randnDynamic (shape : Shape) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Tensor { dtype := dtype, device := device, shape := shape }) := do
  IO.ofExcept ((← Tensor.randn (device := device) shape).castDType? dtype)

def hasGradFn {σ : StaticSpec} (x : Tensor σ) : Bool :=
  RawTensor.lagunaHasGradFn x.raw

end torch.laguna
