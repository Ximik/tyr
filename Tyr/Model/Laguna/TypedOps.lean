import Tyr.Model.Qwen35.TypedOps
import Tyr.RawTensor.Laguna

namespace torch.laguna

variable {device : Device}
private def checked {σ : StaticSpec} (raw : RawTensor) : Tensor σ :=
  match Tensor.ofRaw raw with
  | .ok tensor => tensor
  | .error message => panic! s!"Laguna tensor boundary: {message}"

def mulBroadcast {s t out : Shape} (a : Tensor { device := device, shape := s }) (b : Tensor { device := device, shape := t }) : Tensor { device := device, shape := out } :=
  match Tensor.Promoting.mulB? a b out with
  | .ok result => checked result.tensor.raw
  | .error message => panic! s!"Laguna.mulBroadcast: {message}"

def bmm {b m n k : UInt64} (x : Tensor { device := device, shape := #[b, m, k] }) (y : Tensor { device := device, shape := #[b, k, n] }) :
    Tensor { device := device, shape := #[b, m, n] } :=
  checked (RawTensor.lagunaBmm b m n k x.raw y.raw)

def scatterAdd {s t : Shape} (input : Tensor { device := device, shape := s }) (dim : Int64)
    (indices : Tensor { device := device, shape := t, dtype := .Int64 }) (src : Tensor { device := device, shape := t }) : Tensor { device := device, shape := s } :=
  input.checkedReplacement (RawTensor.lagunaScatterAdd s input.raw dim indices.raw src.raw)

def transposeMatrixBatch {b m n : UInt64} {d : DType}
    (x : Tensor { device := device, shape := #[b, m, n], dtype := d }) : Tensor { device := device, shape := #[b, n, m], dtype := d } := x.transpose 1 2

def mulSame {s : Shape} (x y : Tensor { device := device, shape := s }) : Tensor { device := device, shape := s } := Tensor.Promoting.mul x y
def addSame {s : Shape} (x y : Tensor { device := device, shape := s }) : Tensor { device := device, shape := s } := Tensor.Promoting.add x y

def transposeAttention {b s h d : UInt64} {dt : DType}
    (x : Tensor { device := device, shape := #[b, s, h, d], dtype := dt }) : Tensor { device := device, shape := #[b, h, s, d], dtype := dt } := x.transpose 1 2

def sliceTyped {s : Shape} {d : DType} (x : Tensor { device := device, shape := s, dtype := d })
    (dim : Nat) (start len : UInt64) : Tensor { device := device, shape := (s.set! dim len), dtype := d } :=
  x.sliceChecked dim start len (s.set! dim len)

def randnDynamic (shape : Shape) (device : Device := .CPU) : IO (Tensor { device := device, shape := shape }) := do
  return (← Tensor.randn (device := device) shape).forgetDType

def hasGradFn {σ : StaticSpec} (x : Tensor σ) : Bool :=
  RawTensor.lagunaHasGradFn σ.shape x.raw

end torch.laguna
