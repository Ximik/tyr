import Tyr.Model.Qwen35.TypedOps
import Tyr.RawTensor.Laguna

namespace torch.laguna

private def checked {σ : StaticSpec} (raw : RawTensor) : Tensor σ :=
  match Tensor.ofRaw raw with
  | .ok tensor => tensor
  | .error message => panic! s!"Laguna tensor boundary: {message}"

def mulBroadcast {s t out : Shape} (a : Tensor { shape := s }) (b : Tensor { shape := t }) : Tensor { shape := out } :=
  checked (RawTensor.mul s a.raw b.raw)

def bmm {b m n k : UInt64} (x : Tensor { shape := #[b, m, k] }) (y : Tensor { shape := #[b, k, n] }) :
    Tensor { shape := #[b, m, n] } :=
  checked (RawTensor.lagunaBmm b m n k x.raw y.raw)

def scatterAdd {s t : Shape} (input : Tensor { shape := s }) (dim : Int64)
    (indices : Tensor { shape := t, dtype := .Int64 }) (src : Tensor { shape := t }) : Tensor { shape := s } :=
  input.checkedReplacement (RawTensor.lagunaScatterAdd s input.raw dim indices.raw src.raw)

def transposeMatrixBatch {b m n : UInt64} {d : DType}
    (x : Tensor { shape := #[b, m, n], dtype := d }) : Tensor { shape := #[b, n, m], dtype := d } := x.transpose 1 2

def mulSame {s : Shape} (x y : Tensor { shape := s }) : Tensor { shape := s } := x * y
def addSame {s : Shape} (x y : Tensor { shape := s }) : Tensor { shape := s } := x + y

def transposeAttention {b s h d : UInt64} {dt : DType}
    (x : Tensor { shape := #[b, s, h, d], dtype := dt }) : Tensor { shape := #[b, h, s, d], dtype := dt } := x.transpose 1 2

def sliceTyped {s : Shape} {d : DType} (x : Tensor { shape := s, dtype := d })
    (dim : Nat) (start len : UInt64) : Tensor { shape := (s.set! dim len), dtype := d } :=
  x.sliceChecked dim start len (s.set! dim len)

def randnDynamic (shape : Shape) : IO (Tensor { shape := shape }) := do
  return (← Tensor.randn shape).forgetDType

def hasGradFn {σ : StaticSpec} (x : Tensor σ) : Bool :=
  RawTensor.lagunaHasGradFn σ.shape x.raw

end torch.laguna
