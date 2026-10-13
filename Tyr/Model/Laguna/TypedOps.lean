import Tyr.Model.Qwen35.TypedOps
import Tyr.RawTensor.Laguna

namespace torch.laguna

private def checked {σ : StaticSpec} (raw : RawTensor) : Tensor σ :=
  match Tensor.ofRaw raw with
  | .ok tensor => tensor
  | .error message => panic! s!"Laguna tensor boundary: {message}"

def mulBroadcast {s t out : Shape} (a : STensor s) (b : STensor t) : STensor out :=
  checked (RawTensor.mul s a.raw b.raw)

def bmm {b m n k : UInt64} (x : STensor #[b, m, k]) (y : STensor #[b, k, n]) :
    STensor #[b, m, n] :=
  checked (RawTensor.lagunaBmm b m n k x.raw y.raw)

def scatterAdd {s t : Shape} (input : STensor s) (dim : Int64)
    (indices : DTensor t .Int64) (src : STensor t) : STensor s :=
  input.checkedReplacement (RawTensor.lagunaScatterAdd s input.raw dim indices.raw src.raw)

def transposeMatrixBatch {b m n : UInt64} {d : DType}
    (x : DTensor #[b, m, n] d) : DTensor #[b, n, m] d := x.transpose 1 2

def mulSame {s : Shape} (x y : STensor s) : STensor s := x * y
def addSame {s : Shape} (x y : STensor s) : STensor s := x + y

def transposeAttention {b s h d : UInt64} {dt : DType}
    (x : DTensor #[b, s, h, d] dt) : DTensor #[b, h, s, d] dt := x.transpose 1 2

def sliceTyped {s : Shape} {d : DType} (x : DTensor s d)
    (dim : Nat) (start len : UInt64) : DTensor (s.set! dim len) d :=
  x.sliceChecked dim start len (s.set! dim len)

def randnDynamic (shape : Shape) : IO (STensor shape) := do
  return (← Tensor.randn shape).forgetDType

def hasGradFn {σ : StaticSpec} (x : Tensor σ) : Bool :=
  RawTensor.lagunaHasGradFn σ.shape x.raw

end torch.laguna
