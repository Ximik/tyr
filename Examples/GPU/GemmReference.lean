import Tyr.Typed.Ops

namespace Examples.GPU
open torch

/-- BF16 operands with Float32 accumulation for `A Bᵀ`. -/
def gemmReference {m n k : UInt64}
    (a : Tensor { shape := #[m, k], dtype := .BFloat16 }) (b : Tensor { shape := #[n, k], dtype := .BFloat16 }) :
    Tensor { shape := #[m, n], dtype := .Float32 } :=
  Tensor.mm a.toFloat32 (Tensor.transpose2d b.toFloat32)

end Examples.GPU
