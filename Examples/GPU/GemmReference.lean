import Tyr.Typed.Ops

namespace Examples.GPU
open torch

/-- BF16 operands with Float32 accumulation for `A Bᵀ`. -/
def gemmReference {m n k : UInt64}
    (a : DTensor #[m, k] .BFloat16) (b : DTensor #[n, k] .BFloat16) :
    DTensor #[m, n] .Float32 :=
  Tensor.mm a.toFloat32 (Tensor.transpose2d b.toFloat32)

end Examples.GPU
