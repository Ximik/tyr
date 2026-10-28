import Tyr.Typed.Ops

namespace Examples.GPU
open torch

variable {device : Device}

/-- BF16 operands with Float32 accumulation for `A Bᵀ`. -/
def gemmReference {m n k : Nat}
    (a : Tensor { shape := #[m, k], dtype := .BFloat16, device := device }) (b : Tensor { shape := #[n, k], dtype := .BFloat16, device := device }) :
    Tensor { shape := #[m, n], dtype := .Float32, device := device } :=
  Tensor.mm a.toFloat32 (Tensor.transpose2d b.toFloat32)

end Examples.GPU
