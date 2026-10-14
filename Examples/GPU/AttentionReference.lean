import Tyr.Typed.Neural

namespace Examples.GPU
open torch

/-- Noncausal head-dimension-64 attention with quantized inputs and Float32 reference arithmetic. -/
def attentionReference {qSeq kvSeq : UInt64}
    (q : DTensor #[1, 1, qSeq, 64] .BFloat16)
    (k v : DTensor #[1, 1, kvSeq, 64] .BFloat16) := Id.run do
  let kt := (Tensor.transpose k.toFloat32 2 3).reshapeChecked #[1, 1, 64, kvSeq]
  let scores := Tensor.bmm4d q.toFloat32 kt
  let scaled := Tensor.Preserving.mulScalar scores 0.125
  let probs := Tensor.softmax scaled (-1)
  let output := (Tensor.bmm4d probs v.toFloat32).toBFloat16
  let sumExp := (Tensor.sumDim (Tensor.exp scaled) 3 false).reshapeChecked #[1, 1, qSeq]
  let lse := Tensor.log sumExp
  return (output, lse)

end Examples.GPU
