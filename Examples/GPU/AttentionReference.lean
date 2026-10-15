import Tyr.Typed.Neural

namespace Examples.GPU
open torch

variable {device : Device}

/-- Noncausal head-dimension-64 attention with quantized inputs and Float32 reference arithmetic. -/
def attentionReference {batch heads qSeq kvSeq : UInt64}
    (q : Tensor { shape := #[batch, heads, qSeq, 64], dtype := .BFloat16, device := device })
    (k v : Tensor { shape := #[batch, heads, kvSeq, 64], dtype := .BFloat16, device := device }) := Id.run do
  let kt := (Tensor.transpose k.toFloat32 2 3).reshapeChecked #[batch, heads, 64, kvSeq]
  let scores := Tensor.bmm4d q.toFloat32 kt
  let scaled := Tensor.mulScalar scores 0.125
  let probs := Tensor.softmax scaled (-1)
  let output := (Tensor.bmm4d probs v.toFloat32).toBFloat16
  let sumExp := (Tensor.sumDim (Tensor.exp scaled) 3 false).reshapeChecked #[batch, heads, qSeq]
  let lse := Tensor.log sumExp
  return (output, lse)

/-- BF16 SDPA vector-Jacobian product, returned in Float32 for kernel comparison. -/
def attentionTrainingReference {batch heads seq : UInt64}
    (q k v cotangent : Tensor { shape := #[batch, heads, seq, 64], dtype := .BFloat16, device := device }) :
    IO (Tensor { shape := #[batch, heads, seq, 64], dtype := .BFloat16, device := device } × Tensor { shape := #[batch, heads, seq, 64], dtype := .Float32, device := device } ×
      Tensor { shape := #[batch, heads, seq, 64], dtype := .Float32, device := device } × Tensor { shape := #[batch, heads, seq, 64], dtype := .Float32, device := device }) := do
  let qRef ← Tensor.setRequiresGrad (Tensor.detach q) true
  let kRef ← Tensor.setRequiresGrad (Tensor.detach k) true
  let vRef ← Tensor.setRequiresGrad (Tensor.detach v) true
  let output := Tensor.attentionGQA qRef kRef vRef false
  Tensor.backwardLoss (Tensor.sumAll (output.toFloat32 * cotangent.toFloat32))
  pure (output, (← Tensor.gradOf qRef).toFloat32, (← Tensor.gradOf kRef).toFloat32,
    (← Tensor.gradOf vRef).toFloat32)

end Examples.GPU
