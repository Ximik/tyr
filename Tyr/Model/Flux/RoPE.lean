import Tyr.Model.Flux.TypedOps

namespace torch.flux

variable {device : Device} {dtype : DType}
private def axisIds {batch seq : UInt64} (ids : Tensor { device := device, shape := #[batch, seq, 4], dtype := .Int64 }) (axis : UInt64) : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 } :=
  (sliceTyped ids 2 axis 1).reshapeChecked #[batch, seq]

private def ropeAxis {batch seq : UInt64}
    (ids : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) (dim theta : UInt64)
    : Tensor { device := device, shape := #[batch, seq, dim / 2, 2, 2], dtype := .Float32 } :=
  let scale := Tensor.Preserving.mulScalar ((Tensor.arange (dim / 2)).toFloat32.to ids.device) (2.0 / dim.toFloat)
  let omega := Tensor.Preserving.exp (Tensor.Preserving.mulScalar scale (-Float.log theta.toFloat))
  let pos := Tensor.expand (ids.toFloat32.reshapeChecked #[batch, seq, 1]) #[batch, seq, dim / 2]
  let omega := Tensor.expand (omega.reshapeChecked #[1, 1, dim / 2]) #[batch, seq, dim / 2]
  let angle := pos * omega
  let cos := Tensor.cos angle
  let sin := Tensor.Preserving.sin angle
  let cos := cos.reshapeChecked #[batch, seq, dim / 2, 1]
  let sin := sin.reshapeChecked #[batch, seq, dim / 2, 1]
  let row0 : Tensor { device := device, shape := #[batch, seq, dim / 2, 2], dtype := .Float32 } := qwen35.typedCat cos (Tensor.Preserving.mulScalar sin (-1.0)) 3
  let row1 : Tensor { device := device, shape := #[batch, seq, dim / 2, 2], dtype := .Float32 } := qwen35.typedCat sin cos 3
  let both : Tensor { device := device, shape := #[batch, seq, dim / 2, 4], dtype := .Float32 } := qwen35.typedCat row0 row1 3
  both.reshapeChecked #[batch, seq, dim / 2, 2, 2]

/-- Four-axis rotary matrices. Each axis dimension is even and their sum is the head dimension. -/
def ropeEmbed {batch seq head_dim : UInt64}
    (ids : Tensor { device := device, shape := #[batch, seq, 4], dtype := .Int64 }) (axes_dims : Array UInt64) (theta : UInt64 := 2000)
    : Tensor { device := device, shape := #[batch, 1, seq, head_dim / 2, 2, 2], dtype := .Float32 } :=
  if axes_dims.size != 4 || axes_dims.any (fun d => d % 2 != 0) ||
      axes_dims.foldl (fun n d => n + d.toNat) 0 != head_dim.toNat || theta == 0 then
    nativeInvariantFailure "Flux RoPE requires four even axis dimensions summing to head_dim and a positive theta"
  else
    let d0 := axes_dims[0]!
    let d1 := axes_dims[1]!
    let d2 := axes_dims[2]!
    let d3 := axes_dims[3]!
    let r0 := ropeAxis (axisIds ids 0) d0 theta
    let r1 := ropeAxis (axisIds ids 1) d1 theta
    let r2 := ropeAxis (axisIds ids 2) d2 theta
    let r3 := ropeAxis (axisIds ids 3) d3 theta
    let joined := Tensor.catManyChecked #[r0.packShape, r1.packShape, r2.packShape, r3.packShape] 2 #[batch, seq, head_dim / 2, 2, 2]
    joined.reshapeChecked #[batch, 1, seq, head_dim / 2, 2, 2]

/-- Apply rotary matrices in Float32 and restore the activation dtype. -/
def applyRope {batch seq n_head head_dim : UInt64}
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, n_head, head_dim] })
    (pe : Tensor { device := device, shape := #[batch, 1, seq, head_dim / 2, 2, 2], dtype := .Float32 })
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, n_head, head_dim] } :=
  let xf := (transposeAttention x).toFloat32.reshapeChecked #[batch, n_head, seq, head_dim / 2, 1, 2]
  let x0 := Tensor.expand ((sliceTyped xf 5 0 1).reshapeChecked #[batch, n_head, seq, head_dim / 2, 1]) #[batch, n_head, seq, head_dim / 2, 2]
  let x1 := Tensor.expand ((sliceTyped xf 5 1 1).reshapeChecked #[batch, n_head, seq, head_dim / 2, 1]) #[batch, n_head, seq, head_dim / 2, 2]
  let p0 := Tensor.expand ((sliceTyped pe 5 0 1).reshapeChecked #[batch, 1, seq, head_dim / 2, 2]) #[batch, n_head, seq, head_dim / 2, 2]
  let p1 := Tensor.expand ((sliceTyped pe 5 1 1).reshapeChecked #[batch, 1, seq, head_dim / 2, 2]) #[batch, n_head, seq, head_dim / 2, 2]
  let out := (p0 * x0 + p1 * x1).reshapeChecked #[batch, n_head, seq, head_dim]
  Tensor.castLike x (fromAttention out)

end torch.flux
