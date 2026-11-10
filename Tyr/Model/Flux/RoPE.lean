import Tyr.Model.Flux.TypedOps

namespace torch.flux

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]
private def axisIds {batch seq : Nat} (ids : Tensor { device := device, shape := #[batch, seq, 4], dtype := .Int64 }) (axis : Nat) : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 } :=
  (Tensor.slice! ids 2 axis 1).reshape #[batch, seq]

private def ropeAxis {batch seq : Nat}
    (ids : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) (dim theta : Nat)
    : Tensor { device := device, shape := #[batch, seq, dim / 2, 2, 2], dtype := .Float32 } :=
  let scale := Tensor.mulScalar ((Tensor.arange (dim / 2)).toFloat32.to ids.device) (2.0 / dim.toFloat)
  let omega := Tensor.exp (Tensor.mulScalar scale (-Float.log theta.toFloat))
  let pos := Tensor.broadcastTo (ids.toFloat32.reshape #[batch, seq, 1]) #[batch, seq, dim / 2]
  let omega := Tensor.broadcastTo (omega.reshape #[1, 1, dim / 2]) #[batch, seq, dim / 2]
  let angle := pos * omega
  let cos := Tensor.cos angle
  let sin := Tensor.sin angle
  let cos := cos.reshape #[batch, seq, dim / 2, 1]
  let sin := sin.reshape #[batch, seq, dim / 2, 1]
  let negSin := Tensor.mulScalar sin (-1.0)
  let row0 : Tensor { device := device, shape := #[batch, seq, dim / 2, 2], dtype := .Float32 } := Tensor.cat cos negSin 3
  let row1 : Tensor { device := device, shape := #[batch, seq, dim / 2, 2], dtype := .Float32 } := Tensor.cat sin cos 3
  let both : Tensor { device := device, shape := #[batch, seq, dim / 2, 4], dtype := .Float32 } := Tensor.cat row0 row1 3
  both.reshape #[batch, seq, dim / 2, 2, 2] (by simp [Nat.mul_assoc])

private theorem axesHalves {head_dim : Nat} (axes : Array Nat) (hSize : axes.size = 4)
    (hEven : axes.toList.all (· % 2 == 0)) (hSum : axes.toList.sum = head_dim) :
    axes[0] / 2 + axes[1] / 2 + axes[2] / 2 + axes[3] / 2 = head_dim / 2 := by
  obtain ⟨l⟩ := axes
  rcases l with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, l⟩⟩⟩⟩⟩
  any_goals (simp at hSize <;> omega)
  simp at hEven hSum ⊢
  omega

/-- Four-axis rotary matrices. Each axis dimension is even and their sum is the head dimension. -/
def ropeEmbed {batch seq head_dim : Nat}
    (ids : Tensor { device := device, shape := #[batch, seq, 4], dtype := .Int64 }) (axes_dims : Array Nat)
    (hAxes : axes_dims.size = 4 ∧ axes_dims.toList.all (· % 2 == 0) ∧ axes_dims.toList.sum = head_dim)
    (theta : Nat := 2000) (_hTheta : 0 < theta := by decide)
    : Tensor { device := device, shape := #[batch, 1, seq, head_dim / 2, 2, 2], dtype := .Float32 } :=
  let d0 := axes_dims[0]'(by omega)
  let d1 := axes_dims[1]'(by omega)
  let d2 := axes_dims[2]'(by omega)
  let d3 := axes_dims[3]'(by omega)
  let r0 := ropeAxis (axisIds ids 0) d0 theta
  let r1 := ropeAxis (axisIds ids 1) d1 theta
  let r2 := ropeAxis (axisIds ids 2) d2 theta
  let r3 := ropeAxis (axisIds ids 3) d3 theta
  have hHalves := axesHalves axes_dims hAxes.1 hAxes.2.1 hAxes.2.2
  (Tensor.cat (Tensor.cat (Tensor.cat r0 r1 2) r2 2) r3 2).reshape #[batch, 1, seq, head_dim / 2, 2, 2]
    (by simp only [d0, d1, d2, d3] at *; simp [← hHalves, Nat.mul_assoc])

/-- Apply rotary matrices in Float32 and restore the activation dtype. -/
def applyRope {batch seq n_head head_dim : Nat}
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, n_head, head_dim] })
    (heven : 2 ∣ head_dim)
    (pe : Tensor { device := device, shape := #[batch, 1, seq, head_dim / 2, 2, 2], dtype := .Float32 })
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, n_head, head_dim] } :=
  let xf := (transposeAttention x).toFloat32.reshape #[batch, n_head, seq, head_dim / 2, 1, 2]
    (by simp [Nat.mul_assoc, Nat.div_mul_cancel heven])
  let x0 := Tensor.broadcastTo ((Tensor.slice xf 5 0 1).reshape #[batch, n_head, seq, head_dim / 2, 1]) #[batch, n_head, seq, head_dim / 2, 2]
  let x1 := Tensor.broadcastTo ((Tensor.slice xf 5 1 1).reshape #[batch, n_head, seq, head_dim / 2, 1]) #[batch, n_head, seq, head_dim / 2, 2]
  let p0 := Tensor.broadcastTo ((Tensor.slice pe 5 0 1).reshape #[batch, 1, seq, head_dim / 2, 2]) #[batch, n_head, seq, head_dim / 2, 2]
  let p1 := Tensor.broadcastTo ((Tensor.slice pe 5 1 1).reshape #[batch, 1, seq, head_dim / 2, 2]) #[batch, n_head, seq, head_dim / 2, 2]
  let out := (p0 * x0 + p1 * x1).reshape #[batch, n_head, seq, head_dim]
    (by simp [Nat.mul_assoc, Nat.div_mul_cancel heven])
  Tensor.castLike x (fromAttention out)

end torch.flux
