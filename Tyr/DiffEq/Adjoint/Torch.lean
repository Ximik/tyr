import Tyr.DiffEq.Adjoint.Core

namespace torch
namespace DiffEq

/-! ## Torch Adjoint Backend -/

open _root_.torch

/-- Differentiable scalar pairing of two tensor trees: the tree dot product as
    a Float32 scalar tensor, kept in the autograd graph. -/
private def dotStruct [TensorStruct Y] (x y : Y) : DTensor #[] .Float32 :=
  let prod := TensorStruct.zipWith Mul.mul x y
  TensorStruct.fold (fun t acc => Add.add acc (Tensor.toFloat32 (Tensor.sumAll t)).forgetDevice)
    (Tensor.zeros #[]) prod

private def gradStruct [TensorStruct α] (loss : DTensor #[] .Float32) (x : α) : α :=
  TensorStruct.map (fun t => Tensor.grad loss t (Tensor.onesLike loss)) x

private def vjp
    [TensorStruct Y] [TensorStruct Args]
    (vf : Time → Y → Args → Y)
    (t : Time) (y : Y) (args : Args) (adjY : Y) : (Y × Y × Args) :=
  let yReq := TensorStruct.makeLeafParams y
  let argsReq := TensorStruct.makeLeafParams args
  let fReq := vf t yReq argsReq
  let adjDet := TensorStruct.detach adjY
  let loss := dotStruct fReq adjDet
  let gradY := gradStruct loss yReq
  let gradArgs := gradStruct loss argsReq
  let fDet := TensorStruct.detach fReq
  (fDet, gradY, gradArgs)

instance [TensorStruct Y] [TensorStruct Args] : AdjointBackend Y Args where
  vjp := vjp

private def vjpFn
    [TensorStruct Y] [TensorStruct Args]
    (f : Y → Args → Y)
    (y : Y) (args : Args) (adjY : Y) : (Y × Args) :=
  let yReq := TensorStruct.makeLeafParams y
  let argsReq := TensorStruct.makeLeafParams args
  let out := f yReq argsReq
  let adjDet := TensorStruct.detach adjY
  let loss := dotStruct out adjDet
  let gradY := gradStruct loss yReq
  let gradArgs := gradStruct loss argsReq
  (gradY, gradArgs)

instance [TensorStruct Y] [TensorStruct Args] : AdjointFnBackend Y Args where
  vjpFn := vjpFn

end DiffEq
end torch
