import Tyr.DiffEq.Adjoint.Core

namespace torch
namespace DiffEq

/-! ## Torch Adjoint Backend -/

open _root_.torch

/-- Differentiable scalar pairing of two tensor trees: the tree dot product as
    a Float32 scalar tensor, kept in the autograd graph. -/
private def dotStruct [TensorStruct Y] [FloatingTensorStruct Y] (x y : Y) : Tensor { shape := #[], dtype := .Float32, device := .CPU } :=
  let prod := FloatingTensorStruct.zipWith Tensor.mul x y
  TensorStruct.fold (fun t acc => Tensor.add acc ((Tensor.toFloat32 (Tensor.sumAll t)).to .CPU))
    (Tensor.zeros #[]) prod

private def vjp
    [TensorStruct Y] [FloatingTensorStruct Y] [TensorStruct Args]
    (vf : Time → Y → Args → Y)
    (t : Time) (y : Y) (args : Args) (adjY : Y) : (Y × Y × Args) :=
  let adjDet := TensorStruct.detach adjY
  let (gradY, gradArgs) := TensorStruct.functionalGrad
    (fun (y, args) => dotStruct (vf t y args) adjDet) (y, args)
  -- The primal is evaluated with detached inputs, so no private differentiation
  -- variables can escape through static fields or custom TensorStruct instances.
  let fDet := TensorStruct.detach (vf t (TensorStruct.detach y) (TensorStruct.detach args))
  (fDet, gradY, gradArgs)

instance [TensorStruct Y] [FloatingTensorStruct Y] [TensorStruct Args] : AdjointBackend Y Args where
  vjp := vjp

private def vjpFn
    [TensorStruct Y] [FloatingTensorStruct Y] [TensorStruct Args]
    (f : Y → Args → Y)
    (y : Y) (args : Args) (adjY : Y) : (Y × Args) :=
  let adjDet := TensorStruct.detach adjY
  TensorStruct.functionalGrad (fun (y, args) => dotStruct (f y args) adjDet) (y, args)

instance [TensorStruct Y] [FloatingTensorStruct Y] [TensorStruct Args] : AdjointFnBackend Y Args where
  vjpFn := vjpFn

end DiffEq
end torch
