/-
  Tests/TestDiffEqSdeAdjoint.lean

  Gradient correctness for the Stratonovich SDE backsolve adjoint.

  The linear Stratonovich SDE  dy = a·y dt + b·y ∘dW  has the exact
  solution y(T) = y₀·exp(a·T + b·W(T)) — ordinary chain rule, which is the
  point of Stratonovich calculus — so all gradients are known analytically
  for the realized path:

      ∂y(T)/∂y₀ = exp(a·T + b·W_T)
      ∂y(T)/∂a  = T · y(T)        (drift parameter)
      ∂y(T)/∂b  = W_T · y(T)      (diffusion parameter)

  Both parameter paths (args feeding the drift, args feeding the
  diffusion) exercise their respective VJP contractions in the backward
  augmented solve.
-/
import LeanTest
import Tyr.DiffEq
import Tyr.DiffEq.Adjoint.Torch

open torch
open torch.DiffEq

namespace Tests.DiffEqSdeAdjoint

private def approxRel (a b tol : Float) : Bool :=
  Float.abs (a - b) <= tol * (Float.abs b + 1.0e-12)

private def mkPath (seed : UInt64) :
    (AbstractPath (SpaceTimeLevyArea Time Float) × Float) :=
  let bm : VirtualBrownianTree Float := {
    t0 := 0.0, t1 := 1.0, tol := 1.0e-6, seed := seed, shape := 0.0
  }
  let path := (VirtualBrownianTree.toAbstractSpaceTime bm).toPath
  -- W must come from the same (space-time) sampler the diffusion control
  -- uses; the plain-increment core derives W with a different key fold.
  let wT := (VirtualBrownianTree.incrementSpaceTime bm 0.0 1.0).W
  (path, wT)

@[test] def testSdeBacksolveDriftParam : IO Unit := do
  let y0v := 0.8
  let aV := 0.4
  let bV := 0.3
  let (bmPath, wT) := mkPath 2024
  -- args drive the drift; the diffusion coefficient is baked in
  let drift : ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 }) :=
    { vectorField := fun _t y a => Tensor.mul a y }
  let diffusion : ControlTerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 }) (SpaceTimeLevyArea Time Float) (Tensor { shape := #[], dtype := .Float32 }) :=
    ControlTerm.ofPath
      (fun _t y _ => Tensor.mulScalar y bV)
      bmPath
      (fun vf c => Tensor.mulScalar vf c.W)
  let yTv := y0v * Float.exp (aV + bV * wT)
  let adj :=
    sdeBacksolveAdjointStratonovich drift diffusion 0.0 1.0
      (Tensor.full #[] yTv) (Tensor.full #[] aV) (Tensor.ones #[]) (steps := 512)
  let dy0 := Tensor.item adj.adjY0
  let da := Tensor.item adj.adjArgs
  LeanTest.assertTrue (approxRel dy0 (Float.exp (aV + bV * wT)) 2.0e-2)
    s!"∂y(T)/∂y₀ expected {Float.exp (aV + bV * wT)}, got {dy0}"
  LeanTest.assertTrue (approxRel da yTv 2.0e-2)
    s!"∂y(T)/∂a expected {yTv}, got {da}"

@[test] def testSdeBacksolveDiffusionParam : IO Unit := do
  let y0v := 0.8
  let aV := 0.4
  let bV := 0.3
  let (bmPath, wT) := mkPath 909
  -- args drive the diffusion; the drift coefficient is baked in
  let drift : ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 }) :=
    { vectorField := fun _t y _ => Tensor.mulScalar y aV }
  let diffusion : ControlTerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 }) (SpaceTimeLevyArea Time Float) (Tensor { shape := #[], dtype := .Float32 }) :=
    ControlTerm.ofPath
      (fun _t y b => Tensor.mul b y)
      bmPath
      (fun vf c => Tensor.mulScalar vf c.W)
  let yTv := y0v * Float.exp (aV + bV * wT)
  let adj :=
    sdeBacksolveAdjointStratonovich drift diffusion 0.0 1.0
      (Tensor.full #[] yTv) (Tensor.full #[] bV) (Tensor.ones #[]) (steps := 512)
  let dy0 := Tensor.item adj.adjY0
  let db := Tensor.item adj.adjArgs
  LeanTest.assertTrue (approxRel dy0 (Float.exp (aV + bV * wT)) 2.0e-2)
    s!"∂y(T)/∂y₀ expected {Float.exp (aV + bV * wT)}, got {dy0}"
  LeanTest.assertTrue (approxRel db (wT * yTv) 5.0e-2)
    s!"∂y(T)/∂b expected {wT * yTv}, got {db}"

end Tests.DiffEqSdeAdjoint
