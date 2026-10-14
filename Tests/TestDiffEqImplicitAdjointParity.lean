/-
  Tests/TestDiffEqImplicitAdjointParity.lean

  Gradient correctness for the DIRK discrete adjoint (implicit solvers).
  The reverse sweep solves the transposed stage equations by fixed-point
  iteration using only the term's VJP; these tests validate the resulting
  gradients against central finite differences on a nonlinear problem, and
  cross-check an implicit solver's gradient against an explicit one.
-/
import LeanTest
import Tyr.DiffEq
import Tyr.DiffEq.Adjoint.Torch

open torch
open torch.DiffEq

namespace Tests.DiffEqImplicitAdjointParity

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

/-- Nonlinear scalar problem dy/dt = a·y − y³ (state-dependent Jacobian, so
    the transposed fixed-point solve is genuinely exercised). -/
private def nonlinearTerm : ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 }) :=
  { vectorField := fun _t y a => Tensor.sub (Tensor.mul a y) (Tensor.mul (Tensor.mul y y) y) }

private def solveFinal
    (solver : AbstractSolver (ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 })) (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 }) Time (Tensor { shape := #[], dtype := .Float32 }))
    (y0 a : Tensor { shape := #[], dtype := .Float32 }) (dt0 : Float) : Float :=
  let sol :=
    diffeqsolve
      (Term := ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 }))
      (Y := Tensor { shape := #[], dtype := .Float32 }) (VF := Tensor { shape := #[], dtype := .Float32 }) (Control := Time) (Args := Tensor { shape := #[], dtype := .Float32 })
      (Controller := ConstantStepSize)
      nonlinearTerm solver 0.0 1.0 (some dt0) y0 a (saveat := { t1 := true })
  match sol.ys with
  | some ys => Tensor.item (ys.getD (ys.size - 1) y0)
  | none => 0.0

private def checkAgainstFiniteDifferences
    (label : String)
    (solver : AbstractSolver (ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 })) (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 }) Time (Tensor { shape := #[], dtype := .Float32 }))
    (dt0 : Float) (tol : Float) : IO Unit := do
  let y0 := Tensor.full #[] 0.8
  let a := Tensor.full #[] 0.5
  let adjY1 := Tensor.ones #[]
  let (_, adjOpt) :=
    diffeqsolveDirectAdjoint nonlinearTerm solver 0.0 1.0 (some dt0) y0 a adjY1
      (saveat := { t1 := true })
  match adjOpt with
  | none => LeanTest.fail s!"{label}: expected direct adjoint result"
  | some adj => do
      let eps := 1.0e-4
      let fdY0 :=
        (solveFinal solver (Tensor.addScalar y0 eps) a dt0 -
         solveFinal solver (Tensor.addScalar y0 (-eps)) a dt0) / (2.0 * eps)
      let fdA :=
        (solveFinal solver y0 (Tensor.addScalar a eps) dt0 -
         solveFinal solver y0 (Tensor.addScalar a (-eps)) dt0) / (2.0 * eps)
      let adjY0 := Tensor.item adj.adjY0
      let adjA := Tensor.item adj.adjArgs
      LeanTest.assertTrue (approx adjY0 fdY0 tol)
        s!"{label}: d/dy0 expected {fdY0}, got {adjY0}"
      LeanTest.assertTrue (approx adjA fdA tol)
        s!"{label}: d/da expected {fdA}, got {adjA}"

@[test] def testImplicitEulerDirectAdjointFiniteDiff : IO Unit := do
  checkAgainstFiniteDifferences "implicitEuler"
    (ImplicitEuler.solver
      (Term := ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 }))
      (Y := Tensor { shape := #[], dtype := .Float32 }) (VF := Tensor { shape := #[], dtype := .Float32 }) (Args := Tensor { shape := #[], dtype := .Float32 }))
    0.01 1.0e-2

@[test] def testKvaerno3DirectAdjointFiniteDiff : IO Unit := do
  checkAgainstFiniteDifferences "kvaerno3"
    (Kvaerno3.solver
      (Term := ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 }))
      (Y := Tensor { shape := #[], dtype := .Float32 }) (VF := Tensor { shape := #[], dtype := .Float32 }) (Args := Tensor { shape := #[], dtype := .Float32 }))
    0.05 1.0e-2

@[test] def testKvaerno5MatchesDopri5Adjoint : IO Unit := do
  -- Cross-solver agreement: a 5th-order implicit and a 5th-order explicit
  -- method must agree on the gradients of a smooth problem.
  let y0 := Tensor.full #[] 0.8
  let a := Tensor.full #[] 0.5
  let adjY1 := Tensor.ones #[]
  let dt0 := some 0.05
  let implicitSolver :=
    Kvaerno5.solver
      (Term := ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 }))
      (Y := Tensor { shape := #[], dtype := .Float32 }) (VF := Tensor { shape := #[], dtype := .Float32 }) (Args := Tensor { shape := #[], dtype := .Float32 })
  let explicitSolver :=
    Dopri5.solver
      (Term := ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 }))
      (Y := Tensor { shape := #[], dtype := .Float32 }) (VF := Tensor { shape := #[], dtype := .Float32 }) (Args := Tensor { shape := #[], dtype := .Float32 })
  let (_, adjI) :=
    diffeqsolveDirectAdjoint nonlinearTerm implicitSolver 0.0 1.0 dt0 y0 a adjY1
      (saveat := { t1 := true })
  let (_, adjE) :=
    diffeqsolveDirectAdjoint nonlinearTerm explicitSolver 0.0 1.0 dt0 y0 a adjY1
      (saveat := { t1 := true })
  match adjI, adjE with
  | some i, some e => do
      LeanTest.assertTrue (approx (Tensor.item i.adjY0) (Tensor.item e.adjY0) 1.0e-4)
        s!"kvaerno5 d/dy0 {Tensor.item i.adjY0} vs dopri5 {Tensor.item e.adjY0}"
      LeanTest.assertTrue (approx (Tensor.item i.adjArgs) (Tensor.item e.adjArgs) 1.0e-4)
        s!"kvaerno5 d/da {Tensor.item i.adjArgs} vs dopri5 {Tensor.item e.adjArgs}"
  | _, _ => LeanTest.fail "expected adjoint results from both solvers"

/-! ## IMEX (KenCarp) adjoints -/

private def imexExplicitTerm : ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 }) :=
  { vectorField := fun _t y a => Tensor.mul a y }

private def imexImplicitTerm : ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 }) :=
  { vectorField := fun _t y _ => Tensor.mulScalar (Tensor.mul (Tensor.mul y y) y) (-1.0) }

private def imexTerms : MultiTerm (ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 })) (ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 })) :=
  { term1 := imexExplicitTerm, term2 := imexImplicitTerm }

private def imexSolveFinal
    (solver : AbstractSolver (MultiTerm (ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 })) (ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 })))
      (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 } × Tensor { shape := #[], dtype := .Float32 }) (Time × Time) (Tensor { shape := #[], dtype := .Float32 }))
    (y0 a : Tensor { shape := #[], dtype := .Float32 }) (dt0 : Float) : Float :=
  let sol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 })) (ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 })))
      (Y := Tensor { shape := #[], dtype := .Float32 }) (VF := (Tensor { shape := #[], dtype := .Float32 } × Tensor { shape := #[], dtype := .Float32 })) (Control := (Time × Time)) (Args := Tensor { shape := #[], dtype := .Float32 })
      (Controller := ConstantStepSize)
      imexTerms solver 0.0 1.0 (some dt0) y0 a (saveat := { t1 := true })
  match sol.ys with
  | some ys => Tensor.item (ys.getD (ys.size - 1) y0)
  | none => 0.0

/-- The IMEX split fᴱ = a·y, fᴵ = −y³ totals the same vector field as the
    DIRK test; gradients must match central finite differences. -/
@[test] def testKencarp4ImexAdjointFiniteDiff : IO Unit := do
  let solver :=
    Kencarp4.solver
      (ExplicitTerm := ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 }))
      (ImplicitTerm := ODETerm (Tensor { shape := #[], dtype := .Float32 }) (Tensor { shape := #[], dtype := .Float32 }))
      (Y := Tensor { shape := #[], dtype := .Float32 }) (VFe := Tensor { shape := #[], dtype := .Float32 }) (VFi := Tensor { shape := #[], dtype := .Float32 }) (Args := Tensor { shape := #[], dtype := .Float32 })
  let y0 := Tensor.full #[] 0.8
  let a := Tensor.full #[] 0.5
  let adjY1 := Tensor.ones #[]
  let dt0 := 0.05
  let (_, adjOpt) :=
    diffeqsolveDirectAdjointIMEX imexTerms solver 0.0 1.0 (some dt0) y0 a adjY1
      (saveat := { t1 := true })
  match adjOpt with
  | none => LeanTest.fail "kencarp4 IMEX: expected direct adjoint result"
  | some adj => do
      let eps := 1.0e-4
      let fdY0 :=
        (imexSolveFinal solver (Tensor.addScalar y0 eps) a dt0 -
         imexSolveFinal solver (Tensor.addScalar y0 (-eps)) a dt0) / (2.0 * eps)
      let fdA :=
        (imexSolveFinal solver y0 (Tensor.addScalar a eps) dt0 -
         imexSolveFinal solver y0 (Tensor.addScalar a (-eps)) dt0) / (2.0 * eps)
      let adjY0 := Tensor.item adj.adjY0
      let adjA := Tensor.item adj.adjArgs
      LeanTest.assertTrue (approx adjY0 fdY0 1.0e-2)
        s!"kencarp4 IMEX: d/dy0 expected {fdY0}, got {adjY0}"
      LeanTest.assertTrue (approx adjA fdA 1.0e-2)
        s!"kencarp4 IMEX: d/da expected {fdA}, got {adjA}"

end Tests.DiffEqImplicitAdjointParity
