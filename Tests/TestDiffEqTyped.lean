/-
  Tests/TestDiffEqTyped.lean

  Solving differential equations with the spec-graded typed tensor as the
  solver state: the phantom shape/dtype/device claims ride through the
  whole solve and still validate against the runtime at the end.
-/
import LeanTest
import Tyr.DiffEq

open torch
open torch.DiffEq

namespace Tests.DiffEqTyped

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private abbrev YSpec : StaticSpec := { shape := #[2], dtype := .Float32, device := .CPU }

@[test] def testTypedTensorOdeSolve : IO Unit := do
  -- dy/dt = -y, y(0) = (1, 2)  ⇒  y(1) = e⁻¹ · (1, 2)
  let term : ODETerm (Tensor YSpec) Unit :=
    { vectorField := fun _t y _ => DiffEqSpace.scale (-1.0) y }
  let solver :=
    Heun.solver
      (Term := ODETerm (Tensor YSpec) Unit)
      (Y := Tensor YSpec) (VF := Tensor YSpec) (Args := Unit)
  let y0 : Tensor YSpec := Tensor.full #[2] 1.0
  let y0 := DiffEqSpace.add y0 (Tensor.full #[2] 0.0)
  let sol :=
    diffeqsolve
      (Term := ODETerm (Tensor YSpec) Unit)
      (Y := Tensor YSpec) (VF := Tensor YSpec) (Control := Time) (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.01) y0 () (saveat := { t1 := true })
  LeanTest.assertTrue sol.result.isOkay s!"typed solve failed: {sol.result.message}"
  match sol.ys with
  | none => LeanTest.fail "expected saved final state"
  | some ys => do
      let yT := ys.getD (ys.size - 1) y0
      -- the phantom claims survive the solve
      match yT.validate with
      | .ok () => pure ()
      | .error err => LeanTest.fail s!"typed state failed validation after solve: {err}"
      let vals ← Tensor.toFloatArray yT
      LeanTest.assertEqual vals.size 2 "state keeps its shape"
      let expected := Float.exp (-1.0)
      for v in vals do
        LeanTest.assertTrue (approx v expected 1.0e-3)
          s!"y(1) expected ≈ {expected}, got {v}"

@[test] def testTypedTensorAdaptiveSolve : IO Unit := do
  -- the adaptive controller exercises DiffEqElem/DiffEqSeminorm on the
  -- typed state (error norms, elementwise tolerance scaling)
  let term : ODETerm (Tensor YSpec) Unit :=
    { vectorField := fun _t y _ => DiffEqSpace.scale (-1.0) y }
  let solver :=
    Dopri5.solver
      (Term := ODETerm (Tensor YSpec) Unit)
      (Y := Tensor YSpec) (VF := Tensor YSpec) (Args := Unit)
  let y0 : Tensor YSpec := Tensor.full #[2] 1.0
  let controller : PIDController := { rtol := 1.0e-6, atol := 1.0e-8 }
  let sol :=
    diffeqsolve
      (Term := ODETerm (Tensor YSpec) Unit)
      (Y := Tensor YSpec) (VF := Tensor YSpec) (Control := Time) (Args := Unit)
      (Controller := PIDController)
      term solver 0.0 1.0 (some 0.1) y0 () (saveat := { t1 := true })
      (controller := controller)
  LeanTest.assertTrue sol.result.isOkay s!"adaptive typed solve failed: {sol.result.message}"
  match sol.ys with
  | none => LeanTest.fail "expected saved final state"
  | some ys => do
      let yT := ys.getD (ys.size - 1) y0
      match yT.validate with
      | .ok () => pure ()
      | .error err => LeanTest.fail s!"typed state failed validation: {err}"
      let vals ← Tensor.toFloatArray yT
      for v in vals do
        LeanTest.assertTrue (approx v (Float.exp (-1.0)) 1.0e-4)
          s!"adaptive y(1) expected ≈ {Float.exp (-1.0)}, got {v}"


@[test] def testSolverGlueTimingProbe : IO Unit := do
  -- Informational: per-step cost of Dopri5 on tensor vs Float states at
  -- fixed step count quantifies the FFI/launch glue a fused kernel removes.
  let steps := 500
  let dt := 1.0 / steps.toFloat
  -- tensor state, n = 64
  let tTerm : ODETerm (Tensor { shape := #[64], dtype := .Float32, device := .CPU }) Unit :=
    { vectorField := fun _t y _ => Tensor.mulScalar y (-1.0) }
  let tSolver :=
    Dopri5.solver (Term := ODETerm (Tensor { shape := #[64], dtype := .Float32, device := .CPU }) Unit)
      (Y := Tensor { shape := #[64], dtype := .Float32, device := .CPU }) (VF := Tensor { shape := #[64], dtype := .Float32, device := .CPU }) (Args := Unit)
  let t0 ← IO.monoMsNow
  let tSol :=
    diffeqsolve (Term := ODETerm (Tensor { shape := #[64], dtype := .Float32, device := .CPU }) Unit)
      (Y := Tensor { shape := #[64], dtype := .Float32, device := .CPU }) (VF := Tensor { shape := #[64], dtype := .Float32, device := .CPU }) (Control := Time) (Args := Unit)
      (Controller := ConstantStepSize)
      tTerm tSolver 0.0 1.0 (some dt) (Tensor.ones #[64]) () (saveat := { t1 := true })
  let tFinal := Tensor.item (Tensor.sumAll ((tSol.ys.getD #[]).getD 0 (Tensor.ones #[64])))
  let tensorMs := (← IO.monoMsNow) - t0
  -- Float state (pure Lean, no FFI)
  let fTerm : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let fSolver :=
    Dopri5.solver (Term := ODETerm Float Unit)
      (Y := Float) (VF := Float) (Args := Unit)
  let t1 ← IO.monoMsNow
  let fSol :=
    diffeqsolve (Term := ODETerm Float Unit)
      (Y := Float) (VF := Float) (Control := Time) (Args := Unit)
      (Controller := ConstantStepSize)
      fTerm fSolver 0.0 1.0 (some dt) (1.0 : Float) () (saveat := { t1 := true })
  let fFinal := (fSol.ys.getD #[]).getD 0 0.0
  let floatMs := (← IO.monoMsNow) - t1
  IO.eprintln s!"  [glue-timing] dopri5 {steps} steps: tensor[64] {tensorMs}ms ({(tensorMs.toFloat * 1000.0 / steps.toFloat)}us/step), Float {floatMs}ms (finals {tFinal}, {fFinal})"

end Tests.DiffEqTyped
