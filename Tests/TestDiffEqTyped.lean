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

end Tests.DiffEqTyped
