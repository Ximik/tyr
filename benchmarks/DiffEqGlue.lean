import Tyr.DiffEq

/-!
Times Dopri5 at a fixed step count on a 64-element tensor state against the same
solve on a plain `Float` state, to show the per-step cost of the tensor path.
Run with `lake exe diffeq_glue_bench`.
-/

open torch
open torch.DiffEq

def main : IO Unit := do
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
  IO.println s!"dopri5 {steps} steps: tensor[64] {tensorMs}ms ({(tensorMs.toFloat * 1000.0 / steps.toFloat)}us/step), Float {floatMs}ms (finals {tFinal}, {fFinal})"
