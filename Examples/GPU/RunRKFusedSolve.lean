/-
  Examples/GPU/RunRKFusedSolve.lean

  End-to-end parity + benchmark for the fused Dopri5 integrator:
  - solve dy/dt = -y on a [1, 1, 64, 64] CUDA state with the generic
    (pure, ~60 tensor ops per step) Dopri5 solver and with the fused
    kernel path (~15 launches per step)
  - compare endpoints, report wall-clock for both
-/
import Tyr.Torch
import Tyr.DiffEq
import Tyr.GPU.Ops.RKFused
import Examples.GPU.IndexedParity

namespace Examples.GPU.RunRKFusedSolve

open torch
open torch.DiffEq
open Tyr.GPU.Ops.RKFused

def suiteName : String := "rk_fused_solve"

private abbrev S : Shape := #[1, 1, 64, 64]

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  seedFixtures suiteName 31415
  let device := Device.CUDA 0
  let y0 ← Tensor.uniformLike (Tensor.zeros S false device)
  let steps := 200
  let dt := 1.0 / steps.toFloat

  -- generic pure solver path
  let term : ODETerm (Tensor { shape := S, dtype := .Float32, device := (.CUDA 0) }) Unit :=
    { vectorField := fun _t y _ => Tensor.Preserving.mulScalar y (-1.0) }
  let solver :=
    Dopri5.solver (Term := ODETerm (Tensor { shape := S, dtype := .Float32, device := (.CUDA 0) }) Unit)
      (Y := Tensor { shape := S, dtype := .Float32, device := (.CUDA 0) }) (VF := Tensor { shape := S, dtype := .Float32, device := (.CUDA 0) }) (Args := Unit)
  let tg0 ← IO.monoMsNow
  let sol :=
    diffeqsolve (Term := ODETerm (Tensor { shape := S, dtype := .Float32, device := (.CUDA 0) }) Unit)
      (Y := Tensor { shape := S, dtype := .Float32, device := (.CUDA 0) }) (VF := Tensor { shape := S, dtype := .Float32, device := (.CUDA 0) }) (Control := Time) (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some dt) y0 () (saveat := { t1 := true })
  let yGeneric :=
    match sol.ys with
    | some ys => ys.getD (ys.size - 1) y0
    | none => y0
  let _ ← torch.cuda_synchronize
  let genericMs := (← IO.monoMsNow) - tg0

  -- fused kernel path
  let blackwell ← isBlackwellFamily
  let stream ← torch.cuda_current_stream
  let ws ← Workspace.make 64 device
  let tf0 ← IO.monoMsNow
  let yFused ← dopri5SolveFused (fun _t y => Tensor.Preserving.mulScalar y (-1.0))
    0.0 1.0 steps y0 ws blackwell stream
  let fusedMs := (← IO.monoMsNow) - tf0

  IO.eprintln s!"[{suiteName}] {steps} steps on [1,1,64,64]: generic {genericMs}ms, fused {fusedMs}ms"

  IndexedParity.reportTensorComparison s!"{suiteName}.endpoint" yGeneric yFused 1e-4 1e-4

def main (_ : List String) : IO UInt32 := do
  let ok ← runOnce
  pure (if ok then 0 else 1)

end Examples.GPU.RunRKFusedSolve
