/-
  Examples/GPU/RunEulerMaruyamaFused.lean

  CPU↔GPU parity for the fused Euler–Maruyama SDE solve: geometric
  Brownian motion dy = a·y dt + b·y dW over 16384 independent elements.
  The CPU reference runs the identical update with increments from the
  CPU tensor VBT instance (same seed and keying as the device descent),
  so the two paths differ only by the Float32 kernel arithmetic.
  Also reports the device-vs-CPU per-solve wall-clock.
-/
import Tyr.Torch
import Tyr.DiffEq
import Tyr.GPU.Ops.SDEFused
import Examples.GPU.IndexedParity

namespace Examples.GPU.RunEulerMaruyamaFused

open torch
open torch.DiffEq
open Tyr.GPU.Ops.SDEFused

def suiteName : String := "euler_maruyama_fused"

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  seedFixtures suiteName 271828
  let device ← Device.cuda 0
  let n : UInt64 := 16384
  let treeSeed : UInt64 := 9001
  let tol := 1.0e-4
  let steps := 32
  let t0 := 0.0
  let t1 := 1.0
  let a := -0.5   -- drift rate
  let b := 0.3    -- diffusion rate

  let drift {placement : Device} : Float → Tensor { shape := #[n], dtype := .Float32, device := placement } → Tensor { shape := #[n], dtype := .Float32, device := placement } := fun _ y => Tensor.Preserving.mulScalar y a
  let diffusion {placement : Device} : Float → Tensor { shape := #[n], dtype := .Float32, device := placement } → Tensor { shape := #[n], dtype := .Float32, device := placement } := fun _ y => Tensor.Preserving.mulScalar y b

  -- CPU reference: same EM update, increments from the CPU tensor VBT
  let cpuTree : VirtualBrownianTree (Tensor { shape := #[n], dtype := .Float32, device := .CPU }) := {
    t0 := t0, t1 := t1, tol := tol, seed := treeSeed
    shape := Tensor.zeros #[n]
  }
  let h := (t1 - t0) / Float.ofNat steps
  let y0Host := Tensor.ones #[n]

  let tCpu0 ← IO.monoMsNow
  let mut yCpu := y0Host
  for k in [:steps] do
    let tA := t0 + Float.ofNat k * h
    let tB := t0 + Float.ofNat (k + 1) * h
    let inc := VirtualBrownianTree.increment cpuTree tA tB
    yCpu := Tensor.add (Tensor.add yCpu (Tensor.Preserving.mulScalar (drift tA yCpu) h))
      (Tensor.mul (diffusion tA yCpu) inc.W)
  let cpuVals ← Tensor.toFloatArray yCpu
  let cpuMs := (← IO.monoMsNow) - tCpu0

  let blackwell ← isBlackwellFamily
  let stream ← torch.cuda_current_stream
  let y0Dev := y0Host.to device

  let tGpu0 ← IO.monoMsNow
  let yGpu ← emSolveFused drift diffusion treeSeed t0 t1 tol 24
    t0 t1 steps y0Dev blackwell stream
  let gpuMs := (← IO.monoMsNow) - tGpu0
  let gpuVals ← Tensor.toFloatArray yGpu

  let mut maxErr := 0.0
  for i in [:n.toNat] do
    let err := Float.abs (gpuVals.getD i 0.0 - cpuVals.getD i 0.0)
    maxErr := max maxErr err
  IO.eprintln s!"[{suiteName}] n={n} steps={steps}: cpu {cpuMs}ms, gpu {gpuMs}ms, max|Δ| = {maxErr}"
  let ok := maxErr <= 1.0e-3
  if !ok then
    for i in [:4] do
      IO.eprintln s!"  element {i}: gpu {gpuVals.getD i 0.0}, cpu {cpuVals.getD i 0.0}"
  pure ok

def main (_ : List String) : IO UInt32 := do
  let ok ← runOnce
  pure (if ok then 0 else 1)

end Examples.GPU.RunEulerMaruyamaFused
