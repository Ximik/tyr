/-
  Examples/GPU/RunBrownianDescent.lean

  CPU↔GPU parity for the device virtual-Brownian-tree increment: the
  kernel's per-element descent must reproduce the CPU tensor instance
  (`VirtualBrownianTreeOps (Tensor { shape := s, dtype := .Float32, device := .CPU })`) — identical key derivation, identical
  bridge arithmetic — within Float32 descent tolerance. Also reports the
  device-vs-CPU sampling wall-clock (the CPU path measured ~4.6µs per
  element-increment).
-/
import Tyr.Torch
import Tyr.DiffEq
import Tyr.GPU.Ops.BrownianFused
import Examples.GPU.IndexedParity

namespace Examples.GPU.RunBrownianDescent

open torch
open torch.DiffEq
open Tyr.GPU.Ops.BrownianFused

def suiteName : String := "brownian_descent"

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  seedFixtures suiteName 161803
  let device ← Device.cuda 0
  let n : Nat := 16384
  let treeSeed : UInt64 := 4242
  let tol := 1.0e-4
  let queries : List (Float × Float) := [(0.0, 1.0), (0.13, 0.61), (0.25, 0.5)]

  -- CPU reference: the tensor VBT instance (same keying)
  let cpuTree : VirtualBrownianTree (Tensor { shape := #[n], dtype := .Float32, device := .CPU }) := {
    t0 := 0.0, t1 := 1.0, tol := tol, seed := treeSeed
    shape := Tensor.zeros #[n]
  }

  let blackwell ← isBlackwellFamily
  let stream ← torch.cuda_current_stream
  let mut ok := true

  for (tA, tB) in queries do
    let tCpu0 ← IO.monoMsNow
    let cpuInc := VirtualBrownianTree.increment cpuTree tA tB
    let cpuVals ← Tensor.toFloatArray cpuInc.W
    let cpuMs := (← IO.monoMsNow) - tCpu0

    let out ← Tensor.uniformLike (Tensor.zeros #[n] false device)
    let tGpu0 ← IO.monoMsNow
    vbtIncrementDevice treeSeed 0.0 1.0 tol 24 tA tB out n blackwell stream device
    let _ ← torch.cuda_synchronize
    let gpuMs := (← IO.monoMsNow) - tGpu0
    let gpuVals ← Tensor.toFloatArray out

    let mut maxErr := 0.0
    for i in [:n] do
      let err := Float.abs (gpuVals.getD i 0.0 - cpuVals.getD i 0.0)
      maxErr := max maxErr err
    IO.eprintln s!"[{suiteName}] [{tA}, {tB}] n={n}: cpu {cpuMs}ms, gpu {gpuMs}ms, max|Δ| = {maxErr}"
    if maxErr > 1.0e-4 then
      ok := false
      for i in [:4] do
        IO.eprintln s!"  element {i}: gpu {gpuVals.getD i 0.0}, cpu {cpuVals.getD i 0.0}"
  pure ok

def main (_ : List String) : IO UInt32 := do
  let ok ← runOnce
  pure (if ok then 0 else 1)

end Examples.GPU.RunBrownianDescent
