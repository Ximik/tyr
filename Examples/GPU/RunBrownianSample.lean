/-
  Examples/GPU/RunBrownianSample.lean

  CPU↔GPU parity for the keyed Gaussian sampling kernel: every element's
  draw must match `PRNGKey.normal01 (PRNGKey.foldIn root i) 0` computed on
  the CPU (within Float32 evaluation of log/cos/sqrt).
-/
import Tyr.Torch
import Tyr.PRNG
import Tyr.GPU.Kernels.BrownianSample
import Examples.GPU.Parity

namespace Examples.GPU.RunBrownianSample

open torch
open Tyr.GPU.Kernels

def suiteName : String := "brownian_sample"

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  seedFixtures suiteName 271828
  let device ← Device.cuda 0
  let n : Nat := 4096
  let rootKey := PRNGKey.foldIn (PRNGKey.fromUInt64 99) 0x56544152
  let out ← Tensor.uniformLike (Tensor.zeros #[n] false device) 0 0
  let stream ← torch.cuda_current_stream
  let nArg ← Tyr.GPU.Codegen.launchArg n "n"

  let blackwell ← isBlackwellFamily
  if blackwell then
    keyedNormalBlackwell.launch out.raw rootKey.state nArg 32 1 1 128 1 1 0 stream
  else
    keyedNormal.launch out.raw rootKey.state nArg 32 1 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let got ← out.toFloatArray
  let mut ok := true
  let mut maxErr := 0.0
  for i in [:n] do
    let want := PRNGKey.normal01 (PRNGKey.foldIn rootKey (UInt32.ofNat i)) 0
    let err := Float.abs (got.getD i 0.0 - want)
    maxErr := max maxErr err
    if err > 1.0e-4 then
      if ok then
        IO.eprintln s!"[{suiteName}] element {i}: got {got.getD i 0.0}, want {want}"
      ok := false
  IO.eprintln s!"[{suiteName}] n={n} max |gpu - cpu| = {maxErr}"
  pure ok

def main (_ : List String) : IO UInt32 := do
  let ok ← runOnce
  pure (if ok then 0 else 1)

end Examples.GPU.RunBrownianSample
