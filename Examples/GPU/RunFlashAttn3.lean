/- End-to-end FlashAttention3 validation. -/
import Tyr.Torch
import Tyr.Typed.Neural
import Examples.GPU.AttentionReference
import Tyr.GPU.Kernels.FlashAttn3
import Examples.GPU.FixtureRunner

namespace Examples.GPU

open torch

variable {device : Device}
open Tyr.GPU.Kernels.FlashAttn3

def fa3FixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/flashattn3_256x64"⟩
  names := #["q", "k", "v", "expected_o"]
}

def fa3FixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fa3FixtureSpec name

def generateFa3Fixtures : IO Unit := do
  if !(← torch.cuda_is_available) then
    throw <| IO.userError "CUDA is not available; cannot generate flash attention fixtures."

  IO.FS.createDirAll fa3FixtureSpec.dir
  let device ← Device.cuda 0

  let qf ← Tensor.randn #[1, 1, 256, 64] false device
  let kf ← Tensor.randn #[1, 1, 256, 64] false device
  let vf ← Tensor.randn #[1, 1, 256, 64] false device

  let q := qf.toBFloat16
  let k := kf.toBFloat16
  let v := vf.toBFloat16
  let (expectedOut, _) := attentionReference q k v

  Tensor.saveTensor q (fa3FixtureFile "q").toString
  Tensor.saveTensor k (fa3FixtureFile "k").toString
  Tensor.saveTensor v (fa3FixtureFile "v").toString
  Tensor.saveTensor expectedOut (fa3FixtureFile "expected_o").toString

  let outMean := Tensor.item (Tensor.meanAll expectedOut)
  IO.println s!"Generated FA3 fixtures in {fa3FixtureSpec.dir} outMean={outMean}"

def runFa3Once : IO Bool := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return false

  let device ← Device.cuda 0
  if !(← fixturesPresent fa3FixtureSpec) then
    generateFa3Fixtures

  let q ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 256, 64], dtype := .BFloat16 } (fa3FixtureFile "q").toString
  let k ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 256, 64], dtype := .BFloat16 } (fa3FixtureFile "k").toString
  let v ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 256, 64], dtype := .BFloat16 } (fa3FixtureFile "v").toString
  let expectedOut ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 256, 64], dtype := .BFloat16 } (fa3FixtureFile "expected_o").toString

  let outFwd ← Tensor.uniformLike q 0 0
  let lseOut ← Tensor.uniformLike (Tensor.zeros #[1, 1, 256] false device) 0 0
  let stream ← torch.cuda_current_stream

  -- FA3 kernel expect seqLenQ, seqLenK, headDim as KVal
  -- flashAttn3Fwd (Q_ptr K_ptr V_ptr O_ptr L_ptr seqLenQ seqLenK headDim)
  -- The grid size needs to match CTAs. blockM=64, so 256/64 = 4 CTAs.
  flashAttn3Fwd.launch q.raw k.raw v.raw outFwd.raw lseOut.raw 256 256 64 1 4 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let outOk := Tensor.allclose expectedOut outFwd 3e-2 3e-2
  let outMae := Tensor.item (Tensor.meanAll (Tensor.abs (outFwd - expectedOut)))

  IO.println s!"flashattn3 fwd_ok={outOk} out_mae={outMae}"
  pure outOk

def main (args : List String) : IO UInt32 := do
  runWithFixtures args "flashattn3" fa3FixtureSpec generateFa3Fixtures runFa3Once

end Examples.GPU

def main : List String → IO UInt32 := Examples.GPU.main
