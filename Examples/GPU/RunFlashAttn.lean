/- End-to-end FlashAttention validation for 2-block forward + forward-with-LSE. -/
import Tyr.Torch
import Tyr.Typed.Neural
import Examples.GPU.AttentionReference
import Tyr.GPU.Kernels.MhaH100
import Examples.GPU.IndexedParity
import Examples.GPU.FixtureRunner

namespace Examples.GPU.RunFlashAttn

open torch

variable {device : Device}
open Tyr.GPU.Kernels

def suiteName : String := "flashattn"

def fixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/flashattn128x64"⟩
  names := #["q", "k", "v", "expected_o", "expected_lse"]
}

def fixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fixtureSpec name

def generateFixtures : IO Unit := do
  if !(← requireCuda suiteName) then
    throw <| IO.userError "CUDA is not available; cannot generate flash attention fixtures."

  IO.FS.createDirAll fixtureSpec.dir
  let device ← Device.cuda 0

  let qf ← Tensor.randn #[1, 1, 128, 64] false device
  let kf ← Tensor.randn #[1, 1, 128, 64] false device
  let vf ← Tensor.randn #[1, 1, 128, 64] false device

  let q := qf.toBFloat16
  let k := kf.toBFloat16
  let v := vf.toBFloat16
  let (expectedOut, lse) := attentionReference q k v
  let expectedLse := lse.reshape #[2, 64]

  Tensor.saveTensor q (fixtureFile "q").toString
  Tensor.saveTensor k (fixtureFile "k").toString
  Tensor.saveTensor v (fixtureFile "v").toString
  Tensor.saveTensor expectedOut (fixtureFile "expected_o").toString
  Tensor.saveTensor expectedLse (fixtureFile "expected_lse").toString

  let outMean := Tensor.item (Tensor.meanAll expectedOut)
  let lseMean := Tensor.item (Tensor.meanAll expectedLse)
  IO.println s!"Generated flash attention fixtures in {fixtureSpec.dir} outMean={outMean} lseMean={lseMean}"

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  let device ← Device.cuda 0
  if !(← fixturesPresent fixtureSpec) then
    generateFixtures

  let q ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "q").toString
  let k ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "k").toString
  let v ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "v").toString
  let expectedOut ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "expected_o").toString
  let expectedLse ← Tensor.loadTensorExact { device := device, shape := #[2, 64], dtype := .Float32 } (fixtureFile "expected_lse").toString

  let outFwd ← Tensor.uniformLike q 0 0
  let outFwdLse ← Tensor.uniformLike q 0 0
  let lseOut ← Tensor.uniformLike (Tensor.zeros #[2, 64] false device) 0 0
  let stream ← torch.cuda_current_stream

  -- grid=(1,2,1): one CTA per 64-row query tile.
  tkFlashAttnFwd2Block.launch q.raw k.raw v.raw outFwd.raw 128 64 1 2 1 128 1 1 0 stream
  tkFlashAttnFwd2BlockLse.launch q.raw k.raw v.raw outFwdLse.raw lseOut.raw 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let outCheck := IndexedParity.compareTensors "flashattn.forward" expectedOut outFwd 3e-2 3e-2
  let outLseCheck := IndexedParity.compareTensors "flashattn.forward_lse" expectedOut outFwdLse 3e-2 3e-2
  let lseCheck := IndexedParity.compareTensors "flashattn.lse" expectedLse lseOut 3e-2 3e-2
  logTensorCheck outCheck
  logTensorCheck outLseCheck
  logTensorCheck lseCheck
  pure (outCheck.ok && outLseCheck.ok && lseCheck.ok)

def main (args : List String) : IO UInt32 := do
  runWithFixtures args suiteName fixtureSpec generateFixtures runOnce

end Examples.GPU.RunFlashAttn
