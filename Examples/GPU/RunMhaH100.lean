/- End-to-end `mha_h100`-style forward/backward validation for the reduced
   2-block (`seq=128`, `d=64`) kernel variant in this repo. -/
import Tyr.Torch
import Examples.GPU.AttentionReference
import Tyr.GPU.Kernels.MhaH100
import Examples.GPU.FixtureRunner

namespace Examples.GPU.RunMhaH100

open torch

variable {device : Device}
open Tyr.GPU.Kernels

private abbrev MhaTensor (device : Device) := Tensor { shape := #[1, 1, 128, 64], dtype := .Float32, device := device }
private abbrev LTensor (device : Device) := Tensor { shape := #[2, 64], dtype := .Float32, device := device }
private abbrev GradTiles (device : Device) := Tensor { shape := #[2, 64, 64], dtype := .Float32, device := device }

private def contractLabel : String := "dq_direct_kv_sweep_store_add"
private def seqLen : Nat := 128
private def headDim : Nat := 64
private def kvTiles : Nat := 2
def suiteName : String := "mha_h100"

def fixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/mha_h100_128x64"⟩
  names := #[
    "q", "k", "v", "dO",
    "expected_o", "expected_l",
    "expected_dq", "expected_dk", "expected_dv"
  ]
}

def fixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fixtureSpec name

private def partialDumpFile (name : String) : System.FilePath :=
  fixtureSpec.dir / name

private def dumpAccumulatedGrads (dK dV : MhaTensor (device := device)) : IO Unit := do
  let dKTiles : GradTiles (device := device) := Tensor.reshape dK #[2, 64, 64] (by simp)
  let dVTiles : GradTiles (device := device) := Tensor.reshape dV #[2, 64, 64] (by simp)
  let dKPath := partialDumpFile "diag_dK_accum_tiles.pt"
  let dVPath := partialDumpFile "diag_dV_accum_tiles.pt"
  Tensor.saveTensor dKTiles dKPath.toString
  Tensor.saveTensor dVTiles dVPath.toString
  IO.println s!"mha_h100 grad_dump=true dK_tiles={dKPath} dV_tiles={dVPath}"

def generateFixtures : IO Unit := do
  if !(← torch.cuda_is_available) then
    throw <| IO.userError "CUDA is not available; cannot generate mha_h100 fixtures."

  IO.FS.createDirAll fixtureSpec.dir
  let device ← Device.cuda 0

  let qf ← Tensor.randn #[1, 1, 128, 64] false device
  let kf ← Tensor.randn #[1, 1, 128, 64] false device
  let vf ← Tensor.randn #[1, 1, 128, 64] false device
  let dOf ← Tensor.randn #[1, 1, 128, 64] false device

  let q := qf.toBFloat16
  let k := kf.toBFloat16
  let v := vf.toBFloat16
  let dO := dOf.toBFloat16

  -- Forward reference output.
  let expectedOut := Tensor.attentionGQA q k v false

  -- ThunderKittens-style L vector: L = -8 * lse for d=64.
  let (_, lse) := attentionReference q k v
  let expectedL := (Tensor.mulScalar lse (-8.0)).reshape #[2, 64]

  -- Backward references from PyTorch autograd.
  let (_, expectedDQ, expectedDK, expectedDV) ← attentionTrainingReference q k v dO

  Tensor.saveTensor q (fixtureFile "q").toString
  Tensor.saveTensor k (fixtureFile "k").toString
  Tensor.saveTensor v (fixtureFile "v").toString
  Tensor.saveTensor dO (fixtureFile "dO").toString
  Tensor.saveTensor expectedOut (fixtureFile "expected_o").toString
  Tensor.saveTensor expectedL (fixtureFile "expected_l").toString
  Tensor.saveTensor expectedDQ (fixtureFile "expected_dq").toString
  Tensor.saveTensor expectedDK (fixtureFile "expected_dk").toString
  Tensor.saveTensor expectedDV (fixtureFile "expected_dv").toString

  let outMean := Tensor.item (Tensor.meanAll expectedOut)
  let lMean := Tensor.item (Tensor.meanAll expectedL)
  let dqMean := Tensor.item (Tensor.meanAll expectedDQ)
  IO.println s!"Generated mha_h100 fixtures in {fixtureSpec.dir} outMean={outMean} lMean={lMean} dqMean={dqMean}"

def runOnce (dumpPartials : Bool := false) : IO Bool := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return false

  let device ← Device.cuda 0
  if !(← fixturesPresent fixtureSpec) then
    generateFixtures

  let stream ← torch.cuda_current_stream

  let q ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "q").toString
  let k ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "k").toString
  let v ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "v").toString
  let dO ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "dO").toString
  let expectedOut ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .BFloat16 } (fixtureFile "expected_o").toString
  let expectedL ← Tensor.loadTensorExact { device := device, shape := #[2, 64], dtype := .Float32 } (fixtureFile "expected_l").toString
  let expectedDQ ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .Float32 } (fixtureFile "expected_dq").toString
  let expectedDK ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .Float32 } (fixtureFile "expected_dk").toString
  let expectedDV ← Tensor.loadTensorExact { device := device, shape := #[1, 1, 128, 64], dtype := .Float32 } (fixtureFile "expected_dv").toString

  -- Forward.
  -- Cross-check L against the existing LSE kernel path.
  let outLse ← Tensor.uniformLike q 0 0
  let lseOut ← Tensor.uniformLike (Tensor.zeros #[2, 64] false device) 0 0
  tkFlashAttnFwd2BlockLse.launch q.raw k.raw v.raw outLse.raw lseOut.raw 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize
  let lFromLse : LTensor (device := device) := Tensor.mulScalar lseOut (-8.0)

  let out ← Tensor.uniformLike q 0 0
  let lOut ← Tensor.uniformLike lseOut 0 0
  tkMhaH100Fwd2Block.launch q.raw k.raw v.raw out.raw lOut.raw 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize

  -- Backward prep + backward.
  -- Allocate each mutable gradient buffer independently through IO.
  let dVec ← Tensor.uniformLike lOut 0 0
  tkMhaH100BwdPrep2Block.launch dO.raw out.raw dVec.raw 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let dQ : MhaTensor (device := device) ← Tensor.uniformLike (Tensor.zeros #[1, 1, 128, 64] false device) 0 0
  let dK : MhaTensor (device := device) ← Tensor.uniformLike (Tensor.zeros #[1, 1, 128, 64] false device) 0 0
  let dV : MhaTensor (device := device) ← Tensor.uniformLike (Tensor.zeros #[1, 1, 128, 64] false device) 0 0
  tkMhaH100Bwd2BlockDq.launch q.raw k.raw v.raw dO.raw lOut.raw dVec.raw dQ.raw 128 64 1 2 1 128 1 1 0 stream
  tkMhaH100Bwd2BlockKvSweep.launch q.raw k.raw v.raw dO.raw lOut.raw dVec.raw dQ.raw dK.raw dV.raw 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize
  if dumpPartials then
    dumpAccumulatedGrads dK dV

  -- Validate.
  let outRefOk := Tensor.allclose expectedOut out 3e-2 3e-2
  let lRefOk := Tensor.allclose expectedL lOut 3e-2 3e-2
  let lVsLseOk := Tensor.allclose lFromLse lOut 3e-2 3e-2
  let lseRefOk := Tensor.allclose expectedL lFromLse 3e-2 3e-2
  let dqRefOk := Tensor.allclose expectedDQ dQ 3e-2 3e-2
  let dkRefOk := Tensor.allclose expectedDK dK 3e-2 3e-2
  let dvRefOk := Tensor.allclose expectedDV dV 3e-2 3e-2
  let kernelRefOk := outRefOk && lRefOk && dqRefOk && dkRefOk && dvRefOk
  let lRouteOk := lVsLseOk && lseRefOk
  let overallOk := kernelRefOk && lRouteOk

  let outMae := Tensor.item (Tensor.meanAll (Tensor.abs (out - expectedOut)))
  let outMaxErr := Tensor.item (Tensor.maxAll (Tensor.abs (out - expectedOut)))
  let lMae := Tensor.item (Tensor.meanAll (Tensor.abs (lOut - expectedL)))
  let lMaxErr := Tensor.item (Tensor.maxAll (Tensor.abs (lOut - expectedL)))
  let lKernelMae := Tensor.item (Tensor.meanAll (Tensor.abs (lOut - lFromLse)))
  let lFixtureMae := Tensor.item (Tensor.meanAll (Tensor.abs (lFromLse - expectedL)))
  let dqMae := Tensor.item (Tensor.meanAll (Tensor.abs (dQ - expectedDQ)))
  let dqMaxErr := Tensor.item (Tensor.maxAll (Tensor.abs (dQ - expectedDQ)))
  let dkMae := Tensor.item (Tensor.meanAll (Tensor.abs (dK - expectedDK)))
  let dkMaxErr := Tensor.item (Tensor.maxAll (Tensor.abs (dK - expectedDK)))
  let dvMae := Tensor.item (Tensor.meanAll (Tensor.abs (dV - expectedDV)))
  let dvMaxErr := Tensor.item (Tensor.maxAll (Tensor.abs (dV - expectedDV)))
  IO.println s!"mha_h100 contract={contractLabel} seq={seqLen} head_dim={headDim} kv_tiles={kvTiles} overall_ok={overallOk} kernel_ref_ok={kernelRefOk} l_route_ok={lRouteOk} out_ref_ok={outRefOk} l_ref_ok={lRefOk} l_vs_lse_ok={lVsLseOk} lse_ref_ok={lseRefOk} dq_ref_ok={dqRefOk} dk_ref_ok={dkRefOk} dv_ref_ok={dvRefOk} out_mae={outMae} out_max={outMaxErr} l_mae={lMae} l_max={lMaxErr} l_kernel_mae={lKernelMae} l_fixture_mae={lFixtureMae} dq_mae={dqMae} dq_max={dqMaxErr} dk_mae={dkMae} dk_max={dkMaxErr} dv_mae={dvMae} dv_max={dvMaxErr}"

  pure overallOk

def main (args : List String) : IO UInt32 := do
  let dumpPartials := args.contains "--dump-partials"
  runWithFixtures args suiteName fixtureSpec generateFixtures (runOnce dumpPartials)

end Examples.GPU.RunMhaH100
