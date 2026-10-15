/- Runtime smoke test for the high-level `tyr::flash_attn` bridge. -/
import Tyr.Torch
import Examples.GPU.FlashBenchReference
import Tyr.GPU.Ops.FlashAttn

namespace Examples.GPU

open torch
open Tyr.GPU.Ops.FlashAttn

private def routeName : DispatchRoute → String
  | .tkKernel => "tkKernel"
  | .portable => "portable"

private def reportCase
    (name : String)
    (route : DispatchRoute)
    (routeOk outOk dqOk dkOk dvOk : Bool)
    (outMae dqMae dkMae dvMae : Float)
    (outMax dqMax dkMax dvMax : Float) : IO Unit :=
  IO.println s!"{name} route={routeName route} route_ok={routeOk} out_ok={outOk} dq_ok={dqOk} dk_ok={dkOk} dv_ok={dvOk} out_mae={outMae} dq_mae={dqMae} dk_mae={dkMae} dv_mae={dvMae} out_max={outMax} dq_max={dqMax} dk_max={dkMax} dv_max={dvMax}"

private def nativeDenseCase : IO Bool := do
  let device ← Device.cuda 0
  let q ← Tensor.setRequiresGrad ((← Tensor.randn #[1, 1, 128, 64] false device).toBFloat16) true
  let k ← Tensor.setRequiresGrad ((← Tensor.randn #[1, 1, 128, 64] false device).toBFloat16) true
  let v ← Tensor.setRequiresGrad ((← Tensor.randn #[1, 1, 128, 64] false device).toBFloat16) true
  let dO := (← Tensor.randn #[1, 1, 128, 64] false device).toBFloat16

  let (route, out) := flashAttnWithRoute q k v
  FlashBenchReference.backwardWithCotangent out dO
  let _ ← torch.cuda_synchronize
  let dq := (← Tensor.gradOf q).toFloat32
  let dk := (← Tensor.gradOf k).toFloat32
  let dv := (← Tensor.gradOf v).toFloat32

  let qRef ← Tensor.setRequiresGrad (Tensor.detach q) true
  let kRef ← Tensor.setRequiresGrad (Tensor.detach k) true
  let vRef ← Tensor.setRequiresGrad (Tensor.detach v) true
  let outRef := Tensor.attentionGQA qRef kRef vRef false
  FlashBenchReference.backwardWithCotangent outRef dO
  let _ ← torch.cuda_synchronize
  let dqRef := (← Tensor.gradOf qRef).toFloat32
  let dkRef := (← Tensor.gradOf kRef).toFloat32
  let dvRef := (← Tensor.gradOf vRef).toFloat32

  let routeOk := route == .tkKernel
  let outOk := Tensor.allclose outRef out 3e-2 3e-2
  let dqOk := Tensor.allclose dqRef dq 3e-2 3e-2
  let dkOk := Tensor.allclose dkRef dk 3e-2 3e-2
  let dvOk := Tensor.allclose dvRef dv 3e-2 3e-2
  let outMae := Tensor.item (Tensor.meanAll (Tensor.abs (out - outRef)))
  let dqMae := Tensor.item (Tensor.meanAll (Tensor.abs (dq - dqRef)))
  let dkMae := Tensor.item (Tensor.meanAll (Tensor.abs (dk - dkRef)))
  let dvMae := Tensor.item (Tensor.meanAll (Tensor.abs (dv - dvRef)))
  let outMax := Tensor.item (Tensor.maxAll (Tensor.abs (out - outRef)))
  let dqMax := Tensor.item (Tensor.maxAll (Tensor.abs (dq - dqRef)))
  let dkMax := Tensor.item (Tensor.maxAll (Tensor.abs (dk - dkRef)))
  let dvMax := Tensor.item (Tensor.maxAll (Tensor.abs (dv - dvRef)))
  reportCase "flash_attn_native_dense" route routeOk outOk dqOk dkOk dvOk outMae dqMae dkMae dvMae outMax dqMax dkMax dvMax
  pure (routeOk && outOk && dqOk && dkOk && dvOk)

private def portableCausalCase : IO Bool := do
  let device ← Device.cuda 0
  let q ← Tensor.setRequiresGrad ((← Tensor.randn #[1, 1, 96, 64] false device).toBFloat16) true
  let k ← Tensor.setRequiresGrad ((← Tensor.randn #[1, 1, 96, 64] false device).toBFloat16) true
  let v ← Tensor.setRequiresGrad ((← Tensor.randn #[1, 1, 96, 64] false device).toBFloat16) true
  let dO := (← Tensor.randn #[1, 1, 96, 64] false device).toBFloat16

  let (route, out) := flashAttnWithRoute q k v none true none false
  FlashBenchReference.backwardWithCotangent out dO
  let _ ← torch.cuda_synchronize
  let dq := (← Tensor.gradOf q).toFloat32
  let dk := (← Tensor.gradOf k).toFloat32
  let dv := (← Tensor.gradOf v).toFloat32

  let qRef ← Tensor.setRequiresGrad (Tensor.detach q) true
  let kRef ← Tensor.setRequiresGrad (Tensor.detach k) true
  let vRef ← Tensor.setRequiresGrad (Tensor.detach v) true
  let outRef := Tensor.attentionGQA qRef kRef vRef true
  FlashBenchReference.backwardWithCotangent outRef dO
  let _ ← torch.cuda_synchronize
  let dqRef := (← Tensor.gradOf qRef).toFloat32
  let dkRef := (← Tensor.gradOf kRef).toFloat32
  let dvRef := (← Tensor.gradOf vRef).toFloat32

  let routeOk := route == .portable
  let outOk := Tensor.allclose outRef out 1e-5 1e-5
  let dqOk := Tensor.allclose dqRef dq 1e-5 1e-5
  let dkOk := Tensor.allclose dkRef dk 1e-5 1e-5
  let dvOk := Tensor.allclose dvRef dv 1e-5 1e-5
  let outMae := Tensor.item (Tensor.meanAll (Tensor.abs (out - outRef)))
  let dqMae := Tensor.item (Tensor.meanAll (Tensor.abs (dq - dqRef)))
  let dkMae := Tensor.item (Tensor.meanAll (Tensor.abs (dk - dkRef)))
  let dvMae := Tensor.item (Tensor.meanAll (Tensor.abs (dv - dvRef)))
  let outMax := Tensor.item (Tensor.maxAll (Tensor.abs (out - outRef)))
  let dqMax := Tensor.item (Tensor.maxAll (Tensor.abs (dq - dqRef)))
  let dkMax := Tensor.item (Tensor.maxAll (Tensor.abs (dk - dkRef)))
  let dvMax := Tensor.item (Tensor.maxAll (Tensor.abs (dv - dvRef)))
  reportCase "flash_attn_portable_causal" route routeOk outOk dqOk dkOk dvOk outMae dqMae dkMae dvMae outMax dqMax dkMax dvMax
  pure (routeOk && outOk && dqOk && dkOk && dvOk)

private def portableGqaCase : IO Bool := do
  let device ← Device.cuda 0
  let q ← Tensor.setRequiresGrad ((← Tensor.randn #[1, 4, 96, 64] false device).toBFloat16) true
  let k ← Tensor.setRequiresGrad ((← Tensor.randn #[1, 2, 96, 64] false device).toBFloat16) true
  let v ← Tensor.setRequiresGrad ((← Tensor.randn #[1, 2, 96, 64] false device).toBFloat16) true
  let dO := (← Tensor.randn #[1, 4, 96, 64] false device).toBFloat16

  let (route, out) := flashAttnWithRoute q k v none false none true
  FlashBenchReference.backwardWithCotangent out dO
  let _ ← torch.cuda_synchronize
  let dq := (← Tensor.gradOf q).toFloat32
  let dk := (← Tensor.gradOf k).toFloat32
  let dv := (← Tensor.gradOf v).toFloat32

  let qRef ← Tensor.setRequiresGrad (Tensor.detach q) true
  let kRef ← Tensor.setRequiresGrad (Tensor.detach k) true
  let vRef ← Tensor.setRequiresGrad (Tensor.detach v) true
  let outRef := Tensor.attentionGQAQKV qRef kRef vRef false true
  FlashBenchReference.backwardWithCotangent outRef dO
  let _ ← torch.cuda_synchronize
  let dqRef := (← Tensor.gradOf qRef).toFloat32
  let dkRef := (← Tensor.gradOf kRef).toFloat32
  let dvRef := (← Tensor.gradOf vRef).toFloat32

  let routeOk := route == .portable
  let outOk := Tensor.allclose outRef out 1e-5 1e-5
  let dqOk := Tensor.allclose dqRef dq 1e-5 1e-5
  let dkOk := Tensor.allclose dkRef dk 1e-5 1e-5
  let dvOk := Tensor.allclose dvRef dv 1e-5 1e-5
  let outMae := Tensor.item (Tensor.meanAll (Tensor.abs (out - outRef)))
  let dqMae := Tensor.item (Tensor.meanAll (Tensor.abs (dq - dqRef)))
  let dkMae := Tensor.item (Tensor.meanAll (Tensor.abs (dk - dkRef)))
  let dvMae := Tensor.item (Tensor.meanAll (Tensor.abs (dv - dvRef)))
  let outMax := Tensor.item (Tensor.maxAll (Tensor.abs (out - outRef)))
  let dqMax := Tensor.item (Tensor.maxAll (Tensor.abs (dq - dqRef)))
  let dkMax := Tensor.item (Tensor.maxAll (Tensor.abs (dk - dkRef)))
  let dvMax := Tensor.item (Tensor.maxAll (Tensor.abs (dv - dvRef)))
  reportCase "flash_attn_portable_gqa" route routeOk outOk dqOk dkOk dvOk outMae dqMae dkMae dvMae outMax dqMax dkMax dvMax
  pure (routeOk && outOk && dqOk && dkOk && dvOk)

def main (_args : List String) : IO UInt32 := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return 1

  torch.manualSeed 20260421
  let okNative ← nativeDenseCase
  let okPortableCausal ← portableCausalCase
  let okPortableGqa ← portableGqaCase
  pure <| if okNative && okPortableCausal && okPortableGqa then 0 else 1

end Examples.GPU

def main : List String → IO UInt32 := Examples.GPU.main
