import Tyr.Typed.Neural

namespace Examples.GPU
open torch

variable {device : Device}

abbrev MhaTensor (device : Device := .CUDA 0) := Tensor { shape := #[1, 1, 128, 64], dtype := .BFloat16, device := device }   -- bf16 kernel inputs/outputs
abbrev MasterTensor (device : Device := .CUDA 0) := Tensor { shape := #[1, 1, 128, 64], dtype := .Float32, device := device } -- fp32 optimizer state
abbrev LTensor (device : Device := .CUDA 0) := Tensor { shape := #[2, 64], dtype := .Float32, device := device }
def nElems : Float := 8192.0

def lossAndGradOut (out targetOut : MhaTensor (device := device)) : Tensor { shape := #[], dtype := .Float32, device := device } × MhaTensor (device := device) :=
  let out32 : MasterTensor (device := device) := out.toFloat32
  let tgt32 : MasterTensor (device := device) := targetOut.toFloat32
  let diff : MasterTensor (device := device) := out32 - tgt32
  let lossT : Tensor { shape := #[], dtype := .Float32, device := device } := Tensor.Preserving.mulScalar (Tensor.meanAll (diff * diff)) 0.5
  let dO32 : MasterTensor (device := device) := diff / nElems
  let dO : MhaTensor (device := device) := dO32.toBFloat16
  (lossT, dO)

def sgdUpdate (param grad : MasterTensor (device := device)) (lr : Float) : MasterTensor (device := device) :=
  param - (grad * lr)

def oneTorchTrainStep
    (qM kM vM : MasterTensor (device := device))
    (targetOut32 : MasterTensor (device := device))
    (lr : Float)
    : IO (MasterTensor (device := device) × MasterTensor (device := device) × MasterTensor (device := device) × Tensor { shape := #[], dtype := .Float32, device := device }) := do
  let qLeaf ← Tensor.setRequiresGrad (Tensor.detach qM) true
  let kLeaf ← Tensor.setRequiresGrad (Tensor.detach kM) true
  let vLeaf ← Tensor.setRequiresGrad (Tensor.detach vM) true
  let out : MasterTensor (device := device) := Tensor.attentionGQA qLeaf kLeaf vLeaf 0.0 false
  let diff : MasterTensor (device := device) := out - targetOut32
  let lossT : Tensor { shape := #[], dtype := .Float32, device := device } := Tensor.Preserving.mulScalar (Tensor.meanAll (diff * diff)) 0.5
  Tensor.backwardLoss lossT
  let dQ : MasterTensor (device := device) ← Tensor.gradOf qLeaf
  let dK : MasterTensor (device := device) ← Tensor.gradOf kLeaf
  let dV : MasterTensor (device := device) ← Tensor.gradOf vLeaf
  let qM' := sgdUpdate qM dQ lr
  let kM' := sgdUpdate kM dK lr
  let vM' := sgdUpdate vM dV lr
  pure (qM', kM', vM', lossT)

end Examples.GPU
