import Tyr.Typed.Neural

namespace Examples.GPU
open torch

abbrev MhaTensor := DTensor #[1, 1, 128, 64] .BFloat16   -- bf16 kernel inputs/outputs
abbrev MasterTensor := DTensor #[1, 1, 128, 64] .Float32 -- fp32 optimizer state
abbrev LTensor := DTensor #[2, 64] .Float32
def nElems : Float := 8192.0

def lossAndGradOut (out targetOut : MhaTensor) : DTensor #[] .Float32 × MhaTensor :=
  let out32 : MasterTensor := out.toFloat32
  let tgt32 : MasterTensor := targetOut.toFloat32
  let diff : MasterTensor := out32 - tgt32
  let lossT : DTensor #[] .Float32 := Tensor.Preserving.mulScalar (Tensor.meanAll (diff * diff)) 0.5
  let dO32 : MasterTensor := diff / nElems
  let dO : MhaTensor := dO32.toBFloat16
  (lossT, dO)

def sgdUpdate (param grad : MasterTensor) (lr : Float) : MasterTensor :=
  param - (grad * lr)

def oneTorchTrainStep
    (qM kM vM : MasterTensor)
    (targetOut32 : MasterTensor)
    (lr : Float)
    : IO (MasterTensor × MasterTensor × MasterTensor × DTensor #[] .Float32) := do
  let qLeaf := Tensor.setRequiresGrad (Tensor.detach qM) true
  let kLeaf := Tensor.setRequiresGrad (Tensor.detach kM) true
  let vLeaf := Tensor.setRequiresGrad (Tensor.detach vM) true
  let out : MasterTensor := Tensor.attentionGQA qLeaf kLeaf vLeaf 0.0 false
  let diff : MasterTensor := out - targetOut32
  let lossT : DTensor #[] .Float32 := Tensor.Preserving.mulScalar (Tensor.meanAll (diff * diff)) 0.5
  Tensor.backwardLoss lossT
  let dQ : MasterTensor := Tensor.gradOf qLeaf
  let dK : MasterTensor := Tensor.gradOf kLeaf
  let dV : MasterTensor := Tensor.gradOf vLeaf
  let qM' := sgdUpdate qM dQ lr
  let kM' := sgdUpdate kM dK lr
  let vM' := sgdUpdate vM dV lr
  pure (qM', kM', vM', lossT)

end Examples.GPU
