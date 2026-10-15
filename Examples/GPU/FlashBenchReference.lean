import Tyr.Typed.Neural
import Tyr.Inference.OwnedKV

namespace Examples.GPU.FlashBenchReference
open torch

variable {device : Device}

def relL2
    {s : Shape} {dtype : DType}
    (actual ref : Tensor { shape := s, dtype := dtype, device := device }) : Float :=
  let diffNorm :=
    Tensor.item <| Tensor.sqrt <| Tensor.sumAll <| Tensor.pow ((Tensor.Preserving.sub actual ref).toFloat32) 2.0
  let refNorm :=
    Tensor.item <| Tensor.sqrt <| Tensor.sumAll <| Tensor.pow (ref.toFloat32) 2.0
  if refNorm <= 1.0e-12 then diffNorm else diffNorm / refNorm

/-- Fresh storage makes repeated backward calls independent even under Lean commoning. -/
def makeLeaf {s : Shape} {dtype : DType} (base : Tensor { shape := s, dtype := dtype, device := device }) : IO (Tensor { shape := s, dtype := dtype, device := device }) := do
  let leaf ← torch.data.cloneInferenceIO base
  Tensor.parameter leaf

def backwardWithCotangent {s : Shape} {dtype : DType} (out dOut : Tensor { shape := s, dtype := dtype, device := device }) : IO Unit :=
  Tensor.backwardLoss (Tensor.sumAll (out.toFloat32 * dOut.toFloat32))

end Examples.GPU.FlashBenchReference
