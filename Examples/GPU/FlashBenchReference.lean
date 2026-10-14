import Tyr.Typed.Neural
import Tyr.Inference.OwnedKV

namespace Examples.GPU.FlashBenchReference
open torch

def relL2
    {s : Shape} {dtype : DType}
    (actual ref : DTensor s dtype) : Float :=
  let diffNorm :=
    Tensor.item <| Tensor.sqrt <| Tensor.sumAll <| Tensor.pow ((actual - ref).toFloat32) 2.0
  let refNorm :=
    Tensor.item <| Tensor.sqrt <| Tensor.sumAll <| Tensor.pow (ref.toFloat32) 2.0
  if refNorm <= 1.0e-12 then diffNorm else diffNorm / refNorm

/-- Fresh storage makes repeated backward calls independent even under Lean commoning. -/
def makeLeaf {s : Shape} {dtype : DType} (base : DTensor s dtype) : IO (DTensor s dtype) := do
  let leaf ← torch.data.cloneInferenceIO base
  pure (Tensor.setRequiresGrad leaf true)

def backwardWithCotangent {s : Shape} {dtype : DType} (out dOut : DTensor s dtype) : IO Unit :=
  Tensor.backwardLoss (Tensor.sumAll (out.toFloat32 * dOut.toFloat32))

end Examples.GPU.FlashBenchReference
