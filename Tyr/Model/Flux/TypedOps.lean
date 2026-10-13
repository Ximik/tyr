import Tyr.Model.Qwen35.TypedOps
import Tyr.RawTensor.Flux
import Tyr.Module.LayerNorm

namespace torch.flux

def randomWeight (s : Shape) : IO (STensor s) := do
  return (← Tensor.randn s).forgetDType

def zerosWeight (s : Shape) : STensor s := (Tensor.zeros s).forgetDType

def initLayerNorm (dim : UInt64) (eps : Float := 1e-5) (train : Bool := true) : LayerNorm dim (.Unknown "dynamic") :=
  { weight := (Tensor.ones #[dim] train).forgetDType,
    bias := (Tensor.zeros #[dim] train).forgetDType, eps := ⟨eps⟩ }

def sliceTyped {s : Shape} {d : DType} (x : DTensor s d) (dim : Nat) (start len : UInt64) : DTensor (s.set! dim len) d :=
  x.sliceChecked dim start len (s.set! dim len)

def transposeAttention {b s h d : UInt64} {dt : DType} (x : DTensor #[b, s, h, d] dt) : DTensor #[b, h, s, d] dt := x.transpose 1 2

def fromAttention {b h s d : UInt64} {dt : DType} (x : DTensor #[b, h, s, d] dt) : DTensor #[b, s, h, d] dt := x.transpose 1 2

def saveImagePPM {b h w : UInt64} (image : STensor #[b, 3, h, w]) (path : String) : IO Unit :=
  RawTensor.savePPM image.shape (image.to .CPU).raw path

end torch.flux
