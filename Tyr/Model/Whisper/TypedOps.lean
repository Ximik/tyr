import Tyr.Model.Qwen35.TypedOps
import Tyr.Model.Utils
import Tyr.Module.LayerNorm

namespace torch.whisper

def initWeightDynamic (s : Shape) (fanIn : UInt64) : IO (STensor s) :=
  Model.initWeight s fanIn (.Unknown "dynamic")
def initBiasDynamic (s : Shape) : STensor s := Model.initBias s (.Unknown "dynamic")
def zerosDynamic (s : Shape) (grad : Bool := false) (device : Device := .CPU) : STensor s :=
  (Tensor.zeros s grad device).forgetDType

def initLayerNorm (dim : UInt64) (eps : Float := 1e-5) : LayerNorm dim (.Unknown "dynamic") :=
  { weight := (Tensor.ones #[dim] true).forgetDType,
    bias := (Tensor.zeros #[dim] true).forgetDType, eps := ⟨eps⟩ }

def sliceTyped {s : Shape} {d : DType} (x : DTensor s d) (dim : Nat) (start len : UInt64) : DTensor (s.set! dim len) d :=
  x.sliceChecked dim start len (s.set! dim len)

def transposeAttention {b s h d : UInt64} (x : STensor #[b, s, h, d]) : STensor #[b, h, s, d] := x.transpose 1 2

def fromAttention {b h s d : UInt64} (x : STensor #[b, h, s, d]) : STensor #[b, s, h, d] := x.transpose 1 2

end torch.whisper
