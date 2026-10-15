import Tyr.Typed.Neural
import Tyr.TensorStruct

/-! Typed GPT primitives built from the shared tensor operations.
Dynamic reshapes and native tensor inputs validate their claims. -/
namespace torch.gpt

variable {device : Device}

abbrev Activation (s : Shape) := Tensor { shape := s, dtype := .Float32, device := device }
abbrev Tokens (s : Shape) := Tensor { shape := s, dtype := .Int64, device := device }

namespace TensorOps

def checked {s : Shape} (d : DType) (raw : RawTensor) : IO (Tensor { shape := s, dtype := d, device := device }) :=
  match Tensor.ofRaw (σ := { shape := s, dtype := d, device := device }) raw with
  | .ok t => pure t
  | .error message => throw (IO.userError message)

def leaf {s : Shape} (t : Activation (device := device) s) : IO (Activation (device := device) s) :=
  Tensor.parameter t

def reshape {s : Shape} {d : DType} (t : Tensor { shape := s, dtype := d, device := device }) (target : Shape) : IO (Tensor { shape := target, dtype := d, device := device }) := do
  unless TensorSpec.numelOfShape s == TensorSpec.numelOfShape target do
    throw (IO.userError "GPT reshape changes the element count")
  checked d (RawTensor.reshape s t.raw target)

def dropout {s : Shape} (t : Activation (device := device) s) (p : Float) (training : Bool) : IO (Activation (device := device) s) := do
  Tensor.dropout t p training

def layerNorm {b t n : UInt64} (x : Activation (device := device) #[b,t,n])
    (weight bias : Activation (device := device) #[n]) : Activation (device := device) #[b,t,n] :=
  Tensor.layerNorm x weight bias

def splitHeads {b t h d : UInt64} (x : Activation (device := device) #[b,t,h,d]) : Activation (device := device) #[b,h,t,d] :=
  Tensor.splitHeads x

def joinHeads {b h t d : UInt64} (x : Activation (device := device) #[b,h,t,d]) : Activation (device := device) #[b,t,h,d] :=
  Tensor.joinHeads x

def attention {b h t d : UInt64} (q k v : Activation (device := device) #[b,h,t,d])
    (dropout : Float) : Activation (device := device) #[b,h,t,d] :=
  Tensor.attention q k v dropout true

def embedding {b t v e : UInt64} (idx : Tokens (device := device) #[b,t])
    (weight : Activation (device := device) #[v,e]) : Activation (device := device) #[b,t,e] :=
  Tensor.embedding idx weight

def embedding1d {t v e : UInt64} (idx : Tokens (device := device) #[t])
    (weight : Activation (device := device) #[v,e]) : Activation (device := device) #[t,e] :=
  Tensor.embedding1d idx weight

def crossEntropy {n c : UInt64} (logits : Activation (device := device) #[n,c])
    (targets : Tokens (device := device) #[n]) : Activation (device := device) #[] :=
  Tensor.crossEntropy logits targets

end TensorOps
end torch.gpt
