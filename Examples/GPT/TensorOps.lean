import Tyr.Typed.Neural
import Tyr.TensorStruct

/-! Typed GPT primitives built from the shared tensor operations.
Dynamic reshapes and native tensor inputs validate their claims. -/
namespace torch.gpt

abbrev Activation (s : Shape) := Tensor { shape := s, dtype := .Float32 }
abbrev Tokens (s : Shape) := Tensor { shape := s, dtype := .Int64 }

namespace TensorOps

def checked {s : Shape} (d : DType) (raw : RawTensor) : IO (Tensor { shape := s, dtype := d }) :=
  match Tensor.ofTensor d raw with
  | .ok t => pure t
  | .error message => throw (IO.userError message)

def leaf {s : Shape} (t : Activation s) : Activation s :=
  t.detach.setRequiresGrad true

def reshape {s : Shape} {d : DType} (t : Tensor { shape := s, dtype := d }) (target : Shape) : IO (Tensor { shape := target, dtype := d }) := do
  unless TensorSpec.numelOfShape s == TensorSpec.numelOfShape target do
    throw (IO.userError "GPT reshape changes the element count")
  checked d (RawTensor.reshape s t.raw target)

def dropout {s : Shape} (t : Activation s) (p : Float) (training : Bool) : IO (Activation s) := do
  Tensor.dropout t p training

def layerNorm {b t n : UInt64} (x : Activation #[b,t,n])
    (weight bias : Activation #[n]) : Activation #[b,t,n] :=
  Tensor.layerNorm x weight bias

def splitHeads {b t h d : UInt64} (x : Activation #[b,t,h,d]) : Activation #[b,h,t,d] :=
  Tensor.splitHeads x

def joinHeads {b h t d : UInt64} (x : Activation #[b,h,t,d]) : Activation #[b,t,h,d] :=
  Tensor.joinHeads x

def attention {b h t d : UInt64} (q k v : Activation #[b,h,t,d])
    (dropout : Float) : Activation #[b,h,t,d] :=
  Tensor.attention q k v dropout true

def embedding {b t v e : UInt64} (idx : Tokens #[b,t])
    (weight : Activation #[v,e]) : Activation #[b,t,e] :=
  Tensor.embedding idx weight

def embedding1d {t v e : UInt64} (idx : Tokens #[t])
    (weight : Activation #[v,e]) : Activation #[t,e] :=
  Tensor.embedding1d idx weight

def crossEntropy {n c : UInt64} (logits : Activation #[n,c])
    (targets : Tokens #[n]) : Activation #[] :=
  Tensor.crossEntropy logits targets

end TensorOps
end torch.gpt
