import Tyr.Typed.Ops
import Tyr.TensorStruct

/-! Typed GPT primitives. Raw FFI handles are confined to these adapters.
Dynamic reshapes and native tensor inputs validate their claims. -/
namespace torch.gpt

abbrev Activation (s : Shape) := DTensor s .Float32
abbrev Tokens (s : Shape) := DTensor s .Int64

namespace TensorOps

def checked {s : Shape} (d : DType) (raw : RawTensor) : IO (DTensor s d) :=
  match Tensor.ofTensor d raw with
  | .ok t => pure t
  | .error message => throw (IO.userError message)

def leaf {s : Shape} (t : Activation s) : Activation s :=
  .assumeSpec (RawTensor.setRequiresGrad s (RawTensor.detach s t.raw) true)

def reshape {s : Shape} {d : DType} (t : DTensor s d) (target : Shape) : IO (DTensor target d) := do
  unless TensorSpec.numelOfShape s == TensorSpec.numelOfShape target do
    throw (IO.userError "GPT reshape changes the element count")
  checked d (RawTensor.reshape s t.raw target)

def dropout {s : Shape} (t : Activation s) (p : Float) (training : Bool) : IO (Activation s) := do
  Tensor.dropout t p training

def layerNorm {b t n : UInt64} (x : Activation #[b,t,n])
    (weight bias : Activation #[n]) : Activation #[b,t,n] :=
  .assumeSpec (RawTensor.layerNorm b t n x.raw weight.raw bias.raw)

def splitHeads {b t h d : UInt64} (x : Activation #[b,t,h,d]) : Activation #[b,h,t,d] :=
  .assumeSpec (RawTensor.splitHeads b t h d x.raw)

def joinHeads {b h t d : UInt64} (x : Activation #[b,h,t,d]) : Activation #[b,t,h,d] :=
  .assumeSpec (RawTensor.joinHeads b h t d x.raw)

def attention {b h t d : UInt64} (q k v : Activation #[b,h,t,d])
    (dropout : Float) : Activation #[b,h,t,d] :=
  .assumeSpec (RawTensor.attention b h t d q.raw k.raw v.raw dropout true)

def embedding {b t v e : UInt64} (idx : Tokens #[b,t])
    (weight : Activation #[v,e]) : Activation #[b,t,e] :=
  Tensor.embedding idx weight

def embedding1d {t v e : UInt64} (idx : Tokens #[t])
    (weight : Activation #[v,e]) : Activation #[t,e] :=
  Tensor.embedding1d idx weight

def crossEntropy {n c : UInt64} (logits : Activation #[n,c])
    (targets : Tokens #[n]) : Activation #[] :=
  .assumeSpec (RawTensor.crossEntropy n c logits.raw targets.raw)

end TensorOps
end torch.gpt
