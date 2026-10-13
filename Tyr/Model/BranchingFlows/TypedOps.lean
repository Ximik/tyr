import Tyr.Typed.Neural
import Tyr.RawTensor.BranchingFlows
import Tyr.TensorStruct

namespace torch.branching

def zerosDynamic (shape : Shape) (requiresGrad : Bool := false) (device : Device := .CPU) : STensor shape :=
  (Tensor.zeros shape requiresGrad device).forgetDType
def onesDynamic (shape : Shape) : STensor shape := (Tensor.ones shape).forgetDType
def randnDynamic (shape : Shape) : IO (STensor shape) := do
  return (← Tensor.randn shape).forgetDType
def floatsDynamic (values : Array Float) : STensor #[values.size.toUInt64] :=
  (Tensor.fromFloatArray values).forgetDType
def floatDynamic {s : Shape} {d : DType} (x : DTensor s d) : STensor s :=
  x.toFloat32.forgetDType
def bfloatDynamic {s : Shape} (x : STensor s) : STensor s := x.toBFloat16.forgetDType

def logDynamic {s : Shape} (x : STensor s) : STensor s := x.checkedReplacement (RawTensor.log s x.raw)
def softplusDynamic {s : Shape} (x : STensor s) : STensor s := x.checkedReplacement (RawTensor.branchingSoftplus s x.raw)
def powDynamic {s : Shape} (x : STensor s) (n : Float) : STensor s := x.checkedReplacement (RawTensor.pow s x.raw n)

def sliceTyped {s out : Shape} {d : DType} (x : DTensor s d) (dim : Nat) (start len : UInt64) : DTensor out d :=
  x.sliceChecked dim start len out

def copyIntoChecked {σ τ : StaticSpec} (dst : Tensor σ) (src : Tensor τ) : Tensor σ :=
  if dst.runtimeShape != src.runtimeShape then
    panic! s!"BranchingFlows static buffer shape mismatch: {dst.runtimeShape} vs {src.runtimeShape}"
  else dst.checkedReplacement (RawTensor.copyInplace σ.shape dst.raw src.raw)

def touchTensor {σ : StaticSpec} (t : Tensor σ) : IO Unit := RawTensor.touch σ.shape t.raw

def mulTensorScalar {σ : StaticSpec} {d : DType} (x : Tensor σ) (scalar : DTensor #[] d) : Tensor σ :=
  x.checkedReplacement (RawTensor.mulTensorScalar σ.shape x.raw scalar.raw)

def cosDynamic {s : Shape} (x : STensor s) : STensor s := x.checkedReplacement (RawTensor.cos s x.raw)

def affine2d {batch input output : UInt64} (x : STensor #[batch, input])
    (w : STensor #[output, input]) (bias : STensor #[output]) : STensor #[batch, output] :=
  Tensor.linear x w + (bias.reshapeChecked #[1, output]).expand #[batch, output]

def transposeAttention {b s h d : UInt64} (x : STensor #[b, s, h, d]) : STensor #[b, h, s, d] :=
  x.transpose 1 2

def attentionBias {b h s d : UInt64} (q k v : STensor #[b, h, s, d])
    (bias : STensor #[b, h, s, s]) : STensor #[b, h, s, d] :=
  q.checkedReplacement (RawTensor.branchingAttentionBias b h s d q.raw k.raw v.raw bias.raw 0.0 false)

def fullDynamic (shape : Shape) (value : Float) (requiresGrad : Bool := false) (device : Device := .CPU) : STensor shape :=
  zerosDynamic shape requiresGrad device + value

/-- Recast a tree only when each leaf's static dtype permits BF16. -/
def castTreeBFloat16 {α : Type} [TensorStruct α] (params : α) : IO α :=
  TensorStruct.mapM (fun {σ} t => do
    match Tensor.ofRaw (σ := σ) t.toBFloat16.raw with
    | .ok converted => pure converted
    | .error message => throw <| IO.userError s!"BranchingFlows BF16 parameter conversion: {message}") params

def binaryCrossEntropySum {s : Shape} (input target weight : STensor s) : STensor #[] :=
  match Tensor.ofRaw (RawTensor.branchingBCE s input.raw target.raw (some weight.raw) "sum") with
  | .ok result => result
  | .error message => panic! s!"BranchingFlows binary cross entropy: {message}"

end torch.branching
