import Tyr.Typed.Neural
import Tyr.RawTensor.BranchingFlows
import Tyr.TensorStruct

namespace torch.branching

def zerosDynamic (shape : Shape) (requiresGrad : Bool := false) (device : Device := .CPU) : Tensor { shape := shape } :=
  (Tensor.zeros shape requiresGrad device).forgetDType
def onesDynamic (shape : Shape) : Tensor { shape := shape } := (Tensor.ones shape).forgetDType
def randnDynamic (shape : Shape) : IO (Tensor { shape := shape }) := do
  return (← Tensor.randn shape).forgetDType
def floatsDynamic (values : Array Float) : Tensor { shape := #[values.size.toUInt64] } :=
  (Tensor.fromFloatArray values).forgetDType
def floatDynamic {s : Shape} {d : DType} (x : Tensor { shape := s, dtype := d }) : Tensor { shape := s } :=
  x.toFloat32.forgetDType
def bfloatDynamic {s : Shape} (x : Tensor { shape := s }) : Tensor { shape := s } := x.toBFloat16.forgetDType

def logDynamic {s : Shape} (x : Tensor { shape := s }) : Tensor { shape := s } := x.checkedReplacement (RawTensor.log s x.raw)
def softplusDynamic {s : Shape} (x : Tensor { shape := s }) : Tensor { shape := s } := x.checkedReplacement (RawTensor.branchingSoftplus s x.raw)
def powDynamic {s : Shape} (x : Tensor { shape := s }) (n : Float) : Tensor { shape := s } := x.checkedReplacement (RawTensor.pow s x.raw n)

def sliceTyped {s out : Shape} {d : DType} (x : Tensor { shape := s, dtype := d }) (dim : Nat) (start len : UInt64) : Tensor { shape := out, dtype := d } :=
  x.sliceChecked dim start len out

/-- Return an updated destination after checking native shapes and dtypes.
Retained aliases keep their values. Invalid metadata raises an IO error; this
operation never broadcasts or implicitly casts the source. -/
def copyIntoChecked? {σ τ : StaticSpec} (dst : Tensor σ) (src : Tensor τ) :
    IO (Tensor σ) := do
  match dst.validate, src.validate with
  | .error message, _ => throw (IO.userError s!"BranchingFlows static buffer destination: {message}")
  | _, .error message => throw (IO.userError s!"BranchingFlows static buffer source: {message}")
  | .ok (), .ok () => pure ()
  if dst.runtimeShape != src.runtimeShape then
    throw (IO.userError s!"BranchingFlows static buffer shape mismatch: {dst.runtimeShape} vs {src.runtimeShape}")
  if dst.actualSpec.dtype != src.actualSpec.dtype then
    throw (IO.userError s!"BranchingFlows static buffer dtype mismatch: {dst.actualSpec.dtype} vs {src.actualSpec.dtype}; cast the source explicitly before copying")
  let dtype := dst.raw.dtype
  let device := dst.raw.device
  let raw ← RawTensor.copyInplace σ.shape dst.raw src.raw
  match Tensor.checkedUpdate? dtype device raw with
  | .ok result => pure result
  | .error message => throw (IO.userError s!"BranchingFlows static buffer copy: {message}")

/-- Checked returned-value copy. Errors are reported through IO. -/
def copyIntoChecked {σ τ : StaticSpec} (dst : Tensor σ) (src : Tensor τ) : IO (Tensor σ) :=
  copyIntoChecked? dst src

def touchTensor {σ : StaticSpec} (t : Tensor σ) : IO Unit := RawTensor.touch σ.shape t.raw

def mulTensorScalar {σ : StaticSpec} {d : DType} (x : Tensor σ) (scalar : Tensor { shape := #[], dtype := d }) : Tensor σ :=
  x.checkedReplacement (RawTensor.mulTensorScalar σ.shape x.raw scalar.raw)

def cosDynamic {s : Shape} (x : Tensor { shape := s }) : Tensor { shape := s } := x.checkedReplacement (RawTensor.cos s x.raw)

def affine2d {batch input output : UInt64} (x : Tensor { shape := #[batch, input] })
    (w : Tensor { shape := #[output, input] }) (bias : Tensor { shape := #[output] }) : Tensor { shape := #[batch, output] } :=
  Tensor.linear x w + (bias.reshapeChecked #[1, output]).expand #[batch, output]

def transposeAttention {b s h d : UInt64} (x : Tensor { shape := #[b, s, h, d] }) : Tensor { shape := #[b, h, s, d] } :=
  x.transpose 1 2

def attentionBias {b h s d : UInt64} (q k v : Tensor { shape := #[b, h, s, d] })
    (bias : Tensor { shape := #[b, h, s, s] }) : Tensor { shape := #[b, h, s, d] } :=
  q.checkedReplacement (RawTensor.branchingAttentionBias b h s d q.raw k.raw v.raw bias.raw 0.0 false)

def fullDynamic (shape : Shape) (value : Float) (requiresGrad : Bool := false) (device : Device := .CPU) : Tensor { shape := shape } :=
  zerosDynamic shape requiresGrad device + value

/-- Recast a tree only when each leaf's static dtype permits BF16. -/
def castTreeBFloat16 {α : Type} [TensorStruct α] (params : α) : IO α :=
  TensorStruct.castDType .BFloat16 params

def binaryCrossEntropySum {s : Shape} (input target weight : Tensor { shape := s }) : Tensor { shape := #[] } :=
  match Tensor.ofRaw (RawTensor.branchingBCE s input.raw target.raw (some weight.raw) "sum") with
  | .ok result => result
  | .error message => panic! s!"BranchingFlows binary cross entropy: {message}"

end torch.branching
