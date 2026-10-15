import Tyr.Typed.Neural
import Tyr.RawTensor.BranchingFlows
import Tyr.TensorStruct

namespace torch.branching

def zerosDynamic (shape : Shape) (requiresGrad : Bool := false) (device : Device := .CPU) (noGrad : requiresGrad = false := by decide) : Tensor { shape := shape, device } :=
  (Tensor.zeros shape requiresGrad device noGrad).forgetDType
def onesDynamic (shape : Shape) (device : Device := .CPU) : Tensor { shape := shape, device } :=
  (Tensor.ones shape false device).forgetDType
def randnDynamic (shape : Shape) (device : Device := .CPU) : IO (Tensor { shape := shape, device }) := do
  return (← Tensor.randn shape false device).forgetDType
def floatsDynamic (values : Array Float) : Tensor { shape := #[values.size.toUInt64], device := .CPU } :=
  (Tensor.fromFloatArray values).forgetDType
def floatDynamic {s : Shape} {d : DType} {dev : Device} (x : Tensor { shape := s, dtype := d, device := dev }) : Tensor { shape := s, device := dev } :=
  x.toFloat32.forgetDType
def bfloatDynamic {s : Shape} {dev : Device} (x : Tensor { shape := s, device := dev }) : Tensor { shape := s, device := dev } := x.toBFloat16.forgetDType

def logDynamic {s : Shape} {dev : Device} (x : Tensor { shape := s, device := dev }) : Tensor { shape := s, device := dev } := Tensor.Promoting.log x
def softplusDynamic {s : Shape} {dev : Device} (x : Tensor { shape := s, device := dev }) : Tensor { shape := s, device := dev } := x.checkedReplacement (RawTensor.branchingSoftplus s x.raw)
def powDynamic {s : Shape} {dev : Device} (x : Tensor { shape := s, device := dev }) (n : Float) : Tensor { shape := s, device := dev } := Tensor.Promoting.pow x n

def sliceTyped {s out : Shape} {d : DType} {dev : Device} (x : Tensor { shape := s, dtype := d, device := dev }) (dim : Nat) (start len : UInt64) : Tensor { shape := out, dtype := d, device := dev } :=
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

/-- A tensor learning rate follows rank-aware PyTorch promotion, including
weak scalar precision. The result is never restored to the input dtype. -/
def mulTensorScalar {σ : StaticSpec} {d result : DType} {scalarDevice : Device}
    [TensorSpec.Promotes σ.toTensorSpec ⟨#[], d⟩ result]
    [Device.PointwiseCompatible σ ⟨#[], d, scalarDevice⟩]
    (x : Tensor σ) (scalar : Tensor { shape := #[], dtype := d, device := scalarDevice }) :
    Tensor ⟨σ.shape, result, σ.pointwiseDevice ⟨#[], d, scalarDevice⟩⟩ :=
  let result := do
    Tensor.checkPointwise x scalar
    Tensor.ofRaw (← RawTensor.mulTensorScalarChecked σ.shape x.raw scalar.raw)
  match result with
  | .ok tensor => tensor
  | .error message => panic! s!"BranchingFlows.mulTensorScalar: {message}"

def mulTensorScalar? {σ : StaticSpec} {d : DType} {scalarDevice : Device}
    (x : Tensor σ) (scalar : Tensor { shape := #[], dtype := d, device := scalarDevice }) :=
  Tensor.Promoting.mulB? x scalar σ.shape

/-- Optimizer updates explicitly require native dtype preservation. This
checked adapter rejects any promoted result instead of casting it back. -/
def mulTensorScalarPreserving? {σ : StaticSpec} {d : DType} {scalarDevice : Device}
    (x : Tensor σ) (scalar : Tensor { shape := #[], dtype := d, device := scalarDevice }) :
    Except String (Tensor σ) := do
  let result ← mulTensorScalar? x scalar
  x.checkedReplacement? result.tensor.raw

def mulTensorScalarPreserving {σ : StaticSpec} {d : DType} {scalarDevice : Device}
    (x : Tensor σ) (scalar : Tensor { shape := #[], dtype := d, device := scalarDevice }) : Tensor σ :=
  match mulTensorScalarPreserving? x scalar with
  | .ok tensor => tensor
  | .error message => panic! s!"BranchingFlows.mulTensorScalarPreserving: {message}"

def cosDynamic {s : Shape} {dev : Device} (x : Tensor { shape := s, device := dev }) : Tensor { shape := s, device := dev } := Tensor.Promoting.cos x

def affine2d {batch input output : UInt64} {dev : Device} (x : Tensor { shape := #[batch, input], device := dev })
    (w : Tensor { shape := #[output, input], device := dev }) (bias : Tensor { shape := #[output], device := dev }) : Tensor { shape := #[batch, output], device := dev } :=
  Tensor.Promoting.add (Tensor.linear x w) ((bias.reshapeChecked #[1, output]).expand #[batch, output])

def transposeAttention {b s h d : UInt64} {dev : Device} (x : Tensor { shape := #[b, s, h, d], device := dev }) : Tensor { shape := #[b, h, s, d], device := dev } :=
  x.transpose 1 2

def attentionBias? {b h s d : UInt64} {dev : Device} (q k v : Tensor { shape := #[b, h, s, d], device := dev })
    (bias : Tensor { shape := #[b, h, s, s], device := dev }) :
    Except String (Tensor { shape := #[b, h, s, d], device := dev }) := do
  q.checkedReplacement? (← RawTensor.branchingAttentionBias b h s d q.raw k.raw v.raw bias.raw false)

def attentionBias {b h s d : UInt64} {dev : Device} (q k v : Tensor { shape := #[b, h, s, d], device := dev })
    (bias : Tensor { shape := #[b, h, s, s], device := dev }) : Tensor { shape := #[b, h, s, d], device := dev } :=
  match attentionBias? q k v bias with
  | .ok result => result
  | .error message => panic! s!"BranchingFlows.attentionBias: {message}"

def attentionBiasIO {b h s d : UInt64} {dev : Device} (q k v : Tensor { shape := #[b, h, s, d], device := dev })
    (bias : Tensor { shape := #[b, h, s, s], device := dev })
    (dropout : Float) (training : Bool) (causal : Bool := false) :
    IO (Tensor { shape := #[b, h, s, d], device := dev }) := do
  let effective ← IO.ofExcept (Tensor.attentionDropout? dropout training)
  let raw ← RawTensor.branchingAttentionBiasIO b h s d q.raw k.raw v.raw bias.raw
    effective causal
  IO.ofExcept (q.checkedReplacement? raw)

def fullDynamic (shape : Shape) (value : Float) (requiresGrad : Bool := false) (device : Device := .CPU) (noGrad : requiresGrad = false := by decide) : Tensor { shape := shape, device } :=
  Tensor.Preserving.addScalar (zerosDynamic shape requiresGrad device noGrad) value

/-- Recast a tree only when each leaf's static dtype permits BF16. -/
def castTreeBFloat16 {α : Type} [TensorStruct α] (params : α) : IO α :=
  TensorStruct.castDType .BFloat16 params

def binaryCrossEntropySum {s : Shape} {dev : Device} (input target weight : Tensor { shape := s, device := dev }) : Tensor { shape := #[], device := dev } :=
  match Tensor.ofRaw (RawTensor.branchingBCE s input.raw target.raw (some weight.raw) "sum") with
  | .ok result => result
  | .error message => panic! s!"BranchingFlows binary cross entropy: {message}"

end torch.branching
