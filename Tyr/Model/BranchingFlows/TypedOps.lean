import Tyr.Typed.Neural
import Tyr.ConstantSupport
import Tyr.RawTensor.BranchingFlows
import Tyr.TensorStruct

namespace torch.branching

variable {dtype : DType}
variable [DType.RealFloating dtype]

def zerosDynamic (shape : Shape) (requiresGrad : Bool := false) (device : Device := .CPU)
    (_noGrad : requiresGrad = false := by decide) (dtype : DType := .Float32)
    [DType.ConstantSupported dtype device] :
    Tensor { shape, dtype, device } := Tensor.zerosFor { shape, dtype, device }
def onesDynamic (shape : Shape) (device : Device := .CPU) (dtype : DType := .Float32)
    [DType.ConstantSupported dtype device] :
    Tensor { shape, dtype, device } := Tensor.onesFor { shape, dtype, device }
def randnDynamic (shape : Shape) (device : Device := .CPU) (dtype : DType := .Float32) :
    IO (Tensor { shape, dtype, device }) := do
  IO.ofExcept ((← Tensor.randn shape false device).castDType? dtype)
def floatsDynamic (values : Array Float) : Tensor { shape := #[values.size.toUInt64], dtype := .Float32, device := .CPU } :=
  Tensor.fromFloatArray values
def floatDynamic {s : Shape} {d : DType} {dev : Device} (x : Tensor { shape := s, dtype := d, device := dev }) : Tensor { shape := s, dtype := .Float32, device := dev } :=
  x.toFloat32
def bfloatDynamic {s : Shape} {d : DType} {dev : Device} (x : Tensor { shape := s, dtype := d, device := dev }) : Tensor { shape := s, dtype := .BFloat16, device := dev } := x.toBFloat16

def logDynamic {s : Shape} {dev : Device} (x : Tensor { shape := s, dtype, device := dev }) : Tensor { shape := s, dtype, device := dev } := Tensor.log x
def softplusDynamic {s : Shape} {dev : Device} (x : Tensor { shape := s, dtype, device := dev }) : Tensor { shape := s, dtype, device := dev } := x.checkedReplacement (RawTensor.branchingSoftplus s x.raw)
def powDynamic {s : Shape} {dev : Device} (x : Tensor { shape := s, dtype, device := dev }) (n : Float) : Tensor { shape := s, dtype, device := dev } := Tensor.pow x n

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
  | .error message => nativeInvariantFailure s!"BranchingFlows.mulTensorScalar: {message}"

def mulTensorScalar? {σ : StaticSpec} {d : DType} {scalarDevice : Device}
    (x : Tensor σ) (scalar : Tensor { shape := #[], dtype := d, device := scalarDevice }) :=
  Tensor.Promoting.mulB? x scalar σ.shape

def cosDynamic {s : Shape} {dev : Device} (x : Tensor { dtype := dtype, shape := s, device := dev }) : Tensor { dtype := dtype, shape := s, device := dev } := Tensor.cos x

def affine2d {batch input output : UInt64} {dev : Device} (x : Tensor { dtype := dtype, shape := #[batch, input], device := dev })
    (w : Tensor { dtype := dtype, shape := #[output, input], device := dev }) (bias : Tensor { dtype := dtype, shape := #[output], device := dev }) : Tensor { dtype := dtype, shape := #[batch, output], device := dev } :=
  Tensor.add (Tensor.linear x w) ((bias.reshapeChecked #[1, output]).expand #[batch, output])

def transposeAttention {b s h d : UInt64} {dev : Device} (x : Tensor { dtype := dtype, shape := #[b, s, h, d], device := dev }) : Tensor { dtype := dtype, shape := #[b, h, s, d], device := dev } :=
  x.transpose 1 2

def attentionBias? {biasDType : DType} {b h s d : UInt64} {dev : Device} (q k v : Tensor { dtype := dtype, shape := #[b, h, s, d], device := dev })
    (bias : Tensor { dtype := biasDType, shape := #[b, h, s, s], device := dev }) :
    Except String (Tensor { dtype := dtype, shape := #[b, h, s, d], device := dev }) := do
  q.checkedReplacement? (← RawTensor.branchingAttentionBias b h s d q.raw k.raw v.raw bias.raw false)

def attentionBias {biasDType : DType} {b h s d : UInt64} {dev : Device} (q k v : Tensor { dtype := dtype, shape := #[b, h, s, d], device := dev })
    (bias : Tensor { dtype := biasDType, shape := #[b, h, s, s], device := dev }) : Tensor { dtype := dtype, shape := #[b, h, s, d], device := dev } :=
  match attentionBias? q k v bias with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"BranchingFlows.attentionBias: {message}"

def attentionBiasIO {biasDType : DType} {b h s d : UInt64} {dev : Device} (q k v : Tensor { dtype := dtype, shape := #[b, h, s, d], device := dev })
    (bias : Tensor { dtype := biasDType, shape := #[b, h, s, s], device := dev })
    (dropout : Float) (training : Bool) (causal : Bool := false) :
    IO (Tensor { dtype := dtype, shape := #[b, h, s, d], device := dev }) := do
  let effective ← IO.ofExcept (Tensor.attentionDropout? dropout training)
  let raw ← RawTensor.branchingAttentionBiasIO b h s d q.raw k.raw v.raw bias.raw
    effective causal
  IO.ofExcept (q.checkedReplacement? raw)

def fullDynamic (shape : Shape) (value : Float) (requiresGrad : Bool := false) (device : Device := .CPU) (noGrad : requiresGrad = false := by decide) (dtype : DType := .Float32)
    [DType.ConstantSupported dtype device] [DType.RealFloating dtype] : Tensor { dtype := dtype, shape := shape, device } :=
  Tensor.addScalar (zerosDynamic shape requiresGrad device noGrad dtype) value

def binaryCrossEntropySum {s : Shape} {dev : Device} (input target weight : Tensor { dtype := dtype, shape := s, device := dev }) : Tensor { dtype := dtype, shape := #[], device := dev } :=
  match Tensor.ofRaw (RawTensor.branchingBCE s input.raw target.raw (some weight.raw) "sum") with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"BranchingFlows binary cross entropy: {message}"

end torch.branching
