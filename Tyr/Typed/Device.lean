import Tyr.Typed.Spec

/-!
# Runtime placement contracts

`StaticSpec.device` is an actual placement, possibly selected at runtime.
`TensorContract` adds runtime placement checks to shape/dtype metadata for
boundaries such as file loading and external native handles.
-/

namespace torch

namespace Device

def render : Device → String
  | .CPU => "CPU"
  | .MPS => "MPS"
  | .CUDA idx => s!"CUDA:{idx}"

/-- Validate actual placement for operations that require one device.
The check is independent of the typed indices and also guards raw boundaries. -/
def sameResult? (left right : Device) : Except String Device :=
  if left == right then
    .ok left
  else
    .error s!"Expected the same device, got {left.render} and {right.render}"

/-- Validate binary pointwise placement and determine the result device. PyTorch
allows a rank-zero CPU operand on another operand's device. A one-element
vector is not a scalar, and accelerator scalars do not get this exception. -/
def pointwiseResult? (leftShape : Shape) (left : Device)
    (rightShape : Shape) (right : Device) : Except String Device :=
  if left == right then
    .ok left
  else if left == .CPU && leftShape.isEmpty then
    .ok right
  else if right == .CPU && rightShape.isEmpty then
    .ok left
  else
    .error s!"Incompatible pointwise devices: {left.render} at shape {leftShape} and {right.render} at shape {rightShape}; only rank-zero CPU operands may cross devices"

/-- Placement for PyTorch `where`: select the first non-CPU device in condition,
left, right order, or CPU if all inputs are on CPU. Rank-zero operands on other
devices are relocated by the native operation; non-scalars must already be on
the selected device. Unlike binary arithmetic, this also permits relocation of
an accelerator scalar. This checks placement only, not broadcasting or dtype. -/
def selectionResult? (conditionShape : Shape) (condition : Device)
    (leftShape : Shape) (left : Device) (rightShape : Shape) (right : Device) :
    Except String Device := do
  let result := if condition != .CPU then condition else if left != .CPU then left else right
  for (label, shape, device) in #[
      ("condition", conditionShape, condition), ("left", leftShape, left),
      ("right", rightShape, right)] do
    if device != result && !(result != .CPU && shape.isEmpty) then
      throw s!"Incompatible selection device: {label} is {device.render} at shape {shape}, expected {result.render}; only rank-zero operands may move to the selected device"
  pure result

/-- Equality evidence for a same-device operation. Runtime-selected devices can
be related by checking equality before entering a shared-device operation. -/
class SameDeviceCompatible (left right : Device) : Prop where
  valid : left = right

instance sameDevice (device : Device) : SameDeviceCompatible device device := ⟨rfl⟩

/-- Binary pointwise placement is shared placement or a rank-zero CPU operand. -/
def pointwiseAllowed (left right : StaticSpec) : Prop :=
  left.device = right.device ∨
    (left.shape = #[] ∧ left.device = .CPU) ∨
    (right.shape = #[] ∧ right.device = .CPU)

/-- Erased evidence of valid binary pointwise placement. There is no wildcard
instance: distinct runtime devices need equality evidence or a CPU scalar. -/
class PointwiseCompatible (left right : StaticSpec) : Prop where
  valid : pointwiseAllowed left right

instance compatiblePointwiseDevices (left right : StaticSpec)
    [h : SameDeviceCompatible left.device right.device] :
    PointwiseCompatible left right := ⟨Or.inl h.valid⟩

instance cpuScalarLeft (dtype : DType) (right : StaticSpec) :
    PointwiseCompatible ⟨#[], dtype, .CPU⟩ right :=
  ⟨Or.inr (Or.inl ⟨rfl, rfl⟩)⟩

instance cpuScalarRight (left : StaticSpec) (dtype : DType) :
    PointwiseCompatible left ⟨#[], dtype, .CPU⟩ :=
  ⟨Or.inr (Or.inr ⟨rfl, rfl⟩)⟩

/-- Check placement at a native admission or conversion boundary. -/
def check (expected actual : Device) : Except String Unit :=
  if actual == expected then .ok ()
  else .error s!"Expected device {expected.render}, got {actual.render}"

end Device

inductive TensorRole where
  | activation
  | parameter
  | index
  | mask
  | logits
  | loss
  | cache
  | quantScale
deriving Repr, BEq, DecidableEq

structure TensorContract where
  spec : TensorSpec
  role : TensorRole := .activation
  expectedDevice : Option Device := none
deriving Repr, BEq

namespace TensorContract

def check (contract : TensorContract) (actualSpec : TensorSpec) (actualDevice : Device) :
    Except String Unit := do
  TensorSpec.checkCompatible contract.spec actualSpec
  if let some expected := contract.expectedDevice then
    expected.check actualDevice

end TensorContract

end torch
