import Tyr.Typed.Spec

/-!
# Runtime placement contracts

`StaticSpec.device` carries either an exact placement or an unrestricted policy.
`TensorContract` adds runtime placement checks to shape/dtype metadata for
boundaries such as file loading and external native handles.
-/

namespace torch

namespace Device

def render : Device → String
  | .CPU => "CPU"
  | .MPS => "MPS"
  | .CUDA idx => s!"CUDA:{idx}"

/-- Validate actual placement for operations that require one device. Static
`.any` policies are deliberately absent: equality of policies is not equality
of the devices carried by native handles. -/
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

end Device

namespace DevicePolicy

/-- The policies contain no known conflict for a same-device operation. An
unrestricted policy still requires validation of the native device. -/
def sameAllowed (left right : DevicePolicy) : Prop :=
  left = .any ∨ right = .any ∨ left = right

/-- Static eligibility for a same-device operation, not a proof of actual
placement equality. Callers must also validate native metadata. -/
class SameDeviceCompatible (left right : DevicePolicy) : Prop where
  valid : sameAllowed left right

instance sameDevice (device : DevicePolicy) : SameDeviceCompatible device device :=
  ⟨Or.inr (Or.inr rfl)⟩

instance anyDeviceLeft (right : DevicePolicy) : SameDeviceCompatible .any right :=
  ⟨Or.inl rfl⟩

instance anyDeviceRight (left : DevicePolicy) : SameDeviceCompatible left .any :=
  ⟨Or.inr (Or.inl rfl)⟩

/-- Pointwise placement permits matching/unknown policies or a CPU scalar.
The latter is about rank, not the number of elements. -/
def pointwiseAllowed (left right : StaticSpec) : Prop :=
  sameAllowed left.device right.device ∨
    (left.shape = #[] ∧ left.device = .exact .CPU) ∨
    (right.shape = #[] ∧ right.device = .exact .CPU)

/-- Reject statically known pointwise placement conflicts. In particular,
`.any` does not bypass the runtime `Device.pointwiseResult?` check. -/
class PointwiseCompatible (left right : StaticSpec) : Prop where
  valid : pointwiseAllowed left right

instance compatiblePointwiseDevices (left right : StaticSpec)
    [h : SameDeviceCompatible left.device right.device] :
    PointwiseCompatible left right :=
  ⟨Or.inl h.valid⟩

instance cpuScalarLeft (dtype : DType) (right : StaticSpec) :
    PointwiseCompatible ⟨#[], dtype, .exact .CPU⟩ right :=
  ⟨Or.inr (Or.inl ⟨rfl, rfl⟩)⟩

instance cpuScalarRight (left : StaticSpec) (dtype : DType) :
    PointwiseCompatible left ⟨#[], dtype, .exact .CPU⟩ :=
  ⟨Or.inr (Or.inr ⟨rfl, rfl⟩)⟩

def check (policy : DevicePolicy) (actual : Device) : Except String Unit :=
  match policy with
  | .any => .ok ()
  | .exact expected =>
      if actual == expected then
        .ok ()
      else
        .error s!"Expected device {expected.render}, got {actual.render}"

end DevicePolicy

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
  devicePolicy : DevicePolicy := .any
deriving Repr, BEq

namespace TensorContract

def check (contract : TensorContract) (actualSpec : TensorSpec) (actualDevice : Device) :
    Except String Unit := do
  TensorSpec.checkCompatible contract.spec actualSpec
  contract.devicePolicy.check actualDevice

end TensorContract

end torch
