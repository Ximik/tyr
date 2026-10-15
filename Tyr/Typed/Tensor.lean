import Tyr.RawTensor
import Tyr.Typed.Device

/-!
# Typed tensor facade

`Tensor (σ : StaticSpec)` is a lightweight typed view over Tyr's existing
`RawTensor`, indexed by shape, dtype, and device policy. The index is
preserved during typechecking: a `Tensor σ` contains one native handle,
so the carrier adds no allocation. Checked construction reads native metadata;
the FFI representation is unchanged.

Grading: `dtype` and `device` have "dynamic" defaults, so
`Tensor {shape := s}` claims only the shape, while
`Tensor {shape := s, dtype := .Float32, device := .exact (.CUDA 0)}` pins
everything. `Tensor {shape := s, dtype := d}` pins shape and dtype while
leaving the device unconstrained.

Checked constructors validate runtime metadata when crossing from raw
tensors into the typed facade. There is no public unchecked conversion or
implicit coercion back to `RawTensor`; native adapters use the explicit `.raw`
projection. Only the reserved default dtype `Unknown "dynamic"` makes no dtype
claim; other unknown tokens must match exactly. The `.any` device policy makes
no device claim.
-/

namespace torch

namespace StaticSpec

/-- Forget the device policy (runtime-metadata view). -/
@[reducible] def toTensorSpec (σ : StaticSpec) : TensorSpec :=
  { shape := σ.shape, dtype := σ.dtype }

@[reducible] def withShape (σ : StaticSpec) (shape : Shape) : StaticSpec :=
  { σ with shape }

@[reducible] def withDType (σ : StaticSpec) (dtype : DType) : StaticSpec :=
  { σ with dtype }

/-- A CPU scalar follows the other operand's placement; all other left
operands keep their placement policy. This selects an output policy, assuming
callers separately check static and actual device compatibility. -/
@[reducible] def pointwiseDevice (l r : StaticSpec) : DevicePolicy :=
  match l.shape.toList with
  | _ :: _ => l.device
  | [] =>
      match l.device with
      | .exact .CPU => r.device
      | device => device

/-- Result spec of a broadcasting pointwise op, including the weaker dtype
promotion of zero-dimensional tensors. Dtype promotion must be witnessed;
shape compatibility remains a separate precondition. -/
@[reducible] def pointwise (l r : StaticSpec) {result : DType}
    [TensorSpec.Promotes l.toTensorSpec r.toTensorSpec result] : StaticSpec :=
  { shape := TensorSpec.broadcastShape l.shape r.shape
    dtype := result
    device := l.pointwiseDevice r }

/-- Result spec of true division: pointwise, then floated. -/
@[reducible] def division (l r : StaticSpec) {result : DType}
    [TensorSpec.Promotes l.toTensorSpec r.toTensorSpec result] : StaticSpec :=
  { l.pointwise r with dtype := result.atLeastFloat }

/-- Result spec of a full reduction with sum accumulation rules. -/
@[reducible] def sumSpec (σ : StaticSpec) : StaticSpec :=
  { σ with shape := #[], dtype := σ.dtype.sumResult }

end StaticSpec

namespace Tensor

def toTensor {σ : StaticSpec} (t : Tensor σ) : RawTensor :=
  t.raw

/-- The static spec, as runtime metadata (shape + dtype). -/
def spec {σ : StaticSpec} (_t : Tensor σ) : TensorSpec :=
  σ.toTensorSpec

/-- The full static spec, including the device policy. -/
def staticSpec {σ : StaticSpec} (_t : Tensor σ) : StaticSpec :=
  σ

def dtype {σ : StaticSpec} (_t : Tensor σ) : DType :=
  σ.dtype

def shape {σ : StaticSpec} (_t : Tensor σ) : Shape :=
  σ.shape

/-- Actual native device, including for an unrestricted placement policy. -/
def device {σ : StaticSpec} (t : Tensor σ) : Device :=
  t.raw.device

/-- Actual native shape, as distinct from the static `shape` index. -/
def runtimeShape {σ : StaticSpec} (t : Tensor σ) : Shape :=
  t.raw.runtimeShape

/-- Check actual native placement before binary pointwise execution, including
for two tensors whose static policies are both `.any`. -/
def checkPointwise {σ τ : StaticSpec} (left : Tensor σ) (right : Tensor τ) :
    Except String Unit := do
  let _ ← Device.pointwiseResult? left.runtimeShape left.device
    right.runtimeShape right.device
  pure ()

/-- Check actual native placement for operations without the CPU scalar
exception, such as matrix multiplication and concatenation. -/
def checkSameDevice {σ τ : StaticSpec} (left : Tensor σ) (right : Tensor τ) :
    Except String Unit := do
  let _ ← Device.sameResult? left.device right.device
  pure ()

/-- Validate the selection-specific scalar relocation rule before native
`where` execution. Placement is selected in condition, left, right order. -/
def checkSelection {σ τ υ : StaticSpec} (condition : Tensor σ)
    (left : Tensor τ) (right : Tensor υ) : Except String Unit := do
  let _ ← Device.selectionResult? condition.runtimeShape condition.device
    left.runtimeShape left.device right.runtimeShape right.device
  pure ()

/-- Runtime metadata actually carried by the handle. -/
def actualSpec {σ : StaticSpec} (t : Tensor σ) : TensorSpec :=
  { shape := t.raw.runtimeShape, dtype := t.raw.dtype }

/-- Validate a value update using metadata captured before consuming its input.
The original tensor is deliberately absent: retaining it through result
validation would prevent native ownership transfer and storage reuse.
Dynamic specifications still preserve the actual dtype and placement. -/
def checkedUpdate? {σ : StaticSpec} (expectedDType : DType) (expectedDevice : Device)
    (raw : RawTensor) : Except String (Tensor σ) := do
  TensorSpec.checkDType expectedDType raw.dtype
  (DevicePolicy.exact expectedDevice).check raw.device
  ofRaw raw

/-- Pure value-update adapter. Its panic fallback is the valid default for the
static specification, so the adapter never retains the consumed input. -/
def checkedUpdate {σ : StaticSpec} (expectedDType : DType) (expectedDevice : Device)
    (raw : RawTensor) : Tensor σ :=
  match checkedUpdate? expectedDType expectedDevice raw with
  | .ok tensor => tensor
  | .error message => panic! s!"Tensor update: {message}"

/-- Check a replacement against both the static index and the original native
dtype/device. Dynamic index components must not hide a metadata change in an
operation that promises to preserve the tensor's specification. -/
def checkedReplacement? {σ : StaticSpec} (original : Tensor σ) (raw : RawTensor) :
    Except String (Tensor σ) := do
  TensorSpec.checkDType original.raw.dtype raw.dtype
  (DevicePolicy.exact original.raw.device).check raw.device
  ofRaw raw

/-- Native adapter for operations whose output must preserve an existing
specification. A metadata mismatch is a runtime error; the original tensor is
the well-typed fallback used by Lean's panic mechanism. -/
def checkedReplacement {σ : StaticSpec} (original : Tensor σ) (raw : RawTensor) : Tensor σ :=
  letI : Inhabited (Tensor σ) := ⟨original⟩
  match original.checkedReplacement? raw with
  | .ok tensor => tensor
  | .error message => panic! s!"Tensor replacement: {message}"

def ofTensor? {shape : Shape} (dtype : DType) (raw : RawTensor) :
    Option (Tensor { shape, dtype }) :=
  if raw.runtimeShape == shape && raw.dtype == dtype then
    (ofRaw raw).toOption
  else
    none

def ofTensor {shape : Shape} (dtype : DType) (raw : RawTensor) :
    Except String (Tensor { shape, dtype }) := do
  TensorSpec.checkShape shape raw.runtimeShape
  TensorSpec.checkDType dtype raw.dtype
  ofRaw raw

def ofTensorWithContract {shape : Shape} (contract : TensorContract) (raw : RawTensor) :
    Except String (Tensor { shape, dtype := contract.spec.dtype }) := do
  let actual : TensorSpec := { shape := raw.runtimeShape, dtype := raw.dtype }
  contract.check actual raw.device
  if contract.spec.shape = shape then
    ofRaw raw
  else
    .error s!"Expected static shape {reprStr contract.spec.shape}, got {reprStr shape}"

/-- Package a tensor with its full static specification, preserving the device policy. -/
def toSome {σ : StaticSpec} (tensor : Tensor σ) : Sigma Tensor :=
  ⟨σ, tensor⟩

/-- Recover the native dtype as an existential index without casting values or
forgetting shape/device policy. The resulting tensor passes the checked raw
boundary, including when the input carries the dynamic dtype marker. -/
def reifyDType {σ : StaticSpec} (tensor : Tensor σ) :
    Except String (Sigma fun dtype : DType => Tensor { σ with dtype }) := do
  let dtype := tensor.raw.dtype
  let typed ← ofRaw (σ := { σ with dtype }) tensor.raw
  pure ⟨dtype, typed⟩

end Tensor

end torch
