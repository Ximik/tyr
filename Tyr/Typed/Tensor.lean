import Tyr.RawTensor
import Tyr.Typed.Device

/-!
# Typed tensor facade

`Tensor σ` retains shape, dtype and actual placement. A runtime `Device` value
can be an index; unknown placement is packaged existentially, never erased.
Only `Unknown "dynamic"` relaxes the dtype claim. Checked admission validates
all indices against native metadata. The indexed carrier has no allocation or
public unchecked constructor; FFI adapters use the explicit `.raw` projection.
-/

namespace torch

namespace StaticSpec

/-- Shape and dtype metadata without a placement field. -/
@[reducible] def toTensorSpec (σ : StaticSpec) : TensorSpec :=
  { shape := σ.shape, dtype := σ.dtype }

@[reducible] def withShape (σ : StaticSpec) (shape : Shape) : StaticSpec :=
  { σ with shape }

@[reducible] def withDType (σ : StaticSpec) (dtype : DType) : StaticSpec :=
  { σ with dtype }

/-- A CPU scalar follows the other operand's placement; all other left
operands keep their placement. This selects an output device, assuming
callers separately check static and actual device compatibility. -/
@[reducible] def pointwiseDevice (l r : StaticSpec) : Device :=
  match l.shape.toList with
  | _ :: _ => l.device
  | [] =>
      match l.device with
      | .CPU => r.device
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

/-- The full index, including actual placement. -/
def staticSpec {σ : StaticSpec} (_t : Tensor σ) : StaticSpec :=
  σ

def dtype {σ : StaticSpec} (_t : Tensor σ) : DType :=
  σ.dtype

def shape {σ : StaticSpec} (_t : Tensor σ) : Shape :=
  σ.shape

/-- Placement retained by the tensor index. Raw admission checks it exactly. -/
@[reducible] def device {σ : StaticSpec} (_t : Tensor σ) : Device :=
  σ.device

/-- Native placement, useful for independent boundary validation. -/
def runtimeDevice {σ : StaticSpec} (t : Tensor σ) : Device :=
  t.raw.device

/-- Actual native shape, as distinct from the static `shape` index. -/
def runtimeShape {σ : StaticSpec} (t : Tensor σ) : Shape :=
  t.raw.runtimeShape

/-- Independently validate native placement before binary pointwise execution. -/
def checkPointwise {σ τ : StaticSpec} (left : Tensor σ) (right : Tensor τ) :
    Except String Unit := do
  let _ ← Device.pointwiseResult? left.runtimeShape left.runtimeDevice
    right.runtimeShape right.runtimeDevice
  pure ()

/-- Check actual native placement for operations without the CPU scalar
exception, such as matrix multiplication and concatenation. -/
def checkSameDevice {σ τ : StaticSpec} (left : Tensor σ) (right : Tensor τ) :
    Except String Unit := do
  let _ ← Device.sameResult? left.runtimeDevice right.runtimeDevice
  pure ()

/-- Validate the selection-specific scalar relocation rule before native
`where` execution. Placement is selected in condition, left, right order. -/
def checkSelection {σ τ υ : StaticSpec} (condition : Tensor σ)
    (left : Tensor τ) (right : Tensor υ) : Except String Unit := do
  let _ ← Device.selectionResult? condition.runtimeShape condition.runtimeDevice
    left.runtimeShape left.runtimeDevice right.runtimeShape right.runtimeDevice
  pure ()

/-- Compare runtime-selected indices and return equality evidence, so unpacked
tensors can enter operations that require one shared placement. -/
def sameDevice? {σ τ : StaticSpec} (_left : Tensor σ) (_right : Tensor τ) :
    Except String (PLift (σ.device = τ.device)) :=
  if h : σ.device = τ.device then .ok ⟨h⟩
  else .error s!"Expected the same device, got {σ.device.render} and {τ.device.render}"

/-- Recover a requested placement index by checking equality, with no transfer. -/
def atDevice? {σ : StaticSpec} (tensor : Tensor σ) (device : Device) :
    Except String (Tensor { σ with device }) :=
  if h : σ.device = device then .ok (h ▸ tensor)
  else .error s!"Expected device {device.render}, got {σ.device.render}"

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
  expectedDevice.check raw.device
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
  original.raw.device.check raw.device
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
    Option (Tensor { shape, dtype, device := raw.device }) :=
  if raw.runtimeShape == shape && raw.dtype == dtype then
    (ofRaw raw).toOption
  else
    none

def ofTensor {shape : Shape} (dtype : DType) (raw : RawTensor) :
    Except String (Tensor { shape, dtype, device := raw.device }) := do
  TensorSpec.checkShape shape raw.runtimeShape
  TensorSpec.checkDType dtype raw.dtype
  ofRaw raw

def ofTensorWithContract {shape : Shape} (contract : TensorContract) (raw : RawTensor) :
    Except String (Tensor { shape, dtype := contract.spec.dtype, device := raw.device }) := do
  let actual : TensorSpec := { shape := raw.runtimeShape, dtype := raw.dtype }
  contract.check actual raw.device
  if contract.spec.shape = shape then
    ofRaw raw
  else
    .error s!"Expected static shape {reprStr contract.spec.shape}, got {reprStr shape}"

/-- Package a tensor with its full static specification, preserving actual placement. -/
def toSome {σ : StaticSpec} (tensor : Tensor σ) : Sigma Tensor :=
  ⟨σ, tensor⟩

/-- Recover the native dtype as an existential index without casting values or
forgetting shape or placement. The resulting tensor passes the checked raw
boundary, including when the input carries the dynamic dtype marker. -/
def reifyDType {σ : StaticSpec} (tensor : Tensor σ) :
    Except String (Sigma fun dtype : DType => Tensor { σ with dtype }) := do
  let dtype := tensor.raw.dtype
  let typed ← ofRaw (σ := { σ with dtype }) tensor.raw
  pure ⟨dtype, typed⟩

end Tensor

end torch
