import Tyr.RawTensor
import Tyr.Typed.Device

/-!
# Typed tensor facade

`Tensor (σ : StaticSpec)` is a lightweight typed view over Tyr's existing
`RawTensor`, indexed by shape, dtype, and device policy. The index is
preserved during typechecking: a `Tensor σ` contains one native handle,
so wrappers cost nothing at runtime and the FFI surface is
untouched.

Grading: `dtype` and `device` have "dynamic" defaults, so
`Tensor {shape := s}` claims only the shape, while
`Tensor {shape := s, dtype := .Float32, device := .exact (.CUDA 0)}` pins
everything. `DTensor shape dtype` abbreviates the shape+dtype instantiation.

Checked constructors validate runtime metadata when crossing from raw
tensors into the typed facade; `assumeSpec`/`assumeDType` are available for
code that already established the invariant externally (each such use
encodes an assumption about libtorch behavior — keep them covered by the
parity tests in `Tests/TestTyped.lean`).
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

/-- Result spec of a broadcasting pointwise op. The device policy of the
    left operand propagates. -/
@[reducible] def pointwise (l r : StaticSpec) : StaticSpec :=
  { shape := TensorSpec.broadcastShape l.shape r.shape
    dtype := DType.promote l.dtype r.dtype
    device := l.device }

/-- Result spec of true division: pointwise, then floated. -/
@[reducible] def division (l r : StaticSpec) : StaticSpec :=
  { l.pointwise r with dtype := (DType.promote l.dtype r.dtype).atLeastFloat }

/-- Result spec of a full reduction with sum accumulation rules. -/
@[reducible] def sumSpec (σ : StaticSpec) : StaticSpec :=
  { σ with shape := #[], dtype := σ.dtype.sumResult }

end StaticSpec

abbrev SomeDTensor :=
  Sigma fun spec : TensorSpec => DTensor spec.shape spec.dtype

namespace Tensor

/-- Compat alias for `assumeSpec` (the shape+dtype reading). -/
abbrev assumeDType {σ : StaticSpec} (raw : RawTensor) : Tensor σ :=
  assumeSpec raw

def toTensor {σ : StaticSpec} (t : Tensor σ) : RawTensor :=
  t.raw

instance {σ : StaticSpec} : CoeOut (Tensor σ) (RawTensor) where
  coe := toTensor

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

/-- Actual native shape, as distinct from the static `shape` index. -/
def runtimeShape {σ : StaticSpec} (t : Tensor σ) : Shape :=
  t.raw.runtimeShape

/-- Runtime metadata actually carried by the handle. -/
def actualSpec {σ : StaticSpec} (t : Tensor σ) : TensorSpec :=
  { shape := t.raw.runtimeShape, dtype := t.raw.dtype }

/-- Validate every phantom claim — shape, dtype, and device policy —
    against the runtime handle. Graded components make no claim at their
    dynamic level: an `Unknown` dtype matches any runtime dtype, and the
    `.any` device policy accepts any placement. Used by tests and debug
    assertions at trust boundaries. -/
def validate {σ : StaticSpec} (t : Tensor σ) : Except String Unit := do
  TensorSpec.checkShape σ.shape t.raw.runtimeShape
  match σ.dtype with
  | .Unknown _ => pure ()
  | d => TensorSpec.checkDType d t.raw.dtype
  σ.device.check t.raw.device

/-- Validate a native handle against every component of the destination index. -/
def ofRaw {σ : StaticSpec} (raw : RawTensor) : Except String (Tensor σ) := do
  let tensor : Tensor σ := assumeSpec raw
  tensor.validate
  pure tensor

/-- Native adapter for operations whose output must preserve an existing
specification. A metadata mismatch is a runtime error; the original tensor is
the well-typed fallback used by Lean's panic mechanism. -/
def checkedReplacement {σ : StaticSpec} (original : Tensor σ) (raw : RawTensor) : Tensor σ :=
  letI : Inhabited (Tensor σ) := ⟨original⟩
  match ofRaw raw with
  | .ok tensor => tensor
  | .error message => panic! s!"Tensor replacement: {message}"

def ofTensor? {shape : Shape} (dtype : DType) (raw : RawTensor) : Option (DTensor shape dtype) :=
  if raw.runtimeShape == shape && raw.dtype == dtype then
    some (assumeSpec raw)
  else
    none

def ofTensor {shape : Shape} (dtype : DType) (raw : RawTensor) : Except String (DTensor shape dtype) := do
  TensorSpec.checkShape shape raw.runtimeShape
  TensorSpec.checkDType dtype raw.dtype
  pure (assumeSpec raw)

def ofTensorWithContract {shape : Shape} (contract : TensorContract) (raw : RawTensor) :
    Except String (DTensor shape contract.spec.dtype) := do
  let actual : TensorSpec := { shape := raw.runtimeShape, dtype := raw.dtype }
  contract.check actual raw.device
  if h : contract.spec.shape = shape then
    pure (h ▸ assumeSpec raw)
  else
    .error s!"Expected static shape {reprStr contract.spec.shape}, got {reprStr shape}"

def toSome {shape : Shape} {dtype : DType} (tensor : DTensor shape dtype) : SomeDTensor :=
  ⟨{ shape, dtype }, tensor⟩

end Tensor

end torch
