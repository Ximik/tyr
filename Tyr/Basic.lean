import Tyr.Basic.Types

/-!
# Tyr.Basic

`Tyr.Basic` defines Tyr's foundational tensor-level types and pure shape logic.
It is the lowest-level shared layer used across runtime bindings, model code, and utilities.

## Major Components

- Core types (in `Tyr.Basic.Types`): `Shape`, `DType`, `Device`, the native
  `RawTensor` carrier, `DevicePolicy`, and `StaticSpec`,
  with DType parsing and pure shape-transform utilities — no native functions,
  so the GPU code generator can use them without linking `libTyrC`.
- Tensor metadata extraction hooks (`runtimeShape`, dtype/device introspection,
  stats/values), implemented in `libTyrC` (this module).
- The indexed `Tensor`, checked raw construction, and a trusted fresh zero
  factory. An opaque carrier prevents structural construction and reindexing,
  while GPU code generation continues to import only `Basic.Types`.

## Scope

This module is intentionally kernel-agnostic.
Typed numerical operations live in `Tyr.Typed.Ops`; native adapters live in
`Tyr.RawTensor`. Public numerical APIs use indexed `Tensor` values.
-/

namespace torch

namespace RawTensor

@[extern "lean_torch_zeros"] opaque zeros (shape : Shape) (requiresGrad : Bool := false)
    (device : Device := .CPU) : RawTensor

@[extern "lean_torch_cast_dtype"] opaque castDType (shape : Shape) (t : @& RawTensor)
    (tag : @& String) : RawTensor

-- Preserve the existing C ABI's shape argument without exposing a shape-indexed
-- raw type. Metadata queries inspect the native tensor, never this argument.
@[extern "lean_torch_to_string"] private opaque toStringImpl
    (shape : Shape) (t : @& RawTensor) : String

def toString (t : RawTensor) : String := toStringImpl #[] t

instance : ToString RawTensor := ⟨RawTensor.toString⟩
instance : Repr RawTensor where
  reprPrec t _ := RawTensor.toString t

@[extern "lean_torch_get_shape"] private opaque runtimeShapeImpl
    (shape : Shape) (t : @& RawTensor) : Shape
@[extern "lean_torch_get_dtype"] private opaque dtypeStrImpl
    (shape : Shape) (t : @& RawTensor) : String
@[extern "lean_torch_get_device_enum"] private opaque deviceImpl
    (shape : Shape) (t : @& RawTensor) : Device
@[extern "lean_torch_get_device"] private opaque deviceStrImpl
    (shape : Shape) (t : @& RawTensor) : String
@[extern "lean_torch_get_values"] private opaque getValuesImpl
    (shape : Shape) (t : @& RawTensor) (maxElements : UInt64) : FloatArray
@[extern "lean_torch_get_stats"] private opaque statsImpl
    (shape : Shape) (t : @& RawTensor) : String

/-- Native metadata, independent of any static tensor specification. -/
def runtimeShape (t : RawTensor) : Shape := runtimeShapeImpl #[] t
def dtypeStr (t : RawTensor) : String := dtypeStrImpl #[] t
def dtype (t : RawTensor) : DType := DType.parse t.dtypeStr
def device (t : RawTensor) : Device := deviceImpl #[] t
def deviceStr (t : RawTensor) : String := deviceStrImpl #[] t
def getValues (t : RawTensor) (maxElements : UInt64 := 1000) : FloatArray :=
  getValuesImpl #[] t maxElements
def stats (t : RawTensor) : String := statsImpl #[] t

end RawTensor

-- Keep the representation opaque even to constructor/cases/unfold tactics.
-- The body establishes nonemptiness without adding an axiom; it does not make
-- different specification indices definitionally equal in the kernel.
private opaque TensorCarrier (_σ : StaticSpec) : NonemptyType :=
  ⟨RawTensor, inferInstance⟩

/-- Tensor indexed by a static specification. Safe code admits native handles
through `Tensor.ofRaw`; the carrier has no constructor or structural recursor.
At runtime it is the native handle itself, with no additional allocation. -/
def Tensor (σ : StaticSpec) : Type := (TensorCarrier σ).type

instance {σ : StaticSpec} : Nonempty (Tensor σ) := (TensorCarrier σ).property

namespace Tensor

-- These casts implement the native representation, not kernel proofs. They are
-- unsafe so even discovering their private names cannot provide an unchecked
-- constructor to safe Lean code. Public opaque declarations have safe logical
-- bodies and use these implementations only during execution.
private unsafe def rawImpl {σ : StaticSpec} (tensor : Tensor σ) : RawTensor :=
  unsafeCast tensor

/-- Explicitly expose the native handle, which carries no static claims. -/
@[implemented_by rawImpl]
opaque raw {σ : StaticSpec} (tensor : Tensor σ) : RawTensor :=
  Classical.choice inferInstance

private def renderDevice : Device → String
  | .CPU => "CPU"
  | .MPS => "MPS"
  | .CUDA index => s!"CUDA:{index}"

private def checkSpec (σ : StaticSpec) (raw : RawTensor) : Except String Unit := do
  if raw.runtimeShape != σ.shape then
    throw s!"Expected shape {reprStr σ.shape}, got {reprStr raw.runtimeShape}"
  if σ.dtype != .Unknown "dynamic" && raw.dtype != σ.dtype then
    throw s!"Expected dtype {σ.dtype}, got {raw.dtype}"
  match σ.device with
  | .any => pure ()
  | .exact expected =>
      if raw.device != expected then
        throw s!"Expected device {renderDevice expected}, got {renderDevice raw.device}"

/-- Check shape, dtype and device policy against the native handle. Only the
reserved `Unknown "dynamic"` dtype makes no dtype claim; any other unknown
token must match exactly. The `.any` device policy makes no placement claim. -/
def validate {σ : StaticSpec} (tensor : Tensor σ) : Except String Unit :=
  checkSpec σ tensor.raw

private unsafe def ofRawImpl {σ : StaticSpec} (raw : RawTensor) :
    Except String (Tensor σ) := do
  checkSpec σ raw
  pure (unsafeCast raw)

/-- The raw-to-indexed boundary. Construct a tensor only after validating every
claim in the destination specification against native metadata. -/
@[implemented_by ofRawImpl]
opaque ofRaw {σ : StaticSpec} (raw : RawTensor) : Except String (Tensor σ) :=
  .error "Tensor.ofRaw requires its checked runtime implementation"

private unsafe def zerosForImpl (σ : StaticSpec) : Tensor σ :=
  let device := match σ.device with
    | .any => Device.CPU
    | .exact device => device
  let raw := RawTensor.zeros σ.shape false device
  let raw := match σ.dtype.safeTensorTag? with
    | some tag => RawTensor.castDType σ.shape raw tag
    | none =>
        if σ.dtype == DType.Unknown "dynamic" then raw
        else RawTensor.castDType σ.shape raw σ.dtype.canonicalName
  unsafeCast raw

/-- Trusted native allocation boundary used to bootstrap pure typed factories.
It accepts a specification, never a foreign handle: libtorch creates fresh zeros
on the requested device and casts them to the requested concrete dtype. Native
allocation/cast failures throw; `Unknown "dynamic"` uses the native default and
`.any` placement chooses CPU. Other unknown dtype tokens go through native dtype
parsing and fail instead of silently defaulting. This is the sole unchecked
native construction boundary.
-/
@[implemented_by zerosForImpl]
opaque zerosFor (σ : StaticSpec) : Tensor σ := Classical.choice inferInstance

instance {σ : StaticSpec} : Inhabited (Tensor σ) := ⟨zerosFor σ⟩

private unsafe def forgetDeviceImpl {σ : StaticSpec} (tensor : Tensor σ) :
    Tensor { σ with device := .any } :=
  unsafeCast tensor

/-- Weaken placement without changing the native handle, shape or dtype. -/
@[implemented_by forgetDeviceImpl]
opaque forgetDevice {σ : StaticSpec} (tensor : Tensor σ) :
    Tensor { σ with device := .any } :=
  Classical.choice inferInstance

instance {σ : StaticSpec} : ToString (Tensor σ) where
  toString t := RawTensor.toString t.raw

instance {σ : StaticSpec} : Repr (Tensor σ) where
  reprPrec t _ := RawTensor.toString t.raw

end Tensor

end torch
