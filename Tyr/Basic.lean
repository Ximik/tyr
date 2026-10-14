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
  factory. Keeping its private constructor here prevents unchecked raw wrapping
  elsewhere, while GPU code generation continues to import only `Basic.Types`.

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

/-- Tensor indexed by a static specification. Normal construction uses the
checked native boundary in this module. Constructor privacy is an API convention,
not a proof of the native invariant: constructor tactics can still bypass it. -/
structure Tensor (σ : StaticSpec) where
  private mk ::
  raw : RawTensor

namespace Tensor

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

/-- The raw-to-indexed boundary. Construct a tensor only after validating every
claim in the destination specification against native metadata. -/
def ofRaw {σ : StaticSpec} (raw : RawTensor) : Except String (Tensor σ) := do
  checkSpec σ raw
  pure (.mk raw)

/-- Trusted native allocation boundary used to bootstrap pure typed factories.
It accepts a specification, never a foreign handle: libtorch creates fresh zeros
on the requested device and casts them to the requested concrete dtype. Native
allocation/cast failures throw; `Unknown "dynamic"` uses the native default and
`.any` placement chooses CPU. Other unknown dtype tokens go through native dtype
parsing and fail instead of silently defaulting. This is the sole unchecked
native construction boundary.
-/
def zerosFor (σ : StaticSpec) : Tensor σ :=
  let device := match σ.device with
    | .any => Device.CPU
    | .exact device => device
  let raw := RawTensor.zeros σ.shape false device
  let raw := match σ.dtype.safeTensorTag? with
    | some tag => RawTensor.castDType σ.shape raw tag
    | none =>
        if σ.dtype == .Unknown "dynamic" then raw
        else RawTensor.castDType σ.shape raw σ.dtype.canonicalName
  .mk raw

instance {σ : StaticSpec} : Inhabited (Tensor σ) := ⟨zerosFor σ⟩

/-- Weaken placement without changing the native handle, shape or dtype. -/
def forgetDevice {σ : StaticSpec} (tensor : Tensor σ) :
    Tensor { σ with device := .any } :=
  .mk tensor.raw

instance {σ : StaticSpec} : ToString (Tensor σ) where
  toString t := RawTensor.toString t.raw

instance {σ : StaticSpec} : Repr (Tensor σ) where
  reprPrec t _ := RawTensor.toString t.raw

end Tensor

end torch
