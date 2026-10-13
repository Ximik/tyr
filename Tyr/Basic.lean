import Tyr.Basic.Types

/-!
# Tyr.Basic

`Tyr.Basic` defines Tyr's foundational tensor-level types and pure shape logic.
It is the lowest-level shared layer used across runtime bindings, model code, and utilities.

## Major Components

- Core types (in `Tyr.Basic.Types`): `Shape`, `DType`, `Device`, the native
  `RawTensor` carrier, `DevicePolicy`, `StaticSpec`, and the indexed `Tensor`,
  with DType parsing and pure shape-transform utilities — no native functions,
  so the GPU code generator can use them without linking `libTyrC`.
- Tensor metadata extraction hooks (`runtimeShape`, dtype/device introspection,
  stats/values), implemented in `libTyrC` (this module).

## Scope

This module is intentionally kernel-agnostic.
Typed numerical operations live in `Tyr.Typed.Ops`; native adapters live in
`Tyr.RawTensor`. Legacy shape-erasing declarations are isolated in
`Tyr.Legacy.Tensor`.
-/

namespace torch

namespace RawTensor

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

namespace Tensor

instance {σ : StaticSpec} : ToString (Tensor σ) where
  toString t := RawTensor.toString t.raw

instance {σ : StaticSpec} : Repr (Tensor σ) where
  reprPrec t _ := RawTensor.toString t.raw

end Tensor

end torch
