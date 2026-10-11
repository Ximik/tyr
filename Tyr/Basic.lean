import Tyr.Basic.Types

/-!
# Tyr.Basic

`Tyr.Basic` defines Tyr's foundational tensor-level types and pure shape logic.
It is the lowest-level shared layer used across runtime bindings, model code, and utilities.

## Major Components

- Core aliases/types: `Shape`, `DType`, `Device`, and opaque tensor carrier `T`,
  with DType parsing and pure shape-transform utilities (all in `Tyr.Basic.Types`).
- Tensor metadata extraction hooks (`runtimeShape`, dtype/device introspection,
  stats/values), implemented in `libTyrC`.

## Scope

This module is intentionally kernel-agnostic.
Numerical tensor operations and backend FFI bindings live in `Tyr.Torch`.
-/

namespace torch

@[extern "lean_torch_to_string"] opaque T.toString {s : Shape} (t : @& T s) : String
@[extern "lean_torch_tensor_print"] opaque T.print {s : Shape} (t : @& T s) : IO Unit

/-! ## Tensor Metadata Extraction (for visualization widgets) -/

/-- Get the runtime shape of a tensor (useful for widgets/visualization) -/
@[extern "lean_torch_get_shape"]
opaque T.runtimeShape {s : Shape} (t : @& T s) : Array UInt64

/-- Get the dtype of a tensor as a backend string token. -/
@[extern "lean_torch_get_dtype"]
opaque T.dtypeStr {s : Shape} (t : @& T s) : String

/-- Get the dtype of a tensor as the core `DType`. -/
def T.dtype {s : Shape} (t : @& T s) : DType :=
  DType.parse (t.dtypeStr)

/-- Get the device of a tensor as a Device enum -/
@[extern "lean_torch_get_device_enum"]
opaque T.device {s : Shape} (t : @& T s) : Device

/-- Get the device of a tensor as a string (e.g., "cpu", "cuda:0", "mps") -/
@[extern "lean_torch_get_device"]
opaque T.deviceStr {s : Shape} (t : @& T s) : String

/-- Get tensor values as a flat array of floats (up to maxElements).
    Values are converted to Float for uniform access. -/
@[extern "lean_torch_get_values"]
opaque T.getValues {s : Shape} (t : @& T s) (maxElements : UInt64 := 1000) : FloatArray

/-- Get tensor statistics as a JSON string: {min, max, mean, std} -/
@[extern "lean_torch_get_stats"]
opaque T.stats {s : Shape} (t : @& T s) : String

instance {s : Shape} : ToString (T s) where
  toString t := t.toString

def T.shape {s : Shape} (_t : T s) : Shape := s

instance {s : Shape} : Repr (T s) where
  reprPrec t _ := t.toString
