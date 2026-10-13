import Tyr.Basic

/-! Compatibility declarations for code still using the shape-erasing `T` alias.
New code should import the typed API. This module is removed when its remaining
callers have migrated; it must not be imported by the typed tensor foundation. -/

namespace torch

/-- Legacy shape annotation over a raw handle. The index is NOT enforced.
Pending migration to `Tensor`; do not use this alias in new model APIs. -/
def T (_ : Shape) : Type := RawTensor

instance (s : Shape) : Nonempty (T s) := TSpec.property
instance {s : Shape} : ToString (T s) where
  toString t := RawTensor.toString t
instance {s : Shape} : Repr (T s) where
  reprPrec t _ := RawTensor.toString t

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

def T.shape {s : Shape} (_t : T s) : Shape := s

end torch
