import Tyr.Core.Typed.DType
import Tyr.Core.Typed.Spec
import Tyr.Core.Typed.Device
import Tyr.Core.Typed.Tensor
import Tyr.Core.Typed.Ops

/-!
# Tyr.Core.Typed

LeanMLX-inspired typed facade for Tyr.

The public tensor type is `Tensor σ`, backed by a native `RawTensor` handle.
Shape, dtype, and device specifications survive typechecking. Native operations
are isolated in `Tyr.Core.FFI.RawTensor`; checked constructors validate external handles.
-/
