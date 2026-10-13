import Tyr.Typed.DType
import Tyr.Typed.Spec
import Tyr.Typed.Device
import Tyr.Typed.Tensor
import Tyr.Typed.Ops
import Tyr.Typed.Layers

/-!
# Tyr.Typed

LeanMLX-inspired typed facade for Tyr.

The public tensor type is `Tensor σ`, backed by a native `RawTensor` handle.
Shape, dtype, and device specifications survive typechecking. Native operations
are isolated in `Tyr.RawTensor`; checked constructors validate external handles.
-/
