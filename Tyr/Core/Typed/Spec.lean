import Tyr.Core.Typed.Promotion
import Tyr.Core.Shape.Basic

/-!
# Typed tensor specs

A small, LeanMLX-inspired tensor spec layer for Tyr: shape plus dtype, with
pure result-spec algebra for common tensor operations.
-/

namespace torch

structure TensorSpec where
  shape : Shape
  dtype : DType
deriving Repr, BEq, DecidableEq

namespace TensorSpec

/-- Result dtype of a pointwise operation. A zero-dimensional operand next to a
non-scalar must have the same dtype: libtorch's weaker promotion for it is not
admitted. Unsupported dtype combinations have no result. -/
@[reducible] def pointwiseDType? (lhs rhs : TensorSpec) : Option DType :=
  match lhs.shape.toList, rhs.shape.toList with
  | [], _ :: _ | _ :: _, [] =>
      match DType.sameKnownDType lhs.dtype rhs.dtype with
      | true => DType.promote? lhs.dtype rhs.dtype
      | false => none
  | _, _ => DType.promote? lhs.dtype rhs.dtype

/-- Evidence of the pointwise dtype result.
This does not establish shape compatibility or native kernel availability. -/
class Promotes (lhs rhs : TensorSpec) (result : outParam DType) : Prop where
  valid : pointwiseDType? lhs rhs = some result

instance {lhs rhs : TensorSpec} {result : DType}
    [h : SomeResult (pointwiseDType? lhs rhs) result] : Promotes lhs rhs result :=
  ⟨h.valid⟩

/-- A pointwise dtype index backed by a successful promotion witness. -/
@[reducible] def pointwiseDType (lhs rhs : TensorSpec) {result : DType}
    [Promotes lhs rhs result] : DType := result

def checkShape (expected actual : Shape) : Except String Unit :=
  if actual == expected then
    .ok ()
  else
    .error s!"Expected shape {reprStr expected}, got {reprStr actual}"

def checkDType (expected actual : DType) : Except String Unit :=
  DType.expectedEq expected actual

def checkCompatible (expected actual : TensorSpec) : Except String Unit := do
  checkShape expected.shape actual.shape
  checkDType expected.dtype actual.dtype

end TensorSpec

end torch
