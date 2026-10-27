import Tyr.Typed.DType

/-!
# Dtype promotion witnesses

Promotion is partial. These proof-only classes expose its successful result to
typeclass inference without inventing a dtype for unsupported pairs. They prove
the promotion rule, not that an individual operator has a kernel on a device.
-/

namespace torch

/-- Evidence that an optional computation has the specified result. The only
automatic instance matches `some`; `none` has no instance. Computations used as
instance inputs must unfold at reducible transparency. -/
class SomeResult {α : Type u} (value : Option α) (result : outParam α) : Prop where
  valid : value = some result

instance {α : Type u} (result : α) : SomeResult (some result) result := ⟨rfl⟩

namespace DType

/-- Subtraction rejects Bool operands even when their promotion is defined.
This is an operator precondition, separate from the promotion witness. -/
class NonBool (dtype : DType) : Prop where
  valid : (dtype != .Bool) = true

@[reducible] private def nonBool? : DType → Option Unit
  | .Bool => none
  | _ => some ()

instance {dtype : DType} [h : SomeResult (nonBool? dtype) ()] : NonBool dtype := by
  constructor
  cases dtype <;> first | rfl | cases h.valid

/-- A type-level witness of ordinary tensor dtype promotion. Unsupported
pairs require a checked runtime path rather than a fabricated
instance. Generic functions can accept this class as an explicit constraint. -/
class Promotes (lhs rhs : DType) (result : outParam DType) : Prop where
  valid : promote? lhs rhs = some result

instance {lhs rhs result : DType} [h : SomeResult (promote? lhs rhs) result] :
    Promotes lhs rhs result := ⟨h.valid⟩

/-- A witness of promotion between a non-scalar tensor and a zero-dimensional
tensor. The non-scalar operand's precision takes precedence within a category. -/
class ScalarPromotes (tensor scalar : DType) (result : outParam DType) : Prop where
  valid : promoteScalar? tensor scalar = some result

instance {tensor scalar result : DType}
    [h : SomeResult (promoteScalar? tensor scalar) result] :
    ScalarPromotes tensor scalar result := ⟨h.valid⟩

/-- Standard real floating tensors keep their dtype in homogeneous arithmetic. -/
instance {dtype : DType} [h : RealFloating dtype] : Promotes dtype dtype dtype := by
  constructor
  cases dtype <;> first | rfl | cases h.valid

/-- A host Float is a weak scalar: standard floating tensor precision wins. -/
instance {dtype : DType} [h : RealFloating dtype] : ScalarPromotes dtype .Float32 dtype := by
  constructor
  cases dtype <;> first | rfl | cases h.valid

instance {dtype : DType} [h : RealFloating dtype] : NonBool dtype := by
  constructor
  cases dtype <;> first | rfl | cases h.valid

/-- Homogeneous complex arithmetic preserves its precision as well. -/
instance {dtype : DType} [h : FloatingOrComplex dtype] : Promotes dtype dtype dtype := by
  constructor
  cases dtype <;> first | rfl | cases h.valid

instance {dtype : DType} [h : FloatingOrComplex dtype] : ScalarPromotes dtype .Float32 dtype := by
  constructor
  cases dtype <;> first | rfl | cases h.valid

instance {dtype : DType} [h : FloatingOrComplex dtype] : NonBool dtype := by
  constructor
  cases dtype <;> first | rfl | cases h.valid

/-- The ordinary promotion result, available only with a valid witness. -/
@[reducible] def promote (lhs rhs : DType) {result : DType}
    [Promotes lhs rhs result] : DType := result

/-- The weak scalar promotion result, available only with a valid witness. -/
@[reducible] def promoteScalar (tensor scalar : DType) {result : DType}
    [ScalarPromotes tensor scalar result] : DType := result

end DType

end torch
