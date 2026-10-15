import Tyr.Typed.Ops

open torch

namespace Tests.TestDTypeWitness

/- These are compile-time contracts. In particular, no native Tensor value or
kernel is needed to establish dtype promotion. -/

example : DType.Promotes .UInt8 .Int8 .Int16 := inferInstance
example : DType.Promotes .Int8 .UInt8 .Int16 := inferInstance
example : DType.Promotes .Float16 .BFloat16 .Float32 := inferInstance
example : DType.Promotes .Float32 .Float64 .Float64 := inferInstance
example : DType.Promotes .Bool .Int64 .Int64 := inferInstance
example : DType.Promotes .Complex32 .BFloat16 .Complex64 := inferInstance
example : DType.Promotes .Complex64 .Float64 .Complex128 := inferInstance
example : DType.Promotes .UInt16 .Float32 .Float32 := inferInstance
example : DType.Promotes .UInt64 .UInt64 .UInt64 := inferInstance
example : DType.Promotes .QInt8 .QInt8 .QInt8 := inferInstance
example : DType.NonBool .Float32 := inferInstance
example : DType.NonBool .Int64 := inferInstance

-- No expected result dtype is supplied at the call site.
private def inferredMixed := DType.promote .UInt8 .Int8
example : inferredMixed = .Int16 := rfl

example {lhs rhs result : DType} [h : DType.Promotes lhs rhs result] :
    DType.promote? lhs rhs = some (DType.promote lhs rhs) := h.valid

example {dtype : DType} [DType.Promotes dtype dtype dtype] :
    DType.promote dtype dtype = dtype := rfl

example : DType.ScalarPromotes .Float32 .Float64 .Float32 := inferInstance
example : DType.ScalarPromotes .Int64 .Float32 .Float32 := inferInstance
example : DType.promoteScalar .Float16 .BFloat16 = .Float16 := rfl

example {tensor scalar result : DType}
    [h : DType.ScalarPromotes tensor scalar result] :
    DType.promoteScalar? tensor scalar = some (DType.promoteScalar tensor scalar) :=
  h.valid

-- Scalar precision is weak in either operand order; two scalars promote normally.
example : TensorSpec.Promotes ⟨#[], .Float64⟩ ⟨#[2], .Float32⟩ .Float32 := inferInstance
example : TensorSpec.Promotes ⟨#[2], .Float32⟩ ⟨#[], .Float64⟩ .Float32 := inferInstance
example : TensorSpec.Promotes ⟨#[], .Float64⟩ ⟨#[], .Float32⟩ .Float64 := inferInstance

example {lhs rhs : TensorSpec} {result : DType} [h : TensorSpec.Promotes lhs rhs result] :
    TensorSpec.pointwiseDType? lhs rhs = some (TensorSpec.pointwiseDType lhs rhs) := h.valid

-- Invalid claims must fail elaboration, with no sorry-backed declarations.
example : True := by
  fail_if_success
    have _ : DType.NonBool .Bool := inferInstance
  fail_if_success
    have _ : DType.Promotes .UInt8 .Int8 .Float32 := inferInstance
  fail_if_success
    have _ : DType.Promotes .Float8E4M3FN .Float32 .Float32 := inferInstance
  fail_if_success
    have _ := DType.promote (.Unknown "dynamic") .Float32
  fail_if_success
    have _ := DType.promote (.Unknown "typo") (.Unknown "typo")
  fail_if_success
    have _ : DType.ScalarPromotes .Float32 .Float64 .Float64 := inferInstance
  fail_if_success
    have _ : TensorSpec.Promotes ⟨#[], .Float64⟩ ⟨#[2], .Float32⟩ .Float64 := inferInstance
  trivial

example (_dtype : DType) : True := by
  fail_if_success
    have _ := DType.promote _dtype _dtype
  trivial

-- Shape compatibility cannot rescue undefined promotion. Matrix multiplication
-- additionally rejects unequal dtypes even when ordinary promotion is defined.
example : TensorSpec.pointwise? ⟨#[2], .Unknown "dynamic"⟩ ⟨#[2], .Float32⟩ = none := by decide
example : TensorSpec.division? ⟨#[2], .Float8E4M3FN⟩ ⟨#[2], .Float32⟩ = none := by decide
example : TensorSpec.matmul? ⟨#[2, 3], .Float32⟩ ⟨#[3, 4], .Float64⟩ = none := by decide
example : TensorSpec.matmul? ⟨#[2, 3], .Unknown "dynamic"⟩
    ⟨#[3, 4], .Unknown "dynamic"⟩ = none := by decide
example : TensorSpec.matmul? ⟨#[2, 3], .Float32⟩ ⟨#[3, 4], .Float32⟩ =
    some ⟨#[2, 4], .Float32⟩ := by decide

-- These functions are elaborated without allocating tensors or invoking kernels.
example (a : Tensor ⟨#[2], .UInt8, .CPU⟩) (b : Tensor ⟨#[2], .Int8, .CPU⟩) :
    Tensor ⟨#[2], .Int16, .CPU⟩ := a + b

example (a : Tensor ⟨#[2], .Complex64, .CPU⟩) (b : Tensor ⟨#[2], .Float64, .CPU⟩) :
    Tensor ⟨#[2], .Complex128, .CPU⟩ := Tensor.add a b

example (a : Tensor ⟨#[2, 1], .Float16, .CPU⟩) (b : Tensor ⟨#[3], .BFloat16, .CPU⟩) :
    Tensor ⟨#[2, 3], .Float32, .CPU⟩ := a.addB b

example (a : Tensor ⟨#[], .Float64, .CPU⟩) (b : Tensor ⟨#[2], .Float32, .CPU⟩) :
    Tensor ⟨#[2], .Float32, .CPU⟩ := a.addB b

-- Computed dtype indices must remain transparent to promotion inference. These
-- definitions deliberately omit an expected result type, as normal composed
-- model expressions do. The following examples separately check their indices.
private def addFloatSums (x : Tensor ⟨#[2], .Float32, .CPU⟩) :=
  x.sumAll + x.sumAll

private def addIntegerSums (x : Tensor ⟨#[2], .Int32, .CPU⟩) :=
  x.sumAll + x.sumAll

private def addComplexSums (x : Tensor ⟨#[2], .Complex64, .CPU⟩) :=
  x.sumAll + x.sumAll

private def addMixedSums (x : Tensor ⟨#[2], .Float16, .CPU⟩)
    (y : Tensor ⟨#[2], .BFloat16, .CPU⟩) :=
  x.sumAll + y.sumAll

private def subtractFloatSums (x : Tensor ⟨#[2], .Float32, .CPU⟩) :=
  x.sumAll - x.sumAll

private def addIntegerQuotient (x : Tensor ⟨#[2], .Int32, .CPU⟩)
    (y : Tensor ⟨#[2], .Float32, .CPU⟩) :=
  Tensor.div x x + y

private def addFloatQuotient (x : Tensor ⟨#[2], .Float32, .CPU⟩) :=
  Tensor.div x x + x

private def addScalarToIntegerQuotient (x : Tensor ⟨#[2], .Int32, .CPU⟩) :=
  Tensor.div x x + (1.0 : Float)

private def addComplexMagnitudes (x : Tensor ⟨#[2], .Complex64, .CPU⟩) :=
  x.abs + x.abs

-- Scalar broadcasting keeps the vector's symbolic shape visible to the next
-- operation; both operand orders must compose without an intermediate cast.
private def addWeakScalarThenVector {n : UInt64}
    (vector : Tensor ⟨#[n], .Float32, .CPU⟩) (scalar : Tensor ⟨#[], .Float64, .CPU⟩) :=
  vector.addB scalar + vector

private def addReversedWeakScalarThenVector {n : UInt64}
    (vector : Tensor ⟨#[n], .Float32, .CPU⟩) (scalar : Tensor ⟨#[], .Float64, .CPU⟩) :=
  scalar.addB vector + vector

private def addCPUSignalAfterScalar {n : UInt64}
    (vector : Tensor ⟨#[n], .Float32, .CPU⟩)
    (scalar : Tensor ⟨#[], .Float64, .CPU⟩) :=
  scalar.addB vector + vector

example (x : Tensor ⟨#[2], .Float32, .CPU⟩) : Tensor ⟨#[], .Float32, .CPU⟩ :=
  addFloatSums x
example (x : Tensor ⟨#[2], .Int32, .CPU⟩) : Tensor ⟨#[], .Int64, .CPU⟩ :=
  addIntegerSums x
example (x : Tensor ⟨#[2], .Complex64, .CPU⟩) : Tensor ⟨#[], .Complex64, .CPU⟩ :=
  addComplexSums x
example (x : Tensor ⟨#[2], .Float16, .CPU⟩) (y : Tensor ⟨#[2], .BFloat16, .CPU⟩) :
    Tensor ⟨#[], .Float32, .CPU⟩ := addMixedSums x y
example (x : Tensor ⟨#[2], .Float32, .CPU⟩) : Tensor ⟨#[], .Float32, .CPU⟩ :=
  subtractFloatSums x
example (x : Tensor ⟨#[2], .Int32, .CPU⟩) (y : Tensor ⟨#[2], .Float32, .CPU⟩) :
    Tensor ⟨#[2], .Float32, .CPU⟩ := addIntegerQuotient x y
example (x : Tensor ⟨#[2], .Float32, .CPU⟩) : Tensor ⟨#[2], .Float32, .CPU⟩ :=
  addFloatQuotient x
example (x : Tensor ⟨#[2], .Int32, .CPU⟩) : Tensor ⟨#[2], .Float32, .CPU⟩ :=
  addScalarToIntegerQuotient x
example (x : Tensor ⟨#[2], .Complex64, .CPU⟩) : Tensor ⟨#[2], .Float32, .CPU⟩ :=
  addComplexMagnitudes x
example {n : UInt64} (vector : Tensor ⟨#[n], .Float32, .CPU⟩)
    (scalar : Tensor ⟨#[], .Float64, .CPU⟩) : Tensor ⟨#[n], .Float32, .CPU⟩ :=
  addWeakScalarThenVector vector scalar
example {n : UInt64} (vector : Tensor ⟨#[n], .Float32, .CPU⟩)
    (scalar : Tensor ⟨#[], .Float64, .CPU⟩) : Tensor ⟨#[n], .Float32, .CPU⟩ :=
  addReversedWeakScalarThenVector vector scalar
example {n : UInt64} (vector : Tensor ⟨#[n], .Float32, .CPU⟩)
    (scalar : Tensor ⟨#[], .Float64, .CPU⟩) : Tensor ⟨#[n], .Float32, .CPU⟩ :=
  addCPUSignalAfterScalar vector scalar

example {shape : Shape} {lhs rhs result : DType} {device : Device}
    [DType.Promotes lhs rhs result]
    (a : Tensor ⟨shape, lhs, device⟩) (b : Tensor ⟨shape, rhs, device⟩) :
    Tensor ⟨shape, result, device⟩ := a + b

example {spec : StaticSpec} [DType.Promotes spec.dtype spec.dtype spec.dtype]
    (a b : Tensor spec) : Tensor spec := Add.add a b

-- A runtime-checked promotion can supply the same proof expected by static APIs.
example {shape : Shape} {lhs rhs : DType} {device : Device}
    (a : Tensor ⟨shape, lhs, device⟩) (b : Tensor ⟨shape, rhs, device⟩) :
    Option (Sigma fun result : DType => Tensor ⟨shape, result, device⟩) :=
  match h : DType.promote? lhs rhs with
  | none => none
  | some result =>
      letI : DType.Promotes lhs rhs result := ⟨h⟩
      some ⟨result, Tensor.add a b⟩

example (_a : Tensor ⟨#[2], .Float8E4M3FN, .CPU⟩)
    (_b : Tensor ⟨#[2], .Float32, .CPU⟩) : True := by
  fail_if_success have _ := Tensor.add _a _b
  fail_if_success have _ := _a + _b
  fail_if_success have _ := _a.addB _b
  trivial

example (_a : Tensor ⟨#[2], .QInt8, .CPU⟩)
    (_b : Tensor ⟨#[2], .QUInt8, .CPU⟩) : True := by
  fail_if_success have _ := Tensor.add _a _b
  fail_if_success have _ := _a + _b
  fail_if_success have _ := _a.addB _b
  trivial

example (_a : Tensor ⟨#[2], .UInt16, .CPU⟩)
    (_b : Tensor ⟨#[2], .Int64, .CPU⟩) : True := by
  fail_if_success have _ := Tensor.add _a _b
  fail_if_success have _ := _a + _b
  fail_if_success have _ := _a.addB _b
  trivial

-- The homogeneous instances must not bypass missing promotion evidence.
example {dtype : DType} (_a _b : Tensor { shape := #[2], dtype, device := .CPU }) : True := by
  fail_if_success have _ := Tensor.add _a _b
  fail_if_success have _ := _a + _b
  fail_if_success have _ := _a.addB _b
  fail_if_success have _ := Add.add _a _b
  fail_if_success have _ := Sub.sub _a _b
  fail_if_success have _ := Mul.mul _a _b
  trivial

example {_spec : StaticSpec} (_a _b : Tensor _spec) : True := by
  fail_if_success have _ := Add.add _a _b
  fail_if_success have _ := _a + _b
  trivial

-- Bool promotion is defined, but subtraction has a separate precondition.
example (_a _b : Tensor ⟨#[2], .Bool, .CPU⟩)
    (_float : Tensor ⟨#[2], .Float32, .CPU⟩) : True := by
  fail_if_success have _ := Tensor.sub _a _b
  fail_if_success have _ := _a - _b
  fail_if_success have _ := _a.subB _b
  fail_if_success have _ := Sub.sub _a _b
  fail_if_success have _ := Tensor.sub _float _a
  fail_if_success have _ := _a - _float
  fail_if_success have _ := _float.subB _a
  trivial

end Tests.TestDTypeWitness
