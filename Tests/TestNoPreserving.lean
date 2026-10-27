import Tyr.Typed.Neural

open torch

-- Natural result rules remain visible for non-floating inputs.
example (x y : Tensor ⟨#[2], .Int32, .CPU⟩) : Tensor ⟨#[2], .Float32, .CPU⟩ :=
  Tensor.div x y
example (x : Tensor ⟨#[2], .Int32, .CPU⟩) : Tensor ⟨reduceShape #[2] 0 false, .Int64, .CPU⟩ :=
  Tensor.sumDim x 0
example (x : Tensor ⟨#[2], .Complex64, .CPU⟩) : Tensor ⟨#[2], .Float32, .CPU⟩ :=
  Tensor.abs x
example (x : Tensor ⟨#[2], .Int32, .CPU⟩) : Tensor ⟨#[2], .Float32, .CPU⟩ :=
  Tensor.clampFloat x 0 1
example (x : Tensor ⟨#[2], .Int32, .CPU⟩) :
    Except String (Tensor ⟨#[2], .Float32, .CPU⟩) := Tensor.mulScalar? x 2

-- Generic floating code uses the same operations and erased index equalities.
example {σ : StaticSpec} [DType.RealFloating σ.dtype] (x y : Tensor σ) : Tensor σ :=
  Tensor.add (Tensor.sub x y) (Tensor.mul x y)
example {σ : StaticSpec} [DType.RealFloating σ.dtype] (x : Tensor σ) : Tensor σ :=
  Tensor.mulScalar (Tensor.addScalar x 1) 2
example {σ : StaticSpec} [DType.RealFloating σ.dtype] (x : Tensor σ) : Tensor σ :=
  Tensor.sqrt (Tensor.exp (Tensor.log (Tensor.sin (Tensor.cos (Tensor.atan x)))))
example {σ : StaticSpec} [DType.RealFloating σ.dtype] (x : Tensor σ) : Tensor σ :=
  Tensor.softmax (Tensor.gelu (Tensor.silu (Tensor.sigmoid (Tensor.tanh (Tensor.elu x)))))
example {σ : StaticSpec} [DType.RealFloating σ.dtype] (x y : Tensor σ) : Tensor σ := by
  simpa using Tensor.div x y
example {σ : StaticSpec} [DType.RealFloating σ.dtype] (x : Tensor σ) : Tensor σ := by
  simpa using Tensor.abs x
example {s : Shape} {dtype : DType} {device : Device} [DType.RealFloating dtype]
    (x : Tensor ⟨s, dtype, device⟩) (dim : Nat) (axis : dim < max 1 s.size) :
    Tensor ⟨reduceShape s dim false, dtype, device⟩ := by
  simpa using Tensor.sumDim x dim false axis
example {s : Shape} {dtype : DType} {device : Device} [DType.RealFloating dtype]
    (x : Tensor ⟨s, dtype, device⟩) (dim : Nat) (axis : dim < max 1 s.size) :
    Tensor ⟨reduceShape s dim false, dtype, device⟩ := Tensor.meanDim x dim false axis

example (_x : Tensor ⟨#[2], .Int32, .CPU⟩) : True := by
  fail_if_success have _ := Tensor.sqrt _x
  fail_if_success have _ := Tensor.meanDim _x 0
  fail_if_success have _ : DType.RealFloating .Int32 := inferInstance
  trivial

example : (DType.realFloating? .Float16).isOk = true := by decide
example : (DType.realFloating? .BFloat16).isOk = true := by decide
example : (DType.realFloating? .Float32).isOk = true := by decide
example : (DType.realFloating? .Float64).isOk = true := by decide
example : (DType.realFloating? .Int64).isOk = false := by decide
example : (DType.realFloating? .Complex64).isOk = false := by decide
example : (DType.realFloating? .Float8E4M3FN).isOk = false := by decide

example {σ : StaticSpec} [DType.RealFloating σ.dtype] (x y : Tensor σ) : Tensor σ :=
  Tensor.maximum x y

-- Runtime validation supplies the same proof used by statically typed callers.
example {s : Shape} {dtype : DType} {device : Device}
    (x : Tensor ⟨s, dtype, device⟩) : Except String (Tensor ⟨s, dtype, device⟩) := do
  let support ← DType.realFloating? dtype
  let _ := support.down
  Tensor.mulScalar? x 2

-- Loaded compute-dtype packages retain their proof when the dtype is hidden.
example (selected : RealFloatingDType) {s : Shape} {device : Device}
    (x : Tensor ⟨s, selected.val, device⟩) : Tensor ⟨s, selected.val, device⟩ :=
  Tensor.mulScalar x 2

example : (RealFloatingDType.ofDType? .BFloat16).isOk = true := by decide
example : (RealFloatingDType.ofDType? .Int32).isOk = false := by decide
example : (RealFloatingDType.ofDType? .Complex64).isOk = false := by decide
example : (RealFloatingDType.ofDType? .Float8E4M3FN).isOk = false := by decide
example : True := by
  fail_if_success have _ : RealFloatingDType := ⟨.Int32, rfl⟩
  trivial
