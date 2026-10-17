/-
  TypeCounterexampleRuntime.lean

  Axis arguments the typed facade accepts without checking. Every definition
  below compiles, and each one is wrong: the result type states a shape, but
  libtorch rejects the call at runtime. Deliberately not imported by
  `Examples.lean`. `lake lean` builds the imports and loads Tyr's native code,
  so the `#eval` at the bottom actually runs:

    lake lean Examples/TypeCounterexampleRuntime.lean

  A libtorch error in a pure op aborts the whole process (exit 134, reported
  as "Stack overflow detected"), so only one case can run per invocation.
  Each case was run on its own this way and aborted; the same ops with valid
  axes return the shapes their types claim. The failure is loud, not silent:
  no tensor with a wrong type ever reaches later code.

  Two root causes:
  * The shape functions in `Tyr/Basic/Types.lean` (`transposeShape`,
    `unsqueezeShape`, `squeezeShape`, …) return the input shape unchanged
    when the axis is out of range, so the type claims a no-op.
  * Axis arguments are plain `Int32` / `Nat` with no relation to the
    tensor's rank, and `catShape` never compares the non-concatenated axes.
-/
import Tyr.Typed
open torch

/-- `StaticSpec` has no default device; fix it to CPU so the cases
    read as shape + dtype only. -/
abbrev DTensor (shape : Shape) (dtype : DType) : Type :=
  Tensor { shape, dtype, device := .CPU }

namespace AxisOutOfRange

-- softmax over axis 7 of a rank-2 tensor; `dim : Int32` is never compared
-- with the rank. Aborts at runtime.
def softmax7 (x : DTensor #[2, 3] .Float32) : DTensor #[2, 3] .Float32 :=
  x.softmax 7

-- Same with a negative axis past the front.
def softmaxNeg (x : DTensor #[2, 3] .Float32) : DTensor #[2, 3] .Float32 :=
  x.softmax (-3)

-- transposeShape returns the shape unchanged for an out-of-range axis, so the
-- type says "no-op". Aborts at runtime.
def transpose05 (x : DTensor #[2, 3] .Float32) : DTensor #[2, 3] .Float32 :=
  x.transpose 0 5

-- unsqueezeShape likewise returns the shape unchanged. Aborts at runtime.
def unsqueeze9 (x : DTensor #[2, 3] .Float32) : DTensor #[2, 3] .Float32 :=
  x.unsqueeze 9

-- squeezeShape likewise. Aborts at runtime.
def squeeze4 (x : DTensor #[2, 3] .Float32) : DTensor #[2, 3] .Float32 :=
  x.squeeze 4

end AxisOutOfRange

namespace Cat

-- catShape adds the sizes along `dim` and never compares the other axes.
-- 2 ≠ 5 on axis 0. Aborts at runtime.
def mismatched (x : DTensor #[2, 3] .Float32) (y : DTensor #[5, 7] .Float32) :
    DTensor #[2, 10] .Float32 :=
  x.cat y 1

-- Rank mismatch is not checked either. Aborts at runtime.
def rankMismatch (x : DTensor #[2, 3] .Float32) (y : DTensor #[2, 3, 4] .Float32) :
    DTensor #[2, 6] .Float32 :=
  x.cat y 1

end Cat

-- The type claims `#[2, 3]`; running it aborts the process.
#eval (AxisOutOfRange.transpose05 (Tensor.ones #[2, 3])).raw.runtimeShape
