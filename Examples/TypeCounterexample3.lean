/-
  TypeCounterexample3.lean

  Axis and size arguments the typed facade accepts without checking. Every
  definition below compiles, and each one is wrong: the result type states a
  shape libtorch would never produce, because the call throws at runtime.
  Deliberately not imported by `Examples.lean`; check with:

    lake env lean Examples/TypeCounterexample3.lean

  Two root causes:
  * The shape functions in `Tyr/Basic/Types.lean` (`transposeShape`,
    `unsqueezeShape`, `squeezeShape`, …) return the input shape unchanged
    when the axis is out of range, so the type claims a no-op.
  * Axis arguments are plain `Int32` / `Nat` with no relation to the
    tensor's rank, and `catShape` never compares the non-concatenated axes.
-/
import Tyr.Typed
open torch

namespace AxisOutOfRange

-- softmax over axis 7 of a rank-2 tensor; `dim : Int32` is never compared
-- with the rank. libtorch: IndexError, dimension out of range.
def softmax7 (x : DTensor #[2, 3] .Float32) : DTensor #[2, 3] .Float32 :=
  x.softmax 7

-- Same with a negative axis past the front.
def softmaxNeg (x : DTensor #[2, 3] .Float32) : DTensor #[2, 3] .Float32 :=
  x.softmax (-3)

-- transposeShape returns the shape unchanged for an out-of-range axis, so the
-- type says "no-op". libtorch: IndexError.
def transpose05 (x : DTensor #[2, 3] .Float32) : DTensor #[2, 3] .Float32 :=
  x.transpose 0 5

-- unsqueezeShape likewise returns the shape unchanged. libtorch: IndexError.
def unsqueeze9 (x : DTensor #[2, 3] .Float32) : DTensor #[2, 3] .Float32 :=
  x.unsqueeze 9

-- squeezeShape likewise. libtorch: IndexError.
def squeeze4 (x : DTensor #[2, 3] .Float32) : DTensor #[2, 3] .Float32 :=
  x.squeeze 4

end AxisOutOfRange

namespace Cat

-- catShape adds the sizes along `dim` and never compares the other axes.
-- 2 ≠ 5 on axis 0. libtorch: sizes of tensors must match except in dimension 1.
def mismatched (x : DTensor #[2, 3] .Float32) (y : DTensor #[5, 7] .Float32) :
    DTensor #[2, 10] .Float32 :=
  x.cat y 1

-- Rank mismatch is not checked either.
def rankMismatch (x : DTensor #[2, 3] .Float32) (y : DTensor #[2, 3, 4] .Float32) :
    DTensor #[2, 6] .Float32 :=
  x.cat y 1

end Cat
