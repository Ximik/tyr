/-
  Shape side conditions over generic dimensions and dtypes (#24).

  Correct definitions must elaborate with the default proofs; wrong reshapes must
  be rejected. Everything here is checked when the module is compiled.
-/
import Tyr.Typed

namespace Tests.ShapeProofs

open torch

abbrev DTensor (shape : Shape) (dtype : DType) : Type :=
  Tensor { shape, dtype, device := .CPU }

def splitHeads {b : Nat} (x : DTensor #[b, 512] .Float32) : DTensor #[b, 8, 64] .Float32 :=
  x.reshape #[b, 8, 64]

def splitConcrete (x : DTensor #[2, 512] .Float32) : DTensor #[2, 8, 64] .Float32 :=
  x.reshape #[2, 8, 64]

def split2 {b s : Nat} (x : DTensor #[b, s, 512] .Float32) : DTensor #[b, s, 8, 64] .Float32 :=
  x.reshape #[b, s, 8, 64]

def swap {b s : Nat} (x : DTensor #[b, s] .Float32) : DTensor #[s, b] .Float32 :=
  x.reshape #[s, b]

def flatten {b s : Nat} (x : DTensor #[b, s, 512] .Float32) : DTensor #[b * s, 512] .Float32 :=
  x.reshape #[b * s, 512]

def mergeHeads {b s h d : Nat} (x : DTensor #[b, s, h, d] .Float32) : DTensor #[b, s, h * d] .Float32 :=
  x.reshape #[b, s, h * d]

def mergeHeadsReordered {b h s d : Nat} (x : DTensor #[b, h, s, d] .Float32) :
    DTensor #[b, s, h * d] .Float32 :=
  x.reshape #[b, s, h * d]

def mmGeneric {m k n : Nat} (x : DTensor #[m, k] .Float32) (w : DTensor #[k, n] .Float32) :
    DTensor #[m, n] .Float32 :=
  x.mm w

def matmulGeneric {m k n : Nat} (x : DTensor #[m, k] .Float32) (w : DTensor #[k, n] .Float32) :
    DTensor #[m, n] .Float32 :=
  x.matmul w

def matmulBatched {b m k n : Nat} (x : DTensor #[b, m, k] .Float32) (w : DTensor #[b, k, n] .Float32) :=
  x.matmul w

def matmulBroadcast {b c m k n : Nat} (x : DTensor #[b, c, m, k] .Float32) (w : DTensor #[b, 1, k, n] .Float32) :=
  x.matmul w

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) (w : DTensor #[4, 4] .Float32) : True := by
  fail_if_success have := x.matmul w
  trivial

set_option linter.unusedVariables false in
example {m k j n : Nat} (x : DTensor #[m, k] .Float32) (w : DTensor #[j, n] .Float32) : True := by
  fail_if_success have := x.matmul w
  trivial

def transposeGeneric {b s d : Nat} (x : DTensor #[b, s, d] .Float32) : DTensor #[b, d, s] .Float32 :=
  x.transpose 1 2

def expGeneric {d : DType} [DType.RealFloating d] (x : DTensor #[4] d) : DTensor #[4] d :=
  x.exp

set_option linter.unusedVariables false in
example {b : Nat} (x : DTensor #[b, 512] .Float32) : True := by
  fail_if_success have := x.reshape #[b, 8, 60]
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 512] .Float32) : True := by
  fail_if_success have := x.reshape #[2, 8, 60]
  trivial

set_option linter.unusedVariables false in
example {b s : Nat} (x : DTensor #[b, s, 512] .Float32) : True := by
  fail_if_success have := x.reshape #[b * s, 500]
  trivial

set_option linter.unusedVariables false in
example {b s : Nat} (x : DTensor #[b, s] .Float32) : True := by
  fail_if_success have := x.reshape #[b, b]
  trivial

end Tests.ShapeProofs
