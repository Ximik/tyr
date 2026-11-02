/-
  Shape side conditions over generic dimensions and dtypes (#23, #24).

  Correct definitions must elaborate with the default proofs; wrong reshapes,
  out-of-range axes and incompatible shapes must be rejected. Everything here is
  checked when the module is compiled.
-/
import Tyr.Typed
import Tyr.Typed.Neural

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

/-! ## Axes (#23) -/

def softmaxLast {b s d : Nat} (x : DTensor #[b, s, d] .Float32) : DTensor #[b, s, d] .Float32 :=
  x.softmax

def softmaxAxis {b s d : Nat} (x : DTensor #[b, s, d] .Float32) : DTensor #[b, s, d] .Float32 :=
  x.softmax 1

def softmaxScalar (x : DTensor #[] .Float32) : DTensor #[] .Float32 :=
  x.softmax

def logSoftmaxGeneric {σ : StaticSpec} [DType.RealFloating σ.dtype] (x : Tensor σ) : Tensor σ :=
  x.logSoftmax

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) : True := by
  fail_if_success have := x.softmax 7
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) : True := by
  fail_if_success have := x.softmax (-3)
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) : True := by
  fail_if_success have := x.logSoftmax 2
  trivial

set_option linter.unusedVariables false in
example {b s : Nat} (x : DTensor #[b, s] .Float32) : True := by
  fail_if_success have := x.softmax 2
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) : True := by
  fail_if_success have := x.transpose 0 5
  trivial

set_option linter.unusedVariables false in
example {b s : Nat} (x : DTensor #[b, s] .Float32) : True := by
  fail_if_success have := x.transpose 2 0
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) : True := by
  fail_if_success have := x.unsqueeze 9
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) : True := by
  fail_if_success have := x.squeeze 4
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) : True := by
  fail_if_success have := x.unbind 2
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) : True := by
  fail_if_success have := x.sumDim 5
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) : True := by
  fail_if_success have := x.meanDim 2
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) : True := by
  fail_if_success have := x.argmax 2
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 0] .Float32) : True := by
  fail_if_success have := x.argmax 1
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) : True := by
  fail_if_success have := x.slice 1 2 2
  trivial

/-! ## Concatenation (#23) -/

def catConcrete (x : DTensor #[2, 3] .Float32) (y : DTensor #[2, 7] .Float32) :
    DTensor #[2, 10] .Float32 :=
  x.cat y 1

def catGeneric {b s t d : Nat} (x : DTensor #[b, s, d] .Float32) (y : DTensor #[b, t, d] .Float32) :
    DTensor #[b, s + t, d] .Float32 :=
  x.cat y 1

def catAppend {b s : Nat} (x : DTensor #[b, s] .Int64) (y : DTensor #[b, 1] .Int64) :
    DTensor #[b, s + 1] .Int64 :=
  x.cat y 1

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) (y : DTensor #[5, 7] .Float32) : True := by
  fail_if_success have := x.cat y 1
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) (y : DTensor #[2, 3, 4] .Float32) : True := by
  fail_if_success have := x.cat y 1
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) (y : DTensor #[2, 3] .Float32) : True := by
  fail_if_success have := x.cat y 2
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[0] .Float32) (y : DTensor #[2, 3] .Float32) : True := by
  fail_if_success have := x.cat y 0
  trivial

set_option linter.unusedVariables false in
example {b c s t : Nat} (x : DTensor #[b, s] .Float32) (y : DTensor #[c, t] .Float32) : True := by
  fail_if_success have := x.cat y 1
  trivial

/-! ## Permutation, gather and scatter (#23) -/

def permuteGeneric {b c h w : Nat} (x : DTensor #[b, c, h, w] .Float32) :=
  x.permute #[0, 2, 3, 1]

def gatherGeneric {b v k : Nat} (x : DTensor #[b, v] .Float32) (i : DTensor #[b, k] .Int64) :
    DTensor #[b, k] .Float32 :=
  Tensor.gather x 1 i

def scatterGeneric {b v k : Nat} (x : DTensor #[b, v] .Float32) (i : DTensor #[b, k] .Int64)
    (src : DTensor #[b, k] .Float32) : DTensor #[b, v] .Float32 :=
  Tensor.scatter2d x 1 i src

def scatterRows (x : DTensor #[2, 5] .Float32) (i : DTensor #[2, 3] .Int64)
    (src : DTensor #[2, 3] .Float32) : DTensor #[2, 5] .Float32 :=
  Tensor.scatter2d x 0 i src

def getLeading {n : Nat} (x : DTensor #[n + 1, 3] .Float32) :=
  x.getOp 0

def sliceScatterGeneric {b h c d : Nat} (store : DTensor #[b, h, c, d] .Float32)
    (row : DTensor #[b, h, 1, d] .Float32) (pos : Nat) (_hpos : pos < c) : DTensor #[b, h, c, d] .Float32 :=
  Tensor.sliceScatter store 2 pos row

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) : True := by
  fail_if_success have := x.permute #[0, 0]
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) : True := by
  fail_if_success have := x.permute #[5, 0]
  trivial

set_option linter.unusedVariables false in
example {b c : Nat} (x : DTensor #[b, c] .Float32) : True := by
  fail_if_success have := x.permute #[0, 1, 2]
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) (i : DTensor #[3, 2] .Int64) : True := by
  fail_if_success have := Tensor.gather x 1 i
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) (i : DTensor #[2] .Int64) : True := by
  fail_if_success have := Tensor.gather x 0 i
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) (i : DTensor #[2, 2] .Int64) : True := by
  fail_if_success have := Tensor.gather x 2 i
  trivial

set_option linter.unusedVariables false in
example {b c v k : Nat} (x : DTensor #[b, v] .Float32) (i : DTensor #[c, k] .Int64) : True := by
  fail_if_success have := Tensor.gather x 1 i
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) (i : DTensor #[2, 2] .Int64) (src : DTensor #[2, 2] .Float32) : True := by
  fail_if_success have := Tensor.scatter2d x 2 i src
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) (i : DTensor #[2, 4] .Int64) (src : DTensor #[2, 4] .Float32) : True := by
  fail_if_success have := Tensor.scatter2d x 0 i src
  trivial

set_option linter.unusedVariables false in
example {b v k : Nat} (x : DTensor #[b, v] .Float32) (i : DTensor #[b, k] .Int64) (src : DTensor #[b, k] .Float32) : True := by
  fail_if_success have := Tensor.scatter2d x 0 i src
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[2, 3] .Float32) : True := by
  fail_if_success have := x.getOp 2
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[] .Float32) : True := by
  fail_if_success have := x.getOp 0
  trivial

set_option linter.unusedVariables false in
example {n : Nat} (x : DTensor #[n, 3] .Float32) : True := by
  fail_if_success have := x.getOp 0
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[4, 8] .Float32) (y : DTensor #[1, 7] .Float32) : True := by
  fail_if_success have := Tensor.sliceScatter x 0 0 y
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[4, 8] .Float32) (y : DTensor #[1, 8] .Float32) : True := by
  fail_if_success have := Tensor.sliceScatter x 2 0 y
  trivial

set_option linter.unusedVariables false in
example (x : DTensor #[4, 8] .Float32) (y : DTensor #[8] .Float32) : True := by
  fail_if_success have := Tensor.sliceScatter x 0 0 y
  trivial

/-! ## Generic reductions and bounds -/

def sumDimGeneric {b s d : Nat} (x : DTensor #[b, s, d] .Float32) :=
  x.sumDim 1

def meanDimGeneric {b s d : Nat} (x : DTensor #[b, s, d] .Float32) :=
  x.meanDim 2

def argmaxGeneric {b s : Nat} (x : DTensor #[b, s + 1] .Float32) :=
  x.argmax 1

def sliceInScope {b s : Nat} (x : DTensor #[b, s] .Float32) (start : Nat) : Option Nat :=
  if h : start + 2 ≤ s then some (x.slice 1 start 2).raw.runtimeShape.size else none

def topKGeneric {r c : Nat} (x : DTensor #[r, c + 3] .Float32) :=
  Tensor.topKLast2d x 3

def qrTall {m n : Nat} (x : DTensor #[m + n, n] .Float32) :=
  Tensor.qrReduced x

set_option linter.unusedVariables false in
example {b s d : Nat} (x : DTensor #[b, s, d] .Float32) : True := by
  fail_if_success have := x.sumDim 3
  trivial

set_option linter.unusedVariables false in
example {b s : Nat} (x : DTensor #[b, s] .Float32) : True := by
  fail_if_success have := x.argmax 1
  trivial

set_option linter.unusedVariables false in
example {b s : Nat} (x : DTensor #[b, s] .Float32) (start : Nat) : True := by
  fail_if_success have := x.slice 1 start 2
  trivial

set_option linter.unusedVariables false in
example {r c : Nat} (x : DTensor #[r, c] .Float32) : True := by
  fail_if_success have := Tensor.topKLast2d x 3
  trivial

end Tests.ShapeProofs
