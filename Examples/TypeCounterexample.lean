/-
  TypeCounterexample.lean

  Well-shaped code that the typed facade rejects, next to a small `Nat`-indexed
  model showing which of those rejections are tactic limits and which come from
  `UInt64` dimensions. Deliberately not imported by `Examples.lean`; check with:

    lake env lean Examples/TypeCounterexample.lean

  Every definition is marked ✅ (compiles) or ❌ (fails to elaborate).
-/
import Tyr.Typed
open torch

/-! ## Real facade, default side-condition tactics

`reshape` proves its element-count condition `by decide`, `matmul` its
shape-compatibility condition `by rfl`, and float-only ops `isFloating` by
`decide`. All of them only evaluate closed terms, so any generic dimension or
dtype fails, even though every definition below is correct. -/

namespace Facade

-- ❌ decide: expected type must not contain free variables
def splitHeads {b : UInt64} (x : DTensor #[b, 512] .Float32) : DTensor #[b, 8, 64] .Float32 :=
  x.reshape #[b, 8, 64]

-- ❌ rfl: `(matmulShape? #[m, k] #[k, n]).isSome` does not reduce to `true`
def matmulGeneric {m k n : UInt64} (x : DTensor #[m, k] .Float32) (w : DTensor #[k, n] .Float32) :=
  x.matmul w

-- ❌ decide: `d.isFloating = true` has a free variable
def expGeneric {d : DType} (x : DTensor #[4] d) :=
  x.exp

-- ✅ shape rules stated as patterns in the signature need no tactic
def mmGeneric {m k n : UInt64} (x : DTensor #[m, k] .Float32) (w : DTensor #[k, n] .Float32) :
    DTensor #[m, n] .Float32 :=
  x.mm w

-- ✅ `transposeShape` reduces on a literal-length array
def transposeGeneric {b s d : UInt64} (x : DTensor #[b, s, d] .Float32) :
    DTensor #[b, d, s] .Float32 :=
  x.transpose 1 2

end Facade

/-! ## Real facade, stronger reshape tactic

`reshape'` swaps `decide` for `simp … <;> omega`. Linear cases now go through,
but a product of two `UInt64` dimensions wraps mod 2⁶⁴, so
`(b * s).toNat = b.toNat * s.toNat` is false in general and no tactic can
prove the flatten/merge cases. -/

namespace FacadeOmega

def reshape' {σ : StaticSpec} (t : Tensor σ) (shape' : Shape)
    (_h : TensorSpec.numelOfShape σ.shape = TensorSpec.numelOfShape shape' := by
      simp [TensorSpec.numelOfShape, Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm] <;> omega) :
    Tensor (σ.withShape shape') :=
  t.reshape shape' _h

-- ✅
def splitHeads {b : UInt64} (x : DTensor #[b, 512] .Float32) : DTensor #[b, 8, 64] .Float32 :=
  reshape' x #[b, 8, 64]

-- ✅
def split2 {b s : UInt64} (x : DTensor #[b, s, 512] .Float32) : DTensor #[b, s, 8, 64] .Float32 :=
  reshape' x #[b, s, 8, 64]

-- ✅
def swap {b s : UInt64} (x : DTensor #[b, s] .Float32) : DTensor #[s, b] .Float32 :=
  reshape' x #[s, b]

-- ❌ omega counterexample: b.toNat * s.toNat ≥ 2⁶⁴
def flatten {b s : UInt64} (x : DTensor #[b, s, 512] .Float32) : DTensor #[b * s, 512] .Float32 :=
  reshape' x #[b * s, 512]

-- ❌ same overflow on h * d
def mergeHeads {b s h d : UInt64} (x : DTensor #[b, s, h, d] .Float32) :
    DTensor #[b, s, h * d] .Float32 :=
  reshape' x #[b, s, h * d]

-- ❌ correctly rejected: 512 ≠ 480
def wrongSplit {b : UInt64} (x : DTensor #[b, 512] .Float32) : DTensor #[b, 8, 60] .Float32 :=
  reshape' x #[b, 8, 60]

end FacadeOmega

/-! ## Fake facade with `Nat` dimensions

Same phantom-index idea, dimensions in `Nat`. With the same tactic every
correct reshape is proved and every wrong one rejected. `Nat.mul_comm` /
`Nat.mul_left_comm` are needed because omega treats `b * s` and `s * b` as
unrelated atoms. -/

namespace NatModel

abbrev Dims := List Nat

def numel : Dims → Nat
  | [] => 1
  | d :: ds => d * numel ds

structure Ten (s : Dims) where

def reshape {s : Dims} (_t : Ten s) (s' : Dims)
    (_h : numel s = numel s' := by
      simp only [numel, Nat.mul_assoc, Nat.mul_one, Nat.mul_comm, Nat.mul_left_comm] <;> omega) :
    Ten s' :=
  ⟨⟩

-- ✅
def splitHeads {b : Nat} (x : Ten [b, 512]) : Ten [b, 8, 64] :=
  reshape x [b, 8, 64]

-- ✅
def splitConcrete (x : Ten [2, 512]) : Ten [2, 8, 64] :=
  reshape x [2, 8, 64]

-- ✅
def split2 {b s : Nat} (x : Ten [b, s, 512]) : Ten [b, s, 8, 64] :=
  reshape x [b, s, 8, 64]

-- ✅
def swap {b s : Nat} (x : Ten [b, s]) : Ten [s, b] :=
  reshape x [s, b]

-- ✅
def flatten {b s : Nat} (x : Ten [b, s, 512]) : Ten [b * s, 512] :=
  reshape x [b * s, 512]

-- ✅
def mergeHeads {b s h d : Nat} (x : Ten [b, s, h, d]) : Ten [b, s, h * d] :=
  reshape x [b, s, h * d]

-- ✅
def mergeHeadsReordered {b h s d : Nat} (x : Ten [b, h, s, d]) : Ten [b, s, h * d] :=
  reshape x [b, s, h * d]

-- ❌ correctly rejected
def wrongSplit {b : Nat} (x : Ten [b, 512]) : Ten [b, 8, 60] :=
  reshape x [b, 8, 60]

-- ❌ correctly rejected
def wrongConcrete (x : Ten [2, 512]) : Ten [2, 8, 60] :=
  reshape x [2, 8, 60]

-- ❌ correctly rejected
def wrongFlatten {b s : Nat} (x : Ten [b, s, 512]) : Ten [b * s, 500] :=
  reshape x [b * s, 500]

-- ❌ correctly rejected
def wrongVar {b s : Nat} (x : Ten [b, s]) : Ten [b, b] :=
  reshape x [b, b]

end NatModel
