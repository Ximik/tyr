/-
  TypeCounterexample2.lean

  Same reshape check, two ways of writing the element count. Deliberately not
  imported by `Examples.lean`; check with:

    lake env lean Examples/TypeCounterexample2.lean

  `Foldl.numel` mirrors `TensorSpec.numelOfShape` (a left fold, as in
  `Tyr/Typed/Spec.lean`) with `Nat` dimensions, so `UInt64` overflow is not a
  factor. `Recursive.numel` computes the same number right-nested. Both use the
  identical default tactic.

  The left fold produces `(b * k) * (d / k)`, so when the cancellation step runs
  `k` and `d / k` are not grouped together and `Nat.mul_div_cancel'`
  (`k ∣ d → k * (d / k) = d`) cannot fire. Reassociation happens only after it,
  leaving omega with the nonlinear atom `b * (k * (d / k))`. The right-nested
  form is already `b * (k * (d / k))` when the cancellation runs.

  Running `simp only [Nat.mul_assoc]` before the cancellation step also makes
  `Foldl` pass: the left fold is not wrong, it just depends on tactic order.

  Every definition is marked ✅ (compiles) or ❌ (fails to elaborate).
-/

namespace Foldl

def numel (s : Array Nat) : Nat :=
  s.foldl (· * ·) 1

structure Ten (s : Array Nat) where

def reshape {s : Array Nat} (_t : Ten s) (s' : Array Nat)
    (_h : numel s = numel s' := by
      simp [numel] <;>
      (try simp only [Nat.mul_div_cancel', *]) <;>
      (try simp only [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]) <;> omega) :
    Ten s' :=
  ⟨⟩

-- ✅
def splitHeads {b : Nat} (x : Ten #[b, 512]) : Ten #[b, 8, 64] :=
  reshape x #[b, 8, 64]

-- ❌ correct, but `8` and `d / 8` end up in different parentheses
def heads {b d : Nat} (h : 8 ∣ d) (x : Ten #[b, d]) : Ten #[b, 8, d / 8] :=
  reshape x #[b, 8, d / 8]

-- ❌ same, with a variable divisor
def headsVar {b d k : Nat} (h : k ∣ d) (x : Ten #[b, d]) : Ten #[b, k, d / k] :=
  reshape x #[b, k, d / k]

-- ❌ correctly rejected: no evidence that k divides d
def headsVarNoHyp {b d k : Nat} (x : Ten #[b, d]) : Ten #[b, k, d / k] :=
  reshape x #[b, k, d / k]

end Foldl

namespace Recursive

def numelL : List Nat → Nat
  | [] => 1
  | d :: ds => d * numelL ds

def numel (s : Array Nat) : Nat :=
  numelL s.toList

structure Ten (s : Array Nat) where

def reshape {s : Array Nat} (_t : Ten s) (s' : Array Nat)
    (_h : numel s = numel s' := by
      simp [numel, numelL] <;>
      (try simp only [Nat.mul_div_cancel', *]) <;>
      (try simp only [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]) <;> omega) :
    Ten s' :=
  ⟨⟩

-- ✅
def splitHeads {b : Nat} (x : Ten #[b, 512]) : Ten #[b, 8, 64] :=
  reshape x #[b, 8, 64]

-- ✅
def heads {b d : Nat} (h : 8 ∣ d) (x : Ten #[b, d]) : Ten #[b, 8, d / 8] :=
  reshape x #[b, 8, d / 8]

-- ✅ same definition that fails under `Foldl`
def headsVar {b d k : Nat} (h : k ∣ d) (x : Ten #[b, d]) : Ten #[b, k, d / k] :=
  reshape x #[b, k, d / k]

-- ❌ correctly rejected: no evidence that k divides d
def headsVarNoHyp {b d k : Nat} (x : Ten #[b, d]) : Ten #[b, k, d / k] :=
  reshape x #[b, k, d / k]

end Recursive
