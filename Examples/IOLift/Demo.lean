import Tyr.Typed.Ops
import Examples.IOLift.Linter

/-! Toy: tensor ops as honest `IO` actions, with operators lifted over `IO` so
formulas need a single `←`. See `Examples.IOLift.Linter` for the reuse check. -/

namespace Examples.IOLift
open torch

abbrev T (s : Shape) := Tensor ⟨s, .Float32, .CPU⟩

/-- Operands may be plain tensors or pending actions; the shape flows out of either. -/
class IntoIO (α : Type) (β : outParam Type) where
  into : α → IO β

instance {s : Shape} : IntoIO (T s) (T s) := ⟨pure⟩
instance {s : Shape} : IntoIO (IO (T s)) (T s) := ⟨id⟩

private def ofExcept (op : String) : Except String α → IO α
  | .ok a => pure a
  | .error e => throw (IO.userError s!"{op}: {e}")

/-! Possibly nondeterministic ops (BLAS, reductions) live in `IO`; native errors become `throw`. -/

/-- Counts native matmuls, so the demo can show how often each one actually runs. -/
initialize mmCalls : IO.Ref Nat ← IO.mkRef 0

def mm {m k n : UInt64} [IntoIO α (T #[m, k])] [IntoIO β (T #[k, n])]
    (a : α) (b : β) : IO (T #[m, n]) := do
  mmCalls.modify (· + 1)
  ofExcept "mm" (Tensor.mm? (← IntoIO.into a) (← IntoIO.into b))

def softmax {s : Shape} [IntoIO α (T s)] (x : α) : IO (T s) := do
  ofExcept "softmax" (Tensor.softmax? (← IntoIO.into x))

infixl:70 " ⬝ " => mm

/-! Elementwise ops are deterministic per machine, so they stay pure and lift for free.
Low priority keeps plain `T + T` on the existing pure instance. -/

instance (priority := low) {s : Shape} [IntoIO α (T s)] [IntoIO β (T s)] :
    HAdd α β (IO (T s)) where
  hAdd x y := return (← IntoIO.into x) + (← IntoIO.into y)

instance (priority := low) {s : Shape} [IntoIO α (T s)] [IntoIO β (T s)] :
    HMul α β (IO (T s)) where
  hMul x y := return (← IntoIO.into x) * (← IntoIO.into y)

/-! Model code: formulas read like math, one `←` at the call site. -/

variable {b d h n : UInt64}

def mlp (x : T #[b, d]) (w₁ : T #[d, h]) (w₂ : T #[h, d]) : IO (T #[b, d]) :=
  x + (Tensor.relu <$> (x ⬝ w₁)) ⬝ w₂

def attention (q : T #[n, d]) (kt : T #[d, n]) (v : T #[n, d]) : IO (T #[n, d]) :=
  softmax (q ⬝ kt) ⬝ v

/-- Reuse done right: `←` pins one concrete value. -/
def gated (x : T #[b, d]) (w : T #[d, d]) : IO (T #[b, d]) := do
  let g ← x ⬝ w
  return g * g + g

/-- The trap: `g` is a recipe, so `x ⬝ w` runs three times. The linter flags it. -/
def gatedBad (x : T #[b, d]) (w : T #[d, d]) : IO (T #[b, d]) :=
  let g := x ⬝ w
  g * g + g

/-- The explicit version of the trap: two `←` run the matmul twice. Also flagged. -/
def squaredBad (x : T #[b, d]) (w : T #[d, d]) : IO (T #[b, d]) := do
  return (← x ⬝ w) * (← x ⬝ w)

end Examples.IOLift
