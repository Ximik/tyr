/-! Helpers on core types that core Lean does not provide. -/

/-- `n / d` rounded up; `0` when `d = 0`, like `/`. -/
def Nat.ceilDiv (n d : Nat) : Nat := (n + d - 1) / d

theorem Nat.ceilDiv_one (n : Nat) : n.ceilDiv 1 = n := by simp [Nat.ceilDiv]

theorem Nat.ceilDiv_pos {n d : Nat} (hn : 0 < n) (hd : 0 < d) : 0 < n.ceilDiv d :=
  Nat.div_pos (by omega) hd

/-- `xs` with `x` appended unless it is already present. -/
def Array.pushUnique [BEq α] (xs : Array α) (x : α) : Array α :=
  if xs.contains x then xs else xs.push x

def Float.pi : Float := 3.14159265358979323846
