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

/-- The little-endian `UInt16` at `offset`; `none` past the end. -/
def ByteArray.readUInt16LE? (bytes : ByteArray) (offset : Nat) : Option UInt16 :=
  if offset + 2 > bytes.size then none
  else some (bytes[offset]!.toUInt16 ||| (bytes[offset + 1]!.toUInt16 <<< 8))

/-- The little-endian `UInt32` at `offset`; `none` past the end. -/
def ByteArray.readUInt32LE? (bytes : ByteArray) (offset : Nat) : Option UInt32 :=
  if offset + 4 > bytes.size then none
  else some (bytes[offset]!.toUInt32 ||| (bytes[offset + 1]!.toUInt32 <<< 8) |||
    (bytes[offset + 2]!.toUInt32 <<< 16) ||| (bytes[offset + 3]!.toUInt32 <<< 24))
