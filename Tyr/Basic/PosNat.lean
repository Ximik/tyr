import Lean.Data.Json

/-!
# Tyr.Basic.PosNat

`PosNat`, a positive count. Literals, JSON decoding and string parsing all
reject zero, so a `PosNat` never needs a clamp where it is used. `Option PosNat`
is the optional form; an alias such as `TopK` documents what its `none` means.
-/

namespace torch

/-- A positive count. -/
abbrev PosNat := { n : Nat // 0 < n }

namespace PosNat

instance : OfNat PosNat (n + 1) := ⟨⟨n + 1, Nat.succ_pos n⟩⟩

instance : OfNat (Option PosNat) (n + 1) := ⟨some ⟨n + 1, Nat.succ_pos n⟩⟩

instance : ToString PosNat := ⟨fun n => toString n.val⟩

instance : ToString (Option PosNat) where
  toString
    | none => "none"
    | some n => toString n.val

instance : Lean.ToJson PosNat := ⟨fun n => Lean.toJson n.val⟩

instance : Lean.FromJson PosNat where
  fromJson? json := do
    let n ← Lean.fromJson? (α := Nat) json
    if h : 0 < n then pure ⟨n, h⟩ else throw "expected a positive count, got 0"

/-- A positive count, or `none` for the word `none`. -/
def parseOrNone? (s : String) : Option (Option PosNat) :=
  if s == "none" then
    some none
  else
    match s.toNat? with
    | some (n + 1) => some (some ⟨n + 1, Nat.succ_pos n⟩)
    | _ => none

end PosNat

end torch
