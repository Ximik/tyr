/-
  Tyr/Lib/Text/Nfc.lean

  Unicode Normalization Form C (UAX #15): full canonical decomposition,
  canonical ordering of combining marks, then canonical composition. Tables
  are in `NfcData.lean`; Hangul syllables are decomposed and composed
  algorithmically.
-/
import Std.Data.HashMap
import Tyr.Lib.Text.NfcData

namespace tokenizer.nfc

private def hex (s : String) : Nat :=
  s.foldl (fun n c =>
    let d := if c.isDigit then c.toNat - '0'.toNat else c.toNat - 'A'.toNat + 10
    16 * n + d) 0

private def entries (table : String) : Array String :=
  (table.splitOn ";").toArray

private def fields (entry : String) : Array Nat :=
  ((entry.splitOn " ").filter (!·.isEmpty)).toArray.map hex

private def classTable : Std.HashMap Nat Nat := Id.run do
  let mut out : Std.HashMap Nat Nat := {}
  for entry in entries combiningClasses do
    let f := fields entry
    for cp in [f[0]!:f[1]! + 1] do
      out := out.insert cp f[2]!
  out

private def decompositionTable : Std.HashMap Nat (Array Nat) := Id.run do
  let mut out : Std.HashMap Nat (Array Nat) := {}
  for entry in entries decompositions do
    match entry.splitOn ":" with
    | [cp, full] => out := out.insert (hex cp) (fields full)
    | _ => pure ()
  out

private def compositionTable : Std.HashMap (Nat × Nat) Nat := Id.run do
  let mut out : Std.HashMap (Nat × Nat) Nat := {}
  for entry in entries compositions do
    let f := fields entry
    out := out.insert (f[0]!, f[1]!) f[2]!
  out

private def sBase : Nat := 0xAC00
private def lBase : Nat := 0x1100
private def vBase : Nat := 0x1161
private def tBase : Nat := 0x11A7
private def lCount : Nat := 19
private def vCount : Nat := 21
private def tCount : Nat := 28
private def nCount : Nat := vCount * tCount
private def sCount : Nat := lCount * nCount

def combiningClass (cp : Nat) : Nat :=
  classTable.getD cp 0

private def decompose (cp : Nat) (out : Array Nat) : Array Nat :=
  if sBase ≤ cp && cp < sBase + sCount then
    let s := cp - sBase
    let out := out.push (lBase + s / nCount) |>.push (vBase + (s % nCount) / tCount)
    if s % tCount == 0 then out else out.push (tBase + s % tCount)
  else
    match decompositionTable.get? cp with
    | some full => out ++ full
    | none => out.push cp

/-- Stable sort of each run of nonzero combining class by class. -/
private def reorder (cps : Array Nat) : Array Nat := Id.run do
  let mut out := cps
  for i in [1:out.size] do
    let c := combiningClass out[i]!
    if c != 0 then
      let mut j := i
      while 0 < j && combiningClass out[j - 1]! > c do
        out := out.swapIfInBounds j (j - 1)
        j := j - 1
  out

private def composePair (a b : Nat) : Option Nat :=
  if lBase ≤ a && a < lBase + lCount && vBase ≤ b && b < vBase + vCount then
    some (sBase + ((a - lBase) * vCount + (b - vBase)) * tCount)
  else if sBase ≤ a && a < sBase + sCount && (a - sBase) % tCount == 0 && tBase < b && b < tBase + tCount then
    some (a + (b - tBase))
  else
    compositionTable.get? (a, b)

private def compose (cps : Array Nat) : Array Nat := Id.run do
  let mut out : Array Nat := #[]
  let mut starter : Option Nat := none
  -- Combining class of the last character appended after the starter.
  let mut lastClass : Nat := 0
  for cp in cps do
    let c := combiningClass cp
    if let some s := starter then
      let unblocked := out.size == s + 1 || (lastClass != 0 && lastClass < c)
      if unblocked then
        if let some composite := composePair out[s]! cp then
          out := out.set! s composite
          continue
    if c == 0 then
      starter := some out.size
      lastClass := 0
    else
      lastClass := c
    out := out.push cp
  out

def normalize (text : String) : String :=
  -- No character below U+0300 decomposes into or composes with another.
  if text.all (fun c : Char => c.toNat < 0x300) then text else
    let decomposed := text.foldl (fun (out : Array Nat) (c : Char) => decompose c.toNat out) #[]
    String.ofList ((compose (reorder decomposed)).toList.map Char.ofNat)

end tokenizer.nfc
