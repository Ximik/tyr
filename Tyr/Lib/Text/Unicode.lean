/-
  Tyr/Lib/Text/Unicode.lean

  Unicode character classes: letters (L*), numbers (N*) and White_Space.
  The category tables are in `UnicodeData.lean`.
-/
import Tyr.Lib.Text.UnicodeData

namespace tokenizer.unicode

/-- Binary search in sorted non-overlapping inclusive ranges. -/
private def inRanges (ranges : Array (Nat × Nat)) (c : Nat) : Bool := Id.run do
  let mut lo := 0
  let mut hi := ranges.size
  while lo < hi do
    let mid := (lo + hi) / 2
    let (s, e) := ranges[mid]!
    if c < s then
      hi := mid
    else if c > e then
      lo := mid + 1
    else
      return true
  false

/-- `\p{L}`: Unicode letter. -/
def isLetter (c : Char) : Bool := inRanges letterRanges c.toNat

/-- `\p{N}`: Unicode number (Nd, Nl, No). -/
def isNumber (c : Char) : Bool := inRanges numberRanges c.toNat

/-- `\s`: Unicode White_Space property (25 code points). -/
def isWhitespace (c : Char) : Bool :=
  let n := c.toNat
  (n >= 0x09 && n <= 0x0D) || n == 0x20 || n == 0x85 || n == 0xA0 ||
  n == 0x1680 || (n >= 0x2000 && n <= 0x200A) || n == 0x2028 || n == 0x2029 ||
  n == 0x202F || n == 0x205F || n == 0x3000

end tokenizer.unicode
