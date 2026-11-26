/-
  Tyr/Lib/Tokenizer/WordSplit.lean

  The pre-tokenizer split
  `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
   ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
  with Unicode character classes, each match its own piece.
-/

import Tyr.Lib.Text.Unicode

namespace tokenizer.wordSplit

def isCRLF (c : Char) : Bool := c == '\r' || c == '\n'

def sliceChars (chars : Array Char) (lo hi : Nat) : String :=
  String.ofList (chars.extract lo hi).toList

/-! ## Pre-tokenizer step 2: GPT-2-style Split with Isolated behavior

  `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
   ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
  Alternatives are tried in pattern order (leftmost-first semantics). -/

/-- Match contraction `(?i:'s|'t|'re|'ve|'m|'ll|'d)` (ASCII case fold). -/
private def matchContraction (chars : Array Char) (i : Nat) : Option Nat := Id.run do
  if i >= chars.size then return none
  if chars[i]! != '\'' then return none
  let opts := #["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
  for opt in opts do
    let pat := opt.toList
    let len := pat.length
    if i + len <= chars.size then
      let mut ok := true
      for j in [:len] do
        if chars[i + j]!.toLower != pat[j]!.toLower then
          ok := false
      if ok then
        return some len
  none

/-- Match `[^\r\n\p{L}\p{N}]?\p{L}+`. -/
private def matchWord (chars : Array Char) (i : Nat) : Option Nat := Id.run do
  if i >= chars.size then return none
  let c := chars[i]!
  if unicode.isLetter c then
    let mut j := i
    while j < chars.size && unicode.isLetter chars[j]! do
      j := j + 1
    return some (j - i)
  else if !isCRLF c && !unicode.isLetter c && !unicode.isNumber c then
    if i + 1 < chars.size && unicode.isLetter chars[i + 1]! then
      let mut j := i + 1
      while j < chars.size && unicode.isLetter chars[j]! do
        j := j + 1
      return some (j - i)
    else
      return none
  else
    return none

/-- Match ` ?[^\s\p{L}\p{N}]+[\r\n]*`. -/
private def matchPunct (chars : Array Char) (i : Nat) : Option Nat := Id.run do
  if i >= chars.size then return none
  let mut j := i
  if chars[j]! == ' ' then
    j := j + 1
  if j >= chars.size then
    return none
  else if unicode.isWhitespace chars[j]! || unicode.isLetter chars[j]! || unicode.isNumber chars[j]! then
    return none
  else
    while j < chars.size && !unicode.isWhitespace chars[j]! && !unicode.isLetter chars[j]! && !unicode.isNumber chars[j]! do
      j := j + 1
    while j < chars.size && isCRLF chars[j]! do
      j := j + 1
    return some (j - i)

/-- Match `\s*[\r\n]+` (longest ws prefix ending right after its last newline). -/
private def matchWhitespaceNewline (chars : Array Char) (i : Nat) : Option Nat := Id.run do
  if i >= chars.size then return none
  let mut j := i
  let mut lastNewline : Option Nat := none
  while j < chars.size && unicode.isWhitespace chars[j]! do
    if isCRLF chars[j]! then
      lastNewline := some j
    j := j + 1
  match lastNewline with
  | some idx => return some (idx + 1 - i)
  | none => return none

/-- Match `\s+(?!\S)`: longest whitespace run whose end is *not* followed by a
    non-whitespace char (regex backtracking semantics: a run not reaching the
    end of input leaves its last whitespace char for the next match). -/
private def matchWhitespaceEnd (chars : Array Char) (i : Nat) : Option Nat := Id.run do
  if i >= chars.size then return none
  if !unicode.isWhitespace chars[i]! then
    return none
  let mut j := i
  while j < chars.size && unicode.isWhitespace chars[j]! do
    j := j + 1
  let run := j - i
  if j == chars.size then
    return some run
  else if run >= 2 then
    return some (run - 1)
  else
    return none

/-- Match `\s+`. -/
private def matchWhitespace (chars : Array Char) (i : Nat) : Option Nat := Id.run do
  if i >= chars.size then return none
  if !unicode.isWhitespace chars[i]! then
    return none
  let mut j := i
  while j < chars.size && unicode.isWhitespace chars[j]! do
    j := j + 1
  return some (j - i)

/-- Split `chars` from `start` by the pattern above, each match its own piece. -/
def splitWords (chars : Array Char) (start : Nat := 0) : Array String := Id.run do
  let mut out : Array String := #[]
  let mut i : Nat := start
  while i < chars.size do
    let len :=
      match matchContraction chars i with
      | some n => n
      | none =>
        match matchWord chars i with
        | some n => n
        | none =>
          if i < chars.size && unicode.isNumber chars[i]! then 1 else
          match matchPunct chars i with
          | some n => n
          | none =>
            match matchWhitespaceNewline chars i with
            | some n => n
            | none =>
              match matchWhitespaceEnd chars i with
              | some n => n
              | none =>
                match matchWhitespace chars i with
                | some n => n
                | none => 1
    out := out.push (sliceChars chars i (i + len))
    i := i + len
  out

end tokenizer.wordSplit
