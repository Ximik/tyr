/-
  Token Encoding

  BPE encoding implementation.
-/
import Tyr.Lib.Tokenizer.Types
import Tyr.Lib.Tokenizer.ByteLevel
import Tyr.Lib.Tokenizer.Pretokenize

namespace tokenizer

/-- Match a special token at position `i` in `chars`, preferring earlier entries.
    Call with specials sorted by decreasing length for longest-match behavior. -/
private def matchSpecialAt (chars : Array Char) (i : Nat) (specials : Array String)
    : Option (String × Nat) := Id.run do
  for s in specials do
    let pat := s.toList
    let len := pat.length
    if i + len <= chars.size then
      let mut ok := true
      for j in [:len] do
        if chars[i + j]! != pat[j]! then
          ok := false
      if ok then
        return some (s, len)
  none

/-- Split text into `(isSpecial, segment)` while preserving special tokens. -/
private def splitWithSpecials (text : String) (specials : Array String)
    : Array (Bool × String) := Id.run do
  let chars := text.toList.toArray
  let mut out : Array (Bool × String) := #[]
  let mut buffer : Array Char := #[]
  let mut i : Nat := 0

  while i < chars.size do
    match matchSpecialAt chars i specials with
    | some (tok, len) =>
      if !buffer.isEmpty then
        out := out.push (false, String.ofList buffer.toList)
        buffer := #[]
      out := out.push (true, tok)
      i := i + len
    | none =>
      buffer := buffer.push chars[i]!
      i := i + 1

  if !buffer.isEmpty then
    out := out.push (false, String.ofList buffer.toList)
  out

/-- Convert bytes to base token IDs (one token per byte) -/
def bytesToBaseTokens (tok : BPETokenizer) (bytes : ByteArray) : Array UInt32 := Id.run do
  let mut result := Array.mkEmpty bytes.size
  for b in bytes.toList do
    -- Each byte maps to a single-byte token
    let singleByte := ByteArray.mk #[b]
    match tok.bytesToId.get? singleByte with
    | some id => result := result.push id
    | none => pure ()  -- Skip unknown bytes (shouldn't happen with proper vocab)
  return result

/-- Find the highest priority merge pair in a token sequence.
    Returns the index and the merge rule if found. -/
def findBestMerge (tok : BPETokenizer) (tokens : Array UInt32) : Option (Nat × MergeRule) := Id.run do
  if tokens.size < 2 then return none

  let mut best : Option (Nat × Nat) := none

  -- Check each adjacent pair
  for i in [:tokens.size - 1] do
    let left := tokens[i]!
    let right := tokens[i + 1]!
    match tok.mergePriority.get? (left, right) with
    | some priority =>
      match best with
      | none => best := some (i, priority)
      | some (_, bestPriority) =>
        if priority < bestPriority then
          best := some (i, priority)
    | none => pure ()

  match best with
  | some (idx, priority) =>
    match tok.merges[priority]? with
    | some rule => return some (idx, rule)
    | none => return none
  | none => return none

/-- Apply a single BPE merge at a given position -/
def applyMerge (tokens : Array UInt32) (idx : Nat) (mergeResult : UInt32) : Array UInt32 := Id.run do
  if idx + 1 >= tokens.size then return tokens
  let mut newTokens := Array.mkEmpty (tokens.size - 1)
  for i in [:idx] do
    newTokens := newTokens.push tokens[i]!
  newTokens := newTokens.push mergeResult
  for i in [idx + 2:tokens.size] do
    newTokens := newTokens.push tokens[i]!
  return newTokens

/-- BPE encode a single word (pretokenized chunk) -/
partial def encodeWord (tok : BPETokenizer) (word : String) : Array UInt32 := Id.run do
  -- Convert to byte-level then to base tokens
  let bytes := word.toUTF8
  let mut tokens := bytesToBaseTokens tok bytes

  if tokens.isEmpty then return #[]

  -- Iteratively apply merges until no more can be applied
  let mut changed := true
  let mut iterations := 0
  let maxIter := 10000  -- Safety limit

  while changed && iterations < maxIter do
    changed := false
    iterations := iterations + 1
    match findBestMerge tok tokens with
    | some (idx, rule) =>
      tokens := applyMerge tokens idx rule.result
      changed := true
    | none => pure ()

  return tokens

/-- Encode a full text string to tokens -/
def encode (tok : BPETokenizer) (text : String) : Array UInt32 := Id.run do
  -- Pretokenize first
  let chunks := pretokenizeFull text
  let mut encResult : Array UInt32 := #[]

  for chunk in chunks do
    let tokens := encodeWord tok chunk
    encResult := encResult ++ tokens

  return encResult

/-- Encode with special token handling.
    Special tokens in the input are detected and replaced with their IDs. -/
def encodeWithSpecials (tok : BPETokenizer) (text : String) : Array UInt32 := Id.run do
  if tok.specialTokens.isEmpty then
    return encode tok text

  let specialList := (tok.specialTokens.toList.map Prod.fst).toArray.insertionSort (·.length > ·.length)
  let segments := splitWithSpecials text specialList

  let mut out : Array UInt32 := #[]
  for (isSpecial, segment) in segments do
    if isSpecial then
      match tok.specialTokens.get? segment with
      | some id => out := out.push id
      | none =>
        -- Should not happen because we matched from `specialList`, but keep safe fallback.
        out := out ++ encode tok segment
    else
      out := out ++ encode tok segment
  out

/-- Get the number of tokens for a text (without returning them) -/
def countTokens (tok : BPETokenizer) (text : String) : Nat :=
  (encodeWithSpecials tok text).size

end tokenizer
