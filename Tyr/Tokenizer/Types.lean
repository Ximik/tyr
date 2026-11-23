/-
  Tokenizer Core Types

  BPE tokenizer types for NanoProof.
-/
import Std.Data.HashMap

namespace tokenizer

/-- Token ID type (UInt32 for efficiency) -/
abbrev TokenId := UInt32

/-- A BPE merge rule: left + right → result -/
structure MergeRule where
  left : TokenId
  right : TokenId
  result : TokenId
  deriving Repr, BEq

/-- BPE Tokenizer state -/
structure BPETokenizer where
  /-- Total vocabulary size -/
  vocabSize : Nat
  /-- Token ID → byte sequence for decoding -/
  idToBytes : Array ByteArray
  /-- Byte sequence → token ID for base vocab lookup -/
  bytesToId : Std.HashMap ByteArray TokenId
  /-- Ordered merge rules (by priority) -/
  merges : Array MergeRule
  /-- (left, right) → result for fast merge lookup -/
  mergeLookup : Std.HashMap (TokenId × TokenId) TokenId
  /-- (left, right) → merge priority/rank (lower is better) -/
  mergePriority : Std.HashMap (TokenId × TokenId) Nat
  /-- Special token string → ID -/
  specialTokens : Std.HashMap String TokenId
  /-- Token ID → special token string (for decoding) -/
  idToSpecial : Std.HashMap TokenId String

/-- Check if a token is a special token -/
def BPETokenizer.isSpecialToken (tok : BPETokenizer) (id : TokenId) : Bool :=
  tok.idToSpecial.contains id

/-- Get bytes for a token (returns none if invalid) -/
def BPETokenizer.getTokenBytes (tok : BPETokenizer) (id : TokenId) : Option ByteArray :=
  tok.idToBytes[id.toNat]?

end tokenizer
