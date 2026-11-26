/-
  Tokenizer Core Types

  BPE tokenizer types for NanoProof.
-/
import Std.Data.HashMap

namespace tokenizer

/-- A BPE merge rule: left + right → result -/
structure MergeRule where
  left : UInt32
  right : UInt32
  result : UInt32
  deriving Repr, BEq

/-- BPE Tokenizer state -/
structure BPETokenizer where
  /-- Total vocabulary size -/
  vocabSize : Nat
  /-- Token ID → byte sequence for decoding -/
  idToBytes : Array ByteArray
  /-- Byte sequence → token ID for base vocab lookup -/
  bytesToId : Std.HashMap ByteArray UInt32
  /-- Ordered merge rules (by priority) -/
  merges : Array MergeRule
  /-- (left, right) → result for fast merge lookup -/
  mergeLookup : Std.HashMap (UInt32 × UInt32) UInt32
  /-- (left, right) → merge priority/rank (lower is better) -/
  mergePriority : Std.HashMap (UInt32 × UInt32) Nat
  /-- Special token string → ID -/
  specialTokens : Std.HashMap String UInt32
  /-- Token ID → special token string (for decoding) -/
  idToSpecial : Std.HashMap UInt32 String

/-- Check if a token is a special token -/
def BPETokenizer.isSpecialToken (tok : BPETokenizer) (id : UInt32) : Bool :=
  tok.idToSpecial.contains id

/-- Get bytes for a token (returns none if invalid) -/
def BPETokenizer.getTokenBytes (tok : BPETokenizer) (id : UInt32) : Option ByteArray :=
  tok.idToBytes[id.toNat]?

end tokenizer
