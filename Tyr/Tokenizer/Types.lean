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
  deriving Repr, Inhabited, BEq

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
  deriving Inhabited

/-- Result of tokenization -/
structure TokenizeResult where
  tokens : Array TokenId
  specialPositions : Array (Nat × String)  -- (position, special token)
  deriving Repr, Inhabited

/-- Create an empty tokenizer -/
def BPETokenizer.empty : BPETokenizer :=
  { vocabSize := 0
  , idToBytes := #[]
  , bytesToId := {}
  , merges := #[]
  , mergeLookup := {}
  , mergePriority := {}
  , specialTokens := {}
  , idToSpecial := {}
  }

/-- Get vocabulary size -/
def BPETokenizer.getVocabSize (tok : BPETokenizer) : Nat := tok.vocabSize

/-- Check if a token ID is valid -/
def BPETokenizer.isValidToken (tok : BPETokenizer) (id : TokenId) : Bool :=
  id.toNat < tok.idToBytes.size

/-- Check if a token is a special token -/
def BPETokenizer.isSpecialToken (tok : BPETokenizer) (id : TokenId) : Bool :=
  tok.idToSpecial.contains id

/-- Get bytes for a token (returns none if invalid) -/
def BPETokenizer.getTokenBytes (tok : BPETokenizer) (id : TokenId) : Option ByteArray :=
  tok.idToBytes[id.toNat]?

/-- Get special token string (returns none if not special) -/
def BPETokenizer.getSpecialString (tok : BPETokenizer) (id : TokenId) : Option String :=
  tok.idToSpecial.get? id

end tokenizer
