/-
  Tyr/Tokenizer/Training.lean

  Tokenizer training infrastructure for BPE tokenizers.

  Based on nanochat's tok_train.py:
  - Configurable vocabulary size (default 32768)
  - GPT-4 style pretokenization with modifications
  - Special token support
  - Token bytes mapping generation for BPB evaluation
-/
import Tyr.Tokenizer.Types
import Tyr.Tokenizer.TokenBytes
import Tyr.Tokenizer.SpecialTokens
import Tyr.Log

namespace tokenizer

open torch.Log

/-- GPT-4 style split pattern (modified for smaller vocab).
    Changes \\d{1,3} to \\d{1,2} to handle numbers more granularly. -/
def gpt4SplitPattern : String :=
  "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}{1,2}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"

/-- Default special tokens for chat models.
    Mirrors nanochat's SPECIAL_TOKENS ordering exactly. -/
def defaultChatSpecialTokens : Array String := #[
  "<|bos|>",              -- Beginning of sequence
  "<|user_start|>",       -- User message start
  "<|user_end|>",         -- User message end
  "<|assistant_start|>",  -- Assistant message start
  "<|assistant_end|>",    -- Assistant message end
  "<|python_start|>",     -- Tool (python) call start
  "<|python_end|>",       -- Tool (python) call end
  "<|output_start|>",     -- Tool output start
  "<|output_end|>"        -- Tool output end
]

/-- Configuration for BPE tokenizer training -/
structure TrainConfig where
  /-- Target vocabulary size (excluding special tokens) -/
  vocabSize : Nat := 32768
  /-- Maximum characters to use for training -/
  maxChars : Nat := 10_000_000_000  -- 10B chars default
  /-- Maximum documents to process per batch -/
  docCap : Nat := 10_000
  /-- Regex pattern for pretokenization.
      Uses GPT-4 style but with 1-2 digits instead of 1-3.
      This saves tokens on smaller vocab sizes. -/
  splitPattern : String := gpt4SplitPattern
  /-- Special tokens to add to vocabulary -/
  specialTokens : Array String := defaultChatSpecialTokens
  /-- Random seed for shuffling -/
  seed : UInt64 := 42
  deriving Repr, Inhabited

/-- Training statistics collected during tokenizer training -/
structure TrainStats where
  /-- Total characters processed -/
  totalChars : Nat := 0
  /-- Total documents processed -/
  totalDocs : Nat := 0
  /-- Number of merge operations performed -/
  numMerges : Nat := 0
  /-- Final vocabulary size -/
  finalVocabSize : Nat := 0
  /-- Compression ratio (chars per token) -/
  compressionRatio : Float := 0.0
  deriving Repr, Inhabited

/-- Result of tokenizer training -/
structure TrainResult where
  /-- The trained tokenizer -/
  tokenizer : BPETokenizer
  /-- Token bytes mapping for BPB evaluation -/
  tokenBytes : TokenBytes
  /-- Training statistics -/
  stats : TrainStats

/-! ## BPE Training Algorithm

The BPE training algorithm works as follows:

1. Initialize vocabulary with all single bytes (0-255)
2. Add special tokens to vocabulary
3. While vocabulary size < target:
   a. Count all adjacent token pairs in corpus
   b. Find the most frequent pair
   c. Create a new token by merging the pair
   d. Record the merge rule
   e. Replace all occurrences of the pair with the new token

This implementation provides the infrastructure for training.
The actual training would typically use a fast Rust implementation
(like rustbpe) for performance on large corpora.
-/

/-- Initialize a base tokenizer with byte vocabulary.
    Creates tokens for all 256 bytes, plus special tokens. -/
def initBaseTokenizer (specialTokens : Array String) : BPETokenizer := Id.run do
  let mut idToBytes : Array ByteArray := #[]
  let mut bytesToId : Std.HashMap ByteArray TokenId := {}

  -- Add all 256 byte tokens
  for i in [:256] do
    let ba := ByteArray.mk #[i.toUInt8]
    idToBytes := idToBytes.push ba
    bytesToId := bytesToId.insert ba i.toUInt32

  let mut tok : BPETokenizer := {
    vocabSize := 256
    idToBytes := idToBytes
    bytesToId := bytesToId
    merges := #[]
    mergeLookup := {}
    mergePriority := {}
    specialTokens := {}
    idToSpecial := {}
  }

  -- Add special tokens
  tok := addSpecialTokens tok specialTokens 256

  -- Update vocab size to include special tokens
  tok := { tok with vocabSize := 256 + specialTokens.size }

  return tok

/-- Count pair frequencies in a corpus.
    Returns a map from (left_id, right_id) → count. -/
def countPairs (tokenizedDocs : Array (Array TokenId))
    : Std.HashMap (TokenId × TokenId) Nat := Id.run do
  let mut counts : Std.HashMap (TokenId × TokenId) Nat := {}

  for doc in tokenizedDocs do
    for i in [:doc.size - 1] do
      let pair := (doc[i]!, doc[i + 1]!)
      let currentCount := counts.getD pair 0
      counts := counts.insert pair (currentCount + 1)

  return counts

/-- Find the most frequent pair -/
def findBestPair (counts : Std.HashMap (TokenId × TokenId) Nat)
    : Option ((TokenId × TokenId) × Nat) := Id.run do
  let mut best : Option ((TokenId × TokenId) × Nat) := none

  for (pair, count) in counts do
    match best with
    | none => best := some (pair, count)
    | some (_, bestCount) =>
      if count > bestCount then
        best := some (pair, count)

  return best

/-- Replace all occurrences of a pair with a new token in a document -/
def replacePair (doc : Array TokenId) (left right newToken : TokenId)
    : Array TokenId := Id.run do
  if doc.size < 2 then return doc

  let mut result : Array TokenId := #[]
  let mut i := 0

  while i < doc.size do
    if i + 1 < doc.size && doc[i]! == left && doc[i + 1]! == right then
      result := result.push newToken
      i := i + 2
    else
      result := result.push doc[i]!
      i := i + 1

  return result

/-- Perform one BPE merge step -/
def bpeMergeStep (tok : BPETokenizer) (docs : Array (Array TokenId))
    (pair : TokenId × TokenId)
    : BPETokenizer × Array (Array TokenId) := Id.run do
  let (left, right) := pair
  let newId := tok.vocabSize.toUInt32

  -- Create new token bytes by concatenating the pair's bytes
  let leftBytes := tok.idToBytes[left.toNat]?.getD ByteArray.empty
  let rightBytes := tok.idToBytes[right.toNat]?.getD ByteArray.empty
  let newBytes := leftBytes ++ rightBytes

  -- Update tokenizer
  let mergePriority :=
    match tok.mergePriority.get? (left, right) with
    | some _ => tok.mergePriority
    | none => tok.mergePriority.insert (left, right) tok.merges.size
  let newTok : BPETokenizer := {
    tok with
    vocabSize := tok.vocabSize + 1
    idToBytes := tok.idToBytes.push newBytes
    bytesToId := tok.bytesToId.insert newBytes newId
    merges := tok.merges.push { left, right, result := newId }
    mergeLookup := tok.mergeLookup.insert (left, right) newId
    mergePriority := mergePriority
  }

  -- Replace pair in all documents
  let newDocs := docs.map (replacePair · left right newId)

  return (newTok, newDocs)

/-- Train a BPE tokenizer on a corpus of documents.
    This is a simple implementation for testing.
    For production use, prefer rustbpe or similar. -/
def trainBPE (docs : Array String) (config : TrainConfig) (log : Handlers := {}) : IO TrainResult := do
  -- Each merge's id is the vocab size before it, so every id stays below this bound.
  if config.vocabSize > UInt32.size then
    throw <| IO.userError s!"Target vocab size {config.vocabSize} exceeds the UInt32 token id range"
  log.onInfo s!"Starting BPE training with target vocab size: {config.vocabSize}"

  -- Initialize tokenizer with base vocabulary
  let mut tok := initBaseTokenizer config.specialTokens
  log.onInfo s!"Initialized base vocabulary: {tok.vocabSize} tokens"

  -- Tokenize documents to bytes
  let mut tokenizedDocs : Array (Array TokenId) := #[]
  let mut totalChars := 0

  for doc in docs do
    totalChars := totalChars + doc.length
    let bytes := doc.toUTF8
    let tokenIds : Array TokenId := bytes.toList.toArray.map (·.toUInt32)
    tokenizedDocs := tokenizedDocs.push tokenIds

  log.onInfo s!"Processed {docs.size} documents, {totalChars} characters"

  -- Perform merges until target vocab size reached
  let targetMerges := config.vocabSize - tok.vocabSize
  let mut numMerges := 0

  while tok.vocabSize < config.vocabSize do
    -- Count pairs
    let counts := countPairs tokenizedDocs

    -- Find best pair
    match findBestPair counts with
    | none =>
      log.onInfo "No more pairs to merge"
      break
    | some ((left, right), count) =>
      if count < 2 then
        log.onInfo "Best pair frequency < 2, stopping"
        break

      -- Perform merge
      let (newTok, newDocs) := bpeMergeStep tok tokenizedDocs (left, right)
      tok := newTok
      tokenizedDocs := newDocs
      numMerges := numMerges + 1

      -- Progress update
      if numMerges % 1000 == 0 then
        let progress := (numMerges.toFloat / targetMerges.toFloat) * 100.0
        log.onInfo s!"Merge {numMerges}/{targetMerges} ({progress.floor}%): vocab size = {tok.vocabSize}"

  log.onInfo s!"Training complete: {tok.vocabSize} tokens, {numMerges} merges"

  -- Calculate compression ratio
  let totalTokens := tokenizedDocs.foldl (fun acc doc => acc + doc.size) 0
  let compressionRatio := totalChars.toFloat / totalTokens.toFloat

  -- Build token bytes mapping
  let tokenBytes := TokenBytes.fromTokenizer tok

  return {
    tokenizer := tok
    tokenBytes := tokenBytes
    stats := {
      totalChars := totalChars
      totalDocs := docs.size
      numMerges := numMerges
      finalVocabSize := tok.vocabSize
      compressionRatio := compressionRatio
    }
  }

/-! ## Pretokenization

Pretokenization splits text before BPE encoding.
GPT-4 uses regex patterns to handle:
- Contractions ('s, 'd, etc.)
- Words (letters only)
- Numbers (1-2 digits at a time)
- Punctuation
- Whitespace

This is a placeholder - actual implementation would use
regex via FFI to a Rust/C++ library.
-/

/-- Simple word-level pretokenization (placeholder).
    Real implementation would use regex patterns. -/
def simplePretokenize (text : String) : Array String := Id.run do
  -- Simple split on whitespace and keep punctuation separate
  let mut tokens : Array String := #[]
  let mut current := ""

  for c in text.toList do
    if c.isWhitespace then
      if !current.isEmpty then
        tokens := tokens.push current
        current := ""
      tokens := tokens.push c.toString
    else if c.isAlphanum then
      current := current.push c
    else
      -- Punctuation: emit current word, then punctuation
      if !current.isEmpty then
        tokens := tokens.push current
        current := ""
      tokens := tokens.push c.toString

  if !current.isEmpty then
    tokens := tokens.push current

  return tokens

end tokenizer
