/-
  Tyr/Tokenizer/Laguna.lean

  Poolside Laguna tokenizer (byte-level BPE) matching the `tokenizer.json` of
  `poolside/Laguna-XS-2.1` (revision c5f36269: vocab 100352, 100026 merges, 70
  added tokens) and the HuggingFace `tokenizers` library behavior exactly:

  - Pre-tokenizer: Split `(?:\r?\n)+(?!\r?\n)` with behavior `MergedWithNext`
    (newline runs attach to the following segment), then GPT-2-style Split
    `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
     ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+` with behavior `Isolated`
    (Unicode-aware `\p{L}` / `\p{N}` / `\s`, single char per number), then
    ByteLevel (no prefix space).
  - Post-processor: TemplateProcessing always prepends `〈|EOS|〉` (id 2).
  - Decoder: ByteLevel.
-/
import Tyr.Tokenizer.Types
import Tyr.Tokenizer.ByteLevel
import Tyr.Tokenizer.WordSplit
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace tokenizer.laguna

open Lean

/-- BOS prepended by the post-processor to every encoding (same token as EOS). -/
def bosTokenId : TokenId := 2
/-- `〈|PAD|〉` id. -/
def padTokenId : TokenId := 9

/-- Surface form of the BOS/EOS token (U+3008/U+3009 angle brackets). -/
def eosToken : String := "〈|EOS|〉"
/-- Surface form of the PAD token. -/
def padToken : String := "〈|PAD|〉"

/-- Default system message baked into chat_template.jinja. -/
def defaultSystemMessage : String :=
  "You are a helpful, conversationally-fluent assistant made by Poolside. " ++
  "You are here to be helpful to users through natural language conversations."

/-- Laguna tokenizer state (byte-level BPE + added tokens). -/
structure LagunaTokenizer where
  vocabSize : Nat
  idToToken : Array String
  tokenToId : Std.HashMap String TokenId
  charToId : Std.HashMap Char TokenId
  mergeLookup : Std.HashMap (TokenId × TokenId) TokenId
  mergePriority : Std.HashMap (TokenId × TokenId) Nat
  /-- All added tokens (ids 0-69, both `special: true` and `special: false`):
      matched atomically in input text, like HF AddedVocabulary. -/
  addedTokens : Std.HashMap String TokenId
  idToAdded : Std.HashMap TokenId String
  /-- Added-token surface forms sorted longest-first for greedy matching. -/
  addedList : Array String
  padTokenId' : TokenId

/-- Helper: parse JSON file. -/
private def parseJsonFile (path : String) : IO Json := do
  let contents <- IO.FS.readFile path
  match Json.parse contents with
  | .ok json => pure json
  | .error err => throw (IO.userError s!"Failed to parse JSON at {path}: {err}")

/-- Build a length-descending list of added tokens (for greedy matching). -/
private def sortAddedByLength (added : Array String) : Array String := Id.run do
  let mut out : Array String := #[]
  for s in added do
    let mut inserted := false
    let mut next : Array String := #[]
    for existing in out do
      if !inserted && s.length > existing.length then
        next := next.push s
        inserted := true
      next := next.push existing
    if !inserted then
      next := next.push s
    out := next
  out

/-- Load the Laguna tokenizer from `<modelDir>/tokenizer.json`. -/
def loadTokenizer (modelDir : String) : IO LagunaTokenizer := do
  let tokJson <- parseJsonFile s!"{modelDir}/tokenizer.json"
  let modelJson <-
    match (tokJson.getObjVal? "model").toOption with
    | some v => pure v
    | none => throw (IO.userError "tokenizer.json missing model")
  let vocabJson <-
    match (modelJson.getObjVal? "vocab").toOption with
    | some v => pure v
    | none => throw (IO.userError "tokenizer.json missing model.vocab")
  let mergesJson <-
    match (modelJson.getObjVal? "merges" >>= (·.getArr?)).toOption with
    | some v => pure v
    | none => throw (IO.userError "tokenizer.json missing model.merges")

  -- Build vocab maps
  let mut maxId : Nat := 0
  let mut tokenToId : Std.HashMap String TokenId := {}
  match vocabJson with
  | .obj kvs =>
    for (tok, idJson) in kvs do
      match idJson.getNat?.toOption with
      | some id =>
        if id ≥ UInt32.size then
          throw (IO.userError s!"Vocab id {id} for token {tok} exceeds the UInt32 token id range")
        if id > maxId then maxId := id
        tokenToId := tokenToId.insert tok id.toUInt32
      | none =>
        throw (IO.userError s!"Invalid vocab id for token {tok}")
  | _ => throw (IO.userError "model.vocab is not an object")

  let vocabSize := maxId + 1
  let mut idToToken : Array String := Array.replicate (maxId + 1) ""
  for (tok, id) in tokenToId.toList do
    let idx := id.toNat
    if idx < idToToken.size then
      idToToken := idToToken.set! idx tok

  -- Build char->id map for single-character (byte-level) tokens
  let mut charToId : Std.HashMap Char TokenId := {}
  for (tok, id) in tokenToId.toList do
    if tok.length == 1 then
      match tok.toList with
      | c :: _ => charToId := charToId.insert c id
      | _ => pure ()

  -- Added tokens (all of them, `special` flag irrelevant for matching)
  let mut addedTokens : Std.HashMap String TokenId := {}
  let mut idToAdded : Std.HashMap TokenId String := {}
  match (tokJson.getObjVal? "added_tokens" >>= (·.getArr?)).toOption with
  | some arr =>
    for entry in arr do
      let content := (entry.getObjVal? "content" >>= (·.getStr?)).toOption
      let id := (entry.getObjVal? "id" >>= (·.getNat?)).toOption
      match content, id with
      | some s, some n =>
        let tid := n.toUInt32
        addedTokens := addedTokens.insert s tid
        idToAdded := idToAdded.insert tid s
      | _, _ => pure ()
  | none => pure ()

  let addedList := sortAddedByLength (List.toArray (addedTokens.toList.map Prod.fst))

  let padTokenId' : TokenId := addedTokens.getD padToken laguna.padTokenId

  -- Build merges
  let mut mergeLookup : Std.HashMap (TokenId × TokenId) TokenId := {}
  let mut mergePriority : Std.HashMap (TokenId × TokenId) Nat := {}
  let mut idx : Nat := 0
  for entry in mergesJson do
    let (leftStr, rightStr) ←
      match entry with
      | .str mergeStr =>
        let parts := mergeStr.splitOn " "
        if parts.length >= 2 then
          pure (parts[0]!, parts[1]!)
        else
          throw (IO.userError s!"Invalid merge entry: {mergeStr}")
      | .arr arr =>
        if arr.size >= 2 then
          match arr[0]!.getStr?.toOption, arr[1]!.getStr?.toOption with
          | some l, some r => pure (l, r)
          | _, _ => throw (IO.userError "Invalid merge entry: non-string pair")
        else
          throw (IO.userError "Invalid merge entry: array too short")
      | _ =>
        throw (IO.userError "Invalid merge entry: unsupported JSON type")
    match tokenToId.get? leftStr, tokenToId.get? rightStr with
    | some leftId, some rightId =>
      let resultStr := leftStr ++ rightStr
      match tokenToId.get? resultStr with
      | some resultId =>
        mergeLookup := mergeLookup.insert (leftId, rightId) resultId
        mergePriority := mergePriority.insert (leftId, rightId) idx
        idx := idx + 1
      | none =>
        throw (IO.userError s!"Missing merge result token: {resultStr}")
    | _, _ =>
      throw (IO.userError s!"Missing merge pair tokens: {leftStr} {rightStr}")

  pure {
    vocabSize
    idToToken
    tokenToId
    charToId
    mergeLookup
    mergePriority
    addedTokens
    idToAdded
    addedList
    padTokenId'
  }

/-! ## Added-token splitting -/

/-- Greedy match for any added token at position i (longest first). -/
private def matchAdded (chars : Array Char) (i : Nat) (added : Array String)
    : Option (String × Nat) := Id.run do
  for s in added do
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

/-- Split text into segments, preserving added tokens as atomic segments. -/
private def splitWithAdded (text : String) (added : Array String)
    : Array (Bool × String) := Id.run do
  let chars := text.toList.toArray
  let mut out : Array (Bool × String) := #[]
  let mut buf : Array Char := #[]
  let mut i : Nat := 0
  while i < chars.size do
    match matchAdded chars i added with
    | some (tok, len) =>
      if !buf.isEmpty then
        out := out.push (false, String.ofList buf.toList)
        buf := #[]
      out := out.push (true, tok)
      i := i + len
    | none =>
      buf := buf.push chars[i]!
      i := i + 1
  if !buf.isEmpty then
    out := out.push (false, String.ofList buf.toList)
  out

/-! ## Pre-tokenizer step 1: newline-run Split with MergedWithNext

  Pattern `(?:\r?\n)+(?!\r?\n)`: maximal runs of `\n` / `\r\n` units.
  Each run is attached to the *following* segment (MergedWithNext); a trailing
  run stays as its own segment. -/

/-- Is `i` the start of a newline unit (`\n` or `\r\n`)? -/
private def isNewlineUnitStart (chars : Array Char) (i : Nat) : Bool :=
  chars[i]! == '\n' ||
  (chars[i]! == '\r' && i + 1 < chars.size && chars[i + 1]! == '\n')

/-- End (exclusive) of the maximal newline run starting at `i`. -/
private def newlineRunEnd (chars : Array Char) (i : Nat) : Nat := Id.run do
  let mut j := i
  while j < chars.size do
    if chars[j]! == '\n' then
      j := j + 1
    else if chars[j]! == '\r' && j + 1 < chars.size && chars[j + 1]! == '\n' then
      j := j + 2
    else
      break
  j

/-- Split into segments at newline runs, merging each run with the next segment. -/
private def splitNewlineMerged (chars : Array Char) : Array String := Id.run do
  let mut out : Array String := #[]
  let mut segStart : Nat := 0
  let mut i : Nat := 0
  while i < chars.size do
    if isNewlineUnitStart chars i then
      let runEnd := newlineRunEnd chars i
      if i > segStart then
        out := out.push (wordSplit.sliceChars chars segStart i)
      segStart := i
      i := runEnd
    else
      i := i + 1
  if chars.size > segStart then
    out := out.push (wordSplit.sliceChars chars segStart chars.size)
  out

/-- Full pre-tokenizer: newline-run split (MergedWithNext), then GPT-2 split
    (Isolated) on each resulting segment. -/
private def pretokenize (text : String) : Array String := Id.run do
  let chars := text.toList.toArray
  let segments := splitNewlineMerged chars
  let mut out : Array String := #[]
  for seg in segments do
    let segChars := seg.toList.toArray
    for piece in wordSplit.splitWords segChars do
      out := out.push piece
  out

/-! ## Byte-level BPE -/

/-- Find the best merge (lowest rank) among adjacent pairs. -/
private def findBestMerge (tok : LagunaTokenizer) (tokens : Array TokenId)
    : Option (Nat × TokenId) := Id.run do
  if tokens.size < 2 then return none
  let mut bestIdx : Option Nat := none
  let mut bestRank : Nat := tok.mergePriority.size + 1
  for i in [:tokens.size - 1] do
    let left := tokens[i]!
    let right := tokens[i + 1]!
    match tok.mergePriority.get? (left, right) with
    | some rank =>
      if rank < bestRank then
        bestRank := rank
        bestIdx := some i
    | none => pure ()
  match bestIdx with
  | none => return none
  | some idx =>
    let left := tokens[idx]!
    let right := tokens[idx + 1]!
    let result := tok.mergeLookup.getD (left, right) left
    return some (idx, result)

private def applyMerge (tokens : Array TokenId) (idx : Nat) (mergeResult : TokenId)
    : Array TokenId := Id.run do
  if idx + 1 >= tokens.size then return tokens
  let mut newTokens := Array.mkEmpty (tokens.size - 1)
  for i in [:idx] do
    newTokens := newTokens.push tokens[i]!
  newTokens := newTokens.push mergeResult
  for i in [idx + 2:tokens.size] do
    newTokens := newTokens.push tokens[i]!
  newTokens

/-- BPE-encode a single pretokenized piece (byte-level). -/
private def encodePiece (tok : LagunaTokenizer) (piece : String) : Array TokenId := Id.run do
  if piece.isEmpty then return #[]
  let byteLevel := tokenizer.stringToByteLevel piece
  let mut tokens : Array TokenId := #[]
  for c in byteLevel.toList do
    match tok.charToId.get? c with
    | some id => tokens := tokens.push id
    | none =>
      -- fallback to string lookup (should not happen: all 256 byte chars are in vocab)
      match tok.tokenToId.get? (String.ofList [c]) with
      | some id => tokens := tokens.push id
      | none => pure ()
  if tokens.isEmpty then return #[]
  let mut changed := true
  while changed do
    match findBestMerge tok tokens with
    | some (idx, result) =>
      tokens := applyMerge tokens idx result
      changed := true
    | none =>
      changed := false
  tokens

/-- Encode text without the post-processor (no BOS prepend). -/
def encodeTextRaw (tok : LagunaTokenizer) (text : String) : Array TokenId := Id.run do
  let segments := splitWithAdded text tok.addedList
  let mut out : Array TokenId := #[]
  for (isAdded, seg) in segments do
    if isAdded then
      match tok.addedTokens.get? seg with
      | some id => out := out.push id
      | none => pure ()
    else
      for piece in pretokenize seg do
        out := out ++ encodePiece tok piece
  out

/-- Encode text exactly like HF `Tokenizer.encode`: BOS (`〈|EOS|〉`, id 2)
    prepended by the TemplateProcessing post-processor. -/
def encodeText (tok : LagunaTokenizer) (text : String) : Array TokenId :=
  #[bosTokenId] ++ encodeTextRaw tok text

/-- Decode byte-level text to UTF-8; fall back to the byte-level form on
    invalid UTF-8 streams (HF emits U+FFFD there instead — only reachable when
    decoding arbitrary id sequences, never for decode(encode(x))). -/
private def byteLevelToStringLossy (s : String) : String :=
  let bytes := tokenizer.byteLevelToBytes s
  match String.fromUTF8? bytes with
  | some out => out
  | none => tokenizer.bytesToByteLevel bytes

/-- Decode token ids to text, keeping added tokens as their surface forms
    (like HF `Tokenizer.decode(ids, skip_special_tokens=false)`). -/
def decodeTokens (tok : LagunaTokenizer) (ids : Array TokenId) : String := Id.run do
  let mut out := ""
  let mut byteLevelBuf := ""
  for id in ids do
    match tok.idToAdded.get? id with
    | some added =>
      if !byteLevelBuf.isEmpty then
        out := out ++ byteLevelToStringLossy byteLevelBuf
        byteLevelBuf := ""
      out := out ++ added
    | none =>
      match tok.idToToken[id.toNat]? with
      | some tokStr => byteLevelBuf := byteLevelBuf ++ tokStr
      | none => pure ()
  if !byteLevelBuf.isEmpty then
    out := out ++ byteLevelToStringLossy byteLevelBuf
  out

/-- Decode one token id to display text. -/
def decodeOne (tok : LagunaTokenizer) (id : TokenId) : String :=
  decodeTokens tok #[id]

/-! ## Chat template (single user turn, default system message,
     add_generation_prompt=true; matches chat_template.jinja) -/

/-- Render the Laguna chat template for a single user turn.
    Ends with `<assistant><think>` when `enableThinking` (jinja default),
    else `<assistant></think>`. -/
def chatTemplate (prompt : String) (enableThinking : Bool := true) : String :=
  eosToken ++ "<system>" ++ defaultSystemMessage ++ "</system>\n" ++
  "<user>" ++ prompt ++ "</user>\n" ++
  "<assistant>" ++ (if enableThinking then "<think>" else "</think>")

end tokenizer.laguna
