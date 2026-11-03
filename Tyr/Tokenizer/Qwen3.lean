/-
  Tyr/Tokenizer/Qwen3.lean

  Qwen3 tokenizer loader + encoder for Flux Klein text encoder.
  Matches HuggingFace tokenizer.json + chat template behavior.
-/
import Tyr.Tokenizer.Types
import Tyr.Tokenizer.ByteLevel
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace tokenizer.qwen3

open Lean

/-- Helper: parse JSON file. -/
private def parseJsonFile (path : String) : IO Json := do
  let contents <- IO.FS.readFile path
  match Json.parse contents with
  | .ok json => pure json
  | .error err => throw (IO.userError s!"Failed to parse JSON at {path}: {err}")

private def getObjVal? (j : Json) (key : String) : Option Json :=
  match j with
  | .obj kvs => Std.TreeMap.Raw.get? kvs key
  | _ => none

private def getStr? (j : Json) : Option String :=
  match j with
  | .str s => some s
  | _ => none

private def getArr? (j : Json) : Option (Array Json) :=
  match j with
  | .arr a => some a
  | _ => none

private def fromJson? {α} [FromJson α] (j : Json) : Option α :=
  match (FromJson.fromJson? j : Except String α) with
  | .ok v => some v
  | .error _ => none

private def getNat? (j : Json) : Option Nat :=
  fromJson? j

private def stringCharLength (s : String) : Nat :=
  s.toList.length

/-- Tokenizer for Qwen3 (ByteLevel BPE). -/
structure QwenTokenizer where
  vocabSize : Nat
  idToToken : Array String
  tokenToId : Std.HashMap String TokenId
  charToId : Std.HashMap Char TokenId
  merges : Array MergeRule
  mergeLookup : Std.HashMap (TokenId × TokenId) TokenId
  mergePriority : Std.HashMap (TokenId × TokenId) Nat
  specialTokens : Std.HashMap String TokenId
  idToSpecial : Std.HashMap TokenId String
  specialList : Array String
  unkToken : Option TokenId
  padToken : TokenId
  deriving Inhabited

/-- Build a length-descending list of special tokens (for greedy matching). -/
private def sortSpecials (specials : Array String) : Array String := Id.run do
  let mut out : Array String := #[]
  for s in specials do
    let mut inserted := false
    let mut next : Array String := #[]
    for existing in out do
      if !inserted && stringCharLength s > stringCharLength existing then
        next := next.push s
        inserted := true
      next := next.push existing
    if !inserted then
      next := next.push s
    out := next
  out

/-- Load Qwen3 tokenizer from HF tokenizer.json (+ tokenizer_config.json for pad token). -/
def loadTokenizer (dir : String) : IO QwenTokenizer := do
  let cfgJson <- parseJsonFile s!"{dir}/tokenizer_config.json"
  let tokenizerJsonPath := s!"{dir}/tokenizer.json"
  let hasTokenizerJson ← System.FilePath.pathExists ⟨tokenizerJsonPath⟩

  let (tokJson?, vocabJson, mergesJson, unkTokenName?) ←
    if hasTokenizerJson then
      let tokJson <- parseJsonFile tokenizerJsonPath
      let modelJson <-
        match getObjVal? tokJson "model" with
        | some v => pure v
        | none => throw (IO.userError "tokenizer.json missing model")
      let vocabJson <-
        match getObjVal? modelJson "vocab" with
        | some v => pure v
        | none => throw (IO.userError "tokenizer.json missing model.vocab")
      let mergesJson <-
        match getObjVal? modelJson "merges" >>= getArr? with
        | some v => pure v
        | none => throw (IO.userError "tokenizer.json missing model.merges")
      let unkName :=
        match getObjVal? modelJson "unk_token" with
        | some (.str s) => some s
        | _ => none
      pure (some tokJson, vocabJson, mergesJson, unkName)
    else
      let vocabJson <- parseJsonFile s!"{dir}/vocab.json"
      let mergesText <- IO.FS.readFile s!"{dir}/merges.txt"
      let mut merges : Array Json := #[]
      for raw in mergesText.splitOn "\n" do
        let line := raw.trimAscii.toString
        if !line.isEmpty && !line.startsWith "#" then
          merges := merges.push (.str line)
      let unkName :=
        match getObjVal? cfgJson "unk_token" with
        | some (.str s) => some s
        | _ => none
      pure (none, vocabJson, merges, unkName)

  -- Build vocab maps
  let mut maxId : Nat := 0
  let mut tokenToId : Std.HashMap String TokenId := {}
  match vocabJson with
  | .obj kvs =>
    for (tok, idJson) in kvs do
      match getNat? idJson with
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

  -- Build char->id map for single-character tokens
  let mut charToId : Std.HashMap Char TokenId := {}
  for (tok, id) in tokenToId.toList do
    if stringCharLength tok == 1 then
      match tok.toList with
      | c :: _ => charToId := charToId.insert c id
      | _ => pure ()

  -- Special tokens
  let mut specialTokens : Std.HashMap String TokenId := {}
  let mut idToSpecial : Std.HashMap TokenId String := {}
  match tokJson? with
  | some tokJson =>
      match getObjVal? tokJson "added_tokens" >>= getArr? with
      | some arr =>
        for entry in arr do
          match entry with
          | .obj kvs =>
            let content := getObjVal? (.obj kvs) "content" >>= getStr?
            let id := getObjVal? (.obj kvs) "id" >>= getNat?
            match content, id with
            | some s, some n =>
              let tid := n.toUInt32
              specialTokens := specialTokens.insert s tid
              idToSpecial := idToSpecial.insert tid s
            | _, _ => pure ()
          | _ => pure ()
      | none => pure ()
  | none => pure ()

  -- HF fast-tokenizer fallback format (tokenizer_config.json::added_tokens_decoder)
  match getObjVal? cfgJson "added_tokens_decoder" with
  | some (.obj kvs) =>
      for (idStr, entry) in kvs do
        let content := getObjVal? entry "content" >>= getStr?
        let idNat := (idStr.toNat?) <|> (getObjVal? entry "id" >>= getNat?)
        match content, idNat with
        | some s, some n =>
            let tid := n.toUInt32
            specialTokens := specialTokens.insert s tid
            idToSpecial := idToSpecial.insert tid s
        | _, _ => pure ()
  | _ => pure ()

  let specialList := sortSpecials (List.toArray (specialTokens.toList.map Prod.fst))

  -- unk token (if present)
  let unkToken : Option TokenId :=
    match unkTokenName? with
    | some s => specialTokens.get? s <|> tokenToId.get? s
    | none => none

  -- pad token: prefer config "pad_token" / "pad_token_id" if present, else <|endoftext|>
  let padToken : TokenId :=
    match getObjVal? cfgJson "pad_token_id" >>= getNat? with
    | some id => id.toUInt32
    | none =>
      match specialTokens.get? "<|endoftext|>" with
      | some id => id
      | none =>
        match tokenToId.get? "<|endoftext|>" with
        | some id => id
        | none => 0

  -- Build merges
  let mut merges : Array MergeRule := #[]
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
          match getStr? arr[0]!, getStr? arr[1]! with
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
        let rule : MergeRule := { left := leftId, right := rightId, result := resultId }
        merges := merges.push rule
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
    merges
    mergeLookup
    mergePriority
    specialTokens
    idToSpecial
    specialList
    unkToken
    padToken
  }

/-- Chat template for Qwen3 (single user message, add_generation_prompt=true, enable_thinking=false). -/
def chatTemplate (prompt : String) : String :=
  "<|im_start|>user\n" ++ prompt ++ "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

/-- Qwen3-TTS assistant text wrapper used for synthesis input IDs. -/
def ttsAssistantText (text : String) : String :=
  "<|im_start|>assistant\n" ++ text ++ "<|im_end|>\n<|im_start|>assistant\n"

/-- Qwen3-TTS reference text wrapper used for voice-clone reference IDs. -/
def ttsRefText (text : String) : String :=
  "<|im_start|>assistant\n" ++ text ++ "<|im_end|>\n"

/-- Qwen3-TTS instruct wrapper used for style/speaker instruction IDs. -/
def ttsInstructText (instruct : String) : String :=
  "<|im_start|>user\n" ++ instruct ++ "<|im_end|>\n"

/-- Greedy match for any special token at position i. -/
private def matchSpecial (chars : Array Char) (i : Nat) (specials : Array String)
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

/-- Split text into segments, preserving special tokens as atomic segments. -/
private def splitWithSpecials (text : String) (specials : Array String)
    : Array (Bool × String) := Id.run do
  let chars := text.toList.toArray
  let mut out : Array (Bool × String) := #[]  -- (isSpecial, segment)
  let mut buf : Array Char := #[]
  let mut i : Nat := 0
  while i < chars.size do
    match matchSpecial chars i specials with
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

private def lowerAscii (c : Char) : Char :=
  if c >= 'A' && c <= 'Z' then
    Char.ofNat (c.toNat + 32)
  else
    c

private def isLetter (c : Char) : Bool := c.isAlpha
private def isDigit (c : Char) : Bool := c.isDigit
private def isAlnum (c : Char) : Bool := isLetter c || isDigit c
private def isWhitespace (c : Char) : Bool := c.isWhitespace
private def isCRLF (c : Char) : Bool := c == '\r' || c == '\n'

/-- Match contraction at position i. -/
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
        let a := lowerAscii chars[i + j]!
        let b := lowerAscii pat[j]!
        if a != b then ok := false
      if ok then
        return some len
  none

/-- Match optional prefix + letters pattern. -/
private def matchWord (chars : Array Char) (i : Nat) : Option Nat := Id.run do
  if i >= chars.size then return none
  let c := chars[i]!
  if isLetter c then
    let mut j := i
    while j < chars.size && isLetter chars[j]! do
      j := j + 1
    return some (j - i)
  else if !isCRLF c && !isAlnum c then
    if i + 1 < chars.size && isLetter chars[i + 1]! then
      let mut j := i + 1
      while j < chars.size && isLetter chars[j]! do
        j := j + 1
      return some (j - i)
    else
      return none
  else
    return none

/-- Match punctuation pattern: ` ?[^\s\p{L}\p{N}]+[\r\n]*` -/
private def matchPunct (chars : Array Char) (i : Nat) : Option Nat := Id.run do
  if i >= chars.size then return none
  let mut j := i
  if chars[j]! == ' ' then
    j := j + 1
  if j >= chars.size then
    return none
  else if isWhitespace chars[j]! || isAlnum chars[j]! then
    return none
  else
    while j < chars.size && !isWhitespace chars[j]! && !isAlnum chars[j]! do
      j := j + 1
    while j < chars.size && isCRLF chars[j]! do
      j := j + 1
    return some (j - i)

/-- Match whitespace* + newline+ pattern. -/
private def matchWhitespaceNewline (chars : Array Char) (i : Nat) : Option Nat := Id.run do
  if i >= chars.size then return none
  let mut j := i
  let mut lastNewline : Option Nat := none
  while j < chars.size && isWhitespace chars[j]! do
    if isCRLF chars[j]! then
      lastNewline := some j
    j := j + 1
  match lastNewline with
  | some idx => return some (idx + 1 - i)
  | none => return none

/-- Match trailing whitespace (\s+(?!\S)). -/
private def matchWhitespaceEnd (chars : Array Char) (i : Nat) : Option Nat := Id.run do
  if i >= chars.size then return none
  if !isWhitespace chars[i]! then
    return none
  let mut j := i
  while j < chars.size do
    if !isWhitespace chars[j]! then
      return none
    j := j + 1
  return some (chars.size - i)

/-- Match general whitespace run (\s+). -/
private def matchWhitespace (chars : Array Char) (i : Nat) : Option Nat := Id.run do
  if i >= chars.size then return none
  if !isWhitespace chars[i]! then
    return none
  let mut j := i
  while j < chars.size && isWhitespace chars[j]! do
    j := j + 1
  return some (j - i)

/-- Regex-like pretokenization for Qwen3. -/
private def pretokenize (text : String) : Array String := Id.run do
  let chars := text.toList.toArray
  let mut out : Array String := #[]
  let mut i : Nat := 0
  while i < chars.size do
    let len :=
      match matchContraction chars i with
      | some n => n
      | none =>
        match matchWord chars i with
        | some n => n
        | none =>
          if i < chars.size && isDigit chars[i]! then 1 else
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
    let segment := String.ofList ((chars.extract i (i + len)).toList)
    out := out.push segment
    i := i + len
  out

/-- Find the best merge (lowest priority). -/
private def findBestMerge (tok : QwenTokenizer) (tokens : Array TokenId)
    : Option (Nat × TokenId) := Id.run do
  if tokens.size < 2 then return none
  let mut bestIdx : Option Nat := none
  let mut bestRank : Nat := tok.merges.size
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

private def applyMerge (tokens : Array TokenId) (idx : Nat) (mergeResult : TokenId) : Array TokenId := Id.run do
  if idx + 1 >= tokens.size then return tokens
  let mut newTokens := Array.mkEmpty (tokens.size - 1)
  for i in [:idx] do
    newTokens := newTokens.push tokens[i]!
  newTokens := newTokens.push mergeResult
  for i in [idx + 2:tokens.size] do
    newTokens := newTokens.push tokens[i]!
  newTokens

/-- BPE encode a single pretokenized piece. -/
private def encodePiece (tok : QwenTokenizer) (piece : String) : Array TokenId := Id.run do
  if piece.isEmpty then return #[]
  let byteLevel := tokenizer.stringToByteLevel piece
  let mut tokens : Array TokenId := #[]
  for c in byteLevel.toList do
    match tok.charToId.get? c with
    | some id => tokens := tokens.push id
    | none =>
      -- fallback to string lookup
      let s := String.ofList [c]
      match tok.tokenToId.get? s, tok.unkToken with
      | some id, _ => tokens := tokens.push id
      | none, some unk => tokens := tokens.push unk
      | none, none => pure ()

  if tokens.isEmpty then return #[]

  let mut changed := true
  let mut iterations := 0
  let maxIter := 10000
  while changed && iterations < maxIter do
    iterations := iterations + 1
    match findBestMerge tok tokens with
    | some (idx, result) =>
      tokens := applyMerge tokens idx result
      changed := true
    | none =>
      changed := false
  tokens

/-- Encode text with special token handling. -/
private def encodeTextCore (tok : QwenTokenizer) (text : String) : Array TokenId := Id.run do
  let segments := splitWithSpecials text tok.specialList
  let mut out : Array TokenId := #[]
  for (isSpecial, seg) in segments do
    if isSpecial then
      match tok.specialTokens.get? seg with
      | some id => out := out.push id
      | none =>
        match tok.tokenToId.get? seg, tok.unkToken with
        | some id, _ => out := out.push id
        | none, some unk => out := out.push unk
        | none, none => pure ()
    else
      let pieces := pretokenize seg
      for piece in pieces do
          let ids := encodePiece tok piece
          out := out ++ ids
  out

/-- Public raw text encoding (no prompt template, no padding). -/
def encodeText (tok : QwenTokenizer) (text : String) : Array TokenId :=
  encodeTextCore tok text

/-- Decode byte-level text to UTF-8, preserving byte-level form on invalid streams. -/
private def byteLevelToStringLossy (s : String) : String :=
  let bytes := tokenizer.byteLevelToBytes s
  match String.fromUTF8? bytes with
  | some out => out
  | none => tokenizer.bytesToByteLevel bytes

/-- Decode token IDs to text (HF-style, keeping special tokens by default). -/
def decodeText (tok : QwenTokenizer) (ids : Array TokenId) : String := Id.run do
  let mut out := ""
  let mut byteLevelBuf := ""

  for id in ids do
    match tok.idToSpecial.get? id with
    | some special =>
      if !byteLevelBuf.isEmpty then
        out := out ++ byteLevelToStringLossy byteLevelBuf
        byteLevelBuf := ""
      out := out ++ special
    | none =>
      match tok.idToToken[id.toNat]? with
      | some tokStr => byteLevelBuf := byteLevelBuf ++ tokStr
      | none => pure ()

  if !byteLevelBuf.isEmpty then
    out := out ++ byteLevelToStringLossy byteLevelBuf
  out

/-- Decode one token ID to display text. -/
def decodeOne (tok : QwenTokenizer) (id : TokenId) : String :=
  decodeText tok #[id]

/-- Encode a prompt using the Qwen3 chat template and produce tokens + attention mask. -/
def encodePrompt (tok : QwenTokenizer) (prompt : String) (maxLen : Nat := 512)
    : Array TokenId × Array TokenId := Id.run do
  let text := chatTemplate prompt
  let tokens := encodeTextCore tok text
  let max := maxLen
  let trimmed := if tokens.size > max then tokens.extract 0 max else tokens
  let mut outTokens : Array TokenId := Array.mkEmpty max
  let mut mask : Array TokenId := Array.mkEmpty max
  for i in [:max] do
    if h : i < trimmed.size then
      outTokens := outTokens.push (trimmed[i]'h)
      mask := mask.push (1 : TokenId)
    else
      outTokens := outTokens.push tok.padToken
      mask := mask.push (0 : TokenId)
  (outTokens, mask)

/-- Convert TokenId array to Int64 array (for tensor creation). -/
def toInt64Array (tokens : Array TokenId) : Array Int64 :=
  tokens.map (fun t => (t.toUInt64).toInt64)

end tokenizer.qwen3
