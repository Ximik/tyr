/-
  Tyr/Lib/Tokenizer/ByteLevelBPE.lean

  A byte-level BPE tokenizer in the HuggingFace `tokenizer.json` format: special
  tokens are matched first, other text is optionally NFC-normalized, split by
  `wordSplit.splitWords`, and each piece is merged by rank.
-/
import Tyr.IO.ConfigJson
import Tyr.Lib.Tokenizer.Types
import Tyr.Lib.Tokenizer.ByteLevel
import Tyr.Lib.Tokenizer.WordSplit
import Tyr.Lib.Text.Nfc
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace tokenizer

open Lean

structure ByteLevelBPETokenizer where
  vocabSize : Nat
  idToToken : Array String
  tokenToId : Std.HashMap String UInt32
  charToId : Std.HashMap Char UInt32
  merges : Array MergeRule
  mergeLookup : Std.HashMap (UInt32 × UInt32) UInt32
  mergePriority : Std.HashMap (UInt32 × UInt32) Nat
  specialTokens : Std.HashMap String UInt32
  idToSpecial : Std.HashMap UInt32 String
  specialList : Array String
  unkToken : Option UInt32
  padToken : UInt32
  /-- Whether text is put in Unicode Normalization Form C before splitting. -/
  nfc : Bool

namespace ByteLevelBPETokenizer

/-- Load from `dir`: `tokenizer.json`, or `vocab.json` and `merges.txt`, plus
    `tokenizer_config.json`. -/
def load (dir : String) : IO ByteLevelBPETokenizer := do
  let cfgJson <- torch.ConfigJson.parseFile s!"{dir}/tokenizer_config.json"
  let tokenizerJsonPath := s!"{dir}/tokenizer.json"
  let hasTokenizerJson ← System.FilePath.pathExists ⟨tokenizerJsonPath⟩

  let (tokJson?, vocabJson, mergesJson, unkTokenName?) ←
    if hasTokenizerJson then
      let tokJson <- torch.ConfigJson.parseFile tokenizerJsonPath
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
      let unkName :=
        match (modelJson.getObjVal? "unk_token").toOption with
        | some (.str s) => some s
        | _ => none
      pure (some tokJson, vocabJson, mergesJson, unkName)
    else
      let vocabJson <- torch.ConfigJson.parseFile s!"{dir}/vocab.json"
      let mergesText <- IO.FS.readFile s!"{dir}/merges.txt"
      let mut merges : Array Json := #[]
      for raw in mergesText.splitOn "\n" do
        let line := raw.trimAscii.toString
        if !line.isEmpty && !line.startsWith "#" then
          merges := merges.push (.str line)
      let unkName :=
        match (cfgJson.getObjVal? "unk_token").toOption with
        | some (.str s) => some s
        | _ => none
      pure (none, vocabJson, merges, unkName)

  -- Build vocab maps
  let mut maxId : Nat := 0
  let mut tokenToId : Std.HashMap String UInt32 := {}
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

  -- Build char->id map for single-character tokens
  let mut charToId : Std.HashMap Char UInt32 := {}
  for (tok, id) in tokenToId.toList do
    if tok.length == 1 then
      match tok.toList with
      | c :: _ => charToId := charToId.insert c id
      | _ => pure ()

  -- Special tokens
  let mut specialTokens : Std.HashMap String UInt32 := {}
  let mut idToSpecial : Std.HashMap UInt32 String := {}
  match tokJson? with
  | some tokJson =>
      match (tokJson.getObjVal? "added_tokens" >>= (·.getArr?)).toOption with
      | some arr =>
        for entry in arr do
          match entry with
          | .obj kvs =>
            let content := ((Json.obj kvs).getObjVal? "content" >>= (·.getStr?)).toOption
            let id := ((Json.obj kvs).getObjVal? "id" >>= (·.getNat?)).toOption
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
  match (cfgJson.getObjVal? "added_tokens_decoder").toOption with
  | some (.obj kvs) =>
      for (idStr, entry) in kvs do
        let content := (entry.getObjVal? "content" >>= (·.getStr?)).toOption
        let idNat := (idStr.toNat?) <|> ((entry.getObjVal? "id" >>= (·.getNat?)).toOption)
        match content, idNat with
        | some s, some n =>
            let tid := n.toUInt32
            specialTokens := specialTokens.insert s tid
            idToSpecial := idToSpecial.insert tid s
        | _, _ => pure ()
  | _ => pure ()

  let specialList := (specialTokens.toList.map Prod.fst).toArray.insertionSort (·.length > ·.length)

  -- unk token (if present)
  let unkToken : Option UInt32 :=
    match unkTokenName? with
    | some s => specialTokens.get? s <|> tokenToId.get? s
    | none => none

  -- pad token: prefer config "pad_token" / "pad_token_id" if present, else <|endoftext|>
  let padToken : UInt32 :=
    match (cfgJson.getObjVal? "pad_token_id" >>= (·.getNat?)).toOption with
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
  let mut mergeLookup : Std.HashMap (UInt32 × UInt32) UInt32 := {}
  let mut mergePriority : Std.HashMap (UInt32 × UInt32) Nat := {}
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
        let rule : MergeRule := { left := leftId, right := rightId, result := resultId }
        merges := merges.push rule
        mergeLookup := mergeLookup.insert (leftId, rightId) resultId
        mergePriority := mergePriority.insert (leftId, rightId) idx
        idx := idx + 1
      | none =>
        throw (IO.userError s!"Missing merge result token: {resultStr}")
    | _, _ =>
      throw (IO.userError s!"Missing merge pair tokens: {leftStr} {rightStr}")

  -- Without tokenizer.json, transformers' Qwen2 tokenizer normalizes to NFC.
  let nfc ← match tokJson? with
    | none => pure true
    | some tokJson =>
      match (tokJson.getObjVal? "normalizer").toOption with
      | none | some .null => pure false
      | some normalizer =>
        if (normalizer.getObjVal? "type").toOption == some (.str "NFC") then pure true
        else throw (IO.userError s!"unsupported tokenizer.json normalizer {normalizer.compress}; tyr implements NFC")
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
    nfc
  }

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

/-- Find the best merge (lowest priority). -/
private def findBestMerge (tok : ByteLevelBPETokenizer) (tokens : Array UInt32)
    : Option (Nat × UInt32) := Id.run do
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

private def applyMerge (tokens : Array UInt32) (idx : Nat) (mergeResult : UInt32) : Array UInt32 := Id.run do
  if idx + 1 >= tokens.size then return tokens
  let mut newTokens := Array.mkEmpty (tokens.size - 1)
  for i in [:idx] do
    newTokens := newTokens.push tokens[i]!
  newTokens := newTokens.push mergeResult
  for i in [idx + 2:tokens.size] do
    newTokens := newTokens.push tokens[i]!
  newTokens

/-- BPE encode a single pretokenized piece. -/
private def encodePiece (tok : ByteLevelBPETokenizer) (piece : String) : Array UInt32 := Id.run do
  if piece.isEmpty then return #[]
  let byteLevel := tokenizer.stringToByteLevel piece
  let mut tokens : Array UInt32 := #[]
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

/-- Raw text encoding: no template, no padding. -/
def encodeText (tok : ByteLevelBPETokenizer) (text : String) : Array UInt32 := Id.run do
  let segments := splitWithSpecials text tok.specialList
  let mut out : Array UInt32 := #[]
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
      let seg := if tok.nfc then nfc.normalize seg else seg
      let pieces := wordSplit.splitWords seg.toList.toArray
      for piece in pieces do
          let ids := encodePiece tok piece
          out := out ++ ids
  out

/-- Decode byte-level text to UTF-8, preserving byte-level form on invalid streams. -/
private def byteLevelToStringLossy (s : String) : String :=
  let bytes := tokenizer.byteLevelToBytes s
  match String.fromUTF8? bytes with
  | some out => out
  | none => tokenizer.bytesToByteLevel bytes

/-- Decode token IDs to text (HF-style, keeping special tokens by default). -/
def decodeText (tok : ByteLevelBPETokenizer) (ids : Array UInt32) : String := Id.run do
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
def decodeOne (tok : ByteLevelBPETokenizer) (id : UInt32) : String :=
  decodeText tok #[id]

end ByteLevelBPETokenizer

/-- Convert token id array to Int64 array (for tensor creation). -/
def toInt64Array (tokens : Array UInt32) : Array Int64 :=
  tokens.map (fun t => (t.toUInt64).toInt64)

end tokenizer
