/-
  Tyr/Tokenizer/Gemma4.lean

  Gemma 4 tokenizer loader + encoder/decoder.
  Matches the public Hugging Face `tokenizer.json` format used by Gemma 4:
  SentencePiece-style BPE with `▁` whitespace marker and byte fallback tokens.
-/
import Tyr.Tokenizer.Types
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace tokenizer.gemma4

open Lean

private def parseJsonFile (path : String) : IO Json := do
  let contents ← IO.FS.readFile path
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

structure GemmaTokenizer where
  vocabSize : Nat
  idToToken : Array String
  tokenToId : Std.HashMap String TokenId
  merges : Array MergeRule
  mergeLookup : Std.HashMap (TokenId × TokenId) TokenId
  mergePriority : Std.HashMap (TokenId × TokenId) Nat
  specialTokens : Std.HashMap String TokenId
  idToSpecial : Std.HashMap TokenId String
  specialList : Array String
  unkToken : Option TokenId
  padToken : TokenId
  bosToken : Option TokenId
  eosToken : Option TokenId
  deriving Inhabited

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

private def byteFallbackToken (b : UInt8) : String :=
  let digits := "0123456789ABCDEF".toList.toArray
  let hi := digits[(b.toNat / 16)]!
  let lo := digits[(b.toNat % 16)]!
  String.ofList ['<', '0', 'x', hi, lo, '>']

def loadTokenizer (dir : String) : IO GemmaTokenizer := do
  let cfgJson ← parseJsonFile s!"{dir}/tokenizer_config.json"
  let tokJson ← parseJsonFile s!"{dir}/tokenizer.json"
  let modelJson ←
    match getObjVal? tokJson "model" with
    | some v => pure v
    | none => throw (IO.userError "tokenizer.json missing model")
  let vocabJson ←
    match getObjVal? modelJson "vocab" with
    | some v => pure v
    | none => throw (IO.userError "tokenizer.json missing model.vocab")
  let mergesJson ←
    match getObjVal? modelJson "merges" >>= getArr? with
    | some v => pure v
    | none => throw (IO.userError "tokenizer.json missing model.merges")

  let unkTokenName :=
    match getObjVal? modelJson "unk_token" with
    | some (.str s) => some s
    | _ => none

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
    if id.toNat < idToToken.size then
      idToToken := idToToken.set! id.toNat tok

  let mut specialTokens : Std.HashMap String TokenId := {}
  let mut idToSpecial : Std.HashMap TokenId String := {}

  match getObjVal? tokJson "added_tokens" >>= getArr? with
  | some arr =>
    for entry in arr do
      match entry with
      | .obj _ =>
        let content := getObjVal? entry "content" >>= getStr?
        let id := getObjVal? entry "id" >>= getNat?
        match content, id with
        | some s, some n =>
          let tid := n.toUInt32
          specialTokens := specialTokens.insert s tid
          idToSpecial := idToSpecial.insert tid s
        | _, _ => pure ()
      | _ => pure ()
  | none => pure ()

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

  let unkToken : Option TokenId :=
    match unkTokenName with
    | some s => specialTokens.get? s <|> tokenToId.get? s
    | none => none

  let padToken : TokenId :=
    match getObjVal? cfgJson "pad_token_id" >>= getNat? with
    | some id => id.toUInt32
    | none =>
      match specialTokens.get? "<pad>" with
      | some id => id
      | none => 0

  let bosToken : Option TokenId :=
    match getObjVal? cfgJson "bos_token_id" >>= getNat? with
    | some id => some id.toUInt32
    | none => specialTokens.get? "<bos>"

  let eosToken : Option TokenId :=
    match getObjVal? cfgJson "eos_token_id" >>= getNat? with
    | some id => some id.toUInt32
    | none => specialTokens.get? "<eos>"

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
    merges
    mergeLookup
    mergePriority
    specialTokens
    idToSpecial
    specialList
    unkToken
    padToken
    bosToken
    eosToken
  }

def chatTemplate (prompt : String) : String :=
  "<bos><|turn>user\n" ++ prompt ++ "<turn|>\n<|turn>model\n"

def chatTemplateThinking (prompt : String) : String :=
  "<bos><|turn>system\n<|think|><turn|>\n<|turn>user\n" ++ prompt ++ "<turn|>\n<|turn>model\n"

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

private def splitWithSpecials (text : String) (specials : Array String)
    : Array (Bool × String) := Id.run do
  let chars := text.toList.toArray
  let mut out : Array (Bool × String) := #[]
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

private def findBestMerge (tok : GemmaTokenizer) (tokens : Array TokenId)
    : Option (Nat × TokenId) := Id.run do
  if tokens.size < 2 then
    return none
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
  | some i =>
    let left := tokens[i]!
    let right := tokens[i + 1]!
    some (i, tok.mergeLookup.getD (left, right) left)
  | none => none

private def applyMerge (tokens : Array TokenId) (idx : Nat) (mergeResult : TokenId) : Array TokenId := Id.run do
  if idx + 1 >= tokens.size then
    return tokens
  let mut out := Array.mkEmpty (tokens.size - 1)
  for i in [:idx] do
    out := out.push tokens[i]!
  out := out.push mergeResult
  for i in [idx + 2:tokens.size] do
    out := out.push tokens[i]!
  out

private def encodeCharTokens (tok : GemmaTokenizer) (c : Char) : Array TokenId := Id.run do
  let s := String.ofList [c]
  match tok.tokenToId.get? s with
  | some id => #[id]
  | none =>
    let mut out : Array TokenId := #[]
    for b in s.toUTF8.toList do
      match tok.tokenToId.get? (byteFallbackToken b) with
      | some id => out := out.push id
      | none =>
        match tok.unkToken with
        | some unk => out := out.push unk
        | none => pure ()
    out

private def encodePiece (tok : GemmaTokenizer) (piece : String) : Array TokenId := Id.run do
  if piece.isEmpty then
    return #[]
  let normalized := piece.replace " " "▁"
  let mut tokens : Array TokenId := #[]
  for c in normalized.toList do
    tokens := tokens ++ encodeCharTokens tok c
  if tokens.isEmpty then
    return #[]
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

private def encodeTextCore (tok : GemmaTokenizer) (text : String) : Array TokenId := Id.run do
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
      out := out ++ encodePiece tok seg
  out

def encodeText (tok : GemmaTokenizer) (text : String) : Array TokenId :=
  encodeTextCore tok text

private def hexValue? (c : Char) : Option UInt8 :=
  if c >= '0' && c <= '9' then
    some ((c.toNat - '0'.toNat).toUInt8)
  else if c >= 'A' && c <= 'F' then
    some ((10 + c.toNat - 'A'.toNat).toUInt8)
  else if c >= 'a' && c <= 'f' then
    some ((10 + c.toNat - 'a'.toNat).toUInt8)
  else
    none

private def parseByteFallbackToken? (s : String) : Option UInt8 :=
  let cs := s.toList.toArray
  if cs.size == 6 && cs[0]! == '<' && cs[1]! == '0' && cs[2]! == 'x' && cs[5]! == '>' then
    match hexValue? cs[3]!, hexValue? cs[4]! with
    | some hi, some lo => some (hi * 16 + lo)
    | _, _ => none
  else
    none

private def flushDecodedBytes (bytes : ByteArray) : String :=
  match String.fromUTF8? bytes with
  | some out => out
  | none => String.fromUTF8! bytes

def decodeText (tok : GemmaTokenizer) (ids : Array TokenId) : String := Id.run do
  let mut out := ""
  let mut bytes := ByteArray.empty
  for id in ids do
    match tok.idToSpecial.get? id with
    | some special =>
      if bytes.size > 0 then
        out := out ++ flushDecodedBytes bytes
        bytes := ByteArray.empty
      out := out ++ special
    | none =>
      match tok.idToToken[id.toNat]? with
      | some tokStr =>
        match parseByteFallbackToken? tokStr with
        | some b => bytes := bytes.push b
        | none => bytes := bytes ++ (tokStr.replace "▁" " ").toUTF8
      | none => pure ()
  if bytes.size > 0 then
    out := out ++ flushDecodedBytes bytes
  out

def decodeOne (tok : GemmaTokenizer) (id : TokenId) : String :=
  decodeText tok #[id]

end tokenizer.gemma4
