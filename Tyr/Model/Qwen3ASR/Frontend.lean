/-
  Tyr/Model/Qwen3ASR/Frontend.lean

  Qwen3-ASR audio input: a WAV path, an HTTP(S) URL, base64 WAV bytes or a
  waveform, normalized to a 16 kHz mono waveform.
-/
import Tyr.IO.Audio
import Tyr.Lib.Audio.Waveform

namespace torch.qwen3asr

open torch.audio

/-- Unified ASR audio input surface, matching upstream-style entry forms. -/
inductive ASRAudioInput where
  /-- Local filesystem audio path (WAV PCM16 currently supported by Lean frontend). -/
  | wavPath (path : String)
  /-- HTTP/HTTPS URL to audio content (downloaded then decoded as WAV PCM16). -/
  | url (value : String)
  /-- Base64-encoded audio payload (raw base64 or data URL), decoded as WAV PCM16. -/
  | base64 (value : String)
  /-- In-memory mono waveform with explicit source sampling rate. -/
  | waveform (samples : Array Float) (sampleRate : Nat)
  deriving Repr

private def isHttpUrl (s : String) : Bool :=
  s.startsWith "http://" || s.startsWith "https://"

private def isBase64AlphabetChar (c : Char) : Bool :=
  c.isAlphanum || c == '+' || c == '/' || c == '=' || c == '-' || c == '_'

private def isProbablyBase64 (s : String) : Bool :=
  if s.startsWith "data:audio" then
    true
  else
    let t := s.trimAscii.toString
    let filtered := t.toList.filter (fun c => !(c.isWhitespace))
    if filtered.length < 64 then
      false
    else
      let hasOnlyBase64Chars := filtered.all isBase64AlphabetChar
      let looksLikePath :=
        t.startsWith "/" ||
        t.startsWith "./" ||
        t.startsWith "../" ||
        t.startsWith "~/" ||
        t.startsWith "~\\" ||
        t.contains '\\' ||
        t.contains '.'
      hasOnlyBase64Chars && !looksLikePath

namespace ASRAudioInput

def ofStringAuto (s : String) : ASRAudioInput :=
  if isHttpUrl s then
    .url s
  else if isProbablyBase64 s then
    .base64 s
  else
    .wavPath s

end ASRAudioInput

private def base64Val? (c : Char) : Option UInt8 :=
  let n := c.toNat
  if 'A'.toNat <= n && n <= 'Z'.toNat then
    some (UInt8.ofNat (n - 'A'.toNat))
  else if 'a'.toNat <= n && n <= 'z'.toNat then
    some (UInt8.ofNat (26 + (n - 'a'.toNat)))
  else if '0'.toNat <= n && n <= '9'.toNat then
    some (UInt8.ofNat (52 + (n - '0'.toNat)))
  else if c == '+' || c == '-' then
    some 62
  else if c == '/' || c == '_' then
    some 63
  else
    none

private def stripDataUrlPrefix (s : String) : String :=
  if s.startsWith "data:" then
    match s.splitOn "," with
    | [] => s
    | _ :: rest => String.intercalate "," rest
  else
    s

private def decodeBase64Bytes (input : String) : Except String ByteArray := do
  let s0 := stripDataUrlPrefix input
  let filtered := s0.toList.filter (fun c => !(c.isWhitespace))
  if filtered.isEmpty then
    throw "base64 payload is empty"
  let mut chars := filtered.toArray
  while chars.size % 4 != 0 do
    chars := chars.push '='

  let mut out := ByteArray.empty
  let mut i : Nat := 0
  while i < chars.size do
    if i + 3 >= chars.size then
      throw "invalid base64: truncated quartet"
    let c0 := chars[i]!
    let c1 := chars[i + 1]!
    let c2 := chars[i + 2]!
    let c3 := chars[i + 3]!
    let v0 ←
      match base64Val? c0 with
      | some v => pure v
      | none => throw s!"invalid base64 character: {c0}"
    let v1 ←
      match base64Val? c1 with
      | some v => pure v
      | none => throw s!"invalid base64 character: {c1}"

    let b0 : UInt8 := ((v0 <<< 2) ||| (v1 >>> 4))
    out := out.push b0

    if c2 != '=' then
      let v2 ←
        match base64Val? c2 with
        | some v => pure v
        | none => throw s!"invalid base64 character: {c2}"
      let b1 : UInt8 := (((v1 &&& 0x0f) <<< 4) ||| (v2 >>> 2))
      out := out.push b1
      if c3 != '=' then
        let v3 ←
          match base64Val? c3 with
          | some v => pure v
          | none => throw s!"invalid base64 character: {c3}"
        let b2 : UInt8 := (((v2 &&& 0x03) <<< 6) ||| v3)
        out := out.push b2
    else if c3 != '=' then
      throw "invalid base64: third char padding requires fourth char padding"

    i := i + 4
  pure out

private def ensureParentDir (path : String) : IO Unit := do
  match System.FilePath.parent ⟨path⟩ with
  | some parent =>
      if parent.toString != "" && parent.toString != "." then
        IO.FS.createDirAll parent
  | none => pure ()

private def downloadToFile (url : String) (dest : String) : IO Unit := do
  ensureParentDir dest
  let tmp := s!"{dest}.tmp"
  let out ← IO.Process.output {
    cmd := "curl"
    args := #["-fL", "--retry", "3", "--retry-delay", "1", "-o", tmp, url]
  }
  if out.exitCode == 0 then
    IO.FS.rename ⟨tmp⟩ ⟨dest⟩
  else
    if (← System.FilePath.pathExists ⟨tmp⟩) then
      IO.FS.removeFile ⟨tmp⟩
    throw <| IO.userError s!"Failed to download audio URL (exit={out.exitCode}): {url}\n{out.stderr}"

private def mkTempWavPath (tag : String) : IO String := do
  let nonce ← IO.rand 0 2147483647
  pure s!"/tmp/tyr_qwen3asr_{tag}_{nonce}.wav"

/-- Normalize one unified audio input into mono 16k float waveform in `[-1, 1]`.
    URL/base64 paths currently decode via WAV PCM16 loader after materialization. -/
def normalizeAudioInputTo16k (input : ASRAudioInput) : IO (Array Float) := do
  match input with
  | .waveform samples srcRate =>
      let mono := normalizeFloatRange samples
      if srcRate == 16000 || srcRate == 0 then
        pure mono
      else
        pure (normalizeFloatRange (← data.resampleSoxrHQ mono srcRate 16000))
  | .wavPath path =>
      normalizeAudioTo16kFromWav path
  | .url value => do
      let tmp ← mkTempWavPath "url"
      try
        downloadToFile value tmp
        normalizeAudioTo16kFromWav tmp
      finally
        if (← System.FilePath.pathExists ⟨tmp⟩) then
          IO.FS.removeFile ⟨tmp⟩
  | .base64 value => do
      let bytes ←
        match decodeBase64Bytes value with
        | .ok b => pure b
        | .error e => throw <| IO.userError s!"Invalid base64 audio payload: {e}"
      let (sr, wav) ← loadMonoPcm16WavFromBytes bytes "base64-audio"
      if sr == 16000 || sr == 0 then
        pure (normalizeFloatRange wav)
      else
        pure (normalizeFloatRange (← data.resampleSoxrHQ wav sr 16000))

end torch.qwen3asr
