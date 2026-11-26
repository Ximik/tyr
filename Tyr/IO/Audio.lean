/-
  Tyr/IO/Audio.lean

  PCM16 WAV decoding to a mono waveform in `[-1, 1]` (channels averaged).
-/
import Tyr.Core.Util

namespace torch.audio

private def hasTag4 (bytes : ByteArray) (offset : Nat) (tag : String) : Bool :=
  offset + 4 <= bytes.size && bytes.extract offset (offset + 4) == tag.toUTF8

private def i16ToFloat (u : UInt16) : Float :=
  let s : Int64 :=
    if u.toNat < 32768 then
      Int64.ofNat u.toNat
    else
      (Int64.ofNat u.toNat) - (Int64.ofNat 65536)
  s.toFloat / 32768.0

/-- Parse PCM16 WAV data chunk(s) and return `(sampleRate, monoWaveform)` in `[-1, 1]`. -/
def loadMonoPcm16WavFromBytes
    (bytes : ByteArray)
    (sourceLabel : String := "<bytes>")
    : IO (Nat × Array Float) := do
  if bytes.size < 12 then
    throw <| IO.userError s!"Invalid WAV (too small): {sourceLabel}"
  if !(hasTag4 bytes 0 "RIFF" && hasTag4 bytes 8 "WAVE") then
    throw <| IO.userError s!"Invalid WAV RIFF/WAVE header: {sourceLabel}"

  let mut fmtOffset : Option Nat := none
  let mut fmtSize : Nat := 0
  let mut dataOffset : Option Nat := none
  let mut dataSize : Nat := 0

  let mut off : Nat := 12
  while off + 8 <= bytes.size do
    let csizeU32 ←
      match bytes.readUInt32LE? (off + 4) with
      | some x => pure x
      | none => throw <| IO.userError s!"Invalid WAV chunk header in {sourceLabel}"
    let csize := csizeU32.toNat
    let body := off + 8
    if body + csize > bytes.size then
      throw <| IO.userError s!"Truncated WAV chunk payload in {sourceLabel}"

    if hasTag4 bytes off "fmt " then
      fmtOffset := some body
      fmtSize := csize
    if hasTag4 bytes off "data" then
      dataOffset := some body
      dataSize := csize

    let next := body + csize
    off := if next % 2 == 1 then next + 1 else next

  let fmtOff ←
    match fmtOffset with
    | some x => pure x
    | none => throw <| IO.userError s!"WAV missing fmt chunk: {sourceLabel}"
  let dataOff ←
    match dataOffset with
    | some x => pure x
    | none => throw <| IO.userError s!"WAV missing data chunk: {sourceLabel}"

  if fmtSize < 16 then
    throw <| IO.userError s!"Unsupported fmt chunk size ({fmtSize}) in {sourceLabel}"

  let readFmtU16 (rel : Nat) (name : String) : IO UInt16 := do
    match bytes.readUInt16LE? (fmtOff + rel) with
    | some x => pure x
    | none => throw <| IO.userError s!"Invalid fmt field `{name}` in {sourceLabel}"

  let readFmtU32 (rel : Nat) (name : String) : IO UInt32 := do
    match bytes.readUInt32LE? (fmtOff + rel) with
    | some x => pure x
    | none => throw <| IO.userError s!"Invalid fmt field `{name}` in {sourceLabel}"

  let audioFormat ← readFmtU16 0 "audio_format"
  let channels ← readFmtU16 2 "num_channels"
  let sampleRate ← readFmtU32 4 "sample_rate"
  let bitsPerSample ← readFmtU16 14 "bits_per_sample"

  if audioFormat != 1 then
    throw <| IO.userError s!"Only PCM WAV is supported, got format={audioFormat} in {sourceLabel}"
  if bitsPerSample != 16 then
    throw <| IO.userError s!"Only 16-bit WAV is supported, got bits_per_sample={bitsPerSample} in {sourceLabel}"
  if channels == 0 then
    throw <| IO.userError s!"Invalid WAV channels=0 in {sourceLabel}"
  if dataSize % 2 != 0 then
    throw <| IO.userError s!"Invalid PCM16 data size (odd number of bytes) in {sourceLabel}"

  let chNat := channels.toNat
  let sampleCountTotal := dataSize / 2
  if sampleCountTotal < chNat then
    throw <| IO.userError s!"WAV data is too short for channel count in {sourceLabel}"

  let frameCount := sampleCountTotal / chNat
  let mut mono : Array Float := Array.mkEmpty frameCount
  for frame in [:frameCount] do
    let mut acc : Float := 0.0
    for ch in [:chNat] do
      let sampleIdx := frame * chNat + ch
      let byteOff := dataOff + sampleIdx * 2
      let u ←
        match bytes.readUInt16LE? byteOff with
        | some x => pure x
        | none => throw <| IO.userError s!"Invalid PCM sample in {sourceLabel}"
      acc := acc + i16ToFloat u
    mono := mono.push (acc / channels.toFloat)

  pure (sampleRate.toNat, mono)

/-- Parse PCM16 WAV data chunk(s) and return `(sampleRate, monoWaveform)` in `[-1, 1]`. -/
def loadMonoPcm16Wav (path : String) : IO (Nat × Array Float) := do
  let bytes ← IO.FS.readBinFile path
  loadMonoPcm16WavFromBytes bytes path

end torch.audio
