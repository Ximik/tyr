/-
  Tyr/Model/Qwen3ASR/Frontend.lean

  Lean-native Qwen3-ASR audio frontend:
  - PCM16 WAV loading
  - resample to target sampling rate
  - Whisper-style log-mel extraction
  - feature attention-mask generation
-/
import Tyr.Torch
import Tyr.Typed.Neural
import Tyr.Model.Qwen3ASR.PreprocessorConfig

namespace torch.qwen3asr

variable {device : Device}
structure WhisperFrontendOutput (melBins frames : Nat) where
  inputFeatures : Tensor { device := device, shape := #[1, melBins, frames], dtype := .Float32 }
  featureAttentionMask : Tensor { device := device, shape := #[1, frames], dtype := .Int64 }

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
  deriving Repr, Inhabited

private def readU16LE (bytes : ByteArray) (offset : Nat) : Option UInt16 :=
  if offset + 2 > bytes.size then
    none
  else
    let b0 := bytes[offset]!
    let b1 := bytes[offset + 1]!
    some (b0.toUInt16 ||| (b1.toUInt16 <<< 8))

private def readU32LE (bytes : ByteArray) (offset : Nat) : Option UInt32 :=
  if offset + 4 > bytes.size then
    none
  else
    let b0 := bytes[offset]!
    let b1 := bytes[offset + 1]!
    let b2 := bytes[offset + 2]!
    let b3 := bytes[offset + 3]!
    some (b0.toUInt32 ||| (b1.toUInt32 <<< 8) ||| (b2.toUInt32 <<< 16) ||| (b3.toUInt32 <<< 24))

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
private def loadMonoPcm16WavFromBytes
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
      match readU32LE bytes (off + 4) with
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
    match readU16LE bytes (fmtOff + rel) with
    | some x => pure x
    | none => throw <| IO.userError s!"Invalid fmt field `{name}` in {sourceLabel}"

  let readFmtU32 (rel : Nat) (name : String) : IO UInt32 := do
    match readU32LE bytes (fmtOff + rel) with
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
        match readU16LE bytes byteOff with
        | some x => pure x
        | none => throw <| IO.userError s!"Invalid PCM sample in {sourceLabel}"
      acc := acc + i16ToFloat u
    mono := mono.push (acc / channels.toFloat)

  pure (sampleRate.toNat, mono)

/-- Parse PCM16 WAV data chunk(s) and return `(sampleRate, monoWaveform)` in `[-1, 1]`. -/
def loadMonoPcm16Wav (path : String) : IO (Nat × Array Float) := do
  let bytes ← IO.FS.readBinFile path
  loadMonoPcm16WavFromBytes bytes path

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

private def expandHome (path : String) : IO String := do
  if path == "~" then
    return (← IO.getEnv "HOME").getD path
  else if path.startsWith "~/" then
    return s!"{(← IO.getEnv "HOME").getD ""}/{path.drop 2}"
  else
    return path

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

private def clamp01 (x : Float) : Float :=
  if x > 1.0 then 1.0 else if x < (-1.0) then -1.0 else x

private def normalizeFloatRange (audio : Array Float) : Array Float :=
  if audio.isEmpty then
    audio
  else
    Id.run do
      let mut peak : Float := 0.0
      for x in audio do
        let a := Float.abs x
        if a > peak then
          peak := a
      let scale : Float := if peak > 1.0 then peak else 1.0
      let mut out : Array Float := Array.mkEmpty audio.size
      for x in audio do
        out := out.push (clamp01 (x / scale))
      out

def resampleLinear (audio : Array Float) (srcRate targetRate : Nat) : Array Float :=
  if audio.isEmpty || srcRate == targetRate || srcRate == 0 || targetRate == 0 then
    audio
  else
    let inSize := audio.size
    let outSize := (inSize * targetRate) / srcRate
    if outSize == 0 then
      #[]
    else
      Id.run do
        let mut out : Array Float := Array.mkEmpty outSize
        let srcNat := srcRate
        let dstNat := targetRate
        for i in [:outSize] do
          let srcNum := i * srcNat
          let idx0 := srcNum / dstNat
          let rem := srcNum % dstNat
          let i0 := if idx0 < inSize then idx0 else inSize - 1
          let i1 := if i0 + 1 < inSize then i0 + 1 else i0
          let w := rem.toFloat / targetRate.toFloat
          let v0 := audio[i0]!
          let v1 := audio[i1]!
          out := out.push ((1.0 - w) * v0 + w * v1)
        out

private def fmax (a b : Float) : Float := if a > b then a else b
private def fmin (a b : Float) : Float := if a < b then a else b

private def log10f (x : Float) : Float :=
  Float.log x / Float.log 10.0

private def hertzToMelSlaney (freq : Float) : Float :=
  let minLogHertz := 1000.0
  let minLogMel := 15.0
  let logstep := 27.0 / Float.log 6.4
  if freq >= minLogHertz then
    minLogMel + Float.log (freq / minLogHertz) * logstep
  else
    3.0 * freq / 200.0

private def melToHertzSlaney (mel : Float) : Float :=
  let minLogHertz := 1000.0
  let minLogMel := 15.0
  let logstep := Float.log 6.4 / 27.0
  if mel >= minLogMel then
    minLogHertz * Float.exp (logstep * (mel - minLogMel))
  else
    200.0 * mel / 3.0

private def linspace (start stop : Float) (n : Nat) : Array Float :=
  if n == 0 then
    #[]
  else if n == 1 then
    #[start]
  else
    Id.run do
      let mut out : Array Float := Array.mkEmpty n
      let step := (stop - start) / ((n - 1).toFloat)
      for i in [:n] do
        out := out.push (start + step * i.toFloat)
      out

private def buildSlaneyMelFilterBankFlat
    (numFrequencyBins numMelFilters samplingRate : Nat)
    : Array Float :=
  if numFrequencyBins < 2 || numMelFilters == 0 then
    #[]
  else
    let minHz : Float := 0.0
    let maxHz : Float := (samplingRate / 2).toFloat
    let melMin := hertzToMelSlaney minHz
    let melMax := hertzToMelSlaney maxHz
    let melFreqs := linspace melMin melMax (numMelFilters + 2)
    let filterFreqs := melFreqs.map melToHertzSlaney
    let fftFreqs := linspace 0.0 maxHz numFrequencyBins
    Id.run do
      let mut out : Array Float := Array.mkEmpty (numFrequencyBins * numMelFilters)
      for i in [:numFrequencyBins] do
        let fft := fftFreqs.getD i 0.0
        for j in [:numMelFilters] do
          let f0 := filterFreqs.getD j 0.0
          let f1 := filterFreqs.getD (j + 1) 0.0
          let f2 := filterFreqs.getD (j + 2) 0.0
          let downDen := f1 - f0
          let upDen := f2 - f1
          let down :=
            if downDen == 0.0 then 0.0 else (fft - f0) / downDen
          let up :=
            if upDen == 0.0 then 0.0 else (f2 - fft) / upDen
          let tri := fmax 0.0 (fmin down up)
          let enormDen := f2 - f0
          let enorm := if enormDen == 0.0 then 0.0 else 2.0 / enormDen
          out := out.push (tri * enorm)
      out

private def takePadRight (xs : Array Float) (target : Nat) (padValue : Float) : Array Float × Nat :=
  let taken := if xs.size > target then xs[:target].toArray else xs
  let validLen := Nat.min xs.size target
  if taken.size == target then
    (taken, validLen)
  else
    Id.run do
      let mut out := taken
      let padN := target - taken.size
      for _ in [:padN] do
        out := out.push padValue
      (out, validLen)

private def zeroMeanUnitVar (audio : Array Float) (validLen : Nat) (paddingValue : Float) : Array Float :=
  if audio.isEmpty || validLen == 0 then
    audio
  else
    Id.run do
      let useLen := Nat.min validLen audio.size
      let mut sum : Float := 0.0
      for i in [:useLen] do
        sum := sum + audio[i]!
      let mean := sum / useLen.toFloat

      let mut varAcc : Float := 0.0
      for i in [:useLen] do
        let d := audio[i]! - mean
        varAcc := varAcc + d * d
      let std := Float.sqrt (varAcc / useLen.toFloat + 1e-7)
      let denom := if std <= 0.0 then 1.0 else std

      let mut out : Array Float := Array.mkEmpty audio.size
      for i in [:audio.size] do
        if i < useLen then
          out := out.push ((audio[i]! - mean) / denom)
        else
          out := out.push paddingValue
      out

private def buildFeatureAttentionMask
    (validSamples targetSamples hopLength : Nat)
    (returnAttentionMask : Bool)
    : Array Int64 :=
  if hopLength == 0 then
    #[]
  else
    let frames := targetSamples / hopLength
    let validFrames := validSamples / hopLength
    Id.run do
      let mut out : Array Int64 := Array.mkEmpty frames
      for i in [:frames] do
        let v : Int64 :=
          if returnAttentionMask then
            if i < validFrames then 1 else 0
          else
            1
        out := out.push v
      out

private def adjustMaskLength (vals : Array Int64) (target : Nat) : Array Int64 :=
  if vals.size == target then
    vals
  else if vals.size > target then
    vals.extract 0 target
  else
    Id.run do
      let mut out := vals
      let pad := target - vals.size
      for _ in [:pad] do
        out := out.push 0
      out

/-- Faithful Whisper-style feature extraction on one waveform.
    This mirrors the reference path:
    Hann + STFT -> power -> Slaney mel -> log10 -> dynamic range clamp -> normalize. -/
def waveformToWhisperFeatures
    (cfg : PreprocessorConfig)
    (wave : Array Float)
    : IO (WhisperFrontendOutput (device := .CPU) cfg.featureSize (PreprocessorConfig.expectedFrames cfg)) := do
  let targetSamples := PreprocessorConfig.expectedSampleCount cfg
  let expectedFrames := PreprocessorConfig.expectedFrames cfg
  let nFft := if cfg.nFft == 0 then 400 else cfg.nFft
  let hop := if cfg.hopLength == 0 then 160 else cfg.hopLength
  let melBins := cfg.featureSize

  if melBins == 0 then
    throw <| IO.userError "Preprocessor config has feature_size=0"
  if targetSamples == 0 then
    throw <| IO.userError "Preprocessor config has n_samples=0"

  let (prepared0, validLenNat) := takePadRight wave targetSamples cfg.paddingValue
  let prepared :=
    if cfg.doNormalize then
      zeroMeanUnitVar prepared0 validLenNat cfg.paddingValue
    else
      prepared0

  let wav0 : Tensor { device := .CPU, shape := #[targetSamples], dtype := .Float32 } :=
    Tensor.reshapeChecked (Tensor.fromFloatArray prepared) #[targetSamples]
  let wav ←
    if cfg.dither == 0.0 then
      pure wav0
    else
      let noise ← Tensor.randn (device := .CPU) #[targetSamples]
      pure (wav0 + noise * cfg.dither)

  let window : Tensor { device := .CPU, shape := #[nFft], dtype := .Float32 } := Tensor.hannWindow nFft
  let ⟨_, stftDyn⟩ ← Tensor.stft wav nFft hop window true false
  let stftShape := stftDyn.runtimeShape
  if stftShape.size < 3 then
    throw <| IO.userError s!"Unexpected STFT rank {stftShape.size}, expected 3"

  let freqBins := stftShape.getD 0 0
  let stftFrames := stftShape.getD 1 0
  let packed := stftShape.getD 2 0
  if packed != 2 then
    throw <| IO.userError s!"Unexpected STFT packed size {packed}, expected 2"

  let droppedFrames := if stftFrames > 0 then stftFrames - 1 else 0
  if droppedFrames != expectedFrames then
    throw <| IO.userError
      s!"STFT frames mismatch: got={droppedFrames}, expected={expectedFrames} (stft_frames={stftFrames})"

  let stftVals ← stftDyn.toFloatArray
  let freqNat := freqBins
  let stftFramesNat := stftFrames
  let framesNat := expectedFrames
  let packedNat := packed

  let mut powerFlat : Array Float := Array.mkEmpty (freqNat * framesNat)
  for f in [:freqNat] do
    for t in [:framesNat] do
      let base := ((f * stftFramesNat + t) * packedNat)
      let re := stftVals.getD base 0.0
      let im := stftVals.getD (base + 1) 0.0
      powerFlat := powerFlat.push (re * re + im * im)

  let melFilterFlat := buildSlaneyMelFilterBankFlat freqNat melBins cfg.samplingRate
  let powerSpec : Tensor { device := .CPU, shape := #[freqBins, expectedFrames], dtype := .Float32 } :=
    Tensor.reshapeChecked (Tensor.fromFloatArray powerFlat) #[freqBins, expectedFrames]
  let melFilter : Tensor { device := .CPU, shape := #[freqBins, melBins], dtype := .Float32 } :=
    Tensor.reshapeChecked (Tensor.fromFloatArray melFilterFlat) #[freqBins, melBins]
  let melPower : Tensor { device := .CPU, shape := #[melBins, expectedFrames], dtype := .Float32 } :=
    Tensor.mm (Tensor.transpose2d melFilter) powerSpec
  let logSpec : Tensor { device := .CPU, shape := #[melBins, expectedFrames], dtype := .Float32 } :=
    Tensor.log10 (Tensor.clampFloat melPower (1e-10 : Float) (1e10 : Float))
  let maxVal := Tensor.item (Tensor.maxAllChecked logSpec)
  let floored : Tensor { device := .CPU, shape := #[melBins, expectedFrames], dtype := .Float32 } :=
    Tensor.clampFloat logSpec (maxVal - (8.0 : Float)) (1e10 : Float)
  let normalized : Tensor { device := .CPU, shape := #[melBins, expectedFrames], dtype := .Float32 } := (floored + (4.0 : Float)) / (4.0 : Float)
  let inputFeatures : Tensor { device := .CPU, shape := #[1, melBins, expectedFrames], dtype := .Float32 } :=
    Tensor.reshape normalized #[1, melBins, expectedFrames] (by simp [TensorSpec.numelOfShape])

  let maskVals := buildFeatureAttentionMask validLenNat targetSamples hop cfg.returnAttentionMask
  let featureAttentionMask : Tensor { device := .CPU, shape := #[1, expectedFrames], dtype := .Int64 } :=
    Tensor.reshapeChecked (Tensor.fromInt64Array maskVals) #[1, expectedFrames]

  pure { inputFeatures, featureAttentionMask }

/-- Dynamic-length Whisper-style feature extraction for full-wave inference.
    Unlike `waveformToWhisperFeatures`, this does not force 30s framing.
    It uses the full waveform length (with optional min/max duration constraints). -/
def waveformToWhisperFeaturesDynamic
    (cfg : PreprocessorConfig)
    (wave : Array Float)
    (minSeconds : Float := 0.5)
    (maxSeconds : Option Float := none)
    (device : Device := Device.CPU)
    : IO (Sigma (fun frames => WhisperFrontendOutput (device := device) cfg.featureSize frames)) := do
  let nFft := if cfg.nFft == 0 then 400 else cfg.nFft
  let hop := if cfg.hopLength == 0 then 160 else cfg.hopLength
  let melBins := cfg.featureSize

  if melBins == 0 then
    throw <| IO.userError "Preprocessor config has feature_size=0"
  if cfg.samplingRate == 0 then
    throw <| IO.userError "Preprocessor config has sampling_rate=0"

  let minSamplesRaw := ((minSeconds * cfg.samplingRate.toFloat) + 0.5).toUInt64.toNat
  let minSamples := if minSamplesRaw == 0 then 1 else minSamplesRaw
  -- `stft(center := true)` reflection-pads by `nFft/2`, so very short inputs must
  -- still exceed that padding width to avoid a hard runtime exception in libtorch.
  let minStftSamples := (nFft / 2) + 1
  let waveSamples := wave.size
  let targetSamplesRaw := if waveSamples >= minSamples then waveSamples else minSamples
  let targetSamples :=
    if targetSamplesRaw >= minStftSamples then
      targetSamplesRaw
    else
      minStftSamples

  match maxSeconds with
  | none => pure ()
  | some sec =>
      if sec > 0.0 then
        let maxSamples := ((sec * cfg.samplingRate.toFloat) + 0.5).toUInt64.toNat
        if maxSamples > 0 && targetSamples > maxSamples then
          throw <| IO.userError
            s!"Audio duration exceeds maxSeconds={sec}: samples={targetSamples}, limit={maxSamples}"
      else
        pure ()

  let (prepared0, validLenNat) := takePadRight wave targetSamples cfg.paddingValue
  let prepared :=
    if cfg.doNormalize then
      zeroMeanUnitVar prepared0 validLenNat cfg.paddingValue
    else
      prepared0

  let runOnDevice
      (runDevice : Device)
      : IO (Sigma (fun frames => WhisperFrontendOutput (device := runDevice) cfg.featureSize frames)) := do
    let wav0Cpu : Tensor { device := .CPU, shape := #[targetSamples], dtype := .Float32 } :=
      Tensor.reshapeChecked (Tensor.fromFloatArray prepared) #[targetSamples]
    let wav0 : Tensor { device := runDevice, shape := #[targetSamples], dtype := .Float32 } :=
      wav0Cpu.to runDevice
    let wav ←
      if cfg.dither == 0.0 then
        pure wav0
      else
        let noise0 ← Tensor.randn (device := runDevice) #[targetSamples]
        let noise : Tensor { device := runDevice, shape := #[targetSamples], dtype := .Float32 } :=
          noise0
        pure (wav0 + noise * cfg.dither)

    let windowCpu : Tensor { device := .CPU, shape := #[nFft], dtype := .Float32 } := Tensor.hannWindow nFft
    let window : Tensor { device := runDevice, shape := #[nFft], dtype := .Float32 } :=
      windowCpu.to runDevice
    let ⟨_, stftDyn⟩ ← Tensor.stft wav nFft hop window true false
    let stftShape := stftDyn.runtimeShape
    if stftShape.size < 3 then
      throw <| IO.userError s!"Unexpected STFT rank {stftShape.size}, expected 3"

    let freqBins := stftShape.getD 0 0
    let stftFrames := stftShape.getD 1 0
    let packed := stftShape.getD 2 0
    if packed != 2 then
      throw <| IO.userError s!"Unexpected STFT packed size {packed}, expected 2"

    let frames := if stftFrames > 0 then stftFrames - 1 else 0
    let stft : Tensor { device := runDevice, shape := #[freqBins, stftFrames, 2], dtype := .Float32 } :=
      Tensor.reshapeChecked stftDyn #[freqBins, stftFrames, 2]
    let stftTrim : Tensor { device := runDevice, shape := #[freqBins, frames, 2], dtype := .Float32 } :=
      Tensor.sliceChecked stft 1 0 frames #[freqBins, frames, 2]
    let re3 : Tensor { device := runDevice, shape := #[freqBins, frames, 1], dtype := .Float32 } :=
      Tensor.sliceChecked stftTrim 2 0 1 #[freqBins, frames, 1]
    let im3 : Tensor { device := runDevice, shape := #[freqBins, frames, 1], dtype := .Float32 } :=
      Tensor.sliceChecked stftTrim 2 1 1 #[freqBins, frames, 1]
    let re : Tensor { device := runDevice, shape := #[freqBins, frames], dtype := .Float32 } := Tensor.reshape re3 #[freqBins, frames]
    let im : Tensor { device := runDevice, shape := #[freqBins, frames], dtype := .Float32 } := Tensor.reshape im3 #[freqBins, frames]
    let powerSpec : Tensor { device := runDevice, shape := #[freqBins, frames], dtype := .Float32 } := re * re + im * im

    let melFilterFlat := buildSlaneyMelFilterBankFlat freqBins melBins cfg.samplingRate
    let melFilterCpu : Tensor { device := .CPU, shape := #[freqBins, melBins], dtype := .Float32 } :=
      Tensor.reshapeChecked (Tensor.fromFloatArray melFilterFlat) #[freqBins, melBins]
    let melFilter : Tensor { device := runDevice, shape := #[freqBins, melBins], dtype := .Float32 } :=
      melFilterCpu.to runDevice
    let melPower : Tensor { device := runDevice, shape := #[melBins, frames], dtype := .Float32 } :=
      Tensor.mm (Tensor.transpose2d melFilter) powerSpec
    let logSpec : Tensor { device := runDevice, shape := #[melBins, frames], dtype := .Float32 } :=
      Tensor.log10 (Tensor.clampFloat melPower (1e-10 : Float) (1e10 : Float))
    let maxVal := Tensor.item (Tensor.maxAllChecked logSpec)
    let floored : Tensor { device := runDevice, shape := #[melBins, frames], dtype := .Float32 } :=
      Tensor.clampFloat logSpec (maxVal - (8.0 : Float)) (1e10 : Float)
    let normalized : Tensor { device := runDevice, shape := #[melBins, frames], dtype := .Float32 } := (floored + (4.0 : Float)) / (4.0 : Float)
    let inputFeatures : Tensor { device := runDevice, shape := #[1, melBins, frames], dtype := .Float32 } :=
      Tensor.reshape normalized #[1, melBins, frames] (by simp [TensorSpec.numelOfShape])

    let maskVals0 := buildFeatureAttentionMask validLenNat targetSamples hop cfg.returnAttentionMask
    let maskVals := adjustMaskLength maskVals0 frames
    let featureAttentionMaskCpu : Tensor { device := .CPU, shape := #[1, frames], dtype := .Int64 } :=
      Tensor.reshapeChecked (Tensor.fromInt64Array maskVals) #[1, frames]
    let featureAttentionMask : Tensor { device := runDevice, shape := #[1, frames], dtype := .Int64 } :=
      featureAttentionMaskCpu.to runDevice

    pure ⟨frames, { inputFeatures, featureAttentionMask }⟩

  if device == Device.CPU then
    runOnDevice device
  else
    try
      runOnDevice device
    catch _ =>
      let ⟨frames, output⟩ ← runOnDevice Device.CPU
      pure ⟨frames, {
        inputFeatures := output.inputFeatures.to device
        featureAttentionMask := output.featureAttentionMask.to device }⟩

/-- Load WAV, resample to preprocessor sampling rate, then extract Whisper-style features. -/
def wavToWhisperFeatures
    (cfg : PreprocessorConfig)
    (path : String)
    : IO (WhisperFrontendOutput (device := .CPU) cfg.featureSize (PreprocessorConfig.expectedFrames cfg)) := do
  let (sr, wav) ← loadMonoPcm16Wav path
  let wavTarget ←
    if sr == cfg.samplingRate || sr == 0 || cfg.samplingRate == 0 then
      pure wav
    else
      data.resampleSoxrHQ wav sr cfg.samplingRate
  waveformToWhisperFeatures cfg wavTarget

/-- Dynamic-length WAV frontend path.
    Uses full resampled waveform length (subject to optional min/max seconds). -/
def wavToWhisperFeaturesDynamic
    (cfg : PreprocessorConfig)
    (path : String)
    (minSeconds : Float := 0.5)
    (maxSeconds : Option Float := none)
    (device : Device := Device.CPU)
    : IO (Sigma (fun frames => WhisperFrontendOutput (device := device) cfg.featureSize frames)) := do
  let (sr, wav) ← loadMonoPcm16Wav path
  let wavTarget ←
    if sr == cfg.samplingRate || sr == 0 || cfg.samplingRate == 0 then
      pure wav
    else
      data.resampleSoxrHQ wav sr cfg.samplingRate
  waveformToWhisperFeaturesDynamic cfg wavTarget minSeconds maxSeconds device

/-- Backward-compatible convenience wrapper.
    This now runs the faithful Whisper-style frontend instead of pseudo-mel. -/
def wavToPseudoMel {melBins frames : Nat} (path : String) :
    IO (Tensor { device := .CPU, shape := #[1, melBins, frames], dtype := .Float32 }) := do
  let cfg : PreprocessorConfig := {
    featureSize := melBins
    samplingRate := 16000
    hopLength := 160
    nFft := 400
    nSamples := frames * 160
    nbMaxFrames := frames
    returnAttentionMask := true
    doNormalize := false
    dither := 0.0
  }
  let out ← wavToWhisperFeatures cfg path
  pure (Tensor.reshapeChecked out.inputFeatures #[1, melBins, frames])

def fullFeatureAttentionMask {frames : Nat} : Tensor { device := device, shape := #[1, frames], dtype := .Int64 } :=
  (Tensor.fullInt #[1, frames] 1).to device

def normalizeAudioTo16kFromWav (path : String) : IO (Array Float) := do
  let path ← expandHome path
  let (sr, wav) ← loadMonoPcm16Wav path
  if sr == 16000 || sr == 0 then
    pure (normalizeFloatRange wav)
  else
    pure (normalizeFloatRange (← data.resampleSoxrHQ wav sr 16000))

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
        let resolved ← expandHome tmp
        downloadToFile value resolved
        normalizeAudioTo16kFromWav resolved
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

/-- Normalize a batch of unified audio inputs to mono 16k float waveforms. -/
def normalizeAudioInputsTo16k (inputs : Array ASRAudioInput) : IO (Array (Array Float)) := do
  let mut out : Array (Array Float) := Array.mkEmpty inputs.size
  for x in inputs do
    out := out.push (← normalizeAudioInputTo16k x)
  pure out

end torch.qwen3asr
