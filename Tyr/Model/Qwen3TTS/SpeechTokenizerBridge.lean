/-
  Tyr/Model/Qwen3TTS/SpeechTokenizerBridge.lean

  Qwen3-TTS speech-tokenizer integration helpers:
  - decode codec IDs -> waveform (legacy Python bridge path)
  - encode waveform -> codec IDs (Lean 12Hz-family path)
  - prepare speaker mel from reference audio (Lean-native)

  Design intent:
  Keep inference in Lean and retain Python only for optional decode fallback.
-/
import Tyr.Model.Qwen3TTS.Model
import Tyr.Model.Qwen3TTS.SpeechTokenizerEncoder
import Tyr.Model.Qwen3TTS.SpeechTokenizer25HzEncoder
import Tyr.Model.Qwen3ASR.Frontend
import Tyr.Log

namespace torch.qwen3tts

variable {device : Device}

open torch.Log

structure SpeechTokenizerBridgeConfig where
  pythonExe : String := "uv"
  qwenRepo : String := "../Qwen3-TTS"
  speechTokenizerDir : Option String := none
  decodeScript : String := "scripts/qwen3tts_decode_codes.py"
  /-- Legacy fallback field; encode is now Lean-native. -/
  encodeScript : String := "scripts/qwen3tts_encode_audio.py"
  /-- Legacy fallback field; speaker mel prepare is now Lean-native. -/
  speakerMelScript : String := "scripts/qwen3tts_prepare_speaker_mel.py"
  deviceMap : Option String := none
  deriving Repr, Inhabited

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

private def pythonPrefix (pythonExe : String) : Array String :=
  if pythonExe == "uv" then #["run", "python"] else #[]

private def runBridgeCommand (pythonExe : String) (args : Array String) (errorPrefix : String)
    (log : Handlers := {})
    : IO Unit := do
  let result ← IO.Process.output {
    cmd := pythonExe
    args := args
  }
  if result.exitCode != 0 then
    throw <| IO.userError s!"{errorPrefix} (exit={result.exitCode}):\n{result.stderr}"
  let stdout := result.stdout.trimAscii.toString
  if !stdout.isEmpty then
    log.onInfo stdout

private def resolveSpeechTokenizerDir (bridge : SpeechTokenizerBridgeConfig) (modelDir : String) : IO String := do
  match bridge.speechTokenizerDir with
  | some d => expandHome d
  | none => pure s!"{modelDir}/speech_tokenizer"

private def isTokenizer25Hz (speechTokenizerDir : String) : IO Bool := do
  let cfgPath := s!"{speechTokenizerDir}/config.json"
  let cfgText ← IO.FS.readFile cfgPath
  pure (cfgText.contains "\"qwen3_tts_tokenizer_25hz\"")

private def fmax (a b : Float) : Float := if a > b then a else b
private def fmin (a b : Float) : Float := if a < b then a else b

private partial def reflectIndex (n : Int) (i : Int) : Int :=
  if n <= 1 then
    0
  else if i < 0 then
    reflectIndex n (-i)
  else if i >= n then
    reflectIndex n ((2 * n - 2) - i)
  else
    i

private def reflectPad1d (xs : Array Float) (pad : Nat) : Array Float :=
  if pad == 0 then
    xs
  else
    let n := xs.size
    if n == 0 then
      Array.replicate (pad * 2) 0.0
    else if n == 1 then
      Array.replicate (n + pad * 2) xs[0]!
    else
      Id.run do
        let total := n + pad * 2
        let mut out : Array Float := Array.mkEmpty total
        let nI : Int := Int.ofNat n
        let pI : Int := Int.ofNat pad
        for k in [:total] do
          let srcI : Int := Int.ofNat k - pI
          let idxI := reflectIndex nI srcI
          let idx : Nat := Int.toNat idxI
          out := out.push (xs[idx]!)
        out

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
    (minHz maxHz : Float)
    : Array Float :=
  if numFrequencyBins < 2 || numMelFilters == 0 then
    #[]
  else
    let nyquist : Float := (samplingRate / 2).toFloat
    let lo := fmax 0.0 minHz
    let hi := fmin nyquist (fmax lo maxHz)
    let melMin := hertzToMelSlaney lo
    let melMax := hertzToMelSlaney hi
    let melFreqs := linspace melMin melMax (numMelFilters + 2)
    let filterFreqs := melFreqs.map melToHertzSlaney
    let fftFreqs := linspace 0.0 nyquist numFrequencyBins
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
          let down := if downDen == 0.0 then 0.0 else (fft - f0) / downDen
          let up := if upDen == 0.0 then 0.0 else (f2 - fft) / upDen
          let tri := fmax 0.0 (fmin down up)
          let normDen := f2 - f0
          let norm := if normDen == 0.0 then 0.0 else 2.0 / normDen
          out := out.push (tri * norm)
      out

private def computeSpeakerMelFromWave
    (wave : Array Float)
    (sampleRate : Nat)
    (melDim : Nat)
    (nFft : Nat := 1024)
    (hopSize : Nat := 256)
    (winSize : Nat := 1024)
    (fminHz : Float := 0.0)
    (fmaxHz : Option Float := none)
    : IO (Sigma fun frames => Tensor { shape := #[1, frames, melDim], dtype := .Float32, device := .CPU }) := do
  if wave.isEmpty then
    throw <| IO.userError "Reference audio is empty after loading/resampling."
  if nFft == 0 || hopSize == 0 || winSize == 0 then
    throw <| IO.userError "Invalid STFT config for speaker mel (n_fft/hop_size/win_size must be > 0)."

  let pad : Nat := if nFft > hopSize then (nFft - hopSize) / 2 else 0
  let padded := reflectPad1d wave pad
  let paddedLen : Nat := padded.size
  let wav : Tensor { shape := #[paddedLen], dtype := .Float32, device := .CPU } := Tensor.reshape (Tensor.fromFloatArray padded) #[paddedLen]

  let window : Tensor { shape := #[winSize], dtype := .Float32, device := .CPU } := Tensor.hannWindow winSize
  let ⟨_, stftDyn⟩ ← Tensor.stft wav nFft hopSize window false false
  let stftShape := stftDyn.runtimeShape
  if stftShape.size < 3 then
    throw <| IO.userError s!"Unexpected STFT rank {stftShape.size}, expected 3"

  let freqBins : Nat := stftShape.getD 0 0
  let frames : Nat := stftShape.getD 1 0
  let packed : Nat := stftShape.getD 2 0
  if packed != 2 then
    throw <| IO.userError s!"Unexpected STFT packed size {packed}, expected 2"
  if frames == 0 then
    throw <| IO.userError "Speaker mel extraction produced zero frames."

  let spec : Tensor { shape := #[freqBins, frames, 2], dtype := .Float32, device := .CPU } := Tensor.reshapeChecked stftDyn #[freqBins, frames, 2]
  let re3 : Tensor { shape := #[freqBins, frames, 1], dtype := .Float32, device := .CPU } := Tensor.sliceChecked spec 2 0 1 #[freqBins, frames, 1]
  let im3 : Tensor { shape := #[freqBins, frames, 1], dtype := .Float32, device := .CPU } := Tensor.sliceChecked spec 2 1 1 #[freqBins, frames, 1]
  let re : Tensor { shape := #[freqBins, frames], dtype := .Float32, device := .CPU } := Tensor.reshape re3 #[freqBins, frames]
  let im : Tensor { shape := #[freqBins, frames], dtype := .Float32, device := .CPU } := Tensor.reshape im3 #[freqBins, frames]
  let mag : Tensor { shape := #[freqBins, frames], dtype := .Float32, device := .CPU } := Tensor.sqrt ((re * re) + (im * im) + (1e-9 : Float))

  let melMaxHz : Float := fmaxHz.getD (sampleRate.toFloat / 2.0)
  let melFlat := buildSlaneyMelFilterBankFlat
    freqBins melDim sampleRate fminHz melMaxHz
  let melFilter : Tensor { shape := #[freqBins, melDim], dtype := .Float32, device := .CPU } := Tensor.reshapeChecked (Tensor.fromFloatArray melFlat) #[freqBins, melDim]
  let melPow : Tensor { shape := #[melDim, frames], dtype := .Float32, device := .CPU } := Tensor.mm (Tensor.transpose2d melFilter) mag
  let melLog : Tensor { shape := #[melDim, frames], dtype := .Float32, device := .CPU } := Tensor.log (Tensor.clampFloat melPow (1e-5 : Float) (1e10 : Float))
  let melFrameMajor : Tensor { shape := #[frames, melDim], dtype := .Float32, device := .CPU } := Tensor.transpose2d melLog
  let mel3 : Tensor { shape := #[1, frames, melDim], dtype := .Float32, device := .CPU } := Tensor.reshape melFrameMajor #[1, frames, melDim]
  pure ⟨frames, mel3⟩

private def loadResampledMonoWav (path : String) (targetSr : Nat) : IO (Array Float) := do
  let (sr, wav) ← qwen3asr.loadMonoPcm16Wav path
  if sr == targetSr then
    pure wav
  else
    data.resampleSoxrHQ wav sr targetSr

private def formatCodesRows (flat : Array UInt64) (numCodeGroups : Nat) : String := Id.run do
  let g := numCodeGroups
  if g == 0 then
    return ""
  let rows := flat.size / g
  let mut lines : Array String := Array.mkEmpty rows
  for r in [:rows] do
    let mut toks : Array String := Array.mkEmpty g
    for c in [:g] do
      toks := toks.push (toString (flat[r * g + c]!))
    lines := lines.push (String.intercalate " " toks.toList)
  if lines.isEmpty then
    ""
  else
    String.intercalate "\n" lines.toList ++ "\n"

/-- Decode codec ID rows to a waveform file using the speech tokenizer bridge. -/
def decodeCodesToWav
    (bridge : SpeechTokenizerBridgeConfig)
    (modelDir : String)
    (talkerCfg : TalkerConfig)
    (codesPath wavPath : String)
    (log : Handlers := {})
    : IO Unit := do
  let decodeScript ← expandHome bridge.decodeScript
  let qwenRepo ← expandHome bridge.qwenRepo
  let speechTokenizerDir ← resolveSpeechTokenizerDir bridge modelDir
  let codesPath ← expandHome codesPath
  let wavPath ← expandHome wavPath
  ensureParentDir wavPath

  let baseArgs :=
    pythonPrefix bridge.pythonExe ++
      #[
        decodeScript,
        "--speech-tokenizer-dir", speechTokenizerDir,
        "--codes", codesPath,
        "--num-code-groups", toString talkerCfg.numCodeGroups,
        "--output-wav", wavPath,
        "--qwen3-tts-repo", qwenRepo
      ]
  let args :=
    match bridge.deviceMap with
    | some dm => baseArgs ++ #["--device-map", dm]
    | none => baseArgs
  runBridgeCommand bridge.pythonExe args "Speech tokenizer decode failed" log

/-- Encode an audio input into codec ID rows using the speech tokenizer bridge. -/
def encodeAudioToCodes
    (bridge : SpeechTokenizerBridgeConfig)
    (modelDir : String)
    (audioPath codesPath : String)
    (device : Device := Device.CPU)
    : IO Unit := do
  let speechTokenizerDir ← resolveSpeechTokenizerDir bridge modelDir
  let audioPath ← expandHome audioPath
  let codesPath ← expandHome codesPath
  ensureParentDir codesPath
  if (← isTokenizer25Hz speechTokenizerDir) then
    let ⟨_dtype, encoder25⟩ ← SpeechTokenizer25HzEncoder.loadFromDir speechTokenizerDir device
    let wave ← loadResampledMonoWav audioPath encoder25.inputSampleRate
    if wave.isEmpty then
      throw <| IO.userError s!"Audio encode input is empty: {audioPath}"
    let (groups, vals) ← encoder25.encodeWaveToCodes wave
    IO.FS.writeFile codesPath (formatCodesRows vals groups)
  else
    let ⟨_dtypes, encoder⟩ ← SpeechTokenizer12HzEncoder.loadFromDirFlexible speechTokenizerDir device
    let wave ← loadResampledMonoWav audioPath encoder.inputSampleRate
    if wave.isEmpty then
      throw <| IO.userError s!"Audio encode input is empty: {audioPath}"
    let samples : Nat := wave.size
    -- Keep long-form encode numerically stable by processing in bounded chunks.
    let chunkSamples : Nat := (encoder.inputSampleRate * 10)
    let leftContextSamples : Nat := (encoder.inputSampleRate * 12)
    if wave.size <= chunkSamples then
      let audio0 : Tensor { shape := #[samples], dtype := .Float32, device := .CPU } := Tensor.reshape (Tensor.fromFloatArray wave) #[samples]
      let audio : Tensor { shape := #[samples], dtype := .Float32, device := device } :=
        audio0.to encoder.conv0Weight.device
      let frames : Nat := encodedFrames samples
      let ⟨groups, codes⟩ := encoder.encodeMonoFrameMajorWithRoPEOffsetDynamic (Tensor.castLike encoder.conv0Weight audio)
      let flat : Tensor { shape := #[frames * groups], dtype := .Int64, device := device } := Tensor.reshape codes #[frames * groups]
      let vals ← Tensor.toUInt64Array flat
      IO.FS.writeFile codesPath (formatCodesRows vals groups)
    else
      let expectedRows : Nat := (encodedFrames samples)
      IO.FS.withFile codesPath .write fun h => do
        let mut st :=
          SpeechTokenizer12HzEncoder.initEncodeStreamState chunkSamples leftContextSamples
        let mut off : Nat := 0
        let mut rowsWritten : Nat := 0
        let mut groups? : Option Nat := none
        while _h : off < wave.size && rowsWritten < expectedRows do
          let remaining := wave.size - off
          let takeN := Nat.min chunkSamples remaining
          let inputChunk := wave.extract off (off + takeN)
          let (st', groups, valsKeep) ← SpeechTokenizer12HzEncoder.pushEncodeStreamDynamic encoder st inputChunk
          st := st'
          match groups? with
          | some g0 =>
              if g0 != groups then
                throw <| IO.userError
                  s!"Inconsistent encoder code-group count across chunks ({g0} vs {groups})."
          | none =>
              groups? := some groups
          let groupsNat : Nat := groups
          let availRows := valsKeep.size / groupsNat
          let rowsToWrite := Nat.min availRows (expectedRows - rowsWritten)
          if rowsToWrite > 0 then
            let writeVals := valsKeep.extract 0 (rowsToWrite * groupsNat)
            h.putStr (formatCodesRows writeVals groups)
            rowsWritten := rowsWritten + rowsToWrite
          off := off + takeN

/-- Build speaker-encoder mel tensor from reference audio with upstream mel settings.
    Output tensor is stored as SafeTensors with key `mel` and shape `[1, frames, mel_dim]`. -/
def prepareSpeakerMel
    (_bridge : SpeechTokenizerBridgeConfig)
    (audioPath melOutPath melFramesPath : String)
    (sampleRate : Nat := 24000)
    (melDim : Nat := 128)
    : IO Unit := do
  let audioPath ← expandHome audioPath
  let melOutPath ← expandHome melOutPath
  let melFramesPath ← expandHome melFramesPath
  ensureParentDir melOutPath
  ensureParentDir melFramesPath
  let wave ← loadResampledMonoWav audioPath sampleRate
  let ⟨frames, mel⟩ ← computeSpeakerMelFromWave wave sampleRate melDim 1024 256 1024 0.0 (some (sampleRate.toFloat / 2.0))
  Tensor.saveSafeTensor mel melOutPath "mel"
  IO.FS.writeFile melFramesPath s!"{frames}\n"

/-- End-to-end speaker embedding from reference audio:
    audio -> mel (bridge) -> Lean speaker encoder forward. -/
def extractSpeakerEmbeddingFromAudio {dtypes : DTypeLayout}
    (cfg : Qwen3TTSConfig)
    (model : Qwen3TTSForConditionalGeneration (device := device) cfg dtypes)
    (_bridge : SpeechTokenizerBridgeConfig)
    (audioPath melOutPath melFramesPath : String)
    : IO (Tensor { shape := #[1, 1, cfg.talkerConfig.hiddenSize], dtype := dtypes.speaker.val, device := device }) := do
  if cfg.speakerEncoderConfig.encDim != cfg.talkerConfig.hiddenSize then
    throw <| IO.userError
      s!"speaker embedding dim mismatch: speaker enc_dim={cfg.speakerEncoderConfig.encDim}, talker hidden={cfg.talkerConfig.hiddenSize}"
  let audioPath ← expandHome audioPath
  let wave ← loadResampledMonoWav audioPath cfg.speakerEncoderConfig.sampleRate
  let ⟨frames, mel⟩ ← computeSpeakerMelFromWave
    wave
    cfg.speakerEncoderConfig.sampleRate
    cfg.speakerEncoderConfig.melDim
    1024 256 1024 0.0 (some (cfg.speakerEncoderConfig.sampleRate.toFloat / 2.0))

  let melOutPath := melOutPath.trimAscii.toString
  let melFramesPath := melFramesPath.trimAscii.toString
  if !melOutPath.isEmpty then
    let melOutPath ← expandHome melOutPath
    ensureParentDir melOutPath
    Tensor.saveSafeTensor mel melOutPath "mel"
  if !melFramesPath.isEmpty then
    let melFramesPath ← expandHome melFramesPath
    ensureParentDir melFramesPath
    IO.FS.writeFile melFramesPath s!"{frames}\n"

  let spk : Tensor { shape := #[1, cfg.speakerEncoderConfig.encDim], dtype := dtypes.speaker.val, device := device } ← model.extractSpeakerEmbedding (mel.to device)
  pure (Tensor.reshapeChecked spk #[1, 1, cfg.talkerConfig.hiddenSize])

end torch.qwen3tts
