/-
  Tyr/Lib/Audio/LogMel.lean

  Log-mel features of a mono waveform: optional zero-mean unit-variance
  normalization and dither, Hann-window STFT power spectrum without its last
  frame, Slaney mel filter bank, `log10` clamped to `[1e-10, 1e10]` before the
  log, floored 8 below the maximum, then `(x + 4) / 4`; and a frame mask that is
  1 on frames of the unpadded waveform (all 1 without `returnAttentionMask`).
-/
import Tyr.Core.Torch
import Tyr.Core.Typed.Neural
import Tyr.IO.Audio
import Tyr.Lib.Audio.LogMelConfig

namespace torch.audio

variable {device : Device}
structure LogMelFeatures (melBins frames : Nat) where
  inputFeatures : Tensor { device := device, shape := #[1, melBins, frames], dtype := .Float32 }
  featureAttentionMask : Tensor { device := device, shape := #[1, frames], dtype := .Int64 }
  /-- Feature extraction fails on audio that yields no frames. -/
  framesPos : 0 < frames

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
          let tri := max 0.0 (min down up)
          let enormDen := f2 - f0
          let enorm := if enormDen == 0.0 then 0.0 else 2.0 / enormDen
          out := out.push (tri * enorm)
      out

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

/-- Features of `wave` truncated or padded to `cfg.expectedSampleCount` samples. -/
def logMelFeatures
    (cfg : LogMelConfig)
    (wave : Array Float)
    : IO (LogMelFeatures (device := .CPU) cfg.featureSize (LogMelConfig.expectedFrames cfg)) := do
  let targetSamples := LogMelConfig.expectedSampleCount cfg
  let expectedFrames := LogMelConfig.expectedFrames cfg
  let nFft := cfg.nFft
  let hop := cfg.hopLength
  let melBins := cfg.featureSize
  have melPos : 0 < melBins := cfg.featureSizePos
  if targetSamples == 0 then
    throw <| IO.userError "Preprocessor config has n_samples=0"

  let prepared0 := (wave.take targetSamples).rightpad targetSamples cfg.paddingValue
  let validLenNat := min wave.size targetSamples
  let prepared :=
    if cfg.doNormalize then
      zeroMeanUnitVar prepared0 validLenNat cfg.paddingValue
    else
      prepared0

  let wav0 : Tensor { device := .CPU, shape := #[targetSamples], dtype := .Float32 } ←
    IO.ofExcept (Tensor.fromFloatArrayAs? prepared #[targetSamples])
  let wav ←
    if cfg.dither == 0.0 then
      pure wav0
    else
      let noise ← Tensor.randn (device := .CPU) #[targetSamples]
      pure (wav0 + noise * cfg.dither)

  let window : Tensor { device := .CPU, shape := #[nFft], dtype := .Float32 } := Tensor.hannWindow nFft
  let ⟨hStft⟩ ← if h : Tensor.StftValid targetSamples nFft hop nFft true then pure (PLift.up h) else
    throw <| IO.userError s!"STFT parameters n_fft={nFft} hop_length={hop} do not fit {targetSamples} samples"
  let stftDyn := Tensor.stft wav nFft hop window true false hStft
  let freqBins := nFft / 2 + 1
  let stftFrames := Tensor.stftFrames targetSamples nFft hop true
  let packed := 2
  if stftDyn.runtimeShape != #[freqBins, stftFrames, packed] then
    throw <| IO.userError s!"Unexpected STFT shape {stftDyn.runtimeShape}, expected {#[freqBins, stftFrames, packed]}"

  let droppedFrames := if stftFrames > 0 then stftFrames - 1 else 0
  if droppedFrames != expectedFrames then
    throw <| IO.userError
      s!"STFT frames mismatch: got={droppedFrames}, expected={expectedFrames} (stft_frames={stftFrames})"
  let ⟨framesPos⟩ ← if h : 0 < expectedFrames then pure (PLift.up h) else
    throw <| IO.userError "Preprocessor config yields no feature frames"

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
  let powerSpec : Tensor { device := .CPU, shape := #[freqBins, expectedFrames], dtype := .Float32 } ←
    IO.ofExcept (Tensor.fromFloatArrayAs? powerFlat #[freqBins, expectedFrames])
  let melFilter : Tensor { device := .CPU, shape := #[freqBins, melBins], dtype := .Float32 } ←
    IO.ofExcept (Tensor.fromFloatArrayAs? melFilterFlat #[freqBins, melBins])
  let melPower : Tensor { device := .CPU, shape := #[melBins, expectedFrames], dtype := .Float32 } :=
    Tensor.mm (Tensor.transpose2d melFilter) powerSpec
  let logSpec : Tensor { device := .CPU, shape := #[melBins, expectedFrames], dtype := .Float32 } :=
    Tensor.log10 (Tensor.clampFloat melPower (1e-10 : Float) (1e10 : Float))
  let maxVal := Tensor.item (Tensor.maxAll logSpec)
  let floored : Tensor { device := .CPU, shape := #[melBins, expectedFrames], dtype := .Float32 } :=
    Tensor.clampFloat logSpec (maxVal - (8.0 : Float)) (1e10 : Float)
  let normalized : Tensor { device := .CPU, shape := #[melBins, expectedFrames], dtype := .Float32 } := (floored + (4.0 : Float)) / (4.0 : Float)
  let inputFeatures : Tensor { device := .CPU, shape := #[1, melBins, expectedFrames], dtype := .Float32 } :=
    Tensor.reshape normalized #[1, melBins, expectedFrames] (by simp [Shape.numel])

  let maskVals := buildFeatureAttentionMask validLenNat targetSamples hop cfg.returnAttentionMask
  let featureAttentionMask : Tensor { device := .CPU, shape := #[1, expectedFrames], dtype := .Int64 } ←
    IO.ofExcept (Tensor.fromInt64ArrayAs? maskVals #[1, expectedFrames])

  pure { inputFeatures, featureAttentionMask, framesPos }

/-- Features of the whole of `wave`, at least `minSeconds` long; longer than
    `maxSeconds` is an error. Falls back to the CPU when `device` fails. -/
def logMelFeaturesDynamic
    (cfg : LogMelConfig)
    (wave : Array Float)
    (minSeconds : Float := 0.5)
    (maxSeconds : Option Float := none)
    (device : Device := Device.CPU)
    : IO (Sigma (fun frames => LogMelFeatures (device := device) cfg.featureSize frames)) := do
  let nFft := cfg.nFft
  let hop := cfg.hopLength
  let melBins := cfg.featureSize
  have melPos : 0 < melBins := cfg.featureSizePos

  let minSamplesRaw := (minSeconds * cfg.samplingRate.toFloat).round.toUInt64.toNat
  let minSamples := if minSamplesRaw == 0 then 1 else minSamplesRaw
  -- `stft(center := true)` reflection-pads by `nFft/2`, so very short inputs must
  -- still exceed that padding width to avoid a hard runtime exception in libtorch.
  let minStftSamples := (nFft / 2) + 1
  let waveSamples := wave.size
  let targetSamplesRaw := max waveSamples minSamples
  let targetSamples :=
    if targetSamplesRaw >= minStftSamples then
      targetSamplesRaw
    else
      minStftSamples

  match maxSeconds with
  | none => pure ()
  | some sec =>
      if sec > 0.0 then
        let maxSamples := (sec * cfg.samplingRate.toFloat).round.toUInt64.toNat
        if maxSamples > 0 && targetSamples > maxSamples then
          throw <| IO.userError
            s!"Audio duration exceeds maxSeconds={sec}: samples={targetSamples}, limit={maxSamples}"
      else
        pure ()

  let prepared0 := (wave.take targetSamples).rightpad targetSamples cfg.paddingValue
  let validLenNat := min wave.size targetSamples
  let prepared :=
    if cfg.doNormalize then
      zeroMeanUnitVar prepared0 validLenNat cfg.paddingValue
    else
      prepared0

  let runOnDevice
      (runDevice : Device)
      : IO (Sigma (fun frames => LogMelFeatures (device := runDevice) cfg.featureSize frames)) := do
    let wav0Cpu : Tensor { device := .CPU, shape := #[targetSamples], dtype := .Float32 } ←
      IO.ofExcept (Tensor.fromFloatArrayAs? prepared #[targetSamples])
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
    let ⟨hStft⟩ ← if h : Tensor.StftValid targetSamples nFft hop nFft true then pure (PLift.up h) else
      throw <| IO.userError s!"STFT parameters n_fft={nFft} hop_length={hop} do not fit {targetSamples} samples"
    let stft := Tensor.stft wav nFft hop window true false hStft
    let freqBins := nFft / 2 + 1
    let stftFrames := Tensor.stftFrames targetSamples nFft hop true
    if stft.runtimeShape != #[freqBins, stftFrames, 2] then
      throw <| IO.userError s!"Unexpected STFT shape {stft.runtimeShape}, expected {#[freqBins, stftFrames, 2]}"

    let frames := if stftFrames > 0 then stftFrames - 1 else 0
    let ⟨framesPos⟩ ← if h : 0 < frames then pure (PLift.up h) else
      throw <| IO.userError s!"Audio too short for log-mel features: {targetSamples} samples yield no frames"
    let stftTrim : Tensor { device := runDevice, shape := #[freqBins, frames, 2], dtype := .Float32 } :=
      Tensor.slice stft 1 0 frames (_bounds := by simp only [frames]; split <;> simp <;> omega)
    let re3 : Tensor { device := runDevice, shape := #[freqBins, frames, 1], dtype := .Float32 } :=
      Tensor.slice stftTrim 2 0 1
    let im3 : Tensor { device := runDevice, shape := #[freqBins, frames, 1], dtype := .Float32 } :=
      Tensor.slice stftTrim 2 1 1
    let re : Tensor { device := runDevice, shape := #[freqBins, frames], dtype := .Float32 } := Tensor.reshape re3 #[freqBins, frames]
    let im : Tensor { device := runDevice, shape := #[freqBins, frames], dtype := .Float32 } := Tensor.reshape im3 #[freqBins, frames]
    let powerSpec : Tensor { device := runDevice, shape := #[freqBins, frames], dtype := .Float32 } := re * re + im * im

    let melFilterFlat := buildSlaneyMelFilterBankFlat freqBins melBins cfg.samplingRate
    let melFilterCpu : Tensor { device := .CPU, shape := #[freqBins, melBins], dtype := .Float32 } ←
      IO.ofExcept (Tensor.fromFloatArrayAs? melFilterFlat #[freqBins, melBins])
    let melFilter : Tensor { device := runDevice, shape := #[freqBins, melBins], dtype := .Float32 } :=
      melFilterCpu.to runDevice
    let melPower : Tensor { device := runDevice, shape := #[melBins, frames], dtype := .Float32 } :=
      Tensor.mm (Tensor.transpose2d melFilter) powerSpec
    let logSpec : Tensor { device := runDevice, shape := #[melBins, frames], dtype := .Float32 } :=
      Tensor.log10 (Tensor.clampFloat melPower (1e-10 : Float) (1e10 : Float))
    let maxVal := Tensor.item (Tensor.maxAll logSpec)
    let floored : Tensor { device := runDevice, shape := #[melBins, frames], dtype := .Float32 } :=
      Tensor.clampFloat logSpec (maxVal - (8.0 : Float)) (1e10 : Float)
    let normalized : Tensor { device := runDevice, shape := #[melBins, frames], dtype := .Float32 } := (floored + (4.0 : Float)) / (4.0 : Float)
    let inputFeatures : Tensor { device := runDevice, shape := #[1, melBins, frames], dtype := .Float32 } :=
      Tensor.reshape normalized #[1, melBins, frames] (by simp [Shape.numel])

    let maskVals0 := buildFeatureAttentionMask validLenNat targetSamples hop cfg.returnAttentionMask
    let maskVals := (maskVals0.take frames).rightpad frames 0
    let featureAttentionMaskCpu : Tensor { device := .CPU, shape := #[1, frames], dtype := .Int64 } ←
      IO.ofExcept (Tensor.fromInt64ArrayAs? maskVals #[1, frames])
    let featureAttentionMask : Tensor { device := runDevice, shape := #[1, frames], dtype := .Int64 } :=
      featureAttentionMaskCpu.to runDevice

    pure ⟨frames, { inputFeatures, featureAttentionMask, framesPos }⟩

  if device == Device.CPU then
    runOnDevice device
  else
    try
      runOnDevice device
    catch _ =>
      let ⟨frames, output⟩ ← runOnDevice Device.CPU
      pure ⟨frames, {
        inputFeatures := output.inputFeatures.to device
        featureAttentionMask := output.featureAttentionMask.to device
        framesPos := output.framesPos }⟩

/-- `logMelFeatures` of a PCM16 WAV file resampled to `cfg.samplingRate`. -/
def wavLogMelFeatures
    (cfg : LogMelConfig)
    (path : String)
    : IO (LogMelFeatures (device := .CPU) cfg.featureSize (LogMelConfig.expectedFrames cfg)) := do
  let (sr, wav) ← loadMonoPcm16Wav path
  let wavTarget ←
    if sr == cfg.samplingRate || sr == 0 then
      pure wav
    else
      data.resampleSoxrHQ wav sr cfg.samplingRate
  logMelFeatures cfg wavTarget

end torch.audio
