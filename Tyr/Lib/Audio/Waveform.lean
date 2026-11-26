/-
  Tyr/Lib/Audio/Waveform.lean

  Mono waveform resampling, range normalization and loading at 16 kHz.
-/
import Tyr.Core.FFI.Runtime
import Tyr.IO.Audio

namespace torch.audio

private def expandHome (path : String) : IO String := do
  if path == "~" then
    return (← IO.getEnv "HOME").getD path
  else if path.startsWith "~/" then
    return s!"{(← IO.getEnv "HOME").getD ""}/{path.drop 2}"
  else
    return path

private def clamp01 (x : Float) : Float :=
  if x > 1.0 then 1.0 else if x < (-1.0) then -1.0 else x

/-- `audio` divided by its peak when the peak exceeds 1, clamped to `[-1, 1]`. -/
def normalizeFloatRange (audio : Array Float) : Array Float :=
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

/-- Linear-interpolation resampling from `srcRate` to `targetRate`. -/
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

/-- A PCM16 WAV file as a 16 kHz mono waveform normalized by `normalizeFloatRange`;
    a leading `~` in `path` is the home directory. -/
def normalizeAudioTo16kFromWav (path : String) : IO (Array Float) := do
  let path ← expandHome path
  let (sr, wav) ← loadMonoPcm16Wav path
  if sr == 16000 || sr == 0 then
    pure (normalizeFloatRange wav)
  else
    pure (normalizeFloatRange (← data.resampleSoxrHQ wav sr 16000))

end torch.audio
