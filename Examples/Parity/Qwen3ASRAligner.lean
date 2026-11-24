/-
  Examples/Parity/Qwen3ASRAligner.lean

  Parity probe for `parity/tests/qwen3asr_aligner.py`. Request:
  `{"dtype", "audio": <16 kHz PCM16 WAV path>, "prompt": [token], "words": [word]}`,
  the prompt carrying the audio placeholders and a `<timestamp>` marker around
  every word. Features come from the same frontend as transcription. Writes the
  class predicted at each marker (`classes`, `[[class]]`), each word's start and
  end in milliseconds (`timestamps`, `[[[start, end]]]`), and the Float32 tensors
  `mel` (`[1, mels, frames]`) and `logits` (`[1, seq, classes]`).
-/
import Examples.Parity.LM
import Tyr.Model.Qwen3ASR

namespace Examples.Parity.Qwen3ASRAligner

open Lean torch torch.qwen3asr

structure Request where
  dtype : String
  audio : String
  prompt : Array Nat
  words : Array String
  deriving FromJson

def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let request : Request ← IO.ofExcept (fromJson? json)
  let some dtype := DType.ofString? request.dtype
    | throw <| IO.userError s!"unknown dtype '{request.dtype}'"
  let dtype ← IO.ofExcept (RealFloatingDType.ofDType? dtype)
  let weights := args.weights.toString
  let cfg ← Qwen3ASRConfig.loadFromPretrainedDir weights
  unless ThinkerConfig.isForcedAligner cfg.thinkerConfig do
    throw <| IO.userError s!"{weights} is not a forced-aligner checkpoint"
  let pre ← PreprocessorConfig.loadFromPretrainedDir weights (requiredSamplingRate := some 16000)
  let ⟨hMel⟩ ← if h : pre.featureSize = cfg.thinkerConfig.audioConfig.numMelBins then pure (PLift.up h) else
    throw <| IO.userError s!"feature_size {pre.featureSize} is not num_mel_bins {cfg.thinkerConfig.audioConfig.numMelBins}"
  let ⟨dtype, model⟩ ← Qwen3ASRForConditionalGeneration.loadSharded weights cfg device {} (some dtype)
  have : DType.RealFloating dtype.val := inferInstance

  let ⟨frames, features⟩ ← waveformToWhisperFeaturesDynamic pre (← normalizeAudioTo16kFromWav request.audio)
    (device := device)
  let mel : Tensor { dtype := .Float32, device := device, shape := #[1, cfg.thinkerConfig.audioConfig.numMelBins, frames] } :=
    Tensor.reshape features.inputFeatures #[1, cfg.thinkerConfig.audioConfig.numMelBins, frames] (by simp [hMel])
  let prompt ← LM.tokenMatrix #[request.prompt] 1 request.prompt.size device
  let logits ← model.forward prompt (some mel) (some features.featureAttentionMask)

  let predicted ← (Tensor.argmax! logits 2).toUInt64Array
  let marker := cfg.timestampTokenId.toNat
  let classes := (Array.range request.prompt.size).filterMap fun i =>
    if request.prompt[i]! == marker then some (predicted[i]!.toNat) else none
  let aligned ← model.alignFromLogits prompt logits #[request.words]
  let ms (seconds : Float) : Nat := (seconds * 1000.0).round.toUInt64.toNat
  let timestamps := (aligned.getD 0 { items := #[] }).items.map fun item => #[ms item.startTime, ms item.endTime]
  pure {
    json := Json.mkObj [("classes", toJson #[classes]), ("timestamps", toJson #[timestamps])]
    tensors := #[
      ("mel", Tensor.toTensor mel),
      ("logits", Tensor.toTensor (Tensor.castDType logits .Float32))]
  }

end Examples.Parity.Qwen3ASRAligner

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.Qwen3ASRAligner.probe
