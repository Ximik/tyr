/-
  Examples/Parity/Qwen3ASR.lean

  Parity probe for `parity/tests/qwen3asr.py`. `--weights` is the model
  directory with its tokenizer files. Request:
  `{"dtype", "audio": <16 kHz PCM16 WAV path>, "language", "max_new_tokens", "forced": [token]}`.
  Features and the prompt, forcing `language`, are built as transcription
  builds them. Writes the prompt with its audio placeholders (`prompt`,
  `[token]`), the greedy tokens after it (`tokens`, `[[token]]`) and the
  Float32 tensors `mel` (`[1, mels, frames]`),
  `encoder` (the audio embeddings that fill the placeholders, `[audioTokens, hidden]`)
  and `logits` (`[1, steps, vocab]`) of feeding `forced` after the prompt.
-/
import Examples.Parity.LM
import Tyr.Model.Qwen3ASR
import Tyr.Lib.Tokenizer.ByteLevelBPE

namespace Examples.Parity.Qwen3ASR

open Lean torch torch.qwen3asr

structure Request where
  dtype : String
  audio : String
  language : String
  max_new_tokens : Nat
  forced : Array Nat
  deriving FromJson

def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let request : Request ← IO.ofExcept (fromJson? json)
  let some dtype := DType.ofString? request.dtype
    | throw <| IO.userError s!"unknown dtype '{request.dtype}'"
  let dtype ← IO.ofExcept (RealFloatingDType.ofDType? dtype)
  let weights := args.weights.toString
  let cfg ← Qwen3ASRConfig.loadFromPretrainedDir weights
  let language ← IO.ofExcept (normalizeLanguageName request.language)
  IO.ofExcept (validateLanguage cfg.supportLanguages language)
  let tok ← tokenizer.ByteLevelBPETokenizer.load weights
  let pre ← audio.LogMelConfig.loadFromPretrainedDir weights (requiredSamplingRate := some 16000)
  let ⟨hMel⟩ ← if h : pre.featureSize = cfg.thinkerConfig.audioConfig.numMelBins then pure (PLift.up h) else
    throw <| IO.userError s!"feature_size {pre.featureSize} is not num_mel_bins {cfg.thinkerConfig.audioConfig.numMelBins}"
  let ⟨dtype, model⟩ ← Qwen3ASRForConditionalGeneration.loadSharded weights cfg device {} (some dtype)
  have : DType.RealFloating dtype.val := inferInstance

  let ⟨frames, features⟩ ← audio.logMelFeaturesDynamic pre (← audio.normalizeAudioTo16kFromWav request.audio)
    (device := device)
  let mel : Tensor { dtype := .Float32, device := device, shape := #[1, cfg.thinkerConfig.audioConfig.numMelBins, frames] } :=
    Tensor.reshape features.inputFeatures #[1, cfg.thinkerConfig.audioConfig.numMelBins, frames] (by simp [hMel])
  let mask := features.featureAttentionMask

  let featureLens := (← (Tensor.sumDim mask 1 false).toUInt64Array).map (·.toNat)
  let audioTokens := (AudioEncoderConfig.featExtractOutputLengths featureLens).getD 0 0
  let encoded ← model.thinker.encodeAudioVarLenIO mel featureLens false
  let encoderRaw := Tensor.toTensor (Tensor.castDType encoded .Float32)
  let encoder := RawTensor.reshape! (RawTensor.slice encoderRaw 1 0 audioTokens) #[audioTokens, cfg.thinkerConfig.audioConfig.outputDim]

  let promptIds ← IO.ofExcept <| ({} : Qwen3ASRProcessor).encodeWithExpandedAudioTokenIds tok
    (buildTextPrompt "" (some language)) #[audioTokens] cfg.thinkerConfig.audioTokenId
  let promptIds := promptIds.map (·.toNat)
  let prompt ← LM.tokenMatrix #[promptIds] 1 promptIds.size device
  let forced ← LM.tokenMatrix #[request.forced] 1 request.forced.size device
  let ⟨outSeq, generated⟩ ← model.generateGreedy prompt (some mel) (some mask) request.max_new_tokens #[]
  let flat ← generated.toUInt64Array
  let tokens := (flat.extract promptIds.size outSeq).map (·.toNat)
  let logits ← model.thinker.forcedLogits prompt (some mel) (some mask) forced
  pure {
    json := Json.mkObj [("prompt", toJson promptIds), ("tokens", toJson #[tokens])]
    tensors := #[
      ("mel", Tensor.toTensor mel),
      ("encoder", encoder),
      ("logits", Tensor.toTensor (Tensor.castDType logits .Float32))]
  }

end Examples.Parity.Qwen3ASR

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.Qwen3ASR.probe
