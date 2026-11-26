/-
  Examples/Parity/Qwen3TTSSpeaker.lean

  Parity probe for `parity/tests/qwen3tts_speaker.py`. Request:
  `{"dtype", "mel": <safetensors path>}`, the file holding `mel`
  (`[1, frames, mels]`). Writes the speaker embedding (`embedding`, Float32
  `[1, dim]`).
-/
import Examples.Parity.Common
import Tyr.Model.Qwen3TTS
import Tyr.IO.SafeTensors.Source

namespace Examples.Parity.Qwen3TTSSpeaker

open Lean torch torch.qwen3tts

structure Request where
  dtype : String
  mel : String
  deriving FromJson

def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let request : Request ← IO.ofExcept (fromJson? json)
  let some dtype := DType.ofString? request.dtype
    | throw <| IO.userError s!"unknown dtype '{request.dtype}'"
  let dtype ← IO.ofExcept (RealFloatingDType.ofDType? dtype)
  let weights := args.weights.toString
  let cfg ← Qwen3TTSConfig.loadFromPretrainedDir weights
  let ⟨_, model⟩ ← Qwen3TTSForConditionalGeneration.loadSharded weights cfg device {} (some dtype)
  let src ← safetensors.Source.open request.mel
  let some spec := src.spec? "mel" | throw <| IO.userError s!"{request.mel} has no 'mel'"
  let some frames := spec.shape[1]? | throw <| IO.userError s!"'mel' has shape {spec.shape}"
  let ⟨_, mel⟩ ← src.load "mel" #[1, frames, cfg.speakerEncoderConfig.melDim] device
  let embedding ← model.extractSpeakerEmbedding mel
  pure {
    json := Json.mkObj []
    tensors := #[("embedding", Tensor.toTensor (embedding.castDType .Float32))]
  }

end Examples.Parity.Qwen3TTSSpeaker

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.Qwen3TTSSpeaker.probe
