/-
  Examples/Parity/Qwen3TTSTalker.lean

  Parity probe for `parity/tests/qwen3tts_talker.py`. Request:
  `{"dtype", "tensors": <safetensors path>, "max_frames", "repetition_penalty"}`,
  the file holding the reference's talker inputs: `inputs_embeds`
  (`[1, seq, hidden]`), `trailing_text_hidden` (`[1, steps, hidden]`) and
  `tts_pad_embed` (`[1, 1, hidden]`). Generates greedily, with the reference's
  codec-token suppression and the request's repetition penalty, until EOS or
  `max_frames`; writes the codes before EOS (`codes`, `[frames, groups]`).
-/
import Examples.Parity.Common
import Tyr.Model.Qwen3TTS
import Tyr.IO.SafeTensors.Source

namespace Examples.Parity.Qwen3TTSTalker

open Lean torch torch.qwen3tts

structure Request where
  dtype : String
  tensors : String
  max_frames : Nat
  repetition_penalty : Float
  deriving FromJson

def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let request : Request ← IO.ofExcept (fromJson? json)
  let some dtype := DType.ofString? request.dtype
    | throw <| IO.userError s!"unknown dtype '{request.dtype}'"
  let dtype ← IO.ofExcept (RealFloatingDType.ofDType? dtype)
  let weights := args.weights.toString
  let cfg ← Qwen3TTSConfig.loadFromPretrainedDir weights
  let ⟨dtypes, model⟩ ← Qwen3TTSForConditionalGeneration.loadSharded weights cfg device {} (some dtype)
  let hidden := cfg.talkerConfig.hiddenSize
  let src ← safetensors.Source.open request.tensors
  let length (name : String) : IO Nat := do
    let some spec := src.spec? name | throw <| IO.userError s!"{request.tensors} has no '{name}'"
    let some n := spec.shape[1]? | throw <| IO.userError s!"'{name}' has shape {spec.shape}"
    pure n
  let seq ← length "inputs_embeds"
  let steps ← length "trailing_text_hidden"
  let ⟨_, inputs⟩ ← src.load "inputs_embeds" #[1, seq, hidden] device
  let ⟨_, trailing⟩ ← src.load "trailing_text_hidden" #[1, steps, hidden] device
  let ⟨_, pad⟩ ← src.load "tts_pad_embed" #[1, 1, hidden] device
  -- Greedy: top-1 sampling at temperature 1 always draws the argmax.
  let top1 : TopK := some ⟨1, by decide⟩
  let out ← TalkerForConditionalGeneration.generateCodesWithLengths cfg.talkerConfig model.talker
    (inputs.castDType dtypes.talker.val) request.max_frames (minNewTokens := 2)
    (temperature := 1.0) (topK := top1) (subtalkerTemperature := 1.0) (subtalkerTopK := top1)
    (repetitionPenalty := request.repetition_penalty) (suppressTail := 1024)
    (trailingTextHidden := some ⟨steps, trailing.castDType dtypes.talker.val⟩)
    (ttsPadEmbed := some (pad.castDType dtypes.talker.val))
  let groups := cfg.talkerConfig.numCodeGroups
  let frames := out.lengths.getD 0 0
  let codes ← out.codes.toUInt64Array
  let rows := (Array.range frames).map fun f => (codes.extract (groups * f) (groups * f + groups)).map (·.toNat)
  pure { json := Json.mkObj [("codes", toJson rows)] }

end Examples.Parity.Qwen3TTSTalker

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.Qwen3TTSTalker.probe
