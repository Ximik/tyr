/-
  Examples/Parity/Qwen3TTSDecode.lean

  Parity probe for `parity/tests/qwen3tts_decode.py`. `--weights` is the model
  directory; its `speech_tokenizer/` is read. Request: `{"codes": [[code]]}`,
  `[frames, 16]`, decoded in chunks as the reference does. Writes the decoded
  waveform (`wave`, Float32 `[samples]`).
-/
import Examples.Parity.Common
import Tyr.Model.Qwen3TTS

namespace Examples.Parity.Qwen3TTSDecode

open Lean torch torch.qwen3tts

structure Request where
  codes : Array (Array Nat)
  deriving FromJson

def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let request : Request ← IO.ofExcept (fromJson? json)
  let ⟨dtypes, decoder⟩ ← SpeechTokenizer12HzDecoder.loadFromDir s!"{args.weights}/speech_tokenizer" device
  have : DType.RealFloating dtypes.compute.val := inferInstance
  let frames := request.codes.size
  if frames == 0 then throw <| IO.userError "no codes"
  if request.codes.any (·.size != 16) then
    throw <| IO.userError "every frame needs 16 codes"
  let flat := request.codes.flatMap (·.map Int64.ofNat)
  let codes := (← IO.ofExcept (Tensor.fromInt64ArrayAs? flat #[frames, 16])).to device
  let wave := decoder.decodeFrameMajorChunked codes
  pure {
    json := Json.mkObj []
    tensors := #[("wave", Tensor.toTensor ((wave.castDType .Float32).reshape #[frames * 1920]))]
  }

end Examples.Parity.Qwen3TTSDecode

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.Qwen3TTSDecode.probe
