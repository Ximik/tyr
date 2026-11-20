/-
  Examples/Parity/Qwen3TTSEncode.lean

  Parity probe for `parity/tests/qwen3tts_encode.py`. `--weights` is the model
  directory; its `speech_tokenizer/` is read. Request:
  `{"audio": <24 kHz PCM16 WAV path>}`. Writes the codec codes
  (`codes`, `[frames, groups]`).
-/
import Examples.Parity.Common
import Tyr.Model.Qwen3TTS
import Tyr.Model.Qwen3ASR.Frontend

namespace Examples.Parity.Qwen3TTSEncode

open Lean torch torch.qwen3tts

structure Request where
  audio : String
  deriving FromJson

def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let request : Request ← IO.ofExcept (fromJson? json)
  let ⟨dtypes, encoder⟩ ← SpeechTokenizer12HzEncoder.loadFromDir s!"{args.weights}/speech_tokenizer" device
  have : DType.RealFloating dtypes.compute.val := inferInstance
  let (rate, wave) ← qwen3asr.loadMonoPcm16Wav request.audio
  unless rate == 24000 do
    throw <| IO.userError s!"{request.audio} is {rate} Hz; the speech tokenizer reads 24 kHz"
  let ⟨hs⟩ ← if h : 0 < wave.size then pure (PLift.up h) else throw <| IO.userError "empty audio"
  let ⟨hBooks⟩ ← if h : 1 ≤ encoder.semanticRVQ.codebooks.size ∧ 15 ≤ encoder.acousticRVQ.codebooks.size then
      pure (PLift.up h) else throw <| IO.userError "the encoder needs 1 semantic and 15 acoustic codebooks"
  let audio := ((Tensor.fromFloatArray wave).castDType dtypes.compute.val).to device
  let codes ← (encoder.encodeMonoFrameMajor audio hs hBooks).toUInt64Array
  let rows := (Array.range (codes.size / 16)).map fun f => (codes.extract (16 * f) (16 * f + 16)).map (·.toNat)
  pure { json := Json.mkObj [("codes", toJson rows)] }

end Examples.Parity.Qwen3TTSEncode

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.Qwen3TTSEncode.probe
