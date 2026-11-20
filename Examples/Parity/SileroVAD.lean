/-
  Examples/Parity/SileroVAD.lean

  Parity probe for `parity/tests/silerovad.py`. `--weights` is the safetensors
  file. Request: `{"audio": <16 kHz PCM16 WAV path>}`. Writes the speech segments
  in samples (`segments`, `[[start, end]]`) and the speech probability of each
  512-sample window (`probs`, Float32 `[windows]`).
-/
import Examples.Parity.Common
import Tyr.Model.SileroVAD

namespace Examples.Parity.SileroVAD

open Lean torch torch.silerovad

structure Request where
  audio : String
  deriving FromJson

def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let request : Request ← IO.ofExcept (fromJson? json)
  let model ← SileroVAD.load args.weights.toString device
  let audio ← readAudio request.audio
  let (probs, _) ← (SileroVADRuntime.init model).audioForward audio
  let (segments, _) ← getSpeechTimestamps audio (SileroVADRuntime.init model)
  pure {
    json := Json.mkObj [("segments", toJson (segments.map fun s => #[s.start, s.endPos]))]
    tensors := #[("probs", Tensor.toTensor (Tensor.fromFloatArray probs))]
  }

end Examples.Parity.SileroVAD

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.SileroVAD.probe
