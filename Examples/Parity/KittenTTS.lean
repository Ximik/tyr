/-
  Examples/Parity/KittenTTS.lean

  Parity probe for `parity/tests/kittentts.py`. `--weights` is a Kokoro
  directory with `config.json` and `model.safetensors`. Request:
  `{"phonemes", "voice": <voice safetensors path>, "speed", "seed"}`; the
  generator's noise is drawn after `torch.manual_seed(seed)`. Writes each
  phoneme's duration in frames (`durations`, with the boundary tokens) and the
  Float32 tensors `f0` and `n` (`[1, 1, 2 * frames]`) and `wave` (`[1, 1, samples]`).
-/
import Examples.Parity.Common
import Tyr.Model.KittenTTS

namespace Examples.Parity.KittenTTS

open Lean torch torch.kittentts

structure Request where
  phonemes : String
  voice : String
  speed : Float
  seed : Nat
  deriving FromJson

def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let request : Request ← IO.ofExcept (fromJson? json)
  let dir := args.weights.toString
  let cfg ← KittenTTSConfig.loadFromPretrainedDir dir
  let loaded ← Model.loadAutoConfig s!"{dir}/model.safetensors" device cfg
  let vocab ← loadVocabFromPretrainedDir dir loaded.cfg.nToken
  let bundle : PretrainedBundle := {
    cfg := loaded.cfg, device := loaded.device, model := loaded.model, vocab, modelDir := dir,
    source := dir, revision := "", cacheDir := "" }
  let ids := bundle.inputIdsFromPhonemes request.phonemes
  let inputIds := (Tensor.reshape (Tensor.fromInt64Array (ids.map (·.toInt64))) #[1, ids.size]).to bundle.device
  let style ← bundle.loadVoiceStyle request.voice request.phonemes.length
  torch.manualSeed request.seed.toUInt64
  let out ← bundle.model.debugSynthesizeIds inputIds style request.speed
  pure {
    json := Json.mkObj [("durations", toJson out.predDurations)]
    tensors := #[
      ("f0", Tensor.toTensor out.f0Curve),
      ("n", Tensor.toTensor out.nCurve),
      ("wave", Tensor.toTensor out.audio.2)]
  }

end Examples.Parity.KittenTTS

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.KittenTTS.probe
