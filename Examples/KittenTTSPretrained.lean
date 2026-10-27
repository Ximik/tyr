import Tyr
import Tyr.Model.KittenTTS

open torch
open torch.kittentts

private def usage : String :=
  "Usage: lake exe KittenTTSPretrained <model_dir_or_repo> <phonemes> <voice> <out_audio> [speed]"

private def parseFloatLit? (s : String) : Option Float :=
  match s.splitOn "." with
  | [whole] =>
      whole.toNat?.map (·.toFloat)
  | [whole, frac] =>
      match whole.toNat?, frac.toNat? with
      | some w, some f =>
          let denom : Float := (Nat.pow 10 frac.length).toFloat
          some (w.toFloat + f.toFloat / denom)
      | _, _ => none
  | _ => none

def main (args : List String) : IO UInt32 := do
  let (source, phonemes, voice, outPath, speed) ←
    match args with
    | [source, phonemes, voice, outPath] =>
      pure (source, phonemes, voice, outPath, 1.0)
    | [source, phonemes, voice, outPath, speedStr] =>
      match parseFloatLit? speedStr with
      | some speed => pure (source, phonemes, voice, outPath, speed)
      | none =>
        IO.eprintln s!"Invalid speed: {speedStr}"
        IO.eprintln usage
        pure ("", "", "", "", 1.0)
    | _ =>
      IO.eprintln usage
      pure ("", "", "", "", 1.0)
  if source.isEmpty then
    return 1

  let log : torch.Log.Handlers := {
    onInfo := IO.println
    onWarn := fun msg => IO.eprintln s!"warning: {msg}"
    onError := fun msg => IO.eprintln s!"error: {msg}"
  }
  let bundle ← Model.loadFromPretrained source (log := log)
  let out ← bundle.synthesizePhonemesToFile phonemes voice outPath speed
  IO.println s!"saved_audio={outPath}"
  IO.println s!"alignment_frames={out.alignmentFrames}"
  IO.println s!"pred_durations={out.predDurations}"
  IO.println s!"audio_shape={out.audio.1}"
  pure 0
