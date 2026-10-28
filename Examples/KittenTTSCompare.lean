import Tyr
import Tyr.Model.KittenTTS

open torch
open torch.kittentts

private def usage : String :=
  "Usage: lake exe KittenTTSCompare <model_dir_or_repo> <phonemes> <voice> <lean_wav> <python_wav> [speed] [seed]"

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

private def arraysMatch (a b : Array Nat) : Bool :=
  a.size == b.size && Id.run (do
    for h : i in [:a.size] do
      if a[i] != b[i]! then
        return false
    true)

def main (args : List String) : IO UInt32 := do
  let (source, phonemes, voice, leanWav, pyWav, speed, seed) ←
    match args with
    | [source, phonemes, voice, leanWav, pyWav] =>
      pure (source, phonemes, voice, leanWav, pyWav, 1.0, (0 : UInt64))
    | [source, phonemes, voice, leanWav, pyWav, speedStr] =>
      match parseFloatLit? speedStr with
      | some speed => pure (source, phonemes, voice, leanWav, pyWav, speed, (0 : UInt64))
      | none =>
        IO.eprintln s!"Invalid speed: {speedStr}"
        IO.eprintln usage
        pure ("", "", "", "", "", 1.0, (0 : UInt64))
    | [source, phonemes, voice, leanWav, pyWav, speedStr, seedStr] =>
      match parseFloatLit? speedStr, seedStr.toNat? with
      | some speed, some seed => pure (source, phonemes, voice, leanWav, pyWav, speed, seed.toUInt64)
      | none, _ =>
        IO.eprintln s!"Invalid speed: {speedStr}"
        IO.eprintln usage
        pure ("", "", "", "", "", 1.0, (0 : UInt64))
      | _, none =>
        IO.eprintln s!"Invalid seed: {seedStr}"
        IO.eprintln usage
        pure ("", "", "", "", "", 1.0, (0 : UInt64))
    | _ =>
      IO.eprintln usage
      pure ("", "", "", "", "", 1.0, (0 : UInt64))
  if source.isEmpty then
    return 1

  let log : torch.Log.Handlers := {
    onInfo := IO.println
    onWarn := fun msg => IO.eprintln s!"warning: {msg}"
    onError := fun msg => IO.eprintln s!"error: {msg}"
  }
  let bundle ← Model.loadFromPretrained source (log := log)
  torch.manualSeed seed
  let leanOut ← bundle.synthesizePhonemesToWav phonemes voice leanWav speed
  let refOut ← bundle.synthesizePhonemesToReferenceWav phonemes voice pyWav speed { seed := some seed }

  IO.println s!"seed={seed}"
  IO.println s!"lean_wav={leanWav}"
  IO.println s!"python_wav={refOut.wavPath}"
  IO.println s!"lean_alignment_frames={leanOut.alignmentFrames}"
  IO.println s!"python_alignment_frames={refOut.predDurations.foldl (· + ·) 0}"
  IO.println s!"lean_pred_durations={leanOut.predDurations}"
  IO.println s!"python_pred_durations={refOut.predDurations}"
  IO.println s!"durations_match={arraysMatch leanOut.predDurations refOut.predDurations}"
  IO.println s!"lean_audio_shape={leanOut.audio.1}"
  IO.println s!"python_audio_shape={refOut.audioShape}"
  IO.println s!"python_audio_num_samples={refOut.audioNumSamples}"
  IO.println s!"python_voice_index={refOut.voiceIndex}"
  pure 0
