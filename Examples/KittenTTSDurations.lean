import Tyr
import Tyr.Model.KittenTTS

open torch
open torch.kittentts

private def usage : String :=
  "Usage: lake exe KittenTTSDurations <model_dir_or_repo> <phonemes> <voice> [speed]"

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
  let (source, phonemes, voice, speed) ←
    match args with
    | [source, phonemes, voice] =>
      pure (source, phonemes, voice, 1.0)
    | [source, phonemes, voice, speedStr] =>
      match parseFloatLit? speedStr with
      | some speed => pure (source, phonemes, voice, speed)
      | none =>
        IO.eprintln s!"Invalid speed: {speedStr}"
        IO.eprintln usage
        pure ("", "", "", 1.0)
    | _ =>
      IO.eprintln usage
      pure ("", "", "", 1.0)
  if source.isEmpty then
    return 1

  let log : torch.Log.Handlers := {
    onInfo := IO.println
    onWarn := fun msg => IO.eprintln s!"warning: {msg}"
    onError := fun msg => IO.eprintln s!"error: {msg}"
  }
  try
    let bundle ← Model.loadFromPretrained source (log := log)
    let ids := bundle.inputIdsFromPhonemes phonemes
    let seq := ids.size.toUInt64
    let modelDevice := bundle.model.bert.embeddings.wordEmbeddings.device
    let inputIdsCpu : DTensor #[1, seq] .Int64 := Tensor.reshapeChecked (Tensor.fromInt64Array (ids.map (·.toInt64))) #[1, seq]
    let inputIds : DTensor #[1, seq] .Int64 := inputIdsCpu.to modelDevice
    let refStyleCpu ← bundle.loadVoiceStyle voice phonemes.toList.length.toUInt64
    let refStyle := refStyleCpu.to modelDevice
    IO.println s!"loaded_bundle=true"
    IO.println s!"input_ids={ids}"
    let (bertOut, _pooled) := bundle.model.bert.forward inputIds
    IO.println s!"bert_shape={bertOut.runtimeShape}"
    let dEnSeq : DTensor #[1, seq, bundle.cfg.hiddenDim] .Float32 := bundle.model.bertEncoder.forward3d bertOut
    IO.println s!"bert_encoder_shape={dEnSeq.runtimeShape}"
    let dEn : DTensor #[1, bundle.cfg.hiddenDim, seq] .Float32 := Tensor.reshapeChecked (Tensor.transpose dEnSeq 1 2) #[1, bundle.cfg.hiddenDim, seq]
    IO.println s!"d_en_cf_shape={dEn.runtimeShape}"
    let predictorStyle : DTensor #[1, bundle.cfg.styleDim] .Float32 :=
      sliceTensor refStyle 1 bundle.cfg.styleDim bundle.cfg.styleDim
    let durEnc : DTensor #[1, bundle.cfg.hiddenDim + bundle.cfg.styleDim, seq] .Float32 :=
      bundle.model.predictor.durationEncoding dEn predictorStyle
    IO.println s!"duration_encoder_shape={durEnc.runtimeShape}"
    let durLogits : DTensor #[1, seq, bundle.cfg.maxDur] .Float32 := bundle.model.predictor.durationLogits durEnc
    IO.println s!"duration_logits_shape={durLogits.runtimeShape}"
    let durSig : DTensor #[1, seq, bundle.cfg.maxDur] .Float32 := Tensor.sigmoid durLogits
    let durSum : DTensor #[1, seq] .Float32 := Tensor.reshapeChecked (Tensor.sumDim durSig 2 false) #[1, seq]
    IO.println s!"duration_sum_shape={durSum.runtimeShape}"
    let pred ← bundle.model.predictDurations inputIds refStyle speed
    IO.println s!"dur_vals={pred.durVals}"
    IO.println s!"pred_durations={pred.predDurations}"
    IO.println s!"alignment_frames={pred.alignmentFrames}"
    pure 0
  catch e =>
    IO.eprintln s!"{e.toString}"
    pure 1
