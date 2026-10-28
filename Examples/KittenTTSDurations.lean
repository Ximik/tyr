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
    let seq := ids.size
    let modelDevice := bundle.model.bert.embeddings.wordEmbeddings.device
    let inputIdsCpu : Tensor { shape := #[1, seq], dtype := .Int64, device := .CPU } := Tensor.reshape (Tensor.fromInt64Array (ids.map (·.toInt64))) #[1, seq]
    let inputIds : Tensor { shape := #[1, seq], dtype := .Int64, device := modelDevice } := inputIdsCpu.to modelDevice
    let refStyleCpu ← bundle.loadVoiceStyle voice phonemes.toList.length
    let refStyle := refStyleCpu.to modelDevice
    IO.println s!"loaded_bundle=true"
    IO.println s!"input_ids={ids}"
    let (bertOut, _pooled) := bundle.model.bert.forward inputIds
    IO.println s!"bert_shape={bertOut.runtimeShape}"
    let dEnSeq : Tensor { shape := #[1, seq, bundle.cfg.hiddenDim], dtype := .Float32, device := modelDevice } := bundle.model.bertEncoder.forward3d bertOut
    IO.println s!"bert_encoder_shape={dEnSeq.runtimeShape}"
    let dEn : Tensor { shape := #[1, bundle.cfg.hiddenDim, seq], dtype := .Float32, device := modelDevice } := Tensor.reshape (Tensor.transpose dEnSeq 1 2) #[1, bundle.cfg.hiddenDim, seq]
    IO.println s!"d_en_cf_shape={dEn.runtimeShape}"
    let predictorStyle : Tensor { shape := #[1, bundle.cfg.styleDim], dtype := .Float32, device := modelDevice } :=
      sliceTensor refStyle 1 bundle.cfg.styleDim bundle.cfg.styleDim
    let durEnc : Tensor { shape := #[1, bundle.cfg.hiddenDim + bundle.cfg.styleDim, seq], dtype := .Float32, device := modelDevice } :=
      bundle.model.predictor.durationEncoding dEn predictorStyle
    IO.println s!"duration_encoder_shape={durEnc.runtimeShape}"
    let durLogits : Tensor { shape := #[1, seq, bundle.cfg.maxDur], dtype := .Float32, device := modelDevice } := bundle.model.predictor.durationLogits durEnc
    IO.println s!"duration_logits_shape={durLogits.runtimeShape}"
    let durSig : Tensor { shape := #[1, seq, bundle.cfg.maxDur], dtype := .Float32, device := modelDevice } := Tensor.sigmoid durLogits
    let durSum : Tensor { shape := #[1, seq], dtype := .Float32, device := modelDevice } := Tensor.reshape (Tensor.sumDim durSig 2 false) #[1, seq]
    IO.println s!"duration_sum_shape={durSum.runtimeShape}"
    let pred ← bundle.model.predictDurations inputIds refStyle speed
    IO.println s!"dur_vals={pred.durVals}"
    IO.println s!"pred_durations={pred.predDurations}"
    IO.println s!"alignment_frames={pred.alignmentFrames}"
    pure 0
  catch e =>
    IO.eprintln s!"{e.toString}"
    pure 1
