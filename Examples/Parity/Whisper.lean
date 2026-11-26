/-
  Examples/Parity/Whisper.lean

  Parity probe for `parity/tests/whisper.py`. Request:
  `{"dtype", "audio": <16 kHz PCM16 WAV path>, "prompt": [token], "max_new_tokens", "forced": [token]}`.
  Writes the greedy tokens after the decoder prompt (`tokens`, `[[token]]`) and
  the Float32 tensors `mel` (`[1, mels, frames]`), `encoder` (`[1, encSeq, dModel]`)
  and `logits` (`[1, steps, vocab]`) of feeding `forced` after the prompt.
-/
import Examples.Parity.Common
import Tyr.Core.Typed.Neural
import Tyr.Model.Whisper.ConfigIO
import Tyr.Model.Whisper.Weights
import Tyr.Lib.Audio.LogMel
import Tyr.Lib.Audio.Waveform

namespace Examples.Parity.Whisper

open Lean torch torch.whisper torch.whisper.WhisperForConditionalGeneration

structure Request where
  dtype : String
  audio : String
  prompt : Array Nat
  max_new_tokens : Nat
  forced : Array Nat
  deriving FromJson

variable {device : Device} {dtype : DType} {cfg : WhisperConfig} {encSeq : Nat}

/-- Feed `pending` through the decoder; after the last pending token, record the
logits and take `choose index logits` as the next token, `steps` times. -/
private partial def run [DType.RealFloating dtype] {cap len : Nat}
    (m : WhisperForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (cross : Array (LayerCrossCache (device := device) (dtype := dtype) cfg 1 encSeq))
    (hcap : cap ≤ cfg.maxTargetPositions)
    (caches : SelfKVCaches (device := device) (dtype := dtype) cfg 1 cap len)
    (pending : List Nat) (steps : Nat)
    (choose : Nat → Tensor { dtype := dtype, device := device, shape := #[1, cfg.vocabSize] } → IO Nat)
    (tokens : Array Nat)
    (logits : Array (Tensor { dtype := dtype, device := device, shape := #[1, 1, cfg.vocabSize] }))
    : IO (Array Nat × Array (Tensor { dtype := dtype, device := device, shape := #[1, 1, cfg.vocabSize] })) := do
  match pending with
  | [] => pure (tokens, logits)
  | token :: rest =>
    let ⟨hlen⟩ ← if h : len < cap then pure (PLift.up h) else
      throw <| IO.userError s!"decoder cache of {cap} positions is full"
    let ids : Tensor { device := device, shape := #[1, 1], dtype := .Int64 } :=
      (Tensor.fromInt64ArrayShaped #[Int64.ofNat token] #[1, 1] (by simp)).to device
    let (step, caches') ← m.decodeStepWithCache ids cross caches hlen hcap
    if !rest.isEmpty then
      run m cross hcap caches' rest steps choose tokens logits
    else if steps == 0 then
      pure (tokens, logits)
    else
      let next ← choose tokens.size step
      let tokens := tokens.push next
      let logits := logits.push (step.reshape #[1, 1, cfg.vocabSize])
      if steps == 1 then pure (tokens, logits)
      else run m cross hcap caches' [next] (steps - 1) choose tokens logits

private def argmaxToken {vocab : Nat}
    (logits : Tensor { dtype := dtype, device := device, shape := #[1, vocab] }) : IO Nat := do
  let ids ← (Tensor.argmax! (Tensor.castDType logits .Float32) 1).toUInt64Array
  match ids[0]? with
  | some id => pure id.toNat
  | none => throw <| IO.userError "argmax of an empty row"

def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let request : Request ← IO.ofExcept (fromJson? json)
  let weights := args.weights.toString
  let cfg ← WhisperConfig.loadFromPretrainedDir weights {}
  let pre ← audio.LogMelConfig.loadFromPretrainedDir weights (requiredSamplingRate := some 16000) {
    featureSize := cfg.numMelBins
    featureSizePos := cfg.numMelBinsPos
    hopLength := 160
    chunkLength := 30
    nFft := 400
    returnAttentionMask := true
    doNormalize := false
    dither := 0.0
  }
  let ⟨hMel⟩ ← if h : pre.featureSize = cfg.numMelBins then pure (PLift.up h) else
    throw <| IO.userError s!"feature_size {pre.featureSize} is not num_mel_bins {cfg.numMelBins}"
  let ⟨dtype, model⟩ ← WhisperForConditionalGeneration.loadSharded weights cfg device
  unless DType.ofString? request.dtype == some dtype.val do
    throw <| IO.userError s!"the checkpoint loads as {dtype.val}; the request asks for '{request.dtype}'"
  have : DType.RealFloating dtype.val := inferInstance
  let support ← dtype.val.acquireConstantSupport device
  let _ := support.down

  let features ← audio.logMelFeatures pre (← audio.normalizeAudioTo16kFromWav request.audio)
  let frames := audio.LogMelConfig.expectedFrames pre
  let mel : Tensor { dtype := .Float32, device := device, shape := #[1, cfg.numMelBins, frames] } :=
    (Tensor.reshape features.inputFeatures #[1, cfg.numMelBins, frames] (by simp [hMel, frames])).to device
  let encoder ← model.encode (frames := frames) mel features.framesPos
  let cross ← model.precomputeCrossCaches encoder

  let cap := request.prompt.size + request.max_new_tokens
  let ⟨hcap⟩ ← if h : cap ≤ cfg.maxTargetPositions then pure (PLift.up h) else
    throw <| IO.userError s!"{cap} decoder positions exceed max_target_positions {cfg.maxTargetPositions}"
  let ⟨hpos⟩ ← if h : 0 < cap then pure (PLift.up h) else
    throw <| IO.userError "empty decoder prompt"
  let fresh := model.initLayerKVCaches (batch := 1) cap device hpos
  let (tokens, _) ← run model cross hcap fresh request.prompt.toList request.max_new_tokens
    (fun _ logits => argmaxToken logits) #[] #[]
  let steps := request.forced.size
  let (_, forcedLogits) ← run model cross hcap fresh request.prompt.toList steps
    (fun index _ => pure (request.forced.getD index 0)) #[] #[]
  let logits ← IO.ofExcept <|
    Tensor.catMany? (forcedLogits.map Tensor.packShape) 1 #[1, steps, cfg.vocabSize]
  pure {
    json := Json.mkObj [("tokens", toJson #[tokens])]
    tensors := #[
      ("mel", Tensor.toTensor mel),
      ("encoder", Tensor.toTensor (Tensor.castDType encoder .Float32)),
      ("logits", Tensor.toTensor (Tensor.castDType logits .Float32))]
  }

end Examples.Parity.Whisper

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.Whisper.probe
