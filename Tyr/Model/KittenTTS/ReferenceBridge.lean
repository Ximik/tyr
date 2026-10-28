/-
  Tyr/Model/KittenTTS/ReferenceBridge.lean

  Python Kokoro reference bridge for KittenTTS debugging/parity checks.
  This runs the upstream Python modules against the same converted
  `model.safetensors` + `voice.safetensors` assets used by the Lean port.
-/
import Tyr.Model.KittenTTS.Pretrained
import Lean.Data.Json

namespace torch.kittentts

open Lean

structure ReferenceBridgeConfig where
  pythonExe : String := "uv"
  synthScript : String := "scripts/kittentts_reference_synthesize.py"
  device : String := "cpu"
  repoId : String := hub.defaultRepoId
  disableComplex : Bool := false
  seed : Option UInt64 := none
  deriving Repr, Inhabited

structure ReferenceSynthesisResult where
  wavPath : String
  predDurations : Array Nat
  audioShape : Array Nat
  audioNumSamples : Nat
  sampleRate : Nat
  voiceIndex : Nat
  deriving Repr, Inhabited

private def expandHome (path : String) : IO String := do
  if path == "~" then
    return (← IO.getEnv "HOME").getD path
  else if path.startsWith "~/" then
    return s!"{(← IO.getEnv "HOME").getD ""}/{path.drop 2}"
  else
    return path

private def ensureParentDir (path : String) : IO Unit := do
  match System.FilePath.parent ⟨path⟩ with
  | some parent =>
      if parent.toString != "" && parent.toString != "." then
        IO.FS.createDirAll parent
  | none => pure ()

private def pythonPrefix (pythonExe : String) : Array String :=
  if pythonExe == "uv" || pythonExe.endsWith "/uv" then
    #["run", "--with", "kokoro", "--with", "safetensors", "python"]
  else
    #[]

private def parseNatArrayField (json : Json) (field : String) : IO (Array Nat) := do
  match (json.getObjValAs? (Array Nat) field).toOption with
  | some xs => pure xs
  | none => throw <| IO.userError s!"Missing or invalid JSON array field '{field}'"

private def parseNatField (json : Json) (field : String) : IO Nat := do
  match (json.getObjValAs? Nat field).toOption with
  | some n => pure n
  | none =>
    match (json.getObjValAs? Int field).toOption with
    | some i => pure i.toNat
    | none => throw <| IO.userError s!"Missing or invalid JSON field '{field}'"

private def parseReferenceSummary (stdout : String) (wavPath : String) : IO ReferenceSynthesisResult := do
  let lines := stdout.splitOn "\n"
  let jsonLine? := lines.reverse.find? (fun line => !((line.trimAscii.toString).isEmpty))
  let jsonLine ←
    match jsonLine? with
    | some line => pure line
    | none => throw <| IO.userError "Reference bridge produced no JSON summary."
  let json ←
    match Json.parse jsonLine with
    | .ok j => pure j
    | .error err =>
      throw <| IO.userError s!"Failed to parse reference bridge JSON: {err}\nstdout:\n{stdout}"
  pure {
    wavPath := wavPath
    predDurations := ← parseNatArrayField json "pred_durations"
    audioShape := ← parseNatArrayField json "audio_shape"
    audioNumSamples := ← parseNatField json "audio_num_samples"
    sampleRate := ← parseNatField json "sample_rate"
    voiceIndex := ← parseNatField json "voice_index"
  }

namespace PretrainedBundle

def synthesizePhonemesToReferenceWav
    (bundle : PretrainedBundle)
    (phonemes : String)
    (voice : String)
    (wavPath : String)
    (speed : Float := 1.0)
    (bridge : ReferenceBridgeConfig := {})
    : IO ReferenceSynthesisResult := do
  let pythonExe ← expandHome bridge.pythonExe
  let synthScript ← expandHome bridge.synthScript
  let wavPath ← expandHome wavPath
  let voicePath ← hub.ensureVoiceSafetensors bundle.source bundle.revision bundle.modelDir voice
  let configPath := s!"{bundle.modelDir}/config.json"
  let modelPath := s!"{bundle.modelDir}/model.safetensors"
  ensureParentDir wavPath
  let args :=
    pythonPrefix pythonExe ++ #[
      synthScript,
      "--config", configPath,
      "--model", modelPath,
      "--voice", voicePath,
      "--phonemes", phonemes,
      "--output-wav", wavPath,
      "--speed", toString speed,
      "--sample-rate", toString bundle.cfg.sampleRate,
      "--device", bridge.device,
      "--repo-id", bridge.repoId
    ] ++
    (match bridge.seed with
    | some seed => #["--seed", toString seed]
    | none => #[]) ++
    (if bridge.disableComplex then #["--disable-complex"] else #[])
  let result ← IO.Process.output { cmd := pythonExe, args := args }
  if result.exitCode != 0 then
    throw <| IO.userError
      s!"Kitten reference bridge failed (exit={result.exitCode}):\n{result.stderr}"
  parseReferenceSummary result.stdout wavPath

end PretrainedBundle

end torch.kittentts
