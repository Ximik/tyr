import Tyr.SafeTensors.Schema
/-
  Tyr/Model/KittenTTS/Pretrained.lean

  Resolve/load KittenTTS-compatible Kokoro checkpoints from either:
  - a local model directory, or
  - a HuggingFace repo id (download + convert on demand).
-/
import Tyr.Hub
import Tyr.Model.KittenTTS.ConfigIO
import Tyr.Model.KittenTTS.Weights
import Tyr.Log

namespace torch.kittentts.hub

open torch.Hub

def defaultRepoId : String := "hexgrad/Kokoro-82M"

private def repoModelFilename (repoId : String) : String :=
  if repoId == "hexgrad/Kokoro-82M-v1.1-zh" then
    "kokoro-v1_1-zh.pth"
  else
    "kokoro-v1_0.pth"

private def findRawModelFile? (modelDir : String) : IO (Option String) := do
  let candidates :=
    #[
      s!"{modelDir}/kokoro-v1_0.pth",
      s!"{modelDir}/kokoro-v1_1-zh.pth"
    ]
  for path in candidates do
    if ← fileExists path then
      return some path
  let dir : System.FilePath := ⟨modelDir⟩
  if !(← dir.pathExists) then
    return none
  let entries ← dir.readDir
  for entry in entries do
    if entry.path.extension == some "pth" then
      return some entry.path.toString
  pure none

private partial def findRepoRootFrom (path : System.FilePath) : IO (Option System.FilePath) := do
  if ← (path / "lakefile.lean").pathExists then
    return some path
  match path.parent with
  | some parent =>
    if parent == path then
      pure none
    else
      findRepoRootFrom parent
  | none => pure none

private def converterScriptPath : IO String := do
  let cwd ← IO.currentDir
  match (← findRepoRootFrom cwd) with
  | some root =>
    let script := root / "scripts" / "kokoro_to_safetensors.py"
    if ← script.pathExists then
      pure script.toString
    else
      throw <| IO.userError s!"Missing converter script at {script}"
  | none =>
    throw <| IO.userError "Could not locate Tyr repo root from current working directory."

private def runProcessChecked (cmd : String) (args : Array String) : IO Unit := do
  let out ← IO.Process.output { cmd, args }
  if out.exitCode != 0 then
    throw <| IO.userError s!"Command failed: {cmd} {String.intercalate " " args.toList}\n{out.stderr}"

private def ensureConvertedCheckpoint (modelDir rawPath : String) : IO Unit := do
  let safetensorsPath := s!"{modelDir}/model.safetensors"
  if ← fileExists safetensorsPath then
    pure ()
  else
    let script ← converterScriptPath
    runProcessChecked "uv"
      #[
        "run", "--with", "torch", "--with", "safetensors",
        "python", script,
        "--checkpoint", rawPath,
        "--checkpoint-out", safetensorsPath
      ]

private def ensureConvertedVoice (voicePtPath : String) : IO String := do
  let voiceStPath :=
    if voicePtPath.endsWith ".pt" then
      (voicePtPath.dropEnd 3).toString ++ ".safetensors"
    else
      voicePtPath ++ ".safetensors"
  if ← fileExists voiceStPath then
    pure voiceStPath
  else
    let script ← converterScriptPath
    runProcessChecked "uv"
      #[
        "run", "--with", "torch", "--with", "safetensors",
        "python", script,
        "--voice", voicePtPath,
        "--voice-out", voiceStPath
      ]
    pure voiceStPath

/-- Resolve a source into a local model directory and ensure `model.safetensors` exists. -/
def resolvePretrainedDir (source : String) (opts : Hub.DownloadOptions := {}) : IO String := do
  let sourceExpanded ← expandHome source
  if ← dirExists sourceExpanded then
    if !(← fileExists s!"{sourceExpanded}/config.json") then
      throw <| IO.userError s!"Missing config.json in {sourceExpanded}"
    if !(← fileExists s!"{sourceExpanded}/model.safetensors") then
      if let some rawModel ← findRawModelFile? sourceExpanded then
        ensureConvertedCheckpoint sourceExpanded rawModel
      else
        throw <| IO.userError s!"No model.safetensors or raw .pth checkpoint found in {sourceExpanded}"
    pure sourceExpanded
  else
    try
      Hub.resolvePretrainedDir source { opts with includeTokenizer := false }
    catch _ =>
      let modelDir ← modelDirForRepo opts.cacheDir source opts.revision
      IO.FS.createDirAll ⟨modelDir⟩
      if !(← fileExists s!"{modelDir}/config.json") then
        ensureRemoteFile source opts.revision "config.json" s!"{modelDir}/config.json"
      if !(← fileExists s!"{modelDir}/model.safetensors") then
        if let some rawModel ← findRawModelFile? modelDir then
          ensureConvertedCheckpoint modelDir rawModel
        else
          let rawRel := repoModelFilename source
          let rawPath := s!"{modelDir}/{rawRel}"
          ensureRemoteFile source opts.revision rawRel rawPath
          ensureConvertedCheckpoint modelDir rawPath
      pure modelDir

def ensureVoiceSafetensors (source revision modelDir voice : String) : IO String := do
  let normalized ←
    if voice.endsWith ".safetensors" || voice.endsWith ".pt" then
      pure voice
    else
      do
      let stRel := s!"voices/{voice}.safetensors"
      let ptRel := s!"voices/{voice}.pt"
      let stLocal := s!"{modelDir}/{stRel}"
      pure <| if (← fileExists stLocal) then stRel else ptRel
  let localPath :=
    if normalized.startsWith "/" || normalized.startsWith "~" then
      normalized
    else
      s!"{modelDir}/{normalized}"
  let localPath ← expandHome localPath
  if normalized.endsWith ".safetensors" then
    if ← fileExists localPath then
      pure localPath
    else
      throw <| IO.userError s!"Missing voice safetensors file at {localPath}"
  else
    if !(← fileExists localPath) then
      if source == modelDir then
        throw <| IO.userError s!"Missing voicepack file at {localPath}"
      else
        ensureRemoteFile source revision normalized localPath
    ensureConvertedVoice localPath

end torch.kittentts.hub

namespace torch.kittentts

open torch.Log
open torch.Hub

structure PretrainedBundle where
  cfg : KittenTTSConfig
  model : Model cfg
  vocab : VocabMap
  modelDir : String
  source : String
  revision : String
  cacheDir : String

private def encodePhonemeIds (vocab : VocabMap) (phonemes : String) : Array UInt64 :=
  Id.run do
    let mut ids : Array UInt64 := #[0]
    for c in phonemes.toList do
      match vocab[c]? with
      | some id => ids := ids.push id
      | none => pure ()
    ids := ids.push 0
    ids

private def voiceIndexOfPhonemeCount (phonemeCount : UInt64) : UInt64 :=
  min 509 <| if phonemeCount == 0 then 0 else phonemeCount - 1

namespace PretrainedBundle

def inputIdsFromPhonemes (bundle : PretrainedBundle) (phonemes : String) : Array UInt64 :=
  encodePhonemeIds bundle.vocab phonemes

def loadVoiceStyle (bundle : PretrainedBundle) (voice : String) (phonemeCount : UInt64)
    : IO (DTensor #[1, KittenTTSConfig.fullStyleDim bundle.cfg] .Float32) := do
  let voicePath ← hub.ensureVoiceSafetensors bundle.source bundle.revision bundle.modelDir voice
  let table : DTensor #[510, 1, KittenTTSConfig.fullStyleDim bundle.cfg] .Float32 ←
    safetensors.loadTensorWithSpec voicePath "voice" ⟨#[510, 1, KittenTTSConfig.fullStyleDim bundle.cfg], .Float32⟩
  let idx := voiceIndexOfPhonemeCount phonemeCount
  let style : DTensor #[1, 1, KittenTTSConfig.fullStyleDim bundle.cfg] .Float32 := sliceTensor table 0 idx 1
  pure (Tensor.reshapeChecked style #[1, KittenTTSConfig.fullStyleDim bundle.cfg])

def synthesizePhonemes (bundle : PretrainedBundle) (phonemes : String) (voice : String) (speed : Float := 1.0)
    : IO KittenTTSOutput := do
  let ids := bundle.inputIdsFromPhonemes phonemes
  let seq := ids.size.toUInt64
  let inputIds : DTensor #[1, seq] .Int64 := Tensor.reshapeChecked (Tensor.fromInt64Array (ids.map (·.toInt64))) #[1, seq]
  let refStyle ← bundle.loadVoiceStyle voice phonemes.toList.length.toUInt64
  bundle.model.synthesizeIds inputIds refStyle speed

def synthesizePhonemesToFile
    (bundle : PretrainedBundle)
    (phonemes : String)
    (voice : String)
    (outPath : String)
    (speed : Float := 1.0)
    : IO KittenTTSOutput := do
  let out ← bundle.synthesizePhonemes phonemes voice speed
  let outPath ← expandHome outPath
  ensureParentDir outPath
  if outPath.endsWith ".safetensors" then
    Tensor.saveSafeTensor out.audio.2 outPath "audio"
  else
    Tensor.saveWav out.audio.2 outPath bundle.cfg.sampleRate
  pure out

def synthesizePhonemesToWav
    (bundle : PretrainedBundle)
    (phonemes : String)
    (voice : String)
    (wavPath : String)
    (speed : Float := 1.0)
    : IO KittenTTSOutput :=
  synthesizePhonemesToFile bundle phonemes voice wavPath speed

def predictPhonemeDurations (bundle : PretrainedBundle) (phonemes : String) (voice : String) (speed : Float := 1.0)
    : IO KittenTTSDurationPrediction := do
  let ids := bundle.inputIdsFromPhonemes phonemes
  let seq := ids.size.toUInt64
  let inputIds : DTensor #[1, seq] .Int64 := Tensor.reshapeChecked (Tensor.fromInt64Array (ids.map (·.toInt64))) #[1, seq]
  let refStyle ← bundle.loadVoiceStyle voice phonemes.toList.length.toUInt64
  bundle.model.predictDurations inputIds refStyle speed

def debugPhonemes (bundle : PretrainedBundle) (phonemes : String) (voice : String) (speed : Float := 1.0)
    : IO KittenTTSDebugOutput := do
  let ids := bundle.inputIdsFromPhonemes phonemes
  let seq := ids.size.toUInt64
  let inputIds : DTensor #[1, seq] .Int64 := Tensor.reshapeChecked (Tensor.fromInt64Array (ids.map (·.toInt64))) #[1, seq]
  let refStyle ← bundle.loadVoiceStyle voice phonemes.toList.length.toUInt64
  bundle.model.debugSynthesizeIds inputIds refStyle speed

end PretrainedBundle

namespace Model

def loadFromPretrained
    (source : String := hub.defaultRepoId)
    (defaults : KittenTTSConfig := {})
    (revision : String := "main")
    (cacheDir : String := Hub.defaultCacheDir)
    (log : Handlers := default)
    : IO PretrainedBundle := do
  let modelDir ← hub.resolvePretrainedDir source { revision := revision, cacheDir := cacheDir }
  let cfg0 ← KittenTTSConfig.loadFromPretrainedDir modelDir defaults
  let vocab ← loadVocabFromPretrainedDir modelDir
  let loaded ← Model.loadAutoConfig s!"{modelDir}/model.safetensors" cfg0 (log := log)
  let cfg := loaded.cfg
  let model := loaded.model
  pure { cfg, model, vocab, modelDir, source, revision, cacheDir }

end Model

end torch.kittentts
