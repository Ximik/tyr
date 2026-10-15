/-
  Tyr/Model/Qwen3ASR/Pretrained.lean

  Resolve/load Qwen3-ASR checkpoints from either:
  - local model directory, or
  - HuggingFace repo id (download to local cache on demand).
-/
import Tyr.Hub
import Tyr.Model.Qwen3ASR.ConfigIO
import Tyr.Model.Qwen3ASR.PreprocessorConfig
import Tyr.Model.Qwen3ASR.Weights

namespace torch.qwen3asr.hub

variable {device : Device} {dtype : DType}
open torch.Hub

structure DownloadOptions where
  revision : String := "main"
  cacheDir : String := Hub.defaultCacheDir
  includeTokenizer : Bool := true
  includePreprocessor : Bool := true
  deriving Repr, Inhabited

/-- Official Qwen3-ASR model ids in HF collection `Qwen/Qwen3-ASR` as of 2026-02-28. -/
def qwen3ASRCollectionRepoIds : Array String := #[
  "Qwen/Qwen3-ASR-0.6B",
  "Qwen/Qwen3-ASR-1.7B"
]

def isQwen3ASRCollectionRepoId (repoId : String) : Bool :=
  qwen3ASRCollectionRepoIds.contains repoId

/-- Resolve a source into a local model directory.
    - If `source` is an existing local directory, return it.
    - Otherwise treat it as HF `repo_id` and resolve/download locally. -/
def resolvePretrainedDir (source : String) (opts : DownloadOptions := {}) : IO String := do
  let sourceExpanded ← expandHome source
  if ← dirExists sourceExpanded then
    return sourceExpanded

  if let some snap ← findCachedSnapshot? source opts.revision then
    let hasTok ←
      if opts.includeTokenizer then
        let tokJson ← fileExists s!"{snap}/tokenizer.json"
        let tokCfg ← fileExists s!"{snap}/tokenizer_config.json"
        let vocab ← fileExists s!"{snap}/vocab.json"
        let merges ← fileExists s!"{snap}/merges.txt"
        pure <| tokJson || (tokCfg && vocab && merges)
      else
        pure true
    let hasPre ←
      if opts.includePreprocessor then
        fileExists s!"{snap}/preprocessor_config.json"
      else
        pure true
    if hasTok && hasPre then
      ensureModelWeights source opts.revision snap
      return snap

  let modelDir ← modelDirForRepo opts.cacheDir source opts.revision
  IO.FS.createDirAll ⟨modelDir⟩
  ensureRemoteFile source opts.revision "config.json" s!"{modelDir}/config.json"
  ensureModelWeights source opts.revision modelDir
  if opts.includeTokenizer then
    let _ ← tryRemoteFile source opts.revision "tokenizer.json" s!"{modelDir}/tokenizer.json"
    let _ ← tryRemoteFile source opts.revision "tokenizer_config.json" s!"{modelDir}/tokenizer_config.json"
    let _ ← tryRemoteFile source opts.revision "vocab.json" s!"{modelDir}/vocab.json"
    let _ ← tryRemoteFile source opts.revision "merges.txt" s!"{modelDir}/merges.txt"
    pure ()
  if opts.includePreprocessor then
    let _ ← tryRemoteFile source opts.revision "preprocessor_config.json" s!"{modelDir}/preprocessor_config.json"
    pure ()
  pure modelDir

export torch.Hub (detectWeightLayout)

end torch.qwen3asr.hub

namespace torch.qwen3asr

namespace Qwen3ASRForConditionalGeneration

/-- Load Qwen3-ASR from a local dir or HF repo id. -/
def loadFromPretrained
    (source : String)
    (defaults : Qwen3ASRConfig := {})
    (revision : String := "main")
    (cacheDir : String := Hub.defaultCacheDir)
    : IO (Σ cfg, Σ device : Device, Σ dtype : ExactDType, Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype.val) cfg) := do
  let modelDir ← torch.Hub.resolvePretrainedDir source {
    revision := revision
    cacheDir := cacheDir
    includeTokenizer := false
  }
  let cfg ← Qwen3ASRConfig.loadFromPretrainedDir modelDir defaults
  let m ← Qwen3ASRForConditionalGeneration.loadSharded modelDir cfg
  pure ⟨cfg, m⟩

end Qwen3ASRForConditionalGeneration

end torch.qwen3asr
