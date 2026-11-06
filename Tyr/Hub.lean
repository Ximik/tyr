/-
  Tyr/Hub.lean

  Generic HuggingFace Hub utilities for downloading, caching, and resolving
  pretrained model directories. Used by all `Tyr.Model.*.Pretrained` modules.
-/
import Lean.Data.Json
import Tyr.SafeTensors.File

namespace torch.Hub

open Lean

def defaultCacheDir : String := "~/.cache/huggingface/tyr-models"

/-! ## Configuration -/

structure DownloadOptions where
  revision : String := "main"
  cacheDir : String := defaultCacheDir
  includeTokenizer : Bool := true
  deriving Repr, Inhabited

/-! ## File system helpers -/

def expandHome (path : String) : IO String := do
  if path.startsWith "~" then
    match (← IO.getEnv "HOME") with
    | some home => pure (path.replace "~" home)
    | none => pure path
  else
    pure path

def ensureParentDir (path : String) : IO Unit := do
  let p : System.FilePath := ⟨path⟩
  match p.parent with
  | some parent => IO.FS.createDirAll parent
  | none => pure ()

def fileExists (path : String) : IO Bool := do
  let p : System.FilePath := ⟨path⟩
  p.pathExists

def dirExists (path : String) : IO Bool := do
  let p : System.FilePath := ⟨path⟩
  if !(← p.pathExists) then
    pure false
  else
    p.isDir

/-! ## Shard parsing -/

/-- The unique shard files named by a `model.safetensors.index.json`. -/
def shardFilesFromIndexFile (indexPath : String) : IO (Array String) := do
  let mut out : Array String := #[]
  for (_, shard) in ← safetensors.readWeightMap indexPath do
    unless out.contains shard do
      out := out.push shard
  pure out

/-! ## Download -/

private def downloadFile (url : String) (dest : String) : IO Bool := do
  if ← fileExists dest then
    return true

  ensureParentDir dest
  let tmp := s!"{dest}.tmp"
  let resumeArgs :=
    if ← fileExists tmp then
      #["-C", "-"]
    else
      #[]
  let token? ← IO.getEnv "HF_TOKEN"
  let args :=
    match token? with
    | some tok =>
      #[
        "-fL", "--retry", "3", "--retry-delay", "1",
        "-H", s!"Authorization: Bearer {tok}"
      ] ++ resumeArgs ++ #["-o", tmp, url]
    | none =>
      #[
        "-fL", "--retry", "3", "--retry-delay", "1"
      ] ++ resumeArgs ++ #["-o", tmp, url]

  let out ← IO.Process.output { cmd := "curl", args := args }
  if out.exitCode == 0 then
    IO.FS.rename ⟨tmp⟩ ⟨dest⟩
    pure true
  else
    pure false

def ensureRemoteFile (repoId revision relPath dest : String) : IO Unit := do
  if ← fileExists dest then
    pure ()
  else
    let url := s!"https://huggingface.co/{repoId}/resolve/{revision}/{relPath}"
    let ok ← downloadFile url dest
    if !ok then
      throw <| IO.userError s!"Failed to download {repoId}:{revision}:{relPath}"

def tryRemoteFile (repoId revision relPath dest : String) : IO Bool := do
  if ← fileExists dest then
    pure true
  else
    let url := s!"https://huggingface.co/{repoId}/resolve/{revision}/{relPath}"
    downloadFile url dest

/-! ## Cache layout -/

private def sanitizeRepoId (repoId : String) : String :=
  repoId.replace "/" "__"

def modelDirForRepo (cacheDir repoId revision : String) : IO String := do
  let cacheDir ← expandHome cacheDir
  pure s!"{cacheDir}/{sanitizeRepoId repoId}/{revision}"

def maybeReadRef (repoDir : System.FilePath) (revision : String) : IO (Option String) := do
  let refPath := repoDir / "refs" / revision
  if ← refPath.pathExists then
    let ref := (← IO.FS.readFile refPath).trimAscii.toString
    if ref.isEmpty then pure none else pure (some ref)
  else
    pure none

def defaultHFHubDir : IO String := do
  match (← IO.getEnv "HF_HOME") with
  | some home => pure s!"{home}/hub"
  | none => pure "~/.cache/huggingface/hub"

def hfRepoDirName (repoId : String) : String :=
  s!"models--{repoId.replace "/" "--"}"

/-- Check if a directory contains model weight files. -/
def hasWeightFiles (dir : String) : IO Bool := do
  let sharded := s!"{dir}/model.safetensors.index.json"
  let single := s!"{dir}/model.safetensors"
  pure ((← fileExists sharded) || (← fileExists single))

/-- Try resolving a repo id against existing HuggingFace cache snapshots. -/
def findCachedSnapshot? (repoId : String) (revision : String := "main") : IO (Option String) := do
  let hubDir ← expandHome (← defaultHFHubDir)
  let repoDir : System.FilePath := ⟨s!"{hubDir}/{hfRepoDirName repoId}"⟩
  if !(← repoDir.pathExists) then
    return none

  let snapshotsDir := repoDir / "snapshots"
  if !(← snapshotsDir.pathExists) then
    return none

  let mut candidates : Array System.FilePath := #[]
  let directRevPath := snapshotsDir / revision
  if ← directRevPath.pathExists then
    candidates := candidates.push directRevPath

  if let some resolvedRev ← maybeReadRef repoDir revision then
    let resolvedPath := snapshotsDir / resolvedRev
    if (← resolvedPath.pathExists) && !candidates.contains resolvedPath then
      candidates := candidates.push resolvedPath

  let entries ← snapshotsDir.readDir
  for entry in entries do
    if (← entry.path.isDir) && !candidates.contains entry.path then
      candidates := candidates.push entry.path

  for c in candidates do
    let cStr := c.toString
    if (← fileExists s!"{cStr}/config.json") && (← hasWeightFiles cStr) then
      return some cStr
  pure none

/-! ## Model-specific helpers (parameterized) -/

/-- Download tokenizer files given a list of relative filenames. -/
def ensureTokenizerFiles (repoId revision modelDir : String) (files : Array String) : IO Unit := do
  for file in files do
    let _ ← tryRemoteFile repoId revision file s!"{modelDir}/{file}"
  pure ()

/-- Download model weights (sharded or single-file). -/
def ensureModelWeights (repoId revision modelDir : String) : IO Unit := do
  let indexPath := s!"{modelDir}/model.safetensors.index.json"
  let singlePath := s!"{modelDir}/model.safetensors"

  if ← fileExists indexPath then
    let shardFiles ← shardFilesFromIndexFile indexPath
    for shard in shardFiles do
      ensureRemoteFile repoId revision shard s!"{modelDir}/{shard}"
  else if ← fileExists singlePath then
    pure ()
  else
    let gotIndex ← tryRemoteFile repoId revision "model.safetensors.index.json" indexPath
    if gotIndex then
      let shardFiles ← shardFilesFromIndexFile indexPath
      for shard in shardFiles do
        ensureRemoteFile repoId revision shard s!"{modelDir}/{shard}"
    else
      ensureRemoteFile repoId revision "model.safetensors" singlePath

/-- Resolve a source into a local model directory.
    - If `source` is an existing local directory, return it.
    - Otherwise treat it as HF `repo_id` and resolve/download locally. -/
def resolvePretrainedDir (source : String) (opts : DownloadOptions := {}) (tokenizerFiles : Array String := #[]) : IO String := do
  let sourceExpanded ← expandHome source
  if ← dirExists sourceExpanded then
    return sourceExpanded

  if let some snap ← findCachedSnapshot? source opts.revision then
    return snap

  let modelDir ← modelDirForRepo opts.cacheDir source opts.revision
  IO.FS.createDirAll ⟨modelDir⟩
  ensureRemoteFile source opts.revision "config.json" s!"{modelDir}/config.json"
  ensureModelWeights source opts.revision modelDir
  if opts.includeTokenizer then
    ensureTokenizerFiles source opts.revision modelDir tokenizerFiles
  pure modelDir

/-- Generic helper to load a model from a local directory or HF Hub repo.
    - Resolves/downloads the model directory
    - Loads config
    - Loads the directory via the provided model-specific function
    Returns the resolved config and model as a dependent pair. -/
def loadModelFromPretrained
    {Cfg : Type}
    {Model : Cfg → Type}
    (source : String)
    (revision : String)
    (cacheDir : String)
    (tokenizerFiles : Array String)
    (loadConfig : String → Cfg → IO Cfg)
    (defaults : Cfg)
    (load : String → (cfg : Cfg) → IO (Model cfg))
    : IO (Sigma (fun cfg => Model cfg)) := do
  let modelDir ← resolvePretrainedDir source {
    revision := revision
    cacheDir := cacheDir
    includeTokenizer := true
  } tokenizerFiles
  let cfg ← loadConfig modelDir defaults
  pure ⟨cfg, ← load modelDir cfg⟩

end torch.Hub
