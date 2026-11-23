/-
  Tyr/Data/Download.lean

  Download utilities with retry and caching for data files.

  Based on nanochat's download patterns:
  - Exponential backoff retry for unreliable connections
  - File caching to avoid re-downloads
  - ZIP extraction for evaluation bundles
  - Progress tracking for long downloads
-/
import Tyr.Log

namespace torch.Data.Download

open torch.Log

/-! ## Path Utilities -/

/-- Expand ~ to home directory -/
def expandHome (path : String) : IO String := do
  if path.startsWith "~" then
    let home ← IO.getEnv "HOME"
    match home with
    | some h => return path.replace "~" h
    | none => return path
  else
    return path

/-- Ensure a directory exists -/
def ensureDir (path : String) : IO Unit := do
  let expandedPath ← expandHome path
  IO.FS.createDirAll ⟨expandedPath⟩

/-- Check if a file exists -/
def fileExists (path : String) : IO Bool := do
  let expandedPath ← expandHome path
  System.FilePath.pathExists ⟨expandedPath⟩

/-! ## Download Operations -/

/-- Download a file with exponential backoff retry.
    Returns true if successful, false otherwise. -/
def downloadWithRetry (url : String) (dest : String) (maxRetries : Nat := 5)
    (initialBackoffMs : Nat := 1000) (log : Handlers := {}) : IO Bool := do
  let expandedDest ← expandHome dest

  -- Check if file already exists
  if ← fileExists expandedDest then
    log.onInfo s!"  [cached] {expandedDest}"
    return true

  -- Ensure parent directory exists
  let parentDir := System.FilePath.parent ⟨expandedDest⟩
  match parentDir with
  | some p => IO.FS.createDirAll p
  | none => pure ()

  log.onInfo s!"  Downloading: {url}"

  for attempt in [1:maxRetries+1] do
    let tempPath := s!"{expandedDest}.tmp"

    -- Use curl for download
    let result ← IO.Process.output {
      cmd := "curl"
      args := #["-fsSL", "-o", tempPath, url]
    }

    if result.exitCode == 0 then
      -- Move temp file to final location
      IO.FS.rename ⟨tempPath⟩ ⟨expandedDest⟩
      log.onInfo s!"  Downloaded: {expandedDest}"
      return true
    else
      log.onWarn s!"  Attempt {attempt}/{maxRetries} failed: {result.stderr}"

      -- Clean up temp file
      if ← fileExists tempPath then
        IO.FS.removeFile ⟨tempPath⟩

      if attempt < maxRetries then
        -- Exponential backoff: 2^attempt * initialBackoff
        let waitTime := (2 ^ attempt) * initialBackoffMs
        log.onInfo s!"  Waiting {waitTime / 1000} seconds before retry..."
        IO.sleep waitTime.toUInt32

  log.onError s!"  Failed to download after {maxRetries} attempts: {url}"
  return false

/-- Download and extract a ZIP file.
    Returns path to extracted directory. -/
def downloadAndExtractZip (url : String) (destDir : String) (maxRetries : Nat := 5)
    (log : Handlers := {}) : IO String := do
  let expandedDestDir ← expandHome destDir
  let zipPath := s!"{expandedDestDir}.zip"

  -- Check if already extracted
  if ← fileExists expandedDestDir then
    log.onInfo s!"  [cached] Extracted directory: {expandedDestDir}"
    return expandedDestDir

  -- Download the ZIP file
  let success ← downloadWithRetry url zipPath maxRetries 1000 log
  if !success then
    throw $ IO.userError s!"Failed to download: {url}"

  -- Create destination directory
  IO.FS.createDirAll ⟨expandedDestDir⟩

  -- Extract using unzip
  log.onInfo s!"  Extracting to {expandedDestDir}..."
  let result ← IO.Process.output {
    cmd := "unzip"
    args := #["-q", "-o", zipPath, "-d", expandedDestDir]
  }

  if result.exitCode != 0 then
    throw $ IO.userError s!"Failed to extract ZIP: {result.stderr}"

  -- Clean up ZIP file
  IO.FS.removeFile ⟨zipPath⟩

  log.onInfo s!"  Extracted: {expandedDestDir}"
  return expandedDestDir

/-! ## CORE Evaluation Bundle -/

/-- Ensure CORE evaluation bundle is downloaded and extracted.
    Returns path to the extracted eval_bundle directory. -/
def ensureCOREData (cacheDir : String := "~/.cache/nanochat") (log : Handlers := {}) : IO String := do
  let expandedCacheDir ← expandHome cacheDir
  let evalBundleDir := s!"{expandedCacheDir}/eval_bundle"
  let evalBundleUrl := "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

  let resolveBundleRoot (rootDir : String) : IO String := do
    let directDataDir := s!"{rootDir}/eval_data"
    if ← fileExists directDataDir then
      return rootDir
    let nestedRoot := s!"{rootDir}/eval_bundle"
    let nestedDataDir := s!"{nestedRoot}/eval_data"
    if ← fileExists nestedDataDir then
      return nestedRoot
    return rootDir

  -- Check if already exists
  if ← fileExists evalBundleDir then
    return (← resolveBundleRoot evalBundleDir)

  log.onInfo "Downloading CORE evaluation bundle..."
  let extractedRoot ← downloadAndExtractZip evalBundleUrl evalBundleDir 5 log

  return (← resolveBundleRoot extractedRoot)

/-- Ensure word list is downloaded for spelling tasks.
    Returns path to words_alpha.txt. -/
def ensureWordList (cacheDir : String := "~/.cache/nanochat") (log : Handlers := {}) : IO String := do
  let expandedCacheDir ← expandHome cacheDir
  let wordListPath := s!"{expandedCacheDir}/words_alpha.txt"
  let wordListUrl := "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"

  -- Download if needed
  let _ ← downloadWithRetry wordListUrl wordListPath 3 1000 log

  return wordListPath

/-! ## HuggingFace Utilities -/

/-- Ensure HuggingFace dataset parquet file is downloaded.
    Returns local file path. -/
def ensureHFParquet (repoId : String) (subset : String) (split : String)
    (cacheDir : String := "~/.cache/huggingface") (_fileIdx : Nat := 0)
    (log : Handlers := {})
    : IO String := do
  let expandedCacheDir ← expandHome cacheDir
  let safeRepoId := repoId.replace "/" "_"
  let localDir := s!"{expandedCacheDir}/datasets/{safeRepoId}/{subset}"
  let localPath := s!"{localDir}/{split}.parquet"

  -- Try downloading the file
  -- HuggingFace URLs can have multiple patterns, try common ones
  let urls := #[
    s!"https://huggingface.co/datasets/{repoId}/resolve/main/{subset}/{split}-00000-of-00001.parquet",
    s!"https://huggingface.co/datasets/{repoId}/resolve/main/{split}.parquet",
    s!"https://huggingface.co/datasets/{repoId}/resolve/main/data/{split}.parquet",
    s!"https://huggingface.co/datasets/{repoId}/resolve/main/data/{split}-00000-of-00001.parquet"
  ]

  -- Check if already downloaded
  if ← fileExists localPath then
    return localPath

  -- Try each URL pattern
  for url in urls do
    let success ← downloadWithRetry url localPath 3 1000 log
    if success then
      return localPath

  throw $ IO.userError s!"Failed to download HuggingFace dataset: {repoId}/{subset}/{split}"

end torch.Data.Download
