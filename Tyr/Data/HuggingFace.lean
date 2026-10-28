/-
  Tyr/Data/HuggingFace.lean

  HuggingFace dataset loading via parquet files using FFI.

  Uses the Apache Arrow parquet reader via C++ FFI bindings,
  avoiding the need for Python dependencies.
-/
import Tyr.Data.Download
import Tyr.Data.Pretraining
import Tyr.Log
import Lean.Data.Json

namespace torch.Data.HuggingFace

open torch.Data.Download
open torch.Data.Pretraining (readParquetAsJson readRowGroupAsJson getParquetMetadata)
open torch.Log

/-- HuggingFace dataset configuration -/
structure HFDatasetConfig where
  repoId : String
  subset : String := "default"
  split : String := "train"
  cacheDir : String := "~/.cache/huggingface"
  deriving Repr, Inhabited

/-- Read parquet file and return JSON objects per row using FFI.
    This uses the Apache Arrow library via C++ bindings. -/
def readParquetToJson (filePath : String) (log : Handlers := {}) : IO (Array Lean.Json) := do
  -- Use FFI to read parquet as JSON strings
  let jsonStrings ← readParquetAsJson filePath

  -- Parse each JSON string
  let mut jsons : Array Lean.Json := #[]
  for jsonStr in jsonStrings do
    match Lean.Json.parse jsonStr with
    | .ok json => jsons := jsons.push json
    | .error e =>
      log.onWarn s!"Warning: Failed to parse JSON: {e}"
  return jsons

/-- Read parquet file row group by row group for large files.
    More memory efficient than reading the entire file at once. -/
def readParquetToJsonStreaming (filePath : String) : IO (Array Lean.Json) := do
  let metadata ← getParquetMetadata filePath
  let mut allJsons : Array Lean.Json := #[]

  for rgIdx in [:metadata.numRowGroups] do
    let jsonStrings ← readRowGroupAsJson filePath rgIdx
    for jsonStr in jsonStrings do
      match Lean.Json.parse jsonStr with
      | .ok json => allJsons := allJsons.push json
      | .error _ => pure ()

  return allJsons

/-- Load ARC dataset from HuggingFace -/
def loadARC (subset : String) (split : String)
    (cacheDir : String := "~/.cache/huggingface")
    (log : Handlers := {})
    : IO (Array Lean.Json) := do
  let localPath ← ensureHFParquet "allenai/ai2_arc" subset split cacheDir 0 log
  readParquetToJson localPath log

/-- Load GSM8K dataset from HuggingFace -/
def loadGSM8K (subset : String := "main") (split : String := "train")
    (cacheDir : String := "~/.cache/huggingface")
    (log : Handlers := {})
    : IO (Array Lean.Json) := do
  let localPath ← ensureHFParquet "openai/gsm8k" subset split cacheDir 0 log
  readParquetToJson localPath log

/-- Load MMLU dataset from HuggingFace -/
def loadMMLU (subset : String) (split : String)
    (cacheDir : String := "~/.cache/huggingface")
    (log : Handlers := {})
    : IO (Array Lean.Json) := do
  let localPath ← ensureHFParquet "cais/mmlu" subset split cacheDir 0 log
  readParquetToJson localPath log

private def sortPaths (paths : Array String) : Array String :=
  paths.qsort (· < ·)

private def readSnapshotRef? (repoDir : System.FilePath) : IO (Option String) := do
  let refPath := repoDir / "refs" / "main"
  if ← refPath.pathExists then
    let ref := (← IO.FS.readFile refPath).trimAscii.toString
    if ref.isEmpty then pure none else pure (some ref)
  else
    pure none

private def findSmolTalkLocalParquets (split : String) (cacheDir : String)
    : IO (Array String) := do
  let expanded ← expandHome cacheDir
  let repoDir : System.FilePath := ⟨s!"{expanded}/hub/datasets--HuggingFaceTB--smol-smoltalk"⟩
  if !(← repoDir.pathExists) then
    return #[]
  let snapshotsDir := repoDir / "snapshots"
  if !(← snapshotsDir.pathExists) then
    return #[]

  let preferredRev? ← readSnapshotRef? repoDir
  let mut revDirs : Array System.FilePath := #[]
  match preferredRev? with
  | some rev =>
    let p := snapshotsDir / rev
    if ← p.pathExists then
      revDirs := revDirs.push p
  | none => pure ()

  let entries ← snapshotsDir.readDir
  for entry in entries do
    if (← entry.path.isDir) && !revDirs.contains entry.path then
      revDirs := revDirs.push entry.path

  let mut files : Array String := #[]
  for revDir in revDirs do
    let dataDir := revDir / "data"
    if ← dataDir.pathExists then
      let dataEntries ← dataDir.readDir
      for entry in dataEntries do
        if entry.path.extension == some "parquet" && entry.fileName.startsWith s!"{split}-" then
          files := files.push entry.path.toString
      if !files.isEmpty then
        return sortPaths files
  return #[]

private def ensureSmolTalkParquet (split : String) (cacheDir : String) (log : Handlers := {})
    : IO (Array String) := do
  let localFiles ← findSmolTalkLocalParquets split cacheDir
  if !localFiles.isEmpty then
    return localFiles

  -- Fallback: fetch known parquet shard names from HF.
  let expanded ← expandHome cacheDir
  let outDir := s!"{expanded}/datasets/HuggingFaceTB_smol-smoltalk/default"
  ensureDir outDir

  let urls : Array String :=
    if split == "test" then
      #[
        "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/resolve/main/data/test-00000-of-00001.parquet"
      ]
    else
      #[
        "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/resolve/main/data/train-00000-of-00004.parquet",
        "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/resolve/main/data/train-00001-of-00004.parquet",
        "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/resolve/main/data/train-00002-of-00004.parquet",
        "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/resolve/main/data/train-00003-of-00004.parquet"
      ]

  let mut files : Array String := #[]
  for url in urls do
    let filename := System.FilePath.fileName ⟨url⟩ |>.getD ""
    let localPath := s!"{outDir}/{filename}"
    let ok ← downloadWithRetry url localPath 3 1000 log
    if ok then
      files := files.push localPath

  if files.isEmpty then
    throw <| IO.userError s!"Failed to resolve SmolTalk parquet files for split={split}"
  return sortPaths files

/-- Load SmolTalk dataset from HuggingFace.
    Uses local HF cache snapshots when available; otherwise downloads known parquet shards. -/
def loadSmolTalk (split : String)
    (cacheDir : String := "~/.cache/huggingface")
    (log : Handlers := {})
    : IO (Array Lean.Json) := do
  let files ← ensureSmolTalkParquet split cacheDir log
  let mut rows : Array Lean.Json := #[]
  for file in files do
    let part ← readParquetToJson file log
    rows := rows ++ part
  return rows

/-- Load JSONL file from URL -/
def loadJsonlFromUrl (url : String) (cacheDir : String) (filename : String)
    (log : Handlers := {})
    : IO (Array Lean.Json) := do
  let expandedCacheDir ← expandHome cacheDir
  let localPath := s!"{expandedCacheDir}/{filename}"
  let _ ← downloadWithRetry url localPath 3 1000 log
  let content ← IO.FS.readFile ⟨localPath⟩
  let lines := content.splitOn "\n" |>.filter (!·.isEmpty)
  let mut jsons : Array Lean.Json := #[]
  for line in lines do
    match Lean.Json.parse line with
    | .ok json => jsons := jsons.push json
    | .error _ => pure ()
  return jsons

/-- Load JSONL file from local path -/
def loadJsonlFromFile (path : String) : IO (Array Lean.Json) := do
  let expandedPath ← expandHome path
  let content ← IO.FS.readFile ⟨expandedPath⟩
  let lines := content.splitOn "\n" |>.filter (!·.isEmpty)
  let mut jsons : Array Lean.Json := #[]
  for line in lines do
    match Lean.Json.parse line with
    | .ok json => jsons := jsons.push json
    | .error _ => pure ()
  return jsons

/-- Get string field from JSON object -/
def getJsonString (json : Lean.Json) (field : String) : Option String :=
  (json.getObjValAs? String field).toOption

def getJsonNat (json : Lean.Json) (field : String) : Option Nat :=
  (json.getObjValAs? Nat field).toOption

def getJsonBool (json : Lean.Json) (field : String) : Option Bool :=
  (json.getObjValAs? Bool field).toOption

def getJsonStringArray (json : Lean.Json) (field : String) : Option (Array String) :=
  (json.getObjValAs? (Array String) field).toOption

end torch.Data.HuggingFace
