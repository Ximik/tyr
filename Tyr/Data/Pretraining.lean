/-
  Tyr/Data/Pretraining.lean

  Streaming data loader for pretraining on raw text.

  Based on nanochat's dataloader.py:
  - Iterates parquet files row-group by row-group
  - DDP-aware: each rank processes different row groups
  - Tokenizes in parallel with background threads
  - Accumulates tokens into fixed-size batches
  - Supports approximate resume via state dict
-/
import Tyr.Torch
import Tyr.Typed
import Tyr.Distributed
import Tyr.Log

namespace torch.Data.Pretraining

open torch
open torch.Log

/-! ## Parquet FFI Declarations

FFI declarations for parquet file operations. These are implemented in
`cc/src/tyr_parquet.cpp` using Apache Arrow. When Arrow/Parquet development
headers are unavailable at build time, the C++ side falls back to stubs that
fail with an IO error ("Parquet support unavailable").
-/

/-- Parquet file metadata -/
structure ParquetMetadata where
  /-- Number of row groups in the file -/
  numRowGroups : Nat
  /-- Total number of rows -/
  numRows : Nat
  /-- Column names -/
  columns : Array String
  deriving Repr, Inhabited

/-- Row group data -/
structure RowGroupData where
  /-- Documents in this row group (text column) -/
  documents : Array String
  /-- Number of documents -/
  numDocs : Nat
  deriving Repr

/-- List all parquet files in a directory.
    Returns sorted list of file paths. -/
@[extern "lean_parquet_list_files"]
opaque listParquetFiles (dirPath : @& String) : IO (Array String)

/-- Get metadata for a parquet file -/
@[extern "lean_parquet_get_metadata"]
opaque getParquetMetadata (filePath : @& String) : IO ParquetMetadata

/-- Read a specific row group from a parquet file.
    Returns the text column as an array of strings. -/
@[extern "lean_parquet_read_row_group"]
opaque readRowGroup (filePath : @& String) (rowGroupIdx : UInt64) (textColumn : @& String := "text")
    : IO RowGroupData

/-- Check if a file exists -/
@[extern "lean_parquet_file_exists"]
opaque parquetFileExists (filePath : @& String) : IO Bool

/-- Read all rows from a parquet file as JSON strings.
    Returns an array where each element is a JSON object string for one row.
    This reads all columns and handles various Arrow types (string, int, float, bool, list, struct). -/
@[extern "lean_parquet_read_as_json"]
opaque readParquetAsJson (filePath : @& String) : IO (Array String)

/-- Read a specific row group from a parquet file as JSON strings.
    More memory-efficient for large files. -/
@[extern "lean_parquet_read_row_group_as_json"]
opaque readRowGroupAsJson (filePath : @& String) (rowGroupIdx : UInt64) : IO (Array String)

/-! ## Configuration -/

/-- Configuration for pretraining data loader -/
structure PretrainingConfig where
  /-- Path to data directory containing parquet files -/
  dataPath : String := "data"
  /-- Batch size (number of sequences) -/
  batchSize : UInt64 := 4
  /-- Sequence length -/
  seqLen : UInt64 := 2048
  /-- BOS token ID (prepended to each document) -/
  bosToken : UInt64 := 50256
  /-- Number of tokenizer threads -/
  numThreads : Nat := 4
  /-- Documents to process per tokenizer batch -/
  tokenizerBatchSize : Nat := 128
  /-- Random seed for shuffling -/
  seed : UInt64 := 42
  deriving Repr, Inhabited

/-! ## State for Resumption -/

/-- State for resuming data loading from a checkpoint -/
structure LoaderState where
  /-- Current parquet file index -/
  parquetIdx : Nat
  /-- Current row group index within file -/
  rowGroupIdx : Nat
  /-- Offset within current document batch -/
  docOffset : Nat
  /-- Accumulated tokens not yet consumed -/
  tokenBuffer : Array UInt64
  /-- Total tokens processed -/
  totalTokens : UInt64
  deriving Repr, Inhabited

/-! ## Token Buffer -/

/-- Ring buffer for accumulating tokens -/
structure TokenBuffer where
  /-- Token storage -/
  tokens : Array UInt64
  /-- Read position -/
  readPos : Nat
  /-- Write position -/
  writePos : Nat
  deriving Repr

def TokenBuffer.empty : TokenBuffer := { tokens := #[], readPos := 0, writePos := 0 }

def TokenBuffer.push (buf : TokenBuffer) (t : UInt64) : TokenBuffer :=
  { buf with tokens := buf.tokens.push t, writePos := buf.writePos + 1 }

def TokenBuffer.pushMany (buf : TokenBuffer) (ts : Array UInt64) : TokenBuffer :=
  { buf with tokens := buf.tokens ++ ts, writePos := buf.writePos + ts.size }

def TokenBuffer.available (buf : TokenBuffer) : Nat :=
  buf.tokens.size - buf.readPos

def TokenBuffer.take (buf : TokenBuffer) (n : Nat) : Array UInt64 × TokenBuffer :=
  let endPos := min (buf.readPos + n) buf.tokens.size
  let taken := buf.tokens.extract buf.readPos endPos
  let newBuf := { buf with readPos := endPos }
  (taken, newBuf)

def TokenBuffer.compact (buf : TokenBuffer) : TokenBuffer :=
  if buf.readPos > 0 then
    let remaining := buf.tokens.extract buf.readPos buf.tokens.size
    { tokens := remaining, readPos := 0, writePos := remaining.size }
  else
    buf

/-! ## Pretraining Data Loader -/

/-- Main pretraining data loader -/
structure PretrainingLoader where
  /-- Configuration -/
  config : PretrainingConfig
  /-- Current state -/
  state : LoaderState
  /-- Token buffer -/
  buffer : TokenBuffer
  /-- DDP rank -/
  rank : UInt64
  /-- DDP world size -/
  worldSize : UInt64
  /-- List of parquet file paths -/
  parquetFiles : Array String
  deriving Repr

/-! ## Row Group Iterator -/

/-- Iterator state for streaming through parquet row groups.
    Supports DDP-aware striding where each rank processes
    every worldSize-th row group. -/
structure RowGroupIterator where
  /-- List of parquet file paths -/
  parquetFiles : Array String
  /-- Metadata for each file (cached) -/
  fileMetadata : Array ParquetMetadata
  /-- Current file index -/
  currentFile : Nat
  /-- Current row group index within file -/
  currentRowGroup : Nat
  /-- DDP rank (0 for single-GPU) -/
  rank : UInt64
  /-- DDP world size (1 for single-GPU) -/
  worldSize : UInt64
  /-- Total row groups processed -/
  totalRowGroupsProcessed : Nat := 0
  deriving Repr

/-- Initialize row group iterator from a data directory -/
def RowGroupIterator.init (dataPath : String) (rank worldSize : UInt64) (log : Handlers := {}) : IO RowGroupIterator := do
  -- List and sort parquet files
  let parquetFiles ← listParquetFiles dataPath

  if parquetFiles.isEmpty then
    log.onWarn s!"Warning: No parquet files found in {dataPath}"
    return {
      parquetFiles := #[]
      fileMetadata := #[]
      currentFile := 0
      currentRowGroup := rank.toNat
      rank
      worldSize
    }

  -- Get metadata for all files
  let mut metadata : Array ParquetMetadata := #[]
  for file in parquetFiles do
    let fileMeta ← getParquetMetadata file
    metadata := metadata.push fileMeta

  return {
    parquetFiles
    fileMetadata := metadata
    currentFile := 0
    currentRowGroup := rank.toNat  -- Start at rank offset for DDP
    rank
    worldSize
  }

/-- Get total number of row groups across all files -/
def RowGroupIterator.totalRowGroups (iter : RowGroupIterator) : Nat :=
  iter.fileMetadata.foldl (fun acc m => acc + m.numRowGroups) 0

/-- Check if iterator is exhausted -/
def RowGroupIterator.isDone (iter : RowGroupIterator) : Bool :=
  iter.currentFile >= iter.parquetFiles.size

/-- Get next row group, respecting DDP striding.
    Returns none if all row groups have been processed. -/
def RowGroupIterator.next (iter : RowGroupIterator) : IO (Option RowGroupData × RowGroupIterator) := do
  if iter.isDone then
    return (none, iter)

  -- Find the file containing our current row group
  let mut fileIdx := iter.currentFile
  let mut rgIdx := iter.currentRowGroup

  -- Skip files until we find one with our row group
  while fileIdx < iter.parquetFiles.size do
    let fileMeta := iter.fileMetadata[fileIdx]!
    if rgIdx < fileMeta.numRowGroups then
      break
    rgIdx := rgIdx - fileMeta.numRowGroups
    fileIdx := fileIdx + 1

  -- Check if we've exhausted all files
  if fileIdx >= iter.parquetFiles.size then
    return (none, { iter with currentFile := fileIdx })

  -- Read the row group
  let filePath := iter.parquetFiles[fileIdx]!
  let rowGroupData ← readRowGroup filePath rgIdx.toUInt64

  -- Advance to next row group for this rank (stride by worldSize)
  let nextRg := iter.currentRowGroup + iter.worldSize.toNat
  let newIter := { iter with
    currentRowGroup := nextRg
    totalRowGroupsProcessed := iter.totalRowGroupsProcessed + 1
  }

  return (some rowGroupData, newIter)

/-- Reset iterator to beginning (for new epoch) -/
def RowGroupIterator.reset (iter : RowGroupIterator) : RowGroupIterator :=
  { iter with
    currentFile := 0
    currentRowGroup := iter.rank.toNat
    totalRowGroupsProcessed := 0
  }

/-! ## Streaming Data Loader -/

/-- Initialize pretraining loader -/
def PretrainingLoader.init (config : PretrainingConfig) : IO PretrainingLoader := do
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then dist.getRankAndWorldSize else pure ((0 : UInt64), (1 : UInt64))

  -- List parquet files in data directory
  let parquetFiles ← listParquetFiles config.dataPath

  return {
    config
    state := { parquetIdx := 0, rowGroupIdx := 0, docOffset := 0, tokenBuffer := #[], totalTokens := 0 }
    buffer := TokenBuffer.empty
    rank
    worldSize
    parquetFiles
  }

/-- Create input/target pairs from token sequence.
    input = tokens[:-1], target = tokens[1:] -/
def createInputTarget (tokens : Array UInt64) : (Array UInt64 × Array UInt64) :=
  let input := tokens.extract 0 (tokens.size - 1)
  let target := tokens.extract 1 tokens.size
  (input, target)

/-- Build an Int64 token tensor from host data over the raw FFI boundary.
    `reshapeChecked` validates the result metadata (shape and Int64 dtype). -/
private def int64TensorFromArray (arr : Array Int64) : Tensor { shape := #[arr.size.toUInt64], dtype := .Int64 } :=
  Tensor.fromInt64Array arr

/-- Get next batch of (input, target) pairs.
    Returns Int64 tensors of shape [batchSize, seqLen]. -/
def PretrainingLoader.nextBatch (loader : PretrainingLoader)
    : IO (Option (Tensor { shape := #[loader.config.batchSize, loader.config.seqLen], dtype := .Int64 } ×
                  Tensor { shape := #[loader.config.batchSize, loader.config.seqLen], dtype := .Int64 }) ×
          PretrainingLoader) := do
  let tokensNeeded := (loader.config.batchSize * loader.config.seqLen + 1).toNat

  -- Check if we have enough tokens
  if loader.buffer.available >= tokensNeeded then
    let (batchTokens, newBuffer) := loader.buffer.take tokensNeeded
    let (inputs, targets) := createInputTarget batchTokens

    -- Convert to tensors
    let inputTensor := (int64TensorFromArray (inputs.map (·.toInt64))).reshapeChecked
      #[loader.config.batchSize, loader.config.seqLen]
    let targetTensor := (int64TensorFromArray (targets.map (·.toInt64))).reshapeChecked
      #[loader.config.batchSize, loader.config.seqLen]

    let newLoader := { loader with
      buffer := newBuffer.compact
      state := { loader.state with totalTokens := loader.state.totalTokens + tokensNeeded.toUInt64 }
    }

    return (some (inputTensor, targetTensor), newLoader)
  else
    -- Need more tokens - in real implementation would load from parquet
    -- For now, return none indicating end of data
    return (none, loader)

/-- Get current state for checkpointing -/
def PretrainingLoader.getState (loader : PretrainingLoader) : LoaderState :=
  loader.state

/-- Resume from a checkpoint state -/
def PretrainingLoader.resume (loader : PretrainingLoader) (state : LoaderState)
    : PretrainingLoader :=
  { loader with
    state
    buffer := TokenBuffer.empty.pushMany state.tokenBuffer
  }

/-! ## Validation Data Loader -/

/-- Load validation data (typically last shard of training data) -/
structure ValidationLoader where
  /-- All validation tokens (runtime length) -/
  tokens : Σ n : UInt64, Tensor { shape := #[n], dtype := .Int64 }
  /-- Current position -/
  position : Nat
  /-- Sequence length -/
  seqLen : UInt64
  deriving Repr

def ValidationLoader.init (tokens : Σ n : UInt64, Tensor { shape := #[n], dtype := .Int64 }) (seqLen : UInt64)
    : ValidationLoader :=
  { tokens, position := 0, seqLen }

def ValidationLoader.nextBatch (loader : ValidationLoader) (batchSize : UInt64)
    : IO (Option (Tensor { shape := #[batchSize, loader.seqLen], dtype := .Int64 } ×
                  Tensor { shape := #[batchSize, loader.seqLen], dtype := .Int64 }) × ValidationLoader) := do
  -- Implementation would slice tokens and create batches
  -- Placeholder for now
  return (none, loader)

def ValidationLoader.reset (loader : ValidationLoader) : ValidationLoader :=
  { loader with position := 0 }

/-! ## Bits-Per-Byte Evaluation -/

/-- Calculate bits-per-byte (BPB) from cross-entropy loss.
    BPB = loss * (num_tokens / num_bytes) * log2(e)

    This is a more interpretable metric than perplexity for
    comparing across different tokenizers. -/
def lossToBPB (loss : Float) (numTokens numBytes : Nat) : Float :=
  let log2e := 1.4426950408889634
  loss * (numTokens.toFloat / numBytes.toFloat) * log2e

/-- Compute average bytes per token for a tokenizer.
    Used to convert loss → BPB. -/
def avgBytesPerToken (vocabSize : Nat) (tokenBytes : Array Float) : Float :=
  tokenBytes.foldl (· + ·) 0.0 / vocabSize.toFloat

/-! ## Tokenizing Streaming Data Loader

A complete streaming data loader that:
1. Iterates parquet files row-group by row-group
2. Tokenizes documents in batches
3. Accumulates tokens into a buffer
4. Yields fixed-size batches for training
-/

/-- Configuration for the tokenizing streaming loader -/
structure StreamingLoaderConfig where
  /-- Path to data directory -/
  dataPath : String := "data"
  /-- Batch size in tokens (total across sequences) -/
  batchTokens : UInt64 := 524288  -- 2^19
  /-- Sequence length -/
  seqLen : UInt64 := 2048
  /-- BOS token ID -/
  bosToken : UInt64 := 0
  /-- EOS token ID -/
  eosToken : UInt64 := 1
  /-- Text column name in parquet files -/
  textColumn : String := "text"
  deriving Repr, Inhabited

/-- State for the tokenizing streaming loader.
    Note: In production, this would include a reference to
    an actual tokenizer (e.g., tiktoken via FFI). -/
structure StreamingLoaderState where
  /-- Row group iterator -/
  rowGroupIter : RowGroupIterator
  /-- Token buffer (deque-like) -/
  tokenBuffer : TokenBuffer
  /-- Current epoch -/
  epoch : Nat := 0
  /-- Total tokens yielded -/
  totalTokensYielded : UInt64 := 0
  /-- Checkpoint state for resume -/
  checkpointState : LoaderState
  deriving Repr

/-- Initialize the streaming loader -/
def StreamingLoaderState.init (config : StreamingLoaderConfig) (rank worldSize : UInt64)
    : IO StreamingLoaderState := do
  let rowGroupIter ← RowGroupIterator.init config.dataPath rank worldSize

  return {
    rowGroupIter
    tokenBuffer := TokenBuffer.empty
    epoch := 0
    totalTokensYielded := 0
    checkpointState := {
      parquetIdx := 0
      rowGroupIdx := rank.toNat
      docOffset := 0
      tokenBuffer := #[]
      totalTokens := 0
    }
  }

/-- Tokenize a batch of documents.
    This is a placeholder - actual implementation would use
    a real tokenizer (e.g., tiktoken via FFI).

    Each document is tokenized with:
    - BOS token prepended
    - EOS token appended
    - UTF-8 bytes converted to tokens -/
def tokenizeDocuments (docs : Array String) (bosToken eosToken : UInt64) : Array UInt64 := Id.run do
  let mut allTokens : Array UInt64 := #[]

  for doc in docs do
    -- Add BOS
    allTokens := allTokens.push bosToken

    -- Simple byte-level tokenization (placeholder)
    -- Real implementation would use BPE tokenizer
    for c in doc.toUTF8.toList do
      allTokens := allTokens.push c.toUInt64

    -- Add EOS
    allTokens := allTokens.push eosToken

  return allTokens

/-- Fill the token buffer from row groups until we have enough tokens -/
def fillBuffer (state : StreamingLoaderState) (config : StreamingLoaderConfig) (targetTokens : Nat)
    : IO StreamingLoaderState := do
  let mut st := state

  while st.tokenBuffer.available < targetTokens do
    -- Get next row group
    let (rgDataOpt, newIter) ← st.rowGroupIter.next
    st := { st with rowGroupIter := newIter }

    match rgDataOpt with
    | none =>
      -- End of data - start new epoch
      let resetIter := st.rowGroupIter.reset
      st := { st with
        rowGroupIter := resetIter
        epoch := st.epoch + 1
      }
      -- If we still can't get data, break to avoid infinite loop
      if st.rowGroupIter.isDone then break

    | some rgData =>
      -- Tokenize the documents
      let tokens := tokenizeDocuments rgData.documents config.bosToken config.eosToken
      st := { st with tokenBuffer := st.tokenBuffer.pushMany tokens }

  return st

/-- Get next batch from the streaming loader.
    Returns (input, target) Int64 tensors of shape
    `[batchTokens / seqLen, seqLen]`.

    Following nanochat's pattern:
    - input = tokens[:-1]
    - target = tokens[1:] -/
def StreamingLoaderState.nextBatch (state : StreamingLoaderState) (config : StreamingLoaderConfig)
    : IO (Option (Tensor { shape := #[config.batchTokens / config.seqLen, config.seqLen], dtype := .Int64 } ×
                  Tensor { shape := #[config.batchTokens / config.seqLen, config.seqLen], dtype := .Int64 }) ×
          StreamingLoaderState) := do
  let batchSize := config.batchTokens / config.seqLen
  let tokensNeeded := (batchSize * config.seqLen + 1).toNat

  -- Fill buffer if needed
  let st ← fillBuffer state config tokensNeeded

  -- Check if we have enough tokens
  if st.tokenBuffer.available < tokensNeeded then
    return (none, st)

  -- Take tokens for batch
  let (batchTokens, newBuffer) := st.tokenBuffer.take tokensNeeded
  let (inputs, targets) := createInputTarget batchTokens

  -- Convert to tensors
  let inputTensor := (int64TensorFromArray (inputs.map (·.toInt64))).reshapeChecked
    #[batchSize, config.seqLen]
  let targetTensor := (int64TensorFromArray (targets.map (·.toInt64))).reshapeChecked
    #[batchSize, config.seqLen]

  let newState := { st with
    tokenBuffer := newBuffer.compact
    totalTokensYielded := st.totalTokensYielded + tokensNeeded.toUInt64
    checkpointState := {
      parquetIdx := st.rowGroupIter.currentFile
      rowGroupIdx := st.rowGroupIter.currentRowGroup
      docOffset := 0
      tokenBuffer := newBuffer.tokens
      totalTokens := st.totalTokensYielded + tokensNeeded.toUInt64
    }
  }

  return (some (inputTensor, targetTensor), newState)

/-- Get checkpoint state for saving -/
def StreamingLoaderState.getCheckpoint (state : StreamingLoaderState) : LoaderState :=
  state.checkpointState

/-- Resume from a checkpoint -/
def StreamingLoaderState.resume (state : StreamingLoaderState) (checkpoint : LoaderState)
    : StreamingLoaderState :=
  { state with
    rowGroupIter := { state.rowGroupIter with
      currentFile := checkpoint.parquetIdx
      currentRowGroup := checkpoint.rowGroupIdx
    }
    tokenBuffer := TokenBuffer.empty.pushMany checkpoint.tokenBuffer
    totalTokensYielded := checkpoint.totalTokens
    checkpointState := checkpoint
  }

/-- Get progress through current epoch (0.0 to 1.0) -/
def StreamingLoaderState.progress (state : StreamingLoaderState) : Float :=
  let totalRgs := state.rowGroupIter.totalRowGroups
  if totalRgs == 0 then 1.0
  else state.rowGroupIter.totalRowGroupsProcessed.toFloat / totalRgs.toFloat

end torch.Data.Pretraining
