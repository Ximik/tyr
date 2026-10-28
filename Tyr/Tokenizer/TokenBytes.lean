import Tyr.Typed.Neural
/-
  Tyr/Tokenizer/TokenBytes.lean

  Token byte mapping for bits-per-byte (BPB) evaluation.

  Based on nanochat's token_bytes.pt:
  - Maps each token ID → number of UTF-8 bytes
  - Special tokens → 0 bytes (don't count toward BPB)
  - Enables vocab-size-independent evaluation metric
-/
import Tyr.Tokenizer.Types
import Tyr.Torch
import Tyr.Typed

namespace tokenizer

open torch

variable {device : Device}

/-- Token bytes mapping for BPB calculation.
    Maps each token ID to the number of UTF-8 bytes it represents.
    Special tokens have 0 bytes (don't contribute to BPB). -/
structure TokenBytes where
  /-- Vocabulary size -/
  vocabSize : UInt32
  /-- Bytes per token ID as Int64 tensor of shape [vocabSize] -/
  bytes : Tensor { shape := #[vocabSize.toNat], dtype := .Int64, device := .CPU }
  deriving Repr

/-- Construct the requested Int64 vector, checking its length. -/
private def int64VectorFromArray (arr : Array Int64) (n : Nat) : Tensor { shape := #[n], dtype := .Int64, device := .CPU } :=
  (Tensor.fromInt64Array arr).reshapeChecked #[n]

private def indexSelect1d {n k : Nat} (t : Tensor { shape := #[n], dtype := .Int64, device := device })
    (indices : Tensor { shape := #[k], dtype := .Int64, device := device }) :
    IO (Tensor { shape := #[k], dtype := .Int64, device := device }) :=
  IO.ofExcept (Tensor.indexSelect? t indices)

/-- Transfer the byte table without leaving the IO error boundary. -/
private def bytesAt (tb : TokenBytes) (device : Device) :
    IO (Tensor { shape := #[tb.vocabSize.toNat], dtype := .Int64, device }) := do
  let raw ← RawTensor.convertIO tb.bytes.shape tb.bytes.raw device none
  IO.ofExcept (Tensor.ofRaw raw)

/-- Build token bytes mapping from a BPE tokenizer.
    For each token:
    - Regular tokens: count UTF-8 bytes from idToBytes
    - Special tokens: 0 bytes -/
def TokenBytes.fromTokenizer (tok : BPETokenizer) : TokenBytes := Id.run do
  let mut byteCounts : Array Int64 := Array.mkEmpty tok.vocabSize.toNat

  for id in [:tok.vocabSize.toNat] do
    let tokenId : TokenId := id.toUInt32
    if tok.isSpecialToken tokenId then
      -- Special tokens contribute 0 bytes
      byteCounts := byteCounts.push 0
    else
      match tok.getTokenBytes tokenId with
      | some ba => byteCounts := byteCounts.push ba.size.toInt64
      | none => byteCounts := byteCounts.push 0

  { vocabSize := tok.vocabSize
    bytes := int64VectorFromArray byteCounts tok.vocabSize.toNat }

/-- Compute total bytes from an array of token IDs.
    Uses the token bytes mapping to sum up byte counts. -/
def TokenBytes.totalBytes (tb : TokenBytes) (tokenIds : Array UInt32) : IO UInt64 := do
  if tokenIds.isEmpty then return 0
  -- Convert token IDs to Int64 tensor for indexing
  let idxTensor := int64VectorFromArray (tokenIds.map (·.toNat.toInt64)) tokenIds.size
  -- Use index_select to get byte counts for each token
  let bytesPerToken ← indexSelect1d tb.bytes idxTensor
  -- Sum all bytes
  let total ← IO.ofExcept (Tensor.sumAll? bytesPerToken)
  let totalFloat ← IO.ofExcept (Tensor.item? total)
  return totalFloat.toUInt64

/-- Compute total bytes for a 1D tensor of token IDs.
    More efficient than array-based version for tensor inputs. -/
def TokenBytes.totalBytesFromTensor (tb : TokenBytes) (tokenIds : Tensor { shape := #[n], dtype := .Int64, device := device })
    : IO UInt64 := do
  -- Index into bytes to get byte counts per token
  let bytesPerToken ← indexSelect1d (← bytesAt tb device) tokenIds
  -- Sum all bytes
  let total ← IO.ofExcept (Tensor.sumAll? bytesPerToken)
  let totalFloat ← IO.ofExcept (Tensor.item? total)
  return totalFloat.toUInt64

/-- Compute total bytes with a loss mask.
    Only counts bytes for tokens where mask > 0.
    Used in BPB evaluation where we only count trained tokens. -/
def TokenBytes.totalBytesWithMask (tb : TokenBytes)
    (tokenIds : Tensor { shape := #[b, s], dtype := .Int64, device := device }) (mask : Tensor { shape := #[b, s], dtype := .Float32, device := device })
    : IO UInt64 := do
  -- Flatten for indexing
  let numTokens := b * s

  let flatIds ← IO.ofExcept (tokenIds.reshapeChecked? #[numTokens])
  let flatMask ← IO.ofExcept (mask.reshapeChecked? #[numTokens])

  -- Get byte counts per token
  let bytesPerToken ← indexSelect1d (← bytesAt tb device) flatIds

  -- Multiply by mask and sum (mask is Float32, so cast the counts)
  let bytesFloat ← IO.ofExcept (bytesPerToken.castDType? .Float32)
  let maskedBytes ← IO.ofExcept (Tensor.mul? bytesFloat flatMask)
  let summed ← IO.ofExcept (Tensor.sumAll? maskedBytes)
  let total ← IO.ofExcept (Tensor.item? summed)
  return total.toUInt64

/-- Compute total bytes with a loss mask, with the batch and sequence
    dimensions named explicitly. -/
def TokenBytes.totalBytesWithMask' (tb : TokenBytes)
    {batchSize seqLen : Nat}
    (tokenIds : Tensor { shape := #[batchSize, seqLen], dtype := .Int64, device := device })
    (mask : Tensor { shape := #[batchSize, seqLen], dtype := .Float32, device := device })
    : IO UInt64 :=
  tb.totalBytesWithMask tokenIds mask

/-- Save token bytes mapping to a file -/
def TokenBytes.save (tb : TokenBytes) (path : String) : IO Unit := do
  Tensor.saveTensor tb.bytes path

/-- Load token bytes mapping from a file.

    Unlike the legacy loader, the stored tensor is validated against the
    claimed spec (`[vocabSize]`, Int64) and a mismatch is an IO error. -/
def TokenBytes.load (path : String) (vocabSize : UInt32) : IO TokenBytes := do
  let bytes ← Tensor.loadTensorExact {shape := #[vocabSize.toNat], dtype := .Int64, device := .CPU} path
  return { vocabSize, bytes }

/-- Create a dummy token bytes mapping for testing.
    Assumes each token represents 2 bytes on average (rough approximation). -/
def TokenBytes.dummy (vocabSize : UInt32) : TokenBytes := Id.run do
  let mut byteCounts : Array Int64 := Array.mkEmpty vocabSize.toNat
  for _ in [:vocabSize.toNat] do
    byteCounts := byteCounts.push 2  -- Default 2 bytes per token
  { vocabSize, bytes := int64VectorFromArray byteCounts vocabSize.toNat }

/-- Create a realistic token bytes mapping for testing.
    Uses a distribution typical of BPE tokenizers:
    - Single bytes (0-255): 1 byte each
    - Common words: 3-6 bytes
    - Special tokens (last 100): 0 bytes -/
def TokenBytes.realistic (vocabSize : UInt32) (numSpecialTokens : UInt32 := 100) : TokenBytes := Id.run do
  let mut byteCounts : Array Int64 := Array.mkEmpty vocabSize.toNat

  for id in [:vocabSize.toNat] do
    let byteCount : Int64 :=
      if id < 256 then
        -- Base bytes: 1 byte each
        1
      else if id >= vocabSize.toNat - numSpecialTokens.toNat then
        -- Special tokens: 0 bytes
        0
      else
        -- Regular tokens: average ~3 bytes (BPE merges)
        3
    byteCounts := byteCounts.push byteCount

  { vocabSize, bytes := int64VectorFromArray byteCounts vocabSize.toNat }

end tokenizer
