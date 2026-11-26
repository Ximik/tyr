/-
  Special Tokens

  Registers special tokens in a BPE tokenizer.
-/
import Tyr.Lib.Tokenizer.Types

namespace tokenizer

/-- Add special tokens to a tokenizer -/
def addSpecialTokens (tok : BPETokenizer) (tokens : Array String) (startId : UInt32)
    : BPETokenizer := Id.run do
  let mut result := tok
  let mut id := startId
  for token in tokens do
    let mut idToBytes := result.idToBytes
    while idToBytes.size <= id.toNat do
      -- Reserve token IDs in the regular table to keep vocab indexing dense.
      idToBytes := idToBytes.push ByteArray.empty
    result := { result with
      idToBytes := idToBytes
      specialTokens := result.specialTokens.insert token id
      idToSpecial := result.idToSpecial.insert id token
    }
    id := id + 1
  return { result with vocabSize := result.idToBytes.size }

end tokenizer
