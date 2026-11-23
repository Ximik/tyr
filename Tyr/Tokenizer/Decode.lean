/-
  Token Decoding

  Convert token IDs back to strings.
-/
import Tyr.Tokenizer.Types
import Tyr.Tokenizer.ByteLevel

namespace tokenizer

/-- Decode UTF-8 bytes, falling back to byte-level rendering for invalid streams. -/
private def decodeUtf8Lossy (bytes : ByteArray) : String :=
  match String.fromUTF8? bytes with
  | some s => s
  | none => bytesToByteLevel bytes

/-- Decode a single token ID to bytes -/
def decodeToken (tok : BPETokenizer) (id : TokenId) : ByteArray :=
  -- Check if it's a special token first
  match tok.idToSpecial.get? id with
  | some specialStr => specialStr.toUTF8
  | none =>
    -- Regular token
    match tok.idToBytes[id.toNat]? with
    | some bytes => bytes
    | none => ByteArray.empty

/-- Decode tokens to their concatenated bytes. -/
def decodeToBytes (tok : BPETokenizer) (ids : Array TokenId) : ByteArray :=
  ids.foldl (fun bytes id => bytes ++ decodeToken tok id) ByteArray.empty

/-- Decode an array of token IDs to a string -/
def decode (tok : BPETokenizer) (ids : Array TokenId) : String :=
  decodeUtf8Lossy (decodeToBytes tok ids)

end tokenizer
