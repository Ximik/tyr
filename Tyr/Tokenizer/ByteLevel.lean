/-
  Byte-Level Encoding

  GPT-4 style byte-to-character mapping (256 entries).
  Allows BPE to handle arbitrary binary data.
-/
import Tyr.Tokenizer.Types

namespace tokenizer

/-- GPT-2/GPT-4 byte-to-unicode mapping.
    Maps bytes 0-255 to Unicode code points to make them printable.
    - Printable ASCII (33-126, 161-172, 174-255) maps to itself
    - Non-printable bytes map to 256+ range -/
def byteToChar : Array Char := Id.run do
  let mut arr := Array.mkEmpty 256
  let mut n : UInt32 := 0

  for b in [:256] do
    let byte := b.toUInt8
    -- Check if byte is in printable range
    let isPrintable :=
      (byte >= 33 && byte <= 126) ||  -- Printable ASCII
      (byte >= 161 && byte <= 172) || -- Extended printable
      (byte >= 174)                    -- More extended
    if isPrintable then
      arr := arr.push (Char.ofNat byte.toNat)
    else
      -- Map to 256+ range
      arr := arr.push (Char.ofNat (256 + n.toNat))
      n := n + 1

  return arr

/-- Reverse mapping: char → byte (only for the 256 byte-level tokens) -/
def charToByte : Std.HashMap Char UInt8 := Id.run do
  let mut map : Std.HashMap Char UInt8 := {}
  for idx in [:256] do
    if h : idx < byteToChar.size then
      let c := byteToChar[idx]
      map := map.insert c idx.toUInt8
  return map

/-- Convert a byte array to its byte-level string representation -/
def bytesToByteLevel (bytes : ByteArray) : String := Id.run do
  let mut result := ""
  for b in bytes.toList do
    let idx := b.toNat
    if h : idx < byteToChar.size then
      result := result.push byteToChar[idx]
    else
      result := result.push '?'  -- fallback (should never happen)
  return result

/-- Convert a byte-level string back to bytes -/
def byteLevelToBytes (s : String) : ByteArray := Id.run do
  let mut result := ByteArray.empty
  for c in s.toList do
    match charToByte.get? c with
    | some b => result := result.push b
    | none => pure ()  -- ignore unknown chars
  return result

/-- Convert a regular string to byte-level representation -/
def stringToByteLevel (s : String) : String :=
  bytesToByteLevel s.toUTF8

end tokenizer
