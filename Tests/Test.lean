import Tyr
import Tyr.IO.Checkpoint
import Examples.GPT.GPT
import Examples.GPT.Train
import LeanTest

open torch
open torch.gpt
open torch.train
open torch.checkpoint

-- Shakespeare character vocabulary (65 chars)
def shakespeareChars : String :=
  "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def charToInt (c : Char) : Int64 :=
  match shakespeareChars.toList.findIdx? (· == c) with
  | some idx => idx.toInt64
  | none => 0  -- fallback to newline

def intToChar (i : Int64) : Char :=
  shakespeareChars.toList.getD i.toUInt64.toNat '\n'

def encode (s : String) : Array Int64 :=
  s.toList.toArray.map charToInt

def decode (tokens : Array Int64) : String :=
  String.ofList (tokens.toList.map intToChar)

/-- Test character-level encode/decode -/
@[test]
def testTokenizer : IO Unit := do
  -- Test that vocab size is correct (65 Shakespeare chars)
  LeanTest.assertEqual shakespeareChars.length 65 "Vocab size should be 65"

  -- Test encode produces tokens
  let testText := "Hello world"
  let tokens := encode testText
  LeanTest.assertTrue (tokens.size > 0) "Should produce tokens"
  LeanTest.assertEqual tokens.size testText.length "Token count should match text length"

  -- Test decode recovers original text
  let decoded := decode tokens
  LeanTest.assertEqual decoded testText "Decode should recover original text"
