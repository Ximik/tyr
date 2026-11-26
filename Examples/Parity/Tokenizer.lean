/-
  Examples/Parity/Tokenizer.lean

  Parity probe for `parity/tests/tokenizer_*.py`. `--weights` is a directory
  with the family's tokenizer files. Request:
  `{"family", "texts", "prompts", "reference_ids"}`, `family` one of `qwen3`,
  `qwen35`, `qwen36`, `gemma4`, `laguna`, and `reference_ids` the reference's
  ids of `texts`. Writes `ids` (each text encoded), `decoded` (each of
  `reference_ids` decoded, as UTF-8 bytes) and `chat` (for each prompt, the
  family's single-user-turn generation prompt as UTF-8 bytes: thinking off,
  then thinking on where the family has both).
-/
import Examples.Parity.Common
import Tyr.Model.Qwen35.Qwen36.Tokenizer
import Tyr.Model.Gemma4.Tokenizer
import Tyr.Model.Laguna.Tokenizer

namespace Examples.Parity.Tokenizer

open Lean tokenizer

structure Request where
  family : String
  texts : Array String
  prompts : Array String
  reference_ids : Array (Array Nat)
  deriving FromJson

/-- One family's encode, decode and chat templates. -/
structure Codec where
  encode : String → Array UInt32
  decode : Array UInt32 → String
  chat : String → Array String

def codec (family dir : String) : IO Codec := do
  match family with
  | "qwen3" =>
    let tok ← qwen3.loadTokenizer dir
    pure { encode := qwen3.encodeText tok, decode := qwen3.decodeText tok, chat := fun p => #[qwen3.chatTemplate p] }
  | "qwen35" =>
    let tok ← qwen35.loadTokenizer dir
    pure { encode := qwen35.encodeText tok, decode := qwen35.decodeText tok,
           chat := fun p => #[qwen35.chatTemplate p, qwen35.chatTemplateThinking p] }
  | "qwen36" =>
    let tok ← qwen36.loadTokenizer dir
    pure { encode := qwen36.encodeText tok, decode := qwen36.decodeText tok,
           chat := fun p => #[qwen36.chatTemplate p, qwen36.chatTemplateThinking p] }
  | "gemma4" =>
    let tok ← gemma4.loadTokenizer dir
    pure { encode := gemma4.encodeText tok, decode := gemma4.decodeText tok,
           chat := fun p => #[gemma4.chatTemplate p, gemma4.chatTemplateThinking p] }
  | "laguna" =>
    let tok ← laguna.loadTokenizer dir
    pure { encode := laguna.encodeText tok, decode := laguna.decodeTokens tok,
           chat := fun p => #[laguna.chatTemplate p false, laguna.chatTemplate p true] }
  | other => throw <| IO.userError s!"unknown tokenizer family '{other}'"

private def bytes (s : String) : Array Nat :=
  s.toUTF8.data.map (·.toNat)

def probe (args : Args) (_device : Device) (json : Json) : IO Output := do
  let request : Request ← IO.ofExcept (fromJson? json)
  let codec ← codec request.family args.weights.toString
  let ids := request.texts.map fun text => (codec.encode text).map (·.toNat)
  let decoded := request.reference_ids.map fun row => bytes (codec.decode (row.map (·.toUInt32)))
  let chat := request.prompts.flatMap fun prompt => (codec.chat prompt).map bytes
  pure { json := Json.mkObj [("ids", toJson ids), ("decoded", toJson decoded), ("chat", toJson chat)] }

end Examples.Parity.Tokenizer

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.Tokenizer.probe
