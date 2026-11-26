/-
  Tyr/Model/Qwen/Tokenizer.lean

  Qwen3 chat templates over the byte-level BPE tokenizer.
-/
import Tyr.Lib.Tokenizer.ByteLevelBPE

namespace tokenizer.qwen3

/-- Chat template for Qwen3 (single user message, add_generation_prompt=true, enable_thinking=false). -/
def chatTemplate (prompt : String) : String :=
  "<|im_start|>user\n" ++ prompt ++ "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

/-- Qwen3-TTS assistant text wrapper used for synthesis input IDs. -/
def ttsAssistantText (text : String) : String :=
  "<|im_start|>assistant\n" ++ text ++ "<|im_end|>\n<|im_start|>assistant\n"

/-- Encode a prompt using the Qwen3 chat template and produce tokens + attention mask. -/
def encodePrompt (tok : ByteLevelBPETokenizer) (prompt : String) (maxLen : Nat := 512)
    : Array UInt32 × Array UInt32 := Id.run do
  let text := chatTemplate prompt
  let tokens := tok.encodeText text
  let max := maxLen
  let trimmed := if tokens.size > max then tokens.extract 0 max else tokens
  let mut outTokens : Array UInt32 := Array.mkEmpty max
  let mut mask : Array UInt32 := Array.mkEmpty max
  for i in [:max] do
    if h : i < trimmed.size then
      outTokens := outTokens.push (trimmed[i]'h)
      mask := mask.push (1 : UInt32)
    else
      outTokens := outTokens.push tok.padToken
      mask := mask.push (0 : UInt32)
  (outTokens, mask)

end tokenizer.qwen3
