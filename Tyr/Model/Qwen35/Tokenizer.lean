import Tyr.Lib.Tokenizer.ByteLevelBPE

namespace tokenizer.qwen35

/--
Qwen3.5 uses the same HuggingFace `tokenizer.json` model format as Qwen3 in the
current checkpoints we support, so reuse the full tokenizer loader/codec and
specialize only the chat template surface here.
-/
abbrev QwenTokenizer := tokenizer.ByteLevelBPETokenizer

abbrev loadTokenizer := tokenizer.ByteLevelBPETokenizer.load
abbrev encodeText := tokenizer.ByteLevelBPETokenizer.encodeText
abbrev decodeText := tokenizer.ByteLevelBPETokenizer.decodeText
abbrev decodeOne := tokenizer.ByteLevelBPETokenizer.decodeOne

/-- Qwen3.5 generation prompt for a single user turn with `enable_thinking=false`,
    which is the default branch in the HF tokenizer template when no explicit
    thinking flag is provided. -/
def chatTemplate (prompt : String) : String :=
  "<|im_start|>user\n" ++ prompt ++ "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

/-- Qwen3.5 generation prompt with `enable_thinking=true`. -/
def chatTemplateThinking (prompt : String) : String :=
  "<|im_start|>user\n" ++ prompt ++ "<|im_end|>\n<|im_start|>assistant\n<think>\n"

/-- Prefix up to the start of user content. Useful when injecting multimodal
    placeholders into the user turn before closing it. -/
def userPrefix : String :=
  "<|im_start|>user\n"

/-- Suffix from the end of user content into the assistant generation prompt. -/
def assistantGenerationSuffix : String :=
  "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

/-- Thinking-enabled assistant suffix from the end of user content into the
    assistant generation prompt. -/
def assistantGenerationSuffixThinking : String :=
  "<|im_end|>\n<|im_start|>assistant\n<think>\n"

end tokenizer.qwen35
