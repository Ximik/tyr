import Tyr.Model.Qwen35.Tokenizer

namespace tokenizer.qwen36

/--
Qwen3.6-35B-A3B currently uses the same tokenizer artifacts and chat-template
shape as the shared Qwen3.5 family in Tyr, so re-export that surface under a
Qwen3.6-specific namespace for downstream users.
-/
abbrev QwenTokenizer := tokenizer.qwen35.QwenTokenizer

abbrev loadTokenizer := tokenizer.qwen35.loadTokenizer
abbrev encodeText := tokenizer.qwen35.encodeText
abbrev decodeText := tokenizer.qwen35.decodeText
abbrev decodeOne := tokenizer.qwen35.decodeOne

abbrev chatTemplate := tokenizer.qwen35.chatTemplate
abbrev chatTemplateThinking := tokenizer.qwen35.chatTemplateThinking
abbrev userPrefix := tokenizer.qwen35.userPrefix
abbrev assistantGenerationSuffix := tokenizer.qwen35.assistantGenerationSuffix
abbrev assistantGenerationSuffixThinking := tokenizer.qwen35.assistantGenerationSuffixThinking

end tokenizer.qwen36
