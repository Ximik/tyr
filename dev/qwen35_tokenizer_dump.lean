import Tyr.Tokenizer.Qwen35

def dumpIds (label : String) (ids : Array UInt32) : IO Unit := do
  IO.println s!"{label}={ids.map (fun x => x.toUInt64)}"

def main : IO Unit := do
  let dir := ".model-cache/qwen35/Qwen__Qwen3.5-0.8B/main"
  let tok ← tokenizer.qwen35.loadTokenizer dir
  let prompt := "Write one sentence about Lean."
  let closed := tokenizer.qwen35.chatTemplate prompt
  let thinking := tokenizer.qwen35.chatTemplateThinking prompt
  IO.println s!"closed_text={repr closed}"
  dumpIds "closed_ids" (tokenizer.qwen35.encodeText tok closed)
  IO.println s!"thinking_text={repr thinking}"
  dumpIds "thinking_ids" (tokenizer.qwen35.encodeText tok thinking)
