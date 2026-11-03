/-
  Special Tokens

  NanoProof special tokens for theorem proving.
-/
import Tyr.Tokenizer.Types

namespace tokenizer

/-- Core special tokens -/
def coreSpecialTokens : Array String :=
  #[ "<|pad|>"
   , "<|eos|>"
   , "<|bos|>"
   , "<|tactic|>"
   , "<|value|>"
   , "<|goal|>"
   , "<|proof|>"
   , "<|state|>"
   ]

/-- Value bin tokens (for value head) -/
def valueBinTokens : Array String := Id.run do
  let mut tokens := Array.mkEmpty 64
  for i in [:64] do
    let padded := if i < 10 then s!"0{i}" else s!"{i}"
    tokens := tokens.push s!"<|bin_{padded}|>"
  return tokens

/-- Greek letters for mathematical notation -/
def greekLetters : Array String :=
  #[ "α", "β", "γ", "δ", "ε", "ζ", "η", "θ"
   , "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π"
   , "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω"
   , "Α", "Β", "Γ", "Δ", "Ε", "Ζ", "Η", "Θ"
   , "Ι", "Κ", "Λ", "Μ", "Ν", "Ξ", "Ο", "Π"
   , "Ρ", "Σ", "Τ", "Υ", "Φ", "Χ", "Ψ", "Ω"
   ]

/-- Mathematical operators -/
def mathOperators : Array String :=
  #[ "∀", "∃", "∄", "∈", "∉", "⊂", "⊃", "⊆", "⊇"
   , "∪", "∩", "∅", "∧", "∨", "¬", "⊕", "⊗", "⊙"
   , "→", "←", "↔", "⇒", "⇐", "⇔", "↦", "↣"
   , "≡", "≠", "≤", "≥", "≪", "≫", "≈", "≅", "∼"
   , "∞", "∂", "∇", "∫", "∑", "∏", "√", "∛"
   , "⊢", "⊣", "⊤", "⊥", "⊦", "⊧"
   , "⟨", "⟩", "⟪", "⟫", "⌊", "⌋", "⌈", "⌉"
   , "∘", "⋅", "×", "÷", "±", "∓", "′", "″"
   ]

/-- Lean-specific tokens -/
def leanTokens : Array String :=
  #[ "theorem", "lemma", "def", "structure", "class", "instance"
   , "inductive", "where", "with", "match", "by", "exact"
   , "apply", "intro", "intros", "refl", "rfl", "simp"
   , "rw", "rewrite", "have", "let", "show", "calc"
   , "sorry", "admit", "axiom", "variable", "namespace"
   , "open", "section", "end", "import", "export"
   , "#check", "#eval", "#print", "#reduce"
   , ":=", "=>", "->", "<-", "∘", "∘"
   , "Prop", "Type", "Sort", "Nat", "Int", "Bool"
   , "True", "False", "And", "Or", "Not", "Iff"
   ]

/-- Tool invocation tokens for agentic generation.
    Following nanochat's pattern for tool use:
    - Generic tool markers
    - Calculator tool for math expressions
    - Prover tool for Lean proof checking
    - Eval tool for Lean #eval execution -/
def toolTokens : Array String :=
  #[ "<|tool_start|>"       -- Generic tool invocation start
   , "<|tool_end|>"         -- Generic tool invocation end
   , "<|output_start|>"     -- Tool output injection start
   , "<|output_end|>"       -- Tool output injection end
   , "<|calc_start|>"       -- Calculator tool start
   , "<|calc_end|>"         -- Calculator tool end
   , "<|prover_start|>"     -- Lean prover tool start
   , "<|prover_end|>"       -- Lean prover tool end
   , "<|eval_start|>"       -- Lean eval tool start
   , "<|eval_end|>"         -- Lean eval tool end
   , "<|error|>"            -- Error marker for tool failures
   , "<|success|>"          -- Success marker for tool completion
   ]

/-- Chat/conversation tokens for multi-turn dialogue -/
def chatTokens : Array String :=
  #[ "<|system|>"           -- System message marker
   , "<|user|>"             -- User message marker
   , "<|assistant|>"        -- Assistant message marker
   , "<|eot|>"              -- End of turn
   , "<|header_start|>"     -- Message header start
   , "<|header_end|>"       -- Message header end
   ]

/-- All special tokens combined -/
def allSpecialTokens : Array String :=
  coreSpecialTokens ++ valueBinTokens ++ greekLetters ++ mathOperators ++ leanTokens ++ toolTokens ++ chatTokens

/-- Standard token IDs for core tokens -/
def padTokenId : TokenId := 0
def eosTokenId : TokenId := 1
def bosTokenId : TokenId := 2

/-- Add special tokens to a tokenizer -/
def addSpecialTokens (tok : BPETokenizer) (tokens : Array String) (startId : TokenId)
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
