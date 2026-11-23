/-
  Pre-tokenization

  Simplified splitting without external regex library.
  Handles ~95% of cases for GPT-style tokenization.
-/
import Tyr.Tokenizer.Types

namespace tokenizer

/-- Check if a character is punctuation -/
def isPunctuation (c : Char) : Bool :=
  c == '.' || c == ',' || c == '!' || c == '?' ||
  c == ':' || c == ';' || c == '\'' || c == '"' ||
  c == '(' || c == ')' || c == '[' || c == ']' ||
  c == '{' || c == '}' || c == '-' || c == '_' ||
  c == '/' || c == '\\' || c == '|' || c == '@' ||
  c == '#' || c == '$' || c == '%' || c == '^' ||
  c == '&' || c == '*' || c == '+' || c == '=' ||
  c == '<' || c == '>' || c == '~' || c == '`'

/-- Token type for pretokenization -/
inductive TokenType
  | Word       -- Regular word
  | Number     -- Sequence of digits
  | Whitespace -- Spaces/newlines (kept for now)
  | Punct      -- Punctuation
  | Other      -- Everything else
  deriving Repr, BEq

/-- Classify a character -/
def classifyChar (c : Char) : TokenType :=
  if c.isAlpha then .Word
  else if c.isDigit then .Number
  else if c.isWhitespace then .Whitespace
  else if isPunctuation c then .Punct
  else .Other

/-- Pre-tokenize a string into chunks.
    Returns array of (chunk, preserveLeadingSpace) pairs.
    Leading spaces are attached to the following word. -/
def pretokenize (text : String) : Array String := Id.run do
  if text.isEmpty then return #[]

  let chars := text.toList
  let mut result : Array String := #[]
  let mut current := ""
  let mut currentType : Option TokenType := none
  let mut leadingSpaces := ""

  for c in chars do
    let charType := classifyChar c

    match currentType with
    | none =>
      -- Starting fresh
      if charType == .Whitespace then
        leadingSpaces := leadingSpaces.push c
      else
        current := leadingSpaces ++ current.push c
        leadingSpaces := ""
        currentType := some charType

    | some prevType =>
      if charType == prevType && charType != .Punct then
        -- Continue same token type (except punctuation which is always separate)
        current := current.push c
      else if charType == .Whitespace then
        -- End current token, accumulate whitespace
        if !current.isEmpty then
          result := result.push current
          current := ""
        leadingSpaces := leadingSpaces.push c
        currentType := none
      else
        -- End current token, start new one
        if !current.isEmpty then
          result := result.push current
        current := leadingSpaces ++ String.ofList [c]
        leadingSpaces := ""
        currentType := some charType

  -- Don't forget the last token
  if !current.isEmpty then
    result := result.push current

  return result

/-- Common English contractions -/
def contractions : Array String :=
  #["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]

/-- Split off contractions from a pretokenized word.
    E.g., "don't" → ["don", "'t"] -/
def splitContraction (word : String) : Array String := Id.run do
  for contr in contractions do
    if word.endsWith contr && word.length > contr.length then
      let pre := (word.dropEnd contr.length).toString
      return #[pre, contr]
  return #[word]

/-- Full pretokenization with contraction handling -/
def pretokenizeFull (text : String) : Array String :=
  (pretokenize text).flatMap splitContraction

end tokenizer
