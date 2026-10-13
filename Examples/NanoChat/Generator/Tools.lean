/-
  Tyr/Generator/Tools.lean

  Tool interface and implementations for agentic generation.

  Tools allow the model to:
  - Execute safe mathematical expressions (calculator)
  - Verify Lean proofs (prover)
  - Evaluate Lean expressions (eval)

  Based on nanochat's execution.py and engine.py tool handling.
-/

namespace torch.Generator.Tools

/-- Result of tool execution -/
structure ExecResult where
  /-- Whether execution succeeded -/
  success : Bool
  /-- Output from the tool (stdout or result) -/
  output : String
  /-- Error message if execution failed -/
  error : Option String := none
  /-- Whether execution timed out -/
  timeout : Bool := false
  /-- Whether memory limit was exceeded -/
  memoryExceeded : Bool := false
  deriving Repr, Inhabited

/-- Create a successful result -/
def ExecResult.ok (output : String) : ExecResult :=
  { success := true, output := output, error := none }

/-- Create a failed result -/
def ExecResult.fail (error : String) : ExecResult :=
  { success := false, output := "", error := some error }

/-- Create a timeout result -/
def ExecResult.timedOut : ExecResult :=
  { success := false, output := "", error := some "Execution timed out", timeout := true }

/-- Tool interface -/
class Tool (α : Type) where
  /-- Tool name for identification -/
  name : String
  /-- Check if input is valid for this tool -/
  canExecute : String → Bool
  /-- Execute the tool with given input -/
  execute : String → IO ExecResult

/-- Safe character set for calculator expressions -/
def calcSafeChars : String := "0123456789+-*/.()^ "

/-- Check if a character is in the safe set -/
def isCalcSafeChar (c : Char) : Bool :=
  calcSafeChars.any (· == c)

/-- Check if string contains only safe calculator characters -/
def isCalcSafe (s : String) : Bool :=
  s.all isCalcSafeChar

/-- Check if string contains a substring -/
def stringContains (s : String) (sub : String) : Bool :=
  (s.splitOn sub).length > 1

/-! ## Calculator Expression AST and Evaluator -/

/-- Expression AST for calculator -/
inductive CalcExpr where
  | num : Float → CalcExpr
  | add : CalcExpr → CalcExpr → CalcExpr
  | sub : CalcExpr → CalcExpr → CalcExpr
  | mul : CalcExpr → CalcExpr → CalcExpr
  | div : CalcExpr → CalcExpr → CalcExpr
  | pow : CalcExpr → CalcExpr → CalcExpr
  | neg : CalcExpr → CalcExpr
  deriving Repr, Inhabited

/-- Evaluate a calculator expression -/
partial def evalCalcExpr : CalcExpr → Except String Float
  | .num n => .ok n
  | .neg e => do return -(← evalCalcExpr e)
  | .add a b => do return (← evalCalcExpr a) + (← evalCalcExpr b)
  | .sub a b => do return (← evalCalcExpr a) - (← evalCalcExpr b)
  | .mul a b => do return (← evalCalcExpr a) * (← evalCalcExpr b)
  | .div a b => do
    let bv ← evalCalcExpr b
    if bv == 0.0 then .error "Division by zero"
    else return (← evalCalcExpr a) / bv
  | .pow a b => do
    let av ← evalCalcExpr a
    let bv ← evalCalcExpr b
    return Float.pow av bv

/-! ## Calculator Parser

    Recursive descent parser for arithmetic expressions.
    Grammar (in order of precedence, lowest to highest):
    - expr := term (('+' | '-') term)*
    - term := factor (('*' | '/') factor)*
    - factor := base ('^' factor)?  (right associative)
    - base := number | '(' expr ')' | '-' base
-/

/-- Parser state: remaining characters and position -/
structure ParseState where
  input : String
  pos : Nat := 0
  deriving Repr

/-- Check if at end of input -/
def ParseState.atEnd (s : ParseState) : Bool :=
  s.pos >= s.input.length

/-- Peek at current character -/
def ParseState.peek (s : ParseState) : Option Char :=
  if s.atEnd then none else some (String.Pos.Raw.get s.input ⟨s.pos⟩)

/-- Advance by one character -/
def ParseState.advance (s : ParseState) : ParseState :=
  { s with pos := s.pos + 1 }

/-- Skip whitespace -/
def ParseState.skipWhitespace (s : ParseState) : ParseState := Id.run do
  let mut state := s
  while !state.atEnd && (state.peek.getD ' ').isWhitespace do
    state := state.advance
  return state

/-- Convert Int to Float -/
def intToFloat (n : Int) : Float :=
  if n >= 0 then n.toNat.toFloat
  else -(-n).toNat.toFloat

/-- Parse string to float (simple implementation) -/
def stringToFloat (s : String) : Option Float :=
  -- Try parsing as integer first
  match s.toInt? with
  | some n => some (intToFloat n)
  | none =>
    -- Try parsing as float - handle negative and decimal
    let parts := s.splitOn "."
    if parts.length == 1 then none
    else if parts.length == 2 then
      let intPart := parts[0]!
      let fracPart := parts[1]!
      match intPart.toInt?, fracPart.toNat? with
      | some i, some f =>
        let fracValue := f.toFloat / Float.pow 10.0 fracPart.length.toFloat
        if i >= 0 then some (intToFloat i + fracValue)
        else some (intToFloat i - fracValue)
      | _, _ => none
    else none

/-- Collect digits and optional decimal point from input -/
partial def collectNumber (state : ParseState) (numStr : String) (hasDecimal : Bool)
    : String × ParseState :=
  if state.atEnd then (numStr, state)
  else match state.peek with
  | some c =>
    if c.isDigit then
      collectNumber state.advance (numStr.push c) hasDecimal
    else if c == '.' && !hasDecimal then
      collectNumber state.advance (numStr.push c) true
    else
      (numStr, state)
  | none => (numStr, state)

/-- Parse a number (integer or float) -/
def parseNumber (s : ParseState) : Except String (Float × ParseState) :=
  let s := s.skipWhitespace
  if s.atEnd then .error "Expected number, got end of input"
  else
    -- Handle optional leading minus
    let (numStr, state) :=
      if s.peek == some '-' then
        let (rest, state) := collectNumber s.advance "" false
        ("-" ++ rest, state)
      else
        collectNumber s "" false

    if numStr.isEmpty || numStr == "-" || numStr == "." then
      .error s!"Expected number at position {s.pos}"
    else
      match stringToFloat numStr with
      | some n => .ok (n, state)
      | none => .error s!"Invalid number: {numStr}"

mutual
/-- Parse base: number | '(' expr ')' | '-' base -/
partial def parseBase (s : ParseState) : Except String (CalcExpr × ParseState) := do
  let s := s.skipWhitespace
  match s.peek with
  | some '(' =>
    -- Parenthesized expression
    let s := s.advance.skipWhitespace
    let (expr, s) ← parseExpr s
    let s := s.skipWhitespace
    match s.peek with
    | some ')' => .ok (expr, s.advance)
    | _ => .error s!"Expected ')' at position {s.pos}"
  | some '-' =>
    -- Unary negation
    let s := s.advance
    let (expr, s) ← parseBase s
    .ok (.neg expr, s)
  | some c =>
    if c.isDigit then
      let (n, s) ← parseNumber s
      .ok (.num n, s)
    else
      .error s!"Unexpected character '{c}' at position {s.pos}"
  | none => .error "Unexpected end of input"

/-- Parse factor: base ('^' factor)? -/
partial def parseFactor (s : ParseState) : Except String (CalcExpr × ParseState) := do
  let (left, s) ← parseBase s
  let s := s.skipWhitespace
  match s.peek with
  | some '^' =>
    let s := s.advance
    let (right, s) ← parseFactor s  -- Right associative
    .ok (.pow left right, s)
  | _ => .ok (left, s)

/-- Parse remaining term operators: ('*' | '/') factor ... -/
partial def parseTermRest (left : CalcExpr) (s : ParseState) : Except String (CalcExpr × ParseState) :=
  let s := s.skipWhitespace
  if s.atEnd then .ok (left, s)
  else match s.peek with
  | some '*' => do
    let (right, s') ← parseFactor s.advance
    parseTermRest (.mul left right) s'
  | some '/' => do
    let (right, s') ← parseFactor s.advance
    parseTermRest (.div left right) s'
  | _ => .ok (left, s)

/-- Parse term: factor (('*' | '/') factor)* -/
partial def parseTerm (s : ParseState) : Except String (CalcExpr × ParseState) := do
  let (left, s) ← parseFactor s
  parseTermRest left s

/-- Parse remaining expr operators: ('+' | '-') term ... -/
partial def parseExprRest (left : CalcExpr) (s : ParseState) : Except String (CalcExpr × ParseState) :=
  let s := s.skipWhitespace
  if s.atEnd then .ok (left, s)
  else match s.peek with
  | some '+' => do
    let (right, s') ← parseTerm s.advance
    parseExprRest (.add left right) s'
  | some '-' => do
    let (right, s') ← parseTerm s.advance
    parseExprRest (.sub left right) s'
  | _ => .ok (left, s)

/-- Parse expr: term (('+' | '-') term)* -/
partial def parseExpr (s : ParseState) : Except String (CalcExpr × ParseState) := do
  let (left, s) ← parseTerm s
  parseExprRest left s
end

/-- Parse a complete expression string -/
def parseCalcExpr (input : String) : Except String CalcExpr := do
  let s : ParseState := { input := input }
  let (expr, s) ← parseExpr s
  let s := s.skipWhitespace
  if !s.atEnd then
    .error s!"Unexpected input at position {s.pos}: '{input.drop s.pos}'"
  .ok expr

/-- Evaluate a calculator expression string -/
def evalCalc (input : String) : Except String Float := do
  let expr ← parseCalcExpr input
  evalCalcExpr expr

/-- Format float for output (remove trailing zeros) -/
def formatFloat (f : Float) : String :=
  let s := toString f
  -- If it's an integer, return without decimal part
  if f == f.floor && f.abs < 1e15 then
    toString f.toUInt64
  else
    s

/-- Calculator tool: evaluates safe mathematical expressions -/
@[reducible] def calculatorTool : Tool String := {
  name := "calculator"
  canExecute := fun input =>
    isCalcSafe input && !stringContains input "**"
  execute := fun input => do
    match evalCalc input.trimAscii.toString with
    | .ok result => return ExecResult.ok (formatFloat result)
    | .error e => return ExecResult.fail e
}

/-- Lean prover tool: verifies theorem proofs -/
@[reducible] def leanProverTool : Tool String := {
  name := "prover"
  canExecute := fun input =>
    stringContains input "theorem" ||
    stringContains input "lemma" ||
    stringContains input "by"
  execute := fun _input => do
    -- Placeholder: would use Lean's kernel to check proofs
    return ExecResult.fail "Prover not yet implemented"
}

/-- Lean eval tool: evaluates #eval expressions -/
@[reducible] def leanEvalTool : Tool String := {
  name := "eval"
  canExecute := fun input =>
    input.startsWith "#eval" || input.startsWith "#reduce"
  execute := fun _input => do
    -- Placeholder: would use Lean's evaluation
    return ExecResult.fail "Eval not yet implemented"
}

/-- Registry of available tools -/
structure ToolRegistry where
  /-- Available tools by name -/
  tools : List (String × (String → IO ExecResult))
  deriving Inhabited

/-- Create default tool registry with standard tools -/
def ToolRegistry.default : ToolRegistry := {
  tools := [
    ("calculator", calculatorTool.execute),
    ("prover", leanProverTool.execute),
    ("eval", leanEvalTool.execute)
  ]
}

/-- Execute a tool by name -/
def ToolRegistry.execute (registry : ToolRegistry) (toolName input : String)
    : IO ExecResult := do
  match registry.tools.find? (fun (name, _) => name == toolName) with
  | some (_, execFn) => execFn input
  | none => return ExecResult.fail s!"Unknown tool: {toolName}"

/-- Dispatch tool execution based on input content -/
def ToolRegistry.dispatch (_registry : ToolRegistry) (input : String)
    : IO ExecResult := do
  -- Try each tool's canExecute in order
  if calculatorTool.canExecute input then
    calculatorTool.execute input
  else if leanProverTool.canExecute input then
    leanProverTool.execute input
  else if leanEvalTool.canExecute input then
    leanEvalTool.execute input
  else
    return ExecResult.fail "No tool matched input"

end torch.Generator.Tools
