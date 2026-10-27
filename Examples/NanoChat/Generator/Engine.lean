/-
  Tyr/Generator/Engine.lean

  Main generation engine for autoregressive text generation with tool support.

  Features:
  - Token-by-token generation with KV caching
  - Tool invocation and result injection
  - Batch generation with per-row state tracking
  - Temperature and top-p/top-k sampling

  Based on nanochat's engine.py implementation.
-/
import Tyr.Typed.Neural
import Examples.NanoChat.Generator.State
import Examples.NanoChat.Generator.TypedState
import Examples.NanoChat.Generator.KVCache
import Examples.NanoChat.Generator.Tools

namespace torch.Generator.Engine

open torch

variable {device : Device}
open State
open KVCache
open Tools

/-- Sampling configuration -/
structure SamplingConfig where
  /-- Temperature for softmax (higher = more random) -/
  temperature : Float := 1.0
  /-- Top-p (nucleus) sampling threshold -/
  topP : Float := 0.9
  /-- Top-k sampling (0 = disabled) -/
  topK : Nat := 0
  /-- Maximum number of tokens to generate -/
  maxTokens : Nat := 256
  /-- Stop generation on these token IDs -/
  stopTokens : Array TokenId := #[]
  deriving Repr, Inhabited

/-- Generation engine wrapping model and tokenizer -/
structure Engine where
  /-- Special token IDs -/
  specialTokens : SpecialTokens
  /-- Tool registry -/
  toolRegistry : ToolRegistry
  /-- Sampling configuration -/
  samplingConfig : SamplingConfig
  deriving Inhabited

/-- Apply temperature scaling to logits -/
def applyTemperature {s : Shape} (logits : Tensor { shape := s, dtype := .Float32, device := device }) (temp : Float) : Tensor { shape := s, dtype := .Float32, device := device } :=
  if temp == 1.0 then logits
  else Tensor.mulScalar logits (1.0 / temp)

/-- Sample from logits using argmax (greedy).
    Returns array of token IDs, one per batch element. -/
def sampleGreedy {batch vocabSize : UInt64} (logits : Tensor { shape := #[batch, vocabSize], dtype := .Float32, device := device })
    : IO (Array TokenId) := do
  -- Get argmax along vocab dimension - returns [batch] tensor of indices
  let indices ← IO.ofExcept (Tensor.argmax? logits (1 : UInt64))
  -- Convert tensor to array of token IDs (use dynamic version since shape is computed)
  indices.toUInt64Array

/-- Apply top-k filtering to logits.
    Sets all logits outside the top k to -infinity. -/
def applyTopK {s : Shape} (logits : Tensor { shape := s, dtype := .Float32, device := device }) (k : Nat) : Tensor { shape := s, dtype := .Float32, device := device } :=
  if k == 0 then logits
  else Tensor.topKFilter logits k.toUInt64

/-- Apply top-p (nucleus) filtering to logits.
    Sets logits outside the nucleus (cumulative prob > p) to -infinity. -/
def applyTopP {s : Shape} (logits : Tensor { shape := s, dtype := .Float32, device := device }) (p : Float) : Tensor { shape := s, dtype := .Float32, device := device } :=
  if p >= 1.0 then logits
  else Tensor.topPFilter logits p

/-- Sample from logits with temperature, top-k, and top-p.
    Returns array of token IDs, one per batch element. -/
def sampleWithConfig {batch vocabSize : UInt64}
    (logits : Tensor { shape := #[batch, vocabSize], dtype := .Float32, device := device }) (cfg : SamplingConfig)
    : IO (Array TokenId) := do
  -- Apply temperature
  let scaled := applyTemperature logits cfg.temperature
  -- Apply top-k filtering
  let filtered := applyTopK scaled cfg.topK
  -- Apply top-p filtering
  let filtered := applyTopP filtered cfg.topP
  -- Convert to probabilities
  let probs := Tensor.softmax filtered (-1)
  -- Sample from distribution
  let samples ← Tensor.multinomial probs 1
  -- Squeeze to remove the sample dimension and convert to token IDs
  let squeezed := samples.reshapeChecked #[batch]
  squeezed.toUInt64Array

/-- Process a single generated token for a row, handling tool invocation -/
def processToken (engine : Engine) (state : RowState) (token : TokenId)
    (decode : Array TokenId → String) (encode : String → Array TokenId)
    : IO RowState := do
  -- Check for end token
  if engine.specialTokens.isEndToken token then
    return state.markComplete.appendToken token

  -- Check for tool start
  match engine.specialTokens.isToolStart token with
  | some toolName =>
    return state.enterToolBlock toolName |>.appendToken token
  | none =>
    -- Check for tool end (if in tool block)
    if state.inToolBlock && engine.specialTokens.isToolEnd token then
      -- Execute the tool
      let toolInput := decode state.toolInputTokens
      let result ← engine.toolRegistry.dispatch toolInput

      -- Encode result with markers
      let resultStr := if result.success then result.output else s!"Error: {result.error.getD "unknown"}"
      let outputTokens := #[engine.specialTokens.outputStart] ++
                          encode resultStr ++
                          #[engine.specialTokens.outputEnd]

      return state.exitToolBlock outputTokens |>.appendToken token
    else if state.inToolBlock then
      -- Accumulate tool input
      return state.accumulateToolToken token |>.appendToken token
    else
      -- Normal token
      return state.appendToken token

/-! ## Type-Safe Token Processing

These versions use the typed state machine from TypedState.lean.
Invalid transitions are caught at compile time.
-/

open TypedState in
/-- Process a token using the typed state machine.

    This version uses `BoxedRowState` with type-safe transitions.
    The key benefit is that invalid transitions (e.g., calling `exitToolBlock`
    when not in a tool block) return `none` rather than silently succeeding.

    Example:
    ```
    -- Using typed state, this pattern is enforced:
    let state := BoxedRowState.initial prompt
    let state := state.appendToken tok1  -- OK: appending always valid
    -- state.exitToolBlock ...           -- Returns none: not in tool block
    let state := state.enterToolBlock "calc" |>.getD state  -- Enter tool block
    -- Now exitToolBlock will return some
    ```
-/
def processTokenTyped (engine : Engine) (state : BoxedRowState) (token : TokenId)
    (decode : Array TokenId → String) (encode : String → Array TokenId)
    : IO BoxedRowState := do
  -- Check for end token first
  if engine.specialTokens.isEndToken token then
    -- Can only mark complete from normal phase
    match state.markComplete with
    | some completed => return completed.appendToken token
    | none =>
      -- Not in normal phase, just append and continue
      return state.appendToken token

  -- Check for tool start token
  match engine.specialTokens.isToolStart token with
  | some toolName =>
    -- enterToolBlock only works from normal phase
    match state.enterToolBlock toolName with
    | some toolState => return toolState.appendToken token
    | none =>
      -- Already in tool block or other phase, just append
      return state.appendToken token
  | none =>
    -- Check for tool end (transitions out of tool block)
    if state.isInToolBlock && engine.specialTokens.isToolEnd token then
      -- Execute the tool
      let toolInput := decode state.toolInputTokens
      let result ← engine.toolRegistry.dispatch toolInput

      -- Encode result with markers
      let resultStr := if result.success then result.output else s!"Error: {result.error.getD "unknown"}"
      let outputTokens := #[engine.specialTokens.outputStart] ++
                          encode resultStr ++
                          #[engine.specialTokens.outputEnd]

      -- exitToolBlock only works from tool block phase
      match state.exitToolBlock outputTokens with
      | some forcingState => return forcingState.appendToken token
      | none => return state.appendToken token  -- Shouldn't happen
    else if state.isInToolBlock then
      -- Accumulate tool input token
      match state.accumulateToolToken token with
      | some newState => return newState.appendToken token
      | none => return state.appendToken token
    else
      -- Normal token
      return state.appendToken token

/-- Generate using typed state machine.

    Same as `generate` but uses `TypedBatchState` with compile-time
    phase checking. This version demonstrates how the typed state machine
    integrates with the generation loop. -/
unsafe def generateTyped (engine : Engine) (batch vocabSize : UInt64)
    (promptTokens : Array (Array TokenId))
    (runModel : Array TokenId → IO (Tensor { shape := #[batch, vocabSize], dtype := .Float32, device := device }))
    (decode : Array TokenId → String)
    (encode : String → Array TokenId)
    : IO (Array (Array TokenId)) := do
  let batchSize := promptTokens.size

  -- Initialize typed batch state
  let mut batchState := TypedState.TypedBatchState.fromPrompts promptTokens

  -- Main generation loop
  for _ in [:engine.samplingConfig.maxTokens] do
    -- Check if all rows are complete
    if batchState.allComplete then break

    -- Get last tokens for each row
    let lastTokensArr := batchState.rows.map fun row =>
      row.lastToken.getD 0

    -- Run model to get logits
    let logits ← runModel lastTokensArr

    -- Sample next tokens
    let sampledTokens ← sampleWithConfig logits engine.samplingConfig

    -- Process each token with typed state machine
    for i in [:batchSize] do
      match batchState.rows[i]?, sampledTokens[i]? with
      | some row, some sampledTok =>
        if !row.isCompleted then
          -- Handle forced tokens first (from tool output injection)
          let (tokenToProcess, newRow) :=
            if row.isForcing then
              match row.popForcedToken with
              | some (some forcedTok, nextState) => (forcedTok, nextState)
              | _ => (sampledTok, row)
            else
              (sampledTok, row)

          let processedRow ← processTokenTyped engine newRow tokenToProcess decode encode
          batchState := batchState.updateRow i processedRow
      | _, _ => pure ()

  -- Return generated sequences
  return batchState.rows.map (·.tokens)

/-- Generate tokens for a batch of prompts.

    This is the main generation loop that:
    1. Runs the model to get logits
    2. Samples next tokens using temperature/top-p/top-k
    3. Processes tokens for tool invocation
    4. Updates state and KV cache
    5. Repeats until max tokens or all rows complete

    Parameters:
    - engine: Generation engine with config and tools
    - promptTokens: Array of prompt token arrays, one per batch element
    - runModel: Model callback that takes input token IDs and returns logits tensor [batch, vocabSize]
    - decode: Decoder function for tool execution
    - encode: Encoder function for tool output injection -/
def generate (engine : Engine) (batch vocabSize : UInt64)
    (promptTokens : Array (Array TokenId))
    (runModel : Array TokenId → IO (Tensor { shape := #[batch, vocabSize], dtype := .Float32, device := device }))
    (decode : Array TokenId → String)
    (encode : String → Array TokenId)
    : IO (Array (Array TokenId)) := do
  let batchSize := promptTokens.size

  -- Initialize batch state
  let mut batchState := BatchState.fromPrompts promptTokens

  -- Main generation loop
  for _ in [:engine.samplingConfig.maxTokens] do
    -- Check if all rows are complete
    if batchState.allComplete then break

    -- Get last tokens for each row (for next prediction)
    let lastTokensArr := batchState.rows.map fun row =>
      row.lastToken.getD 0

    -- Run model to get logits [batch, vocabSize]
    let logits ← runModel lastTokensArr

    -- Sample next tokens using configured sampling strategy
    let sampledTokens ← sampleWithConfig logits engine.samplingConfig

    -- Get next tokens (either forced from tool output or sampled)
    let (nextTokens, newBatchState) := batchState.getNextTokens sampledTokens
    batchState := newBatchState

    -- Process each token for tool handling
    for i in [:batchSize] do
      let row := batchState.rows[i]!
      if !row.completed then
        let newRow ← processToken engine row nextTokens[i]! decode encode
        batchState := batchState.updateRow i newRow

  -- Return generated sequences
  return batchState.rows.map (·.tokens)

/-- Generate a single completion from a prompt -/
def generateOne (engine : Engine) (vocabSize : UInt64)
    (promptTokens : Array TokenId)
    (runModel : Array TokenId → IO (Tensor { shape := #[1, vocabSize], dtype := .Float32, device := device }))
    (decode : Array TokenId → String)
    (encode : String → Array TokenId)
    : IO (Array TokenId) := do
  let results ← generate engine 1 vocabSize #[promptTokens] runModel decode encode
  return results[0]!

/-- Configuration for the generation engine -/
structure EngineConfig where
  /-- Number of transformer layers -/
  numLayers : UInt64
  /-- Batch size -/
  batchSize : UInt64
  /-- Maximum sequence length -/
  maxSeqLen : UInt64
  /-- Number of KV heads -/
  numKvHeads : UInt64
  /-- Head dimension -/
  headDim : UInt64
  /-- Sampling configuration -/
  sampling : SamplingConfig := {}
  deriving Repr, Inhabited

/-- Create engine with configuration -/
def Engine.create (cfg : EngineConfig) (specialTokens : SpecialTokens) : Engine := {
  specialTokens := specialTokens
  toolRegistry := ToolRegistry.default
  samplingConfig := cfg.sampling
}

/-- Simple interface: generate text from string prompt -/
def generateText (engine : Engine) (vocabSize : UInt64)
    (prompt : String)
    (encode : String → Array TokenId)
    (decode : Array TokenId → String)
    (runModel : Array TokenId → IO (Tensor { shape := #[1, vocabSize], dtype := .Float32, device := device }))
    : IO String := do
  let promptTokens := encode prompt
  let outputTokens ← generateOne engine vocabSize promptTokens runModel decode encode
  -- Decode only the generated part (after prompt)
  let generatedTokens := outputTokens.extract promptTokens.size outputTokens.size
  return decode generatedTokens

end torch.Generator.Engine
