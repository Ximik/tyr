/-
  Examples/NanoChat/Generator/TypedState.lean

  Type-indexed generation state machine.

  This module provides a typed state machine where the generation phase
  is tracked at the type level. Invalid state transitions become type errors.

  Phases:
  - Normal: Regular token generation
  - InToolBlock: Accumulating tokens for tool input
  - Forcing: Injecting forced tokens (tool output)
  - Completed: Generation finished

  Example:
  ```
  -- Type-safe transitions
  let state : TypedRowState .normal := TypedRowState.initial promptTokens
  let state' : TypedRowState (.inToolBlock "calculator") := state.enterToolBlock "calculator"
  let state'' : TypedRowState .forcing := state'.exitToolBlock resultTokens
  -- state.exitToolBlock ...  -- Type error! Can't exit tool block from normal phase
  ```
-/
import Tyr.Torch
import Examples.NanoChat.Generator.State

namespace torch.Generator.TypedState

open torch
open State

/-! ## Generation Phases -/

/-- Generation phase as a type-level tag.
    Each phase represents a distinct mode of operation. -/
inductive GenerationPhase where
  /-- Normal generation - sampling tokens from model -/
  | normal
  /-- Inside a tool invocation block - accumulating input -/
  | inToolBlock (tool : String)
  /-- Forcing tokens - injecting tool output or other forced sequence -/
  | forcing
  /-- Generation complete - no more tokens will be generated -/
  | completed
  deriving Repr, BEq, Inhabited

/-! ## Phase-Specific Data -/

/-- Data that varies by phase.
    Only `inToolBlock` needs extra state (accumulated tokens). -/
inductive PhaseData : GenerationPhase → Type where
  | normal : PhaseData .normal
  | inToolBlock : (toolInputTokens : Array TokenId) → PhaseData (.inToolBlock tool)
  | forcing : PhaseData .forcing
  | completed : PhaseData .completed

instance : Inhabited (PhaseData GenerationPhase.normal) where
  default := .normal

instance : Inhabited (PhaseData GenerationPhase.forcing) where
  default := .forcing

instance : Inhabited (PhaseData GenerationPhase.completed) where
  default := .completed

instance {tool : String} : Inhabited (PhaseData (GenerationPhase.inToolBlock tool)) where
  default := .inToolBlock #[]

/-! ## Typed Row State -/

/-- Row state indexed by generation phase.
    The phase determines what operations are valid. -/
structure TypedRowState (phase : GenerationPhase) where
  /-- Full token sequence generated so far -/
  tokens : Array TokenId
  /-- Queue of tokens to force-inject (non-empty only in forcing phase) -/
  forcedTokens : Array TokenId
  /-- Number of tokens generated (excluding prompt) -/
  generatedCount : Nat
  /-- Phase-specific data -/
  phaseData : PhaseData phase

instance : Inhabited (TypedRowState .normal) where
  default := {
    tokens := #[]
    forcedTokens := #[]
    generatedCount := 0
    phaseData := .normal
  }

instance : Inhabited (TypedRowState .forcing) where
  default := {
    tokens := #[]
    forcedTokens := #[]
    generatedCount := 0
    phaseData := .forcing
  }

instance : Inhabited (TypedRowState .completed) where
  default := {
    tokens := #[]
    forcedTokens := #[]
    generatedCount := 0
    phaseData := .completed
  }

instance {tool : String} : Inhabited (TypedRowState (.inToolBlock tool)) where
  default := {
    tokens := #[]
    forcedTokens := #[]
    generatedCount := 0
    phaseData := .inToolBlock #[]
  }

/-! ## State Construction -/

/-- Create initial state from prompt tokens (in normal phase) -/
def TypedRowState.initial (promptTokens : Array TokenId) : TypedRowState .normal := {
  tokens := promptTokens
  forcedTokens := #[]
  generatedCount := 0
  phaseData := .normal
}

/-! ## Phase Transitions

These functions encode the valid state transitions as types.
Invalid transitions (e.g., exiting a tool block from normal phase) are type errors.
-/

/-- Append a token to the sequence (valid in any non-completed phase) -/
def TypedRowState.appendToken {phase : GenerationPhase}
    (state : TypedRowState phase) (token : TokenId)
    (_h : phase ≠ .completed := by trivial)
    : TypedRowState phase :=
  { state with
    tokens := state.tokens.push token
    generatedCount := state.generatedCount + 1
  }

/-- Enter tool invocation mode (only valid from normal phase) -/
def TypedRowState.enterToolBlock (state : TypedRowState .normal) (toolName : String)
    : TypedRowState (.inToolBlock toolName) := {
  tokens := state.tokens
  forcedTokens := state.forcedTokens
  generatedCount := state.generatedCount
  phaseData := .inToolBlock #[]
}

/-- Accumulate a token for tool input (only valid in tool block phase) -/
def TypedRowState.accumulateToolToken {tool : String}
    (state : TypedRowState (.inToolBlock tool)) (token : TokenId)
    : TypedRowState (.inToolBlock tool) :=
  match state.phaseData with
  | .inToolBlock accum =>
    { state with phaseData := .inToolBlock (accum.push token) }

/-- Get accumulated tool input tokens -/
def TypedRowState.getToolInput {tool : String}
    (state : TypedRowState (.inToolBlock tool)) : Array TokenId :=
  match state.phaseData with
  | .inToolBlock accum => accum

/-- Exit tool block and enter forcing phase (only valid from tool block) -/
def TypedRowState.exitToolBlock {tool : String}
    (state : TypedRowState (.inToolBlock tool)) (resultTokens : Array TokenId)
    : TypedRowState .forcing := {
  tokens := state.tokens
  forcedTokens := resultTokens
  generatedCount := state.generatedCount
  phaseData := .forcing
}

/-- Pop a forced token (only valid in forcing phase).
    Transitions to normal phase when tokens exhausted. -/
def TypedRowState.popForcedToken (state : TypedRowState .forcing)
    : (Option TokenId) × (TypedRowState .normal ⊕ TypedRowState .forcing) :=
  if state.forcedTokens.size > 0 then
    let token := state.forcedTokens[0]!
    let newForced := state.forcedTokens.extract 1 state.forcedTokens.size
    if newForced.size > 0 then
      (some token, .inr { state with forcedTokens := newForced })
    else
      (some token, .inl {
        tokens := state.tokens
        forcedTokens := #[]
        generatedCount := state.generatedCount
        phaseData := .normal
      })
  else
    (none, .inl {
      tokens := state.tokens
      forcedTokens := #[]
      generatedCount := state.generatedCount
      phaseData := .normal
    })

/-- Check if forcing phase has more tokens -/
def TypedRowState.hasForcedTokens (state : TypedRowState .forcing) : Bool :=
  state.forcedTokens.size > 0

/-- Mark generation as complete (only valid from normal phase) -/
def TypedRowState.markComplete (state : TypedRowState .normal)
    : TypedRowState .completed := {
  tokens := state.tokens
  forcedTokens := #[]
  generatedCount := state.generatedCount
  phaseData := .completed
}

/-- Get the last token in the sequence -/
def TypedRowState.lastToken {phase : GenerationPhase}
    (state : TypedRowState phase) : Option TokenId :=
  state.tokens.back?

/-! ## Boxed State (Existential Wrapper)

For heterogeneous collections where the phase varies per element.
-/

/-- Operations on a boxed (phase-erased) row state -/
structure RowStateOps where
  /-- Get current phase -/
  phase : GenerationPhase
  /-- Get token sequence -/
  tokens : Array TokenId
  /-- Get forced token queue -/
  forcedTokens : Array TokenId
  /-- Get generated count -/
  generatedCount : Nat
  /-- Get last token -/
  lastToken : Option TokenId
  /-- Check if completed -/
  isCompleted : Bool
  /-- Check if forcing tokens -/
  isForcing : Bool
  /-- Check if in tool block -/
  isInToolBlock : Bool
  /-- Get tool name (if in tool block) -/
  currentTool : Option String
  /-- Get tool input tokens (if in tool block) -/
  toolInputTokens : Array TokenId

instance : Inhabited RowStateOps where
  default := {
    phase := .normal
    tokens := #[]
    forcedTokens := #[]
    generatedCount := 0
    lastToken := none
    isCompleted := false
    isForcing := false
    isInToolBlock := false
    currentTool := none
    toolInputTokens := #[]
  }

/-- Create ops from typed state -/
def RowStateOps.ofTyped {phase : GenerationPhase} (state : TypedRowState phase) : RowStateOps := {
  phase := phase
  tokens := state.tokens
  forcedTokens := state.forcedTokens
  generatedCount := state.generatedCount
  lastToken := state.tokens.back?
  isCompleted := phase == .completed
  isForcing := phase == .forcing
  isInToolBlock := match phase with | .inToolBlock _ => true | _ => false
  currentTool := match phase with | .inToolBlock tool => some tool | _ => none
  toolInputTokens := match phase, state.phaseData with
    | .inToolBlock _, .inToolBlock toks => toks
    | _, _ => #[]
}

/-- Boxed row state that hides the phase at type level -/
structure BoxedRowState where
  ops : RowStateOps
  -- We also need to store update functions for transitions
  /-- Append token and get new boxed state -/
  appendTokenFn : TokenId -> BoxedRowState
  /-- Enter tool block (if in normal phase) -/
  enterToolBlockFn : String -> Option BoxedRowState
  /-- Accumulate tool token (if in tool block) -/
  accumulateToolTokenFn : TokenId -> Option BoxedRowState
  /-- Exit tool block with result (if in tool block) -/
  exitToolBlockFn : Array TokenId -> Option BoxedRowState
  /-- Pop forced token (if in forcing phase) -/
  popForcedTokenFn : Option (Prod (Option TokenId) BoxedRowState)
  /-- Mark complete (if in normal phase) -/
  markCompleteFn : Option BoxedRowState

/- Forward accessors -/
namespace BoxedRowState

def phase (s : BoxedRowState) : GenerationPhase := s.ops.phase
def tokens (s : BoxedRowState) : Array TokenId := s.ops.tokens
def forcedTokens (s : BoxedRowState) : Array TokenId := s.ops.forcedTokens
def generatedCount (s : BoxedRowState) : Nat := s.ops.generatedCount
def lastToken (s : BoxedRowState) : Option TokenId := s.ops.lastToken
def isCompleted (s : BoxedRowState) : Bool := s.ops.isCompleted
def isForcing (s : BoxedRowState) : Bool := s.ops.isForcing
def isInToolBlock (s : BoxedRowState) : Bool := s.ops.isInToolBlock
def currentTool (s : BoxedRowState) : Option String := s.ops.currentTool
def toolInputTokens (s : BoxedRowState) : Array TokenId := s.ops.toolInputTokens

def appendToken (s : BoxedRowState) (token : TokenId) : BoxedRowState :=
  s.appendTokenFn token

def enterToolBlock (s : BoxedRowState) (tool : String) : Option BoxedRowState :=
  s.enterToolBlockFn tool

def accumulateToolToken (s : BoxedRowState) (token : TokenId) : Option BoxedRowState :=
  s.accumulateToolTokenFn token

def exitToolBlock (s : BoxedRowState) (resultTokens : Array TokenId) : Option BoxedRowState :=
  s.exitToolBlockFn resultTokens

def popForcedToken (s : BoxedRowState) : Option (Prod (Option TokenId) BoxedRowState) :=
  s.popForcedTokenFn

def markComplete (s : BoxedRowState) : Option BoxedRowState :=
  s.markCompleteFn

end BoxedRowState

mutual
  /-- Box a normal phase state -/
  unsafe def boxNormal (state : TypedRowState .normal) : BoxedRowState := {
    ops := RowStateOps.ofTyped state
    appendTokenFn := fun tok =>
      boxNormal (state.appendToken tok (_h := by
        intro hEq
        cases hEq))
    enterToolBlockFn := fun tool => some (boxInToolBlock (state.enterToolBlock tool))
    accumulateToolTokenFn := fun _ => none  -- Not in tool block
    exitToolBlockFn := fun _ => none  -- Not in tool block
    popForcedTokenFn := none  -- Not forcing
    markCompleteFn := some (boxCompleted state.markComplete)
  }

  /-- Box an in-tool-block phase state -/
  unsafe def boxInToolBlock {tool : String} (state : TypedRowState (.inToolBlock tool)) : BoxedRowState := {
    ops := RowStateOps.ofTyped state
    appendTokenFn := fun tok =>
      let state' := state.accumulateToolToken tok
      let state'' : TypedRowState (.inToolBlock tool) := {
        tokens := state'.tokens.push tok
        forcedTokens := state'.forcedTokens
        generatedCount := state'.generatedCount + 1
        phaseData := state'.phaseData
      }
      boxInToolBlock state''
    enterToolBlockFn := fun _ => none  -- Already in tool block
    accumulateToolTokenFn := fun tok => some (boxInToolBlock (state.accumulateToolToken tok))
    exitToolBlockFn := fun resultTokens => some (boxForcing (state.exitToolBlock resultTokens))
    popForcedTokenFn := none  -- Not forcing
    markCompleteFn := none  -- Can't complete from tool block
  }

  /-- Box a forcing phase state -/
  unsafe def boxForcing (state : TypedRowState .forcing) : BoxedRowState := {
    ops := RowStateOps.ofTyped state
    appendTokenFn := fun tok =>
      -- In forcing, append just updates the tokens array
      let state' : TypedRowState .forcing := {
        tokens := state.tokens.push tok
        forcedTokens := state.forcedTokens
        generatedCount := state.generatedCount + 1
        phaseData := .forcing
      }
      boxForcing state'
    enterToolBlockFn := fun _ => none  -- Must finish forcing first
    accumulateToolTokenFn := fun _ => none  -- Not in tool block
    exitToolBlockFn := fun _ => none  -- Not in tool block
    popForcedTokenFn :=
      let (tokOpt, nextState) := state.popForcedToken
      match nextState with
      | .inl normalState => some (tokOpt, boxNormal normalState)
      | .inr forcingState => some (tokOpt, boxForcing forcingState)
    markCompleteFn := none  -- Can't complete while forcing
  }

  /-- Box a completed phase state -/
  unsafe def boxCompleted (state : TypedRowState .completed) : BoxedRowState := {
    ops := RowStateOps.ofTyped state
    appendTokenFn := fun _ => boxCompleted state  -- No-op when completed
    enterToolBlockFn := fun _ => none
    accumulateToolTokenFn := fun _ => none
    exitToolBlockFn := fun _ => none
    popForcedTokenFn := none
    markCompleteFn := none  -- Already completed
  }
end

/-- Box any typed state -/
unsafe def boxTypedState {phase : GenerationPhase} (state : TypedRowState phase) : BoxedRowState :=
  match phase with
  | .normal => boxNormal state
  | .inToolBlock tool => boxInToolBlock (tool := tool) state
  | .forcing => boxForcing state
  | .completed => boxCompleted state

/-- Create boxed state from prompt tokens -/
unsafe def BoxedRowState.initial (promptTokens : Array TokenId) : BoxedRowState :=
  boxNormal (TypedRowState.initial promptTokens)

/-! ## Conversion to/from Untyped State -/

/-- Convert untyped RowState to BoxedRowState -/
unsafe def fromRowState (state : RowState) : BoxedRowState :=
  if state.completed then
    boxCompleted {
      tokens := state.tokens
      forcedTokens := #[]
      generatedCount := state.generatedCount
      phaseData := .completed
    }
  else if state.forcedTokens.size > 0 then
    boxForcing {
      tokens := state.tokens
      forcedTokens := state.forcedTokens
      generatedCount := state.generatedCount
      phaseData := .forcing
    }
  else if state.inToolBlock then
    let tool := state.currentTool.getD "unknown"
    boxInToolBlock (tool := tool) {
      tokens := state.tokens
      forcedTokens := state.forcedTokens
      generatedCount := state.generatedCount
      phaseData := .inToolBlock state.toolInputTokens
    }
  else
    boxNormal {
      tokens := state.tokens
      forcedTokens := state.forcedTokens
      generatedCount := state.generatedCount
      phaseData := .normal
    }

/-- Convert BoxedRowState back to untyped RowState -/
def toRowState (state : BoxedRowState) : RowState := {
  tokens := state.tokens
  forcedTokens := state.forcedTokens
  inToolBlock := state.isInToolBlock
  toolInputTokens := state.toolInputTokens
  currentTool := state.currentTool
  completed := state.isCompleted
  generatedCount := state.generatedCount
}

/-! ## Typed Batch State -/

/-- Batch state using boxed rows -/
structure TypedBatchState where
  /-- Per-row states -/
  rows : Array BoxedRowState
  /-- Total tokens generated across all rows -/
  totalGenerated : Nat

instance : Inhabited TypedBatchState where
  default := {
    rows := #[]
    totalGenerated := 0
  }

/-- Create batch state from prompts -/
unsafe def TypedBatchState.fromPrompts (prompts : Array (Array TokenId)) : TypedBatchState := {
  rows := prompts.map BoxedRowState.initial
  totalGenerated := 0
}

/-- Check if all rows are complete -/
def TypedBatchState.allComplete (state : TypedBatchState) : Bool :=
  state.rows.all (·.isCompleted)

/-- Count completed rows -/
def TypedBatchState.completedCount (state : TypedBatchState) : Nat :=
  state.rows.foldl (fun acc row => if row.isCompleted then acc + 1 else acc) 0

/-- Update row at index -/
def TypedBatchState.updateRow (state : TypedBatchState) (idx : Nat) (newRow : BoxedRowState)
    : TypedBatchState :=
  if idx < state.rows.size then
    { state with rows := state.rows.set! idx newRow }
  else
    state

end torch.Generator.TypedState
