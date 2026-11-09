/-
  Tyr/Model/ConfigJson.lean

  Strict field access for model configuration files. A key that is absent takes
  its default; a key that is present must decode, so a malformed value is a load
  error instead of a silent default.
-/
import Lean.Data.Json
import Tyr.Tokenizer.Types

namespace torch.ConfigJson

open Lean

def parseFile (path : System.FilePath) : IO Json := do
  match Json.parse (← IO.FS.readFile path) with
  | .ok json => pure json
  | .error err => throw (IO.userError s!"Failed to parse JSON at {path}: {err}")

/-- The member at `key`, `none` when absent; `j` must be an object. -/
def member? (j : Json) (key : String) : Except String (Option Json) :=
  match j with
  | .obj kvs => pure (kvs.get? key)
  | _ => throw s!"expected a JSON object holding '{key}', got {j.compress}"

/-- The member at `key` as a nested object, `none` when absent. -/
def objectField? (j : Json) (key : String) : Except String (Option Json) := do
  match ← member? j key with
  | none => pure none
  | some value@(.obj _) => pure (some value)
  | some value => throw s!"invalid '{key}' = {value.compress}: expected a JSON object"

/-- The decoded member at `key`, `none` when absent. -/
def field? {α : Type} [FromJson α] (j : Json) (key : String) : Except String (Option α) := do
  match ← member? j key with
  | none => pure none
  | some value =>
    match fromJson? value with
    | .ok x => pure (some x)
    | .error err => throw s!"invalid '{key}' = {value.compress}: {err}"

/-- The decoded member at `key`, `default` when absent. -/
def fieldD? {α : Type} [FromJson α] (j : Json) (key : String) (default : α) : Except String α :=
  return (← field? j key).getD default

/-- A token id at `key`, `none` when absent. JSON carries it as a natural number;
one outside the 32-bit token-id space is an error. -/
def tokenIdField? (j : Json) (key : String) : Except String (Option tokenizer.TokenId) := do
  let some (n : Nat) ← field? j key | pure none
  unless n < UInt32.size do
    throw s!"invalid '{key}' = {n}: outside the 32-bit token-id space"
  pure (some n.toUInt32)

/-- A token id at `key`, `default` when absent. -/
def tokenIdFieldD? (j : Json) (key : String) (default : tokenizer.TokenId) :
    Except String tokenizer.TokenId :=
  return (← tokenIdField? j key).getD default

/-- An array of token ids at `key`, `default` when absent. -/
def tokenIdArrayFieldD? (j : Json) (key : String) (default : Array tokenizer.TokenId) :
    Except String (Array tokenizer.TokenId) := do
  let some (ns : Array Nat) ← field? j key | pure default
  ns.mapM fun n => do
    unless n < UInt32.size do
      throw s!"invalid '{key}' entry {n}: outside the 32-bit token-id space"
    pure n.toUInt32

/-- The first of `keys` that is present, decoded; `default` when none is. -/
def firstFieldD? {α : Type} [FromJson α] (j : Json) (keys : List String) (default : α) :
    Except String α := do
  for key in keys do
    if let some x ← field? j key then return x
  pure default

/-- A string member that must be one of `allowed` when present. Used for keys whose
behavior the model hard-codes, so an unsupported value is rejected at load. -/
def requireOneOf (j : Json) (key : String) (allowed : List String) : Except String Unit := do
  if let some (value : String) ← field? j key then
    unless allowed.contains value do
      throw s!"unsupported '{key}' = \"{value}\"; this model implements {allowed}"

/-- A member that must equal `expected` when present. -/
def requireEq {α : Type} [FromJson α] [BEq α] [ToString α] (j : Json) (key : String) (expected : α) :
    Except String Unit := do
  if let some (value : α) ← field? j key then
    unless value == expected do
      throw s!"unsupported '{key}' = {value}; this model implements {expected}"

end torch.ConfigJson
