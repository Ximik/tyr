import Tyr.IO.ConfigJson
import Tyr.Lib.Tokenizer.Types

/-! Token ids in model configuration files. -/

namespace torch.ConfigJson

open Lean

/-- A token id at `key`, `none` when absent. JSON carries it as a natural number;
one outside the 32-bit token-id space is an error. -/
def tokenIdField? (j : Json) (key : String) : Except String (Option UInt32) := do
  let some (n : Nat) ← field? j key | pure none
  unless n < UInt32.size do
    throw s!"invalid '{key}' = {n}: outside the 32-bit token-id space"
  pure (some n.toUInt32)

/-- A token id at `key`, `default` when absent. -/
def tokenIdFieldD? (j : Json) (key : String) (default : UInt32) :
    Except String UInt32 :=
  return (← tokenIdField? j key).getD default

/-- An array of token ids at `key`, `default` when absent. -/
def tokenIdArrayFieldD? (j : Json) (key : String) (default : Array UInt32) :
    Except String (Array UInt32) := do
  let some (ns : Array Nat) ← field? j key | pure default
  ns.mapM fun n => do
    unless n < UInt32.size do
      throw s!"invalid '{key}' entry {n}: outside the 32-bit token-id space"
    pure n.toUInt32

end torch.ConfigJson
