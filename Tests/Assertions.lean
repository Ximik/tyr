import LeanTest

namespace Tests

/-- Assert that the action throws, optionally matching part of its error message.
Assertion failures are raised outside the action's catch block. -/
def assertThrows (action : IO α) (msgPattern : Option String := none) : IO Unit := do
  let caught ← try
    let _ ← action
    pure none
  catch error =>
    pure (some error)
  match caught with
  | none =>
    throw <| IO.userError "Assertion failed: expected exception but none was thrown"
  | some error =>
    if let some pattern := msgPattern then
      let message := error.toString
      unless message.containsSubstr pattern do
        throw <| IO.userError
          s!"Assertion failed: exception '{message}' did not contain pattern '{pattern}'"

/-- Element `i` of `xs`; a missing element fails the test. -/
def get (xs : Array α) (i : Nat) : IO α :=
  match xs[i]? with
  | some x => pure x
  | none => throw <| IO.userError s!"Assertion failed: index {i} out of bounds for size {xs.size}"

end Tests
