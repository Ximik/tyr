import Tests.Assertions

namespace Tests.AssertionsRegression

private def captureError (action : IO α) : IO (Option IO.Error) := do
  try
    let _ ← action
    pure none
  catch error =>
    pure (some error)

@[test] def testAssertThrowsRejectsNonthrowingAction : IO Unit := do
  for pattern in #[none, some "expected exception"] do
    let error ← captureError (Tests.assertThrows (pure (42 : Nat)) pattern)
    match error with
    | none => LeanTest.fail "assertThrows accepted an action that returned normally"
    | some error =>
      LeanTest.assertTrue (error.toString.containsSubstr "expected exception but none was thrown")

@[test] def testAssertThrowsAcceptsMatchingException : IO Unit := do
  Tests.assertThrows (throw (IO.userError "sentinel failure") : IO Unit)
  Tests.assertThrows (throw (IO.userError "sentinel failure") : IO Unit) (some "sentinel")

@[test] def testAssertThrowsRejectsWrongPattern : IO Unit := do
  let error ← captureError
    (Tests.assertThrows (throw (IO.userError "sentinel failure") : IO Unit) (some "different"))
  match error with
  | none => LeanTest.fail "assertThrows accepted the wrong exception message"
  | some error =>
    LeanTest.assertTrue (error.toString.containsSubstr "did not contain pattern 'different'")

end Tests.AssertionsRegression
