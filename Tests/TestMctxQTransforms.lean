import LeanTest
import Tyr.Mctx

open torch.mctx

private def approx (a b : Float) (tol : Float := 1e-6) : Bool :=
  Float.abs (a - b) < tol

@[test]
def testMctxQTransformMixValue : IO Unit := do
  let rawValue : Float := -0.8
  let priorLogits : Array Float := #[-1e30, -1.0, 2.0, -1e30]
  let probs := softmax priorLogits
  let visitCounts : Array Nat := #[0, 4, 4, 0]
  let scale := 10.0 / 54.0
  let qvalues : Array Float := #[20.0 * scale, 3.0 * scale, -1.0 * scale, 10.0 * scale]

  let mixed := computeMixedValue rawValue qvalues visitCounts probs

  let numSimulations := (visitCounts.foldl (init := 0) (· + ·)).toFloat
  let expected :=
    (rawValue + numSimulations *
      (probs.getD 1 0.0 * qvalues.getD 1 0.0 + probs.getD 2 0.0 * qvalues.getD 2 0.0)) /
    (numSimulations + 1.0)

  LeanTest.assertTrue (approx mixed expected 1e-6)
    s!"Expected mix value {expected}, got {mixed}"

@[test]
def testMctxQTransformMixValueWithZeroVisits : IO Unit := do
  let rawValue : Float := -0.8
  let priorLogits : Array Float := #[-1e30, -1.0, 2.0, -1e30]
  let probs := softmax priorLogits
  let visitCounts : Array Nat := #[0, 0, 0, 0]
  let qvalues : Array Float := #[0.0, 0.0, 0.0, 0.0]

  let mixed := computeMixedValue rawValue qvalues visitCounts probs
  LeanTest.assertTrue (approx mixed rawValue 1e-6)
    s!"Expected raw value {rawValue} when all actions unvisited, got {mixed}"
