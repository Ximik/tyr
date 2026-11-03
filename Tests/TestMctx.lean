import LeanTest
import Tyr.Mctx

open torch.mctx

private def approx (a b : Float) (tol : Float := 1e-6) : Bool :=
  Float.abs (a - b) < tol

@[test]
def testSeqHalvingSchedule : IO Unit := do
  let got := getSequenceOfConsideredVisits 8 13
  let expected : Array Nat := #[
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1,
    2
  ]
  LeanTest.assertEqual got expected "Sequential halving schedule should match reference sequence"

@[test]
def testMixedValue : IO Unit := do
  let rawValue : Float := -0.8
  let priorLogits : Array Float := #[-1e30, -1.0, 2.0, -1e30]
  let probs := softmax priorLogits
  let visitCounts : Array Nat := #[0, 4, 4, 0]
  let scale := 10.0 / 54.0
  let qvalues : Array Float := #[20.0 * scale, 3.0 * scale, -1.0 * scale, 10.0 * scale]
  let mixed := computeMixedValue rawValue qvalues visitCounts probs

  let numSimulations := (visitCounts.foldl (init := 0) (· + ·)).toFloat
  let expected := (rawValue + numSimulations *
    (probs.getD 1 0.0 * qvalues.getD 1 0.0 + probs.getD 2 0.0 * qvalues.getD 2 0.0)) /
    (numSimulations + 1.0)

  LeanTest.assertTrue (approx mixed expected 1e-6) s!"Expected mixed value {expected}, got {mixed}"

@[test]
def testMuZeroPolicyBandit : IO Unit := do
  let root : RootFnOutput Unit := {
    priorLogits := #[-1.0, 0.0, 2.0, 3.0]
    value := 0.0
    embedding := ()
  }

  let rewards : Array Float := #[0.0, 0.0, 0.0, 0.0]
  let recurrentFn : RecurrentFn Unit Unit := fun _params _rng action _embedding =>
    ({
      reward := rewards.getD action 0.0
      discount := 0.0
      priorLogits := #[0.0, 0.0, 0.0, 0.0]
      value := 0.0
    }, ())

  let invalidActions : Array Bool := #[false, false, false, true]

  let out := muzeroPolicy
    (params := ())
    (rngKey := 0)
    (root := root)
    (recurrentFn := recurrentFn)
    (numSimulations := 1)
    (invalidActions := some invalidActions)
    (dirichletFraction := 0.0)

  LeanTest.assertEqual out.action 2 "MuZero should pick the best valid action"

  let expected : Array Float := #[0.0, 0.0, 1.0, 0.0]
  let weightsOk := (List.range expected.size).all fun i =>
    approx (out.actionWeights.getD i 0.0) (expected.getD i 0.0) 1e-6
  LeanTest.assertTrue weightsOk "Visit-prob action weights should be one-hot after 1 simulation"

@[test]
def testGumbelPolicyRespectsInvalidMask : IO Unit := do
  let root : RootFnOutput Unit := {
    priorLogits := #[0.0, -1.0, 2.0, 3.0]
    value := -1.0
    embedding := ()
  }

  let rewards : Array Float := #[20.0, 3.0, -1.0, 10.0]
  let recurrentFn : RecurrentFn Unit Unit := fun _params _rng action _embedding =>
    ({
      reward := rewards.getD action 0.0
      discount := 0.0
      priorLogits := #[0.0, 0.0, 0.0, 0.0]
      value := 0.0
    }, ())

  let invalidActions : Array Bool := #[true, false, false, true]

  let out := gumbelMuZeroPolicy
    (params := ())
    (rngKey := 42)
    (root := root)
    (recurrentFn := recurrentFn)
    (numSimulations := 12)
    (invalidActions := some invalidActions)

  LeanTest.assertTrue (out.action = 1 || out.action = 2)
    s!"Action should be valid under mask, got {out.action}"
