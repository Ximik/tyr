import Tyr.Mctx

/-!
Native, tensor-free benchmarks for the MCTS scheduling and allocation changes.
The baseline functions intentionally retain the pre-optimization algorithms.
Each comparison verifies equal search statistics or allocation checksums before
reporting timings. Results measure CPU overhead, not planning quality or GPU work.
-/

open torch.mctx

@[noinline] private def legacyNextNodeIndex (tree : Tree S E) : Nat := Id.run do
  for i in [:tree.nodeVisits.size] do
    if tree.nodeVisits.getD i 0 == 0 then return i
  return tree.nodeVisits.size

@[noinline] private def allocate (capacity : Nat) (scan : Bool) : Nat := Id.run do
  let root : RootFnOutput Unit := { priorLogits := #[0.0], value := 0.0, embedding := () }
  let mut tree := instantiateTreeFromRootWithCapacity root capacity #[false] ()
  let mut checksum := 0
  for _ in [:capacity - 1] do
    let next := if scan then legacyNextNodeIndex tree else tree.nextNodeIndex
    checksum := checksum + next
    tree := { tree with nodeVisits := tree.nodeVisits.set! next 1, numAllocated := next + 1 }
  return checksum + (if scan then legacyNextNodeIndex tree else tree.nextNodeIndex)

-- The old root selector rebuilt schedules for every possible considered-action
-- count on every simulation. All other work is identical to the cached selector.
private def legacyRoot (budget : Nat) (tree : Tree UInt64 GumbelMuZeroExtraData) (node : Nat) : Nat :=
  let visits := tree.childrenVisits[node]!
  let priors := tree.childrenPriorLogits[node]!
  let q := qtransformCompletedByMixValue tree node
  let table := getTableOfConsideredVisits 16 budget
  let considered := countConsideredActions 16 priors.size tree.rootInvalidActions
  let iteration := visits.foldl (· + ·) 0
  let round := (table.getD considered #[]).getD iteration 0
  maskedArgmax (scoreConsidered round tree.extraData.rootGumbel priors q visits)
    (some tree.rootInvalidActions)

private def model : RecurrentFn Unit UInt64 := fun _ _ action state =>
  let next := state * 5 + action.toUInt64 + 1
  let reward := (next % 17).toFloat / 17.0 - 0.4
  let value := (next % 13).toFloat / 13.0 - 0.3
  ({ reward, discount := 0.9, priorLogits := #[0.4, 0.2, 0.1, -0.3], value }, next)

@[noinline] private def runSearch (budget : Nat) (cached : Bool) : Tree UInt64 GumbelMuZeroExtraData :=
  let root : RootFnOutput UInt64 :=
    { priorLogits := #[0.4, 0.2, 0.1, -0.3], value := 0.2, embedding := 0 }
  let extra : GumbelMuZeroExtraData := {
    rootGumbel := #[0.0, 0.0, 0.0, 0.0]
    consideredVisitSchedule := if cached then some (ConsideredVisitSchedule.create 4 budget) else none
  }
  let rootFn : RootActionSelectionFn UInt64 GumbelMuZeroExtraData := fun _ tree node =>
    if cached then gumbelMuZeroRootActionSelection tree node budget 16 else legacyRoot budget tree node
  let interior : InteriorActionSelectionFn UInt64 GumbelMuZeroExtraData := fun _ tree node _ =>
    gumbelMuZeroInteriorActionSelection tree node
  search () 0 root model rootFn interior budget (some 12) none extra

private def timed (repeats : Nat) (work : Nat → Nat) : IO (Float × Nat) := do
  let start ← IO.monoNanosNow
  let mut checksum := 0
  for i in [:repeats] do checksum := checksum + work i
  let stop ← IO.monoNanosNow
  return ((stop.toFloat - start.toFloat) / 1000000.0 / repeats.toFloat, checksum)

private def report (kind : String) (size : Nat) (before after : Float) (checksum : Nat) : IO Unit :=
  IO.println s!"{kind},{size},{before},{after},{before / after},{checksum}"

def main : IO Unit := do
  IO.println "case,size,baseline_ms,optimized_ms,speedup,checksum"
  for capacity in #[128, 512, 2048, 8192] do
    let (before, a) ← timed 10 fun i => allocate (capacity + i % 2) true
    let (after, b) ← timed 10 fun i => allocate (capacity + i % 2) false
    unless a == b do throw (IO.userError "Allocation checksums differ")
    report "allocation" capacity before after a
  for budget in #[128, 512, 1024] do
    for checkedBudget in #[budget, budget + 1] do
      let old := runSearch checkedBudget false
      let new := runSearch checkedBudget true
      unless old.nodeVisits == new.nodeVisits && old.nodeValues == new.nodeValues &&
          old.childrenIndex == new.childrenIndex && old.childrenVisits == new.childrenVisits &&
          old.summary.qvalues == new.summary.qvalues do
        throw (IO.userError s!"Search statistics differ at budget {checkedBudget}")
      unless new.nodeVisits[0]! == checkedBudget + 1 do
        throw (IO.userError "Benchmark did not complete its simulation budget")
    let (before, a) ← timed 5 fun i => (runSearch (budget + i % 2) false).nodeVisits[0]!
    let (after, b) ← timed 5 fun i => (runSearch (budget + i % 2) true).nodeVisits[0]!
    unless a == b do throw (IO.userError "Search checksums differ")
    report "gumbel_search" budget before after a
