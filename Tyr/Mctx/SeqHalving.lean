import Tyr.Mctx.Math

/-!
# Tyr.Mctx.SeqHalving

Sequential-halving visit schedules for Gumbel MuZero: how many simulations
each considered action receives (`getSequenceOfConsideredVisits`), plus the
considered-action scoring used by Gumbel root action selection.
-/

namespace torch.mctx

private def updateAt (xs : Array α) (i : Nat) (v : α) : Array α :=
  if i < xs.size then xs.set! i v else xs

private def ceilLog2 (n : Nat) : Nat := Id.run do
  if n <= 1 then
    return 0
  let mut p := 1
  let mut k := 0
  while p < n do
    p := p * 2
    k := k + 1
  return k

/-- Sequential-halving considered-visit schedule. -/
def getSequenceOfConsideredVisits (maxNumConsideredActions numSimulations : Nat) : Array Nat := Id.run do
  if maxNumConsideredActions <= 1 then
    return (List.range numSimulations).toArray

  let log2max := ceilLog2 maxNumConsideredActions
  let mut sequence : Array Nat := #[]
  let mut visits : Array Nat := Array.replicate maxNumConsideredActions 0
  let mut numConsidered := maxNumConsideredActions

  while sequence.size < numSimulations do
    let denom := Nat.max 1 (log2max * numConsidered)
    let mut numExtraVisits := numSimulations / denom
    if numExtraVisits = 0 then
      numExtraVisits := 1

    for _ in [:numExtraVisits] do
      for i in [:numConsidered] do
        sequence := sequence.push (visits.getD i 0)
      for i in [:numConsidered] do
        let v := visits.getD i 0
        visits := updateAt visits i (v + 1)

    numConsidered := Nat.max 2 (numConsidered / 2)

  return sequence.extract 0 (Nat.min sequence.size numSimulations)

/-- Table of schedules for `0..maxNumConsideredActions`. -/
def getTableOfConsideredVisits (maxNumConsideredActions numSimulations : Nat) : Array (Array Nat) :=
  (List.range (maxNumConsideredActions + 1)).toArray.map fun m =>
    getSequenceOfConsideredVisits m numSimulations

/-- The one sequential-halving schedule used by a policy invocation. -/
structure ConsideredVisitSchedule where
  numConsideredActions : Nat
  visits : Array Nat
  deriving Repr, Inhabited

/-- Count considered legal actions, treating missing mask entries as valid. -/
def countConsideredActions (maximum numActions : Nat) (invalid : Array Bool) : Nat :=
  min maximum ((List.range numActions).countP fun i => !invalid.getD i false)

def ConsideredVisitSchedule.create (numConsideredActions numSimulations : Nat) : ConsideredVisitSchedule :=
  { numConsideredActions, visits := getSequenceOfConsideredVisits numConsideredActions numSimulations }

/-- Look up a cached schedule. Direct selector callers may omit it; stale
    metadata is ignored when the legal-action count or simulation budget changes.
    The fallback constructs only the requested row, never the entire table. -/
def consideredVisitAt (cached : Option ConsideredVisitSchedule)
    (numConsideredActions numSimulations simulationIndex : Nat) : Nat :=
  match cached with
  | some schedule =>
    if schedule.numConsideredActions == numConsideredActions && schedule.visits.size == numSimulations then
      schedule.visits.getD simulationIndex 0
    else
      (getSequenceOfConsideredVisits numConsideredActions numSimulations).getD simulationIndex 0
  | none =>
    (getSequenceOfConsideredVisits numConsideredActions numSimulations).getD simulationIndex 0

/-- Score used by Gumbel MuZero root action selection. -/
def scoreConsidered
    (consideredVisit : Nat)
    (gumbel logits normalizedQvalues : Array Float)
    (visitCounts : Array Nat)
    : Array Float :=
  let lowLogit : Float := -1e9
  let logits' :=
    let m := maxD logits (logits.getD 0 0.0)
    logits.map (fun x => x - m)
  (List.range logits'.size).toArray.map fun i =>
    let penalty := if visitCounts.getD i 0 = consideredVisit then 0.0 else -1e30
    let raw := (gumbel.getD i 0.0) + (logits'.getD i 0.0) + (normalizedQvalues.getD i 0.0)
    let clipped := if raw < lowLogit then lowLogit else raw
    clipped + penalty

end torch.mctx
