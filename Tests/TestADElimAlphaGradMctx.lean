import LeanTest
import Tyr.AD.Elim

namespace Tests.ADElimAlphaGradMctx

open LeanTest
open Tyr.AD.Elim
open Tyr.AD.JaxprLike
open torch.mctxdag

private def mkEdge (src dst : Nat) (repr : String) : LocalJacEdge :=
  { src := src, dst := dst, map := { repr := Tyr.AD.Sparse.SparseMapTag.namedStr repr } }

private def approx (a b : Float) (tol : Float := 1e-8) : Bool :=
  Float.abs (a - b) < tol

private def chooseLeastVisitedAction [BEq K] [Hashable K]
    (tree : DagTree S K E) (nodeIndex : Nat) : Nat :=
  let visits := tree.childrenVisits.getD nodeIndex #[]
  if visits.isEmpty then
    0
  else
    let init : Nat × Nat := (0, visits.getD 0 0)
    let (best, _) := (List.range visits.size).foldl (init := init) fun (acc : Nat × Nat) a =>
      let c := visits.getD a acc.2
      if c < acc.2 then (a, c) else acc
    best

private def chooseLeastVisitedFeasible [BEq K] [Hashable K]
    (tree : DagTree S K E) (nodeIndex : Nat) : Nat :=
  let visits := tree.childrenVisits.getD nodeIndex #[]
  let logits := tree.childrenPriorLogits.getD nodeIndex #[]
  let feasible := (List.range visits.size).filter fun a => logits.getD a (-1.0e30) > -1.0e20
  if feasible.isEmpty then
    chooseLeastVisitedAction tree nodeIndex
  else
    let initA := feasible.getD 0 0
    let init : Nat × Nat := (initA, visits.getD initA 0)
    let (best, _) := feasible.foldl (init := init) fun (acc : Nat × Nat) a =>
      let c := visits.getD a acc.2
      if c < acc.2 then (a, c) else acc
    best

private def assertCompleteEpisode (label : String) (numVertices : Nat) (res : AlphaGradEpisodeResult) : IO Unit := do
  LeanTest.assertEqual res.actions0.size numVertices
    s!"{label} should emit exactly one action per eliminable vertex"
  LeanTest.assertEqual res.order1.size numVertices
    s!"{label} should emit exactly one vertex per eliminable vertex"
  LeanTest.assertTrue (Tyr.AD.Elim.hasNoDuplicates res.actions0)
    s!"{label} actions should form a duplicate-free permutation"
  LeanTest.assertTrue (Tyr.AD.Elim.hasNoDuplicates res.order1)
    s!"{label} vertices should form a duplicate-free permutation"
  LeanTest.assertTrue (res.finalState.violation?.isNone)
    s!"{label} should complete without violation diagnostics"

@[test]
def testReplayPreservesActionZero : IO Unit := do
  let edges : Array LocalJacEdge := #[(mkEdge 1 2 "m12"), (mkEdge 2 3 "m23")]
  let cfg : AlphaGradMctxConfig := {}

  match initAlphaGradStateFromEdges? edges 3 with
  | .error msg =>
    LeanTest.fail s!"State init should succeed, got: {msg}"
  | .ok s0 =>
    match replayActions? cfg s0 #[0, 2, 1] with
    | .error msg =>
      LeanTest.fail s!"Replay should succeed for valid action sequence, got: {msg}"
    | .ok s =>
      LeanTest.assertEqual s.actionTrace #[0, 2, 1]
        "Action trace must preserve action 0 explicitly (no sentinel ambiguity)"
      LeanTest.assertEqual s.vertexTrace #[1, 3, 2]
        "Vertex trace should convert actions via vertex = action + 1"
      LeanTest.assertTrue (s.eliminatedCount = 3)
        "All vertices should be marked eliminated after full replay"

@[test]
def testConstraintMaskingAtRoot : IO Unit := do
  let edges : Array LocalJacEdge := #[(mkEdge 1 2 "m12"), (mkEdge 2 3 "m23")]
  let cfg : AlphaGradMctxConfig := {
    constraints := { hardPrecedence := #[(1, 2)] }
  }

  match initAlphaGradStateFromEdges? edges 3 with
  | .error msg =>
    LeanTest.fail s!"State init should succeed, got: {msg}"
  | .ok s0 =>
    let mask := invalidActionMask cfg s0
    LeanTest.assertEqual mask #[false, true, false]
      "Hard precedence 1 -> 2 should invalidate action 1 at root"

    match rootFromState? cfg s0 with
    | .error msg =>
      LeanTest.fail s!"Root extraction should succeed on feasible state, got: {msg}"
    | .ok (_root, rootMask) =>
      LeanTest.assertEqual rootMask #[false, true, false]
        "Root invalid-action mask should match constraint feasibility"

@[test]
def testSearchStepRespectsHardConstraintMask : IO Unit := do
  let edges : Array LocalJacEdge := #[(mkEdge 1 2 "m12"), (mkEdge 2 3 "m23"), (mkEdge 1 3 "m13")]
  let envCfg : AlphaGradMctxConfig := {
    constraints := { hardPrecedence := #[(1, 2)] }
  }
  let mctsCfg : AlphaGradMctsConfig := {
    numSimulations := 24
    maxNumConsideredActions := 3
    gumbelScale := 0.0
  }

  match initAlphaGradStateFromEdges? edges 3 with
  | .error msg =>
    LeanTest.fail s!"State init should succeed, got: {msg}"
  | .ok s0 =>
    match searchStep? envCfg mctsCfg 7 s0 with
    | .error msg =>
      LeanTest.fail s!"searchStep? should succeed on feasible constrained state, got: {msg}"
    | .ok decision =>
      LeanTest.assertTrue (decision.action != 1)
        "Constraint mask should prevent selecting action 1 (vertex 2) before vertex 1"
      LeanTest.assertTrue (decision.actionWeights.getD 1 1.0 < 1e-8)
        s!"Masked action should have near-zero weight, got {decision.actionWeights.getD 1 0.0}"

@[test]
def testInfeasibleConstraintsFailWithoutFallback : IO Unit := do
  let edges : Array LocalJacEdge := #[(mkEdge 1 2 "m12")]
  let envCfg : AlphaGradMctxConfig := {
    constraints := { hardPrecedence := #[(1, 2), (2, 1)] }
  }
  let mctsCfg : AlphaGradMctsConfig := {
    numSimulations := 8
    maxNumConsideredActions := 2
    gumbelScale := 0.0
  }

  match searchEpisodeFromEdges? envCfg mctsCfg 11 edges 2 with
  | .ok _ =>
    LeanTest.fail "Infeasible hard constraints must fail (no fallback order allowed)"
  | .error msg =>
    LeanTest.assertTrue (msg.contains "No feasible actions")
      s!"Expected infeasible-state diagnostic, got: {msg}"

@[test]
def testSearchEpisodeProducesFullOrder : IO Unit := do
  let edges : Array LocalJacEdge := #[(mkEdge 1 2 "m12"), (mkEdge 2 3 "m23"), (mkEdge 1 3 "m13")]
  let envCfg : AlphaGradMctxConfig := {
    terminalBonus := 0.5
  }
  let mctsCfg : AlphaGradMctsConfig := {
    numSimulations := 16
    maxNumConsideredActions := 3
    gumbelScale := 0.0
  }

  match searchEpisodeFromEdges? envCfg mctsCfg 19 edges 3 with
  | .error msg =>
    LeanTest.fail s!"searchEpisodeFromEdges? should succeed, got: {msg}"
  | .ok res =>
    LeanTest.assertEqual res.actions0.size 3
      "Episode should emit exactly one action per eliminable vertex"
    LeanTest.assertEqual res.order1.size 3
      "Episode should emit exactly one vertex per eliminable vertex"
    LeanTest.assertTrue (Tyr.AD.Elim.hasNoDuplicates res.actions0)
      "Episode actions should form a duplicate-free permutation"
    LeanTest.assertTrue (Tyr.AD.Elim.hasNoDuplicates res.order1)
      "Episode vertices should form a duplicate-free permutation"
    LeanTest.assertTrue (res.finalState.violation?.isNone)
      "Successful episode should have no violation diagnostics"

    let rewardFromSteps := res.stepRewards.foldl (init := 0.0) (· + ·)
    LeanTest.assertTrue (approx rewardFromSteps res.totalReward 1e-6)
      s!"Episode totalReward should match cumulative step rewards, got steps={rewardFromSteps}, total={res.totalReward}"

@[test]
def testCommAwareRewardPenalizesCrossGroupCut : IO Unit := do
  let edges : Array LocalJacEdge := #[(mkEdge 1 2 "m12"), (mkEdge 2 3 "m23")]
  let cfg : AlphaGradMctxConfig := {
    constraints := { groups := #[#[1], #[2], #[3]] }
    costWeights := {
      alphaFlops := 0.0
      betaLocalBytes := 0.0
      gammaCommBytes := 1.0
      deltaCollectives := 0.0
      epsilonP2PMsgs := 0.0
    }
  }

  match initAlphaGradStateFromEdges? edges 3 with
  | .error msg =>
    LeanTest.fail s!"State init should succeed, got: {msg}"
  | .ok s0 =>
    let (_sMid, rMid) := applyAction cfg s0 1
    let (_sEdge, rEdge) := applyAction cfg s0 0
    LeanTest.assertTrue (rMid < rEdge)
      s!"Cross-group elimination should be more penalized, got rMid={rMid}, rEdge={rEdge}"

@[test]
def testCommAwarePriorsPenalizeCrossGroupCut : IO Unit := do
  let edges : Array LocalJacEdge := #[(mkEdge 1 2 "m12"), (mkEdge 2 3 "m23")]
  let cfg : AlphaGradMctxConfig := {
    constraints := { groups := #[#[1], #[2], #[3]] }
    costWeights := {
      alphaFlops := 0.0
      betaLocalBytes := 0.0
      gammaCommBytes := 1.0
      deltaCollectives := 0.0
      epsilonP2PMsgs := 0.0
    }
  }

  match initAlphaGradStateFromEdges? edges 3 with
  | .error msg =>
    LeanTest.fail s!"State init should succeed, got: {msg}"
  | .ok s0 =>
    let priors := heuristicPriorLogits cfg s0
    let pMid := priors.getD 1 (0.0 - 1.0e30)
    let pEdge := priors.getD 0 (0.0 - 1.0e30)
    LeanTest.assertTrue (pMid < pEdge)
      s!"Comm-aware priors should down-rank high-cut action, got pMid={pMid}, pEdge={pEdge}"

@[test]
def testCommHintContributesToReward : IO Unit := do
  let edges : Array LocalJacEdge := #[(mkEdge 1 2 "m12")]
  let cfg : AlphaGradMctxConfig := {
    constraints := {
      commHints := #[{ pattern := .allReduce, bytes := 128, collectiveCount := 2 }]
    }
    costWeights := {
      alphaFlops := 0.0
      betaLocalBytes := 0.0
      gammaCommBytes := 1.0
      deltaCollectives := 1.0
      epsilonP2PMsgs := 0.0
    }
  }

  match initAlphaGradStateFromEdges? edges 2 with
  | .error msg =>
    LeanTest.fail s!"State init should succeed, got: {msg}"
  | .ok s0 =>
    let (_s1, r1) := applyAction cfg s0 0
    let expected := -(Float.ofNat 128 + Float.ofNat 2)
    LeanTest.assertTrue (approx r1 expected 1e-6)
      s!"Comm hint penalty mismatch: got {r1}, expected {expected}"

@[test]
def testCommHintLowersHeuristicValue : IO Unit := do
  let edges : Array LocalJacEdge := #[(mkEdge 1 2 "m12")]
  let baseCfg : AlphaGradMctxConfig := {
    costWeights := {
      alphaFlops := 0.0
      betaLocalBytes := 0.0
      gammaCommBytes := 1.0
      deltaCollectives := 1.0
      epsilonP2PMsgs := 0.0
    }
  }
  let hintCfg : AlphaGradMctxConfig := {
    constraints := {
      commHints := #[{ pattern := .allReduce, bytes := 128, collectiveCount := 2 }]
    }
    costWeights := baseCfg.costWeights
  }

  match initAlphaGradStateFromEdges? edges 2 with
  | .error msg =>
    LeanTest.fail s!"State init should succeed, got: {msg}"
  | .ok s0 =>
    let vBase := heuristicValue baseCfg s0
    let vHint := heuristicValue hintCfg s0
    LeanTest.assertTrue (vHint < vBase)
      s!"Comm hints should lower non-terminal heuristic value, got withHint={vHint}, base={vBase}"

@[test]
def testSoftPrecedenceShapesPriorValueAndReward : IO Unit := do
  let edges : Array LocalJacEdge := #[(mkEdge 1 2 "m12")]
  let baseCfg : AlphaGradMctxConfig := {
    costWeights := {
      alphaFlops := 0.0
      betaLocalBytes := 0.0
      gammaCommBytes := 0.0
      deltaCollectives := 0.0
      epsilonP2PMsgs := 0.0
    }
  }
  let softCfg : AlphaGradMctxConfig := {
    constraints := { softPrecedence := #[(1, 2, 5.0)] }
    costWeights := baseCfg.costWeights
  }

  match initAlphaGradStateFromEdges? edges 2 with
  | .error msg =>
    LeanTest.fail s!"State init should succeed, got: {msg}"
  | .ok s0 =>
    let priors := heuristicPriorLogits softCfg s0
    let p1 := priors.getD 0 (0.0 - 1.0e30)
    let p2 := priors.getD 1 (0.0 - 1.0e30)
    LeanTest.assertTrue (p2 < p1)
      s!"Soft precedence should down-rank violating action in priors, got p1={p1}, p2={p2}"

    let vBase := heuristicValue baseCfg s0
    let vSoft := heuristicValue softCfg s0
    LeanTest.assertTrue (vSoft < vBase)
      s!"Soft precedence should lower non-terminal heuristic value, got soft={vSoft}, base={vBase}"

    let (_sGood, rGood) := applyAction softCfg s0 0
    let (_sBad, rBad) := applyAction softCfg s0 1
    LeanTest.assertTrue (rBad < rGood)
      s!"Soft precedence should penalize violating step reward, got bad={rBad}, good={rGood}"
    LeanTest.assertTrue (approx (rGood - rBad) 5.0 1e-6)
      s!"Soft precedence reward delta should match weight 5.0, got delta={(rGood - rBad)}"

@[test]
def testAlphaGradNopsRewardMatchesComposedPairs : IO Unit := do
  let edges : Array LocalJacEdge := #[(mkEdge 1 2 "m12"), (mkEdge 2 3 "m23")]
  let cfg : AlphaGradMctxConfig := {
    rewardMode := .alphaGradNops
  }

  match initAlphaGradStateFromEdges? edges 3 with
  | .error msg =>
    LeanTest.fail s!"State init should succeed, got: {msg}"
  | .ok s0 =>
    let (s1, r1) := applyAction cfg s0 1
    LeanTest.assertTrue (s1.violation?.isNone)
      "Valid AlphaGrad nops step should not stamp a violation."
    LeanTest.assertEqual s1.cumulativeNops 1
      "Eliminating the middle vertex of a length-2 chain should contribute one composed pair."
    LeanTest.assertTrue (approx r1 (- Float.ofNat s1.cumulativeNops) 1e-6)
      s!"AlphaGrad nops reward should equal -cumulative step nops for a single step, got reward={r1}, nops={s1.cumulativeNops}"

@[test]
def testAlphaGradNopsRewardIgnoresSoftAndCommPenalties : IO Unit := do
  let edges : Array LocalJacEdge := #[(mkEdge 1 2 "m12"), (mkEdge 2 3 "m23")]
  let baseCfg : AlphaGradMctxConfig := {
    rewardMode := .alphaGradNops
    costWeights := {
      alphaFlops := 1.0
      betaLocalBytes := 1.0
      gammaCommBytes := 1.0
      deltaCollectives := 1.0
      epsilonP2PMsgs := 1.0
    }
  }
  let hintedCfg : AlphaGradMctxConfig := {
    rewardMode := .alphaGradNops
    constraints := {
      softPrecedence := #[(1, 2, 5.0)]
      groups := #[#[1], #[2], #[3]]
      commHints := #[{ pattern := .allReduce, bytes := 128, collectiveCount := 2 }]
    }
    costWeights := baseCfg.costWeights
  }

  match initAlphaGradStateFromEdges? edges 3 with
  | .error msg =>
    LeanTest.fail s!"State init should succeed, got: {msg}"
  | .ok s0 =>
    let (_sBase, rBase) := applyAction baseCfg s0 1
    let (_sHinted, rHinted) := applyAction hintedCfg s0 1
    LeanTest.assertTrue (approx rHinted rBase 1e-6)
      s!"AlphaGrad nops reward should ignore soft/comm penalties, got base={rBase}, hinted={rHinted}"

@[test]
def testAlphaGradTerminalNopsProxyDefersRewardUntilSuccess : IO Unit := do
  let edges : Array LocalJacEdge := #[(mkEdge 1 2 "m12"), (mkEdge 2 3 "m23")]
  let cfg : AlphaGradMctxConfig := {
    rewardMode := .alphaGradTerminalNopsProxy
  }

  match initAlphaGradStateFromEdges? edges 3 with
  | .error msg =>
    LeanTest.fail s!"State init should succeed, got: {msg}"
  | .ok s0 =>
    let t1 := transition cfg s0 1
    LeanTest.assertTrue (approx t1.reward 0.0 1e-6)
      s!"Terminal nops proxy should defer reward on non-terminal steps, got {t1.reward}"
    LeanTest.assertTrue (!t1.done)
      "First terminal-proxy step on a 3-vertex chain should not terminate."
    LeanTest.assertEqual t1.nextState.cumulativeNops 1
      "Deferred terminal proxy should still track cumulative nops."

    let t2 := transition cfg t1.nextState 0
    LeanTest.assertTrue (approx t2.reward 0.0 1e-6)
      s!"Terminal nops proxy should keep intermediate rewards at zero, got {t2.reward}"
    LeanTest.assertTrue (!t2.done)
      "Second terminal-proxy step should still leave one action."

    let t3 := transition cfg t2.nextState 2
    LeanTest.assertTrue t3.done
      "Third step should terminate the episode."
    LeanTest.assertTrue (approx t3.reward (-1.0) 1e-6)
      s!"Terminal nops proxy should emit -total_nops on success, got {t3.reward}"
    LeanTest.assertEqual t3.nextState.cumulativeNops 1
      "Final cumulative nops should still reflect the one composed pair from the chain."
    LeanTest.assertTrue (approx t3.nextState.cumulativeReward (-1.0) 1e-6)
      s!"Terminal proxy cumulative reward should equal -total_nops on success, got {t3.nextState.cumulativeReward}"

@[test]
def testRewardModeChangesSameReplayTraceEvaluation : IO Unit := do
  let edges : Array LocalJacEdge := #[(mkEdge 1 2 "m12"), (mkEdge 2 3 "m23")]
  let heuristicCfg : AlphaGradMctxConfig := {
    constraints := { softPrecedence := #[(1, 2, 5.0)] }
    costWeights := {
      alphaFlops := 0.0
      betaLocalBytes := 0.0
      gammaCommBytes := 0.0
      deltaCollectives := 0.0
      epsilonP2PMsgs := 0.0
    }
  }
  let nopsCfg : AlphaGradMctxConfig := {
    rewardMode := .alphaGradNops
    constraints := heuristicCfg.constraints
    costWeights := heuristicCfg.costWeights
  }
  let actions : Array ActionId0 := #[1, 0, 2]

  match initAlphaGradStateFromEdges? edges 3 with
  | .error msg =>
    LeanTest.fail s!"State init should succeed, got: {msg}"
  | .ok s0 =>
    match replayActions? heuristicCfg s0 actions, replayActions? nopsCfg s0 actions with
    | .ok sHeuristic, .ok sNops =>
      LeanTest.assertTrue (!approx sHeuristic.cumulativeReward sNops.cumulativeReward 1e-6)
        s!"Same trace should evaluate differently under heuristic vs AlphaGrad nops rewards, got heuristic={sHeuristic.cumulativeReward}, nops={sNops.cumulativeReward}"
      LeanTest.assertTrue (approx sNops.cumulativeReward (-1.0) 1e-6)
        s!"AlphaGrad nops replay should total -1 on the chain trace, got {sNops.cumulativeReward}"
    | .error msg, _ =>
      LeanTest.fail s!"Heuristic replay should succeed, got: {msg}"
    | _, .error msg =>
      LeanTest.fail s!"AlphaGrad nops replay should succeed, got: {msg}"

@[test]
def testDagTranspositionReuseOnReconvergentElimination : IO Unit := do
  -- No edge structure needed; with 2 vertices both orders end in the same eliminated set.
  let edges : Array LocalJacEdge := #[]
  let envCfg : AlphaGradMctxConfig := {}
  let mctsCfg : AlphaGradMctsConfig := {
    numSimulations := 4
    maxDepth := some 2
    dagDirichletFraction := 0.0
  }

  match initAlphaGradStateFromEdges? edges 2 with
  | .error msg =>
    LeanTest.fail s!"State init should succeed, got: {msg}"
  | .ok s0 =>
    match rootFromState? envCfg s0 with
    | .error msg =>
      LeanTest.fail s!"rootFromState? should succeed, got: {msg}"
    | .ok (root, invalid) =>
      let tree0 := instantiateDagTreeFromRoot root (dagStateKey s0) mctsCfg.numSimulations invalid ()
      let rootFn : RootActionSelectionFn AlphaGradState AlphaGradDagKey Unit := fun _ tree nodeIndex =>
        chooseLeastVisitedFeasible tree nodeIndex
      let interiorFn : InteriorActionSelectionFn AlphaGradState AlphaGradDagKey Unit := fun _ tree nodeIndex _ =>
        chooseLeastVisitedFeasible tree nodeIndex

      let tree := searchWithDag
        (params := envCfg)
        (rngKey := 23)
        (tree := tree0)
        (recurrentFn := recurrentFn)
        (keyFn := dagStateKey)
        (rootActionSelectionFn := rootFn)
        (interiorActionSelectionFn := interiorFn)
        (numSimulations := mctsCfg.numSimulations)
        (maxDepth := mctsCfg.maxDepth)

      -- Expected merged DAG:
      -- root + two single-elimination states + one shared full-elimination state.
      LeanTest.assertEqual tree.numAllocated 4
        s!"Expected transposition merge to allocate 4 nodes, got {tree.numAllocated}"

@[test]
def testSearchEpisodeDagPolicySelectableEntrypoint : IO Unit := do
  let edges : Array LocalJacEdge := #[(mkEdge 1 2 "m12"), (mkEdge 2 3 "m23"), (mkEdge 1 3 "m13")]
  let envCfg : AlphaGradMctxConfig := {}
  let mctsCfg : AlphaGradMctsConfig := {
    numSimulations := 12
    maxNumConsideredActions := 3
    dagDirichletFraction := 0.0
    dagTemperature := 1.0
    gumbelScale := 1.0
  }

  match searchEpisodeDagWithPolicyFromEdges? .alphaZero envCfg mctsCfg 31 edges 3 with
  | .error msg =>
    LeanTest.fail s!"alphaZero DAG policy entrypoint should succeed, got: {msg}"
  | .ok res =>
    assertCompleteEpisode "alphaZero DAG policy entrypoint" 3 res

  match searchEpisodeDagWithPolicyFromEdges? .gumbelMuZero envCfg mctsCfg 32 edges 3 with
  | .error msg =>
    LeanTest.fail s!"gumbelMuZero DAG policy entrypoint should succeed, got: {msg}"
  | .ok res =>
    assertCompleteEpisode "gumbelMuZero DAG policy entrypoint" 3 res

@[test]
def testSearchStepDagGumbelRespectsConstraintMask : IO Unit := do
  let edges : Array LocalJacEdge := #[(mkEdge 1 2 "m12"), (mkEdge 2 3 "m23"), (mkEdge 1 3 "m13")]
  let envCfg : AlphaGradMctxConfig := {
    constraints := { hardPrecedence := #[(1, 2)] }
  }
  let mctsCfg : AlphaGradMctsConfig := {
    numSimulations := 24
    maxNumConsideredActions := 3
    gumbelScale := 1.0
  }

  match initAlphaGradStateFromEdges? edges 3 with
  | .error msg =>
    LeanTest.fail s!"State init should succeed, got: {msg}"
  | .ok s0 =>
    match searchStepDagGumbel? envCfg mctsCfg 37 s0 with
    | .error msg =>
      LeanTest.fail s!"searchStepDagGumbel? should succeed, got: {msg}"
    | .ok decision =>
      LeanTest.assertTrue (decision.action != 1)
        "DAG Gumbel mask should prevent selecting action 1 (vertex 2) before vertex 1"
      LeanTest.assertTrue (decision.actionWeights.getD 1 1.0 < 1e-8)
        s!"Masked action should have near-zero weight, got {decision.actionWeights.getD 1 0.0}"

@[test]
def testDagStateKeyIgnoresMapReprForEquivalentSparseMaps : IO Unit := do
  let mapA : SparseLinearMap := {
    repr := Tyr.AD.Sparse.SparseMapTag.namedStr "semantic-a"
    inDim? := some 4
    outDim? := some 3
    entries := #[
      { src := 1, dst := 2, weight := 0.5 },
      { src := 0, dst := 1, weight := 1.0 }
    ]
  }
  let mapB : SparseLinearMap := {
    repr := Tyr.AD.Sparse.SparseMapTag.namedStr "semantic-b"
    inDim? := some 4
    outDim? := some 3
    entries := #[
      { src := 0, dst := 1, weight := 1.0 },
      { src := 1, dst := 2, weight := 0.5 }
    ]
  }
  let gA := insertEdge ({} : ElimGraph) 1 2 mapA
  let gB := insertEdge ({} : ElimGraph) 1 2 mapB
  let sA : AlphaGradState := {
    graph := gA
    numVertices := 2
    actionVertices := #[1, 2]
    eliminatedActions := #[false, false]
  }
  let sB : AlphaGradState := {
    graph := gB
    numVertices := 2
    actionVertices := #[1, 2]
    eliminatedActions := #[false, false]
  }
  LeanTest.assertTrue (dagStateKey sA == dagStateKey sB)
    "DAG state key should ignore sparse-map repr and canonicalize by shape+entries."

@[test]
def testPartitionedGraphMasksOutputsAndTerminatesOnEliminableCount : IO Unit := do
  let edges : Array LocalJacEdge := #[(mkEdge 1 2 "m12")]
  let graphRes := ofLocalJacEdgesWithPartitions edges #[] #[2] #[1]
  match graphRes with
  | .error msg =>
    LeanTest.fail s!"Partitioned graph construction should succeed, got: {msg}"
  | .ok graph =>
    match initAlphaGradState? graph 2 with
    | .error msg =>
      LeanTest.fail s!"State init should succeed on partitioned graph, got: {msg}"
    | .ok s0 =>
      LeanTest.assertEqual s0.numActions 2
        "Fixed action slots should still cover the full vertex domain."
      LeanTest.assertEqual s0.numEliminableActions 1
        "Only the declared eliminable vertex should count toward rollout length."
      LeanTest.assertEqual s0.outputActionMask #[false, true]
        "Output slot should be marked in the output-action mask."
      LeanTest.assertEqual (invalidActionMask {} s0) #[false, true]
        "Output vertex should be invalid from the root mask in fixed-slot compatibility mode."

      let t := transition {} s0 0
      LeanTest.assertTrue t.done
        "Eliminating the single eliminable vertex should terminate even when output slots remain masked."
      LeanTest.assertEqual t.nextState.actionTrace #[0]
        "Action trace should record only the eliminable action."
      LeanTest.assertEqual t.nextState.vertexTrace #[1]
        "Vertex trace should record the eliminable vertex."
      LeanTest.assertTrue (t.nextState.violation?.isNone)
        "Partitioned fixed-slot transition should not stamp a violation."

@[test]
def testExplicitActionSpaceSubsetCompatibility : IO Unit := do
  let edges : Array LocalJacEdge := #[(mkEdge 1 2 "m12"), (mkEdge 2 4 "m24")]
  let cfg : AlphaGradMctxConfig := {
    actionSpace := .explicitVertices #[2, 4]
  }

  match initAlphaGradStateFromEdges? edges 4 #[] (some #[2, 4]) with
  | .error msg =>
    LeanTest.fail s!"State init should succeed for explicit action-space subset, got: {msg}"
  | .ok s0 =>
    LeanTest.assertEqual s0.numActions 2
      "Explicit action-space mode should size action IDs by configured subset."
    LeanTest.assertEqual s0.actionVertices #[2, 4]
      "Action-space vertex table should be preserved in state."

    let mask := invalidActionMask cfg s0
    LeanTest.assertEqual mask #[false, false]
      "Explicit action-space subset should produce a mask sized by subset cardinality."

    let (s1, _r1) := applyAction cfg s0 0
    LeanTest.assertEqual s1.vertexTrace #[2]
      "Action 0 should map to first explicit action-space vertex."

    match replayActions? cfg s0 #[0, 1] with
    | .error msg =>
      LeanTest.fail s!"Replay should support explicit action-space IDs, got: {msg}"
    | .ok sDone =>
      LeanTest.assertEqual sDone.vertexTrace #[2, 4]
        "Replay should map explicit action IDs through the configured action-space table."

    match replayActions? cfg s0 #[0, 2] with
    | .ok _ =>
      LeanTest.fail "Out-of-range action IDs must fail under explicit action-space sizing."
    | .error msg =>
      LeanTest.assertTrue (msg.contains "Invalid ActionId0 2")
        s!"Expected out-of-range action diagnostic for explicit action-space, got: {msg}"

end Tests.ADElimAlphaGradMctx
