import Tyr
import Tyr.Manifolds.Orthogonal
import Tyr.Manifolds.Grassmann
import Tyr.Model.BranchingFlows
import Tyr.Model.BranchingFlows.DiffEq
import Tyr.Model.BranchingFlows.Discrete
import Tyr.Model.BranchingFlows.Molecule
import Tyr.Model.BranchingFlows.QM9
import Tyr.Model.BranchingFlows.MoleculeTrain
import Tyr.Model.BranchingFlows.MoleculeTransformer
import Tyr.Model.BranchingFlowsTrain
import LeanTest

open torch
open torch.branching
open torch.DiffEq
open Tyr.EventSkeleton
open Tyr.AD

private def clamp01 (x : Float) : Float :=
  max 0.0 (min x 1.0)

private def uniformTimeDist : TimeDist :=
  { cdf := fun t => clamp01 t,
    pdf := fun t => if t < 0.0 || t > 1.0 then 0.0 else 1.0,
    quantile := fun p => clamp01 p }

private def noEventTimeDist : TimeDist :=
  { cdf := fun _ => 0.0,
    pdf := fun _ => 0.0,
    quantile := fun _ => 0.0 }

private def sumFloatArray (xs : Array Float) : Float :=
  xs.foldl (fun acc x => acc + x) 0.0

private def expectOk {α : Type} (what : String) (r : Except String α) : IO α := do
  match r with
  | .ok x => pure x
  | .error e => LeanTest.fail s!"{what}: {e}"

private def expectError {α : Type} (what : String) (r : Except String α) : IO String := do
  match r with
  | .ok _ => LeanTest.fail s!"{what}: expected error"
  | .error e => pure e

private def nearSureEventTimeDist : TimeDist :=
  { cdf := fun _ => 0.0,
    pdf := fun _ => 1.0e9,
    quantile := fun _ => 0.0 }

private def mkState (vals : Array Nat) (groups : Array Int) : BranchingState Nat :=
  let n := vals.size
  { state := vals,
    groupings := groups,
    del := Array.replicate n false,
    ids := (Array.range n).map (fun i => Int.ofNat (i + 1)),
    branchmask := Array.replicate n true,
    flowmask := Array.replicate n true,
    padmask := Array.replicate n true }

private def mkPolicyNode (value : Float) (weight : Nat) : FlowNode Float :=
  { (FlowNode.leaf 1.0 value 0 true false true (Int.ofNat weight)) with
    weight := weight }

private def expectSelectedPair (what : String) (selected : Option (Nat × Nat)) (expected : Nat × Nat) :
    IO Unit := do
  match selected with
  | none => LeanTest.fail s!"{what}: expected {expected}, got none"
  | some got =>
      LeanTest.assertEqual got.1 expected.1 s!"{what}: selected left index"
      LeanTest.assertEqual got.2 expected.2 s!"{what}: selected right index"

private def countTrue (xs : Array Bool) : Nat :=
  xs.foldl (fun acc x => if x then acc + 1 else acc) 0

@[test]
def testBranchingSampleForestSimple : IO Unit := do
  let elements : Array Nat := #[1, 2]
  let groupings : Array Int := #[0, 0]
  let branchable : Array Bool := #[true, true]
  let flowable : Array Bool := #[true, true]
  let deleted : Array Bool := #[false, false]
  let ids : Array Int := #[1, 2]
  let merger := fun (a b : Nat) (_w1 _w2 : Nat) => a + b
  let (roots, times, _rng) :=
    sampleForest elements groupings branchable flowable deleted ids uniformTimeDist
      (sequentialUniformPolicy Nat) 1.0 merger none { state := 1 }
  LeanTest.assertEqual roots.size 1 "Forest collapses to one root"
  LeanTest.assertEqual times.size 1 "Two leaves yield one split time"
  let t := times[0]!
  LeanTest.assertTrue (t >= 0.0 && t <= 1.0) "Split time in [0,1]"
  let root := roots[0]!
  LeanTest.assertEqual root.weight 2 "Merged weight is 2"
  LeanTest.assertEqual root.group 0 "Group preserved"
  LeanTest.assertEqual root.children.size 2 "Root has two children"

@[test]
def testBranchingRichGetRicherSequentialPolicy : IO Unit := do
  let nodes := #[
    mkPolicyNode 0.0 1,
    mkPolicyNode 1.0 1,
    mkPolicyNode 2.0 1000
  ]
  let policy := richGetRicherSequentialPolicy Float (alpha := 4.0)
  let (selected, _rng) := policy.select nodes none { state := 1 }
  expectSelectedPair "Rich-get-richer policy should choose the much heavier adjacent pair"
    selected (1, 2)

@[test]
def testBranchingSequentialProximityPolicy : IO Unit := do
  let nodes := #[
    mkPolicyNode 0.0 1,
    mkPolicyNode 10.0 1,
    mkPolicyNode 10.1 1
  ]
  let policy := sequentialProximityPolicy Float (fun a b => Float.abs (a - b))
  let (selected, _rng) := policy.select nodes none { state := 2 }
  expectSelectedPair "Proximity policy should choose the closest adjacent features"
    selected (1, 2)

@[test]
def testBranchingSequentialDeepLineagePolicy : IO Unit := do
  let nodes := #[
    mkPolicyNode 0.0 1,
    mkPolicyNode 1.0 1,
    mkPolicyNode 2.0 3
  ]
  let policy := sequentialDeepLineagePolicy Float (minCount := 2) (targetTrunks := 1)
  let (selected, _rng) := policy.select nodes none { state := 3 }
  expectSelectedPair "Deep-lineage policy should require an existing deep lineage once active"
    selected (1, 2)

@[test]
def testBranchingFixedcountInsertionsLength : IO Unit := do
  let x := mkState #[10, 20, 30] #[0, 0, 1]
  let (x', _rng) := fixedcountDelInsertions x 2 { state := 42 }
  LeanTest.assertEqual x'.state.size (x.state.size + 2) "Adds exactly numEvents elements"
  LeanTest.assertEqual x'.del.size x'.state.size "Deletion flags match length"
  LeanTest.assertEqual x'.groupings.size x'.state.size "Groupings match length"
  LeanTest.assertEqual x'.ids.size x'.state.size "Ids match length"
  LeanTest.assertEqual x'.branchmask.size x'.state.size "Branchmask match length"
  LeanTest.assertEqual x'.flowmask.size x'.state.size "Flowmask match length"
  LeanTest.assertEqual x'.padmask.size x'.state.size "Padmask match length"

@[test]
def testBranchingDeletionPadFloorSemantics : IO Unit := do
  let x1 := mkState #[10, 20, 30] #[0, 0, 0]
  let bridge (_ : Unit) (_x0 x1 : Nat) (_t0 _t1 : Float) : Nat := x1
  let x0Sampler (_root : FlowNode Nat) : Nat := 0
  let merger (a _b : Nat) (_w1 _w2 : Nat) : Nat := a
  let run (deletionPad : Float) (seed : UInt64) :=
    branchingBridge bridge () x0Sampler #[x1] #[0.0]
      noEventTimeDist noEventTimeDist (sequentialUniformPolicy Nat) merger
      (coalescenceFactor := 0.0)
      (lengthMins := .uniform 10)
      (deletionPad := deletionPad)
      (rng := { state := seed })

  let (noPad, _rng) := run 0.0 11
  LeanTest.assertEqual noPad.Xt[0]!.state.size 3
    "deletionPad=0 should not pad solely because lengthMins is larger"
  LeanTest.assertEqual (countTrue noPad.del[0]!) 0
    "deletionPad=0 should not mark deletion targets"

  let (floorPad, _rng) := run 1.0 11
  LeanTest.assertEqual floorPad.Xt[0]!.state.size 10
    "deletionPad=1 should deterministically pad to the group length floor"
  LeanTest.assertEqual (countTrue floorPad.del[0]!) 7
    "deletionPad=1 should create exactly the floor deficit as deletion targets"

  let (extraPad, _rng) := run 2.0 11
  LeanTest.assertTrue (extraPad.Xt[0]!.state.size >= floorPad.Xt[0]!.state.size)
    "deletionPad>1 should preserve the deterministic floor"
  LeanTest.assertTrue (extraPad.Xt[0]!.state.size > 10)
    s!"deletionPad>1 should add stochastic excess for this seed, got length {extraPad.Xt[0]!.state.size}"

@[test]
def testBranchingTimeDistDefaults : IO Unit := do
  LeanTest.assertTrue (Float.abs ((TimeDist.uniform).quantile 0.25 - 0.25) < 1.0e-12)
    "Uniform time distribution should invert directly"
  let q13 := (TimeDist.betaOneThreeHalves).quantile ((TimeDist.betaOneThreeHalves).cdf 0.25)
  LeanTest.assertTrue (Float.abs (q13 - 0.25) < 1.0e-10)
    s!"Beta(1,3/2) quantile should invert cdf, got {q13}"
  let q22 := (TimeDist.betaTwoTwo).quantile 0.5
  LeanTest.assertTrue (Float.abs (q22 - 0.5) < 1.0e-8)
    s!"Beta(2,2) median should be near 0.5, got {q22}"
  LeanTest.assertTrue ((TimeDist.betaOneThreeHalves).hazard 0.5 > 0.0)
    "Beta split-time distribution should expose a positive hazard inside support"

@[test]
def testBranchingLossHelpers : IO Unit := do
  LeanTest.assertTrue (Float.abs (defaultSplitTransform 0.0 - 1.0) < 1.0e-12)
    "Default split transform should map zero logits to unit intensity"
  LeanTest.assertTrue (Float.abs (splitCountLoss defaultSplitTransform 0.0 1) < 1.0e-12)
    "Unit predicted intensity should have zero shifted Poisson loss for count one"
  LeanTest.assertTrue (Float.abs (splitCountLoss defaultSplitTransform 0.0 0 - 1.0) < 1.0e-12)
    "Unit predicted intensity should pay loss one for count zero"
  let splitLoss := maskedSplitCountLoss defaultSplitTransform #[0.0, 0.0] #[1, 0] #[true, false]
  LeanTest.assertTrue (Float.abs splitLoss < 1.0e-12)
    s!"Masked split loss should ignore inactive positions, got {splitLoss}"
  let bce := logitBinaryCrossEntropy 0.0 true
  LeanTest.assertTrue (Float.abs (bce - Float.log 2.0) < 1.0e-12)
    s!"Zero-logit deletion BCE should equal log(2), got {bce}"
  let delLoss := maskedDeletionLoss #[0.0, 100.0] #[true, false] #[true, false]
  LeanTest.assertTrue (Float.abs (delLoss - Float.log 2.0) < 1.0e-12)
    s!"Masked deletion loss should ignore inactive positions, got {delLoss}"

@[test]
def testBranchingTensorTrainingLosses : IO Unit := do
  let logits : Tensor { dtype := .Float32, shape := #[1, 2], device := .CPU } := Tensor.reshape (floatsDynamic #[0.0, 0.0]) #[1, 2]
  let splitTarget : Tensor { dtype := .Float32, shape := #[1, 2], device := .CPU } := Tensor.reshape (floatsDynamic #[1.0, 0.0]) #[1, 2]
  let firstOnly : Tensor { dtype := .Float32, shape := #[1, 2], device := .CPU } := Tensor.reshape (floatsDynamic #[1.0, 0.0]) #[1, 2]
  let splitLoss := Tensor.item (maskedSplitPoissonLoss logits splitTarget firstOnly)
  LeanTest.assertTrue (Float.abs splitLoss < 1.0e-6)
    s!"Unit split intensity should have zero tensor Poisson loss for count one, got {splitLoss}"
  let delTarget : Tensor { dtype := .Float32, shape := #[1, 2], device := .CPU } := Tensor.reshape (floatsDynamic #[1.0, 0.0]) #[1, 2]
  let delLoss := Tensor.item (maskedBCEWithLogits logits delTarget firstOnly)
  LeanTest.assertTrue (Float.abs (delLoss - Float.log 2.0) < 1.0e-6)
    s!"Zero-logit tensor deletion BCE should equal log(2), got {delLoss}"

private def targetStep (_ : Unit) (x : BranchingState Nat) (targets : Array Nat)
    (_s1 _s2 : Float) : Array Nat :=
  (Array.range x.state.size).map (fun i => targets.getD i (x.state.getD i 0))

@[test]
def testBranchingStepAppliesBaseStepAndSplitsAdjacently : IO Unit := do
  let flow : CoalescentFlow Unit Nat :=
    CoalescentFlow.mkDefault () uniformTimeDist noEventTimeDist (fun _ => 30.0)
  let x := mkState #[1] #[0]
  let prediction : BranchingStepPrediction Nat := {
    targets := #[10]
    splitLogits := #[0.0]
    delLogits := #[-100.0]
  }
  let (result, _rng) := branchingStep targetStep flow x prediction 0.0 1.0 (rng := { state := 7 })
  LeanTest.assertTrue (result.state.state.size > 1)
    s!"High split intensity should create adjacent duplicates, got length {result.state.state.size}"
  LeanTest.assertTrue (result.state.state.all (fun v => v == 10))
    s!"All generated elements should use the base-step target, got {result.state.state}"
  LeanTest.assertEqual result.state.groupings.size result.state.state.size
    "Generated groupings should match state length"
  LeanTest.assertEqual result.state.branchmask.size result.state.state.size
    "Generated branchmask should match state length"
  LeanTest.assertTrue (result.events.size == 1 && result.events[0]!.splitCount > 0)
    s!"Split event should be recorded, got {reprStr result.events}"

@[test]
def testBranchingStepDeletionKeepsNonemptyState : IO Unit := do
  let flow : CoalescentFlow Unit Nat := {
    base := ()
    branchTime := uniformTimeDist
    splitTransform := fun _ => 0.0
    policy := sequentialUniformPolicy Nat
    deletionTime := nearSureEventTimeDist
  }
  let x := mkState #[1, 2] #[0, 0]
  let prediction : BranchingStepPrediction Nat := {
    targets := #[11, 22]
    splitLogits := #[0.0, 0.0]
    delLogits := #[100.0, 100.0]
  }
  let (result, _rng) := branchingStep targetStep flow x prediction 0.0 1.0 (rng := { state := 9 })
  LeanTest.assertEqual result.state.state.size 1
    "Deletion step should remove elements but preserve a nonempty state"
  LeanTest.assertTrue (result.state.state[0]! == 11 || result.state.state[0]! == 22)
    s!"Surviving element should still be base-stepped, got {result.state.state}"
  LeanTest.assertTrue (result.events.size == 1 && result.events[0]!.deleted)
    s!"Exactly one committed deletion should be recorded after the nonempty guard, got {reprStr result.events}"

@[test]
def testBranchingGenerateRunsSchedule : IO Unit := do
  let flow : CoalescentFlow Unit Nat := {
    base := ()
    branchTime := uniformTimeDist
    splitTransform := fun _ => 0.0
    policy := sequentialUniformPolicy Nat
    deletionTime := noEventTimeDist
  }
  let model (_t : Float) (state : BranchingState Nat) : BranchingStepPrediction Nat := {
    targets := state.state.map (fun v => v + 1)
    splitLogits := Array.replicate state.state.size 0.0
    delLogits := Array.replicate state.state.size (-100.0)
  }
  let (result, _rng) :=
    branchingGenerate targetStep flow (mkState #[3, 4] #[0, 0]) model #[0.0, 0.5, 1.0]
      (rng := { state := 11 })
  LeanTest.assertEqual result.finalState.state #[5, 6]
    s!"Two generation steps should apply the model twice, got {result.finalState.state}"
  LeanTest.assertEqual result.trajectory.size 3
    "Trajectory should include initial state plus one state per schedule interval"
  LeanTest.assertEqual result.events.size 2
    "Event batches should align with schedule intervals"

@[test]
def testBranchingDiffEqODEBridge : IO Unit := do
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Control := Time) (Args := Unit)
  let cfg : ODEBridgeConfig Float Unit ConstantStepSize :=
    ODEBridgeConfig.mk
      (Y := Float)
      (Args := Unit)
      (Controller := ConstantStepSize)
      (fun anchor _t y _ => anchor - y)
      solver
      ()
      ({} : ConstantStepSize)
      (dt0 := some 0.25)
  let bridgedFinal := DiffEqBridgeConfig.bridge cfg 0.0 1.0 0.0 1.0
  LeanTest.assertTrue (bridgedFinal > 0.6 && bridgedFinal < 1.0)
    s!"ODE bridge should move toward the endpoint anchor, got {bridgedFinal}"
  let bridgedHalf := DiffEqBridgeConfig.bridge cfg 0.0 1.0 0.0 0.5

  let x1 : BranchingState Float := BranchingState.mkDefault #[1.0] #[0]
  let (result, _rng) :=
    branchingBridge
      (fun cfg x0 x1 t0 t1 => DiffEqBridgeConfig.bridge cfg x0 x1 t0 t1)
      cfg
      (fun _root => 0.0)
      #[x1]
      #[0.5]
      uniformTimeDist
      uniformTimeDist
      (sequentialUniformPolicy Float)
      canonicalAnchorMerge
      (coalescenceFactor := 0.0)
      (rng := { state := 123 })
  LeanTest.assertEqual result.Xt.size 1 "Expected one bridged batch"
  LeanTest.assertEqual result.Xt[0]!.state.size 1 "Expected one bridged segment"
  LeanTest.assertTrue (Float.abs (result.Xt[0]!.state[0]! - bridgedHalf) < 1.0e-12)
    s!"Branching bridge should use the DiffEq bridge result, got {result.Xt[0]!.state[0]!}"

@[test]
def testBranchingDiffEqSDEBridge : IO Unit := do
  let bm : ScalarBrownianPath := { t0 := 0.0, t1 := 1.0, seed := 991 }
  let bmPath := (ScalarBrownianPath.toAbstract bm).toPath
  let drift : ODETerm Float Unit := { vectorField := fun _t _y _ => 0.0 }
  let diffusion : ControlTerm Float Float Float Unit :=
    ControlTerm.ofPath (fun _t _y _ => 1.0) bmPath (fun vf control => vf * control)
  let solver :=
    EulerMaruyama.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := ControlTerm Float Float Float Unit)
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Control := Float)
      (Args := Unit)
  let cfg :
      SDEBridgeConfig
        (ODETerm Float Unit)
        (ControlTerm Float Float Float Unit)
        Float
        Float
        Float
        Float
        Unit
        ConstantStepSize :=
    SDEBridgeConfig.mk
      (Drift := ODETerm Float Unit)
      (Diffusion := ControlTerm Float Float Float Unit)
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Control := Float)
      (Args := Unit)
      (Controller := ConstantStepSize)
      (fun _anchor => drift)
      (fun _anchor => diffusion)
      solver
      ()
      ({} : ConstantStepSize)
      (dt0 := some 1.0)
  let bridged := DiffEqBridgeConfig.bridge cfg 0.0 0.0 0.0 1.0
  let expected := (ScalarBrownianPath.increment bm 0.0 1.0).W
  LeanTest.assertTrue (Float.abs (bridged - expected) < 1.0e-12)
    s!"SDE bridge should reproduce the deterministic Brownian increment, got {bridged}, expected {expected}"

@[test]
def testBranchingOUBridgeBrownianLimit : IO Unit := do
  let cfg := OUBridgeConfig.constantVariance 0.0 1.0
  let mean := OUBridgeConfig.bridgeMean cfg 0.0 10.0 0.0 0.5
  LeanTest.assertTrue (Float.abs (mean - 5.0) < 1.0e-12)
    s!"Zero-theta OU bridge should reduce to Brownian linear interpolation, got {mean}"
  let variance := OUBridgeConfig.bridgeVariance cfg 0.0 0.5
  LeanTest.assertTrue (Float.abs (variance - 0.25) < 1.0e-12)
    s!"Zero-theta OU bridge variance should reduce to Brownian bridge variance, got {variance}"
  let endpoint := OUBridgeConfig.bridgeMean cfg 0.0 10.0 0.0 1.0
  LeanTest.assertTrue (Float.abs (endpoint - 10.0) < 1.0e-12)
    s!"OU bridge should hit the terminal anchor, got {endpoint}"

@[test]
def testBranchingOUBridgeMeanMatchesGaussianConditioning : IO Unit := do
  let cfg := OUBridgeConfig.constantVariance 1.0 1.0
  let got := OUBridgeConfig.bridgeMean cfg 0.0 1.0 0.0 0.5
  let v0t := (1.0 - Float.exp (-1.0)) / 2.0
  let v0T := (1.0 - Float.exp (-2.0)) / 2.0
  let expected := Float.exp (-0.5) * v0t / v0T
  LeanTest.assertTrue (Float.abs (got - expected) < 1.0e-4)
    s!"OU bridge mean should match Gaussian conditioning, got {got}, expected {expected}"

@[test]
def testBranchingOUBridgeIntegratesWithBranchingBridge : IO Unit := do
  let cfg := OUBridgeConfig.constantVariance 0.0 1.0
  let x1 : BranchingState Float := BranchingState.mkDefault #[1.0] #[0]
  let expected := OUBridgeConfig.bridge cfg 0.0 1.0 0.0 0.5
  let (result, _rng) :=
    branchingBridge
      (fun cfg x0 x1 t0 t => OUBridgeConfig.bridge cfg x0 x1 t0 t)
      cfg
      (fun _root => 0.0)
      #[x1]
      #[0.5]
      uniformTimeDist
      noEventTimeDist
      (sequentialUniformPolicy Float)
      canonicalAnchorMerge
      (coalescenceFactor := 0.0)
      (rng := { state := 321 })
  LeanTest.assertEqual result.Xt.size 1 "Expected one bridged batch"
  LeanTest.assertEqual result.Xt[0]!.state.size 1 "Expected one OU-bridged segment"
  LeanTest.assertTrue (Float.abs (result.Xt[0]!.state[0]! - expected) < 1.0e-12)
    s!"Branching bridge should use the analytic OU bridge result, got {result.Xt[0]!.state[0]!}, expected {expected}"

@[test]
def testBranchingDistNoisyDiscreteSchedule : IO Unit := do
  let cfg := DistNoisyDiscreteConfig.qm9 10 9
  let (w10, w20, w30) := cfg.weights 0.0
  LeanTest.assertTrue (Float.abs w10 < 1.0e-12 && Float.abs w20 < 1.0e-12)
    s!"DFM t=0 should have no target/uniform mass, got {(w10, w20, w30)}"
  LeanTest.assertTrue (Float.abs (w30 - 1.0) < 1.0e-12)
    s!"DFM t=0 should stay at source, got source weight {w30}"
  let (w11, w21, w31) := cfg.weights 1.0
  LeanTest.assertTrue (Float.abs (w11 - 1.0) < 1.0e-12)
    s!"DFM t=1 should hit target, got target weight {w11}"
  LeanTest.assertTrue (Float.abs w21 < 1.0e-12 && Float.abs w31 < 1.0e-12)
    s!"DFM t=1 should remove uniform/source mass, got {(w11, w21, w31)}"
  let (wm1, wm2, wm3) := cfg.weights 0.5
  LeanTest.assertTrue (Float.abs (wm1 - 0.5) < 1.0e-12)
    s!"Beta(2,2) midpoint should give target mass 0.5, got {wm1}"
  LeanTest.assertTrue (Float.abs (wm2 - 0.05) < 1.0e-12)
    s!"QM9 DFM midpoint should give uniform mass 0.05, got {wm2}"
  LeanTest.assertTrue (Float.abs (wm3 - 0.45) < 1.0e-12)
    s!"QM9 DFM midpoint should leave source mass 0.45, got {wm3}"

@[test]
def testBranchingDistNoisyDiscreteConditionalBridgeAndStep : IO Unit := do
  let cfg := DistNoisyDiscreteConfig.qm9 10 9
  let (sameTarget, sameUniform, sameSource) := cfg.conditionalWeights 0.25 0.25
  LeanTest.assertTrue (Float.abs sameTarget < 1.0e-12 && Float.abs sameUniform < 1.0e-12)
    s!"Conditional DFM should not move when t0=t, got {(sameTarget, sameUniform, sameSource)}"
  LeanTest.assertTrue (Float.abs (sameSource - 1.0) < 1.0e-12)
    s!"Conditional DFM should keep current source when t0=t, got {sameSource}"
  LeanTest.assertEqual (cfg.modeBridgeFrom 9 2 0.0 0.0) 9
    "Deterministic DFM representative should use source at t=0"
  LeanTest.assertEqual (cfg.modeBridgeFrom 9 2 0.0 1.0) 2
    "Deterministic DFM representative should use target at t=1"

  let logits := (Array.replicate 10 (-10.0)).set! 2 10.0
  let probs := cfg.stepDistribution 9 logits 0.5 0.6
  LeanTest.assertEqual probs.size 10 "DFM Euler step should preserve vocabulary size"
  LeanTest.assertTrue (Float.abs (sumFloatArray probs - 1.0) < 1.0e-12)
    s!"DFM Euler step should normalize probabilities, got {sumFloatArray probs}"
  LeanTest.assertTrue (probs[2]! > 0.25)
    s!"DFM Euler step should move probability toward model target, got target prob {probs[2]!}"
  LeanTest.assertTrue (probs[9]! > 0.5)
    s!"DFM Euler step should retain source/mask mass for a small step, got mask prob {probs[9]!}"

@[test]
def testBranchingMoleculeAnchorMergeMasksLabelAndAveragesCoords : IO Unit := do
  let cfg : MoleculeBridgeConfig := {
    coordOU := OUBridgeConfig.constantVariance 0.0 1.0
    maskToken := 99
  }
  let a : MoleculeAtom := { coord := { x := 0.0, y := 2.0, z := 4.0 }, label := 6 }
  let b : MoleculeAtom := { coord := { x := 10.0, y := 20.0, z := 30.0 }, label := 8 }
  let merged := MoleculeBridgeConfig.anchorMerge cfg a b 1 3
  LeanTest.assertTrue (Float.abs (merged.coord.x - 7.5) < 1.0e-12)
    s!"Molecule anchor x coordinate should be weighted, got {merged.coord.x}"
  LeanTest.assertTrue (Float.abs (merged.coord.y - 15.5) < 1.0e-12)
    s!"Molecule anchor y coordinate should be weighted, got {merged.coord.y}"
  LeanTest.assertTrue (Float.abs (merged.coord.z - 23.5) < 1.0e-12)
    s!"Molecule anchor z coordinate should be weighted, got {merged.coord.z}"
  LeanTest.assertEqual merged.label 99
    "Molecule anchor labels should be replaced with the mask token"

@[test]
def testBranchingMoleculeBridgeLiftsOUOverCoordinates : IO Unit := do
  let cfg : MoleculeBridgeConfig := {
    coordOU := OUBridgeConfig.constantVariance 0.0 1.0
    maskToken := 99
  }
  let x0 := MoleculeBridgeConfig.maskedAtom cfg { x := 0.0, y := 0.0, z := 0.0 }
  let x1 : MoleculeAtom := { coord := { x := 10.0, y := 20.0, z := 30.0 }, label := 6 }
  let half := MoleculeBridgeConfig.bridge cfg x0 x1 0.0 0.5
  LeanTest.assertTrue (Float.abs (half.coord.x - 5.0) < 1.0e-12)
    s!"Molecule OU bridge x coordinate should interpolate, got {half.coord.x}"
  LeanTest.assertTrue (Float.abs (half.coord.y - 10.0) < 1.0e-12)
    s!"Molecule OU bridge y coordinate should interpolate, got {half.coord.y}"
  LeanTest.assertTrue (Float.abs (half.coord.z - 15.0) < 1.0e-12)
    s!"Molecule OU bridge z coordinate should interpolate, got {half.coord.z}"
  LeanTest.assertEqual half.label 99
    "Intermediate molecule label should stay masked when no DFM label bridge is configured"
  let terminal := MoleculeBridgeConfig.bridge cfg x0 x1 0.0 1.0
  LeanTest.assertEqual terminal.label 6
    "Molecule coordinate bridge should still expose the terminal label at t=1"

@[test]
def testBranchingMoleculeBridgeUsesDiscreteDFMWhenConfigured : IO Unit := do
  let cfg := MoleculeBridgeConfig.qm9 10 9
  let x0 := MoleculeBridgeConfig.maskedAtom cfg { x := 0.0, y := 0.0, z := 0.0 }
  let x1 : MoleculeAtom := { coord := { x := 10.0, y := 20.0, z := 30.0 }, label := 6 }
  let start := MoleculeBridgeConfig.bridge cfg x0 x1 0.0 0.0
  LeanTest.assertEqual start.label 9
    "QM9 molecule bridge should start at the mask/source label"
  let nearTerminal := MoleculeBridgeConfig.bridge cfg x0 x1 0.0 0.9
  LeanTest.assertEqual nearTerminal.label 6
    "QM9 molecule bridge should use the DFM target label when target mass dominates"
  let terminal := MoleculeBridgeConfig.bridge cfg x0 x1 0.0 1.0
  LeanTest.assertEqual terminal.label 6
    "QM9 molecule bridge should expose the endpoint label at t=1"

@[test]
def testBranchingMoleculeModelPredictionUsesDFMStep : IO Unit := do
  let labelDFM := DistNoisyDiscreteConfig.qm9 10 9
  let cfg : MoleculeBridgeConfig := {
    coordOU := OUBridgeConfig.constantVariance 0.0 1.0
    maskToken := 9
    labelDFM := some labelDFM
  }
  let x0Atom := MoleculeBridgeConfig.maskedAtom cfg { x := 0.0, y := 0.0, z := 0.0 }
  let x0 : BranchingState MoleculeAtom := BranchingState.mkDefault #[x0Atom] #[0]
  let labelLogits := (Array.replicate 10 (-10.0)).set! 6 10.0
  let prediction : MoleculeModelPrediction := {
    coordTargets := #[{ x := 10.0, y := 0.0, z := 0.0 }]
    labelLogits := #[labelLogits]
    splitLogits := #[-100.0]
    delLogits := #[-100.0]
  }
  let flow : CoalescentFlow MoleculeBridgeConfig MoleculeAtom :=
    CoalescentFlow.mkDefault cfg noEventTimeDist noEventTimeDist (fun _ => 0.0)
  let (result, _rng) := moleculeBranchingStep flow x0 prediction 0.8 0.9 (rng := { state := 777 })
  LeanTest.assertEqual result.state.state.size 1
    "Molecule DFM model step should preserve one atom when split/deletion hazards are disabled"
  let got := result.state.state[0]!
  LeanTest.assertTrue (Float.abs (got.coord.x - 5.0) < 1.0e-12)
    s!"Molecule model prediction should OU-step coordinates, got x={got.coord.x}"
  LeanTest.assertEqual got.label 6
    "Molecule model prediction should DFM-step labels from model logits"

@[test]
def testBranchingMoleculeFixedLabelStepBypassesDFM : IO Unit := do
  let fixedCfg : MoleculeBridgeConfig := {
    coordOU := OUBridgeConfig.constantVariance 0.0 1.0
    maskToken := 0
    labelDFM := none
  }
  let logits := #[-10.0, 10.0]
  let initialRng : Rng := { state := 20260709 }
  let (label, finalRng) :=
    fixedCfg.sampleStepLabelFromLogits 0 logits 0.2 0.3 initialRng
  LeanTest.assertEqual label 1
    "A fixed-label generation step should use the model mode directly"
  LeanTest.assertEqual finalRng.state initialRng.state
    "A fixed-label generation step should not consume DFM randomness"

  let ordinaryCfg := MoleculeBridgeConfig.qm9 2 0
  LeanTest.assertTrue ordinaryCfg.labelDFM.isSome
    "The ordinary molecule configuration should retain its QM9 DFM default"

@[test]
def testBranchingMoleculeSampledStepSeparatesIdenticalParticles : IO Unit := do
  let cfg : MoleculeBridgeConfig := {
    coordOU := OUBridgeConfig.constantVariance 0.0 1.0
    maskToken := 9
  }
  let atom := MoleculeBridgeConfig.maskedAtom cfg { x := 0.0, y := 0.0, z := 0.0 }
  let x0 : BranchingState MoleculeAtom := BranchingState.mkDefault #[atom, atom] #[0, 0]
  let prediction : MoleculeModelPrediction := {
    coordTargets := #[
      { x := 1.0, y := 0.0, z := 0.0 },
      { x := 1.0, y := 0.0, z := 0.0 }
    ]
    labelLogits := #[#[], #[]]
    splitLogits := #[-100.0, -100.0]
    delLogits := #[-100.0, -100.0]
  }
  let flow : CoalescentFlow MoleculeBridgeConfig MoleculeAtom :=
    CoalescentFlow.mkDefault cfg noEventTimeDist noEventTimeDist (fun _ => 0.0)
  let (result, _rng) :=
    moleculeBranchingStepSampled flow x0 prediction 0.1 0.2 (rng := { state := 20260709 })
  LeanTest.assertEqual result.state.state.size 2
    "Sampled molecule step should preserve both particles when event hazards are disabled"
  let first := result.state.state[0]!.coord
  let second := result.state.state[1]!.coord
  let separation :=
    Float.abs (first.x - second.x) + Float.abs (first.y - second.y) + Float.abs (first.z - second.z)
  LeanTest.assertTrue (separation > 1.0e-8)
    s!"Independent OU samples should separate identical descendants, got separation={separation}"

@[test]
def testBranchingMoleculeSuppressesSplitWhenLabelChanges : IO Unit := do
  let cfg : MoleculeBridgeConfig := {
    coordOU := OUBridgeConfig.constantVariance 0.0 1.0
    maskToken := 0
    labelDFM := none
  }
  let atom : MoleculeAtom := { coord := default, label := 0 }
  let x0 : BranchingState MoleculeAtom := BranchingState.mkDefault #[atom] #[0]
  let prediction : MoleculeModelPrediction := {
    coordTargets := #[default]
    labelLogits := #[#[-20.0, 20.0]]
    splitLogits := #[8.0]
    delLogits := #[-100.0]
  }
  let flow : CoalescentFlow MoleculeBridgeConfig MoleculeAtom :=
    CoalescentFlow.mkDefault cfg uniformTimeDist noEventTimeDist
  let seed : Rng := { state := 20260718 }
  let (suppressed, _) := moleculeBranchingStep flow x0 prediction 0.0 0.5 (rng := seed)
  let (allowed, _) := moleculeBranchingStep flow x0 prediction 0.0 0.5
    (splitAllowedAfterBaseStep := fun _ _ => true) (rng := seed)
  LeanTest.assertEqual suppressed.state.state.size 1
    "A molecule label change should suppress a split in the same interval"
  LeanTest.assertTrue (allowed.state.state.size > 1)
    "The test fixture should generate splits when the Julia compatibility predicate is bypassed"

@[test]
def testMoleculeTrainingBridgeSamplesGaussianRoots : IO Unit := do
  let cfg := MoleculeBridgeConfig.qm9 10 9
  let targetAtom : MoleculeAtom := {
    coord := { x := 0.0, y := 0.0, z := 0.0 }
    label := 8
  }
  let target : BranchingState MoleculeAtom := BranchingState.mkDefault #[targetAtom] #[0]
  let (result, _rng) ←
    sampleMoleculeBridgeBatch cfg #[target] 2 { state := 20260709 }
      (branchTime := noEventTimeDist) (deletionTime := noEventTimeDist)
      (coalescenceFactor := 0.0) (deletionPad := 0.0)
  LeanTest.assertEqual result.Xt.size 2 "Expected two sampled molecule bridge items"
  let first := result.Xt[0]!.state[0]!.coord
  let second := result.Xt[1]!.state[0]!.coord
  let magnitude :=
    Float.abs first.x + Float.abs first.y + Float.abs first.z +
    Float.abs second.x + Float.abs second.y + Float.abs second.z
  let separation :=
    Float.abs (first.x - second.x) + Float.abs (first.y - second.y) + Float.abs (first.z - second.z)
  LeanTest.assertTrue (magnitude > 1.0e-8)
    "Molecule training bridge should no longer start every item at coordinate zero"
  LeanTest.assertTrue (separation > 1.0e-8)
    "Molecule training bridge should draw independent base-process samples"

@[test]
def testMoleculeTrainingBridgeCanOversampleBranchTimes : IO Unit := do
  let cfg := MoleculeBridgeConfig.qm9 10 9
  let atoms : Array MoleculeAtom := #[
    { coord := { x := 0.0, y := 0.0, z := 0.0 }, label := 6 },
    { coord := { x := 1.0, y := 0.0, z := 0.0 }, label := 8 }
  ]
  let target : BranchingState MoleculeAtom := BranchingState.mkDefault atoms #[0, 0]
  let seed : Rng := { state := 20260718 }
  let (ordinary, _) ← sampleMoleculeBridgeBatch cfg #[target] 1 seed
    (branchTime := uniformTimeDist) (deletionTime := noEventTimeDist)
    (coalescenceFactor := 1.0) (useBranchingTimeProb := 0.0) (deletionPad := 0.0)
  let (atBranch, _) ← sampleMoleculeBridgeBatch cfg #[target] 1 seed
    (branchTime := uniformTimeDist) (deletionTime := noEventTimeDist)
    (coalescenceFactor := 1.0) (useBranchingTimeProb := 1.0) (deletionPad := 0.0)
  LeanTest.assertTrue (Float.abs (ordinary.t[0]! - atBranch.t[0]!) > 1.0e-8)
    "Branch-time oversampling should replace the initially sampled uniform training time"

@[test]
def testBranchingMoleculeGenerateUsesPredictionIntervals : IO Unit := do
  let cfg : MoleculeBridgeConfig := {
    coordOU := OUBridgeConfig.constantVariance 0.0 1.0
    maskToken := 9
    labelDFM := some (DistNoisyDiscreteConfig.qm9 10 9)
  }
  let x0Atom := MoleculeBridgeConfig.maskedAtom cfg { x := 0.0, y := 0.0, z := 0.0 }
  let x0 : BranchingState MoleculeAtom := BranchingState.mkDefault #[x0Atom] #[0]
  let labelLogits := (Array.replicate 10 (-10.0)).set! 6 10.0
  let model (_t : Float) (_state : BranchingState MoleculeAtom) : MoleculeModelPrediction := {
    coordTargets := #[{ x := 10.0, y := 0.0, z := 0.0 }]
    labelLogits := #[labelLogits]
    splitLogits := #[-100.0]
    delLogits := #[-100.0]
  }
  let flow : CoalescentFlow MoleculeBridgeConfig MoleculeAtom :=
    CoalescentFlow.mkDefault cfg noEventTimeDist noEventTimeDist (fun _ => 0.0)
  let (result, _rng) := moleculeBranchingGenerate flow x0 model #[0.0, 0.5, 1.0] (rng := { state := 888 })
  LeanTest.assertEqual result.trajectory.size 3
    "Molecule generation should preserve one trajectory state per schedule time"
  LeanTest.assertEqual result.finalState.state.size 1
    "Molecule generation should preserve one atom when split/deletion hazards are disabled"
  let got := result.finalState.state[0]!
  LeanTest.assertTrue (Float.abs (got.coord.x - 10.0) < 1.0e-12)
    s!"Molecule generation should apply the final interval endpoint, got x={got.coord.x}"
  LeanTest.assertEqual got.label 6
    "Molecule generation should DFM-step labels across schedule intervals"

@[test]
def testBranchingQM9JsonLoaderBuildsMoleculeState : IO Unit := do
  let raw :=
    "{\"name\":\"water\",\"smiles\":\"O\",\"atoms\":[{\"label\":8,\"x\":0.0,\"y\":0.1,\"z\":-0.2},{\"atom_label\":1,\"coord\":[0.9,0.0,0.3]}]}"
  let mol ← expectOk "parse preprocessed QM9 molecule" (parseQM9MoleculeJson raw)
  LeanTest.assertTrue (mol.name? == some "water")
    s!"Expected molecule name metadata, got {mol.name?}"
  LeanTest.assertTrue (mol.smiles? == some "O")
    s!"Expected molecule smiles metadata, got {mol.smiles?}"
  let cfg : QM9StateConfig := {
    group := 3
    firstId := 41
    vocabSize? := some 10
    maskToken? := some 9
  }
  let state ← expectOk "convert preprocessed QM9 molecule" (mol.toBranchingState cfg)
  LeanTest.assertEqual state.state.size 2 "QM9 loader should preserve atom count"
  LeanTest.assertEqual state.groupings #[3, 3]
    "QM9 loader should assign the configured group to all atoms"
  LeanTest.assertEqual state.ids #[41, 42]
    "QM9 loader should assign stable sequential ids"
  LeanTest.assertTrue (state.branchmask.all id && state.flowmask.all id && state.padmask.all id)
    "QM9 loader should mark all terminal atoms active"
  LeanTest.assertEqual state.state[0]!.label 8
    "QM9 loader should parse numeric labels"
  LeanTest.assertEqual state.state[1]!.label 1
    "QM9 loader should accept atom_label as a label-field alias"
  LeanTest.assertTrue (Float.abs (state.state[0]!.coord.y - 0.1) < 1.0e-12)
    s!"QM9 loader should parse x/y/z coordinates, got {reprStr state.state[0]!.coord}"
  LeanTest.assertTrue (Float.abs (state.state[1]!.coord.x - 0.9) < 1.0e-12)
    s!"QM9 loader should parse coord-array coordinates, got {reprStr state.state[1]!.coord}"
  let xyz := moleculeStateToXYZ state "water fixture"
  LeanTest.assertTrue (xyz.startsWith "2\nwater fixture\n")
    s!"XYZ export should include atom count and comment header, got {xyz}"
  LeanTest.assertTrue (xyz.contains "O ")
    s!"XYZ export should map atomic-number labels to element symbols, got {xyz}"
  let tokenXyz := moleculeStateToXYZ state "token fixture" (labelSymbolFromArray #["X", "H", "C", "N", "O", "F", "G", "I", "O", "*"])
  LeanTest.assertTrue (tokenXyz.contains "H ")
    s!"XYZ export should accept token-vocabulary label maps, got {tokenXyz}"

  let vocabErr ← expectError "reject label outside configured vocabulary"
    (mol.toBranchingState { cfg with vocabSize? := some 8 })
  LeanTest.assertTrue (!vocabErr.isEmpty)
    "Invalid QM9 labels should produce a useful error"

  let maskRaw := "{\"atoms\":[{\"label\":9,\"coord\":[0.0,0.0,0.0]}]}"
  let maskMol ← expectOk "parse mask-token molecule" (parseQM9MoleculeJson maskRaw)
  let maskErr ← expectError "reject terminal mask token"
    (maskMol.toBranchingState { vocabSize? := some 10, maskToken? := some 9 })
  LeanTest.assertTrue (!maskErr.isEmpty)
    "Terminal QM9 records should reject the reserved mask token by default"
  let deletedState : BranchingState MoleculeAtom := {
    state with
    del := #[false, true]
  }
  let maskedDeleted := maskDeletedMoleculeLabels (MoleculeBridgeConfig.qm9 10 9) deletedState
  LeanTest.assertEqual maskedDeleted.state[0]!.label state.state[0]!.label
    "Deleted-label masking should preserve non-deleted molecule labels"
  LeanTest.assertEqual maskedDeleted.state[1]!.label 9
    "Deleted-label masking should replace deleted molecule labels with the mask token"

@[test]
def testBranchingQM9JsonlLoaderAndMaskedInitialState : IO Unit := do
  let raw :=
    "{\"atoms\":[{\"label\":6,\"coord\":[0.0,0.0,0.0]}]}\n\n{\"canonical_smiles\":\"N\",\"atoms\":[{\"label\":7,\"x\":1.0,\"y\":2.0,\"z\":3.0}]}"
  let records ← expectOk "parse preprocessed QM9 jsonl" (parseQM9MoleculeJsonl raw)
  LeanTest.assertEqual records.size 2
    "QM9 JSONL parser should ignore blank lines and parse both molecules"
  LeanTest.assertTrue (records[1]!.smiles? == some "N")
    s!"QM9 JSONL parser should accept canonical_smiles metadata, got {records[1]!.smiles?}"
  let states ← expectOk "convert preprocessed QM9 jsonl batch"
    (qm9RecordsToBranchingStates records { vocabSize? := some 10, maskToken? := some 9 })
  LeanTest.assertEqual states.size 2 "QM9 JSONL state conversion should preserve batch size"
  LeanTest.assertEqual states[0]!.state[0]!.label 6
    "QM9 JSONL state conversion should preserve first atom label"
  LeanTest.assertTrue (Float.abs (states[1]!.state[0]!.coord.z - 3.0) < 1.0e-12)
    s!"QM9 JSONL state conversion should preserve coordinates, got {reprStr states[1]!.state[0]!.coord}"

  let bridgeCfg := MoleculeBridgeConfig.qm9 10 9
  let x0 := qm9InitialMaskedState bridgeCfg (group := 5) (coord := { x := 0.25, y := -0.5, z := 0.75 })
  LeanTest.assertEqual x0.state.size 1
    "QM9 generation source state should have length one"
  LeanTest.assertEqual x0.groupings #[5]
    "QM9 generation source state should use the requested group"
  LeanTest.assertEqual x0.state[0]!.label 9
    "QM9 generation source state should use the mask token"
  LeanTest.assertTrue (Float.abs (x0.state[0]!.coord.x - 0.25) < 1.0e-12)
    s!"QM9 generation source state should preserve the requested initial coordinate, got {reprStr x0.state[0]!.coord}"

@[test]
def testBranchingMoleculeBridgeIntegratesWithBranchingBridge : IO Unit := do
  let cfg : MoleculeBridgeConfig := {
    coordOU := OUBridgeConfig.constantVariance 0.0 1.0
    maskToken := 99
  }
  let x1Atom : MoleculeAtom := { coord := { x := 2.0, y := 4.0, z := 6.0 }, label := 6 }
  let x1 : BranchingState MoleculeAtom := BranchingState.mkDefault #[x1Atom] #[0]
  let expected := MoleculeBridgeConfig.bridge cfg (MoleculeBridgeConfig.maskedAtom cfg) x1Atom 0.0 0.5
  let (result, _rng) :=
    branchingBridge
      (fun cfg x0 x1 t0 t => MoleculeBridgeConfig.bridge cfg x0 x1 t0 t)
      cfg
      (fun _root => MoleculeBridgeConfig.maskedAtom cfg)
      #[x1]
      #[0.5]
      uniformTimeDist
      noEventTimeDist
      (sequentialUniformPolicy MoleculeAtom)
      (MoleculeBridgeConfig.anchorMerge cfg)
      (coalescenceFactor := 0.0)
      (rng := { state := 654 })
  LeanTest.assertEqual result.Xt.size 1 "Expected one molecule-bridged batch"
  LeanTest.assertEqual result.Xt[0]!.state.size 1 "Expected one molecule-bridged segment"
  let got := result.Xt[0]!.state[0]!
  LeanTest.assertTrue (Float.abs (got.coord.x - expected.coord.x) < 1.0e-12)
    s!"Branching bridge should use molecule OU x coordinate, got {got.coord.x}"
  LeanTest.assertTrue (Float.abs (got.coord.y - expected.coord.y) < 1.0e-12)
    s!"Branching bridge should use molecule OU y coordinate, got {got.coord.y}"
  LeanTest.assertTrue (Float.abs (got.coord.z - expected.coord.z) < 1.0e-12)
    s!"Branching bridge should use molecule OU z coordinate, got {got.coord.z}"
  LeanTest.assertEqual got.label expected.label
    "Branching bridge should preserve molecule label semantics"

@[test]
def testBranchingMoleculePackingAndLosses : IO Unit := do
  let atom : MoleculeAtom := { coord := { x := 1.0, y := 2.0, z := 3.0 }, label := 9 }
  let anchor : MoleculeAtom := { coord := { x := 4.0, y := 5.0, z := 6.0 }, label := 6 }
  let state : BranchingState MoleculeAtom := {
    state := #[atom]
    groupings := #[0]
    del := #[false]
    ids := #[1]
    branchmask := #[true]
    flowmask := #[true]
    padmask := #[true]
  }
  let result : BranchingBridgeResult MoleculeAtom := {
    t := #[0.25]
    segments := #[#[]]
    Xt := #[state]
    X1anchor := #[#[anchor]]
    descendants := #[#[1]]
    del := #[#[true]]
    splitsTarget := #[#[2]]
    prevCoalescence := #[#[0.0]]
  }
  let cfg : BranchingTrainConfig := { maxLen := 2, padToken := 0 }
  let ⟨batch, packed⟩ ← packBranchingMolecule cfg result
  LeanTest.assertEqual batch (1 : Nat) "Packed molecule batch should preserve batch size"
  LeanTest.assertTrue (Float.abs (Tensor.item (Tensor.sumAll packed.coord) - 6.0) < 1.0e-6)
    "Packed molecule coordinates should contain the bridge state coordinates"
  LeanTest.assertTrue (Float.abs (Tensor.item (Tensor.sumAll packed.coordAnchor) - 15.0) < 1.0e-6)
    "Packed molecule anchor coordinates should contain endpoint coordinates"
  LeanTest.assertTrue (Float.abs (Tensor.item (Tensor.sumAll (floatDynamic packed.label)) - 9.0) < 1.0e-6)
    "Packed molecule labels should contain atom labels and pad with the configured pad token"
  LeanTest.assertTrue (Float.abs (Tensor.item (Tensor.sumAll (floatDynamic packed.labelAnchor)) - 6.0) < 1.0e-6)
    "Packed molecule anchor labels should contain atom-label targets"
  LeanTest.assertTrue (Float.abs (Tensor.item (Tensor.sumAll packed.padmask) - 1.0) < 1.0e-6)
    "Packed molecule pad mask should mark only real atoms"
  LeanTest.assertTrue (Float.abs (Tensor.item (Tensor.sumAll packed.splitsTarget) - 2.0) < 1.0e-6)
    "Packed molecule split targets should preserve BranchingBridgeResult targets"
  LeanTest.assertTrue (Float.abs (Tensor.item (Tensor.sumAll packed.delTarget) - 1.0) < 1.0e-6)
    "Packed molecule deletion targets should preserve BranchingBridgeResult targets"

  let totalLogits := batch * cfg.maxLen * 10
  let labelLogits : Tensor { dtype := .Float32, shape := #[batch, cfg.maxLen, 10], device := .CPU } :=
    Tensor.reshapeChecked (floatsDynamic (Array.replicate totalLogits 0.0)) #[batch, cfg.maxLen, 10]
  let flat := batch * cfg.maxLen
  let splitLogits : Tensor { dtype := .Float32, shape := #[batch, cfg.maxLen], device := .CPU } :=
    Tensor.reshape (floatsDynamic (Array.replicate flat 0.0)) #[batch, cfg.maxLen]
  let delLogits : Tensor { dtype := .Float32, shape := #[batch, cfg.maxLen], device := .CPU } :=
    Tensor.reshape (floatsDynamic (Array.replicate flat 0.0)) #[batch, cfg.maxLen]
  let (_loss, report) := moleculeLosses cfg packed packed.coordAnchor labelLogits splitLogits delLogits
  LeanTest.assertTrue (Float.abs report.coord < 1.0e-6)
    s!"Coordinate loss should be zero when prediction equals anchor, got {report.coord}"
  LeanTest.assertTrue (report.label > 2.0 && report.label < 2.5)
    s!"Uniform 10-way label logits should produce about log(10), got {report.label}"
  LeanTest.assertTrue (report.splits > 0.0 && report.del > 0.0)
    s!"Molecule loss report should include split and deletion terms, got {reprStr report}"

  let labelDFM := DistNoisyDiscreteConfig.qm9 10 9
  let ⟨batchDfm, packedDfm⟩ ← packBranchingMoleculeWithDFM cfg labelDFM result
  LeanTest.assertEqual batchDfm (1 : Nat) "DFM-packed molecule batch should preserve batch size"
  let scaleSum := Tensor.item (Tensor.sumAll packedDfm.labelLossScale)
  LeanTest.assertTrue (Float.abs (scaleSum - (1.0 / 0.95)) < 1.0e-5)
    s!"DFM-packed molecule batch should carry Flowfusion label scale at t=0.25, got {scaleSum}"
  let totalDfmLogits := batchDfm * cfg.maxLen * 10
  let labelLogitsDfm : Tensor { dtype := .Float32, shape := #[batchDfm, cfg.maxLen, 10], device := .CPU } :=
    Tensor.reshapeChecked (floatsDynamic (Array.replicate totalDfmLogits 0.0)) #[batchDfm, cfg.maxLen, 10]
  let flatDfm := batchDfm * cfg.maxLen
  let splitLogitsDfm : Tensor { dtype := .Float32, shape := #[batchDfm, cfg.maxLen], device := .CPU } :=
    Tensor.reshape (floatsDynamic (Array.replicate flatDfm 0.0)) #[batchDfm, cfg.maxLen]
  let delLogitsDfm : Tensor { dtype := .Float32, shape := #[batchDfm, cfg.maxLen], device := .CPU } :=
    Tensor.reshape (floatsDynamic (Array.replicate flatDfm 0.0)) #[batchDfm, cfg.maxLen]
  let (_dfmLoss, dfmReport) :=
    moleculeLosses cfg packedDfm packedDfm.coordAnchor labelLogitsDfm splitLogitsDfm delLogitsDfm
  LeanTest.assertTrue (dfmReport.label > report.label)
    s!"DFM label loss should apply the time scale, unscaled={report.label}, scaled={dfmReport.label}"

private def singleAtomMoleculeLossFixture (t : Float) : BranchingBridgeResult MoleculeAtom :=
  let atom : MoleculeAtom := { coord := default, label := 0 }
  let anchor : MoleculeAtom := { coord := { x := 1.0, y := 0.0, z := 0.0 }, label := 0 }
  let state : BranchingState MoleculeAtom := BranchingState.mkDefault #[atom] #[0]
  {
    t := #[t]
    segments := #[#[]]
    Xt := #[state]
    X1anchor := #[#[anchor]]
    descendants := #[#[1]]
    del := #[#[false]]
    splitsTarget := #[#[0]]
    prevCoalescence := #[#[0.0]]
  }

@[test]
def testMoleculeLossUsesJuliaTimeScalingAndIndependentWeights : IO Unit := do
  let dfm := DistNoisyDiscreteConfig.qm9 10 0
  let labelScaleRatio := dfm.lossScale 0.9 1.0 0.2 / dfm.lossScale 0.0 1.0 0.2
  LeanTest.assertTrue (Float.abs (labelScaleRatio - 4.0) < 1.0e-8)
    s!"Discrete labels should use the Julia demo's linear 0.2-epsilon scale, ratio={labelScaleRatio}"
  let cfg : BranchingTrainConfig := {
    maxLen := 1
    padToken := 0
    coordWeight := 2.0
    labelWeight := 0.0
    splitsWeight := 1.0
    delWeight := 1.0
  }
  let ⟨batchEarly, early⟩ ← packBranchingMolecule cfg (singleAtomMoleculeLossFixture 0.0)
  let ⟨batchLate, late⟩ ← packBranchingMolecule cfg (singleAtomMoleculeLossFixture 0.9)
  let labelEarly : Tensor { dtype := .Float32, shape := #[batchEarly, cfg.maxLen, 1], device := .CPU } := zerosDynamic #[batchEarly, cfg.maxLen, 1]
  let splitEarly : Tensor { dtype := .Float32, shape := #[batchEarly, cfg.maxLen], device := .CPU } := zerosDynamic #[batchEarly, cfg.maxLen]
  let delEarly : Tensor { dtype := .Float32, shape := #[batchEarly, cfg.maxLen], device := .CPU } := zerosDynamic #[batchEarly, cfg.maxLen]
  let (totalEarly, reportEarly) :=
    moleculeLosses cfg early early.coord labelEarly splitEarly delEarly
  let labelLate : Tensor { dtype := .Float32, shape := #[batchLate, cfg.maxLen, 1], device := .CPU } := zerosDynamic #[batchLate, cfg.maxLen, 1]
  let splitLate : Tensor { dtype := .Float32, shape := #[batchLate, cfg.maxLen], device := .CPU } := zerosDynamic #[batchLate, cfg.maxLen]
  let delLate : Tensor { dtype := .Float32, shape := #[batchLate, cfg.maxLen], device := .CPU } := zerosDynamic #[batchLate, cfg.maxLen]
  let (_, reportLate) :=
    moleculeLosses cfg late late.coord labelLate splitLate delLate
  LeanTest.assertTrue (Float.abs (reportLate.coord / reportEarly.coord - 4.0) < 1.0e-4)
    s!"Late coordinate supervision should receive Julia scalefloss amplification, early={reportEarly.coord}, late={reportLate.coord}"
  LeanTest.assertTrue (Float.abs (reportLate.splits / reportEarly.splits - 4.0) < 1.0e-4)
    s!"Late split supervision should receive Julia scalefloss amplification, early={reportEarly.splits}, late={reportLate.splits}"
  LeanTest.assertTrue (Float.abs (reportLate.del / reportEarly.del - 4.0) < 1.0e-4)
    s!"Late deletion supervision should receive Julia scalefloss amplification, early={reportEarly.del}, late={reportLate.del}"
  let expectedTotal := reportEarly.coord * 2.0 + reportEarly.splits + reportEarly.del
  LeanTest.assertTrue (Float.abs (Tensor.item totalEarly - expectedTotal) < 1.0e-5)
    s!"Independent molecule loss weights should determine total loss, expected={expectedTotal}, got={Tensor.item totalEarly}"

@[test]
def testMoleculeMuonUpdatesMatrixAndVectorLeaves : IO Unit := do
  let matrix : Tensor { dtype := .Float32, shape := #[2, 2], device := .CPU } := Tensor.reshape (floatsDynamic #[1.0, 2.0, 3.0, 4.0]) #[2, 2]
  let vector : Tensor { dtype := .Float32, shape := #[2], device := .CPU } := floatsDynamic #[1.0, -1.0]
  let grads : Tensor { dtype := .Float32, shape := #[2, 2], device := .CPU } × Tensor { dtype := .Float32, shape := #[2], device := .CPU } := (onesDynamic #[2, 2], onesDynamic #[2])
  let params : Tensor { dtype := .Float32, shape := #[2, 2], device := .CPU } × Tensor { dtype := .Float32, shape := #[2], device := .CPU } := (matrix, vector)
  let state := initMoleculeMuonState params
  let (updated, state') ← moleculeMuonStep params grads state 0.01 0.0 0.95 5
  let matrixDelta := Tensor.item (Tensor.sumAll (Tensor.abs (Tensor.sub updated.1 params.1)))
  let vectorDelta := Tensor.item (Tensor.sumAll (Tensor.abs (Tensor.sub updated.2 params.2)))
  LeanTest.assertTrue (Float.isFinite matrixDelta && matrixDelta > 0.0)
    s!"Muon should update matrix leaves, delta={matrixDelta}"
  LeanTest.assertTrue (Float.isFinite vectorDelta && vectorDelta > 0.0)
    s!"Muon should flatten and update vector leaves like the Julia rule, delta={vectorDelta}"
  LeanTest.assertEqual state'.step 1 "Muon optimizer state should advance one step"

private def moleculeCoordCell {maxLen : Nat}
    (coord : Tensor { dtype := .Float32, shape := #[1, maxLen, 3], device := .CPU }) (j k : Nat) : Float :=
  let row : Tensor { dtype := .Float32, shape := #[1, 1, 3], device := .CPU } := sliceTyped coord 1 j 1
  let cell : Tensor { dtype := .Float32, shape := #[1, 1, 1], device := .CPU } := sliceTyped row 2 k 1
  Tensor.item (cell.reshape #[])

@[test]
def testMoleculeTransformerCoordTargetCapAppliesInTrainingAndInference : IO Unit := do
  let maxLen : Nat := 1
  let vocab : Nat := 2
  let heads : Nat := 1
  let headDim : Nat := 2
  let mlp : Nat := 2
  let cap := 2.0
  torch.manualSeed 20260709
  let initialized ← MoleculeTransformerParams.init vocab heads headDim mlp
  let params := { initialized with
    coordHeadW := zerosDynamic #[3, heads * headDim]
    coordHeadB := floatsDynamic #[100.0, -100.0, 4.0]
  }

  let coord : Tensor { dtype := .Float32, shape := #[1, maxLen, 3], device := .CPU } := zerosDynamic #[1, maxLen, 3]
  let label : Tensor { shape := #[1, maxLen], dtype := .Int64, device := .CPU } := Tensor.reshape (Tensor.fromInt64Array #[0]) #[1, maxLen]
  let time : Tensor { dtype := .Float32, shape := #[1], device := .CPU } := zerosDynamic #[1]
  let padmask : Tensor { shape := #[1, maxLen], dtype := .Int64, device := .CPU } := Tensor.fullInt #[1, maxLen] 1
  let trainingModel :=
    moleculeTransformerModel (maxLen := maxLen) (vocab := vocab)
      (heads := heads) (headDim := headDim) (mlp := mlp)
      (coordTargetCap? := some cap)
  let (bounded, _, _, _) ← trainingModel.forward params coord label time padmask
  for k in [:3] do
    let got := moleculeCoordCell bounded 0 k
    LeanTest.assertTrue (Float.abs got <= cap + 1.0e-5)
      s!"Training model coordinate {k} should be bounded by {cap}, got {got}"
  LeanTest.assertTrue (moleculeCoordCell bounded 0 0 > 1.99)
    s!"Positive oversized training target should approach +cap, got {moleculeCoordCell bounded 0 0}"
  LeanTest.assertTrue (moleculeCoordCell bounded 0 1 < -1.99)
    s!"Negative oversized training target should approach -cap, got {moleculeCoordCell bounded 0 1}"

  let atom : MoleculeAtom := { coord := default, label := 0 }
  let state : BranchingState MoleculeAtom := BranchingState.mkDefault #[atom] #[0]
  let inferenceModel :=
    moleculeTransformerIOModel (maxLen := maxLen) (vocab := vocab)
      (heads := heads) (headDim := headDim) (mlp := mlp)
      0 params (coordTargetCap? := some cap)
  let prediction ← inferenceModel 0.0 state
  let target := prediction.coordTargets[0]!
  for got in #[target.x, target.y, target.z] do
    LeanTest.assertTrue (Float.abs got <= cap + 1.0e-5)
      s!"Inference coordinate target should be bounded by {cap}, got {got}"
  LeanTest.assertTrue (target.x > 1.99 && target.y < -1.99)
    s!"Inference should use the same signed cap as training, got {reprStr target}"

@[test]
def testBranchingSegmentsProjectToEventSkeleton : IO Unit := do
  let seg : Segment Float := {
    Xt := 0.5
    t := 0.75
    anchor := 1.0
    descendants := 3
    del := false
    branchable := true
    flowable := true
    group := 0
    lastCoalescence := 0.25
    id := 11
  }
  let interval := Segment.toAcceptedStepSegment 0 0 seg
  LeanTest.assertTrue (Float.abs (interval.tStart - 0.25) < 1.0e-12)
    s!"Projected interval should start at last coalescence, got {interval.tStart}"
  LeanTest.assertTrue (Float.abs (interval.tAfter - 0.75) < 1.0e-12)
    s!"Projected interval should end at the segment time, got {interval.tAfter}"
  LeanTest.assertTrue interval.crossedJumpFlag
    "Segments with multiple descendants should mark a branch crossing"

  let graph := graphFromBranchingSegments #[seg]
  LeanTest.assertEqual graph.vertices.size 2
    "One interval plus one branch vertex should be projected"
  LeanTest.assertTrue (graph.containsMoveKind .intervalAdjoint)
    "Projected graph should include interval adjoint work"
  LeanTest.assertTrue (graph.containsMoveKind .checkpointBoundary)
    "Projected graph should include checkpoint boundaries by default"
  LeanTest.assertTrue (graph.containsMoveKind .branchAggregate)
    "Projected graph should include branch aggregation for descendant-count segments"

@[test]
def testReconstructLineageAssignsStableParentChildIds : IO Unit := do
  let split0 : BranchingStepEvent := {
    sourceIndex := 0
    sourceId := 1
    group := 0
    splitCount := 1
    deleted := false
    t0 := 0.0
    t1 := 0.25
  }
  let split1 : BranchingStepEvent := {
    sourceIndex := 0
    sourceId := 1
    group := 0
    splitCount := 1
    deleted := false
    t0 := 0.25
    t1 := 0.5
  }
  let generated : BranchingGenerateResult Nat := {
    finalState := mkState #[20, 21, 10] #[0, 0, 0]
    trajectory := #[
      mkState #[0] #[0],
      mkState #[10, 11] #[0, 0],
      mkState #[20, 21, 10] #[0, 0, 0]
    ]
    events := #[#[split0], #[split1]]
    times := #[0.0, 0.25, 0.5]
  }
  let trace ←
    match reconstructLineage generated with
    | .ok trace => pure trace
    | .error message => throw (IO.userError message)
  LeanTest.assertEqual trace.frames.size 3 "Expected one lineage frame per generated state"
  LeanTest.assertEqual trace.events.size 2 "Expected both split events in the lineage trace"
  LeanTest.assertEqual (trace.frames[0]!.particles.map (fun particle => particle.particleId)) #[1]
    "Initial particle should receive the first runtime id"
  LeanTest.assertEqual (trace.frames[1]!.particles.map (fun particle => particle.particleId)) #[2, 3]
    "First split should terminate the root and create two child ids"
  LeanTest.assertEqual (trace.frames[2]!.particles.map (fun particle => particle.particleId)) #[4, 5, 3]
    "Second split should replace its parent while preserving the unaffected sibling"
  LeanTest.assertEqual trace.events[0]!.parentId 1 "First split should reference the root"
  LeanTest.assertEqual trace.events[0]!.childIds #[2, 3] "First split should record both children"
  LeanTest.assertEqual trace.events[1]!.parentId 2 "Second split should reference the first child"
  LeanTest.assertEqual trace.events[1]!.childIds #[4, 5] "Second split should record new child ids"

@[test]
def testReconstructLineageRejectsMalformedFrameCounts : IO Unit := do
  let malformed : BranchingGenerateResult Nat := {
    finalState := mkState #[1] #[0]
    trajectory := #[mkState #[0] #[0], mkState #[1] #[0]]
    events := #[]
    times := #[0.0, 1.0]
  }
  let rejected :=
    match reconstructLineage malformed with
    | .error _ => true
    | .ok _ => false
  LeanTest.assertTrue rejected
    "Lineage reconstruction must reject a trajectory with a missing event step"

@[test]
def testMoleculeTrajectoryJsonlRejectsNonfiniteCoordinates : IO Unit := do
  let badAtom : MoleculeAtom := {
    coord := { x := 0.0 / 0.0, y := 0.0, z := 0.0 }
    label := 8
  }
  let badState : BranchingState MoleculeAtom :=
    BranchingState.mkDefault #[badAtom] #[0]
  let generated : BranchingGenerateResult MoleculeAtom := {
    finalState := badState
    trajectory := #[badState]
    events := #[]
    times := #[0.0]
  }
  let rejected :=
    match moleculeTrajectoryJsonl generated with
    | .error _ => true
    | .ok _ => false
  LeanTest.assertTrue rejected
    "Trajectory JSONL must reject NaN/Inf coordinates rather than emitting invalid JSON"

@[test]
def testBranchingStepEventsProjectToEventSkeleton : IO Unit := do
  let event : BranchingStepEvent := {
    sourceIndex := 0
    sourceId := 21
    group := 0
    splitCount := 2
    deleted := false
    t0 := 0.0
    t1 := 0.5
  }
  let graph := graphFromBranchingStepEvents #[event]
  LeanTest.assertTrue (graph.containsMoveKind .intervalAdjoint)
    "Step-event graph should include interval adjoint work"
  LeanTest.assertTrue (graph.containsMoveKind .checkpointBoundary)
    "Step-event graph should include checkpoint boundaries by default"
  LeanTest.assertTrue (graph.containsMoveKind .branchAggregate)
    "Step-event graph should include branch aggregation for split events"

  let generated : BranchingGenerateResult Nat := {
    finalState := mkState #[1] #[0]
    trajectory := #[mkState #[0] #[0], mkState #[0] #[0], mkState #[1] #[0]]
    events := #[#[], #[event]]
    times := #[0.0, 0.5, 1.0]
  }
  let generatedGraph := graphFromBranchingGenerateResult generated
  LeanTest.assertTrue (generatedGraph.vertices.size >= 3)
    s!"Generation graph should include a quiet interval plus split interval/branch vertices, got {generatedGraph.vertices.size}"
  LeanTest.assertTrue (generatedGraph.containsMoveKind .branchAggregate)
    "Generation graph should preserve branch aggregation moves"

@[test]
def testGeodesicInterpolateFloat : IO Unit := do
  let x := 0.0
  let y := 10.0
  let z := canonicalAnchorMerge x y 1 3
  LeanTest.assertTrue (Float.abs (z - 7.5) < 1.0e-6) "Geodesic interpolate matches weighted average"

@[test]
def testOrthogonalLogExp : IO Unit := do
  let Y ← Orthogonal.random 2
  let prod := Tensor.mm (Tensor.transpose2d Y.matrix) Y.matrix
  LeanTest.assertTrue (Tensor.allclose prod (Tensor.eye 2) 1.0e-4 1.0e-4) "exp preserves orthogonality"

@[test]
def testGrassmannDistanceSelf : IO Unit := do
  let X ← Grassmann.random 3 1
  let d := Grassmann.distance X X
  LeanTest.assertTrue (Float.abs d < 1.0e-4) "Grassmann self-distance is near zero"
