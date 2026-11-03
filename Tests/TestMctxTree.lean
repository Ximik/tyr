import LeanTest
import Lean.Data.Json
import Tyr.Mctx

open torch.mctx

private def approx (a b : Float) (tol : Float := 1e-6) : Bool :=
  Float.abs (a - b) < tol

private def parseJsonFileOrFail (path : System.FilePath) : IO Lean.Json := do
  let raw ← IO.FS.readFile path
  match Lean.Json.parse raw with
  | .ok j => pure j
  | .error e => LeanTest.fail s!"JSON parse failed for {path}: {e}"

private def lcgA : UInt64 := 6364136223846793005
private def lcgC : UInt64 := 1442695040888963407

private def mix (x : UInt64) : UInt64 :=
  x * lcgA + lcgC

private def uniform01 (x : UInt64) : Float :=
  let mant := (x >>> 11).toNat
  let denom : Float := Float.ofNat (Nat.pow 2 53)
  Float.ofNat mant / denom

private def signed01 (x : UInt64) : Float :=
  2.0 * uniform01 x - 1.0

private def priorLogitsFromSeed (seed : UInt64) (numActions : Nat) : Array Float :=
  (List.range numActions).toArray.map fun i =>
    let k := mix (seed + UInt64.ofNat (i + 1) * 0x9e3779b97f4a7c15)
    signed01 k

/- The reference model and expected statistics come from the unmodified upstream
   mctx wheel, not from Tyr. See MctxData/README.md for provenance/regeneration. -/
private structure ReferenceModel where
  root_state : Nat
  prior_logits : Array (Array Float)
  values : Array Float
  next_states : Array (Array Nat)
  rewards : Array (Array Float)
  deriving Lean.FromJson

private structure ReferenceExpected where
  node_visits : Array Nat
  raw_values : Array Float
  node_values : Array Float
  parents : Array Int
  action_from_parent : Array Int
  children_index : Array (Array Int)
  children_visits : Array (Array Nat)
  children_rewards : Array (Array Float)
  children_discounts : Array (Array Float)
  children_values : Array (Array Float)
  embeddings : Array Nat
  prior_probs : Array (Array Float)
  qvalues : Array (Array Float)
  action : Nat
  action_weights : Array Float
  transformed_root_qvalues : Array Float
  deriving Lean.FromJson

private structure ReferenceCase where
  name : String
  algorithm : String
  num_simulations : Nat
  max_depth : Nat
  seed : Nat
  discount : Float
  algorithm_config : Lean.Json
  expected : ReferenceExpected
  deriving Lean.FromJson

private structure ReferenceFile where
  schema_version : Nat
  model : ReferenceModel
  cases : Array ReferenceCase
  deriving Lean.FromJson

private def checkEqual [BEq α] [Repr α]
    (label : String) (actual expected : α) : Except String Unit :=
  if actual == expected then .ok ()
  else .error s!"{label}: expected {repr expected}, got {repr actual}"

private def checkFloats (label : String) (actual expected : Array Float) : Except String Unit := do
  checkEqual s!"{label}.size" actual.size expected.size
  for i in [:actual.size] do
    let a := actual[i]!
    let e := expected[i]!
    unless a.isFinite && e.isFinite && Float.abs (a - e) ≤ 1e-10 do
      throw s!"{label}[{i}]: expected {e}, got {a}"

private def checkFloatRows (label : String)
    (actual expected : Array (Array Float)) : Except String Unit := do
  checkEqual s!"{label}.size" actual.size expected.size
  for i in [:actual.size] do
    checkFloats s!"{label}[{i}]" actual[i]! expected[i]!

private def checkModel (model : ReferenceModel) : Except String Unit := do
  let states := model.values.size
  unless model.root_state < states do throw "model.root_state out of range"
  checkEqual "model.prior_logits.size" model.prior_logits.size states
  checkEqual "model.next_states.size" model.next_states.size states
  checkEqual "model.rewards.size" model.rewards.size states
  let actions := model.prior_logits[model.root_state]!.size
  unless actions > 0 do throw "model must have actions"
  checkFloats "model.values" model.values model.values
  for state in [:states] do
    checkEqual "model.prior_logits width" model.prior_logits[state]!.size actions
    checkEqual "model.next_states width" model.next_states[state]!.size actions
    checkEqual "model.rewards width" model.rewards[state]!.size actions
    checkFloats "model.prior_logits" model.prior_logits[state]! model.prior_logits[state]!
    checkFloats "model.rewards" model.rewards[state]! model.rewards[state]!
    unless model.next_states[state]!.all (· < states) do
      throw "model.next_states contains out-of-range state"

private def referenceQTransform (config : Lean.Json) : Except String (QTransform Nat E) := do
  let name ← config.getObjValAs? String "qtransform"
  let args ← config.getObjVal? "qtransform_kwargs"
  match name with
  | "qtransform_by_min_max" =>
    let minValue ← args.getObjValAs? Float "min_value"
    let maxValue ← args.getObjValAs? Float "max_value"
    pure (fun tree node => qtransformByMinMax tree node minValue maxValue)
  | "qtransform_by_parent_and_siblings" =>
    let epsilon ← args.getObjValAs? Float "epsilon"
    pure (fun tree node => qtransformByParentAndSiblings tree node epsilon)
  | "qtransform_completed_by_mix_value" =>
    let valueScale ← args.getObjValAs? Float "value_scale"
    let maxvisitInit ← args.getObjValAs? Float "maxvisit_init"
    let rescaleValues ← args.getObjValAs? Bool "rescale_values"
    let useMixedValue ← args.getObjValAs? Bool "use_mixed_value"
    let epsilon ← args.getObjValAs? Float "epsilon"
    pure (fun tree node => qtransformCompletedByMixValue tree node
      valueScale maxvisitInit rescaleValues useMixedValue epsilon)
  | _ => throw s!"Unsupported reference qtransform '{name}'"

/-- Upstream mctx stores a missing parent or child as -1. -/
private def toUpstreamIndex : Option Nat → Int
  | none => -1
  | some n => Int.ofNat n

private def compareReference (expected : ReferenceExpected)
    (output : PolicyOutput (Tree Nat E)) (qtransform : QTransform Nat E) : Except String Unit := do
  let tree := output.searchTree
  checkEqual "action" output.action expected.action
  checkFloats "action_weights" output.actionWeights expected.action_weights
  checkEqual "node_visits" tree.nodeVisits expected.node_visits
  checkEqual "parents" (tree.parents.map toUpstreamIndex) expected.parents
  checkEqual "action_from_parent" (tree.actionFromParent.map toUpstreamIndex) expected.action_from_parent
  checkEqual "children_index" (tree.childrenIndex.map (·.map toUpstreamIndex)) expected.children_index
  checkEqual "children_visits" tree.childrenVisits expected.children_visits
  checkEqual "embeddings" tree.embeddings expected.embeddings
  checkFloats "raw_values" tree.rawValues expected.raw_values
  checkFloats "node_values" tree.nodeValues expected.node_values
  checkFloatRows "prior_probs" (tree.childrenPriorLogits.map softmax) expected.prior_probs
  checkFloatRows "children_rewards" tree.childrenRewards expected.children_rewards
  checkFloatRows "children_discounts" tree.childrenDiscounts expected.children_discounts
  checkFloatRows "children_values" tree.childrenValues expected.children_values
  checkFloatRows "qvalues" ((List.range tree.nodeVisits.size).toArray.map tree.qvalues) expected.qvalues
  checkFloats "transformed_root_qvalues" (qtransform tree ROOT_INDEX) expected.transformed_root_qvalues

private def runReference (model : ReferenceModel) (fixture : ReferenceCase) : Except String Unit := do
  checkModel model
  unless fixture.discount.isFinite do throw "Nonfinite fixture discount"
  let root : RootFnOutput Nat := {
    priorLogits := model.prior_logits[model.root_state]!
    value := model.values[model.root_state]!
    embedding := model.root_state
  }
  let recurrent : RecurrentFn Unit Nat := fun _ _ action state =>
    let nextState := model.next_states[state]![action]!
    ({ reward := model.rewards[state]![action]!, discount := fixture.discount,
       priorLogits := model.prior_logits[nextState]!, value := model.values[nextState]! }, nextState)
  let config := fixture.algorithm_config
  match fixture.algorithm with
  | "muzero" =>
    let qtransform ← referenceQTransform (E := Unit) config
    let dirichletFraction ← config.getObjValAs? Float "dirichlet_fraction"
    let dirichletAlpha ← config.getObjValAs? Float "dirichlet_alpha"
    let pbCInit ← config.getObjValAs? Float "pb_c_init"
    let pbCBase ← config.getObjValAs? Float "pb_c_base"
    let temperature ← config.getObjValAs? Float "temperature"
    let output := muzeroPolicy () fixture.seed.toUInt64 root recurrent fixture.num_simulations
      (maxDepth := some fixture.max_depth) (qtransform := qtransform)
      (dirichletFraction := dirichletFraction) (dirichletAlpha := dirichletAlpha)
      (pbCInit := pbCInit) (pbCBase := pbCBase) (temperature := temperature)
    compareReference fixture.expected output qtransform
  | "gumbel_muzero" =>
    let qtransform ← referenceQTransform (E := GumbelMuZeroExtraData) config
    let maxNumConsideredActions ← config.getObjValAs? Nat "max_num_considered_actions"
    let gumbelScale ← config.getObjValAs? Float "gumbel_scale"
    let output := gumbelMuZeroPolicy () fixture.seed.toUInt64 root recurrent fixture.num_simulations
      (maxDepth := some fixture.max_depth) (qtransform := qtransform)
      (maxNumConsideredActions := maxNumConsideredActions) (gumbelScale := gumbelScale)
    compareReference fixture.expected output qtransform
  | _ => throw s!"Unsupported reference algorithm '{fixture.algorithm}'"

private def loadReference : IO ReferenceFile := do
  let json ← parseJsonFileOrFail ⟨"Tests/MctxData/deterministic_reference.json"⟩
  let fixture ← match Lean.fromJson? json with
    | .ok f => pure (f : ReferenceFile)
    | .error error => throw (IO.userError s!"Invalid MCTS reference: {error}")
  LeanTest.assertEqual fixture.schema_version 1 "MCTS reference schema version"
  pure fixture

private def findReference (file : ReferenceFile) (name : String) : IO ReferenceCase := do
  match file.cases.find? (·.name == name) with
  | some fixture => pure fixture
  | none => throw (IO.userError s!"Missing MCTS reference '{name}'")

private def runFixture (name : String) : IO Unit := do
  let file ← loadReference
  let fixture ← findReference file name
  match runReference file.model fixture with
  | .ok () => pure ()
  | .error error => LeanTest.fail s!"MCTS reference {name}: {error}"

@[test]
def testMctxTreeFixtureMuZero : IO Unit := runFixture "muzero_min_max"

@[test]
def testMctxTreeFixtureMuZeroQTransform : IO Unit := runFixture "muzero_parent_siblings"

@[test]
def testMctxTreeFixtureGumbelMuZero : IO Unit := runFixture "gumbel_no_rescale"

@[test]
def testMctxTreeFixtureGumbelMuZeroReward : IO Unit := runFixture "gumbel_rescale"

private def expectReferenceFailure (model : ReferenceModel) (fixture : ReferenceCase)
    (field : String) : IO Unit := do
  match runReference model fixture with
  | .ok () => LeanTest.fail s!"Reference accepted mutated {field}"
  | .error error =>
    LeanTest.assertTrue (error.contains field)
      s!"Expected {field} mismatch, got unrelated failure: {error}"

@[test]
def testMctxTreeReferenceRejectsChangedExpectations : IO Unit := do
  let file ← loadReference
  let fixture ← findReference file "muzero_min_max"
  let expected := fixture.expected
  expectReferenceFailure file.model
    { fixture with expected := { expected with action := (expected.action + 1) % 4 } } "action"
  let visits := expected.children_visits
  let changedVisits := visits.set! 0 ((visits[0]!).set! 0 (visits[0]![0]! + 1))
  expectReferenceFailure file.model
    { fixture with expected := { expected with children_visits := changedVisits } } "children_visits"
  let qvalues := expected.qvalues
  let changedQ := qvalues.set! 0 ((qvalues[0]!).set! 0 (qvalues[0]![0]! + 0.25))
  expectReferenceFailure file.model
    { fixture with expected := { expected with qvalues := changedQ } } "qvalues"

@[test]
def testMctxTreeReferenceHonorsAlgorithmConfiguration : IO Unit := do
  let file ← loadReference
  let fixture ← findReference file "muzero_min_max"
  let config := fixture.algorithm_config.setObjVal! "pb_c_init" (Lean.toJson (0.0 : Float))
  LeanTest.assertTrue (runReference file.model { fixture with algorithm_config := config }).toOption.isNone
    "Changed PUCT config must change the independently recorded search"
  let config := fixture.algorithm_config.setObjVal! "qtransform" (Lean.toJson "unsupported")
  expectReferenceFailure file.model { fixture with algorithm_config := config } "Unsupported reference qtransform"
  let gumbel ← findReference file "gumbel_rescale"
  let config := gumbel.algorithm_config.setObjVal! "max_num_considered_actions" (Lean.toJson (1 : Nat))
  LeanTest.assertTrue (runReference file.model { gumbel with algorithm_config := config }).toOption.isNone
    "Changed sequential-halving config must change the independently recorded search"

@[test]
def testMctxGetSubtreeCarriesChildAsRoot : IO Unit := do
  let root : RootFnOutput UInt64 := {
    priorLogits := #[0.1, 0.3, 0.2, -0.1]
    value := 0.4
    embedding := 0
  }
  let recurrent : RecurrentFn Unit UInt64 := fun _ _ action emb =>
    let nextEmb := mix (emb + UInt64.ofNat (action + 1) * 0x9e3779b97f4a7c15)
    ({
      reward := signed01 (mix (nextEmb + 7))
      discount := 0.9
      priorLogits := priorLogitsFromSeed nextEmb 4
      value := signed01 (mix (nextEmb + 13))
    }, nextEmb)

  let out := muzeroPolicy
    (params := ())
    (rngKey := 9)
    (root := root)
    (recurrentFn := recurrent)
    (numSimulations := 8)
    (dirichletFraction := 0.0)

  let chosen := out.action
  let some childIdx := (out.searchTree.childrenIndex.getD ROOT_INDEX #[]).getD chosen none
    | LeanTest.fail "Chosen action should be expanded"

  let subtree := getSubtree out.searchTree chosen

  LeanTest.assertEqual (subtree.parents.getD ROOT_INDEX (some 123)) none
    "Subtree root should have no parent"
  LeanTest.assertEqual (subtree.actionFromParent.getD ROOT_INDEX (some 123)) none
    "Subtree root should not have incoming action"
  LeanTest.assertTrue ((List.range subtree.rootInvalidActions.size).all fun i =>
      subtree.rootInvalidActions.getD i true = false)
    "Subtree root invalid-action mask should be reset"

  let oldVisits := out.searchTree.nodeVisits.getD childIdx 0
  let newVisits := subtree.nodeVisits.getD ROOT_INDEX 0
  LeanTest.assertEqual newVisits oldVisits "Subtree root visits should match selected child visits"

  let oldRaw := out.searchTree.rawValues.getD childIdx 0.0
  let newRaw := subtree.rawValues.getD ROOT_INDEX 0.0
  LeanTest.assertTrue (approx newRaw oldRaw 1e-6)
    s!"Subtree root raw value mismatch, expected {oldRaw}, got {newRaw}"

  let next := subtree.nextNodeIndex
  let trailingZero := (List.range subtree.nodeVisits.size).all fun i =>
    if i < next then true else subtree.nodeVisits.getD i 0 = 0
  LeanTest.assertTrue trailingZero "Subtree should compact retained nodes contiguously from index 0"

@[test]
def testMctxGetSubtreeOnUnvisitedActionResets : IO Unit := do
  let root : RootFnOutput Unit := {
    priorLogits := #[0.0, 1.0, 2.0, 3.0]
    value := 0.0
    embedding := ()
  }
  let recurrent : RecurrentFn Unit Unit := fun _ _ _ _ =>
    ({ reward := 0.0, discount := 0.0, priorLogits := #[0.0, 0.0, 0.0, 0.0], value := 0.0 }, ())
  let out := muzeroPolicy
    (params := ())
    (rngKey := 0)
    (root := root)
    (recurrentFn := recurrent)
    (numSimulations := 1)
    (dirichletFraction := 0.0)

  let mut unvisited : Option Nat := none
  for a in [:4] do
    let idx := (out.searchTree.childrenIndex.getD ROOT_INDEX #[]).getD a (some 0)
    if idx.isNone && unvisited.isNone then
      unvisited := some a
  LeanTest.assertTrue unvisited.isSome "Expected at least one unvisited root action after one simulation"
  let action := unvisited.getD 0

  let subtree := getSubtree out.searchTree action
  LeanTest.assertTrue ((List.range subtree.nodeVisits.size).all fun i => subtree.nodeVisits.getD i 1 = 0)
    "Subtree of an unvisited root action should be reset"

@[test]
def testMctxResetSearchTreePreservesExtraData : IO Unit := do
  let root : RootFnOutput Unit := {
    priorLogits := #[0.1, 0.2]
    value := 1.0
    embedding := ()
  }
  let tree := instantiateTreeFromRoot root 4 #[false, true] (42 : Nat)
  let reset := resetSearchTree tree

  LeanTest.assertEqual reset.extraData 42 "Reset should preserve extraData"
  LeanTest.assertTrue ((List.range reset.nodeVisits.size).all fun i => reset.nodeVisits.getD i 1 = 0)
    "Reset tree should clear node visits"
  LeanTest.assertTrue ((List.range reset.rootInvalidActions.size).all fun i => reset.rootInvalidActions.getD i true = false)
    "Reset tree should clear root invalid-action mask"
