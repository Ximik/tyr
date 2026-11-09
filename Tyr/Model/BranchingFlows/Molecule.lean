import Tyr.Model.BranchingFlows.DiffEq
import Tyr.Model.BranchingFlows.Discrete

/-!
  Molecule-shaped helpers for BranchingFlows.

  QM9 elements are atom records: a continuous 3D coordinate and a discrete atom
  label.  This module keeps that state boundary explicit while reusing the
  generic BranchingFlows sampler and the scalar OU bridge from the DiffEq
  adapter module.
-/

namespace torch.branching

/-! ## 3D coordinates -/

structure Vec3 where
  x : Float
  y : Float
  z : Float
  deriving Repr, BEq, Inhabited

namespace Vec3

def zero : Vec3 := { x := 0.0, y := 0.0, z := 0.0 }

def map2 (f : Float → Float → Float) (a b : Vec3) : Vec3 :=
  { x := f a.x b.x, y := f a.y b.y, z := f a.z b.z }

def scale (c : Float) (v : Vec3) : Vec3 :=
  { x := c * v.x, y := c * v.y, z := c * v.z }

def add (a b : Vec3) : Vec3 :=
  map2 (fun x y => x + y) a b

def weightedAverage (a b : Vec3) (w1 w2 : Nat) : Vec3 :=
  let total := (w1 + w2).toFloat
  if total == 0.0 then
    a
  else
    scale (1.0 / total) (add (scale w1.toFloat a) (scale w2.toFloat b))

def ouBridge (cfg : OUBridgeConfig) (x0 anchor : Vec3) (t0 t : Float) : Vec3 :=
  { x := OUBridgeConfig.bridge cfg x0.x anchor.x t0 t,
    y := OUBridgeConfig.bridge cfg x0.y anchor.y t0 t,
    z := OUBridgeConfig.bridge cfg x0.z anchor.z t0 t }

def sampleOUBridge (cfg : OUBridgeConfig) (x0 anchor : Vec3) (t0 t : Float)
    (rng : Rng) : Vec3 × Rng :=
  let (x, rng) := OUBridgeConfig.sampleBridge cfg x0.x anchor.x t0 t rng
  let (y, rng) := OUBridgeConfig.sampleBridge cfg x0.y anchor.y t0 t rng
  let (z, rng) := OUBridgeConfig.sampleBridge cfg x0.z anchor.z t0 t rng
  ({ x, y, z }, rng)

end Vec3

/-! ## Atom records -/

structure MoleculeAtom where
  coord : Vec3
  label : Nat
  deriving Repr, BEq, Inhabited

structure MoleculeBridgeConfig where
  coordOU : OUBridgeConfig
  maskToken : Nat
  labelDFM : Option DistNoisyDiscreteConfig := none

namespace MoleculeBridgeConfig

/-- QM9 continuous/discrete base-process defaults from the BranchingFlows appendix. -/
def qm9 (vocabSize maskToken : Nat) (hVocab : 0 < vocabSize := by decide)
    (hMask : maskToken < vocabSize := by decide) : MoleculeBridgeConfig :=
  { coordOU := OUBridgeConfig.logLinearVariance 5.0 10.0 0.001,
    maskToken,
    labelDFM := some (DistNoisyDiscreteConfig.qm9 vocabSize maskToken hVocab hMask) }

def maskedAtom (cfg : MoleculeBridgeConfig) (coord : Vec3 := Vec3.zero) : MoleculeAtom :=
  { coord, label := cfg.maskToken }

/--
Anchor merge for molecule atoms.

Coordinates use the continuous weighted average.  The discrete label is replaced
by the mask token, matching the Julia `canonical_anchor_merge` behavior for
discrete states and avoiding endpoint-label leakage through internal anchors.
-/
def anchorMerge (cfg : MoleculeBridgeConfig)
    (a b : MoleculeAtom) (w1 w2 : Nat) : MoleculeAtom :=
  { coord := Vec3.weightedAverage a.coord b.coord w1 w2,
    label := cfg.maskToken }

/--
Molecule atom bridge.

Coordinates use the endpoint-conditioned OU bridge.  Labels use the optional
Flowfusion-style DFM bridge when configured, otherwise they retain the source
label until the terminal endpoint.
-/
def bridge (cfg : MoleculeBridgeConfig)
    (x0 anchor : MoleculeAtom) (t0 t : Float) : MoleculeAtom :=
  let terminal := max cfg.coordOU.terminalTime t0
  let label :=
    if t >= terminal then
      anchor.label
    else
      match cfg.labelDFM with
      | some dfm => dfm.modeBridgeFrom x0.label anchor.label t0 t
      | none => x0.label
  { coord := Vec3.ouBridge cfg.coordOU x0.coord anchor.coord t0 t,
    label }

def sampleBridge (cfg : MoleculeBridgeConfig)
    (x0 anchor : MoleculeAtom) (t0 t : Float) (rng : Rng) : MoleculeAtom × Rng :=
  let terminal := max cfg.coordOU.terminalTime t0
  let (coord, rng) := Vec3.sampleOUBridge cfg.coordOU x0.coord anchor.coord t0 t rng
  if t >= terminal then
    ({ coord, label := anchor.label }, rng)
  else
    match cfg.labelDFM with
    | some dfm =>
        let (label, rng) := dfm.bridgeFrom x0.label anchor.label t0 t rng
        ({ coord, label }, rng)
    | none =>
        ({ coord, label := x0.label }, rng)

def sampleInitialAtom (cfg : MoleculeBridgeConfig) (rng : Rng) : MoleculeAtom × Rng :=
  let (x, rng) := randNormal rng
  let (y, rng) := randNormal rng
  let (z, rng) := randNormal rng
  ({ coord := { x, y, z }, label := cfg.maskToken }, rng)

private def modeOfLogits (logits : Array Float) (fallback : Nat) : Nat :=
  if logits.isEmpty then
    fallback
  else Id.run do
    let mut out := 0
    let mut best := logits[0]!
    for i in [:logits.size] do
      let x := logits[i]!
      if x > best then
        best := x
        out := i
    return out

def stepLabelFromLogits
    (cfg : MoleculeBridgeConfig)
    (currentLabel : Nat)
    (targetLogits : Array Float)
    (t0 t1 : Float) : Nat :=
  match cfg.labelDFM with
  | some dfm => dfm.stepLabelMode currentLabel targetLogits t0 t1
  | none => modeOfLogits targetLogits currentLabel

def sampleStepLabelFromLogits
    (cfg : MoleculeBridgeConfig)
    (currentLabel : Nat)
    (targetLogits : Array Float)
    (t0 t1 : Float)
    (rng : Rng) : Nat × Rng :=
  match cfg.labelDFM with
  | some dfm => dfm.stepLabel currentLabel targetLogits t0 t1 rng
  | none => (modeOfLogits targetLogits currentLabel, rng)

def stepAtomFromLogits
    (cfg : MoleculeBridgeConfig)
    (current : MoleculeAtom)
    (coordTarget : Vec3)
    (labelLogits : Array Float)
    (t0 t1 : Float) : MoleculeAtom :=
  { coord := Vec3.ouBridge cfg.coordOU current.coord coordTarget t0 t1,
    label := cfg.stepLabelFromLogits current.label labelLogits t0 t1 }

def sampleStepAtomFromLogits
    (cfg : MoleculeBridgeConfig)
    (current : MoleculeAtom)
    (coordTarget : Vec3)
    (labelLogits : Array Float)
    (t0 t1 : Float)
    (rng : Rng) : MoleculeAtom × Rng :=
  let (coord, rng) := Vec3.sampleOUBridge cfg.coordOU current.coord coordTarget t0 t1 rng
  let (label, rng) := cfg.sampleStepLabelFromLogits current.label labelLogits t0 t1 rng
  ({ coord, label }, rng)

end MoleculeBridgeConfig

def maskDeletedMoleculeLabels
    (cfg : MoleculeBridgeConfig)
    (state : BranchingState MoleculeAtom) :
    BranchingState MoleculeAtom :=
  { state with
    state := state.state.mapIdx (fun i atom =>
      if state.del.getD i false then
        { atom with label := cfg.maskToken }
      else
        atom) }

structure MoleculeModelPrediction where
  coordTargets : Array Vec3
  labelLogits : Array (Array Float)
  splitLogits : Array Float
  delLogits : Array Float
  deriving Repr, Inhabited

namespace MoleculeModelPrediction

def toBranchingStepPrediction
    (prediction : MoleculeModelPrediction)
    (cfg : MoleculeBridgeConfig)
    (x : BranchingState MoleculeAtom)
    (s1 s2 : Float) : BranchingStepPrediction MoleculeAtom :=
  let targets := (Array.range x.state.size).map (fun i =>
    let current := x.state.getD i default
    let coordTarget := prediction.coordTargets.getD i current.coord
    let labelLogits := prediction.labelLogits.getD i #[]
    cfg.stepAtomFromLogits current coordTarget labelLogits s1 s2)
  { targets,
    splitLogits := prediction.splitLogits,
    delLogits := prediction.delLogits }

/-- Sample the continuous OU and discrete DFM base processes independently for
each current particle before applying split and deletion events. -/
def sampleBranchingStepPrediction
    (prediction : MoleculeModelPrediction)
    (cfg : MoleculeBridgeConfig)
    (x : BranchingState MoleculeAtom)
    (s1 s2 : Float)
    (rng : Rng) : BranchingStepPrediction MoleculeAtom × Rng := Id.run do
  let mut rng := rng
  let mut targets : Array MoleculeAtom := #[]
  for i in [:x.state.size] do
    let current := x.state.getD i default
    let coordTarget := prediction.coordTargets.getD i current.coord
    let labelLogits := prediction.labelLogits.getD i #[]
    let (target, rng') := cfg.sampleStepAtomFromLogits current coordTarget labelLogits s1 s2 rng
    rng := rng'
    targets := targets.push target
  return ({
    targets,
    splitLogits := prediction.splitLogits,
    delLogits := prediction.delLogits
  }, rng)

end MoleculeModelPrediction

/--
Julia's forward `CoalescentFlow` step suppresses a split whenever a discrete
component changes in the same integration interval.  Molecule states carry the
discrete component as `label`, so equality of the old/new labels is precisely
the corresponding admissibility predicate.
-/
def moleculeSplitAllowedAfterBaseStep (old new : MoleculeAtom) : Bool :=
  old.label == new.label

/--
Base-step adapter for `MoleculeModelPrediction.toBranchingStepPrediction`.

The prediction already contains the one-step OU/DFM molecule state, so the
generic BranchingFlows forward step only needs to apply split/deletion events.
-/
def moleculePredictedBaseStep
    (_cfg : MoleculeBridgeConfig)
    (_x : BranchingState MoleculeAtom)
    (targets : Array MoleculeAtom)
    (_s1 _s2 : Float) : Array MoleculeAtom :=
  targets

def moleculeBranchingStep
    (flow : CoalescentFlow MoleculeBridgeConfig MoleculeAtom)
    (x : BranchingState MoleculeAtom)
    (prediction : MoleculeModelPrediction)
    (s1 s2 : Float)
    (splitAllowedAfterBaseStep : MoleculeAtom → MoleculeAtom → Bool := moleculeSplitAllowedAfterBaseStep)
    (rng : Rng := { state := 0 }) :
    BranchingStepResult MoleculeAtom × Rng :=
  branchingStep moleculePredictedBaseStep flow x
    (prediction.toBranchingStepPrediction flow.base x s1 s2)
    s1 s2 splitAllowedAfterBaseStep rng

def moleculeBranchingStepSampled
    (flow : CoalescentFlow MoleculeBridgeConfig MoleculeAtom)
    (x : BranchingState MoleculeAtom)
    (prediction : MoleculeModelPrediction)
    (s1 s2 : Float)
    (splitAllowedAfterBaseStep : MoleculeAtom → MoleculeAtom → Bool := moleculeSplitAllowedAfterBaseStep)
    (rng : Rng := { state := 0 }) :
    BranchingStepResult MoleculeAtom × Rng :=
  let (sampledPrediction, rng) :=
    prediction.sampleBranchingStepPrediction flow.base x s1 s2 rng
  branchingStep moleculePredictedBaseStep flow x sampledPrediction
    s1 s2 splitAllowedAfterBaseStep rng

def moleculeBranchingGenerate
    (flow : CoalescentFlow MoleculeBridgeConfig MoleculeAtom)
    (x0 : BranchingState MoleculeAtom)
    (model : Float → BranchingState MoleculeAtom → MoleculeModelPrediction)
    (schedule : Array Float)
    (splitAllowedAfterBaseStep : MoleculeAtom → MoleculeAtom → Bool := moleculeSplitAllowedAfterBaseStep)
    (rng : Rng := { state := 0 }) :
    BranchingGenerateResult MoleculeAtom × Rng := Id.run do
  if schedule.size <= 1 then
    return ({ finalState := x0, trajectory := #[x0], events := #[], times := schedule }, rng)
  let mut state := x0
  let mut trajectory : Array (BranchingState MoleculeAtom) := #[x0]
  let mut events : Array (Array BranchingStepEvent) := #[]
  let mut rng := rng
  for i in [:schedule.size - 1] do
    let s1 := schedule[i]!
    let s2 := schedule[i + 1]!
    let prediction := model s1 state
    let (result, rng') :=
      moleculeBranchingStep flow state prediction s1 s2 splitAllowedAfterBaseStep rng
    rng := rng'
    state := result.state
    trajectory := trajectory.push state
    events := events.push result.events
  ({ finalState := state, trajectory, events, times := schedule }, rng)

def moleculeBranchingGenerateIO
    (flow : CoalescentFlow MoleculeBridgeConfig MoleculeAtom)
    (x0 : BranchingState MoleculeAtom)
    (model : Float → BranchingState MoleculeAtom → IO MoleculeModelPrediction)
    (schedule : Array Float)
    (splitAllowedAfterBaseStep : MoleculeAtom → MoleculeAtom → Bool := moleculeSplitAllowedAfterBaseStep)
    (maxStateLen? : Option Nat := none)
    (rng : Rng := { state := 0 }) :
    IO (BranchingGenerateResult MoleculeAtom × Rng) := do
  if schedule.size <= 1 then
    return ({ finalState := x0, trajectory := #[x0], events := #[], times := schedule }, rng)
  let mut state := x0
  let mut trajectory : Array (BranchingState MoleculeAtom) := #[x0]
  let mut events : Array (Array BranchingStepEvent) := #[]
  let mut rng := rng
  for i in [:schedule.size - 1] do
    let s1 := schedule[i]!
    let s2 := schedule[i + 1]!
    let prediction ← model s1 state
    let (result, rng') :=
      moleculeBranchingStepSampled flow state prediction s1 s2 splitAllowedAfterBaseStep rng
    rng := rng'
    match maxStateLen? with
    | some limit =>
        if result.state.state.size > limit then
          throw (IO.userError s!"molecule generation exceeded maxStateLen={limit} at step {i + 1}: produced {result.state.state.size} particles")
    | none => pure ()
    state := result.state
    trajectory := trajectory.push state
    events := events.push result.events
  return ({ finalState := state, trajectory, events, times := schedule }, rng)

private def lineageKindName : LineageEventKind → String
  | .split => "split"
  | .delete => "delete"

private def jsonNatOption : Option Nat → String
  | some value => toString value
  | none => "null"

/--
Serialize raw molecule states together with reconstructed runtime lineage.

Coordinates are written without display offsets or inferred correspondences.
Every state row carries the stable particle id assigned by
`reconstructLineage`; split/delete event rows carry the corresponding parent
and child ids.
-/
def moleculeTrajectoryJsonl
    (result : BranchingGenerateResult MoleculeAtom)
    (appendOnSplit : Bool := false) : Except String String := do
  let trace ← reconstructLineage result appendOnSplit
  if trace.frames.size != result.trajectory.size then
    throw s!"lineage frame count {trace.frames.size} does not match trajectory frame count {result.trajectory.size}"
  let mut rows : Array String := #[
    "{\"type\":\"metadata\",\"schema\":\"tyr.branching-trajectory.v1\",\"coordinates\":\"raw\"}"
  ]
  for frameIndex in [:trace.frames.size] do
    let frame := trace.frames[frameIndex]!
    let state := result.trajectory[frameIndex]!
    if !Float.isFinite frame.time then
      throw s!"non-finite trajectory time while serializing frame {frameIndex}"
    if frame.particles.size != state.state.size then
      throw s!"lineage/state size mismatch while serializing frame {frameIndex}"
    for stateIndex in [:state.state.size] do
      let particle := frame.particles[stateIndex]!
      let atom := state.state[stateIndex]!
      if !(Float.isFinite atom.coord.x && Float.isFinite atom.coord.y &&
          Float.isFinite atom.coord.z) then
        throw s!"non-finite molecule coordinate while serializing frame {frameIndex}, state index {stateIndex}"
      rows := rows.push ("{" ++ s!"\"type\":\"state\",\"step\":{frame.step},\"t\":{frame.time},\"state_index\":{stateIndex},\"particle_id\":{particle.particleId},\"parent_id\":{jsonNatOption particle.parentId?},\"birth_event_id\":{jsonNatOption particle.birthEventId?},\"label\":{atom.label},\"x\":{atom.coord.x},\"y\":{atom.coord.y},\"z\":{atom.coord.z}" ++ "}")
  for event in trace.events do
    if !(Float.isFinite event.t0 && Float.isFinite event.t1) then
      throw s!"non-finite trajectory interval while serializing event {event.eventId}"
    let children := String.intercalate "," (event.childIds.map toString).toList
    rows := rows.push ("{" ++ s!"\"type\":\"event\",\"event_id\":{event.eventId},\"kind\":\"{lineageKindName event.kind}\",\"t0\":{event.t0},\"t1\":{event.t1},\"parent_id\":{event.parentId},\"child_ids\":[{children}]" ++ "}")
  return String.intercalate "\n" rows.toList ++ "\n"

def writeMoleculeTrajectoryJsonl
    (path : System.FilePath)
    (result : BranchingGenerateResult MoleculeAtom)
    (appendOnSplit : Bool := false) : IO Unit := do
  match moleculeTrajectoryJsonl result appendOnSplit with
  | .ok content => IO.FS.writeFile path content
  | .error message => throw (IO.userError message)

end torch.branching
