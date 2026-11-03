import Tyr.DiffEq
import Tyr.EventSkeleton.Interval
import Tyr.Model.BranchingFlows

/-!
  DiffEq adapters for the BranchingFlows port.

  The core BranchingFlows sampler accepts a user-supplied bridge
  `P -> x0 -> x1 -> t0 -> t1 -> xt`.  This module makes Tyr DiffEq solvers
  usable at that seam without teaching the sampler about ODEs, SDEs, or event
  localization internals.
-/

namespace torch.branching

open torch.DiffEq
open Tyr.EventSkeleton

/-- Configuration for using a Tyr `DiffEq` solver as a BranchingFlows bridge.

`term` is parameterized by the future anchor `x1`, matching Flowfusion-style
conditional bridges where the base process may depend on the endpoint.  For an
unconditioned process, ignore the anchor.
-/
structure DiffEqBridgeConfig
    (Term Y VF Control Args Controller : Type) where
  term : Y → Term
  solver : AbstractSolver Term Y VF Control Args
  args : Args
  dt0 : Option Time := none
  maxSteps : Nat := 4096
  controller : Controller
  events : Array (EventSpec Y Args) := #[]
  eventTree : Option (EventTree Y Args) := none
  saveat : SaveAt := { t1 := true }
  throwOnFailure : Bool := false

namespace DiffEqBridgeConfig

def solve
    {Term Y VF Control Args Controller : Type}
    [DiffEqSpace Y]
    [DiffEqSeminorm Y]
    [DiffEqElem Y]
    [Inhabited Y]
    [StepSizeController Controller]
    [StepSizeControllerValidation Controller]
    [Inhabited Controller]
    (cfg : DiffEqBridgeConfig Term Y VF Control Args Controller)
    (x0 anchor : Y)
    (t0 t1 : Float) :
    Solution Y cfg.solver.SolverState (StepSizeController.State (C := Controller)) :=
  diffeqsolve
    (Term := Term)
    (Y := Y)
    (VF := VF)
    (Control := Control)
    (Args := Args)
    (Controller := Controller)
    (cfg.term anchor)
    cfg.solver
    t0
    t1
    cfg.dt0
    x0
    cfg.args
    (saveat := cfg.saveat)
    (maxSteps := cfg.maxSteps)
    (controller := cfg.controller)
    (events := cfg.events)
    (eventTree := cfg.eventTree)
    (throwOnFailure := cfg.throwOnFailure)

def finalStateOr
    {Y SolverState ControllerState : Type}
    [Inhabited Y]
    (fallback : Y)
    (sol : Solution Y SolverState ControllerState) : Y :=
  match sol.ys with
  | some ys =>
      match ys.back? with
      | some y => y
      | none => fallback
  | none =>
      match sol.interpolation with
      | some interp => interp.evaluate sol.t1 none true
      | none => fallback

/-- Drop-in BranchingFlows bridge backed by `diffeqsolve`. -/
def bridge
    {Term Y VF Control Args Controller : Type}
    [DiffEqSpace Y]
    [DiffEqSeminorm Y]
    [DiffEqElem Y]
    [Inhabited Y]
    [StepSizeController Controller]
    [StepSizeControllerValidation Controller]
    [Inhabited Controller]
    (cfg : DiffEqBridgeConfig Term Y VF Control Args Controller)
    (x0 anchor : Y)
    (t0 t1 : Float) : Y :=
  finalStateOr x0 (cfg.solve x0 anchor t0 t1)

end DiffEqBridgeConfig

/-- ODE-specialized bridge configuration. -/
abbrev ODEBridgeConfig (Y Args Controller : Type) :=
  DiffEqBridgeConfig (ODETerm Y Args) Y Y Time Args Controller

namespace ODEBridgeConfig

def mk
    {Y Args Controller : Type}
    [DiffEqSpace Y]
    (vectorField : Y → Time → Y → Args → Y)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (args : Args)
    (controller : Controller)
    (dt0 : Option Time := none)
    (maxSteps : Nat := 4096)
    (events : Array (EventSpec Y Args) := #[])
    (eventTree : Option (EventTree Y Args) := none)
    (saveat : SaveAt := { t1 := true }) :
    ODEBridgeConfig Y Args Controller :=
  {
    term := fun anchor => { vectorField := fun t y args => vectorField anchor t y args }
    solver := solver
    args := args
    dt0 := dt0
    maxSteps := maxSteps
    controller := controller
    events := events
    eventTree := eventTree
    saveat := saveat
  }

end ODEBridgeConfig

/-- SDE-specialized bridge configuration using Tyr's drift+diffusion `MultiTerm`. -/
abbrev SDEBridgeConfig
    (Drift Diffusion Y VFd VFg Control Args Controller : Type) :=
  DiffEqBridgeConfig
    (SDETerm Drift Diffusion)
    Y
    (VFd × VFg)
    (Time × Control)
    Args
    Controller

namespace SDEBridgeConfig

def mk
    {Drift Diffusion Y VFd VFg Control Args Controller : Type}
    (drift : Y → Drift)
    (diffusion : Y → Diffusion)
    (solver :
      AbstractSolver
        (SDETerm Drift Diffusion)
        Y
        (VFd × VFg)
        (Time × Control)
        Args)
    (args : Args)
    (controller : Controller)
    (dt0 : Option Time := none)
    (maxSteps : Nat := 4096)
    (events : Array (EventSpec Y Args) := #[])
    (eventTree : Option (EventTree Y Args) := none)
    (saveat : SaveAt := { t1 := true }) :
    SDEBridgeConfig Drift Diffusion Y VFd VFg Control Args Controller :=
  {
    term := fun anchor => SDETerm.mk (drift anchor) (diffusion anchor)
    solver := solver
    args := args
    dt0 := dt0
    maxSteps := maxSteps
    controller := controller
    events := events
    eventTree := eventTree
    saveat := saveat
  }

end SDEBridgeConfig

/-! ## Analytic endpoint-conditioned OU bridge -/

private def clampTime (lo hi t : Float) : Float :=
  max lo (min hi t)

/--
Scalar endpoint-conditioned Ornstein-Uhlenbeck bridge.

The underlying process is
`dX_t = -theta * X_t dt + sqrt(diffusionVariance t) dW_t`, conditioned on
`X_terminalTime = anchor`.  The returned bridge mean is analytic up to numerical
quadrature of the time-varying transition variance, so it can be used as a
paper-faithful continuous base-process target without running a generic solver.
-/
structure OUBridgeConfig where
  theta : Float
  diffusionVariance : Float → Float
  terminalTime : Float := 1.0
  integrationSteps : Nat := 128
  integrationStepsPos : 0 < integrationSteps := by decide
  minVariance : Float := 1.0e-12

namespace OUBridgeConfig

def constantVariance (theta variance : Float) : OUBridgeConfig :=
  { theta := theta,
    diffusionVariance := fun _ => max variance 0.0 }

def logLinearVariance
    (theta startVariance endVariance : Float)
    (terminalTime : Float := 1.0)
    (integrationSteps : Nat := 128)
    (integrationStepsPos : 0 < integrationSteps := by decide) :
    OUBridgeConfig :=
  let startVariance := max startVariance 1.0e-12
  let endVariance := max endVariance 1.0e-12
  let logStart := Float.log startVariance
  let logEnd := Float.log endVariance
  let horizon := max terminalTime 1.0e-12
  { theta := theta,
    diffusionVariance := fun t =>
      let u := clampTime 0.0 1.0 (t / horizon)
      Float.exp (logStart + (logEnd - logStart) * u),
    terminalTime := terminalTime,
    integrationSteps := integrationSteps,
    integrationStepsPos := integrationStepsPos }

def transitionCoeff (cfg : OUBridgeConfig) (t0 t1 : Float) : Float :=
  if t1 <= t0 then 1.0 else Float.exp (-(cfg.theta) * (t1 - t0))

def transitionVariance (cfg : OUBridgeConfig) (t0 t1 : Float) : Float := Id.run do
  if t1 <= t0 then
    return 0.0
  let steps := cfg.integrationSteps
  let dt := (t1 - t0) / steps.toFloat
  let mut acc := 0.0
  for i in [:steps] do
    let u := t0 + (i.toFloat + 0.5) * dt
    let kernel := Float.exp (-2.0 * cfg.theta * (t1 - u))
    acc := acc + kernel * max (cfg.diffusionVariance u) 0.0 * dt
  return max acc 0.0

/-- Conditional OU bridge mean from `(t0, x0)` to `anchor` at `terminalTime`. -/
def bridgeMean (cfg : OUBridgeConfig) (x0 anchor : Float) (t0 t : Float) : Float :=
  let terminal := max cfg.terminalTime t0
  let t := clampTime t0 terminal t
  if t <= t0 then
    x0
  else if t >= terminal then
    anchor
  else
    let v0T := max (cfg.transitionVariance t0 terminal) cfg.minVariance
    let v0t := cfg.transitionVariance t0 t
    let a0t := cfg.transitionCoeff t0 t
    let a0T := cfg.transitionCoeff t0 terminal
    let atT := cfg.transitionCoeff t terminal
    let gain := atT * v0t / v0T
    a0t * x0 + gain * (anchor - a0T * x0)

/-- Conditional OU bridge variance at `t` given `(t0, x0)` and terminal anchor. -/
def bridgeVariance (cfg : OUBridgeConfig) (t0 t : Float) : Float :=
  let terminal := max cfg.terminalTime t0
  let t := clampTime t0 terminal t
  if t <= t0 || t >= terminal then
    0.0
  else
    let v0T := max (cfg.transitionVariance t0 terminal) cfg.minVariance
    let v0t := cfg.transitionVariance t0 t
    let atT := cfg.transitionCoeff t terminal
    max (v0t - (atT * v0t) * (atT * v0t) / v0T) 0.0

/-- Deterministic bridge value, using the conditional mean. -/
def bridge (cfg : OUBridgeConfig) (x0 anchor : Float) (t0 t : Float) : Float :=
  cfg.bridgeMean x0 anchor t0 t

def sampleBridge
    (cfg : OUBridgeConfig)
    (x0 anchor : Float)
    (t0 t : Float)
    (rng : Rng) : Float × Rng :=
  let mean := cfg.bridgeMean x0 anchor t0 t
  let variance := cfg.bridgeVariance t0 t
  if variance <= 0.0 then
    (mean, rng)
  else
    let (z, rng') := randNormal rng
    (mean + Float.sqrt variance * z, rng')

end OUBridgeConfig

namespace Segment

/-- Project one BranchingFlows segment into the interval skeleton vocabulary. -/
def toAcceptedStepSegment
    (segmentId attemptIndex : Nat)
    (seg : Segment α) :
    AcceptedStepSegment :=
  {
    id := segmentId
    attemptIndex := attemptIndex
    tStart := seg.lastCoalescence
    tAttempt := seg.t
    tAfter := seg.t
    madeJumpBefore := false
    madeJumpAfter := seg.descendants > 1
    label := s!"branching-segment:{seg.id}"
  }

def branchVertex?
    (vertexId : VertexId)
    (seg : Segment α) :
    Option SkeletonVertex :=
  if seg.descendants > 1 then
    some {
      id := vertexId
      kind := .branch
      label := s!"branching-split:{seg.id}:w={seg.descendants}"
    }
  else
    none

def branchAggregateMove?
    (vertexId : VertexId)
    (seg : Segment α) :
    Option SkeletonMove :=
  if seg.descendants > 1 then
    some {
      kind := .branchAggregate
      targets := #[vertexId]
      label := s!"branching-aggregate:{seg.id}:w={seg.descendants}"
    }
  else
    none

end Segment

namespace BranchingStepEvent

def hasTopologyEvent (event : BranchingStepEvent) : Bool :=
  event.splitCount > 0 || event.deleted

/-- Project one forward-generation step event into the interval skeleton vocabulary. -/
def toAcceptedStepSegment
    (segmentId attemptIndex : Nat)
    (event : BranchingStepEvent) :
    AcceptedStepSegment :=
  {
    id := segmentId
    attemptIndex := attemptIndex
    tStart := event.t0
    tAttempt := event.t1
    tAfter := event.t1
    madeJumpBefore := false
    madeJumpAfter := event.hasTopologyEvent
    label := s!"branching-step:{event.sourceId}:i={event.sourceIndex}"
  }

def branchVertex?
    (vertexId : VertexId)
    (event : BranchingStepEvent) :
    Option SkeletonVertex :=
  if event.hasTopologyEvent then
    some {
      id := vertexId
      kind := .branch
      label :=
        if event.deleted then
          s!"branching-delete:{event.sourceId}:splits={event.splitCount}"
        else
          s!"branching-split:{event.sourceId}:splits={event.splitCount}"
    }
  else
    none

def branchAggregateMove?
    (vertexId : VertexId)
    (event : BranchingStepEvent) :
    Option SkeletonMove :=
  if event.hasTopologyEvent then
    some {
      kind := .branchAggregate
      targets := #[vertexId]
      label :=
        if event.deleted then
          s!"branching-delete-aggregate:{event.sourceId}"
        else
          s!"branching-split-aggregate:{event.sourceId}:splits={event.splitCount}"
    }
  else
    none

end BranchingStepEvent

/-- Event-skeleton graph for a flat BranchingFlows segment collection. -/
def graphFromBranchingSegments
    (segments : Array (Segment α))
    (checkpoint : Bool := true)
    (branchMoves : Bool := true) :
    SkeletonGraph := Id.run do
  let intervalSegments :=
    segments.mapIdx (fun i seg => Segment.toAcceptedStepSegment i i seg)
  let mut g := graphFromAcceptedSegments intervalSegments checkpoint
  if branchMoves then
    for i in [:segments.size] do
      match segments[i]? with
      | none => pure ()
      | some seg =>
          let branchVertexId := segments.size + i
          match Segment.branchVertex? branchVertexId seg with
          | some v => g := g.addVertex v
          | none => pure ()
          match Segment.branchAggregateMove? branchVertexId seg with
          | some m => g := g.addMove m
          | none => pure ()
  return g

def graphFromBranchingBridgeBatch?
    (result : BranchingBridgeResult α)
    (batchIndex : Nat)
    (checkpoint : Bool := true)
    (branchMoves : Bool := true) :
    Option SkeletonGraph :=
  match result.segments[batchIndex]? with
  | some segments => some (graphFromBranchingSegments segments checkpoint branchMoves)
  | none => none

def graphsFromBranchingBridgeResult
    (result : BranchingBridgeResult α)
    (checkpoint : Bool := true)
    (branchMoves : Bool := true) :
    Array SkeletonGraph :=
  result.segments.map (fun segments =>
    graphFromBranchingSegments segments checkpoint branchMoves)

def graphFromBranchingStepEvents
    (events : Array BranchingStepEvent)
    (checkpoint : Bool := true)
    (branchMoves : Bool := true) :
    SkeletonGraph := Id.run do
  let intervalSegments :=
    events.mapIdx (fun i event => BranchingStepEvent.toAcceptedStepSegment i i event)
  let mut g := graphFromAcceptedSegments intervalSegments checkpoint
  if branchMoves then
    for i in [:events.size] do
      match events[i]? with
      | none => pure ()
      | some event =>
          let branchVertexId := events.size + i
          match BranchingStepEvent.branchVertex? branchVertexId event with
          | some v => g := g.addVertex v
          | none => pure ()
          match BranchingStepEvent.branchAggregateMove? branchVertexId event with
          | some m => g := g.addMove m
          | none => pure ()
  return g

def graphFromBranchingGenerateResult
    (result : BranchingGenerateResult α)
    (checkpoint : Bool := true)
    (branchMoves : Bool := true) :
    SkeletonGraph := Id.run do
  let mut g : SkeletonGraph := {}
  if result.times.size <= 1 then
    return g
  let mut segmentId := 0
  for i in [:result.times.size - 1] do
    let events := result.events.getD i #[]
    if events.isEmpty then
      let seg : AcceptedStepSegment := {
        id := segmentId
        attemptIndex := i
        tStart := result.times[i]!
        tAttempt := result.times[i + 1]!
        tAfter := result.times[i + 1]!
        label := s!"branching-step-quiet:{i}"
      }
      g := g.addVertex seg.intervalVertex
      for move in planForAcceptedSegment seg checkpoint do
        g := g.addMove move
      segmentId := segmentId + 1
    else
      for event in events do
        let seg := BranchingStepEvent.toAcceptedStepSegment segmentId i event
        g := g.addVertex seg.intervalVertex
        for move in planForAcceptedSegment seg checkpoint do
          g := g.addMove move
        if branchMoves then
          let branchVertexId := result.times.size + segmentId
          match BranchingStepEvent.branchVertex? branchVertexId event with
          | some v => g := g.addVertex v
          | none => pure ()
          match BranchingStepEvent.branchAggregateMove? branchVertexId event with
          | some m => g := g.addMove m
          | none => pure ()
        segmentId := segmentId + 1
  return g

structure BranchingSkeletonSummary where
  batches : Nat
  segments : Nat
  branchSegments : Nat
  deletionSegments : Nat
  frozenSegments : Nat
  deriving Repr, Inhabited

def summarizeBranchingBridgeResult
    (result : BranchingBridgeResult α) :
    BranchingSkeletonSummary := Id.run do
  let mut segments := 0
  let mut branchSegments := 0
  let mut deletionSegments := 0
  let mut frozenSegments := 0
  for batch in result.segments do
    for seg in batch do
      segments := segments + 1
      if seg.descendants > 1 then
        branchSegments := branchSegments + 1
      if seg.del then
        deletionSegments := deletionSegments + 1
      if !seg.flowable then
        frozenSegments := frozenSegments + 1
  return {
    batches := result.segments.size
    segments := segments
    branchSegments := branchSegments
    deletionSegments := deletionSegments
    frozenSegments := frozenSegments
  }

end torch.branching
