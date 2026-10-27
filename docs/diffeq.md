# Differential equations

`Tyr/DiffEq/` is a backend-agnostic ODE/SDE integration library modeled on JAX's
diffrax (the tests are explicit parity suites against diffrax behavior). It
provides a term/path problem representation, ~35 solvers (explicit, implicit,
IMEX, stochastic, symplectic, underdamped-Langevin), adaptive step-size control,
dense output, event handling, and several adjoint modes for differentiating
through solves. Use it for neural ODEs/SDEs, continuous-time generative models,
physics simulation, and training loops that differentiate through an integrator.

Everything lives in namespace `torch.DiffEq`. One umbrella import pulls it all in:

```lean
import Tyr.DiffEq
-- For differentiating through solves on tensor states, also:
import Tyr.DiffEq.Adjoint.Torch
```

`Tyr.DiffEq.Adjoint` re-exports only `Adjoint/Core`; the `AdjointBackend`
instances for `TensorStruct` states live in `Adjoint/Torch.lean` and need the
separate import (see `Tests/TestDiffEqAdjoint.lean:3`).

## Architecture and main abstractions

The design mirrors diffrax's layering: *state algebra* → *terms* → *solvers* →
*step-size controllers* → *`diffeqsolve`*; adjoints and events hook into the
same solve loop.

### State algebra

Solvers are generic over the state type `Y`. Any type with the three instances
below can be a solver state (`Tyr/DiffEq/Types.lean:21`):

```lean
class DiffEqSpace (α : Type) where
  add : α → α → α
  sub : α → α → α
  scale : Scalar → α → α

class DiffEqSeminorm (α : Type) where
  rms : α → Scalar            -- used by adaptive controllers / error estimates

class DiffEqElem (α : Type) where
  abs : α → α
  max : α → α → α
  addScalar : Scalar → α → α
  div : α → α → α             -- elementwise; used for scaled error norms
```

`Time` and `Scalar` are both `Float` abbreviations. Instances exist for
`Float`, indexed `Tensor σ` values, `[TensorStruct α]` model trees (priority 50, so
network parameters can be integrated directly), pairs, `Fin n → α`,
`Vector n α`, `List`, `Array`, `Option`. `Tyr/DiffEq/Typed.lean` adds the same
instances for `Tensor (σ : StaticSpec)` using indexed arithmetic and metadata-checked
preserving scalar operations.
Arithmetic operators are opt-in: bring `DiffEqArithmetic.hAddInst`,
`hSubInst`, `hMulInst`, `smulInst` in as `local instance`s for `+`, `-`, `*`.

### Terms

A *term* is a vector field, a control, and their product (`Tyr/DiffEq/Term.lean:21`):

```lean
class TermLike (τ Y VF Control Args : Type) where
  vf : τ → Time → Y → Args → VF
  contr : τ → Time → Time → Control
  prod : τ → VF → Control → Y
  vf_prod : τ → Time → Y → Args → Control → Y
  is_vf_expensive : τ → Time → Time → Y → Args → Bool
```

Concrete terms:

- `ODETerm Y Args` — `vectorField : Time → Y → Args → Y`; the control is `Time`
  itself (`Term.lean:146`).
- `ControlTerm Y VF Control Args` — custom control path and product; optional
  `controlDerivative?` enables conversion to an ODE via `toODE?`
  (`Term.lean:167`). `ControlTerm.ofPath` builds one from an `AbstractPath`.
- `DiffusionTerm` — a `ControlTerm`-like term carrying a Jacobian-vector
  product (`jacobianProd`) for Milstein-type solvers (`Term.lean:270`).
- `UnderdampedLangevinDriftTerm` / `UnderdampedLangevinDiffusionTerm` —
  underdamped Langevin dynamics on pair states `(position, velocity)` with
  friction `gamma` and scale `u`; the diffusion consumes
  `SpaceTimeTimeLevyArea` controls (`Term.lean:306`, `Term.lean:366`).

Terms compose additively: the `(T1 × T2)` `TermLike` instance sums the two
`prod`s, so a drift + diffusion SDE is just a pair. `MultiTerm`, the
`MultiTerm3..6` abbreviations, and `MultiTermArray` generalize this
(`Term.lean:645`, `Term.lean:742`). `Solver/SDE.lean` defines the aliases:

```lean
abbrev SDETerm (Drift Diffusion : Type) : Type := MultiTerm Drift Diffusion
def SDETerm.mk (drift : Drift) (diffusion : Diffusion) : SDETerm Drift Diffusion
```

### Paths and Brownian controls

Controls are *paths* (`Tyr/DiffEq/Path.lean:11`):

```lean
structure AbstractPath (Control : Type) where
  t0 : Time
  t1 : Time
  evaluate : Time → Option Time → Bool → Control   -- point value or increment
  derivativeFn? : Option (Time → Bool → Control) := none
```

`LinearPath` and `CubicHermitePath` are the reusable interpolants; `compose`
concatenates two paths, `restrict` re-windows one.

Brownian controls (`Tyr/DiffEq/Brownian.lean`) come as increment structures —
`BrownianIncrement` (`dt`, `W`), `SpaceTimeLevyArea` (+ `H`),
`SpaceTimeTimeLevyArea` (+ `K`) — sampled from one of:

- `AbstractBrownianPath` — the interface (`evaluate`, `increment`);
  `.toPath` views it as an `AbstractPath`.
- `UnsafeBrownianPath` — hash-keyed sampling; cheap, but not consistent across
  overlapping intervals.
- `VirtualBrownianTree` — Brownian-bridge tree (`tol`, `maxDepth := 24`,
  `seed`, `shape`); interval-consistent like diffrax's. Entry points:
  `VirtualBrownianTree.increment`, `incrementSpaceTime`, `incrementSpaceTimeTime`.
- `ScalarBrownianPath` — `Float` convenience wrapper (`t0`, `t1`, `seed`).

Sampling is keyed (`PRNGKey.foldIn` over seed and interval endpoints), so
re-querying the same interval reproduces the same increment.

### Solvers

A solver is a *structure*, not a class (`Tyr/DiffEq/Solver/Base.lean:193`):

```lean
structure AbstractSolver (Term Y VF Control Args : Type) where
  SolverState : Type
  DenseInfo : Type
  termStructure : TermStructure
  odeStepAdjoint? : Option ODEStepAdjoint   -- discrete-adjoint tableau data
  order : Term → Nat
  strongOrder : Term → Float
  errorOrder : Term → Float
  init : Term → Time → Time → Y → Args → SolverState
  step : Term → Time → Time → Y → Args → SolverState → Bool → StepOutput Y DenseInfo SolverState
  func : Term → Time → Y → Args → VF
  interpolation : DenseInfo → DenseInterpolation Y
```

Each concrete solver is a thin `structure X` plus `def X.solver : AbstractSolver ...`.
Explicit RK solvers are built from a shared engine (`Solver/RungeKutta.lean`:
`ButcherTableau s`, `ExplicitRK s` with Hermite or quartic-polynomial dense
output); implicit ones from `ImplicitRungeKutta.lean` with a configurable root
finder. Marker traits (`ExplicitSolver`, `ImplicitSolver`, `AdaptiveSolver`,
`ItoSolver`, `StratonovichSolver`, `SymplecticSolver`, `ReversibleSolver`,
`WrappedSolver`) are trivial `Prop` classes per solver — they classify, they
don't enforce laws.

Roster:

| Category | Solvers |
|---|---|
| Explicit ODE | `Euler`, `Heun`, `Midpoint`, `Ralston`, `Bosh3`, `RK4`, `ReversibleHeun`, `Dopri5`, `Tsit5`, `Dopri8`, `LeapfrogMidpoint` |
| Implicit / IMEX | `ImplicitEuler`, `Kvaerno3/4/5` (SDIRK), `Kencarp3/4/5` (IMEX additive RK), `SIL3` |
| SDE (Itô) | `EulerMaruyama` (strong 0.5), `Milstein` (needs `DiffusionTerm`'s JVP) |
| SDE (Stratonovich) | `EulerHeun`, `StratonovichMilstein`, `SRA1`, `SEA`, `ShARK`, `GeneralShARK`, `SPaRK`, `SlowRK` |
| Symplectic | `SemiImplicitEuler` |
| Underdamped Langevin | `ALIGN` (order 2), `ShOULD` (order 3), `QUICSORT` — staged SRKs over `SpaceTimeTimeLevyArea` controls |
| Wrapper | `HalfSolver` — wraps any solver, estimates error from one full step vs two half-steps |

Embedded error estimates (for adaptive control) come from solvers marked
`AdaptiveSolver` (`Tsit5`, `Dopri5`, `Dopri8`, Kvaerno/Kencarp families) and
from `HalfSolver` for everything else.

### Step-size controllers

```lean
class StepSizeController (C : Type) where
  State : Type
  init : ... → C → Term → ... → Option Time → ... → StepSizeState State
  adapt : ... → C → StepSizeState State → ... → Option Y → Float →
    StepSizeDecision State
```

(`Tyr/DiffEq/StepSizeController.lean:60`.) Implementations:

- `ConstantStepSize` — fixed `dt0`; panics if `dt0 = none`.
- `PIDController` — adaptive PID on the scaled RMS error. Defaults: `rtol :=
  1.0e-5`, `atol := 1.0e-8`, `pcoeff := 0.7`, `icoeff := 0.4`, `dcoeff := 0.0`,
  `safety := 0.9`, `factormin := 0.2`, `factormax := 10.0`, optional
  `dtmin`/`dtmax`, `force_dtmin`.
- `StepTo` — step exactly onto a given `ts` grid (validated: endpoints must
  match `t0`/`t1`, strictly monotone in solve direction).
- `ClipStepSizeController Base` — wraps another controller with `dt_min`/`dt_max`
  clamping, `step_ts`/`jump_ts` clipping, rejected-step tracking
  (`store_rejected_steps`), and `madeJump` flags.

When `dt0 = none` and the term is an `ODETerm`, an `InitialStepSelector`
instance picks the first step with diffrax's ODE heuristic; otherwise the
controller falls back to `0.01`.

### SaveAt and Solution

`SaveAt` (`Tyr/DiffEq/SaveAt.lean:49`) controls what the solve records:

| Field | Default | Meaning |
|---|---|---|
| `t0` / `t1` | `false` / `true` | save endpoints |
| `ts` | `none` | save at these times (dense interpolation is used between steps) |
| `steps` | `0` | `StepCadence`: save every *n* accepted steps (`true` coerces to every step) |
| `dense` | `false` | keep per-step dense info, yielding `sol.interpolation` |
| `solverState` / `controllerState` / `madeJump` | `false` | save final internal states / jump flag |
| `subs` | `#[]` | tree-structured `SubSaveAt`s; explicitly empty leaves are rejected (`Result.invalidInput`) for diffrax parity |

The result is a `Solution` (`Tyr/DiffEq/Solution.lean:45`): `t0`, `t1`,
`ts : Option (Array Time)`, `ys : Option (Array Y)`,
`interpolation : Option (DenseInterpolation Y)`, `stats : List (String × Nat)`
(`num_steps`, `num_accepted_steps`, `num_rejected_steps`), `result : Result`,
optional solver/controller states, `madeJump`, and event masks. `Result` has
constructors `successful`, `maxStepsReached`, `dtMinReached`, `eventOccurred`,
`maxStepsRejected`, `internalError`, `invalidInput`; `Result.isOkay` accepts
`successful` and `eventOccurred`. `Solution.evaluate`/`derivative` query the
dense output (they panic unless `dense := true` was saved).

Dense interpolation and `SaveAt.ts` also support reverse-time solves. Interpolation
knots may be increasing or decreasing; `left := true` selects the limit from
smaller physical times, and `left := false` the limit from larger physical times,
including at a discontinuous knot. This convention is independent of solve direction.

### Events

`diffeqsolve` accepts terminating or recording events (`Tyr/DiffEq/Integrate.lean:11`):

```lean
inductive EventCondition (Y Args : Type) where
  | boolean : (Time → Y → Args → Bool) → EventCondition Y Args
  | real : (Time → Y → Args → Float) → EventCondition Y Args

structure EventSpec (Y Args : Type) where
  condition : EventCondition Y Args
  terminate : Bool := true
  direction : Option Bool := none      -- filter up/down crossings
  rootMaxIters : Nat := 24
  rootTol : Time := 1.0e-6
```

Real-valued conditions are localized by bisection on the dense interpolation.
`EventSpec.steadyState` builds a boolean "vector field norm below tolerance"
event. `EventTree` / `EventMaskTree` plus `diffeqsolveEventTree` give
diffrax-style pytree event masks. The read-only bridge
`Tyr/DiffEq/EventSkeleton.lean` projects accepted `SolveLoopAttempt`s into
`Tyr.EventSkeleton` interval segments — see [Event skeleton](event-skeleton.md).

### Adjoints

Differentiating through a solve goes through a VJP backend (`Tyr/DiffEq/Adjoint/Core.lean:22`):

```lean
class AdjointBackend (Y Args : Type) where
  vjp : (Time → Y → Args → Y) → Time → Y → Args → Y → (Y × Y × Args)

structure AdjointResult (Y Args : Type) where
  adjY0 : Y
  adjArgs : Args
```

`Adjoint/Torch.lean` implements this for any `[TensorStruct Y] [TensorStruct Args]`
via `TensorStruct.functionalGrad`: temporary differentiation leaves stay inside a
pure callback boundary and only detached derivatives escape. The primal is evaluated
separately on detached inputs, preserving static metadata without exporting temporary
leaf identities. Entry points:

| Function | Mode |
|---|---|
| `backsolveAdjoint` | continuous adjoint: re-solves the augmented `(y, adjY, adjArgs)` ODE backward over the primal's saved steps (single-term ODEs; primal must be saved with `steps := true`; no events) |
| `diffeqsolveAdjoint`, `diffeqsolveBacksolveAdjoint` | forward solve + backsolve in one call; manages saving internally, so rejects `steps`/`dense`/`subs` in `saveat` |
| `diffeqsolveDirectAdjoint`, `diffeqsolveDirectAdjointIMEX` | discrete adjoint replaying the accepted-step trace using the solver's `odeStepAdjoint?` tableau (explicit RK, DIRK, KenCarp IMEX) |
| `diffeqsolveForwardMode` | forward-mode tangents |
| `diffeqsolveRecursiveCheckpointAdjoint` | checkpointed backsolve for long horizons |
| `diffeqsolveImplicitAdjoint` | implicit-function-theorem style via a whole-solve VJP |
| `sdeBacksolveAdjointStratonovich` | fixed-step Stratonovich SDE backsolve |

Each mode has an `...UnsupportedReason` validator that reports an explicit
error string instead of failing opaquely.

### Root finders

Implicit solvers solve their stage equations with `Tyr/DiffEq/RootFinder.lean`:
the `RootFinder` class plus `FixedPoint` (default for `ImplicitEuler`),
`AdaptiveFixedPoint`, and `NormRatioFixedPoint`, all with scaled-RMS convergence
tests and `rtol`/`atol`/`maxIters` configs. The latter two apply scalar relaxation
to `step(y) - y`: `AdaptiveFixedPoint` adapts the step using successive residual
norms, while `NormRatioFixedPoint` uses scaled iterate/residual norm ratios.
They do not solve or reuse a Jacobian system, and noncontractive stage equations
can fail to converge. Select them with `RootFindMethod.adaptiveFixedPoint` or
`RootFindMethod.normRatioFixedPoint`.

Migration: the former names `VeryChord` / `.veryChord` and `Newton` / `.newton`
were removed because the implementations are fixed-point heuristics, not chord
or Newton methods. Use the corresponding names above; the iteration algorithms
and configuration fields are unchanged.

## Key APIs

The entry point (`Tyr/DiffEq/Integrate.lean:834`):

```lean
def diffeqsolve {Term Y VF Control Args Controller : Type}
    [DiffEqSpace Y] [DiffEqSeminorm Y] [DiffEqElem Y] [Inhabited Y]
    [StepSizeController Controller] [StepSizeControllerValidation Controller]
    [Inhabited Controller]
    (terms : Term) (solver : AbstractSolver Term Y VF Control Args)
    (t0 t1 : Time) (dt0 : Option Time) (y0 : Y) (args : Args)
    (saveat : SaveAt := {}) (maxSteps : Nat := 4096)
    (controller : Controller := default)
    (event : Option (EventSpec Y Args) := none)
    (events : Array (EventSpec Y Args) := #[])
    (eventTree : Option (EventTree Y Args) := none)
    (saveFn : Option (Time → Y → Args → Y) := none)
    (throwOnFailure : Bool := false)
    (progress_meter : ProgressMeter := .none)
    -- plus initialSolverState / initialControllerState / initialMadeJump /
    -- maxStepsOpt for resuming and unbounded solves
    : Solution Y solver.SolverState (StepSizeController.State (C := Controller))
```

Variants: `diffeqsolveOrError` (returns `Except SolveError`),
`diffeqsolveEventTree` (tree-shaped event masks). `saveFn` transforms saved
values (e.g. an observation instead of the full state); `maxSteps` bounds both
attempted and rejected steps; `t1 < t0` integrates backward in time.

## Usage example

Reconstructed example (from `Tests/TestDiffEq.lean:40-62`, `:203-241`,
`:1093-1117`, `:1652-1682`, and `Tests/TestDiffEqAdjoint.lean:27-58`):

```lean
import Tyr.DiffEq
import Tyr.DiffEq.Adjoint.Torch   -- AdjointBackend instances for tensor states

open torch torch.DiffEq

-- 1. Scalar ODE, adaptive: dy/dt = -y with Dopri5 + PID controller.
-- (The tests spell out (Term := ...) (Y := ...) ...; concrete args infer them.)
def adaptiveODE : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Dopri5.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let controller : PIDController := { rtol := 1.0e-4, atol := 1.0e-6 }
  let sol := diffeqsolve term solver 0.0 1.0 none (1.0 : Float) ()
    (saveat := { t1 := true }) (controller := controller)
  -- dt0 = none triggers the initial-step heuristic.
  match sol.ys with
  | some ys => IO.println s!"y(1) = {ys[ys.size - 1]!}"   -- ≈ exp (-1)
  | none => pure ()

-- 2. Itô SDE: dy = -y dt + y dW, Euler–Maruyama over a scalar Brownian path.
def sdeExample : IO Unit := do
  let drift : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let bm : ScalarBrownianPath := { t0 := 0.0, t1 := 1.0, seed := 54321 }
  let diffusion : ControlTerm Float Float Float Unit :=
    ControlTerm.ofPath (fun _t y _ => y) bm.toAbstract.toPath (fun vf c => vf * c)
  let terms := SDETerm.mk drift diffusion
  let solver :=
    EulerMaruyama.solver
      (Drift := ODETerm Float Unit) (Diffusion := ControlTerm Float Float Float Unit)
      (Y := Float) (VFd := Float) (VFg := Float) (Control := Float) (Args := Unit)
  let _sol := diffeqsolve terms solver 0.0 1.0 (some 1.0) (2.0 : Float) ()
    (saveat := { t1 := true }) (controller := ({} : ConstantStepSize))
  pure ()

-- 3. Terminating event: stop when y crosses 0.5 upward, root localized.
def eventExample : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let ev : EventSpec Float Unit := {
    condition := .real (fun _t y _ => y - 0.5)
    direction := some true
    terminate := true
  }
  let sol := diffeqsolve term solver 0.0 1.0 (some 0.2) (0.0 : Float) ()
    (saveat := { t1 := true }) (controller := ({} : ConstantStepSize)) (event := some ev)
  -- sol.result == Result.eventOccurred, root saved near t = 0.5.
  IO.println s!"result: {reprStr sol.result}"

-- 4. Neural-ODE-style gradient: backsolve adjoint through an RK4 solve
--    of dy/dt = a * y on scalar tensors.
def adjointExample : IO Unit := do
  let term : ODETerm (Tensor { shape := #[], dtype := .Float32, device := .CPU }) (Tensor { shape := #[], dtype := .Float32, device := .CPU }) := { vectorField := fun _t y a => Tensor.mul a y }
  let solver :=
    RK4.solver (Term := ODETerm (Tensor { shape := #[], dtype := .Float32, device := .CPU }) (Tensor { shape := #[], dtype := .Float32, device := .CPU })) (Y := Tensor { shape := #[], dtype := .Float32, device := .CPU }) (VF := Tensor { shape := #[], dtype := .Float32, device := .CPU }) (Args := Tensor { shape := #[], dtype := .Float32, device := .CPU })
  let adjSolver :=
    RK4.solver
      (Term := ODETerm (AdjointState (Tensor { shape := #[], dtype := .Float32, device := .CPU }) (Tensor { shape := #[], dtype := .Float32, device := .CPU })) (Tensor { shape := #[], dtype := .Float32, device := .CPU }))
      (Y := AdjointState (Tensor { shape := #[], dtype := .Float32, device := .CPU }) (Tensor { shape := #[], dtype := .Float32, device := .CPU })) (VF := AdjointState (Tensor { shape := #[], dtype := .Float32, device := .CPU }) (Tensor { shape := #[], dtype := .Float32, device := .CPU }))
      (Args := Tensor { shape := #[], dtype := .Float32, device := .CPU })
  let a := Tensor.full #[] 0.3
  let sol := diffeqsolve term solver 0.0 1.0 (some 0.01) (Tensor.full #[] 2.0) a
    (saveat := { steps := true }) (controller := ({} : ConstantStepSize))
  match backsolveAdjoint term adjSolver sol a (Tensor.ones #[]) with
  | some _adj => IO.println s!"∂L/∂y₀ and ∂L/∂a computed"  -- adj.adjY0, adj.adjArgs
  | none => IO.println "adjoint solve failed"
```

For a larger system, `Tyr/EventSkeleton/Examples/Pendulum.lean:1306-1350`
integrates a custom record state (hand-written `DiffEqSpace` instances) with
`RK4`; `Tyr/Model/BranchingFlows/DiffEq.lean` wraps `diffeqsolve` for
generative modeling; `Examples/GPU/RunRKFusedSolve.lean` is a fused CUDA bench.

## Related guides

- [Automatic differentiation](autodiff.md) — the `autograd.grad` engine behind `Adjoint/Torch.lean`
- [Tensors](core/tensors.md) — `Tensor σ`, the indexed tensor state type
- [TensorStruct](core/tensorstruct.md) — model-tree states for neural ODEs
- [Typed tensors](core/typed.md) — `Tensor σ` states with static shape/dtype specs
- [Event skeleton](event-skeleton.md) — the hybrid-systems representation the DiffEq bridge projects into
- [Generative models](models/generative.md) — BranchingFlows, a downstream `diffeqsolve` consumer
- [GPU kernels](gpu/kernels.md) — fused RK solve kernels
- [Examples and testing](examples-and-testing.md) — the DiffEq parity test suites (`Tests/TestDiffEq*.lean`)

Exhaustive, per-symbol API documentation is generated separately by doc-gen4
(see `docbuild/`); this chapter is a guide, not a symbol dump.
