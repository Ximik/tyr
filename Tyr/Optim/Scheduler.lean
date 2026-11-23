/-
  Tyr/Optim/Scheduler.lean

  Composable learning rate and hyperparameter schedules.

  All schedules are pure functions `Nat → Float`, matching the `Schedule` type
  from `Tyr.Optim`. They can be:
  - Used standalone to query a value at any step
  - Composed into optimizer chains via `scale_by_schedule`
  - Combined with schedule combinators (`join`, `scale_schedule`, etc.)
-/
import Tyr.Optim

namespace torch.Optim.Scheduler

open torch.Optim (Schedule)

private def pi : Float := 3.14159265358979323846

private def maxFloat (a b : Float) : Float := if a > b then a else b

/-! ## Primitive Schedule Constructors

Each constructor returns a `Schedule` (i.e., `Nat → Float`) directly.
-/

/-- Constant schedule: returns the same value at every step. -/
def constant (value : Float) : Schedule := fun _ => value

/-- Cosine annealing with optional linear warmup.
    - Warmup: linear increase from 0 to `peak`
    - Annealing: cosine decay from `peak` to `end_value` -/
def cosine_decay (peak : Float) (totalSteps : Nat) (end_value : Float := 0.0)
    (warmupSteps : Nat := 0) : Schedule := fun step =>
  if step < warmupSteps then
    peak * (step.toFloat / maxFloat warmupSteps.toFloat 1.0)
  else if step >= totalSteps then
    end_value
  else
    let progress := (step - warmupSteps).toFloat / maxFloat (totalSteps - warmupSteps).toFloat 1.0
    let coeff := 0.5 * (1.0 + Float.cos (pi * progress))
    end_value + coeff * (peak - end_value)

/-- Linear warmup then linear decay. -/
def linear_decay (peak : Float) (totalSteps : Nat) (end_value : Float := 0.0)
    (warmupSteps : Nat := 0) : Schedule := fun step =>
  if step < warmupSteps then
    peak * (step.toFloat / maxFloat warmupSteps.toFloat 1.0)
  else if step >= totalSteps then
    end_value
  else
    let progress := (step - warmupSteps).toFloat / maxFloat (totalSteps - warmupSteps).toFloat 1.0
    peak - progress * (peak - end_value)

/-- Step decay: value = initial * gamma^(step / stepSize) -/
def step_decay (initial : Float) (gamma : Float := 0.1) (stepSize : Nat := 30) : Schedule :=
  fun step =>
    let numDecays := step / stepSize
    initial * Float.pow gamma numDecays.toFloat

/-- One-cycle policy (Smith & Topin, 2017).
    - Phase 1 (pctStart): Linear increase from `min_value` to `peak`
    - Phase 2 (remaining): Cosine decrease from `peak` to `min_value` -/
def one_cycle (peak : Float) (totalSteps : Nat) (min_value : Float := 0.0)
    (pctStart : Float := 0.3) : Schedule := fun step =>
  if step >= totalSteps then
    min_value
  else
    let warmupSteps := (pctStart * totalSteps.toFloat).toUInt64.toNat
    if step < warmupSteps then
      let progress := step.toFloat / maxFloat warmupSteps.toFloat 1.0
      min_value + progress * (peak - min_value)
    else
      let progress := (step - warmupSteps).toFloat / maxFloat (totalSteps - warmupSteps).toFloat 1.0
      let coeff := 0.5 * (1.0 + Float.cos (pi * progress))
      min_value + coeff * (peak - min_value)

/-- Warmup → Plateau → Cosine decay. -/
def warmup_plateau_cosine (peak : Float) (totalSteps : Nat)
    (warmupSteps : Nat) (plateauSteps : Nat) (end_value : Float := 0.0) : Schedule := fun step =>
  let decayStart := warmupSteps + plateauSteps
  if step < warmupSteps then
    peak * (step.toFloat / maxFloat warmupSteps.toFloat 1.0)
  else if step < decayStart then
    peak
  else if step >= totalSteps then
    end_value
  else
    let progress := (step - decayStart).toFloat / maxFloat (totalSteps - decayStart).toFloat 1.0
    let coeff := 0.5 * (1.0 + Float.cos (pi * progress))
    end_value + coeff * (peak - end_value)

/-- Exponential decay: value = initial * gamma^step -/
def exponential_decay (initial : Float) (gamma : Float := 0.99) : Schedule :=
  fun step => initial * Float.pow gamma step.toFloat

/-- Polynomial decay: value = (initial - end_value) * (1 - step/total)^power + end_value -/
def polynomial_decay (initial : Float) (totalSteps : Nat) (end_value : Float := 0.0)
    (power : Float := 1.0) : Schedule := fun step =>
  if step >= totalSteps then
    end_value
  else
    let progress := step.toFloat / totalSteps.toFloat
    let decay := Float.pow (1.0 - progress) power
    end_value + decay * (initial - end_value)

/-- Linear warmup only: ramps from 0 to `peak` over `warmupSteps`, then stays at `peak`. -/
def warmup (peak : Float) (warmupSteps : Nat) : Schedule := fun step =>
  if warmupSteps == 0 then peak
  else if step < warmupSteps then
    peak * (step.toFloat / warmupSteps.toFloat)
  else peak

/-! ## Schedule Combinators

Combine schedules to build complex training policies.
-/

/-- Join two schedules at a boundary step.
    Uses `s1` for steps `< boundary`, `s2` for steps `>= boundary`. -/
def join (s1 s2 : Schedule) (boundary : Nat) : Schedule := fun step =>
  if step < boundary then s1 step else s2 step

/-- Sequentially compose schedules: run `s1` for `duration` steps, then `s2`
    (with its step count starting from 0). -/
def sequence (s1 : Schedule) (duration : Nat) (s2 : Schedule) : Schedule := fun step =>
  if step < duration then s1 step else s2 (step - duration)

/-- Multiply a schedule's output by a constant factor. -/
def scale_schedule (s : Schedule) (factor : Float) : Schedule := fun step =>
  s step * factor

/-- Multiply two schedules pointwise. Useful for combining a base LR schedule
    with a warmup multiplier. -/
def multiply (s1 s2 : Schedule) : Schedule := fun step =>
  s1 step * s2 step

/-- Add two schedules pointwise. -/
def add (s1 s2 : Schedule) : Schedule := fun step =>
  s1 step + s2 step

/-- Clamp a schedule's output between `lo` and `hi`. -/
def clamp (s : Schedule) (lo hi : Float) : Schedule := fun step =>
  let v := s step
  if v < lo then lo else if v > hi then hi else v

/-! ## Weight Decay Schedules -/

/-- Linear weight decay: decays from baseWd to 0 over training.
    Following nanochat's approach where weight decay goes to zero. -/
def linear_weight_decay (baseWd : Float) (totalSteps : Nat) : Schedule := fun step =>
  if step >= totalSteps then
    0.0
  else
    baseWd * (1.0 - step.toFloat / totalSteps.toFloat)

/-- Cosine weight decay: smooth decay from baseWd to 0. -/
def cosine_weight_decay (baseWd : Float) (totalSteps : Nat) : Schedule := fun step =>
  if step >= totalSteps then
    0.0
  else
    let progress := step.toFloat / totalSteps.toFloat
    baseWd * 0.5 * (1.0 + Float.cos (pi * progress))

end torch.Optim.Scheduler
