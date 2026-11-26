import Tyr.Lib.Optim.Modular.Norm

namespace Tyr.Lib.Optim.Modular

/-- Configuration for compiling learning-rate multipliers from modular sensitivities. -/
structure BudgetConfig where
  /-- Minimum positive value used to avoid division by zero. -/
  epsilon : Float := 1e-6
  /-- Lower clamp for produced multipliers. -/
  minMultiplier : Float := 1e-3
  /-- Upper clamp for produced multipliers. -/
  maxMultiplier : Float := 1e3
  /-- Global factor applied to all multipliers. -/
  globalScale : Float := 1.0
  deriving Repr, BEq

private def clamp (lo hi x : Float) : Float :=
  if x < lo then lo else if x > hi then hi else x

/-- Convert a raw sensitivity into a finite positive value. -/
def sanitizeSensitivity (cfg : BudgetConfig) (x : Float) : Float :=
  let x := if Float.isFinite x then Float.abs x else cfg.epsilon
  if x < cfg.epsilon then cfg.epsilon else x

/-- Compile a sensitivity into an LR multiplier (inverse-sensitivity scaling). -/
def multiplierFromSensitivity (cfg : BudgetConfig) (sensitivity : Float) : Float :=
  let s := sanitizeSensitivity cfg sensitivity
  let raw := cfg.globalScale / s
  let raw := if Float.isFinite raw then raw else cfg.maxMultiplier
  clamp cfg.minMultiplier cfg.maxMultiplier raw

/-- Compile per-layer multipliers for a sequential stack.
    For layer `i`, sensitivity is `mu_i * (Π_{j>i} nu_j)`. -/
def sequentialDownstreamScales [NormedModule M]
    (cfg : BudgetConfig) (modules : Array M) : Array Float :=
  let (_, scales) :=
    modules.foldr
      (fun m (state : Float × List Float) =>
        let downstreamNu := state.1
        let μ := sanitizeSensitivity cfg (NormedModule.mu m)
        let sensitivity := μ * downstreamNu
        let scale := multiplierFromSensitivity cfg sensitivity
        let ν := sanitizeSensitivity cfg (NormedModule.nu m)
        let nextDownstreamNu := downstreamNu * ν
        (nextDownstreamNu, scale :: state.2))
      (1.0, [])
  scales.toArray

/-- Apply one multiplier to a base learning rate. -/
def applyBudget (baseLR multiplier : Float) : Float :=
  baseLR * multiplier

/-- Compile budgeted learning rates for a sequential stack. -/
def budgetedSequentialLRs [NormedModule M]
    (cfg : BudgetConfig) (baseLR : Float) (modules : Array M) : Array Float :=
  (sequentialDownstreamScales cfg modules).map (applyBudget baseLR)

/-- Grouped LR multipliers for common parameter families. -/
structure GroupBudget where
  matrix : Float := 1.0
  embedding : Float := 1.0
  lmHead : Float := 1.0
  scalar : Float := 1.0
  deriving Repr, BEq

namespace GroupBudget

/-- Apply grouped multipliers to a single base learning rate. -/
def applyToBaseLR (baseLR : Float) (g : GroupBudget) : GroupBudget :=
  { matrix := applyBudget baseLR g.matrix
    embedding := applyBudget baseLR g.embedding
    lmHead := applyBudget baseLR g.lmHead
    scalar := applyBudget baseLR g.scalar }

/-- Compile grouped multipliers directly from sensitivities. -/
def ofSensitivities (cfg : BudgetConfig)
    (matrix embedding lmHead scalar : Float) : GroupBudget :=
  { matrix := multiplierFromSensitivity cfg matrix
    embedding := multiplierFromSensitivity cfg embedding
    lmHead := multiplierFromSensitivity cfg lmHead
    scalar := multiplierFromSensitivity cfg scalar }

end GroupBudget

end Tyr.Lib.Optim.Modular
