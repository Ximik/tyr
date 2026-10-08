import Tyr.DiffEq.Types

namespace torch
namespace DiffEq

/-! ## Root Finders -/

structure RootFindResult (Y : Type) where
  value : Y
  converged : Bool
  iterations : Nat

structure FixedPoint where
  rtol : Float := 1.0e-6
  atol : Float := 1.0e-6
  maxIters : Nat := 20
  deriving Inhabited

/-- Fixed-point iteration with a scalar relaxation factor chosen from ratios of
scaled iterate/residual norms. This does not compute or invert a Jacobian and
has no Newton convergence guarantee for noncontractive stage equations. -/
structure NormRatioFixedPoint where
  rtol : Float := 1.0e-6
  atol : Float := 1.0e-6
  maxIters : Nat := 20
  damping : Float := 1.0
  stepMin : Float := 1.0e-3
  stepMax : Float := 10.0
  deriving Inhabited

/-- Relaxed fixed-point iteration that reduces its scalar step according to
successive residual norms. This does not form or reuse a chord Jacobian. -/
structure AdaptiveFixedPoint where
  rtol : Float := 1.0e-6
  atol : Float := 1.0e-6
  maxIters : Nat := 20
  step : Float := 1.0
  stepMin : Float := 1.0e-3
  stepMax : Float := 10.0
  safety : Float := 0.9
  deriving Inhabited

inductive RootFindMethod where
  | fixedPoint (cfg : FixedPoint)
  | normRatioFixedPoint (cfg : NormRatioFixedPoint)
  | adaptiveFixedPoint (cfg : AdaptiveFixedPoint)
  deriving Inhabited

class RootFinder (R : Type) where
  solve : {Y : Type} → [DiffEqSpace Y] → [DiffEqSeminorm Y] → [DiffEqElem Y] →
    R → (Y → Y) → Y → RootFindResult Y

def FixedPoint.withTolerances (cfg : FixedPoint) (rtol atol : Option Float)
    (maxIters : Option Nat) : FixedPoint := {
  rtol := rtol.getD cfg.rtol
  atol := atol.getD cfg.atol
  maxIters := maxIters.getD cfg.maxIters
}

def NormRatioFixedPoint.withTolerances (cfg : NormRatioFixedPoint) (rtol atol : Option Float)
    (maxIters : Option Nat) : NormRatioFixedPoint := {
  rtol := rtol.getD cfg.rtol
  atol := atol.getD cfg.atol
  maxIters := maxIters.getD cfg.maxIters
  damping := cfg.damping
  stepMin := cfg.stepMin
  stepMax := cfg.stepMax
}

def AdaptiveFixedPoint.withTolerances (cfg : AdaptiveFixedPoint) (rtol atol : Option Float)
    (maxIters : Option Nat) : AdaptiveFixedPoint := {
  rtol := rtol.getD cfg.rtol
  atol := atol.getD cfg.atol
  maxIters := maxIters.getD cfg.maxIters
  step := cfg.step
  stepMin := cfg.stepMin
  stepMax := cfg.stepMax
  safety := cfg.safety
}

def RootFindMethod.withTolerances (cfg : RootFindMethod) (rtol atol : Option Float)
    (maxIters : Option Nat) : RootFindMethod :=
  match cfg with
  | .fixedPoint fp => .fixedPoint (fp.withTolerances rtol atol maxIters)
  | .normRatioFixedPoint n => .normRatioFixedPoint (n.withTolerances rtol atol maxIters)
  | .adaptiveFixedPoint cfg => .adaptiveFixedPoint (cfg.withTolerances rtol atol maxIters)

private def scaledError {Y : Type} [DiffEqSpace Y] [DiffEqSeminorm Y] [DiffEqElem Y]
    (rtol atol : Float) (yPrev yNext : Y) : Float :=
  let diff := DiffEqSpace.sub yNext yPrev
  let scale :=
    DiffEqElem.addScalar atol (DiffEqSpace.scale rtol (DiffEqElem.max (DiffEqElem.abs yPrev) (DiffEqElem.abs yNext)))
  DiffEqSeminorm.rms (DiffEqElem.div diff scale)

private def residual {Y : Type} [DiffEqSpace Y] (step : Y → Y) (y : Y) : Y :=
  DiffEqSpace.sub (step y) y

private def scaledResidual {Y : Type} [DiffEqSpace Y] [DiffEqSeminorm Y] [DiffEqElem Y]
    (rtol atol : Float) (step : Y → Y) (y : Y) : Float :=
  let yNext := step y
  let res := DiffEqSpace.sub yNext y
  let scale :=
    DiffEqElem.addScalar atol
      (DiffEqSpace.scale rtol (DiffEqElem.max (DiffEqElem.abs y) (DiffEqElem.abs yNext)))
  DiffEqSeminorm.rms (DiffEqElem.div res scale)

private def clampFloat (x lo hi : Float) : Float :=
  if x < lo then lo else if x > hi then hi else x

instance : RootFinder FixedPoint where
  solve {Y} _ _ _ cfg step y0 :=
    let rec loop (i : Nat) (yPrev : Y) : RootFindResult Y :=
      if i >= cfg.maxIters then
        { value := yPrev, converged := false, iterations := i }
      else
        let yNext := step yPrev
        let err := scaledError cfg.rtol cfg.atol yPrev yNext
        if err < 1.0 then
          { value := yNext, converged := true, iterations := i + 1 }
        else
          loop (i + 1) yNext
    loop 0 y0

instance : RootFinder AdaptiveFixedPoint where
  solve {Y} _ _ _ cfg step y0 :=
    let step0 := clampFloat cfg.step cfg.stepMin cfg.stepMax
    let rec loop (i : Nat) (yPrev : Y) (prevErr : Float) (stepScale : Float) :
        RootFindResult Y :=
      if i >= cfg.maxIters then
        { value := yPrev, converged := false, iterations := i }
      else
        let err := scaledResidual cfg.rtol cfg.atol step yPrev
        if err < 1.0 then
          { value := yPrev, converged := true, iterations := i }
        else
          let ratio :=
            if prevErr <= 0.0 then 1.0 else err / prevErr
          let rawScale :=
            if i == 0 then stepScale
            else stepScale * cfg.safety / (1.0 + ratio)
          let nextScale := clampFloat rawScale cfg.stepMin cfg.stepMax
          let res := residual step yPrev
          let yNext := DiffEqSpace.add yPrev (DiffEqSpace.scale nextScale res)
          loop (i + 1) yNext err nextScale
    loop 0 y0 1.0e30 step0

instance : RootFinder NormRatioFixedPoint where
  solve {Y} _ _ _ cfg step y0 :=
    let damping := clampFloat cfg.damping 0.0 cfg.stepMax
    let rec loop (i : Nat) (yPrev : Y) (prev : Option (Y × Y)) : RootFindResult Y :=
      if i >= cfg.maxIters then
        { value := yPrev, converged := false, iterations := i }
      else
        let res := residual step yPrev
        let err := scaledResidual cfg.rtol cfg.atol step yPrev
        if err < 1.0 then
          { value := yPrev, converged := true, iterations := i }
        else
          let rawScale :=
            match prev with
            | none => 1.0
            | some (yPrevPrev, resPrev) =>
                let dy := DiffEqSpace.sub yPrev yPrevPrev
                let dr := DiffEqSpace.sub res resPrev
                let scaleY :=
                  DiffEqElem.addScalar cfg.atol
                    (DiffEqSpace.scale cfg.rtol (DiffEqElem.max (DiffEqElem.abs yPrev) (DiffEqElem.abs yPrevPrev)))
                let scaleR :=
                  DiffEqElem.addScalar cfg.atol
                    (DiffEqSpace.scale cfg.rtol (DiffEqElem.max (DiffEqElem.abs res) (DiffEqElem.abs resPrev)))
                let numer := DiffEqSeminorm.rms (DiffEqElem.div dy scaleY)
                let denom := DiffEqSeminorm.rms (DiffEqElem.div dr scaleR)
                if denom <= 1.0e-12 then 1.0 else numer / denom
          let stepScale := clampFloat (damping * rawScale) cfg.stepMin cfg.stepMax
          let yNext := DiffEqSpace.add yPrev (DiffEqSpace.scale stepScale res)
          loop (i + 1) yNext (some (yPrev, res))
    loop 0 y0 none

instance : RootFinder RootFindMethod where
  solve {_Y} _ _ _ cfg step y0 :=
    match cfg with
    | .fixedPoint fp => RootFinder.solve fp step y0
    | .normRatioFixedPoint n => RootFinder.solve n step y0
    | .adaptiveFixedPoint cfg => RootFinder.solve cfg step y0

end DiffEq
end torch
