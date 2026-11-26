import LeanTest
import Tyr.Lib.Optim.Modular.Budget

open LeanTest
open Tyr.Lib.Optim.Modular

/-! Tests for modular LR budget compilation. -/

structure MockLayer where
  nuVal : Float
  muVal : Float
  deriving Repr, BEq

instance : torch.TensorStruct MockLayer where
  map _ x := x
  mapM _ x := pure x
  zipWith _ x _ := x
  fold _ init _ := init

instance : NormedModule MockLayer where
  norm m := Float.abs m.muVal
  dualNorm m := Float.abs m.muVal
  nu m := m.nuVal
  mu m := m.muVal
  normalize m := m
  normalizeDual m := m

@[test]
def testSequentialDownstreamScalesNonEmpty : IO Unit := do
  let cfg : BudgetConfig := {}
  let layers : Array MockLayer := #[
    { nuVal := 1.2, muVal := 0.8 },
    { nuVal := 1.1, muVal := 0.6 },
    { nuVal := 0.9, muVal := 0.7 }
  ]
  let scales := sequentialDownstreamScales cfg layers
  LeanTest.assertTrue (scales.size == layers.size && decide (scales.size > 0))
    s!"Expected non-empty scales with matching size, got {scales.size}"

@[test]
def testSequentialDownstreamScalesPositiveFinite : IO Unit := do
  let cfg : BudgetConfig := {
    minMultiplier := 1e-4
    maxMultiplier := 1e2
    globalScale := 0.5
  }
  let layers : Array MockLayer := #[
    { nuVal := 10.0, muVal := 0.2 },
    { nuVal := 0.3, muVal := 5.0 },
    { nuVal := 2.5, muVal := 1.7 }
  ]
  let scales := sequentialDownstreamScales cfg layers
  LeanTest.assertTrue (decide (scales.size > 0)) "Expected at least one compiled scale"
  for s in scales do
    LeanTest.assertTrue (decide (s > 0.0)) s!"Scale should be positive, got {s}"
    LeanTest.assertTrue (Float.isFinite s) s!"Scale should be finite, got {s}"

@[test]
def testBudgetMultiplierMonotonicWithSensitivity : IO Unit := do
  let cfg : BudgetConfig := {}
  let lowSens : Array MockLayer := #[{ nuVal := 1.0, muVal := 0.5 }]
  let highSens : Array MockLayer := #[{ nuVal := 1.0, muVal := 2.0 }]
  let lowMul := (sequentialDownstreamScales cfg lowSens).foldl (fun _ s => s) 0.0
  let highMul := (sequentialDownstreamScales cfg highSens).foldl (fun _ s => s) 0.0
  LeanTest.assertTrue (decide (lowMul > highMul))
    s!"Expected lower sensitivity to get higher multiplier, got {lowMul} vs {highMul}"
