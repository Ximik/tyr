import Tyr.DiffEq.Solver.Base

namespace torch
namespace DiffEq

/-! ## Midpoint Solver (Explicit RK2) -/

attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hAddInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hSubInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hMulInst

@[reducible] private def hDivRightScalarInst [DiffEqSpace Y] : HDiv Y Scalar Y where
  hDiv y a := (1.0 / a) * y

attribute [local instance] hDivRightScalarInst

structure Midpoint where
  deriving Inhabited

def Midpoint.solver {Term Y VF Args : Type}
    [TermLike Term Y VF Time Args]
    [DiffEqSpace Y] : AbstractSolver Term Y VF Time Args := {
  SolverState := Unit
  DenseInfo := LocalLinearDenseInfo Y
  termStructure := TermStructure.single
  odeStepAdjoint? := some (.explicitRK {
    a := #[#[], #[0.5]]
    b := #[0.0, 1.0]
    c := #[0.0, 0.5]
  })
  order := fun _ => 2
  strongOrder := fun _ => 0.0
  init := fun _ _ _ _ _ => ()
  step := fun term t0 t1 y0 args state _madeJump =>
    let inst := (inferInstance : TermLike Term Y VF Time Args)
    let dt := inst.contr term t0 t1
    let k1 := inst.vf_prod term t0 y0 args dt
    let yMid := y0 + (k1 / 2.0)
    let tMid := t0 + dt / 2.0
    let k2 := inst.vf_prod term tMid yMid args dt
    let y1 := y0 + k2
    let dense := { t0 := t0, t1 := t1, y0 := y0, y1 := y1 }
    {
      y1 := y1
      yError := none
      denseInfo := dense
      solverState := state
      result := Result.successful
    }
  func := fun term t y args =>
    let inst := (inferInstance : TermLike Term Y VF Time Args)
    inst.vf term t y args
  interpolation := fun info => info.toInterpolation
}

instance : ExplicitSolver Midpoint := ⟨True.intro⟩
instance : StratonovichSolver Midpoint := ⟨True.intro⟩

end DiffEq
end torch
