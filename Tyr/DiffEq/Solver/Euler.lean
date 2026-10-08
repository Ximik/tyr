import Tyr.DiffEq.Solver.Base

namespace torch
namespace DiffEq

/-! ## Euler Solver -/

attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hAddInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hSubInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hMulInst

@[reducible] private def hDivRightScalarInst [DiffEqSpace Y] : HDiv Y Scalar Y where
  hDiv y a := (1.0 / a) * y

attribute [local instance] hDivRightScalarInst

structure Euler where
  deriving Inhabited

def Euler.solver {Term Y VF Control Args : Type}
    [TermLike Term Y VF Control Args]
    [DiffEqSpace Y] : AbstractSolver Term Y VF Control Args := {
  SolverState := Unit
  DenseInfo := LocalLinearDenseInfo Y
  termStructure := TermStructure.single
  odeStepAdjoint? := some (.explicitRK {
    a := #[#[]]
    b := #[1.0]
    c := #[0.0]
  })
  order := fun _ => 1
  strongOrder := fun _ => 0.0
  init := fun _ _ _ _ _ => ()
  step := fun term t0 t1 y0 args state _madeJump =>
    let inst := (inferInstance : TermLike Term Y VF Control Args)
    let control := inst.contr term t0 t1
    let dy := inst.vf_prod term t0 y0 args control
    let y1 := y0 + dy
    let dense := { t0 := t0, t1 := t1, y0 := y0, y1 := y1 }
    {
      y1 := y1
      yError := none
      denseInfo := dense
      solverState := state
      result := Result.successful
    }
  func := fun term t y args =>
    let inst := (inferInstance : TermLike Term Y VF Control Args)
    inst.vf term t y args
  interpolation := fun info => info.toInterpolation
}

instance : ExplicitSolver Euler := ⟨True.intro⟩
instance : ItoSolver Euler := ⟨True.intro⟩

end DiffEq
end torch
