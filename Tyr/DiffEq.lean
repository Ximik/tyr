import Tyr.DiffEq.Types
import Tyr.DiffEq.Typed
import Tyr.DiffEq.Term
import Tyr.DiffEq.Path
import Tyr.DiffEq.Brownian
import Tyr.DiffEq.Interpolation
import Tyr.DiffEq.Solution
import Tyr.DiffEq.SaveAt
import Tyr.DiffEq.RootFinder
import Tyr.DiffEq.Solver.Base
import Tyr.DiffEq.Solver.ImplicitRungeKutta
import Tyr.DiffEq.Solver.SDE
import Tyr.DiffEq.Solver.Euler
import Tyr.DiffEq.Solver.Heun
import Tyr.DiffEq.Solver.Midpoint
import Tyr.DiffEq.Solver.ReversibleHeun
import Tyr.DiffEq.Solver.EulerMaruyama
import Tyr.DiffEq.Solver.EulerHeun
import Tyr.DiffEq.Solver.Milstein
import Tyr.DiffEq.Solver.StratonovichMilstein
import Tyr.DiffEq.Solver.SRK
import Tyr.DiffEq.Solver.SEA
import Tyr.DiffEq.Solver.SRA1
import Tyr.DiffEq.Solver.ShARK
import Tyr.DiffEq.Solver.GeneralShARK
import Tyr.DiffEq.Solver.SemiImplicitEuler
import Tyr.DiffEq.Solver.SlowRK
import Tyr.DiffEq.Solver.SPaRK
import Tyr.DiffEq.Solver.HalfSolver
import Tyr.DiffEq.Solver.ALIGN
import Tyr.DiffEq.Solver.ShOULD
import Tyr.DiffEq.Solver.QUICSORT
import Tyr.DiffEq.Solver.RungeKutta
import Tyr.DiffEq.Solver.Ralston
import Tyr.DiffEq.Solver.Bosh3
import Tyr.DiffEq.Solver.RK4
import Tyr.DiffEq.Solver.Dopri5
import Tyr.DiffEq.Solver.Dopri8
import Tyr.DiffEq.Solver.Tsit5
import Tyr.DiffEq.Solver.ImplicitEuler
import Tyr.DiffEq.Solver.Kvaerno3
import Tyr.DiffEq.Solver.Kvaerno4
import Tyr.DiffEq.Solver.Kvaerno5
import Tyr.DiffEq.Solver.Kencarp3
import Tyr.DiffEq.Solver.Kencarp4
import Tyr.DiffEq.Solver.Kencarp5
import Tyr.DiffEq.Solver.SIL3
import Tyr.DiffEq.Solver.LeapfrogMidpoint
import Tyr.DiffEq.StepSizeController
import Tyr.DiffEq.Integrate
import Tyr.DiffEq.Adjoint

/-!
# Tyr.DiffEq

`Tyr.DiffEq` is the umbrella import for Tyr's differential-equation stack
(ODEs, SDEs, and adjoint-based differentiation workflows).

## Major Components

- Core problem representation: types, terms, paths, Brownian processes.
- Numerical infrastructure: interpolation, saving rules, root finding, step control.
- Solver families: explicit, implicit, adaptive, stochastic, and symplectic variants.
- Integration and adjoint backends for end-to-end differentiable simulation.

## Scope

Use this as the canonical entrypoint when you need the full DiffEq toolbox.
Specialized projects may import solver submodules directly for tighter dependencies.
-/
