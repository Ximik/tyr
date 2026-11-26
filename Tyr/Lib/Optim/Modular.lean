import Tyr.Lib.Optim.Modular.Norm
import Tyr.Lib.Optim.Modular.Atomic
import Tyr.Lib.Optim.Modular.Compose
import Tyr.Lib.Optim.Modular.Budget
import Tyr.Lib.Optim.Modular.MetricFactor
import Tyr.Lib.Optim.Modular.Manifold
import Tyr.Lib.Optim.Modular.RiemannianModule

/-!
# Tyr.Lib.Optim.Modular

`Tyr.Lib.Optim.Modular` is the umbrella import for modular-norm abstractions in Tyr.
It re-exports the components used to define normed modules and compose them in a way
that supports width/depth-robust optimization scaling.

## Major Components

- `Norm`: foundational modular-norm abstractions.
- `Atomic`: atomic module instances (for example linear/layer-style primitives).
- `Compose`: composition rules for sequential/product/container module structures.
- `Budget`: modular sensitivity -> LR budget compilation utilities.
- `Manifold`: manifold-native module wrappers and modular-to-optimizer bridges.

## Scope

This module collects the modular-norm stack behind one import.
It targets optimizer/module scaling workflows rather than general tensor math.
-/
