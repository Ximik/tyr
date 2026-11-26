import Tyr.Lib.Module.Core
import Tyr.Core.TensorStruct.Derive
import Tyr.Lib.Module.Linear
import Tyr.Lib.Module.LayerNorm
import Tyr.Lib.Module.RMSNorm
import Tyr.Lib.Module.Init

/-!
# Tyr.Lib.Module

`Tyr.Lib.Module` is the umbrella import for Tyr's core neural-module layer.
It re-exports base module interfaces, deriving support, and common layer implementations.

## Major Components

- `Tyr.Lib.Module.Core`: the `Module` typeclass.
- `Tyr.Core.TensorStruct.Derive`: deriving support for module/tensor-structure ergonomics.
- Common layers: `Linear`, `LayerNorm` and `RMSNorm`.
- `Tyr.Lib.Module.Init`: parameter initialization.

## Scope

Use this module as the standard entrypoint for building model layers in Tyr.
Specialized layer families may live in additional submodules.
-/
