import Tyr.SafeTensors.File
import Tyr.SafeTensors.Load
import Tyr.SafeTensors.Exact
import Tyr.SafeTensors.Schema
import Tyr.SafeTensors.TypeProvider

/-!
# Tyr.SafeTensors

`Tyr.SafeTensors` is the umbrella import for SafeTensors integration in Tyr.
It combines runtime schema/metadata support with elaboration-time type-provider tooling.

## Major Components

- `Tyr.SafeTensors.File`: header parsing and writing, and the raw single-file,
  open-handle and sharded loaders.
- `Tyr.SafeTensors.Load`: shared try-load and candidate-name fallback helpers
  used by per-model weight loaders.
- `Tyr.SafeTensors.Exact`: loaders retaining the stored dtype as an exact
  existential index, without implicit dtype conversion.
- `Tyr.SafeTensors.Schema`: runtime schema parsing and typed metadata surface.
- `Tyr.SafeTensors.TypeProvider`: compile-time/type-provider integration for
  generated typed loaders/accessors.

## Scope

Use this module when working with checkpoint schema introspection and typed loading.
Low-level tensor runtime operations remain in `Tyr.Torch`.
-/
