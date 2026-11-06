import Tyr.SafeTensors.File
import Tyr.SafeTensors.Schema
import Tyr.SafeTensors.Source
import Tyr.SafeTensors.TypeProvider

/-!
# Tyr.SafeTensors

`Tyr.SafeTensors` is the umbrella import for SafeTensors integration in Tyr.
It combines runtime schema/metadata support with elaboration-time type-provider tooling.

## Major Components

- `Tyr.SafeTensors.File`: header parsing and writing, and per-file and
  per-directory header caches.
- `Tyr.SafeTensors.Source`: a checkpoint (one file or a shard directory) and
  its loaders, checked against the header before any bytes are read.
- `Tyr.SafeTensors.Schema`: runtime schema parsing and typed metadata surface.
- `Tyr.SafeTensors.TypeProvider`: compile-time/type-provider integration for
  generated typed loaders/accessors.

## Scope

Use this module when working with checkpoint schema introspection and typed loading.
Low-level tensor runtime operations remain in `Tyr.Torch`.
-/
