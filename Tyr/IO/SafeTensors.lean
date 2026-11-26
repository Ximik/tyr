import Tyr.IO.SafeTensors.File
import Tyr.IO.SafeTensors.Schema
import Tyr.IO.SafeTensors.Source

/-!
# Tyr.IO.SafeTensors

`Tyr.IO.SafeTensors` is the umbrella import for SafeTensors integration in Tyr.
It combines checkpoint loading with runtime schema and metadata support.

## Major Components

- `Tyr.IO.SafeTensors.File`: header parsing and writing, and per-file and
  per-directory header caches.
- `Tyr.IO.SafeTensors.Source`: a checkpoint (one file or a shard directory) and
  its loaders, checked against the header before any bytes are read.
- `Tyr.IO.SafeTensors.Schema`: runtime schema parsing and typed metadata surface.

## Scope

Use this module when working with checkpoint schema introspection and typed loading.
Low-level tensor runtime operations remain in `Tyr.Core.Torch`.
-/
