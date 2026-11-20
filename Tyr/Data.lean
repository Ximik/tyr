import Tyr.Data.Task
import Tyr.Data.TaskClass
import Tyr.Data.Tasks
import Tyr.Data.Download
import Tyr.Data.HuggingFace

/-!
# Tyr.Data

`Tyr.Data` is the umbrella import for Tyr's task/data pipeline surface.
It re-exports task abstractions, task registries, and pretraining pipeline utilities.

## Major Components

- `Tyr.Data.Task`: core task abstractions.
- `Tyr.Data.TaskClass`: shared task typing/evaluation contracts.
- `Tyr.Data.Tasks`: concrete built-in tasks.
- `Tyr.Data.Download`: download utilities with retry and caching.
- `Tyr.Data.HuggingFace`: HuggingFace dataset loading via parquet FFI.

## Scope

Use this module when you want the standard data/task stack through one import.
Domain-specific suites can still live in separate modules and be composed on top.
-/
