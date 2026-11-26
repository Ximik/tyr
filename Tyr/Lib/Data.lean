import Tyr.Lib.Data.Task
import Tyr.Lib.Data.TaskClass
import Tyr.IO.Download
import Tyr.IO.HuggingFace

/-!
# Tyr.Lib.Data

`Tyr.Lib.Data` is the umbrella import for Tyr's task/data pipeline surface.
It re-exports task abstractions, task registries, and pretraining pipeline utilities.

## Major Components

- `Tyr.Lib.Data.Task`: core task abstractions.
- `Tyr.Lib.Data.TaskClass`: shared task typing/evaluation contracts.
- `Tyr.Lib.Data.Tasks`: concrete built-in tasks.
- `Tyr.IO.Download`: download utilities with retry and caching.
- `Tyr.IO.HuggingFace`: HuggingFace dataset loading via parquet FFI.

## Scope

Use this module when you want the standard data/task stack through one import.
Domain-specific suites can still live in separate modules and be composed on top.
-/
