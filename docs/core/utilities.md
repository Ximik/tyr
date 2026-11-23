# Supporting utilities

Two small support modules: silent-by-default log handlers (`Tyr/Log.lean`) and VS Code infoview widgets for tensors and modules (`Tyr/Widget.lean`).

## Purpose & when to use

- Use `Tyr.Log` when writing library code that should report progress, warnings, or errors without doing console I/O itself. Handlers default to no-ops, so libraries stay silent unless the executable wires up sinks.
- Use `Tyr.Widget` during development to inspect tensor values and module parameter trees directly in the Lean infoview, instead of printing arrays.

## Architecture & main abstractions

### Log handlers (`Tyr/Log.lean`)

One record of three sinks in `namespace torch.Log`, no imports:

```lean
abbrev Sink := String → IO Unit

structure Handlers where
  onInfo : Sink := fun _ => pure ()
  onWarn : Sink := fun _ => pure ()
  onError : Sink := fun _ => pure ()
```

All sinks default to no-ops, so the library-side pattern is to accept `(log : Handlers := {})` and call `log.onInfo ...` etc. — checkpointing is a typical example (`Tyr/Checkpoint.lean`):

```lean
def saveParams [TensorStruct α]
    (params : α)
    (dir : String)
    (namePrefix : String := "param")
    (log : Handlers := {})
    : IO Unit
```

`Handlers.combine` runs two handler sets in sequence, e.g. to tee output to both console and a log file. `Tyr.Log` is re-exported by the root `Tyr.lean` (`Tyr.lean`). Consumers include `Tyr/Checkpoint.lean`, the model weight loaders under `Tyr/Model/*/Weights.lean`, and the data/download code under `Tyr/Data/`.

### Infoview widgets (`Tyr/Widget.lean`)

Three RPC-encodable data types, one JavaScript frontend, two commands. The data model (`Tyr/Widget.lean`):

```lean
structure TensorDisplayProps where
  shape : Array UInt64
  dtype : String
  device : String
  values : Array Float
  requiresGrad : Bool
  numel : UInt64
  stats : String
  gradValues : Option (Array Float) := none
  gradStats : Option String := none
  name : Option String := none
  axisNames : Option (Array String) := none

inductive ModuleNode where
  | tensor : TensorDisplayProps → ModuleNode
  | group : String → Array (String × ModuleNode) → ModuleNode
  | static : String → String → ModuleNode

structure ModuleDisplayProps where
  root : ModuleNode
```

`tensorToProps` snapshots a tensor for display: shape, dtype, device, summary stats, up to 10k values, and gradient values/stats when `requires_grad` is set (`Tyr/Widget.lean`). The single `@[widget_module] TensorWidget` JavaScript bundle (`Tyr/Widget.lean`) renders both views: a heatmap with diverging colormap, faceted layout for 3D+ tensors, hover tooltips, and theme awareness; and a collapsible module tree with a search box and per-group trainable-parameter counts.

For module trees there is a typeclass plus a deriving handler:

```lean
class ToModuleDisplay (α : Type) where
  toModuleNode : α → String → IO ModuleNode

def toModuleDisplayProps [ToModuleDisplay α] (value : α) (name : String := "root") : IO ModuleDisplayProps
```

Instances exist for `Tensor σ`, `Option α`, `Array α`, `List α`, `α × β`, and `Static α` (rendered as a plain `name: value` label). The deriving handler (`Tyr/Widget.lean`) generates an instance for any structure, producing one `group` node whose children are the fields. The built-in modules already use it — e.g. `Linear` derives `TensorStruct, ToModuleDisplay` (`Tyr/Module/Linear.lean`), as do `RMSNorm` and `LayerNorm`.

The two elaboration commands:

- `#tensor t` — accepts a pure `Tensor σ` term or an `IO (Tensor σ)` term (e.g. `Tensor.randn`, which is in `IO`).
- `#module t` — accepts `ModuleDisplayProps` or `IO ModuleDisplayProps`.

Both elaborate the term, evaluate it inside the Lean server via `Meta.evalExpr` (behind `unsafe`/`@[implemented_by]` wrappers, `Tyr/Widget.lean`), and pin the panel to the command line. Caveat: this runs the given `IO` in the compiler process at elaboration time, so treat `#tensor`/`#module` as development tools for trusted code, not as something to leave in library files.

`Tyr.Widget` is not re-exported by the root `Tyr.lean`; import it directly. The import also registers the `deriving ToModuleDisplay` handler.

## Key APIs

### `Tyr/Log.lean` (`namespace torch.Log`)

| Signature | Purpose |
| --- | --- |
| `Sink := String → IO Unit` | Log sink type. |
| `Handlers` | Record of `onInfo`/`onWarn`/`onError`, all defaulting to no-ops. |
| `Handlers.combine (lhs rhs : Handlers) : Handlers` | Invoke both handler sets, in order. |

### `Tyr/Widget.lean` (`namespace torch`)

| Signature | Purpose |
| --- | --- |
| `tensorToProps (t : Tensor σ) : IO TensorDisplayProps` | Snapshot a tensor and its current gradient for display. |
| `tensorToPropsNamed (t : Tensor σ) (name : String) (axisNames : Array String := #[]) : IO TensorDisplayProps` | Same, with name and axis labels. |
| `toModuleDisplayProps [ToModuleDisplay α] (value : α) (name : String := "root") : IO ModuleDisplayProps` | Build a display tree from any derivable structure. |
| `deriving ToModuleDisplay` | Generate a tree instance from a structure's fields. |
| `#tensor <term>` | Infoview heatmap for `Tensor σ` or `IO (Tensor σ)`. |
| `#module <term>` | Infoview tree for `ModuleDisplayProps` or `IO ModuleDisplayProps`. |

## Usage examples

### Wiring log handlers

Reconstructed example (from `Tyr/Checkpoint.lean` and `Examples/KittenTTSPretrained.lean`):

```lean
-- Library side: accept handlers with a silent default (Tyr/Checkpoint.lean).
def saveCheckpoint [TensorStruct α]
    (params : α)
    (iteration : Nat)
    (bestValLoss : Float)
    (trainLoss : Float)
    (dir : String)
    (namePrefix : String := "param")
    (log : Handlers := {})
    : IO Unit

-- Executable side: wire sinks to the console (Examples/KittenTTSPretrained.lean).
let log : torch.Log.Handlers := {
  onInfo := IO.println
  onWarn := fun msg => IO.eprintln s!"warning: {msg}"
  onError := fun msg => IO.eprintln s!"error: {msg}"
}
let bundle ← Model.loadFromPretrained source (log := log)
```

### Inspecting tensors and modules in the infoview

Reconstructed example (from `Examples/TestGuard.lean`):

```lean
import Tyr.Widget  -- also enables `deriving ToModuleDisplay`
import Tyr.Module.Linear

open torch

-- Heatmap of any tensor; IO tensor expressions work too.
#tensor Tensor.randn #[20, 100, 4]

-- Derive a module tree for your own structure.
structure MLP (in_dim hidden out_dim : UInt64) where
  fc1 : Linear in_dim hidden
  fc2 : Linear hidden out_dim
  deriving ToModuleDisplay

def exampleMLP : IO ModuleDisplayProps := do
  let fc1 ← Linear.init 64 256
  let fc2 ← Linear.init 256 10
  let mlp : MLP 64 256 10 := { fc1, fc2 }
  toModuleDisplayProps mlp "MLP(64 → 256 → 10)"

#module exampleMLP
```

Place the cursor on a `#tensor` or `#module` line in VS Code to open the panel. For full control over the hierarchy you can also build `ModuleNode.group`/`ModuleNode.tensor` trees manually, as `mlpModule` does in `Examples/TestGuard.lean`.

## Related guides

- [Tensors](tensors.md) — `Tensor σ`, creation ops like `Tensor.randn`/`Tensor.zeros`, and the FFI surface behind them.
- [TensorStruct](tensorstruct.md) — parameter-tree traversal; `Static` fields show up as labels in module widgets.
- [Modules](../modules.md) — `Linear` and friends, which already derive `ToModuleDisplay`.
- [Examples and testing](../examples-and-testing.md) — where `Examples/TestGuard.lean` and the parity harnesses live.

Exhaustive per-symbol documentation for these modules is generated by doc-gen4 (see `docbuild/`); this chapter is a guide, not an API dump.
