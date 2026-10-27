/-
  Tyr/SafeTensors/Load.lean

  Shared checkpoint-loading helpers used by the per-model `Weights`
  modules. Centralizes the try-load and candidate-name fallback patterns
  needed when reading Hugging Face SafeTensors checkpoints (single-file
  and sharded), so model families don't re-implement them.

  Shapes and dtypes come from file metadata at runtime, so the FFI edge
  keeps raw handles; the public helpers validate each loaded handle against
  the requested shape and hand out exact dtype packages.
  The dtype follows the file and remains available as a dependent witness.
-/
import Tyr.SafeTensors.Exact

namespace torch.safetensors

/-- Try to load `name` from a single SafeTensors file.
    Returns `none` on any failure (missing tensor, shape mismatch,
    unreadable file). Use `tryLoadOptionalTensor` to swallow only
    missing-tensor errors. -/
def tryLoadTensor (path : String) (name : String) (s : Shape)
    (device : Device := Device.CPU)
    : IO (Option (SomeDTypeTensor s device)) := do
  try
    pure (some (← loadTensorExact path name s device))
  catch _ =>
    pure none

/-- Try to load `name` from a sharded SafeTensors checkpoint directory.
    Returns `none` on any failure (missing tensor, shape mismatch,
    unreadable file). Use `tryLoadOptionalTensorSharded` to swallow only
    missing-tensor errors. -/
def tryLoadTensorSharded (dir : String) (name : String) (s : Shape)
    (device : Device := Device.CPU)
    : IO (Option (SomeDTypeTensor s device)) := do
  try
    pure (some (← loadTensorShardedExact dir name s device))
  catch _ =>
    pure none

/-- Heuristic check whether an error message reports `name` as absent from
    the checkpoint (as opposed to e.g. a shape mismatch). -/
def isMissingTensorError (msg : String) (name : String) : Bool :=
  (msg.contains name) &&
  ((msg.contains "not found") || (msg.contains "missing") || (msg.contains "No tensor"))

/-- Like `tryLoadTensor`, but only a missing tensor yields `none`;
    other errors (shape mismatch, I/O) are rethrown. -/
def tryLoadOptionalTensor (path : String) (name : String) (s : Shape)
    (device : Device := Device.CPU)
    : IO (Option (SomeDTypeTensor s device)) :=
  loadOptionalTensorExact path name s device

/-- Like `tryLoadTensorSharded`, but only a missing tensor yields `none`;
    other errors (shape mismatch, I/O) are rethrown. -/
def tryLoadOptionalTensorSharded (dir : String) (name : String) (s : Shape)
    (device : Device := Device.CPU)
    : IO (Option (SomeDTypeTensor s device)) :=
  loadOptionalTensorShardedExact dir name s device

/-- Try candidate names, ignoring only absent entries. A present tensor with
invalid metadata, an unreadable checkpoint, or a device error is not absent. -/
def tryLoadOptionalTensorCandidates (path : String) (names : Array String) (s : Shape)
    (device : Device := Device.CPU)
    : IO (Option (SomeDTypeTensor s device)) := do
  for name in names do
    if let some tensor ← tryLoadOptionalTensor path name s device then
      return some tensor
  pure none

/-- Sharded counterpart of `tryLoadOptionalTensorCandidates`. -/
def tryLoadOptionalTensorShardedCandidates (dir : String) (names : Array String) (s : Shape)
    (device : Device := Device.CPU)
    : IO (Option (SomeDTypeTensor s device)) := do
  for name in names do
    if let some tensor ← tryLoadOptionalTensorSharded dir name s device then
      return some tensor
  pure none

/-- Load the first of `names` that resolves in a single SafeTensors file;
    `none` if no candidate loads. -/
def tryLoadTensorCandidates (path : String) (names : Array String) (s : Shape)
    (device : Device := Device.CPU)
    : IO (Option (SomeDTypeTensor s device)) := do
  for n in names do
    if let some t ← tryLoadTensor path n s device then
      return some t
  pure none

/-- Load the first of `names` that resolves in a sharded checkpoint directory;
    `none` if no candidate loads. -/
def tryLoadTensorShardedCandidates (dir : String) (names : Array String) (s : Shape)
    (device : Device := Device.CPU)
    : IO (Option (SomeDTypeTensor s device)) := do
  for n in names do
    if let some t ← tryLoadTensorSharded dir n s device then
      return some t
  pure none

/-- Load the first present candidate in a single SafeTensors file.
Malformed present entries remain errors; no candidates means a missing error. -/
def loadTensorCandidates (path : String) (names : Array String) (s : Shape)
    (device : Device := Device.CPU)
    : IO (SomeDTypeTensor s device) := do
  match (← tryLoadOptionalTensorCandidates path names s device) with
  | some t => pure t
  | none => throw <| IO.userError s!"Failed to load tensor: {names}"

/-- Sharded counterpart of `loadTensorCandidates`, with the same missing-only
candidate fallback. -/
def loadTensorShardedCandidates (dir : String) (names : Array String) (s : Shape)
    (device : Device := Device.CPU)
    : IO (SomeDTypeTensor s device) := do
  match (← tryLoadOptionalTensorShardedCandidates dir names s device) with
  | some t => pure t
  | none => throw <| IO.userError s!"Failed to load tensor (sharded): {names}"

/-- Append `x` to `xs` unless already present. Used by checkpoint
    tensor-name candidate generators. -/
def pushUnique (xs : Array String) (x : String) : Array String :=
  if xs.contains x then xs else xs.push x

end torch.safetensors
