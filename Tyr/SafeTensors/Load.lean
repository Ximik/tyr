/-
  Tyr/SafeTensors/Load.lean

  Shared checkpoint-loading helpers used by the per-model `Weights`
  modules. Centralizes the try-load and candidate-name fallback patterns
  needed when reading Hugging Face SafeTensors checkpoints (single-file
  and sharded), so model families don't re-implement them.

  Shapes and dtypes come from file metadata at runtime, so the FFI edge
  keeps raw handles; the public helpers validate each loaded handle against
  the requested shape and hand out shape-indexed tensors (`STensor`).
  The dtype follows the file and stays dynamic.
-/
import Tyr.Torch
import Tyr.Typed.Tensor

namespace torch.safetensors

/-- Validate a freshly loaded raw handle against the requested shape. Loaders below
request the stored layout: passing the expected shape to the legacy native loader
would reshape equal-numel tensors before this check, hiding layout mismatches. -/
private def checkLoadedShape (name : String) (s : Shape) (raw : RawTensor) : IO (STensor s) :=
  match Tensor.ofRaw (σ := { shape := s }) raw with
  | .ok tensor => pure tensor
  | .error message =>
      throw <| IO.userError s!"SafeTensors tensor '{name}': {message}"

/-- Try to load `name` from a single SafeTensors file.
    Returns `none` on any failure (missing tensor, shape mismatch,
    unreadable file). Use `tryLoadOptionalTensor` to swallow only
    missing-tensor errors. -/
def tryLoadTensor (path : String) (name : String) (s : Shape)
    (device : Device := Device.CPU)
    : IO (Option (STensor s)) := do
  try
    let raw ← loadTensorOnDevice path name #[] device
    pure (some (← checkLoadedShape name s raw))
  catch _ =>
    pure none

/-- Try to load `name` from a sharded SafeTensors checkpoint directory.
    Returns `none` on any failure (missing tensor, shape mismatch,
    unreadable file). Use `tryLoadOptionalTensorSharded` to swallow only
    missing-tensor errors. -/
def tryLoadTensorSharded (dir : String) (name : String) (s : Shape)
    (device : Device := Device.CPU)
    : IO (Option (STensor s)) := do
  try
    let raw ← loadTensorShardedOnDevice dir name #[] device
    pure (some (← checkLoadedShape name s raw))
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
    : IO (Option (STensor s)) := do
  try
    let raw ← loadTensorOnDevice path name #[] device
    pure (some (← checkLoadedShape name s raw))
  catch e =>
    if isMissingTensorError (toString e) name then
      pure none
    else
      throw e

/-- Like `tryLoadTensorSharded`, but only a missing tensor yields `none`;
    other errors (shape mismatch, I/O) are rethrown. -/
def tryLoadOptionalTensorSharded (dir : String) (name : String) (s : Shape)
    (device : Device := Device.CPU)
    : IO (Option (STensor s)) := do
  try
    let raw ← loadTensorShardedOnDevice dir name #[] device
    pure (some (← checkLoadedShape name s raw))
  catch e =>
    if isMissingTensorError (toString e) name then
      pure none
    else
      throw e

/-- Load the first of `names` that resolves in a single SafeTensors file;
    `none` if no candidate loads. -/
def tryLoadTensorCandidates (path : String) (names : Array String) (s : Shape)
    (device : Device := Device.CPU)
    : IO (Option (STensor s)) := do
  for n in names do
    if let some t ← tryLoadTensor path n s device then
      return some t
  pure none

/-- Load the first of `names` that resolves in a sharded checkpoint directory;
    `none` if no candidate loads. -/
def tryLoadTensorShardedCandidates (dir : String) (names : Array String) (s : Shape)
    (device : Device := Device.CPU)
    : IO (Option (STensor s)) := do
  for n in names do
    if let some t ← tryLoadTensorSharded dir n s device then
      return some t
  pure none

/-- Load the first of `names` that resolves in a single SafeTensors file;
    throws listing all candidates if none loads. -/
def loadTensorCandidates (path : String) (names : Array String) (s : Shape)
    (device : Device := Device.CPU)
    : IO (STensor s) := do
  match (← tryLoadTensorCandidates path names s device) with
  | some t => pure t
  | none => throw <| IO.userError s!"Failed to load tensor: {names}"

/-- Load the first of `names` that resolves in a sharded checkpoint directory;
    throws listing all candidates if none loads. -/
def loadTensorShardedCandidates (dir : String) (names : Array String) (s : Shape)
    (device : Device := Device.CPU)
    : IO (STensor s) := do
  match (← tryLoadTensorShardedCandidates dir names s device) with
  | some t => pure t
  | none => throw <| IO.userError s!"Failed to load tensor (sharded): {names}"

/-- Append `x` to `xs` unless already present. Used by checkpoint
    tensor-name candidate generators. -/
def pushUnique (xs : Array String) (x : String) : Array String :=
  if xs.contains x then xs else xs.push x

end torch.safetensors
