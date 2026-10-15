import Tyr.RawTensor.SafeTensors
import Tyr.Typed.Tensor

/-! Exact-dtype checkpoint admission. These loaders preserve the stored dtype
in a dependent package; they never reinterpret layout or erase dtype evidence.
Use the WithSpec loaders when the expected dtype is already known. -/

namespace torch.safetensors

private def admitExact (name : String) (shape : Shape) (device : Device)
    (raw : RawTensor) : IO (SomeDTypeTensor shape device) := do
  -- Validate the stored layout before any device conversion. An empty shape at
  -- the native loading boundary requests that layout rather than a reshape.
  let stored ← match SomeDTypeTensor.ofRaw (shape := shape) (device := raw.device) raw with
    | .ok tensor => pure tensor
    | .error message => throw <| IO.userError s!"SafeTensors tensor '{name}': {message}"
  let moved ← RawTensor.convertIO shape stored.2.raw device none
  -- Conversion retains dtype; keep that same witness and verify the result.
  match Tensor.ofRaw (σ := ⟨shape, stored.1.val, device⟩) moved with
  | .ok tensor => pure ⟨stored.1, tensor⟩
  | .error message => throw <| IO.userError s!"SafeTensors tensor '{name}': {message}"

/-- Load the original stored shape and dtype from one SafeTensors file.
The requested device is concrete; dtype conversion is the caller's decision. -/
def loadTensorExact (path name : String) (shape : Shape) (device : Device := .CPU) :
    IO (SomeDTypeTensor shape device) := do
  admitExact name shape device (← RawTensor.safeTensorsLoad path name #[])

/-- Sharded counterpart of `loadTensorExact`. -/
def loadTensorShardedExact (dir name : String) (shape : Shape) (device : Device := .CPU) :
    IO (SomeDTypeTensor shape device) := do
  admitExact name shape device (← RawTensor.safeTensorsLoadSharded dir name #[])

-- Match the native missing-entry diagnostics exactly. Errors opening a file,
-- parsing metadata, validating dimensions or transferring devices are failures.
private def missingEntry (error : IO.Error) (name : String) : Bool :=
  error.toString == s!"Tensor not found: {name}" ||
    error.toString == s!"Tensor not found in any shard: {name}"

private def optionalRaw (load : IO RawTensor) (name : String) : IO (Option RawTensor) := do
  try
    pure (some (← load))
  catch error =>
    if missingEntry error name then pure none else throw error

/-- Load an optional entry, returning `none` only when its name is absent.
Stored-layout, dtype, file and device errors remain recoverable IO failures. -/
def loadOptionalTensorExact (path name : String) (shape : Shape) (device : Device := .CPU) :
    IO (Option (SomeDTypeTensor shape device)) := do
  match ← optionalRaw (RawTensor.safeTensorsLoad path name #[]) name with
  | none => pure none
  | some raw => pure (some (← admitExact name shape device raw))

/-- Sharded counterpart of `loadOptionalTensorExact`. -/
def loadOptionalTensorShardedExact (dir name : String) (shape : Shape)
    (device : Device := .CPU) : IO (Option (SomeDTypeTensor shape device)) := do
  match ← optionalRaw (RawTensor.safeTensorsLoadSharded dir name #[]) name with
  | none => pure none
  | some raw => pure (some (← admitExact name shape device raw))

end torch.safetensors
