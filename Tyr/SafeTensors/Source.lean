import Tyr.SafeTensors.File
import Tyr.Typed.Tensor

/-! A SafeTensors checkpoint opened once, single file or shard directory alike.
Shape and dtype are checked against the header before any bytes are read. -/

namespace torch.safetensors

structure Source where
  path : String
  tensors : Std.HashMap String Stored

namespace Source

/-- A single parsed `.safetensors` file. -/
def ofHandle (handle : SafeTensorsHandle) : Source :=
  { path := handle.path, tensors := storedIn handle }

/-- Open a `.safetensors` file or a Hugging Face model directory (see `openModelDir`). -/
def «open» (path : String) : IO Source := do
  if ← System.FilePath.isDir path then
    return { path, tensors := ← openModelDir path }
  return ofHandle (← openHandle path)

def contains (src : Source) (name : String) : Bool :=
  src.tensors.contains name

/-- The first of `names` present in the checkpoint. -/
def find? (src : Source) (names : Array String) : Option String :=
  names.find? src.contains

def resolve (src : Source) (names : Array String) : IO String :=
  match src.find? names with
  | some name => return name
  | none => throw <| IO.userError s!"SafeTensors: none of {names} in {src.path}"

private def entry? (src : Source) (name : String) : Option (SafeTensorsHandle × Entry) :=
  src.tensors[name]?.map fun stored => (stored.handle, stored.entry)

/-- Stored dtype and shape, without reading the tensor. -/
def spec? (src : Source) (name : String) : Option TensorSpec :=
  (src.entry? name).map fun (_, entry) => { shape := entry.shape, dtype := entry.dtype }

private def fail {α : Type} (src : Source) (name message : String) : IO α :=
  throw <| IO.userError s!"SafeTensors tensor '{name}' in {src.path}: {message}"

private def read (src : Source) (name : String) (handle : SafeTensorsHandle) (entry : Entry)
    (shape : Shape) (device : Device) : IO (Tensor ⟨shape, entry.dtype, device⟩) := do
  if entry.shape != shape then
    src.fail name s!"Expected shape {shape}, got {entry.shape}"
  let raw ← RawTensor.safeTensorsRead handle.path (handle.dataOffset + entry.dataStart)
    (entry.dataEnd - entry.dataStart) entry.dtype entry.shape
  return Tensor.ofRaw! (← RawTensor.convertIO shape raw device none)

/-- Load `name` with its stored dtype; `none` only when the checkpoint has no such tensor. -/
def load? (src : Source) (name : String) (shape : Shape) (device : Device := .CPU) :
    IO (Option (SomeDTypeTensor shape device)) := do
  let some (handle, entry) := src.entry? name | return none
  return some ⟨entry.dtype, ← src.read name handle entry shape device⟩

def load (src : Source) (name : String) (shape : Shape) (device : Device := .CPU) :
    IO (SomeDTypeTensor shape device) := do
  let some tensor ← src.load? name shape device | src.fail name "not found"
  return tensor

/-- Like `load?`, additionally requiring the stored dtype to be `spec.dtype`. -/
def loadAs? (src : Source) (name : String) (spec : TensorSpec) (device : Device := .CPU) :
    IO (Option (Tensor ⟨spec.shape, spec.dtype, device⟩)) := do
  let some (handle, entry) := src.entry? name | return none
  if h : entry.dtype = spec.dtype then
    return some (h ▸ (← src.read name handle entry spec.shape device))
  src.fail name s!"Expected dtype {spec.dtype}, got {entry.dtype}"

def loadAs (src : Source) (name : String) (spec : TensorSpec) (device : Device := .CPU) :
    IO (Tensor ⟨spec.shape, spec.dtype, device⟩) := do
  let some tensor ← src.loadAs? name spec device | src.fail name "not found"
  return tensor

end Source

/-- Append `x` to `xs` unless already present. Used by checkpoint
    tensor-name candidate generators. -/
def pushUnique (xs : Array String) (x : String) : Array String :=
  if xs.contains x then xs else xs.push x

end torch.safetensors
