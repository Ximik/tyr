/-
  Tyr/IO/SafeTensors/Schema.lean

  SafeTensors schema introspection for Lean:
  - parse tensor headers from `.safetensors` files
  - introspect single-file or sharded-directory layouts
  - support HuggingFace `model.safetensors.index.json` when present
-/
import Tyr.IO.SafeTensors.File
import Tyr.Core.Typed.Tensor

namespace torch.safetensors


/-- One tensor entry discovered in a SafeTensors source. -/
structure TensorSchema where
  name : String
  dtype : DType
  shape : Shape
  /-- For directory sources this is the shard filename; for single-file sources it is empty. -/
  sourceFile : String := ""
  deriving Repr, BEq

namespace TensorSchema

/-- Convert SafeTensors metadata into Tyr's shared tensor spec. -/
def toSpec (schema : TensorSchema) : TensorSpec :=
  { shape := schema.shape, dtype := schema.dtype }

end TensorSchema

/-- Full schema discovered from a SafeTensors source path. -/
structure Schema where
  source : String
  sourceIsDirectory : Bool
  tensors : Array TensorSchema
  deriving Repr

/-- Look up a tensor schema by exact tensor name. -/
def Schema.find? (schema : Schema) (tensorName : String) : Option TensorSchema :=
  schema.tensors.findSome? fun t => if t.name == tensorName then some t else none

private def sortedByName (tensors : Array TensorSchema) : Array TensorSchema :=
  tensors.qsort (fun a b => a.name < b.name)

private def schemasOf (tensors : Std.HashMap String Stored) : Array TensorSchema :=
  sortedByName <| tensors.toArray.map fun (name, stored) =>
    { name, dtype := stored.entry.dtype, shape := stored.entry.shape, sourceFile := stored.file }

/-- Introspect the tensor schema of a `.safetensors` file or a Hugging Face model
    directory (see `openModelDir`). -/
def introspect (source : String) : IO Schema := do
  let sourcePath : System.FilePath := ⟨source⟩
  if !(← sourcePath.pathExists) then
    throw <| IO.userError s!"SafeTensors source does not exist: {source}"
  if ← sourcePath.isDir then
    return { source, sourceIsDirectory := true, tensors := schemasOf (← openModelDir source) }
  if sourcePath.extension != some "safetensors" then
    throw <| IO.userError s!"SafeTensors file must use '.safetensors' extension: {source}"
  return { source, sourceIsDirectory := false, tensors := schemasOf (storedIn (← readHeader source)) }

end torch.safetensors
