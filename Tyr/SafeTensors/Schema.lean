/-
  Tyr/SafeTensors/Schema.lean

  SafeTensors schema introspection for Lean:
  - parse tensor headers from `.safetensors` files
  - introspect single-file or sharded-directory layouts
  - support HuggingFace `model.safetensors.index.json` when present
-/
import Tyr.SafeTensors.File
import Tyr.Typed.Tensor
import Lean.Data.Json
import Lean.Data.Json.FromToJson

namespace torch.safetensors

open Lean

instance : ToJson DType where
  toJson dt := Json.str dt.canonicalName

instance : FromJson DType where
  fromJson?
    | .str s => match DType.ofString? s with
      | some dtype => pure dtype
      | none => .error s!"unknown dtype {s}"
    | _ => .error "expected dtype string"

/-- One tensor entry discovered in a SafeTensors source. -/
structure TensorSchema where
  name : String
  dtype : DType
  shape : Shape
  /-- For directory sources this is the shard filename; for single-file sources it is empty. -/
  sourceFile : String := ""
  deriving Inhabited, Repr, BEq

-- Dimensions stay string-encoded, the format schema files were written in when shapes were `UInt64`.
instance : ToJson TensorSchema where
  toJson t := Json.mkObj [("name", toJson t.name), ("dtype", toJson t.dtype),
    ("shape", Json.arr (t.shape.map bignumToJson)), ("sourceFile", toJson t.sourceFile)]

instance : FromJson TensorSchema where
  fromJson? j := do
    let shape ← (← j.getObjValAs? (Array Json) "shape").mapM bignumFromJson?
    pure { name := ← j.getObjValAs? String "name", dtype := ← j.getObjValAs? DType "dtype", shape,
           sourceFile := ← j.getObjValAs? String "sourceFile" }

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
  deriving Inhabited, Repr, ToJson, FromJson

/-- Look up a tensor schema by exact tensor name. -/
def Schema.find? (schema : Schema) (tensorName : String) : Option TensorSchema :=
  schema.tensors.findSome? fun t => if t.name == tensorName then some t else none

private def ensureUniqueNames (tensors : Array TensorSchema) : IO Unit := do
  let mut seen : Std.HashMap String String := {}
  for t in tensors do
    match seen.get? t.name with
    | some otherSource =>
      throw <| IO.userError
        s!"Duplicate tensor name '{t.name}' in sources '{otherSource}' and '{t.sourceFile}'"
    | none =>
      seen := seen.insert t.name t.sourceFile

private def sortedByName (tensors : Array TensorSchema) : Array TensorSchema :=
  tensors.qsort (fun a b => a.name < b.name)

private def isSchemaSnapshotPath (path : System.FilePath) : Bool :=
  let s := path.toString
  s.endsWith ".schema.json" || s.endsWith ".safetensors.schema.json"

/-- Load a previously saved SafeTensors schema snapshot. -/
def loadSnapshot (path : String) : IO Schema := do
  let text ← IO.FS.readFile ⟨path⟩
  let json ←
    match Json.parse text with
    | Except.ok j => pure j
    | Except.error err =>
        throw <| IO.userError s!"Invalid SafeTensors schema snapshot '{path}': {err}"
  let schema : Schema ←
    match (FromJson.fromJson? json : Except String Schema) with
    | .ok s => pure s
    | .error err =>
        throw <| IO.userError s!"Invalid SafeTensors schema snapshot '{path}': {err}"
  ensureUniqueNames schema.tensors
  pure { schema with tensors := sortedByName schema.tensors }

/-- Persist a SafeTensors schema snapshot as JSON for later type-provider use. -/
def saveSnapshot (path : String) (schema : Schema) : IO Unit := do
  let normalized := { schema with tensors := sortedByName schema.tensors }
  IO.FS.writeFile ⟨path⟩ (toJson normalized).compress

private def schemasOf (tensors : Std.HashMap String Stored) : Array TensorSchema :=
  sortedByName <| tensors.toArray.map fun (name, stored) =>
    { name, dtype := stored.entry.dtype, shape := stored.entry.shape, sourceFile := stored.file }

/-- Introspect the tensor schema of a `.safetensors` file, a Hugging Face model
    directory (see `openModelDir`), or a `.schema.json` snapshot. -/
def introspect (source : String) : IO Schema := do
  let sourcePath : System.FilePath := ⟨source⟩
  if !(← sourcePath.pathExists) then
    throw <| IO.userError s!"SafeTensors source does not exist: {source}"
  if isSchemaSnapshotPath sourcePath then
    return ← loadSnapshot source
  if ← sourcePath.isDir then
    return { source, sourceIsDirectory := true, tensors := schemasOf (← openModelDir source) }
  if sourcePath.extension != some "safetensors" then
    throw <| IO.userError s!"SafeTensors file must use '.safetensors' extension: {source}"
  return { source, sourceIsDirectory := false, tensors := schemasOf (storedIn (← readHeader source)) }

end torch.safetensors
