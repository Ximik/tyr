/-
  Tyr/SafeTensors/Schema.lean

  SafeTensors schema introspection for Lean:
  - parse tensor headers from `.safetensors` files
  - introspect single-file or sharded-directory layouts
  - support HuggingFace `model.safetensors.index.json` when present
-/
import Tyr.RawTensor.SafeTensors
import Tyr.Typed.Tensor
import Lean.Data.Json
import Lean.Data.Json.FromToJson

namespace torch.safetensors

open Lean

instance : ToJson DType where
  toJson dt := Json.str dt.canonicalName

instance : FromJson DType where
  fromJson?
    | .str s => pure (DType.parse s)
    | _ => .error "expected dtype string"

/-- One tensor entry discovered in a SafeTensors source. -/
structure TensorSchema where
  name : String
  dtype : DType
  shape : Shape
  /-- For directory sources this is the shard filename; for single-file sources it is empty. -/
  sourceFile : String := ""
  deriving Inhabited, Repr, BEq, ToJson, FromJson

namespace TensorSchema

/-- Convert SafeTensors metadata into Tyr's shared tensor spec. -/
def toSpec (schema : TensorSchema) : TensorSpec :=
  { shape := schema.shape, dtype := schema.dtype }

/-- Build a boundary contract for loading this tensor. -/
def contract
    (schema : TensorSchema)
    (role : TensorRole := .parameter)
    (devicePolicy : DevicePolicy := .any)
    : TensorContract :=
  { spec := schema.toSpec, role, devicePolicy }

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

private def checkLoadedWithContractCore
    {shape : Shape}
    (context : String)
    (contract : TensorContract)
    (raw : RawTensor)
    : IO (Tensor { shape := shape, dtype := contract.spec.dtype }) := do
  match Tensor.ofTensorWithContract contract raw with
  | .ok tensor => pure tensor
  | .error err => throw <| IO.userError s!"{context}: {err}"

/--
Check a loaded raw tensor against a shape/dtype/device contract and return a
dtype-indexed view over the same runtime tensor.
-/
def checkLoadedWithContract
    (context : String)
    (contract : TensorContract)
    (raw : RawTensor)
    : IO (Tensor { shape := contract.spec.shape, dtype := contract.spec.dtype }) :=
  checkLoadedWithContractCore context contract raw

/-- Load from an open SafeTensors handle and check against an explicit contract. -/
def loadFromHandleWithContract
    (handle : @& SafeTensorsHandle)
    (name : String)
    (contract : TensorContract)
    (device : Device := Device.CPU)
    : IO (Tensor { shape := contract.spec.shape, dtype := contract.spec.dtype }) := do
  let raw ← RawTensor.safeTensorsLoadFromHandleOnDevice handle name device
  checkLoadedWithContract s!"SafeTensors tensor '{name}'" contract raw

/-- Load from a SafeTensors file and check against an explicit contract. -/
def loadTensorWithContract
    (path : String)
    (name : String)
    (contract : TensorContract)
    (device : Device := Device.CPU)
    : IO (Tensor { shape := contract.spec.shape, dtype := contract.spec.dtype }) := do
  let raw ← RawTensor.safeTensorsLoadOnDevice path name device
  checkLoadedWithContract s!"SafeTensors tensor '{name}'" contract raw

/-- Load from a sharded SafeTensors directory and check against an explicit contract. -/
def loadTensorShardedWithContract
    (dir : String)
    (name : String)
    (contract : TensorContract)
    (device : Device := Device.CPU)
    : IO (Tensor { shape := contract.spec.shape, dtype := contract.spec.dtype }) := do
  let raw ← RawTensor.safeTensorsLoadShardedOnDevice dir name device
  checkLoadedWithContract s!"SafeTensors tensor '{name}'" contract raw

/-- Load from an open SafeTensors handle and check shape, dtype, and target device. -/
def loadFromHandleWithSpec
    (handle : @& SafeTensorsHandle)
    (name : String)
    (spec : TensorSpec)
    (device : Device := Device.CPU)
    : IO (Tensor { shape := spec.shape, dtype := spec.dtype }) :=
  loadFromHandleWithContract handle name
    { spec, role := .parameter, devicePolicy := .exact device }
    device

/-- Load from a SafeTensors file and check shape, dtype, and target device. -/
def loadTensorWithSpec
    (path : String)
    (name : String)
    (spec : TensorSpec)
    (device : Device := Device.CPU)
    : IO (Tensor { shape := spec.shape, dtype := spec.dtype }) :=
  loadTensorWithContract path name
    { spec, role := .parameter, devicePolicy := .exact device }
    device

/-- Load from a sharded SafeTensors directory and check shape, dtype, and target device. -/
def loadTensorShardedWithSpec
    (dir : String)
    (name : String)
    (spec : TensorSpec)
    (device : Device := Device.CPU)
    : IO (Tensor { shape := spec.shape, dtype := spec.dtype }) :=
  loadTensorShardedWithContract dir name
    { spec, role := .parameter, devicePolicy := .exact device }
    device

/-- Load using a discovered tensor schema from an open handle. -/
def loadFromHandleWithSchema
    (handle : @& SafeTensorsHandle)
    (schema : TensorSchema)
    (device : Device := Device.CPU)
    : IO (Tensor { shape := schema.shape, dtype := schema.dtype }) :=
  loadFromHandleWithSpec handle schema.name schema.toSpec device

/-- Load using a discovered tensor schema from a SafeTensors file. -/
def loadTensorWithSchema
    (path : String)
    (schema : TensorSchema)
    (device : Device := Device.CPU)
    : IO (Tensor { shape := schema.shape, dtype := schema.dtype }) :=
  loadTensorWithSpec path schema.name schema.toSpec device

/-- Load using a discovered tensor schema from a sharded SafeTensors directory. -/
def loadTensorShardedWithSchema
    (dir : String)
    (schema : TensorSchema)
    (device : Device := Device.CPU)
    : IO (Tensor { shape := schema.shape, dtype := schema.dtype }) :=
  loadTensorShardedWithSpec dir schema.name schema.toSpec device

private def getObjVal? (j : Json) (key : String) : Option Json :=
  match j with
  | .obj kvs => Std.TreeMap.Raw.get? kvs key
  | _ => none

private def getArr? (j : Json) : Option (Array Json) :=
  match j with
  | .arr xs => some xs
  | _ => none

private def getStr? (j : Json) : Option String :=
  match j with
  | .str s => some s
  | _ => none

private def getObjPairs? (j : Json) : Option (List (String × Json)) :=
  match j with
  | .obj kvs => some kvs.toList
  | _ => none

private def getNat? (j : Json) : Option Nat :=
  match (FromJson.fromJson? j : Except String Nat) with
  | .ok n => some n
  | .error _ => none

private def readU64LE? (bytes : ByteArray) (offset : Nat) : Option UInt64 :=
  if offset + 8 > bytes.size then
    none
  else
    let b0 := bytes[offset]!
    let b1 := bytes[offset + 1]!
    let b2 := bytes[offset + 2]!
    let b3 := bytes[offset + 3]!
    let b4 := bytes[offset + 4]!
    let b5 := bytes[offset + 5]!
    let b6 := bytes[offset + 6]!
    let b7 := bytes[offset + 7]!
    some (
      b0.toUInt64 |||
      (b1.toUInt64 <<< 8) |||
      (b2.toUInt64 <<< 16) |||
      (b3.toUInt64 <<< 24) |||
      (b4.toUInt64 <<< 32) |||
      (b5.toUInt64 <<< 40) |||
      (b6.toUInt64 <<< 48) |||
      (b7.toUInt64 <<< 56)
    )

private def parseTensorEntry (tensorName sourceFile : String) (entryJson : Json)
    : Except String TensorSchema := do
  let dtypeJson ←
    match getObjVal? entryJson "dtype" with
    | some j => pure j
    | none => .error s!"SafeTensors entry '{tensorName}' is missing 'dtype'"
  let dtypeRaw ←
    match getStr? dtypeJson with
    | some s => pure s
    | none => .error s!"SafeTensors entry '{tensorName}' has non-string 'dtype'"
  let dtype ←
    match DType.ofString? dtypeRaw with
    | some dt => pure dt
    | none => .error s!"SafeTensors entry '{tensorName}' has unsupported dtype '{dtypeRaw}'"

  let shapeJson ←
    match getObjVal? entryJson "shape" with
    | some j => pure j
    | none => .error s!"SafeTensors entry '{tensorName}' is missing 'shape'"
  let dimsJson ←
    match getArr? shapeJson with
    | some xs => pure xs
    | none => .error s!"SafeTensors entry '{tensorName}' has non-array 'shape'"

  let mut shape : Shape := #[]
  for dimJson in dimsJson do
    let n ←
      match getNat? dimJson with
      | some x => pure x
      | none => .error s!"SafeTensors entry '{tensorName}' has non-natural shape dimension"
    shape := shape.push n.toUInt64

  pure {
    name := tensorName
    dtype := dtype
    shape := shape
    sourceFile := sourceFile
  }

private def parseHeaderEntries (sourceFile : String) (headerJson : Json)
    : Except String (Array TensorSchema) := do
  let pairs ←
    match headerJson with
    | .obj kvs => pure kvs.toList
    | _ => .error "SafeTensors header is not a JSON object"

  let mut entries : Array TensorSchema := #[]
  for (tensorName, entryJson) in pairs do
    if tensorName != "__metadata__" then
      let parsed ← parseTensorEntry tensorName sourceFile entryJson
      entries := entries.push parsed
  pure entries

/-- Maximum JSON header allocation during schema discovery (100 MiB). -/
def maxHeaderBytes : Nat := 100 * 1024 * 1024

private def readHeaderBytes (handle : IO.FS.Handle) (count : Nat) : IO ByteArray := do
  let mut bytes := ByteArray.empty
  while bytes.size < count do
    let chunk ← handle.read (count - bytes.size).toUSize
    if chunk.isEmpty then break
    bytes := bytes ++ chunk
  pure bytes

private def parseSafeTensorFile (path sourceFile : String) : IO (Array TensorSchema) := do
  let handle ← IO.FS.Handle.mk path .read
  let bytes ← readHeaderBytes handle 8
  let headerSize ←
    match readU64LE? bytes 0 with
    | some n => pure n
    | none => throw <| IO.userError s!"Invalid SafeTensors file '{path}': missing 8-byte header size"
  let headerSizeNat := headerSize.toNat
  if headerSizeNat > maxHeaderBytes then
    throw <| IO.userError
      s!"Invalid SafeTensors file '{path}': header exceeds {maxHeaderBytes}-byte limit"
  let fileSize := (← (System.FilePath.mk path).metadata).byteSize.toNat
  if 8 + headerSizeNat > fileSize then
    throw <| IO.userError
      s!"Invalid SafeTensors file '{path}': header exceeds file size ({headerSizeNat} bytes)"

  let headerBytes ← readHeaderBytes handle headerSizeNat
  if headerBytes.size != headerSizeNat then
    throw <| IO.userError s!"Invalid SafeTensors file '{path}': truncated header"
  let headerStr ←
    match String.fromUTF8? headerBytes with
    | some s => pure s
    | none => throw <| IO.userError s!"Invalid SafeTensors file '{path}': header is not UTF-8"
  let headerJson ←
    match Json.parse headerStr with
    | .ok j => pure j
    | .error err =>
      throw <| IO.userError s!"Invalid SafeTensors file '{path}': failed to parse header JSON: {err}"

  match parseHeaderEntries sourceFile headerJson with
  | .ok entries => pure entries
  | .error err => throw <| IO.userError s!"Invalid SafeTensors file '{path}': {err}"

private def parseJsonFile (path : String) : IO Json := do
  let contents ← IO.FS.readFile path
  match Json.parse contents with
  | .ok j => pure j
  | .error err =>
      throw <| IO.userError s!"Invalid JSON file '{path}': {err}"

private def parseWeightMap (indexPath : String) : IO (Array (String × String)) := do
  let root ← parseJsonFile indexPath
  let weightMapJson ←
    match getObjVal? root "weight_map" with
    | some j => pure j
    | none =>
        throw <| IO.userError s!"Invalid index file '{indexPath}': missing 'weight_map'"
  let pairs ←
    match getObjPairs? weightMapJson with
    | some kvs => pure kvs
    | none =>
        throw <| IO.userError
          s!"Invalid index file '{indexPath}': 'weight_map' must be a JSON object"
  if pairs.isEmpty then
    throw <| IO.userError s!"Invalid index file '{indexPath}': 'weight_map' is empty"

  let mut mappings : Array (String × String) := #[]
  for (tensorName, shardJson) in pairs do
    let shardFile ←
      match getStr? shardJson with
      | some shard =>
          if shard.isEmpty then
            throw <| IO.userError
              s!"Invalid index file '{indexPath}': tensor '{tensorName}' maps to an empty shard filename"
          else
            pure shard
      | none =>
          throw <| IO.userError
            s!"Invalid index file '{indexPath}': tensor '{tensorName}' has non-string shard filename"
    mappings := mappings.push (tensorName, shardFile)
  pure mappings

private def hasWindowsDrivePrefix (path : String) : Bool :=
  match path.toList with
  | c :: ':' :: _ => c.isAlpha
  | _ => false

private def hasPathSegment (path segment : String) : Bool :=
  (path.splitOn "/").contains segment || (path.splitOn "\\").contains segment

private def isUnsafeShardPath (shardFile : String) : Bool :=
  shardFile.startsWith "/" ||
  shardFile.startsWith "\\" ||
  hasWindowsDrivePrefix shardFile ||
  hasPathSegment shardFile ".." ||
  hasPathSegment shardFile "."

private def listShardFiles (dir : System.FilePath) : IO (Array String) := do
  let entries ← dir.readDir
  let mut shardFiles : Array String := #[]
  for entry in entries do
    if !(← entry.path.isDir) && entry.path.extension == some "safetensors" then
      shardFiles := shardFiles.push entry.fileName
  if shardFiles.isEmpty then
    throw <| IO.userError s!"No '.safetensors' files found in directory '{dir}'"
  pure <| shardFiles.qsort (· < ·)

private def pushUnique (xs : Array String) (x : String) : Array String :=
  if xs.contains x then xs else xs.push x

private def mapByTensorName (entries : Array TensorSchema) : Std.HashMap String TensorSchema :=
  Id.run do
    let mut out : Std.HashMap String TensorSchema := {}
    for entry in entries do
      out := out.insert entry.name entry
    pure out

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

/-- Introspect tensor schema from either:
    - a single `.safetensors` file, or
    - a directory containing shard `.safetensors` files. -/
def introspect (source : String) : IO Schema := do
  let sourcePath : System.FilePath := ⟨source⟩
  if !(← sourcePath.pathExists) then
    throw <| IO.userError s!"SafeTensors source does not exist: {source}"

  if isSchemaSnapshotPath sourcePath then
    return ← loadSnapshot source

  if ← sourcePath.isDir then
    let indexPath := sourcePath / "model.safetensors.index.json"
    let tensors ←
      if ← indexPath.pathExists then
        let weightMap ← parseWeightMap indexPath.toString
        let mut referencedShards : Array String := #[]
        for (_, shardFile) in weightMap do
          if isUnsafeShardPath shardFile then
            throw <| IO.userError
              s!"Invalid index file '{indexPath}': unsafe shard path '{shardFile}' (must be relative and stay within source directory)"
          referencedShards := pushUnique referencedShards shardFile
        let referencedShardList := referencedShards.qsort (· < ·)

        let mut shardTensorMaps : Std.HashMap String (Std.HashMap String TensorSchema) := {}
        for shardFile in referencedShardList do
          let shardPath := sourcePath / shardFile
          let shardPathRel : System.FilePath := ⟨shardFile⟩
          if shardPathRel.extension != some "safetensors" then
            throw <| IO.userError
              s!"Invalid index file '{indexPath}': shard '{shardFile}' must use '.safetensors' extension"
          if !(← shardPath.pathExists) then
            throw <| IO.userError
              s!"Invalid index file '{indexPath}': referenced shard does not exist: '{shardFile}'"
          if ← shardPath.isDir then
            throw <| IO.userError
              s!"Invalid index file '{indexPath}': referenced shard is a directory: '{shardFile}'"
          let entries ← parseSafeTensorFile shardPath.toString shardFile
          shardTensorMaps := shardTensorMaps.insert shardFile (mapByTensorName entries)

        let mut fromIndex : Array TensorSchema := #[]
        for (tensorName, shardFile) in weightMap do
          let shardMap ←
            match shardTensorMaps.get? shardFile with
            | some m => pure m
            | none =>
                throw <| IO.userError
                  s!"Invalid index file '{indexPath}': shard '{shardFile}' was not loaded"
          let entry ←
            match shardMap.get? tensorName with
            | some t => pure t
            | none =>
                throw <| IO.userError
                  s!"Invalid index file '{indexPath}': tensor '{tensorName}' not found in shard '{shardFile}'"
          fromIndex := fromIndex.push { entry with sourceFile := shardFile }
        pure fromIndex
      else
        let shardFiles ← listShardFiles sourcePath
        let mut fromShards : Array TensorSchema := #[]
        for shardFile in shardFiles do
          let fullPath := (sourcePath / shardFile).toString
          let shardEntries ← parseSafeTensorFile fullPath shardFile
          fromShards := fromShards ++ shardEntries
        pure fromShards

    ensureUniqueNames tensors
    pure {
      source := source
      sourceIsDirectory := true
      tensors := sortedByName tensors
    }
  else
    if sourcePath.extension != some "safetensors" then
      throw <| IO.userError
        s!"SafeTensors file must use '.safetensors' extension: {source}"
    let tensors ← parseSafeTensorFile source ""
    pure {
      source := source
      sourceIsDirectory := false
      tensors := sortedByName tensors
    }

end torch.safetensors
