import Tyr.RawTensor.SafeTensors
import Lean.Data.Json

/-!
SafeTensors files: an 8-byte little-endian header size, a JSON header giving
each tensor's dtype, shape and byte range, then the tensor data. Lean parses
and writes the header; `Tyr.RawTensor.SafeTensors` only moves tensor bytes.
Dtype names and element widths come from `DType.safeTensorFormat?`.
-/

namespace torch.safetensors

open Lean

/-- Maximum JSON header size accepted (100 MiB). -/
def maxHeaderBytes : Nat := 100 * 1024 * 1024

/-- One tensor's header entry. The byte range is relative to the data section. -/
structure Entry where
  dtype : DType
  shape : Shape
  dataStart : Nat
  dataEnd : Nat
  deriving Inhabited, Repr

/-- A parsed SafeTensors file. -/
structure SafeTensorsHandle where
  path : String
  /-- File offset of the data section. -/
  dataOffset : Nat
  entries : Std.HashMap String Entry
  deriving Inhabited

private def getObjVal? (j : Json) (key : String) : Option Json :=
  match j with
  | .obj kvs => Std.TreeMap.Raw.get? kvs key
  | _ => none

private def getNats? (j : Json) : Option (Array Nat) :=
  match j with
  | .arr xs => xs.mapM fun x => (FromJson.fromJson? x : Except String Nat).toOption
  | _ => none

private def readU64LE? (bytes : ByteArray) (offset : Nat) : Option UInt64 :=
  if offset + 8 > bytes.size then
    none
  else
    some <| (List.range 8).foldl
      (fun acc i => acc ||| (bytes[offset + i]!.toUInt64 <<< (8 * i).toUInt64)) 0

private def readBytes (handle : IO.FS.Handle) (count : Nat) : IO ByteArray := do
  let mut bytes := ByteArray.empty
  while bytes.size < count do
    let chunk ← handle.read (count - bytes.size).toUSize
    if chunk.isEmpty then break
    bytes := bytes ++ chunk
  pure bytes

private def parseEntry (name : String) (json : Json) (dataSize : Nat) : Except String Entry := do
  let some dtypeJson := getObjVal? json "dtype"
    | throw s!"SafeTensors entry '{name}' is missing 'dtype'"
  let .str tag := dtypeJson
    | throw s!"SafeTensors entry '{name}' has non-string 'dtype'"
  let some (dtype, width) := DType.all.findSome? fun dtype =>
      dtype.safeTensorFormat?.bind fun (t, w) => if t == tag then some (dtype, w) else none
    | throw s!"SafeTensors entry '{name}' has unsupported dtype '{tag}'"
  let some shapeJson := getObjVal? json "shape"
    | throw s!"SafeTensors entry '{name}' is missing 'shape'"
  let some dims := getNats? shapeJson
    | throw s!"SafeTensors entry '{name}' has a malformed 'shape'"
  let some offsetsJson := getObjVal? json "data_offsets"
    | throw s!"SafeTensors entry '{name}' is missing 'data_offsets'"
  let some #[dataStart, dataEnd] := getNats? offsetsJson
    | throw s!"SafeTensors entry '{name}' has malformed 'data_offsets'"
  if dataStart > dataEnd then
    throw s!"SafeTensors entry '{name}' has an invalid data_offsets range"
  if dataEnd > dataSize then
    throw s!"SafeTensors entry '{name}' exceeds the file's data section (truncated file)"
  if dataEnd - dataStart != dims.foldl (· * ·) 1 * width then
    throw s!"SafeTensors entry '{name}': data size mismatch (offset range {dataEnd - dataStart} bytes)"
  pure { dtype, shape := dims, dataStart, dataEnd }

/-- Parse and validate the header of the SafeTensors file at `path`. -/
def readHeader (path : String) : IO SafeTensorsHandle := do
  let invalid (message : String) : IO.Error :=
    IO.userError s!"Invalid SafeTensors file '{path}': {message}"
  let handle ← IO.FS.Handle.mk path .read
  let some headerSize := readU64LE? (← readBytes handle 8) 0
    | throw (invalid "missing 8-byte header size")
  let headerSize := headerSize.toNat
  if headerSize > maxHeaderBytes then
    throw (invalid s!"header exceeds {maxHeaderBytes}-byte limit")
  let fileSize := (← (System.FilePath.mk path).metadata).byteSize.toNat
  if 8 + headerSize > fileSize then
    throw (invalid s!"header exceeds file size ({headerSize} bytes)")
  let headerBytes ← readBytes handle headerSize
  if headerBytes.size != headerSize then
    throw (invalid "truncated header")
  let some text := String.fromUTF8? headerBytes
    | throw (invalid "header is not UTF-8")
  let json ← match Json.parse text with
    | .ok json => pure json
    | .error err => throw (invalid s!"failed to parse header JSON: {err}")
  let .obj kvs := json
    | throw (invalid "header is not a JSON object")
  let dataOffset := 8 + headerSize
  let mut entries : Std.HashMap String Entry := {}
  for (name, entryJson) in kvs.toList do
    if name != "__metadata__" then
      match parseEntry name entryJson (fileSize - dataOffset) with
      | .ok entry => entries := entries.insert name entry
      | .error message => throw (invalid message)
  pure { path, dataOffset, entries }

initialize handleCache : IO.Ref (Std.HashMap String SafeTensorsHandle) ← IO.mkRef {}

/-- The parsed header of `path`, read once per path and then reused. -/
def openHandle (path : String) : IO SafeTensorsHandle := do
  if let some handle := (← handleCache.get)[path]? then
    return handle
  let handle ← readHeader path
  handleCache.modify (·.insert path handle)
  pure handle

initialize shardCache : IO.Ref (Std.HashMap String (Std.HashMap String SafeTensorsHandle)) ←
  IO.mkRef {}

/-- The file holding each tensor among the `.safetensors` files directly in `dir`. -/
private def openShards (dir : String) : IO (Std.HashMap String SafeTensorsHandle) := do
  if let some shards := (← shardCache.get)[dir]? then
    return shards
  let mut paths : Array String := #[]
  for entry in ← System.FilePath.readDir dir do
    if entry.path.extension == some "safetensors" && !(← entry.path.isDir) then
      paths := paths.push entry.path.toString
  if paths.isEmpty then
    throw <| IO.userError s!"No .safetensors files found in directory: {dir}"
  let mut byName : Std.HashMap String SafeTensorsHandle := {}
  for path in paths.qsort (· < ·) do
    let handle ← openHandle path
    for (name, _) in handle.entries do
      if byName.contains name then
        throw <| IO.userError s!"Duplicate tensor found across safetensors shards: {name}"
      byName := byName.insert name handle
  shardCache.modify (·.insert dir byName)
  pure byName

end torch.safetensors

namespace torch.RawTensor

open Lean torch.safetensors

/-- Load `name` with its stored dtype and shape on CPU. A nonempty `shape`
reshapes it, which must preserve the element count. -/
def safeTensorsLoadFromHandle (handle : SafeTensorsHandle) (name : String) (shape : Shape) :
    IO RawTensor := do
  let some entry := handle.entries[name]?
    | throw <| IO.userError s!"Tensor not found: {name}"
  let raw ← safeTensorsRead handle.path (handle.dataOffset + entry.dataStart)
    (entry.dataEnd - entry.dataStart) entry.dtype entry.shape
  if shape.isEmpty then
    return raw
  match reshapeChecked raw shape with
  | .ok reshaped => pure reshaped
  | .error _ =>
    throw <| IO.userError "SafeTensors load error: loaded tensor size does not match requested shape."

def safeTensorsLoad (path name : String) (shape : Shape) : IO RawTensor := do
  safeTensorsLoadFromHandle (← openHandle path) name shape

def safeTensorsLoadSharded (dir name : String) (shape : Shape) : IO RawTensor := do
  let some handle := (← openShards dir)[name]?
    | throw <| IO.userError s!"Tensor not found in any shard: {name}"
  safeTensorsLoadFromHandle handle name shape

def safeTensorsLoadOnDevice (path name : String) (device : Device) : IO RawTensor := do
  let raw ← safeTensorsLoad path name #[]
  pure (toDevice raw device)

def safeTensorsLoadShardedOnDevice (dir name : String) (device : Device) : IO RawTensor := do
  let raw ← safeTensorsLoadSharded dir name #[]
  pure (toDevice raw device)

def safeTensorsLoadFromHandleOnDevice (handle : SafeTensorsHandle) (name : String)
    (device : Device) : IO RawTensor := do
  let raw ← safeTensorsLoadFromHandle handle name #[]
  pure (toDevice raw device)

/-- Write `input` as the only tensor of a new SafeTensors file at `path`. -/
def saveSafeTensor (path name : String) (input : RawTensor) : IO Unit := do
  if name.isEmpty then
    throw <| IO.userError "SafeTensors save error: tensor name cannot be empty."
  let some (tag, width) := input.dtype.safeTensorFormat?
    | throw <| IO.userError s!"SafeTensors save error: unsupported dtype for tensor '{name}'."
  let shape := input.runtimeShape
  let length := shape.foldl (· * ·) 1 * width
  let entry := Json.mkObj [("dtype", Json.str tag), ("shape", toJson shape),
    ("data_offsets", toJson #[0, length])]
  safeTensorsWrite path (Json.mkObj [(name, entry)]).compress #[input] #[length.toUInt64]

end torch.RawTensor
