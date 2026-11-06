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

private def hasWindowsDrivePrefix (path : String) : Bool :=
  match path.toList with
  | c :: ':' :: _ => c.isAlpha
  | _ => false

private def hasPathSegment (path segment : String) : Bool :=
  (path.splitOn "/").contains segment || (path.splitOn "\\").contains segment

/-- The `weight_map` of a `model.safetensors.index.json`: each tensor's shard file.
Shard files must be relative `.safetensors` paths that stay inside the index's directory. -/
def readWeightMap (indexPath : System.FilePath) : IO (Array (String × String)) := do
  let invalid (message : String) : IO.Error :=
    IO.userError s!"Invalid SafeTensors index '{indexPath}': {message}"
  let json ← match Json.parse (← IO.FS.readFile indexPath) with
    | .ok json => pure json
    | .error err => throw (invalid s!"failed to parse JSON: {err}")
  let some weightMapJson := getObjVal? json "weight_map"
    | throw (invalid "missing 'weight_map'")
  let .obj weightMap := weightMapJson
    | throw (invalid "'weight_map' must be a JSON object")
  if weightMap.isEmpty then
    throw (invalid "'weight_map' is empty")
  let mut out : Array (String × String) := #[]
  for (name, fileJson) in weightMap.toList do
    let .str file := fileJson
      | throw (invalid s!"tensor '{name}' has non-string shard filename")
    if file.isEmpty then
      throw (invalid s!"tensor '{name}' maps to an empty shard filename")
    if file.startsWith "/" || file.startsWith "\\" || hasWindowsDrivePrefix file ||
        hasPathSegment file ".." || hasPathSegment file "." then
      throw (invalid s!"unsafe shard path '{file}' (must be relative and stay within the directory)")
    if (System.FilePath.mk file).extension != some "safetensors" then
      throw (invalid s!"shard '{file}' must use '.safetensors' extension")
    out := out.push (name, file)
  pure out

/-- Where a checkpoint tensor is stored. -/
structure Stored where
  /-- The file relative to the checkpoint directory; empty for a single-file checkpoint. -/
  file : String
  handle : SafeTensorsHandle
  entry : Entry

/-- Every tensor of one parsed file. -/
def storedIn (handle : SafeTensorsHandle) (file : String := "") : Std.HashMap String Stored :=
  handle.entries.fold (fun m name entry => m.insert name { file, handle, entry }) {}

initialize modelDirCache : IO.Ref (Std.HashMap String (Std.HashMap String Stored)) ←
  IO.mkRef {}

/-- The tensors of a Hugging Face model directory: the shards listed by its
`model.safetensors.index.json`, each tensor in its listed shard, or else its
`model.safetensors`. No other file in the directory is read. -/
def openModelDir (dir : String) : IO (Std.HashMap String Stored) := do
  if let some tensors := (← modelDirCache.get)[dir]? then
    return tensors
  let root := System.FilePath.mk dir
  let indexPath := root / "model.safetensors.index.json"
  let tensors ←
    if ← indexPath.pathExists then
      let invalid (message : String) : IO.Error :=
        IO.userError s!"Invalid SafeTensors index '{indexPath}': {message}"
      let mut tensors : Std.HashMap String Stored := {}
      for (name, file) in ← readWeightMap indexPath do
        let path := root / file
        unless ← path.pathExists do
          throw (invalid s!"referenced shard does not exist: '{file}'")
        if ← path.isDir then
          throw (invalid s!"referenced shard is a directory: '{file}'")
        let handle ← openHandle path.toString
        let some entry := handle.entries[name]?
          | throw (invalid s!"tensor '{name}' not found in shard '{file}'")
        tensors := tensors.insert name { file, handle, entry }
      pure tensors
    else if ← (root / "model.safetensors").pathExists then
      pure (storedIn (← openHandle (root / "model.safetensors").toString) "model.safetensors")
    else
      throw <| IO.userError s!"No model.safetensors.index.json or model.safetensors in {dir}"
  modelDirCache.modify (·.insert dir tensors)
  pure tensors

end torch.safetensors

namespace torch.RawTensor

open Lean torch.safetensors

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
