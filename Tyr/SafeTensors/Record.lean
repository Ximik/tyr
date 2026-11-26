import Tyr.SafeTensors.Schema

/-! The typed record of a SafeTensors schema: dotted tensor names become nested
records, and contiguous numeric segments with identical subtrees become arrays.
Building the record only describes it; rendering it is up to the caller. -/

namespace torch.safetensors

/-- The type of a record field. -/
inductive FieldType where
  | tensor (dtype : DType) (shape : Shape)
  | record (typeName : String)
  | array (elem : FieldType) (count : Nat)
  deriving Repr, BEq

/-- A record field. Its tensors are named under the record's prefix extended by
`segment`, or under the prefix itself when `segment` is `none`; array elements
add their index. -/
structure Field where
  name : String
  segment : Option String
  type : FieldType
  deriving Repr, BEq

structure RecordType where
  name : String
  fields : Array Field
  deriving Repr, BEq

/-- The record types of a schema, each after the types its fields use; the root
type is last. -/
structure Record where
  types : Array RecordType
  deriving Repr, BEq

private def leanKeywords : List String :=
  [ "abbrev", "axiom", "class", "def", "deriving", "do", "else", "end", "example", "forall"
  , "fun", "if", "import", "in", "inductive", "instance", "let", "macro", "match", "mutual"
  , "namespace", "noncomputable", "opaque", "open", "private", "protected", "section", "set_option"
  , "structure", "syntax", "term", "then", "theorem", "universe", "unsafe", "where", "with"
  ]

private def replaceInvalidChars (raw : String) : String :=
  String.ofList <| raw.toList.map fun c => if c.isAlphanum || c == '_' then c else '_'

private def ensureValidStart (raw : String) : String :=
  match raw.toList with
  | [] => "tensor"
  | c :: _ => if c.isAlpha || c == '_' then raw else s!"t_{raw}"

private def sanitizeIdent (raw fallback : String) : String :=
  let cleaned := replaceInvalidChars raw
  let cleaned := if cleaned.isEmpty then fallback else cleaned
  let cleaned := ensureValidStart cleaned
  if leanKeywords.contains cleaned then cleaned ++ "_" else cleaned

private partial def freshNameAux (used : List String) (base : String) (idx : Nat) : String :=
  let candidate := if idx == 0 then base else s!"{base}_{idx}"
  if used.contains candidate then
    freshNameAux used base (idx + 1)
  else
    candidate

private def freshName (used : List String) (base : String) : String :=
  freshNameAux used base 0

private def splitWords (raw : String) : List String :=
  let spaced := String.ofList <| raw.toList.map fun c => if c.isAlphanum then c else ' '
  (spaced.splitOn " ").filter (fun part => !part.isEmpty)

private def capitalizeWord (word : String) : String :=
  match word.toList with
  | [] => ""
  | c :: cs => String.ofList (c.toUpper :: cs.map Char.toLower)

private def typeNameBase (path : List String) : String :=
  if path.isEmpty then
    "Weights"
  else
    let suffix :=
      path.foldl (init := "") fun acc part =>
        let merged := (splitWords part).foldl (fun a w => a ++ capitalizeWord w) ""
        acc ++ capitalizeWord (if merged.isEmpty then part else merged)
    "Weights" ++ suffix

private structure LeafInfo where
  tensorName : String
  dtype : DType
  shape : Shape

private structure PathNode where
  leaf? : Option LeafInfo := none
  named : Array (String × PathNode) := #[]
  indexed : Array (Nat × PathNode) := #[]

private def emptyPathNode : PathNode := {}

private def upsertNamed (children : Array (String × PathNode)) (key : String) (child : PathNode)
    : Array (String × PathNode) :=
  let updated :=
    match children.findIdx? (fun pair => pair.1 == key) with
    | some i => children.set! i (key, child)
    | none => children.push (key, child)
  updated.qsort (fun a b => a.1 < b.1)

private def upsertIndexed (children : Array (Nat × PathNode)) (key : Nat) (child : PathNode)
    : Array (Nat × PathNode) :=
  let updated :=
    match children.findIdx? (fun pair => pair.1 == key) with
    | some i => children.set! i (key, child)
    | none => children.push (key, child)
  updated.qsort (fun a b => a.1 < b.1)

private partial def insertPath (node : PathNode) (segments : List String) (leaf : LeafInfo)
    : Except String PathNode := do
  match segments with
  | [] =>
      match node.leaf? with
      | some _ => .error s!"duplicate tensor path '{leaf.tensorName}'"
      | none => pure { node with leaf? := some leaf }
  | seg :: rest =>
      match seg.toNat? with
      | some idx =>
          let child := (node.indexed.findSome? fun (k, c) => if k == idx then some c else none).getD emptyPathNode
          let child' ← insertPath child rest leaf
          pure { node with indexed := upsertIndexed node.indexed idx child' }
      | none =>
          let child := (node.named.findSome? fun (k, c) => if k == seg then some c else none).getD emptyPathNode
          let child' ← insertPath child rest leaf
          pure { node with named := upsertNamed node.named seg child' }

private def namedChildren (node : PathNode) : Array (String × PathNode) := node.named
private def indexedChildren (node : PathNode) : Array (Nat × PathNode) := node.indexed

private def shapeFingerprint (shape : Shape) : String :=
  "[" ++ String.intercalate "," (shape.toList.map toString) ++ "]"

private partial def nodeFingerprint (node : PathNode) : String :=
  let leafPart :=
    match node.leaf? with
    | none => "N"
    | some leaf => s!"L({leaf.dtype}:{shapeFingerprint leaf.shape})"
  let namedPart :=
    String.intercalate "," <| (namedChildren node).toList.map fun (k, child) =>
      s!"{k}:{nodeFingerprint child}"
  let indexedPart :=
    String.intercalate "," <| (indexedChildren node).toList.map fun (i, child) =>
      s!"{i}:{nodeFingerprint child}"
  leafPart ++ "{" ++ namedPart ++ "}[" ++ indexedPart ++ "]"

private def pathLabel (path : List String) : String :=
  if path.isEmpty then "<root>" else String.intercalate "." path

private def validateIndexedContiguous (path : List String) (children : Array (Nat × PathNode))
    : Except String Unit := do
  let mut expected : Nat := 0
  for (idx, _) in children do
    if idx != expected then
      throw s!
        "SafeTensors record: non-contiguous numeric path segment under '{pathLabel path}'. expected index {expected}, found {idx}."
    expected := expected + 1

private def indexedChildrenUniform (children : Array (Nat × PathNode)) : Bool :=
  Id.run do
    let mut firstFp? : Option String := none
    for (_, child) in children do
      let fp := nodeFingerprint child
      match firstFp? with
      | none => firstFp? := some fp
      | some firstFp =>
          if fp != firstFp then
            return false
    true

private def fieldBase (raw : String) : String :=
  sanitizeIdent (replaceInvalidChars raw |>.map Char.toLower) "field"

private partial def buildType (node : PathNode) (path : List String) (used : List String)
    (types : Array RecordType) : Except String (FieldType × List String × Array RecordType) := do
  let named := namedChildren node
  let indexed := indexedChildren node
  if let some leaf := node.leaf? then
    if named.isEmpty && indexed.isEmpty then
      return (.tensor leaf.dtype leaf.shape, used, types)
  let mut used := used
  let mut types := types
  if node.leaf?.isNone && named.isEmpty && !indexed.isEmpty then
    validateIndexedContiguous path indexed
    if indexedChildrenUniform indexed then
      let some (_, first) := indexed[0]?
        | throw s!"SafeTensors record: empty indexed collection under '{pathLabel path}'"
      let (elem, used', types') ← buildType first (path ++ ["item"]) used types
      return (.array elem indexed.size, used', types')
  let mut fields : Array Field := #[]
  let mut fieldNames : List String := []
  if let some leaf := node.leaf? then
    let name := freshName fieldNames "value"
    fieldNames := name :: fieldNames
    fields := fields.push { name, segment := none, type := .tensor leaf.dtype leaf.shape }
  for (seg, child) in named do
    let name := freshName fieldNames (fieldBase seg)
    fieldNames := name :: fieldNames
    let (type, used', types') ← buildType child (path ++ [name]) used types
    used := used'; types := types'
    fields := fields.push { name, segment := some seg, type }
  if !indexed.isEmpty then
    validateIndexedContiguous (if named.isEmpty && node.leaf?.isNone then path else path ++ ["<index>"]) indexed
    if indexedChildrenUniform indexed && !(named.isEmpty && node.leaf?.isNone) then
      let name := freshName fieldNames "items"
      fieldNames := name :: fieldNames
      let some (_, first) := indexed[0]?
        | throw s!"SafeTensors record: empty indexed collection under '{pathLabel path}'"
      let (elem, used', types') ← buildType first (path ++ [name]) used types
      used := used'; types := types'
      fields := fields.push { name, segment := none, type := .array elem indexed.size }
    else
      for (idx, child) in indexed do
        let name := freshName fieldNames s!"i{idx}"
        fieldNames := name :: fieldNames
        let (type, used', types') ← buildType child (path ++ [name]) used types
        used := used'; types := types'
        fields := fields.push { name, segment := some (toString idx), type }
  let typeName := freshName used (typeNameBase path)
  return (.record typeName, typeName :: used, types.push { name := typeName, fields })

/-- The record of `tensors`. Fails on a duplicate or empty tensor name, and on
numeric segments that are not `0, 1, …`. -/
def Record.ofSchema (tensors : Array TensorSchema) : Except String Record := do
  let mut root : PathNode := {}
  for tensor in tensors do
    let segments := (tensor.name.splitOn ".").filter (!·.isEmpty)
    if segments.isEmpty then
      throw s!"SafeTensors record: empty tensor name"
    root ← insertPath root segments { tensorName := tensor.name, dtype := tensor.dtype, shape := tensor.shape }
  match ← buildType root [] [] #[] with
  | (.record _, _, types) => return { types }
  | (other, _, types) =>
      return { types := types.push { name := typeNameBase [], fields := #[{ name := "value", segment := none, type := other }] } }

end torch.safetensors
