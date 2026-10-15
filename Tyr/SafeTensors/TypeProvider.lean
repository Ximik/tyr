/-
  Tyr/SafeTensors/TypeProvider.lean

  Command-level SafeTensors type provider:

    safetensors_type_provider "path/to/model.safetensors" as MyWeights
    safetensors_type_provider "path/to/sharded_dir" as MyWeights

  The command introspects tensor schema at elaboration time and emits:
  - typed tensor specs
  - typed per-tensor load helpers
  - hierarchical typed checkpoint record (`Weights`) with `loadAll`
  - generated schema table for introspection from Lean
-/
import Lean
import Tyr.SafeTensors.Schema

open Lean Elab Command

namespace torch.safetensors

syntax (name := safetensorsTypeProviderCmd)
  "safetensors_type_provider " str " as " ident : command

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

private def tensorDeclBaseName (raw : String) : String :=
  sanitizeIdent (replaceInvalidChars raw |>.map Char.toLower) "tensor"

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

private def escapeLeanStringChars : List Char → List Char
  | [] => []
  | c :: cs =>
      let escaped :=
        match c with
        | '\"' => ['\\', '\"']
        | '\\' => ['\\', '\\']
        | '\n' => ['\\', 'n']
        | '\r' => ['\\', 'r']
        | '\t' => ['\\', 't']
        | other => [other]
      escaped ++ escapeLeanStringChars cs

private def escapeLeanString (raw : String) : String :=
  String.ofList <| escapeLeanStringChars raw.toList

private def leanStringLit (raw : String) : String :=
  "\"" ++ escapeLeanString raw ++ "\""

private def renderShapeLit (shape : Shape) : String :=
  let dims := shape.toList.map (fun d => toString d)
  "#[" ++ String.intercalate ", " dims ++ "]"

private def renderDTypeExpr : DType → String
  | .Unknown raw => s!"(_root_.torch.DType.Unknown {leanStringLit raw})"
  | dtype => s!"_root_.torch.DType.{dtype.canonicalName}"

private def parseGeneratedCommand (source : String) : CommandElabM Syntax := do
  let env ← getEnv
  match Parser.runParserCategory env `command source "<safetensors_type_provider>" with
  | .ok stx => pure stx
  | .error err =>
    throwError "safetensors_type_provider generated invalid Lean command:\n{err}\n\n{source}"

private def elabGeneratedCommand (source : String) : CommandElabM Unit := do
  elabCommand (← parseGeneratedCommand source)

private def indentBlock (n : Nat) (source : String) : String :=
  let pad := String.ofList (List.replicate n ' ')
  String.intercalate "\n" <| (source.splitOn "\n").map (fun line => pad ++ line)

private def renderTensorDecls (sourceIsDirectory : Bool) (tensor : TensorSchema) (declName : String) : Array String :=
  let shapeLit := renderShapeLit tensor.shape
  let dtypeExpr := renderDTypeExpr tensor.dtype
  let typedReturnType :=
    "(_root_.torch.Tensor { shape := " ++ s!"{declName}Shape, dtype := {dtypeExpr}, " ++
    "device := _root_.torch.DevicePolicy.any })"
  let shapeDecl :=
    s!"/-- Static shape for tensor `{tensor.name}`. -/\n" ++
    s!"abbrev {declName}Shape : _root_.torch.Shape := {shapeLit}\n"
  let specDecl :=
    s!"/-- Tensor schema for `{tensor.name}`. -/\n" ++
    s!"def {declName}Spec : _root_.torch.safetensors.TensorSchema :=\n" ++
    "  {\n" ++
    s!"    name := {leanStringLit tensor.name},\n" ++
    s!"    dtype := {dtypeExpr},\n" ++
    s!"    shape := {declName}Shape,\n" ++
    s!"    sourceFile := {leanStringLit tensor.sourceFile}\n" ++
    "  }\n"
  let tensorSpecDecl :=
    s!"/-- Shared tensor spec for `{tensor.name}`. -/\n" ++
    s!"def {declName}TensorSpec : _root_.torch.TensorSpec :=\n" ++
    s!"  {declName}Spec.toSpec\n"
  let loadFromHandleDecl :=
    s!"/-- Load `{tensor.name}` from an open SafeTensors handle with checked shape and dtype. -/\n" ++
    s!"def load_{declName}FromHandle (handle : _root_.torch.safetensors.SafeTensorsHandle) : IO {typedReturnType} :=\n" ++
    s!"  _root_.torch.safetensors.loadFromHandleWithSpec handle {declName}Spec.name {declName}TensorSpec\n"
  let loadDecl :=
    if sourceIsDirectory then
      s!"/-- Load `{tensor.name}` with checked shape and dtype. -/\n" ++
      s!"def load_{declName} (source : String := defaultSource) : IO {typedReturnType} := do\n" ++
      "  let filePath : String :=\n" ++
      "    if " ++ declName ++ "Spec.sourceFile.isEmpty then source else source ++ \"/\" ++ " ++ declName ++ "Spec.sourceFile\n" ++
      s!"  _root_.torch.safetensors.loadTensorWithSpec filePath {declName}Spec.name {declName}TensorSpec\n"
    else
      s!"/-- Load `{tensor.name}` with checked shape and dtype. -/\n" ++
      s!"def load_{declName} (source : String := defaultSource) : IO {typedReturnType} := do\n" ++
      "  let handle ← _root_.torch.safetensors.openHandle source\n" ++
      s!"  load_{declName}FromHandle handle\n"
  let typedLoadFromHandleOnDeviceDecl :=
    s!"/-- Load `{tensor.name}` from an open handle, checking shape, dtype, and target device. -/\n" ++
    s!"def load_{declName}TypedFromHandleOnDevice (handle : _root_.torch.safetensors.SafeTensorsHandle) (device : _root_.torch.Device := _root_.torch.Device.CPU) : IO {typedReturnType} :=\n" ++
    s!"  _root_.torch.safetensors.loadFromHandleWithSpec handle {declName}Spec.name {declName}TensorSpec device\n"
  let typedLoadFromHandleDecl :=
    s!"/-- Load `{tensor.name}` from an open handle, checking shape and dtype on CPU. -/\n" ++
    s!"def load_{declName}TypedFromHandle (handle : _root_.torch.safetensors.SafeTensorsHandle) : IO {typedReturnType} :=\n" ++
    s!"  load_{declName}TypedFromHandleOnDevice handle\n"
  let typedLoadOnDeviceDecl :=
    if sourceIsDirectory then
      s!"/-- Load `{tensor.name}`, checking shape, dtype, and target device. -/\n" ++
      s!"def load_{declName}TypedOnDevice (source : String := defaultSource) (device : _root_.torch.Device := _root_.torch.Device.CPU) : IO {typedReturnType} := do\n" ++
      "  let filePath : String :=\n" ++
      "    if " ++ declName ++ "Spec.sourceFile.isEmpty then source else source ++ \"/\" ++ " ++ declName ++ "Spec.sourceFile\n" ++
      s!"  _root_.torch.safetensors.loadTensorWithSpec filePath {declName}Spec.name {declName}TensorSpec device\n"
    else
      s!"/-- Load `{tensor.name}`, checking shape, dtype, and target device. -/\n" ++
      s!"def load_{declName}TypedOnDevice (source : String := defaultSource) (device : _root_.torch.Device := _root_.torch.Device.CPU) : IO {typedReturnType} := do\n" ++
      "  let handle ← _root_.torch.safetensors.openHandle source\n" ++
      s!"  load_{declName}TypedFromHandleOnDevice handle device\n"
  let typedLoadDecl :=
    s!"/-- Load `{tensor.name}`, checking shape and dtype on CPU. -/\n" ++
    s!"def load_{declName}Typed (source : String := defaultSource) : IO {typedReturnType} :=\n" ++
    s!"  load_{declName}TypedOnDevice source\n"
  if sourceIsDirectory then
    #[shapeDecl, specDecl, tensorSpecDecl, loadDecl, typedLoadOnDeviceDecl, typedLoadDecl]
  else
    #[shapeDecl, specDecl, tensorSpecDecl, loadFromHandleDecl, loadDecl,
      typedLoadFromHandleOnDeviceDecl, typedLoadFromHandleDecl, typedLoadOnDeviceDecl, typedLoadDecl]

private structure LeafInfo where
  declName : String
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
    : CommandElabM Unit := do
  let mut expected : Nat := 0
  for (idx, _) in children do
    if idx != expected then
      throwError
        "safetensors_type_provider: non-contiguous numeric path segment under '{pathLabel path}'. expected index {expected}, found {idx}."
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

private inductive FieldSource where
  | value
  | named (segment : String)
  | indexedCollection
  | indexedAt (idx : Nat)

private def indexFieldBase (idx : Nat) : String :=
  s!"i{idx}"

private inductive TypeRepr where
  | leaf (declName : String)
  | array (elem : TypeRepr) (count : Nat)
  | struct (typeName : String) (fields : Array (String × FieldSource × TypeRepr))

private partial def TypeRepr.typeExpr : TypeRepr → String
  | .leaf declName =>
      "_root_.torch.Tensor { shape := " ++ s!"{declName}Shape, dtype := {declName}Spec.dtype, " ++
      "device := _root_.torch.DevicePolicy.any }"
  | .array elem _ => s!"Array ({elem.typeExpr})"
  | .struct typeName _ => typeName

private def loaderNameForType (typeName : String) : String :=
  "load_node_" ++ sanitizeIdent (replaceInvalidChars typeName) "node"

private def handleLoaderNameForType (typeName : String) : String :=
  loaderNameForType typeName ++ "_fromHandle"

private partial def buildTypeRepr
    (node : PathNode)
    (path : List String)
    (usedTypeNames : List String)
    : CommandElabM (TypeRepr × Array String × List String) := do
  let hasLeaf := node.leaf?.isSome
  let named := namedChildren node
  let indexed := indexedChildren node

  if hasLeaf && named.isEmpty && indexed.isEmpty then
    let some leaf := node.leaf?
      | throwError "safetensors_type_provider internal error: missing leaf at '{pathLabel path}'"
    return (.leaf leaf.declName, #[], usedTypeNames)

  if !hasLeaf && named.isEmpty && !indexed.isEmpty then
    validateIndexedContiguous path indexed
    if indexedChildrenUniform indexed then
      let some (_, firstChild) := indexed[0]?
        | throwError "safetensors_type_provider internal error: missing first indexed child"
      let (elemRepr, elemDecls, used') ← buildTypeRepr firstChild (path ++ ["item"]) usedTypeNames
      return (.array elemRepr indexed.size, elemDecls, used')
    else
      let mut decls : Array String := #[]
      let mut used := usedTypeNames
      let mut fields : Array (String × FieldSource × TypeRepr) := #[]
      let mut usedFieldNames : List String := []
      for (idx, child) in indexed do
        let fieldName := freshName usedFieldNames (indexFieldBase idx)
        usedFieldNames := fieldName :: usedFieldNames
        let (childRepr, childDecls, used') ← buildTypeRepr child (path ++ [fieldName]) used
        used := used'
        decls := decls ++ childDecls
        fields := fields.push (fieldName, .indexedAt idx, childRepr)
      let typeName := freshName used (typeNameBase path)
      used := typeName :: used
      let fieldLines :=
        fields.map fun (fieldName, _, childRepr) => s!"  {fieldName} : {childRepr.typeExpr}"
      let fieldsBlock := if fieldLines.isEmpty then "" else String.intercalate "\n" fieldLines.toList ++ "\n"
      let structDecl :=
        s!"/-- Generated hierarchical checkpoint node `{typeName}`. -/\n" ++
        s!"structure {typeName} where\n" ++
        fieldsBlock
      decls := decls.push structDecl
      return (.struct typeName fields, decls, used)

  let mut decls : Array String := #[]
  let mut used := usedTypeNames
  let mut fields : Array (String × FieldSource × TypeRepr) := #[]
  let mut usedFieldNames : List String := []

  if let some leaf := node.leaf? then
    let fieldName := freshName usedFieldNames "value"
    usedFieldNames := fieldName :: usedFieldNames
    fields := fields.push (fieldName, .value, .leaf leaf.declName)

  for (seg, child) in named do
    let baseField := sanitizeIdent (replaceInvalidChars seg |>.map Char.toLower) "field"
    let fieldName := freshName usedFieldNames baseField
    usedFieldNames := fieldName :: usedFieldNames
    let (childRepr, childDecls, used') ← buildTypeRepr child (path ++ [fieldName]) used
    used := used'
    decls := decls ++ childDecls
    fields := fields.push (fieldName, .named seg, childRepr)

  if !indexed.isEmpty then
    validateIndexedContiguous (path ++ ["<index>"]) indexed
    if indexedChildrenUniform indexed then
      let fieldName := freshName usedFieldNames "items"
      usedFieldNames := fieldName :: usedFieldNames
      let some (_, firstChild) := indexed[0]?
        | throwError "safetensors_type_provider internal error: missing first indexed child for struct field"
      let (elemRepr, elemDecls, used') ← buildTypeRepr firstChild (path ++ [fieldName]) used
      used := used'
      decls := decls ++ elemDecls
      fields := fields.push (fieldName, .indexedCollection, .array elemRepr indexed.size)
    else
      for (idx, child) in indexed do
        let fieldName := freshName usedFieldNames (indexFieldBase idx)
        usedFieldNames := fieldName :: usedFieldNames
        let (childRepr, childDecls, used') ← buildTypeRepr child (path ++ [fieldName]) used
        used := used'
        decls := decls ++ childDecls
        fields := fields.push (fieldName, .indexedAt idx, childRepr)

  let typeName := freshName used (typeNameBase path)
  used := typeName :: used
  let fieldLines :=
    fields.map fun (fieldName, _, childRepr) => s!"  {fieldName} : {childRepr.typeExpr}"
  let fieldsBlock := if fieldLines.isEmpty then "" else String.intercalate "\n" fieldLines.toList ++ "\n"
  let structDecl :=
    s!"/-- Generated hierarchical checkpoint node `{typeName}`. -/\n" ++
    s!"structure {typeName} where\n" ++
    fieldsBlock
  decls := decls.push structDecl
  return (.struct typeName fields, decls, used)

private def childNodeForFieldSource
    (node : PathNode)
    (source : FieldSource)
    : CommandElabM PathNode := do
  match source with
  | .value => pure node
  | .named segment =>
      match node.named.findSome? (fun (k, c) => if k == segment then some c else none) with
      | some child => pure child
      | none =>
          throwError
            "safetensors_type_provider internal error: missing named child segment '{segment}' while building hierarchical checkpoint type."
  | .indexedCollection => pure node
  | .indexedAt idx =>
      match node.indexed.findSome? (fun (k, c) => if k == idx then some c else none) with
      | some child => pure child
      | none =>
          throwError
            "safetensors_type_provider internal error: missing indexed child '{idx}' while building hierarchical checkpoint type."

private partial def buildInlineLoadExpr
    (node : PathNode)
    (repr : TypeRepr)
    (sourceExpr : String)
    (hint : String)
    (useHandle : Bool)
    : CommandElabM String := do
  match repr with
  | .leaf _ =>
      let some leaf := node.leaf?
        | throwError "safetensors_type_provider internal error: missing leaf while building load expression"
      pure <| if useHandle then s!"load_{leaf.declName}FromHandle {sourceExpr}" else s!"load_{leaf.declName} {sourceExpr}"
  | .array elem expectedCount =>
      let indexed := indexedChildren node
      if indexed.size != expectedCount then
        throwError
          "safetensors_type_provider internal error: indexed collection size mismatch (expected {expectedCount}, got {indexed.size})."
      let mut lines : Array String := #["do"]
      let mut valueNames : Array String := #[]
      let mut i : Nat := 0
      for (idx, child) in indexed do
        let valueName := sanitizeIdent s!"{hint}_{idx}" s!"item_{i}"
        let childExpr ← buildInlineLoadExpr child elem sourceExpr valueName useHandle
        lines := lines.push s!"  let {valueName} ←"
        lines := lines.push (indentBlock 4 childExpr)
        valueNames := valueNames.push valueName
        i := i + 1
      let arrayLiteral := "#[" ++ String.intercalate ", " valueNames.toList ++ "]"
      lines := lines.push s!"  pure {arrayLiteral}"
      pure <| String.intercalate "\n" lines.toList
  | .struct typeName fields =>
      let mut lines : Array String := #["do"]
      let mut assignments : Array String := #[]
      let mut i : Nat := 0
      for (fieldName, source, childRepr) in fields do
        let childNode ← childNodeForFieldSource node source
        let valueName := sanitizeIdent s!"{hint}_{fieldName}" s!"field_{i}"
        let childExpr ← buildInlineLoadExpr childNode childRepr sourceExpr valueName useHandle
        lines := lines.push s!"  let {valueName} ←"
        lines := lines.push (indentBlock 4 childExpr)
        assignments := assignments.push s!"{fieldName} := {valueName}"
        i := i + 1
      let recordLiteral := "{" ++ String.intercalate ", " assignments.toList ++ "}"
      lines := lines.push s!"  pure ({recordLiteral} : {typeName})"
      pure <| String.intercalate "\n" lines.toList

private partial def buildLoadExpr
    (node : PathNode)
    (repr : TypeRepr)
    (sourceExpr : String)
    (hint : String)
    (useHandle : Bool)
    : CommandElabM String := do
  match repr with
  | .struct typeName _ =>
      pure <| if useHandle then s!"{handleLoaderNameForType typeName} {sourceExpr}" else s!"{loaderNameForType typeName} {sourceExpr}"
  | _ =>
      buildInlineLoadExpr node repr sourceExpr hint useHandle

private partial def buildStructLoaderDecls
    (node : PathNode)
    (repr : TypeRepr)
    (sourceIsDirectory : Bool)
    : CommandElabM (Array String) := do
  match repr with
  | .leaf _ => pure #[]
  | .array elem _ =>
      match (indexedChildren node)[0]? with
      | some (_, firstChild) => buildStructLoaderDecls firstChild elem sourceIsDirectory
      | none => pure #[]
  | .struct typeName fields =>
      let mut decls : Array String := #[]
      for (_, source, childRepr) in fields do
        let childNode ← childNodeForFieldSource node source
        decls := decls ++ (← buildStructLoaderDecls childNode childRepr sourceIsDirectory)
      if sourceIsDirectory then
        let mut lines : Array String := #["do"]
        let mut assignments : Array String := #[]
        let mut i : Nat := 0
        for (fieldName, source, childRepr) in fields do
          let childNode ← childNodeForFieldSource node source
          let valueName := sanitizeIdent s!"{typeName}_{fieldName}" s!"field_{i}"
          let childExpr ← buildLoadExpr childNode childRepr "source" valueName false
          lines := lines.push s!"  let {valueName} ←"
          lines := lines.push (indentBlock 4 childExpr)
          assignments := assignments.push s!"{fieldName} := {valueName}"
          i := i + 1
        let recordLiteral := "{" ++ String.intercalate ", " assignments.toList ++ "}"
        lines := lines.push s!"  pure ({recordLiteral} : {typeName})"
        let loaderDecl :=
          s!"/-- Load typed checkpoint node `{typeName}`. -/\n" ++
          s!"def {loaderNameForType typeName} (source : String := defaultSource) : IO {typeName} :=\n" ++
          indentBlock 2 (String.intercalate "\n" lines.toList) ++ "\n"
        pure (decls.push loaderDecl)
      else
        let mut lines : Array String := #["do"]
        let mut assignments : Array String := #[]
        let mut i : Nat := 0
        for (fieldName, source, childRepr) in fields do
          let childNode ← childNodeForFieldSource node source
          let valueName := sanitizeIdent s!"{typeName}_{fieldName}" s!"field_{i}"
          let childExpr ← buildLoadExpr childNode childRepr "handle" valueName true
          lines := lines.push s!"  let {valueName} ←"
          lines := lines.push (indentBlock 4 childExpr)
          assignments := assignments.push s!"{fieldName} := {valueName}"
          i := i + 1
        let recordLiteral := "{" ++ String.intercalate ", " assignments.toList ++ "}"
        lines := lines.push s!"  pure ({recordLiteral} : {typeName})"
        let fromHandleDecl :=
          s!"/-- Load typed checkpoint node `{typeName}` from an open SafeTensors handle. -/\n" ++
          s!"def {handleLoaderNameForType typeName} (handle : _root_.torch.safetensors.SafeTensorsHandle) : IO {typeName} :=\n" ++
          indentBlock 2 (String.intercalate "\n" lines.toList) ++ "\n"
        let loaderDecl :=
          s!"/-- Load typed checkpoint node `{typeName}`. -/\n" ++
          s!"def {loaderNameForType typeName} (source : String := defaultSource) : IO {typeName} := do\n" ++
          "  let handle ← _root_.torch.safetensors.openHandle source\n" ++
          s!"  {handleLoaderNameForType typeName} handle\n"
        pure (decls.push fromHandleDecl |>.push loaderDecl)

private partial def buildNamespaceDeclsForChild
    (nsName : String)
    (node : PathNode)
    (repr : TypeRepr)
    (sourceIsDirectory : Bool)
    : CommandElabM (Array String) := do
  match repr with
  | .leaf _ => pure #[]
  | .array _ _ =>
      if sourceIsDirectory then
        let loadExpr ← buildLoadExpr node repr "source" nsName false
        pure #[
          s!"namespace {nsName}\n",
          s!"abbrev Weights := {repr.typeExpr}\n",
          "/-- Load this checkpoint subtree. -/\n" ++
            "def load (source : String := defaultSource) : IO Weights :=\n" ++
            indentBlock 2 loadExpr ++ "\n",
          s!"end {nsName}\n"
        ]
      else
        let loadFromHandleExpr ← buildLoadExpr node repr "handle" nsName true
        pure #[
          s!"namespace {nsName}\n",
          s!"abbrev Weights := {repr.typeExpr}\n",
          "/-- Load this checkpoint subtree from an open SafeTensors handle. -/\n" ++
            "def loadFromHandle (handle : _root_.torch.safetensors.SafeTensorsHandle) : IO Weights :=\n" ++
            indentBlock 2 loadFromHandleExpr ++ "\n",
          "/-- Load this checkpoint subtree. -/\n" ++
            "def load (source : String := defaultSource) : IO Weights := do\n" ++
            "  let handle ← _root_.torch.safetensors.openHandle source\n" ++
            "  loadFromHandle handle\n",
          s!"end {nsName}\n"
        ]
  | .struct _ fields =>
      let decls0 : Array String ←
        if sourceIsDirectory then do
          let loadExpr ← buildLoadExpr node repr "source" nsName false
          pure #[
            s!"namespace {nsName}\n",
            s!"abbrev Weights := {repr.typeExpr}\n",
            "/-- Load this checkpoint subtree. -/\n" ++
              "def load (source : String := defaultSource) : IO Weights :=\n" ++
              indentBlock 2 loadExpr ++ "\n"
          ]
        else do
          let loadFromHandleExpr ← buildLoadExpr node repr "handle" nsName true
          pure #[
            s!"namespace {nsName}\n",
            s!"abbrev Weights := {repr.typeExpr}\n",
            "/-- Load this checkpoint subtree from an open SafeTensors handle. -/\n" ++
              "def loadFromHandle (handle : _root_.torch.safetensors.SafeTensorsHandle) : IO Weights :=\n" ++
              indentBlock 2 loadFromHandleExpr ++ "\n",
            "/-- Load this checkpoint subtree. -/\n" ++
              "def load (source : String := defaultSource) : IO Weights := do\n" ++
              "  let handle ← _root_.torch.safetensors.openHandle source\n" ++
              "  loadFromHandle handle\n"
          ]
      let mut decls : Array String := decls0
      for (fieldName, source, childRepr) in fields do
        let childNode ← childNodeForFieldSource node source
        decls := decls ++ (← buildNamespaceDeclsForChild fieldName childNode childRepr sourceIsDirectory)
      decls := decls.push s!"end {nsName}\n"
      pure decls

private def buildRootNamespaceDecls
    (node : PathNode)
    (repr : TypeRepr)
    (sourceIsDirectory : Bool)
    : CommandElabM (Array String) := do
  match repr with
  | .struct _ fields =>
      let mut decls : Array String := #[]
      for (fieldName, source, childRepr) in fields do
        let childNode ← childNodeForFieldSource node source
        decls := decls ++ (← buildNamespaceDeclsForChild fieldName childNode childRepr sourceIsDirectory)
      pure decls
  | _ => pure #[]

private def elabSafeTensorsTypeProviderCore
    (source : String)
    (ns : Name)
    : CommandElabM Unit := do
      let discovered ←
        try
          liftIO <| introspect source
        catch _ =>
          throwError "safetensors_type_provider failed while introspecting source '{source}'."

      let namespaceName := ns.toString
      elabGeneratedCommand s!"namespace {namespaceName}"
      try
        elabGeneratedCommand <|
          s!"/-- Provider default source path (file or directory). -/\n" ++
          s!"def defaultSource : String := {leanStringLit discovered.source}\n"
        elabGeneratedCommand <|
          s!"/-- True when `defaultSource` is a directory containing shard files. -/\n" ++
          s!"def sourceIsDirectory : Bool := {if discovered.sourceIsDirectory then "true" else "false"}\n"

        let mut usedDeclNames : List String := []
        let mut specNames : Array String := #[]
        let mut tensorFieldPairs : Array (String × String) := #[]
        let mut root : PathNode := emptyPathNode

        for tensor in discovered.tensors do
          let baseName := tensorDeclBaseName tensor.name
          let declName := freshName usedDeclNames baseName
          usedDeclNames := declName :: usedDeclNames
          specNames := specNames.push s!"{declName}Spec"
          tensorFieldPairs := tensorFieldPairs.push (declName, tensor.name)
          for decl in renderTensorDecls discovered.sourceIsDirectory tensor declName do
            elabGeneratedCommand decl

          let segments := (tensor.name.splitOn ".").filter (fun s => !s.isEmpty)
          if segments.isEmpty then
            throwError "safetensors_type_provider found invalid empty tensor path."
          let leaf : LeafInfo := {
            declName := declName
            tensorName := tensor.name
            dtype := tensor.dtype
            shape := tensor.shape
          }
          match insertPath root segments leaf with
          | .ok root' => root := root'
          | .error err =>
              throwError "safetensors_type_provider failed while building hierarchy: {err}"

        if tensorFieldPairs.isEmpty then
          elabGeneratedCommand <|
            "/-- Typed checkpoint record generated by `safetensors_type_provider`. -/\n" ++
            "abbrev Weights := Unit\n"
          elabGeneratedCommand <|
            "/-- Load all tensors into a typed checkpoint record. -/\n" ++
            "def loadAll (source : String := defaultSource) : IO Weights :=\n" ++
            "  pure ()\n"
        else
          let (weightsRepr, weightsTypeDecls, _) ← buildTypeRepr root [] []
          for decl in weightsTypeDecls do
            elabGeneratedCommand decl
          for decl in (← buildStructLoaderDecls root weightsRepr discovered.sourceIsDirectory) do
            elabGeneratedCommand decl
          for decl in (← buildRootNamespaceDecls root weightsRepr discovered.sourceIsDirectory) do
            elabGeneratedCommand decl
          if weightsRepr.typeExpr != "Weights" then
            elabGeneratedCommand s!"abbrev Weights := {weightsRepr.typeExpr}\n"
          if discovered.sourceIsDirectory then
            let loadExpr ← buildLoadExpr root weightsRepr "source" "weights" false
            let loadAllDecl :=
              "/-- Load all tensors into a typed checkpoint record. -/\n" ++
              "def loadAll (source : String := defaultSource) : IO Weights :=\n" ++
              indentBlock 2 loadExpr ++ "\n"
            elabGeneratedCommand loadAllDecl
          else
            let loadFromHandleExpr ← buildLoadExpr root weightsRepr "handle" "weights" true
            let loadAllFromHandleDecl :=
              "/-- Load all tensors into a typed checkpoint record from an open SafeTensors handle. -/\n" ++
              "def loadAllFromHandle (handle : _root_.torch.safetensors.SafeTensorsHandle) : IO Weights :=\n" ++
              indentBlock 2 loadFromHandleExpr ++ "\n"
            elabGeneratedCommand loadAllFromHandleDecl
            let loadAllDecl :=
              "/-- Load all tensors into a typed checkpoint record. -/\n" ++
              "def loadAll (source : String := defaultSource) : IO Weights := do\n" ++
              "  let handle ← _root_.torch.safetensors.openHandle source\n" ++
              "  loadAllFromHandle handle\n"
            elabGeneratedCommand loadAllDecl

        let fieldPairsLit := tensorFieldPairs.map fun (fieldName, tensorName) =>
          s!"({leanStringLit fieldName}, {leanStringLit tensorName})"
        elabGeneratedCommand <|
          "/-- Mapping from generated flat tensor identifiers to original tensor names. -/\n" ++
          "def fieldToTensorName : List (String × String) :=\n" ++
          "  [" ++ String.intercalate ", " fieldPairsLit.toList ++ "]\n"

        let schemaArray :=
          if specNames.isEmpty then
            "#[]"
          else
            "#[" ++ String.intercalate ", " specNames.toList ++ "]"
        elabGeneratedCommand <|
          s!"/-- All tensors discovered by the provider. -/\n" ++
          s!"def schema : Array _root_.torch.safetensors.TensorSchema := {schemaArray}\n"
        elabGeneratedCommand <|
          s!"/-- Number of discovered tensors. -/\n" ++
          s!"def tensorCount : Nat := {discovered.tensors.size}\n"
        elabGeneratedCommand <|
          "/-- Check whether a tensor exists in the generated schema. -/\n" ++
          "def hasTensor (tensorName : String) : Bool :=\n" ++
          "  schema.any (fun t => t.name == tensorName)\n"
        elabGeneratedCommand <|
          "/-- Look up tensor schema by tensor name. -/\n" ++
          "def find? (tensorName : String) : Option _root_.torch.safetensors.TensorSchema :=\n" ++
          "  schema.findSome? (fun t => if t.name == tensorName then some t else none)\n"
      finally
        elabGeneratedCommand s!"end {namespaceName}"

@[command_elab safetensorsTypeProviderCmd]
def elabSafeTensorsTypeProvider : CommandElab
  | `(safetensors_type_provider $sourcePath:str as $ns:ident) =>
      elabSafeTensorsTypeProviderCore sourcePath.getString ns.getId
  | _ => throwUnsupportedSyntax

end torch.safetensors
