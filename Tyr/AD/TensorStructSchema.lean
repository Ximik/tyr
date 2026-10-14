import Lean
import Tyr.TensorStruct

/-!
# Tyr.AD.TensorStructSchema

Schema-oriented structural metadata for `TensorStruct` values.

This layer is intentionally separate from `TensorStruct.map/fold`: it records
stable leaf ordering, field/index paths, and AD participation roles, and it
provides a schema-aware flatten/rebuild boundary for structured AD frontends.
-/

namespace torch

open Lean

/-- Path to a structural leaf inside a `TensorStruct` value. -/
structure TensorLeafPath where
  segments : Array String := #[]
  deriving Repr, BEq, Inhabited, DecidableEq

namespace TensorLeafPath

def push (path : TensorLeafPath) (segment : String) : TensorLeafPath :=
  { segments := path.segments.push segment }

def pushIndex (path : TensorLeafPath) (idx : Nat) : TensorLeafPath :=
  path.push s!"[{idx}]"

def toString (path : TensorLeafPath) : String :=
  path.segments.foldl (init := "") fun acc segment =>
    if acc.isEmpty then
      segment
    else if segment.startsWith "[" then
      acc ++ segment
    else
      acc ++ "." ++ segment

instance : ToString TensorLeafPath := ⟨TensorLeafPath.toString⟩

end TensorLeafPath

private def renderPath (path : TensorLeafPath) : String :=
  let rendered := toString path
  if rendered.isEmpty then "<root>" else rendered

private def shapeToString (s : Shape) : String :=
  "[" ++ String.intercalate ", " (s.toList.map toString) ++ "]"

private def shapeToNatArray (s : Shape) : Array Nat :=
  s.map UInt64.toNat

/-- AD participation role for one structural leaf. -/
inductive TensorLeafRole where
  | diff
  | static
  | frozen
  deriving Repr, BEq, Inhabited, DecidableEq

/-- One structural leaf in a `TensorStruct` schema. -/
structure TensorLeafSpec where
  path : TensorLeafPath := {}
  role : TensorLeafRole := .diff
  shape : Option (Array Nat) := none
  dtype : Option String := none
  deriving Repr, BEq, Inhabited, DecidableEq

/-- Structural schema for a `TensorStruct` value. -/
structure TensorStructSchema where
  typeName : Name := Name.anonymous
  leaves : Array TensorLeafSpec := #[]
  deriving Repr, Inhabited

namespace TensorStructSchema

def leafPaths (schema : TensorStructSchema) : Array TensorLeafPath :=
  schema.leaves.map (·.path)

def renderedLeafPaths (schema : TensorStructSchema) : Array String :=
  schema.leafPaths.map (fun path => toString path)

/-- Render a human-readable summary of the model structure. -/
def summary (schema : TensorStructSchema) : String :=
  let header := s!"Model Structure: {schema.typeName}\n"
  let line := "--------------------------------------------------\n"
  let leaves := schema.leaves.foldl (init := "") fun acc leaf =>
    let shapeStr := match leaf.shape with
      | some s => s!"{s}"
      | none => "static"
    acc ++ s!"{renderPath leaf.path} : {shapeStr} ({reprStr leaf.role})\n"
  header ++ line ++ leaves ++ line

end TensorStructSchema

/--
Controls which tensor leaves participate in flatten/rebuild operations.

- `.diffOnly` keeps only differentiable leaves.
- `.diffAndFrozen` keeps differentiable and frozen tensor leaves.
-/
inductive TensorLeafSelection where
  | diffOnly
  | diffAndFrozen
  deriving Repr, BEq, Inhabited, DecidableEq

namespace TensorLeafSelection

def includesRole : TensorLeafSelection → TensorLeafRole → Bool
  | .diffOnly, .diff => true
  | .diffOnly, _ => false
  | .diffAndFrozen, .diff => true
  | .diffAndFrozen, .frozen => true
  | .diffAndFrozen, .static => false

end TensorLeafSelection

/-- Runtime tensor payload for one flattened structured leaf. -/
structure TensorLeafValue where
  role : TensorLeafRole := .diff
  payload : Sigma Tensor

instance : Repr TensorLeafValue where
  reprPrec leaf _ := s!"TensorLeafValue({reprStr leaf.role}, {reprStr leaf.payload.1})"

namespace TensorLeafValue

def ofTensor {σ : StaticSpec} (role : TensorLeafRole) (t : Tensor σ) : TensorLeafValue :=
  { role := role, payload := ⟨σ, t⟩ }

def typedShape (leaf : TensorLeafValue) : Shape :=
  leaf.payload.1.shape

def shape (leaf : TensorLeafValue) : Array Nat :=
  shapeToNatArray leaf.typedShape

def dtype (leaf : TensorLeafValue) : DType :=
  leaf.payload.2.dtype

end TensorLeafValue

/--
Schema-oriented structural view of a value. Unlike `TensorStruct`, this is
allowed to depend on the runtime shape-of-structure for containers such as
`Array`, `List`, and `Option`.
-/
class ToTensorStructSchema (α : Type) where
  typeName : Name := Name.anonymous
  describeLeaves : α → TensorLeafPath → Array TensorLeafSpec

namespace ToTensorStructSchema

def schema [ToTensorStructSchema α] (x : α) : TensorStructSchema :=
  { typeName := ToTensorStructSchema.typeName (α := α)
    leaves := ToTensorStructSchema.describeLeaves x {} }

/-- Render a human-readable summary of a value's structure. -/
def summary [ToTensorStructSchema α] (x : α) : String :=
  (schema x).summary

end ToTensorStructSchema

/--
Schema-aware flatten/rebuild support for structured AD boundaries.

`rebuildAt` uses an existing value as a structural template. This preserves
static fields and dynamic container shape while replacing only the selected
tensor leaves.
-/
class TensorStructFlatten (α : Type) where
  flattenAt : α → TensorLeafPath → TensorLeafSelection → Array TensorLeafValue
  rebuildAt :
    α →
    TensorLeafPath →
    TensorLeafSelection →
    Array TensorLeafValue →
    StateT Nat (Except String) α

namespace TensorStructFlatten

def flatten [TensorStructFlatten α]
    (x : α)
    (selection : TensorLeafSelection := .diffAndFrozen) :
    Array TensorLeafValue :=
  TensorStructFlatten.flattenAt x {} selection

def count [TensorStructFlatten α]
    (x : α)
    (selection : TensorLeafSelection := .diffAndFrozen) :
    Nat :=
  (flatten x selection).size

def rebuildFrom [TensorStructFlatten α]
    (template : α)
    (leaves : Array TensorLeafValue)
    (selection : TensorLeafSelection := .diffAndFrozen) :
    Except String α := do
  let (rebuilt, nextIdx) ← (TensorStructFlatten.rebuildAt template {} selection leaves).run 0
  if nextIdx = leaves.size then
    pure rebuilt
  else
    throw <| s!"Unexpected extra tensor leaves after rebuilding {renderPath {}}: " ++
      s!"consumed {nextIdx} leaves but received {leaves.size}."

end TensorStructFlatten

private def takeLeaf
    (path : TensorLeafPath)
    (leaves : Array TensorLeafValue) :
    StateT Nat (Except String) TensorLeafValue := do
  let idx ← get
  match leaves[idx]? with
  | some leaf =>
      set (idx + 1)
      pure leaf
  | none =>
      throw <| s!"Missing tensor leaf for {renderPath path}: " ++
        s!"needed leaf #{idx}, but only {leaves.size} were provided."

private def expectLeaf
    {σ : StaticSpec}
    (expectedRole : TensorLeafRole)
    (path : TensorLeafPath)
    (leaf : TensorLeafValue) :
    Except String (Tensor σ) := do
  if leaf.role != expectedRole then
    throw <| s!"Tensor leaf role mismatch at {renderPath path}: " ++
      s!"expected {reprStr expectedRole}, got {reprStr leaf.role}."
  let ⟨leafSpec, leafTensor⟩ := leaf.payload
  if leafSpec.shape != σ.shape then
    throw <| s!"Tensor leaf shape mismatch at {renderPath path}: " ++
      s!"expected {shapeToString σ.shape}, got {shapeToString leafSpec.shape}."
  -- Validate both the incoming claim and the destination contract. This also
  -- catches malformed handles constructed through unsafe native adapters.
  match leafTensor.validate *> Tensor.ofRaw (σ := σ) leafTensor.raw with
  | .ok tensor => pure tensor
  | .error message => throw s!"Tensor leaf mismatch at {renderPath path}: {message}"

private def knownDType : DType → Option String
  | .Unknown _ => none
  | dtype => some dtype.canonicalName

instance {σ : StaticSpec} : ToTensorStructSchema (Tensor σ) where
  describeLeaves _ path := #[{
    path, role := .diff, shape := some (shapeToNatArray σ.shape), dtype := knownDType σ.dtype
  }]

instance {σ : StaticSpec} : TensorStructFlatten (Tensor σ) where
  flattenAt tensor _ selection :=
    if selection.includesRole .diff then #[TensorLeafValue.ofTensor .diff tensor] else #[]
  rebuildAt _ path selection leaves := do
    if selection.includesRole .diff then
      let leaf ← takeLeaf path leaves
      match expectLeaf .diff path leaf with
      | .ok tensor => pure tensor
      | .error message => throw message
    else
      throw s!"Internal error: selection {reprStr selection} excluded diff leaf at {renderPath path}."

instance {dtype : DType} : ToTensorStructSchema (ShapePackedTensor dtype) where
  describeLeaves tensor path := ToTensorStructSchema.describeLeaves tensor.2 path

instance {dtype : DType} : TensorStructFlatten (ShapePackedTensor dtype) where
  flattenAt tensor path selection := TensorStructFlatten.flattenAt tensor.2 path selection
  rebuildAt template path selection leaves := do
    let value ← TensorStructFlatten.rebuildAt template.2 path selection leaves
    pure ⟨template.1, value⟩

private def describeListLeaves
    [ToTensorStructSchema α]
    (xs : List α)
    (path : TensorLeafPath)
    (idx : Nat := 0) :
    Array TensorLeafSpec :=
  match xs with
  | [] => #[]
  | x :: rest =>
    ToTensorStructSchema.describeLeaves x (path.pushIndex idx) ++
      describeListLeaves rest path (idx + 1)

instance {σ : StaticSpec} : ToTensorStructSchema (Frozen σ) where
  typeName := Name.anonymous
  describeLeaves _ path := #[{
    path := path
    role := .frozen
    shape := some (shapeToNatArray σ.shape)
    dtype := knownDType σ.dtype
  }]

instance {σ : StaticSpec} : TensorStructFlatten (Frozen σ) where
  flattenAt fr _ selection :=
    if TensorLeafSelection.includesRole selection .frozen then
      #[TensorLeafValue.ofTensor .frozen fr.tensor]
    else
      #[]
  rebuildAt fr path selection leaves := do
    if TensorLeafSelection.includesRole selection .frozen then
      let leaf ← takeLeaf path leaves
      match expectLeaf (σ := σ) .frozen path leaf with
      | .ok t => pure { tensor := t }
      | .error err => throw err
    else
      pure fr

instance {α : Type} : ToTensorStructSchema (Static α) where
  typeName := Name.anonymous
  describeLeaves _ path := #[{
    path := path
    role := .static
    shape := none
  }]

instance {α : Type} : TensorStructFlatten (Static α) where
  flattenAt _ _ _ := #[]
  rebuildAt x _ _ _ := pure x

instance [ToTensorStructSchema α] [ToTensorStructSchema β] : ToTensorStructSchema (α × β) where
  typeName := Name.anonymous
  describeLeaves pair path :=
    ToTensorStructSchema.describeLeaves pair.1 (path.push "fst") ++
      ToTensorStructSchema.describeLeaves pair.2 (path.push "snd")

instance [TensorStructFlatten α] [TensorStructFlatten β] : TensorStructFlatten (α × β) where
  flattenAt pair path selection :=
    TensorStructFlatten.flattenAt pair.1 (path.push "fst") selection ++
      TensorStructFlatten.flattenAt pair.2 (path.push "snd") selection
  rebuildAt pair path selection leaves := do
    let fst' ← TensorStructFlatten.rebuildAt pair.1 (path.push "fst") selection leaves
    let snd' ← TensorStructFlatten.rebuildAt pair.2 (path.push "snd") selection leaves
    pure (fst', snd')

instance {α : Type} [ToTensorStructSchema α] : ToTensorStructSchema (Array α) where
  typeName := Name.anonymous
  describeLeaves xs path := Id.run do
    let mut out : Array TensorLeafSpec := #[]
    for h : i in [:xs.size] do
      let x := xs[i]
      out := out ++ ToTensorStructSchema.describeLeaves x (path.pushIndex i)
    return out

instance {α : Type} [TensorStructFlatten α] : TensorStructFlatten (Array α) where
  flattenAt xs path selection := Id.run do
    let mut out : Array TensorLeafValue := #[]
    for h : i in [:xs.size] do
      let x := xs[i]
      out := out ++ TensorStructFlatten.flattenAt x (path.pushIndex i) selection
    return out
  rebuildAt xs path selection leaves := do
    let mut out : Array α := #[]
    for h : i in [:xs.size] do
      let x := xs[i]
      let x' ← TensorStructFlatten.rebuildAt x (path.pushIndex i) selection leaves
      out := out.push x'
    pure out

instance {α : Type} [ToTensorStructSchema α] : ToTensorStructSchema (List α) where
  typeName := Name.anonymous
  describeLeaves xs path := describeListLeaves xs path

private def rebuildListLeaves
    [TensorStructFlatten α]
    (xs : List α)
    (path : TensorLeafPath)
    (selection : TensorLeafSelection)
    (leaves : Array TensorLeafValue)
    (idx : Nat := 0) :
    StateT Nat (Except String) (List α) := do
  match xs with
  | [] => pure []
  | x :: rest =>
      let x' ← TensorStructFlatten.rebuildAt x (path.pushIndex idx) selection leaves
      let rest' ← rebuildListLeaves rest path selection leaves (idx + 1)
      pure (x' :: rest')

private def rebuildVectorLeaves
    {n : Nat}
    [TensorStructFlatten α]
    (xs : Vector n α)
    (path : TensorLeafPath)
    (selection : TensorLeafSelection)
    (leaves : Array TensorLeafValue)
    (acc : Array α)
    (idx : Nat)
    (hacc : acc.size = idx)
    (hbound : idx ≤ n) :
    StateT Nat (Except String) { arr : Array α // arr.size = n } := do
  if h : idx < n then
    let x := xs.data[idx]'(by rw [xs.size_eq]; exact h)
    let x' ← TensorStructFlatten.rebuildAt x (path.pushIndex idx) selection leaves
    let acc' := acc.push x'
    have hacc' : acc'.size = idx + 1 := by
      simp [acc', hacc]
    rebuildVectorLeaves xs path selection leaves acc' (idx + 1) hacc' h
  else
    have hEq : idx = n := Nat.le_antisymm hbound (Nat.ge_of_not_lt h)
    pure ⟨acc, by rw [hacc, hEq]⟩

instance {α : Type} [TensorStructFlatten α] : TensorStructFlatten (List α) where
  flattenAt xs path selection := Id.run do
    let mut out : Array TensorLeafValue := #[]
    let mut idx := 0
    for x in xs do
      out := out ++ TensorStructFlatten.flattenAt x (path.pushIndex idx) selection
      idx := idx + 1
    return out
  rebuildAt xs path selection leaves :=
    rebuildListLeaves xs path selection leaves

instance {n : Nat} {α : Type} [ToTensorStructSchema α] : ToTensorStructSchema (Vector n α) where
  typeName := Name.anonymous
  describeLeaves xs path := describeListLeaves xs.data.toList path

instance {n : Nat} {α : Type} [TensorStructFlatten α] : TensorStructFlatten (Vector n α) where
  flattenAt xs path selection :=
    TensorStructFlatten.flattenAt xs.data path selection
  rebuildAt xs path selection leaves := do
    let ⟨data', hsize⟩ ← rebuildVectorLeaves xs path selection leaves #[] 0 rfl (Nat.zero_le n)
    pure ⟨data', hsize⟩

instance {α : Type} [ToTensorStructSchema α] : ToTensorStructSchema (Option α) where
  typeName := Name.anonymous
  describeLeaves x path :=
    match x with
    | some y => ToTensorStructSchema.describeLeaves y path
    | none => #[]

instance {α : Type} [TensorStructFlatten α] : TensorStructFlatten (Option α) where
  flattenAt x path selection :=
    match x with
    | some y => TensorStructFlatten.flattenAt y path selection
    | none => #[]
  rebuildAt x path selection leaves :=
    match x with
    | some y => do
        let y' ← TensorStructFlatten.rebuildAt y path selection leaves
        pure (some y')
    | none => pure none

instance : ToTensorStructSchema Bool where
  typeName := Name.anonymous
  describeLeaves _ path := #[{ path := path, role := .static }]

instance : TensorStructFlatten Bool where
  flattenAt _ _ _ := #[]
  rebuildAt x _ _ _ := pure x

instance : ToTensorStructSchema Float where
  typeName := Name.anonymous
  describeLeaves _ path := #[{ path := path, role := .static }]

instance : TensorStructFlatten Float where
  flattenAt _ _ _ := #[]
  rebuildAt x _ _ _ := pure x

instance : ToTensorStructSchema UInt8 where
  typeName := Name.anonymous
  describeLeaves _ path := #[{ path := path, role := .static }]

instance : TensorStructFlatten UInt8 where
  flattenAt _ _ _ := #[]
  rebuildAt x _ _ _ := pure x

instance : ToTensorStructSchema UInt64 where
  typeName := Name.anonymous
  describeLeaves _ path := #[{ path := path, role := .static }]

instance : TensorStructFlatten UInt64 where
  flattenAt _ _ _ := #[]
  rebuildAt x _ _ _ := pure x

instance : ToTensorStructSchema Nat where
  typeName := Name.anonymous
  describeLeaves _ path := #[{ path := path, role := .static }]

instance : TensorStructFlatten Nat where
  flattenAt _ _ _ := #[]
  rebuildAt x _ _ _ := pure x

instance : ToTensorStructSchema Int where
  typeName := Name.anonymous
  describeLeaves _ path := #[{ path := path, role := .static }]

instance : TensorStructFlatten Int where
  flattenAt _ _ _ := #[]
  rebuildAt x _ _ _ := pure x

instance : ToTensorStructSchema String where
  typeName := Name.anonymous
  describeLeaves _ path := #[{ path := path, role := .static }]

instance : TensorStructFlatten String where
  flattenAt _ _ _ := #[]
  rebuildAt x _ _ _ := pure x

private def leafSegmentOfName : Name → String
  | .anonymous => "_"
  | .str _ s => s
  | .num _ n => toString n

open Lean Elab Command Term Meta in
private def mkTensorStructSchemaInstanceCmd (typeName : Name) : CommandElabM Unit := do
  let env ← getEnv

  let some structInfo := getStructureInfo? env typeName
    | throwError "{typeName} is not a structure"

  let some (.inductInfo indInfo) := env.find? typeName
    | throwError "{typeName} is not an inductive type"

  let fieldNames := structInfo.fieldNames
  let fieldLeaves : Array (TSyntax `term) ← fieldNames.mapM fun fname => do
    let fieldLabel := leafSegmentOfName fname
    `(term|
      ToTensorStructSchema.describeLeaves x.$(mkIdent fname)
        (TensorLeafPath.push path $(Lean.quote fieldLabel)))

  let mut leavesExpr : TSyntax `term := ← `(term| (#[] : Array TensorLeafSpec))
  for fieldLeaf in fieldLeaves do
    leavesExpr := ← `(term| $leavesExpr ++ $fieldLeaf)

  let numParams := indInfo.numParams

  if numParams == 0 then
    let typeIdent := mkIdent typeName
    let instCmd ← `(command|
      instance : ToTensorStructSchema $typeIdent where
        typeName := $(Lean.quote typeName)
        describeLeaves x path := $leavesExpr
    )
    elabCommand instCmd
  else
    let instCmd ← liftTermElabM do
      Meta.forallTelescopeReducing indInfo.type fun params _ => do
        let paramBinders := params[:numParams]

        let binderStxs ← paramBinders.toArray.mapM fun param => do
          let paramDecl ← param.fvarId!.getDecl
          let paramName := paramDecl.userName
          let paramType ← Meta.inferType param
          let paramTypeStx ← PrettyPrinter.delab paramType
          `(bracketedBinder| {$(mkIdent paramName) : $paramTypeStx})

        let paramIdents ← paramBinders.toArray.mapM fun param => do
          let paramDecl ← param.fvarId!.getDecl
          pure (mkIdent paramDecl.userName)

        let appliedType ← `($(mkIdent typeName) $paramIdents*)

        `(command|
          instance $[$binderStxs]* : ToTensorStructSchema $appliedType where
            typeName := $(Lean.quote typeName)
            describeLeaves x path := $leavesExpr
        )

    elabCommand instCmd

open Lean Elab Command Deriving in
def mkToTensorStructSchemaHandler (typeNames : Array Name) : CommandElabM Bool := do
  if typeNames.isEmpty then
    return false
  for typeName in typeNames do
    mkTensorStructSchemaInstanceCmd typeName
  return true

open Lean Elab Command Term Meta in
private def mkTensorStructFlattenInstanceCmd (typeName : Name) : CommandElabM Unit := do
  let env ← getEnv

  let some structInfo := getStructureInfo? env typeName
    | throwError "{typeName} is not a structure"

  let some (.inductInfo indInfo) := env.find? typeName
    | throwError "{typeName} is not an inductive type"

  let fieldNames := structInfo.fieldNames

  let flattenFields : Array (TSyntax `term) ← fieldNames.mapM fun fname => do
    let fieldLabel := leafSegmentOfName fname
    `(term|
      TensorStructFlatten.flattenAt x.$(mkIdent fname)
        (TensorLeafPath.push path $(Lean.quote fieldLabel))
        selection)

  let mut flattenExpr : TSyntax `term := ← `(term| (#[] : Array TensorLeafValue))
  for fieldFlat in flattenFields do
    flattenExpr := ← `(term| $flattenExpr ++ $fieldFlat)

  let rebuildBinds ← fieldNames.mapM fun fname => do
    let fnameId := mkIdent fname
    let fieldLabel := leafSegmentOfName fname
    `(doElem|
      let $fnameId ← TensorStructFlatten.rebuildAt x.$fnameId
        (TensorLeafPath.push path $(Lean.quote fieldLabel))
        selection
        leaves)

  let rebuildFields ← fieldNames.mapM fun fname => do
    let fnameId := mkIdent fname
    `(Lean.Parser.Term.structInstField| $fnameId:ident := $fnameId)

  let numParams := indInfo.numParams

  if numParams == 0 then
    let typeIdent := mkIdent typeName
    let instCmd ← `(command|
      instance : TensorStructFlatten $typeIdent where
        flattenAt x path selection := $flattenExpr
        rebuildAt x path selection leaves := do
          $[$rebuildBinds:doElem]*
          pure { $[$rebuildFields:structInstField],* }
    )
    elabCommand instCmd
  else
    let instCmd ← liftTermElabM do
      Meta.forallTelescopeReducing indInfo.type fun params _ => do
        let paramBinders := params[:numParams]

        let binderStxs ← paramBinders.toArray.mapM fun param => do
          let paramDecl ← param.fvarId!.getDecl
          let paramName := paramDecl.userName
          let paramType ← Meta.inferType param
          let paramTypeStx ← PrettyPrinter.delab paramType
          `(bracketedBinder| {$(mkIdent paramName) : $paramTypeStx})

        let paramIdents ← paramBinders.toArray.mapM fun param => do
          let paramDecl ← param.fvarId!.getDecl
          pure (mkIdent paramDecl.userName)

        let appliedType ← `($(mkIdent typeName) $paramIdents*)

        `(command|
          instance $[$binderStxs]* : TensorStructFlatten $appliedType where
            flattenAt x path selection := $flattenExpr
            rebuildAt x path selection leaves := do
              $[$rebuildBinds:doElem]*
              pure { $[$rebuildFields:structInstField],* }
        )

    elabCommand instCmd

open Lean Elab Command Deriving in
def mkTensorStructFlattenHandler (typeNames : Array Name) : CommandElabM Bool := do
  if typeNames.isEmpty then
    return false
  for typeName in typeNames do
    mkTensorStructFlattenInstanceCmd typeName
  return true

open Lean Elab Deriving in
initialize
  registerDerivingHandler ``ToTensorStructSchema mkToTensorStructSchemaHandler
  registerDerivingHandler ``TensorStructFlatten mkTensorStructFlattenHandler

end torch
