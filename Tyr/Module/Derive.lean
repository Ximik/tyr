/-
  Tyr/Module/Derive.lean

  Macro-based approach for deriving TensorStruct instances.
  Provides `derive_tensor_struct` command for simple structures.
-/
import Lean
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.TensorStruct.DeriveCast
import Tyr.AD.TensorStructSchema

open Lean Elab Command Term Meta

namespace torch

/-! ## TensorStruct Instance Generation

This module provides an automatic `deriving` handler for `torch.TensorStruct` on
structure types.

What it handles well:

- field-wise traversal for structures,
- parametric structures (it generates binders for type/shape parameters),
- recursive use of existing `TensorStruct` instances on fields,
- capability-aware `TensorStructFor P` traversal when every field supports `P`,
- explicit conversions between parameterizations of the same record through
  derived `TensorTreeCast` instances.

Current limits:

- it is not a fully customizable deriving framework,
- it assumes each field already has a suitable `TensorStruct` instance,
- non-structure or highly custom traversal policies still require manual instances.

For a record that already has a custom `TensorStruct` instance,
`derive_tensor_struct_for RecordName` generates only the capability-aware
field traversal. It does not replace the existing structural instance.

### Field Type Handling

When generating a TensorStruct instance, each field is handled based on its type:

- `Tensor σ` → delegate to the checked tensor leaf instance
- `Static α` → pass through: `field`
- `Frozen σ` → delegate to the indexed frozen leaf instance
- `Array α` → recursive: `field.map (TensorStruct.map f)`
- `Option α` → recursive: `field.map (TensorStruct.map f)`
- Nested `TensorStruct` → recursive: `TensorStruct.map f field`

### Example Usage

```lean
-- Define your structure
structure MyLayer where
  weight : Tensor { shape := #[n, m], dtype := .Float32, device := .CPU }
  bias : Tensor { shape := #[n], dtype := .Float32, device := .CPU }
  config : Static Config

-- Manually create instance (the macro would generate this)
instance : TensorStruct MyLayer where
  map f l := { weight := TensorStruct.map f l.weight, bias := TensorStruct.map f l.bias, config := l.config }
  mapM f l := do pure { weight := ← TensorStruct.mapM f l.weight, bias := ← TensorStruct.mapM f l.bias, config := l.config }
  zipWith f l1 l2 := { weight := TensorStruct.zipWith f l1.weight l2.weight, bias := TensorStruct.zipWith f l1.bias l2.bias, config := l1.config }
  fold f init l := TensorStruct.fold f (TensorStruct.fold f init l.bias) l.weight
```
-/

/-! ## Helper Functions for Manual Instances -/

/-- Helper: map a function over a tensor field -/
@[inline] def mapTensor {σ : StaticSpec}
    (f : ∀ {σ}, Tensor σ → Tensor σ) (t : Tensor σ) : Tensor σ := f t

/-- Helper: map a function over a frozen field -/
@[inline] def mapFrozen {σ : StaticSpec}
    (f : ∀ {σ}, Tensor σ → Tensor σ) (fr : Frozen σ) : Frozen σ :=
  Frozen.map f fr

/-- Helper: skip a static field -/
@[inline] def mapStatic {α : Type} (_ : ∀ {σ}, Tensor σ → Tensor σ) (s : Static α) : Static α := s

/-! ## Instance for Single Tensor

A single tensor is trivially a TensorStruct.
-/

/-! ## Tuple Instances

Allow pairs of TensorStructs to be TensorStructs.
-/

instance [TensorStruct α] [TensorStruct β] : TensorStruct (α × β) where
  map f pair := (TensorStruct.map f pair.1, TensorStruct.map f pair.2)
  mapM f pair := do pure (← TensorStruct.mapM f pair.1, ← TensorStruct.mapM f pair.2)
  zipWith f p1 p2 := (TensorStruct.zipWith f p1.1 p2.1, TensorStruct.zipWith f p1.2 p2.2)
  fold f init pair := TensorStruct.fold f (TensorStruct.fold f init pair.1) pair.2

/-! ## Deriving Handler for TensorStruct -/

open Lean Elab Command Term Meta in
/-- Generate TensorStruct instance for a structure.
    All fields are assumed to have TensorStruct instances (Tensor σ, Option, Array, Static, etc.) -/
private def mkTensorStructInstanceCmd (typeName : Name) : CommandElabM Unit := do
  let env ← getEnv

  -- Get structure info
  let some structInfo := getStructureInfo? env typeName
    | throwError "{typeName} is not a structure"

  -- Get the inductive info to find parameters
  let some (.inductInfo indInfo) := env.find? typeName
    | throwError "{typeName} is not an inductive type"

  let fieldNames := structInfo.fieldNames

  -- Build field entries for `map`: field := TensorStruct.map f x.field
  let mapFields ← fieldNames.mapM fun fname => do
    let fnameId := mkIdent fname
    `(Lean.Parser.Term.structInstField| $fnameId:ident := TensorStruct.map f x.$fnameId)

  -- Build field entries for `mapM`: let field ← TensorStruct.mapM f x.field
  let mapMBinds ← fieldNames.mapM fun fname => do
    let fnameId := mkIdent fname
    `(doElem| let $fnameId ← TensorStruct.mapM f x.$fnameId)

  -- Build struct literal for mapM result
  let mapMFields ← fieldNames.mapM fun fname => do
    let fnameId := mkIdent fname
    `(Lean.Parser.Term.structInstField| $fnameId:ident := $fnameId)

  -- Build field entries for `zipWith`: field := TensorStruct.zipWith f x.field y.field
  let zipFields ← fieldNames.mapM fun fname => do
    let fnameId := mkIdent fname
    `(Lean.Parser.Term.structInstField| $fnameId:ident := TensorStruct.zipWith f x.$fnameId y.$fnameId)

  -- Build fold expression: TensorStruct.fold f (TensorStruct.fold f ... init x.field1) x.field2
  -- We build this as a right-to-left chain
  let mut foldExpr ← `(init)
  for fname in fieldNames do
    let fnameId := mkIdent fname
    foldExpr ← `(TensorStruct.fold f $foldExpr x.$fnameId)

  -- Parameters of the structure, bound by every generated declaration.
  let (binderStxs, appliedType) ← liftTermElabM do
    Meta.forallTelescopeReducing indInfo.type fun params _ => do
      let params := params[:indInfo.numParams].toArray
      let binders ← params.mapM fun param => do
        let paramDecl ← param.fvarId!.getDecl
        let paramTypeStx ← PrettyPrinter.delab (← Meta.inferType param)
        if paramDecl.binderInfo == .instImplicit then
          `(bracketedBinder| [$(mkIdent paramDecl.userName) : $paramTypeStx])
        else
          `(bracketedBinder| {$(mkIdent paramDecl.userName) : $paramTypeStx})
      let paramIdents ← params.mapM fun param => do
        pure (mkIdent (← param.fvarId!.getDecl).userName)
      -- `params` includes implicit record parameters such as a shared device.
      -- Apply every parameter explicitly so it cannot be mistaken for the
      -- next explicit shape/type argument.
      return (binders, ← `(@$(mkIdent typeName) $paramIdents*))

  -- The traversals are separate non-inlined functions that the instance calls.
  -- Inlined instance methods would paste every nested structure's traversal
  -- into its parent, making a model's traversal one function of its whole tree.
  let helper (suffix : String) :=
    mkIdent (`_root_ ++ privateToUserName typeName ++ Name.mkSimple s!"tensorStruct{suffix}")
  let (mapFn, mapMFn, zipFn, foldFn) :=
    (helper "Map", helper "MapM", helper "ZipWith", helper "Fold")
  elabCommand (← `(command|
    @[noinline] private def $mapFn $[$binderStxs]*
        (f : ∀ {σ : _root_.torch.StaticSpec}, _root_.torch.Tensor σ → _root_.torch.Tensor σ)
        (x : $appliedType) : $appliedType :=
      { $[$mapFields:structInstField],* }))
  elabCommand (← `(command|
    @[noinline] private def $mapMFn $[$binderStxs]* {m : Type → Type} [Monad m]
        (f : ∀ {σ : _root_.torch.StaticSpec}, _root_.torch.Tensor σ → m (_root_.torch.Tensor σ))
        (x : $appliedType) : m $appliedType := do
      $[$mapMBinds:doElem]*
      pure { $[$mapMFields:structInstField],* }))
  elabCommand (← `(command|
    @[noinline] private def $zipFn $[$binderStxs]*
        (f : ∀ {σ : _root_.torch.StaticSpec},
          _root_.torch.Tensor σ → _root_.torch.Tensor σ → _root_.torch.Tensor σ)
        (x y : $appliedType) : $appliedType :=
      { $[$zipFields:structInstField],* }))
  elabCommand (← `(command|
    @[noinline] private def $foldFn $[$binderStxs]* {β : Type}
        (f : ∀ {σ : _root_.torch.StaticSpec}, _root_.torch.Tensor σ → β → β)
        (init : β) (x : $appliedType) : β :=
      $foldExpr))
  elabCommand (← `(command|
    instance $[$binderStxs]* : TensorStruct $appliedType where
      map f x := $mapFn f x
      mapM f x := $mapMFn f x
      zipWith f x y := $zipFn f x y
      fold f init x := $foldFn f init x))

/-- Derive constrained traversal independently of structural traversal. Each
field contributes its own capability requirement, resolved only when the
resulting record instance is used with a particular leaf predicate. -/
private def mkTensorStructForInstanceCmd (typeName : Name) : CommandElabM Unit := do
  let env ← getEnv
  let some info := getStructureInfo? env typeName
    | throwError "{typeName} is not a structure"
  let some (.inductInfo inductiveInfo) := env.find? typeName
    | throwError "{typeName} is not an inductive type"
  let mapFields ← info.fieldNames.mapM fun field => do
    let field := mkIdent field
    `(Lean.Parser.Term.structInstField| $field:ident := TensorStructFor.map f x.$field)
  let mapMBinds ← info.fieldNames.mapM fun field => do
    let field := mkIdent field
    `(doElem| let $field ← TensorStructFor.mapM f x.$field)
  let mapMFields ← info.fieldNames.mapM fun field => do
    let field := mkIdent field
    `(Lean.Parser.Term.structInstField| $field:ident := $field)
  let zipFields ← info.fieldNames.mapM fun field => do
    let field := mkIdent field
    `(Lean.Parser.Term.structInstField| $field:ident := TensorStructFor.zipWith f x.$field y.$field)
  let mut foldExpr ← `(init)
  for field in info.fieldNames do
    let field := mkIdent field
    foldExpr ← `(TensorStructFor.fold f $foldExpr x.$field)
  let (binders, predicate, appliedType) ← liftTermElabM do
    forallTelescopeReducing inductiveInfo.type fun parameters _ => do
      let parameters := parameters.extract 0 inductiveInfo.numParams
      let predicate := mkIdent `_tensorLeafPredicate
      let mut binders : Array (TSyntax ``Parser.Term.bracketedBinder) := #[
        ← `(bracketedBinder| {$predicate : StaticSpec → Prop})]
      for parameter in parameters do
        let declaration ← parameter.fvarId!.getDecl
        let type ← withOptions (·.setBool `pp.explicit true) <|
          PrettyPrinter.delab declaration.type
        let binder ← if declaration.binderInfo == .instImplicit then
          `(bracketedBinder| [$(mkIdent declaration.userName) : $type])
        else
          `(bracketedBinder| {$(mkIdent declaration.userName) : $type})
        binders := binders.push binder
      let constructor := Lean.mkConst inductiveInfo.ctors.head!
        (inductiveInfo.levelParams.map Level.param)
      let fieldBinders ← forallTelescopeReducing
          (← inferType (mkAppN constructor parameters)) fun fields _ =>
        fields.mapM fun field => do
          let type ← inferType field
          if fields.any (fun previous => type.containsFVar previous.fvarId!) then
            throwError "Automatic TensorStructFor derivation does not support dependent fields; use a manual instance."
          let type ← withOptions (·.setBool `pp.explicit true) <| PrettyPrinter.delab type
          `(bracketedBinder| [TensorStructFor $predicate $type])
      binders := binders ++ fieldBinders
      let arguments ← parameters.mapM fun parameter =>
        return mkIdent (← parameter.fvarId!.getDecl).userName
      return (binders, predicate, ← `(@$(mkIdent typeName) $arguments*))
  -- Separate non-inlined traversals, as for `TensorStruct` above.
  let helper (suffix : String) :=
    mkIdent (`_root_ ++ privateToUserName typeName ++ Name.mkSimple s!"tensorStructFor{suffix}")
  let (mapFn, mapMFn, zipFn, foldFn) :=
    (helper "Map", helper "MapM", helper "ZipWith", helper "Fold")
  elabCommand (← `(command|
    @[noinline] private def $mapFn $[$binders]*
        (f : ∀ {σ : _root_.torch.StaticSpec}, $predicate σ →
          _root_.torch.Tensor σ → _root_.torch.Tensor σ)
        (x : $appliedType) : $appliedType :=
      { $[$mapFields:structInstField],* }))
  elabCommand (← `(command|
    @[noinline] private def $mapMFn $[$binders]* {m : Type → Type} [Monad m]
        (f : ∀ {σ : _root_.torch.StaticSpec}, $predicate σ →
          _root_.torch.Tensor σ → m (_root_.torch.Tensor σ))
        (x : $appliedType) : m $appliedType := do
      $[$mapMBinds:doElem]*
      pure { $[$mapMFields:structInstField],* }))
  elabCommand (← `(command|
    @[noinline] private def $zipFn $[$binders]*
        (f : ∀ {σ : _root_.torch.StaticSpec}, $predicate σ →
          _root_.torch.Tensor σ → _root_.torch.Tensor σ → _root_.torch.Tensor σ)
        (x y : $appliedType) : $appliedType :=
      { $[$zipFields:structInstField],* }))
  elabCommand (← `(command|
    @[noinline] private def $foldFn $[$binders]* {β : Type}
        (f : ∀ {σ : _root_.torch.StaticSpec}, $predicate σ → _root_.torch.Tensor σ → β → β)
        (init : β) (x : $appliedType) : β :=
      $foldExpr))
  elabCommand (← `(command|
    instance $[$binders]* : TensorStructFor $predicate $appliedType where
      map f x := $mapFn f x
      mapM f x := $mapMFn f x
      zipWith f x y := $zipFn f x y
      fold f init x := $foldFn f init x))

/-- Add only capability-aware traversal to a record with a custom TensorStruct
instance. Field traversal remains structural and uses each field's instances. -/
syntax (name := deriveTensorStructFor) "derive_tensor_struct_for " ident : command

elab_rules : command
  | `(derive_tensor_struct_for $typeName:ident) => do
    let typeName ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo typeName
    mkTensorStructForInstanceCmd typeName

open Lean Elab Deriving in
/-- Deriving handler for TensorStruct -/
def mkTensorStructHandler (typeNames : Array Name) : CommandElabM Bool := do
  if typeNames.isEmpty then return false

  for typeName in typeNames do
    mkTensorStructInstanceCmd typeName
    deriveTensorTreeCastFor typeName
    mkTensorStructForInstanceCmd typeName

  return true

/-- Unified deriving handler for full Tyr model support.
    Registers TensorStruct, ToTensorStructSchema, and TensorStructFlatten. -/
def mkModelHandler (typeNames : Array Name) : CommandElabM Bool := do
  if typeNames.isEmpty then return false

  -- Call individual handlers
  let _ ← mkTensorStructHandler typeNames
  let _ ← mkToTensorStructSchemaHandler typeNames
  let _ ← mkTensorStructFlattenHandler typeNames

  return true

-- Register the deriving handlers
open Lean Elab Deriving in
initialize
  registerDerivingHandler ``TensorStruct mkTensorStructHandler
  registerDerivingHandler ``Model mkModelHandler

end torch
