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
- explicit conversions between parameterizations of the same record through
  derived `TensorTreeCast` instances.

Current limits:

- it is not a fully customizable deriving framework,
- it assumes each field already has a suitable `TensorStruct` instance,
- non-structure or highly custom traversal policies still require manual instances.

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

  -- Get the type parameters
  let numParams := indInfo.numParams

  if numParams == 0 then
    -- No parameters - simple instance
    let typeId := mkIdent typeName
    let instCmd ← `(command|
      instance : TensorStruct $typeId where
        map f x := { $[$mapFields:structInstField],* }
        mapM f x := do
          $[$mapMBinds:doElem]*
          pure { $[$mapMFields:structInstField],* }
        zipWith f x y := { $[$zipFields:structInstField],* }
        fold f init x := $foldExpr
    )
    elabCommand instCmd
  else
    -- Has parameters - need to build binders
    let instCmd ← liftTermElabM do
      Meta.forallTelescopeReducing indInfo.type fun params _ => do
        let paramBinders := params[:numParams]

        -- Build implicit binder syntax for each parameter
        let binderStxs ← paramBinders.toArray.mapM fun param => do
          let paramDecl ← param.fvarId!.getDecl
          let paramName := paramDecl.userName
          let paramType ← Meta.inferType param
          let paramTypeStx ← PrettyPrinter.delab paramType
          if paramDecl.binderInfo == .instImplicit then
            `(bracketedBinder| [$(mkIdent paramName) : $paramTypeStx])
          else
            `(bracketedBinder| {$(mkIdent paramName) : $paramTypeStx})

        -- Build the applied type
        let paramIdents ← paramBinders.toArray.mapM fun param => do
          let paramDecl ← param.fvarId!.getDecl
          pure (mkIdent paramDecl.userName)

        -- `params` includes implicit record parameters such as a shared device.
        -- Apply every parameter explicitly so it cannot be mistaken for the
        -- next explicit shape/type argument.
        let appliedType ← `(@$(mkIdent typeName) $paramIdents*)

        -- Generate instance command with binders
        `(command|
          instance $[$binderStxs]* : TensorStruct $appliedType where
            map f x := { $[$mapFields:structInstField],* }
            mapM f x := do
              $[$mapMBinds:doElem]*
              pure { $[$mapMFields:structInstField],* }
            zipWith f x y := { $[$zipFields:structInstField],* }
            fold f init x := $foldExpr
        )

    elabCommand instCmd

open Lean Elab Deriving in
/-- Deriving handler for TensorStruct -/
def mkTensorStructHandler (typeNames : Array Name) : CommandElabM Bool := do
  if typeNames.isEmpty then return false

  for typeName in typeNames do
    mkTensorStructInstanceCmd typeName
    deriveTensorTreeCastFor typeName

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
