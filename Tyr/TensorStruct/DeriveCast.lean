import Lean
import Tyr.TensorStruct.Cast

/-!
# Explicit tensor-tree casts between record types

`derive_tensor_tree_cast Source => Target` creates a `TensorTreeCast Source Target`
instance by matching record fields by name and declaration order. Both sides can
be instantiated structure types or aliases for them. Each field delegates to its
own checked cast instance; static metadata is carried from the source.

No destination model or tensor template is constructed. Ordinary
`deriving TensorStruct` and `deriving Model` provide conversion between
parameterizations of the same record. This explicit command is for conversions
between different record types.

For parameterized records, declare the shared parameters before a colon:
`derive_tensor_tree_cast {source target : Device} {size : UInt64} :
  Source (device := source) size => Target (device := target) size`.
These binders scope over both record types and the generated instance.
-/

namespace torch

open Lean Elab Command Term Meta

/-- Introduce a distinct parameter telescope, including dependent parameter
types. Source and target arguments are paired by position, never by their names. -/
private partial def withCastParameters (parameters : Array Expr) (parameterPrefix : String)
    (k : Array Expr → TermElabM α) (introduced : Array Expr := #[]) : TermElabM α := do
  if introduced.size == parameters.size then
    k introduced
  else
    let parameter := parameters[introduced.size]!
    let declaration ← parameter.fvarId!.getDecl
    let type := declaration.type.replaceFVars
      (parameters.extract 0 introduced.size) introduced
    let name := Name.mkSimple s!"{parameterPrefix}_{introduced.size}_{declaration.userName}"
    withLocalDecl name .implicit type fun fresh =>
      withCastParameters parameters parameterPrefix k (introduced.push fresh)

private def castFieldTypes (constructor : Expr) (parameters : Array Expr) :
    TermElabM (Array Expr) := do
  forallTelescopeReducing (← inferType (mkAppN constructor parameters)) fun fields _ => do
    fields.mapM fun field => do
      let type ← inferType field
      if fields.any (fun previous => type.containsFVar previous.fvarId!) then
        throwError "Automatic TensorTreeCast derivation does not support fields whose types depend on earlier field values; use a manual TensorTreeCast instance."
      pure type

/-- Generate the same-record conversion provided by `deriving TensorStruct`.
Every parameter has distinct source and target values; conversion is available
only when each corresponding field has a checked `TensorTreeCast` instance. -/
def deriveTensorTreeCastFor (typeName : Name) : CommandElabM Unit := do
  let env ← getEnv
  let some info := getStructureInfo? env typeName
    | throwError "{typeName} is not a structure"
  let some (.inductInfo inductiveInfo) := env.find? typeName
    | throwError "{typeName} is not an inductive type"
  let fields ← info.fieldNames.mapM fun field => do
    let fieldId := mkIdent field
    let label := field.getString!
    `(Lean.Parser.Term.structInstField|
      $fieldId:ident := ← TensorTreeCast.castAt
        (TensorTreeCast.fieldPath _path $(Lean.quote label)) _source.$fieldId)
  let command ← liftTermElabM do
    forallTelescopeReducing inductiveInfo.type fun parameters _ => do
      let parameters := parameters.extract 0 inductiveInfo.numParams
      withCastParameters parameters "source" fun source =>
        withCastParameters parameters "target" fun target => do
          let constructor := Lean.mkConst inductiveInfo.ctors.head! (inductiveInfo.levelParams.map Level.param)
          let sourceTypes ← castFieldTypes constructor source
          let targetTypes ← castFieldTypes constructor target
          let mut binders : Array (TSyntax ``Parser.Term.bracketedBinder) := #[]
          for parameter in source ++ target do
            let declaration ← parameter.fvarId!.getDecl
            let type ← withOptions (·.setBool `pp.explicit true) <|
              PrettyPrinter.delab declaration.type
            binders := binders.push (← `(bracketedBinder|
              {$(mkIdent declaration.userName) : $type}))
          for sourceType in sourceTypes, targetType in targetTypes do
            let sourceType ← withOptions (·.setBool `pp.explicit true) <|
              PrettyPrinter.delab sourceType
            let targetType ← withOptions (·.setBool `pp.explicit true) <|
              PrettyPrinter.delab targetType
            binders := binders.push (← `(bracketedBinder|
              [TensorTreeCast $sourceType $targetType]))
          let sourceArguments ← source.mapM fun parameter =>
            return mkIdent (← parameter.fvarId!.getDecl).userName
          let targetArguments ← target.mapM fun parameter =>
            return mkIdent (← parameter.fvarId!.getDecl).userName
          let sourceType ← `(@$(mkIdent typeName) $sourceArguments*)
          let targetType ← `(@$(mkIdent typeName) $targetArguments*)
          `(command|
            instance $[$binders]* : TensorTreeCast $sourceType $targetType where
              castAt _path _source := do
                pure { $[$fields:structInstField],* })
  elabCommand command

private def castFieldLabel : Name → String
  | .anonymous => "_"
  | .str _ field => field
  | .num _ field => toString field

private def castStructureInfo (typeSyntax : TSyntax `term) : TermElabM StructureInfo := do
  let type ← Term.elabType typeSyntax
  Term.synthesizeSyntheticMVarsNoPostponing
  let type ← Meta.whnf (← instantiateMVars type)
  let .const typeName _ := type.getAppFn
    | throwErrorAt typeSyntax "Tensor tree cast requires an instantiated structure type."
  let some info := getStructureInfo? (← getEnv) typeName
    | throwErrorAt typeSyntax "Tensor tree cast requires a structure type, got {type}."
  return info

/-- Derive a checked field-wise cast between two explicitly selected record types.
Fields must have identical names and declaration order; leaf and container casts
must already have `TensorTreeCast` instances. -/
syntax (name := deriveTensorTreeCast)
  "derive_tensor_tree_cast " term " => " term : command

syntax (name := deriveTensorTreeCastWithBinders)
  "derive_tensor_tree_cast " bracketedBinder+ " : " term " => " term : command

private def elabTensorTreeCast (binders : Array (TSyntax ``Parser.Term.bracketedBinder))
    (sourceType targetType : TSyntax `term) : CommandElabM Unit := do
      let (sourceFields, targetFields) ← liftTermElabM do
        Term.elabBinders (binders.map (·.raw)) fun _ => do
          let sourceInfo ← castStructureInfo sourceType
          let targetInfo ← castStructureInfo targetType
          return (sourceInfo.fieldNames, targetInfo.fieldNames)
      let sourceLabels := sourceFields.map castFieldLabel
      let targetLabels := targetFields.map castFieldLabel
      unless sourceLabels == targetLabels do
        throwErrorAt targetType
          "Tensor tree cast requires the same field names in the same order; source fields are {sourceLabels}, target fields are {targetLabels}."
      let fields ← targetFields.mapM fun field => do
        let fieldIdent := mkIdent field
        let label := castFieldLabel field
        `(Lean.Parser.Term.structInstField|
          $fieldIdent:ident := ← TensorTreeCast.castAt
            (TensorTreeCast.fieldPath _path $(Lean.quote label)) _source.$fieldIdent)
      let instanceCommand ← `(command|
        instance $[$binders]* : TensorTreeCast $sourceType $targetType where
          castAt _path _source := do
            pure { $[$fields:structInstField],* })
      elabCommand instanceCommand

elab_rules : command
  | `(derive_tensor_tree_cast $sourceType => $targetType) =>
      elabTensorTreeCast #[] sourceType targetType
  | `(derive_tensor_tree_cast $binders:bracketedBinder* : $sourceType => $targetType) =>
      elabTensorTreeCast binders sourceType targetType

end torch
