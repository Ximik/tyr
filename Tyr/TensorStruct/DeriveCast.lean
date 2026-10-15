import Lean
import Tyr.TensorStruct.Cast

/-!
# Explicit tensor-tree casts between record types

`derive_tensor_tree_cast Source => Target` creates a `TensorTreeCast Source Target`
instance by matching record fields by name and declaration order. Both sides can
be instantiated structure types or aliases for them. Each field delegates to its
own checked cast instance; static metadata is carried from the source.

No destination model or tensor template is constructed. This command is explicit:
ordinary `deriving TensorStruct` and `deriving Model` keep their existing meaning.

For parameterized records, declare the shared parameters before a colon:
`derive_tensor_tree_cast {source target : Device} {size : UInt64} :
  Source (device := source) size => Target (device := target) size`.
These binders scope over both record types and the generated instance.
-/

namespace torch

open Lean Elab Command Term Meta

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
