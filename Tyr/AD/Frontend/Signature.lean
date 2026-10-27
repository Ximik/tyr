import Tyr.AD.TensorStructSchema
import Tyr.AD.JaxprLike.Core

/-!
# Tyr.AD.Frontend.Signature

Structured signature metadata for schema-aware AD frontends.

This layer sits between `TensorStruct`-derived boundary schemas and the flat
`LeanJaxpr` variable space. It gives frontends a stable ordered view of
parameter/input/output leaves before any tracing-specific lowering happens.
-/

namespace Tyr.AD.Frontend

open torch
open Tyr.AD.JaxprLike

private def renderPath (path : TensorLeafPath) : String :=
  let rendered := toString path
  if rendered.isEmpty then "<root>" else rendered

private def shapeToString (shape : Array Nat) : String :=
  "[" ++ String.intercalate ", " (shape.toList.map toString) ++ "]"

/-- Boundary class for one structured frontend section. -/
inductive FrontendBoundaryKind where
  | param
  | input
  | output
  deriving Repr, BEq, Inhabited, DecidableEq

instance : ToString FrontendBoundaryKind := ⟨fun
  | .param => "param"
  | .input => "input"
  | .output => "output"⟩

/--
One structured frontend boundary together with the tensor-leaf selection used to
map it into flat frontend variables.
-/
structure FrontendBoundary where
  schema : TensorStructSchema
  selection : TensorLeafSelection := .diffAndFrozen
  deriving Repr, Inhabited

namespace FrontendBoundary

def ofValue [ToTensorStructSchema α]
    (x : α)
    (selection : TensorLeafSelection := .diffAndFrozen) :
    FrontendBoundary :=
  { schema := ToTensorStructSchema.schema x, selection := selection }

def typeName (boundary : FrontendBoundary) : Lean.Name :=
  boundary.schema.typeName

def selectedLeafSpecs (boundary : FrontendBoundary) : Array TensorLeafSpec :=
  boundary.schema.leaves.filter fun leaf =>
    TensorLeafSelection.includesRole boundary.selection leaf.role

def selectedLeafCount (boundary : FrontendBoundary) : Nat :=
  boundary.selectedLeafSpecs.size

def renderedSelectedLeafPaths (boundary : FrontendBoundary) : Array String :=
  boundary.selectedLeafSpecs.map (fun leaf => renderPath leaf.path)

private def validateSelectedSpecs
    (expected current : FrontendBoundary) :
    Except String Unit := do
  if expected.typeName != current.typeName then
    throw <| s!"Frontend boundary type mismatch: expected `{expected.typeName}`, " ++
      s!"got `{current.typeName}`."
  let expectedSpecs := expected.selectedLeafSpecs
  let currentSpecs := current.selectedLeafSpecs
  if expectedSpecs.size != currentSpecs.size then
    throw <| s!"Frontend boundary leaf-count mismatch for `{expected.typeName}`: " ++
      s!"expected {expectedSpecs.size} selected leaves, got {currentSpecs.size}."
  for h : i in [:expectedSpecs.size] do
    let expectedLeaf := expectedSpecs[i]
    let some currentLeaf := currentSpecs[i]? | throw <|
      s!"Internal frontend boundary validation error for `{expected.typeName}`: " ++
        s!"missing current selected leaf #{i} after count validation."
    if expectedLeaf.path != currentLeaf.path then
      throw <| s!"Frontend boundary leaf-path mismatch for `{expected.typeName}` at selected leaf #{i}: " ++
        s!"expected {renderPath expectedLeaf.path}, got {renderPath currentLeaf.path}."
    if expectedLeaf.role != currentLeaf.role then
      throw <| s!"Frontend boundary leaf-role mismatch for `{expected.typeName}` at " ++
        s!"{renderPath expectedLeaf.path}: expected {reprStr expectedLeaf.role}, got {reprStr currentLeaf.role}."
    if expectedLeaf.shape != currentLeaf.shape then
      throw <| s!"Frontend boundary leaf-shape metadata mismatch for `{expected.typeName}` at " ++
        s!"{renderPath expectedLeaf.path}: expected {reprStr expectedLeaf.shape}, got {reprStr currentLeaf.shape}."
    if expectedLeaf.dtype != currentLeaf.dtype then
      throw <| s!"Frontend boundary leaf-dtype metadata mismatch for `{expected.typeName}` at " ++
        s!"{renderPath expectedLeaf.path}: expected {reprStr expectedLeaf.dtype}, got {reprStr currentLeaf.dtype}."

def validateValue [ToTensorStructSchema α]
    (boundary : FrontendBoundary)
    (x : α) :
    Except String Unit :=
  validateSelectedSpecs boundary (FrontendBoundary.ofValue x boundary.selection)

def validateFlattenedLeaves
    (boundary : FrontendBoundary)
    (leaves : Array TensorLeafValue) :
    Except String Unit := do
  let specs := boundary.selectedLeafSpecs
  if leaves.size != specs.size then
    throw <| s!"Flattened leaf-count mismatch for `{boundary.typeName}`: " ++
      s!"expected {specs.size} selected leaves, got {leaves.size}."
  for h : i in [:specs.size] do
    let spec := specs[i]
    let some leaf := leaves[i]? | throw <|
      s!"Internal flattened-leaf validation error for `{boundary.typeName}`: " ++
        s!"missing flattened leaf #{i} after count validation."
    if spec.role != leaf.role then
      throw <| s!"Flattened leaf role mismatch for `{boundary.typeName}` at " ++
        s!"{renderPath spec.path}: expected {reprStr spec.role}, got {reprStr leaf.role}."
    match spec.shape with
    | some expectedShape =>
        if leaf.shape != expectedShape then
          throw <| s!"Flattened leaf shape mismatch for `{boundary.typeName}` at " ++
            s!"{renderPath spec.path}: expected {shapeToString expectedShape}, got {shapeToString leaf.shape}."
    | none => pure ()
    match leaf.payload.2.validate with
    | .ok () => pure ()
    | .error message => throw s!"Invalid flattened tensor at {renderPath spec.path}: {message}"
    if let some expectedDType := spec.dtype then
      let actualDType := leaf.payload.2.raw.dtype.canonicalName
      if actualDType != expectedDType then
        throw s!"Flattened leaf dtype mismatch at {renderPath spec.path}: expected {expectedDType}, got {actualDType}."

def flattenValue [ToTensorStructSchema α] [TensorStructFlatten α]
    (boundary : FrontendBoundary)
    (x : α) :
    Except String (Array TensorLeafValue) := do
  boundary.validateValue x
  let leaves := TensorStructFlatten.flatten x boundary.selection
  boundary.validateFlattenedLeaves leaves
  pure leaves

def rebuildValue [ToTensorStructSchema α] [TensorStructFlatten α]
    (boundary : FrontendBoundary)
    (template : α)
    (leaves : Array TensorLeafValue) :
    Except String α := do
  boundary.validateValue template
  boundary.validateFlattenedLeaves leaves
  TensorStructFlatten.rebuildFrom template leaves boundary.selection

end FrontendBoundary

/-- One selected structured leaf in a frontend signature. -/
structure FrontendLeafBinding where
  kind : FrontendBoundaryKind
  boundaryIndex : Nat
  leafIndex : Nat
  ownerTypeName : Lean.Name := .anonymous
  path : TensorLeafPath := {}
  role : TensorLeafRole := .diff
  shape : Option (Array Nat) := none
  dtype : Option String := none
  deriving Repr, Inhabited, BEq, DecidableEq

namespace FrontendLeafBinding

def renderedPath (binding : FrontendLeafBinding) : String :=
  renderPath binding.path

def participation (binding : FrontendLeafBinding) : DiffParticipation :=
  match binding.role with
  | .diff => .diff
  | .static => .static
  | .frozen => .frozen

def toVarMeta (binding : FrontendLeafBinding) : VarMeta :=
  { participation := binding.participation
    shape := binding.shape
    dtype := binding.dtype }

def toJVar
    (binding : FrontendLeafBinding)
    (id : JVarId)
    (ty : Lean.IR.IRType := .object) :
    JVar :=
  { id := id, ty := ty, metaInfo := binding.toVarMeta }

def validateJVarsAgainstBindings
    (label : String)
    (bindings : Array FrontendLeafBinding)
    (vars : Array JVar) :
    Except String Unit := do
  if vars.size != bindings.size then
    throw <| s!"Frontend signature {label}-count mismatch: expected {bindings.size}, got {vars.size}."
  for h : i in [:bindings.size] do
    let binding := bindings[i]
    let some v := vars[i]? | throw <|
      s!"Internal frontend signature validation error: missing {label} #{i} after count validation."
    if v.metaInfo.participation != FrontendLeafBinding.participation binding then
      throw <| s!"Frontend signature participation mismatch for {label} " ++
        s!"{renderPath binding.path}: expected {reprStr (FrontendLeafBinding.participation binding)}, got {reprStr v.metaInfo.participation}."
    if binding.shape != v.metaInfo.shape then
      throw <| s!"Frontend signature shape mismatch for {label} " ++
        s!"{renderPath binding.path}: expected {reprStr binding.shape}, got {reprStr v.metaInfo.shape}."
    match binding.dtype with
    | some dtype =>
        if v.metaInfo.dtype != some dtype then
          throw <| s!"Frontend signature dtype mismatch for {label} " ++
            s!"{renderPath binding.path}: expected {dtype}, got {reprStr v.metaInfo.dtype}."
    | none => pure ()

end FrontendLeafBinding

/-- Structured signature metadata for a frontend-traced declaration. -/
structure FrontendADSignature where
  params : Array FrontendBoundary := #[]
  inputs : Array FrontendBoundary := #[]
  outputs : Array FrontendBoundary := #[]
  deriving Repr, Inhabited

namespace FrontendADSignature

private def bindingsFor
    (kind : FrontendBoundaryKind)
    (boundaries : Array FrontendBoundary) :
    Array FrontendLeafBinding := Id.run do
  let mut out : Array FrontendLeafBinding := #[]
  for h : boundaryIdx in [:boundaries.size] do
    let boundary := boundaries[boundaryIdx]
    let specs := boundary.selectedLeafSpecs
    for hLeaf : leafIdx in [:specs.size] do
      let leaf := specs[leafIdx]
      out := out.push {
        kind := kind
        boundaryIndex := boundaryIdx
        leafIndex := leafIdx
        ownerTypeName := boundary.typeName
        path := leaf.path
        role := leaf.role
        shape := leaf.shape
        dtype := leaf.dtype
      }
  return out

def paramBindings (sig : FrontendADSignature) : Array FrontendLeafBinding :=
  bindingsFor .param sig.params

def inputBindings (sig : FrontendADSignature) : Array FrontendLeafBinding :=
  bindingsFor .input sig.inputs

def outputBindings (sig : FrontendADSignature) : Array FrontendLeafBinding :=
  bindingsFor .output sig.outputs

def invarBindings (sig : FrontendADSignature) : Array FrontendLeafBinding :=
  sig.paramBindings ++ sig.inputBindings

def paramLeafCount (sig : FrontendADSignature) : Nat :=
  sig.paramBindings.size

def inputLeafCount (sig : FrontendADSignature) : Nat :=
  sig.inputBindings.size

def outputLeafCount (sig : FrontendADSignature) : Nat :=
  sig.outputBindings.size

def renderedInvarPaths (sig : FrontendADSignature) : Array String :=
  sig.invarBindings.map (·.renderedPath)

def renderedOutputPaths (sig : FrontendADSignature) : Array String :=
  sig.outputBindings.map (·.renderedPath)

private def mkJVarsFromBindings
    (bindings : Array FrontendLeafBinding)
    (startId : JVarId := 0)
    (ty : Lean.IR.IRType := .object) :
    Array JVar := Id.run do
  let mut out : Array JVar := #[]
  for h : i in [:bindings.size] do
    out := out.push (bindings[i].toJVar (startId + i) ty)
  return out

def invars
    (sig : FrontendADSignature)
    (startId : JVarId := 0)
    (ty : Lean.IR.IRType := .object) :
    Array JVar :=
  mkJVarsFromBindings sig.invarBindings startId ty

def outvars
    (sig : FrontendADSignature)
    (startId : JVarId := 0)
    (ty : Lean.IR.IRType := .object) :
    Array JVar :=
  mkJVarsFromBindings sig.outputBindings startId ty

private def boundaryLeafSpans
    (boundaries : Array FrontendBoundary)
    (startOffset : Nat := 0) :
    Array (Nat × Nat) := Id.run do
  let mut out : Array (Nat × Nat) := #[]
  let mut start := startOffset
  for boundary in boundaries do
    let count := boundary.selectedLeafCount
    out := out.push (start, count)
    start := start + count
  return out

def paramBoundaryLeafSpans (sig : FrontendADSignature) : Array (Nat × Nat) :=
  boundaryLeafSpans sig.params

def inputBoundaryLeafSpans (sig : FrontendADSignature) : Array (Nat × Nat) :=
  boundaryLeafSpans sig.inputs sig.paramLeafCount

def outputBoundaryLeafSpans (sig : FrontendADSignature) : Array (Nat × Nat) :=
  boundaryLeafSpans sig.outputs

private def sliceBoundaryLeaves
    (label : String)
    (boundaries : Array FrontendBoundary)
    (spans : Array (Nat × Nat))
    (expectedTotal : Nat)
    (boundaryIndex : Nat)
    (leaves : Array TensorLeafValue) :
    Except String (Array TensorLeafValue) := do
  let some boundary := boundaries[boundaryIndex]? | throw <|
    s!"Unknown {label} boundary index {boundaryIndex}; signature has {boundaries.size} {label} boundaries."
  let some (start, count) := spans[boundaryIndex]? | throw <|
    s!"Internal {label}-boundary span error at index {boundaryIndex}."
  if leaves.size != expectedTotal then
    throw <| s!"Flattened {label} leaf-count mismatch: expected {expectedTotal}, got {leaves.size}."
  let slice := leaves.extract start (start + count)
  boundary.validateFlattenedLeaves slice
  pure slice

def sliceParamLeaves
    (sig : FrontendADSignature)
    (paramBoundaryIndex : Nat)
    (invarLeaves : Array TensorLeafValue) :
    Except String (Array TensorLeafValue) :=
  sliceBoundaryLeaves "param"
    sig.params
    sig.paramBoundaryLeafSpans
    (sig.paramLeafCount + sig.inputLeafCount)
    paramBoundaryIndex
    invarLeaves

def sliceInputLeaves
    (sig : FrontendADSignature)
    (inputBoundaryIndex : Nat)
    (invarLeaves : Array TensorLeafValue) :
    Except String (Array TensorLeafValue) :=
  sliceBoundaryLeaves "input"
    sig.inputs
    sig.inputBoundaryLeafSpans
    (sig.paramLeafCount + sig.inputLeafCount)
    inputBoundaryIndex
    invarLeaves

def sliceOutputLeaves
    (sig : FrontendADSignature)
    (outputBoundaryIndex : Nat)
    (leaves : Array TensorLeafValue) :
    Except String (Array TensorLeafValue) :=
  sliceBoundaryLeaves "output"
    sig.outputs
    sig.outputBoundaryLeafSpans
    sig.outputLeafCount
    outputBoundaryIndex
    leaves

def rebuildParamValue
    [ToTensorStructSchema α]
    [TensorStructFlatten α]
    (sig : FrontendADSignature)
    (paramBoundaryIndex : Nat)
    (template : α)
    (invarLeaves : Array TensorLeafValue) :
    Except String α := do
  let some boundary := sig.params[paramBoundaryIndex]? | throw <|
    s!"Unknown param boundary index {paramBoundaryIndex}; signature has {sig.params.size} params."
  let slice ← sig.sliceParamLeaves paramBoundaryIndex invarLeaves
  boundary.rebuildValue template slice

def rebuildInputValue
    [ToTensorStructSchema α]
    [TensorStructFlatten α]
    (sig : FrontendADSignature)
    (inputBoundaryIndex : Nat)
    (template : α)
    (invarLeaves : Array TensorLeafValue) :
    Except String α := do
  let some boundary := sig.inputs[inputBoundaryIndex]? | throw <|
    s!"Unknown input boundary index {inputBoundaryIndex}; signature has {sig.inputs.size} inputs."
  let slice ← sig.sliceInputLeaves inputBoundaryIndex invarLeaves
  boundary.rebuildValue template slice

def rebuildOutputValue
    [ToTensorStructSchema α]
    [TensorStructFlatten α]
    (sig : FrontendADSignature)
    (outputBoundaryIndex : Nat)
    (template : α)
    (leaves : Array TensorLeafValue) :
    Except String α := do
  let some boundary := sig.outputs[outputBoundaryIndex]? | throw <|
    s!"Unknown output boundary index {outputBoundaryIndex}; signature has {sig.outputs.size} outputs."
  let slice ← sig.sliceOutputLeaves outputBoundaryIndex leaves
  boundary.rebuildValue template slice

def validateJaxprBoundaryMetadata
    (sig : FrontendADSignature)
    (jaxpr : LeanJaxpr) :
    Except String Unit := do
  FrontendLeafBinding.validateJVarsAgainstBindings "invar"
    sig.invarBindings
    jaxpr.invars
  FrontendLeafBinding.validateJVarsAgainstBindings "outvar"
    sig.outputBindings
    jaxpr.outvars

end FrontendADSignature

end Tyr.AD.Frontend
