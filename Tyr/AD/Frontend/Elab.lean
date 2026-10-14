import Lean
import Tyr.Typed.Tensor
import Tyr.AD.JaxprLike.Elab

/-!
# Tyr.AD.Frontend.Elab

Restricted definition-site frontend derivation helpers.

This is the first bridge from ordinary Lean definitions to direct frontend AD
IR: it inspects definition bodies in a deliberately small supported subset and
registers `LeanJaxpr` automatically. The initial subset is one primitive call
over tensor-typed parameters with no extra computation around it.
-/

namespace Tyr.AD.Frontend

open Lean Meta
open torch
open Tyr.AD.JaxprLike

/--
Frontend lowering spec for one ordinary Lean primitive wrapper.

`leanConst` is the constant that appears in the Lean definition body.
`op` is the normalized frontend/Jaxpr primitive name to emit.
`sourceOp` optionally preserves a higher-level source identity distinct from the
normalized op.
-/
structure PrimitiveFrontendSpec where
  leanConst : Name
  op : OpName
  sourceOp : Option OpName := none
  fixedParams : OpParams := #[]
  deriving Inhabited, Repr

private def shapeToNatArray (shape : Shape) : Array Nat :=
  Array.map UInt64.toNat shape

private unsafe def evalClosedShapeExpr? (shapeExpr : Expr) : MetaM (Option (Array Nat)) := do
  let shapeExpr ← whnf (← instantiateMVars shapeExpr)
  if !shapeExpr.hasFVar then
    try
      let shape ← evalExpr Shape (mkConst ``torch.Shape) shapeExpr
      return some (shapeToNatArray shape)
    catch _ =>
      return none
  else
    return none

private unsafe def indexedVarMeta (spec : Expr) (participation : DiffParticipation) : MetaM VarMeta := do
  let shape? ← evalClosedShapeExpr? (mkProj ``torch.StaticSpec 0 spec)
  let dtypeExpr ← whnf (← instantiateMVars (mkProj ``torch.StaticSpec 1 spec))
  let dtype? ← if dtypeExpr.hasFVar then pure none else do
    try
      let dtype ← evalExpr torch.DType (mkConst ``torch.DType) dtypeExpr
      pure <| match dtype with
        | .Unknown _ => none
        | dtype => some dtype.canonicalName
    catch _ => pure none
  return { participation, shape := shape?, dtype := dtype? }

private unsafe def tensorLikeVarMeta? (type : Expr) : MetaM (Option VarMeta) := do
  -- Normalize aliases without unfolding Tensor into its opaque carrier. Its
  -- public application retains the specification needed for frontend metadata.
  let type ← whnfHeadPred (← instantiateMVars type) fun type =>
    pure (!(type.isAppOfArity ``torch.Tensor 1 || type.isAppOfArity ``torch.Frozen 1))
  if type.isAppOfArity ``torch.Tensor 1 then
    return some (← indexedVarMeta (type.getArg! 0) .diff)
  else if type.isAppOfArity ``torch.Frozen 1 then
    return some (← indexedVarMeta (type.getArg! 0) .frozen)
  else
    return none

private def unsupportedParamMessage (param : Expr) (type : Expr) : MetaM String := do
  let paramDecl ← param.fvarId!.getDecl
  return s!"single-primitive frontend derivation only supports tensor-like parameters, " ++
    s!"but `{paramDecl.userName}` has type `{type}`"

private partial def letInline (expr : Expr) : Expr :=
  match expr.consumeMData with
  | .letE _ _ value body _ => letInline (body.instantiate1 value)
  | other => other

private def lookupPrimitiveSpec?
    (specs : Array PrimitiveFrontendSpec)
    (leanConst : Name) :
    Option PrimitiveFrontendSpec :=
  specs.find? (fun spec => spec.leanConst == leanConst)

private unsafe def buildInvars
    (params : Array Expr) :
    MetaM (Array JVar × Std.HashMap FVarId JVar) := do
  let mut nextId : JVarId := 1
  let mut invars : Array JVar := #[]
  let mut env : Std.HashMap FVarId JVar := {}
  for param in params do
    let type ← inferType param
    let some metaInfo ← tensorLikeVarMeta? type
      | throwError (← unsupportedParamMessage param type)
    let v : JVar := { id := nextId, ty := .object, metaInfo := metaInfo }
    invars := invars.push v
    env := env.insert param.fvarId! v
    nextId := nextId + 1
  return (invars, env)

private def requireConstApp
    (declName : Name)
    (body : Expr) :
    MetaM (Name × Array Expr) := do
  let body := letInline body
  let fn := body.getAppFn
  let args := body.getAppArgs
  match fn with
  | .const name _ => pure (name, args)
  | _ =>
      throwError
        m!"single-primitive frontend derivation for `{declName}` only supports a direct constant application body"

private def argsToJVars
    (declName : Name)
    (paramEnv : Std.HashMap FVarId JVar)
    (args : Array Expr) :
    MetaM (Array JVar) := do
  let mut invars : Array JVar := #[]
  for arg in args do
    let arg := arg.consumeMData
    let .fvar fvarId := arg
      | throwError
          m!"single-primitive frontend derivation for `{declName}` only supports primitive arguments that are direct parameters"
    let some v := paramEnv[fvarId]?
      | throwError
          m!"single-primitive frontend derivation for `{declName}` encountered a non-parameter primitive argument"
    invars := invars.push v
  return invars

/--
Derive a direct `FrontendRegistration` from an ordinary Lean definition in the
initial supported subset:

- tensor-like parameters only (`torch.Tensor` / `torch.Frozen`)
- body is a single registered primitive call
- result type is tensor-like
-/
unsafe def deriveSinglePrimitiveFrontendRegistration
    (declName : Name)
    (specs : Array PrimitiveFrontendSpec) :
    CoreM FrontendRegistration :=
  MetaM.run' do
    let info ← getConstInfo declName
    let value ←
      match info with
      | .defnInfo info => pure info.value
      | .thmInfo info => pure info.value
      | _ =>
          throwError
            m!"single-primitive frontend derivation only supports definitions/theorems, but `{declName}` is `{info.name}`"
    lambdaTelescope value fun params body => do
      let (primName, args) ← requireConstApp declName body
      let some spec := lookupPrimitiveSpec? specs primName
        | throwError
            m!"single-primitive frontend derivation for `{declName}` has no primitive spec for `{primName}`"
      let (jaxprInvars, paramEnv) ← buildInvars params
      let eqnInvars ← argsToJVars declName paramEnv args
      let outputType ← inferType body
      let some outputMeta ← tensorLikeVarMeta? outputType
        | throwError
            m!"single-primitive frontend derivation for `{declName}` only supports tensor-like outputs, got `{outputType}`"
      let outvar : JVar := {
        id := jaxprInvars.size + 1
        ty := .object
        metaInfo := outputMeta
      }
      let extraParams :=
        match spec.sourceOp with
        | some sourceOp => #[OpParam.mkName .sourceOp sourceOp] ++ spec.fixedParams
        | none => spec.fixedParams
      let jaxpr : LeanJaxpr := {
        invars := jaxprInvars
        eqns := #[{
          op := spec.op
          invars := eqnInvars
          outvars := #[outvar]
          params := extraParams
          source := { decl := declName }
        }]
        outvars := #[outvar]
      }
      pure { jaxpr := jaxpr.withDefaultSourceDecl declName }

/--
Derive and register a direct frontend bundle for an ordinary Lean definition in
the initial single-primitive supported subset.
-/
unsafe def registerDerivedSinglePrimitiveFrontend
    (declName : Name)
    (specs : Array PrimitiveFrontendSpec) :
    CoreM FrontendRegistration := do
  let registration ← deriveSinglePrimitiveFrontendRegistration declName specs
  registerFrontendRegistration declName registration
  pure registration

end Tyr.AD.Frontend
