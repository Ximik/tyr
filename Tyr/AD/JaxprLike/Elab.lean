import Lean
import Tyr.AD.JaxprLike.HintRegistry
import Tyr.AD.JaxprLike.Validate
import Tyr.AD.Frontend.API

/-!
# Tyr.AD.JaxprLike.Elab

Elaboration-time registration helpers for frontends that can emit structured
frontend AD artifacts directly instead of relying on `FnBody` recovery.

When a registered frontend bundle also provides a runtime frontend constant,
the `attribute [ad_frontend frontendSpec] f` path synthesizes importable
structured companions such as `f.frontend`, `f.linearize`, `f.vjp`,
`f.valueAndGrad`, and `f.grad`.
-/

namespace Tyr.AD.JaxprLike

open Lean Elab Command Meta

private structure RuntimeFrontendTypeInfo where
  paramsType : Expr
  inputsType : Expr
  outputsType : Expr

private structure FrontendCompanionNames where
  frontend : Name
  linearize : Name
  vjp : Name
  valueAndGrad : Name
  grad : Name

private def mkFrontendCompanionNames (declName : Name) : FrontendCompanionNames :=
  { frontend := Name.mkStr declName "frontend"
    linearize := Name.mkStr declName "linearize"
    vjp := Name.mkStr declName "vjp"
    valueAndGrad := Name.mkStr declName "valueAndGrad"
    grad := Name.mkStr declName "grad" }

private def inferRuntimeFrontendTypeInfoMeta
    (runtimeFrontend : Name) :
    MetaM RuntimeFrontendTypeInfo := do
  let info ← getConstInfo runtimeFrontend
  forallTelescopeReducing info.type fun xs body => do
    if !xs.isEmpty then
      throwError m!"runtime frontend `{runtimeFrontend}` must be a closed constant"
    let body ← whnf body
    let fn := body.getAppFn
    let args := body.getAppArgs
    let fnName ←
      match fn with
      | .const name _ => pure name
      | _ =>
          throwError
            m!"runtime frontend `{runtimeFrontend}` must have type `Tyr.AD.Frontend.StructuredFrontendFunction ...`"
    unless fnName == ``Tyr.AD.Frontend.StructuredFrontendFunction do
      throwError
        m!"runtime frontend `{runtimeFrontend}` must have type `Tyr.AD.Frontend.StructuredFrontendFunction ...`, got `{fnName}`"
    let some paramsType := args[0]? | throwError
      m!"runtime frontend `{runtimeFrontend}` is missing its parameter type argument"
    let some inputsType := args[1]? | throwError
      m!"runtime frontend `{runtimeFrontend}` is missing its input type argument"
    let some outputsType := args[2]? | throwError
      m!"runtime frontend `{runtimeFrontend}` is missing its output type argument"
    if args.size != 3 then
      throwError m!"runtime frontend `{runtimeFrontend}` must have exactly three type arguments"
    pure { paramsType := paramsType, inputsType := inputsType, outputsType := outputsType }

private def inferRuntimeFrontendTypeInfo
    (runtimeFrontend : Name) :
    CoreM RuntimeFrontendTypeInfo :=
  Lean.Meta.MetaM.run' <| inferRuntimeFrontendTypeInfoMeta runtimeFrontend

private def inferRuntimeFrontendTypeSyntax
    (runtimeFrontend : Name) :
    CommandElabM (TSyntax `term × TSyntax `term × TSyntax `term) := do
  liftTermElabM do
    let info ← inferRuntimeFrontendTypeInfoMeta runtimeFrontend
    let paramsType ← PrettyPrinter.delab info.paramsType
    let inputsType ← PrettyPrinter.delab info.inputsType
    let outputsType ← PrettyPrinter.delab info.outputsType
    pure (paramsType, inputsType, outputsType)

private def ensureCompanionNamesAreFresh
    (declName : Name)
    (names : FrontendCompanionNames) :
    CommandElabM Unit := do
  let env ← getEnv
  for name in #[names.frontend, names.linearize, names.vjp, names.valueAndGrad, names.grad] do
    if env.find? name |>.isSome then
      throwError m!"cannot synthesize structured frontend companion `{name}` for `{declName}`: declaration already exists"

private def synthesizeStructuredFrontendCompanions
    (declName runtimeFrontend : Name) :
    CommandElabM Unit := do
  let names := mkFrontendCompanionNames declName
  ensureCompanionNamesAreFresh declName names
  let (paramsType, inputsType, outputsType) ← inferRuntimeFrontendTypeSyntax runtimeFrontend
  let runtimeId := mkIdent runtimeFrontend
  let frontendId := mkIdent names.frontend
  let addCompanionDef
      (name : Name)
      (typeStx valueStx : TSyntax `term) :
      CommandElabM Unit := do
    let decl ← liftTermElabM do
      let type ← Term.elabType typeStx
      let value ← Term.elabTermEnsuringType valueStx type
      let type ← instantiateMVars type
      let value ← instantiateMVars value
      let defn ← mkDefinitionValInferringUnsafe name [] type value ReducibilityHints.abbrev
      pure (Declaration.defnDecl defn)
    liftCoreM <| addAndCompile decl

  let frontendType ← `(term|
    Tyr.AD.Frontend.StructuredFrontendFunction $paramsType $inputsType $outputsType)
  let frontendValue ← `(term| $runtimeId)
  addCompanionDef names.frontend frontendType frontendValue

  let linearizeType ← `(term|
    $paramsType → $inputsType →
      Except String (Tyr.AD.Frontend.StructuredPullback $paramsType $inputsType $outputsType))
  let linearizeValue ← `(term|
    fun params inputs =>
      Tyr.AD.Frontend.StructuredFrontendFunction.linearize $frontendId params inputs)
  addCompanionDef names.linearize linearizeType linearizeValue

  let vjpType ← `(term|
    $paramsType → $inputsType → $outputsType →
      Except String (Tyr.AD.Frontend.StructuredVJPResult $paramsType $inputsType))
  let vjpValue ← `(term|
    fun params inputs outputCotangent =>
      Tyr.AD.Frontend.StructuredFrontendFunction.vjp $frontendId params inputs outputCotangent)
  addCompanionDef names.vjp vjpType vjpValue

  let valueAndGradType ← `(term|
    $paramsType → $inputsType →
      Except String ($outputsType × Tyr.AD.Frontend.StructuredGradResult $paramsType))
  let valueAndGradValue ← `(term|
    fun params inputs =>
      Tyr.AD.Frontend.StructuredFrontendFunction.valueAndGrad $frontendId params inputs)
  addCompanionDef names.valueAndGrad valueAndGradType valueAndGradValue

  let gradType ← `(term|
    $paramsType → $inputsType →
      Except String (Tyr.AD.Frontend.StructuredGradResult $paramsType))
  let gradValue ← `(term|
    fun params inputs =>
      Tyr.AD.Frontend.StructuredFrontendFunction.grad $frontendId params inputs)
  addCompanionDef names.grad gradType gradValue

/-- Attach direct frontend AD artifacts to a declaration at elaboration time. -/
syntax (name := adFrontendAttr) "ad_frontend" ident : attr
syntax (name := applyAdFrontendCmd)
  "attribute" "[" "ad_frontend" ident "]" ident : command

private def validateFrontendJaxpr
    (declName : Name)
    (jaxpr : LeanJaxpr) :
    CoreM LeanJaxpr := do
  let jaxpr := jaxpr.withDefaultSourceDecl declName
  match validate jaxpr with
  | .ok () => pure jaxpr
  | .error errs =>
    throwError m!"invalid frontend jaxpr for `{declName}`:\n{String.intercalate "\n" errs.toList}"

private def validateFrontendRegistration
    (declName : Name)
    (registration : FrontendRegistration) :
    CoreM FrontendRegistration := do
  let jaxpr ← validateFrontendJaxpr declName registration.jaxpr
  match registration.runtimeFrontend with
  | some runtimeFrontend =>
      discard <| inferRuntimeFrontendTypeInfo runtimeFrontend
  | none => pure ()
  match registration.signature with
  | none =>
      pure { registration with jaxpr := jaxpr }
  | some sig =>
      match sig.validateJaxprBoundaryMetadata jaxpr with
      | .ok () =>
          pure { registration with jaxpr := jaxpr }
      | .error err =>
          throwError m!"invalid `ad_frontend` for `{declName}`:\n{err}"

initialize frontendAttr : ParametricAttribute FrontendRegistration ←
  registerParametricAttribute {
    name := `adFrontendAttr
    descr := "Register direct frontend AD artifacts for elimination-based AD"
    getParam := fun declName stx => do
      match stx with
      | `(attr| ad_frontend $frontendId:ident) =>
        let frontendDecl ← realizeGlobalConstNoOverload frontendId
        let registration ← unsafe evalConstCheck FrontendRegistration ``FrontendRegistration frontendDecl
        validateFrontendRegistration declName registration
      | _ =>
        throwError "invalid ad_frontend attribute"
  }

/--
Register frontend-produced AD artifacts for a declaration. `buildFromDecl`
will prefer this direct high-level IR bundle over `FnBody` recovery when
present.
-/
def registerFrontendRegistration
    (declName : Lean.Name)
    (registration : FrontendRegistration) :
    Lean.CoreM Unit := do
  let env ← Lean.getEnv
  match frontendAttr.setParam env declName registration with
  | .ok env' => Lean.setEnv env'
  | .error err => throwError err

def getRegisteredFrontendRegistration?
    (declName : Lean.Name) :
    Lean.CoreM (Option FrontendRegistration) := do
  return frontendAttr.getParam? (← Lean.getEnv) declName

def getRegisteredFrontendLeanJaxpr?
    (declName : Lean.Name) :
    Lean.CoreM (Option LeanJaxpr) := do
  return (← getRegisteredFrontendRegistration? declName).map (·.jaxpr)

def getRegisteredFrontendADSignature?
    (declName : Lean.Name) :
    Lean.CoreM (Option Tyr.AD.Frontend.FrontendADSignature) := do
  return (← getRegisteredFrontendRegistration? declName).bind (·.signature)

private def resolveFrontendDeclName (frontendId : Syntax) : CommandElabM Name :=
  liftCoreM <| realizeGlobalConstNoOverloadWithInfo frontendId

private def resolveTargetDeclName (declId : Syntax) : CommandElabM Name :=
  liftCoreM <| realizeGlobalConstNoOverloadWithInfo declId

private def loadFrontendRegistration
    (frontendId : Syntax) :
    CommandElabM FrontendRegistration := do
  let frontendDecl ← resolveFrontendDeclName frontendId
  liftCoreM <| unsafe evalConstCheck FrontendRegistration ``FrontendRegistration frontendDecl

elab_rules (kind := applyAdFrontendCmd) : command
  | `(attribute [ad_frontend $frontendId:ident] $declId:ident) => do
      let declName ← resolveTargetDeclName declId
      let registration ← loadFrontendRegistration frontendId
      let registration ← liftCoreM <| validateFrontendRegistration declName registration
      liftCoreM <| registerFrontendRegistration declName registration
      match registration.runtimeFrontend with
      | some runtimeFrontend =>
          synthesizeStructuredFrontendCompanions declName runtimeFrontend
      | none =>
          pure ()

end Tyr.AD.JaxprLike
