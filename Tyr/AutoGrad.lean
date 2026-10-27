import Lean.Compiler.IR.Basic
import Lean.Compiler.IR.FreeVars
import Lean.Compiler.IR.NormIds
import Lean.Compiler.IR.Format
import Lean.Compiler.IR.CompilerM
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Var
import Tyr.Manifolds

/-!
# Tyr.AutoGrad

`Tyr.AutoGrad` implements automatic differentiation utilities over Lean IR for Tyr.
It provides IR-level transformation machinery used to build differentiable programs,
including handling for differentiable, static, and frozen parameters.

## Major Components

- Lean IR rewriting helpers (`replaceArg`, `replaceExpr`, `replaceVar`).
- AD state/context management (`ADContext`, `ADM`) for tangent/cotangent bookkeeping.
- Parameter participation control via `ParamKind` (`diff`, `static`, `frozen`).
- Integration points with Tyr GPU codegen IR/variables and manifold abstractions.

## Scope

This is a backend/compiler-facing module used by AD and transformation infrastructure.
Most model code should consume higher-level APIs rather than manipulating this layer directly.
-/

namespace Tyr.AD

open torch (Static TensorStruct)
open Lean.IR

-- Manifold types (DifferentiableManifold, Stiefel, Orthogonal, Grassmann, etc.)
-- are available via `import Tyr.Manifolds`

/-! ## Parameter Kinds for Differentiation

Parameters can be:
- `diff`: Differentiable (receives tangent/cotangent)
- `static`: Non-differentiable, passed through unchanged (like `torch.Static`)
- `frozen`: Forward-only tensor, no gradient update (like `torch.Frozen`)
-/

/-- Describes how a parameter participates in differentiation -/
inductive ParamKind where
  | diff      -- Differentiable: needs tangent (JVP) or cotangent (VJP)
  | static    -- Non-differentiable: passed through unchanged
  | frozen    -- Forward-only: participates in forward, but no gradient update
  deriving Repr, BEq, Inhabited

/-- Replace variable in Argument -/
def replaceArg (a : Arg) (old new : VarId) : Arg :=
  match a with
  | .var x => if x.idx == old.idx then .var new else a
  | .erased => .erased

/-- Replace variable in Expression -/
def replaceExpr (e : Expr) (old new : VarId) : Expr :=
  match e with
  | .ctor i args => .ctor i (args.map (replaceArg · old new))
  | .reset n x => .reset n (if x.idx == old.idx then new else x)
  | .reuse x i u args => .reuse (if x.idx == old.idx then new else x) i u (args.map (replaceArg · old new))
  | .proj i x => .proj i (if x.idx == old.idx then new else x)
  | .uproj i x => .uproj i (if x.idx == old.idx then new else x)
  | .sproj n o x => .sproj n o (if x.idx == old.idx then new else x)
  | .fap c args => .fap c (args.map (replaceArg · old new))
  | .pap c args => .pap c (args.map (replaceArg · old new))
  | .ap x args => .ap (if x.idx == old.idx then new else x) (args.map (replaceArg · old new))
  | .box t x => .box t (if x.idx == old.idx then new else x)
  | .unbox x => .unbox (if x.idx == old.idx then new else x)
  | .lit l => .lit l
  | .isShared x => .isShared (if x.idx == old.idx then new else x)

/-- Replace variable in FnBody -/
partial def replaceVar (b : FnBody) (old new : VarId) : FnBody :=
  match b with
  | .vdecl x ty e rest =>
    .vdecl x ty (replaceExpr e old new) (replaceVar rest old new)
  | .jdecl j xs v rest =>
    .jdecl j xs (replaceVar v old new) (replaceVar rest old new)
  | .set x i y rest =>
    .set (if x.idx == old.idx then new else x) i (replaceArg y old new) (replaceVar rest old new)
  | .setTag x i rest =>
    .setTag (if x.idx == old.idx then new else x) i (replaceVar rest old new)
  | .uset x i y rest =>
    .uset (if x.idx == old.idx then new else x) i (if y.idx == old.idx then new else y) (replaceVar rest old new)
  | .sset x i o y ty rest =>
    .sset (if x.idx == old.idx then new else x) i o (if y.idx == old.idx then new else y) ty (replaceVar rest old new)
  | .inc x n c p rest =>
    .inc (if x.idx == old.idx then new else x) n c p (replaceVar rest old new)
  | .dec x n c p rest =>
    .dec (if x.idx == old.idx then new else x) n c p (replaceVar rest old new)
  | .del x rest =>
    .del (if x.idx == old.idx then new else x) (replaceVar rest old new)
  | .case tid x xType alts =>
    .case tid (if x.idx == old.idx then new else x) xType (alts.map fun
      | .ctor info b => .ctor info (replaceVar b old new)
      | .default b => .default (replaceVar b old new))
  | .jmp j args =>
    .jmp j (args.map (replaceArg · old new))
  | .ret x =>
    .ret (replaceArg x old new)
  | .unreachable => .unreachable

-- AD Context and Monad
structure ADContext where
  tangents : Std.HashMap Nat Nat := {}
  cotangents : Std.HashMap Nat Nat := {}
  varTypes : Std.HashMap Nat IRType := {}
  nextVarIdx : Nat := 0
  /-- When true, missing tangents are treated as an AD error. -/
  strictMissingTangents : Bool := false
  /-- Accumulator for backward pass -/
  pullbackStack : Array (FnBody → FnBody) := #[]
  /-- Variables that are static (non-differentiable) - skip tangent/cotangent -/
  staticVars : Std.HashSet Nat := {}
  deriving Inhabited

abbrev ADM := StateT ADContext Lean.CoreM

def getFreshVarIdx : ADM Nat := do
  let s ← get
  let idx := s.nextVarIdx
  modify fun s => { s with nextVarIdx := idx + 1 }
  return idx

def getFreshVar : ADM VarId := do
  return { idx := ← getFreshVarIdx }

def getFreshGpuVar : ADM Tyr.GPU.Codegen.VarId := do
  return { idx := ← getFreshVarIdx }

def setVarType (x : VarId) (ty : IRType) : ADM Unit := do
  modify fun s => { s with varTypes := s.varTypes.insert x.idx ty }

def getVarType? (x : VarId) : ADM (Option IRType) := do
  return (← get).varTypes.get? x.idx

def getVarTypeD (x : VarId) (fallback : IRType := .object) : ADM IRType := do
  return (← getVarType? x).getD fallback

def getArgTypeD (arg : Arg) (fallback : IRType := .object) : ADM IRType := do
  match arg with
  | .var v => getVarTypeD v fallback
  | .erased => return .erased

def getTangentIdx (idx : Nat) : ADM Nat := do
  let s ← get
  match s.tangents.get? idx with
  | some t => return t
  | none =>
    if s.strictMissingTangents then
      throwError s!"AD transform failed: missing tangent for IR variable {idx}. " ++
        "Ensure all operations on the path to the function output have JVP rules."
    else
      return idx

def getTangent (x : VarId) : ADM VarId := do
  return { idx := ← getTangentIdx x.idx }

def getGpuTangent (x : Tyr.GPU.Codegen.VarId) : ADM Tyr.GPU.Codegen.VarId := do
  return { idx := ← getTangentIdx x.idx }

def setTangentIdx (x idx : Nat) : ADM Unit := do
  modify fun s => { s with tangents := s.tangents.insert x idx }

def setTangent (x dx : VarId) : ADM Unit := do
  setTangentIdx x.idx dx.idx
  match (← getVarType? x), (← getVarType? dx) with
  | some ty, none => setVarType dx ty
  | _, _ => pure ()

def setGpuTangent (x dx : Tyr.GPU.Codegen.VarId) : ADM Unit := do
  setTangentIdx x.idx dx.idx

def getCotangentIdx (idx : Nat) : ADM Nat := do
  let s ← get
  match s.cotangents.get? idx with
  | some dx => return dx
  | none => do
    let dx ← getFreshVarIdx
    modify fun s => { s with cotangents := s.cotangents.insert idx dx }
    return dx

def getCotangent (x : VarId) : ADM VarId := do
  let s ← get
  match s.cotangents.get? x.idx with
  | some dx => return { idx := dx }
  | none =>
    let dx ← getFreshVar
    modify fun s => { s with cotangents := s.cotangents.insert x.idx dx.idx }
    match (← getVarType? x) with
    | some ty => setVarType dx ty
    | none => pure ()
    return dx

def getGpuCotangent (x : Tyr.GPU.Codegen.VarId) : ADM Tyr.GPU.Codegen.VarId := do
  return { idx := ← getCotangentIdx x.idx }

def setCotangentIdx (x idx : Nat) : ADM Unit := do
  modify fun s => { s with cotangents := s.cotangents.insert x idx }

def setCotangent (x dx : VarId) : ADM Unit := do
  setCotangentIdx x.idx dx.idx
  match (← getVarType? x), (← getVarType? dx) with
  | some ty, none => setVarType dx ty
  | _, _ => pure ()

def setGpuCotangent (x dx : Tyr.GPU.Codegen.VarId) : ADM Unit := do
  setCotangentIdx x.idx dx.idx

/-- Check if a variable is marked as static (non-differentiable) -/
def isStaticVar (idx : Nat) : ADM Bool := do
  return (← get).staticVars.contains idx

/-- Mark a variable as static (non-differentiable) -/
def markStatic (idx : Nat) : ADM Unit := do
  modify fun s => { s with staticVars := s.staticVars.insert idx }

/-- Mark a VarId as static -/
def markStaticVar (x : VarId) : ADM Unit := markStatic x.idx

-- AD Rules for Lean IR
abbrev JVPBuilder := FnBody → FnBody
abbrev JVPRule := Array Arg → Array Arg → IRType → ADM (JVPBuilder × VarId × VarId)

abbrev VJPBuilder := FnBody → FnBody
abbrev VJPRule := Array Arg → VarId → IRType → ADM (VJPBuilder × Array VarId)

structure ADRegistry where
  jvpRules : Std.HashMap Lean.Name JVPRule := {}
  vjpRules : Std.HashMap Lean.Name VJPRule := {}
  deriving Inhabited

initialize adRegistry : Lean.EnvExtension ADRegistry ←
  Lean.registerEnvExtension (pure {})

-- GPU AD Registry
/--
  GPU VJP Rule Type:
  Arguments:
  - dout : Cotangent of the output
  - dins : Array of cotangents of the inputs
  - ctx  : Array of primal variables preserved from forward pass
  Returns:
  - Array KStmt: The statements implementing the backward pass
-/
abbrev GpuVJPRule := Tyr.GPU.Codegen.VarId → Array Tyr.GPU.Codegen.VarId → Array Tyr.GPU.Codegen.VarId → ADM (Array Tyr.GPU.Codegen.KStmt)

/--
  GPU AD Registry for custom operations.
  Unlike Lean IR which uses Name, GPU IR operations are typically Enums (BinaryOp, UnaryOp).
  We map a String key (e.g. "UnaryOp.Sin") to the rule.
-/
structure GpuADRegistry where
  /-- Map from Op Name to VJP Rule -/
  vjpRules : Std.HashMap String GpuVJPRule := {}
  deriving Inhabited

initialize gpuAdRegistry : Lean.EnvExtension GpuADRegistry ←
  Lean.registerEnvExtension (pure {})

def registerJVPRule (fn : Lean.Name) (rule : JVPRule) : Lean.CoreM Unit := do
  Lean.modifyEnv fun env => adRegistry.modifyState env fun s =>
    { s with jvpRules := s.jvpRules.insert fn rule }

def registerVJPRule (fn : Lean.Name) (rule : VJPRule) : Lean.CoreM Unit := do
  Lean.modifyEnv fun env => adRegistry.modifyState env fun s =>
    { s with vjpRules := s.vjpRules.insert fn rule }

def getJVPRule (fn : Lean.Name) : Lean.CoreM (Option JVPRule) := do
  return (adRegistry.getState (← Lean.getEnv)).jvpRules.get? fn

def getVJPRule (fn : Lean.Name) : Lean.CoreM (Option VJPRule) := do
  return (adRegistry.getState (← Lean.getEnv)).vjpRules.get? fn

-- Lean VJP Registration Helper

def boxTupleFieldIfScalar (v : VarId) : ADM (Arg × (FnBody → FnBody)) := do
  let ty ← getVarTypeD v .tobject
  if ty.isScalar then
    let boxed ← getFreshVar
    setVarType boxed .tobject
    let builder := fun b =>
      FnBody.vdecl boxed IRType.tobject (Expr.box ty v) b
    return (.var boxed, builder)
  else
    return (.var v, id)

def coerceTupleFieldToType (field : VarId) (targetTy : IRType) : ADM (VarId × (FnBody → FnBody)) := do
  if targetTy.isScalar then
    let unboxed ← getFreshVar
    setVarType unboxed targetTy
    let builder := fun b =>
      FnBody.vdecl unboxed targetTy (Expr.unbox field) b
    return (unboxed, builder)
  else
    return (field, id)

partial def unpackTupleValues (res : VarId) (tys : List IRType) : ADM (List VarId × (FnBody → FnBody)) := do
  match tys with
  | [] => return ([], id)
  | [ty] =>
    let (val, valBuilder) ← coerceTupleFieldToType res ty
    return ([val], valBuilder)
  | ty :: tys' =>
    let headField ← getFreshVar
    setVarType headField .tobject
    let (headVal, headValBuilder) ← coerceTupleFieldToType headField ty
    let tailField ← getFreshVar
    setVarType tailField .tobject
    let (tailVals, tailBuilder) ← unpackTupleValues tailField tys'
    let builder := fun b =>
      FnBody.vdecl headField IRType.tobject (Expr.proj 0 res) <|
      headValBuilder <|
      FnBody.vdecl tailField IRType.tobject (Expr.proj 1 res) <|
      tailBuilder b
    return (headVal :: tailVals, builder)

/-- Register a VJP rule with parameter kinds.
    Static parameters are excluded from cotangent computation. -/
def registerLeanVJPRuleWithKinds (primalFn vjpFn : Lean.Name)
    (paramKinds : Array ParamKind := #[]) : Lean.CoreM Unit := do
  registerVJPRule primalFn fun args dy _resTy => do
    let callArgs := args.push (Arg.var dy)
    let mut diffTys : List IRType := []

    -- Create fresh vars for cotangents (only for diff params)
    for i in [:args.size] do
      let kind := paramKinds.getD i .diff
      if kind == .diff then
        let dxTy ← getArgTypeD args[i]! .tobject
        diffTys := diffTys ++ [dxTy]
      else
        -- Also mark the arg var itself as static if it's a var
        match args[i]! with
        | .var v => markStatic v.idx
        | _ => pure ()

    let (diffVals, builder) ←
      match diffTys with
      | [] =>
        let res ← getFreshVar
        setVarType res IRType.object
        pure ([], fun rest => FnBody.vdecl res IRType.object (Expr.fap vjpFn callArgs) rest)
      | [dxTy] =>
        let resTy := dxTy
        let res ← getFreshVar
        setVarType res resTy
        pure ([res], fun rest => FnBody.vdecl res resTy (Expr.fap vjpFn callArgs) rest)
      | _ =>
        let res ← getFreshVar
        setVarType res IRType.object
        let (vals, unpacker) ← unpackTupleValues res diffTys
        pure (vals, fun rest =>
          FnBody.vdecl res IRType.object (Expr.fap vjpFn callArgs) <|
          unpacker rest)

    let mut dxs : Array VarId := Array.replicate args.size { idx := 0 }
    let mut nextDiff := diffVals
    for i in [:args.size] do
      let kind := paramKinds.getD i .diff
      if kind == .diff then
        match nextDiff with
        | [] =>
          throwError s!"VJP rule arity mismatch for `{primalFn}`: missing cotangent value for argument {i}"
        | v :: vs =>
          dxs := dxs.set! i v
          nextDiff := vs
      else
        let dummyTy ← getArgTypeD args[i]! .tobject
        let dummyVar ← getFreshVar
        setVarType dummyVar dummyTy
        markStatic dummyVar.idx
        dxs := dxs.set! i dummyVar

    if !nextDiff.isEmpty then
      throwError s!"VJP rule arity mismatch for `{primalFn}`: too many cotangent outputs"

    return (builder, dxs)

/-- Register a VJP rule (all params differentiable) -/
def registerLeanVJPRule (primalFn vjpFn : Lean.Name) : Lean.CoreM Unit := do
  registerLeanVJPRuleWithKinds primalFn vjpFn #[]

/-- Register a JVP rule with parameter kinds.
    Static parameters don't receive tangent arguments. -/
def registerLeanJVPRuleWithKinds (primalFn jvpFn : Lean.Name)
    (paramKinds : Array ParamKind := #[]) : Lean.CoreM Unit := do
  registerJVPRule primalFn fun args tans resTy => do
    let resPair ← getFreshVar
    let expectedTangents :=
      (Array.range args.size).foldl (init := 0) fun n i =>
        if paramKinds.getD i .diff == .diff then n + 1 else n
    if tans.size != expectedTangents then
      throwError s!"JVP rule arity mismatch for `{primalFn}`: expected {expectedTangents} tangents, got {tans.size}"

    -- Build call args: primal args interleaved with tangents for diff params only
    let mut callArgs : Array Arg := #[]
    let mut tanIdx := 0
    for i in [:args.size] do
      callArgs := callArgs.push args[i]!
      let kind := paramKinds.getD i .diff
      if kind == .diff then
        -- Only add tangent for differentiable params
        if h : tanIdx < tans.size then
          callArgs := callArgs.push tans[tanIdx]
          tanIdx := tanIdx + 1
      -- static/frozen params: no tangent arg added

    let boxedPrim ← getFreshVar
    setVarType boxedPrim .tobject
    let prim ←
      if resTy.isScalar then do
        let p ← getFreshVar
        setVarType p resTy
        pure p
      else
        pure boxedPrim
    let boxedTan ← getFreshVar
    setVarType boxedTan .tobject
    let tan ←
      if resTy.isScalar then do
        let t ← getFreshVar
        setVarType t resTy
        pure t
      else
        pure boxedTan

    let builder := fun rest =>
      FnBody.vdecl resPair IRType.object (Expr.fap jvpFn callArgs) <|
      FnBody.vdecl boxedPrim IRType.tobject (Expr.proj 0 resPair) <|
      if resTy.isScalar then
        FnBody.vdecl prim resTy (Expr.unbox boxedPrim) <|
        FnBody.vdecl boxedTan IRType.tobject (Expr.proj 1 resPair) <|
        FnBody.vdecl tan resTy (Expr.unbox boxedTan) <|
        rest
      else
        FnBody.vdecl boxedTan IRType.tobject (Expr.proj 1 resPair) <|
        rest

    return (builder, prim, tan)

/-- Register a JVP rule (all params differentiable) -/
def registerLeanJVPRule (primalFn jvpFn : Lean.Name) : Lean.CoreM Unit := do
  registerLeanJVPRuleWithKinds primalFn jvpFn #[]

-- JVP Interpreter
partial def jvp (b : FnBody) : ADM FnBody := do
  match b with
  | .vdecl x ty e rest =>
    setVarType x ty
    match e with
    | .fap fn args =>
      let rule? ← getJVPRule fn
      match rule? with
      | some rule =>
        -- Only get tangents for non-static args
        let tanArgs ← args.filterMapM fun
          | Arg.var v => do
            if (← isStaticVar v.idx) then
              return none  -- Skip tangent for static vars
            else
              return some (Arg.var (← getTangent v))
          | Arg.erased => return none
        let (builder, prim, tan) ← rule args tanArgs ty
        setVarType prim ty
        setVarType tan ty
        setTangent x tan
        let rest' ← jvp rest
        return builder (replaceVar rest' x prim)
      | none =>
        let rest' ← jvp rest
        return .vdecl x ty e rest'
    | _ =>
      let rest' ← jvp rest
      return .vdecl x ty e rest'
  | .ret (.var x) =>
    let dx ← getTangent x
    let (xArg, xBuilder) ← boxTupleFieldIfScalar x
    let (dxArg, dxBuilder) ← boxTupleFieldIfScalar dx
    let pair ← getFreshVar
    setVarType pair IRType.object
    let ctorInfo : CtorInfo := { name := `Prod.mk, cidx := 0, size := 2, usize := 0, ssize := 0 }
    return xBuilder <| dxBuilder <|
      .vdecl pair IRType.object (.ctor ctorInfo #[xArg, dxArg]) (.ret (.var pair))
  | other => return other

-- VJP Interpreter
partial def vjp (b : FnBody) : ADM FnBody := do
  match b with
  | .vdecl x ty e rest =>
    setVarType x ty
    match e with
    | .fap fn args =>
      let rule? ← getVJPRule fn
      match rule? with
      | some rule =>
        let dx ← getCotangent x
        let (pbBuilder, argCotans) ← rule args dx ty

        -- Reject repeated differentiable arguments to avoid silently dropping gradient contributions.
        -- Proper accumulation for aliasing arguments needs explicit add-accumulation support.
        let mut seen : Std.HashSet Nat := {}
        for i in [:args.size] do
          match args[i]! with
          | .var v =>
            if !(← isStaticVar v.idx) then
              if seen.contains v.idx then
                throwError s!"AD transform failed: VJP for `{fn}` uses repeated differentiable argument {v.idx}. " ++
                  "Aliased-argument cotangent accumulation is not implemented yet."
              seen := seen.insert v.idx
          | _ => pure ()

        -- Register cotangents for arguments (skip static vars)
        for i in [:args.size] do
           match args[i]! with
           | .var v =>
             -- Skip cotangent for static vars
             if !(← isStaticVar v.idx) then
               setCotangent v argCotans[i]!
           | _ => pure ()

        modify fun s => { s with pullbackStack := s.pullbackStack.push pbBuilder }
        let rest' ← vjp rest
        return .vdecl x ty e rest'
      | none =>
        let rest' ← vjp rest
        return .vdecl x ty e rest'
    | _ =>
      let rest' ← vjp rest
      return .vdecl x ty e rest'
  | .ret (.var x) =>
    let dx ← getCotangent x
    let s ← get
    -- Construct backward pass
    let backward := s.pullbackStack.foldl (fun acc pb => pb acc) (.ret (.var dx))
    return backward
  | other => return other

/-- Find the returned variable for simple linear IR bodies. -/
partial def findReturnVar? (b : FnBody) : Option VarId :=
  match b with
  | .vdecl _ _ _ rest => findReturnVar? rest
  | .jdecl _ _ _ rest => findReturnVar? rest
  | .set _ _ _ rest => findReturnVar? rest
  | .setTag _ _ rest => findReturnVar? rest
  | .uset _ _ _ rest => findReturnVar? rest
  | .sset _ _ _ _ _ rest => findReturnVar? rest
  | .inc _ _ _ _ rest => findReturnVar? rest
  | .dec _ _ _ _ rest => findReturnVar? rest
  | .del _ rest => findReturnVar? rest
  | .ret (.var x) => some x
  | _ => none

/-- Replace each return with the provided function body. -/
partial def replaceRetWith (b replacement : FnBody) : FnBody :=
  match b with
  | .vdecl x ty e rest =>
    .vdecl x ty e (replaceRetWith rest replacement)
  | .jdecl j xs v rest =>
    .jdecl j xs (replaceRetWith v replacement) (replaceRetWith rest replacement)
  | .set x i y rest =>
    .set x i y (replaceRetWith rest replacement)
  | .setTag x i rest =>
    .setTag x i (replaceRetWith rest replacement)
  | .uset x i y rest =>
    .uset x i y (replaceRetWith rest replacement)
  | .sset x i o y ty rest =>
    .sset x i o y ty (replaceRetWith rest replacement)
  | .inc x n c p rest =>
    .inc x n c p (replaceRetWith rest replacement)
  | .dec x n c p rest =>
    .dec x n c p (replaceRetWith rest replacement)
  | .del x rest =>
    .del x (replaceRetWith rest replacement)
  | .case tid x xType alts =>
    .case tid x xType (alts.map fun
      | .ctor info body => .ctor info (replaceRetWith body replacement)
      | .default body => .default (replaceRetWith body replacement))
  | .ret _ => replacement
  | .jmp _ _ => b
  | .unreachable => b

partial def mkTupleReturnAux (vars : List VarId) : ADM (VarId × (FnBody → FnBody)) := do
  match vars with
  | [] =>
    throwError "AD transform failed: cannot build tuple return for empty cotangent list"
  | [v] =>
    return (v, id)
  | v :: vs =>
    let (tail, tailBuilder) ← mkTupleReturnAux vs
    let pair ← getFreshVar
    setVarType pair IRType.object
    let (vArg, vArgBuilder) ← boxTupleFieldIfScalar v
    let (tailArg, tailArgBuilder) ← boxTupleFieldIfScalar tail
    let ctorInfo : CtorInfo := { name := `Prod.mk, cidx := 0, size := 2, usize := 0, ssize := 0 }
    let builder := fun b =>
      tailBuilder <|
      vArgBuilder <|
      tailArgBuilder <|
      .vdecl pair IRType.object (.ctor ctorInfo #[vArg, tailArg]) b
    return (pair, builder)

def mkTupleReturn (vars : Array VarId) : ADM FnBody := do
  let (root, builder) ← mkTupleReturnAux vars.toList
  return builder (.ret (.var root))

def normalizeDeclParamIds (decl : Decl) : Decl :=
  match decl with
  | .fdecl f params ty body info =>
    let (_, body', newParams) :=
      params.foldl
        (fun (nextIdx, bodyAcc, paramsAcc) p =>
          let newVar : VarId := { idx := nextIdx }
          let bodyAcc :=
            if p.x.idx == newVar.idx then bodyAcc else replaceVar bodyAcc p.x newVar
          let newParam := { p with x := newVar }
          (nextIdx + 1, bodyAcc, paramsAcc.push newParam))
        (1, body, #[])
    .fdecl f newParams ty body' info
  | other => other

def linearizeWithKinds (decl : Decl) (paramKinds : Array ParamKind := #[]) : ADM Decl := do
  let startIdx := decl.maxIndex + 1
  modify fun s => {
    s with
      nextVarIdx := max s.nextVarIdx startIdx
      strictMissingTangents := true
  }
  match decl with
  | Decl.fdecl f params _ty body info =>
    let mut tanParams := #[]
    for i in [:params.size] do
      let p := params[i]!
      setVarType p.x p.ty
      let kind := paramKinds.getD i .diff
      if kind == .diff then
        let dp ← getFreshVar
        setVarType dp p.ty
        setTangent p.x dp
        tanParams := tanParams.push { p with x := dp }
      else
        markStatic p.x.idx
    let body' ← jvp body
    return Decl.fdecl (Lean.Name.mkStr f "jvp") (params ++ tanParams) IRType.object body' info
  | other => return other

def linearize (decl : Decl) : ADM Decl := do
  linearizeWithKinds decl #[]

def transposeWithKinds (decl : Decl) (paramKinds : Array ParamKind := #[]) : ADM Decl := do
  let startIdx := decl.maxIndex + 1
  modify fun s => { s with nextVarIdx := max s.nextVarIdx startIdx }
  match decl with
  | Decl.fdecl f params ty body info =>
    for p in params do
      setVarType p.x p.ty
    for i in [:params.size] do
      let kind := paramKinds.getD i .diff
      if kind != .diff then
        markStatic params[i]!.x.idx

    let some outVar := findReturnVar? body
      | throwError s!"AD transform failed: `{f}` does not end in `ret (.var _)`"

    let dy ← getFreshVar
    setVarType dy ty
    setCotangent outVar dy
    let transformedBody ← vjp body

    let mut paramCotans := #[]
    let stateAfter ← get
    for i in [:params.size] do
      let p := params[i]!
      let kind := paramKinds.getD i .diff
      if kind == .diff then
        match stateAfter.cotangents.get? p.x.idx with
        | some dxIdx => paramCotans := paramCotans.push { idx := dxIdx }
        | none =>
          throwError s!"AD transform failed for `{f}`: missing cotangent for parameter {i}. " ++
            "Mark non-differentiable parameters as `static` or provide a custom VJP rule."

    if paramCotans.isEmpty then
      throwError s!"AD transform failed for `{f}`: no differentiable parameters for VJP generation"

    let retBody ← mkTupleReturn paramCotans
    let finalBody := replaceRetWith transformedBody retBody
    let retTy ←
      match paramCotans.toList with
      | [dx] => getVarTypeD dx IRType.object
      | _ => pure IRType.object
    let dyParam : Param := { x := dy, borrow := false, ty := ty }
    return Decl.fdecl (Lean.Name.mkStr f "vjp") (params.push dyParam) retTy finalBody info
  | other => return other

def deriveAndRegisterADRules (primalFn : Lean.Name) (paramKinds : Array ParamKind := #[]) : Lean.CoreM Unit := do
  let primalDecl ← Lean.IR.getDecl primalFn
  let (jvpDeclRaw, _) ← (linearizeWithKinds primalDecl paramKinds).run {}
  let (vjpDeclRaw, _) ← (transposeWithKinds primalDecl paramKinds).run {}
  let jvpDecl := (normalizeDeclParamIds jvpDeclRaw).normalizeIds
  let vjpDecl := (normalizeDeclParamIds vjpDeclRaw).normalizeIds

  let jvpFn := jvpDecl.name
  let vjpFn := vjpDecl.name

  if !jvpDecl.uniqueIds then
    throwError m!"autodiff internal error: non-unique IDs in `{jvpFn}`\n{jvpDecl}"
  if !vjpDecl.uniqueIds then
    throwError m!"autodiff internal error: non-unique IDs in `{vjpFn}`\n{vjpDecl}"

  if (← Lean.IR.containsDecl jvpFn) then
    throwError s!"cannot derive AD for `{primalFn}`: declaration `{jvpFn}` already exists"
  if (← Lean.IR.containsDecl vjpFn) then
    throwError s!"cannot derive AD for `{primalFn}`: declaration `{vjpFn}` already exists"

  Lean.IR.addDecl jvpDecl
  Lean.IR.addDecl vjpDecl

  registerLeanJVPRuleWithKinds primalFn jvpFn paramKinds
  registerLeanVJPRuleWithKinds primalFn vjpFn paramKinds

-- GPU Registration helpers
def registerGpuVJPRule (opName : String) (rule : GpuVJPRule) : Lean.CoreM Unit := do
  Lean.modifyEnv fun env => gpuAdRegistry.modifyState env fun s =>
    { s with vjpRules := s.vjpRules.insert opName rule }

def getGpuVJPRule (opName : String) : Lean.CoreM (Option GpuVJPRule) := do
  return (gpuAdRegistry.getState (← Lean.getEnv)).vjpRules.get? opName

-- Attributes

/-- Parse a list of static parameter indices into ParamKind array -/
def parseStaticIndices (staticIndices : Array Nat) (numParams : Nat) : Array ParamKind :=
  Array.range numParams |>.map fun i =>
    if staticIndices.contains i then .static else .diff

def resolvePrimalNameForAttr (ruleDecl primalFn : Lean.Name) : Lean.CoreM Lean.Name := do
  let env ← Lean.getEnv
  if env.find? primalFn |>.isSome then
    return primalFn
  let scopedName := ruleDecl.getPrefix ++ primalFn
  if env.find? scopedName |>.isSome then
    return scopedName
  throwError s!"unknown declaration `{primalFn}` in AD attribute"

def getExplicitParamCount (fn : Lean.Name) : Lean.CoreM Nat := do
  let env ← Lean.getEnv
  let some declInfo := env.find? fn
    | throwError s!"unknown declaration `{fn}` in AD attribute"
  Lean.Meta.MetaM.run' do
    Lean.Meta.forallTelescopeReducing declInfo.type fun xs _ => do
      let mut count := 0
      for x in xs do
        let localDecl ← x.fvarId!.getDecl
        if localDecl.binderInfo.isExplicit then
          count := count + 1
      return count

def parseStaticIndicesChecked (fn : Lean.Name) (staticIndices : Array Nat) : Lean.CoreM (Array ParamKind) := do
  let numParams ← getExplicitParamCount fn
  for idx in staticIndices do
    if idx >= numParams then
      throwError s!"invalid static parameter index {idx} for `{fn}`; expected index < {numParams}"
  return parseStaticIndices staticIndices numParams

-- Basic syntax: @[jvp primalFn] or @[vjp primalFn]
syntax (name := jvpAttr) "jvp" ident : attr
syntax (name := vjpAttr) "vjp" ident : attr

-- Extended syntax with static params: @[jvp primalFn, static := [1, 3]]
syntax (name := jvpAttrStatic) "jvp" ident "," "static" ":=" "[" num,* "]" : attr
syntax (name := vjpAttrStatic) "vjp" ident "," "static" ":=" "[" num,* "]" : attr

-- Autodiff syntax: @[autodiff] or @[autodiff, static := [1, 3]]
syntax (name := autodiffAttr) "autodiff" : attr
syntax (name := autodiffAttrStatic) "autodiff" "," "static" ":=" "[" num,* "]" : attr

initialize
  -- Basic JVP (all params differentiable)
  Lean.registerBuiltinAttribute {
    name := `jvpAttr
    descr := "Register JVP rule"
    add := fun declName stx _ => do
      match stx with
      | `(attr| jvp $id) =>
        let primalFn ← resolvePrimalNameForAttr declName id.getId
        registerLeanJVPRule primalFn declName
      | _ => throwError "invalid jvp attribute"
  }

  -- Basic VJP (all params differentiable)
  Lean.registerBuiltinAttribute {
    name := `vjpAttr
    descr := "Register VJP rule"
    add := fun declName stx _ => do
      match stx with
      | `(attr| vjp $id) =>
        let primalFn ← resolvePrimalNameForAttr declName id.getId
        registerLeanVJPRule primalFn declName
      | _ => throwError "invalid vjp attribute"
  }

  -- JVP with static params
  Lean.registerBuiltinAttribute {
    name := `jvpAttrStatic
    descr := "Register JVP rule with static (non-differentiable) parameters"
    add := fun declName stx _ => do
      match stx with
      | `(attr| jvp $id, static := [$indices,*]) =>
        let primalFn ← resolvePrimalNameForAttr declName id.getId
        let staticIndices := indices.getElems.map fun n => n.getNat
        let paramKinds ← parseStaticIndicesChecked primalFn staticIndices
        registerLeanJVPRuleWithKinds primalFn declName paramKinds
      | _ => throwError "invalid jvp attribute with static"
  }

  -- VJP with static params
  Lean.registerBuiltinAttribute {
    name := `vjpAttrStatic
    descr := "Register VJP rule with static (non-differentiable) parameters"
    add := fun declName stx _ => do
      match stx with
      | `(attr| vjp $id, static := [$indices,*]) =>
        let primalFn ← resolvePrimalNameForAttr declName id.getId
        let staticIndices := indices.getElems.map fun n => n.getNat
        let paramKinds ← parseStaticIndicesChecked primalFn staticIndices
        registerLeanVJPRuleWithKinds primalFn declName paramKinds
      | _ => throwError "invalid vjp attribute with static"
  }

  -- Auto-derive and register both JVP and VJP from IR
  Lean.registerBuiltinAttribute {
    name := `autodiffAttr
    descr := "Automatically derive and register JVP/VJP companions for this declaration"
    applicationTime := .afterCompilation
    add := fun declName stx _ => do
      match stx with
      | `(attr| autodiff) =>
        let numParams ← getExplicitParamCount declName
        let paramKinds := parseStaticIndices #[] numParams
        deriveAndRegisterADRules declName paramKinds
      | _ => throwError "invalid autodiff attribute"
  }

  -- Auto-derive with static parameter annotations
  Lean.registerBuiltinAttribute {
    name := `autodiffAttrStatic
    descr := "Automatically derive and register JVP/VJP companions with static parameters"
    applicationTime := .afterCompilation
    add := fun declName stx _ => do
      match stx with
      | `(attr| autodiff, static := [$indices,*]) =>
        let staticIndices := indices.getElems.map fun n => n.getNat
        let paramKinds ← parseStaticIndicesChecked declName staticIndices
        deriveAndRegisterADRules declName paramKinds
      | _ => throwError "invalid autodiff attribute with static"
  }

end Tyr.AD
