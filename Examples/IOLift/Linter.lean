import Lean
import Tyr.Typed.Tensor

/-! Warns when a binding of type `IO (Tensor _)` is referenced more than once.
Such a binding is an action, not a value: every reference runs the native op again. -/

open Lean Elab Command Linter

register_option linter.ioTensorReuse : Bool := {
  defValue := true
  descr := "warn when an `IO (Tensor _)` binding is used more than once"
}

namespace Examples.IOLift

private structure Binder where
  stx : Syntax
  type : Expr
  ctx : ContextInfo
  lctx : LocalContext

private def isIOTensor (ty : Expr) : MetaM Bool := do
  -- No `whnf` here: `IO` is itself reducible and would unfold to `EStateM`.
  let ty := ty.consumeMData
  unless ty.isAppOfArity ``IO 1 do return false
  -- Unfold aliases such as `abbrev T s := Tensor ⟨s, …⟩` before checking the head.
  return (← Meta.withReducible (Meta.whnf ty.appArg!)).isAppOf ``torch.Tensor

/-- `(← e)` actions owned by one `do` statement, skipping nested statements. -/
private partial def liftedActions (stx : Syntax) : Array Syntax :=
  stx.getArgs.foldl (init := #[]) fun acc arg =>
    if arg.isOfKind ``Lean.Parser.Term.doSeqItem then acc
    else if arg.isOfKind ``Lean.Parser.Term.liftMethod then acc.push arg[1] ++ liftedActions arg[1]
    else acc ++ liftedActions arg

private partial def doItems (stx : Syntax) : Array Syntax :=
  let inner := stx.getArgs.foldl (· ++ doItems ·) #[]
  if stx.isOfKind ``Lean.Parser.Term.doSeqItem then inner.push stx else inner

/-- Flags the same `IO (Tensor _)` action lifted twice in one statement, e.g. `(← f x) + (← f x)`. -/
private def checkRepeatedLifts (stx : Syntax) : CommandElabM Unit := do
  let terms := (← getInfoTrees).foldl (init := #[]) fun acc tree =>
    tree.foldInfo (init := acc) fun ctx info acc =>
      match info with
      | .ofTermInfo ti => acc.push (ctx, ti)
      | _ => acc
  for item in doItems stx do
    let actions := liftedActions item
    for h : i in [0:actions.size] do
      let e := actions[i]
      unless actions[:i].toArray.any (·.structEq e) do
        let n := actions.filter (·.structEq e) |>.size
        if n < 2 then continue
        let some (ctx, ti) := terms.find? fun (_, ti) =>
            ti.stx.getPos? == e.getPos? && ti.stx.getTailPos? == e.getTailPos?
          | continue
        let isTensor ← ctx.runMetaM ti.lctx do
          isIOTensor (← instantiateMVars (← Meta.inferType ti.expr))
        if isTensor then
          logLint linter.ioTensorReuse e
            m!"`{e}` is run {n} times in one statement. Bind it once with `let _ ← …` to reuse the value."

def ioTensorReuse : Linter where
  run stx := do
    unless getLinterValue linter.ioTensorReuse (← getLinterOptions) do return
    checkRepeatedLifts stx
    for tree in ← getInfoTrees do
      let mut binders : Std.HashMap FVarId Binder := {}
      let mut uses : Std.HashMap FVarId (Std.HashSet String.Pos.Raw) := {}
      let mut mctxs : Array MetavarContext := #[]
      for (ctx, ti) in tree.foldInfo (init := #[]) fun ctx info acc =>
          match info with
          | .ofTermInfo ti => acc.push (ctx, ti)
          | _ => acc do
        mctxs := mctxs.push ctx.mctx
        let .fvar id := ti.expr | continue
        if ti.isBinder then
          if let some decl := ti.lctx.find? id then
            binders := binders.insertIfNew id ⟨ti.stx, decl.type, ctx, ti.lctx⟩
        else if let some pos := ti.stx.getPos? then
          -- Elaboration can record one reference several times; count source positions.
          uses := uses.alter id fun s => some ((s.getD {}).insert pos)
      for (id, b) in binders do
        let n := (uses.getD id {}).size
        if n < 2 then continue
        -- Binder types are often assigned after the binder itself is recorded.
        let some type := mctxs.reverse.findSome? fun m =>
            let t := (instantiateMVarsCore m b.type).1
            if t.hasExprMVar then none else some t
          | continue
        if ← b.ctx.runMetaM b.lctx (isIOTensor type) then
          logLint linter.ioTensorReuse b.stx
            m!"`{b.stx}` is an IO action used {n} times; each use reruns it. Bind it with `←` to reuse the value."

initialize addLinter ioTensorReuse

end Examples.IOLift
