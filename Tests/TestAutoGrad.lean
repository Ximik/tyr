import Tyr.AutoGrad
import Tests.Test
import LeanTest
import Lean.CoreM

namespace Tests.AutoGrad

open Tyr.AD
open LeanTest
open Lean
open Lean.IR

def runCoreMResult (x : CoreM α) : IO (Except String α) := do
  let env ← mkEmptyEnvironment
  let ctx : Core.Context := { fileName := "<test>", fileMap := default }
  let state : Core.State := { env := env }
  let eio := x.run ctx state
  let res ← EIO.toBaseIO eio
  match res with
  | .ok (value, _) => pure (.ok value)
  | .error err =>
    let msg ← err.toMessageData.toString
    pure (.error msg)

def runCoreM (x : CoreM α) : IO α := do
  match (← runCoreMResult x) with
  | .ok value => pure value
  | .error msg => throw (IO.userError msg)

def runCoreM? (x : CoreM α) : IO (Except String α) := do
  runCoreMResult x

-- 1. Define Primal Functions
@[noinline]
def square (x : Float) : Float := x * x
@[noinline]
def mul (x y : Float) : Float := x * y

-- 2. Define and Register Rules

@[vjp square]
def square_bwd (x dy : Float) : Float := 
  2.0 * x * dy

@[jvp square]
def square_fwd (x dx : Float) : Float × Float :=
  (x * x, 2.0 * x * dx)

@[vjp mul]
def mul_bwd (x y dz : Float) : Float × Float :=
  (dz * y, dz * x)

@[jvp mul]
def mul_fwd (x y dx dy : Float) : Float × Float :=
  (x * y, x * dy + y * dx)

def objId (x : Nat) : Nat := x

@[jvp objId]
def objId_fwd (x dx : Nat) : Nat × Nat := (x, dx)

@[vjp objId]
def objId_bwd (_x dy : Nat) : Nat := dy

@[autodiff]
def autoObjId (x : Nat) : Nat :=
  objId x

@[autodiff]
def autoChain (x : Float) : Float :=
  square (square x)

@[test]
def testAttributes : IO Unit := do
  -- This test passes if the file compiles and attributes are processed
  -- We could inspect the environment to see if rules are registered,
  -- but access to `getEnv` requires `CoreM` or similar, not `IO`.
  -- For now, compilation is the main check.
  return

@[test]
def testLinearize : IO Unit := do
  let x : VarId := { idx := 0 }
  let y : VarId := { idx := 5 }
  let p : Param := { x := x, borrow := false, ty := IRType.object }
  let body :=
    FnBody.vdecl y IRType.object (Expr.fap `test.square #[Arg.var x]) (
      .ret (.var y)
    )
  let decl : Decl := .fdecl `test.main #[p] IRType.object body {}

  let jvpDecl ← runCoreM (do
    registerJVPRule `test.square fun args tans _ => do
      let prim ← getFreshVar
      let tan ← getFreshVar
      let builder := fun rest =>
        FnBody.vdecl prim IRType.object (Expr.fap `test.square args) (
          FnBody.vdecl tan IRType.object (Expr.fap `test.square_tan tans) rest
        )
      return (builder, prim, tan)
    let (res, _) ← (linearize decl).run {}
    return res)

  match jvpDecl with
  | .fdecl _ params _ _ _ =>
    LeanTest.assertTrue (params.size == 2) s!"Expected primal+tangent params, got {params.size}"
    let tanParam := params[1]!
    LeanTest.assertTrue (tanParam.x.idx > y.idx)
      s!"Tangent param index should be fresh, got {tanParam.x.idx} with max primal index {y.idx}"
  | _ =>
    LeanTest.fail "Expected linearize to produce a function declaration"

@[test]
def testAutodiffAttributeCompiles : IO Unit := do
  -- Compilation validates that @[autodiff] ran after IR generation.
  return

@[test]
def testAutodiffFloatAttributeCompiles : IO Unit := do
  -- Compilation validates float autodiff companion generation.
  return

@[test]
def testDeriveAndRegisterADRulesCore : IO Unit := do
  let ok ← runCoreM (do
    registerJVPRule `test.foo fun args tans _ => do
      let prim ← getFreshVar
      let tan ← getFreshVar
      let builder := fun rest =>
        FnBody.vdecl prim IRType.object (Expr.fap `test.foo args) <|
        FnBody.vdecl tan IRType.object (Expr.fap `test.foo_tan tans) <|
        rest
      return (builder, prim, tan)

    registerVJPRule `test.foo fun args dy _ => do
      let dx ← getFreshVar
      let callArgs := args.push (.var dy)
      let builder := fun rest =>
        FnBody.vdecl dx IRType.object (Expr.fap `test.foo_bwd callArgs) rest
      return (builder, #[dx])

    let x : VarId := { idx := 1 }
    let y : VarId := { idx := 2 }
    let p : Param := { x := x, borrow := false, ty := IRType.object }
    let body := FnBody.vdecl y IRType.object (Expr.fap `test.foo #[Arg.var x]) (.ret (.var y))
    let decl : Decl := .fdecl `test.primal #[p] IRType.object body {}
    Lean.IR.addDecl decl
    deriveAndRegisterADRules `test.primal #[]

    let hasJVP := (← getJVPRule `test.primal).isSome
    let hasVJP := (← getVJPRule `test.primal).isSome
    let hasJVPDecl := (← Lean.IR.findDecl (Lean.Name.mkStr `test.primal "jvp")).isSome
    let hasVJPDecl := (← Lean.IR.findDecl (Lean.Name.mkStr `test.primal "vjp")).isSome
    return hasJVP && hasVJP && hasJVPDecl && hasVJPDecl
  )
  LeanTest.assertTrue ok "deriveAndRegisterADRules should register rules and add companion decls"

@[test]
def testVJPPullbackOrder : IO Unit := do
  let x : VarId := { idx := 0 }
  let y : VarId := { idx := 1 }
  let z : VarId := { idx := 2 }
  let body :=
    FnBody.vdecl y IRType.object (Expr.fap `rule_f #[Arg.var x]) (
      FnBody.vdecl z IRType.object (Expr.fap `rule_g #[Arg.var y]) (
        .ret (.var z)
      )
    )

  let transformed ← runCoreM (do
    registerVJPRule `rule_f fun _args dy _ => do
      let dx ← getFreshVar
      let builder := fun rest =>
        FnBody.vdecl dx IRType.object (Expr.fap `pb_f #[Arg.var dy]) rest
      return (builder, #[dx])
    registerVJPRule `rule_g fun _args dy _ => do
      let dx ← getFreshVar
      let builder := fun rest =>
        FnBody.vdecl dx IRType.object (Expr.fap `pb_g #[Arg.var dy]) rest
      return (builder, #[dx])
    let (res, _) ← (vjp body).run {}
    return res)

  let backwardStart? :=
    match transformed with
    | .vdecl _ _ _ (.vdecl _ _ _ rest) => some rest
    | _ => none

  match backwardStart? with
  | some (.vdecl _ _ (.fap fn1 _) (.vdecl _ _ (.fap fn2 _) _)) =>
    LeanTest.assertTrue (fn1 == `pb_g) s!"Expected first pullback to be pb_g, got {fn1}"
    LeanTest.assertTrue (fn2 == `pb_f) s!"Expected second pullback to be pb_f, got {fn2}"
  | _ =>
    LeanTest.fail "Expected backward pass declarations after forward declarations"

@[test]
def testVJPRejectsAliasedArgs : IO Unit := do
  let x : VarId := { idx := 0 }
  let y : VarId := { idx := 1 }
  let body :=
    FnBody.vdecl y IRType.object (Expr.fap `dup #[Arg.var x, Arg.var x]) (
      .ret (.var y)
    )

  let result ← runCoreM? (do
    registerVJPRule `dup fun _args dy _ => do
      let dx0 ← getFreshVar
      let dx1 ← getFreshVar
      let builder := fun rest =>
        FnBody.vdecl dx0 IRType.object (Expr.fap `pb0 #[Arg.var dy]) (
          FnBody.vdecl dx1 IRType.object (Expr.fap `pb1 #[Arg.var dy]) rest
        )
      return (builder, #[dx0, dx1])
    let _ ← (vjp body).run {}
    return ())

  match result with
  | .ok _ =>
    LeanTest.fail "vjp should fail for repeated differentiable arguments"
  | .error _ =>
    pure ()

@[test]
def testParseStaticIndices : IO Unit := do
  let kinds := parseStaticIndices #[1, 3] 5
  LeanTest.assertTrue (kinds.size == 5) s!"Expected 5 parameter kinds, got {kinds.size}"
  LeanTest.assertTrue (kinds[1]! == .static) "Expected parameter 1 to be static"
  LeanTest.assertTrue (kinds[3]! == .static) "Expected parameter 3 to be static"
  LeanTest.assertTrue (kinds[0]! == .diff) "Expected parameter 0 to be differentiable"

@[test]
def testNoGradRestoresOnException : IO Unit := do
  let before ← torch.autograd.is_grad_enabled
  let _ ← torch.autograd.set_grad_enabled true
  let threw ←
    try
      let (_ : Unit) ← torch.autograd.no_grad (α := Unit) do
        let enabled ← torch.autograd.is_grad_enabled
        LeanTest.assertTrue (!enabled) "grad should be disabled inside no_grad"
        throw (IO.userError "intentional failure")
      pure false
    catch _ =>
      pure true

  LeanTest.assertTrue threw "no_grad test should throw from inner action"
  let after ← torch.autograd.is_grad_enabled
  LeanTest.assertTrue after "grad mode should be restored after exception"
  let _ ← torch.autograd.set_grad_enabled before

@[test]
def testNoGradNestedRestore : IO Unit := do
  let before ← torch.autograd.is_grad_enabled
  let _ ← torch.autograd.set_grad_enabled true

  let _ ← torch.autograd.no_grad do
    let outerEnabled ← torch.autograd.is_grad_enabled
    LeanTest.assertTrue (!outerEnabled) "outer no_grad should disable grad"
    let _ ← torch.autograd.no_grad do
      let innerEnabled ← torch.autograd.is_grad_enabled
      LeanTest.assertTrue (!innerEnabled) "inner no_grad should keep grad disabled"
      pure ()
    let outerAfterInner ← torch.autograd.is_grad_enabled
    LeanTest.assertTrue (!outerAfterInner) "outer no_grad should remain disabled after inner"
    pure ()

  let after ← torch.autograd.is_grad_enabled
  LeanTest.assertTrue after "grad mode should be restored after nested no_grad"
  let _ ← torch.autograd.set_grad_enabled before

/-! ## JVP Execution Tests -/

@[test]
def testSquareJVPExecution : IO Unit := do
  -- Test that JVP rules produce correct tangents
  -- square: f(x) = x², f'(x) = 2x
  -- At x=3, dx=1: f(3)=9, f'(3)*1=6
  let (y, dy) := square_fwd 3.0 1.0
  LeanTest.assertTrue (y == 9.0) s!"square(3) should be 9, got {y}"
  LeanTest.assertTrue (dy == 6.0) s!"d/dx[x²] at x=3 should be 6, got {dy}"

@[test]
def testSquareJVPAtZero : IO Unit := do
  -- At x=0: f(0)=0, f'(0)*1=0
  let (y, dy) := square_fwd 0.0 1.0
  LeanTest.assertTrue (y == 0.0) s!"square(0) should be 0"
  LeanTest.assertTrue (dy == 0.0) s!"d/dx[x²] at x=0 should be 0"

@[test]
def testSquareJVPNegative : IO Unit := do
  -- At x=-2: f(-2)=4, f'(-2)*1=-4
  let (y, dy) := square_fwd (-2.0) 1.0
  LeanTest.assertTrue (y == 4.0) s!"square(-2) should be 4"
  LeanTest.assertTrue (dy == -4.0) s!"d/dx[x²] at x=-2 should be -4, got {dy}"

/-! ## VJP Execution Tests -/

@[test]
def testSquareVJPExecution : IO Unit := do
  -- Test that VJP rules produce correct cotangents
  -- square backward: dx = 2*x*dy
  -- At x=3, dy=1: dx = 6
  let dx := square_bwd 3.0 1.0
  LeanTest.assertTrue (dx == 6.0) s!"grad of x² at x=3 should be 6, got {dx}"

@[test]
def testSquareVJPScaled : IO Unit := do
  -- At x=3, dy=2: dx = 2*3*2 = 12
  let dx := square_bwd 3.0 2.0
  LeanTest.assertTrue (dx == 12.0) s!"grad of x² at x=3 with dy=2 should be 12, got {dx}"

/-! ## Multiplication Tests -/

@[test]
def testMulJVPExecution : IO Unit := do
  -- f(x,y) = x*y, df = x*dy + y*dx
  -- At (2,3), (dx=1,dy=0): f=6, df=3 (only contribution from dx)
  let (f, df) := mul_fwd 2.0 3.0 1.0 0.0
  LeanTest.assertTrue (f == 6.0) s!"2*3 should be 6"
  LeanTest.assertTrue (df == 3.0) s!"d/dx[x*y] at (2,3) should be 3, got {df}"

@[test]
def testMulJVPBothTangents : IO Unit := do
  -- At (2,3), (dx=1,dy=1): f=6, df = 2*1 + 3*1 = 5
  let (f, df) := mul_fwd 2.0 3.0 1.0 1.0
  LeanTest.assertTrue (f == 6.0) s!"2*3 should be 6"
  LeanTest.assertTrue (df == 5.0) s!"df should be x*dy + y*dx = 2+3 = 5, got {df}"

@[test]
def testMulVJPExecution : IO Unit := do
  -- Backward: dx = dz*y, dy = dz*x
  -- At (2,3), dz=1: dx=3, dy=2
  let (dx, dy) := mul_bwd 2.0 3.0 1.0
  LeanTest.assertTrue (dx == 3.0) s!"dx should be y=3, got {dx}"
  LeanTest.assertTrue (dy == 2.0) s!"dy should be x=2, got {dy}"

@[test]
def testMulVJPScaled : IO Unit := do
  -- At (2,3), dz=5: dx=15, dy=10
  let (dx, dy) := mul_bwd 2.0 3.0 5.0
  LeanTest.assertTrue (dx == 15.0) s!"dx should be dz*y=15, got {dx}"
  LeanTest.assertTrue (dy == 10.0) s!"dy should be dz*x=10, got {dy}"

/-! ## Chain Rule Tests -/

-- Manually compute chain rule: d/dx[f(g(x))] = f'(g(x)) * g'(x)
-- For h(x) = square(square(x)) = x^4
-- h'(x) = 4x^3

@[test]
def testChainRuleManual : IO Unit := do
  -- h(x) = (x²)² = x⁴
  -- At x=2: h(2)=16, h'(2)=4*8=32
  let x := 2.0
  let dx := 1.0

  -- Forward pass through first square
  let (y1, dy1) := square_fwd x dx      -- y1=4, dy1=4

  -- Forward pass through second square
  let (y2, dy2) := square_fwd y1 dy1    -- y2=16, dy2=2*4*4=32

  LeanTest.assertTrue (y2 == 16.0) s!"x⁴ at x=2 should be 16, got {y2}"
  LeanTest.assertTrue (dy2 == 32.0) s!"d/dx[x⁴] at x=2 should be 32, got {dy2}"

/-! ## Static Parameter Tests -/

-- A function with a static (non-differentiable) parameter
-- Multiplies x by a constant factor n
def scale (x : Float) (n : Nat) : Float := x * n.toFloat

@[autodiff, static := [1]]
def autoFloatKeep (x : Float) (_n : Nat) : Float :=
  square x

-- JVP rule: only x has a tangent, n is static
-- Note: n doesn't get a tangent parameter
@[jvp scale, static := [1]]
def scale_fwd (x dx : Float) (n : Nat) : Float × Float :=
  (x * n.toFloat, dx * n.toFloat)

-- VJP rule: only x gets a cotangent, n is static
-- Note: returns only dx, not (dx, dn)
@[vjp scale, static := [1]]
def scale_bwd (_x : Float) (n : Nat) (dy : Float) : Float :=
  dy * n.toFloat

def objKeep (x : Nat) (_tag : String) : Nat := x

@[jvp objKeep, static := [1]]
def objKeep_fwd (x dx : Nat) (_tag : String) : Nat × Nat := (x, dx)

@[vjp objKeep, static := [1]]
def objKeep_bwd (_x : Nat) (_tag : String) (dy : Nat) : Nat := dy

@[autodiff, static := [1]]
def autoObjKeep (x : Nat) (tag : String) : Nat :=
  objKeep x tag

@[test]
def testStaticJVP : IO Unit := do
  -- scale(3.0, 5) = 15.0
  -- d/dx[scale(x, 5)] at x=3 with dx=1 should be 5
  let (y, dy) := scale_fwd 3.0 1.0 5
  LeanTest.assertTrue (y == 15.0) s!"scale(3, 5) should be 15, got {y}"
  LeanTest.assertTrue (dy == 5.0) s!"d/dx[scale] at x=3, n=5 should be 5, got {dy}"

@[test]
def testStaticJVPDifferentScale : IO Unit := do
  -- scale(2.0, 10) = 20.0
  -- d/dx[scale(x, 10)] at x=2 with dx=1 should be 10
  let (y, dy) := scale_fwd 2.0 1.0 10
  LeanTest.assertTrue (y == 20.0) s!"scale(2, 10) should be 20, got {y}"
  LeanTest.assertTrue (dy == 10.0) s!"d/dx[scale] with n=10 should be 10, got {dy}"

@[test]
def testStaticVJP : IO Unit := do
  -- Backward: dx = dy * n
  -- At x=3, n=5, dy=1: dx = 5
  let dx := scale_bwd 3.0 5 1.0
  LeanTest.assertTrue (dx == 5.0) s!"grad of scale(x, 5) should be 5, got {dx}"

@[test]
def testStaticVJPScaled : IO Unit := do
  -- At x=3, n=5, dy=2: dx = 10
  let dx := scale_bwd 3.0 5 2.0
  LeanTest.assertTrue (dx == 10.0) s!"grad of scale with dy=2 should be 10, got {dx}"

-- Function with mixed parameters: (diff, static, diff)
def weightedAdd (x : Float) (weight : Nat) (y : Float) : Float :=
  x + (weight.toFloat * y)

-- JVP: x and y get tangents, weight is static
@[jvp weightedAdd, static := [1]]
def weightedAdd_fwd (x dx : Float) (weight : Nat) (y dy : Float) : Float × Float :=
  (x + weight.toFloat * y, dx + weight.toFloat * dy)

-- VJP: returns (dx, dy), no gradient for weight
@[vjp weightedAdd, static := [1]]
def weightedAdd_bwd (_x : Float) (weight : Nat) (_y : Float) (dz : Float) : Float × Float :=
  (dz, dz * weight.toFloat)

@[test]
def testMixedStaticJVP : IO Unit := do
  -- weightedAdd(2, 3, 4) = 2 + 3*4 = 14
  -- With dx=1, dy=1: dz = 1 + 3*1 = 4
  let (z, dz) := weightedAdd_fwd 2.0 1.0 3 4.0 1.0
  LeanTest.assertTrue (z == 14.0) s!"weightedAdd(2, 3, 4) should be 14, got {z}"
  LeanTest.assertTrue (dz == 4.0) s!"dz should be 1 + 3*1 = 4, got {dz}"

@[test]
def testMixedStaticVJP : IO Unit := do
  -- Backward: dx = dz, dy = dz * weight
  -- At dz=1, weight=3: dx=1, dy=3
  let (dx, dy) := weightedAdd_bwd 2.0 3 4.0 1.0
  LeanTest.assertTrue (dx == 1.0) s!"dx should be 1, got {dx}"
  LeanTest.assertTrue (dy == 3.0) s!"dy should be 3, got {dy}"

@[test]
def testStaticAttributeCompiles : IO Unit := do
  -- This test verifies that the static := [...] attribute syntax compiles
  -- The main check is that the file compiles with the static attribute syntax
  return

@[test]
def testAutodiffStaticAttributeCompiles : IO Unit := do
  -- Compilation validates that @[autodiff, static := [...]] ran successfully.
  return

@[test]
def testAutodiffFloatStaticAttributeCompiles : IO Unit := do
  -- Compilation validates float autodiff with static parameters.
  return

@[test]
def testDeriveAndRegisterADRulesCoreStatic : IO Unit := do
  let ok ← runCoreM (do
    registerJVPRule `test.foo fun args tans _ => do
      let prim ← getFreshVar
      let tan ← getFreshVar
      let builder := fun rest =>
        FnBody.vdecl prim IRType.object (Expr.fap `test.foo args) <|
        FnBody.vdecl tan IRType.object (Expr.fap `test.foo_tan tans) <|
        rest
      return (builder, prim, tan)

    registerVJPRule `test.foo fun args dy _ => do
      let dx ← getFreshVar
      let callArgs := args.push (.var dy)
      let builder := fun rest =>
        FnBody.vdecl dx IRType.object (Expr.fap `test.foo_bwd callArgs) rest
      return (builder, #[dx])

    let x : VarId := { idx := 1 }
    let s : VarId := { idx := 2 }
    let y : VarId := { idx := 3 }
    let px : Param := { x := x, borrow := false, ty := IRType.object }
    let ps : Param := { x := s, borrow := false, ty := IRType.object }
    let body := FnBody.vdecl y IRType.object (Expr.fap `test.foo #[Arg.var x]) (.ret (.var y))
    let decl : Decl := .fdecl `test.primalStatic #[px, ps] IRType.object body {}
    Lean.IR.addDecl decl
    let paramKinds := parseStaticIndices #[1] 2
    deriveAndRegisterADRules `test.primalStatic paramKinds

    let hasJVP := (← getJVPRule `test.primalStatic).isSome
    let hasVJP := (← getVJPRule `test.primalStatic).isSome
    let hasJVPDecl := (← Lean.IR.findDecl (Lean.Name.mkStr `test.primalStatic "jvp")).isSome
    let hasVJPDecl := (← Lean.IR.findDecl (Lean.Name.mkStr `test.primalStatic "vjp")).isSome
    return hasJVP && hasVJP && hasJVPDecl && hasVJPDecl
  )
  LeanTest.assertTrue ok "static deriveAndRegisterADRules should register rules and add companion decls"

/-! ## DifferentiableManifold Typeclass Tests -/

open Tyr.AD.DifferentiableManifold in
@[test]
def testFloatEuclidean : IO Unit := do
  -- Float as EuclideanSpace operations
  let zero : Float := EuclideanSpace.zero
  LeanTest.assertTrue (zero == 0.0) s!"Float zero should be 0.0"

  let sum := EuclideanSpace.add 2.0 3.0
  LeanTest.assertTrue (sum == 5.0) s!"Float add should work, got {sum}"

  let scaled := EuclideanSpace.smul 2.0 3.0
  LeanTest.assertTrue (scaled == 6.0) s!"Float smul should work, got {scaled}"

  let inner := EuclideanSpace.inner 3.0 4.0
  LeanTest.assertTrue (inner == 12.0) s!"Float inner should work, got {inner}"

open Tyr.AD.DifferentiableManifold in
@[test]
def testFloatManifold : IO Unit := do
  -- Float as DifferentiableManifold - test exp map
  let x : Float := 3.0
  let v : Float := 2.0  -- Tangent vector
  let moved := DifferentiableManifold.exp x v
  LeanTest.assertTrue (moved == 5.0) s!"exp(3, 2) should be 5, got {moved}"

  -- Test retract (same as exp for Euclidean)
  let retracted := DifferentiableManifold.retract x v
  LeanTest.assertTrue (retracted == 5.0) s!"retract(3, 2) should be 5, got {retracted}"

open Tyr.AD in
@[test]
def testStaticManifold : IO Unit := do
  -- Static has Unit tangent/cotangent
  let x : torch.Static Float := ⟨42.0⟩
  let zeroTan := DifferentiableManifold.zeroTangent x
  let zeroCot := DifferentiableManifold.zeroCotangent x

  -- Adding units should work
  let unitSum := DifferentiableManifold.addTangent zeroTan zeroTan
  -- Check they are unit
  match zeroTan, zeroCot, unitSum with
  | (), (), () => pure ()

  -- Static exp should not move the point
  let moved := DifferentiableManifold.exp x ()
  LeanTest.assertTrue (moved.val == 42.0) s!"Static exp should not move, got {moved.val}"

open Tyr.AD.DifferentiableManifold in
@[test]
def testGradientStep : IO Unit := do
  -- Test gradient descent step using retraction
  let x : Float := 10.0
  let grad : Float := 2.0  -- Cotangent (gradient)
  let lr := 0.1
  let x' := gradientStep x grad lr
  -- x' = exp(x, -lr * sharp(grad))
  -- For Euclidean: x' = x + (-lr * grad) = 10 - 0.2 = 9.8
  LeanTest.assertTrue (x' == 9.8) s!"Expected 9.8, got {x'}"

open Tyr.AD.DifferentiableManifold in
@[test]
def testMusicalIsomorphisms : IO Unit := do
  -- For Euclidean spaces, sharp and flat are identity
  let x : Float := 5.0
  let cotangent : Float := 3.0
  -- For Float, Tangent x = Float, so sharp/flat are id
  let tangent : Float := sharp (x := x) cotangent
  LeanTest.assertTrue (tangent == 3.0) s!"sharp should be identity for Euclidean, got {tangent}"

  let flatted : Float := flat (x := x) tangent
  LeanTest.assertTrue (flatted == 3.0) s!"flat should be identity for Euclidean, got {flatted}"

/-! ## Stiefel Manifold Tests -/

open Tyr.AD.DifferentiableManifold in
@[test]
def testStiefelIdentity : IO Unit := do
  -- Test Stiefel identity element St(3, 3) = O(3)
  let X := Stiefel.identity 3
  -- Identity should satisfy X^T @ X = I
  let XtX := torch.nn.mm (torch.nn.transpose2d X.matrix) X.matrix
  let I := torch.eye 3
  LeanTest.assertTrue (torch.allclose XtX I) "Stiefel identity should satisfy X^T X = I"

open Tyr.AD.DifferentiableManifold in
@[test]
def testStiefelConstraint : IO Unit := do
  -- Random Stiefel element should satisfy X^T @ X = I
  let X ← Stiefel.random 4 4
  let XtX := torch.nn.mm (torch.nn.transpose2d X.matrix) X.matrix
  let I := torch.eye 4
  LeanTest.assertTrue (torch.allclose XtX I (rtol := 1e-5) (atol := 1e-6))
    "Random Stiefel element should satisfy X^T X = I"

open Tyr.AD.DifferentiableManifold in
@[test]
def testStiefelProjection : IO Unit := do
  -- Project a random matrix to Stiefel and verify constraint
  let mat ← torch.randn #[4, 4]
  let X := Stiefel.project 4 4 mat
  let XtX := torch.nn.mm (torch.nn.transpose2d X.matrix) X.matrix
  let I := torch.eye 4
  LeanTest.assertTrue (torch.allclose XtX I (rtol := 1e-5) (atol := 1e-6))
    "Projected matrix should satisfy X^T X = I"

open Tyr.AD.DifferentiableManifold in
@[test]
def testStiefelRetractionPreservesConstraint : IO Unit := do
  -- After retraction, matrix should still be on Stiefel manifold
  let X := Stiefel.identity 3
  -- Create a small tangent perturbation
  let z := StiefelTangent.smul 0.01 (StiefelTangent.zero 3 3)
  let X' := DifferentiableManifold.exp X z
  let XtX := torch.nn.mm (torch.nn.transpose2d X'.matrix) X'.matrix
  let I := torch.eye 3
  LeanTest.assertTrue (torch.allclose XtX I (rtol := 1e-5) (atol := 1e-6))
    "Retraction should preserve Stiefel constraint"

/-! ## Orthogonal Manifold Tests -/

open Tyr.AD.DifferentiableManifold in
@[test]
def testOrthogonalIdentityConstraint : IO Unit := do
  -- Test that Orthogonal identity satisfies Q^T @ Q = I and Q @ Q^T = I
  let Q := Orthogonal.identity 3
  let QtQ := torch.nn.mm (torch.nn.transpose2d Q.matrix) Q.matrix
  let QQt := torch.nn.mm Q.matrix (torch.nn.transpose2d Q.matrix)
  let I := torch.eye 3
  LeanTest.assertTrue (torch.allclose QtQ I) "Q^T Q should be I"
  LeanTest.assertTrue (torch.allclose QQt I) "Q Q^T should be I"

open Tyr.AD.DifferentiableManifold in
@[test]
def testOrthogonalRandomConstraint : IO Unit := do
  -- Random orthogonal matrix should satisfy Q^T @ Q = I
  let Q ← Orthogonal.random 4
  let QtQ := torch.nn.mm (torch.nn.transpose2d Q.matrix) Q.matrix
  let I := torch.eye 4
  LeanTest.assertTrue (torch.allclose QtQ I (rtol := 1e-5) (atol := 1e-6))
    "Random orthogonal matrix should satisfy Q^T Q = I"

open Tyr.AD.DifferentiableManifold in
@[test]
def testOrthogonalProjection : IO Unit := do
  -- Project a random matrix to O(n) and verify constraint
  let mat ← torch.randn #[3, 3]
  let Q := Orthogonal.project 3 mat
  let QtQ := torch.nn.mm (torch.nn.transpose2d Q.matrix) Q.matrix
  let I := torch.eye 3
  LeanTest.assertTrue (torch.allclose QtQ I (rtol := 1e-5) (atol := 1e-6))
    "Projected matrix should satisfy Q^T Q = I"

open Tyr.AD.DifferentiableManifold in
@[test]
def testSkewSymmetricProperty : IO Unit := do
  -- Test that SkewSymmetric.fromMatrix produces S + S^T = 0
  let mat ← torch.randn #[3, 3]
  let S := SkewSymmetric.fromMatrix mat
  let St := torch.nn.transpose2d S.matrix
  let sum := torch.add S.matrix St
  let zero := torch.zeros #[3, 3]
  LeanTest.assertTrue (torch.allclose sum zero (atol := 1e-6))
    "Skew-symmetric matrix should satisfy S + S^T = 0"

open Tyr.AD.DifferentiableManifold in
@[test]
def testOrthogonalTranspose : IO Unit := do
  -- Test that transpose gives the inverse for orthogonal matrices
  let Q ← Orthogonal.random 3
  let Qt := Orthogonal.transpose Q
  -- Q @ Q^T should be identity
  let QQt := Orthogonal.mul Q Qt
  let I := torch.eye 3
  LeanTest.assertTrue (torch.allclose QQt.matrix I (rtol := 1e-5) (atol := 1e-6))
    "Q @ Q^T should be I"

open Tyr.AD.DifferentiableManifold in
@[test]
def testOrthogonalGroupClosure : IO Unit := do
  -- Product of orthogonal matrices should be orthogonal
  let Q1 ← Orthogonal.random 3
  let Q2 ← Orthogonal.random 3
  let Q3 := Orthogonal.mul Q1 Q2
  let Q3tQ3 := torch.nn.mm (torch.nn.transpose2d Q3.matrix) Q3.matrix
  let I := torch.eye 3
  LeanTest.assertTrue (torch.allclose Q3tQ3 I (rtol := 1e-4) (atol := 1e-6))
    "Product of orthogonal matrices should be orthogonal"

open Tyr.AD.DifferentiableManifold in
@[test]
def testOrthogonalExpZero : IO Unit := do
  -- exp(Q, 0) should return Q (unchanged)
  let Q := Orthogonal.identity 3
  let z := DifferentiableManifold.zeroTangent Q
  let Q' := DifferentiableManifold.exp Q z
  LeanTest.assertTrue (torch.allclose Q.matrix Q'.matrix (rtol := 1e-5) (atol := 1e-6))
    "exp(Q, 0) should return Q unchanged"

open Tyr.AD.DifferentiableManifold in
@[test]
def testOrthogonalExpPreservesConstraint : IO Unit := do
  -- After exponential map with small tangent, result should still be orthogonal
  let Q ← Orthogonal.random 3
  -- Create a small skew-symmetric perturbation
  let mat ← torch.randn #[3, 3]
  let S := SkewSymmetric.fromMatrix mat
  let smallS := SkewSymmetric.smul 0.01 S
  let v := OrthogonalTangent.fromSkew smallS
  let Q' := DifferentiableManifold.exp Q v
  let QtQ := torch.nn.mm (torch.nn.transpose2d Q'.matrix) Q'.matrix
  let I := torch.eye 3
  LeanTest.assertTrue (torch.allclose QtQ I (rtol := 1e-4) (atol := 1e-6))
    "Exponential map should preserve orthogonality"

/-! ## Grassmann Manifold Tests -/

open Tyr.AD.DifferentiableManifold in
@[test]
def testGrassmannConstraint : IO Unit := do
  -- Random Grassmann element should satisfy X^T @ X = I_p
  let X ← Grassmann.random 5 3
  let XtX := torch.nn.mm (torch.nn.transpose2d X.matrix) X.matrix
  let I := torch.eye 3
  LeanTest.assertTrue (torch.allclose XtX I (rtol := 1e-5) (atol := 1e-6))
    "Random Grassmann element should satisfy X^T X = I"

open Tyr.AD.DifferentiableManifold in
@[test]
def testGrassmannProjection : IO Unit := do
  -- Project a random matrix to Gr(n, p) and verify constraint
  let mat ← torch.randn #[5, 3]
  let X := Grassmann.project 5 3 mat
  let XtX := torch.nn.mm (torch.nn.transpose2d X.matrix) X.matrix
  let I := torch.eye 3
  LeanTest.assertTrue (torch.allclose XtX I (rtol := 1e-5) (atol := 1e-6))
    "Projected Grassmann element should satisfy X^T X = I"

open Tyr.AD.DifferentiableManifold in
@[test]
def testGrassmannTangentOrthogonality : IO Unit := do
  -- Tangent vectors should satisfy X^T @ Z = 0
  let X ← Grassmann.random 5 3
  let V ← torch.randn #[5, 3]
  let Z := GrassmannTangent.project X V
  -- X^T @ Z should be zero
  let XtZ := torch.nn.mm (torch.nn.transpose2d X.matrix) Z.vec
  let zero := torch.zeros #[3, 3]
  LeanTest.assertTrue (torch.allclose XtZ zero (atol := 1e-6))
    "Tangent projection should satisfy X^T Z = 0"

open Tyr.AD.DifferentiableManifold in
@[test]
def testGrassmannRetractionPreservesConstraint : IO Unit := do
  -- After retraction, result should still be on the manifold
  let X ← Grassmann.random 5 3
  -- Create a random tangent vector
  let V ← torch.randn #[5, 3]
  let Z := GrassmannTangent.project X V
  let smallZ := GrassmannTangent.smul 0.1 Z
  let X' := DifferentiableManifold.exp X smallZ
  let XtX := torch.nn.mm (torch.nn.transpose2d X'.matrix) X'.matrix
  let I := torch.eye 3
  LeanTest.assertTrue (torch.allclose XtX I (rtol := 1e-4) (atol := 1e-6))
    "Retraction should preserve Grassmann constraint"

open Tyr.AD.DifferentiableManifold in
@[test]
def testGrassmannPrincipalAngles : IO Unit := do
  -- Principal angles of same subspace with itself should be all 1s (cos(0) = 1)
  let X ← Grassmann.random 5 3
  let cosines := Grassmann.principalAngles X X
  let ones := torch.ones #[3]
  LeanTest.assertTrue (torch.allclose cosines ones (rtol := 1e-5) (atol := 1e-6))
    "Principal angles of subspace with itself should be all zeros (cosines all 1)"

open Tyr.AD.DifferentiableManifold in
@[test]
def testGrassmannDistanceSelf : IO Unit := do
  -- Distance of a subspace to itself should be 0
  let X ← Grassmann.random 5 3
  let d := Grassmann.distance X X
  LeanTest.assertTrue (d < 1e-5) "Distance of subspace to itself should be 0"

/-! ## Hyperbolic Manifold Tests -/

open Tyr.AD.DifferentiableManifold in
@[test]
def testHyperbolicConstraint : IO Unit := do
  -- Random hyperbolic point should satisfy Minkowski norm ≈ -1
  let X ← Hyperbolic.random 3
  let inner := Hyperbolic.minkowskiInner X.coords X.coords
  LeanTest.assertTrue (Float.abs (inner + 1.0) < 1e-4)
    s!"Hyperbolic point should satisfy <x,x>_L = -1, got {inner}"

open Tyr.AD.DifferentiableManifold in
@[test]
def testHyperbolicTangentOrthogonality : IO Unit := do
  -- Projected tangent should be Minkowski-orthogonal to base point
  let X ← Hyperbolic.random 3
  let V ← torch.randn #[4]
  let Z := HyperbolicTangent.project X V
  let inner := Hyperbolic.minkowskiInner X.coords Z.vec
  LeanTest.assertTrue (Float.abs inner < 1e-4)
    s!"Tangent projection should satisfy <x,z>_L = 0, got {inner}"

open Tyr.AD.DifferentiableManifold in
@[test]
def testHyperbolicRetractionPreservesConstraint : IO Unit := do
  -- Retraction should keep points on the hyperboloid
  let X ← Hyperbolic.random 3
  let V ← torch.randn #[4]
  let Z := HyperbolicTangent.project X V
  let smallZ := HyperbolicTangent.smul 0.05 Z
  let X' := DifferentiableManifold.exp X smallZ
  let inner := Hyperbolic.minkowskiInner X'.coords X'.coords
  LeanTest.assertTrue (Float.abs (inner + 1.0) < 1e-4)
    s!"Retraction should preserve <x,x>_L = -1, got {inner}"

end Tests.AutoGrad
