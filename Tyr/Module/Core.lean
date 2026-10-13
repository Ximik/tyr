/-
  Tyr/Module/Core.lean

  Equinox-style Module typeclasses for neural networks.
  Provides callable module abstractions with TensorStruct integration.
-/
import Tyr.TensorStruct

namespace torch

/-! ## Module Typeclasses

These typeclasses define the interface for neural network modules:
- `Module`: Pure modules with no side effects in forward pass
- `ModuleIO`: Modules with IO effects (dropout, batch norm in training mode)
- `ModuleCtx`: Modules that take additional context (e.g., training flag, cache)

`torch.TensorStruct` is the parameter-tree substrate for modules:
- `Module`/`ModuleIO`/`ModuleCtx` require a `torch.TensorStruct` instance for `M`,
- this enables generic parameter transforms (e.g. detach, grads, checkpoint traversal)
  independently of the forward API.

### Example: Pure Module

```lean
import Tyr.Module.Derive

open torch

structure MyLinear (inDim outDim : UInt64) where
  weight : T #[outDim, inDim]
  bias : Option (T #[outDim]) := none
  deriving TensorStruct

instance {inDim outDim batch : UInt64} :
    Module (MyLinear inDim outDim) (T #[batch, inDim]) (T #[batch, outDim]) where
  forward m x :=
    let y := linear x m.weight
    match m.bias with
    | some b => add y (nn.expand b #[batch, outDim])
    | none => y
```

### Example: IO Module

```lean
open torch

structure MyDropout where
  p : Static Float := ⟨0.1⟩
  deriving TensorStruct

instance {s : Shape} : ModuleIO MyDropout (T s) (T s) where
  forward m x := dropout x m.p.val true
```

### Example: Contextual Module

```lean
open torch

structure MyLinearCtx (inDim outDim : UInt64) where
  lin : MyLinear inDim outDim
  deriving TensorStruct

instance {inDim outDim batch : UInt64} :
    ModuleCtx (MyLinearCtx inDim outDim) TrainingCtx
      (T #[batch, inDim]) (T #[batch, outDim]) where
  forward m ctx x := do
    let y := m.lin |> x
    dropout y ctx.dropout_p ctx.training
```
-/

/-- A pure module: a parameterized function with no side effects.
    The module type `M` must have a `torch.TensorStruct` instance for parameter traversal. -/
class Module (M : Type) (In : Type) (Out : Type) extends TensorStruct M where
  forward : M → In → Out

/-- A module with IO effects (dropout, random sampling, etc.) -/
class ModuleIO (M : Type) (In : Type) (Out : Type) extends TensorStruct M where
  forward : M → In → IO Out

/-- A module that takes additional context (training flag, cache, etc.) -/
class ModuleCtx (M : Type) (Ctx : Type) (In : Type) (Out : Type) extends TensorStruct M where
  forward : M → Ctx → In → IO Out

/-! ## Lifting Instances

Automatically lift pure Module to ModuleIO.
-/

/-- Lift a pure Module to ModuleIO -/
instance [inst : Module M In Out] : ModuleIO M In Out where
  toTensorStruct := inst.toTensorStruct
  forward m x := pure (Module.forward m x)

/-! ## Forward Application

Explicit forward pass functions. CoeFun doesn't work well with multi-parameter
typeclasses, so we use explicit syntax instead.

### Example: Applying Modules

```lean
open torch

def runPure [Module M In Out] (m : M) (x : In) : Out :=
  m |> x

def runIO [ModuleIO M In Out] (m : M) (x : In) : IO Out := do
  m |>! x

def runCtx [ModuleCtx M Ctx In Out] (m : M) (ctx : Ctx) (x : In) : IO Out := do
  ModuleCtx.forward m ctx x
```
-/

/-- Apply a pure module to input -/
@[inline] def Module.apply [Module M In Out] (m : M) (x : In) : Out :=
  Module.forward m x

/-- Apply an IO module to input -/
@[inline] def ModuleIO.apply [ModuleIO M In Out] (m : M) (x : In) : IO Out :=
  ModuleIO.forward m x

/-- Infix notation for module application: `m |> x` -/
scoped infixl:90 " |> " => Module.apply

/-- Infix notation for IO module application: `m |>! x` -/
scoped infixl:90 " |>! " => ModuleIO.apply

/-! ## Training Context

Common context type for modules that behave differently in training vs inference.
-/

/-- Training context for modules like Dropout, BatchNorm -/
structure TrainingCtx where
  training : Bool := true
  dropout_p : Float := 0.0
  deriving Repr, BEq, Inhabited

namespace TrainingCtx

def train : TrainingCtx := { training := true }
def eval : TrainingCtx := { training := false, dropout_p := 0.0 }

def withDropout (ctx : TrainingCtx) (p : Float) : TrainingCtx :=
  { ctx with dropout_p := if ctx.training then p else 0.0 }

end TrainingCtx

end torch
