# Automatic differentiation

Tyr has three AD-relevant layers, and which one you want depends on the job:

- **Indexed LibTorch autograd** — runtime, tape-based reverse mode on `Tensor σ`, exposed through `Tensor.backwardLoss`, `Tensor.backward`, and `Tensor.gradOf`. See [core/tensors.md](core/tensors.md). Global gradient-mode controls remain in `torch.autograd`; `torch.differentiable` is a separate lightweight gradient interface.
- **IR-level JVP/VJP** (`Tyr/AutoGrad.lean`, namespace `Tyr.AD`) — compile-time, source-to-source forward- and reverse-mode transforms over Lean compiler IR (`Lean.IR.FnBody`), driven by rule registries and the `@[jvp]` / `@[vjp]` / `@[autodiff]` attributes.
- **Elimination-based AD** (`Tyr/AD/`, 41 files) — a Graphax/JAX-style stack that normalizes code into a jaxpr-like equation IR, extracts local Jacobians as sparse linear maps, and accumulates the full Jacobian by vertex elimination on a bipartite graph, with pluggable elimination orders including an AlphaGrad-style MCTS planner.

Use the FFI path for training models. Use `@[autodiff]` for compile-time-derived gradients of pure Lean functions (e.g. `Float` arithmetic). Reach for the `Tyr/AD/` stack when working on GPU codegen or elimination-order research — several layers are still scaffolding, see [Current limitations](#current-limitations).

## Architecture and main abstractions

### IR-level AD (`Tyr.AutoGrad`)

The substrate is Lean's compiler IR. A declaration's body is a `FnBody` tree; full applications show up as `Expr.fap callee args`, and rules are keyed on `callee`:

```lean
inductive ParamKind where
  | diff      -- differentiable: gets a tangent (JVP) / cotangent (VJP)
  | static    -- non-differentiable: passed through unchanged
  | frozen    -- forward-only: no gradient update
                                                    -- Tyr/AutoGrad.lean:47

abbrev ADM := StateT ADContext Lean.CoreM           -- :121

abbrev JVPRule := Array Arg → Array Arg → IRType
  → ADM (JVPBuilder × VarId × VarId)                -- :227
abbrev VJPRule := Array Arg → VarId → IRType
  → ADM (VJPBuilder × Array VarId)                  -- :230
```

`ADContext` tracks primal→tangent and primal→cotangent variable maps, variable types, a pullback stack for the reverse pass, and the set of static variables. Rules live in a Lean environment extension (`adRegistry`, `:237`) and are stored per function name.

The two transforms walk a `Decl` and build companion declarations:

- `linearizeWithKinds (decl : Decl) (paramKinds : Array ParamKind := #[]) : ADM Decl` (`:626`) — forward mode. Produces `f.jvp`, taking primal parameters followed by one tangent per `diff` parameter and returning a boxed `(primal, tangent)` pair. Missing tangents are hard errors here (`strictMissingTangents` is set).
- `transposeWithKinds (decl : Decl) (paramKinds : Array ParamKind := #[]) : ADM Decl` (`:654`) — reverse mode. Runs the `vjp` interpreter, which pushes a pullback builder per rule application while walking forward, then emits the stack in reverse at the `ret`. Produces `f.vjp`, taking the primal parameters plus the output cotangent and returning a tuple of the `diff`-parameter cotangents.
- `deriveAndRegisterADRules (primalFn : Name) (paramKinds := #[]) : CoreM Unit` (`:699`) — compiles both companions from `primalFn`'s IR, adds them as IR declarations, and registers them as JVP/VJP rules so enclosing transforms can differentiate through `primalFn`.

The user-facing surface is the attribute layer (`:770-779`):

```lean
@[jvp primalFn]   @[vjp primalFn]                        -- register this def as the rule
@[jvp primalFn, static := [1, 3]]                        -- same, with static parameters
@[autodiff]                                              -- derive both companions from IR
@[autodiff, static := [1]]
```

Two semantics worth knowing:

- `@[jvp]` / `@[vjp]` only *register* the attributed definition; it remains an ordinary Lean function you can call and test directly.
- `@[autodiff]` runs `.afterCompilation` and its `f.jvp` / `f.vjp` companions exist only as IR declarations plus rule-registry entries — they are inputs to further AD transforms, not Lean constants you can reference from source.

There is also a GPU hook: `GpuVJPRule` and a separate `gpuAdRegistry` keyed by op-name strings (`registerGpuVJPRule`, `:726`), consumed by `Tyr/GPU/AutoGrad.lean`. Nothing in the repository registers GPU rules today. Handwritten Lean-IR rules for `torch.add`/`sub`/`mul`/`matmul` live in `Tyr.GPU.AD.init` (`Tyr/GPU/AD.lean:37`) but are not wired into any standard import path.

### Jaxpr-like IR (`Tyr.AD.JaxprLike`)

The elimination stack's normalized form is a flat equation list in the style of a JAX jaxpr:

```lean
structure LeanJaxpr where               -- Tyr/AD/JaxprLike/Core.lean:255
  constvars : Array JVar := #[]
  invars    : Array JVar := #[]
  eqns      : Array JEqn := #[]
  outvars   : Array JVar := #[]

structure JEqn where                    -- :234
  op      : OpName                      -- abbrev OpName := Name
  invars  : Array JVar
  outvars : Array JVar
  params  : OpParams := #[]             -- typed metadata bag (axis, padLow, sourceOp, …)
  source  : SourceRef := {}
```

Each `JVar` (`:227`) carries a `VarMeta` (`:173`): a `DiffParticipation` marker (`diff | static | frozen`, `:166`) plus optional shape/dtype/sharding hints. Three frontends produce a `LeanJaxpr`:

- `FromFnBody.lean` — from Lean IR; very conservative, only direct `Expr.fap` value-declaration chains are lowered.
- `FromKStmt.lean` — from the GPU `KStmt` IR (see [gpu/dsl-codegen.md](gpu/dsl-codegen.md)); canonical op names come from `KStmtNames.lean` (`kstmtUnaryOpName`, `kstmtBinaryOpName`, …).
- `@[ad_frontend]` — direct registration of a hand-built jaxpr per declaration (below); preferred over `FnBody` recovery when present.

`Pipeline.buildFromDecl (cfg : BuildConfig := {}) (decl : Decl) : CoreM (Except BuildError LeanJaxpr)` (`Pipeline.lean:78`) ties these together: convert, validate structural invariants, then gate on local-Jacobian rule coverage (`BuildConfig.requireRuleCoverage` defaults to `true`).

Local Jacobians attach to equations through rules:

```lean
structure LocalJacEdge where            -- JaxprLike/Rules.lean:17
  src dst : JVarId
  map : SparseLinearMap := {}

abbrev LocalJacRule :=
  JEqn → RuleContext → Except RuleError (Array LocalJacEdge)   -- :35

registerLocalJacRule (op : OpName) (rule : LocalJacRule)
    : Lean.CoreM Unit                   -- JaxprLike/RuleRegistry.lean:18
```

Built-in rule packs for all `KStmt` ops live in `RulePackKStmt.lean`; `registerKStmtAllSupportedSemanticsRules` (`:1582`) is the one with real derivative semantics.

### Sparse linear maps (`Tyr.AD.Sparse`)

Edges carry sparse linear maps — the local Jacobians:

```lean
structure SparseLinearMap where         -- Sparse/Map.lean:221
  repr    : SparseMapTag := .placeholder
  inDim?  : Option DimSize := none
  outDim? : Option DimSize := none
  entries : Array SparseEntry := #[]    -- COO: src, dst, weight : Float
```

`SparseMapTag` (`:182`) records provenance (`placeholder`, `identityLike`, `zero`, `identity n`, `semantic tag`, `named label`, plus `add`/`compose` combinators). The `.semantic` variant wraps a `SparseSemanticTag` (`:110`) describing what the map means (e.g. `.binary op .lhs .rhsValue`, `.dotGeneral spec`) for diagnostics and downstream interpretation. `Tyr.AD.Sparse.compose` and `Tyr.AD.Sparse.add` (`Sparse/Compose.lean`, `Sparse/Add.lean`) are the dimension-checked algebra used during elimination.

### Vertex elimination (`Tyr.AD.Elim`)

```lean
structure ElimGraph where               -- Elim/Graph.lean:19
  forward  : AdjMap                     -- src → dst → SparseLinearMap
  backward : AdjMap                     -- dst → src → SparseLinearMap
  inputs outputs eliminable : Array JVarId

eliminateVertex (g : ElimGraph) (v : JVarId)
    : Except String (ElimGraph × ElimStepStats)   -- Elim/Eliminate.lean:45
```

Eliminating a vertex composes every incoming edge with every outgoing edge (`outMap ∘ inMap`), merges parallel edges with `add`, and prunes the vertex. A complete run (`runElimination`, `runForwardElimination`, `runReverseElimination`) leaves only input→output edges: the accumulated Jacobian, as an `ElimRunResult` (`Eliminate.lean:36`) with per-step statistics.

The elimination order is a policy:

```lean
inductive OrderPolicy where             -- Elim/OrderPolicy.lean:42
  | forward | reverse
  | explicitVertex (order1 : Array VertexId1)
  | constrainedVertex (base : Option (Array VertexId1)) (constraints : ConstraintSpec := {})
  | alphaGradAction (actions0 : Array ActionId0) (constraints : Option ConstraintSpec := none)
  | heuristic (name : String)
```

Vertex IDs are 1-based (`VertexId1`); AlphaGrad action IDs are 0-based (`ActionId0`), and the two spaces cross only through checked adapters. `Elim/AlphaGradMctx.lean` wraps elimination as an MCTS environment over `Tyr.Mctx` (`recurrentFn` plus a `searchEpisode?` family, alphaZero and gumbelMuZero variants) — an AlphaGrad-style learned planner for elimination orders, exercised from `Examples/AlphaGradPort/`. See [mctx.md](mctx.md) for the search infrastructure.

End-to-end adapters in `Elim/FromJaxpr.lean`:

```lean
buildElimGraphFromJaxpr (jaxpr : LeanJaxpr)
    : CoreM (Except String ElimGraph)                       -- :39
runEliminationOnJaxpr (jaxpr : LeanJaxpr) (order : Array JVarId)
    : CoreM (Except String ElimRunResult)                   -- :50
runEliminationOnJaxprWithPolicy (jaxpr : LeanJaxpr) (policy : OrderPolicy)
    : CoreM (Except String ElimRunResult)                   -- :61
runForwardEliminationOnJaxpr / runReverseEliminationOnJaxpr -- :75, :88
runEliminationOnKStmts (cfg : BuildConfig := {}) (stmts : Array KStmt) …  -- :101
```

Round-trip note: `Elim/LowerKStmt.lean` and `Elim/LowerFnBody.lean` lower a normalized jaxpr *back* to `KStmt` / `FnBody`. They share file names with `JaxprLike/Lower*.lean`, which go the opposite direction — check the namespace before importing.

### Structured frontend (`Tyr.AD.Frontend`)

The top layer rebuilds ordinary Lean values (model structures, not flat variable arrays) across the AD boundary. It rests on two derivable schema classes (`Tyr/AD/TensorStructSchema.lean`, namespace `torch`):

```lean
class ToTensorStructSchema (α : Type) where    -- TensorStructSchema.lean:144
  typeName : Name := Name.anonymous
  describeLeaves : α → TensorLeafPath → Array TensorLeafSpec

class TensorStructFlatten (α : Type) where     -- :167
  flattenAt : α → TensorLeafPath → TensorLeafSelection → Array TensorLeafValue
  rebuildAt : α → TensorLeafPath → TensorLeafSelection → Array TensorLeafValue
    → StateT Nat (Except String) α
```

Both have `deriving` handlers and are typically derived alongside `TensorStruct` (see [core/tensorstruct.md](core/tensorstruct.md)). Leaves are tagged with a `TensorLeafRole` (`diff | static | frozen`, `:55`); `TensorLeafSelection` (`:101`) picks which roles cross the boundary.

A `FrontendADSignature` (`Frontend/Signature.lean:205`) is an ordered set of `FrontendBoundary`s for params, inputs, and outputs — the schema-aware bridge between a structured value and the flat jaxpr invars. A `StructuredFrontendFunction` (`Frontend/API.lean:33`) bundles that signature with flat eval/linearize callbacks and exposes the user-facing API:

```lean
structure StructuredFrontendFunction (Params Inputs Outputs : Type) where
  signature     : FrontendADSignature
  evalFlat      : Array TensorLeafValue → Except String (FlatFrontendEvalResult Outputs)
  linearizeFlat : Array TensorLeafValue → Except String (FlatFrontendLinearizedResult Outputs)

StructuredFrontendFunction.call        … : Except String Outputs                              -- :98
StructuredFrontendFunction.linearize   … : Except String (StructuredPullback Params Inputs Outputs)   -- :116
StructuredFrontendFunction.vjp         … (outputCotangent : Outputs)
                                       : Except String (StructuredVJPResult Params Inputs)  -- :138
StructuredFrontendFunction.valueAndGrad… : Except String (Outputs × StructuredGradResult Params)    -- :158
StructuredFrontendFunction.grad        … : Except String (StructuredGradResult Params)      -- :186
```

`grad` is scalar-loss style: it requires exactly one differentiable output leaf of shape `#[]` and seeds the cotangent with `ones_like`. The result types (`StructuredPullback`, `StructuredVJPResult`, `StructuredGradResult`, `Frontend/Companion.lean:18-29`) rebuild cotangents into the same structures as the inputs, with static fields preserved from a template.

Registration ties a frontend to a declaration:

```lean
structure FrontendRegistration where    -- JaxprLike/HintRegistry.lean:23
  jaxpr : LeanJaxpr
  signature : Option FrontendADSignature := none
  runtimeFrontend : Option Lean.Name := none   -- a StructuredFrontendFunction constant
```

`attribute [ad_frontend spec] f` (`JaxprLike/Elab.lean:186`) validates and registers the bundle; when `runtimeFrontend` is present it also synthesizes ordinary Lean defs `f.frontend`, `f.linearize`, `f.vjp`, `f.valueAndGrad`, and `f.grad`. `buildFromDecl` then prefers this jaxpr over `FnBody` recovery.

## Key APIs

### Differentiating Lean functions (`Tyr.AutoGrad`, namespace `Tyr.AD`)

| API | Kind | Purpose |
| --- | --- | --- |
| `@[jvp f]` / `@[vjp f]` | attribute | register the def as `f`'s forward/reverse rule |
| `@[jvp f, static := [i, …]]` (same for `vjp`) | attribute | rule with non-differentiable parameters by index |
| `@[autodiff]` / `@[autodiff, static := […]]` | attribute | derive `f.jvp` + `f.vjp` from compiled IR, register both |
| `registerLeanJVPRule[WithKinds]` / `registerLeanVJPRule[WithKinds]` | `CoreM Unit` | programmatic rule registration (`Tyr/AutoGrad.lean:390`, `:324`) |
| `linearizeWithKinds` / `transposeWithKinds` | `ADM Decl` | the forward/reverse IR transforms (`:626`, `:654`) |
| `deriveAndRegisterADRules` | `CoreM Unit` | compile + add + register both companions (`:699`) |
| `getJVPRule` / `getVJPRule` | `CoreM (Option …)` | registry lookup |

### Elimination stack (`Tyr.AD.JaxprLike`, `Tyr.AD.Sparse`, `Tyr.AD.Elim`)

| API | Purpose |
| --- | --- |
| `LeanJaxpr` / `JEqn` / `JVar` / `VarMeta` | normalized equation IR with AD metadata |
| `buildFromDecl` / `buildFromFnBody` / `buildFromKStmts` | convert + validate + coverage gate (`Pipeline.lean`) |
| `LocalJacRule`, `registerLocalJacRule` | per-op local-Jacobian contract and registry |
| `registerKStmtAllSupportedSemanticsRules` | built-in rules for all `KStmt` ops (`RulePackKStmt.lean:1582`) |
| `SparseLinearMap`, `Sparse.compose`, `Sparse.add` | checked sparse Jacobian algebra |
| `ElimGraph`, `eliminateVertex`, `ElimRunResult` | elimination graph and single-step eliminator |
| `OrderPolicy` | forward/reverse/explicit/constrained/AlphaGrad/heuristic orders |
| `runEliminationOnJaxpr[WithPolicy]`, `run{Forward,Reverse}EliminationOnJaxpr`, `runEliminationOnKStmts[WithPolicy]` | end-to-end runs (`FromJaxpr.lean`) |

### Structured frontend (`Tyr.AD.Frontend`, namespace `torch` for schema classes)

| API | Purpose |
| --- | --- |
| `ToTensorStructSchema` / `TensorStructFlatten` | derivable schema + flatten/rebuild classes |
| `FrontendBoundary.ofValue x selection` | boundary from a sample value (`Signature.lean:49`) |
| `FrontendADSignature` | ordered param/input/output boundaries |
| `StructuredFrontendFunction.{call, linearize, vjp, valueAndGrad, grad}` | structured AD entry points (`API.lean`) |
| `FrontendRegistration`, `attribute [ad_frontend spec] f` | per-decl registration + companion synthesis |

## Usage examples

### Custom rules and `@[autodiff]` (Reconstructed example from `Tests/TestAutoGrad.lean:34-71` and `:369-385`)

```lean
import Tyr.AutoGrad

open Tyr.AD

@[noinline] def square (x : Float) : Float := x * x

-- Hand-written rules: the attributed defs stay ordinary, callable functions.
@[vjp square]
def square_bwd (x dy : Float) : Float := 2.0 * x * dy

@[jvp square]
def square_fwd (x dx : Float) : Float × Float := (x * x, 2.0 * x * dx)

-- Automatic derivation of f.jvp / f.vjp from compiled IR:
@[autodiff]
def autoChain (x : Float) : Float := square (square x)

-- Parameter 1 is static: no tangent/cotangent is built for it.
@[autodiff, static := [1]]
def autoFloatKeep (x : Float) (_n : Nat) : Float := square x
```

The primals are marked `@[noinline]` on purpose: rules match on `Expr.fap` callees, so the call must survive to IR instead of being inlined away.

### Vertex elimination over a jaxpr (Reconstructed example from `Tests/TestADElimFromJaxpr.lean:32-76`)

```lean
import Tyr.AD.Elim
import Tyr.AD.JaxprLike
import Tyr.GPU.Codegen.IR

open Tyr.AD.Elim Tyr.AD.JaxprLike Tyr.GPU.Codegen

-- out = exp(x0) + x1 as two equations; vertex 2 is the intermediate.
def jaxpr : LeanJaxpr := {
  invars := #[{ id := 0 }, { id := 1 }]
  eqns   := #[
    { op := kstmtUnaryOpName .Exp,  invars := #[{ id := 0 }],
      outvars := #[{ id := 2 }] },
    { op := kstmtBinaryOpName .Add, invars := #[{ id := 2 }, { id := 1 }],
      outvars := #[{ id := 3 }] } ]
  outvars := #[{ id := 3 }] }

def run : Lean.CoreM Unit := do
  registerKStmtAllSupportedSemanticsRules   -- required; without rules extraction fails
  let result ← runEliminationOnJaxpr jaxpr #[2]
  -- result : Except String ElimRunResult
  -- after elimination, vertex 2 is gone and only input→output Jacobian edges remain
  pure ()
```

Policy-driven variants: `runEliminationOnJaxprWithPolicy jaxpr .reverse`, `runForwardEliminationOnJaxpr jaxpr`, or MCTS-planned orders via `AlphaGradMctx.searchEpisode?`.

### Structured frontend with `@[ad_frontend]` (Reconstructed example from `Tests/TestADJaxprLikeElabFixture.lean:112-225`, condensed)

```lean
import Tyr

open torch Tyr.AD.Frontend Tyr.AD.JaxprLike

structure Params where
  padv : Tensor { shape := #[1], dtype := .Float32 }
  label : Static String
  deriving TensorStruct, ToTensorStructSchema, TensorStructFlatten

structure Inputs where
  base : Tensor { shape := #[2], dtype := .Float32 }
  tag : Static String
  deriving TensorStruct, ToTensorStructSchema, TensorStructFlatten

structure Outputs where
  loss : Tensor { shape := #[], dtype := .Float32 }
  label : Static String
  deriving TensorStruct, ToTensorStructSchema, TensorStructFlatten

def sig : FrontendADSignature := {
  params  := #[FrontendBoundary.ofValue sampleParams .diffOnly]
  inputs  := #[FrontendBoundary.ofValue sampleInputs .diffOnly]
  outputs := #[FrontendBoundary.ofValue sampleOutputs .diffOnly] }

-- Flat eval/linearize callbacks over leaf buffers:
def fn : StructuredFrontendFunction Params Inputs Outputs := {
  signature := sig
  evalFlat := fun invarLeaves => …
  linearizeFlat := fun invarLeaves => … }  -- also returns a flat pullback

def registration : FrontendRegistration := {
  jaxpr := …                               -- flat LeanJaxpr view of the same computation
  signature := some sig
  runtimeFrontend := some `fn }

def lossFn (_params : Params) (_inp : Inputs) : Outputs := …

attribute [ad_frontend registration] lossFn
-- synthesizes lossFn.frontend / .linearize / .vjp / .valueAndGrad / .grad

-- let g ← lossFn.grad params inputs   -- Except String (StructuredGradResult Params)
```

The executable fixture supplies matching Float32 dtype metadata for the indexed boundaries. Its generated signature, graph metadata, and runtime companions must agree; missing dtype metadata is rejected. The condensed sketch above omits the callbacks and sample values; use the fixture for a complete implementation.

## The indexed LibTorch autograd path

Day-to-day training uses libtorch's tape through the FFI. The idiom is
`Tensor.backwardLoss loss` followed by `let gradient ← Tensor.gradOf param`
(usually `let gradients ← TensorStruct.grads params`). Gradient reads return
independent snapshots; `TensorStruct.zeroGrads params` clears accumulated
gradients in IO without changing those snapshots. Leaf parameters are prepared
with `Tensor.setRequiresGrad (Tensor.detach t) true`; distinct native allocations
require an effectful allocation boundary. Evaluation uses `autograd.no_grad`.
See [core/tensors.md](core/tensors.md), [core/tensorstruct.md](core/tensorstruct.md),
and [optimization.md](optimization.md).

## Current limitations

Read these before building on the stack:

- **Placeholder rule packs are not derivatives.** `registerKStmt*PlaceholderRules` and the hybrid variants (`RulePackKStmt.lean:1469-1580`) install identity-like maps; results are structurally valid but numerically meaningless. Only the `…SemanticsRules` packs carry real local Jacobian semantics, and even then the payload is scalar COO weights plus semantic tags, not dense tensor Jacobians.
- **Scaffolding headers are literal.** `Elim/Eliminate.lean` ("vertex elimination skeleton"), `Elim/Cost.lean` ("cost model scaffolding"), and `FromFnBody.lean` ("conservative conversion skeleton") self-describe as partial.
- **`@[autodiff]` companions are IR-only.** They cannot be called from Lean source, and the test suite only checks that they compile — they are never executed against known derivatives.
- **Reverse mode rejects aliasing.** The VJP interpreter hard-errors on repeated differentiable arguments (`Tyr/AutoGrad.lean:504-515`); cotangent accumulation for aliased arguments is not implemented.
- **Silent tangent fallback.** Outside `linearizeWithKinds`, a missing tangent falls back to the primal variable (`getTangentIdx`, `:149-158`) — wrong JVPs can compile silently.
- **`FromFnBody` coverage is narrow.** Only direct `Expr.fap` chains convert; anything else is a hard error. `@[ad_frontend]` exists to bypass this, which puts registration burden on each frontend.
- **Frontend adoption is test-only.** `@[ad_frontend]` and `StructuredFrontendFunction.grad` appear only in `Tests/`; no example or model uses them end-to-end.
- **GPU rule registries are empty.** No `registerGpuVJPRule` call sites exist, and `Tyr.GPU.AD.init` (handwritten torch-op rules) is not invoked from library code.

## Related guides

- [core/tensors.md](core/tensors.md) — `torch.autograd` FFI bindings and the `differentiable` class
- [core/tensorstruct.md](core/tensorstruct.md) — parameter-tree traversal (`TensorStruct.grads`, `zeroGrads`)
- [modules.md](modules.md) — layers and the `deriving Model` pipeline
- [optimization.md](optimization.md) — optimizers and training loops built on the FFI path
- [gpu/dsl-codegen.md](gpu/dsl-codegen.md) — the `KStmt` GPU IR the elimination stack consumes
- [mctx.md](mctx.md) — MCTS infrastructure under the AlphaGrad elimination planner

This chapter is a guide, not a reference. Exhaustive symbol-level documentation for `Tyr.AD`, `Tyr.AD.JaxprLike`, `Tyr.AD.Sparse`, `Tyr.AD.Elim`, and `Tyr.AD.Frontend` is generated by doc-gen4 (see `docbuild/`).
