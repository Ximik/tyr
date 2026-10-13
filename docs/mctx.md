# Monte Carlo tree search

`Tyr/Mctx/` is a pure-Lean port of the policy side of DeepMind's JAX `mctx`
library: MuZero-style PUCT search, an AlphaZero variant with persistent
subtrees, and Gumbel MuZero with sequential halving. `Tyr/MctxDag/` is a second
backend whose search structure is a DAG with a transposition table, so
different action sequences that reach the same state share a single node. Use
these modules for discrete planning over a learned or hand-written dynamics
model; everything is `Array Float`-based — no tensors and no libtorch FFI are
involved. The main in-tree consumer is the AlphaGrad vertex-elimination
planner (`Tyr/AD/Elim/AlphaGradMctx.lean`) with demos and trainers under
`Examples/AlphaGradPort/`.

Both stacks live in the `torch.mctx` / `torch.mctxdag` namespaces (a naming
leftover; the code is tensor-free). `Tyr.lean` re-exports `Tyr.Mctx` but not
`Tyr.MctxDag` — import the DAG backend explicitly.

```lean
import Tyr.Mctx      -- tree backend: torch.mctx
import Tyr.MctxDag   -- DAG backend: torch.mctxdag
```

## Architecture and main abstractions

### Model interface

The environment is injected as functions over an opaque embedding type `S`
and a params type `P` (`Tyr/Mctx/Base.lean`). You supply a root evaluation
(prior + value + embedding) and a recurrent dynamics step:

```lean
structure RootFnOutput (S : Type) where
  priorLogits : Array Float
  value : Float
  embedding : S

structure RecurrentFnOutput where
  reward : Float
  discount : Float
  priorLogits : Array Float
  value : Float

abbrev RecurrentFn (P S : Type) :=
  P → UInt64 → Action → S → RecurrentFnOutput × S
```

`Action` is `Nat`, the `UInt64` is a manually threaded RNG key (the search
derives per-simulation keys as `rngKey + sim + 1`). Search returns a
`PolicyOutput TreeType` carrying the chosen `action`, the `actionWeights`
training target, and the `searchTree` itself; `Tree.summary` exposes a
`SearchSummary` with root `visitCounts`, `visitProbs`, `value`, and `qvalues`.

### Search tree

`Tree S E` (`Tyr/Mctx/Tree.lean`) is a flat, fixed-capacity store — one
array per node attribute (`nodeVisits`, `rawValues`, `nodeValues`, `parents`,
`embeddings`) and one nested array per node-per-action edge attribute
(`childrenIndex`, `childrenPriorLogits`, `childrenVisits`, `childrenRewards`,
`childrenDiscounts`, `childrenValues`). Fresh searches reserve
`numSimulations + 1` nodes. The arena stays fixed during search, and
`numAllocated` tracks its contiguous occupied prefix for constant-time node
allocation. Persistent arrays and model evaluation can still allocate memory.
Sentinels:
`ROOT_INDEX = 0`, `NO_PARENT = -1`, `UNVISITED = -1`. The `E` parameter is
policy-specific extra data — Gumbel MuZero uses it to stash the root Gumbel
sample and cached sequential-halving schedule (`GumbelMuZeroExtraData`); the
other policies use `Unit`.

Useful tree operations:

- `Tree.qvalues tree nodeIndex` — expected complete return per action, including
  evaluated edges without an allocated child.
- `Tree.summary tree` — root statistics for policies.
- `resetSearchTree tree` — wipe a tree back to empty, keeping capacity.
- `getSubtree tree childAction` (`Tyr/Mctx/Tree.lean`) — extract the
  subtree under a root action with remapped indices; this is how AlphaZero
  reuses search across environment steps.

### Search loop

`search` (`Tyr/Mctx/Search.lean`) instantiates a tree from the root and
hands off to `searchWithTree`, which repeats the classic three steps
`numSimulations` times:

1. `simulate` — descend from the root with the action-selection
   function until an unvisited edge or the depth cutoff. Root and interior
   selectors are combined by `switchingActionSelectionWrapper`.
2. Call `recurrentFn` on the chosen `(parent, action)` edge to obtain its
   reward, discount, prior, value and embedding.
3. `expandWithStepAndBackup` — allocate the next free slot when available and
   back the evaluated return up along parent pointers, updating running-average
   node values and edge visit counts. Evaluations at full capacity still count.

Because the tree is persistent (immutable Lean structures), passing an
existing tree to `searchWithTree` continues a previous search — the mechanism
behind AlphaZero subtree reuse.

### Capacity and continuation

`maxNodes` is a hard arena bound; constructors reserve at least the root slot,
including when passed zero. When no child slot remains, the model is still
evaluated and its complete return `reward + discount * value` is backed up.
The edge remains `UNVISITED` because it has no stored child, but its visits and
mean return contribute to the policy and Q-values. Each ancestor receives the
current rollout return, keeping its running average distinct from that return.
This applies to tree, batched, and DAG searches.

For these evaluated but unallocated edges, `childrenValues` stores the mean
complete return; for allocated children it stores the continuation value.
Custom Q-transforms should use `tree.qvalues` to handle both representations.
If rerooting frees a slot and such an edge later acquires a child, its cached
value changes back to the new child's continuation value. Prior visit counts
and ancestor averages are retained.

`getSubtree` compacts reachable nodes with the chosen child at index zero,
including DAG transpositions to earlier nodes and cycles back to the old root.
Reset and reroot maintain the allocation counter. Manually constructed tree
records infer that counter once from the occupied visit-count prefix; manual
record updates must preserve the prefix and `numAllocated` together.

For batched continuation with no explicit `maxDepth`, each row uses its own
capacity-derived depth limit. Reordering heterogeneous trees therefore cannot
change another row's cutoff. An explicit depth limit applies to every row.

### Action selection and Q-transforms

Selectors have two signatures (`Tyr/Mctx/ActionSelection.lean`):
`RootActionSelectionFn` (no depth) and `InteriorActionSelectionFn`. Provided
implementations:

- `muzeroActionSelection` — PUCT: `argmax` of normalized Q plus
  `√N · pbC · π / (n + 1)`, with `pbCInit = 1.25`, `pbCBase = 19652`.
  Invalid actions are masked at the root only.
- `gumbelMuZeroRootActionSelection` — sequential-halving schedule
  (from `Tyr/Mctx/SeqHalving.lean`) scores
  `gumbel + logits + normalizedQ` among actions on the current visit round.
  Each policy caches its one required schedule in `extraData`; direct selector
  calls without matching cache metadata construct only the needed schedule.
- `gumbelMuZeroInteriorActionSelection` — deterministic:
  `argmax` of `softmax(logits + completedQ) − visits / (1 + Σvisits)`.

Q-transforms normalize raw Q-values before selection
(`Tyr/Mctx/QTransforms.lean`): `qtransformByMinMax` (known value bounds),
`qtransformByParentAndSiblings` (default for MuZero/AlphaZero), and
`qtransformCompletedByMixValue` (default for Gumbel MuZero; completes
unvisited Q-values with the mixed value of Appendix D of the Gumbel MuZero
paper via `computeMixedValue`).

### Policies

The public entry points are in `Tyr/Mctx/Policies.lean`:

- `muzeroPolicy` — fresh tree each call, symmetric Dirichlet root noise,
  PUCT selection everywhere.
- `alphazeroPolicy` — same selection, but takes an optional
  `searchTree` to continue from and a `maxNodes` capacity override, enabling
  subtree persistence across environment steps.
- `gumbelMuZeroPolicy` — Gumbel root with sequential halving,
  deterministic interior; `maxNumConsideredActions := 16`,
  `gumbelScale := 1.0`. `actionWeights` are
  `softmax(priorLogits + completedQvalues)` rather than raw visit
  probabilities.

All three accept `invalidActions : Option (Array Bool)` (`true` = invalid),
`maxDepth`, a `qtransform` override, and exploration hyperparameters.

### Batched API

`Tyr/Mctx/Batched.lean` mirrors the stack for a batch of independent search
problems. `BatchedTree S E` is literally an `Array` of per-row trees; the
point is the `BatchedRecurrentFn`:

```lean
abbrev BatchedRecurrentFn (P S : Type) :=
  P → UInt64 → Array Action → Array S → BatchedRecurrentFnOutput × Array S
```

`searchBatchedWithTrees` advances every row one simulation per model call, so
a neural dynamics model sees one batched query per simulation step instead of
one per row. `muzeroPolicyBatched`, `alphazeroPolicyBatched`,
`gumbelMuZeroPolicyBatched`, `resetSearchTreeBatched`, and
`getSubtreeBatched` complete the surface.

### DAG backend

`Tyr/MctxDag/` re-implements the same pipeline over `DagTree S K E`
(`Tyr/MctxDag/Tree.lean`). The differences from `Tree`:

- A `keys : Array K` slot per node plus `keyToNode : Std.HashMap K NodeIndex`
  and a `numAllocated` bump counter (requires `[BEq K] [Hashable K]`).
- No `parents`/`actionFromParent` arrays: a node can have several parents, so
  backup uses the concrete simulated path (`simulatePath` / `backwardPath` in
  `Tyr/MctxDag/Search.lean`).
- `expandEdgeForBackup` hashes the successor embedding with a user-supplied
  `keyFn : S → K`; if the key is already in the table, the existing node is
  reused as the edge target instead of allocating a new one. At capacity it
  returns the evaluated continuation separately for `backwardPath`. The older
  pair-returning `expandEdge` remains available for compatibility.

The three policies `muzeroPolicyDag`, `alphazeroPolicyDag`,
`gumbelMuZeroPolicyDag` (`Tyr/MctxDag/Policies.lean`) take the same
arguments as their tree counterparts plus `keyFn`, and return
`PolicyOutput (DagTree S K _)`. There is no batched DAG API.

### AlphaGrad bridge

`Tyr/AD/Elim/AlphaGradMctx.lean` instantiates the framework for AlphaGrad-style
vertex elimination over local-Jacobian graphs: `S := AlphaGradState`
(elimination graph + action trace), `P := AlphaGradMctxConfig`,
`K := AlphaGradDagKey` (canonical edge/elimination key via `dagStateKey`, used
as the DAG transposition key). It provides a deterministic
`recurrentFn : RecurrentFn AlphaGradMctxConfig AlphaGradState`, heuristic
priors/values (`heuristicPriorLogits`, `heuristicValue`), an
`invalidActionMask`, and ready-made entry points:

- `searchStep?` — one tree-Gumbel-guided step.
- `searchStepDagWithPolicy?` — one DAG step with
  `AlphaGradDagMctsPolicy.alphaZero | .gumbelMuZero`; AlphaZero returns the
  tree for carry-over.
- `searchEpisode?` / `searchEpisodeDag?` / `searchEpisodeDagGumbel?`
  — run a full elimination episode, returning
  `Except String AlphaGradEpisodeResult` (`actions0`, `order1`, `stepRewards`,
  `totalReward`). `...FromEdges?` / `...FromGraph?` variants build the initial
  state for you; `replayActions?` replays a fixed action sequence.

## Key APIs

Policies (unbatched, `import Tyr.Mctx`, `open torch.mctx`):

| Function | Search tree | Distinctive arguments |
| --- | --- | --- |
| `muzeroPolicy` | `Tree S Unit` (fresh) | `dirichletFraction := 0.25`, `dirichletAlpha := 0.3`, `pbCInit`, `pbCBase`, `temperature` |
| `alphazeroPolicy` | `Tree S Unit` (persistent) | `searchTree : Option (Tree S Unit)`, `maxNodes` |
| `gumbelMuZeroPolicy` | `Tree S GumbelMuZeroExtraData` | `maxNumConsideredActions := 16`, `gumbelScale := 1.0` |

All take `params`, `rngKey : UInt64`, `root : RootFnOutput S`,
`recurrentFn`, `numSimulations`, and optional `invalidActions`, `maxDepth`,
`qtransform`; all return `PolicyOutput _`.

DAG policies (`import Tyr.MctxDag`, `open torch.mctxdag`):
`muzeroPolicyDag`, `alphazeroPolicyDag`, `gumbelMuZeroPolicyDag` — same shape,
plus `keyFn : S → K` and `[BEq K] [Hashable K] [Inhabited K]`.

Lower-level pieces you touch when writing a custom policy:

| Function | Location | Purpose |
| --- | --- | --- |
| `search` / `searchWithTree` | `Tyr/Mctx/Search.lean` | run search from a root / existing tree |
| `instantiateTreeFromRoot(WithCapacity)` | `Tyr/Mctx/Search.lean` | allocate a tree |
| `updateTreeWithRoot` | `Tyr/Mctx/Search.lean` | refresh root prior/value on a reused tree |
| `getSubtree` / `resetSearchTree` | `Tyr/Mctx/Tree.lean` | subtree reuse, tree wipe |
| `muzeroActionSelection` etc. | `Tyr/Mctx/ActionSelection.lean` | plug-in selectors |
| `qtransformBy*` | `Tyr/Mctx/QTransforms.lean` | plug-in Q normalizers |
| `searchDag` / `searchWithDag` | `Tyr/MctxDag/Search.lean` | DAG equivalents |
| `searchBatched(WithTrees)` | `Tyr/Mctx/Batched.lean` | batched search loop |

AlphaGrad configs (`Tyr/AD/Elim/AlphaGradMctx.lean`):

```lean
structure AlphaGradMctsConfig where
  numSimulations : Nat := 32
  maxDepth : Option Nat := none
  maxNumConsideredActions : Nat := 16
  gumbelScale : Float := 1.0
  dagMaxNodes : Option Nat := none        -- DAG AlphaZero capacity override
  dagDirichletFraction : Float := 0.0     -- DAG AlphaZero root noise
  dagTemperature : Float := 1.0
```

`AlphaGradMctxConfig` carries the environment knobs: `rewardMode`,
`discount := 1.0`, `invalidActionPenalty := -1.0e6`,
`infeasibleStatePenalty := -1.0e4`, `terminalBonus`, `maxEpisodeSteps`,
plus constraint/cost specifications.

## Usage example

Reconstructed example (from `Tests/TestMctx.lean:37-69`): a one-step bandit
where action 3 is masked invalid and the prior decides.

```lean
import Tyr.Mctx

open torch.mctx

def root : RootFnOutput Unit := {
  priorLogits := #[-1.0, 0.0, 2.0, 3.0]
  value := 0.0
  embedding := ()
}

def banditStep : RecurrentFn Unit Unit := fun _params _rng action _emb =>
  let rewards := #[0.0, 0.0, 0.0, 0.0]
  ({ reward := rewards.getD action 0.0
     discount := 0.0
     priorLogits := #[0.0, 0.0, 0.0, 0.0]
     value := 0.0 }, ())

-- The test asserts: out.action = 2 (highest-prior valid action) and
-- out.actionWeights = #[0.0, 0.0, 1.0, 0.0] after a single simulation.
def out :=
  muzeroPolicy
    (params := ()) (rngKey := 0) (root := root)
    (recurrentFn := banditStep)
    (numSimulations := 1)
    (invalidActions := some #[false, false, false, true])
    (dirichletFraction := 0.0)
```

Subtree reuse across environment steps, as the AlphaZero self-play loop does
(from `Examples/AlphaGradPort/PolicyTrain.lean:638-661`):

```lean
let out := alphazeroPolicy
  (params := searchParams) (rngKey := key)
  (root := root) (recurrentFn := recurrentFromNet)
  (numSimulations := cfg.numSimulations)
  (searchTree := tree?)                 -- continue from last step's subtree
  (invalidActions := some invalid)
-- ... apply out.action in the real environment ...
tree? := if t.done then none else some (getSubtree out.searchTree out.action)
```

Full AlphaGrad episode planning (from `Examples/AlphaGradPort/A0Train.lean:64-75`):

```lean
open Tyr.AD.Elim
-- task : Examples.AlphaGradPort.TaskSpec with envCfg, mctsCfg, graph, numVertices
let result := searchEpisodeDagGumbelFromGraph?   -- or searchEpisodeFromGraph? / searchEpisodeDagFromGraph?
  task.envCfg task.mctsCfg seed task.graph task.numVertices
-- result : Except String AlphaGradEpisodeResult
```

Runnable front ends (registered in `lakefile.lean`):

```bash
lake exe AlphaGradRoeFlux1dA0 [episodes]     # RoeFlux_1d elimination-planning demo
lake exe AlphaGradPortSweep                  # task sweep
lake exe AlphaGradPolicyTrain <mode> [task] [epochs] [episodes-per-epoch]
```

Tests live in `Tests/TestMctx*.lean` and `Tests/TestMctxDag.lean` (run via
`lake exe test_runner`); `Tests/MctxData/` holds reference data with upstream
provenance. The strict reference tests compare deterministic search topology,
actions, visits, values, and Q-values. The original large dumps remain unchanged
as historical data. See the [fixture guide](../Tests/MctxData/README.md) for
generation and migration details.

The native `mctx_bench` executable compares cached scheduling and constant-time
allocation with the previous algorithms, checking equivalent results before
reporting CPU timings. See [the benchmark guide](../benchmarks/README-mcts.md).

## Exploration and reproducibility

MuZero and AlphaZero use the same exploration controls in the tree, batched,
and DAG backends:

- `dirichletAlpha` is the symmetric Dirichlet concentration. Small values
  produce sparse root noise; large values concentrate it near uniform.
  `dirichletFraction` mixes that noise with the network prior, over legal
  actions only. Set the fraction to zero to disable root noise.
- At positive `temperature`, acting samples from
  `visitCounts ** (1 / temperature)`. Nonpositive temperatures choose the
  first maximum deterministically. The returned `actionWeights` remain the
  untempered visit probabilities, normalized over legal actions, for training.
  This sampling rule follows [upstream mctx's policy API](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py).
- Gumbel MuZero uses independent Gumbel draws for root sequential halving and
  retains its completed-Q policy target. Setting `gumbelScale := 0.0`
  disables Gumbel noise.

`Tyr/Mctx/Sampling.lean` provides shared SplitMix64 streams, gamma-based
Dirichlet draws, and categorical/Gumbel sampling. Root noise, recurrent search,
and final action selection use separate streams; batched rows derive separate
keys. Repeating a key and inputs reproduces a run on the same platform, but the
PRNG is not bit-compatible with JAX and seeded trajectories differ from the old
placeholder sampler. Floating-point library differences can affect draws
across platforms. The root-noise parameter is now named `dirichletAlpha`,
replacing the previously ignored `_dirichletAlpha` argument.

Invalid actions have exactly zero output weight when at least one legal action
exists, including with zero simulations (uniform fallback over legal actions).
Zero-visit actions otherwise remain impossible even at high temperature.
Callers must provide a nonempty action set with at least one legal action for a
valid selection: the total API retains its legacy uniform fallback for an
all-invalid mask and returns action 0 for an empty action set.

For numerical robustness, Dirichlet samples are normalized in log space,
including a scaled-log calculation below concentration one. Nonpositive or
nonfinite alpha disables root noise; the standalone `Sampling.dirichlet`
helper returns uniform weights for such alpha. Fractions are clamped to [0, 1]
and NaN fractions disable noise. NaN temperature chooses the greedy action;
positive infinity samples uniformly over positive visit weights. Model logits
should be finite, with negative infinity permitted for zero prior mass provided
at least one legal action has a finite logit.

## Related guides

- [Getting started](getting-started.md) — build and run the test suite.
- [Tensors](core/tensors.md) — the tensor stack; Mctx is deliberately
  tensor-free, `Examples/AlphaGradPort/PolicyTrain.lean` shows the bridge
  (network outputs feed `RootFnOutput`).
- [Autodiff](autodiff.md) — the `Tyr/AD/Elim` elimination machinery the
  AlphaGrad bridge plans over.
- [Modules](modules.md) and [Optimization](optimization.md) — used by the
  policy trainer to fit the prior/value network.
- [Examples and testing](examples-and-testing.md) — how the `Tests/` suites
  and example executables are wired.

Exhaustive per-symbol documentation for `Tyr.Mctx`, `Tyr.MctxDag`, and the
AlphaGrad bridge is generated by doc-gen4 (see `docbuild/`).
