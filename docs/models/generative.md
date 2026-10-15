# Generative models

This chapter covers Tyr's generative model families: the **BranchingFlows** flow-matching
framework (coalescent/branching generative processes over particle sets, ported from the
Julia `BranchingFlows.jl`, with a QM9 molecule instantiation), the minimal **Flowfusion**
state wrappers it builds on, and the **Flux + VAE** text-to-image stack (a Flux2 Klein 4B
rectified-flow diffusion transformer plus its BFL latent decoder). Use BranchingFlows when
your generative process creates, deletes, and merges particles or tokens during sampling;
use Flux for image generation in a learned latent space.

Everything here is shape-typed Lean 4 over the libtorch FFI. Namespaces:
`torch.branching` (BranchingFlows), `torch.flowfusion`, `torch.flux`, `torch.vae`.

## Architecture and main abstractions

### BranchingFlows core (`Tyr/Model/BranchingFlows.lean`)

The core module is deliberately generic: it implements the combinatorics of branching
(coalescent forest sampling, split/deletion and trajectory bookkeeping) and leaves the
base process (how a particle moves between two times) as a user-supplied function.

**Deterministic RNG.** `Rng` (`:26`) is a single `UInt64` LCG state; every sampler in the
framework is a pure function `Rng → α × Rng`. Samplers include `randFloat`, `randNat`,
`randBernoulli`, `randBinomial`, `randPoisson`, `randExponential`, `randNormal`.

**Time distributions.** `TimeDist` (`:105`) packs `cdf pdf quantile : Float → Float` and
drives split-time and deletion-time sampling. Built-ins: `TimeDist.uniform`,
`betaOneTwo`, `betaOneThreeHalves` (the QM9 split-time distribution), `betaTwoTwo` (used
by the discrete-flow hazards); combinators `survival`, `hazard`, `truncatedPdfFrom`.

**Particles and states.** A `FlowNode α` (`:248`) is a node of the coalescent tree (time,
data, descendant weight, group, branch/del/flow flags, id, children). A
`BranchingState α` (`:280`) is the flat runtime form: one array of particle data plus
parallel bookkeeping arrays (`groupings`, `del`, `ids`, `branchmask`, `flowmask`,
`padmask`). `BranchingState.mkDefault state groupings` builds one with all flags set.

**Anchor merging.** `class AnchorMerge α where merge : α → α → Nat → Nat → α` (`:308`)
combines two coalescing anchors weighted by descendant counts. Instances exist for
`Float`, indexed tensors, products, and `flowfusion.MaskedState α cmaskSpec lmaskSpec`.

**Coalescence policies.** Which pairs of adjacent same-group particles merge is decided
by a `CoalescencePolicy α` (`:393`):

```lean
structure CoalescencePolicy (α : Type) where
  select : Array (FlowNode α) → Option GroupMins → Rng → Option (Nat × Nat) × Rng
  maxCoalescences : Array (FlowNode α) → Nat
  init : Array (FlowNode α) → Rng → Rng
  update : Array (FlowNode α) → Nat → Nat → Nat → Rng → Rng
  reorder : Array (FlowNode α) → Array (FlowNode α)
  shouldAppendOnSplit : Bool := false
```

Built-ins: `sequentialUniformPolicy`, `sequentialUniformBlockMinPolicy`,
`balancedSequentialPolicy`, `richGetRicherSequentialPolicy`,
`sequentialProximityPolicy distance`, `sequentialDeepLineagePolicy`. `GroupMinsSpec`
(`none | uniform min | perGroup`) floors how far coalescence may shrink each group.

**The flow record.** `CoalescentFlow P α` (`:931`) bundles the base-process parameters
`base : P`, the branch and deletion `TimeDist`s, a `splitTransform : Float → Float`
(split-logit → rate, default `defaultSplitTransform`), and the policy. Build with
`CoalescentFlow.mkDefault base branchTime deletionTime` or `mkWithPolicy`.

**Training-side bridge.** Given training endpoints `x1`, `sampleForest` (`:680`) samples
a coalescent forest backward from `x1`, `treeBridge` (`:1301`) walks one tree forward
through a user bridge `P → α → α → Float → Float → α`, and `forestBridge` (`:1357`) /
`branchingBridge` (`:1436`) batch the whole thing:

```lean
def branchingBridge
    (bridge : P → α → α → Float → Float → α)      -- base x0 x1 t0 t1 ↦ xt
    (base : P) (x0Sampler : FlowNode α → α)
    (x1s : Array (BranchingState α)) (times : Array Float)
    (branchTime deletionTime : TimeDist) (policy : CoalescencePolicy α)
    (merger : α → α → Nat → Nat → α)
    ...                                           -- maxLen, deletionPad, rng, samplers
    : BranchingBridgeResult α × Rng
```

`BranchingBridgeResult α` (`:1425`) returns, per batch item, the time actually used, the
raw `Segment`s, the packed intermediate state `Xt : BranchingState α`, the `X1anchor`s,
`descendants`, `del` flags, `splitsTarget` (`descendants - 1`), and `prevCoalescence`
times — everything a model needs as training targets. `deletionPad` pads the batch with
extra deletion events so the deletion head sees positives.

**Forward generation.** The model side supplies a `BranchingStepPrediction α`
(`:962`: `targets`, `splitLogits`, `delLogits` per particle); `branchingStep` (`:1156`)
advances one time interval (base step, Poisson splits, Bernoulli deletions) and
`branchingGenerate` (`:1274`) folds it over a schedule:

```lean
def branchingGenerate
    (baseStep : P → BranchingState α → Array α → Float → Float → Array α)
    (flow : CoalescentFlow P α) (x0 : BranchingState α)
    (model : Float → BranchingState α → BranchingStepPrediction α)
    (schedule : Array Float) ... : BranchingGenerateResult α × Rng
```

`BranchingGenerateResult α` (`:983`) carries `finalState`, the full `trajectory`, per-step
`BranchingStepEvent`s (source index/id, split count, deletion flag, interval), and the
`times`. Because `BranchingState.ids` are anchor ids, not particle identities,
`reconstructLineage result (appendOnSplit := false)` (`:1037`) assigns stable runtime
particle ids across splits and deletes; `appendOnSplit` must match the policy's
`shouldAppendOnSplit`.

**Loss helpers** (`:189-244`): `shiftedPoissonBregmanLoss`/`splitCountLoss` for split
counts, `logitBinaryCrossEntropy` for deletions, masked mean combiners. Tensor-level
versions live in `BranchingFlowsTrain` (below).

**Caveat.** Custom policies must return the selected pair in ascending order:
`sampleForest` (`Tyr/Model/BranchingFlows.lean:709-710`) normalizes `(i, j)` with a
shadowing pattern that collapses a reversed pair `(5, 3)` to `(3, 3)`. All built-in
policies return `(i, i+1)`, so this only bites hand-written policies.

### Bridge adapters (`Tyr/Model/BranchingFlows/`)

- **`Discrete.lean`** — `DistNoisyDiscreteConfig` (`:13`) ports the Flowfusion
  `DistNoisyInterpolatingDiscreteFlow` schedule: mixture weights `k1` (endpoint target),
  `k2` (uniform noise), `k3` (source/mask) with derivatives, conditional weights,
  categorical bridges (`bridge`, `bridgeFrom`, `modeBridgeFrom`), and Euler
  `stepDistribution`s. `DistNoisyDiscreteConfig.qm9 vocabSize maskToken` gives the QM9
  defaults (`omegaUniform := 0.2`).
- **`DiffEq.lean`** — `DiffEqBridgeConfig Term Y VF Control Args Controller` (`:25`)
  makes any `Tyr.DiffEq` solver usable as a BranchingFlows bridge, with the vector field
  parameterized by the future anchor; `ODEBridgeConfig` (`:107`) and `SDEBridgeConfig`
  (`:140`) are specialized abbrevs with `mk` constructors. `OUBridgeConfig` (`:199`) is an
  analytic scalar endpoint-conditioned Ornstein–Uhlenbeck bridge used for molecular
  coordinates. The same file also projects branching results into the event-skeleton
  vocabulary: `graphFromBranchingGenerateResult` (`:458`),
  `graphsFromBranchingBridgeResult`, `summarizeBranchingBridgeResult`.

### Molecule instantiation (`Molecule.lean`, `QM9.lean`)

QM9 elements are `MoleculeAtom`s — `{ coord : Vec3, label : Nat }` (`Molecule.lean:59`).
`MoleculeBridgeConfig` (`:64`) pairs the continuous OU coordinate bridge with an optional
discrete DFM label bridge and a reserved `maskToken`; `MoleculeBridgeConfig.qm9 vocabSize
maskToken` (`:72`) gives the paper defaults. Its `anchorMerge` averages coordinates and
relabels internal anchors with the mask token, and `bridge`/`sampleBridge` combine the OU
coordinate bridge with the DFM (or keep-source) label process.

`MoleculeModelPrediction` (`:197`) is the model-output record (`coordTargets`,
`labelLogits`, `splitLogits`, `delLogits`); generation entry points are
`moleculeBranchingGenerate` (`:283`) for pure models and `moleculeBranchingGenerateIO`
(`:309`) for tensor models in `IO` with an optional `maxStateLen?` guard.
`writeMoleculeTrajectoryJsonl` (`:387`) exports a trajectory plus lineage as JSONL.

`QM9.lean` is the data boundary: `QM9MoleculeRecord` (`:30`) parsed from JSON/JSONL
(`parseQM9MoleculeJsonl`, `loadQM9MoleculeJsonl`), validated and converted by
`qm9RecordsToBranchingStates records (cfg : QM9StateConfig)` (`:238`), and written back
out as XYZ point clouds by `writeMoleculeXYZ` (`:288`) for downstream OpenBabel/RDKit
checks. Chemistry preprocessing (canonicalization etc.) happens outside Lean.

### Training on tensors (`BranchingFlowsTrain.lean`, `MoleculeTrain.lean`, `MoleculeTransformer.lean`)

`BranchingFlowsTrain.lean` packs `BranchingBridgeResult`s into fixed-shape tensors and
provides the generic AdamW training path. `BranchingTrainConfig` holds `maxLen`,
`padToken`, `timeScale`, independent coordinate/label/event loss weights,
`weightDecay`, and `gradClip`. The model interface for the
molecule case (`BranchingModel` in `BranchingFlowsTrain.lean:50` is the discrete-token
analog):

```lean
variable {device : Device}

structure BranchingMoleculeModel (maxLen vocab : UInt64) (Params : Type) where
  forward : {batch : UInt64} → Params → Tensor { shape := #[batch, maxLen, 3], device := device } → Tensor { shape := #[batch, maxLen], dtype := .Int64, device := device } → Tensor { shape := #[batch], device := device }
    → Tensor { shape := #[batch, maxLen], dtype := .Int64, device := device }
    → IO (Tensor { shape := #[batch, maxLen, 3], device := device } × Tensor { shape := #[batch, maxLen, vocab], device := device } × Tensor { shape := #[batch, maxLen], device := device } × Tensor { shape := #[batch, maxLen], device := device })
```

`packBranchingNat` / `packBranchingTensor` (`:107`, `:188`) pack discrete-token and
continuous-tensor states; `packBranchingMolecule` (`MoleculeTrain.lean:62`) packs the
mixed coordinate+label batch (`BranchingMoleculeBatch`, `:18`). Training steps:
`trainStep` (discrete), `trainStepContinuous`, and `trainStepMolecule` (`:221`), all
`[TensorStruct Params]`-generic and returning a per-head loss report.
`sampleMoleculeBridgeBatch` (`MoleculeTrain.lean:271`) is the one-call data loader: pick
random terminal molecules, sample times (optionally oversampling exact branch
times), run `branchingBridge` with the molecule bridges, deletion padding, and
masked-x0 sampling. `trainStepMoleculeMuon` is the Julia-compatible executable
path; `trainStepMolecule` remains available for AdamW callers.

`MoleculeTransformer.lean` provides two ready-made `BranchingMoleculeModel`s:
`MoleculeTransformerParams` (compact; one spatial-attention block with pair-distance
attention bias) via `moleculeTransformerModel` (`:191`), and
`FullMoleculeTransformerParams` (RFF coordinate features, per-layer pair-distance bias,
RoPE, coordinate-update layers) via `fullMoleculeTransformerModel` (`:473`). Both
`derive TensorStruct`. For generation, `moleculeTransformerIOModel` (`:567`) and
`fullMoleculeTransformerIOModel` (`:611`) wrap trained params into the
`Float → BranchingState MoleculeAtom → IO MoleculeModelPrediction` shape that
`moleculeBranchingGenerateIO` consumes (clamping split logits to `-100` at `maxLen`).

### Flowfusion (`Tyr/Model/Flowfusion.lean`)

Minimal wrappers, not a full Flowfusion port. `MaskedState α cmaskSpec lmaskSpec` pairs a state with
conditioning/loss masks stored as `Static (Tensor cmaskSpec)` and
`Static (Tensor lmaskSpec)` so `TensorStruct` traversal skips
them; `Guide α cmaskSpec lmaskSpec` wraps an alternative prediction target with optional masks.
`unmask` (`:67`) and `maskLike` (`:70`) convert. `Guide` has no call sites outside its
own module; `MaskedState` is used only by the `AnchorMerge` instance in
`BranchingFlows.lean:344`.

### Flux (`Tyr/Model/Flux/`)

A Flux2 (Klein 4B) rectified-flow image diffusion transformer. `FluxConfig` (`Config.lean:13`)
carries the architecture dimensions; its field defaults *are* Klein 4B, also exposed as
`FluxConfig.klein4B`.

`FluxModel cfg` (`Model.lean:62`) is one flat `deriving TensorStruct` record: input
projections (`img_in`, `txt_in`), a timestep `MLPEmbedder`, arrays of
`DoubleStreamBlock`s (separate img/txt QKV+SwiGLU paths, joint attention) and
`SingleStreamBlock`s (fused QKV+MLP over the concatenated sequence), three model-level
adaLN `Modulation` layers, and a `LastLayer`. Q/K are RMS-normalized (`QKNorm`); positions
use 4-axis RoPE (`ropeEmbed`/`applyRope`, axes `#[t, h, w, l]`, `theta := 2000`), and the
timestep embedding is a C++ FFI extern (`@[extern "lean_torch_timestep_embedding"]`,
`Modulation.lean:19`). The forward (`Model.lean:165`) takes packed image latents, text
embeddings, timesteps, and both position-ID tensors, returning the velocity prediction:

```lean
variable {device : Device}

def forward {batch img_seq txt_seq : UInt64} (cfg : FluxConfig)
    (model : FluxModel (device := device) cfg)
    (img : Tensor { shape := #[batch, img_seq, cfg.in_channels], device := device })
    (txt : Tensor { shape := #[batch, txt_seq, cfg.context_in_dim], device := device })
    (timesteps : Tensor { shape := #[batch], dtype := .Float32, device := device })
    (img_ids : Tensor { shape := #[batch, img_seq, 4], dtype := .Int64, device := device })
    (txt_ids : Tensor { shape := #[batch, txt_seq, 4], dtype := .Int64, device := device })
    : Tensor { shape := #[batch, img_seq, cfg.in_channels], device := device }
```

`Sampling.lean` implements the Flux2 schedule (`computeTimesteps image_seq_len num_steps`,
empirical-μ time-SNR shift) and Euler integration: `eulerStep` (`:61`), `denoise` (`:74`)
over an explicit noise tensor, `sample` (`:94`) drawing fresh noise.
`computeImagePositionIds height width` and `computeTextPositionIds seq_len` produce the
`[1, seq, 4]` RoPE inputs; `packLatents`/`unpackLatents` (`:156-187`) convert between VAE
spatial format `[batch, 32, H, W]` and the 2×2-patchified sequence format
`[batch, (H/2)·(W/2), 128]`. Weights load from a Flux2 HuggingFace SafeTensors file with
`loadFluxModel path (cfg := FluxConfig.klein4B)` (`Weights.lean:119`). Text conditioning
is **not** part of this module: the model consumes pre-computed embeddings
(`context_in_dim := 7680`), produced in the demo by a Qwen encoder — see
[LLM model families](llms.md).

### VAE (`Tyr/Model/VAE.lean`, `Tyr/Model/VAE/`)

The BFL latent autoencoder decoder for Flux. `Decoder` (`Decoder.lean:28`) is the BFL
stack — `post_quant_conv`, `conv_in`, mid `ResnetBlock`/`AttnBlock`/`ResnetBlock`, four
up-groups with `Upsample`s, GroupNorm + swish + `conv_out` — decoding
`[batch, 32, H, W]` to `[batch, 3, 8H, 8W]`. Building blocks (`ResnetBlock.lean`):
`GroupNormParams`, `Conv2dParams`, `ResnetBlock in_ch out_ch` (with optional
`nin_shortcut`), `swish`.

`AutoEncoder` (`VAE.lean:21`) adds the BatchNorm running statistics used to normalize
packed latents, and the full decode pipeline:

```lean
variable {device : Device}

structure AutoEncoder where
  decoder : Decoder (device := device)
  bn_running_mean : Tensor { shape := #[128], device := device }
  bn_running_var : Tensor { shape := #[128], device := device }
  deriving TensorStruct

def AutoEncoder.invNormalize (ae : AutoEncoder (device := device)) (z : Tensor { shape := #[1, 128, 16, 16], device := device }) : Tensor { shape := #[1, 128, 16, 16], device := device }
def AutoEncoder.decode (ae : AutoEncoder (device := device)) (z : Tensor { shape := #[1, 128, 16, 16], device := device }) : Tensor { shape := #[1, 3, 256, 256], device := device }
def loadAutoEncoder (path : String) (log : Handlers := {})
    (device : Device := .CPU) : IO (AutoEncoder (device := device))
```

**Caveat.** `invNormalize`, `unpackLatents16x16`, `decode`, and `Decoder.forward1` are
hardcoded to the 256×256 path (concrete literal shapes to avoid an FFI issue with type
variables). Other output resolutions need new code — `Decoder.forward` is shape-generic
over `batch` only, not over spatial size.

## Key APIs

BranchingFlows, training a bridge model:

| Entry point | Signature (abridged) | Location |
| --- | --- | --- |
| `branchingBridge` | sample bridge batch from `x1` states | `BranchingFlows.lean:1436` |
| `packBranchingMolecule` | bridge result → `BranchingMoleculeBatch batch maxLen` | `MoleculeTrain.lean:62` |
| `trainStepMolecule` | one AdamW step, returns params + report | `MoleculeTrain.lean:221` |
| `trainStepMoleculeMuon` | one Julia-compatible Muon step, returns params + momentum state + report | `MoleculeTrain.lean` |
| `evalMoleculeLoss` | loss report on a fixed bridge batch | `MoleculeTrain.lean:254` |
| `sampleMoleculeBridgeBatch` | one-call molecule bridge data loader | `MoleculeTrain.lean:271` |
| `moleculeTransformerModel` / `fullMoleculeTransformerModel` | ready `BranchingMoleculeModel`s | `MoleculeTransformer.lean:191,473` |

BranchingFlows, generation:

| Entry point | Signature (abridged) | Location |
| --- | --- | --- |
| `CoalescentFlow.mkDefault` | `base → branchTime → deletionTime → CoalescentFlow P α` | `BranchingFlows.lean:949` |
| `branchingGenerate` | pure model over a schedule | `BranchingFlows.lean:1274` |
| `moleculeBranchingGenerateIO` | IO tensor model, `maxStateLen?` guard | `Molecule.lean:309` |
| `reconstructLineage` | stable particle ids over a trajectory | `BranchingFlows.lean:1037` |
| `writeMoleculeTrajectoryJsonl` / `writeMoleculeXYZ` | export lineage JSONL / XYZ | `Molecule.lean:387`, `QM9.lean:288` |
| `parseQM9MoleculeJsonl` / `qm9RecordsToBranchingStates` | dataset ingestion | `QM9.lean:225,238` |

Flux + VAE:

| Entry point | Signature (abridged) | Location |
| --- | --- | --- |
| `loadFluxModel` | `(path : String) (cfg := FluxConfig.klein4B) : IO (FluxModel cfg)` | `Flux/Weights.lean:119` |
| `computeImagePositionIds` / `computeTextPositionIds` | `[1, seq, 4]` RoPE ids | `Flux/Sampling.lean:109,131` |
| `denoise` | Euler rectified-flow loop, `num_steps := 4` | `Flux/Sampling.lean:74` |
| `sample` | fresh noise → `denoise` | `Flux/Sampling.lean:94` |
| `packLatents` / `unpackLatents` | VAE spatial ↔ packed sequence | `Flux/Sampling.lean:156,174` |
| `loadAutoEncoder` | decoder + BN stats from SafeTensors | `VAE.lean:77` |
| `AutoEncoder.decode` | `Tensor { shape := #[1,128,16,16], device := device } → Tensor { shape := #[1,3,256,256], device := device }` | `VAE.lean:66` |

## Usage examples

Reconstructed example (from `Examples/BranchingFlows/BranchingFlowsDemo.lean`) — toy
bridge over `Float` states, the minimal end-to-end use of the core:

```lean
import Tyr
import Tyr.Model.BranchingFlows

open torch torch.branching

def linBridge (_ : Unit) (x0 x1 : Float) (t0 t1 : Float) : Float :=
  x0 + (x1 - x0) * (t1 - t0)

def runDemo : IO Unit := do
  let x1 := BranchingState.mkDefault #[1.0, 2.0, 3.0, 4.0] #[0, 0, 1, 1]
  let (result, _rng) :=
    branchingBridge linBridge () (fun node => node.data) #[x1] #[0.5]
      TimeDist.uniform TimeDist.uniform (sequentialUniformPolicy Float) canonicalAnchorMerge
      (rng := { state := 123 })
  IO.println s!"segments={result.segments[0]!.size} t={result.t[0]!}"
```

Reconstructed example (from `Examples/BranchingFlows/MoleculeTrainGenerate.lean`) —
dataset-backed molecule training and generation with the compact transformer:

```lean
open torch torch.branching

let bridgeCfg := MoleculeBridgeConfig.qm9 vocabSize maskToken
-- qm9RecordsToBranchingStates returns Except String _; unwrap into IO
let states ← qm9RecordsToBranchingStates records
  { vocabSize? := some vocabSize, maskToken? := some maskToken }
let trainCfg : BranchingTrainConfig := {
  maxLen := 16, coordWeight := 10.0, labelWeight := 1.0 / 3.0,
  splitsWeight := 1.0, delWeight := 1.0, weightDecay := 0.01
}
let model := moleculeTransformerModel (maxLen := 16) (vocab := 10)
  (heads := 2) (headDim := 8) (mlp := 48)
let mut params ← MoleculeTransformerParams.init 10 2 8 48

-- train on freshly sampled bridge batches
let mut rng : Rng := { state := 20260618 }
let mut optState := initMoleculeMuonState params
for step in [:steps] do
  let (batch, rng') ← sampleMoleculeBridgeBatch bridgeCfg states 4 rng
    (useBranchingTimeProb := 0.5) (maxLen := some 16) (deletionPad := 1.2)
  rng := rng'
  let (params', optState', report) ←
    trainStepMoleculeMuon trainCfg model params optState batch 8.0e-3 bridgeCfg.labelDFM
  params := params' ; optState := optState'

-- generate from a single masked atom
let flow := CoalescentFlow.mkDefault bridgeCfg TimeDist.betaOneThreeHalves TimeDist.uniform
let learnedModel := moleculeTransformerIOModel (maxLen := 16) (vocab := 10)
  (heads := 2) (headDim := 8) (mlp := 48) trainCfg.padToken params
let (x0Atom, rng) := MoleculeBridgeConfig.sampleInitialAtom bridgeCfg rng
let x0 := BranchingState.mkDefault #[x0Atom] #[0]
let (generated, _) ← moleculeBranchingGenerateIO flow x0 learnedModel schedule
  (maxStateLen? := some 16) (rng := rng)
writeMoleculeXYZ ⟨"generated.xyz"⟩ generated.finalState "sample" labelSymbol
writeMoleculeTrajectoryJsonl ⟨"trajectory.jsonl"⟩ generated
```

The full executable adds model and Muon-momentum checkpoint resume, LR warmup/cooldown,
profiles (`--profile smoke|paper-qm9-main|paper-qm9-appendix`), and the
`fullMoleculeTransformerModel` architecture (`lake exe BranchingFlowsMoleculeTrainGenerate`).

Reconstructed example (from `Examples/Flux/FluxDemo.lean`) — text-to-image with Flux
Klein 4B (256×256 path, 4-step distilled schedule):

```lean
import Tyr.Model.Qwen.Weights
import Tyr.Model.VAE
import Tyr.Model.Flux.Weights
import Tyr.Model.Flux
open torch torch.qwen torch.vae torch.flux
def fluxExample (qwenDir : String) (tokens : Tensor { shape := #[1, 512], dtype := .Int64, device := .CPU })
    (attnMask : Tensor { shape := #[1, 512], dtype := .Bool, device := .CPU }) : IO Unit := do

  -- text encoder (Qwen), diffusion model, and latent decoder
  let qwen ← loadQwenFluxEmbedderSharded qwenDir QwenConfig.fluxKleinTextEncoder 512 #[8, 17, 26]
  let flux ← loadFluxModel "weights/flux.safetensors" FluxConfig.klein4B
  let vae ← loadAutoEncoder "weights/ae.safetensors"

  -- encode prompt, prepare noise and RoPE position ids
  let txtEmb := qwen.encodeMasked QwenConfig.fluxKleinTextEncoder 512 tokens attnMask
  let txtEmb : Tensor { shape := #[1, 512, FluxConfig.klein4B.context_in_dim], device := .CPU } :=
    (txtEmb.reshapeChecked #[1, 512, FluxConfig.klein4B.context_in_dim]).forgetDType
  let noiseSeq := Tensor.permute (← randomWeight #[1, 128, 16, 16]) #[0, 2, 3, 1]
  let noise : Tensor { shape := #[1, 256, 128], device := .CPU } := Tensor.reshapeChecked noiseSeq #[1, 256, 128]
  let imgIds ← computeImagePositionIds 16 16
  let txtIds ← computeTextPositionIds 512

  -- denoise and decode
  let latents := denoise FluxConfig.klein4B flux noise txtEmb imgIds txtIds 4
  let packed : Tensor { shape := #[1, 128, 16, 16], device := .CPU } :=
    (Tensor.permute (Tensor.reshapeChecked latents #[1, 16, 16, 128]) #[0, 3, 1, 2]).reshapeChecked #[1, 128, 16, 16]
  let image := vae.decoder.forward1 (AutoEncoder.unpackLatents16x16 (vae.invNormalize packed))
  saveImagePPM image "output.ppm"
```

## Related guides

- [core/tensors.md](../core/tensors.md) — `Tensor σ`, shapes, and indexed operations behind every forward here
- [core/tensorstruct.md](../core/tensorstruct.md) — parameter traversal, checkpointing, and target-type conversion
- [autodiff.md](../autodiff.md) — `Tensor.backwardLoss` as used by `trainStepMolecule`
- [optimization.md](../optimization.md) — `Optim.adamw` and optimizer state
- [serialization.md](../serialization.md) — SafeTensors loading and checkpoint save/resume
- [diffeq.md](../diffeq.md) — the solvers behind `DiffEqBridgeConfig`/`ODEBridgeConfig`/`SDEBridgeConfig`
- [event-skeleton.md](../event-skeleton.md) — the interval-skeleton graphs branching results project into
- [llms.md](llms.md) — the Qwen text encoder used for Flux conditioning
- [examples-and-testing.md](../examples-and-testing.md) — the demo executables and `Tests/TestBranchingFlows.lean`

Exhaustive symbol-level documentation for everything in this chapter is generated by doc-gen4 (see `docbuild/`); this guide covers the concepts and the main entry points only.
