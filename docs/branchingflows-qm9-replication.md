# BranchingFlows QM9 Molecule Generation Replication

This note records how to reproduce the molecule generation example described by
the Branching Flows paper in this Tyr checkout.

## Sources Checked

- `../BranchingFlows.jl`: contains the generic Julia `CoalescentFlow`,
  `branching_bridge`, and forward-time `Flowfusion.step`/`gen` pattern, plus a
  toy mixed continuous/discrete demo. It does not contain a QM9 molecule example.
- `../MoleculeFlow.jl`: RDKit/PythonCall wrapper for molecule IO, fingerprints,
  descriptors, drawing, and analysis. It is useful for evaluation/export, not as
  the generator implementation.
- arXiv source for `2511.09465`: the QM9 recipe is in
  `refactored_appendices/F2_analyses.tex`.

## Published QM9 Setup

Data:

- Use QM9 coordinate data.
- Keep the canonical-SMILES heavy atom order.
- Move hydrogens so each hydrogen is inserted before its nearest heavy atom,
  ordered by distance if multiple hydrogens attach to the same heavy atom.
- Each sequence element is one atom: 3D coordinates plus a discrete atom label.

Branching flow:

- Initial state length is always 1.
- Initial coordinates are sampled from `N(0, 1)` per coordinate.
- Initial discrete label is the mask/dummy token.
- Continuous base process: endpoint-conditioned time-inhomogeneous OU bridge
  with mean reversion `theta = 5`; variance decays from `10` at `t = 0` to
  `0.001` by `t = 1`.
- Discrete base process: DFM convex interpolation between `x0`, uniform noise,
  and `x1`, using `Beta(2, 2)` for both hazards and `omega_u = 0.2`.
- Branching hazard distribution: `Beta(1, 3/2)`.
- Deletion hazard distribution: `Uniform(0, 1)`.
- Deletion padding: sample a Poisson number of duplicate-to-delete atoms with
  mean `20%` of the `x1` length, with replacement, inserting duplicates on
  either side of the original. In the Julia/Tyr bridge parameterization this is
  `deletionPad := 1.2`; the value is the expected padded-length multiplier
  when it is at least `1.0`, not the extra fraction.
- Anchors: weighted average for continuous coordinates, mask token for discrete
  atom labels.

Model/training:

- 12-layer transformer.
- 12 heads, head dimension 64, embedding dimension 384.
- Atom positions use Random Fourier Features.
- Pairwise spatial features feed each layer through learnable layer/head-specific
  attention bias.
- RoPE encodes primary sequence order.
- Final six layers additively update atom positions and recompute spatial pair
  features.
- Heads predict endpoint coordinates, endpoint atom-label logits, log expected
  future bifurcations by `t = 1`, and deletion logits.
- The appendix says 500k batches of size 128 with Muon, post-warmup LR `0.005`,
  and linear cooldown for the last 50k batches. A later ablation section says the
  main unconditional figure models used 800k iterations, so parity target should
  be chosen explicitly.
- Sampling uses the schedule
  `t = 1 - (cos(pi * s) + 1) / 2` for `s = 0, 1/1000, ..., 1`.

Evaluation:

- Export atom point clouds to `.xyz`.
- Use OpenBabel to infer bonds and convert `.xyz` to `.sdf`.
- Use RDKit, locally available through `../MoleculeFlow.jl`, for SMILES,
  descriptors, fingerprints, validity, uniqueness, and UMAP inputs.
- PoseBusters metrics are used in the appendix, in addition to distributional
  comparisons of atom counts and molecular descriptors.

## Indexed tensor validation

The BranchingFlows models and training interfaces use indexed tensors. Labels
and molecule masks are Int64; floating weights retain their checkpoint dtype.
Float32/BF16 CPU tests cover forward/backward, Muon rounding, parameter
checkpoint round trips and inference. Mutable static batches allocate each
field independently through IO, preventing aliasing between equal-shaped
label and mask buffers. CUDA capture/replay still needs hardware qualification.

The small standalone training demonstrations can be run with:

```sh
lake exe BranchingFlowsDiscreteTrain
lake exe BranchingFlowsContinuousTrain
lake exe BranchingFlowsMixedTrain
lake exe BranchingFlowsMoleculeTrain
lake exe BranchingFlowsMoleculeTransformerTrain
```

The mixed demo writes four `examples_branching_mixed_*.pt` plotting tensors in
its working directory. These smoke examples and the small dataset CLI
checkpoint/resume check do not establish paper-scale QM9 quality.

## Tyr Mapping

Already close:

- `Tyr/Model/BranchingFlows.lean` has forest sampling, bridge sampling,
  group minima, deletion padding, coalescence policies, and generic anchor
  merging.
- It also now has Julia-compatible time distributions for `Uniform(0, 1)`,
  `Beta(1, 2)`, `Beta(1, 3/2)`, and `Beta(2, 2)`, the default split-intensity
  transform, and scalar/array loss helpers for split counts and deletion logits.
- `Tyr/Model/BranchingFlowsTrain.lean` has discrete and continuous batch packers
  and losses.
- `Examples/BranchingFlows/MixedTrainDemo.lean` is the local toy analogue of the
  Julia mixed continuous/discrete demo.
- `Tyr/Model/BranchingFlows/DiffEq.lean` makes Tyr DiffEq solvers usable as
  BranchingFlows bridges for ODE/SDE prototypes.
- It also contains a scalar endpoint-conditioned OU bridge with constant or
  log-linear variance schedules. This covers the continuous base-process formula
  needed for one coordinate channel and can be lifted over 3D atom positions.
- `Tyr/Model/BranchingFlows/Discrete.lean` ports the Flowfusion
  `DistNoisyInterpolatingDiscreteFlow` schedule: target, uniform, and
  source/mask mixture weights; conditional bridge weights; categorical bridge
  sampling; Euler-style label stepping; and the label-loss time scaling.
- `Tyr/Model/BranchingFlows/Molecule.lean` adds a molecule-shaped atom record:
  3D coordinates plus a discrete label, QM9-style OU defaults, a mask-token
  anchor merge, and a bridge that runs OU coordinates plus optional DFM labels
  through `branchingBridge`.
- `Tyr/Model/BranchingFlows/MoleculeTrain.lean` packs molecule bridge results
  into one tensor batch with coordinates, coordinate anchors, labels,
  label-anchor targets, masks, split targets, and deletion targets.
  `moleculeLosses` applies the Julia/Flowfusion time factors: the DFM label
  scale for labels and `1 / (1.2 - t)` for coordinate, split, and deletion
  losses. This is the demo's explicit `scalefloss(..., 1, 0.2)` override, not
  Flowfusion's squared default. Coordinate and label weights remain
  independently configurable.
- It exposes both the compatibility `trainStepMolecule` AdamW path and
  `trainStepMoleculeMuon`. The latter matches the Julia demo's momentum,
  Nesterov blend, first-dimension matrix flattening, Newton--Schulz
  orthogonalization, aspect scaling, and decoupled weight decay, including for
  vector parameter leaves.
- `Tyr/Model/BranchingFlows/Molecule.lean` also has `MoleculeModelPrediction`,
  `moleculeBranchingStep`, and `moleculeBranchingGenerate`, which adapt
  coordinate endpoint predictions plus atom-label logits into the generic
  forward event path using OU coordinate stepping and DFM label stepping. By
  default they suppress a split when the atom label changed in the same step,
  matching Julia's discrete-state admissibility rule.
- `Tyr/Model/BranchingFlows/QM9.lean` defines the preprocessing boundary for
  QM9 records.  It parses single JSON molecules, JSON molecule arrays, and
  JSONL batches; validates atom count, finite coordinates, label vocabulary,
  and reserved mask-token use; and converts records into
  `BranchingState MoleculeAtom` plus the length-one masked source state used by
  generation.
- It also contains the molecule-specific deletion-padding hook
  `maskDeletedMoleculeLabels` and raw `.xyz` export helpers
  `moleculeStateToXYZ`/`writeMoleculeXYZ`, including optional token-to-symbol
  mapping for mask labels.
- `Examples/BranchingFlows/MoleculeGenerationDemo.lean` is the runnable local
  smoke example. It parses an inline preprocessed QM9-style JSONL fixture, runs
  a bridge sample with `deletionPad := 1.2`, applies the deleted-label mask
  modifier, runs `moleculeBranchingGenerate` with an oracle prediction model,
  and writes target, bridge, and generated `.xyz` files.
- `Examples/BranchingFlows/MoleculeTrainDemo.lean` is the runnable training
  smoke. It parses an inline preprocessed molecule fixture, builds a
  QM9-shaped bridge batch, trains a small coordinate/label model through
  `trainStepMolecule`, and fails if total, coordinate, or label loss does not
  decrease.
- `Tyr/Model/BranchingFlows/MoleculeTransformer.lean` adds a reusable
  Torch-backed molecule transformer with head-specific pairwise
  coordinate-distance attention bias and endpoint-coordinate, atom-label,
  split-count, and deletion heads.
- `Examples/BranchingFlows/MoleculeTransformerTrainDemo.lean` trains that
  transformer on the same QM9-shaped bridge path, with nonzero split and
  deletion loss weights, and fails if any of the four head losses does not
  decrease.
- `Examples/BranchingFlows/MoleculeTrainGenerate.lean` is the dataset-backed
  end-to-end path. It uses Muon, samples half of training times directly from
  branching events, applies Julia's `10 : 1/3 : 1 : 1` coordinate/label/split/
  deletion loss balance, has no extra generation-only split cap by default,
  and saves/restores Muon momentum together with the model checkpoint.

Preprocessed molecule records should look like:

```json
{"name":"example","smiles":"O","atoms":[{"label":8,"coord":[0.0,0.1,-0.2]}]}
```

Atom coordinates may also be emitted as `x`, `y`, and `z` fields.  The label
field should be the integer atom-token id used by the model vocabulary; the
loader also accepts `atom_label` as an alias.

Run the local smoke example with:

```bash
lake env lean --run Examples/BranchingFlows/MoleculeGenerationDemo.lean /tmp/tyr_branching_molecule_demo
```

It writes:

- `/tmp/tyr_branching_molecule_demo_target.xyz`
- `/tmp/tyr_branching_molecule_demo_bridge.xyz`
- `/tmp/tyr_branching_molecule_demo_generated.xyz`

The generated file should contain a three-atom water-shaped sample from the
oracle model. This is a shape and event-path check, not a trained molecular
model.

Run the local molecule training smoke with:

```bash
lake exe BranchingFlowsMoleculeTrain
```

This is a trainability check, not the paper-scale architecture. It should print
initial/final total, coordinate, and label losses and fail if the tiny model
does not overfit the fixed molecule fixture.

Run the transformer training smoke with:

```bash
lake exe BranchingFlowsMoleculeTransformerTrain
```

This is still a small one-block overfit check, but it proves that the
QM9-shaped transformer, pairwise spatial attention bias, and all four molecule
heads train through `trainStepMolecule`.

Remaining for a quantitative paper replication:

- Run the full QM9 preprocessing/training pipeline at the chosen 500k or 800k
  budget. `scripts/qm9_xyz_to_branching_jsonl.py` and
  `scripts/branchingflows/run_qm9_paper.sh` provide the local path, but no
  paper-scale GPU result is checked into the repository.
- Decide whether deterministic conditional-mean coordinate targets are
  sufficient or whether the intended experiment requires stochastic
  bridge-variance targets.
- Add the full OpenBabel/RDKit/PoseBusters evaluation suite and compare the
  resulting validity, uniqueness, geometry, and descriptor distributions with
  the paper's reported figures.

## Practical Replication Order

1. Run `lake exe BranchingFlowsMoleculeTrainGenerate --profile smoke` to check
   sampled bridges, all four losses, Muon, checkpointing, forward events,
   lineage, and `.xyz` export together.
2. Convert a small QM9 shard with `scripts/qm9_xyz_to_branching_jsonl.py` and run
   `scripts/branchingflows/run_qm9_paper.sh` with smoke-sized overrides.
3. Launch the full architecture with either the 500k appendix profile or the
   800k main-figure profile and retain its generated run manifest.
4. Convert generated `.xyz` files with OpenBabel and evaluate them through
   `../MoleculeFlow.jl`/RDKit plus PoseBusters.

The shortest useful local integration check is step 1. A paper result still
requires the full sequence above and a substantial GPU training run.
