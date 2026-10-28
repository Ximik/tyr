import Tyr
import Tyr.Optim
import Tyr.Model.BranchingFlows.MoleculeTransformer
import Tyr.Model.BranchingFlows.QM9

/-!
  Examples/BranchingFlows/MoleculeTransformerTrainDemo.lean

  QM9-shaped molecule transformer overfit training check.
  This uses the same bridge packing and trainStepMolecule loss path as the
  molecule example, but the trainable model is a transformer with pairwise
  coordinate-distance attention bias and all four molecule heads.
-/

namespace Examples.BranchingFlows

open torch
open torch.branching

private def demoJson : String :=
  "{\"name\":\"water\",\"smiles\":\"O\",\"atoms\":[{\"label\":8,\"coord\":[0.0,0.0,0.0]},{\"label\":1,\"coord\":[0.95,0.0,0.0]},{\"label\":1,\"coord\":[-0.24,0.93,0.0]}]}"

private def exceptToIO {α : Type} (context : String) (x : Except String α) : IO α :=
  match x with
  | .ok value => pure value
  | .error e => throw (IO.userError s!"{context}: {e}")

def runDemo : IO Unit := do
  let vocabSize : Nat := 10
  let vocab : Nat := 10
  let maskToken : Nat := 9
  let maxLen : Nat := 12
  let heads : Nat := 2
  let headDim : Nat := 8
  let mlp : Nat := 32
  let steps : Nat := 160
  let lr : Float := 2.5e-2
  let bridgeCfg := MoleculeBridgeConfig.qm9 vocabSize maskToken
  let labelDFM := DistNoisyDiscreteConfig.qm9 vocabSize maskToken
  let record ← exceptToIO "parse molecule transformer train fixture" (parseQM9MoleculeJson demoJson)
  let target ← exceptToIO "convert molecule transformer train fixture"
    (record.toBranchingState { vocabSize? := some vocabSize, maskToken? := some maskToken })

  let (bridgeResult, _rng) :=
    branchingBridge
      (fun cfg x0 x1 t0 t => MoleculeBridgeConfig.bridge cfg x0 x1 t0 t)
      bridgeCfg
      (fun _root => MoleculeBridgeConfig.maskedAtom bridgeCfg)
      #[target]
      #[0.5]
      TimeDist.betaOneThreeHalves
      TimeDist.uniform
      (sequentialUniformPolicy MoleculeAtom)
      (MoleculeBridgeConfig.anchorMerge bridgeCfg)
      (coalescenceFactor := 1.0)
      (maxLen := some maxLen)
      (deletionPad := 1.5)
      (rng := { state := 20260618 })

  let cfg : BranchingTrainConfig := {
    maxLen := maxLen
    padToken := 0
    anchorWeight := 1.0
    splitsWeight := 0.25
    delWeight := 0.25
    weightDecay := 0.0
    gradClip := 0.0
  }

  torch.manualSeed 20260618
  let params ← MoleculeTransformerParams.init vocab heads headDim mlp
  let opt := Optim.adamw (lr := lr) (weight_decay := 0.0)
  let optState := opt.init params
  let mut params := params
  let mut optState := optState
  let mut firstTotal := 0.0
  let mut firstCoord := 0.0
  let mut firstLabel := 0.0
  let mut firstSplits := 0.0
  let mut firstDel := 0.0
  let mut lastTotal := 0.0
  let mut lastCoord := 0.0
  let mut lastLabel := 0.0
  let mut lastSplits := 0.0
  let mut lastDel := 0.0

  for step in [:steps] do
    let (params', optState', report) ←
      trainStepMolecule (maxLen := maxLen) (vocab := vocab) cfg
        (moleculeTransformerModel (maxLen := maxLen) (vocab := vocab)
          (heads := heads) (headDim := headDim) (mlp := mlp))
        params optState bridgeResult lr (some labelDFM)
    params := params'
    optState := optState'
    if step == 0 then
      firstTotal := report.total
      firstCoord := report.coord
      firstLabel := report.label
      firstSplits := report.splits
      firstDel := report.del
    lastTotal := report.total
    lastCoord := report.coord
    lastLabel := report.label
    lastSplits := report.splits
    lastDel := report.del
    if step % 40 == 0 then
      IO.println s!"molecule_transformer step={step} total={report.total} coord={report.coord} label={report.label} splits={report.splits} del={report.del}"

  if !(lastTotal < firstTotal * 0.6) then
    throw (IO.userError s!"molecule transformer training did not reduce total loss enough: first={firstTotal}, last={lastTotal}")
  if !(lastCoord < firstCoord * 0.8) then
    throw (IO.userError s!"molecule transformer training did not reduce coordinate loss enough: first={firstCoord}, last={lastCoord}")
  if !(lastLabel < firstLabel * 0.9) then
    throw (IO.userError s!"molecule transformer training did not reduce label loss enough: first={firstLabel}, last={lastLabel}")
  if !(lastSplits < firstSplits * 0.95) then
    throw (IO.userError s!"molecule transformer training did not reduce split loss enough: first={firstSplits}, last={lastSplits}")
  if !(lastDel < firstDel * 0.95) then
    throw (IO.userError s!"molecule transformer training did not reduce deletion loss enough: first={firstDel}, last={lastDel}")

  IO.println s!"molecule_transformer init_total={firstTotal} final_total={lastTotal}"
  IO.println s!"molecule_transformer init_coord={firstCoord} final_coord={lastCoord}"
  IO.println s!"molecule_transformer init_label={firstLabel} final_label={lastLabel}"
  IO.println s!"molecule_transformer init_splits={firstSplits} final_splits={lastSplits}"
  IO.println s!"molecule_transformer init_del={firstDel} final_del={lastDel}"

def _root_.main (_args : List String) : IO UInt32 := do
  runDemo
  pure 0

end Examples.BranchingFlows
