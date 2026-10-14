import Tyr
import Tyr.Optim
import Tyr.Model.BranchingFlows.MoleculeTrain
import Tyr.Model.BranchingFlows.QM9

/-!
  Examples/BranchingFlows/MoleculeTrainDemo.lean

  QM9-shaped molecule overfit training check.
  Parses a tiny preprocessed molecule fixture, builds a BranchingFlows bridge
  batch, trains a minimal coordinate/label model, and checks that the loss drops.
-/

namespace Examples.BranchingFlows

open torch
open torch.branching

structure MoleculeToyParams (vocab : UInt64) where
  coordW : Tensor { shape := #[3, 3] }
  coordB : Tensor { shape := #[3] }
  labelTable : Tensor { shape := #[vocab, vocab] }
  deriving TensorStruct

namespace MoleculeToyParams

def init (vocab : UInt64) : IO (MoleculeToyParams vocab) := do
  let coordW := zerosDynamic #[3, 3]
  let coordB := zerosDynamic #[3]
  let labelTable := zerosDynamic #[vocab, vocab]
  pure { coordW, coordB, labelTable }

end MoleculeToyParams

def moleculeToyModel {maxLen vocab : UInt64} :
    BranchingMoleculeModel maxLen vocab (MoleculeToyParams vocab) :=
  { forward := fun {batch} params coord label _t _padmask => do
      let coordPred := Tensor.affine3d coord params.coordW params.coordB
      let labelLogits :=
        Tensor.embedding (batch := batch) (seq := maxLen) (vocab := vocab) (embed := vocab)
          label params.labelTable
      let splitLogits := zerosDynamic #[batch, maxLen]
      let delLogits := zerosDynamic #[batch, maxLen]
      pure (coordPred, labelLogits, splitLogits, delLogits) }

private def demoJson : String :=
  "{\"name\":\"water\",\"smiles\":\"O\",\"atoms\":[{\"label\":8,\"coord\":[0.0,0.0,0.0]},{\"label\":1,\"coord\":[0.95,0.0,0.0]},{\"label\":1,\"coord\":[-0.24,0.93,0.0]}]}"

private def exceptToIO {α : Type} (context : String) (x : Except String α) : IO α :=
  match x with
  | .ok value => pure value
  | .error e => throw (IO.userError s!"{context}: {e}")

def runDemo : IO Unit := do
  let vocabSize : Nat := 10
  let vocab : UInt64 := 10
  let maskToken : Nat := 9
  let maxLen : UInt64 := 8
  let steps : Nat := 100
  let lr : Float := 7.5e-2
  let bridgeCfg := MoleculeBridgeConfig.qm9 vocabSize maskToken
  let labelDFM := DistNoisyDiscreteConfig.qm9 vocabSize maskToken
  let record ← exceptToIO "parse molecule train fixture" (parseQM9MoleculeJson demoJson)
  let target ← exceptToIO "convert molecule train fixture"
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
      (coalescenceFactor := 0.0)
      (maxLen := some maxLen.toNat)
      (rng := { state := 20260617 })

  let cfg : BranchingTrainConfig := {
    maxLen := maxLen
    padToken := 0
    anchorWeight := 1.0
    splitsWeight := 0.0
    delWeight := 0.0
    weightDecay := 0.0
    gradClip := 0.0
  }

  let params ← MoleculeToyParams.init vocab
  let opt := Optim.adamw (lr := lr) (weight_decay := 0.0)
  let optState := opt.init params
  let mut params := params
  let mut optState := optState
  let mut firstTotal := 0.0
  let mut firstCoord := 0.0
  let mut firstLabel := 0.0
  let mut lastTotal := 0.0
  let mut lastCoord := 0.0
  let mut lastLabel := 0.0

  for step in [:steps] do
    let (params', optState', report) ←
      trainStepMolecule (maxLen := maxLen) (vocab := vocab) cfg
        (moleculeToyModel (maxLen := maxLen) (vocab := vocab))
        params optState bridgeResult lr (some labelDFM)
    params := params'
    optState := optState'
    if step == 0 then
      firstTotal := report.total
      firstCoord := report.coord
      firstLabel := report.label
    lastTotal := report.total
    lastCoord := report.coord
    lastLabel := report.label
    if step % 20 == 0 then
      IO.println s!"molecule_branching step={step} total={report.total} coord={report.coord} label={report.label}"

  if !(lastTotal < firstTotal * 0.5) then
    throw (IO.userError s!"molecule BranchingFlows training did not reduce total loss enough: first={firstTotal}, last={lastTotal}")
  if !(lastCoord < firstCoord * 0.5) then
    throw (IO.userError s!"molecule BranchingFlows training did not reduce coordinate loss enough: first={firstCoord}, last={lastCoord}")
  if !(lastLabel < firstLabel * 0.8) then
    throw (IO.userError s!"molecule BranchingFlows training did not reduce label loss enough: first={firstLabel}, last={lastLabel}")

  IO.println s!"molecule_branching init_total={firstTotal} final_total={lastTotal}"
  IO.println s!"molecule_branching init_coord={firstCoord} final_coord={lastCoord}"
  IO.println s!"molecule_branching init_label={firstLabel} final_label={lastLabel}"

def _root_.main (_args : List String) : IO UInt32 := do
  runDemo
  pure 0

end Examples.BranchingFlows
