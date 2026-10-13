import Tyr
import Tyr.Optim
import Tyr.Model.BranchingFlows
import Tyr.Model.BranchingFlowsTrain

/-!
  Examples/BranchingFlows/ContinuousTrainDemo.lean

  Continuous-flow training demo using BranchingFlows + Torch.
  Builds a fixed bridge batch, trains a tiny affine endpoint predictor, and
  checks that the anchor loss decreases.
-/

namespace Examples.BranchingFlows

open torch
open torch.branching

private def clamp01 (x : Float) : Float :=
  max 0.0 (min x 1.0)

private def uniformTimeDist : TimeDist :=
  { cdf := fun t => clamp01 t,
    pdf := fun t => if t < 0.0 || t > 1.0 then 0.0 else 1.0,
    quantile := fun p => clamp01 p }

private def linBridge {dim : UInt64} (_ : Unit)
    (x0 x1 : STensor #[dim]) (t0 t1 : Float) : STensor #[dim] :=
  let dt := t1 - t0
  x0 + (x1 - x0) * dt

structure ToyParams (dim : UInt64) where
  W : STensor #[dim, dim]
  b : STensor #[dim]
  deriving TensorStruct

namespace ToyParams

def init (dim : UInt64) : IO (ToyParams dim) := do
  let W := zerosDynamic #[dim, dim]
  let b := zerosDynamic #[dim]
  pure { W, b }

end ToyParams

def toyModel {maxLen dim : UInt64} : BranchingModelContinuous maxLen dim (ToyParams dim) :=
  { forward := fun {batch} params state _t => do
      let anchorPred := Tensor.affine3d state params.W params.b
      let splitLogits := zerosDynamic #[batch, maxLen]
      let delLogits := zerosDynamic #[batch, maxLen]
      pure (anchorPred, splitLogits, delLogits) }

private def mkVec2 (x y : Float) : STensor #[2] :=
  Tensor.reshapeChecked (floatsDynamic #[x, y]) #[2]

private def fixedState : BranchingState (STensor #[2]) :=
  BranchingState.mkDefault
    #[mkVec2 1.0 2.0, mkVec2 (-1.0) 0.5, mkVec2 0.25 (-0.75), mkVec2 2.0 (-1.0)]
    #[0, 0, 0, 0]

def runDemo : IO Unit := do
  let dim : UInt64 := 2
  let maxLen : UInt64 := 8
  let steps : Nat := 80
  let lr : Float := 5.0e-2
  let x1 := fixedState
  let x1s := #[x1]
  let times := #[0.5]
  let x0Sampler : FlowNode (STensor #[dim]) → STensor #[dim] := fun _node => zerosDynamic #[dim]

  let (bridgeResult, _rng) :=
    branchingBridge (linBridge (dim := dim)) () x0Sampler x1s times
      uniformTimeDist uniformTimeDist (sequentialUniformPolicy (STensor #[dim]))
      canonicalAnchorMerge
      (coalescenceFactor := 0.0)
      (maxLen := some maxLen.toNat)
      (rng := { state := 123 })

  let cfg : BranchingTrainConfig := {
    maxLen := maxLen
    anchorWeight := 1.0
    splitsWeight := 0.0
    delWeight := 0.0
    weightDecay := 0.0
    gradClip := 0.0
  }

  let params ← ToyParams.init dim
  let opt := Optim.adamw (lr := lr) (weight_decay := 0.0)
  let optState := opt.init params
  let mut params := params
  let mut optState := optState
  let mut firstLoss := 0.0
  let mut lastLoss := 0.0
  for step in [:steps] do
    let (params', optState', report) ←
      trainStepContinuous (maxLen := maxLen) (dim := dim) cfg
        (toyModel (maxLen := maxLen) (dim := dim)) params optState bridgeResult lr
    params := params'
    optState := optState'
    if step == 0 then
      firstLoss := report.anchor
    lastLoss := report.anchor
    if step % 20 == 0 then
      IO.println s!"continuous_branching step={step} anchor_loss={report.anchor}"
  if !(lastLoss < firstLoss * 0.5) then
    throw (IO.userError s!"continuous BranchingFlows training did not reduce anchor loss enough: first={firstLoss}, last={lastLoss}")
  IO.println s!"continuous_branching init_loss={firstLoss} final_loss={lastLoss}"

def _root_.main (_args : List String) : IO UInt32 := do
  runDemo
  pure 0

end Examples.BranchingFlows
