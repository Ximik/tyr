import Tyr
import Tyr.Optim
import Tyr.Model.BranchingFlows
import Tyr.Model.BranchingFlowsTrain

/-!
  Examples/BranchingFlows/DiscreteTrainDemo.lean

  Minimal discrete-flow training demo using BranchingFlows + Torch.
  This mirrors the Julia demo structure with a lightweight token model.
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

private def discreteBridge (_ : Unit) (_x0 x1 : Nat) (_t0 _t1 : Float) : Nat :=
  x1

structure ToyParams (vocab hidden : Nat) where
  embed : Tensor { dtype := .Float32, shape := #[vocab, hidden], device := .CPU }
  proj : Tensor { dtype := .Float32, shape := #[vocab, hidden], device := .CPU }
  splitW : Tensor { dtype := .Float32, shape := #[1, hidden], device := .CPU }
  delW : Tensor { dtype := .Float32, shape := #[1, hidden], device := .CPU }
  deriving TensorStruct

namespace ToyParams

def init (vocab hidden : Nat) : IO (ToyParams vocab hidden) := do
  let embed ← randnDynamic #[vocab, hidden]
  let proj ← randnDynamic #[vocab, hidden]
  let splitW ← randnDynamic #[1, hidden]
  let delW ← randnDynamic #[1, hidden]
  pure { embed, proj, splitW, delW }

end ToyParams

def toyModel {maxLen vocab hidden : Nat} : BranchingModel (device := .CPU) maxLen vocab (ToyParams vocab hidden) :=
  { forward := fun {batch} params state _t => do
      let emb := Tensor.embedding (batch := batch) (seq := maxLen) (vocab := vocab) (embed := hidden) state params.embed
      let bias : Tensor { dtype := .Float32, shape := #[vocab], device := .CPU } := zerosDynamic #[vocab]
      let logits := Tensor.affine3d emb params.proj bias
      let split3 := Tensor.linear3d (x := emb) (weight := params.splitW)
      let del3 := Tensor.linear3d (x := emb) (weight := params.delW)
      let splitLogits := split3.reshape #[batch, maxLen]
      let delLogits := del3.reshape #[batch, maxLen]
      pure (logits, splitLogits, delLogits) }

def runDemo : IO Unit := do
  let vocab : Nat := 6
  let hidden : Nat := 8
  let maxLen : Nat := 16

  let tokens : Array Nat := #[1, 2, 3, 4, 2, 1]
  let groups : Array Int := Array.replicate tokens.size 0
  let x1 : BranchingState Nat := BranchingState.mkDefault tokens groups
  let x1s := #[x1]
  let times := #[0.4]
  let x0Sampler : FlowNode Nat → Nat := fun node => node.data
  let merger := fun (a _b : Nat) (_w1 _w2 : Nat) => a

  let (bridgeResult, _rng) :=
    branchingBridge discreteBridge () x0Sampler x1s times
      uniformTimeDist uniformTimeDist (sequentialUniformPolicy Nat) merger
      (maxLen := some maxLen)
      (rng := { state := 7 })

  let cfg : BranchingTrainConfig := {
    maxLen := maxLen
    padToken := 0
    anchorWeight := 1.0
    splitsWeight := 0.1
    delWeight := 0.1
    gradClip := 1.0
  }

  let params ← ToyParams.init vocab hidden
  let opt := Optim.adamw (lr := 1.0e-3)
  let optState := opt.init params
  let (_params', _optState', report) ←
    trainStep (maxLen := maxLen) (vocab := vocab) cfg (toyModel (maxLen := maxLen) (vocab := vocab) (hidden := hidden))
      params optState bridgeResult 1.0e-3
  IO.println s!"Discrete branching training step: {reprStr report}"

def _root_.main (_args : List String) : IO UInt32 := do
  runDemo
  pure 0

end Examples.BranchingFlows
