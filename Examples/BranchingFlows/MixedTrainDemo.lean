import Tyr
import Tyr.Optim
import Tyr.Model.BranchingFlows
import Tyr.Model.BranchingFlowsTrain

/-!
  Examples/BranchingFlows/MixedTrainDemo.lean

  Mixed continuous + discrete branching demo (full Julia-style loop).
  Generates synthetic data, trains a tiny model, and writes .pt tensors for plotting.
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

private def linBridge2d (_ : Unit) (x0 x1 : Tensor { shape := #[2], device := .CPU }) (t0 t1 : Float) : Tensor { shape := #[2], device := .CPU } :=
  let dt := t1 - t0
  Tensor.Promoting.add x0 (Tensor.Promoting.mulScalar (Tensor.Promoting.sub x1 x0) dt)

private def discreteBridge (_ : Unit) (_x0 x1 : Nat) (_t0 _t1 : Float) : Nat :=
  x1

structure MixParams (vocab hidden : UInt64) where
  contW : Tensor { shape := #[hidden, 2], device := .CPU }
  contB : Tensor { shape := #[hidden], device := .CPU }
  timeW : Tensor { shape := #[hidden, 1], device := .CPU }
  timeB : Tensor { shape := #[hidden], device := .CPU }
  embed : Tensor { shape := #[vocab, hidden], device := .CPU }
  locW : Tensor { shape := #[2, hidden], device := .CPU }
  locB : Tensor { shape := #[2], device := .CPU }
  proj : Tensor { shape := #[vocab, hidden], device := .CPU }
  tokB : Tensor { shape := #[vocab], device := .CPU }
  splitW : Tensor { shape := #[1, hidden], device := .CPU }
  splitB : Tensor { shape := #[1], device := .CPU }
  delW : Tensor { shape := #[1, hidden], device := .CPU }
  delB : Tensor { shape := #[1], device := .CPU }
  deriving TensorStruct

namespace MixParams

def init (vocab hidden : UInt64) : IO (MixParams vocab hidden) := do
  let contW ← randnDynamic #[hidden, 2]
  let contB := zerosDynamic #[hidden]
  let timeW ← randnDynamic #[hidden, 1]
  let timeB := zerosDynamic #[hidden]
  let embed ← randnDynamic #[vocab, hidden]
  let locW ← randnDynamic #[2, hidden]
  let locB := zerosDynamic #[2]
  let proj ← randnDynamic #[vocab, hidden]
  let tokB := zerosDynamic #[vocab]
  let splitW ← randnDynamic #[1, hidden]
  let splitB := zerosDynamic #[1]
  let delW ← randnDynamic #[1, hidden]
  let delB := zerosDynamic #[1]
  pure { contW, contB, timeW, timeB, embed, locW, locB, proj, tokB, splitW, splitB, delW, delB }

end MixParams

private def randLen (rng : Rng) : Nat × Rng := Id.run do
  let (u, rng') := randFloat rng
  if u < 0.5 then
    let (k, rng'') := randPoisson rng' 4.0
    (k + 1, rng'')
  else
    let (k, rng'') := randPoisson rng' 20.0
    (k + 1, rng'')

private def randBounds (rng : Rng) : (Float × Float) × Rng := Id.run do
  let (u1, rng') := randFloat rng
  let (u2, rng'') := randFloat rng'
  let left := -u1
  let right := u2
  ((left, right), rng'')

private def f (x : Float) : Float :=
  x + Float.sin (3.0 * x)

private def mkPoint (x y : Float) : Tensor { shape := #[2], device := .CPU } :=
  Tensor.reshapeChecked (floatsDynamic #[x, y]) #[2]

private def X1target (rng : Rng) : BranchingState (Tensor { shape := #[2], device := .CPU } × Nat) × Rng := Id.run do
  let (n, rng') := randLen rng
  let ((left, right), rng'') := randBounds rng'
  let step := (right - left) / (n.toFloat)
  let mut points : Array (Tensor { shape := #[2], device := .CPU }) := #[]
  for i in [:n] do
    let x := left + step * i.toFloat
    let y := f x
    points := points.push (mkPoint x y)
  let mut toks : Array Nat := Array.replicate n 1
  let (u, rng''') := randFloat rng''
  let odd : Nat := if u < 0.5 then 0 else 1
  for i in [:n] do
    if i % 2 == odd then
      toks := toks.set! i 2
  let state := Array.zipWith (fun p t => (p, t)) points toks
  let groups := Array.replicate n 0
  ({ state := state, groupings := groups, del := Array.replicate n false,
     ids := (Array.range n).map (fun i => Int.ofNat (i + 1)),
     branchmask := Array.replicate n true, flowmask := Array.replicate n true, padmask := Array.replicate n true }, rng''')

private def mergeTuple (a b : (Tensor { shape := #[2], device := .CPU } × Nat)) (w1 w2 : Nat) : (Tensor { shape := #[2], device := .CPU } × Nat) :=
  let cont := canonicalAnchorMerge a.1 b.1 w1 w2
  (cont, 0)

private def splitResult
    (res : BranchingBridgeResult (Tensor { shape := #[2], device := .CPU } × Nat))
    : BranchingBridgeResult (Tensor { shape := #[2], device := .CPU }) × BranchingBridgeResult Nat := Id.run do
  let mapSegCont := fun (s : Segment (Tensor { shape := #[2], device := .CPU } × Nat)) =>
    { Xt := s.Xt.1, t := s.t, anchor := s.anchor.1, descendants := s.descendants, del := s.del,
      branchable := s.branchable, flowable := s.flowable, group := s.group, lastCoalescence := s.lastCoalescence, id := s.id }
  let mapSegDisc := fun (s : Segment (Tensor { shape := #[2], device := .CPU } × Nat)) =>
    { Xt := s.Xt.2, t := s.t, anchor := s.anchor.2, descendants := s.descendants, del := s.del,
      branchable := s.branchable, flowable := s.flowable, group := s.group, lastCoalescence := s.lastCoalescence, id := s.id }
  let segsCont := res.segments.map (fun arr => arr.map mapSegCont)
  let segsDisc := res.segments.map (fun arr => arr.map mapSegDisc)
  let XtCont := res.Xt.map (fun x =>
    { state := x.state.map (fun v => v.1), groupings := x.groupings, del := x.del, ids := x.ids,
      branchmask := x.branchmask, flowmask := x.flowmask, padmask := x.padmask })
  let XtDisc := res.Xt.map (fun x =>
    { state := x.state.map (fun v => v.2), groupings := x.groupings, del := x.del, ids := x.ids,
      branchmask := x.branchmask, flowmask := x.flowmask, padmask := x.padmask })
  let anchorCont := res.X1anchor.map (fun arr => arr.map (fun v => v.1))
  let anchorDisc := res.X1anchor.map (fun arr => arr.map (fun v => v.2))
  let resCont : BranchingBridgeResult (Tensor { shape := #[2], device := .CPU }) :=
    { t := res.t, segments := segsCont, Xt := XtCont, X1anchor := anchorCont,
      descendants := res.descendants, del := res.del, splitsTarget := res.splitsTarget, prevCoalescence := res.prevCoalescence }
  let resDisc : BranchingBridgeResult Nat :=
    { t := res.t, segments := segsDisc, Xt := XtDisc, X1anchor := anchorDisc,
      descendants := res.descendants, del := res.del, splitsTarget := res.splitsTarget, prevCoalescence := res.prevCoalescence }
  (resCont, resDisc)

private def maskedCrossEntropy {batch maxLen vocab : UInt64}
    (logits : Tensor { shape := #[batch, maxLen, vocab], device := .CPU })
    (targets : Tensor { shape := #[batch, maxLen], dtype := .Int64, device := .CPU })
    (mask : Tensor { shape := #[batch, maxLen], device := .CPU }) : Tensor { shape := #[], device := .CPU } :=
  let n : UInt64 := batch * maxLen
  let logits2 := Tensor.reshapeChecked logits #[n, vocab]
  let targets2 := Tensor.reshapeChecked targets #[n]
  let mask2 := Tensor.reshapeChecked mask #[n]
  let per := Tensor.crossEntropyNone logits2 targets2
  let masked := Tensor.Promoting.mul per mask2
  let denom := Tensor.sumAll mask2
  Tensor.Promoting.div (Tensor.sumAll masked) (Tensor.Promoting.addScalar denom 1.0e-8)

private def maskedMSE {batch maxLen : UInt64}
    (pred target mask : Tensor { shape := #[batch, maxLen], device := .CPU }) : Tensor { shape := #[], device := .CPU } :=
  let diff := Tensor.Promoting.sub pred target
  let sq := Tensor.Promoting.mul diff diff
  let masked := Tensor.Promoting.mul sq mask
  let denom := Tensor.sumAll mask
  Tensor.Promoting.div (Tensor.sumAll masked) (Tensor.Promoting.addScalar denom 1.0e-8)

private def maskedMSE3d {batch maxLen dim : UInt64}
    (pred target : Tensor { shape := #[batch, maxLen, dim], device := .CPU }) (mask : Tensor { shape := #[batch, maxLen], device := .CPU }) : Tensor { shape := #[], device := .CPU } :=
  let mask3 := Tensor.expand (Tensor.unsqueeze mask 2) #[batch, maxLen, dim]
  let diff := Tensor.Promoting.sub pred target
  let sq := Tensor.Promoting.mul diff diff
  let masked := Tensor.Promoting.mul sq mask3
  let denom := Tensor.sumAll mask3
  Tensor.Promoting.div (Tensor.sumAll masked) (Tensor.Promoting.addScalar denom 1.0e-8)

private def maskedBCEWithLogits {batch maxLen : UInt64}
    (logits target mask : Tensor { shape := #[batch, maxLen], device := .CPU }) : Tensor { shape := #[], device := .CPU } :=
  let probs := Tensor.Preserving.sigmoid logits
  let loss := binaryCrossEntropySum probs target mask
  let denom := Tensor.sumAll mask
  Tensor.Promoting.div loss (Tensor.Promoting.addScalar denom 1.0e-8)

def runDemo : IO Unit := do
  let vocab : UInt64 := 3
  let hidden : UInt64 := 32
  let maxLen : UInt64 := 64
  let steps : Nat := 200
  let batch : Nat := 16

  let cfg : BranchingTrainConfig := {
    maxLen := maxLen
    padToken := 0
    anchorWeight := 1.0
    splitsWeight := 0.1
    delWeight := 0.1
    gradClip := 1.0
  }

  let params ← MixParams.init vocab hidden
  let params ← TensorStruct.makeLeafParams params
  let opt := Optim.adamw (lr := 1.0e-3)
  let mut optState := opt.init params
  let mut params := params
  let mut rng : Rng := { state := 0 }

  for i in [:steps] do
    let mut batchStates : Array (BranchingState (Tensor { shape := #[2], device := .CPU } × Nat)) := #[]
    for _ in [:batch] do
      let (x1, rng') := X1target rng
      rng := rng'
      batchStates := batchStates.push x1
    let mut times : Array Float := #[]
    for _ in [:batch] do
      let (u, rng') := randFloat rng
      rng := rng'
      times := times.push u

    let (bridgeRes, rng') :=
      branchingBridge
        (fun _ x0 x1 t0 t1 => (linBridge2d () x0.1 x1.1 t0 t1, discreteBridge () x0.2 x1.2 t0 t1))
        ()
        (fun root => (root.data.1, 0))
        batchStates times
        uniformTimeDist uniformTimeDist
        (sequentialUniformPolicy (Tensor { shape := #[2], device := .CPU } × Nat))
        mergeTuple
        (useBranchingTimeProb := 0.5)
        (deletionPad := 1.5)
        (maxLen := some maxLen.toNat)
        (rng := rng)
    rng := rng'

    let (resCont, resDisc) := splitResult bridgeRes
    let ⟨batchU, packedCont⟩ ← packBranchingTensor (dim := 2) cfg resCont
    let ⟨batchD, packedDisc0⟩ ← packBranchingNat cfg resDisc
    let packedDisc : BranchingBatch (device := .CPU) batchU cfg.maxLen ←
      if h : batchD = batchU then pure (h ▸ packedDisc0)
      else throw <| IO.userError "Mixed bridge batch sizes differ"

    let paramsTrain ← TensorStruct.makeLeafParams params

    TensorStruct.zeroGrads paramsTrain
    let t := packedDisc.t
    let contEmb := Tensor.affine3d packedCont.state paramsTrain.contW paramsTrain.contB
    let discEmb := Tensor.embedding (batch := batchU) (seq := maxLen) (vocab := vocab) (embed := hidden) packedDisc.state paramsTrain.embed
    let t1 := Tensor.unsqueeze t 1
    let t2 := Tensor.unsqueeze t1 1
    let tExp := Tensor.expand t2 #[batchU, maxLen, 1]
    let timeEmb := Tensor.affine3d tExp paramsTrain.timeW paramsTrain.timeB
    let hiddenState := Tensor.Preserving.tanh (Tensor.Promoting.add (Tensor.Promoting.add contEmb discEmb) timeEmb)

    let predLoc := Tensor.affine3d hiddenState paramsTrain.locW paramsTrain.locB
    let predTok := Tensor.affine3d hiddenState paramsTrain.proj paramsTrain.tokB
    let split3 := Tensor.affine3d hiddenState paramsTrain.splitW paramsTrain.splitB
    let del3 := Tensor.affine3d hiddenState paramsTrain.delW paramsTrain.delB
    let splitLogits := split3.reshapeChecked #[batchU, maxLen]
    let delLogits := del3.reshapeChecked #[batchU, maxLen]

    let mask := Tensor.Promoting.mul packedDisc.padmask packedDisc.flowmask
    let contLoss := maskedMSE3d predLoc packedCont.anchor mask
    let discLoss := maskedCrossEntropy predTok packedDisc.anchor mask
    let splitLoss := maskedMSE splitLogits packedDisc.splitsTarget mask
    let delLoss := maskedBCEWithLogits delLogits packedDisc.delTarget mask
    let total := Tensor.Promoting.add (Tensor.Promoting.add (Tensor.Promoting.add contLoss discLoss) splitLoss) delLoss

    torch.Tensor.backwardLoss total
    if cfg.gradClip > 0 then
      pure ()

    let grads ← TensorStruct.grads paramsTrain
    let (params', optState') ← Optim.step opt paramsTrain grads optState
    params := params'
    optState := optState'

    if i % 20 == 0 then
      let tl := torch.Tensor.item total
      IO.println s!"step {i}: loss={tl}"

  -- Write plotting tensors for one batch
  let (bridgeRes, _rng') :=
    branchingBridge
      (fun _ x0 x1 t0 t1 => (linBridge2d () x0.1 x1.1 t0 t1, discreteBridge () x0.2 x1.2 t0 t1))
      ()
      (fun root => (root.data.1, 0))
      #[(X1target rng).1] #[0.5]
      uniformTimeDist uniformTimeDist
      (sequentialUniformPolicy (Tensor { shape := #[2], device := .CPU } × Nat))
      mergeTuple
      (maxLen := some maxLen.toNat)
      (rng := rng)
  let (resCont, resDisc) := splitResult bridgeRes
  let ⟨batchU, packedCont⟩ ← packBranchingTensor (dim := 2) cfg resCont
  let ⟨batchD, packedDisc0⟩ ← packBranchingNat cfg resDisc
  let packedDisc : BranchingBatch (device := .CPU) batchU cfg.maxLen ←
    if h : batchD = batchU then pure (h ▸ packedDisc0)
    else throw <| IO.userError "Mixed bridge batch sizes differ"
  let contEmb := Tensor.affine3d packedCont.state params.contW params.contB
  let discEmb := Tensor.embedding (batch := batchU) (seq := maxLen) (vocab := vocab) (embed := hidden) packedDisc.state params.embed
  let t1 := Tensor.unsqueeze packedDisc.t 1
  let t2 := Tensor.unsqueeze t1 1
  let tExp := Tensor.expand t2 #[batchU, maxLen, 1]
  let timeEmb := Tensor.affine3d tExp params.timeW params.timeB
  let hiddenState := Tensor.Preserving.tanh (Tensor.Promoting.add (Tensor.Promoting.add contEmb discEmb) timeEmb)
  let predLoc := Tensor.affine3d hiddenState params.locW params.locB

  Tensor.saveTensor predLoc "examples_branching_mixed_pred.pt"
  Tensor.saveTensor packedCont.anchor "examples_branching_mixed_anchor.pt"
  Tensor.saveTensor packedDisc.anchor "examples_branching_mixed_tokens.pt"
  let plotMask := Tensor.Promoting.mul packedDisc.padmask packedDisc.flowmask
  Tensor.saveTensor plotMask "examples_branching_mixed_mask.pt"
  IO.println "Wrote examples_branching_mixed_pred.pt, examples_branching_mixed_anchor.pt, examples_branching_mixed_tokens.pt, examples_branching_mixed_mask.pt"

def _root_.main (_args : List String) : IO UInt32 := do
  runDemo
  pure 0

end Examples.BranchingFlows
