import Tyr
import Tyr.Model.BranchingFlows.TypedOps
import Tyr.Optim
import Tyr.Model.BranchingFlows

/-!
  Torch-integrated training helpers for the BranchingFlows port.

  This module focuses on discrete-token branching states and provides:
  - Packing utilities to convert `BranchingBridgeResult` into fixed-size tensors.
  - Masked losses for anchor prediction, split counts, and deletion flags.
  - A minimal training step using `Tyr.Optim`.
-/

namespace torch.branching

open torch

structure BranchingTrainConfig where
  maxLen : UInt64
  padToken : Int64 := 0
  timeScale : Float := 1.0e6
  anchorWeight : Float := 1.0
  /-- Molecule-coordinate multiplier within the anchor objective. -/
  coordWeight : Float := 1.0
  /-- Molecule-label multiplier within the anchor objective. -/
  labelWeight : Float := 1.0
  splitsWeight : Float := 1.0
  delWeight : Float := 1.0
  weightDecay : Float := 0.0
  gradClip : Float := 0.0
  deriving Repr

structure BranchingBatch (batch maxLen : UInt64) where
  t : Tensor { shape := #[batch] }
  state : Tensor { shape := #[batch, maxLen], dtype := .Int64 }
  anchor : Tensor { shape := #[batch, maxLen], dtype := .Int64 }
  padmask : Tensor { shape := #[batch, maxLen] }
  flowmask : Tensor { shape := #[batch, maxLen] }
  splitsTarget : Tensor { shape := #[batch, maxLen] }
  delTarget : Tensor { shape := #[batch, maxLen] }
  deriving Repr

structure BranchingContinuousBatch (batch maxLen dim : UInt64) where
  t : Tensor { shape := #[batch] }
  state : Tensor { shape := #[batch, maxLen, dim] }
  anchor : Tensor { shape := #[batch, maxLen, dim] }
  padmask : Tensor { shape := #[batch, maxLen] }
  flowmask : Tensor { shape := #[batch, maxLen] }
  splitsTarget : Tensor { shape := #[batch, maxLen] }
  delTarget : Tensor { shape := #[batch, maxLen] }
  deriving Repr

structure BranchingModel (maxLen vocab : UInt64) (Params : Type) where
  forward : {batch : UInt64} → Params → Tensor { shape := #[batch, maxLen], dtype := .Int64 } → Tensor { shape := #[batch] }
    → IO (Tensor { shape := #[batch, maxLen, vocab] } × Tensor { shape := #[batch, maxLen] } × Tensor { shape := #[batch, maxLen] })

structure BranchingModelContinuous (maxLen dim : UInt64) (Params : Type) where
  forward : {batch : UInt64} → Params → Tensor { shape := #[batch, maxLen, dim] } → Tensor { shape := #[batch] }
    → IO (Tensor { shape := #[batch, maxLen, dim] } × Tensor { shape := #[batch, maxLen] } × Tensor { shape := #[batch, maxLen] })

private def floatMask {batch maxLen : UInt64} (pad flow : Tensor { shape := #[batch, maxLen] }) : Tensor { shape := #[batch, maxLen] } :=
  pad * flow

private def maskedCrossEntropy {batch maxLen vocab : UInt64}
    (logits : Tensor { shape := #[batch, maxLen, vocab] })
    (targets : Tensor { shape := #[batch, maxLen], dtype := .Int64 })
    (mask : Tensor { shape := #[batch, maxLen] }) : Tensor { shape := #[] } :=
  let n : UInt64 := batch * maxLen
  let logits2 := Tensor.reshapeChecked logits #[n, vocab]
  let targets2 := Tensor.reshapeChecked targets #[n]
  let mask2 := Tensor.reshapeChecked mask #[n]
  let per := Tensor.crossEntropyNone logits2 targets2
  let masked := per * mask2
  let denom := Tensor.sumAll mask2
  Tensor.Preserving.div (Tensor.sumAll masked) (denom + (1.0e-8 : Float))

private def maskedMSE {batch maxLen : UInt64}
    (pred target mask : Tensor { shape := #[batch, maxLen] }) : Tensor { shape := #[] } :=
  let diff := pred - target
  let sq := diff * diff
  let masked := sq * mask
  let denom := Tensor.sumAll mask
  Tensor.Preserving.div (Tensor.sumAll masked) (denom + (1.0e-8 : Float))

private def maskedMSE3d {batch maxLen dim : UInt64}
    (pred target : Tensor { shape := #[batch, maxLen, dim] }) (mask : Tensor { shape := #[batch, maxLen] }) : Tensor { shape := #[] } :=
  let mask3 := Tensor.expand (Tensor.unsqueeze mask 2) #[batch, maxLen, dim]
  let diff := pred - target
  let sq := diff * diff
  let masked := sq * mask3
  let denom := Tensor.sumAll mask3
  Tensor.Preserving.div (Tensor.sumAll masked) (denom + (1.0e-8 : Float))

def maskedSplitPoissonLoss {batch maxLen : UInt64}
    (logits target mask : Tensor { shape := #[batch, maxLen] }) : Tensor { shape := #[] } :=
  let mu := Tensor.Preserving.exp (Tensor.Preserving.clampFloat logits (-100.0) 11.0)
  let safeTarget := Tensor.Preserving.clampFloat target 1.0e-8 1.0e20
  let per := mu - target * logDynamic mu - (target - target * logDynamic safeTarget)
  let masked := per * mask
  let denom := Tensor.sumAll mask
  Tensor.Preserving.div (Tensor.sumAll masked) (denom + (1.0e-8 : Float))

def maskedBCEWithLogits {batch maxLen : UInt64}
    (logits target mask : Tensor { shape := #[batch, maxLen] }) : Tensor { shape := #[] } :=
  let per := softplusDynamic logits - target * logits
  let loss := Tensor.sumAll (per * mask)
  let denom := Tensor.sumAll mask
  Tensor.Preserving.div loss (denom + (1.0e-8 : Float))

def packBranchingNat (cfg : BranchingTrainConfig) (result : BranchingBridgeResult Nat)
    : IO (Sigma fun batch => BranchingBatch batch cfg.maxLen) := do
  let batchNat := result.t.size
  if result.Xt.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.Xt size mismatch")
  if result.X1anchor.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.X1anchor size mismatch")
  if result.splitsTarget.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.splitsTarget size mismatch")
  if result.del.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.del size mismatch")
  let maxLenNat := cfg.maxLen.toNat
  let total := batchNat * maxLenNat

  let mut stateArr : Array Int64 := Array.replicate total cfg.padToken
  let mut anchorArr : Array Int64 := Array.replicate total cfg.padToken
  let mut padArr : Array Int64 := Array.replicate total 0
  let mut flowArr : Array Int64 := Array.replicate total 0
  let mut splitsArr : Array Int64 := Array.replicate total 0
  let mut delArr : Array Int64 := Array.replicate total 0

  for bi in [:batchNat] do
    let x := result.Xt[bi]!
    let anchors := result.X1anchor[bi]!
    let splits := result.splitsTarget[bi]!
    let dels := result.del[bi]!
    if anchors.size != x.state.size then
      throw (IO.userError "X1anchor length mismatch")
    if splits.size != x.state.size then
      throw (IO.userError "splitsTarget length mismatch")
    if dels.size != x.state.size then
      throw (IO.userError "del length mismatch")
    if x.state.size > maxLenNat then
      throw (IO.userError "Branching sample exceeds maxLen; increase maxLen or resample")
    for j in [:maxLenNat] do
      let idx := bi * maxLenNat + j
      if h : j < x.state.size then
        let sVal := x.state[j]'h
        let aVal := anchors.getD j 0
        let spVal := splits.getD j 0
        let dVal := dels.getD j false
        let fVal := x.flowmask.getD j false
        stateArr := stateArr.set! idx (Int64.ofNat sVal)
        anchorArr := anchorArr.set! idx (Int64.ofNat aVal)
        splitsArr := splitsArr.set! idx (Int64.ofNat spVal)
        delArr := delArr.set! idx (if dVal then 1 else 0)
        padArr := padArr.set! idx 1
        flowArr := flowArr.set! idx (if fVal then 1 else 0)
      else
        pure ()

  let batchU : UInt64 := batchNat.toUInt64
  let state := Tensor.reshapeChecked (Tensor.fromInt64Array stateArr) #[batchU, cfg.maxLen]
  let anchor := Tensor.reshapeChecked (Tensor.fromInt64Array anchorArr) #[batchU, cfg.maxLen]
  let padmask := floatDynamic (Tensor.reshapeChecked (Tensor.fromInt64Array padArr) #[batchU, cfg.maxLen])
  let flowmask := floatDynamic (Tensor.reshapeChecked (Tensor.fromInt64Array flowArr) #[batchU, cfg.maxLen])
  let splitsTarget := floatDynamic (Tensor.reshapeChecked (Tensor.fromInt64Array splitsArr) #[batchU, cfg.maxLen])
  let delTarget := floatDynamic (Tensor.reshapeChecked (Tensor.fromInt64Array delArr) #[batchU, cfg.maxLen])

  let timeScaled := result.t.map (fun t => ((t * cfg.timeScale).toUInt64).toInt64)
  let t := Tensor.reshapeChecked (Tensor.fromInt64Array timeScaled) #[batchU]
  let t := (floatDynamic t) / cfg.timeScale

  return ⟨batchU, { t, state, anchor, padmask, flowmask, splitsTarget, delTarget }⟩

private def stack1dDyn {dim : UInt64} (tensors : Array (Tensor { shape := #[dim] })) (len : UInt64)
    : Tensor { shape := #[len, dim] } :=
  if tensors.isEmpty then
    Tensor.reshapeChecked (zerosDynamic #[0, dim]) #[len, dim]
  else
    let unsq := tensors.map (fun t => Tensor.unsqueeze t 0) -- [1, dim]
    Tensor.catManyChecked (unsq.map Tensor.packShape) 0 #[len, dim]

private def stackBatch2d {batch maxLen dim : UInt64} (tensors : Array (Tensor { shape := #[maxLen, dim] })) :
    Tensor { shape := #[batch, maxLen, dim] } :=
  if tensors.isEmpty then
    Tensor.reshapeChecked (zerosDynamic #[0, maxLen, dim]) #[batch, maxLen, dim]
  else
    let unsq := tensors.map (fun t => Tensor.unsqueeze t 0) -- [1, maxLen, dim]
    Tensor.catManyChecked (unsq.map Tensor.packShape) 0 #[batch, maxLen, dim]

def packBranchingTensor {dim : UInt64} (cfg : BranchingTrainConfig)
    (result : BranchingBridgeResult (Tensor { shape := #[dim] }))
    : IO (Sigma fun batch => BranchingContinuousBatch batch cfg.maxLen dim) := do
  let batchNat := result.t.size
  if result.Xt.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.Xt size mismatch")
  if result.X1anchor.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.X1anchor size mismatch")
  if result.splitsTarget.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.splitsTarget size mismatch")
  if result.del.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.del size mismatch")

  let maxLenNat := cfg.maxLen.toNat
  let mut paddedStates : Array (Tensor { shape := #[cfg.maxLen, dim] }) := #[]
  let mut paddedAnchors : Array (Tensor { shape := #[cfg.maxLen, dim] }) := #[]
  let mut padArr : Array Int64 := Array.replicate (batchNat * maxLenNat) 0
  let mut flowArr : Array Int64 := Array.replicate (batchNat * maxLenNat) 0
  let mut splitsArr : Array Int64 := Array.replicate (batchNat * maxLenNat) 0
  let mut delArr : Array Int64 := Array.replicate (batchNat * maxLenNat) 0

  for bi in [:batchNat] do
    let x := result.Xt[bi]!
    let anchors := result.X1anchor[bi]!
    let splits := result.splitsTarget[bi]!
    let dels := result.del[bi]!
    if anchors.size != x.state.size then
      throw (IO.userError "X1anchor length mismatch")
    if splits.size != x.state.size then
      throw (IO.userError "splitsTarget length mismatch")
    if dels.size != x.state.size then
      throw (IO.userError "del length mismatch")
    if x.state.size > maxLenNat then
      throw (IO.userError "Branching sample exceeds maxLen; increase maxLen or resample")

    let lenU := x.state.size.toUInt64
    let xs := stack1dDyn (dim := dim) x.state lenU
    let xa := stack1dDyn (dim := dim) anchors lenU
    let padU := cfg.maxLen - lenU
    let padT : Tensor { shape := #[padU, dim] } := zerosDynamic #[padU, dim]
    let xsPad := Tensor.reshapeChecked (Tensor.cat xs padT 0) #[cfg.maxLen, dim]
    let xaPad := Tensor.reshapeChecked (Tensor.cat xa padT 0) #[cfg.maxLen, dim]
    paddedStates := paddedStates.push xsPad
    paddedAnchors := paddedAnchors.push xaPad

    for j in [:maxLenNat] do
      let idx := bi * maxLenNat + j
      if j < x.state.size then
        let spVal := splits.getD j 0
        let dVal := dels.getD j false
        let fVal := x.flowmask.getD j false
        padArr := padArr.set! idx 1
        flowArr := flowArr.set! idx (if fVal then 1 else 0)
        splitsArr := splitsArr.set! idx (Int64.ofNat spVal)
        delArr := delArr.set! idx (if dVal then 1 else 0)
      else
        pure ()

  let batchU : UInt64 := batchNat.toUInt64
  let state := stackBatch2d (batch := batchU) (maxLen := cfg.maxLen) (dim := dim) paddedStates
  let anchor := stackBatch2d (batch := batchU) (maxLen := cfg.maxLen) (dim := dim) paddedAnchors
  let padmask := floatDynamic (Tensor.reshapeChecked (Tensor.fromInt64Array padArr) #[batchU, cfg.maxLen])
  let flowmask := floatDynamic (Tensor.reshapeChecked (Tensor.fromInt64Array flowArr) #[batchU, cfg.maxLen])
  let splitsTarget := floatDynamic (Tensor.reshapeChecked (Tensor.fromInt64Array splitsArr) #[batchU, cfg.maxLen])
  let delTarget := floatDynamic (Tensor.reshapeChecked (Tensor.fromInt64Array delArr) #[batchU, cfg.maxLen])

  let timeScaled := result.t.map (fun t => ((t * cfg.timeScale).toUInt64).toInt64)
  let t := Tensor.reshapeChecked (Tensor.fromInt64Array timeScaled) #[batchU]
  let t := (floatDynamic t) / cfg.timeScale

  return ⟨batchU, { t, state, anchor, padmask, flowmask, splitsTarget, delTarget }⟩

structure BranchingLossReport where
  total : Float
  anchor : Float
  splits : Float
  del : Float
  deriving Repr

def trainStep {maxLen vocab : UInt64} {Params : Type} [TensorStruct Params]
    (cfg : BranchingTrainConfig)
    (model : BranchingModel maxLen vocab Params)
    (params : Params)
    (optState : Optim.AdamWState Params)
    (result : BranchingBridgeResult Nat)
    (lr : Float)
    (clipGrads : Params → Float → IO Unit := fun _ _ => pure ())
    : IO (Params × Optim.AdamWState Params × BranchingLossReport) := do
  let params := TensorStruct.makeLeafParams params
  TensorStruct.zeroGrads params
  if cfg.maxLen != maxLen then throw <| IO.userError "BranchingFlows model/config maxLen mismatch"
  let ⟨batch, packed⟩ ← packBranchingNat { cfg with maxLen := maxLen } result
  let (anchorLogits, splitLogits, delLogits) ← model.forward (batch := batch) params packed.state packed.t
  let mask := floatMask packed.padmask packed.flowmask

  let anchorLoss := maskedCrossEntropy anchorLogits packed.anchor mask
  let splitLoss := maskedSplitPoissonLoss splitLogits packed.splitsTarget mask
  let delLoss := maskedBCEWithLogits delLogits packed.delTarget mask
  let totalLoss := anchorLoss * cfg.anchorWeight + splitLoss * cfg.splitsWeight + delLoss * cfg.delWeight

  Tensor.backwardLoss totalLoss
  if cfg.gradClip > 0 then
    clipGrads params cfg.gradClip

  let grads ← TensorStruct.grads params
  let opt := Optim.adamw (lr := lr) (weight_decay := cfg.weightDecay)
  let (params', optState') := Optim.step opt params grads optState

  let report : BranchingLossReport := {
    total := Tensor.item totalLoss
    anchor := Tensor.item anchorLoss
    splits := Tensor.item splitLoss
    del := Tensor.item delLoss
  }
  return (params', optState', report)

def trainStepContinuous {maxLen dim : UInt64} {Params : Type} [TensorStruct Params]
    (cfg : BranchingTrainConfig)
    (model : BranchingModelContinuous maxLen dim Params)
    (params : Params)
    (optState : Optim.AdamWState Params)
    (result : BranchingBridgeResult (Tensor { shape := #[dim] }))
    (lr : Float)
    (clipGrads : Params → Float → IO Unit := fun _ _ => pure ())
    : IO (Params × Optim.AdamWState Params × BranchingLossReport) := do
  let params := TensorStruct.makeLeafParams params
  TensorStruct.zeroGrads params
  if cfg.maxLen != maxLen then throw <| IO.userError "BranchingFlows model/config maxLen mismatch"
  let ⟨batch, packed⟩ ← packBranchingTensor (dim := dim) { cfg with maxLen := maxLen } result
  let (anchorPred, splitLogits, delLogits) ← model.forward (batch := batch) params packed.state packed.t
  let mask := floatMask packed.padmask packed.flowmask

  let anchorLoss := maskedMSE3d anchorPred packed.anchor mask
  let splitLoss := maskedSplitPoissonLoss splitLogits packed.splitsTarget mask
  let delLoss := maskedBCEWithLogits delLogits packed.delTarget mask
  let totalLoss := anchorLoss * cfg.anchorWeight + splitLoss * cfg.splitsWeight + delLoss * cfg.delWeight

  Tensor.backwardLoss totalLoss
  if cfg.gradClip > 0 then
    clipGrads params cfg.gradClip

  let grads ← TensorStruct.grads params
  let opt := Optim.adamw (lr := lr) (weight_decay := cfg.weightDecay)
  let (params', optState') := Optim.step opt params grads optState

  let report : BranchingLossReport := {
    total := Tensor.item totalLoss
    anchor := Tensor.item anchorLoss
    splits := Tensor.item splitLoss
    del := Tensor.item delLoss
  }
  return (params', optState', report)

end torch.branching
