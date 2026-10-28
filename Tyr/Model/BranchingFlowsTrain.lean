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

variable {device : Device}
open torch

structure BranchingTrainConfig where
  maxLen : Nat
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

structure BranchingBatch (batch maxLen : Nat) where
  t : Tensor { dtype := .Float32, device := device, shape := #[batch] }
  state : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 }
  anchor : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 }
  padmask : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] }
  flowmask : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] }
  splitsTarget : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] }
  delTarget : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] }
  deriving Repr

structure BranchingContinuousBatch (batch maxLen dim : Nat) where
  t : Tensor { dtype := .Float32, device := device, shape := #[batch] }
  state : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen, dim] }
  anchor : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen, dim] }
  padmask : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] }
  flowmask : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] }
  splitsTarget : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] }
  delTarget : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] }
  deriving Repr

structure BranchingModel (maxLen vocab : Nat) (Params : Type) where
  forward : {batch : Nat} → Params → Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 } → Tensor { dtype := .Float32, device := device, shape := #[batch] }
    → IO (Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen, vocab] } × Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] } × Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] })

structure BranchingModelContinuous (maxLen dim : Nat) (Params : Type) where
  forward : {batch : Nat} → Params → Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen, dim] } → Tensor { dtype := .Float32, device := device, shape := #[batch] }
    → IO (Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen, dim] } × Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] } × Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] })

private def floatMask {batch maxLen : Nat} (pad flow : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] }) : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] } :=
  Tensor.mul pad flow

private def maskedCrossEntropy {batch maxLen vocab : Nat}
    (logits : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen, vocab] })
    (targets : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 })
    (mask : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] }) : Tensor { dtype := .Float32, device := device, shape := #[] } :=
  let n : Nat := batch * maxLen
  let logits2 := Tensor.reshape logits #[n, vocab]
  let targets2 := Tensor.reshape targets #[n]
  let mask2 := Tensor.reshape mask #[n]
  let per := Tensor.crossEntropyNoneChecked logits2 targets2
  let masked := Tensor.mul per mask2
  let denom := Tensor.sumAll mask2
  Tensor.div (Tensor.sumAll masked) (Tensor.addScalar denom 1.0e-8)

private def maskedMSE {batch maxLen : Nat}
    (pred target mask : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] }) : Tensor { dtype := .Float32, device := device, shape := #[] } :=
  let diff := Tensor.sub pred target
  let sq := Tensor.mul diff diff
  let masked := Tensor.mul sq mask
  let denom := Tensor.sumAll mask
  Tensor.div (Tensor.sumAll masked) (Tensor.addScalar denom 1.0e-8)

private def maskedMSE3d {batch maxLen dim : Nat}
    (pred target : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen, dim] }) (mask : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] }) : Tensor { dtype := .Float32, device := device, shape := #[] } :=
  let mask3 := Tensor.expand (Tensor.unsqueeze mask 2) #[batch, maxLen, dim]
  let diff := Tensor.sub pred target
  let sq := Tensor.mul diff diff
  let masked := Tensor.mul sq mask3
  let denom := Tensor.sumAll mask3
  Tensor.div (Tensor.sumAll masked) (Tensor.addScalar denom 1.0e-8)

def maskedSplitPoissonLoss {batch maxLen : Nat}
    (logits target mask : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] }) : Tensor { dtype := .Float32, device := device, shape := #[] } :=
  let mu := Tensor.exp (Tensor.clampFloat logits (-100.0) 11.0)
  let safeTarget := Tensor.clampFloat target 1.0e-8 1.0e20
  let per := Tensor.sub (Tensor.sub mu (Tensor.mul target (logDynamic mu)))
    (Tensor.sub target (Tensor.mul target (logDynamic safeTarget)))
  let masked := Tensor.mul per mask
  let denom := Tensor.sumAll mask
  Tensor.div (Tensor.sumAll masked) (Tensor.addScalar denom 1.0e-8)

def maskedBCEWithLogits {batch maxLen : Nat}
    (logits target mask : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] }) : Tensor { dtype := .Float32, device := device, shape := #[] } :=
  let per := Tensor.sub (softplusDynamic logits) (Tensor.mul target logits)
  let loss := Tensor.sumAll (Tensor.mul per mask)
  let denom := Tensor.sumAll mask
  Tensor.div loss (Tensor.addScalar denom 1.0e-8)

def packBranchingNat (cfg : BranchingTrainConfig) (result : BranchingBridgeResult Nat)
    (device : Device := .CPU)
    : IO (Sigma fun batch => BranchingBatch (device := device) batch cfg.maxLen) := do
  let batchNat := result.t.size
  if result.Xt.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.Xt size mismatch")
  if result.X1anchor.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.X1anchor size mismatch")
  if result.splitsTarget.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.splitsTarget size mismatch")
  if result.del.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.del size mismatch")
  let maxLenNat := cfg.maxLen
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

  let batchU : Nat := batchNat
  let state := Tensor.reshapeChecked ((Tensor.fromInt64Array stateArr).to device) #[batchU, cfg.maxLen]
  let anchor := Tensor.reshapeChecked ((Tensor.fromInt64Array anchorArr).to device) #[batchU, cfg.maxLen]
  let padmask := floatDynamic (Tensor.reshapeChecked ((Tensor.fromInt64Array padArr).to device) #[batchU, cfg.maxLen])
  let flowmask := floatDynamic (Tensor.reshapeChecked ((Tensor.fromInt64Array flowArr).to device) #[batchU, cfg.maxLen])
  let splitsTarget := floatDynamic (Tensor.reshapeChecked ((Tensor.fromInt64Array splitsArr).to device) #[batchU, cfg.maxLen])
  let delTarget := floatDynamic (Tensor.reshapeChecked ((Tensor.fromInt64Array delArr).to device) #[batchU, cfg.maxLen])

  let timeScaled := result.t.map (fun t => ((t * cfg.timeScale).toUInt64).toInt64)
  let t := Tensor.reshape ((Tensor.fromInt64Array timeScaled).to device) #[batchU]
  let t := Tensor.divScalar (floatDynamic t) cfg.timeScale

  return ⟨batchU, { t, state, anchor, padmask, flowmask, splitsTarget, delTarget }⟩

private def stack1dDyn {dim : Nat} (tensors : Array (Tensor { dtype := .Float32, device := device, shape := #[dim] })) (len : Nat)
    : Tensor { dtype := .Float32, device := device, shape := #[len, dim] } :=
  if tensors.isEmpty then
    Tensor.reshapeChecked (zerosDynamic (device := device) #[0, dim]) #[len, dim]
  else
    let unsq := tensors.map (fun t => Tensor.unsqueeze t 0) -- [1, dim]
    Tensor.catManyChecked (unsq.map Tensor.packShape) 0 #[len, dim]

private def stackBatch2d {batch maxLen dim : Nat} (tensors : Array (Tensor { dtype := .Float32, device := device, shape := #[maxLen, dim] })) :
    Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen, dim] } :=
  if tensors.isEmpty then
    Tensor.reshapeChecked (zerosDynamic (device := device) #[0, maxLen, dim]) #[batch, maxLen, dim]
  else
    let unsq := tensors.map (fun t => Tensor.unsqueeze t 0) -- [1, maxLen, dim]
    Tensor.catManyChecked (unsq.map Tensor.packShape) 0 #[batch, maxLen, dim]

def packBranchingTensor {dim : Nat} (cfg : BranchingTrainConfig)
    (result : BranchingBridgeResult (Tensor { dtype := .Float32, device := device, shape := #[dim] }))
    : IO (Sigma fun batch => BranchingContinuousBatch (device := device) batch cfg.maxLen dim) := do
  let batchNat := result.t.size
  if result.Xt.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.Xt size mismatch")
  if result.X1anchor.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.X1anchor size mismatch")
  if result.splitsTarget.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.splitsTarget size mismatch")
  if result.del.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.del size mismatch")

  let maxLenNat := cfg.maxLen
  let mut paddedStates : Array (Tensor { dtype := .Float32, device := device, shape := #[cfg.maxLen, dim] }) := #[]
  let mut paddedAnchors : Array (Tensor { dtype := .Float32, device := device, shape := #[cfg.maxLen, dim] }) := #[]
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

    let lenU := x.state.size
    let xs := stack1dDyn (dim := dim) x.state lenU
    let xa := stack1dDyn (dim := dim) anchors lenU
    let padU := cfg.maxLen - lenU
    let padT : Tensor { dtype := .Float32, device := device, shape := #[padU, dim] } := zerosDynamic (device := device) #[padU, dim]
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

  let batchU : Nat := batchNat
  let state := stackBatch2d (batch := batchU) (maxLen := cfg.maxLen) (dim := dim) paddedStates
  let anchor := stackBatch2d (batch := batchU) (maxLen := cfg.maxLen) (dim := dim) paddedAnchors
  let padmask := floatDynamic (Tensor.reshapeChecked ((Tensor.fromInt64Array padArr).to device) #[batchU, cfg.maxLen])
  let flowmask := floatDynamic (Tensor.reshapeChecked ((Tensor.fromInt64Array flowArr).to device) #[batchU, cfg.maxLen])
  let splitsTarget := floatDynamic (Tensor.reshapeChecked ((Tensor.fromInt64Array splitsArr).to device) #[batchU, cfg.maxLen])
  let delTarget := floatDynamic (Tensor.reshapeChecked ((Tensor.fromInt64Array delArr).to device) #[batchU, cfg.maxLen])

  let timeScaled := result.t.map (fun t => ((t * cfg.timeScale).toUInt64).toInt64)
  let t := Tensor.reshape ((Tensor.fromInt64Array timeScaled).to device) #[batchU]
  let t := Tensor.divScalar (floatDynamic t) cfg.timeScale

  return ⟨batchU, { t, state, anchor, padmask, flowmask, splitsTarget, delTarget }⟩

structure BranchingLossReport where
  total : Float
  anchor : Float
  splits : Float
  del : Float
  deriving Repr

def trainStep {maxLen vocab : Nat} {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params]
    (cfg : BranchingTrainConfig)
    (model : BranchingModel (device := device) maxLen vocab Params)
    (params : Params)
    (optState : Optim.AdamWState Params)
    (result : BranchingBridgeResult Nat)
    (lr : Float)
    (clipGrads : Params → Float → IO Unit := fun _ _ => pure ())
    : IO (Params × Optim.AdamWState Params × BranchingLossReport) := do
  let params ← TensorStruct.makeLeafParams params
  TensorStruct.zeroGrads params
  if cfg.maxLen != maxLen then throw <| IO.userError "BranchingFlows model/config maxLen mismatch"
  let ⟨batch, packed⟩ ← packBranchingNat { cfg with maxLen := maxLen } result (device := device)
  let (anchorLogits, splitLogits, delLogits) ← model.forward (batch := batch) params packed.state packed.t
  let mask := floatMask packed.padmask packed.flowmask

  let anchorLoss := maskedCrossEntropy anchorLogits packed.anchor mask
  let splitLoss := maskedSplitPoissonLoss splitLogits packed.splitsTarget mask
  let delLoss := maskedBCEWithLogits delLogits packed.delTarget mask
  let totalLoss := Tensor.add
    (Tensor.add (Tensor.mulScalar anchorLoss cfg.anchorWeight)
      (Tensor.mulScalar splitLoss cfg.splitsWeight))
    (Tensor.mulScalar delLoss cfg.delWeight)

  Tensor.backwardLoss totalLoss
  if cfg.gradClip > 0 then
    clipGrads params cfg.gradClip

  let grads ← TensorStruct.grads params
  let opt := Optim.adamw (lr := lr) (weight_decay := cfg.weightDecay)
  let (params', optState') ← Optim.step opt params grads optState

  let report : BranchingLossReport := {
    total := Tensor.item totalLoss
    anchor := Tensor.item anchorLoss
    splits := Tensor.item splitLoss
    del := Tensor.item delLoss
  }
  return (params', optState', report)

def trainStepContinuous {maxLen dim : Nat} {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params]
    (cfg : BranchingTrainConfig)
    (model : BranchingModelContinuous (device := device) maxLen dim Params)
    (params : Params)
    (optState : Optim.AdamWState Params)
    (result : BranchingBridgeResult (Tensor { dtype := .Float32, device := device, shape := #[dim] }))
    (lr : Float)
    (clipGrads : Params → Float → IO Unit := fun _ _ => pure ())
    : IO (Params × Optim.AdamWState Params × BranchingLossReport) := do
  let params ← TensorStruct.makeLeafParams params
  TensorStruct.zeroGrads params
  if cfg.maxLen != maxLen then throw <| IO.userError "BranchingFlows model/config maxLen mismatch"
  let ⟨batch, packed⟩ ← packBranchingTensor (dim := dim) { cfg with maxLen := maxLen } result
  let (anchorPred, splitLogits, delLogits) ← model.forward (batch := batch) params packed.state packed.t
  let mask := floatMask packed.padmask packed.flowmask

  let anchorLoss := maskedMSE3d anchorPred packed.anchor mask
  let splitLoss := maskedSplitPoissonLoss splitLogits packed.splitsTarget mask
  let delLoss := maskedBCEWithLogits delLogits packed.delTarget mask
  let totalLoss := Tensor.add
    (Tensor.add (Tensor.mulScalar anchorLoss cfg.anchorWeight)
      (Tensor.mulScalar splitLoss cfg.splitsWeight))
    (Tensor.mulScalar delLoss cfg.delWeight)

  Tensor.backwardLoss totalLoss
  if cfg.gradClip > 0 then
    clipGrads params cfg.gradClip

  let grads ← TensorStruct.grads params
  let opt := Optim.adamw (lr := lr) (weight_decay := cfg.weightDecay)
  let (params', optState') ← Optim.step opt params grads optState

  let report : BranchingLossReport := {
    total := Tensor.item totalLoss
    anchor := Tensor.item anchorLoss
    splits := Tensor.item splitLoss
    del := Tensor.item delLoss
  }
  return (params', optState', report)

end torch.branching
