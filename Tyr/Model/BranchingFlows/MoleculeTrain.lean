import Tyr.Model.BranchingFlows.Molecule
import Tyr.Model.BranchingFlowsTrain
import Tyr.Optim
import Tyr.Optim.PolarExpress
import Tyr.Inference.OwnedKV

/-!
  Tensor packing helpers for molecule-shaped BranchingFlows batches.

  The generic training module has separate packers for token states and
  continuous coordinate states.  Molecule generation needs both in one element:
  3D coordinates plus a discrete atom label.
-/

namespace torch.branching

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]
open torch

structure BranchingMoleculeBatch (batch maxLen : Nat) (dtype : DType := .Float32) where
  t : Tensor { dtype := dtype, device := device, shape := #[batch] }
  coord : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen, 3] }
  coordAnchor : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen, 3] }
  label : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 }
  labelAnchor : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 }
  labelLossScale : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen] }
  padmask : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 }
  flowmask : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 }
  splitsTarget : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen] }
  delTarget : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen] }
  deriving Repr, TensorStruct

namespace BranchingMoleculeBatch

def toDevice {batch maxLen : Nat}
    (targetDevice : Device) (x : BranchingMoleculeBatch (dtype := dtype) (device := device) batch maxLen) :
    BranchingMoleculeBatch (dtype := dtype) (device := targetDevice) batch maxLen :=
  { t := x.t.to targetDevice
    coord := x.coord.to targetDevice
    coordAnchor := x.coordAnchor.to targetDevice
    label := x.label.to targetDevice
    labelAnchor := x.labelAnchor.to targetDevice
    labelLossScale := x.labelLossScale.to targetDevice
    padmask := x.padmask.to targetDevice
    flowmask := x.flowmask.to targetDevice
    splitsTarget := x.splitsTarget.to targetDevice
    delTarget := x.delTarget.to targetDevice }

/-- Explicit conversion of floating batch fields; labels and masks remain Int64. -/
def castDType {batch maxLen : Nat} (target : DType)
    (x : BranchingMoleculeBatch (device := device) batch maxLen dtype) :
    IO (BranchingMoleculeBatch (device := device) batch maxLen target) :=
  TensorStruct.castTo _ x

def castBFloat16 {batch maxLen : Nat}
    (x : BranchingMoleculeBatch (device := device) batch maxLen dtype) :
    IO (BranchingMoleculeBatch (device := device) batch maxLen .BFloat16) :=
  x.castDType .BFloat16

/-- Allocate independent buffers at the requested floating dtype, with Int64 labels and masks. -/
def zeros (batch maxLen : Nat) (device : Device) (dtype : DType := .Float32) :
    IO (BranchingMoleculeBatch (dtype := dtype) (device := device) batch maxLen) := do
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  let f (s : Shape) : IO (Tensor { dtype := dtype, device := device, shape := s }) := data.cloneInferenceIO (zerosDynamic s false device (dtype := dtype))
  let i (s : Shape) : IO (Tensor { device := device, shape := s, dtype := .Int64 }) := data.cloneInferenceIO ((Tensor.fullInt s 0).to device)
  let b : BranchingMoleculeBatch (dtype := dtype) (device := device) batch maxLen := {
    t := ← f #[batch]
    coord := ← f #[batch, maxLen, 3]
    coordAnchor := ← f #[batch, maxLen, 3]
    label := ← i #[batch, maxLen]
    labelAnchor := ← i #[batch, maxLen]
    labelLossScale := ← f #[batch, maxLen]
    padmask := ← i #[batch, maxLen]
    flowmask := ← i #[batch, maxLen]
    splitsTarget := ← f #[batch, maxLen]
    delTarget := ← f #[batch, maxLen]
  }
  pure b

/-- Return a buffer batch containing the freshly packed batch. Retained
destination aliases preserve their values. The batch
    and maxLen indices may differ at the type level (the packed batch carries
    existentials / structure projections); runtime shapes must match. Float
    fields are explicitly cast to the buffer dtype for BF16 staging; integral
    fields must already match the destination dtype. -/
def copyInto {source : DType} {b1 b2 m1 m2 : Nat}
    (dst : BranchingMoleculeBatch (dtype := dtype) (device := device) b1 m1)
    (src : BranchingMoleculeBatch (dtype := source) (device := device) b2 m2) :
    IO (BranchingMoleculeBatch (dtype := dtype) (device := device) b1 m1) := do
  return {
    t := ← copyIntoChecked dst.t (Tensor.castLike dst.t src.t)
    coord := ← copyIntoChecked dst.coord (Tensor.castLike dst.coord src.coord)
    coordAnchor := ← copyIntoChecked dst.coordAnchor (Tensor.castLike dst.coordAnchor src.coordAnchor)
    label := ← copyIntoChecked dst.label src.label
    labelAnchor := ← copyIntoChecked dst.labelAnchor src.labelAnchor
    labelLossScale := ← copyIntoChecked dst.labelLossScale (Tensor.castLike dst.labelLossScale src.labelLossScale)
    padmask := ← copyIntoChecked dst.padmask src.padmask
    flowmask := ← copyIntoChecked dst.flowmask src.flowmask
    splitsTarget := ← copyIntoChecked dst.splitsTarget (Tensor.castLike dst.splitsTarget src.splitsTarget)
    delTarget := ← copyIntoChecked dst.delTarget (Tensor.castLike dst.delTarget src.delTarget) }

/-- Force every constructed field eagerly (see `touchTensor`). -/
def eval {batch maxLen : Nat}
    (b : BranchingMoleculeBatch (dtype := dtype) (device := device) batch maxLen) : IO (BranchingMoleculeBatch (dtype := dtype) (device := device) batch maxLen) := do
  touchTensor b.t
  touchTensor b.coord
  touchTensor b.coordAnchor
  touchTensor b.label
  touchTensor b.labelAnchor
  touchTensor b.labelLossScale
  touchTensor b.padmask
  touchTensor b.flowmask
  touchTensor b.splitsTarget
  touchTensor b.delTarget
  pure b

end BranchingMoleculeBatch

structure BranchingMoleculeModel (maxLen vocab : Nat) (Params : Type) (dtype : DType := .Float32) where
  forward : {batch : Nat} → Params → Tensor { dtype := dtype, device := device, shape := #[batch, maxLen, 3] } → Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 } → Tensor { dtype := dtype, device := device, shape := #[batch] }
    → Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 }
    → IO (Tensor { dtype := dtype, device := device, shape := #[batch, maxLen, 3] } × Tensor { dtype := dtype, device := device, shape := #[batch, maxLen, vocab] } × Tensor { dtype := dtype, device := device, shape := #[batch, maxLen] } × Tensor { dtype := dtype, device := device, shape := #[batch, maxLen] })

private def coordOffset (maxLenNat bi j k : Nat) : Nat :=
  (bi * maxLenNat + j) * 3 + k

private def writeVec3 (arr : Array Float) (offset : Nat) (v : Vec3) : Array Float :=
  let arr := arr.set! offset v.x
  let arr := arr.set! (offset + 1) v.y
  arr.set! (offset + 2) v.z

def packBranchingMolecule (cfg : BranchingTrainConfig)
    (result : BranchingBridgeResult MoleculeAtom)
    (labelDFM : Option DistNoisyDiscreteConfig := none)
    : IO (Sigma fun batch => BranchingMoleculeBatch (dtype := .Float32) (device := .CPU) batch cfg.maxLen) := do
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
  let coordTotal := total * 3

  let mut coordArr : Array Float := Array.replicate coordTotal 0.0
  let mut coordAnchorArr : Array Float := Array.replicate coordTotal 0.0
  let mut labelArr : Array Int64 := Array.replicate total cfg.padToken
  let mut labelAnchorArr : Array Int64 := Array.replicate total cfg.padToken
  let mut labelScaleArr : Array Float := Array.replicate total 0.0
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
        let atom := x.state[j]'h
        let anchor := anchors.getD j default
        let coordIdx := coordOffset maxLenNat bi j 0
        coordArr := writeVec3 coordArr coordIdx atom.coord
        coordAnchorArr := writeVec3 coordAnchorArr coordIdx anchor.coord
        labelArr := labelArr.set! idx (Int64.ofNat atom.label)
        labelAnchorArr := labelAnchorArr.set! idx (Int64.ofNat anchor.label)
        labelScaleArr := labelScaleArr.set! idx
          (match labelDFM with
           | some dfm => dfm.lossScale (result.t.getD bi 0.0) 1.0 0.2
           | none => 1.0)
        splitsArr := splitsArr.set! idx (Int64.ofNat (splits.getD j 0))
        delArr := delArr.set! idx (if dels.getD j false then 1 else 0)
        padArr := padArr.set! idx 1
        flowArr := flowArr.set! idx (if x.flowmask.getD j false then 1 else 0)
      else
        pure ()

  let batchU : Nat := batchNat
  let coord := (← IO.ofExcept (Tensor.fromFloatArrayAs? coordArr #[batchU, cfg.maxLen, 3]))
  let coordAnchor := (← IO.ofExcept (Tensor.fromFloatArrayAs? coordAnchorArr #[batchU, cfg.maxLen, 3]))
  let label := (← IO.ofExcept (Tensor.fromInt64ArrayAs? labelArr #[batchU, cfg.maxLen]))
  let labelAnchor := (← IO.ofExcept (Tensor.fromInt64ArrayAs? labelAnchorArr #[batchU, cfg.maxLen]))
  let labelLossScale := (← IO.ofExcept (Tensor.fromFloatArrayAs? labelScaleArr #[batchU, cfg.maxLen]))
  let padmask := (← IO.ofExcept (Tensor.fromInt64ArrayAs? padArr #[batchU, cfg.maxLen]))
  let flowmask := (← IO.ofExcept (Tensor.fromInt64ArrayAs? flowArr #[batchU, cfg.maxLen]))
  let splitsTarget := floatDynamic ((← IO.ofExcept (Tensor.fromInt64ArrayAs? splitsArr #[batchU, cfg.maxLen])))
  let delTarget := floatDynamic ((← IO.ofExcept (Tensor.fromInt64ArrayAs? delArr #[batchU, cfg.maxLen])))

  let timeScaled := result.t.map (fun t => ((t * cfg.timeScale).toUInt64).toInt64)
  let t := Tensor.reshape (Tensor.fromInt64Array timeScaled) #[batchU]
  let t := Tensor.divScalar (floatDynamic t) cfg.timeScale

  return ⟨batchU, {
    t,
    coord,
    coordAnchor,
    label,
    labelAnchor,
    labelLossScale,
    padmask,
    flowmask,
    splitsTarget,
    delTarget
  }⟩

def packBranchingMoleculeWithDFM (cfg : BranchingTrainConfig)
    (labelDFM : DistNoisyDiscreteConfig)
    (result : BranchingBridgeResult MoleculeAtom)
    : IO (Sigma fun batch => BranchingMoleculeBatch (dtype := .Float32) (device := .CPU) batch cfg.maxLen) :=
  packBranchingMolecule cfg result (labelDFM := some labelDFM)

private def moleculeMask {batch maxLen : Nat}
    (pad flow : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 }) : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] } :=
  floatDynamic (pad * flow)

-- F32 masks intentionally promote BF16/F16 losses to F32 without converting operands.
variable [DType.Promotes dtype .Float32 .Float32] [DType.Promotes .Float32 dtype .Float32]

private def moleculeMaskedWeightedCrossEntropy {batch maxLen vocab : Nat}
    (logits : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen, vocab] })
    (targets : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 })
    (mask : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] })
    (weights : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen] }) : Tensor { dtype := .Float32, device := device, shape := #[] } :=
  let n : Nat := batch * maxLen
  let logits2 := Tensor.reshape logits #[n, vocab]
  let targets2 := Tensor.reshape targets #[n]
  let mask2 := Tensor.reshape mask #[n]
  let weights2 := Tensor.reshape weights #[n]
  let per := Tensor.crossEntropyNoneChecked logits2 targets2
  let masked := Tensor.mul (Tensor.mul per mask2) weights2
  let denom := Tensor.sumAll mask2
  Tensor.div (Tensor.sumAll masked) (Tensor.addScalar denom 1.0e-8)

private def moleculeFlowLossScale {batch maxLen : Nat}
    (t : Tensor { dtype := dtype, device := device, shape := #[batch] }) : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen] } :=
  let remaining : Tensor { dtype := dtype, device := device, shape := #[batch] } := Tensor.addScalar (Tensor.mulScalar t (-1.0)) 1.2
  let scale : Tensor { dtype := dtype, device := device, shape := #[batch] } := powDynamic remaining (-1.0)
  Tensor.expand (Tensor.unsqueeze scale 1) #[batch, maxLen]

private def moleculeMaskedWeightedMSE3d {batch maxLen dim : Nat}
    (pred target : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen, dim] })
    (mask : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] })
    (weights : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen] }) : Tensor { dtype := .Float32, device := device, shape := #[] } :=
  let mask3 := Tensor.expand (Tensor.unsqueeze mask 2) #[batch, maxLen, dim]
  let weights3 := Tensor.expand (Tensor.unsqueeze weights 2) #[batch, maxLen, dim]
  let diff := Tensor.sub pred target
  let sq := Tensor.mul diff diff
  let masked := Tensor.mul (Tensor.mul sq mask3) weights3
  let denom := Tensor.sumAll mask3
  Tensor.div (Tensor.sumAll masked) (Tensor.addScalar denom 1.0e-8)

private def moleculeMaskedWeightedSplitPoissonLoss {batch maxLen : Nat}
    (logits target : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen] })
    (mask : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] })
    (weights : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen] }) : Tensor { dtype := .Float32, device := device, shape := #[] } :=
  let mu := Tensor.exp (Tensor.clampFloat logits (-100.0) 11.0)
  let safeTarget := Tensor.clampFloat target 1.0e-8 1.0e20
  let per := Tensor.sub (Tensor.sub mu (Tensor.mul target (logDynamic mu)))
    (Tensor.sub target (Tensor.mul target (logDynamic safeTarget)))
  let masked := Tensor.mul (Tensor.mul per mask) weights
  let denom := Tensor.sumAll mask
  Tensor.div (Tensor.sumAll masked) (Tensor.addScalar denom 1.0e-8)

private def moleculeMaskedWeightedBCEWithLogits {batch maxLen : Nat}
    (logits target : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen] })
    (mask : Tensor { dtype := .Float32, device := device, shape := #[batch, maxLen] })
    (weights : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen] }) : Tensor { dtype := .Float32, device := device, shape := #[] } :=
  let per := Tensor.sub (softplusDynamic logits) (Tensor.mul target logits)
  let loss := Tensor.sumAll (Tensor.mul (Tensor.mul per mask) weights)
  let denom := Tensor.sumAll mask
  Tensor.div loss (Tensor.addScalar denom 1.0e-8)

structure BranchingMoleculeLossReport where
  total : Float
  coord : Float
  label : Float
  splits : Float
  del : Float
  deriving Repr

/-- Loss tensors for the molecule four-head objective, without any host
    readback — safe to run inside a CUDA graph capture region. -/
def moleculeLossTensors {batch maxLen vocab : Nat}
    (cfg : BranchingTrainConfig)
    (packed : BranchingMoleculeBatch (dtype := dtype) (device := device) batch maxLen)
    (coordPred : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen, 3] })
    (labelLogits : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen, vocab] })
    (splitLogits : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen] })
    (delLogits : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen] }) :
    Tensor { dtype := .Float32, device := device, shape := #[] } × Tensor { dtype := .Float32, device := device, shape := #[] } × Tensor { dtype := .Float32, device := device, shape := #[] } × Tensor { dtype := .Float32, device := device, shape := #[] } × Tensor { dtype := .Float32, device := device, shape := #[] } :=
  let mask := moleculeMask packed.padmask packed.flowmask
  let flowScale := moleculeFlowLossScale (maxLen := maxLen) packed.t
  let coordLoss := moleculeMaskedWeightedMSE3d coordPred packed.coordAnchor mask flowScale
  let labelLoss := moleculeMaskedWeightedCrossEntropy labelLogits packed.labelAnchor mask packed.labelLossScale
  let splitLoss := moleculeMaskedWeightedSplitPoissonLoss splitLogits packed.splitsTarget mask flowScale
  let delLoss := moleculeMaskedWeightedBCEWithLogits delLogits packed.delTarget mask flowScale
  let totalLoss :=
    Tensor.add
      (Tensor.add
        (Tensor.add
          (Tensor.mulScalar (Tensor.mulScalar coordLoss cfg.anchorWeight) cfg.coordWeight)
          (Tensor.mulScalar (Tensor.mulScalar labelLoss cfg.anchorWeight) cfg.labelWeight))
        (Tensor.mulScalar splitLoss cfg.splitsWeight))
      (Tensor.mulScalar delLoss cfg.delWeight)
  (totalLoss, coordLoss, labelLoss, splitLoss, delLoss)

/-- Host-side loss report from the loss tensors (syncs on read). -/
def moleculeLossReport
    (total coord label splits del : Tensor { dtype := .Float32, device := device, shape := #[] }) : BranchingMoleculeLossReport := {
  total := Tensor.item total,
  coord := Tensor.item coord,
  label := Tensor.item label,
  splits := Tensor.item splits,
  del := Tensor.item del
}

def moleculeLosses {batch maxLen vocab : Nat}
    (cfg : BranchingTrainConfig)
    (packed : BranchingMoleculeBatch (dtype := dtype) (device := device) batch maxLen)
    (coordPred : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen, 3] })
    (labelLogits : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen, vocab] })
    (splitLogits : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen] })
    (delLogits : Tensor { dtype := dtype, device := device, shape := #[batch, maxLen] }) :
    Tensor { dtype := .Float32, device := device, shape := #[] } × BranchingMoleculeLossReport :=
  let (totalLoss, coordLoss, labelLoss, splitLoss, delLoss) :=
    moleculeLossTensors cfg packed coordPred labelLogits splitLogits delLogits
  (totalLoss, moleculeLossReport totalLoss coordLoss labelLoss splitLoss delLoss)

def trainStepMolecule {maxLen vocab : Nat} {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params]
    (cfg : BranchingTrainConfig)
    (model : BranchingMoleculeModel (dtype := dtype) (device := device) maxLen vocab Params)
    (params : Params)
    (optState : Optim.AdamWState Params)
    (result : BranchingBridgeResult MoleculeAtom)
    (lr : Float)
    (labelDFM : Option DistNoisyDiscreteConfig := none)
    (clipGrads : Params → Float → IO Unit := fun params maxNorm => do
      let _ ← TensorStruct.mapM (fun tensor => do
        let _ ← Tensor.clipGradNorm tensor maxNorm
        pure tensor) params
      pure ())
    : IO (Params × Optim.AdamWState Params × BranchingMoleculeLossReport) := do
  let params ← TensorStruct.makeLeafParams params
  TensorStruct.zeroGrads params
  if cfg.maxLen != maxLen then throw <| IO.userError "BranchingFlows model/config maxLen mismatch"
  let ⟨batch, packedCpu⟩ ← packBranchingMolecule { cfg with maxLen := maxLen } result labelDFM
  let packed ← (packedCpu.toDevice device).castDType dtype
  let (coordPred, labelLogits, splitLogits, delLogits) ←
    model.forward (batch := batch) params packed.coord packed.label packed.t packed.padmask
  let (totalLoss, report) :=
    moleculeLosses (vocab := vocab) cfg packed coordPred labelLogits splitLogits delLogits

  Tensor.backwardLoss totalLoss
  if cfg.gradClip > 0 then
    clipGrads params cfg.gradClip

  let grads ← TensorStruct.grads params
  let opt := Optim.adamw (lr := lr) (weight_decay := cfg.weightDecay)
  let (params', optState') ← Optim.step opt params grads optState

  return (params', optState', report)

/-! ## Julia-compatible Muon optimizer path -/

/-- Momentum-only state used by the Julia-compatible molecule Muon update. -/
structure MoleculeMuonState (Params : Type) where
  momentum : Params
  step : Nat := 0

def initMoleculeMuonState [TensorStruct Params] (params : Params) : MoleculeMuonState Params :=
  { momentum := TensorStruct.map Tensor.zerosLike params }

private def flattenedMatrixShape (s : Shape) : Nat × Nat :=
  if s.isEmpty then
    (1, 1)
  else
    let rows := s[0]!
    let cols := (s.extract 1 s.size).foldl (fun acc d => acc * d) 1
    (rows, cols)

private theorem foldl_mul_acc (l : List Nat) (acc : Nat) :
    l.foldl (fun a d => a * d) acc = acc * l.foldl (fun a d => a * d) 1 := by
  induction l generalizing acc with
  | nil => simp
  | cons x xs ih => simp only [List.foldl]; rw [ih, ih (1 * x)]; simp [Nat.mul_assoc]

private theorem flattenedMatrixShape_numel (s : Shape) :
    (flattenedMatrixShape s).1 * (flattenedMatrixShape s).2 = TensorSpec.numelOfShape s := by
  rcases s with ⟨_ | ⟨a, rest⟩⟩
  · simp [flattenedMatrixShape]
  · simp [flattenedMatrixShape, TensorSpec.numelOfShape]
    rw [foldl_mul_acc rest a]

/--
Apply the same first-dimension-versus-rest flattening used by the Julia Muon
rule before Newton–Schulz orthogonalization.  This also makes vector leaves a
valid `n × 1` matrix for the IO-backed Tyr kernel.
-/
private def moleculeMuonOrthogonalize {σ : StaticSpec}
    (grad : Tensor σ) (numIters : Nat) : IO (Tensor (σ.withDType .BFloat16)) := do
  let dims := flattenedMatrixShape σ.shape
  have hDims : TensorSpec.numelOfShape #[dims.1, dims.2] = TensorSpec.numelOfShape σ.shape := by
    rw [← flattenedMatrixShape_numel σ.shape]; simp [dims]
  let flat := grad.reshape #[dims.1, dims.2] hDims.symm
  -- The C++ Newton–Schulz normalizes with an epsilon guard (`X / (norm + 1e-7)`,
  -- `cc/src/tyr_polar.cpp`), so a zero gradient orthogonalizes safely to zero.
  -- A host-side zero check here would force a GPU→CPU sync per leaf per step.
  let orth ← Optim.PolarExpress.muonOrthogonalize flat numIters
  pure (orth.reshape σ.shape hDims)

/--
One Muon update over an arbitrary tensor tree.

This follows the Julia demo's `CannotWaitForTheseOptimisers.Muon`: momentum and
Nesterov blending on every tensor leaf, first-dimension matrix flattening,
Newton–Schulz orthogonalization, aspect-ratio scaling, and decoupled weight
decay.
-/
def moleculeMuonStep [TensorStruct Params] [FloatingTensorStruct Params]
    (params grads : Params)
    (state : MoleculeMuonState Params)
    (lr : Float)
    (weightDecay : Float := 0.01)
    (momentumCoeff : Float := 0.95)
    (numIters : Nat := 5) : IO (Params × MoleculeMuonState Params) := do
  let rawGrads := TensorStruct.map Tensor.detach grads
  let momentum := FloatingTensorStruct.zipWith (fun old grad =>
    Tensor.add (Tensor.mulScalar old momentumCoeff) (Tensor.mulScalar grad (1.0 - momentumCoeff)))
    state.momentum rawGrads
  let nesterov := FloatingTensorStruct.zipWith (fun grad mom =>
    Tensor.add (Tensor.mulScalar grad (1.0 - momentumCoeff)) (Tensor.mulScalar mom momentumCoeff))
    rawGrads momentum
  let orth ← TensorStruct.mapM (fun {σ} grad => do
    let direction ← moleculeMuonOrthogonalize grad numIters
    let (rows, cols) := flattenedMatrixShape σ.shape
    let aspect := Float.sqrt (max 1.0 (rows.toFloat / cols.toFloat))
    -- Preserve native BF16 scaling before restoring the parameter dtype.
    let scaled := Tensor.mulScalar direction (lr * aspect)
    pure (Tensor.castLike (σ := σ.withDType .BFloat16) (τ := σ) grad scaled)) nesterov
  let params' := FloatingTensorStruct.zipWith (fun param direction =>
    let p := Tensor.detach param
    let update :=
      Tensor.add direction (Tensor.mulScalar p (lr * weightDecay))
    Tensor.sub p update) params orth
  let params' ← TensorStruct.makeLeafParams params'
  return (params', { momentum, step := state.step + 1 })

/-- The tensor learning-rate update operates on Float32 parameters on its device. -/
abbrev TensorLearningRateTree (device : Device) (Params : Type) :=
  TensorStructFor (fun σ => σ.dtype = .Float32 ∧ σ.device = device) Params

instance {shape : Shape} {device : Device} :
    TensorStructFor.Fact ((fun σ : StaticSpec => σ.dtype = .Float32 ∧ σ.device = device)
      ⟨shape, .Float32, device⟩) := ⟨⟨rfl, rfl⟩⟩

/-- `moleculeMuonStep` with the learning rate supplied as a 0-dim tensor, so a
    captured CUDA graph can read it from a static buffer (a baked Float would
    be frozen at capture time). BF16 aspect and learning-rate multiplications
    retain their separate rounding steps. -/
def moleculeMuonStepT [TensorStruct Params] [FloatingTensorStruct Params]
    [TensorLearningRateTree device Params]
    (params grads : Params)
    (state : MoleculeMuonState Params)
    (lrT : Tensor { dtype := .Float32, device := device, shape := #[] })
    (weightDecay : Float := 0.01)
    (momentumCoeff : Float := 0.95)
    (numIters : Nat := 5) : IO (Params × MoleculeMuonState Params) := do
  let rawGrads := TensorStruct.map Tensor.detach grads
  let momentum := FloatingTensorStruct.zipWith (fun old grad =>
    Tensor.add (Tensor.mulScalar old momentumCoeff) (Tensor.mulScalar grad (1.0 - momentumCoeff)))
    state.momentum rawGrads
  let nesterov := FloatingTensorStruct.zipWith (fun grad mom =>
    Tensor.add (Tensor.mulScalar grad (1.0 - momentumCoeff)) (Tensor.mulScalar mom momentumCoeff))
    rawGrads momentum
  let orth ← TensorStruct.mapM (fun {σ} grad => do
    let direction ← moleculeMuonOrthogonalize grad numIters
    let (rows, cols) := flattenedMatrixShape σ.shape
    let aspect := Float.sqrt (max 1.0 (rows.toFloat / cols.toFloat))
    let promoted ← IO.ofExcept (mulTensorScalar? (Tensor.mulScalar direction aspect) lrT)
    let scaled ← IO.ofExcept (Tensor.castLike? grad promoted.tensor)
    IO.ofExcept (Tensor.ofRaw scaled.raw)) nesterov
  let params' := TensorStructFor.zipWith
      (P := fun σ => σ.dtype = .Float32 ∧ σ.device = device)
      (fun {σ} proof param direction => by
        rcases σ with ⟨shape, dtype, placement⟩
        obtain ⟨rfl, rfl⟩ := proof
        letI : TensorSpec.Promotes ⟨shape, .Float32⟩ ⟨#[], .Float32⟩ .Float32 := by
          constructor
          unfold TensorSpec.pointwiseDType?
          cases shape.toList <;> rfl
        have placementEq : (StaticSpec.mk shape .Float32 placement).pointwiseDevice
            ⟨#[], .Float32, placement⟩ = placement := by
          unfold StaticSpec.pointwiseDevice
          split <;> first | rfl | (split <;> rfl)
        let scaled : Tensor ⟨shape, .Float32, placement⟩ := by
          simpa only [placementEq] using mulTensorScalar (Tensor.detach param) lrT
        exact Tensor.sub (Tensor.detach param)
          (Tensor.add direction (Tensor.mulScalar scaled weightDecay)))
      params orth
  let params' ← TensorStruct.makeLeafParams params'
  return (params', { momentum, step := state.step + 1 })

/-- Graph-safe core of `trainStepMoleculeMuon`: takes a pre-packed batch (the
    static input buffer) and a tensor learning rate, performs no host
    readbacks, and returns the loss tensors (host reporting happens outside
    the capture region). Used by the CUDA-graph training loop. -/
def trainStepMoleculeMuonCoreT {batch maxLen vocab : Nat} {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params]
    [TensorLearningRateTree device Params]
    (cfg : BranchingTrainConfig)
    (model : BranchingMoleculeModel (dtype := dtype) (device := device) maxLen vocab Params)
    (params : Params)
    (optState : MoleculeMuonState Params)
    (packed : BranchingMoleculeBatch (dtype := dtype) (device := device) batch maxLen)
    (lrT : Tensor { dtype := .Float32, device := device, shape := #[] })
    (_labelDFM : Option DistNoisyDiscreteConfig := none)
    (momentumCoeff : Float := 0.95)
    (numIters : Nat := 5)
    (clipGrads : Params → Float → IO Unit := fun params maxNorm => do
      let _ ← TensorStruct.mapM (fun tensor => do
        let _ ← Tensor.clipGradNorm tensor maxNorm
        pure tensor) params
      pure ())
    : IO (Params × MoleculeMuonState Params × (Tensor { dtype := .Float32, device := device, shape := #[] } × Tensor { dtype := .Float32, device := device, shape := #[] } × Tensor { dtype := .Float32, device := device, shape := #[] } × Tensor { dtype := .Float32, device := device, shape := #[] } × Tensor { dtype := .Float32, device := device, shape := #[] })) := do
  let params ← TensorStruct.makeLeafParams params
  TensorStruct.zeroGrads params
  let (coordPred, labelLogits, splitLogits, delLogits) ←
    model.forward (batch := batch) params packed.coord packed.label packed.t packed.padmask
  let (totalLoss, coordLoss, labelLoss, splitLoss, delLoss) :=
    moleculeLossTensors (vocab := vocab) cfg packed coordPred labelLogits splitLogits delLogits
  Tensor.backwardLoss totalLoss
  if cfg.gradClip > 0 then
    clipGrads params cfg.gradClip
  let grads ← TensorStruct.grads params
  let (params', optState') ←
    moleculeMuonStepT params grads optState lrT cfg.weightDecay momentumCoeff numIters
  return (params', optState', (totalLoss, coordLoss, labelLoss, splitLoss, delLoss))

def trainStepMoleculeMuon {maxLen vocab : Nat} {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params]
    (cfg : BranchingTrainConfig)
    (model : BranchingMoleculeModel (dtype := dtype) (device := device) maxLen vocab Params)
    (params : Params)
    (optState : MoleculeMuonState Params)
    (result : BranchingBridgeResult MoleculeAtom)
    (lr : Float)
    (labelDFM : Option DistNoisyDiscreteConfig := none)
    (momentumCoeff : Float := 0.95)
    (numIters : Nat := 5)
    (clipGrads : Params → Float → IO Unit := fun params maxNorm => do
      let _ ← TensorStruct.mapM (fun tensor => do
        let _ ← Tensor.clipGradNorm tensor maxNorm
        pure tensor) params
      pure ())
    (timePhases : Bool := false)
    : IO (Params × MoleculeMuonState Params × BranchingMoleculeLossReport) := do
  let t0 ← IO.monoMsNow
  let params ← TensorStruct.makeLeafParams params
  TensorStruct.zeroGrads params
  if cfg.maxLen != maxLen then throw <| IO.userError "BranchingFlows model/config maxLen mismatch"
  let ⟨batch, packedCpu⟩ ← packBranchingMolecule { cfg with maxLen := maxLen } result labelDFM
  let packedDev := packedCpu.toDevice device
  let packed ← packedDev.castDType dtype
  if timePhases then torch.cuda_synchronize
  let t1 ← IO.monoMsNow
  let (coordPred, labelLogits, splitLogits, delLogits) ←
    model.forward (batch := batch) params packed.coord packed.label packed.t packed.padmask
  let (totalLoss, report) :=
    moleculeLosses (vocab := vocab) cfg packed coordPred labelLogits splitLogits delLogits
  if timePhases then torch.cuda_synchronize
  let t2 ← IO.monoMsNow

  Tensor.backwardLoss totalLoss
  if cfg.gradClip > 0 then
    clipGrads params cfg.gradClip
  if timePhases then torch.cuda_synchronize
  let t3 ← IO.monoMsNow

  let grads ← TensorStruct.grads params
  let (params', optState') ←
    moleculeMuonStep params grads optState lr cfg.weightDecay momentumCoeff numIters
  if timePhases then torch.cuda_synchronize
  let t4 ← IO.monoMsNow
  if timePhases && optState.step % 50 == 0 then
    IO.println s!"molecule_train_phases step={optState.step} pack_ms={t1 - t0} fwd_ms={t2 - t1} bwd_ms={t3 - t2} muon_ms={t4 - t3}"
  return (params', optState', report)

def evalMoleculeLoss {maxLen vocab : Nat} {Params : Type} [TensorStruct Params]
    (cfg : BranchingTrainConfig)
    (model : BranchingMoleculeModel (dtype := dtype) (device := device) maxLen vocab Params)
    (params : Params)
    (result : BranchingBridgeResult MoleculeAtom)
    (labelDFM : Option DistNoisyDiscreteConfig := none)
    : IO BranchingMoleculeLossReport := do
  let params := TensorStruct.detach params
  if cfg.maxLen != maxLen then throw <| IO.userError "BranchingFlows model/config maxLen mismatch"
  let ⟨batch, packedCpu⟩ ← packBranchingMolecule { cfg with maxLen := maxLen } result labelDFM
  let packedDev := packedCpu.toDevice device
  let packed ← packedDev.castDType dtype
  let (coordPred, labelLogits, splitLogits, delLogits) ←
    model.forward (batch := batch) params packed.coord packed.label packed.t packed.padmask
  let (_, report) :=
    moleculeLosses (vocab := vocab) cfg packed coordPred labelLogits splitLogits delLogits
  pure report

/-- Run the molecule `branchingBridge` over a (sub-)batch of targets. Pure. -/
private def runMoleculeBridgeBatch
    (bridgeCfg : MoleculeBridgeConfig)
    (branchTime : TimeDist) (deletionTime : TimeDist)
    (policy : CoalescencePolicy MoleculeAtom)
    (coalescenceFactor : Float) (useBranchingTimeProb : Float)
    (maxLen : Option Nat) (maxResamples : Nat)
    (lengthMins : GroupMinsSpec) (deletionPad : Float)
    (targets : Array (BranchingState MoleculeAtom))
    (times : Array Float)
    (rng : Rng) : BranchingBridgeResult MoleculeAtom × Rng :=
  branchingBridge
    (fun cfg x0 x1 t0 t => MoleculeBridgeConfig.bridge cfg x0 x1 t0 t)
    bridgeCfg
    (fun _root => MoleculeBridgeConfig.maskedAtom bridgeCfg)
    targets
    times
    branchTime
    deletionTime
    policy
    (MoleculeBridgeConfig.anchorMerge bridgeCfg)
    (coalescenceFactor := coalescenceFactor)
    (useBranchingTimeProb := useBranchingTimeProb)
    (maxLen := maxLen)
    (maxResamples := maxResamples)
    (lengthMins := lengthMins)
    (deletionPad := deletionPad)
    (x1Modifier := maskDeletedMoleculeLabels bridgeCfg)
    (rng := rng)
    (sampleBridge? := some (fun cfg x0 x1 t0 t rng =>
      MoleculeBridgeConfig.sampleBridge cfg x0 x1 t0 t rng))
    (sampleX0? := some (fun _root rng =>
      MoleculeBridgeConfig.sampleInitialAtom bridgeCfg rng))

/-- Golden-ratio jump constant for decorrelating per-chunk LCG streams. -/
private def chunkRng (rng : Rng) (chunk : Nat) : Rng :=
  { state := rng.state + (chunk.toUInt64 + 1) * 0x9E3779B97F4A7C15 }

/-- Concatenate two bridge results (field-wise). -/
private def appendBridgeResults (a b : BranchingBridgeResult MoleculeAtom) :
    BranchingBridgeResult MoleculeAtom :=
  { t := a.t ++ b.t
    segments := a.segments ++ b.segments
    Xt := a.Xt ++ b.Xt
    X1anchor := a.X1anchor ++ b.X1anchor
    descendants := a.descendants ++ b.descendants
    del := a.del ++ b.del
    splitsTarget := a.splitsTarget ++ b.splitsTarget
    prevCoalescence := a.prevCoalescence ++ b.prevCoalescence }

/-- Pure core of `sampleMoleculeBridgeBatch` (no validation throws), so the
    train loop can pre-spawn the next batch's sampling as a `Task` and overlap
    it with the current train step. All randomness flows through `rng`, so a
    pipelined call sequence is deterministic and identical to the sequential
    one. The parallel-chunk path itself uses `Task.spawn`/`Task.get`, which
    are pure and nest fine inside a worker thread. -/
def sampleMoleculeBridgeBatchPure
    (bridgeCfg : MoleculeBridgeConfig)
    (states : Array (BranchingState MoleculeAtom))
    (batchSize : Nat)
    (rng : Rng)
    (branchTime : TimeDist := TimeDist.betaOneThreeHalves)
    (deletionTime : TimeDist := TimeDist.uniform)
    (policy : CoalescencePolicy MoleculeAtom := sequentialUniformPolicy MoleculeAtom)
    (coalescenceFactor : Float := 1.0)
    (useBranchingTimeProb : Float := 0.0)
    (maxLen : Option Nat := none)
    (maxResamples : Nat := 8)
    (lengthMins : GroupMinsSpec := .uniform 1)
    (deletionPad : Float := 0.0)
    (parallelism : Nat := 1)
    : BranchingBridgeResult MoleculeAtom × Rng := Id.run do
  let mut rng := rng
  let mut targets : Array (BranchingState MoleculeAtom) := #[]
  let mut times : Array Float := #[]
  for _ in [:batchSize] do
    let (idx, rng') := randNat rng states.size
    rng := rng'
    let (u, rng') := randFloat rng
    rng := rng'
    let t := max 1.0e-4 (min 0.9999 u)
    targets := targets.push states[idx]!
    times := times.push t
  let numChunks := min parallelism batchSize
  if numChunks > 1 then
    -- Embarrassingly parallel across chunk tasks: each chunk gets its own
    -- decorrelated LCG stream; results are concatenated in chunk order.
    let chunkSize := (batchSize + numChunks - 1) / numChunks
    let mut tasks : Array (Task (BranchingBridgeResult MoleculeAtom × Rng)) := #[]
    for c in [:numChunks] do
      let lo := c * chunkSize
      let hi := min (lo + chunkSize) batchSize
      if hi > lo then
        let targetsC := targets.extract lo hi
        let timesC := times.extract lo hi
        let rngC := chunkRng rng c
        -- `Task.spawn` takes a thunk so the bridge computation runs on the
        -- worker thread; `IO.asTask (pure …)` would evaluate eagerly here.
        let task := Task.spawn fun () =>
          runMoleculeBridgeBatch bridgeCfg branchTime deletionTime policy
            coalescenceFactor useBranchingTimeProb maxLen maxResamples lengthMins
            deletionPad targetsC timesC rngC
        tasks := tasks.push task
    let mut merged : BranchingBridgeResult MoleculeAtom :=
      { t := #[], segments := #[], Xt := #[], X1anchor := #[], descendants := #[],
        del := #[], splitsTarget := #[], prevCoalescence := #[] }
    for task in tasks do
      let (res, _) := task.get
      merged := appendBridgeResults merged res
    -- The sequential path threads one rng through the whole batch; the parallel
    -- path uses decorrelated per-chunk streams, so the returned rng simply
    -- advances past the target/time draws above.
    pure (merged, rng)
  else
    pure (runMoleculeBridgeBatch bridgeCfg branchTime deletionTime policy
      coalescenceFactor useBranchingTimeProb maxLen maxResamples lengthMins
      deletionPad targets times rng)

def sampleMoleculeBridgeBatch
    (bridgeCfg : MoleculeBridgeConfig)
    (states : Array (BranchingState MoleculeAtom))
    (batchSize : Nat)
    (rng : Rng)
    (branchTime : TimeDist := TimeDist.betaOneThreeHalves)
    (deletionTime : TimeDist := TimeDist.uniform)
    (policy : CoalescencePolicy MoleculeAtom := sequentialUniformPolicy MoleculeAtom)
    (coalescenceFactor : Float := 1.0)
    (useBranchingTimeProb : Float := 0.0)
    (maxLen : Option Nat := none)
    (maxResamples : Nat := 8)
    (lengthMins : GroupMinsSpec := .uniform 1)
    (deletionPad : Float := 0.0)
    (parallelism : Nat := 1)
    : IO (BranchingBridgeResult MoleculeAtom × Rng) := do
  if states.isEmpty then
    throw (IO.userError "cannot sample a molecule bridge batch from an empty dataset")
  if batchSize == 0 then
    throw (IO.userError "molecule bridge batch size must be positive")
  pure (sampleMoleculeBridgeBatchPure bridgeCfg states batchSize rng branchTime deletionTime
    policy coalescenceFactor useBranchingTimeProb maxLen maxResamples lengthMins deletionPad
    parallelism)

end torch.branching
