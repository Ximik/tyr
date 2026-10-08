import Tyr.Model.BranchingFlows.Molecule
import Tyr.Model.BranchingFlowsTrain
import Tyr.Optim
import Tyr.Optim.PolarExpress

/-!
  Tensor packing helpers for molecule-shaped BranchingFlows batches.

  The generic training module has separate packers for token states and
  continuous coordinate states.  Molecule generation needs both in one element:
  3D coordinates plus a discrete atom label.
-/

namespace torch.branching

open torch
open torch.nn

structure BranchingMoleculeBatch (batch maxLen : UInt64) where
  t : T #[batch]
  coord : T #[batch, maxLen, 3]
  coordAnchor : T #[batch, maxLen, 3]
  label : T #[batch, maxLen]
  labelAnchor : T #[batch, maxLen]
  labelLossScale : T #[batch, maxLen]
  padmask : T #[batch, maxLen]
  flowmask : T #[batch, maxLen]
  splitsTarget : T #[batch, maxLen]
  delTarget : T #[batch, maxLen]
  deriving Repr

namespace BranchingMoleculeBatch

def toDevice {batch maxLen : UInt64}
    (device : Device) (x : BranchingMoleculeBatch batch maxLen) :
    BranchingMoleculeBatch batch maxLen :=
  { t := x.t.to device
    coord := x.coord.to device
    coordAnchor := x.coordAnchor.to device
    label := x.label.to device
    labelAnchor := x.labelAnchor.to device
    labelLossScale := x.labelLossScale.to device
    padmask := x.padmask.to device
    flowmask := x.flowmask.to device
    splitsTarget := x.splitsTarget.to device
    delTarget := x.delTarget.to device }

/-- Cast the float fields of a packed batch to bfloat16 (labels/masks stay
    integral). Used by the `--dtype bf16` experiment: halves bandwidth on the
    model's hot path; the optimizer then also runs in bf16. -/
def castBFloat16 {batch maxLen : UInt64}
    (x : BranchingMoleculeBatch batch maxLen) : BranchingMoleculeBatch batch maxLen :=
  { t := toBFloat16' x.t
    coord := toBFloat16' x.coord
    coordAnchor := toBFloat16' x.coordAnchor
    label := x.label
    labelAnchor := x.labelAnchor
    labelLossScale := toBFloat16' x.labelLossScale
    padmask := x.padmask
    flowmask := x.flowmask
    splitsTarget := toBFloat16' x.splitsTarget
    delTarget := toBFloat16' x.delTarget }

/-- Allocate a static batch of zero buffers (float fields f32, integral fields
    i64), optionally bf16-cast — the CUDA-graph static input buffer. -/
def zeros (batch maxLen : UInt64) (device : Device) (bf16 : Bool := false) :
    BranchingMoleculeBatch batch maxLen :=
  let f (s : Shape) : T s := zerosOn device
  let i (s : Shape) : T s := (full_int s 0).to device
  let b : BranchingMoleculeBatch batch maxLen := {
    t := f #[batch]
    coord := f #[batch, maxLen, 3]
    coordAnchor := f #[batch, maxLen, 3]
    label := i #[batch, maxLen]
    labelAnchor := i #[batch, maxLen]
    labelLossScale := f #[batch, maxLen]
    padmask := i #[batch, maxLen]
    flowmask := i #[batch, maxLen]
    splitsTarget := f #[batch, maxLen]
    delTarget := f #[batch, maxLen]
  }
  if bf16 then b.castBFloat16 else b

/-- Copy a freshly packed batch into the static buffer in place. The batch
    and maxLen indices may differ at the type level (the packed batch carries
    existentials / structure projections); runtime shapes must match —
    `copy_` enforces that, and the sampler always produces exactly the
    configured batch size. -/
def copyInto {b1 b2 m1 m2 : UInt64}
    (dst : BranchingMoleculeBatch b1 m1)
    (src : BranchingMoleculeBatch b2 m2) :
    BranchingMoleculeBatch b1 m1 :=
  { t := copyInplace dst.t src.t
    coord := copyInplace dst.coord src.coord
    coordAnchor := copyInplace dst.coordAnchor src.coordAnchor
    label := copyInplace dst.label src.label
    labelAnchor := copyInplace dst.labelAnchor src.labelAnchor
    labelLossScale := copyInplace dst.labelLossScale src.labelLossScale
    padmask := copyInplace dst.padmask src.padmask
    flowmask := copyInplace dst.flowmask src.flowmask
    splitsTarget := copyInplace dst.splitsTarget src.splitsTarget
    delTarget := copyInplace dst.delTarget src.delTarget }

/-- Force every field eagerly (see `torch.touch`): batch construction and
    `copyInto` are lazy, and without forcing they would be evaluated inside
    the CUDA graph capture region / after the replay launch. -/
def eval {batch maxLen : UInt64}
    (b : BranchingMoleculeBatch batch maxLen) : IO (BranchingMoleculeBatch batch maxLen) := do
  torch.touch b.t
  torch.touch b.coord
  torch.touch b.coordAnchor
  torch.touch b.label
  torch.touch b.labelAnchor
  torch.touch b.labelLossScale
  torch.touch b.padmask
  torch.touch b.flowmask
  torch.touch b.splitsTarget
  torch.touch b.delTarget
  pure b

end BranchingMoleculeBatch

structure BranchingMoleculeModel (maxLen vocab : UInt64) (Params : Type) where
  forward : {batch : UInt64} → Params → T #[batch, maxLen, 3] → T #[batch, maxLen] → T #[batch]
    → T #[batch, maxLen]
    → IO (T #[batch, maxLen, 3] × T #[batch, maxLen, vocab] × T #[batch, maxLen] × T #[batch, maxLen])

private def coordOffset (maxLenNat bi j k : Nat) : Nat :=
  (bi * maxLenNat + j) * 3 + k

private def writeVec3 (arr : Array Float) (offset : Nat) (v : Vec3) : Array Float :=
  let arr := arr.set! offset v.x
  let arr := arr.set! (offset + 1) v.y
  arr.set! (offset + 2) v.z

def packBranchingMolecule (cfg : BranchingTrainConfig)
    (result : BranchingBridgeResult MoleculeAtom)
    (labelDFM : Option DistNoisyDiscreteConfig := none)
    : IO (Sigma fun batch => BranchingMoleculeBatch batch cfg.maxLen) := do
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

  let batchU : UInt64 := batchNat.toUInt64
  let coord := reshape (data.fromFloatArray coordArr) #[batchU, cfg.maxLen, 3]
  let coordAnchor := reshape (data.fromFloatArray coordAnchorArr) #[batchU, cfg.maxLen, 3]
  let label := reshape (data.fromInt64Array labelArr) #[batchU, cfg.maxLen]
  let labelAnchor := reshape (data.fromInt64Array labelAnchorArr) #[batchU, cfg.maxLen]
  let labelLossScale := reshape (data.fromFloatArray labelScaleArr) #[batchU, cfg.maxLen]
  let padmask := toFloat' (reshape (data.fromInt64Array padArr) #[batchU, cfg.maxLen])
  let flowmask := toFloat' (reshape (data.fromInt64Array flowArr) #[batchU, cfg.maxLen])
  let splitsTarget := toFloat' (reshape (data.fromInt64Array splitsArr) #[batchU, cfg.maxLen])
  let delTarget := toFloat' (reshape (data.fromInt64Array delArr) #[batchU, cfg.maxLen])

  let timeScaled := result.t.map (fun t => ((t * cfg.timeScale).toUInt64).toInt64)
  let t := reshape (data.fromInt64Array timeScaled) #[batchU]
  let t := (toFloat' t) / cfg.timeScale

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
    : IO (Sigma fun batch => BranchingMoleculeBatch batch cfg.maxLen) :=
  packBranchingMolecule cfg result (labelDFM := some labelDFM)

private def moleculeMask {batch maxLen : UInt64}
    (pad flow : T #[batch, maxLen]) : T #[batch, maxLen] :=
  pad * flow

private def moleculeMaskedWeightedCrossEntropy {batch maxLen vocab : UInt64}
    (logits : T #[batch, maxLen, vocab])
    (targets : T #[batch, maxLen])
    (mask : T #[batch, maxLen])
    (weights : T #[batch, maxLen]) : T #[] :=
  let n : UInt64 := batch * maxLen
  let logits2 := reshape logits #[n, vocab]
  let targets2 := reshape targets #[n]
  let mask2 := reshape mask #[n]
  let weights2 := reshape weights #[n]
  let per := nn.cross_entropy_none logits2 targets2
  let masked := per * mask2 * weights2
  let denom := nn.sumAll mask2
  nn.div (nn.sumAll masked) (denom + (1.0e-8 : Float))

private def moleculeFlowLossScale {batch maxLen : UInt64}
    (t : T #[batch]) : T #[batch, maxLen] :=
  let remaining : T #[batch] := torch.add_scalar (torch.mul_scalar t (-1.0)) 1.2
  let scale : T #[batch] := nn.pow remaining (-1.0)
  nn.expand (nn.unsqueeze scale 1) #[batch, maxLen]

private def moleculeMaskedWeightedMSE3d {batch maxLen dim : UInt64}
    (pred target : T #[batch, maxLen, dim])
    (mask weights : T #[batch, maxLen]) : T #[] :=
  let mask3 := nn.expand (nn.unsqueeze mask 2) #[batch, maxLen, dim]
  let weights3 := nn.expand (nn.unsqueeze weights 2) #[batch, maxLen, dim]
  let diff := pred - target
  let sq := diff * diff
  let masked := sq * mask3 * weights3
  let denom := nn.sumAll mask3
  nn.div (nn.sumAll masked) (denom + (1.0e-8 : Float))

private def moleculeMaskedWeightedSplitPoissonLoss {batch maxLen : UInt64}
    (logits target mask weights : T #[batch, maxLen]) : T #[] :=
  let mu := nn.exp (torch.clampFloat logits (-100.0) 11.0)
  let safeTarget := torch.clampFloat target 1.0e-8 1.0e20
  let per := mu - target * nn.log mu - (target - target * nn.log safeTarget)
  let masked := per * mask * weights
  let denom := nn.sumAll mask
  nn.div (nn.sumAll masked) (denom + (1.0e-8 : Float))

private def moleculeMaskedWeightedBCEWithLogits {batch maxLen : UInt64}
    (logits target mask weights : T #[batch, maxLen]) : T #[] :=
  let per := nn.softplus logits - target * logits
  let loss := nn.sumAll (per * mask * weights)
  let denom := nn.sumAll mask
  nn.div loss (denom + (1.0e-8 : Float))

structure BranchingMoleculeLossReport where
  total : Float
  coord : Float
  label : Float
  splits : Float
  del : Float
  deriving Repr

/-- Loss tensors for the molecule four-head objective, without any host
    readback — safe to run inside a CUDA graph capture region. -/
def moleculeLossTensors {batch maxLen vocab : UInt64}
    (cfg : BranchingTrainConfig)
    (packed : BranchingMoleculeBatch batch maxLen)
    (coordPred : T #[batch, maxLen, 3])
    (labelLogits : T #[batch, maxLen, vocab])
    (splitLogits : T #[batch, maxLen])
    (delLogits : T #[batch, maxLen]) :
    T #[] × T #[] × T #[] × T #[] × T #[] :=
  let mask := moleculeMask packed.padmask packed.flowmask
  let flowScale := moleculeFlowLossScale (maxLen := maxLen) packed.t
  let coordLoss := moleculeMaskedWeightedMSE3d coordPred packed.coordAnchor mask flowScale
  let labelLoss := moleculeMaskedWeightedCrossEntropy labelLogits packed.labelAnchor mask packed.labelLossScale
  let splitLoss := moleculeMaskedWeightedSplitPoissonLoss splitLogits packed.splitsTarget mask flowScale
  let delLoss := moleculeMaskedWeightedBCEWithLogits delLogits packed.delTarget mask flowScale
  let totalLoss :=
    coordLoss * cfg.anchorWeight * cfg.coordWeight +
    labelLoss * cfg.anchorWeight * cfg.labelWeight +
    splitLoss * cfg.splitsWeight +
    delLoss * cfg.delWeight
  (totalLoss, coordLoss, labelLoss, splitLoss, delLoss)

/-- Host-side loss report from the loss tensors (syncs on read). -/
def moleculeLossReport
    (total coord label splits del : T #[]) : BranchingMoleculeLossReport := {
  total := nn.item total,
  coord := nn.item coord,
  label := nn.item label,
  splits := nn.item splits,
  del := nn.item del
}

def moleculeLosses {batch maxLen vocab : UInt64}
    (cfg : BranchingTrainConfig)
    (packed : BranchingMoleculeBatch batch maxLen)
    (coordPred : T #[batch, maxLen, 3])
    (labelLogits : T #[batch, maxLen, vocab])
    (splitLogits : T #[batch, maxLen])
    (delLogits : T #[batch, maxLen]) :
    T #[] × BranchingMoleculeLossReport :=
  let (totalLoss, coordLoss, labelLoss, splitLoss, delLoss) :=
    moleculeLossTensors cfg packed coordPred labelLogits splitLogits delLogits
  (totalLoss, moleculeLossReport totalLoss coordLoss labelLoss splitLoss delLoss)

def trainStepMolecule {maxLen vocab : UInt64} {Params : Type} [TensorStruct Params]
    (cfg : BranchingTrainConfig)
    (model : BranchingMoleculeModel maxLen vocab Params)
    (params : Params)
    (optState : Optim.AdamWState Params)
    (result : BranchingBridgeResult MoleculeAtom)
    (lr : Float)
    (labelDFM : Option DistNoisyDiscreteConfig := none)
    (clipGrads : Params → Float → IO Unit := fun params maxNorm => do
      let _ ← TensorStruct.mapM (fun tensor => do
        let _ ← nn.clip_grad_norm_ tensor maxNorm
        pure tensor) params
      pure ())
    (device : Device := Device.CPU)
    : IO (Params × Optim.AdamWState Params × BranchingMoleculeLossReport) := do
  let params := TensorStruct.zeroGrads (TensorStruct.makeLeafParams params)
  let ⟨batch, packedCpu⟩ ← packBranchingMolecule cfg result labelDFM
  let packed := packedCpu.toDevice device
  let (coordPred, labelLogits, splitLogits, delLogits) ←
    model.forward (batch := batch) params packed.coord packed.label packed.t packed.padmask
  let (totalLoss, report) :=
    moleculeLosses (vocab := vocab) cfg packed coordPred labelLogits splitLogits delLogits

  autograd.backwardLoss totalLoss
  if cfg.gradClip > 0 then
    clipGrads params cfg.gradClip

  let grads := TensorStruct.grads params
  let opt := Optim.adamw (lr := lr) (weight_decay := cfg.weightDecay)
  let (params', optState') := Optim.step opt params grads optState

  return (params', optState', report)

/-! ## Julia-compatible Muon optimizer path -/

/-- Momentum-only state used by the Julia-compatible molecule Muon update. -/
structure MoleculeMuonState (Params : Type) where
  momentum : Params
  step : Nat := 0

def initMoleculeMuonState [TensorStruct Params] (params : Params) : MoleculeMuonState Params :=
  { momentum := TensorStruct.map torch.zeros_like params }

private def flattenedMatrixShape (s : Shape) : UInt64 × UInt64 :=
  if s.isEmpty then
    (1, 1)
  else
    let rows := s[0]!
    let cols := (s.extract 1 s.size).foldl (fun acc d => acc * d) 1
    (rows, cols)

/--
Apply the same first-dimension-versus-rest flattening used by the Julia Muon
rule before Newton–Schulz orthogonalization.  This also makes vector leaves a
valid `n × 1` matrix for the IO-backed Tyr kernel.
-/
private def moleculeMuonOrthogonalize {s : Shape}
    (grad : T s) (numIters : UInt64) : IO (T s) := do
  let (rows, cols) := flattenedMatrixShape s
  let flat : T #[rows, cols] := reshape grad #[rows, cols]
  -- The C++ Newton–Schulz normalizes with an epsilon guard (`X / (norm + 1e-7)`,
  -- `cc/src/tyr_polar.cpp`), so a zero gradient orthogonalizes safely to zero.
  -- A host-side zero check here would force a GPU→CPU sync per leaf per step.
  let orth ← Optim.PolarExpress.muonOrthogonalize flat numIters
  pure (reshape orth s)

/--
One Muon update over an arbitrary tensor tree.

This follows the Julia demo's `CannotWaitForTheseOptimisers.Muon`: momentum and
Nesterov blending on every tensor leaf, first-dimension matrix flattening,
Newton–Schulz orthogonalization, aspect-ratio scaling, and decoupled weight
decay.
-/
def moleculeMuonStep [TensorStruct Params]
    (params grads : Params)
    (state : MoleculeMuonState Params)
    (lr : Float)
    (weightDecay : Float := 0.01)
    (momentumCoeff : Float := 0.95)
    (numIters : UInt64 := 5) : IO (Params × MoleculeMuonState Params) := do
  let rawGrads := TensorStruct.map autograd.detach grads
  let momentum := TensorStruct.zipWith (fun old grad =>
    torch.mul_scalar old momentumCoeff + torch.mul_scalar grad (1.0 - momentumCoeff))
    state.momentum rawGrads
  let nesterov := TensorStruct.zipWith (fun grad mom =>
    torch.mul_scalar grad (1.0 - momentumCoeff) + torch.mul_scalar mom momentumCoeff)
    rawGrads momentum
  let orth ← TensorStruct.mapM (fun grad => moleculeMuonOrthogonalize grad numIters) nesterov
  let params' := TensorStruct.zipWith (fun {s} param direction =>
    let (rows, cols) := flattenedMatrixShape s
    let aspect := Float.sqrt (max 1.0 (rows.toFloat / cols.toFloat))
    let p := autograd.detach param
    let update :=
      torch.mul_scalar direction (lr * aspect) +
      torch.mul_scalar p (lr * weightDecay)
    autograd.set_requires_grad (p - update) true) params orth
  return (params', { momentum, step := state.step + 1 })

/-- `moleculeMuonStep` with the learning rate supplied as a 0-dim tensor, so a
    captured CUDA graph can read it from a static buffer (a baked Float would
    be frozen at capture time). Numerically identical. -/
def moleculeMuonStepT [TensorStruct Params]
    (params grads : Params)
    (state : MoleculeMuonState Params)
    (lrT : T #[])
    (weightDecay : Float := 0.01)
    (momentumCoeff : Float := 0.95)
    (numIters : UInt64 := 5) : IO (Params × MoleculeMuonState Params) := do
  let rawGrads := TensorStruct.map autograd.detach grads
  let momentum := TensorStruct.zipWith (fun old grad =>
    torch.mul_scalar old momentumCoeff + torch.mul_scalar grad (1.0 - momentumCoeff))
    state.momentum rawGrads
  let nesterov := TensorStruct.zipWith (fun grad mom =>
    torch.mul_scalar grad (1.0 - momentumCoeff) + torch.mul_scalar mom momentumCoeff)
    rawGrads momentum
  let orth ← TensorStruct.mapM (fun grad => moleculeMuonOrthogonalize grad numIters) nesterov
  let params' := TensorStruct.zipWith (fun {s} param direction =>
    let (rows, cols) := flattenedMatrixShape s
    let aspect := Float.sqrt (max 1.0 (rows.toFloat / cols.toFloat))
    let p := autograd.detach param
    let update :=
      torch.mulTensorScalar (torch.mul_scalar direction aspect) lrT +
      torch.mul_scalar (torch.mulTensorScalar p lrT) weightDecay
    autograd.set_requires_grad (p - update) true) params orth
  return (params', { momentum, step := state.step + 1 })

/-- Graph-safe core of `trainStepMoleculeMuon`: takes a pre-packed batch (the
    static input buffer) and a tensor learning rate, performs no host
    readbacks, and returns the loss tensors (host reporting happens outside
    the capture region). Used by the CUDA-graph training loop. -/
def trainStepMoleculeMuonCoreT {batch maxLen vocab : UInt64} {Params : Type} [TensorStruct Params]
    (cfg : BranchingTrainConfig)
    (model : BranchingMoleculeModel maxLen vocab Params)
    (params : Params)
    (optState : MoleculeMuonState Params)
    (packed : BranchingMoleculeBatch batch maxLen)
    (lrT : T #[])
    (_labelDFM : Option DistNoisyDiscreteConfig := none)
    (momentumCoeff : Float := 0.95)
    (numIters : UInt64 := 5)
    (clipGrads : Params → Float → IO Unit := fun params maxNorm => do
      let _ ← TensorStruct.mapM (fun tensor => do
        let _ ← nn.clip_grad_norm_ tensor maxNorm
        pure tensor) params
      pure ())
    : IO (Params × MoleculeMuonState Params × (T #[] × T #[] × T #[] × T #[] × T #[])) := do
  let params := TensorStruct.zeroGrads (TensorStruct.makeLeafParams params)
  let (coordPred, labelLogits, splitLogits, delLogits) ←
    model.forward (batch := batch) params packed.coord packed.label packed.t packed.padmask
  let (totalLoss, coordLoss, labelLoss, splitLoss, delLoss) :=
    moleculeLossTensors (vocab := vocab) cfg packed coordPred labelLogits splitLogits delLogits
  autograd.backwardLoss totalLoss
  if cfg.gradClip > 0 then
    clipGrads params cfg.gradClip
  let grads := TensorStruct.grads params
  let (params', optState') ←
    moleculeMuonStepT params grads optState lrT cfg.weightDecay momentumCoeff numIters
  return (params', optState', (totalLoss, coordLoss, labelLoss, splitLoss, delLoss))

def trainStepMoleculeMuon {maxLen vocab : UInt64} {Params : Type} [TensorStruct Params]
    (cfg : BranchingTrainConfig)
    (model : BranchingMoleculeModel maxLen vocab Params)
    (params : Params)
    (optState : MoleculeMuonState Params)
    (result : BranchingBridgeResult MoleculeAtom)
    (lr : Float)
    (labelDFM : Option DistNoisyDiscreteConfig := none)
    (momentumCoeff : Float := 0.95)
    (numIters : UInt64 := 5)
    (clipGrads : Params → Float → IO Unit := fun params maxNorm => do
      let _ ← TensorStruct.mapM (fun tensor => do
        let _ ← nn.clip_grad_norm_ tensor maxNorm
        pure tensor) params
      pure ())
    (device : Device := Device.CPU)
    (timePhases : Bool := false)
    (bf16 : Bool := false)
    : IO (Params × MoleculeMuonState Params × BranchingMoleculeLossReport) := do
  let t0 ← IO.monoMsNow
  let params := TensorStruct.zeroGrads (TensorStruct.makeLeafParams params)
  let ⟨batch, packedCpu⟩ ← packBranchingMolecule cfg result labelDFM
  let packedDev := packedCpu.toDevice device
  let packed := if bf16 then packedDev.castBFloat16 else packedDev
  if timePhases then torch.cuda_synchronize
  let t1 ← IO.monoMsNow
  let (coordPred, labelLogits, splitLogits, delLogits) ←
    model.forward (batch := batch) params packed.coord packed.label packed.t packed.padmask
  let (totalLoss, report) :=
    moleculeLosses (vocab := vocab) cfg packed coordPred labelLogits splitLogits delLogits
  if timePhases then torch.cuda_synchronize
  let t2 ← IO.monoMsNow

  autograd.backwardLoss totalLoss
  if cfg.gradClip > 0 then
    clipGrads params cfg.gradClip
  if timePhases then torch.cuda_synchronize
  let t3 ← IO.monoMsNow

  let grads := TensorStruct.grads params
  let (params', optState') ←
    moleculeMuonStep params grads optState lr cfg.weightDecay momentumCoeff numIters
  if timePhases then torch.cuda_synchronize
  let t4 ← IO.monoMsNow
  if timePhases && optState.step % 50 == 0 then
    IO.println s!"molecule_train_phases step={optState.step} pack_ms={t1 - t0} fwd_ms={t2 - t1} bwd_ms={t3 - t2} muon_ms={t4 - t3}"
  return (params', optState', report)

def evalMoleculeLoss {maxLen vocab : UInt64} {Params : Type}
    (cfg : BranchingTrainConfig)
    (model : BranchingMoleculeModel maxLen vocab Params)
    (params : Params)
    (result : BranchingBridgeResult MoleculeAtom)
    (labelDFM : Option DistNoisyDiscreteConfig := none)
    (device : Device := Device.CPU)
    (bf16 : Bool := false)
    : IO BranchingMoleculeLossReport := do
  torch.autograd.no_grad do
    let ⟨batch, packedCpu⟩ ← packBranchingMolecule cfg result labelDFM
    let packedDev := packedCpu.toDevice device
    let packed := if bf16 then packedDev.castBFloat16 else packedDev
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
