import Tyr.Model.BranchingFlows.MoleculeTrain

/-!
  A small trainable molecule transformer for BranchingFlows batches.

  The paper-scale QM9 model stacks many transformer layers with richer spatial
  features.  This module keeps the same training boundary: coordinates, labels,
  time, padding mask, and heads for coordinate endpoints, label logits, split
  logits, and deletion logits.
-/

namespace torch.branching

variable {device : Device}
open torch

structure MoleculeTransformerParams (vocab heads headDim mlp : UInt64) where
  coordW : Tensor { device := device, shape := #[heads * headDim, 3] }
  coordB : Tensor { device := device, shape := #[heads * headDim] }
  labelEmbed : Tensor { device := device, shape := #[vocab, heads * headDim] }
  timeW : Tensor { device := device, shape := #[heads * headDim, 1] }
  timeB : Tensor { device := device, shape := #[heads * headDim] }
  attnLnWeight : Tensor { device := device, shape := #[heads * headDim] }
  attnLnBias : Tensor { device := device, shape := #[heads * headDim] }
  qW : Tensor { device := device, shape := #[heads * headDim, heads * headDim] }
  qB : Tensor { device := device, shape := #[heads * headDim] }
  kW : Tensor { device := device, shape := #[heads * headDim, heads * headDim] }
  kB : Tensor { device := device, shape := #[heads * headDim] }
  vW : Tensor { device := device, shape := #[heads * headDim, heads * headDim] }
  vB : Tensor { device := device, shape := #[heads * headDim] }
  oW : Tensor { device := device, shape := #[heads * headDim, heads * headDim] }
  oB : Tensor { device := device, shape := #[heads * headDim] }
  pairDistSlope : Tensor { device := device, shape := #[heads] }
  mlpLnWeight : Tensor { device := device, shape := #[heads * headDim] }
  mlpLnBias : Tensor { device := device, shape := #[heads * headDim] }
  fc1W : Tensor { device := device, shape := #[mlp, heads * headDim] }
  fc1B : Tensor { device := device, shape := #[mlp] }
  fc2W : Tensor { device := device, shape := #[heads * headDim, mlp] }
  fc2B : Tensor { device := device, shape := #[heads * headDim] }
  coordHeadW : Tensor { device := device, shape := #[3, heads * headDim] }
  coordHeadB : Tensor { device := device, shape := #[3] }
  labelHeadW : Tensor { device := device, shape := #[vocab, heads * headDim] }
  labelHeadB : Tensor { device := device, shape := #[vocab] }
  splitHeadW : Tensor { device := device, shape := #[1, heads * headDim] }
  splitHeadB : Tensor { device := device, shape := #[1] }
  delHeadW : Tensor { device := device, shape := #[1, heads * headDim] }
  delHeadB : Tensor { device := device, shape := #[1] }
  deriving TensorStruct

namespace MoleculeTransformerParams

private def boundCoordTarget {s : Shape} (coord : Tensor { device := device, shape := s }) (cap? : Option Float) : Tensor { device := device, shape := s } :=
  match cap? with
  | some cap =>
      if cap > 0.0 then
        Tensor.Promoting.mulScalar (Tensor.Preserving.tanh (Tensor.Promoting.divScalar coord cap)) cap
      else
        coord
  | none => coord

private def randnScaled (shape : Shape) (scale : Float) : IO (Tensor { device := device, shape := shape }) := do
  let x ← randnDynamic (device := device) shape
  pure (Tensor.Promoting.mulScalar x scale)

def init (vocab heads headDim mlp : UInt64) (device : Device := .CPU) : IO (MoleculeTransformerParams (device := device) vocab heads headDim mlp) := do
  let initScale := 0.05
  let coordW ← randnScaled #[heads * headDim, 3] initScale
  let labelEmbed ← randnScaled #[vocab, heads * headDim] initScale
  let timeW ← randnScaled #[heads * headDim, 1] initScale
  let qW ← randnScaled #[heads * headDim, heads * headDim] initScale
  let kW ← randnScaled #[heads * headDim, heads * headDim] initScale
  let vW ← randnScaled #[heads * headDim, heads * headDim] initScale
  let oW ← randnScaled #[heads * headDim, heads * headDim] initScale
  let fc1W ← randnScaled #[mlp, heads * headDim] initScale
  let fc2W ← randnScaled #[heads * headDim, mlp] initScale
  let coordHeadW ← randnScaled #[3, heads * headDim] initScale
  let labelHeadW ← randnScaled #[vocab, heads * headDim] initScale
  let splitHeadW ← randnScaled #[1, heads * headDim] initScale
  let delHeadW ← randnScaled #[1, heads * headDim] initScale
  pure {
    coordW,
    coordB := zerosDynamic (device := device) #[heads * headDim],
    labelEmbed,
    timeW,
    timeB := zerosDynamic (device := device) #[heads * headDim],
    attnLnWeight := onesDynamic (device := device) #[heads * headDim],
    attnLnBias := zerosDynamic (device := device) #[heads * headDim],
    qW,
    qB := zerosDynamic (device := device) #[heads * headDim],
    kW,
    kB := zerosDynamic (device := device) #[heads * headDim],
    vW,
    vB := zerosDynamic (device := device) #[heads * headDim],
    oW,
    oB := zerosDynamic (device := device) #[heads * headDim],
    pairDistSlope := Tensor.Promoting.subScalar (zerosDynamic (device := device) #[heads]) 4.0,
    mlpLnWeight := onesDynamic (device := device) #[heads * headDim],
    mlpLnBias := zerosDynamic (device := device) #[heads * headDim],
    fc1W,
    fc1B := zerosDynamic (device := device) #[mlp],
    fc2W,
    fc2B := zerosDynamic (device := device) #[heads * headDim],
    coordHeadW,
    coordHeadB := zerosDynamic (device := device) #[3],
    labelHeadW,
    labelHeadB := zerosDynamic (device := device) #[vocab],
    splitHeadW,
    splitHeadB := zerosDynamic (device := device) #[1],
    delHeadW,
    delHeadB := zerosDynamic (device := device) #[1]
  }

private def pairwiseDistanceSq {batch maxLen : UInt64}
    (coord : Tensor { device := device, shape := #[batch, maxLen, 3] }) : Tensor { device := device, shape := #[batch, maxLen, maxLen] } :=
  let ci0 : Tensor { device := device, shape := #[batch, maxLen, 1, 3] } := coord.reshapeChecked #[batch, maxLen, 1, 3]
  let cj0 : Tensor { device := device, shape := #[batch, 1, maxLen, 3] } := coord.reshapeChecked #[batch, 1, maxLen, 3]
  let ci : Tensor { device := device, shape := #[batch, maxLen, maxLen, 3] } := Tensor.expand ci0 #[batch, maxLen, maxLen, 3]
  let cj : Tensor { device := device, shape := #[batch, maxLen, maxLen, 3] } := Tensor.expand cj0 #[batch, maxLen, maxLen, 3]
  let diff := Tensor.Promoting.sub ci cj
  (Tensor.Preserving.sumDim (Tensor.Promoting.mul diff diff) 3 false).reshapeChecked #[batch, maxLen, maxLen]

private def spatialAttention {batch maxLen vocab heads headDim mlp : UInt64}
    (params : MoleculeTransformerParams (device := device) vocab heads headDim mlp)
    (coord : Tensor { device := device, shape := #[batch, maxLen, 3] })
    (padmask : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 })
    (h : Tensor { device := device, shape := #[batch, maxLen, heads * headDim] }) : Tensor { device := device, shape := #[batch, maxLen, heads * headDim] } :=
  let q0 : Tensor { device := device, shape := #[batch, maxLen, heads * headDim] } := Tensor.affine3d h params.qW params.qB
  let k0 : Tensor { device := device, shape := #[batch, maxLen, heads * headDim] } := Tensor.affine3d h params.kW params.kB
  let v0 : Tensor { device := device, shape := #[batch, maxLen, heads * headDim] } := Tensor.affine3d h params.vW params.vB
  let q : Tensor { device := device, shape := #[batch, heads, maxLen, headDim] } :=
    transposeAttention (Tensor.reshapeChecked q0 #[batch, maxLen, heads, headDim])
  let k : Tensor { device := device, shape := #[batch, heads, maxLen, headDim] } :=
    transposeAttention (Tensor.reshapeChecked k0 #[batch, maxLen, heads, headDim])
  let v : Tensor { device := device, shape := #[batch, heads, maxLen, headDim] } :=
    transposeAttention (Tensor.reshapeChecked v0 #[batch, maxLen, heads, headDim])
  let dist : Tensor { device := device, shape := #[batch, maxLen, maxLen] } := pairwiseDistanceSq coord
  let slope : Tensor { device := device, shape := #[heads] } := softplusDynamic params.pairDistSlope
  let distBias : Tensor { device := device, shape := #[batch, heads, maxLen, maxLen] } :=
    Tensor.Promoting.mul
      (Tensor.expand (Tensor.reshapeChecked slope #[1, heads, 1, 1]) #[batch, heads, maxLen, maxLen])
      (Tensor.expand (Tensor.reshapeChecked dist #[batch, 1, maxLen, maxLen]) #[batch, heads, maxLen, maxLen])
  let keyMask0 : Tensor { device := device, shape := #[batch, 1, 1, maxLen] } := (floatDynamic padmask).reshapeChecked #[batch, 1, 1, maxLen]
  -- (1 - keyMask) · -1e9, computed on-device (a `zerosDynamic`-based masked_fill
  -- would be CPU-resident and crash on CUDA inputs).
  let maskTerm : Tensor { device := device, shape := #[batch, 1, 1, maxLen] } :=
    Tensor.Promoting.mulScalar (Tensor.Promoting.addScalar (Tensor.Promoting.mulScalar keyMask0 (-1.0)) 1.0) (-1.0e9)
  let bias : Tensor { device := device, shape := #[batch, heads, maxLen, maxLen] } :=
    Tensor.Promoting.sub (Tensor.expand maskTerm #[batch, heads, maxLen, maxLen]) distBias
  let ctx : Tensor { device := device, shape := #[batch, heads, maxLen, headDim] } :=
    attentionBias q k v bias
  let out4 : Tensor { device := device, shape := #[batch, maxLen, heads, headDim] } := transposeAttention ctx
  let out3 : Tensor { device := device, shape := #[batch, maxLen, heads * headDim] } := Tensor.reshapeChecked out4 #[batch, maxLen, heads * headDim]
  Tensor.affine3d out3 params.oW params.oB

def forward {batch maxLen vocab heads headDim mlp : UInt64}
    (params : MoleculeTransformerParams (device := device) vocab heads headDim mlp)
    (coord : Tensor { device := device, shape := #[batch, maxLen, 3] })
    (label : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 })
    (t : Tensor { device := device, shape := #[batch] })
    (padmask : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 })
    (coordTargetCap? : Option Float := none) :
    IO (Tensor { device := device, shape := #[batch, maxLen, 3] } × Tensor { device := device, shape := #[batch, maxLen, vocab] } × Tensor { device := device, shape := #[batch, maxLen] } × Tensor { device := device, shape := #[batch, maxLen] }) := do
  let coord := Tensor.castLike params.coordW coord
  let t := Tensor.castLike params.timeW t
  let coordEmb : Tensor { device := device, shape := #[batch, maxLen, heads * headDim] } := Tensor.affine3d coord params.coordW params.coordB
  let labelEmb : Tensor { device := device, shape := #[batch, maxLen, heads * headDim] } :=
    Tensor.embedding (batch := batch) (seq := maxLen) (vocab := vocab) (embed := heads * headDim)
      label params.labelEmbed
  let time0 : Tensor { device := device, shape := #[batch, 1, 1] } := t.reshapeChecked #[batch, 1, 1]
  let timeIn : Tensor { device := device, shape := #[batch, maxLen, 1] } := Tensor.expand time0 #[batch, maxLen, 1]
  let timeEmb : Tensor { device := device, shape := #[batch, maxLen, heads * headDim] } := Tensor.affine3d timeIn params.timeW params.timeB
  let h0 := Tensor.Promoting.add (Tensor.Promoting.add coordEmb labelEmb) timeEmb
  let h1n := Tensor.layerNorm h0 params.attnLnWeight params.attnLnBias 1.0e-5
  let attnOut := spatialAttention params coord padmask h1n
  let h1 := Tensor.Promoting.add h0 attnOut
  let h2n := Tensor.layerNorm h1 params.mlpLnWeight params.mlpLnBias 1.0e-5
  let mlp1 := Tensor.Preserving.gelu (Tensor.affine3d h2n params.fc1W params.fc1B)
  let mlp2 := Tensor.affine3d mlp1 params.fc2W params.fc2B
  let h := Tensor.Promoting.add h1 mlp2
  let coordRaw : Tensor { device := device, shape := #[batch, maxLen, 3] } := Tensor.affine3d h params.coordHeadW params.coordHeadB
  let coordPred := boundCoordTarget coordRaw coordTargetCap?
  let labelLogits : Tensor { device := device, shape := #[batch, maxLen, vocab] } := Tensor.affine3d h params.labelHeadW params.labelHeadB
  let split3 : Tensor { device := device, shape := #[batch, maxLen, 1] } := Tensor.affine3d h params.splitHeadW params.splitHeadB
  let del3 : Tensor { device := device, shape := #[batch, maxLen, 1] } := Tensor.affine3d h params.delHeadW params.delHeadB
  let splitLogits : Tensor { device := device, shape := #[batch, maxLen] } := Tensor.reshapeChecked split3 #[batch, maxLen]
  let delLogits : Tensor { device := device, shape := #[batch, maxLen] } := Tensor.reshapeChecked del3 #[batch, maxLen]
  pure (coordPred, labelLogits, splitLogits, delLogits)

end MoleculeTransformerParams

def moleculeTransformerModel {maxLen vocab heads headDim mlp : UInt64}
    (coordTargetCap? : Option Float := none) :
    BranchingMoleculeModel (device := device) maxLen vocab (MoleculeTransformerParams (device := device) vocab heads headDim mlp) :=
  { forward := fun {batch} params coord label t padmask =>
      MoleculeTransformerParams.forward (batch := batch) params coord label t padmask coordTargetCap? }

structure FullMoleculeTransformerLayerParams
    (hidden heads headDim mlp rff : UInt64) where
  attnLnWeight : Tensor { device := device, shape := #[hidden] }
  attnLnBias : Tensor { device := device, shape := #[hidden] }
  qW : Tensor { device := device, shape := #[heads * headDim, hidden] }
  qB : Tensor { device := device, shape := #[heads * headDim] }
  kW : Tensor { device := device, shape := #[heads * headDim, hidden] }
  kB : Tensor { device := device, shape := #[heads * headDim] }
  vW : Tensor { device := device, shape := #[heads * headDim, hidden] }
  vB : Tensor { device := device, shape := #[heads * headDim] }
  oW : Tensor { device := device, shape := #[hidden, heads * headDim] }
  oB : Tensor { device := device, shape := #[hidden] }
  pairW : Tensor { device := device, shape := #[heads, rff * 2] }
  pairB : Tensor { device := device, shape := #[heads] }
  mlpLnWeight : Tensor { device := device, shape := #[hidden] }
  mlpLnBias : Tensor { device := device, shape := #[hidden] }
  fc1W : Tensor { device := device, shape := #[mlp, hidden] }
  fc1B : Tensor { device := device, shape := #[mlp] }
  fc2W : Tensor { device := device, shape := #[hidden, mlp] }
  fc2B : Tensor { device := device, shape := #[hidden] }
  coordLnWeight : Tensor { device := device, shape := #[hidden] }
  coordLnBias : Tensor { device := device, shape := #[hidden] }
  coordUpdateW : Tensor { device := device, shape := #[3, hidden] }
  coordUpdateB : Tensor { device := device, shape := #[3] }
  deriving TensorStruct

structure FullMoleculeTransformerParams
    (vocab hidden heads headDim mlp rff layers : UInt64) where
  coordRffW : Tensor { device := device, shape := #[rff, 3] }
  coordRffB : Tensor { device := device, shape := #[rff] }
  coordProjW : Tensor { device := device, shape := #[hidden, rff * 2] }
  coordProjB : Tensor { device := device, shape := #[hidden] }
  pairRffW : Tensor { device := device, shape := #[rff, 1] }
  pairRffB : Tensor { device := device, shape := #[rff] }
  labelEmbed : Tensor { device := device, shape := #[vocab, hidden] }
  timeW : Tensor { device := device, shape := #[hidden, 1] }
  timeB : Tensor { device := device, shape := #[hidden] }
  blocks : Array (FullMoleculeTransformerLayerParams (device := device) hidden heads headDim mlp rff)
  finalLnWeight : Tensor { device := device, shape := #[hidden] }
  finalLnBias : Tensor { device := device, shape := #[hidden] }
  coordHeadW : Tensor { device := device, shape := #[3, hidden] }
  coordHeadB : Tensor { device := device, shape := #[3] }
  labelHeadW : Tensor { device := device, shape := #[vocab, hidden] }
  labelHeadB : Tensor { device := device, shape := #[vocab] }
  splitHeadW : Tensor { device := device, shape := #[1, hidden] }
  splitHeadB : Tensor { device := device, shape := #[1] }
  delHeadW : Tensor { device := device, shape := #[1, hidden] }
  delHeadB : Tensor { device := device, shape := #[1] }
  deriving TensorStruct

namespace FullMoleculeTransformerLayerParams

private def randnScaled (shape : Shape) (scale : Float) : IO (Tensor { device := device, shape := shape }) := do
  let x ← randnDynamic (device := device) shape
  pure (Tensor.Promoting.mulScalar x scale)

def init (hidden heads headDim mlp rff : UInt64) (device : Device := .CPU) :
    IO (FullMoleculeTransformerLayerParams (device := device) hidden heads headDim mlp rff) := do
  let initScale := 0.02
  let qW ← randnScaled #[heads * headDim, hidden] initScale
  let kW ← randnScaled #[heads * headDim, hidden] initScale
  let vW ← randnScaled #[heads * headDim, hidden] initScale
  let oW ← randnScaled #[hidden, heads * headDim] initScale
  let pairW ← randnScaled #[heads, rff * 2] initScale
  let fc1W ← randnScaled #[mlp, hidden] initScale
  let fc2W ← randnScaled #[hidden, mlp] initScale
  let coordUpdateW ← randnScaled #[3, hidden] 0.001
  pure {
    attnLnWeight := onesDynamic (device := device) #[hidden],
    attnLnBias := zerosDynamic (device := device) #[hidden],
    qW,
    qB := zerosDynamic (device := device) #[heads * headDim],
    kW,
    kB := zerosDynamic (device := device) #[heads * headDim],
    vW,
    vB := zerosDynamic (device := device) #[heads * headDim],
    oW,
    oB := zerosDynamic (device := device) #[hidden],
    pairW,
    pairB := zerosDynamic (device := device) #[heads],
    mlpLnWeight := onesDynamic (device := device) #[hidden],
    mlpLnBias := zerosDynamic (device := device) #[hidden],
    fc1W,
    fc1B := zerosDynamic (device := device) #[mlp],
    fc2W,
    fc2B := zerosDynamic (device := device) #[hidden],
    coordLnWeight := onesDynamic (device := device) #[hidden],
    coordLnBias := zerosDynamic (device := device) #[hidden],
    coordUpdateW,
    coordUpdateB := zerosDynamic (device := device) #[3]
  }

end FullMoleculeTransformerLayerParams

namespace FullMoleculeTransformerParams

private def boundCoordTarget {s : Shape} (coord : Tensor { device := device, shape := s }) (cap? : Option Float) : Tensor { device := device, shape := s } :=
  match cap? with
  | some cap =>
      if cap > 0.0 then
        Tensor.Promoting.mulScalar (Tensor.Preserving.tanh (Tensor.Promoting.divScalar coord cap)) cap
      else
        coord
  | none => coord

private def randnScaled (shape : Shape) (scale : Float) : IO (Tensor { device := device, shape := shape }) := do
  let x ← randnDynamic (device := device) shape
  pure (Tensor.Promoting.mulScalar x scale)

def init (vocab hidden heads headDim mlp rff layers : UInt64) (device : Device := .CPU) :
    IO (FullMoleculeTransformerParams (device := device) vocab hidden heads headDim mlp rff layers) := do
  let coordRffW ← randnScaled #[rff, 3] 1.0
  let coordRffB ← randnScaled #[rff] 1.0
  let coordProjW ← randnScaled #[hidden, rff * 2] 0.02
  let pairRffW ← randnScaled #[rff, 1] 1.0
  let pairRffB ← randnScaled #[rff] 1.0
  let labelEmbed ← randnScaled #[vocab, hidden] 0.02
  let timeW ← randnScaled #[hidden, 1] 0.02
  let mut blocks := #[]
  for _ in [:layers.toNat] do
    blocks := blocks.push (← FullMoleculeTransformerLayerParams.init (device := device) hidden heads headDim mlp rff)
  let coordHeadW ← randnScaled #[3, hidden] 0.02
  let labelHeadW ← randnScaled #[vocab, hidden] 0.02
  let splitHeadW ← randnScaled #[1, hidden] 0.02
  let delHeadW ← randnScaled #[1, hidden] 0.02
  pure {
    coordRffW,
    coordRffB,
    coordProjW,
    coordProjB := zerosDynamic (device := device) #[hidden],
    pairRffW,
    pairRffB,
    labelEmbed,
    timeW,
    timeB := zerosDynamic (device := device) #[hidden],
    blocks,
    finalLnWeight := onesDynamic (device := device) #[hidden],
    finalLnBias := zerosDynamic (device := device) #[hidden],
    coordHeadW,
    coordHeadB := zerosDynamic (device := device) #[3],
    labelHeadW,
    labelHeadB := zerosDynamic (device := device) #[vocab],
    splitHeadW,
    splitHeadB := zerosDynamic (device := device) #[1],
    delHeadW,
    delHeadB := zerosDynamic (device := device) #[1]
  }

private def pairwiseDistanceSq {batch maxLen : UInt64}
    (coord : Tensor { device := device, shape := #[batch, maxLen, 3] }) : Tensor { device := device, shape := #[batch, maxLen, maxLen] } :=
  let ci0 : Tensor { device := device, shape := #[batch, maxLen, 1, 3] } := coord.reshapeChecked #[batch, maxLen, 1, 3]
  let cj0 : Tensor { device := device, shape := #[batch, 1, maxLen, 3] } := coord.reshapeChecked #[batch, 1, maxLen, 3]
  let ci : Tensor { device := device, shape := #[batch, maxLen, maxLen, 3] } := Tensor.expand ci0 #[batch, maxLen, maxLen, 3]
  let cj : Tensor { device := device, shape := #[batch, maxLen, maxLen, 3] } := Tensor.expand cj0 #[batch, maxLen, maxLen, 3]
  let diff := Tensor.Promoting.sub ci cj
  (Tensor.Preserving.sumDim (Tensor.Promoting.mul diff diff) 3 false).reshapeChecked #[batch, maxLen, maxLen]

private def coordRffEmbedding {batch maxLen vocab hidden heads headDim mlp rff layers : UInt64}
    (params : FullMoleculeTransformerParams (device := device) vocab hidden heads headDim mlp rff layers)
    (coord : Tensor { device := device, shape := #[batch, maxLen, 3] }) : Tensor { device := device, shape := #[batch, maxLen, hidden] } :=
  let flat : Tensor { device := device, shape := #[batch * maxLen, 3] } := Tensor.reshapeChecked coord #[batch * maxLen, 3]
  let phase : Tensor { device := device, shape := #[batch * maxLen, rff] } := affine2d flat params.coordRffW params.coordRffB
  let featFlat : Tensor { device := device, shape := #[batch * maxLen, rff * 2] } := (Tensor.cat (Tensor.Preserving.sin phase) (cosDynamic phase) 1).reshapeChecked #[batch * maxLen, rff * 2]
  let features : Tensor { device := device, shape := #[batch, maxLen, rff * 2] } := Tensor.reshapeChecked featFlat #[batch, maxLen, rff * 2]
  Tensor.affine3d features params.coordProjW params.coordProjB

private def pairRffFeatures {batch maxLen vocab hidden heads headDim mlp rff layers : UInt64}
    (params : FullMoleculeTransformerParams (device := device) vocab hidden heads headDim mlp rff layers)
    (coord : Tensor { device := device, shape := #[batch, maxLen, 3] }) : Tensor { device := device, shape := #[batch * maxLen * maxLen, rff * 2] } :=
  let dist : Tensor { device := device, shape := #[batch, maxLen, maxLen] } := pairwiseDistanceSq coord
  let distFlat : Tensor { device := device, shape := #[batch * maxLen * maxLen, 1] } := Tensor.reshapeChecked dist #[batch * maxLen * maxLen, 1]
  let phase : Tensor { device := device, shape := #[batch * maxLen * maxLen, rff] } := affine2d distFlat params.pairRffW params.pairRffB
  (Tensor.cat (Tensor.Preserving.sin phase) (cosDynamic phase) 1).reshapeChecked #[batch * maxLen * maxLen, rff * 2]

private def pairAttentionBiasFrom {batch maxLen hidden heads headDim mlp rff : UInt64}
    (layer : FullMoleculeTransformerLayerParams (device := device) hidden heads headDim mlp rff)
    (features : Tensor { device := device, shape := #[batch * maxLen * maxLen, rff * 2] }) : Tensor { device := device, shape := #[batch, heads, maxLen, maxLen] } :=
  let biasFlat : Tensor { device := device, shape := #[batch * maxLen * maxLen, heads] } := affine2d features layer.pairW layer.pairB
  let bias0 : Tensor { device := device, shape := #[batch, maxLen, maxLen, heads] } := Tensor.reshapeChecked biasFlat #[batch, maxLen, maxLen, heads]
  let bias1 : Tensor { device := device, shape := #[batch, maxLen, heads, maxLen] } := Tensor.transpose bias0 2 3
  Tensor.transpose bias1 1 2

private def spatialAttention {batch maxLen vocab hidden heads headDim mlp rff layers : UInt64}
    (_params : FullMoleculeTransformerParams (device := device) vocab hidden heads headDim mlp rff layers)
    (layer : FullMoleculeTransformerLayerParams (device := device) hidden heads headDim mlp rff)
    (pairFeatures : Tensor { device := device, shape := #[batch * maxLen * maxLen, rff * 2] })
    (padmask : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 })
    (ropeCos : Tensor { device := device, shape := #[maxLen, headDim / 2] })
    (ropeSin : Tensor { device := device, shape := #[maxLen, headDim / 2] })
    (h : Tensor { device := device, shape := #[batch, maxLen, hidden] }) : Tensor { device := device, shape := #[batch, maxLen, hidden] } :=
  let q0 : Tensor { device := device, shape := #[batch, maxLen, heads * headDim] } := Tensor.affine3d h layer.qW layer.qB
  let k0 : Tensor { device := device, shape := #[batch, maxLen, heads * headDim] } := Tensor.affine3d h layer.kW layer.kB
  let v0 : Tensor { device := device, shape := #[batch, maxLen, heads * headDim] } := Tensor.affine3d h layer.vW layer.vB
  let qR : Tensor { device := device, shape := #[batch, maxLen, heads, headDim] } :=
    Tensor.applyRotary (Tensor.reshapeChecked q0 #[batch, maxLen, heads, headDim])
      (Tensor.castLike q0 ropeCos) (Tensor.castLike q0 ropeSin)
  let kR : Tensor { device := device, shape := #[batch, maxLen, heads, headDim] } :=
    Tensor.applyRotary (Tensor.reshapeChecked k0 #[batch, maxLen, heads, headDim])
      (Tensor.castLike k0 ropeCos) (Tensor.castLike k0 ropeSin)
  let q : Tensor { device := device, shape := #[batch, heads, maxLen, headDim] } := transposeAttention qR
  let k : Tensor { device := device, shape := #[batch, heads, maxLen, headDim] } := transposeAttention kR
  let v : Tensor { device := device, shape := #[batch, heads, maxLen, headDim] } :=
    transposeAttention (Tensor.reshapeChecked v0 #[batch, maxLen, heads, headDim])
  let pairBias : Tensor { device := device, shape := #[batch, heads, maxLen, maxLen] } := pairAttentionBiasFrom layer pairFeatures
  let keyMask0 : Tensor { device := device, shape := #[batch, 1, 1, maxLen] } := (floatDynamic padmask).reshapeChecked #[batch, 1, 1, maxLen]
  -- (1 - keyMask) · -1e9, computed on-device (a `zerosDynamic`-based masked_fill
  -- would be CPU-resident and crash on CUDA inputs).
  let maskTerm : Tensor { device := device, shape := #[batch, 1, 1, maxLen] } :=
    Tensor.Promoting.mulScalar (Tensor.Promoting.addScalar (Tensor.Promoting.mulScalar keyMask0 (-1.0)) 1.0) (-1.0e9)
  let bias : Tensor { device := device, shape := #[batch, heads, maxLen, maxLen] } :=
    Tensor.Promoting.add pairBias (Tensor.expand maskTerm #[batch, heads, maxLen, maxLen])
  let ctx : Tensor { device := device, shape := #[batch, heads, maxLen, headDim] } :=
    attentionBias q k v bias
  let out4 : Tensor { device := device, shape := #[batch, maxLen, heads, headDim] } := transposeAttention ctx
  let out3 : Tensor { device := device, shape := #[batch, maxLen, heads * headDim] } := Tensor.reshapeChecked out4 #[batch, maxLen, heads * headDim]
  Tensor.affine3d out3 layer.oW layer.oB

private def forwardLayer {batch maxLen vocab hidden heads headDim mlp rff layers : UInt64}
    (params : FullMoleculeTransformerParams (device := device) vocab hidden heads headDim mlp rff layers)
    (layer : FullMoleculeTransformerLayerParams (device := device) hidden heads headDim mlp rff)
    (applyCoordUpdate : Bool)
    (pairFeatures : Tensor { device := device, shape := #[batch * maxLen * maxLen, rff * 2] })
    (ropeCos : Tensor { device := device, shape := #[maxLen, headDim / 2] })
    (ropeSin : Tensor { device := device, shape := #[maxLen, headDim / 2] })
    (padmask : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 })
    (h : Tensor { device := device, shape := #[batch, maxLen, hidden] })
    (coord : Tensor { device := device, shape := #[batch, maxLen, 3] }) :
    Tensor { device := device, shape := #[batch, maxLen, hidden] } × Tensor { device := device, shape := #[batch, maxLen, 3] } :=
  let h1n := Tensor.layerNorm h layer.attnLnWeight layer.attnLnBias 1.0e-5
  let attnOut := spatialAttention params layer pairFeatures padmask ropeCos ropeSin h1n
  let h1 := Tensor.Promoting.add h attnOut
  let h2n := Tensor.layerNorm h1 layer.mlpLnWeight layer.mlpLnBias 1.0e-5
  let mlp1 := Tensor.Preserving.gelu (Tensor.affine3d h2n layer.fc1W layer.fc1B)
  let mlp2 := Tensor.affine3d mlp1 layer.fc2W layer.fc2B
  let h2 := Tensor.Promoting.add h1 mlp2
  if applyCoordUpdate then
    let coordN := Tensor.layerNorm h2 layer.coordLnWeight layer.coordLnBias 1.0e-5
    let delta : Tensor { device := device, shape := #[batch, maxLen, 3] } := Tensor.affine3d coordN layer.coordUpdateW layer.coordUpdateB
    (h2, Tensor.Promoting.add coord delta)
  else
    (h2, coord)

/-- Forward with precomputed rotary frequencies. Split out so CUDA-graph
    capture can hoist `rotary.computeFreqs` (which does a host→device copy,
    illegal during stream capture) out of the captured region. -/
def forwardFreqs {batch maxLen vocab hidden heads headDim mlp rff layers : UInt64}
    (params : FullMoleculeTransformerParams (device := device) vocab hidden heads headDim mlp rff layers)
    (coord : Tensor { device := device, shape := #[batch, maxLen, 3] })
    (label : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 })
    (t : Tensor { device := device, shape := #[batch] })
    (padmask : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 })
    (ropeCos : Tensor { device := device, shape := #[maxLen, headDim / 2] })
    (ropeSin : Tensor { device := device, shape := #[maxLen, headDim / 2] })
    (coordUpdateLayers : Nat := 6)
    (coordTargetCap? : Option Float := none) :
    IO (Tensor { device := device, shape := #[batch, maxLen, 3] } × Tensor { device := device, shape := #[batch, maxLen, vocab] } × Tensor { device := device, shape := #[batch, maxLen] } × Tensor { device := device, shape := #[batch, maxLen] }) := do
  let coord := Tensor.castLike params.coordRffW coord
  let t := Tensor.castLike params.timeW t
  let coordEmb : Tensor { device := device, shape := #[batch, maxLen, hidden] } := coordRffEmbedding params coord
  let labelEmb : Tensor { device := device, shape := #[batch, maxLen, hidden] } :=
    Tensor.embedding (batch := batch) (seq := maxLen) (vocab := vocab) (embed := hidden)
      label params.labelEmbed
  let time0 : Tensor { device := device, shape := #[batch, 1, 1] } := t.reshapeChecked #[batch, 1, 1]
  let timeIn : Tensor { device := device, shape := #[batch, maxLen, 1] } := Tensor.expand time0 #[batch, maxLen, 1]
  let timeEmb : Tensor { device := device, shape := #[batch, maxLen, hidden] } := Tensor.affine3d timeIn params.timeW params.timeB
  let mut h := Tensor.Promoting.add (Tensor.Promoting.add coordEmb labelEmb) timeEmb
  let mut runningCoord := coord
  let updateStart := if params.blocks.size > coordUpdateLayers then params.blocks.size - coordUpdateLayers else 0
  -- RFF pair features depend only on the running coordinates: shared by all
  -- layers until the next coordinate update, so recompute only then
  -- (7 computations instead of 12 at layers=12, coordUpdateLayers=6).
  let mut pairFeatures := pairRffFeatures params runningCoord
  let mut i := 0
  for block in params.blocks do
    if i > updateStart then
      pairFeatures := pairRffFeatures params runningCoord
    let (h', coord') :=
      forwardLayer params block (i >= updateStart) pairFeatures ropeCos ropeSin padmask h runningCoord
    h := h'
    runningCoord := coord'
    i := i + 1
  let hFinal := Tensor.layerNorm h params.finalLnWeight params.finalLnBias 1.0e-5
  let coordDelta : Tensor { device := device, shape := #[batch, maxLen, 3] } := Tensor.affine3d hFinal params.coordHeadW params.coordHeadB
  let coordPred := boundCoordTarget (Tensor.Promoting.add runningCoord coordDelta) coordTargetCap?
  let labelLogits : Tensor { device := device, shape := #[batch, maxLen, vocab] } := Tensor.affine3d hFinal params.labelHeadW params.labelHeadB
  let split3 : Tensor { device := device, shape := #[batch, maxLen, 1] } := Tensor.affine3d hFinal params.splitHeadW params.splitHeadB
  let del3 : Tensor { device := device, shape := #[batch, maxLen, 1] } := Tensor.affine3d hFinal params.delHeadW params.delHeadB
  let splitLogits : Tensor { device := device, shape := #[batch, maxLen] } := Tensor.reshapeChecked split3 #[batch, maxLen]
  let delLogits : Tensor { device := device, shape := #[batch, maxLen] } := Tensor.reshapeChecked del3 #[batch, maxLen]
  pure (coordPred, labelLogits, splitLogits, delLogits)

/-- Forward computing rotary frequencies on the fly (one host→device copy per
    call); see `forwardFreqs` for the graph-safe variant. -/
def forward {batch maxLen vocab hidden heads headDim mlp rff layers : UInt64}
    (params : FullMoleculeTransformerParams (device := device) vocab hidden heads headDim mlp rff layers)
    (coord : Tensor { device := device, shape := #[batch, maxLen, 3] })
    (label : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 })
    (t : Tensor { device := device, shape := #[batch] })
    (padmask : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 })
    (coordUpdateLayers : Nat := 6)
    (coordTargetCap? : Option Float := none) :
    IO (Tensor { device := device, shape := #[batch, maxLen, 3] } × Tensor { device := device, shape := #[batch, maxLen, vocab] } × Tensor { device := device, shape := #[batch, maxLen] } × Tensor { device := device, shape := #[batch, maxLen] }) := do
  let (ropeCos, ropeSin) := Tensor.rotaryFreqs maxLen headDim 10000.0
  let ropeCos := (ropeCos.to device).forgetDType
  let ropeSin := (ropeSin.to device).forgetDType
  forwardFreqs params coord label t padmask ropeCos ropeSin
    (coordUpdateLayers := coordUpdateLayers) (coordTargetCap? := coordTargetCap?)

end FullMoleculeTransformerParams

def fullMoleculeTransformerModel {maxLen vocab hidden heads headDim mlp rff layers : UInt64}
    (coordUpdateLayers : Nat := 6)
    (coordTargetCap? : Option Float := none) :
    BranchingMoleculeModel (device := device) maxLen vocab
      (FullMoleculeTransformerParams (device := device) vocab hidden heads headDim mlp rff layers) :=
  { forward := fun {batch} params coord label t padmask =>
      FullMoleculeTransformerParams.forward (batch := batch) params coord label t padmask
        (coordUpdateLayers := coordUpdateLayers) (coordTargetCap? := coordTargetCap?) }

private def clampUpper? (cap? : Option Float) (x : Float) : Float :=
  match cap? with
  | some cap => min x cap
  | none => x

private def scalar3d {maxLen channels : UInt64}
    (x : Tensor { device := device, shape := #[1, maxLen, channels] }) (j k : Nat) : Float :=
  let row : Tensor { device := device, shape := #[1, 1, channels] } := sliceTyped x 1 j.toUInt64 1
  let cell : Tensor { device := device, shape := #[1, 1, 1] } := sliceTyped row 2 k.toUInt64 1
  Tensor.item (Tensor.reshapeChecked cell #[])

private def scalar2d {maxLen : UInt64}
    (x : Tensor { device := device, shape := #[1, maxLen] }) (j : Nat) : Float :=
  let cell : Tensor { device := device, shape := #[1, 1] } := sliceTyped x 1 j.toUInt64 1
  Tensor.item (Tensor.reshapeChecked cell #[])

private def packMoleculeStateForTransformer {maxLen vocab : UInt64}
    (padToken : Int64)
    (t : Float)
    (state : BranchingState MoleculeAtom)
    (device : Device := Device.CPU) :
    IO (Tensor { device := device, shape := #[1, maxLen, 3] } × Tensor { device := device, shape := #[1, maxLen], dtype := .Int64 } × Tensor { device := device, shape := #[1] } × Tensor { device := device, shape := #[1, maxLen], dtype := .Int64 }) := do
  if state.state.size > maxLen.toNat then
    throw (IO.userError s!"molecule state length {state.state.size} exceeds transformer maxLen {maxLen}")
  let maxLenNat := maxLen.toNat
  let mut coordArr : Array Float := Array.replicate (maxLenNat * 3) 0.0
  let mut labelArr : Array Int64 := Array.replicate maxLenNat padToken
  let mut padArr : Array Int64 := Array.replicate maxLenNat 0
  for j in [:state.state.size] do
    let atom := state.state[j]!
    if atom.label >= vocab.toNat then
      throw (IO.userError s!"molecule label {atom.label} is outside transformer vocab {vocab}")
    let offset := j * 3
    coordArr := coordArr.set! offset atom.coord.x
    coordArr := coordArr.set! (offset + 1) atom.coord.y
    coordArr := coordArr.set! (offset + 2) atom.coord.z
    labelArr := labelArr.set! j (Int64.ofNat atom.label)
    padArr := padArr.set! j 1
  let coord := Tensor.reshapeChecked (floatsDynamic coordArr) #[1, maxLen, 3]
  let label := Tensor.reshapeChecked (Tensor.fromInt64Array labelArr) #[1, maxLen]
  let time := Tensor.reshapeChecked (floatsDynamic #[t]) #[1]
  let padmask := Tensor.reshapeChecked (Tensor.fromInt64Array padArr) #[1, maxLen]
  pure (coord.to device, label.to device, time.to device, padmask.to device)

private def tensorVec3 {maxLen : UInt64}
    (coordPred : Tensor { device := device, shape := #[1, maxLen, 3] }) (j : Nat) : Vec3 :=
  { x := scalar3d coordPred j 0,
    y := scalar3d coordPred j 1,
    z := scalar3d coordPred j 2 }

private def tensorLabelLogits {maxLen vocab : UInt64}
    (labelLogits : Tensor { device := device, shape := #[1, maxLen, vocab] }) (j : Nat) : Array Float := Id.run do
  let mut out : Array Float := #[]
  for k in [:vocab.toNat] do
    out := out.push (scalar3d labelLogits j k)
  out

def moleculeTransformerPrediction {maxLen vocab heads headDim mlp : UInt64}
    (padToken : Int64)
    (params : MoleculeTransformerParams (device := device) vocab heads headDim mlp)
    (t : Float)
    (state : BranchingState MoleculeAtom)
    (splitLogitCap? : Option Float := none)
    (coordTargetCap? : Option Float := none)
    :
    IO MoleculeModelPrediction := do
  let params := TensorStruct.detach params
  let (coord, label, time, padmask) ←
    packMoleculeStateForTransformer (maxLen := maxLen) (vocab := vocab) padToken t state device
  let (coordPred, labelLogits, splitLogits, delLogits) ←
    MoleculeTransformerParams.forward (batch := 1) params coord label time padmask coordTargetCap?
  let n := state.state.size
  let canSplit := n < maxLen.toNat
  let coordTargets := (Array.range n).map (fun j => tensorVec3 coordPred j)
  let labelTargets := (Array.range n).map (fun j => tensorLabelLogits labelLogits j)
  let splitTargets := (Array.range n).map (fun j =>
    if canSplit then clampUpper? splitLogitCap? (scalar2d splitLogits j) else -100.0)
  let delTargets := (Array.range n).map (fun j => scalar2d delLogits j)
  pure {
    coordTargets
    labelLogits := labelTargets
    splitLogits := splitTargets
    delLogits := delTargets
  }

def moleculeTransformerIOModel {maxLen vocab heads headDim mlp : UInt64}
    (padToken : Int64)
    (params : MoleculeTransformerParams (device := device) vocab heads headDim mlp)
    (splitLogitCap? : Option Float := none)
    (coordTargetCap? : Option Float := none)
    :
    Float → BranchingState MoleculeAtom → IO MoleculeModelPrediction :=
  fun t state =>
    moleculeTransformerPrediction (maxLen := maxLen) (vocab := vocab)
      (heads := heads) (headDim := headDim) (mlp := mlp)
      padToken params t state (splitLogitCap? := splitLogitCap?)
      (coordTargetCap? := coordTargetCap?) (device := device)

def fullMoleculeTransformerPrediction
    {maxLen vocab hidden heads headDim mlp rff layers : UInt64}
    (padToken : Int64)
    (params : FullMoleculeTransformerParams (device := device) vocab hidden heads headDim mlp rff layers)
    (t : Float)
    (state : BranchingState MoleculeAtom)
    (coordUpdateLayers : Nat := 6)
    (splitLogitCap? : Option Float := none)
    (coordTargetCap? : Option Float := none)
    :
    IO MoleculeModelPrediction := do
  let params := TensorStruct.detach params
  let (coord, label, time, padmask) ←
    packMoleculeStateForTransformer (maxLen := maxLen) (vocab := vocab) padToken t state device
  let (coordPred, labelLogits, splitLogits, delLogits) ←
    FullMoleculeTransformerParams.forward (batch := 1) params coord label time padmask
      (coordUpdateLayers := coordUpdateLayers) (coordTargetCap? := coordTargetCap?)
  let n := state.state.size
  let canSplit := n < maxLen.toNat
  let coordTargets := (Array.range n).map (fun j => tensorVec3 coordPred j)
  let labelTargets := (Array.range n).map (fun j => tensorLabelLogits labelLogits j)
  let splitTargets := (Array.range n).map (fun j =>
    if canSplit then clampUpper? splitLogitCap? (scalar2d splitLogits j) else -100.0)
  let delTargets := (Array.range n).map (fun j => scalar2d delLogits j)
  pure {
    coordTargets
    labelLogits := labelTargets
    splitLogits := splitTargets
    delLogits := delTargets
  }

def fullMoleculeTransformerIOModel
    {maxLen vocab hidden heads headDim mlp rff layers : UInt64}
    (padToken : Int64)
    (params : FullMoleculeTransformerParams (device := device) vocab hidden heads headDim mlp rff layers)
    (coordUpdateLayers : Nat := 6)
    (splitLogitCap? : Option Float := none)
    (coordTargetCap? : Option Float := none)
    :
    Float → BranchingState MoleculeAtom → IO MoleculeModelPrediction :=
  fun t state =>
    fullMoleculeTransformerPrediction (maxLen := maxLen) (vocab := vocab)
      (hidden := hidden) (heads := heads) (headDim := headDim) (mlp := mlp)
      (rff := rff) (layers := layers)
      padToken params t state (coordUpdateLayers := coordUpdateLayers)
      (splitLogitCap? := splitLogitCap?) (coordTargetCap? := coordTargetCap?)
      (device := device)

end torch.branching
