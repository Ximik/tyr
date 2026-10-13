import Tyr.Model.BranchingFlows.MoleculeTrain

/-!
  A small trainable molecule transformer for BranchingFlows batches.

  The paper-scale QM9 model stacks many transformer layers with richer spatial
  features.  This module keeps the same training boundary: coordinates, labels,
  time, padding mask, and heads for coordinate endpoints, label logits, split
  logits, and deletion logits.
-/

namespace torch.branching

open torch
open torch.nn

structure MoleculeTransformerParams (vocab heads headDim mlp : UInt64) where
  coordW : T #[heads * headDim, 3]
  coordB : T #[heads * headDim]
  labelEmbed : T #[vocab, heads * headDim]
  timeW : T #[heads * headDim, 1]
  timeB : T #[heads * headDim]
  attnLnWeight : T #[heads * headDim]
  attnLnBias : T #[heads * headDim]
  qW : T #[heads * headDim, heads * headDim]
  qB : T #[heads * headDim]
  kW : T #[heads * headDim, heads * headDim]
  kB : T #[heads * headDim]
  vW : T #[heads * headDim, heads * headDim]
  vB : T #[heads * headDim]
  oW : T #[heads * headDim, heads * headDim]
  oB : T #[heads * headDim]
  pairDistSlope : T #[heads]
  mlpLnWeight : T #[heads * headDim]
  mlpLnBias : T #[heads * headDim]
  fc1W : T #[mlp, heads * headDim]
  fc1B : T #[mlp]
  fc2W : T #[heads * headDim, mlp]
  fc2B : T #[heads * headDim]
  coordHeadW : T #[3, heads * headDim]
  coordHeadB : T #[3]
  labelHeadW : T #[vocab, heads * headDim]
  labelHeadB : T #[vocab]
  splitHeadW : T #[1, heads * headDim]
  splitHeadB : T #[1]
  delHeadW : T #[1, heads * headDim]
  delHeadB : T #[1]
  deriving TensorStruct

namespace MoleculeTransformerParams

private def boundCoordTarget {s : Shape} (coord : T s) (cap? : Option Float) : T s :=
  match cap? with
  | some cap =>
      if cap > 0.0 then
        (nn.tanh (coord / cap)) * cap
      else
        coord
  | none => coord

private def randnScaled (shape : Shape) (scale : Float) : IO (T shape) := do
  let x ← torch.randn shape
  pure (x * scale)

def init (vocab heads headDim mlp : UInt64) : IO (MoleculeTransformerParams vocab heads headDim mlp) := do
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
    coordB := torch.zeros #[heads * headDim],
    labelEmbed,
    timeW,
    timeB := torch.zeros #[heads * headDim],
    attnLnWeight := torch.ones #[heads * headDim],
    attnLnBias := torch.zeros #[heads * headDim],
    qW,
    qB := torch.zeros #[heads * headDim],
    kW,
    kB := torch.zeros #[heads * headDim],
    vW,
    vB := torch.zeros #[heads * headDim],
    oW,
    oB := torch.zeros #[heads * headDim],
    pairDistSlope := torch.full #[heads] (-4.0),
    mlpLnWeight := torch.ones #[heads * headDim],
    mlpLnBias := torch.zeros #[heads * headDim],
    fc1W,
    fc1B := torch.zeros #[mlp],
    fc2W,
    fc2B := torch.zeros #[heads * headDim],
    coordHeadW,
    coordHeadB := torch.zeros #[3],
    labelHeadW,
    labelHeadB := torch.zeros #[vocab],
    splitHeadW,
    splitHeadB := torch.zeros #[1],
    delHeadW,
    delHeadB := torch.zeros #[1]
  }

private def pairwiseDistanceSq {batch maxLen : UInt64}
    (coord : T #[batch, maxLen, 3]) : T #[batch, maxLen, maxLen] :=
  let ci0 : T #[batch, maxLen, 1, 3] := nn.unsqueeze coord 2
  let cj0 : T #[batch, 1, maxLen, 3] := nn.unsqueeze coord 1
  let ci : T #[batch, maxLen, maxLen, 3] := nn.expand ci0 #[batch, maxLen, maxLen, 3]
  let cj : T #[batch, maxLen, maxLen, 3] := nn.expand cj0 #[batch, maxLen, maxLen, 3]
  let diff := ci - cj
  nn.sumDim (diff * diff) 3 false

private def spatialAttention {batch maxLen vocab heads headDim mlp : UInt64}
    (params : MoleculeTransformerParams vocab heads headDim mlp)
    (coord : T #[batch, maxLen, 3])
    (padmask : T #[batch, maxLen])
    (h : T #[batch, maxLen, heads * headDim]) : T #[batch, maxLen, heads * headDim] :=
  let q0 : T #[batch, maxLen, heads * headDim] := torch.affine3d h params.qW params.qB
  let k0 : T #[batch, maxLen, heads * headDim] := torch.affine3d h params.kW params.kB
  let v0 : T #[batch, maxLen, heads * headDim] := torch.affine3d h params.vW params.vB
  let q : T #[batch, heads, maxLen, headDim] :=
    nn.transpose_for_attention (reshape q0 #[batch, maxLen, heads, headDim])
  let k : T #[batch, heads, maxLen, headDim] :=
    nn.transpose_for_attention (reshape k0 #[batch, maxLen, heads, headDim])
  let v : T #[batch, heads, maxLen, headDim] :=
    nn.transpose_for_attention (reshape v0 #[batch, maxLen, heads, headDim])
  let dist : T #[batch, maxLen, maxLen] := pairwiseDistanceSq coord
  let slope : T #[heads] := nn.softplus params.pairDistSlope
  let distBias : T #[batch, heads, maxLen, maxLen] :=
    nn.expand (reshape slope #[1, heads, 1, 1]) #[batch, heads, maxLen, maxLen] *
      nn.expand (reshape dist #[batch, 1, maxLen, maxLen]) #[batch, heads, maxLen, maxLen]
  let keyMask0 : T #[batch, 1, 1, maxLen] := nn.unsqueeze (nn.unsqueeze padmask 1) 1
  -- (1 - keyMask) · -1e9, computed on-device (a `torch.zeros`-based masked_fill
  -- would be CPU-resident and crash on CUDA inputs).
  let maskTerm : T #[batch, 1, 1, maxLen] :=
    torch.mul_scalar (torch.add_scalar (torch.mul_scalar keyMask0 (-1.0)) 1.0) (-1.0e9)
  let bias : T #[batch, heads, maxLen, maxLen] :=
    nn.expand maskTerm #[batch, heads, maxLen, maxLen] - distBias
  let ctx : T #[batch, heads, maxLen, headDim] :=
    nn.scaled_dot_product_attention_bias q k v bias
  let out4 : T #[batch, maxLen, heads, headDim] := nn.transpose_from_attention ctx
  let out3 : T #[batch, maxLen, heads * headDim] := reshape out4 #[batch, maxLen, heads * headDim]
  torch.affine3d out3 params.oW params.oB

def forward {batch maxLen vocab heads headDim mlp : UInt64}
    (params : MoleculeTransformerParams vocab heads headDim mlp)
    (coord : T #[batch, maxLen, 3])
    (label : T #[batch, maxLen])
    (t : T #[batch])
    (padmask : T #[batch, maxLen])
    (coordTargetCap? : Option Float := none) :
    IO (T #[batch, maxLen, 3] × T #[batch, maxLen, vocab] × T #[batch, maxLen] × T #[batch, maxLen]) := do
  let coordEmb : T #[batch, maxLen, heads * headDim] := torch.affine3d coord params.coordW params.coordB
  let labelEmb : T #[batch, maxLen, heads * headDim] :=
    nn.embedding (batch := batch) (seq := maxLen) (vocab := vocab) (embed := heads * headDim)
      label params.labelEmbed
  let time0 : T #[batch, 1, 1] := nn.unsqueeze (nn.unsqueeze t 1) 1
  let timeIn : T #[batch, maxLen, 1] := nn.expand time0 #[batch, maxLen, 1]
  let timeEmb : T #[batch, maxLen, heads * headDim] := torch.affine3d timeIn params.timeW params.timeB
  let h0 := coordEmb + labelEmb + timeEmb
  let h1n := nn.layer_norm h0 params.attnLnWeight params.attnLnBias 1.0e-5
  let attnOut := spatialAttention params coord padmask h1n
  let h1 := h0 + attnOut
  let h2n := nn.layer_norm h1 params.mlpLnWeight params.mlpLnBias 1.0e-5
  let mlp1 := nn.gelu (torch.affine3d h2n params.fc1W params.fc1B)
  let mlp2 := torch.affine3d mlp1 params.fc2W params.fc2B
  let h := h1 + mlp2
  let coordRaw : T #[batch, maxLen, 3] := torch.affine3d h params.coordHeadW params.coordHeadB
  let coordPred := boundCoordTarget coordRaw coordTargetCap?
  let labelLogits : T #[batch, maxLen, vocab] := torch.affine3d h params.labelHeadW params.labelHeadB
  let split3 : T #[batch, maxLen, 1] := torch.affine3d h params.splitHeadW params.splitHeadB
  let del3 : T #[batch, maxLen, 1] := torch.affine3d h params.delHeadW params.delHeadB
  let splitLogits : T #[batch, maxLen] := reshape split3 #[batch, maxLen]
  let delLogits : T #[batch, maxLen] := reshape del3 #[batch, maxLen]
  pure (coordPred, labelLogits, splitLogits, delLogits)

end MoleculeTransformerParams

def moleculeTransformerModel {maxLen vocab heads headDim mlp : UInt64}
    (coordTargetCap? : Option Float := none) :
    BranchingMoleculeModel maxLen vocab (MoleculeTransformerParams vocab heads headDim mlp) :=
  { forward := fun {batch} params coord label t padmask =>
      MoleculeTransformerParams.forward (batch := batch) params coord label t padmask coordTargetCap? }

structure FullMoleculeTransformerLayerParams
    (hidden heads headDim mlp rff : UInt64) where
  attnLnWeight : T #[hidden]
  attnLnBias : T #[hidden]
  qW : T #[heads * headDim, hidden]
  qB : T #[heads * headDim]
  kW : T #[heads * headDim, hidden]
  kB : T #[heads * headDim]
  vW : T #[heads * headDim, hidden]
  vB : T #[heads * headDim]
  oW : T #[hidden, heads * headDim]
  oB : T #[hidden]
  pairW : T #[heads, rff * 2]
  pairB : T #[heads]
  mlpLnWeight : T #[hidden]
  mlpLnBias : T #[hidden]
  fc1W : T #[mlp, hidden]
  fc1B : T #[mlp]
  fc2W : T #[hidden, mlp]
  fc2B : T #[hidden]
  coordLnWeight : T #[hidden]
  coordLnBias : T #[hidden]
  coordUpdateW : T #[3, hidden]
  coordUpdateB : T #[3]
  deriving TensorStruct

structure FullMoleculeTransformerParams
    (vocab hidden heads headDim mlp rff layers : UInt64) where
  coordRffW : T #[rff, 3]
  coordRffB : T #[rff]
  coordProjW : T #[hidden, rff * 2]
  coordProjB : T #[hidden]
  pairRffW : T #[rff, 1]
  pairRffB : T #[rff]
  labelEmbed : T #[vocab, hidden]
  timeW : T #[hidden, 1]
  timeB : T #[hidden]
  blocks : Array (FullMoleculeTransformerLayerParams hidden heads headDim mlp rff)
  finalLnWeight : T #[hidden]
  finalLnBias : T #[hidden]
  coordHeadW : T #[3, hidden]
  coordHeadB : T #[3]
  labelHeadW : T #[vocab, hidden]
  labelHeadB : T #[vocab]
  splitHeadW : T #[1, hidden]
  splitHeadB : T #[1]
  delHeadW : T #[1, hidden]
  delHeadB : T #[1]
  deriving TensorStruct

namespace FullMoleculeTransformerLayerParams

private def randnScaled (shape : Shape) (scale : Float) : IO (T shape) := do
  let x ← torch.randn shape
  pure (x * scale)

def init (hidden heads headDim mlp rff : UInt64) :
    IO (FullMoleculeTransformerLayerParams hidden heads headDim mlp rff) := do
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
    attnLnWeight := torch.ones #[hidden],
    attnLnBias := torch.zeros #[hidden],
    qW,
    qB := torch.zeros #[heads * headDim],
    kW,
    kB := torch.zeros #[heads * headDim],
    vW,
    vB := torch.zeros #[heads * headDim],
    oW,
    oB := torch.zeros #[hidden],
    pairW,
    pairB := torch.zeros #[heads],
    mlpLnWeight := torch.ones #[hidden],
    mlpLnBias := torch.zeros #[hidden],
    fc1W,
    fc1B := torch.zeros #[mlp],
    fc2W,
    fc2B := torch.zeros #[hidden],
    coordLnWeight := torch.ones #[hidden],
    coordLnBias := torch.zeros #[hidden],
    coordUpdateW,
    coordUpdateB := torch.zeros #[3]
  }

end FullMoleculeTransformerLayerParams

namespace FullMoleculeTransformerParams

private def boundCoordTarget {s : Shape} (coord : T s) (cap? : Option Float) : T s :=
  match cap? with
  | some cap =>
      if cap > 0.0 then
        (nn.tanh (coord / cap)) * cap
      else
        coord
  | none => coord

private def randnScaled (shape : Shape) (scale : Float) : IO (T shape) := do
  let x ← torch.randn shape
  pure (x * scale)

def init (vocab hidden heads headDim mlp rff layers : UInt64) :
    IO (FullMoleculeTransformerParams vocab hidden heads headDim mlp rff layers) := do
  let coordRffW ← randnScaled #[rff, 3] 1.0
  let coordRffB ← randnScaled #[rff] 1.0
  let coordProjW ← randnScaled #[hidden, rff * 2] 0.02
  let pairRffW ← randnScaled #[rff, 1] 1.0
  let pairRffB ← randnScaled #[rff] 1.0
  let labelEmbed ← randnScaled #[vocab, hidden] 0.02
  let timeW ← randnScaled #[hidden, 1] 0.02
  let mut blocks := #[]
  for _ in [:layers.toNat] do
    blocks := blocks.push (← FullMoleculeTransformerLayerParams.init hidden heads headDim mlp rff)
  let coordHeadW ← randnScaled #[3, hidden] 0.02
  let labelHeadW ← randnScaled #[vocab, hidden] 0.02
  let splitHeadW ← randnScaled #[1, hidden] 0.02
  let delHeadW ← randnScaled #[1, hidden] 0.02
  pure {
    coordRffW,
    coordRffB,
    coordProjW,
    coordProjB := torch.zeros #[hidden],
    pairRffW,
    pairRffB,
    labelEmbed,
    timeW,
    timeB := torch.zeros #[hidden],
    blocks,
    finalLnWeight := torch.ones #[hidden],
    finalLnBias := torch.zeros #[hidden],
    coordHeadW,
    coordHeadB := torch.zeros #[3],
    labelHeadW,
    labelHeadB := torch.zeros #[vocab],
    splitHeadW,
    splitHeadB := torch.zeros #[1],
    delHeadW,
    delHeadB := torch.zeros #[1]
  }

private def pairwiseDistanceSq {batch maxLen : UInt64}
    (coord : T #[batch, maxLen, 3]) : T #[batch, maxLen, maxLen] :=
  let ci0 : T #[batch, maxLen, 1, 3] := nn.unsqueeze coord 2
  let cj0 : T #[batch, 1, maxLen, 3] := nn.unsqueeze coord 1
  let ci : T #[batch, maxLen, maxLen, 3] := nn.expand ci0 #[batch, maxLen, maxLen, 3]
  let cj : T #[batch, maxLen, maxLen, 3] := nn.expand cj0 #[batch, maxLen, maxLen, 3]
  let diff := ci - cj
  nn.sumDim (diff * diff) 3 false

private def coordRffEmbedding {batch maxLen vocab hidden heads headDim mlp rff layers : UInt64}
    (params : FullMoleculeTransformerParams vocab hidden heads headDim mlp rff layers)
    (coord : T #[batch, maxLen, 3]) : T #[batch, maxLen, hidden] :=
  let flat : T #[batch * maxLen, 3] := reshape coord #[batch * maxLen, 3]
  let phase : T #[batch * maxLen, rff] := torch.affine flat params.coordRffW params.coordRffB
  let featFlat : T #[batch * maxLen, rff * 2] := nn.cat (nn.sin phase) (nn.cos phase) 1
  let features : T #[batch, maxLen, rff * 2] := reshape featFlat #[batch, maxLen, rff * 2]
  torch.affine3d features params.coordProjW params.coordProjB

private def pairRffFeatures {batch maxLen vocab hidden heads headDim mlp rff layers : UInt64}
    (params : FullMoleculeTransformerParams vocab hidden heads headDim mlp rff layers)
    (coord : T #[batch, maxLen, 3]) : T #[batch * maxLen * maxLen, rff * 2] :=
  let dist : T #[batch, maxLen, maxLen] := pairwiseDistanceSq coord
  let distFlat : T #[batch * maxLen * maxLen, 1] := reshape dist #[batch * maxLen * maxLen, 1]
  let phase : T #[batch * maxLen * maxLen, rff] := torch.affine distFlat params.pairRffW params.pairRffB
  nn.cat (nn.sin phase) (nn.cos phase) 1

private def pairAttentionBiasFrom {batch maxLen hidden heads headDim mlp rff : UInt64}
    (layer : FullMoleculeTransformerLayerParams hidden heads headDim mlp rff)
    (features : T #[batch * maxLen * maxLen, rff * 2]) : T #[batch, heads, maxLen, maxLen] :=
  let biasFlat : T #[batch * maxLen * maxLen, heads] := torch.affine features layer.pairW layer.pairB
  let bias0 : T #[batch, maxLen, maxLen, heads] := reshape biasFlat #[batch, maxLen, maxLen, heads]
  let bias1 : T #[batch, maxLen, heads, maxLen] := nn.transpose bias0 2 3
  nn.transpose bias1 1 2

private def spatialAttention {batch maxLen vocab hidden heads headDim mlp rff layers : UInt64}
    (_params : FullMoleculeTransformerParams vocab hidden heads headDim mlp rff layers)
    (layer : FullMoleculeTransformerLayerParams hidden heads headDim mlp rff)
    (pairFeatures : T #[batch * maxLen * maxLen, rff * 2])
    (padmask : T #[batch, maxLen])
    (ropeCos : T #[maxLen, headDim / 2])
    (ropeSin : T #[maxLen, headDim / 2])
    (h : T #[batch, maxLen, hidden]) : T #[batch, maxLen, hidden] :=
  let q0 : T #[batch, maxLen, heads * headDim] := torch.affine3d h layer.qW layer.qB
  let k0 : T #[batch, maxLen, heads * headDim] := torch.affine3d h layer.kW layer.kB
  let v0 : T #[batch, maxLen, heads * headDim] := torch.affine3d h layer.vW layer.vB
  let qR : T #[batch, maxLen, heads, headDim] :=
    rotary.applyRotaryEmb (reshape q0 #[batch, maxLen, heads, headDim])
      (castLike q0 ropeCos) (castLike q0 ropeSin)
  let kR : T #[batch, maxLen, heads, headDim] :=
    rotary.applyRotaryEmb (reshape k0 #[batch, maxLen, heads, headDim])
      (castLike k0 ropeCos) (castLike k0 ropeSin)
  let q : T #[batch, heads, maxLen, headDim] := nn.transpose_for_attention qR
  let k : T #[batch, heads, maxLen, headDim] := nn.transpose_for_attention kR
  let v : T #[batch, heads, maxLen, headDim] :=
    nn.transpose_for_attention (reshape v0 #[batch, maxLen, heads, headDim])
  let pairBias : T #[batch, heads, maxLen, maxLen] := pairAttentionBiasFrom layer pairFeatures
  let keyMask0 : T #[batch, 1, 1, maxLen] := nn.unsqueeze (nn.unsqueeze padmask 1) 1
  -- (1 - keyMask) · -1e9, computed on-device (a `torch.zeros`-based masked_fill
  -- would be CPU-resident and crash on CUDA inputs).
  let maskTerm : T #[batch, 1, 1, maxLen] :=
    torch.mul_scalar (torch.add_scalar (torch.mul_scalar keyMask0 (-1.0)) 1.0) (-1.0e9)
  let bias : T #[batch, heads, maxLen, maxLen] :=
    pairBias + nn.expand maskTerm #[batch, heads, maxLen, maxLen]
  let ctx : T #[batch, heads, maxLen, headDim] :=
    nn.scaled_dot_product_attention_bias q k v bias
  let out4 : T #[batch, maxLen, heads, headDim] := nn.transpose_from_attention ctx
  let out3 : T #[batch, maxLen, heads * headDim] := reshape out4 #[batch, maxLen, heads * headDim]
  torch.affine3d out3 layer.oW layer.oB

private def forwardLayer {batch maxLen vocab hidden heads headDim mlp rff layers : UInt64}
    (params : FullMoleculeTransformerParams vocab hidden heads headDim mlp rff layers)
    (layer : FullMoleculeTransformerLayerParams hidden heads headDim mlp rff)
    (applyCoordUpdate : Bool)
    (pairFeatures : T #[batch * maxLen * maxLen, rff * 2])
    (ropeCos : T #[maxLen, headDim / 2])
    (ropeSin : T #[maxLen, headDim / 2])
    (padmask : T #[batch, maxLen])
    (h : T #[batch, maxLen, hidden])
    (coord : T #[batch, maxLen, 3]) :
    T #[batch, maxLen, hidden] × T #[batch, maxLen, 3] :=
  let h1n := nn.layer_norm h layer.attnLnWeight layer.attnLnBias 1.0e-5
  let attnOut := spatialAttention params layer pairFeatures padmask ropeCos ropeSin h1n
  let h1 := h + attnOut
  let h2n := nn.layer_norm h1 layer.mlpLnWeight layer.mlpLnBias 1.0e-5
  let mlp1 := nn.gelu (torch.affine3d h2n layer.fc1W layer.fc1B)
  let mlp2 := torch.affine3d mlp1 layer.fc2W layer.fc2B
  let h2 := h1 + mlp2
  if applyCoordUpdate then
    let coordN := nn.layer_norm h2 layer.coordLnWeight layer.coordLnBias 1.0e-5
    let delta : T #[batch, maxLen, 3] := torch.affine3d coordN layer.coordUpdateW layer.coordUpdateB
    (h2, coord + delta)
  else
    (h2, coord)

/-- Forward with precomputed rotary frequencies. Split out so CUDA-graph
    capture can hoist `rotary.computeFreqs` (which does a host→device copy,
    illegal during stream capture) out of the captured region. -/
def forwardFreqs {batch maxLen vocab hidden heads headDim mlp rff layers : UInt64}
    (params : FullMoleculeTransformerParams vocab hidden heads headDim mlp rff layers)
    (coord : T #[batch, maxLen, 3])
    (label : T #[batch, maxLen])
    (t : T #[batch])
    (padmask : T #[batch, maxLen])
    (ropeCos : T #[maxLen, headDim / 2])
    (ropeSin : T #[maxLen, headDim / 2])
    (coordUpdateLayers : Nat := 6)
    (coordTargetCap? : Option Float := none) :
    IO (T #[batch, maxLen, 3] × T #[batch, maxLen, vocab] × T #[batch, maxLen] × T #[batch, maxLen]) := do
  let coordEmb : T #[batch, maxLen, hidden] := coordRffEmbedding params coord
  let labelEmb : T #[batch, maxLen, hidden] :=
    nn.embedding (batch := batch) (seq := maxLen) (vocab := vocab) (embed := hidden)
      label params.labelEmbed
  let time0 : T #[batch, 1, 1] := nn.unsqueeze (nn.unsqueeze t 1) 1
  let timeIn : T #[batch, maxLen, 1] := nn.expand time0 #[batch, maxLen, 1]
  let timeEmb : T #[batch, maxLen, hidden] := torch.affine3d timeIn params.timeW params.timeB
  let mut h := coordEmb + labelEmb + timeEmb
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
  let hFinal := nn.layer_norm h params.finalLnWeight params.finalLnBias 1.0e-5
  let coordDelta : T #[batch, maxLen, 3] := torch.affine3d hFinal params.coordHeadW params.coordHeadB
  let coordPred := boundCoordTarget (runningCoord + coordDelta) coordTargetCap?
  let labelLogits : T #[batch, maxLen, vocab] := torch.affine3d hFinal params.labelHeadW params.labelHeadB
  let split3 : T #[batch, maxLen, 1] := torch.affine3d hFinal params.splitHeadW params.splitHeadB
  let del3 : T #[batch, maxLen, 1] := torch.affine3d hFinal params.delHeadW params.delHeadB
  let splitLogits : T #[batch, maxLen] := reshape split3 #[batch, maxLen]
  let delLogits : T #[batch, maxLen] := reshape del3 #[batch, maxLen]
  pure (coordPred, labelLogits, splitLogits, delLogits)

/-- Forward computing rotary frequencies on the fly (one host→device copy per
    call); see `forwardFreqs` for the graph-safe variant. -/
def forward {batch maxLen vocab hidden heads headDim mlp rff layers : UInt64}
    (params : FullMoleculeTransformerParams vocab hidden heads headDim mlp rff layers)
    (coord : T #[batch, maxLen, 3])
    (label : T #[batch, maxLen])
    (t : T #[batch])
    (padmask : T #[batch, maxLen])
    (coordUpdateLayers : Nat := 6)
    (coordTargetCap? : Option Float := none) :
    IO (T #[batch, maxLen, 3] × T #[batch, maxLen, vocab] × T #[batch, maxLen] × T #[batch, maxLen]) := do
  let (ropeCos, ropeSin) ← rotary.computeFreqs maxLen headDim 10000.0
  forwardFreqs params coord label t padmask ropeCos ropeSin
    (coordUpdateLayers := coordUpdateLayers) (coordTargetCap? := coordTargetCap?)

end FullMoleculeTransformerParams

def fullMoleculeTransformerModel {maxLen vocab hidden heads headDim mlp rff layers : UInt64}
    (coordUpdateLayers : Nat := 6)
    (coordTargetCap? : Option Float := none) :
    BranchingMoleculeModel maxLen vocab
      (FullMoleculeTransformerParams vocab hidden heads headDim mlp rff layers) :=
  { forward := fun {batch} params coord label t padmask =>
      FullMoleculeTransformerParams.forward (batch := batch) params coord label t padmask
        (coordUpdateLayers := coordUpdateLayers) (coordTargetCap? := coordTargetCap?) }

private def clampUpper? (cap? : Option Float) (x : Float) : Float :=
  match cap? with
  | some cap => min x cap
  | none => x

private def scalar3d {maxLen channels : UInt64}
    (x : T #[1, maxLen, channels]) (j k : Nat) : Float :=
  let row : T #[1, 1, channels] := data.slice x 1 j.toUInt64 1
  let cell : T #[1, 1, 1] := data.slice row 2 k.toUInt64 1
  nn.item (reshape cell #[])

private def scalar2d {maxLen : UInt64}
    (x : T #[1, maxLen]) (j : Nat) : Float :=
  let cell : T #[1, 1] := data.slice x 1 j.toUInt64 1
  nn.item (reshape cell #[])

private def packMoleculeStateForTransformer {maxLen vocab : UInt64}
    (padToken : Int64)
    (t : Float)
    (state : BranchingState MoleculeAtom)
    (device : Device := Device.CPU) :
    IO (T #[1, maxLen, 3] × T #[1, maxLen] × T #[1] × T #[1, maxLen]) := do
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
  let coord := reshape (data.fromFloatArray coordArr) #[1, maxLen, 3]
  let label := reshape (data.fromInt64Array labelArr) #[1, maxLen]
  let time := reshape (data.fromFloatArray #[t]) #[1]
  let padmask := toFloat' (reshape (data.fromInt64Array padArr) #[1, maxLen])
  pure (coord.to device, label.to device, time.to device, padmask.to device)

private def tensorVec3 {maxLen : UInt64}
    (coordPred : T #[1, maxLen, 3]) (j : Nat) : Vec3 :=
  { x := scalar3d coordPred j 0,
    y := scalar3d coordPred j 1,
    z := scalar3d coordPred j 2 }

private def tensorLabelLogits {maxLen vocab : UInt64}
    (labelLogits : T #[1, maxLen, vocab]) (j : Nat) : Array Float := Id.run do
  let mut out : Array Float := #[]
  for k in [:vocab.toNat] do
    out := out.push (scalar3d labelLogits j k)
  out

def moleculeTransformerPrediction {maxLen vocab heads headDim mlp : UInt64}
    (padToken : Int64)
    (params : MoleculeTransformerParams vocab heads headDim mlp)
    (t : Float)
    (state : BranchingState MoleculeAtom)
    (splitLogitCap? : Option Float := none)
    (coordTargetCap? : Option Float := none)
    (device : Device := Device.CPU) :
    IO MoleculeModelPrediction := do
  torch.autograd.no_grad do
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
    (params : MoleculeTransformerParams vocab heads headDim mlp)
    (splitLogitCap? : Option Float := none)
    (coordTargetCap? : Option Float := none)
    (device : Device := Device.CPU) :
    Float → BranchingState MoleculeAtom → IO MoleculeModelPrediction :=
  fun t state =>
    moleculeTransformerPrediction (maxLen := maxLen) (vocab := vocab)
      (heads := heads) (headDim := headDim) (mlp := mlp)
      padToken params t state (splitLogitCap? := splitLogitCap?)
      (coordTargetCap? := coordTargetCap?) (device := device)

def fullMoleculeTransformerPrediction
    {maxLen vocab hidden heads headDim mlp rff layers : UInt64}
    (padToken : Int64)
    (params : FullMoleculeTransformerParams vocab hidden heads headDim mlp rff layers)
    (t : Float)
    (state : BranchingState MoleculeAtom)
    (coordUpdateLayers : Nat := 6)
    (splitLogitCap? : Option Float := none)
    (coordTargetCap? : Option Float := none)
    (device : Device := Device.CPU) :
    IO MoleculeModelPrediction := do
  torch.autograd.no_grad do
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
    (params : FullMoleculeTransformerParams vocab hidden heads headDim mlp rff layers)
    (coordUpdateLayers : Nat := 6)
    (splitLogitCap? : Option Float := none)
    (coordTargetCap? : Option Float := none)
    (device : Device := Device.CPU) :
    Float → BranchingState MoleculeAtom → IO MoleculeModelPrediction :=
  fun t state =>
    fullMoleculeTransformerPrediction (maxLen := maxLen) (vocab := vocab)
      (hidden := hidden) (heads := heads) (headDim := headDim) (mlp := mlp)
      (rff := rff) (layers := layers)
      padToken params t state (coordUpdateLayers := coordUpdateLayers)
      (splitLogitCap? := splitLogitCap?) (coordTargetCap? := coordTargetCap?)
      (device := device)

end torch.branching
