/-
  Tyr/Model/Qwen3ASR/AudioEncoder.lean

  Lean4 port of Qwen3-ASR audio encoder:
  - 3x Conv2d downsampler
  - sinusoidal absolute position embedding
  - transformer encoder stack
  - output projection to thinker hidden space
-/
import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Module.LayerNorm
import Tyr.Model.Qwen3ASR.Config
import Tyr.Model.Utils

namespace torch.qwen3asr

open torch.Model

private def initWeight (shape : Shape) (fanIn : UInt64) : IO (STensor shape) :=
  Model.initWeight shape fanIn (.Unknown "dynamic")

private def initBias (shape : Shape) : STensor shape :=
  Model.initBias shape (.Unknown "dynamic")

private def initLayerNorm (dim : UInt64) (eps : Float := 1e-5) : LayerNorm dim (.Unknown "dynamic") :=
  let norm := LayerNorm.init dim eps
  { weight := norm.weight.forgetDType, bias := norm.bias.forgetDType, eps := norm.eps }


private def activate {s : Shape} (name : String) (x : STensor s) : STensor s :=
  if name == "relu" then
    Tensor.relu x
  else if name == "silu" then
    Tensor.Preserving.silu x
  else
    Tensor.Preserving.gelu x

private def sinusoidPosition {seq dim : UInt64}
    (maxTimescale : Float := 10000.0)
    (device : Device := Device.CPU)
    : STensor #[seq, dim] :=
  if dim % 2 == 0 then
    let half := dim / 2
    let pos := (Tensor.arange seq).toFloat32
    let idx := (Tensor.arange half).toFloat32
    let denom : Float := if half <= 1 then 1.0 else (half - 1).toFloat
    let logInc := Float.log maxTimescale / denom
    let invTimescales := Tensor.exp (idx * (-logInc))
    let scaled := Tensor.mm (pos.reshapeChecked #[seq, 1]) (invTimescales.reshapeChecked #[1, half])
    let out := (Tensor.cat scaled.sin scaled.cos 1).reshapeChecked #[seq, dim]
    (out.to device).forgetDType
  else
    (Tensor.zeros #[seq, dim] false device).forgetDType

/-- Multi-head self-attention used by the audio encoder. -/
structure AudioAttention (cfg : AudioEncoderConfig) where
  qProjWeight : STensor #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg, cfg.dModel]
  qProjBias : STensor #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  kProjWeight : STensor #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg, cfg.dModel]
  kProjBias : STensor #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  vProjWeight : STensor #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg, cfg.dModel]
  vProjBias : STensor #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  outProjWeight : STensor #[cfg.dModel, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  outProjBias : STensor #[cfg.dModel]
  deriving TensorStruct

namespace AudioAttention

def init (cfg : AudioEncoderConfig) : IO (AudioAttention cfg) := do
  let qProjWeight ← initWeight
    #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg, cfg.dModel] cfg.dModel
  let qProjBias := initBias #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  let kProjWeight ← initWeight
    #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg, cfg.dModel] cfg.dModel
  let kProjBias := initBias #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  let vProjWeight ← initWeight
    #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg, cfg.dModel] cfg.dModel
  let vProjBias := initBias #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  let outProjWeight ← initWeight
    #[cfg.dModel, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
    (cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg)
  let outProjBias := initBias #[cfg.dModel]
  pure {
    qProjWeight, qProjBias, kProjWeight, kProjBias, vProjWeight, vProjBias, outProjWeight, outProjBias
  }

def forward {batch seq : UInt64}
    (m : AudioAttention cfg)
    (x : STensor #[batch, seq, cfg.dModel])
    (attnMask : Option (STensor #[batch, seq]) := none)
    : STensor #[batch, seq, cfg.dModel] :=
  let q : STensor #[batch, seq, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg] :=
    Tensor.affine3d x m.qProjWeight m.qProjBias
  let k : STensor #[batch, seq, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg] :=
    Tensor.affine3d x m.kProjWeight m.kProjBias
  let v : STensor #[batch, seq, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg] :=
    Tensor.affine3d x m.vProjWeight m.vProjBias

  let q : STensor #[batch, seq, cfg.encoderAttentionHeads, AudioEncoderConfig.headDim cfg] :=
    Tensor.reshapeChecked q #[batch, seq, cfg.encoderAttentionHeads, AudioEncoderConfig.headDim cfg]
  let k : STensor #[batch, seq, cfg.encoderAttentionHeads, AudioEncoderConfig.headDim cfg] :=
    Tensor.reshapeChecked k #[batch, seq, cfg.encoderAttentionHeads, AudioEncoderConfig.headDim cfg]
  let v : STensor #[batch, seq, cfg.encoderAttentionHeads, AudioEncoderConfig.headDim cfg] :=
    Tensor.reshapeChecked v #[batch, seq, cfg.encoderAttentionHeads, AudioEncoderConfig.headDim cfg]

  let q := Tensor.splitHeads q
  let k := Tensor.splitHeads k
  let v := Tensor.splitHeads v

  let out :=
    match attnMask with
    | some mask =>
      Tensor.attentionGQAMask
        q k v mask cfg.attentionDropout false true
    | none =>
      Tensor.attentionGQA
        q k v cfg.attentionDropout false true

  let out := Tensor.joinHeads out
  let out : STensor #[batch, seq, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg] :=
    Tensor.reshapeChecked out #[batch, seq, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  Tensor.affine3d out m.outProjWeight m.outProjBias

end AudioAttention

/-- One transformer block in the audio encoder. -/
structure AudioEncoderLayer (cfg : AudioEncoderConfig) where
  selfAttn : AudioAttention cfg
  selfAttnLayerNorm : LayerNorm cfg.dModel (.Unknown "dynamic")
  fc1Weight : STensor #[cfg.encoderFfnDim, cfg.dModel]
  fc1Bias : STensor #[cfg.encoderFfnDim]
  fc2Weight : STensor #[cfg.dModel, cfg.encoderFfnDim]
  fc2Bias : STensor #[cfg.dModel]
  finalLayerNorm : LayerNorm cfg.dModel (.Unknown "dynamic")
  deriving TensorStruct

namespace AudioEncoderLayer

def init (cfg : AudioEncoderConfig) : IO (AudioEncoderLayer cfg) := do
  let selfAttn ← AudioAttention.init cfg
  let selfAttnLayerNorm := initLayerNorm cfg.dModel 1e-5
  let fc1Weight ← initWeight #[cfg.encoderFfnDim, cfg.dModel] cfg.dModel
  let fc1Bias := initBias #[cfg.encoderFfnDim]
  let fc2Weight ← initWeight #[cfg.dModel, cfg.encoderFfnDim] cfg.encoderFfnDim
  let fc2Bias := initBias #[cfg.dModel]
  let finalLayerNorm := initLayerNorm cfg.dModel 1e-5
  pure { selfAttn, selfAttnLayerNorm, fc1Weight, fc1Bias, fc2Weight, fc2Bias, finalLayerNorm }

def forward {batch seq : UInt64}
    (m : AudioEncoderLayer cfg)
    (x : STensor #[batch, seq, cfg.dModel])
    (attnMask : Option (STensor #[batch, seq]) := none)
    : STensor #[batch, seq, cfg.dModel] :=
  let residual0 := x
  let h0 := m.selfAttnLayerNorm.forward3d x
  let h1 := m.selfAttn.forward h0 attnMask
  let h2 := residual0 + h1

  let residual1 := h2
  let h3 := m.finalLayerNorm.forward3d h2
  let h4 : STensor #[batch, seq, cfg.encoderFfnDim] := Tensor.affine3d h3 m.fc1Weight m.fc1Bias
  let h5 := activate cfg.activationFunction h4
  let h6 : STensor #[batch, seq, cfg.dModel] := Tensor.affine3d h5 m.fc2Weight m.fc2Bias
  residual1 + h6

end AudioEncoderLayer

/-- Full Qwen3-ASR audio encoder. -/
structure AudioEncoder (cfg : AudioEncoderConfig) where
  conv2d1Weight : STensor #[cfg.downsampleHiddenSize, 1, 3, 3]
  conv2d1Bias : STensor #[cfg.downsampleHiddenSize]
  conv2d2Weight : STensor #[cfg.downsampleHiddenSize, cfg.downsampleHiddenSize, 3, 3]
  conv2d2Bias : STensor #[cfg.downsampleHiddenSize]
  conv2d3Weight : STensor #[cfg.downsampleHiddenSize, cfg.downsampleHiddenSize, 3, 3]
  conv2d3Bias : STensor #[cfg.downsampleHiddenSize]
  convOutWeight : STensor #[cfg.dModel, AudioEncoderConfig.convOutInDim cfg]
  layers : Array (AudioEncoderLayer cfg)
  lnPost : LayerNorm cfg.dModel (.Unknown "dynamic")
  proj1Weight : STensor #[cfg.dModel, cfg.dModel]
  proj1Bias : STensor #[cfg.dModel]
  proj2Weight : STensor #[cfg.outputDim, cfg.dModel]
  proj2Bias : STensor #[cfg.outputDim]
  deriving TensorStruct

namespace AudioEncoder

/-- The frontend computes mel features in Float32. Match the first convolution's
    FP32/BF16 dtype and device before padding or convolution; matching dtypes are
    left unchanged. Positional frequencies are still computed in Float32. -/
private def alignFeatures {batch frames : UInt64} (m : AudioEncoder cfg)
    (features : STensor #[batch, cfg.numMelBins, frames]) : STensor #[batch, cfg.numMelBins, frames] :=
  let features := Tensor.castLike m.conv2d1Weight features
  features.to m.conv2d1Weight.device

def init (cfg : AudioEncoderConfig) : IO (AudioEncoder cfg) := do
  if cfg.dModel % cfg.encoderAttentionHeads != 0 then
    throw <| IO.userError
      s!"Audio encoder requires dModel divisible by encoderAttentionHeads, got {cfg.dModel} and {cfg.encoderAttentionHeads}"

  let conv2d1Weight ← initWeight #[cfg.downsampleHiddenSize, 1, 3, 3] 1
  let conv2d1Bias := initBias #[cfg.downsampleHiddenSize]
  let conv2d2Weight ← initWeight #[cfg.downsampleHiddenSize, cfg.downsampleHiddenSize, 3, 3] cfg.downsampleHiddenSize
  let conv2d2Bias := initBias #[cfg.downsampleHiddenSize]
  let conv2d3Weight ← initWeight #[cfg.downsampleHiddenSize, cfg.downsampleHiddenSize, 3, 3] cfg.downsampleHiddenSize
  let conv2d3Bias := initBias #[cfg.downsampleHiddenSize]

  let convOutWeight ← initWeight #[cfg.dModel, AudioEncoderConfig.convOutInDim cfg] (AudioEncoderConfig.convOutInDim cfg)

  let mut layers : Array (AudioEncoderLayer cfg) := #[]
  for _ in [:cfg.encoderLayers.toNat] do
    let layer ← AudioEncoderLayer.init cfg
    layers := layers.push layer

  let lnPost := initLayerNorm cfg.dModel 1e-5
  let proj1Weight ← initWeight #[cfg.dModel, cfg.dModel] cfg.dModel
  let proj1Bias := initBias #[cfg.dModel]
  let proj2Weight ← initWeight #[cfg.outputDim, cfg.dModel] cfg.dModel
  let proj2Bias := initBias #[cfg.outputDim]

  pure {
    conv2d1Weight, conv2d1Bias, conv2d2Weight, conv2d2Bias, conv2d3Weight, conv2d3Bias,
    convOutWeight, layers, lnPost, proj1Weight, proj1Bias, proj2Weight, proj2Bias
  }

/-- Port of reference chunk planner used before convolutional downsampling.
    For each feature length `L`, emits `ceil(L / (2*n_window))` chunk lengths where
    all chunks are `2*n_window` except the last chunk (`L % (2*n_window)` or full). -/
def buildChunkLengths (cfg : AudioEncoderConfig) (featureLens : Array UInt64) : Array UInt64 :=
  let step := cfg.nWindow * 2
  if step == 0 then
    featureLens
  else
    Id.run do
      let mut out : Array UInt64 := #[]
      for len in featureLens do
        if len == 0 then
          pure ()
        else
          let chunkNum := (len + step - 1) / step
          for j in [:chunkNum.toNat] do
            if j + 1 == chunkNum.toNat then
              let tail := len % step
              out := out.push (if tail == 0 then step else tail)
            else
              out := out.push step
      out

/-- Per-chunk length transform through `_get_feat_extract_output_lengths`. -/
def chunkLengthsAfterCnn (_cfg : AudioEncoderConfig) (chunkLens : Array UInt64) : Array UInt64 :=
  chunkLens.map AudioEncoderConfig.featExtractOutputLength

/-- Port of `cu_chunk_lens` creation in reference audio encoder forward.
    Returns inclusive prefix-sum source with initial `0`. -/
def buildCuChunkLensCumsum
    (cfg : AudioEncoderConfig)
    (afterCnnLens : Array UInt64)
    (paddedAfterCnnWidth : UInt64)
    : Array UInt64 :=
  let denom := cfg.nWindow * 2
  let ratio := if denom == 0 then 1 else cfg.nWindowInfer / denom
  let windowAfterCnn0 := paddedAfterCnnWidth * (if ratio == 0 then 1 else ratio)
  let windowAfterCnn := if windowAfterCnn0 == 0 then 1 else windowAfterCnn0
  Id.run do
    let mut segs : Array UInt64 := #[0]
    for cnnLen in afterCnnLens do
      let full := cnnLen / windowAfterCnn
      for _ in [:full.toNat] do
        segs := segs.push windowAfterCnn
      let rem := cnnLen % windowAfterCnn
      if rem != 0 then
        segs := segs.push rem
    let mut csum : UInt64 := 0
    let mut out : Array UInt64 := #[]
    for s in segs do
      csum := csum + s
      out := out.push csum
    out

private def maxOr (xs : Array UInt64) (fallback : UInt64 := 0) : UInt64 :=
  xs.foldl (fun acc x => if x > acc then x else acc) fallback

private def sumU64 (xs : Array UInt64) : UInt64 :=
  xs.foldl (· + ·) 0

private def chunkLensFromCumsum (cs : Array UInt64) : Array UInt64 :=
  Id.run do
    let mut out : Array UInt64 := #[]
    let mut prev : UInt64 := 0
    for cur in cs do
      if cur >= prev then
        let d := cur - prev
        if d != 0 then
          out := out.push d
        prev := cur
    out

private def padRightFrames {cfg : AudioEncoderConfig} {frames maxFrames : UInt64}
    (x : STensor #[1, cfg.numMelBins, frames])
    : STensor #[1, cfg.numMelBins, maxFrames] :=
  let padFrames := if maxFrames >= frames then maxFrames - frames else 0
  let pad : STensor #[1, cfg.numMelBins, padFrames] :=
    Tensor.castLike x ((Tensor.zeros #[1, cfg.numMelBins, padFrames] false x.device).forgetDType)
  let cat := Tensor.cat x pad 2
  Tensor.reshapeChecked cat #[1, cfg.numMelBins, maxFrames]

private def encodePaddedChunk {maxFrames : UInt64}
    (m : AudioEncoder cfg)
    (chunk : STensor #[1, cfg.numMelBins, maxFrames])
    : STensor #[1, AudioEncoderConfig.framesAfterConv3 cfg maxFrames, cfg.dModel] :=
  let t1 := AudioEncoderConfig.downsampleOnce maxFrames
  let t2 := AudioEncoderConfig.downsampleTwice maxFrames
  let t3 := AudioEncoderConfig.framesAfterConv3 cfg maxFrames
  let x0 : STensor #[1, 1, cfg.numMelBins, maxFrames] := Tensor.reshapeChecked chunk #[1, 1, cfg.numMelBins, maxFrames]
  let x1 : STensor #[1, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv1 cfg, t1] :=
    Tensor.reshapeChecked (Tensor.Preserving.gelu (Tensor.conv2dBias x0 m.conv2d1Weight m.conv2d1Bias #[2, 2] #[1, 1]))
      #[1, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv1 cfg, t1]
  let x2 : STensor #[1, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv2 cfg, t2] :=
    Tensor.reshapeChecked (Tensor.Preserving.gelu (Tensor.conv2dBias x1 m.conv2d2Weight m.conv2d2Bias #[2, 2] #[1, 1]))
      #[1, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv2 cfg, t2]
  let x3 : STensor #[1, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg, t3] :=
    Tensor.reshapeChecked (Tensor.Preserving.gelu (Tensor.conv2dBias x2 m.conv2d3Weight m.conv2d3Bias #[2, 2] #[1, 1]))
      #[1, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg, t3]
  let x3t : STensor #[1, t3, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg] :=
    Tensor.reshapeChecked (Tensor.permute x3 #[0, 3, 1, 2]) #[1, t3, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg]
  let xFlat : STensor #[1, t3, AudioEncoderConfig.convOutInDim cfg] :=
    Tensor.reshapeChecked x3t #[1, t3, AudioEncoderConfig.convOutInDim cfg]
  Tensor.linear3d xFlat m.convOutWeight

private def runLayersChunkLocal {seq : UInt64}
    (m : AudioEncoder cfg)
    (hidden : STensor #[1, seq, cfg.dModel])
    (chunkLens : Array UInt64)
    : STensor #[1, seq, cfg.dModel] :=
  if seq == 0 then
    hidden
  else
    let parts : Array (Sigma STensor) := Id.run do
      let mut out : Array (Sigma STensor) := #[]
      let mut start : UInt64 := 0
      for len in chunkLens do
        if len != 0 then
          let seg : STensor #[1, len, cfg.dModel] := Tensor.sliceChecked hidden 1 start len #[1, len, cfg.dModel]
          let segOut := m.layers.foldl (fun h layer => layer.forward h none) seg
          out := out.push (Tensor.packShape segOut)
          start := start + len
      out
    if parts.isEmpty then
      Tensor.castLike hidden ((Tensor.zeros #[1, seq, cfg.dModel] false hidden.device).forgetDType)
    else
      Tensor.catManyChecked parts 1 #[1, seq, cfg.dModel]

private def encodeOneSampleVarLen {frames : UInt64}
    (m : AudioEncoder cfg)
    (sample : STensor #[1, cfg.numMelBins, frames])
    (featureLen : UInt64)
    : STensor #[1, AudioEncoderConfig.framesAfterConv3 cfg frames, cfg.outputDim] :=
  let outSeq := AudioEncoderConfig.framesAfterConv3 cfg frames
  let featureLen := if featureLen <= frames then featureLen else frames
  if featureLen == 0 then
    Tensor.castLike sample ((Tensor.zeros #[1, outSeq, cfg.outputDim] false sample.device).forgetDType)
  else
    let chunkLens := buildChunkLengths cfg #[featureLen]
    let maxChunkLen := maxOr chunkLens 1
    let tChunk := AudioEncoderConfig.framesAfterConv3 cfg maxChunkLen
    let chunkLensAfterCnn := chunkLengthsAfterCnn cfg chunkLens
    let maxChunkAfterCnn := maxOr chunkLensAfterCnn tChunk
    let pos : STensor #[tChunk, cfg.dModel] :=
      sinusoidPosition (seq := tChunk) (dim := cfg.dModel) (device := sample.device)
    let posBatch : STensor #[1, tChunk, cfg.dModel] := Tensor.reshapeChecked pos #[1, tChunk, cfg.dModel]

    let hiddenParts : Array (Sigma STensor) := Id.run do
      let mut out : Array (Sigma STensor) := #[]
      let mut start : UInt64 := 0
      for i in [:chunkLens.size] do
        let len := chunkLens.getD i 0
        if len != 0 then
          let desiredLen := chunkLensAfterCnn.getD i 0
          let chunk : STensor #[1, cfg.numMelBins, len] := Tensor.sliceChecked sample 2 start len #[1, cfg.numMelBins, len]
          let padded : STensor #[1, cfg.numMelBins, maxChunkLen] :=
            padRightFrames (cfg := cfg) (frames := len) (maxFrames := maxChunkLen) chunk
          let emb : STensor #[1, tChunk, cfg.dModel] := encodePaddedChunk m padded
          let embPos := emb + Tensor.castLike emb posBatch
          let availLen := if desiredLen <= tChunk then desiredLen else tChunk
          let seg0 : STensor #[1, availLen, cfg.dModel] := Tensor.sliceChecked embPos 1 0 availLen #[1, availLen, cfg.dModel]
          let segDyn : Sigma STensor :=
            if desiredLen <= availLen then
              Tensor.packShape seg0
            else
              let extraLen := desiredLen - availLen
              let extra : STensor #[1, extraLen, cfg.dModel] :=
                Tensor.castLike seg0 ((Tensor.zeros #[1, extraLen, cfg.dModel] false seg0.device).forgetDType)
              Tensor.packShape (Tensor.reshapeChecked (Tensor.cat seg0 extra 1) #[1, desiredLen, cfg.dModel])
          out := out.push segDyn
          start := start + len
      out

    let totalValid := sumU64 chunkLensAfterCnn
    let hiddenValid : STensor #[1, totalValid, cfg.dModel] :=
      if hiddenParts.isEmpty then
        Tensor.castLike sample ((Tensor.zeros #[1, totalValid, cfg.dModel] false sample.device).forgetDType)
      else
        Tensor.catManyChecked hiddenParts 1 #[1, totalValid, cfg.dModel]

    let cuCumsum := buildCuChunkLensCumsum cfg #[totalValid] maxChunkAfterCnn
    let attnChunkLens := chunkLensFromCumsum cuCumsum
    let hiddenChunked := runLayersChunkLocal m hiddenValid attnChunkLens

    let hidden := m.lnPost.forward3d hiddenChunked
    let hidden : STensor #[1, totalValid, cfg.dModel] := Tensor.affine3d hidden m.proj1Weight m.proj1Bias
    let hidden := activate cfg.activationFunction hidden
    let hidden : STensor #[1, totalValid, cfg.outputDim] := Tensor.affine3d hidden m.proj2Weight m.proj2Bias

    if totalValid >= outSeq then
      hidden.sliceChecked 1 0 outSeq #[1, outSeq, cfg.outputDim]
    else
      let padLen := outSeq - totalValid
      let pad : STensor #[1, padLen, cfg.outputDim] :=
        Tensor.castLike hidden ((Tensor.zeros #[1, padLen, cfg.outputDim] false hidden.device).forgetDType)
      Tensor.reshapeChecked (Tensor.cat hidden pad 1) #[1, outSeq, cfg.outputDim]

/-- Varlen/chunked execution path mirroring reference `cu_seqlens` behavior:
    each sample is chunked from `featureLens`, convolved chunk-wise, then transformed
    with chunk-local self-attention blocks and finally re-padded to fixed output width. -/
def forwardVarLen {batch frames : UInt64}
    (m : AudioEncoder cfg)
    (inputFeatures : STensor #[batch, cfg.numMelBins, frames])
    (featureLens : Array UInt64)
    : STensor #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames, cfg.outputDim] :=
  let inputFeatures := alignFeatures m inputFeatures
  let outSeq := AudioEncoderConfig.framesAfterConv3 cfg frames
  if batch == 0 then
    Tensor.castLike inputFeatures ((Tensor.zeros #[batch, outSeq, cfg.outputDim] false inputFeatures.device).forgetDType)
  else
    let rows : Array (Sigma STensor) := Id.run do
      let mut out : Array (Sigma STensor) := #[]
      for b in [:batch.toNat] do
        let sample : STensor #[1, cfg.numMelBins, frames] := Tensor.sliceChecked inputFeatures 0 b.toUInt64 1 #[1, cfg.numMelBins, frames]
        let len := featureLens.getD b frames
        let row : STensor #[1, outSeq, cfg.outputDim] := encodeOneSampleVarLen m sample len
        out := out.push (Tensor.packShape row)
      out
    Tensor.catManyChecked rows 0 #[batch, outSeq, cfg.outputDim]

def forward {batch frames : UInt64}
    (m : AudioEncoder cfg)
    (inputFeatures : STensor #[batch, cfg.numMelBins, frames])
    (attnMask : Option (STensor #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames]) := none)
    : STensor #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames, cfg.outputDim] :=
  let inputFeatures := alignFeatures m inputFeatures
  let t1 := AudioEncoderConfig.downsampleOnce frames
  let t2 := AudioEncoderConfig.downsampleTwice frames
  let t3 := AudioEncoderConfig.framesAfterConv3 cfg frames

  let x0 : STensor #[batch, 1, cfg.numMelBins, frames] := Tensor.reshapeChecked inputFeatures #[batch, 1, cfg.numMelBins, frames]

  let x1 : STensor #[batch, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv1 cfg, t1] :=
    Tensor.reshapeChecked (Tensor.Preserving.gelu (Tensor.conv2dBias x0 m.conv2d1Weight m.conv2d1Bias #[2, 2] #[1, 1]))
      #[batch, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv1 cfg, t1]
  let x2 : STensor #[batch, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv2 cfg, t2] :=
    Tensor.reshapeChecked (Tensor.Preserving.gelu (Tensor.conv2dBias x1 m.conv2d2Weight m.conv2d2Bias #[2, 2] #[1, 1]))
      #[batch, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv2 cfg, t2]
  let x3 : STensor #[batch, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg, t3] :=
    Tensor.reshapeChecked (Tensor.Preserving.gelu (Tensor.conv2dBias x2 m.conv2d3Weight m.conv2d3Bias #[2, 2] #[1, 1]))
      #[batch, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg, t3]

  let x3t : STensor #[batch, t3, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg] :=
    Tensor.reshapeChecked (Tensor.permute x3 #[0, 3, 1, 2]) #[batch, t3, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg]

  let xFlat : STensor #[batch, t3, AudioEncoderConfig.convOutInDim cfg] :=
    Tensor.reshapeChecked x3t #[batch, t3, AudioEncoderConfig.convOutInDim cfg]
  let xEmb : STensor #[batch, t3, cfg.dModel] := Tensor.linear3d xFlat m.convOutWeight

  let pos : STensor #[t3, cfg.dModel] := sinusoidPosition (seq := t3) (dim := cfg.dModel) (device := xEmb.device)
  let posBatch : STensor #[batch, t3, cfg.dModel] := Tensor.expand (Tensor.reshapeChecked pos #[1, t3, cfg.dModel]) #[batch, t3, cfg.dModel]
  let hidden0 := xEmb + Tensor.castLike xEmb posBatch

  let hidden := match attnMask with
    | some mask => m.layers.foldl (fun h layer => layer.forward h (some mask)) hidden0
    | none => m.layers.foldl (fun h layer => layer.forward h none) hidden0

  let hidden := m.lnPost.forward3d hidden
  let hidden : STensor #[batch, t3, cfg.dModel] := Tensor.affine3d hidden m.proj1Weight m.proj1Bias
  let hidden := activate cfg.activationFunction hidden
  Tensor.affine3d hidden m.proj2Weight m.proj2Bias

end AudioEncoder

instance {cfg : AudioEncoderConfig} {batch frames : UInt64} :
    Module (AudioEncoder cfg)
      (STensor #[batch, cfg.numMelBins, frames])
      (STensor #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames, cfg.outputDim]) where
  forward := fun m x => m.forward x none

end torch.qwen3asr
