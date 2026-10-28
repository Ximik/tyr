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

variable {device : Device} {dtype featureDType : DType}
variable [DType.RealFloating dtype]
open torch.Model

private def initWeight (shape : Shape) (fanIn : Nat) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Tensor { dtype := dtype, device := device, shape := shape }) :=
  Model.initWeight (device := device) shape fanIn dtype

private def initBias (shape : Shape) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Tensor { dtype := dtype, device := device, shape := shape }) :=
  Model.initBias (device := device) shape dtype

private def initLayerNorm (dim : Nat) (eps : Float := 1e-5) (device : Device := .CPU) (dtype : DType := .Float32) : IO (LayerNorm (device := device) dim dtype) := do
  IO.ofExcept (Shape.validateAllocation #[dim])
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  pure {
    weight := (← (Tensor.onesFor { shape := #[dim], dtype, device }).parameter)
    bias := (← (Tensor.zerosFor { shape := #[dim], dtype, device }).parameter)
    eps := ⟨eps⟩ }

private def activate {s : Shape} (name : String) (x : Tensor { dtype := dtype, device := device, shape := s }) : Tensor { dtype := dtype, device := device, shape := s } :=
  if name == "relu" then
    Tensor.relu x
  else if name == "silu" then
    Tensor.silu x
  else
    Tensor.gelu x

private def sinusoidPosition {seq dim : Nat}
    (maxTimescale : Float := 10000.0)
    (device : Device := Device.CPU)
    : Tensor { dtype := .Float32, device := device, shape := #[seq, dim] } :=
  if dim % 2 == 0 then
    let half := dim / 2
    let pos := (Tensor.arange seq).toFloat32
    let idx := (Tensor.arange half).toFloat32
    let denom : Float := if half <= 1 then 1.0 else (half - 1).toFloat
    let logInc := Float.log maxTimescale / denom
    let invTimescales := Tensor.exp (idx * (-logInc))
    let scaled := Tensor.mm (pos.reshape #[seq, 1]) (invTimescales.reshape #[1, half])
    let out := (Tensor.cat scaled.sin scaled.cos 1).reshapeChecked #[seq, dim]
    (out.to device)
  else
    (Tensor.zeros #[seq, dim] false device)

/-- Multi-head self-attention used by the audio encoder. -/
structure AudioAttention (cfg : AudioEncoderConfig) (dtype : DType := .Float32) where
  qProjWeight : Tensor { dtype := dtype, device := device, shape := #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg, cfg.dModel] }
  qProjBias : Tensor { dtype := dtype, device := device, shape := #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg] }
  kProjWeight : Tensor { dtype := dtype, device := device, shape := #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg, cfg.dModel] }
  kProjBias : Tensor { dtype := dtype, device := device, shape := #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg] }
  vProjWeight : Tensor { dtype := dtype, device := device, shape := #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg, cfg.dModel] }
  vProjBias : Tensor { dtype := dtype, device := device, shape := #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg] }
  outProjWeight : Tensor { dtype := dtype, device := device, shape := #[cfg.dModel, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg] }
  outProjBias : Tensor { dtype := dtype, device := device, shape := #[cfg.dModel] }
  deriving TensorStruct

namespace AudioAttention

def init (cfg : AudioEncoderConfig) (device : Device := .CPU) (dtype : DType := .Float32) : IO (AudioAttention (device := device) (dtype := dtype) cfg) := do
  let qProjWeight ← initWeight (dtype := dtype) (device := device)
    #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg, cfg.dModel] cfg.dModel
  let qProjBias ← initBias (dtype := dtype) (device := device) #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  let kProjWeight ← initWeight (dtype := dtype) (device := device)
    #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg, cfg.dModel] cfg.dModel
  let kProjBias ← initBias (dtype := dtype) (device := device) #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  let vProjWeight ← initWeight (dtype := dtype) (device := device)
    #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg, cfg.dModel] cfg.dModel
  let vProjBias ← initBias (dtype := dtype) (device := device) #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  let outProjWeight ← initWeight (dtype := dtype) (device := device)
    #[cfg.dModel, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
    (cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg)
  let outProjBias ← initBias (dtype := dtype) (device := device) #[cfg.dModel]
  pure {
    qProjWeight, qProjBias, kProjWeight, kProjBias, vProjWeight, vProjBias, outProjWeight, outProjBias
  }

private def forwardWith {M : Type → Type} [Monad M] {batch seq : Nat}
    (attend : (q k v : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.encoderAttentionHeads, seq, AudioEncoderConfig.headDim cfg] }) →
      Option (Tensor { dtype := .Int64, device := device, shape := #[batch, seq] }) →
      M (Tensor { dtype := dtype, device := device, shape := #[batch, cfg.encoderAttentionHeads, seq, AudioEncoderConfig.headDim cfg] }))
    (m : AudioAttention (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] })
    (attnMask : Option (Tensor { dtype := .Int64, device := device, shape := #[batch, seq] }) := none)
    : M (Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] }) := do
  let q : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg] } :=
    Tensor.affine3d x m.qProjWeight m.qProjBias
  let k : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg] } :=
    Tensor.affine3d x m.kProjWeight m.kProjBias
  let v : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg] } :=
    Tensor.affine3d x m.vProjWeight m.vProjBias

  let q : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.encoderAttentionHeads, AudioEncoderConfig.headDim cfg] } :=
    Tensor.reshape q #[batch, seq, cfg.encoderAttentionHeads, AudioEncoderConfig.headDim cfg]
  let k : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.encoderAttentionHeads, AudioEncoderConfig.headDim cfg] } :=
    Tensor.reshape k #[batch, seq, cfg.encoderAttentionHeads, AudioEncoderConfig.headDim cfg]
  let v : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.encoderAttentionHeads, AudioEncoderConfig.headDim cfg] } :=
    Tensor.reshape v #[batch, seq, cfg.encoderAttentionHeads, AudioEncoderConfig.headDim cfg]

  let q := Tensor.splitHeads q
  let k := Tensor.splitHeads k
  let v := Tensor.splitHeads v

  let out ← attend q k v attnMask

  let out := Tensor.joinHeads out
  let out : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg] } :=
    Tensor.reshape out #[batch, seq, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  return Tensor.affine3d out m.outProjWeight m.outProjBias

/-- Deterministic attention: configured dropout is disabled for inference. -/
def forwardInference {batch seq : Nat}
    (m : AudioAttention (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] })
    (attnMask : Option (Tensor { dtype := .Int64, device := device, shape := #[batch, seq] }) := none)
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] } :=
  Id.run <| forwardWith (M := Id) (fun q k v mask => pure <| match mask with
    | some mask => Tensor.attentionGQAMask q k v mask false true
    | none => Tensor.attentionGQA q k v false true) m x attnMask

/-- Effectful attention. Training honors `cfg.attentionDropout`; evaluation disables it. -/
def forwardIO {batch seq : Nat}
    (m : AudioAttention (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] })
    (training : Bool)
    (attnMask : Option (Tensor { dtype := .Int64, device := device, shape := #[batch, seq] }) := none)
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] }) :=
  forwardWith (fun q k v mask => match mask with
    | some mask => Tensor.attentionGQAMaskIO q k v mask cfg.attentionDropout training false true
    | none => Tensor.attentionGQAIO q k v cfg.attentionDropout training false true) m x attnMask

/-- Compatibility alias for deterministic inference. Use `forwardIO` for training. -/
def forward {batch seq : Nat}
    (m : AudioAttention (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] })
    (attnMask : Option (Tensor { dtype := .Int64, device := device, shape := #[batch, seq] }) := none)
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] } :=
  m.forwardInference x attnMask

end AudioAttention

/-- One transformer block in the audio encoder. -/
structure AudioEncoderLayer (cfg : AudioEncoderConfig) (dtype : DType := .Float32) where
  selfAttn : AudioAttention (device := device) (dtype := dtype) cfg
  selfAttnLayerNorm : LayerNorm (device := device) cfg.dModel dtype
  fc1Weight : Tensor { dtype := dtype, device := device, shape := #[cfg.encoderFfnDim, cfg.dModel] }
  fc1Bias : Tensor { dtype := dtype, device := device, shape := #[cfg.encoderFfnDim] }
  fc2Weight : Tensor { dtype := dtype, device := device, shape := #[cfg.dModel, cfg.encoderFfnDim] }
  fc2Bias : Tensor { dtype := dtype, device := device, shape := #[cfg.dModel] }
  finalLayerNorm : LayerNorm (device := device) cfg.dModel dtype
  deriving TensorStruct

namespace AudioEncoderLayer

def init (cfg : AudioEncoderConfig) (device : Device := .CPU) (dtype : DType := .Float32) : IO (AudioEncoderLayer (device := device) (dtype := dtype) cfg) := do
  let selfAttn ← AudioAttention.init (device := device) (dtype := dtype) cfg
  let selfAttnLayerNorm ← initLayerNorm (dtype := dtype) (device := device) cfg.dModel 1e-5
  let fc1Weight ← initWeight (dtype := dtype) (device := device) #[cfg.encoderFfnDim, cfg.dModel] cfg.dModel
  let fc1Bias ← initBias (dtype := dtype) (device := device) #[cfg.encoderFfnDim]
  let fc2Weight ← initWeight (dtype := dtype) (device := device) #[cfg.dModel, cfg.encoderFfnDim] cfg.encoderFfnDim
  let fc2Bias ← initBias (dtype := dtype) (device := device) #[cfg.dModel]
  let finalLayerNorm ← initLayerNorm (dtype := dtype) (device := device) cfg.dModel 1e-5
  pure { selfAttn, selfAttnLayerNorm, fc1Weight, fc1Bias, fc2Weight, fc2Bias, finalLayerNorm }

private def forwardWith {M : Type → Type} [Monad M] {batch seq : Nat}
    (attend : AudioAttention (device := device) (dtype := dtype) cfg →
      Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] } →
      Option (Tensor { dtype := .Int64, device := device, shape := #[batch, seq] }) →
      M (Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] }))
    (m : AudioEncoderLayer (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] })
    (attnMask : Option (Tensor { dtype := .Int64, device := device, shape := #[batch, seq] }) := none)
    : M (Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] }) := do
  let residual0 := x
  let h0 := m.selfAttnLayerNorm.forward3d x
  let h1 ← attend m.selfAttn h0 attnMask
  let h2 := Tensor.add residual0 h1

  let residual1 := h2
  let h3 := m.finalLayerNorm.forward3d h2
  let h4 : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.encoderFfnDim] } := Tensor.affine3d h3 m.fc1Weight m.fc1Bias
  let h5 := activate cfg.activationFunction h4
  let h6 : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] } := Tensor.affine3d h5 m.fc2Weight m.fc2Bias
  return Tensor.add residual1 h6

/-- Deterministic encoder-layer inference, with attention dropout disabled. -/
def forwardInference {batch seq : Nat}
    (m : AudioEncoderLayer (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] })
    (attnMask : Option (Tensor { dtype := .Int64, device := device, shape := #[batch, seq] }) := none)
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] } :=
  Id.run <| forwardWith (M := Id) (fun a x mask => pure (a.forwardInference x mask)) m x attnMask

/-- Training/evaluation execution with explicit attention-dropout effects. -/
def forwardIO {batch seq : Nat}
    (m : AudioEncoderLayer (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] })
    (training : Bool)
    (attnMask : Option (Tensor { dtype := .Int64, device := device, shape := #[batch, seq] }) := none)
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] }) :=
  forwardWith (fun a x mask => a.forwardIO x training mask) m x attnMask

/-- Compatibility alias for deterministic inference. Use `forwardIO` for training. -/
def forward {batch seq : Nat}
    (m : AudioEncoderLayer (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] })
    (attnMask : Option (Tensor { dtype := .Int64, device := device, shape := #[batch, seq] }) := none)
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] } :=
  m.forwardInference x attnMask

end AudioEncoderLayer

/-- Full Qwen3-ASR audio encoder. -/
structure AudioEncoder (cfg : AudioEncoderConfig) (dtype : DType := .Float32) where
  conv2d1Weight : Tensor { dtype := dtype, device := device, shape := #[cfg.downsampleHiddenSize, 1, 3, 3] }
  conv2d1Bias : Tensor { dtype := dtype, device := device, shape := #[cfg.downsampleHiddenSize] }
  conv2d2Weight : Tensor { dtype := dtype, device := device, shape := #[cfg.downsampleHiddenSize, cfg.downsampleHiddenSize, 3, 3] }
  conv2d2Bias : Tensor { dtype := dtype, device := device, shape := #[cfg.downsampleHiddenSize] }
  conv2d3Weight : Tensor { dtype := dtype, device := device, shape := #[cfg.downsampleHiddenSize, cfg.downsampleHiddenSize, 3, 3] }
  conv2d3Bias : Tensor { dtype := dtype, device := device, shape := #[cfg.downsampleHiddenSize] }
  convOutWeight : Tensor { dtype := dtype, device := device, shape := #[cfg.dModel, AudioEncoderConfig.convOutInDim cfg] }
  layers : Array (AudioEncoderLayer (device := device) (dtype := dtype) cfg)
  lnPost : LayerNorm (device := device) cfg.dModel dtype
  proj1Weight : Tensor { dtype := dtype, device := device, shape := #[cfg.dModel, cfg.dModel] }
  proj1Bias : Tensor { dtype := dtype, device := device, shape := #[cfg.dModel] }
  proj2Weight : Tensor { dtype := dtype, device := device, shape := #[cfg.outputDim, cfg.dModel] }
  proj2Bias : Tensor { dtype := dtype, device := device, shape := #[cfg.outputDim] }
  deriving TensorStruct

namespace AudioEncoder

/-- The frontend computes mel features in Float32. Match the first convolution's
    FP32/BF16 dtype and device before padding or convolution; matching dtypes are
    left unchanged. Positional frequencies are still computed in Float32. -/
private def alignFeatures {batch frames : Nat} (m : AudioEncoder (device := device) (dtype := dtype) cfg)
    (features : Tensor { dtype := featureDType, device := device, shape := #[batch, cfg.numMelBins, frames] }) : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.numMelBins, frames] } :=
  let features := Tensor.castLike m.conv2d1Weight features
  features.to m.conv2d1Weight.device

def init (cfg : AudioEncoderConfig) (device : Device := .CPU) (dtype : DType := .Float32) : IO (AudioEncoder (device := device) (dtype := dtype) cfg) := do
  if cfg.dModel % cfg.encoderAttentionHeads != 0 then
    throw <| IO.userError
      s!"Audio encoder requires dModel divisible by encoderAttentionHeads, got {cfg.dModel} and {cfg.encoderAttentionHeads}"

  let conv2d1Weight ← initWeight (dtype := dtype) (device := device) #[cfg.downsampleHiddenSize, 1, 3, 3] 1
  let conv2d1Bias ← initBias (dtype := dtype) (device := device) #[cfg.downsampleHiddenSize]
  let conv2d2Weight ← initWeight (dtype := dtype) (device := device) #[cfg.downsampleHiddenSize, cfg.downsampleHiddenSize, 3, 3] cfg.downsampleHiddenSize
  let conv2d2Bias ← initBias (dtype := dtype) (device := device) #[cfg.downsampleHiddenSize]
  let conv2d3Weight ← initWeight (dtype := dtype) (device := device) #[cfg.downsampleHiddenSize, cfg.downsampleHiddenSize, 3, 3] cfg.downsampleHiddenSize
  let conv2d3Bias ← initBias (dtype := dtype) (device := device) #[cfg.downsampleHiddenSize]

  let convOutWeight ← initWeight (dtype := dtype) (device := device) #[cfg.dModel, AudioEncoderConfig.convOutInDim cfg] (AudioEncoderConfig.convOutInDim cfg)

  let mut layers : Array (AudioEncoderLayer (device := device) (dtype := dtype) cfg) := #[]
  for _ in [:cfg.encoderLayers] do
    let layer ← AudioEncoderLayer.init (device := device) (dtype := dtype) cfg
    layers := layers.push layer

  let lnPost ← initLayerNorm (dtype := dtype) (device := device) cfg.dModel 1e-5
  let proj1Weight ← initWeight (dtype := dtype) (device := device) #[cfg.dModel, cfg.dModel] cfg.dModel
  let proj1Bias ← initBias (dtype := dtype) (device := device) #[cfg.dModel]
  let proj2Weight ← initWeight (dtype := dtype) (device := device) #[cfg.outputDim, cfg.dModel] cfg.dModel
  let proj2Bias ← initBias (dtype := dtype) (device := device) #[cfg.outputDim]

  pure {
    conv2d1Weight, conv2d1Bias, conv2d2Weight, conv2d2Bias, conv2d3Weight, conv2d3Bias,
    convOutWeight, layers, lnPost, proj1Weight, proj1Bias, proj2Weight, proj2Bias
  }

/-- Port of reference chunk planner used before convolutional downsampling.
    For each feature length `L`, emits `ceil(L / (2*n_window))` chunk lengths where
    all chunks are `2*n_window` except the last chunk (`L % (2*n_window)` or full). -/
def buildChunkLengths (cfg : AudioEncoderConfig) (featureLens : Array Nat) : Array Nat :=
  let step := cfg.nWindow * 2
  if step == 0 then
    featureLens
  else
    Id.run do
      let mut out : Array Nat := #[]
      for len in featureLens do
        if len == 0 then
          pure ()
        else
          let chunkNum := (len + step - 1) / step
          for j in [:chunkNum] do
            if j + 1 == chunkNum then
              let tail := len % step
              out := out.push (if tail == 0 then step else tail)
            else
              out := out.push step
      out

/-- Per-chunk length transform through `_get_feat_extract_output_lengths`. -/
def chunkLengthsAfterCnn (_cfg : AudioEncoderConfig) (chunkLens : Array Nat) : Array Nat :=
  chunkLens.map AudioEncoderConfig.featExtractOutputLength

/-- Port of `cu_chunk_lens` creation in reference audio encoder forward.
    Returns inclusive prefix-sum source with initial `0`. -/
def buildCuChunkLensCumsum
    (cfg : AudioEncoderConfig)
    (afterCnnLens : Array Nat)
    (paddedAfterCnnWidth : Nat)
    : Array Nat :=
  let denom := cfg.nWindow * 2
  let ratio := if denom == 0 then 1 else cfg.nWindowInfer / denom
  let windowAfterCnn0 := paddedAfterCnnWidth * (if ratio == 0 then 1 else ratio)
  let windowAfterCnn := if windowAfterCnn0 == 0 then 1 else windowAfterCnn0
  Id.run do
    let mut segs : Array Nat := #[0]
    for cnnLen in afterCnnLens do
      let full := cnnLen / windowAfterCnn
      for _ in [:full] do
        segs := segs.push windowAfterCnn
      let rem := cnnLen % windowAfterCnn
      if rem != 0 then
        segs := segs.push rem
    let mut csum : Nat := 0
    let mut out : Array Nat := #[]
    for s in segs do
      csum := csum + s
      out := out.push csum
    out

private def maxOr (xs : Array Nat) (fallback : Nat := 0) : Nat :=
  xs.foldl (fun acc x => if x > acc then x else acc) fallback

private def sumU64 (xs : Array Nat) : Nat :=
  xs.foldl (· + ·) 0

private def chunkLensFromCumsum (cs : Array Nat) : Array Nat :=
  Id.run do
    let mut out : Array Nat := #[]
    let mut prev : Nat := 0
    for cur in cs do
      if cur >= prev then
        let d := cur - prev
        if d != 0 then
          out := out.push d
        prev := cur
    out

private def padRightFrames {cfg : AudioEncoderConfig} {frames maxFrames : Nat}
    (x : Tensor { dtype := dtype, device := device, shape := #[1, cfg.numMelBins, frames] })
    : Tensor { dtype := dtype, device := device, shape := #[1, cfg.numMelBins, maxFrames] } :=
  let padFrames := if maxFrames >= frames then maxFrames - frames else 0
  let pad : Tensor { dtype := dtype, device := device, shape := #[1, cfg.numMelBins, padFrames] } :=
    Tensor.castLike x ((Tensor.zeros #[1, cfg.numMelBins, padFrames] false x.device))
  let cat := Tensor.cat x pad 2
  Tensor.reshapeChecked cat #[1, cfg.numMelBins, maxFrames]

private def encodePaddedChunk {maxFrames : Nat}
    (m : AudioEncoder (device := device) (dtype := dtype) cfg)
    (chunk : Tensor { dtype := dtype, device := device, shape := #[1, cfg.numMelBins, maxFrames] })
    : Tensor { dtype := dtype, device := device, shape := #[1, AudioEncoderConfig.framesAfterConv3 cfg maxFrames, cfg.dModel] } :=
  let t1 := AudioEncoderConfig.downsampleOnce maxFrames
  let t2 := AudioEncoderConfig.downsampleTwice maxFrames
  let t3 := AudioEncoderConfig.framesAfterConv3 cfg maxFrames
  let x0 : Tensor { dtype := dtype, device := device, shape := #[1, 1, cfg.numMelBins, maxFrames] } := Tensor.reshape chunk #[1, 1, cfg.numMelBins, maxFrames]
  let x1 : Tensor { dtype := dtype, device := device, shape := #[1, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv1 cfg, t1] } :=
    Tensor.reshapeChecked (Tensor.gelu (Tensor.conv2dBias x0 m.conv2d1Weight m.conv2d1Bias #[2, 2] #[1, 1]))
      #[1, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv1 cfg, t1]
  let x2 : Tensor { dtype := dtype, device := device, shape := #[1, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv2 cfg, t2] } :=
    Tensor.reshapeChecked (Tensor.gelu (Tensor.conv2dBias x1 m.conv2d2Weight m.conv2d2Bias #[2, 2] #[1, 1]))
      #[1, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv2 cfg, t2]
  let x3 : Tensor { dtype := dtype, device := device, shape := #[1, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg, t3] } :=
    Tensor.reshapeChecked (Tensor.gelu (Tensor.conv2dBias x2 m.conv2d3Weight m.conv2d3Bias #[2, 2] #[1, 1]))
      #[1, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg, t3]
  let x3t : Tensor { dtype := dtype, device := device, shape := #[1, t3, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg] } :=
    Tensor.reshape (Tensor.permute x3 #[0, 3, 1, 2]) #[1, t3, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg]
  let xFlat : Tensor { dtype := dtype, device := device, shape := #[1, t3, AudioEncoderConfig.convOutInDim cfg] } :=
    Tensor.reshapeChecked x3t #[1, t3, AudioEncoderConfig.convOutInDim cfg]
  Tensor.linear3d xFlat m.convOutWeight

private abbrev LayerForward (M : Type → Type) (cfg : AudioEncoderConfig) (device : Device) (dtype : DType) :=
  {batch seq : Nat} → AudioEncoderLayer (device := device) (dtype := dtype) cfg →
    Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] } →
    Option (Tensor { dtype := .Int64, device := device, shape := #[batch, seq] }) →
    M (Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] })

private def runLayersChunkLocalWith {M : Type → Type} [Monad M] {seq : Nat}
    (layerForward : LayerForward M cfg device dtype)
    (m : AudioEncoder (device := device) (dtype := dtype) cfg)
    (hidden : Tensor { dtype := dtype, device := device, shape := #[1, seq, cfg.dModel] })
    (chunkLens : Array Nat)
    : M (Tensor { dtype := dtype, device := device, shape := #[1, seq, cfg.dModel] }) := do
  if seq == 0 then
    return hidden
  let mut parts : Array (Sigma (fun shape => Tensor { dtype := dtype, device := device, shape := shape })) := #[]
  let mut start : Nat := 0
  for len in chunkLens do
    if len != 0 then
      let seg : Tensor { dtype := dtype, device := device, shape := #[1, len, cfg.dModel] } :=
        Tensor.sliceChecked hidden 1 start len #[1, len, cfg.dModel]
      let segOut ← m.layers.foldlM (fun h layer => layerForward layer h none) seg
      parts := parts.push (Tensor.packShape segOut)
      start := start + len
  return if parts.isEmpty then
    Tensor.castLike hidden ((Tensor.zeros #[1, seq, cfg.dModel] false hidden.device))
  else
    Tensor.catManyChecked parts 1 #[1, seq, cfg.dModel]

private def encodeOneSampleVarLenWith {M : Type → Type} [Monad M] {frames : Nat}
    (layerForward : LayerForward M cfg device dtype)
    (m : AudioEncoder (device := device) (dtype := dtype) cfg)
    (sample : Tensor { dtype := dtype, device := device, shape := #[1, cfg.numMelBins, frames] })
    (featureLen : Nat)
    : M (Tensor { dtype := dtype, device := device, shape := #[1, AudioEncoderConfig.framesAfterConv3 cfg frames, cfg.outputDim] }) := do
  let outSeq := AudioEncoderConfig.framesAfterConv3 cfg frames
  let featureLen := if featureLen <= frames then featureLen else frames
  if featureLen == 0 then
    return Tensor.castLike sample ((Tensor.zeros #[1, outSeq, cfg.outputDim] false sample.device))
  else
    let chunkLens := buildChunkLengths cfg #[featureLen]
    let maxChunkLen := maxOr chunkLens 1
    let tChunk := AudioEncoderConfig.framesAfterConv3 cfg maxChunkLen
    let chunkLensAfterCnn := chunkLengthsAfterCnn cfg chunkLens
    let maxChunkAfterCnn := maxOr chunkLensAfterCnn tChunk
    let pos : Tensor { dtype := .Float32, device := device, shape := #[tChunk, cfg.dModel] } :=
      sinusoidPosition (seq := tChunk) (dim := cfg.dModel) (device := sample.device)
    let posBatch : Tensor { dtype := .Float32, device := device, shape := #[1, tChunk, cfg.dModel] } := Tensor.reshape pos #[1, tChunk, cfg.dModel]

    let hiddenParts : Array (Sigma (fun shape => Tensor { dtype := dtype, device := device, shape := shape })) := Id.run do
      let mut out : Array (Sigma (fun shape => Tensor { dtype := dtype, device := device, shape := shape })) := #[]
      let mut start : Nat := 0
      for i in [:chunkLens.size] do
        let len := chunkLens.getD i 0
        if len != 0 then
          let desiredLen := chunkLensAfterCnn.getD i 0
          let chunk : Tensor { dtype := dtype, device := device, shape := #[1, cfg.numMelBins, len] } := Tensor.sliceChecked sample 2 start len #[1, cfg.numMelBins, len]
          let padded : Tensor { dtype := dtype, device := device, shape := #[1, cfg.numMelBins, maxChunkLen] } :=
            padRightFrames (cfg := cfg) (frames := len) (maxFrames := maxChunkLen) chunk
          let emb : Tensor { dtype := dtype, device := device, shape := #[1, tChunk, cfg.dModel] } := encodePaddedChunk m padded
          let embPos := Tensor.add emb (Tensor.castLike emb posBatch)
          let availLen := if desiredLen <= tChunk then desiredLen else tChunk
          let seg0 : Tensor { dtype := dtype, device := device, shape := #[1, availLen, cfg.dModel] } := Tensor.sliceChecked embPos 1 0 availLen #[1, availLen, cfg.dModel]
          let segDyn : Sigma (fun shape => Tensor { dtype := dtype, device := device, shape := shape }) :=
            if desiredLen <= availLen then
              Tensor.packShape seg0
            else
              let extraLen := desiredLen - availLen
              let extra : Tensor { dtype := dtype, device := device, shape := #[1, extraLen, cfg.dModel] } :=
                Tensor.castLike seg0 ((Tensor.zeros #[1, extraLen, cfg.dModel] false seg0.device))
              Tensor.packShape (Tensor.reshapeChecked (Tensor.cat seg0 extra 1) #[1, desiredLen, cfg.dModel])
          out := out.push segDyn
          start := start + len
      out

    let totalValid := sumU64 chunkLensAfterCnn
    let hiddenValid : Tensor { dtype := dtype, device := device, shape := #[1, totalValid, cfg.dModel] } :=
      if hiddenParts.isEmpty then
        Tensor.castLike sample ((Tensor.zeros #[1, totalValid, cfg.dModel] false sample.device))
      else
        Tensor.catManyChecked hiddenParts 1 #[1, totalValid, cfg.dModel]

    let cuCumsum := buildCuChunkLensCumsum cfg #[totalValid] maxChunkAfterCnn
    let attnChunkLens := chunkLensFromCumsum cuCumsum
    let hiddenChunked ← runLayersChunkLocalWith layerForward m hiddenValid attnChunkLens

    let hidden := m.lnPost.forward3d hiddenChunked
    let hidden : Tensor { dtype := dtype, device := device, shape := #[1, totalValid, cfg.dModel] } := Tensor.affine3d hidden m.proj1Weight m.proj1Bias
    let hidden := activate cfg.activationFunction hidden
    let hidden : Tensor { dtype := dtype, device := device, shape := #[1, totalValid, cfg.outputDim] } := Tensor.affine3d hidden m.proj2Weight m.proj2Bias

    return if totalValid >= outSeq then
      hidden.sliceChecked 1 0 outSeq #[1, outSeq, cfg.outputDim]
    else
      let padLen := outSeq - totalValid
      let pad : Tensor { dtype := dtype, device := device, shape := #[1, padLen, cfg.outputDim] } :=
        Tensor.castLike hidden ((Tensor.zeros #[1, padLen, cfg.outputDim] false hidden.device))
      Tensor.reshapeChecked (Tensor.cat hidden pad 1) #[1, outSeq, cfg.outputDim]

private def forwardVarLenWith {M : Type → Type} [Monad M] {batch frames : Nat}
    (layerForward : LayerForward M cfg device dtype)
    (m : AudioEncoder (device := device) (dtype := dtype) cfg)
    (inputFeatures : Tensor { dtype := featureDType, device := device, shape := #[batch, cfg.numMelBins, frames] })
    (featureLens : Array Nat)
    : M (Tensor { dtype := dtype, device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames, cfg.outputDim] }) := do
  let inputFeatures := alignFeatures m inputFeatures
  let outSeq := AudioEncoderConfig.framesAfterConv3 cfg frames
  if batch == 0 then
    return Tensor.castLike inputFeatures ((Tensor.zeros #[batch, outSeq, cfg.outputDim] false inputFeatures.device))
  let mut rows : Array (Sigma (fun shape => Tensor { dtype := dtype, device := device, shape := shape })) := #[]
  for b in [:batch] do
    let sample : Tensor { dtype := dtype, device := device, shape := #[1, cfg.numMelBins, frames] } :=
      Tensor.sliceChecked inputFeatures 0 b 1 #[1, cfg.numMelBins, frames]
    let len := featureLens.getD b frames
    let row ← encodeOneSampleVarLenWith layerForward m sample len
    rows := rows.push (Tensor.packShape row)
  return Tensor.catManyChecked rows 0 #[batch, outSeq, cfg.outputDim]

/-- Deterministic chunk-local inference, re-padded to the fixed output width. -/
def forwardVarLenInference {batch frames : Nat}
    (m : AudioEncoder (device := device) (dtype := dtype) cfg)
    (inputFeatures : Tensor { dtype := featureDType, device := device, shape := #[batch, cfg.numMelBins, frames] })
    (featureLens : Array Nat)
    : Tensor { dtype := dtype, device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames, cfg.outputDim] } :=
  Id.run <| forwardVarLenWith (M := Id)
    (fun layer x mask => pure (layer.forwardInference x mask)) m inputFeatures featureLens

/-- Chunk-local training/evaluation. Every attention chunk honors the training flag. -/
def forwardVarLenIO {batch frames : Nat}
    (m : AudioEncoder (device := device) (dtype := dtype) cfg)
    (inputFeatures : Tensor { dtype := featureDType, device := device, shape := #[batch, cfg.numMelBins, frames] })
    (featureLens : Array Nat)
    (training : Bool)
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames, cfg.outputDim] }) :=
  forwardVarLenWith (fun layer x mask => layer.forwardIO x training mask) m inputFeatures featureLens

/-- Compatibility alias for deterministic chunk-local inference. -/
def forwardVarLen {batch frames : Nat}
    (m : AudioEncoder (device := device) (dtype := dtype) cfg)
    (inputFeatures : Tensor { dtype := featureDType, device := device, shape := #[batch, cfg.numMelBins, frames] })
    (featureLens : Array Nat)
    : Tensor { dtype := dtype, device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames, cfg.outputDim] } :=
  m.forwardVarLenInference inputFeatures featureLens

private def forwardWith {M : Type → Type} [Monad M] {batch frames : Nat}
    (layerForward : LayerForward M cfg device dtype)
    (m : AudioEncoder (device := device) (dtype := dtype) cfg)
    (inputFeatures : Tensor { dtype := featureDType, device := device, shape := #[batch, cfg.numMelBins, frames] })
    (attnMask : Option (Tensor { dtype := .Int64, device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames] }) := none)
    : M (Tensor { dtype := dtype, device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames, cfg.outputDim] }) := do
  let inputFeatures := alignFeatures m inputFeatures
  let t1 := AudioEncoderConfig.downsampleOnce frames
  let t2 := AudioEncoderConfig.downsampleTwice frames
  let t3 := AudioEncoderConfig.framesAfterConv3 cfg frames

  let x0 : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.numMelBins, frames] } := Tensor.reshape inputFeatures #[batch, 1, cfg.numMelBins, frames]

  let x1 : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv1 cfg, t1] } :=
    Tensor.reshapeChecked (Tensor.gelu (Tensor.conv2dBias x0 m.conv2d1Weight m.conv2d1Bias #[2, 2] #[1, 1]))
      #[batch, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv1 cfg, t1]
  let x2 : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv2 cfg, t2] } :=
    Tensor.reshapeChecked (Tensor.gelu (Tensor.conv2dBias x1 m.conv2d2Weight m.conv2d2Bias #[2, 2] #[1, 1]))
      #[batch, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv2 cfg, t2]
  let x3 : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg, t3] } :=
    Tensor.reshapeChecked (Tensor.gelu (Tensor.conv2dBias x2 m.conv2d3Weight m.conv2d3Bias #[2, 2] #[1, 1]))
      #[batch, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg, t3]

  let x3t : Tensor { dtype := dtype, device := device, shape := #[batch, t3, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg] } :=
    Tensor.reshape (Tensor.permute x3 #[0, 3, 1, 2]) #[batch, t3, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg]

  let xFlat : Tensor { dtype := dtype, device := device, shape := #[batch, t3, AudioEncoderConfig.convOutInDim cfg] } :=
    Tensor.reshapeChecked x3t #[batch, t3, AudioEncoderConfig.convOutInDim cfg]
  let xEmb : Tensor { dtype := dtype, device := device, shape := #[batch, t3, cfg.dModel] } := Tensor.linear3d xFlat m.convOutWeight

  let pos : Tensor { dtype := .Float32, device := device, shape := #[t3, cfg.dModel] } := sinusoidPosition (seq := t3) (dim := cfg.dModel) (device := xEmb.device)
  let posBatch : Tensor { dtype := .Float32, device := device, shape := #[batch, t3, cfg.dModel] } := Tensor.expand (Tensor.reshape pos #[1, t3, cfg.dModel]) #[batch, t3, cfg.dModel]
  let hidden0 := Tensor.add xEmb (Tensor.castLike xEmb posBatch)

  let hidden ← m.layers.foldlM (fun h layer => layerForward layer h attnMask) hidden0

  let hidden := m.lnPost.forward3d hidden
  let hidden : Tensor { dtype := dtype, device := device, shape := #[batch, t3, cfg.dModel] } := Tensor.affine3d hidden m.proj1Weight m.proj1Bias
  let hidden := activate cfg.activationFunction hidden
  return Tensor.affine3d hidden m.proj2Weight m.proj2Bias

/-- Deterministic encoder inference, with attention dropout disabled. -/
def forwardInference {batch frames : Nat}
    (m : AudioEncoder (device := device) (dtype := dtype) cfg)
    (inputFeatures : Tensor { dtype := featureDType, device := device, shape := #[batch, cfg.numMelBins, frames] })
    (attnMask : Option (Tensor { dtype := .Int64, device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames] }) := none)
    : Tensor { dtype := dtype, device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames, cfg.outputDim] } :=
  Id.run <| forwardWith (M := Id)
    (fun layer x mask => pure (layer.forwardInference x mask)) m inputFeatures attnMask

/-- Encoder training/evaluation; training uses the configured attention dropout. -/
def forwardIO {batch frames : Nat}
    (m : AudioEncoder (device := device) (dtype := dtype) cfg)
    (inputFeatures : Tensor { dtype := featureDType, device := device, shape := #[batch, cfg.numMelBins, frames] })
    (training : Bool)
    (attnMask : Option (Tensor { dtype := .Int64, device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames] }) := none)
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames, cfg.outputDim] }) :=
  forwardWith (fun layer x mask => layer.forwardIO x training mask) m inputFeatures attnMask

/-- Compatibility alias for deterministic inference. Use `forwardIO` for training. -/
def forward {batch frames : Nat}
    (m : AudioEncoder (device := device) (dtype := dtype) cfg)
    (inputFeatures : Tensor { dtype := featureDType, device := device, shape := #[batch, cfg.numMelBins, frames] })
    (attnMask : Option (Tensor { dtype := .Int64, device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames] }) := none)
    : Tensor { dtype := dtype, device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames, cfg.outputDim] } :=
  m.forwardInference inputFeatures attnMask

end AudioEncoder

instance {cfg : AudioEncoderConfig} {batch frames : Nat} :
    Module (AudioEncoder (device := device) (dtype := dtype) cfg)
      (Tensor { dtype := dtype, device := device, shape := #[batch, cfg.numMelBins, frames] })
      (Tensor { dtype := dtype, device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames, cfg.outputDim] }) where
  forward := fun m x => m.forwardInference x none

end torch.qwen3asr
