/-
  Tyr/Model/Qwen3TTS/SpeakerEncoder.lean

  Lean4 port of the Qwen3-TTS speaker encoder (ECAPA-TDNN style):
  - TimeDelayNetBlock (Conv1d + ReLU)
  - Res2Net block
  - Squeeze-Excitation block
  - Attentive statistics pooling
  - Final 1x1 projection to speaker embedding
-/
import Tyr.Typed.Neural
import Tyr.Model.Qwen3TTS.DTypes
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Qwen3TTS.Config
import Tyr.Model.Utils

namespace torch.qwen3tts

variable {device : Device} {dtype : RealFloatingDType}

private def initWeight (shape : Shape) (fanIn : UInt64) : IO (Tensor { shape := shape, dtype := dtype.val, device := device }) :=
  Model.initWeight (device := device) shape fanIn dtype.val
private def initBias (shape : Shape) : IO (Tensor { shape := shape, dtype := dtype.val, device := device }) :=
  Model.initBias (device := device) shape dtype.val

private def c0 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encChannels.getD 0 512
private def c1 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encChannels.getD 1 512
private def c2 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encChannels.getD 2 512
private def c3 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encChannels.getD 3 512
private def c4 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encChannels.getD 4 1536

private def k0 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encKernelSizes.getD 0 5
private def k1 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encKernelSizes.getD 1 3
private def k2 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encKernelSizes.getD 2 3
private def k3 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encKernelSizes.getD 3 3
private def k4 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encKernelSizes.getD 4 1

private def d0 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encDilations.getD 0 1
private def d1 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encDilations.getD 1 2
private def d2 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encDilations.getD 2 3
private def d3 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encDilations.getD 3 4
private def d4 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encDilations.getD 4 1

private def mfaInChannels (cfg : SpeakerEncoderConfig) : UInt64 := c1 cfg + c2 cfg + c3 cfg

private def samePadding (kernelSize dilation : UInt64) : UInt64 :=
  (dilation * (kernelSize - 1)) / 2

private def addBias3d {batch channels frames : UInt64}
    (x : Tensor { shape := #[batch, channels, frames], dtype := dtype.val, device := device })
    (bias : Tensor { shape := #[channels], dtype := dtype.val, device := device }) : Tensor { shape := #[batch, channels, frames], dtype := dtype.val, device := device } :=
  let b : Tensor { shape := #[1, channels, 1], dtype := dtype.val, device := device } := Tensor.reshapeChecked bias #[1, channels, 1]
  let be : Tensor { shape := #[batch, channels, frames], dtype := dtype.val, device := device } := Tensor.expand b #[batch, channels, frames]
  Tensor.add x be

private def getOrFirst! [Nonempty α] (xs : Array α) (i : Nat) : α :=
  match xs[i]? with
  | some x => x
  | none =>
    match xs[0]? with
    | some x => x
    | none => nativeInvariantFailure "Qwen3TTS speaker encoder: empty array"

/-- TDNN block used throughout ECAPA speaker encoder: Conv1d + ReLU. -/
structure TimeDelayNetBlock (inChannels outChannels kernelSize dilation : UInt64) (dtype : RealFloatingDType := float32DType) where
  weight : Tensor { shape := #[outChannels, inChannels, kernelSize], dtype := dtype.val, device := device }
  bias : Tensor { shape := #[outChannels], dtype := dtype.val, device := device }
  deriving TensorStruct, Nonempty

namespace TimeDelayNetBlock

def init (inChannels outChannels kernelSize dilation : UInt64) (device : Device := .CPU) (dtype : RealFloatingDType := float32DType)
    : IO (TimeDelayNetBlock (dtype := dtype) (device := device) inChannels outChannels kernelSize dilation) := do
  let weight ← initWeight (device := device) #[outChannels, inChannels, kernelSize] inChannels
  let bias ← initBias (device := device) #[outChannels]
  pure { weight, bias }

def forward {batch frames inChannels outChannels kernelSize dilation : UInt64}
    (m : TimeDelayNetBlock (dtype := dtype) (device := device) inChannels outChannels kernelSize dilation)
    (x : Tensor { shape := #[batch, inChannels, frames], dtype := dtype.val, device := device }) : Tensor { shape := #[batch, outChannels, frames], dtype := dtype.val, device := device } :=
  let y : Tensor { shape := #[batch, outChannels, frames], dtype := dtype.val, device := device } :=
    Tensor.reshapeChecked (Tensor.conv1d x m.weight 1 (samePadding kernelSize dilation) dilation)
      #[batch, outChannels, frames]
  Tensor.relu (addBias3d y m.bias)

end TimeDelayNetBlock

/-- SE block: channel-wise gating from global temporal summary. -/
structure SqueezeExcitationBlock (channels seChannels : UInt64) (dtype : RealFloatingDType := float32DType) where
  conv1Weight : Tensor { shape := #[seChannels, channels, 1], dtype := dtype.val, device := device }
  conv1Bias : Tensor { shape := #[seChannels], dtype := dtype.val, device := device }
  conv2Weight : Tensor { shape := #[channels, seChannels, 1], dtype := dtype.val, device := device }
  conv2Bias : Tensor { shape := #[channels], dtype := dtype.val, device := device }
  deriving TensorStruct

namespace SqueezeExcitationBlock

def init (channels seChannels : UInt64) (device : Device := .CPU) (dtype : RealFloatingDType := float32DType) : IO (SqueezeExcitationBlock (dtype := dtype) (device := device) channels seChannels) := do
  let conv1Weight ← initWeight (device := device) #[seChannels, channels, 1] channels
  let conv2Weight ← initWeight (device := device) #[channels, seChannels, 1] seChannels
  let conv1Bias ← initBias (device := device) #[seChannels]
  let conv2Bias ← initBias (device := device) #[channels]
  pure { conv1Weight, conv1Bias, conv2Weight, conv2Bias }

def forward {batch frames channels seChannels : UInt64}
    (m : SqueezeExcitationBlock (dtype := dtype) (device := device) channels seChannels)
    (x : Tensor { shape := #[batch, channels, frames], dtype := dtype.val, device := device }) : Tensor { shape := #[batch, channels, frames], dtype := dtype.val, device := device } :=
  let mean : Tensor { shape := #[batch, channels, 1], dtype := dtype.val, device := device } := Tensor.reshapeChecked (Tensor.meanDim x 2 true) #[batch, channels, 1]
  let h1 : Tensor { shape := #[batch, seChannels, 1], dtype := dtype.val, device := device } :=
    Tensor.reshapeChecked (Tensor.conv1d mean m.conv1Weight 1 0 1) #[batch, seChannels, 1]
  let h1 := Tensor.relu (addBias3d h1 m.conv1Bias)
  let h2 : Tensor { shape := #[batch, channels, 1], dtype := dtype.val, device := device } :=
    Tensor.reshapeChecked (Tensor.conv1d h1 m.conv2Weight 1 0 1) #[batch, channels, 1]
  let gate1 := Tensor.sigmoid (addBias3d h2 m.conv2Bias)
  let gate : Tensor { shape := #[batch, channels, frames], dtype := dtype.val, device := device } := Tensor.expand gate1 #[batch, channels, frames]
  Tensor.mul x gate

end SqueezeExcitationBlock

/-- Res2Net block split over channel groups. -/
structure Res2NetBlock (channels scale kernelSize dilation : UInt64) (dtype : RealFloatingDType := float32DType) where
  blocks : Array (TimeDelayNetBlock (dtype := dtype) (device := device) (channels / scale) (channels / scale) kernelSize dilation)
  deriving TensorStruct

namespace Res2NetBlock

def init (channels scale kernelSize dilation : UInt64) (device : Device := .CPU) (dtype : RealFloatingDType := float32DType)
    : IO (Res2NetBlock (dtype := dtype) (device := device) channels scale kernelSize dilation) := do
  let nBlocks := if scale.toNat <= 1 then 0 else scale.toNat - 1
  let mut blocks : Array (TimeDelayNetBlock (dtype := dtype) (device := device) (channels / scale) (channels / scale) kernelSize dilation) := #[]
  for _ in [:nBlocks] do
    let b ← TimeDelayNetBlock.init (dtype := dtype) (device := device) (channels / scale) (channels / scale) kernelSize dilation
    blocks := blocks.push b
  pure { blocks }

def forward {batch frames channels scale kernelSize dilation : UInt64}
    (m : Res2NetBlock (dtype := dtype) (device := device) channels scale kernelSize dilation)
    (x : Tensor { shape := #[batch, channels, frames], dtype := dtype.val, device := device }) : Tensor { shape := #[batch, channels, frames], dtype := dtype.val, device := device } :=
  if scale.toNat <= 1 then
    x
  else
    Id.run do
      let partChannels := channels / scale
      let mut parts : Array (Tensor { shape := #[batch, partChannels, frames], dtype := dtype.val, device := device }) := #[]
      for i in [:scale.toNat] do
        let start := i.toUInt64 * partChannels
        let part : Tensor { shape := #[batch, partChannels, frames], dtype := dtype.val, device := device } := Tensor.sliceChecked x 1 start partChannels #[batch, partChannels, frames]
        parts := parts.push part

      let mut outputs : Array (Tensor { shape := #[batch, partChannels, frames], dtype := dtype.val, device := device }) := #[]
      let first := getOrFirst! parts 0
      outputs := outputs.push first
      let mut prev := first

      for i in [1:scale.toNat] do
        let part := getOrFirst! parts i
        let block := getOrFirst! m.blocks (i - 1)
        let input := if i == 1 then part else Tensor.add part prev
        let out := TimeDelayNetBlock.forward block input
        outputs := outputs.push out
        prev := out

      let merged : Tensor { shape := #[batch, channels, frames], dtype := dtype.val, device := device } := Tensor.catManyChecked (outputs.map Tensor.packShape) 1 #[batch, channels, frames]
      merged

end Res2NetBlock

/-- ECAPA building block: TDNN -> Res2Net -> TDNN -> SE (+ residual). -/
structure SqueezeExcitationRes2NetBlock
    (inChannels outChannels res2netScale seChannels kernelSize dilation : UInt64) (dtype : RealFloatingDType := float32DType) where
  tdnn1 : TimeDelayNetBlock (dtype := dtype) (device := device) inChannels outChannels 1 1
  res2net : Res2NetBlock (dtype := dtype) (device := device) outChannels res2netScale kernelSize dilation
  tdnn2 : TimeDelayNetBlock (dtype := dtype) (device := device) outChannels outChannels 1 1
  seBlock : SqueezeExcitationBlock (dtype := dtype) (device := device) outChannels seChannels
  deriving TensorStruct

namespace SqueezeExcitationRes2NetBlock

def init (inChannels outChannels res2netScale seChannels kernelSize dilation : UInt64) (device : Device := .CPU) (dtype : RealFloatingDType := float32DType)
    : IO (SqueezeExcitationRes2NetBlock (dtype := dtype) (device := device) inChannels outChannels res2netScale seChannels kernelSize dilation) := do
  let tdnn1 ← TimeDelayNetBlock.init (dtype := dtype) (device := device) inChannels outChannels 1 1
  let res2net ← Res2NetBlock.init (dtype := dtype) (device := device) outChannels res2netScale kernelSize dilation
  let tdnn2 ← TimeDelayNetBlock.init (dtype := dtype) (device := device) outChannels outChannels 1 1
  let seBlock ← SqueezeExcitationBlock.init (dtype := dtype) (device := device) outChannels seChannels
  pure { tdnn1, res2net, tdnn2, seBlock }

def forward {batch frames inChannels outChannels res2netScale seChannels kernelSize dilation : UInt64}
    (m : SqueezeExcitationRes2NetBlock (dtype := dtype) (device := device) inChannels outChannels res2netScale seChannels kernelSize dilation)
    (x : Tensor { shape := #[batch, inChannels, frames], dtype := dtype.val, device := device }) : Tensor { shape := #[batch, outChannels, frames], dtype := dtype.val, device := device } :=
  let residual : Tensor { shape := #[batch, outChannels, frames], dtype := dtype.val, device := device } := Tensor.reshapeChecked x #[batch, outChannels, frames]
  let h1 := TimeDelayNetBlock.forward m.tdnn1 x
  let h2 := Res2NetBlock.forward m.res2net h1
  let h3 := TimeDelayNetBlock.forward m.tdnn2 h2
  let h4 := SqueezeExcitationBlock.forward m.seBlock h3
  Tensor.add h4 residual

end SqueezeExcitationRes2NetBlock

/-- Attentive statistics pooling (mean+std with learned temporal attention). -/
structure AttentiveStatisticsPooling (channels attentionChannels : UInt64) (dtype : RealFloatingDType := float32DType) where
  tdnn : TimeDelayNetBlock (dtype := dtype) (device := device) (channels * 3) attentionChannels 1 1
  convWeight : Tensor { shape := #[channels, attentionChannels, 1], dtype := dtype.val, device := device }
  convBias : Tensor { shape := #[channels], dtype := dtype.val, device := device }
  eps : Float := 1e-12
  deriving TensorStruct

namespace AttentiveStatisticsPooling

def init (channels attentionChannels : UInt64) (device : Device := .CPU) (dtype : RealFloatingDType := float32DType)
    : IO (AttentiveStatisticsPooling (dtype := dtype) (device := device) channels attentionChannels) := do
  let tdnn ← TimeDelayNetBlock.init (dtype := dtype) (device := device) (channels * 3) attentionChannels 1 1
  let convWeight ← initWeight (device := device) #[channels, attentionChannels, 1] attentionChannels
  let convBias ← initBias (device := device) #[channels]
  pure { tdnn, convWeight, convBias }

def forward {batch frames channels attentionChannels : UInt64}
    (m : AttentiveStatisticsPooling (dtype := dtype) (device := device) channels attentionChannels)
    (x : Tensor { shape := #[batch, channels, frames], dtype := dtype.val, device := device }) : Tensor { shape := #[batch, channels * 2, 1], dtype := dtype.val, device := device } :=
  let mean : Tensor { shape := #[batch, channels], dtype := dtype.val, device := device } := Tensor.reshapeChecked (Tensor.meanDim x 2 false) #[batch, channels]
  let mean1 : Tensor { shape := #[batch, channels, 1], dtype := dtype.val, device := device } := Tensor.reshapeChecked mean #[batch, channels, 1]
  let meanRep : Tensor { shape := #[batch, channels, frames], dtype := dtype.val, device := device } := Tensor.expand mean1 #[batch, channels, frames]

  let diff := Tensor.sub x meanRep
  let var : Tensor { shape := #[batch, channels], dtype := dtype.val, device := device } := Tensor.reshapeChecked (Tensor.meanDim (Tensor.mul diff diff) 2 false) #[batch, channels]
  let std : Tensor { shape := #[batch, channels], dtype := dtype.val, device := device } := Tensor.sqrt (Tensor.addScalar var m.eps)
  let std1 : Tensor { shape := #[batch, channels, 1], dtype := dtype.val, device := device } := Tensor.reshapeChecked std #[batch, channels, 1]
  let stdRep : Tensor { shape := #[batch, channels, frames], dtype := dtype.val, device := device } := Tensor.expand std1 #[batch, channels, frames]

  let attIn12 : Tensor { shape := #[batch, channels + channels, frames], dtype := dtype.val, device := device } := Tensor.reshapeChecked (Tensor.cat x meanRep 1) #[batch, channels + channels, frames]
  let attIn : Tensor { shape := #[batch, channels * 3, frames], dtype := dtype.val, device := device } := Tensor.reshapeChecked (Tensor.cat attIn12 stdRep 1) #[batch, channels * 3, frames]
  let att0 := TimeDelayNetBlock.forward m.tdnn attIn
  let att1 := Tensor.tanh att0
  let att2 : Tensor { shape := #[batch, channels, frames], dtype := dtype.val, device := device } :=
    Tensor.reshapeChecked (Tensor.conv1d att1 m.convWeight 1 0 1) #[batch, channels, frames]
  let att3 := addBias3d att2 m.convBias
  let attention : Tensor { shape := #[batch, channels, frames], dtype := dtype.val, device := device } := Tensor.softmax att3 2

  let weightedMean : Tensor { shape := #[batch, channels], dtype := dtype.val, device := device } := by simpa using (Tensor.reshapeChecked (Tensor.sumDim (Tensor.mul attention x) 2 false) #[batch, channels])
  let weightedMean1 : Tensor { shape := #[batch, channels, 1], dtype := dtype.val, device := device } := Tensor.reshapeChecked weightedMean #[batch, channels, 1]
  let weightedMeanRep : Tensor { shape := #[batch, channels, frames], dtype := dtype.val, device := device } := Tensor.expand weightedMean1 #[batch, channels, frames]

  let centered := Tensor.sub x weightedMeanRep
  let weightedVar : Tensor { shape := #[batch, channels], dtype := dtype.val, device := device } := by simpa using (Tensor.reshapeChecked (Tensor.sumDim (Tensor.mul attention (Tensor.mul centered centered)) 2 false) #[batch, channels])
  let weightedStd : Tensor { shape := #[batch, channels], dtype := dtype.val, device := device } := Tensor.sqrt (Tensor.addScalar weightedVar m.eps)

  let pooled : Tensor { shape := #[batch, channels * 2], dtype := dtype.val, device := device } := Tensor.reshapeChecked (Tensor.cat weightedMean weightedStd 1) #[batch, channels * 2]
  Tensor.reshapeChecked pooled #[batch, channels * 2, 1]

end AttentiveStatisticsPooling

/-- Qwen3-TTS speaker encoder module. -/
structure SpeakerEncoder (cfg : SpeakerEncoderConfig) (dtype : RealFloatingDType := float32DType) where
  tdnn0 : TimeDelayNetBlock (dtype := dtype) (device := device) cfg.melDim (c0 cfg) (k0 cfg) (d0 cfg)
  block1 : SqueezeExcitationRes2NetBlock (dtype := dtype) (device := device)
    (c0 cfg) (c1 cfg) cfg.encRes2NetScale cfg.encSeChannels (k1 cfg) (d1 cfg)
  block2 : SqueezeExcitationRes2NetBlock (dtype := dtype) (device := device)
    (c1 cfg) (c2 cfg) cfg.encRes2NetScale cfg.encSeChannels (k2 cfg) (d2 cfg)
  block3 : SqueezeExcitationRes2NetBlock (dtype := dtype) (device := device)
    (c2 cfg) (c3 cfg) cfg.encRes2NetScale cfg.encSeChannels (k3 cfg) (d3 cfg)
  mfa : TimeDelayNetBlock (dtype := dtype) (device := device) (mfaInChannels cfg) (c4 cfg) (k4 cfg) (d4 cfg)
  asp : AttentiveStatisticsPooling (dtype := dtype) (device := device) (c4 cfg) cfg.encAttentionChannels
  fcWeight : Tensor { shape := #[cfg.encDim, (c4 cfg) * 2, 1], dtype := dtype.val, device := device }
  fcBias : Tensor { shape := #[cfg.encDim], dtype := dtype.val, device := device }
  deriving TensorStruct

namespace SpeakerEncoder

def init (cfg : SpeakerEncoderConfig) (device : Device := .CPU) (dtype : RealFloatingDType := float32DType) : IO (SpeakerEncoder (dtype := dtype) (device := device) cfg) := do
  if cfg.encChannels.size < 5 then
    throw <| IO.userError "speaker encoder requires at least 5 enc_channels entries"
  if cfg.encKernelSizes.size < 5 then
    throw <| IO.userError "speaker encoder requires at least 5 enc_kernel_sizes entries"
  if cfg.encDilations.size < 5 then
    throw <| IO.userError "speaker encoder requires at least 5 enc_dilations entries"
  if cfg.encRes2NetScale < 2 then
    throw <| IO.userError "speaker encoder requires enc_res2net_scale >= 2"
  if c0 cfg != c1 cfg || c1 cfg != c2 cfg || c2 cfg != c3 cfg then
    throw <| IO.userError "speaker encoder expects enc_channels[0..3] to match for residual SE-Res2Net blocks"
  if c4 cfg != mfaInChannels cfg then
    throw <| IO.userError "speaker encoder expects enc_channels[4] = enc_channels[1] + enc_channels[2] + enc_channels[3]"

  let tdnn0 ← TimeDelayNetBlock.init (dtype := dtype) (device := device) cfg.melDim (c0 cfg) (k0 cfg) (d0 cfg)
  let block1 ← SqueezeExcitationRes2NetBlock.init (dtype := dtype) (device := device)
    (c0 cfg) (c1 cfg) cfg.encRes2NetScale cfg.encSeChannels (k1 cfg) (d1 cfg)
  let block2 ← SqueezeExcitationRes2NetBlock.init (dtype := dtype) (device := device)
    (c1 cfg) (c2 cfg) cfg.encRes2NetScale cfg.encSeChannels (k2 cfg) (d2 cfg)
  let block3 ← SqueezeExcitationRes2NetBlock.init (dtype := dtype) (device := device)
    (c2 cfg) (c3 cfg) cfg.encRes2NetScale cfg.encSeChannels (k3 cfg) (d3 cfg)
  let mfa ← TimeDelayNetBlock.init (dtype := dtype) (device := device) (mfaInChannels cfg) (c4 cfg) (k4 cfg) (d4 cfg)
  let asp ← AttentiveStatisticsPooling.init (dtype := dtype) (device := device) (c4 cfg) cfg.encAttentionChannels
  let fcWeight ← initWeight (device := device) #[cfg.encDim, (c4 cfg) * 2, 1] ((c4 cfg) * 2)
  let fcBias ← initBias (device := device) #[cfg.encDim]

  pure { tdnn0, block1, block2, block3, mfa, asp, fcWeight, fcBias }

def forward {batch frames : UInt64}
    (m : SpeakerEncoder (dtype := dtype) (device := device) cfg)
    (mel : Tensor { shape := #[batch, frames, cfg.melDim], dtype := dtype.val, device := device }) : Tensor { shape := #[batch, cfg.encDim], dtype := dtype.val, device := device } :=
  let x0 : Tensor { shape := #[batch, cfg.melDim, frames], dtype := dtype.val, device := device } := Tensor.reshapeChecked (Tensor.transpose mel 1 2) #[batch, cfg.melDim, frames]
  let x1 := TimeDelayNetBlock.forward m.tdnn0 x0
  let x2 := SqueezeExcitationRes2NetBlock.forward m.block1 x1
  let x3 := SqueezeExcitationRes2NetBlock.forward m.block2 x2
  let x4 := SqueezeExcitationRes2NetBlock.forward m.block3 x3

  let agg12 : Tensor { shape := #[batch, c1 cfg + c2 cfg, frames], dtype := dtype.val, device := device } := Tensor.reshapeChecked (Tensor.cat x2 x3 1) #[batch, c1 cfg + c2 cfg, frames]
  let agg123 : Tensor { shape := #[batch, mfaInChannels cfg, frames], dtype := dtype.val, device := device } := Tensor.reshapeChecked (Tensor.cat agg12 x4 1) #[batch, mfaInChannels cfg, frames]
  let mfaOut := TimeDelayNetBlock.forward m.mfa agg123
  let pooled := AttentiveStatisticsPooling.forward m.asp mfaOut

  let y0 : Tensor { shape := #[batch, cfg.encDim, 1], dtype := dtype.val, device := device } :=
    Tensor.reshapeChecked (Tensor.conv1d pooled m.fcWeight 1 0 1) #[batch, cfg.encDim, 1]
  let y1 := addBias3d y0 m.fcBias
  Tensor.reshapeChecked y1 #[batch, cfg.encDim]

end SpeakerEncoder

instance {cfg : SpeakerEncoderConfig} {batch frames : UInt64} :
    Module (SpeakerEncoder (dtype := dtype) (device := device) cfg) (Tensor { shape := #[batch, frames, cfg.melDim], dtype := dtype.val, device := device }) (Tensor { shape := #[batch, cfg.encDim], dtype := dtype.val, device := device }) where
  forward := SpeakerEncoder.forward

end torch.qwen3tts
