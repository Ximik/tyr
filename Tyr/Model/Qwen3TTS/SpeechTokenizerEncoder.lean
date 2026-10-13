/- 
  Tyr/Model/Qwen3TTS/SpeechTokenizerEncoder.lean

  Lean-native Qwen3-TTS 12Hz speech-tokenizer encoder (waveform -> codec IDs).
  This ports the upstream 12Hz encoder path built on Mimi:
  - causal SEANet-style conv encoder
  - causal transformer stack with RoPE + sliding-window attention
  - split RVQ encode (semantic + acoustic)

  Scope in this module:
  - pure Lean compute for encoder forward
  - real weight loading from `speech_tokenizer/model.safetensors`
  - first 16 code groups (1 semantic + 15 acoustic), matching
    Qwen3-TTS `encoder_valid_num_quantizers`.

  Input expected by `encode`:
  - mono waveform tensor `[batch, 1, samples]` at 24kHz.
-/
import Tyr.Typed.Neural
import Tyr.SafeTensors.Load
import Tyr.TensorStruct
import Tyr.Model.Qwen.RoPE
import Tyr.Model.Qwen3TTS.SpeechTokenizer

namespace torch.qwen3tts

private def requireTrue (ok : Bool) (msg : String) : IO Unit := do
  unless ok do
    throw (IO.userError msg)

private def freeze {s : Shape} (t : STensor s) : STensor s :=
  Tensor.setRequiresGrad t false

private def loadFrozen (path : String) (name : String) (s : Shape) : IO (STensor s) := do
  let t ← safetensors.loadTensorCandidates path #[name] s
  pure (freeze t)

private def ceilDiv (x d : UInt64) : UInt64 :=
  if d == 0 then 0 else (x + d - 1) / d

private def framesAfterDown1 (samples : UInt64) : UInt64 := ceilDiv samples 4
private def framesAfterDown2 (samples : UInt64) : UInt64 := ceilDiv (framesAfterDown1 samples) 5
private def framesAfterDown3 (samples : UInt64) : UInt64 := ceilDiv (framesAfterDown2 samples) 6
def framesBeforeFinalDownsample (samples : UInt64) : UInt64 := ceilDiv (framesAfterDown3 samples) 8
private def framesAfterDown4 (samples : UInt64) : UInt64 := framesBeforeFinalDownsample samples
def encodedFrames (samples : UInt64) : UInt64 := ceilDiv (framesAfterDown4 samples) 2

private def effectiveKernel (kernel dilation : UInt64) : UInt64 :=
  (kernel - 1) * dilation + 1

private def paddingTotal (kernel stride dilation : UInt64) : UInt64 :=
  effectiveKernel kernel dilation - stride

private def extraPaddingForCausalConv (inputLen kernel stride dilation : UInt64) : UInt64 :=
  let outLen := ceilDiv inputLen stride
  if outLen == 0 then
    0
  else
    let neededLen := (outLen - 1) * stride + effectiveKernel kernel dilation
    let baseLen := inputLen + paddingTotal kernel stride dilation
    if neededLen >= baseLen then neededLen - baseLen else 0

private def addBias3d {batch channels frames : UInt64}
    (x : STensor #[batch, channels, frames])
    (bias : STensor #[channels]) : STensor #[batch, channels, frames] :=
  let b : STensor #[1, channels, 1] := Tensor.reshapeChecked bias #[1, channels, 1]
  x + Tensor.expand b #[batch, channels, frames]

private def scale3d {batch seq dim : UInt64}
    (x : STensor #[batch, seq, dim])
    (scale : STensor #[dim]) : STensor #[batch, seq, dim] :=
  let s : STensor #[1, 1, dim] := Tensor.reshapeChecked scale #[1, 1, dim]
  x * Tensor.expand s #[batch, seq, dim]

private def clampMin1d {n : UInt64} (x : STensor #[n]) (minVal : Float) : STensor #[n] :=
  let minT := Tensor.castLike x (Tensor.full #[n] minVal false x.device)
  Tensor.whereSelect (Tensor.ltScalar x minVal) minT x

private def stack2dDyn {batch frames k : UInt64}
    (tensors : Array (DTensor #[batch, frames] .Int64)) : DTensor #[k, batch, frames] .Int64 :=
  if tensors.isEmpty then
    Tensor.reshapeChecked (Tensor.fullInt #[0, batch, frames] 0) #[k, batch, frames]
  else
    let unsq := tensors.map (fun t => Tensor.unsqueeze t 0)
    Tensor.catManyChecked (unsq.map Tensor.packShape) 0 #[k, batch, frames]

/-! ## Parameters -/

structure EncoderResBlock (inC hiddenC : UInt64) where
  conv1Weight : STensor #[hiddenC, inC, 3]
  conv1Bias : STensor #[hiddenC]
  conv2Weight : STensor #[inC, hiddenC, 1]
  conv2Bias : STensor #[inC]
  deriving TensorStruct

structure EncoderTransformerLayer where
  inputLayerNormWeight : STensor #[512]
  inputLayerNormBias : STensor #[512]
  qProj : STensor #[512, 512]
  kProj : STensor #[512, 512]
  vProj : STensor #[512, 512]
  oProj : STensor #[512, 512]
  selfAttnLayerScale : STensor #[512]
  postAttentionLayerNormWeight : STensor #[512]
  postAttentionLayerNormBias : STensor #[512]
  fc1 : STensor #[2048, 512]
  fc2 : STensor #[512, 2048]
  mlpLayerScale : STensor #[512]
  deriving TensorStruct

structure EncoderCodebook where
  clusterUsage : STensor #[2048]
  embedSum : STensor #[2048, 256]
  deriving TensorStruct, Inhabited

structure EncodeRVQ where
  inputProj : STensor #[256, 512, 1]
  codebooks : Array EncoderCodebook
  deriving TensorStruct

structure SpeechTokenizer12HzEncoder where
  conv0Weight : STensor #[64, 1, 7]
  conv0Bias : STensor #[64]

  res1 : EncoderResBlock 64 32
  down1Weight : STensor #[128, 64, 8]
  down1Bias : STensor #[128]

  res2 : EncoderResBlock 128 64
  down2Weight : STensor #[256, 128, 10]
  down2Bias : STensor #[256]

  res3 : EncoderResBlock 256 128
  down3Weight : STensor #[512, 256, 12]
  down3Bias : STensor #[512]

  res4 : EncoderResBlock 512 256
  down4Weight : STensor #[1024, 512, 16]
  down4Bias : STensor #[1024]

  finalConvWeight : STensor #[512, 1024, 3]
  finalConvBias : STensor #[512]

  transformerLayers : Array EncoderTransformerLayer
  downsampleWeight : STensor #[512, 512, 4]

  semanticRVQ : EncodeRVQ
  acousticRVQ : EncodeRVQ

  inputSampleRate : UInt64 := 24000
  encodeDownsampleRate : UInt64 := 1920
  deriving TensorStruct

namespace SpeechTokenizer12HzEncoder

private def loadResBlock
    (weightsPath : String)
    (namePrefix : String)
    (inC hiddenC : UInt64)
    : IO (EncoderResBlock inC hiddenC) := do
  let conv1Weight ← loadFrozen weightsPath s!"{namePrefix}.1.conv.weight" #[hiddenC, inC, 3]
  let conv1Bias ← loadFrozen weightsPath s!"{namePrefix}.1.conv.bias" #[hiddenC]
  let conv2Weight ← loadFrozen weightsPath s!"{namePrefix}.3.conv.weight" #[inC, hiddenC, 1]
  let conv2Bias ← loadFrozen weightsPath s!"{namePrefix}.3.conv.bias" #[inC]
  pure { conv1Weight, conv1Bias, conv2Weight, conv2Bias }

private def loadTransformerLayer (weightsPath : String) (i : Nat) : IO EncoderTransformerLayer := do
  let p := s!"encoder.encoder_transformer.layers.{i}"
  let inputLayerNormWeight ← loadFrozen weightsPath s!"{p}.input_layernorm.weight" #[512]
  let inputLayerNormBias ← loadFrozen weightsPath s!"{p}.input_layernorm.bias" #[512]
  let qProj ← loadFrozen weightsPath s!"{p}.self_attn.q_proj.weight" #[512, 512]
  let kProj ← loadFrozen weightsPath s!"{p}.self_attn.k_proj.weight" #[512, 512]
  let vProj ← loadFrozen weightsPath s!"{p}.self_attn.v_proj.weight" #[512, 512]
  let oProj ← loadFrozen weightsPath s!"{p}.self_attn.o_proj.weight" #[512, 512]
  let selfAttnLayerScale ← loadFrozen weightsPath s!"{p}.self_attn_layer_scale.scale" #[512]
  let postAttentionLayerNormWeight ← loadFrozen weightsPath s!"{p}.post_attention_layernorm.weight" #[512]
  let postAttentionLayerNormBias ← loadFrozen weightsPath s!"{p}.post_attention_layernorm.bias" #[512]
  let fc1 ← loadFrozen weightsPath s!"{p}.mlp.fc1.weight" #[2048, 512]
  let fc2 ← loadFrozen weightsPath s!"{p}.mlp.fc2.weight" #[512, 2048]
  let mlpLayerScale ← loadFrozen weightsPath s!"{p}.mlp_layer_scale.scale" #[512]
  pure {
    inputLayerNormWeight
    inputLayerNormBias
    qProj
    kProj
    vProj
    oProj
    selfAttnLayerScale
    postAttentionLayerNormWeight
    postAttentionLayerNormBias
    fc1
    fc2
    mlpLayerScale
  }

private def loadEncoderCodebook (weightsPath : String) (namePrefix : String) : IO EncoderCodebook := do
  let clusterUsage ← loadFrozen weightsPath s!"{namePrefix}.cluster_usage" #[2048]
  let embedSum ← loadFrozen weightsPath s!"{namePrefix}.embed_sum" #[2048, 256]
  pure { clusterUsage, embedSum }

private def loadEncodeRVQ
    (weightsPath : String)
    (namePrefix : String)
    (numCodebooks : Nat)
    : IO EncodeRVQ := do
  let inputProj ← loadFrozen weightsPath s!"{namePrefix}.input_proj.weight" #[256, 512, 1]
  let mut codebooks : Array EncoderCodebook := #[]
  for i in [:numCodebooks] do
    let cb ← loadEncoderCodebook weightsPath s!"{namePrefix}.layers.{i}.codebook"
    codebooks := codebooks.push cb
  pure { inputProj, codebooks }

/-- Load 12Hz speech-tokenizer encoder weights from `speech_tokenizer` directory. -/
def loadFromDir (speechTokenizerDir : String) (device : Device := Device.CPU) : IO SpeechTokenizer12HzEncoder := do
  let cfg ← SpeechTokenizer12HzConfig.loadFromFile s!"{speechTokenizerDir}/config.json"
  let counts ← cfg.encoderCodebookCounts
  requireTrue (counts == (1, 15))
    "Fixed 16-group encoder requires encoder_valid_num_quantizers=16 and one semantic quantizer"

  let weightsPath := s!"{speechTokenizerDir}/model.safetensors"

  let conv0Weight ← loadFrozen weightsPath "encoder.encoder.layers.0.conv.weight" #[64, 1, 7]
  let conv0Bias ← loadFrozen weightsPath "encoder.encoder.layers.0.conv.bias" #[64]
  let res1 ← loadResBlock weightsPath "encoder.encoder.layers.1.block" 64 32
  let down1Weight ← loadFrozen weightsPath "encoder.encoder.layers.3.conv.weight" #[128, 64, 8]
  let down1Bias ← loadFrozen weightsPath "encoder.encoder.layers.3.conv.bias" #[128]

  let res2 ← loadResBlock weightsPath "encoder.encoder.layers.4.block" 128 64
  let down2Weight ← loadFrozen weightsPath "encoder.encoder.layers.6.conv.weight" #[256, 128, 10]
  let down2Bias ← loadFrozen weightsPath "encoder.encoder.layers.6.conv.bias" #[256]

  let res3 ← loadResBlock weightsPath "encoder.encoder.layers.7.block" 256 128
  let down3Weight ← loadFrozen weightsPath "encoder.encoder.layers.9.conv.weight" #[512, 256, 12]
  let down3Bias ← loadFrozen weightsPath "encoder.encoder.layers.9.conv.bias" #[512]

  let res4 ← loadResBlock weightsPath "encoder.encoder.layers.10.block" 512 256
  let down4Weight ← loadFrozen weightsPath "encoder.encoder.layers.12.conv.weight" #[1024, 512, 16]
  let down4Bias ← loadFrozen weightsPath "encoder.encoder.layers.12.conv.bias" #[1024]

  let finalConvWeight ← loadFrozen weightsPath "encoder.encoder.layers.14.conv.weight" #[512, 1024, 3]
  let finalConvBias ← loadFrozen weightsPath "encoder.encoder.layers.14.conv.bias" #[512]

  let mut transformerLayers : Array EncoderTransformerLayer := #[]
  for i in [:8] do
    let layer ← loadTransformerLayer weightsPath i
    transformerLayers := transformerLayers.push layer

  let downsampleWeight ← loadFrozen weightsPath "encoder.downsample.conv.weight" #[512, 512, 4]
  let semanticRVQ ← loadEncodeRVQ weightsPath "encoder.quantizer.semantic_residual_vector_quantizer" 1
  -- Match Qwen3-TTS `encoder_valid_num_quantizers=16`: first 15 acoustic groups.
  let acousticRVQ ← loadEncodeRVQ weightsPath "encoder.quantizer.acoustic_residual_vector_quantizer" 15

  let enc : SpeechTokenizer12HzEncoder := {
    conv0Weight
    conv0Bias
    res1
    down1Weight
    down1Bias
    res2
    down2Weight
    down2Bias
    res3
    down3Weight
    down3Bias
    res4
    down4Weight
    down4Bias
    finalConvWeight
    finalConvBias
    transformerLayers
    downsampleWeight
    semanticRVQ
    acousticRVQ
    inputSampleRate := 24000
    encodeDownsampleRate := 1920
  }
  pure (TensorStruct.map (fun t => t.to device) enc)

/-- Load 12Hz-family speech-tokenizer encoder weights while allowing
    tokenizer variants with different quantizer counts. -/
def loadFromDirFlexible (speechTokenizerDir : String) (device : Device := Device.CPU) : IO SpeechTokenizer12HzEncoder := do
  let cfg ← SpeechTokenizer12HzConfig.loadFromFile s!"{speechTokenizerDir}/config.json"
  let (semanticLayers, acousticLayers) ← cfg.encoderCodebookCounts

  let weightsPath := s!"{speechTokenizerDir}/model.safetensors"

  let conv0Weight ← loadFrozen weightsPath "encoder.encoder.layers.0.conv.weight" #[64, 1, 7]
  let conv0Bias ← loadFrozen weightsPath "encoder.encoder.layers.0.conv.bias" #[64]
  let res1 ← loadResBlock weightsPath "encoder.encoder.layers.1.block" 64 32
  let down1Weight ← loadFrozen weightsPath "encoder.encoder.layers.3.conv.weight" #[128, 64, 8]
  let down1Bias ← loadFrozen weightsPath "encoder.encoder.layers.3.conv.bias" #[128]

  let res2 ← loadResBlock weightsPath "encoder.encoder.layers.4.block" 128 64
  let down2Weight ← loadFrozen weightsPath "encoder.encoder.layers.6.conv.weight" #[256, 128, 10]
  let down2Bias ← loadFrozen weightsPath "encoder.encoder.layers.6.conv.bias" #[256]

  let res3 ← loadResBlock weightsPath "encoder.encoder.layers.7.block" 256 128
  let down3Weight ← loadFrozen weightsPath "encoder.encoder.layers.9.conv.weight" #[512, 256, 12]
  let down3Bias ← loadFrozen weightsPath "encoder.encoder.layers.9.conv.bias" #[512]

  let res4 ← loadResBlock weightsPath "encoder.encoder.layers.10.block" 512 256
  let down4Weight ← loadFrozen weightsPath "encoder.encoder.layers.12.conv.weight" #[1024, 512, 16]
  let down4Bias ← loadFrozen weightsPath "encoder.encoder.layers.12.conv.bias" #[1024]

  let finalConvWeight ← loadFrozen weightsPath "encoder.encoder.layers.14.conv.weight" #[512, 1024, 3]
  let finalConvBias ← loadFrozen weightsPath "encoder.encoder.layers.14.conv.bias" #[512]

  let mut transformerLayers : Array EncoderTransformerLayer := #[]
  for i in [:8] do
    let layer ← loadTransformerLayer weightsPath i
    transformerLayers := transformerLayers.push layer

  let downsampleWeight ← loadFrozen weightsPath "encoder.downsample.conv.weight" #[512, 512, 4]
  let semanticRVQ ← loadEncodeRVQ weightsPath "encoder.quantizer.semantic_residual_vector_quantizer" semanticLayers
  let acousticRVQ ← loadEncodeRVQ weightsPath "encoder.quantizer.acoustic_residual_vector_quantizer" acousticLayers

  let enc : SpeechTokenizer12HzEncoder := {
    conv0Weight
    conv0Bias
    res1
    down1Weight
    down1Bias
    res2
    down2Weight
    down2Bias
    res3
    down3Weight
    down3Bias
    res4
    down4Weight
    down4Bias
    finalConvWeight
    finalConvBias
    transformerLayers
    downsampleWeight
    semanticRVQ
    acousticRVQ
    inputSampleRate := cfg.inputSampleRate
    encodeDownsampleRate := cfg.encodeDownsampleRate
  }
  pure (TensorStruct.map (fun t => t.to device) enc)

def numSemanticCodeGroups (m : SpeechTokenizer12HzEncoder) : UInt64 :=
  m.semanticRVQ.codebooks.size.toUInt64

def numAcousticCodeGroups (m : SpeechTokenizer12HzEncoder) : UInt64 :=
  m.acousticRVQ.codebooks.size.toUInt64

def numCodeGroups (m : SpeechTokenizer12HzEncoder) : UInt64 :=
  numSemanticCodeGroups m + numAcousticCodeGroups m

private def causalConvConstant {batch inC outC inLen outLen kernel : UInt64}
    (x : STensor #[batch, inC, inLen])
    (weight : STensor #[outC, inC, kernel])
    (bias : Option (STensor #[outC]) := none)
    (stride : UInt64 := 1)
    (dilation : UInt64 := 1)
    : STensor #[batch, outC, outLen] :=
  let leftPad : UInt64 := paddingTotal kernel stride dilation
  let extraPad : UInt64 := extraPaddingForCausalConv inLen kernel stride dilation
  let left : STensor #[batch, inC, leftPad] := Tensor.castLike x (Tensor.zeros #[batch, inC, leftPad] false x.device)
  let right : STensor #[batch, inC, extraPad] := Tensor.castLike x (Tensor.zeros #[batch, inC, extraPad] false x.device)
  let xPad0 : STensor #[batch, inC, inLen + leftPad] := Tensor.reshapeChecked (Tensor.cat left x 2) #[batch, inC, inLen + leftPad]
  let xPad : STensor #[batch, inC, inLen + leftPad + extraPad] := Tensor.reshapeChecked (Tensor.cat xPad0 right 2) #[batch, inC, inLen + leftPad + extraPad]
  let y0 : STensor #[batch, outC, outLen] :=
    Tensor.reshapeChecked (Tensor.conv1d xPad weight stride 0 dilation) #[batch, outC, outLen]
  match bias with
  | some b => addBias3d y0 b
  | none => y0

private def causalConvReplicate {batch inC outC inLen outLen kernel : UInt64}
    (x : STensor #[batch, inC, inLen])
    (weight : STensor #[outC, inC, kernel])
    (bias : Option (STensor #[outC]) := none)
    (stride : UInt64 := 1)
    (dilation : UInt64 := 1)
    : STensor #[batch, outC, outLen] :=
  let leftPad : UInt64 := paddingTotal kernel stride dilation
  let extraPad : UInt64 := extraPaddingForCausalConv inLen kernel stride dilation
  let first : STensor #[batch, inC, 1] := Tensor.sliceChecked x 2 0 1 #[batch, inC, 1]
  let last : STensor #[batch, inC, 1] := Tensor.sliceChecked x 2 (inLen - 1) 1 #[batch, inC, 1]
  let left : STensor #[batch, inC, leftPad] := Tensor.expand first #[batch, inC, leftPad]
  let right : STensor #[batch, inC, extraPad] :=
    if extraPad == 0 then
      Tensor.castLike x (Tensor.zeros #[batch, inC, extraPad] false x.device)
    else
      Tensor.expand last #[batch, inC, extraPad]
  let xPad0 : STensor #[batch, inC, inLen + leftPad] := Tensor.reshapeChecked (Tensor.cat left x 2) #[batch, inC, inLen + leftPad]
  let xPad : STensor #[batch, inC, inLen + leftPad + extraPad] := Tensor.reshapeChecked (Tensor.cat xPad0 right 2) #[batch, inC, inLen + leftPad + extraPad]
  let y0 : STensor #[batch, outC, outLen] :=
    Tensor.reshapeChecked (Tensor.conv1d xPad weight stride 0 dilation) #[batch, outC, outLen]
  match bias with
  | some b => addBias3d y0 b
  | none => y0

private def forwardResBlock {batch frames inC hiddenC : UInt64}
    (blk : EncoderResBlock inC hiddenC)
    (x : STensor #[batch, inC, frames]) : STensor #[batch, inC, frames] :=
  let residual := x
  let h0 : STensor #[batch, inC, frames] := Tensor.Preserving.elu x
  let h1 : STensor #[batch, hiddenC, frames] :=
    causalConvConstant h0 blk.conv1Weight (some blk.conv1Bias) 1 1
  let h2 : STensor #[batch, hiddenC, frames] := Tensor.Preserving.elu h1
  let h3 : STensor #[batch, inC, frames] :=
    causalConvConstant h2 blk.conv2Weight (some blk.conv2Bias) 1 1
  residual + h3

private def forwardAttention {batch seq : UInt64}
    (layer : EncoderTransformerLayer)
    (x : STensor #[batch, seq, 512])
    (cos sin : DTensor #[seq, 32] .Float32) : STensor #[batch, seq, 512] :=
  let q0 : STensor #[batch, seq, 512] := Tensor.linear3d x layer.qProj
  let k0 : STensor #[batch, seq, 512] := Tensor.linear3d x layer.kProj
  let v0 : STensor #[batch, seq, 512] := Tensor.linear3d x layer.vProj

  let q : STensor #[batch, seq, 8, 64] := Tensor.reshapeChecked q0 #[batch, seq, 8, 64]
  let k : STensor #[batch, seq, 8, 64] := Tensor.reshapeChecked k0 #[batch, seq, 8, 64]
  let v : STensor #[batch, seq, 8, 64] := Tensor.reshapeChecked v0 #[batch, seq, 8, 64]

  let q := Tensor.applyRotary q cos sin
  let k := Tensor.applyRotary k cos sin

  let qh : STensor #[batch, 8, seq, 64] := Tensor.splitHeads q
  let kh : STensor #[batch, 8, seq, 64] := Tensor.splitHeads k
  let vh : STensor #[batch, 8, seq, 64] := Tensor.splitHeads v

  let attn : STensor #[batch, 8, seq, 64] :=
    Tensor.attentionWindow qh kh vh 0.0 true true 250
  let attn : STensor #[batch, seq, 8, 64] := Tensor.joinHeads attn
  let attn : STensor #[batch, seq, 512] := Tensor.reshapeChecked attn #[batch, seq, 512]
  Tensor.linear3d attn layer.oProj

private def forwardTransformerLayer {batch seq : UInt64}
    (layer : EncoderTransformerLayer)
    (x : STensor #[batch, seq, 512])
    (cos sin : DTensor #[seq, 32] .Float32)
    : STensor #[batch, seq, 512] :=
  let residual1 := x
  let h1 : STensor #[batch, seq, 512] :=
    Tensor.layerNorm x layer.inputLayerNormWeight layer.inputLayerNormBias 1e-5
  let h2 : STensor #[batch, seq, 512] := forwardAttention layer h1 cos sin
  let h3 : STensor #[batch, seq, 512] := residual1 + scale3d h2 layer.selfAttnLayerScale

  let residual2 := h3
  let h4 : STensor #[batch, seq, 512] :=
    Tensor.layerNorm h3 layer.postAttentionLayerNormWeight layer.postAttentionLayerNormBias 1e-5
  let h5 : STensor #[batch, seq, 2048] := Tensor.linear3d h4 layer.fc1
  let h6 : STensor #[batch, seq, 2048] := Tensor.Preserving.gelu h5
  let h7 : STensor #[batch, seq, 512] := Tensor.linear3d h6 layer.fc2
  residual2 + scale3d h7 layer.mlpLayerScale

private def codebookEmbedding (cb : EncoderCodebook) : STensor #[2048, 256] :=
  let usage : STensor #[2048] := clampMin1d cb.clusterUsage 1e-5
  let usage2d : STensor #[2048, 256] := Tensor.expand (Tensor.reshapeChecked usage #[2048, 1]) #[2048, 256]
  Tensor.Preserving.div cb.embedSum usage2d

private def quantizeWithCodebook {batch frames : UInt64}
    (cb : EncoderCodebook)
    (x : STensor #[batch, 256, frames])
    : DTensor #[batch, frames] .Int64 × STensor #[batch, 256, frames] :=
  let embed : STensor #[2048, 256] := codebookEmbedding cb
  let xFlat : STensor #[batch * frames, 256] :=
    Tensor.reshapeChecked (Tensor.transpose x 1 2) #[batch * frames, 256]

  let x2 : STensor #[batch * frames, 1] := Tensor.reshapeChecked (Tensor.Preserving.sumDim (xFlat * xFlat) 1 true) #[batch * frames, 1]
  let e2 : STensor #[2048, 1] := Tensor.reshapeChecked (Tensor.Preserving.sumDim (embed * embed) 1 true) #[2048, 1]
  let dot : STensor #[batch * frames, 2048] := Tensor.mm xFlat (Tensor.transpose2d embed)
  let x2e : STensor #[batch * frames, 2048] := Tensor.expand x2 #[batch * frames, 2048]
  let e2Row : STensor #[1, 2048] := Tensor.reshapeChecked (Tensor.transpose2d e2) #[1, 2048]
  let e2e : STensor #[batch * frames, 2048] := Tensor.expand e2Row #[batch * frames, 2048]
  let dist : STensor #[batch * frames, 2048] := x2e + e2e - Tensor.Preserving.mulScalar dot 2.0

  let idxFlat : DTensor #[batch * frames] .Int64 := Tensor.reshapeChecked (Tensor.argmax (Tensor.Preserving.mulScalar dist (-1.0)) 1) #[batch * frames]
  let quantFlat : STensor #[batch * frames, 256] := Tensor.embedding1d idxFlat embed
  let quant : STensor #[batch, 256, frames] :=
    Tensor.reshapeChecked (Tensor.transpose (Tensor.reshapeChecked quantFlat #[batch, frames, 256]) 1 2) #[batch, 256, frames]
  let idx : DTensor #[batch, frames] .Int64 := Tensor.reshapeChecked idxFlat #[batch, frames]
  (idx, quant)

private def rvqEncode {batch frames k : UInt64}
    (rvq : EncodeRVQ)
    (embeddings : STensor #[batch, 512, frames])
    (numLayers : Nat)
    : DTensor #[k, batch, frames] .Int64 := Id.run do
  let mut residual : STensor #[batch, 256, frames] :=
    Tensor.reshapeChecked (Tensor.conv1d embeddings rvq.inputProj 1 0 1) #[batch, 256, frames]
  let mut allIndices : Array (DTensor #[batch, frames] .Int64) := #[]
  for i in [:numLayers] do
    let cb := rvq.codebooks[i]!
    let (idx, quantized) := quantizeWithCodebook cb residual
    residual := residual - quantized
    allIndices := allIndices.push idx
  stack2dDyn (batch := batch) (frames := frames) (k := k) allIndices

/-- Encode waveform `[batch,1,samples]` to codec IDs `[batch,16,encodedFrames(samples)]`.
    `ropeStartPos` is in the pre-downsample transformer timeline (stride 960 samples). -/
def encodeWithRoPEOffset {batch samples : UInt64}
    (m : SpeechTokenizer12HzEncoder)
    (audio : STensor #[batch, 1, samples])
    (ropeStartPos : UInt64 := 0)
    : DTensor #[batch, 16, encodedFrames samples] .Int64 :=
  let t1 := framesAfterDown1 samples
  let t2 := framesAfterDown2 samples
  let t3 := framesAfterDown3 samples
  let t4 := framesAfterDown4 samples
  let t5 := encodedFrames samples

  let h0 : STensor #[batch, 64, samples] := causalConvConstant audio m.conv0Weight (some m.conv0Bias) 1 1
  let h1 : STensor #[batch, 64, samples] := forwardResBlock m.res1 h0

  let h2In : STensor #[batch, 64, samples] := Tensor.Preserving.elu h1
  let h2 : STensor #[batch, 128, t1] := causalConvConstant h2In m.down1Weight (some m.down1Bias) 4 1
  let h3 : STensor #[batch, 128, t1] := forwardResBlock m.res2 h2

  let h4In : STensor #[batch, 128, t1] := Tensor.Preserving.elu h3
  let h4 : STensor #[batch, 256, t2] := causalConvConstant h4In m.down2Weight (some m.down2Bias) 5 1
  let h5 : STensor #[batch, 256, t2] := forwardResBlock m.res3 h4

  let h6In : STensor #[batch, 256, t2] := Tensor.Preserving.elu h5
  let h6 : STensor #[batch, 512, t3] := causalConvConstant h6In m.down3Weight (some m.down3Bias) 6 1
  let h7 : STensor #[batch, 512, t3] := forwardResBlock m.res4 h6

  let h8In : STensor #[batch, 512, t3] := Tensor.Preserving.elu h7
  let h8 : STensor #[batch, 1024, t4] := causalConvConstant h8In m.down4Weight (some m.down4Bias) 8 1
  let h9In : STensor #[batch, 1024, t4] := Tensor.Preserving.elu h8
  let h9 : STensor #[batch, 512, t4] := causalConvConstant h9In m.finalConvWeight (some m.finalConvBias) 1 1

  let x0 : STensor #[batch, t4, 512] := Tensor.reshapeChecked (Tensor.transpose h9 1 2) #[batch, t4, 512]
  let ropeTotal : UInt64 := ropeStartPos + t4
  let (cosAll, sinAll) := Tensor.rotaryFreqs ropeTotal 64 10000.0
  let cosAll : DTensor #[ropeTotal, 32] .Float32 :=
    if cosAll.device == x0.device then cosAll else cosAll.to x0.device
  let sinAll : DTensor #[ropeTotal, 32] .Float32 :=
    if sinAll.device == x0.device then sinAll else sinAll.to x0.device
  let cos : DTensor #[t4, 32] .Float32 := Tensor.sliceChecked cosAll 0 ropeStartPos t4 #[t4, 32]
  let sin : DTensor #[t4, 32] .Float32 := Tensor.sliceChecked sinAll 0 ropeStartPos t4 #[t4, 32]
  let x1 := m.transformerLayers.foldl (fun h layer => forwardTransformerLayer layer h cos sin) x0
  let x2 : STensor #[batch, 512, t4] := Tensor.reshapeChecked (Tensor.transpose x1 1 2) #[batch, 512, t4]

  let x3 : STensor #[batch, 512, t5] := causalConvReplicate x2 m.downsampleWeight none 2 1

  let semanticCodes : DTensor #[1, batch, t5] .Int64 := rvqEncode (k := 1) m.semanticRVQ x3 1
  let acousticCodes : DTensor #[15, batch, t5] .Int64 := rvqEncode (k := 15) m.acousticRVQ x3 15
  let codes : DTensor #[16, batch, t5] .Int64 := Tensor.reshapeChecked (Tensor.cat semanticCodes acousticCodes 0) #[16, batch, t5]
  Tensor.reshapeChecked (Tensor.transpose codes 0 1) #[batch, 16, t5]

/-- Encode waveform `[batch,1,samples]` to tokenizer-configured codec IDs
    `[batch,codeGroups,encodedFrames(samples)]`.
    `ropeStartPos` is in the pre-downsample transformer timeline (stride 960 samples). -/
def encodeWithRoPEOffsetDynamic {batch samples : UInt64}
    (m : SpeechTokenizer12HzEncoder)
    (audio : STensor #[batch, 1, samples])
    (ropeStartPos : UInt64 := 0)
    : Sigma (fun codeGroups => DTensor #[batch, codeGroups, encodedFrames samples] .Int64) :=
  let t1 := framesAfterDown1 samples
  let t2 := framesAfterDown2 samples
  let t3 := framesAfterDown3 samples
  let t4 := framesAfterDown4 samples
  let t5 := encodedFrames samples

  let h0 : STensor #[batch, 64, samples] := causalConvConstant audio m.conv0Weight (some m.conv0Bias) 1 1
  let h1 : STensor #[batch, 64, samples] := forwardResBlock m.res1 h0

  let h2In : STensor #[batch, 64, samples] := Tensor.Preserving.elu h1
  let h2 : STensor #[batch, 128, t1] := causalConvConstant h2In m.down1Weight (some m.down1Bias) 4 1
  let h3 : STensor #[batch, 128, t1] := forwardResBlock m.res2 h2

  let h4In : STensor #[batch, 128, t1] := Tensor.Preserving.elu h3
  let h4 : STensor #[batch, 256, t2] := causalConvConstant h4In m.down2Weight (some m.down2Bias) 5 1
  let h5 : STensor #[batch, 256, t2] := forwardResBlock m.res3 h4

  let h6In : STensor #[batch, 256, t2] := Tensor.Preserving.elu h5
  let h6 : STensor #[batch, 512, t3] := causalConvConstant h6In m.down3Weight (some m.down3Bias) 6 1
  let h7 : STensor #[batch, 512, t3] := forwardResBlock m.res4 h6

  let h8In : STensor #[batch, 512, t3] := Tensor.Preserving.elu h7
  let h8 : STensor #[batch, 1024, t4] := causalConvConstant h8In m.down4Weight (some m.down4Bias) 8 1
  let h9In : STensor #[batch, 1024, t4] := Tensor.Preserving.elu h8
  let h9 : STensor #[batch, 512, t4] := causalConvConstant h9In m.finalConvWeight (some m.finalConvBias) 1 1

  let x0 : STensor #[batch, t4, 512] := Tensor.reshapeChecked (Tensor.transpose h9 1 2) #[batch, t4, 512]
  let ropeTotal : UInt64 := ropeStartPos + t4
  let (cosAll, sinAll) := Tensor.rotaryFreqs ropeTotal 64 10000.0
  let cosAll : DTensor #[ropeTotal, 32] .Float32 :=
    if cosAll.device == x0.device then cosAll else cosAll.to x0.device
  let sinAll : DTensor #[ropeTotal, 32] .Float32 :=
    if sinAll.device == x0.device then sinAll else sinAll.to x0.device
  let cos : DTensor #[t4, 32] .Float32 := Tensor.sliceChecked cosAll 0 ropeStartPos t4 #[t4, 32]
  let sin : DTensor #[t4, 32] .Float32 := Tensor.sliceChecked sinAll 0 ropeStartPos t4 #[t4, 32]
  let x1 := m.transformerLayers.foldl (fun h layer => forwardTransformerLayer layer h cos sin) x0
  let x2 : STensor #[batch, 512, t4] := Tensor.reshapeChecked (Tensor.transpose x1 1 2) #[batch, 512, t4]

  let x3 : STensor #[batch, 512, t5] := causalConvReplicate x2 m.downsampleWeight none 2 1

  let semLayers : Nat := m.semanticRVQ.codebooks.size
  let acLayers : Nat := m.acousticRVQ.codebooks.size
  let semK : UInt64 := semLayers.toUInt64
  let acK : UInt64 := acLayers.toUInt64
  let semanticCodes : DTensor #[semK, batch, t5] .Int64 := rvqEncode (k := semK) m.semanticRVQ x3 semLayers
  let acousticCodes : DTensor #[acK, batch, t5] .Int64 := rvqEncode (k := acK) m.acousticRVQ x3 acLayers
  let codes : DTensor #[semK + acK, batch, t5] .Int64 := Tensor.reshapeChecked (Tensor.cat semanticCodes acousticCodes 0) #[semK + acK, batch, t5]
  let out : DTensor #[batch, semK + acK, t5] .Int64 := Tensor.reshapeChecked (Tensor.transpose codes 0 1) #[batch, semK + acK, t5]
  ⟨semK + acK, out⟩

/-- Encode waveform `[batch,1,samples]` with zero RoPE offset. -/
def encode {batch samples : UInt64}
    (m : SpeechTokenizer12HzEncoder)
    (audio : STensor #[batch, 1, samples]) : DTensor #[batch, 16, encodedFrames samples] .Int64 :=
  encodeWithRoPEOffset m audio 0

/-- Convenience encode for mono waveform `[samples]` into frame-major `[frames,16]`.
    RoPE offset is in the pre-downsample transformer timeline. -/
def encodeMonoFrameMajorWithRoPEOffset {samples : UInt64}
    (m : SpeechTokenizer12HzEncoder)
    (audio : STensor #[samples])
    (ropeStartPos : UInt64 := 0)
    : DTensor #[encodedFrames samples, 16] .Int64 :=
  let x0 : STensor #[1, 1, samples] := Tensor.reshapeChecked audio #[1, 1, samples]
  let codes : DTensor #[1, 16, encodedFrames samples] .Int64 := encodeWithRoPEOffset m x0 ropeStartPos
  let c0 : DTensor #[1, encodedFrames samples, 16] .Int64 := Tensor.reshapeChecked (Tensor.transpose codes 1 2) #[1, encodedFrames samples, 16]
  Tensor.reshapeChecked c0 #[encodedFrames samples, 16]

/-- Convenience encode for mono waveform `[samples]` into frame-major
    `[frames,codeGroups]` with tokenizer-configured code-group count.
    RoPE offset is in the pre-downsample transformer timeline. -/
def encodeMonoFrameMajorWithRoPEOffsetDynamic {samples : UInt64}
    (m : SpeechTokenizer12HzEncoder)
    (audio : STensor #[samples])
    (ropeStartPos : UInt64 := 0)
    : Sigma (fun codeGroups => DTensor #[encodedFrames samples, codeGroups] .Int64) :=
  let x0 : STensor #[1, 1, samples] := Tensor.reshapeChecked audio #[1, 1, samples]
  let ⟨codeGroups, codes⟩ := encodeWithRoPEOffsetDynamic m x0 ropeStartPos
  let c0 : DTensor #[1, encodedFrames samples, codeGroups] .Int64 :=
    Tensor.reshapeChecked (Tensor.transpose codes 1 2) #[1, encodedFrames samples, codeGroups]
  ⟨codeGroups, Tensor.reshapeChecked c0 #[encodedFrames samples, codeGroups]⟩

/-- Convenience encode for mono waveform `[samples]` into frame-major `[frames,16]`. -/
def encodeMonoFrameMajor {samples : UInt64}
    (m : SpeechTokenizer12HzEncoder)
    (audio : STensor #[samples]) : DTensor #[encodedFrames samples, 16] .Int64 :=
  encodeMonoFrameMajorWithRoPEOffset m audio 0

/-- Stateful streaming encode state for incremental waveform -> codec conversion. -/
structure EncodeStreamState where
  chunkSamples : Nat
  leftContextSamples : Nat
  offsetSamples : Nat := 0
  history : Array Float := #[]

/-- Initialize streaming encode state. -/
def initEncodeStreamState
    (chunkSamples : Nat := 240000)
    (leftContextSamples : Nat := 288000)
    : EncodeStreamState :=
  { chunkSamples, leftContextSamples, offsetSamples := 0, history := #[] }

/-- Encode one streaming chunk and emit flat codec IDs (`rows * 16` values). -/
def pushEncodeStream
    (m : SpeechTokenizer12HzEncoder)
    (st : EncodeStreamState)
    (newSamples : Array Float)
    : IO (EncodeStreamState × Array UInt64) := do
  if newSamples.isEmpty then
    return (st, #[])

  let ctxStart : Nat := st.offsetSamples - st.history.size
  let chunk := st.history ++ newSamples
  let chunkSamplesU64 : UInt64 := chunk.size.toUInt64
  let audioChunk0 : DTensor #[chunkSamplesU64] .Float32 := Tensor.reshapeChecked (Tensor.fromFloatArray chunk) #[chunkSamplesU64]
  let audioChunk : STensor #[chunkSamplesU64] :=
    Tensor.castLike m.conv0Weight (if audioChunk0.device == m.conv0Weight.device then audioChunk0 else audioChunk0.to m.conv0Weight.device)
  let framesChunk : UInt64 := encodedFrames chunkSamplesU64
  let ropeStartPos : UInt64 := framesBeforeFinalDownsample ctxStart.toUInt64
  let codesChunk : DTensor #[framesChunk, 16] .Int64 := encodeMonoFrameMajorWithRoPEOffset m audioChunk ropeStartPos
  let flatChunk : DTensor #[framesChunk * 16] .Int64 := Tensor.reshapeChecked codesChunk #[framesChunk * 16]
  let valsChunk ← Tensor.toUInt64Array flatChunk

  let dropRows : Nat := (encodedFrames st.history.size.toUInt64).toNat
  let keepRows : Nat := (encodedFrames newSamples.size.toUInt64).toNat
  let startIdx : Nat := dropRows * 16
  let endIdx : Nat := Nat.min valsChunk.size ((dropRows + keepRows) * 16)
  let valsKeep := valsChunk.extract startIdx endIdx

  let nextOffset := st.offsetSamples + newSamples.size
  let histStart :=
    if chunk.size > st.leftContextSamples then
      chunk.size - st.leftContextSamples
    else
      0
  let nextHist := chunk.extract histStart chunk.size
  let st' : EncodeStreamState := {
    st with
      offsetSamples := nextOffset
      history := nextHist
  }
  return (st', valsKeep)

/-- Encode one streaming chunk and emit flat codec IDs (`rows * codeGroups` values). -/
def pushEncodeStreamDynamic
    (m : SpeechTokenizer12HzEncoder)
    (st : EncodeStreamState)
    (newSamples : Array Float)
    : IO (EncodeStreamState × UInt64 × Array UInt64) := do
  if newSamples.isEmpty then
    return (st, numCodeGroups m, #[])

  let ctxStart : Nat := st.offsetSamples - st.history.size
  let chunk := st.history ++ newSamples
  let chunkSamplesU64 : UInt64 := chunk.size.toUInt64
  let audioChunk0 : DTensor #[chunkSamplesU64] .Float32 := Tensor.reshapeChecked (Tensor.fromFloatArray chunk) #[chunkSamplesU64]
  let audioChunk : STensor #[chunkSamplesU64] :=
    Tensor.castLike m.conv0Weight (if audioChunk0.device == m.conv0Weight.device then audioChunk0 else audioChunk0.to m.conv0Weight.device)
  let framesChunk : UInt64 := encodedFrames chunkSamplesU64
  let ropeStartPos : UInt64 := framesBeforeFinalDownsample ctxStart.toUInt64
  let ⟨codeGroups, codesChunk⟩ := encodeMonoFrameMajorWithRoPEOffsetDynamic m audioChunk ropeStartPos
  let flatChunk : DTensor #[framesChunk * codeGroups] .Int64 := Tensor.reshapeChecked codesChunk #[framesChunk * codeGroups]
  let valsChunk ← Tensor.toUInt64Array flatChunk

  let groupsNat : Nat := codeGroups.toNat
  let dropRows : Nat := (encodedFrames st.history.size.toUInt64).toNat
  let keepRows : Nat := (encodedFrames newSamples.size.toUInt64).toNat
  let startIdx : Nat := dropRows * groupsNat
  let endIdx : Nat := Nat.min valsChunk.size ((dropRows + keepRows) * groupsNat)
  let valsKeep := valsChunk.extract startIdx endIdx

  let nextOffset := st.offsetSamples + newSamples.size
  let histStart :=
    if chunk.size > st.leftContextSamples then
      chunk.size - st.leftContextSamples
    else
      0
  let nextHist := chunk.extract histStart chunk.size
  let st' : EncodeStreamState := {
    st with
      offsetSamples := nextOffset
      history := nextHist
  }
  return (st', codeGroups, valsKeep)

end SpeechTokenizer12HzEncoder

end torch.qwen3tts
