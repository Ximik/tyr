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
import Tyr.SafeTensors.Exact
import Tyr.Model.Qwen3TTS.SpeechTokenizerDTypes
import Tyr.TensorStruct
import Tyr.Model.Qwen.RoPE
import Tyr.Model.Qwen3TTS.SpeechTokenizer

namespace torch.qwen3tts

variable {device : Device} {dtypes : SpeechTokenizerDTypes}

private def requireTrue (ok : Bool) (msg : String) : IO Unit := do
  unless ok do
    throw (IO.userError msg)

private def loadFrozen {dtype : ExactDType} (path : String) (name : String) (s : Shape)
    : IO (Tensor { shape := s, dtype := dtype.val, device := device }) := do
  let packed ← safetensors.loadTensorExact path name s device
  match packed.atDType? dtype with
  | .ok t => pure (Tensor.detach t)
  | .error message => throw <| IO.userError s!"Speech tokenizer tensor '{name}': {message}"

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
    (x : Tensor { shape := #[batch, channels, frames], dtype := dtypes.compute.val, device := device })
    (bias : Tensor { shape := #[channels], dtype := dtypes.compute.val, device := device }) : Tensor { shape := #[batch, channels, frames], dtype := dtypes.compute.val, device := device } :=
  let b : Tensor { shape := #[1, channels, 1], dtype := dtypes.compute.val, device := device } := Tensor.reshapeChecked bias #[1, channels, 1]
  Tensor.add x (Tensor.expand b #[batch, channels, frames])

private def scale3d {batch seq dim : UInt64}
    (x : Tensor { shape := #[batch, seq, dim], dtype := dtypes.compute.val, device := device })
    (scale : Tensor { shape := #[dim], dtype := dtypes.compute.val, device := device }) : Tensor { shape := #[batch, seq, dim], dtype := dtypes.compute.val, device := device } :=
  let s : Tensor { shape := #[1, 1, dim], dtype := dtypes.compute.val, device := device } := Tensor.reshapeChecked scale #[1, 1, dim]
  Tensor.mul x (Tensor.expand s #[batch, seq, dim])

private def clampMin1d {n : UInt64} {dtype : DType}
    [DType.Ordered dtype] [DType.Promotes dtype dtype dtype]
    (x : Tensor { shape := #[n], dtype, device := device }) (minVal : Float)
    : Tensor { shape := #[n], dtype, device := device } :=
  let minT := Tensor.castLike x (Tensor.full #[n] minVal false x.device)
  Tensor.maximum x minT

private def stack2dDyn {batch frames k : UInt64}
    (tensors : Array (Tensor { shape := #[batch, frames], dtype := .Int64, device := device })) : Tensor { shape := #[k, batch, frames], dtype := .Int64, device := device } :=
  if tensors.isEmpty then
    (Tensor.reshapeChecked (Tensor.fullInt #[0, batch, frames] 0) #[k, batch, frames]).to device
  else
    let unsq := tensors.map (fun t => Tensor.unsqueeze t 0)
    Tensor.catManyChecked (unsq.map Tensor.packShape) 0 #[k, batch, frames]

/-! ## Parameters -/

structure EncoderResBlock (inC hiddenC : UInt64) (dtypes : SpeechTokenizerDTypes := float32SpeechTokenizerDTypes) where
  conv1Weight : Tensor { shape := #[hiddenC, inC, 3], dtype := dtypes.compute.val, device := device }
  conv1Bias : Tensor { shape := #[hiddenC], dtype := dtypes.compute.val, device := device }
  conv2Weight : Tensor { shape := #[inC, hiddenC, 1], dtype := dtypes.compute.val, device := device }
  conv2Bias : Tensor { shape := #[inC], dtype := dtypes.compute.val, device := device }
  deriving TensorStruct

structure EncoderTransformerLayer (dtypes : SpeechTokenizerDTypes := float32SpeechTokenizerDTypes) where
  inputLayerNormWeight : Tensor { shape := #[512], dtype := dtypes.compute.val, device := device }
  inputLayerNormBias : Tensor { shape := #[512], dtype := dtypes.compute.val, device := device }
  qProj : Tensor { shape := #[512, 512], dtype := dtypes.compute.val, device := device }
  kProj : Tensor { shape := #[512, 512], dtype := dtypes.compute.val, device := device }
  vProj : Tensor { shape := #[512, 512], dtype := dtypes.compute.val, device := device }
  oProj : Tensor { shape := #[512, 512], dtype := dtypes.compute.val, device := device }
  selfAttnLayerScale : Tensor { shape := #[512], dtype := dtypes.compute.val, device := device }
  postAttentionLayerNormWeight : Tensor { shape := #[512], dtype := dtypes.compute.val, device := device }
  postAttentionLayerNormBias : Tensor { shape := #[512], dtype := dtypes.compute.val, device := device }
  fc1 : Tensor { shape := #[2048, 512], dtype := dtypes.compute.val, device := device }
  fc2 : Tensor { shape := #[512, 2048], dtype := dtypes.compute.val, device := device }
  mlpLayerScale : Tensor { shape := #[512], dtype := dtypes.compute.val, device := device }
  deriving TensorStruct

structure EncoderCodebook (dtypes : SpeechTokenizerDTypes := float32SpeechTokenizerDTypes) where
  clusterUsage : Tensor { shape := #[2048], dtype := dtypes.usage.val, device := device }
  embedSum : Tensor { shape := #[2048, 256], dtype := dtypes.sum.val, device := device }
  deriving TensorStruct, Nonempty

structure EncodeRVQ (dtypes : SpeechTokenizerDTypes := float32SpeechTokenizerDTypes) where
  inputProj : Tensor { shape := #[256, 512, 1], dtype := dtypes.compute.val, device := device }
  codebooks : Array (EncoderCodebook (device := device) (dtypes := dtypes))
  deriving TensorStruct

structure SpeechTokenizer12HzEncoder (dtypes : SpeechTokenizerDTypes := float32SpeechTokenizerDTypes) where
  conv0Weight : Tensor { shape := #[64, 1, 7], dtype := dtypes.compute.val, device := device }
  conv0Bias : Tensor { shape := #[64], dtype := dtypes.compute.val, device := device }

  res1 : EncoderResBlock (device := device) (dtypes := dtypes) 64 32
  down1Weight : Tensor { shape := #[128, 64, 8], dtype := dtypes.compute.val, device := device }
  down1Bias : Tensor { shape := #[128], dtype := dtypes.compute.val, device := device }

  res2 : EncoderResBlock (device := device) (dtypes := dtypes) 128 64
  down2Weight : Tensor { shape := #[256, 128, 10], dtype := dtypes.compute.val, device := device }
  down2Bias : Tensor { shape := #[256], dtype := dtypes.compute.val, device := device }

  res3 : EncoderResBlock (device := device) (dtypes := dtypes) 256 128
  down3Weight : Tensor { shape := #[512, 256, 12], dtype := dtypes.compute.val, device := device }
  down3Bias : Tensor { shape := #[512], dtype := dtypes.compute.val, device := device }

  res4 : EncoderResBlock (device := device) (dtypes := dtypes) 512 256
  down4Weight : Tensor { shape := #[1024, 512, 16], dtype := dtypes.compute.val, device := device }
  down4Bias : Tensor { shape := #[1024], dtype := dtypes.compute.val, device := device }

  finalConvWeight : Tensor { shape := #[512, 1024, 3], dtype := dtypes.compute.val, device := device }
  finalConvBias : Tensor { shape := #[512], dtype := dtypes.compute.val, device := device }

  transformerLayers : Array (EncoderTransformerLayer (device := device) (dtypes := dtypes))
  downsampleWeight : Tensor { shape := #[512, 512, 4], dtype := dtypes.compute.val, device := device }

  semanticRVQ : (EncodeRVQ (device := device) (dtypes := dtypes))
  acousticRVQ : (EncodeRVQ (device := device) (dtypes := dtypes))

  inputSampleRate : UInt64 := 24000
  encodeDownsampleRate : UInt64 := 1920
  deriving TensorStruct

namespace SpeechTokenizer12HzEncoder

private def loadResBlock
    (weightsPath : String)
    (namePrefix : String)
    (inC hiddenC : UInt64)
    : IO (EncoderResBlock (device := device) (dtypes := dtypes) inC hiddenC) := do
  let conv1Weight ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath s!"{namePrefix}.1.conv.weight" #[hiddenC, inC, 3]
  let conv1Bias ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath s!"{namePrefix}.1.conv.bias" #[hiddenC]
  let conv2Weight ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath s!"{namePrefix}.3.conv.weight" #[inC, hiddenC, 1]
  let conv2Bias ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath s!"{namePrefix}.3.conv.bias" #[inC]
  pure { conv1Weight, conv1Bias, conv2Weight, conv2Bias }

private def loadTransformerLayer (weightsPath : String) (i : Nat) : IO (EncoderTransformerLayer (device := device) (dtypes := dtypes)) := do
  let p := s!"encoder.encoder_transformer.layers.{i}"
  let inputLayerNormWeight ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath s!"{p}.input_layernorm.weight" #[512]
  let inputLayerNormBias ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath s!"{p}.input_layernorm.bias" #[512]
  let qProj ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath s!"{p}.self_attn.q_proj.weight" #[512, 512]
  let kProj ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath s!"{p}.self_attn.k_proj.weight" #[512, 512]
  let vProj ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath s!"{p}.self_attn.v_proj.weight" #[512, 512]
  let oProj ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath s!"{p}.self_attn.o_proj.weight" #[512, 512]
  let selfAttnLayerScale ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath s!"{p}.self_attn_layer_scale.scale" #[512]
  let postAttentionLayerNormWeight ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath s!"{p}.post_attention_layernorm.weight" #[512]
  let postAttentionLayerNormBias ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath s!"{p}.post_attention_layernorm.bias" #[512]
  let fc1 ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath s!"{p}.mlp.fc1.weight" #[2048, 512]
  let fc2 ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath s!"{p}.mlp.fc2.weight" #[512, 2048]
  let mlpLayerScale ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath s!"{p}.mlp_layer_scale.scale" #[512]
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

private def loadEncoderCodebook (weightsPath : String) (namePrefix : String) : IO (EncoderCodebook (device := device) (dtypes := dtypes)) := do
  let clusterUsage ← loadFrozen (dtype := dtypes.usage) weightsPath s!"{namePrefix}.cluster_usage" #[2048]
  let embedSum ← loadFrozen (dtype := dtypes.sum) weightsPath s!"{namePrefix}.embed_sum" #[2048, 256]
  pure { clusterUsage, embedSum }

private def loadEncodeRVQ
    (weightsPath : String)
    (namePrefix : String)
    (numCodebooks : Nat)
    : IO (EncodeRVQ (device := device) (dtypes := dtypes)) := do
  let inputProj ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath s!"{namePrefix}.input_proj.weight" #[256, 512, 1]
  let mut codebooks : Array (EncoderCodebook (device := device) (dtypes := dtypes)) := #[]
  for i in [:numCodebooks] do
    let cb ← loadEncoderCodebook weightsPath s!"{namePrefix}.layers.{i}.codebook"
    codebooks := codebooks.push cb
  pure { inputProj, codebooks }

/-- Load 12Hz speech-tokenizer encoder weights from `speech_tokenizer` directory. -/
def loadFromDir (speechTokenizerDir : String) (device : Device := Device.CPU) : IO (Σ dtypes : SpeechTokenizerDTypes, SpeechTokenizer12HzEncoder (device := device) dtypes) := do
  let cfg ← SpeechTokenizer12HzConfig.loadFromFile s!"{speechTokenizerDir}/config.json"
  let counts ← cfg.encoderCodebookCounts
  requireTrue (counts == (1, 15))
    "Fixed 16-group encoder requires encoder_valid_num_quantizers=16 and one semantic quantizer"

  let weightsPath := s!"{speechTokenizerDir}/model.safetensors"
  let dtypes ← SpeechTokenizerDTypes.load weightsPath "encoder.encoder.layers.0.conv.weight" #[64, 1, 7]
    "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.cluster_usage" "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum" device

  let conv0Weight ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.0.conv.weight" #[64, 1, 7]
  let conv0Bias ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.0.conv.bias" #[64]
  let res1 ← loadResBlock weightsPath "encoder.encoder.layers.1.block" 64 32
  let down1Weight ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.3.conv.weight" #[128, 64, 8]
  let down1Bias ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.3.conv.bias" #[128]

  let res2 ← loadResBlock weightsPath "encoder.encoder.layers.4.block" 128 64
  let down2Weight ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.6.conv.weight" #[256, 128, 10]
  let down2Bias ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.6.conv.bias" #[256]

  let res3 ← loadResBlock weightsPath "encoder.encoder.layers.7.block" 256 128
  let down3Weight ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.9.conv.weight" #[512, 256, 12]
  let down3Bias ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.9.conv.bias" #[512]

  let res4 ← loadResBlock weightsPath "encoder.encoder.layers.10.block" 512 256
  let down4Weight ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.12.conv.weight" #[1024, 512, 16]
  let down4Bias ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.12.conv.bias" #[1024]

  let finalConvWeight ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.14.conv.weight" #[512, 1024, 3]
  let finalConvBias ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.14.conv.bias" #[512]

  let mut transformerLayers : Array (EncoderTransformerLayer (device := device) (dtypes := dtypes)) := #[]
  for i in [:8] do
    let layer ← loadTransformerLayer weightsPath i
    transformerLayers := transformerLayers.push layer

  let downsampleWeight ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.downsample.conv.weight" #[512, 512, 4]
  let semanticRVQ ← loadEncodeRVQ weightsPath "encoder.quantizer.semantic_residual_vector_quantizer" 1
  -- Match Qwen3-TTS `encoder_valid_num_quantizers=16`: first 15 acoustic groups.
  let acousticRVQ ← loadEncodeRVQ weightsPath "encoder.quantizer.acoustic_residual_vector_quantizer" 15

  let enc : (SpeechTokenizer12HzEncoder (device := device) (dtypes := dtypes)) := {
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
  pure ⟨dtypes, enc⟩

/-- Load 12Hz-family speech-tokenizer encoder weights while allowing
    tokenizer variants with different quantizer counts. -/
def loadFromDirFlexible (speechTokenizerDir : String) (device : Device := Device.CPU) : IO (Σ dtypes : SpeechTokenizerDTypes, SpeechTokenizer12HzEncoder (device := device) dtypes) := do
  let cfg ← SpeechTokenizer12HzConfig.loadFromFile s!"{speechTokenizerDir}/config.json"
  let (semanticLayers, acousticLayers) ← cfg.encoderCodebookCounts

  let weightsPath := s!"{speechTokenizerDir}/model.safetensors"
  let dtypes ← SpeechTokenizerDTypes.load weightsPath "encoder.encoder.layers.0.conv.weight" #[64, 1, 7]
    "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.cluster_usage" "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum" device

  let conv0Weight ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.0.conv.weight" #[64, 1, 7]
  let conv0Bias ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.0.conv.bias" #[64]
  let res1 ← loadResBlock weightsPath "encoder.encoder.layers.1.block" 64 32
  let down1Weight ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.3.conv.weight" #[128, 64, 8]
  let down1Bias ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.3.conv.bias" #[128]

  let res2 ← loadResBlock weightsPath "encoder.encoder.layers.4.block" 128 64
  let down2Weight ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.6.conv.weight" #[256, 128, 10]
  let down2Bias ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.6.conv.bias" #[256]

  let res3 ← loadResBlock weightsPath "encoder.encoder.layers.7.block" 256 128
  let down3Weight ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.9.conv.weight" #[512, 256, 12]
  let down3Bias ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.9.conv.bias" #[512]

  let res4 ← loadResBlock weightsPath "encoder.encoder.layers.10.block" 512 256
  let down4Weight ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.12.conv.weight" #[1024, 512, 16]
  let down4Bias ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.12.conv.bias" #[1024]

  let finalConvWeight ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.14.conv.weight" #[512, 1024, 3]
  let finalConvBias ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.encoder.layers.14.conv.bias" #[512]

  let mut transformerLayers : Array (EncoderTransformerLayer (device := device) (dtypes := dtypes)) := #[]
  for i in [:8] do
    let layer ← loadTransformerLayer weightsPath i
    transformerLayers := transformerLayers.push layer

  let downsampleWeight ← loadFrozen (dtype := dtypes.compute.toExactDType) weightsPath "encoder.downsample.conv.weight" #[512, 512, 4]
  let semanticRVQ ← loadEncodeRVQ weightsPath "encoder.quantizer.semantic_residual_vector_quantizer" semanticLayers
  let acousticRVQ ← loadEncodeRVQ weightsPath "encoder.quantizer.acoustic_residual_vector_quantizer" acousticLayers

  let enc : (SpeechTokenizer12HzEncoder (device := device) (dtypes := dtypes)) := {
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
  pure ⟨dtypes, enc⟩

def numSemanticCodeGroups (m : (SpeechTokenizer12HzEncoder (device := device) (dtypes := dtypes))) : UInt64 :=
  m.semanticRVQ.codebooks.size.toUInt64

def numAcousticCodeGroups (m : (SpeechTokenizer12HzEncoder (device := device) (dtypes := dtypes))) : UInt64 :=
  m.acousticRVQ.codebooks.size.toUInt64

def numCodeGroups (m : (SpeechTokenizer12HzEncoder (device := device) (dtypes := dtypes))) : UInt64 :=
  numSemanticCodeGroups m + numAcousticCodeGroups m

private def causalConvConstant {batch inC outC inLen outLen kernel : UInt64}
    (x : Tensor { shape := #[batch, inC, inLen], dtype := dtypes.compute.val, device := device })
    (weight : Tensor { shape := #[outC, inC, kernel], dtype := dtypes.compute.val, device := device })
    (bias : Option (Tensor { shape := #[outC], dtype := dtypes.compute.val, device := device }) := none)
    (stride : UInt64 := 1)
    (dilation : UInt64 := 1)
    : Tensor { shape := #[batch, outC, outLen], dtype := dtypes.compute.val, device := device } :=
  let leftPad : UInt64 := paddingTotal kernel stride dilation
  let extraPad : UInt64 := extraPaddingForCausalConv inLen kernel stride dilation
  let left : Tensor { shape := #[batch, inC, leftPad], dtype := dtypes.compute.val, device := device } := Tensor.castLike x (Tensor.zeros #[batch, inC, leftPad] false x.device)
  let right : Tensor { shape := #[batch, inC, extraPad], dtype := dtypes.compute.val, device := device } := Tensor.castLike x (Tensor.zeros #[batch, inC, extraPad] false x.device)
  let xPad0 : Tensor { shape := #[batch, inC, inLen + leftPad], dtype := dtypes.compute.val, device := device } := Tensor.reshapeChecked (Tensor.cat left x 2) #[batch, inC, inLen + leftPad]
  let xPad : Tensor { shape := #[batch, inC, inLen + leftPad + extraPad], dtype := dtypes.compute.val, device := device } := Tensor.reshapeChecked (Tensor.cat xPad0 right 2) #[batch, inC, inLen + leftPad + extraPad]
  let y0 : Tensor { shape := #[batch, outC, outLen], dtype := dtypes.compute.val, device := device } :=
    Tensor.reshapeChecked (Tensor.conv1d xPad weight stride 0 dilation) #[batch, outC, outLen]
  match bias with
  | some b => addBias3d y0 b
  | none => y0

private def causalConvReplicate {batch inC outC inLen outLen kernel : UInt64}
    (x : Tensor { shape := #[batch, inC, inLen], dtype := dtypes.compute.val, device := device })
    (weight : Tensor { shape := #[outC, inC, kernel], dtype := dtypes.compute.val, device := device })
    (bias : Option (Tensor { shape := #[outC], dtype := dtypes.compute.val, device := device }) := none)
    (stride : UInt64 := 1)
    (dilation : UInt64 := 1)
    : Tensor { shape := #[batch, outC, outLen], dtype := dtypes.compute.val, device := device } :=
  let leftPad : UInt64 := paddingTotal kernel stride dilation
  let extraPad : UInt64 := extraPaddingForCausalConv inLen kernel stride dilation
  let first : Tensor { shape := #[batch, inC, 1], dtype := dtypes.compute.val, device := device } := Tensor.sliceChecked x 2 0 1 #[batch, inC, 1]
  let last : Tensor { shape := #[batch, inC, 1], dtype := dtypes.compute.val, device := device } := Tensor.sliceChecked x 2 (inLen - 1) 1 #[batch, inC, 1]
  let left : Tensor { shape := #[batch, inC, leftPad], dtype := dtypes.compute.val, device := device } := Tensor.expand first #[batch, inC, leftPad]
  let right : Tensor { shape := #[batch, inC, extraPad], dtype := dtypes.compute.val, device := device } :=
    if extraPad == 0 then
      Tensor.castLike x (Tensor.zeros #[batch, inC, extraPad] false x.device)
    else
      Tensor.expand last #[batch, inC, extraPad]
  let xPad0 : Tensor { shape := #[batch, inC, inLen + leftPad], dtype := dtypes.compute.val, device := device } := Tensor.reshapeChecked (Tensor.cat left x 2) #[batch, inC, inLen + leftPad]
  let xPad : Tensor { shape := #[batch, inC, inLen + leftPad + extraPad], dtype := dtypes.compute.val, device := device } := Tensor.reshapeChecked (Tensor.cat xPad0 right 2) #[batch, inC, inLen + leftPad + extraPad]
  let y0 : Tensor { shape := #[batch, outC, outLen], dtype := dtypes.compute.val, device := device } :=
    Tensor.reshapeChecked (Tensor.conv1d xPad weight stride 0 dilation) #[batch, outC, outLen]
  match bias with
  | some b => addBias3d y0 b
  | none => y0

private def forwardResBlock {batch frames inC hiddenC : UInt64}
    (blk : EncoderResBlock (device := device) (dtypes := dtypes) inC hiddenC)
    (x : Tensor { shape := #[batch, inC, frames], dtype := dtypes.compute.val, device := device }) : Tensor { shape := #[batch, inC, frames], dtype := dtypes.compute.val, device := device } :=
  let residual := x
  let h0 : Tensor { shape := #[batch, inC, frames], dtype := dtypes.compute.val, device := device } := Tensor.elu x
  let h1 : Tensor { shape := #[batch, hiddenC, frames], dtype := dtypes.compute.val, device := device } :=
    causalConvConstant h0 blk.conv1Weight (some blk.conv1Bias) 1 1
  let h2 : Tensor { shape := #[batch, hiddenC, frames], dtype := dtypes.compute.val, device := device } := Tensor.elu h1
  let h3 : Tensor { shape := #[batch, inC, frames], dtype := dtypes.compute.val, device := device } :=
    causalConvConstant h2 blk.conv2Weight (some blk.conv2Bias) 1 1
  Tensor.add residual h3

private def forwardAttention {batch seq : UInt64}
    (layer : (EncoderTransformerLayer (device := device) (dtypes := dtypes)))
    (x : Tensor { shape := #[batch, seq, 512], dtype := dtypes.compute.val, device := device })
    (cos sin : Tensor { shape := #[seq, 32], dtype := .Float32, device := device }) : Tensor { shape := #[batch, seq, 512], dtype := dtypes.compute.val, device := device } :=
  let q0 : Tensor { shape := #[batch, seq, 512], dtype := dtypes.compute.val, device := device } := Tensor.linear3d x layer.qProj
  let k0 : Tensor { shape := #[batch, seq, 512], dtype := dtypes.compute.val, device := device } := Tensor.linear3d x layer.kProj
  let v0 : Tensor { shape := #[batch, seq, 512], dtype := dtypes.compute.val, device := device } := Tensor.linear3d x layer.vProj

  let q : Tensor { shape := #[batch, seq, 8, 64], dtype := dtypes.compute.val, device := device } := Tensor.reshapeChecked q0 #[batch, seq, 8, 64]
  let k : Tensor { shape := #[batch, seq, 8, 64], dtype := dtypes.compute.val, device := device } := Tensor.reshapeChecked k0 #[batch, seq, 8, 64]
  let v : Tensor { shape := #[batch, seq, 8, 64], dtype := dtypes.compute.val, device := device } := Tensor.reshapeChecked v0 #[batch, seq, 8, 64]

  let q := Tensor.applyRotary q cos sin
  let k := Tensor.applyRotary k cos sin

  let qh : Tensor { shape := #[batch, 8, seq, 64], dtype := dtypes.compute.val, device := device } := Tensor.splitHeads q
  let kh : Tensor { shape := #[batch, 8, seq, 64], dtype := dtypes.compute.val, device := device } := Tensor.splitHeads k
  let vh : Tensor { shape := #[batch, 8, seq, 64], dtype := dtypes.compute.val, device := device } := Tensor.splitHeads v

  let attn : Tensor { shape := #[batch, 8, seq, 64], dtype := dtypes.compute.val, device := device } :=
    Tensor.attentionWindow qh kh vh true true 250
  let attn : Tensor { shape := #[batch, seq, 8, 64], dtype := dtypes.compute.val, device := device } := Tensor.joinHeads attn
  let attn : Tensor { shape := #[batch, seq, 512], dtype := dtypes.compute.val, device := device } := Tensor.reshapeChecked attn #[batch, seq, 512]
  Tensor.linear3d attn layer.oProj

private def forwardTransformerLayer {batch seq : UInt64}
    (layer : (EncoderTransformerLayer (device := device) (dtypes := dtypes)))
    (x : Tensor { shape := #[batch, seq, 512], dtype := dtypes.compute.val, device := device })
    (cos sin : Tensor { shape := #[seq, 32], dtype := .Float32, device := device })
    : Tensor { shape := #[batch, seq, 512], dtype := dtypes.compute.val, device := device } :=
  let residual1 := x
  let h1 : Tensor { shape := #[batch, seq, 512], dtype := dtypes.compute.val, device := device } :=
    Tensor.layerNorm x layer.inputLayerNormWeight layer.inputLayerNormBias 1e-5
  let h2 : Tensor { shape := #[batch, seq, 512], dtype := dtypes.compute.val, device := device } := forwardAttention layer h1 cos sin
  let h3 : Tensor { shape := #[batch, seq, 512], dtype := dtypes.compute.val, device := device } := Tensor.add residual1 (scale3d h2 layer.selfAttnLayerScale)

  let residual2 := h3
  let h4 : Tensor { shape := #[batch, seq, 512], dtype := dtypes.compute.val, device := device } :=
    Tensor.layerNorm h3 layer.postAttentionLayerNormWeight layer.postAttentionLayerNormBias 1e-5
  let h5 : Tensor { shape := #[batch, seq, 2048], dtype := dtypes.compute.val, device := device } := Tensor.linear3d h4 layer.fc1
  let h6 : Tensor { shape := #[batch, seq, 2048], dtype := dtypes.compute.val, device := device } := Tensor.gelu h5
  let h7 : Tensor { shape := #[batch, seq, 512], dtype := dtypes.compute.val, device := device } := Tensor.linear3d h6 layer.fc2
  Tensor.add residual2 (scale3d h7 layer.mlpLayerScale)

private def codebookEmbedding (cb : (EncoderCodebook (device := device) (dtypes := dtypes))) : Tensor { shape := #[2048, 256], dtype := dtypes.compute.val, device := device } :=
  let usage : Tensor { shape := #[2048], dtype := dtypes.usage.val, device := device } := clampMin1d cb.clusterUsage 1e-5
  let usage2d : Tensor { shape := #[2048, 256], dtype := dtypes.usage.val, device := device } := Tensor.expand (Tensor.reshapeChecked usage #[2048, 1]) #[2048, 256]
  dtypes.divideCodebook cb.embedSum usage2d

private def quantizeWithCodebook {batch frames : UInt64}
    (cb : (EncoderCodebook (device := device) (dtypes := dtypes)))
    (x : Tensor { shape := #[batch, 256, frames], dtype := dtypes.compute.val, device := device })
    : Tensor { shape := #[batch, frames], dtype := .Int64, device := device } × Tensor { shape := #[batch, 256, frames], dtype := dtypes.compute.val, device := device } :=
  let embed : Tensor { shape := #[2048, 256], dtype := dtypes.compute.val, device := device } := codebookEmbedding cb
  let xFlat : Tensor { shape := #[batch * frames, 256], dtype := dtypes.compute.val, device := device } :=
    Tensor.reshapeChecked (Tensor.transpose x 1 2) #[batch * frames, 256]

  let x2 : Tensor { shape := #[batch * frames, 1], dtype := dtypes.compute.val, device := device } := by simpa using (Tensor.reshapeChecked (Tensor.sumDim (Tensor.mul xFlat xFlat) 1 true) #[batch * frames, 1])
  let e2 : Tensor { shape := #[2048, 1], dtype := dtypes.compute.val, device := device } := by simpa using (Tensor.reshapeChecked (Tensor.sumDim (Tensor.mul embed embed) 1 true) #[2048, 1])
  let dot : Tensor { shape := #[batch * frames, 2048], dtype := dtypes.compute.val, device := device } := Tensor.mm xFlat (Tensor.transpose2d embed)
  let x2e : Tensor { shape := #[batch * frames, 2048], dtype := dtypes.compute.val, device := device } := Tensor.expand x2 #[batch * frames, 2048]
  let e2Row : Tensor { shape := #[1, 2048], dtype := dtypes.compute.val, device := device } := Tensor.reshapeChecked (Tensor.transpose2d e2) #[1, 2048]
  let e2e : Tensor { shape := #[batch * frames, 2048], dtype := dtypes.compute.val, device := device } := Tensor.expand e2Row #[batch * frames, 2048]
  let dist : Tensor { shape := #[batch * frames, 2048], dtype := dtypes.compute.val, device := device } := Tensor.sub (Tensor.add x2e e2e) (Tensor.mulScalar dot 2.0)

  let idxFlat : Tensor { shape := #[batch * frames], dtype := .Int64, device := device } := Tensor.reshapeChecked (Tensor.argmaxChecked (Tensor.mulScalar dist (-1.0)) 1) #[batch * frames]
  let quantFlat : Tensor { shape := #[batch * frames, 256], dtype := dtypes.compute.val, device := device } := Tensor.embedding1d idxFlat embed
  let quant : Tensor { shape := #[batch, 256, frames], dtype := dtypes.compute.val, device := device } :=
    Tensor.reshapeChecked (Tensor.transpose (Tensor.reshapeChecked quantFlat #[batch, frames, 256]) 1 2) #[batch, 256, frames]
  let idx : Tensor { shape := #[batch, frames], dtype := .Int64, device := device } := Tensor.reshapeChecked idxFlat #[batch, frames]
  (idx, quant)

private def rvqEncode {batch frames k : UInt64}
    (rvq : (EncodeRVQ (device := device) (dtypes := dtypes)))
    (embeddings : Tensor { shape := #[batch, 512, frames], dtype := dtypes.compute.val, device := device })
    (numLayers : Nat)
    : Tensor { shape := #[k, batch, frames], dtype := .Int64, device := device } := Id.run do
  let mut residual : Tensor { shape := #[batch, 256, frames], dtype := dtypes.compute.val, device := device } :=
    Tensor.reshapeChecked (Tensor.conv1d embeddings rvq.inputProj 1 0 1) #[batch, 256, frames]
  let mut allIndices : Array (Tensor { shape := #[batch, frames], dtype := .Int64, device := device }) := #[]
  for i in [:numLayers] do
    let cb := match rvq.codebooks[i]? with
      | some codebook => codebook
      | none => nativeInvariantFailure s!"Speech tokenizer: missing codebook {i}"
    let (idx, quantized) := quantizeWithCodebook cb residual
    residual := Tensor.sub residual quantized
    allIndices := allIndices.push idx
  stack2dDyn (batch := batch) (frames := frames) (k := k) allIndices

/-- Encode waveform `[batch,1,samples]` to codec IDs `[batch,16,encodedFrames(samples)]`.
    `ropeStartPos` is in the pre-downsample transformer timeline (stride 960 samples). -/
def encodeWithRoPEOffset {batch samples : UInt64}
    (m : (SpeechTokenizer12HzEncoder (device := device) (dtypes := dtypes)))
    (audio : Tensor { shape := #[batch, 1, samples], dtype := dtypes.compute.val, device := device })
    (ropeStartPos : UInt64 := 0)
    : Tensor { shape := #[batch, 16, encodedFrames samples], dtype := .Int64, device := device } :=
  let t1 := framesAfterDown1 samples
  let t2 := framesAfterDown2 samples
  let t3 := framesAfterDown3 samples
  let t4 := framesAfterDown4 samples
  let t5 := encodedFrames samples

  let h0 : Tensor { shape := #[batch, 64, samples], dtype := dtypes.compute.val, device := device } := causalConvConstant audio m.conv0Weight (some m.conv0Bias) 1 1
  let h1 : Tensor { shape := #[batch, 64, samples], dtype := dtypes.compute.val, device := device } := forwardResBlock m.res1 h0

  let h2In : Tensor { shape := #[batch, 64, samples], dtype := dtypes.compute.val, device := device } := Tensor.elu h1
  let h2 : Tensor { shape := #[batch, 128, t1], dtype := dtypes.compute.val, device := device } := causalConvConstant h2In m.down1Weight (some m.down1Bias) 4 1
  let h3 : Tensor { shape := #[batch, 128, t1], dtype := dtypes.compute.val, device := device } := forwardResBlock m.res2 h2

  let h4In : Tensor { shape := #[batch, 128, t1], dtype := dtypes.compute.val, device := device } := Tensor.elu h3
  let h4 : Tensor { shape := #[batch, 256, t2], dtype := dtypes.compute.val, device := device } := causalConvConstant h4In m.down2Weight (some m.down2Bias) 5 1
  let h5 : Tensor { shape := #[batch, 256, t2], dtype := dtypes.compute.val, device := device } := forwardResBlock m.res3 h4

  let h6In : Tensor { shape := #[batch, 256, t2], dtype := dtypes.compute.val, device := device } := Tensor.elu h5
  let h6 : Tensor { shape := #[batch, 512, t3], dtype := dtypes.compute.val, device := device } := causalConvConstant h6In m.down3Weight (some m.down3Bias) 6 1
  let h7 : Tensor { shape := #[batch, 512, t3], dtype := dtypes.compute.val, device := device } := forwardResBlock m.res4 h6

  let h8In : Tensor { shape := #[batch, 512, t3], dtype := dtypes.compute.val, device := device } := Tensor.elu h7
  let h8 : Tensor { shape := #[batch, 1024, t4], dtype := dtypes.compute.val, device := device } := causalConvConstant h8In m.down4Weight (some m.down4Bias) 8 1
  let h9In : Tensor { shape := #[batch, 1024, t4], dtype := dtypes.compute.val, device := device } := Tensor.elu h8
  let h9 : Tensor { shape := #[batch, 512, t4], dtype := dtypes.compute.val, device := device } := causalConvConstant h9In m.finalConvWeight (some m.finalConvBias) 1 1

  let x0 : Tensor { shape := #[batch, t4, 512], dtype := dtypes.compute.val, device := device } := Tensor.reshapeChecked (Tensor.transpose h9 1 2) #[batch, t4, 512]
  let ropeTotal : UInt64 := ropeStartPos + t4
  let (cosAll, sinAll) := Tensor.rotaryFreqs ropeTotal 64 10000.0
  let cosAll : Tensor { shape := #[ropeTotal, 32], dtype := .Float32, device := device } :=
    cosAll.to x0.device
  let sinAll : Tensor { shape := #[ropeTotal, 32], dtype := .Float32, device := device } :=
    sinAll.to x0.device
  let cos : Tensor { shape := #[t4, 32], dtype := .Float32, device := device } := Tensor.sliceChecked cosAll 0 ropeStartPos t4 #[t4, 32]
  let sin : Tensor { shape := #[t4, 32], dtype := .Float32, device := device } := Tensor.sliceChecked sinAll 0 ropeStartPos t4 #[t4, 32]
  let x1 := m.transformerLayers.foldl (fun h layer => forwardTransformerLayer layer h cos sin) x0
  let x2 : Tensor { shape := #[batch, 512, t4], dtype := dtypes.compute.val, device := device } := by simpa using (Tensor.reshapeChecked (Tensor.transpose x1 1 2) #[batch, 512, t4])

  let x3 : Tensor { shape := #[batch, 512, t5], dtype := dtypes.compute.val, device := device } := causalConvReplicate x2 m.downsampleWeight none 2 1

  let semanticCodes : Tensor { shape := #[1, batch, t5], dtype := .Int64, device := device } := rvqEncode (k := 1) m.semanticRVQ x3 1
  let acousticCodes : Tensor { shape := #[15, batch, t5], dtype := .Int64, device := device } := rvqEncode (k := 15) m.acousticRVQ x3 15
  let codes : Tensor { shape := #[16, batch, t5], dtype := .Int64, device := device } := Tensor.reshapeChecked (Tensor.cat semanticCodes acousticCodes 0) #[16, batch, t5]
  Tensor.reshapeChecked (Tensor.transpose codes 0 1) #[batch, 16, t5]

/-- Encode waveform `[batch,1,samples]` to tokenizer-configured codec IDs
    `[batch,codeGroups,encodedFrames(samples)]`.
    `ropeStartPos` is in the pre-downsample transformer timeline (stride 960 samples). -/
def encodeWithRoPEOffsetDynamic {batch samples : UInt64}
    (m : (SpeechTokenizer12HzEncoder (device := device) (dtypes := dtypes)))
    (audio : Tensor { shape := #[batch, 1, samples], dtype := dtypes.compute.val, device := device })
    (ropeStartPos : UInt64 := 0)
    : Sigma (fun codeGroups => Tensor { shape := #[batch, codeGroups, encodedFrames samples], dtype := .Int64, device := device }) :=
  let t1 := framesAfterDown1 samples
  let t2 := framesAfterDown2 samples
  let t3 := framesAfterDown3 samples
  let t4 := framesAfterDown4 samples
  let t5 := encodedFrames samples

  let h0 : Tensor { shape := #[batch, 64, samples], dtype := dtypes.compute.val, device := device } := causalConvConstant audio m.conv0Weight (some m.conv0Bias) 1 1
  let h1 : Tensor { shape := #[batch, 64, samples], dtype := dtypes.compute.val, device := device } := forwardResBlock m.res1 h0

  let h2In : Tensor { shape := #[batch, 64, samples], dtype := dtypes.compute.val, device := device } := Tensor.elu h1
  let h2 : Tensor { shape := #[batch, 128, t1], dtype := dtypes.compute.val, device := device } := causalConvConstant h2In m.down1Weight (some m.down1Bias) 4 1
  let h3 : Tensor { shape := #[batch, 128, t1], dtype := dtypes.compute.val, device := device } := forwardResBlock m.res2 h2

  let h4In : Tensor { shape := #[batch, 128, t1], dtype := dtypes.compute.val, device := device } := Tensor.elu h3
  let h4 : Tensor { shape := #[batch, 256, t2], dtype := dtypes.compute.val, device := device } := causalConvConstant h4In m.down2Weight (some m.down2Bias) 5 1
  let h5 : Tensor { shape := #[batch, 256, t2], dtype := dtypes.compute.val, device := device } := forwardResBlock m.res3 h4

  let h6In : Tensor { shape := #[batch, 256, t2], dtype := dtypes.compute.val, device := device } := Tensor.elu h5
  let h6 : Tensor { shape := #[batch, 512, t3], dtype := dtypes.compute.val, device := device } := causalConvConstant h6In m.down3Weight (some m.down3Bias) 6 1
  let h7 : Tensor { shape := #[batch, 512, t3], dtype := dtypes.compute.val, device := device } := forwardResBlock m.res4 h6

  let h8In : Tensor { shape := #[batch, 512, t3], dtype := dtypes.compute.val, device := device } := Tensor.elu h7
  let h8 : Tensor { shape := #[batch, 1024, t4], dtype := dtypes.compute.val, device := device } := causalConvConstant h8In m.down4Weight (some m.down4Bias) 8 1
  let h9In : Tensor { shape := #[batch, 1024, t4], dtype := dtypes.compute.val, device := device } := Tensor.elu h8
  let h9 : Tensor { shape := #[batch, 512, t4], dtype := dtypes.compute.val, device := device } := causalConvConstant h9In m.finalConvWeight (some m.finalConvBias) 1 1

  let x0 : Tensor { shape := #[batch, t4, 512], dtype := dtypes.compute.val, device := device } := Tensor.reshapeChecked (Tensor.transpose h9 1 2) #[batch, t4, 512]
  let ropeTotal : UInt64 := ropeStartPos + t4
  let (cosAll, sinAll) := Tensor.rotaryFreqs ropeTotal 64 10000.0
  let cosAll : Tensor { shape := #[ropeTotal, 32], dtype := .Float32, device := device } :=
    cosAll.to x0.device
  let sinAll : Tensor { shape := #[ropeTotal, 32], dtype := .Float32, device := device } :=
    sinAll.to x0.device
  let cos : Tensor { shape := #[t4, 32], dtype := .Float32, device := device } := Tensor.sliceChecked cosAll 0 ropeStartPos t4 #[t4, 32]
  let sin : Tensor { shape := #[t4, 32], dtype := .Float32, device := device } := Tensor.sliceChecked sinAll 0 ropeStartPos t4 #[t4, 32]
  let x1 := m.transformerLayers.foldl (fun h layer => forwardTransformerLayer layer h cos sin) x0
  let x2 : Tensor { shape := #[batch, 512, t4], dtype := dtypes.compute.val, device := device } := by simpa using (Tensor.reshapeChecked (Tensor.transpose x1 1 2) #[batch, 512, t4])

  let x3 : Tensor { shape := #[batch, 512, t5], dtype := dtypes.compute.val, device := device } := causalConvReplicate x2 m.downsampleWeight none 2 1

  let semLayers : Nat := m.semanticRVQ.codebooks.size
  let acLayers : Nat := m.acousticRVQ.codebooks.size
  let semK : UInt64 := semLayers.toUInt64
  let acK : UInt64 := acLayers.toUInt64
  let semanticCodes : Tensor { shape := #[semK, batch, t5], dtype := .Int64, device := device } := rvqEncode (k := semK) m.semanticRVQ x3 semLayers
  let acousticCodes : Tensor { shape := #[acK, batch, t5], dtype := .Int64, device := device } := rvqEncode (k := acK) m.acousticRVQ x3 acLayers
  let codes : Tensor { shape := #[semK + acK, batch, t5], dtype := .Int64, device := device } := Tensor.reshapeChecked (Tensor.cat semanticCodes acousticCodes 0) #[semK + acK, batch, t5]
  let out : Tensor { shape := #[batch, semK + acK, t5], dtype := .Int64, device := device } := Tensor.reshapeChecked (Tensor.transpose codes 0 1) #[batch, semK + acK, t5]
  ⟨semK + acK, out⟩

/-- Encode waveform `[batch,1,samples]` with zero RoPE offset. -/
def encode {batch samples : UInt64}
    (m : (SpeechTokenizer12HzEncoder (device := device) (dtypes := dtypes)))
    (audio : Tensor { shape := #[batch, 1, samples], dtype := dtypes.compute.val, device := device }) : Tensor { shape := #[batch, 16, encodedFrames samples], dtype := .Int64, device := device } :=
  encodeWithRoPEOffset m audio 0

/-- Convenience encode for mono waveform `[samples]` into frame-major `[frames,16]`.
    RoPE offset is in the pre-downsample transformer timeline. -/
def encodeMonoFrameMajorWithRoPEOffset {samples : UInt64}
    (m : (SpeechTokenizer12HzEncoder (device := device) (dtypes := dtypes)))
    (audio : Tensor { shape := #[samples], dtype := dtypes.compute.val, device := device })
    (ropeStartPos : UInt64 := 0)
    : Tensor { shape := #[encodedFrames samples, 16], dtype := .Int64, device := device } :=
  let x0 : Tensor { shape := #[1, 1, samples], dtype := dtypes.compute.val, device := device } := Tensor.reshapeChecked audio #[1, 1, samples]
  let codes : Tensor { shape := #[1, 16, encodedFrames samples], dtype := .Int64, device := device } := encodeWithRoPEOffset m x0 ropeStartPos
  let c0 : Tensor { shape := #[1, encodedFrames samples, 16], dtype := .Int64, device := device } := Tensor.reshapeChecked (Tensor.transpose codes 1 2) #[1, encodedFrames samples, 16]
  Tensor.reshapeChecked c0 #[encodedFrames samples, 16]

/-- Convenience encode for mono waveform `[samples]` into frame-major
    `[frames,codeGroups]` with tokenizer-configured code-group count.
    RoPE offset is in the pre-downsample transformer timeline. -/
def encodeMonoFrameMajorWithRoPEOffsetDynamic {samples : UInt64}
    (m : (SpeechTokenizer12HzEncoder (device := device) (dtypes := dtypes)))
    (audio : Tensor { shape := #[samples], dtype := dtypes.compute.val, device := device })
    (ropeStartPos : UInt64 := 0)
    : Sigma (fun codeGroups => Tensor { shape := #[encodedFrames samples, codeGroups], dtype := .Int64, device := device }) :=
  let x0 : Tensor { shape := #[1, 1, samples], dtype := dtypes.compute.val, device := device } := Tensor.reshapeChecked audio #[1, 1, samples]
  let ⟨codeGroups, codes⟩ := encodeWithRoPEOffsetDynamic m x0 ropeStartPos
  let c0 : Tensor { shape := #[1, encodedFrames samples, codeGroups], dtype := .Int64, device := device } :=
    Tensor.reshapeChecked (Tensor.transpose codes 1 2) #[1, encodedFrames samples, codeGroups]
  ⟨codeGroups, Tensor.reshapeChecked c0 #[encodedFrames samples, codeGroups]⟩

/-- Convenience encode for mono waveform `[samples]` into frame-major `[frames,16]`. -/
def encodeMonoFrameMajor {samples : UInt64}
    (m : (SpeechTokenizer12HzEncoder (device := device) (dtypes := dtypes)))
    (audio : Tensor { shape := #[samples], dtype := dtypes.compute.val, device := device }) : Tensor { shape := #[encodedFrames samples, 16], dtype := .Int64, device := device } :=
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
    (m : (SpeechTokenizer12HzEncoder (device := device) (dtypes := dtypes)))
    (st : EncodeStreamState)
    (newSamples : Array Float)
    : IO (EncodeStreamState × Array UInt64) := do
  if newSamples.isEmpty then
    return (st, #[])

  let ctxStart : Nat := st.offsetSamples - st.history.size
  let chunk := st.history ++ newSamples
  let chunkSamplesU64 : UInt64 := chunk.size.toUInt64
  let audioChunk0 : Tensor { shape := #[chunkSamplesU64], dtype := .Float32, device := .CPU } := Tensor.reshapeChecked (Tensor.fromFloatArray chunk) #[chunkSamplesU64]
  let audioChunk : Tensor { shape := #[chunkSamplesU64], dtype := dtypes.compute.val, device := device } :=
    Tensor.castLike m.conv0Weight (audioChunk0.to m.conv0Weight.device)
  let framesChunk : UInt64 := encodedFrames chunkSamplesU64
  let ropeStartPos : UInt64 := framesBeforeFinalDownsample ctxStart.toUInt64
  let codesChunk : Tensor { shape := #[framesChunk, 16], dtype := .Int64, device := device } := encodeMonoFrameMajorWithRoPEOffset m audioChunk ropeStartPos
  let flatChunk : Tensor { shape := #[framesChunk * 16], dtype := .Int64, device := device } := Tensor.reshapeChecked codesChunk #[framesChunk * 16]
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
    (m : (SpeechTokenizer12HzEncoder (device := device) (dtypes := dtypes)))
    (st : EncodeStreamState)
    (newSamples : Array Float)
    : IO (EncodeStreamState × UInt64 × Array UInt64) := do
  if newSamples.isEmpty then
    return (st, numCodeGroups m, #[])

  let ctxStart : Nat := st.offsetSamples - st.history.size
  let chunk := st.history ++ newSamples
  let chunkSamplesU64 : UInt64 := chunk.size.toUInt64
  let audioChunk0 : Tensor { shape := #[chunkSamplesU64], dtype := .Float32, device := .CPU } := Tensor.reshapeChecked (Tensor.fromFloatArray chunk) #[chunkSamplesU64]
  let audioChunk : Tensor { shape := #[chunkSamplesU64], dtype := dtypes.compute.val, device := device } :=
    Tensor.castLike m.conv0Weight (audioChunk0.to m.conv0Weight.device)
  let framesChunk : UInt64 := encodedFrames chunkSamplesU64
  let ropeStartPos : UInt64 := framesBeforeFinalDownsample ctxStart.toUInt64
  let ⟨codeGroups, codesChunk⟩ := encodeMonoFrameMajorWithRoPEOffsetDynamic m audioChunk ropeStartPos
  let flatChunk : Tensor { shape := #[framesChunk * codeGroups], dtype := .Int64, device := device } := Tensor.reshapeChecked codesChunk #[framesChunk * codeGroups]
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
