/-
  Tyr/Model/Qwen3TTS/SpeechTokenizer.lean

  Lean-native Qwen3-TTS 12Hz speech-tokenizer decoder (codes -> waveform).
  This mirrors the Python `Qwen3TTSTokenizerV2Decoder` compute path using real
  `speech_tokenizer/model.safetensors` weights.

  Scope in this module:
  - pure Lean decoder compute
  - real weight loading from SafeTensors
  - native WAV emission via Tyr runtime

  Note: this implementation currently targets the released 12Hz config
  (16 quantizers, 1024 latent dim, 2x convnext upsample, 4 decoder blocks).
-/
import Tyr.Typed.Neural
import Tyr.SafeTensors.Load
import Tyr.TensorStruct
import Tyr.Module.RMSNorm
import Tyr.Model.Qwen.RoPE
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace torch.qwen3tts

variable {device : Device}

open Lean

/-! ## Config parsing -/

structure SpeechTokenizer12HzDecoderConfig where
  codebookSize : UInt64 := 2048
  codebookDim : UInt64 := 512
  latentDim : UInt64 := 1024
  hiddenSize : UInt64 := 512
  intermediateSize : UInt64 := 1024
  layerScaleInitialScale : Float := 0.01
  rmsNormEps : Float := 1e-5
  ropeTheta : Float := 10000.0
  headDim : UInt64 := 64
  numAttentionHeads : UInt64 := 16
  numKeyValueHeads : UInt64 := 16
  numHiddenLayers : UInt64 := 8
  numQuantizers : UInt64 := 16
  numSemanticQuantizers : UInt64 := 1
  slidingWindow : UInt64 := 72
  upsampleRates : Array UInt64 := #[8, 5, 4, 3]
  upsamplingRatios : Array UInt64 := #[2, 2]
  decoderDim : UInt64 := 1536
  deriving Repr, Inhabited

/-- Mimi encoder metadata. Its attention, MLP and quantizer dimensions are
independent of the waveform decoder's `decoder_config`. -/
structure SpeechTokenizer12HzEncoderConfig where
  codebookSize : UInt64 := 2048
  codebookDim : UInt64 := 256
  hiddenSize : UInt64 := 512
  intermediateSize : UInt64 := 2048
  headDim : UInt64 := 64
  numAttentionHeads : UInt64 := 8
  numKeyValueHeads : UInt64 := 8
  numHiddenLayers : UInt64 := 8
  numQuantizers : UInt64 := 32
  numSemanticQuantizers : UInt64 := 1
  slidingWindow : UInt64 := 250
  upsamplingRatios : Array UInt64 := #[8, 6, 5, 4]
  normEps : Float := 1e-5
  ropeTheta : Float := 10000.0
  samplingRate : UInt64 := 24000
  deriving Repr, Inhabited

structure SpeechTokenizer12HzConfig where
  decoder : SpeechTokenizer12HzDecoderConfig := {}
  encoder : SpeechTokenizer12HzEncoderConfig := {}
  /-- Number of available encoder quantizers actually emitted by Qwen. -/
  encoderValidNumQuantizers : UInt64 := 16
  inputSampleRate : UInt64 := 24000
  encodeDownsampleRate : UInt64 := 1920
  outputSampleRate : UInt64 := 24000
  decodeUpsampleRate : UInt64 := 1920
  deriving Repr, Inhabited

private def parseJsonFile (path : String) : IO Json := do
  let contents ← IO.FS.readFile path
  match Json.parse contents with
  | .ok json => pure json
  | .error err => throw (IO.userError s!"Failed to parse JSON at {path}: {err}")

private def getObjVal? (j : Json) (key : String) : Option Json :=
  match j with
  | .obj kvs => Std.TreeMap.Raw.get? kvs key
  | _ => none

private def getArr? (j : Json) : Option (Array Json) :=
  match j with
  | .arr a => some a
  | _ => none

private def getNat? (j : Json) : Option Nat :=
  match (FromJson.fromJson? j : Except String Nat) with
  | .ok n => some n
  | .error _ => none

private def getFloat? (j : Json) : Option Float :=
  match (FromJson.fromJson? j : Except String Float) with
  | .ok x => some x
  | .error _ => (getNat? j).map (·.toFloat)

private def getNatFieldD (j : Json) (key : String) (d : UInt64) : UInt64 :=
  match getObjVal? j key >>= getNat? with
  | some n => n.toUInt64
  | none => d

private def getFloatFieldD (j : Json) (key : String) (d : Float) : Float :=
  match getObjVal? j key >>= getFloat? with
  | some x => x
  | none => d

private def getUInt64ArrayFieldD (j : Json) (key : String) (d : Array UInt64) : Array UInt64 :=
  match getObjVal? j key >>= getArr? with
  | some arr =>
    Id.run do
      let mut out : Array UInt64 := #[]
      for item in arr do
        match getNat? item with
        | some n => out := out.push n.toUInt64
        | none => pure ()
      if out.isEmpty then d else out
  | none => d

namespace SpeechTokenizer12HzConfig

private def parseEncoderConfig (j : Json) (d : SpeechTokenizer12HzEncoderConfig := {})
    : SpeechTokenizer12HzEncoderConfig := {
  codebookSize := getNatFieldD j "codebook_size" d.codebookSize
  codebookDim := getNatFieldD j "codebook_dim" d.codebookDim
  hiddenSize := getNatFieldD j "hidden_size" d.hiddenSize
  intermediateSize := getNatFieldD j "intermediate_size" d.intermediateSize
  headDim := getNatFieldD j "head_dim" d.headDim
  numAttentionHeads := getNatFieldD j "num_attention_heads" d.numAttentionHeads
  numKeyValueHeads := getNatFieldD j "num_key_value_heads" d.numKeyValueHeads
  numHiddenLayers := getNatFieldD j "num_hidden_layers" d.numHiddenLayers
  numQuantizers := getNatFieldD j "num_quantizers" d.numQuantizers
  numSemanticQuantizers := getNatFieldD j "num_semantic_quantizers" d.numSemanticQuantizers
  slidingWindow := getNatFieldD j "sliding_window" d.slidingWindow
  upsamplingRatios := getUInt64ArrayFieldD j "upsampling_ratios" d.upsamplingRatios
  normEps := getFloatFieldD j "norm_eps" d.normEps
  ropeTheta := getFloatFieldD j "rope_theta" d.ropeTheta
  samplingRate := getNatFieldD j "sampling_rate" d.samplingRate
}

private def parseDecoderConfig (j : Json) (d : SpeechTokenizer12HzDecoderConfig := {})
    : SpeechTokenizer12HzDecoderConfig := {
  codebookSize := getNatFieldD j "codebook_size" d.codebookSize
  codebookDim := getNatFieldD j "codebook_dim" d.codebookDim
  latentDim := getNatFieldD j "latent_dim" d.latentDim
  hiddenSize := getNatFieldD j "hidden_size" d.hiddenSize
  intermediateSize := getNatFieldD j "intermediate_size" d.intermediateSize
  layerScaleInitialScale := getFloatFieldD j "layer_scale_initial_scale" d.layerScaleInitialScale
  rmsNormEps := getFloatFieldD j "rms_norm_eps" d.rmsNormEps
  ropeTheta := getFloatFieldD j "rope_theta" d.ropeTheta
  headDim := getNatFieldD j "head_dim" d.headDim
  numAttentionHeads := getNatFieldD j "num_attention_heads" d.numAttentionHeads
  numKeyValueHeads := getNatFieldD j "num_key_value_heads" d.numKeyValueHeads
  numHiddenLayers := getNatFieldD j "num_hidden_layers" d.numHiddenLayers
  numQuantizers := getNatFieldD j "num_quantizers" d.numQuantizers
  numSemanticQuantizers := getNatFieldD j "num_semantic_quantizers" d.numSemanticQuantizers
  slidingWindow := getNatFieldD j "sliding_window" d.slidingWindow
  upsampleRates := getUInt64ArrayFieldD j "upsample_rates" d.upsampleRates
  upsamplingRatios := getUInt64ArrayFieldD j "upsampling_ratios" d.upsamplingRatios
  decoderDim := getNatFieldD j "decoder_dim" d.decoderDim
}

private def requireTrue (ok : Bool) (msg : String) : IO Unit := do
  unless ok do
    throw (IO.userError msg)

/-- Parse encoder and decoder metadata independently, preserving defaults for
omitted fields as in the upstream configuration classes. -/
def fromJson (root : Json) (defaults : SpeechTokenizer12HzConfig := {})
    : SpeechTokenizer12HzConfig := Id.run do
  let decoder :=
    match getObjVal? root "decoder_config" with
    | some j => parseDecoderConfig j defaults.decoder
    | none => defaults.decoder
  let encoder :=
    match getObjVal? root "encoder_config" with
    | some j => parseEncoderConfig j defaults.encoder
    | none => defaults.encoder
  return {
    decoder
    encoder
    encoderValidNumQuantizers := getNatFieldD root "encoder_valid_num_quantizers" defaults.encoderValidNumQuantizers
    inputSampleRate := getNatFieldD root "input_sample_rate" defaults.inputSampleRate
    encodeDownsampleRate := getNatFieldD root "encode_downsample_rate" defaults.encodeDownsampleRate
    outputSampleRate := getNatFieldD root "output_sample_rate" defaults.outputSampleRate
    decodeUpsampleRate := getNatFieldD root "decode_upsample_rate" defaults.decodeUpsampleRate
  }

/-- Load tokenizer config from `speech_tokenizer/config.json`. -/
def loadFromFile (path : String) (defaults : SpeechTokenizer12HzConfig := {})
    : IO SpeechTokenizer12HzConfig := do
  return fromJson (← parseJsonFile path) defaults

/-- Validate the fixed Mimi encoder architecture and select the emitted
semantic/acoustic codebooks. Decoder metadata does not control encoding. -/
def encoderCodebookCounts (cfg : SpeechTokenizer12HzConfig) : IO (Nat × Nat) := do
  let e := cfg.encoder
  for (name, actual, expected) in #[
      ("codebook_size", e.codebookSize, 2048), ("codebook_dim", e.codebookDim, 256),
      ("hidden_size", e.hiddenSize, 512), ("intermediate_size", e.intermediateSize, 2048),
      ("head_dim", e.headDim, 64), ("num_attention_heads", e.numAttentionHeads, 8),
      ("num_key_value_heads", e.numKeyValueHeads, 8), ("num_hidden_layers", e.numHiddenLayers, 8),
      ("sliding_window", e.slidingWindow, 250), ("sampling_rate", e.samplingRate, 24000)] do
    requireTrue (actual == expected)
      s!"Unsupported speech tokenizer encoder {name}={actual} (expected {expected})"
  requireTrue (e.upsamplingRatios == #[8, 6, 5, 4])
    s!"Unsupported speech tokenizer encoder upsampling_ratios={e.upsamplingRatios} (expected #[8,6,5,4])"
  requireTrue (e.normEps == 1e-5 && e.ropeTheta == 10000.0)
    s!"Unsupported speech tokenizer encoder norm_eps={e.normEps}/rope_theta={e.ropeTheta}"
  requireTrue (cfg.inputSampleRate == 24000 && cfg.encodeDownsampleRate == 1920)
    s!"Unsupported encoder input_sample_rate={cfg.inputSampleRate}/encode_downsample_rate={cfg.encodeDownsampleRate}"
  let valid := cfg.encoderValidNumQuantizers
  requireTrue (valid <= e.numQuantizers && e.numSemanticQuantizers >= 1 && e.numSemanticQuantizers < valid)
    s!"Unsupported encoder quantizer split: semantic={e.numSemanticQuantizers}, valid={valid}, available={e.numQuantizers} (need 1 <= semantic < valid <= available)"
  return (e.numSemanticQuantizers.toNat, (valid - e.numSemanticQuantizers).toNat)

/-- Validate the waveform decoder architecture, independently of the encoder. -/
def validateSupported (cfg : SpeechTokenizer12HzConfig) : IO Unit := do
  let d := cfg.decoder
  requireTrue (d.codebookSize == 2048) s!"Unsupported speech tokenizer codebook_size={d.codebookSize} (expected 2048)"
  requireTrue (d.codebookDim == 512) s!"Unsupported speech tokenizer codebook_dim={d.codebookDim} (expected 512)"
  requireTrue (d.latentDim == 1024) s!"Unsupported speech tokenizer latent_dim={d.latentDim} (expected 1024)"
  requireTrue (d.hiddenSize == 512) s!"Unsupported speech tokenizer hidden_size={d.hiddenSize} (expected 512)"
  requireTrue (d.intermediateSize == 1024) s!"Unsupported speech tokenizer intermediate_size={d.intermediateSize} (expected 1024)"
  requireTrue (d.headDim == 64) s!"Unsupported speech tokenizer head_dim={d.headDim} (expected 64)"
  requireTrue (d.numAttentionHeads == 16) s!"Unsupported speech tokenizer num_attention_heads={d.numAttentionHeads} (expected 16)"
  requireTrue (d.numKeyValueHeads == 16) s!"Unsupported speech tokenizer num_key_value_heads={d.numKeyValueHeads} (expected 16)"
  requireTrue (d.numHiddenLayers == 8) s!"Unsupported speech tokenizer num_hidden_layers={d.numHiddenLayers} (expected 8)"
  requireTrue (d.numQuantizers == 16) s!"Unsupported speech tokenizer num_quantizers={d.numQuantizers} (expected 16)"
  requireTrue (d.numSemanticQuantizers == 1) s!"Unsupported speech tokenizer num_semantic_quantizers={d.numSemanticQuantizers} (expected 1)"
  requireTrue (d.slidingWindow == 72) s!"Unsupported speech tokenizer sliding_window={d.slidingWindow} (expected 72)"
  requireTrue (d.upsampleRates == #[8, 5, 4, 3]) s!"Unsupported speech tokenizer upsample_rates={d.upsampleRates} (expected #[8,5,4,3])"
  requireTrue (d.upsamplingRatios == #[2, 2]) s!"Unsupported speech tokenizer upsampling_ratios={d.upsamplingRatios} (expected #[2,2])"
  requireTrue (d.decoderDim == 1536) s!"Unsupported speech tokenizer decoder_dim={d.decoderDim} (expected 1536)"
  requireTrue (cfg.decodeUpsampleRate == 1920) s!"Unsupported decode_upsample_rate={cfg.decodeUpsampleRate} (expected 1920)"

end SpeechTokenizer12HzConfig

/-! ## Decoder parameters -/

private def freeze {s : Shape} (t : Tensor { shape := s, device := device }) : Tensor { shape := s, device := device } :=
  Tensor.detach t

private def loadFrozen (path : String) (name : String) (s : Shape) : IO (Tensor { shape := s, device := device }) := do
  let t ← safetensors.loadTensorCandidates path #[name] s device
  pure (freeze t)

private def addBias3d {batch channels frames : UInt64}
    (x : Tensor { shape := #[batch, channels, frames], device := device })
    (bias : Tensor { shape := #[channels], device := device }) : Tensor { shape := #[batch, channels, frames], device := device } :=
  let b : Tensor { shape := #[1, channels, 1], device := device } := Tensor.reshapeChecked bias #[1, channels, 1]
  Tensor.Promoting.add x (Tensor.expand b #[batch, channels, frames])

private def scale3d {batch seq dim : UInt64}
    (x : Tensor { shape := #[batch, seq, dim], device := device })
    (scale : Tensor { shape := #[dim], device := device }) : Tensor { shape := #[batch, seq, dim], device := device } :=
  let s : Tensor { shape := #[1, 1, dim], device := device } := Tensor.reshapeChecked scale #[1, 1, dim]
  Tensor.Promoting.mul x (Tensor.expand s #[batch, seq, dim])

private def clampWave {s : Shape} (x : Tensor { shape := s, device := device }) : Tensor { shape := s, device := device } :=
  Tensor.Preserving.clampFloat x (-1.0) 1.0

private def clampMin1d {n : UInt64} (x : Tensor { shape := #[n], device := device }) (minVal : Float) : Tensor { shape := #[n], device := device } :=
  let minT := Tensor.castLike x (Tensor.full #[n] minVal false x.device)
  Tensor.Promoting.whereSelect (Tensor.ltScalar x minVal) minT x

structure QuantizerCodebook where
  clusterUsage : Tensor { shape := #[2048], device := device }
  embeddingSum : Tensor { shape := #[2048, 256], device := device }
  deriving TensorStruct

structure RVQDecode where
  outputProj : Tensor { shape := #[512, 256, 1], device := device }
  codebooks : Array (QuantizerCodebook (device := device))
  deriving TensorStruct

structure DecoderTransformerLayer where
  inputLayernormWeight : Tensor { shape := #[512], device := device }
  qProj : Tensor { shape := #[1024, 512], device := device }
  kProj : Tensor { shape := #[1024, 512], device := device }
  vProj : Tensor { shape := #[1024, 512], device := device }
  oProj : Tensor { shape := #[512, 1024], device := device }
  selfAttnLayerScale : Tensor { shape := #[512], device := device }
  postAttentionLayernormWeight : Tensor { shape := #[512], device := device }
  gateProj : Tensor { shape := #[1024, 512], device := device }
  upProj : Tensor { shape := #[1024, 512], device := device }
  downProj : Tensor { shape := #[512, 1024], device := device }
  mlpLayerScale : Tensor { shape := #[512], device := device }
  deriving TensorStruct

structure UpsampleConvNeXtBlock where
  dwconvWeight : Tensor { shape := #[1024, 1, 7], device := device }
  dwconvBias : Tensor { shape := #[1024], device := device }
  normWeight : Tensor { shape := #[1024], device := device }
  normBias : Tensor { shape := #[1024], device := device }
  pwconv1Weight : Tensor { shape := #[4096, 1024], device := device }
  pwconv1Bias : Tensor { shape := #[4096], device := device }
  pwconv2Weight : Tensor { shape := #[1024, 4096], device := device }
  pwconv2Bias : Tensor { shape := #[1024], device := device }
  gamma : Tensor { shape := #[1024], device := device }
  deriving TensorStruct

structure UpsampleStage where
  transConvWeight : Tensor { shape := #[1024, 1024, 2], device := device }
  transConvBias : Tensor { shape := #[1024], device := device }
  convNeXt : (UpsampleConvNeXtBlock (device := device))
  deriving TensorStruct

structure DecoderResidualUnit (dim : UInt64) where
  conv1Dilation : UInt64 := 1
  act1Alpha : Tensor { shape := #[dim], device := device }
  act1Beta : Tensor { shape := #[dim], device := device }
  conv1Weight : Tensor { shape := #[dim, dim, 7], device := device }
  conv1Bias : Tensor { shape := #[dim], device := device }
  act2Alpha : Tensor { shape := #[dim], device := device }
  act2Beta : Tensor { shape := #[dim], device := device }
  conv2Weight : Tensor { shape := #[dim, dim, 1], device := device }
  conv2Bias : Tensor { shape := #[dim], device := device }
  deriving TensorStruct

structure DecoderBlock (inDim outDim upRate : UInt64) where
  snakeAlpha : Tensor { shape := #[inDim], device := device }
  snakeBeta : Tensor { shape := #[inDim], device := device }
  transWeight : Tensor { shape := #[inDim, outDim, 2 * upRate], device := device }
  transBias : Tensor { shape := #[outDim], device := device }
  res1 : DecoderResidualUnit (device := device) outDim
  res2 : DecoderResidualUnit (device := device) outDim
  res3 : DecoderResidualUnit (device := device) outDim
  deriving TensorStruct

structure SpeechTokenizer12HzDecoder where
  rvqFirst : (RVQDecode (device := device))
  rvqRest : (RVQDecode (device := device))

  preConvWeight : Tensor { shape := #[1024, 512, 3], device := device }
  preConvBias : Tensor { shape := #[1024], device := device }

  preTransformerInputProj : Tensor { shape := #[512, 1024], device := device }
  preTransformerInputBias : Tensor { shape := #[512], device := device }
  preTransformerLayers : Array (DecoderTransformerLayer (device := device))
  preTransformerNormWeight : Tensor { shape := #[512], device := device }
  preTransformerOutputProj : Tensor { shape := #[1024, 512], device := device }
  preTransformerOutputBias : Tensor { shape := #[1024], device := device }

  upsample0 : (UpsampleStage (device := device))
  upsample1 : (UpsampleStage (device := device))

  decoderConv0Weight : Tensor { shape := #[1536, 1024, 7], device := device }
  decoderConv0Bias : Tensor { shape := #[1536], device := device }

  decoderBlock1 : DecoderBlock (device := device) 1536 768 8
  decoderBlock2 : DecoderBlock (device := device) 768 384 5
  decoderBlock3 : DecoderBlock (device := device) 384 192 4
  decoderBlock4 : DecoderBlock (device := device) 192 96 3

  finalSnakeAlpha : Tensor { shape := #[96], device := device }
  finalSnakeBeta : Tensor { shape := #[96], device := device }
  finalConvWeight : Tensor { shape := #[1, 96, 7], device := device }
  finalConvBias : Tensor { shape := #[1], device := device }

  outputSampleRate : UInt64 := 24000
  decodeUpsampleRate : UInt64 := 1920
  deriving TensorStruct

namespace SpeechTokenizer12HzDecoder

private def loadCodebook (weightsPath : String) (namePrefix : String) : IO (QuantizerCodebook (device := device)) := do
  let clusterUsage ← loadFrozen weightsPath s!"{namePrefix}.cluster_usage" #[2048]
  let embeddingSum ← loadFrozen weightsPath s!"{namePrefix}.embedding_sum" #[2048, 256]
  pure { clusterUsage, embeddingSum }

private def loadRVQ (weightsPath : String) (namePrefix : String) (nCodebooks : Nat) : IO (RVQDecode (device := device)) := do
  let outputProj ← loadFrozen weightsPath s!"{namePrefix}.output_proj.weight" #[512, 256, 1]
  let mut codebooks : Array (QuantizerCodebook (device := device)) := #[]
  for i in [:nCodebooks] do
    let cb ← loadCodebook weightsPath s!"{namePrefix}.vq.layers.{i}._codebook"
    codebooks := codebooks.push cb
  pure { outputProj, codebooks }

private def loadTransformerLayer (weightsPath : String) (i : Nat) : IO (DecoderTransformerLayer (device := device)) := do
  let p := s!"decoder.pre_transformer.layers.{i}"
  let inputLayernormWeight ← loadFrozen weightsPath s!"{p}.input_layernorm.weight" #[512]
  let qProj ← loadFrozen weightsPath s!"{p}.self_attn.q_proj.weight" #[1024, 512]
  let kProj ← loadFrozen weightsPath s!"{p}.self_attn.k_proj.weight" #[1024, 512]
  let vProj ← loadFrozen weightsPath s!"{p}.self_attn.v_proj.weight" #[1024, 512]
  let oProj ← loadFrozen weightsPath s!"{p}.self_attn.o_proj.weight" #[512, 1024]
  let selfAttnLayerScale ← loadFrozen weightsPath s!"{p}.self_attn_layer_scale.scale" #[512]
  let postAttentionLayernormWeight ← loadFrozen weightsPath s!"{p}.post_attention_layernorm.weight" #[512]
  let gateProj ← loadFrozen weightsPath s!"{p}.mlp.gate_proj.weight" #[1024, 512]
  let upProj ← loadFrozen weightsPath s!"{p}.mlp.up_proj.weight" #[1024, 512]
  let downProj ← loadFrozen weightsPath s!"{p}.mlp.down_proj.weight" #[512, 1024]
  let mlpLayerScale ← loadFrozen weightsPath s!"{p}.mlp_layer_scale.scale" #[512]
  pure {
    inputLayernormWeight
    qProj
    kProj
    vProj
    oProj
    selfAttnLayerScale
    postAttentionLayernormWeight
    gateProj
    upProj
    downProj
    mlpLayerScale
  }

private def loadUpsampleConvNeXt (weightsPath : String) (namePrefix : String) : IO (UpsampleConvNeXtBlock (device := device)) := do
  let dwconvWeight ← loadFrozen weightsPath s!"{namePrefix}.dwconv.conv.weight" #[1024, 1, 7]
  let dwconvBias ← loadFrozen weightsPath s!"{namePrefix}.dwconv.conv.bias" #[1024]
  let normWeight ← loadFrozen weightsPath s!"{namePrefix}.norm.weight" #[1024]
  let normBias ← loadFrozen weightsPath s!"{namePrefix}.norm.bias" #[1024]
  let pwconv1Weight ← loadFrozen weightsPath s!"{namePrefix}.pwconv1.weight" #[4096, 1024]
  let pwconv1Bias ← loadFrozen weightsPath s!"{namePrefix}.pwconv1.bias" #[4096]
  let pwconv2Weight ← loadFrozen weightsPath s!"{namePrefix}.pwconv2.weight" #[1024, 4096]
  let pwconv2Bias ← loadFrozen weightsPath s!"{namePrefix}.pwconv2.bias" #[1024]
  let gamma ← loadFrozen weightsPath s!"{namePrefix}.gamma" #[1024]
  pure {
    dwconvWeight
    dwconvBias
    normWeight
    normBias
    pwconv1Weight
    pwconv1Bias
    pwconv2Weight
    pwconv2Bias
    gamma
  }

private def loadUpsampleStage (weightsPath : String) (idx : Nat) : IO (UpsampleStage (device := device)) := do
  let p := s!"decoder.upsample.{idx}"
  let transConvWeight ← loadFrozen weightsPath s!"{p}.0.conv.weight" #[1024, 1024, 2]
  let transConvBias ← loadFrozen weightsPath s!"{p}.0.conv.bias" #[1024]
  let convNeXt ← loadUpsampleConvNeXt weightsPath s!"{p}.1"
  pure { transConvWeight, transConvBias, convNeXt }

private def loadResidualUnit (weightsPath : String) (namePrefix : String) (dim : UInt64) (dilation : UInt64)
    : IO (DecoderResidualUnit (device := device) dim) := do
  let act1Alpha ← loadFrozen weightsPath s!"{namePrefix}.act1.alpha" #[dim]
  let act1Beta ← loadFrozen weightsPath s!"{namePrefix}.act1.beta" #[dim]
  let conv1Weight ← loadFrozen weightsPath s!"{namePrefix}.conv1.conv.weight" #[dim, dim, 7]
  let conv1Bias ← loadFrozen weightsPath s!"{namePrefix}.conv1.conv.bias" #[dim]
  let act2Alpha ← loadFrozen weightsPath s!"{namePrefix}.act2.alpha" #[dim]
  let act2Beta ← loadFrozen weightsPath s!"{namePrefix}.act2.beta" #[dim]
  let conv2Weight ← loadFrozen weightsPath s!"{namePrefix}.conv2.conv.weight" #[dim, dim, 1]
  let conv2Bias ← loadFrozen weightsPath s!"{namePrefix}.conv2.conv.bias" #[dim]
  pure {
    conv1Dilation := dilation
    act1Alpha
    act1Beta
    conv1Weight
    conv1Bias
    act2Alpha
    act2Beta
    conv2Weight
    conv2Bias
  }

private def loadDecoderBlock
    (weightsPath : String)
    (idx : Nat)
    (inDim outDim upRate : UInt64)
    : IO (DecoderBlock (device := device) inDim outDim upRate) := do
  let p := s!"decoder.decoder.{idx}.block"
  let snakeAlpha ← loadFrozen weightsPath s!"{p}.0.alpha" #[inDim]
  let snakeBeta ← loadFrozen weightsPath s!"{p}.0.beta" #[inDim]
  let transWeight ← loadFrozen weightsPath s!"{p}.1.conv.weight" #[inDim, outDim, 2 * upRate]
  let transBias ← loadFrozen weightsPath s!"{p}.1.conv.bias" #[outDim]
  let res1 ← loadResidualUnit weightsPath s!"{p}.2" outDim 1
  let res2 ← loadResidualUnit weightsPath s!"{p}.3" outDim 3
  let res3 ← loadResidualUnit weightsPath s!"{p}.4" outDim 9
  pure { snakeAlpha, snakeBeta, transWeight, transBias, res1, res2, res3 }

/-- Load decoder weights from `speech_tokenizer` directory. -/
def loadFromDir (speechTokenizerDir : String) (device : Device := Device.CPU) : IO (SpeechTokenizer12HzDecoder (device := device)) := do
  let cfg ← SpeechTokenizer12HzConfig.loadFromFile s!"{speechTokenizerDir}/config.json"
  SpeechTokenizer12HzConfig.validateSupported cfg

  let weightsPath := s!"{speechTokenizerDir}/model.safetensors"

  let rvqFirst ← loadRVQ weightsPath "decoder.quantizer.rvq_first" 1
  let rvqRest ← loadRVQ weightsPath "decoder.quantizer.rvq_rest" 15

  let preConvWeight ← loadFrozen weightsPath "decoder.pre_conv.conv.weight" #[1024, 512, 3]
  let preConvBias ← loadFrozen weightsPath "decoder.pre_conv.conv.bias" #[1024]

  let preTransformerInputProj ← loadFrozen weightsPath "decoder.pre_transformer.input_proj.weight" #[512, 1024]
  let preTransformerInputBias ← loadFrozen weightsPath "decoder.pre_transformer.input_proj.bias" #[512]

  let mut preTransformerLayers : Array (DecoderTransformerLayer (device := device)) := #[]
  for i in [:8] do
    let layer ← loadTransformerLayer weightsPath i
    preTransformerLayers := preTransformerLayers.push layer

  let preTransformerNormWeight ← loadFrozen weightsPath "decoder.pre_transformer.norm.weight" #[512]
  let preTransformerOutputProj ← loadFrozen weightsPath "decoder.pre_transformer.output_proj.weight" #[1024, 512]
  let preTransformerOutputBias ← loadFrozen weightsPath "decoder.pre_transformer.output_proj.bias" #[1024]

  let upsample0 ← loadUpsampleStage weightsPath 0
  let upsample1 ← loadUpsampleStage weightsPath 1

  let decoderConv0Weight ← loadFrozen weightsPath "decoder.decoder.0.conv.weight" #[1536, 1024, 7]
  let decoderConv0Bias ← loadFrozen weightsPath "decoder.decoder.0.conv.bias" #[1536]

  let decoderBlock1 ← loadDecoderBlock weightsPath 1 1536 768 8
  let decoderBlock2 ← loadDecoderBlock weightsPath 2 768 384 5
  let decoderBlock3 ← loadDecoderBlock weightsPath 3 384 192 4
  let decoderBlock4 ← loadDecoderBlock weightsPath 4 192 96 3

  let finalSnakeAlpha ← loadFrozen weightsPath "decoder.decoder.5.alpha" #[96]
  let finalSnakeBeta ← loadFrozen weightsPath "decoder.decoder.5.beta" #[96]
  let finalConvWeight ← loadFrozen weightsPath "decoder.decoder.6.conv.weight" #[1, 96, 7]
  let finalConvBias ← loadFrozen weightsPath "decoder.decoder.6.conv.bias" #[1]

  let dec : (SpeechTokenizer12HzDecoder (device := device)) := {
    rvqFirst
    rvqRest
    preConvWeight
    preConvBias
    preTransformerInputProj
    preTransformerInputBias
    preTransformerLayers
    preTransformerNormWeight
    preTransformerOutputProj
    preTransformerOutputBias
    upsample0
    upsample1
    decoderConv0Weight
    decoderConv0Bias
    decoderBlock1
    decoderBlock2
    decoderBlock3
    decoderBlock4
    finalSnakeAlpha
    finalSnakeBeta
    finalConvWeight
    finalConvBias
    outputSampleRate := cfg.outputSampleRate
    decodeUpsampleRate := cfg.decodeUpsampleRate
  }
  pure dec

private def decodeCodebook {batch frames : UInt64}
    (cb : (QuantizerCodebook (device := device)))
    (codes : Tensor { shape := #[batch, frames], dtype := .Int64, device := device }) : Tensor { shape := #[batch, 256, frames], device := device } :=
  let usage : Tensor { shape := #[2048], device := device } := clampMin1d cb.clusterUsage 1e-5
  let usage2d : Tensor { shape := #[2048, 256], device := device } := Tensor.expand (Tensor.reshapeChecked usage #[2048, 1]) #[2048, 256]
  let embedding : Tensor { shape := #[2048, 256], device := device } := Tensor.Promoting.div cb.embeddingSum usage2d
  let q : Tensor { shape := #[batch, frames, 256], device := device } := Tensor.embedding codes embedding
  Tensor.reshapeChecked (Tensor.transpose q 1 2) #[batch, 256, frames]

private def decodeRVQSlice {batch groups frames : UInt64}
    (rvq : (RVQDecode (device := device)))
    (codes : Tensor { shape := #[batch, groups, frames], dtype := .Int64, device := device })
    (start : UInt64)
    : Tensor { shape := #[batch, 512, frames], device := device } :=
  Id.run do
    let mut quantized : Tensor { shape := #[batch, 256, frames], device := device } := Tensor.castLike rvq.outputProj (Tensor.zeros #[batch, 256, frames] false codes.device)
    let mut off : UInt64 := 0
    for cb in rvq.codebooks do
      let idx := start + off
      let code3 : Tensor { shape := #[batch, 1, frames], dtype := .Int64, device := device } := Tensor.sliceChecked codes 1 idx 1 #[batch, 1, frames]
      let code2 : Tensor { shape := #[batch, frames], dtype := .Int64, device := device } := Tensor.reshapeChecked code3 #[batch, frames]
      let q := decodeCodebook cb code2
      quantized := Tensor.Promoting.add quantized q
      off := off + 1
    let out : Tensor { shape := #[batch, 512, frames], device := device } :=
      Tensor.reshapeChecked (Tensor.conv1d quantized rvq.outputProj 1 0 1) #[batch, 512, frames]
    out

private def decodeSplitRVQ {batch frames : UInt64}
    (m : (SpeechTokenizer12HzDecoder (device := device)))
    (codes : Tensor { shape := #[batch, 16, frames], dtype := .Int64, device := device }) : Tensor { shape := #[batch, 512, frames], device := device } :=
  let first := decodeRVQSlice m.rvqFirst codes 0
  let rest := decodeRVQSlice m.rvqRest codes 1
  Tensor.Promoting.add first rest

private def causalConv1d {batch inC outC frames kernel : UInt64}
    (x : Tensor { shape := #[batch, inC, frames], device := device })
    (weight : Tensor { shape := #[outC, inC, kernel], device := device })
    (bias : Tensor { shape := #[outC], device := device })
    (dilation : UInt64 := 1)
    : Tensor { shape := #[batch, outC, frames], device := device } :=
  let leftPad : UInt64 := dilation * (kernel - 1)
  let left : Tensor { shape := #[batch, inC, leftPad], device := device } := Tensor.castLike x (Tensor.zeros #[batch, inC, leftPad] false x.device)
  let xPad : Tensor { shape := #[batch, inC, frames + leftPad], device := device } := Tensor.reshapeChecked (Tensor.cat left x 2) #[batch, inC, frames + leftPad]
  let y0 : Tensor { shape := #[batch, outC, frames], device := device } := Tensor.reshapeChecked (Tensor.conv1d xPad weight 1 0 dilation) #[batch, outC, frames]
  addBias3d y0 bias

private def causalDepthwiseConv1d {batch channels frames : UInt64}
    (x : Tensor { shape := #[batch, channels, frames], device := device })
    (weight : Tensor { shape := #[channels, 1, 7], device := device })
    (bias : Tensor { shape := #[channels], device := device })
    : Tensor { shape := #[batch, channels, frames], device := device } :=
  let left : Tensor { shape := #[batch, channels, 6], device := device } := Tensor.castLike x (Tensor.zeros #[batch, channels, 6] false x.device)
  let xPad : Tensor { shape := #[batch, channels, frames + 6], device := device } := Tensor.reshapeChecked (Tensor.cat left x 2) #[batch, channels, frames + 6]
  let y : Tensor { shape := #[batch, channels, frames], device := device } :=
    Tensor.reshapeChecked (Tensor.conv1dGroups xPad weight bias 1 0 1 channels) #[batch, channels, frames]
  y

private def causalTransConv1d {batch inC outC frames kernel stride rightPad : UInt64}
    (x : Tensor { shape := #[batch, inC, frames], device := device })
    (weight : Tensor { shape := #[inC, outC, kernel], device := device })
    (bias : Tensor { shape := #[outC], device := device })
    : Tensor { shape := #[batch, outC, ((frames - 1) * stride + kernel) - rightPad], device := device } :=
  let outLen : UInt64 := (frames - 1) * stride + kernel
  let y0 : Tensor { shape := #[batch, outC, outLen], device := device } :=
    Tensor.reshapeChecked (Tensor.convTranspose1d x weight bias stride 0 0 1) #[batch, outC, outLen]
  Tensor.sliceChecked y0 2 0 (outLen - rightPad) #[batch, outC, outLen - rightPad]

private def snakeBeta {batch channels frames : UInt64}
    (x : Tensor { shape := #[batch, channels, frames], device := device })
    (alpha : Tensor { shape := #[channels], device := device })
    (beta : Tensor { shape := #[channels], device := device }) : Tensor { shape := #[batch, channels, frames], device := device } :=
  let alphaE : Tensor { shape := #[batch, channels, frames], device := device } :=
    Tensor.expand (Tensor.reshapeChecked (Tensor.Preserving.exp alpha) #[1, channels, 1]) #[batch, channels, frames]
  let betaE : Tensor { shape := #[batch, channels, frames], device := device } :=
    Tensor.expand (Tensor.reshapeChecked (Tensor.Preserving.exp beta) #[1, channels, 1]) #[batch, channels, frames]
  let sin2 : Tensor { shape := #[batch, channels, frames], device := device } :=
    let z := Tensor.Preserving.sin (Tensor.Promoting.mul x alphaE)
    Tensor.Promoting.mul z z
  let eps : Tensor { shape := #[batch, channels, frames], device := device } := Tensor.castLike x (Tensor.full #[batch, channels, frames] 0.000000001 false x.device)
  let denom := Tensor.Promoting.add betaE eps
  let invDenom : Tensor { shape := #[batch, channels, frames], device := device } := Tensor.Promoting.div (Tensor.castLike x (Tensor.ones #[batch, channels, frames] false x.device)) denom
  Tensor.Promoting.add x (Tensor.Promoting.mul invDenom sin2)

private def forwardConvNeXt {batch frames : UInt64}
    (blk : (UpsampleConvNeXtBlock (device := device)))
    (x : Tensor { shape := #[batch, 1024, frames], device := device }) : Tensor { shape := #[batch, 1024, frames], device := device } :=
  let residual := x
  let h0 := causalDepthwiseConv1d x blk.dwconvWeight blk.dwconvBias
  let h1 : Tensor { shape := #[batch, frames, 1024], device := device } := Tensor.reshapeChecked (Tensor.transpose h0 1 2) #[batch, frames, 1024]
  let h2 : Tensor { shape := #[batch, frames, 1024], device := device } := Tensor.layerNorm h1 blk.normWeight blk.normBias 1e-6
  let h3 : Tensor { shape := #[batch, frames, 4096], device := device } := Tensor.affine3d h2 blk.pwconv1Weight blk.pwconv1Bias
  let h4 : Tensor { shape := #[batch, frames, 4096], device := device } := Tensor.Preserving.gelu h3
  let h5 : Tensor { shape := #[batch, frames, 1024], device := device } := Tensor.affine3d h4 blk.pwconv2Weight blk.pwconv2Bias
  let h6 : Tensor { shape := #[batch, frames, 1024], device := device } := scale3d h5 blk.gamma
  let h7 : Tensor { shape := #[batch, 1024, frames], device := device } := Tensor.reshapeChecked (Tensor.transpose h6 1 2) #[batch, 1024, frames]
  Tensor.Promoting.add residual h7

private def forwardAttention {batch seq : UInt64}
    (layer : (DecoderTransformerLayer (device := device)))
    (x : Tensor { shape := #[batch, seq, 512], device := device })
    (cos sin : Tensor { shape := #[seq, 32], dtype := .Float32, device := device }) : Tensor { shape := #[batch, seq, 512], device := device } :=
  let q0 : Tensor { shape := #[batch, seq, 1024], device := device } := Tensor.linear3d x layer.qProj
  let k0 : Tensor { shape := #[batch, seq, 1024], device := device } := Tensor.linear3d x layer.kProj
  let v0 : Tensor { shape := #[batch, seq, 1024], device := device } := Tensor.linear3d x layer.vProj

  let q : Tensor { shape := #[batch, seq, 16, 64], device := device } := Tensor.reshapeChecked q0 #[batch, seq, 16, 64]
  let k : Tensor { shape := #[batch, seq, 16, 64], device := device } := Tensor.reshapeChecked k0 #[batch, seq, 16, 64]
  let v : Tensor { shape := #[batch, seq, 16, 64], device := device } := Tensor.reshapeChecked v0 #[batch, seq, 16, 64]

  let q := Tensor.applyRotary q cos sin
  let k := Tensor.applyRotary k cos sin

  let qh : Tensor { shape := #[batch, 16, seq, 64], device := device } := Tensor.splitHeads q
  let kh : Tensor { shape := #[batch, 16, seq, 64], device := device } := Tensor.splitHeads k
  let vh : Tensor { shape := #[batch, 16, seq, 64], device := device } := Tensor.splitHeads v

  let attn : Tensor { shape := #[batch, 16, seq, 64], device := device } :=
    Tensor.attentionWindow qh kh vh 0.0 true true 72
  let attn : Tensor { shape := #[batch, seq, 16, 64], device := device } := Tensor.joinHeads attn
  let attn : Tensor { shape := #[batch, seq, 1024], device := device } := Tensor.reshapeChecked attn #[batch, seq, 1024]
  Tensor.linear3d attn layer.oProj

private def forwardTransformerLayer {batch seq : UInt64}
    (layer : (DecoderTransformerLayer (device := device)))
    (x : Tensor { shape := #[batch, seq, 512], device := device })
    (cos sin : Tensor { shape := #[seq, 32], dtype := .Float32, device := device })
    : Tensor { shape := #[batch, seq, 512], device := device } :=
  let residual1 := x
  let norm1 : RMSNorm (device := device) 512 (.Unknown "dynamic") := { weight := layer.inputLayernormWeight, eps := ⟨1e-5⟩ }
  let h1 : Tensor { shape := #[batch, seq, 512], device := device } := norm1.forward3d x
  let h2 : Tensor { shape := #[batch, seq, 512], device := device } := forwardAttention layer h1 cos sin
  let h3 : Tensor { shape := #[batch, seq, 512], device := device } := Tensor.Promoting.add residual1 (scale3d h2 layer.selfAttnLayerScale)

  let residual2 := h3
  let norm2 : RMSNorm (device := device) 512 (.Unknown "dynamic") := { weight := layer.postAttentionLayernormWeight, eps := ⟨1e-5⟩ }
  let h4 : Tensor { shape := #[batch, seq, 512], device := device } := norm2.forward3d h3
  let gate : Tensor { shape := #[batch, seq, 1024], device := device } := Tensor.linear3d h4 layer.gateProj
  let up : Tensor { shape := #[batch, seq, 1024], device := device } := Tensor.linear3d h4 layer.upProj
  let mlpIn : Tensor { shape := #[batch, seq, 1024], device := device } := Tensor.Promoting.mul (Tensor.Preserving.silu gate) up
  let mlpOut : Tensor { shape := #[batch, seq, 512], device := device } := Tensor.linear3d mlpIn layer.downProj
  Tensor.Promoting.add residual2 (scale3d mlpOut layer.mlpLayerScale)

private def forwardPreTransformer {batch frames : UInt64}
    (m : (SpeechTokenizer12HzDecoder (device := device)))
    (x : Tensor { shape := #[batch, 1024, frames], device := device }) : Tensor { shape := #[batch, 1024, frames], device := device } :=
  Id.run do
    let x0 : Tensor { shape := #[batch, frames, 1024], device := device } := Tensor.reshapeChecked (Tensor.transpose x 1 2) #[batch, frames, 1024]
    let mut h : Tensor { shape := #[batch, frames, 512], device := device } := Tensor.affine3d x0 m.preTransformerInputProj m.preTransformerInputBias
    let (cos0, sin0) := Tensor.rotaryFreqs frames 64 10000.0
    let cos : Tensor { shape := #[frames, 32], dtype := .Float32, device := device } := cos0.to h.device
    let sin : Tensor { shape := #[frames, 32], dtype := .Float32, device := device } := sin0.to h.device
    for layer in m.preTransformerLayers do
      h := forwardTransformerLayer layer h cos sin
    let norm : RMSNorm (device := device) 512 (.Unknown "dynamic") := { weight := m.preTransformerNormWeight, eps := ⟨1e-5⟩ }
    let hNorm : Tensor { shape := #[batch, frames, 512], device := device } := norm.forward3d h
    let hOut : Tensor { shape := #[batch, frames, 1024], device := device } := Tensor.affine3d hNorm m.preTransformerOutputProj m.preTransformerOutputBias
    Tensor.reshapeChecked (Tensor.transpose hOut 1 2) #[batch, 1024, frames]

private def forwardResidualUnit {batch frames dim : UInt64}
    (u : DecoderResidualUnit (device := device) dim)
    (x : Tensor { shape := #[batch, dim, frames], device := device }) : Tensor { shape := #[batch, dim, frames], device := device } :=
  let residual := x
  let h1 := snakeBeta x u.act1Alpha u.act1Beta
  let h2 := causalConv1d h1 u.conv1Weight u.conv1Bias u.conv1Dilation
  let h3 := snakeBeta h2 u.act2Alpha u.act2Beta
  let h4 := causalConv1d h3 u.conv2Weight u.conv2Bias
  Tensor.Promoting.add residual h4

private def forwardDecoderBlock {batch frames inDim outDim upRate : UInt64}
    (blk : DecoderBlock (device := device) inDim outDim upRate)
    (x : Tensor { shape := #[batch, inDim, frames], device := device }) : Tensor { shape := #[batch, outDim, frames * upRate], device := device } :=
  let h0 := snakeBeta x blk.snakeAlpha blk.snakeBeta
  let h1 : Tensor { shape := #[batch, outDim, frames * upRate], device := device } :=
    Tensor.reshapeChecked (causalTransConv1d (batch := batch) (inC := inDim) (outC := outDim)
      (frames := frames) (kernel := 2 * upRate) (stride := upRate) (rightPad := upRate)
      h0 blk.transWeight blk.transBias) #[batch, outDim, frames * upRate]
  let h2 := forwardResidualUnit blk.res1 h1
  let h3 := forwardResidualUnit blk.res2 h2
  forwardResidualUnit blk.res3 h3

private def forwardUpsampleStage {batch frames : UInt64}
    (stage : (UpsampleStage (device := device)))
    (x : Tensor { shape := #[batch, 1024, frames], device := device }) : Tensor { shape := #[batch, 1024, frames * 2], device := device } :=
  let h0 : Tensor { shape := #[batch, 1024, frames * 2], device := device } :=
    Tensor.reshapeChecked (causalTransConv1d (batch := batch) (inC := 1024) (outC := 1024)
      (frames := frames) (kernel := 2) (stride := 2) (rightPad := 0)
      x stage.transConvWeight stage.transConvBias) #[batch, 1024, frames * 2]
  forwardConvNeXt stage.convNeXt h0

/-- Decode quantizer codes `[batch, 16, frames]` to waveform `[batch, 1, frames*1920]`. -/
def decode {batch frames : UInt64}
    (m : (SpeechTokenizer12HzDecoder (device := device)))
    (codes : Tensor { shape := #[batch, 16, frames], dtype := .Int64, device := device }) : Tensor { shape := #[batch, 1, frames * 1920], device := device } :=
  let q : Tensor { shape := #[batch, 512, frames], device := device } := decodeSplitRVQ m codes
  let h0 : Tensor { shape := #[batch, 1024, frames], device := device } := causalConv1d q m.preConvWeight m.preConvBias
  let h1 : Tensor { shape := #[batch, 1024, frames], device := device } := forwardPreTransformer m h0

  let u0 : Tensor { shape := #[batch, 1024, frames * 2], device := device } := forwardUpsampleStage m.upsample0 h1
  let u1 : Tensor { shape := #[batch, 1024, frames * 4], device := device } := Tensor.reshapeChecked (forwardUpsampleStage m.upsample1 u0) #[batch, 1024, frames * 4]

  let d0 : Tensor { shape := #[batch, 1536, frames * 4], device := device } := causalConv1d u1 m.decoderConv0Weight m.decoderConv0Bias
  let d1 : Tensor { shape := #[batch, 768, frames * 32], device := device } :=
    Tensor.reshapeChecked (forwardDecoderBlock m.decoderBlock1 d0) #[batch, 768, frames * 32]
  let d2 : Tensor { shape := #[batch, 384, frames * 160], device := device } :=
    Tensor.reshapeChecked (forwardDecoderBlock m.decoderBlock2 d1) #[batch, 384, frames * 160]
  let d3 : Tensor { shape := #[batch, 192, frames * 640], device := device } :=
    Tensor.reshapeChecked (forwardDecoderBlock m.decoderBlock3 d2) #[batch, 192, frames * 640]
  let d4 : Tensor { shape := #[batch, 96, frames * 1920], device := device } :=
    Tensor.reshapeChecked (forwardDecoderBlock m.decoderBlock4 d3) #[batch, 96, frames * 1920]

  let f0 : Tensor { shape := #[batch, 96, frames * 1920], device := device } := snakeBeta d4 m.finalSnakeAlpha m.finalSnakeBeta
  let f1 : Tensor { shape := #[batch, 1, frames * 1920], device := device } := causalConv1d f0 m.finalConvWeight m.finalConvBias
  clampWave f1

/-- Chunked decoder for long-form/streaming inference.
    Mirrors upstream chunk strategy:
    - decode chunks of `chunkSize` codec frames
    - prepend up to `leftContextSize` previous frames for continuity
    - drop context samples from each chunk output before concatenation -/
def decodeChunked {batch frames : UInt64}
    (m : (SpeechTokenizer12HzDecoder (device := device)))
    (codes : Tensor { shape := #[batch, 16, frames], dtype := .Int64, device := device })
    (chunkSize : UInt64 := 300)
    (leftContextSize : UInt64 := 25)
    : Tensor { shape := #[batch, 1, frames * 1920], device := device } :=
  if frames == 0 then
    Tensor.reshapeChecked (Tensor.castLike m.finalConvWeight (Tensor.zeros #[batch, 1, 0] false codes.device)) #[batch, 1, frames * 1920]
  else
    Id.run do
      let chunkSize := if chunkSize == 0 then 1 else chunkSize
      let mut start : UInt64 := 0
      let mut chunks : Array (Sigma (fun shape => Tensor { shape := shape, device := device })) := #[]
      while start < frames do
        let remaining := frames - start
        let curChunk := if remaining < chunkSize then remaining else chunkSize
        let context := if start < leftContextSize then start else leftContextSize
        let chunkStart := start - context
        let chunkFrames := context + curChunk
        let codesChunk : Tensor { shape := #[batch, 16, chunkFrames], dtype := .Int64, device := device } := Tensor.sliceChecked codes 2 chunkStart chunkFrames #[batch, 16, chunkFrames]
        let wavChunk : Tensor { shape := #[batch, 1, chunkFrames * 1920], device := device } := decode m codesChunk
        let keepStart : UInt64 := context * 1920
        let keepSamples : UInt64 := curChunk * 1920
        let wavKeep : Tensor { shape := #[batch, 1, keepSamples], device := device } := Tensor.sliceChecked wavChunk 2 keepStart keepSamples #[batch, 1, keepSamples]
        chunks := chunks.push wavKeep.packShape
        start := start + curChunk
      let wavDyn := Tensor.catManyChecked chunks 2 #[batch, 1, frames * 1920]
      Tensor.reshapeChecked wavDyn #[batch, 1, frames * 1920]

/-- Streaming decoder state for incremental 12Hz codec-frame ingestion. -/
structure DecodeStreamState (batch : UInt64) where
  totalFrames : UInt64 := 0
  codesDyn : Tensor { shape := #[batch, 16, totalFrames], dtype := .Int64, device := device }
  emittedFrames : UInt64 := 0
  chunkSize : UInt64 := 1
  leftContextSize : UInt64 := 25

/-- Initialize an empty incremental decoder state. -/
def initDecodeStreamState {batch : UInt64}
    (chunkSize : UInt64 := 1)
    (leftContextSize : UInt64 := 25)
    (device : Device := Device.CPU)
    : DecodeStreamState (device := device) batch :=
  let emptyCodes0 : Tensor { shape := #[batch, 16, 0], dtype := .Int64, device := .CPU } := Tensor.fullInt #[batch, 16, 0] 0
  let emptyCodes : Tensor { shape := #[batch, 16, 0], dtype := .Int64, device := device } := emptyCodes0.to device
  {
    codesDyn := emptyCodes
    totalFrames := 0
    emittedFrames := 0
    chunkSize := if chunkSize == 0 then 1 else chunkSize
    leftContextSize
  }

private def decodeOneStreamingChunk {batch : UInt64}
    (m : (SpeechTokenizer12HzDecoder (device := device)))
    (st : DecodeStreamState (device := device) batch)
    (force : Bool := false)
    : DecodeStreamState (device := device) batch × Option (Sigma fun samples => Tensor { shape := #[batch, 1, samples], device := device }) :=
  let remaining := st.totalFrames - st.emittedFrames
  if remaining == 0 then
    (st, none)
  else if !force && remaining < st.chunkSize then
    (st, none)
  else
    let curChunk := if remaining < st.chunkSize then remaining else st.chunkSize
    let start := st.emittedFrames
    let context := if start < st.leftContextSize then start else st.leftContextSize
    let chunkStart := start - context
    let chunkFrames := context + curChunk
    let allCodes : Tensor { shape := #[batch, 16, st.totalFrames], dtype := .Int64, device := device } := Tensor.reshapeChecked st.codesDyn #[batch, 16, st.totalFrames]
    let codesChunk : Tensor { shape := #[batch, 16, chunkFrames], dtype := .Int64, device := device } := Tensor.sliceChecked allCodes 2 chunkStart chunkFrames #[batch, 16, chunkFrames]
    let wavChunk : Tensor { shape := #[batch, 1, chunkFrames * 1920], device := device } := decode m codesChunk
    let keepStart : UInt64 := context * 1920
    let keepSamples : UInt64 := curChunk * 1920
    let wavKeep : Tensor { shape := #[batch, 1, keepSamples], device := device } := Tensor.sliceChecked wavChunk 2 keepStart keepSamples #[batch, 1, keepSamples]
    let st' := { st with emittedFrames := st.emittedFrames + curChunk }
    (st', some ⟨keepSamples, wavKeep⟩)

/-- Push new codec frames into decoder state and emit all ready waveform chunks.
    `newCodes` has shape `[batch, 16, newFrames]`. -/
def pushDecodeStream {batch newFrames : UInt64}
    (m : (SpeechTokenizer12HzDecoder (device := device)))
    (st : DecodeStreamState (device := device) batch)
    (newCodes : Tensor { shape := #[batch, 16, newFrames], dtype := .Int64, device := device })
    : DecodeStreamState (device := device) batch × Array (Sigma fun samples => Tensor { shape := #[batch, 1, samples], device := device }) :=
  Id.run do
    let curCodes : Tensor { shape := #[batch, 16, st.totalFrames], dtype := .Int64, device := device } := Tensor.reshapeChecked st.codesDyn #[batch, 16, st.totalFrames]
    let allCodes : Tensor { shape := #[batch, 16, st.totalFrames + newFrames], dtype := .Int64, device := device } := Tensor.cat curCodes newCodes 2
    let mut st : DecodeStreamState (device := device) batch := {
      st with
      codesDyn := allCodes
      totalFrames := st.totalFrames + newFrames
    }
    let mut out : Array (Sigma fun samples => Tensor { shape := #[batch, 1, samples], device := device }) := #[]
    let mut done := false
    while !done do
      let (stNext, chunk?) := decodeOneStreamingChunk m st false
      st := stNext
      match chunk? with
      | some chunk => out := out.push chunk
      | none => done := true
    (st, out)

/-- Flush remaining pending frames from a decode stream. -/
def flushDecodeStream {batch : UInt64}
    (m : (SpeechTokenizer12HzDecoder (device := device)))
    (st : DecodeStreamState (device := device) batch)
    : DecodeStreamState (device := device) batch × Array (Sigma fun samples => Tensor { shape := #[batch, 1, samples], device := device }) :=
  Id.run do
    let mut st := st
    let mut out : Array (Sigma fun samples => Tensor { shape := #[batch, 1, samples], device := device }) := #[]
    let mut done := false
    while !done do
      let (stNext, chunk?) := decodeOneStreamingChunk m st true
      st := stNext
      match chunk? with
      | some chunk => out := out.push chunk
      | none => done := true
    (st, out)

/-- Convenience decode for frame-major codec tensor `[frames, 16]`. -/
def decodeFrameMajor {frames : UInt64}
    (m : (SpeechTokenizer12HzDecoder (device := device)))
    (codes : Tensor { shape := #[frames, 16], dtype := .Int64, device := device }) : Tensor { shape := #[1, 1, frames * 1920], device := device } :=
  let c0 : Tensor { shape := #[1, frames, 16], dtype := .Int64, device := device } := Tensor.reshapeChecked codes #[1, frames, 16]
  let c1 : Tensor { shape := #[1, 16, frames], dtype := .Int64, device := device } := Tensor.reshapeChecked (Tensor.transpose c0 1 2) #[1, 16, frames]
  decode m c1

/-- Chunked convenience decode for frame-major codec tensor `[frames, 16]`. -/
def decodeFrameMajorChunked {frames : UInt64}
    (m : (SpeechTokenizer12HzDecoder (device := device)))
    (codes : Tensor { shape := #[frames, 16], dtype := .Int64, device := device })
    (chunkSize : UInt64 := 300)
    (leftContextSize : UInt64 := 25)
    : Tensor { shape := #[1, 1, frames * 1920], device := device } :=
  let c0 : Tensor { shape := #[1, frames, 16], dtype := .Int64, device := device } := Tensor.reshapeChecked codes #[1, frames, 16]
  let c1 : Tensor { shape := #[1, 16, frames], dtype := .Int64, device := device } := Tensor.reshapeChecked (Tensor.transpose c0 1 2) #[1, 16, frames]
  decodeChunked m c1 chunkSize leftContextSize

/-- Decode `[frames,16]` codec tensor and write mono WAV. -/
def decodeFrameMajorToWav {frames : UInt64}
    (m : (SpeechTokenizer12HzDecoder (device := device)))
    (codes : Tensor { shape := #[frames, 16], dtype := .Int64, device := device })
    (wavPath : String)
    : IO Unit := do
  let wav : Tensor { shape := #[1, 1, frames * 1920], device := device } := decodeFrameMajor m codes
  Tensor.saveWav wav wavPath m.outputSampleRate

/-- Chunked decode `[frames,16]` codec tensor and write mono WAV. -/
def decodeFrameMajorChunkedToWav {frames : UInt64}
    (m : (SpeechTokenizer12HzDecoder (device := device)))
    (codes : Tensor { shape := #[frames, 16], dtype := .Int64, device := device })
    (wavPath : String)
    (chunkSize : UInt64 := 300)
    (leftContextSize : UInt64 := 25)
    : IO Unit := do
  let wav : Tensor { shape := #[1, 1, frames * 1920], device := device } := decodeFrameMajorChunked m codes chunkSize leftContextSize
  Tensor.saveWav wav wavPath m.outputSampleRate

end SpeechTokenizer12HzDecoder

end torch.qwen3tts
