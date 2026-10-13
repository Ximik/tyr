/-
  Tyr/Model/Qwen3TTS/SpeechTokenizer25HzEncoder.lean

  Lean-native Qwen3-TTS 25Hz speech-tokenizer encoder (waveform -> codec IDs).

  Scope:
  - 25Hz encoder-side architecture (Whisper-style frontend + transformer + VQ encode)
  - real weight loading from `speech_tokenizer/model.safetensors`
  - inference-only codec ID emission

  Notes:
  - This module targets the released 25Hz encoder defaults used by
    `Qwen3TTSTokenizerV1Encoder`.
  - Decoder-side 25Hz architecture (DiT + BigVGAN) is handled separately.
-/
import Tyr.Typed.Neural
import Tyr.SafeTensors.Load
import Tyr.TensorStruct
import Tyr.Module.Derive
import Tyr.Model.Qwen3ASR.Frontend
import Tyr.Model.Qwen3ASR.PreprocessorConfig
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace torch.qwen3tts

open Lean
open torch.qwen3asr

/-! ## Config parsing -/

structure SpeechTokenizer25HzEncoderConfig where
  modelType : String := "qwen3_tts_tokenizer_25hz"
  inputSampleRate : UInt64 := 24000
  encodeDownsampleRate : UInt64 := 1920
  nMels : UInt64 := 128
  nCtx : UInt64 := 1500
  nState : UInt64 := 1280
  nHead : UInt64 := 20
  nLayer : UInt64 := 32
  nWindow : UInt64 := 100
  audioVqType : String := "GRVQ"
  audioVqLayers : UInt64 := 6
  audioVqCodebookSize : UInt64 := 32768
  audioVqCodebookDim : UInt64 := 1280
  audioVqDsRate : UInt64 := 2
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

private def getStr? (j : Json) : Option String :=
  match j with
  | .str s => some s
  | _ => none

private def getNat? (j : Json) : Option Nat :=
  match (FromJson.fromJson? j : Except String Nat) with
  | .ok n => some n
  | .error _ => none

private def getUInt64FieldD (j : Json) (key : String) (d : UInt64) : UInt64 :=
  match getObjVal? j key >>= getNat? with
  | some n => n.toUInt64
  | none => d

private def getStrFieldD (j : Json) (key : String) (d : String) : String :=
  match getObjVal? j key >>= getStr? with
  | some s => s
  | none => d

private def requireTrue (ok : Bool) (msg : String) : IO Unit := do
  unless ok do
    throw (IO.userError msg)

namespace SpeechTokenizer25HzEncoderConfig

def loadFromFile (path : String) (defaults : SpeechTokenizer25HzEncoderConfig := {})
    : IO SpeechTokenizer25HzEncoderConfig := do
  let root ← parseJsonFile path
  let encCfg := (getObjVal? root "encoder_config").getD (Json.obj ∅)
  pure {
    modelType := getStrFieldD root "model_type" defaults.modelType
    inputSampleRate := getUInt64FieldD root "input_sample_rate" defaults.inputSampleRate
    encodeDownsampleRate := getUInt64FieldD root "encode_downsample_rate" defaults.encodeDownsampleRate
    nMels := getUInt64FieldD encCfg "n_mels" defaults.nMels
    nCtx := getUInt64FieldD encCfg "n_ctx" defaults.nCtx
    nState := getUInt64FieldD encCfg "n_state" defaults.nState
    nHead := getUInt64FieldD encCfg "n_head" defaults.nHead
    nLayer := getUInt64FieldD encCfg "n_layer" defaults.nLayer
    nWindow := getUInt64FieldD encCfg "n_window" defaults.nWindow
    audioVqType := getStrFieldD encCfg "audio_vq_type" defaults.audioVqType
    audioVqLayers := getUInt64FieldD encCfg "audio_vq_layers" defaults.audioVqLayers
    audioVqCodebookSize := getUInt64FieldD encCfg "audio_vq_codebook_size" defaults.audioVqCodebookSize
    audioVqCodebookDim := getUInt64FieldD encCfg "audio_vq_codebook_dim" defaults.audioVqCodebookDim
    audioVqDsRate := getUInt64FieldD encCfg "audio_vq_ds_rate" defaults.audioVqDsRate
  }

def validateSupported (cfg : SpeechTokenizer25HzEncoderConfig) : IO Unit := do
  requireTrue (cfg.modelType == "qwen3_tts_tokenizer_25hz")
    s!"Unsupported speech tokenizer model_type={cfg.modelType} (expected qwen3_tts_tokenizer_25hz)"
  requireTrue (cfg.nMels == 128)
    s!"Unsupported n_mels={cfg.nMels} (expected 128)"
  requireTrue (cfg.nCtx == 1500)
    s!"Unsupported n_ctx={cfg.nCtx} (expected 1500)"
  requireTrue (cfg.nState == 1280)
    s!"Unsupported n_state={cfg.nState} (expected 1280)"
  requireTrue (cfg.nHead == 20)
    s!"Unsupported n_head={cfg.nHead} (expected 20)"
  requireTrue (cfg.nLayer >= cfg.audioVqLayers)
    s!"Invalid n_layer={cfg.nLayer}, audio_vq_layers={cfg.audioVqLayers}"
  requireTrue (cfg.nWindow == 100)
    s!"Unsupported n_window={cfg.nWindow} (expected 100)"
  requireTrue (cfg.audioVqType == "GRVQ")
    s!"Unsupported audio_vq_type={cfg.audioVqType} (expected GRVQ)"
  requireTrue (cfg.audioVqLayers == 6)
    s!"Unsupported audio_vq_layers={cfg.audioVqLayers} (expected 6)"
  requireTrue (cfg.audioVqCodebookSize == 32768)
    s!"Unsupported audio_vq_codebook_size={cfg.audioVqCodebookSize} (expected 32768)"
  requireTrue (cfg.audioVqCodebookDim == 1280)
    s!"Unsupported audio_vq_codebook_dim={cfg.audioVqCodebookDim} (expected 1280)"
  requireTrue (cfg.audioVqDsRate == 2)
    s!"Unsupported audio_vq_ds_rate={cfg.audioVqDsRate} (expected 2)"

end SpeechTokenizer25HzEncoderConfig

/-! ## Parameters -/

private def freeze {s : Shape} (t : STensor s) : STensor s :=
  Tensor.setRequiresGrad t false

private def loadFrozen (path : String) (name : String) (s : Shape) : IO (STensor s) := do
  let t ← safetensors.loadTensorCandidates path #[name] s
  pure (freeze t)

private def loadFrozenOrZeros (path : String) (name : String) (s : Shape) : IO (STensor s) := do
  try
    loadFrozen path name s
  catch _ =>
    pure (freeze (Tensor.zeros s).forgetDType)

private def addBias2d {n d : UInt64}
    (x : STensor #[n, d])
    (bias : STensor #[d]) : STensor #[n, d] :=
  let b : STensor #[1, d] := Tensor.reshapeChecked bias #[1, d]
  x + Tensor.expand b #[n, d]

private def addBias3d {batch channels frames : UInt64}
    (x : STensor #[batch, channels, frames])
    (bias : STensor #[channels]) : STensor #[batch, channels, frames] :=
  let b : STensor #[1, channels, 1] := Tensor.reshapeChecked bias #[1, channels, 1]
  x + Tensor.expand b #[batch, channels, frames]

private def conv1dWithBias {batch inC outC inFrames outFrames kernel : UInt64}
    (x : STensor #[batch, inC, inFrames])
    (w : STensor #[outC, inC, kernel])
    (b : STensor #[outC])
    (stride pad dilation : UInt64) : STensor #[batch, outC, outFrames] :=
  let y0 : STensor #[batch, outC, outFrames] := Tensor.reshapeChecked (Tensor.conv1d x w stride pad dilation) #[batch, outC, outFrames]
  addBias3d y0 b

private def affine3d {batch seq inDim outDim : UInt64}
    (x : STensor #[batch, seq, inDim])
    (w : STensor #[outDim, inDim])
    (b : STensor #[outDim]) : STensor #[batch, seq, outDim] :=
  let x2 : STensor #[batch * seq, inDim] := Tensor.reshapeChecked x #[batch * seq, inDim]
  let y0 : STensor #[batch * seq, outDim] := Tensor.mm x2 (Tensor.transpose2d w)
  let y1 : STensor #[batch * seq, outDim] := addBias2d y0 b
  Tensor.reshapeChecked y1 #[batch, seq, outDim]

private def scale3d {batch seq dim : UInt64}
    (x : STensor #[batch, seq, dim])
    (scale : STensor #[dim]) : STensor #[batch, seq, dim] :=
  let s : STensor #[1, 1, dim] := Tensor.reshapeChecked scale #[1, 1, dim]
  x * Tensor.expand s #[batch, seq, dim]

private def conv2Frames (melFrames : UInt64) : UInt64 :=
  (melFrames + 1) / 2

private def vqFrames (melFrames : UInt64) : UInt64 :=
  (conv2Frames melFrames) / 2

structure EncoderBlock25Hz where
  attnLnWeight : STensor #[1280]
  attnLnBias : STensor #[1280]
  qWeight : STensor #[1280, 1280]
  qBias : STensor #[1280]
  kWeight : STensor #[1280, 1280]
  kBias : STensor #[1280]
  vWeight : STensor #[1280, 1280]
  vBias : STensor #[1280]
  oWeight : STensor #[1280, 1280]
  oBias : STensor #[1280]
  mlpLnWeight : STensor #[1280]
  mlpLnBias : STensor #[1280]
  fc1Weight : STensor #[5120, 1280]
  fc1Bias : STensor #[5120]
  fc2Weight : STensor #[1280, 5120]
  fc2Bias : STensor #[1280]
  deriving TensorStruct

structure SpeechTokenizer25HzEncoder where
  conv1Weight : STensor #[1280, 128, 3]
  conv1Bias : STensor #[1280]
  conv2Weight : STensor #[1280, 1280, 3]
  conv2Bias : STensor #[1280]
  positionalEmbedding : STensor #[1500, 1280]
  blocks : Array EncoderBlock25Hz
  vqDownWeight : STensor #[1280, 1280, 2]
  vqDownBias : STensor #[1280]
  vqCodebook : STensor #[32768, 1280]
  inputSampleRate : UInt64 := 24000
  encodeDownsampleRate : UInt64 := 1920
  deriving TensorStruct

namespace SpeechTokenizer25HzEncoder

private def loadBlock (weightsPath : String) (i : Nat) : IO EncoderBlock25Hz := do
  let p := s!"encoder.tokenizer.blocks.{i}"
  let attnLnWeight ← loadFrozen weightsPath s!"{p}.attn_ln.weight" #[1280]
  let attnLnBias ← loadFrozen weightsPath s!"{p}.attn_ln.bias" #[1280]
  let qWeight ← loadFrozen weightsPath s!"{p}.attn.query.weight" #[1280, 1280]
  let qBias ← loadFrozen weightsPath s!"{p}.attn.query.bias" #[1280]
  let kWeight ← loadFrozen weightsPath s!"{p}.attn.key.weight" #[1280, 1280]
  let kBias ← loadFrozenOrZeros weightsPath s!"{p}.attn.key.bias" #[1280]
  let vWeight ← loadFrozen weightsPath s!"{p}.attn.value.weight" #[1280, 1280]
  let vBias ← loadFrozen weightsPath s!"{p}.attn.value.bias" #[1280]
  let oWeight ← loadFrozen weightsPath s!"{p}.attn.out.weight" #[1280, 1280]
  let oBias ← loadFrozen weightsPath s!"{p}.attn.out.bias" #[1280]

  let mlpLnWeight ← loadFrozen weightsPath s!"{p}.mlp_ln.weight" #[1280]
  let mlpLnBias ← loadFrozen weightsPath s!"{p}.mlp_ln.bias" #[1280]
  let fc1Weight ← loadFrozen weightsPath s!"{p}.mlp.0.weight" #[5120, 1280]
  let fc1Bias ← loadFrozen weightsPath s!"{p}.mlp.0.bias" #[5120]
  let fc2Weight ← loadFrozen weightsPath s!"{p}.mlp.2.weight" #[1280, 5120]
  let fc2Bias ← loadFrozen weightsPath s!"{p}.mlp.2.bias" #[1280]

  pure {
    attnLnWeight
    attnLnBias
    qWeight
    qBias
    kWeight
    kBias
    vWeight
    vBias
    oWeight
    oBias
    mlpLnWeight
    mlpLnBias
    fc1Weight
    fc1Bias
    fc2Weight
    fc2Bias
  }

/-- Load 25Hz encoder-side tokenizer weights from `speech_tokenizer` directory. -/
def loadFromDir (speechTokenizerDir : String) (device : Device := Device.CPU) : IO SpeechTokenizer25HzEncoder := do
  let cfg ← SpeechTokenizer25HzEncoderConfig.loadFromFile s!"{speechTokenizerDir}/config.json"
  SpeechTokenizer25HzEncoderConfig.validateSupported cfg

  let weightsPath := s!"{speechTokenizerDir}/model.safetensors"
  let conv1Weight ← loadFrozen weightsPath "encoder.tokenizer.conv1.weight" #[1280, 128, 3]
  let conv1Bias ← loadFrozen weightsPath "encoder.tokenizer.conv1.bias" #[1280]
  let conv2Weight ← loadFrozen weightsPath "encoder.tokenizer.conv2.weight" #[1280, 1280, 3]
  let conv2Bias ← loadFrozen weightsPath "encoder.tokenizer.conv2.bias" #[1280]
  let positionalEmbedding ← loadFrozen weightsPath "encoder.tokenizer.positional_embedding" #[1500, 1280]

  let mut blocks : Array EncoderBlock25Hz := #[]
  for i in [:6] do
    let b ← loadBlock weightsPath i
    blocks := blocks.push b

  let vqDownWeight ← loadFrozen weightsPath "encoder.tokenizer.audio_vq_downsample.weight" #[1280, 1280, 2]
  let vqDownBias ← loadFrozen weightsPath "encoder.tokenizer.audio_vq_downsample.bias" #[1280]
  let vqEmbedRaw ← loadFrozen weightsPath "encoder.tokenizer.audio_quantizer.rvqs.0.embed" #[1, 32768, 1280]
  let vqEmbed3 : STensor #[1, 32768, 1280] := Tensor.sliceChecked vqEmbedRaw 0 0 1 #[1, 32768, 1280]
  let vqCodebook : STensor #[32768, 1280] := Tensor.reshapeChecked vqEmbed3 #[32768, 1280]

  let enc : SpeechTokenizer25HzEncoder := {
    conv1Weight
    conv1Bias
    conv2Weight
    conv2Bias
    positionalEmbedding
    blocks
    vqDownWeight
    vqDownBias
    vqCodebook
    inputSampleRate := cfg.inputSampleRate
    encodeDownsampleRate := cfg.encodeDownsampleRate
  }
  pure (TensorStruct.map (fun t => t.to device) enc)

private def forwardAttention {seq : UInt64}
    (blk : EncoderBlock25Hz)
    (x : STensor #[1, seq, 1280]) : STensor #[1, seq, 1280] :=
  let q0 : STensor #[1, seq, 1280] := affine3d x blk.qWeight blk.qBias
  let k0 : STensor #[1, seq, 1280] := affine3d x blk.kWeight blk.kBias
  let v0 : STensor #[1, seq, 1280] := affine3d x blk.vWeight blk.vBias

  let q : STensor #[1, seq, 20, 64] := Tensor.reshapeChecked q0 #[1, seq, 20, 64]
  let k : STensor #[1, seq, 20, 64] := Tensor.reshapeChecked k0 #[1, seq, 20, 64]
  let v : STensor #[1, seq, 20, 64] := Tensor.reshapeChecked v0 #[1, seq, 20, 64]

  let qh : STensor #[1, 20, seq, 64] := Tensor.splitHeads q
  let kh : STensor #[1, 20, seq, 64] := Tensor.splitHeads k
  let vh : STensor #[1, 20, seq, 64] := Tensor.splitHeads v

  let attn : STensor #[1, 20, seq, 64] := Tensor.attentionGQA qh kh vh 0.0 false
  let attn2 : STensor #[1, seq, 20, 64] := Tensor.joinHeads attn
  let attn3 : STensor #[1, seq, 1280] := Tensor.reshapeChecked attn2 #[1, seq, 1280]
  affine3d attn3 blk.oWeight blk.oBias

private def forwardBlock {seq : UInt64}
    (blk : EncoderBlock25Hz)
    (x : STensor #[1, seq, 1280]) : STensor #[1, seq, 1280] :=
  let h0 : STensor #[1, seq, 1280] := Tensor.layerNorm x blk.attnLnWeight blk.attnLnBias 1e-5
  let a : STensor #[1, seq, 1280] := forwardAttention blk h0
  let x1 : STensor #[1, seq, 1280] := x + a

  let h1 : STensor #[1, seq, 1280] := Tensor.layerNorm x1 blk.mlpLnWeight blk.mlpLnBias 1e-5
  let m0 : STensor #[1, seq, 5120] := affine3d h1 blk.fc1Weight blk.fc1Bias
  let m1 : STensor #[1, seq, 5120] := Tensor.Preserving.gelu m0
  let m2 : STensor #[1, seq, 1280] := affine3d m1 blk.fc2Weight blk.fc2Bias
  x1 + m2

private def encodeMelChunk {melFrames : UInt64}
    (m : SpeechTokenizer25HzEncoder)
    (mel : STensor #[128, melFrames]) : DTensor #[vqFrames melFrames] .Int64 :=
  let c2Frames : UInt64 := conv2Frames melFrames
  let qFrames : UInt64 := vqFrames melFrames
  if melFrames < 2 || qFrames == 0 then
    Tensor.reshapeChecked (Tensor.fullInt #[0] 0) #[qFrames]
  else
    Id.run do
      let x0 : STensor #[1, 128, melFrames] := Tensor.reshapeChecked mel #[1, 128, melFrames]
      let h1 : STensor #[1, 1280, melFrames] :=
        Tensor.Preserving.gelu (conv1dWithBias x0 m.conv1Weight m.conv1Bias 1 1 1)
      let h2 : STensor #[1, 1280, c2Frames] :=
        Tensor.Preserving.gelu (conv1dWithBias h1 m.conv2Weight m.conv2Bias 2 1 1)

      let xSeq0 : STensor #[1, c2Frames, 1280] := Tensor.reshapeChecked (Tensor.transpose h2 1 2) #[1, c2Frames, 1280]
      let pos : STensor #[c2Frames, 1280] := Tensor.sliceChecked m.positionalEmbedding 0 0 c2Frames #[c2Frames, 1280]
      let pos3 : STensor #[1, c2Frames, 1280] := Tensor.reshapeChecked pos #[1, c2Frames, 1280]
      let mut xSeq : STensor #[1, c2Frames, 1280] := xSeq0 + pos3

      for blk in m.blocks do
        xSeq := forwardBlock blk xSeq

      let vqIn : STensor #[1, 1280, c2Frames] := Tensor.reshapeChecked (Tensor.transpose xSeq 1 2) #[1, 1280, c2Frames]
      let vqDown : STensor #[1, 1280, qFrames] :=
        conv1dWithBias vqIn m.vqDownWeight m.vqDownBias 2 0 1
      let vqSeq : STensor #[qFrames, 1280] :=
        Tensor.reshapeChecked (Tensor.transpose (Tensor.reshapeChecked vqDown #[1, 1280, qFrames]) 1 2) #[qFrames, 1280]

      let x2 : STensor #[qFrames, 1] := Tensor.reshapeChecked (Tensor.Preserving.sumDim (vqSeq * vqSeq) 1 true) #[qFrames, 1]
      let e2 : STensor #[32768, 1] := Tensor.reshapeChecked (Tensor.Preserving.sumDim (m.vqCodebook * m.vqCodebook) 1 true) #[32768, 1]
      let dot : STensor #[qFrames, 32768] := Tensor.mm vqSeq (Tensor.transpose2d m.vqCodebook)
      let x2e : STensor #[qFrames, 32768] := Tensor.expand x2 #[qFrames, 32768]
      let e2Row : STensor #[1, 32768] := Tensor.reshapeChecked (Tensor.transpose2d e2) #[1, 32768]
      let e2e : STensor #[qFrames, 32768] := Tensor.expand e2Row #[qFrames, 32768]
      let dist : STensor #[qFrames, 32768] := x2e + e2e - Tensor.Preserving.mulScalar dot 2.0
      Tensor.reshapeChecked (Tensor.argmax (Tensor.Preserving.mulScalar dist (-1.0)) 1) #[vqFrames melFrames]

private def appendAll (dst src : Array UInt64) : Array UInt64 :=
  Id.run do
    let mut out := dst
    for x in src do
      out := out.push x
    out

/-- Encode mono waveform to flat codec IDs for the 25Hz tokenizer architecture.
    Returns `(numCodeGroups, flatCodes)` where `numCodeGroups = 1`. -/
def encodeWaveToCodes
    (m : SpeechTokenizer25HzEncoder)
    (wave : Array Float)
    : IO (UInt64 × Array UInt64) := do
  if wave.isEmpty then
    throw <| IO.userError "Audio encode input is empty"

  let preCfg : PreprocessorConfig := {
    featureSize := 128
    samplingRate := m.inputSampleRate
    hopLength := 160
    nFft := 400
    paddingValue := 0.0
    returnAttentionMask := false
    doNormalize := false
    dither := 0.0
  }

  let ⟨melFrames, frontendOut⟩ ← waveformToWhisperFeaturesDynamic preCfg wave 0.0 none
  let feats : STensor #[1, 128, melFrames] := Tensor.castLike m.conv1Weight frontendOut.inputFeatures

  let chunkMel : UInt64 := 200
  let mut start : UInt64 := 0
  let mut outVals : Array UInt64 := #[]
  while start < melFrames do
    let remaining := melFrames - start
    let cur := if remaining < chunkMel then remaining else chunkMel
    if cur >= 2 then
      let melChunk3 : STensor #[1, 128, cur] := Tensor.sliceChecked feats 2 start cur #[1, 128, cur]
      let melChunk : STensor #[128, cur] := Tensor.reshapeChecked melChunk3 #[128, cur]
      let idx : DTensor #[vqFrames cur] .Int64 := encodeMelChunk m melChunk
      let vals ← Tensor.toUInt64Array idx
      outVals := appendAll outVals vals
    start := start + cur

  pure (1, outVals)

end SpeechTokenizer25HzEncoder

end torch.qwen3tts
