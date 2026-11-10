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
import Tyr.SafeTensors.Source
import Tyr.Model.Qwen3TTS.DTypes
import Tyr.TensorStruct
import Tyr.Module.Derive
import Tyr.Model.Qwen3ASR.Frontend
import Tyr.Model.Qwen3ASR.PreprocessorConfig
import Tyr.Model.ConfigJson

namespace torch.qwen3tts

variable {device : Device} {dtype : RealFloatingDType}

open Lean
open torch.qwen3asr

/-! ## Config parsing -/

/-- Qwen3-TTS 25 Hz tokenizer encoder metadata. The encoder architecture is
fixed and `loadFromFile` rejects any other value; only the waveform rates vary. -/
structure SpeechTokenizer25HzEncoderConfig where
  inputSampleRate : Nat := 24000
  encodeDownsampleRate : Nat := 1920
  /-- Audio is framed at a positive rate; checked when the config is built. -/
  ratesPos : 0 < inputSampleRate ∧ 0 < encodeDownsampleRate := by decide
  deriving Repr

namespace SpeechTokenizer25HzEncoderConfig

open ConfigJson

def loadFromJson (root : Json) (defaults : SpeechTokenizer25HzEncoderConfig := {}) :
    Except String SpeechTokenizer25HzEncoderConfig := do
  requireOneOf root "model_type" ["qwen3_tts_tokenizer_25hz"]
  if let some e ← objectField? root "encoder_config" then
    for (key, value) in [("n_mels", 128), ("n_ctx", 1500), ("n_state", 1280), ("n_head", 20),
        ("n_window", 100), ("audio_vq_layers", 6), ("audio_vq_codebook_size", 32768),
        ("audio_vq_codebook_dim", 1280), ("audio_vq_ds_rate", 2)] do
      requireEq e key (value : Nat)
    requireOneOf e "audio_vq_type" ["GRVQ"]
    let nLayer : Nat ← fieldD? e "n_layer" 32
    unless 6 ≤ nLayer do
      throw s!"Invalid n_layer={nLayer}: the six quantized encoder blocks come first"
  let inputSampleRate ← fieldD? root "input_sample_rate" defaults.inputSampleRate
  let encodeDownsampleRate ← fieldD? root "encode_downsample_rate" defaults.encodeDownsampleRate
  let ⟨hRates⟩ ← if h : 0 < inputSampleRate ∧ 0 < encodeDownsampleRate then pure (PLift.up h) else
    throw s!"Invalid input_sample_rate={inputSampleRate}/encode_downsample_rate={encodeDownsampleRate}: both must be positive"
  pure { inputSampleRate, encodeDownsampleRate, ratesPos := hRates }

def loadFromFile (path : String) (defaults : SpeechTokenizer25HzEncoderConfig := {}) :
    IO SpeechTokenizer25HzEncoderConfig := do
  IO.ofExcept (loadFromJson (← parseFile path) defaults)

end SpeechTokenizer25HzEncoderConfig

/-! ## Parameters -/

private def loadFrozen (path : String) (name : String) (s : Shape)
    : IO (Tensor { shape := s, dtype := dtype.val, device := device }) := do
  let packed ← (← safetensors.Source.open path).load name s device
  match packed.atDType? dtype.val with
  | .ok t => pure (Tensor.detach t)
  | .error message => throw <| IO.userError s!"Speech tokenizer tensor '{name}': {message}"

private def loadFrozenOrZeros (path : String) (name : String) (s : Shape)
    : IO (Tensor { shape := s, dtype := dtype.val, device := device }) := do
  match ← (← safetensors.Source.open path).load? name s device with
  | none =>
    let support ← dtype.val.acquireConstantSupport device
    let _ := support.down
    pure (Tensor.zerosFor (σ := ⟨s, dtype.val, device⟩))
  | some packed =>
    match packed.atDType? dtype.val with
    | .ok t => pure (Tensor.detach t)
    | .error message => throw <| IO.userError s!"Speech tokenizer tensor '{name}': {message}"

private def addBias2d {n d : Nat}
    (x : Tensor { shape := #[n, d], dtype := dtype.val, device := device })
    (bias : Tensor { shape := #[d], dtype := dtype.val, device := device }) : Tensor { shape := #[n, d], dtype := dtype.val, device := device } :=
  let b : Tensor { shape := #[1, d], dtype := dtype.val, device := device } := Tensor.reshape bias #[1, d]
  Tensor.add x (Tensor.broadcastTo b #[n, d])

private def addBias3d {batch channels frames : Nat}
    (x : Tensor { shape := #[batch, channels, frames], dtype := dtype.val, device := device })
    (bias : Tensor { shape := #[channels], dtype := dtype.val, device := device }) : Tensor { shape := #[batch, channels, frames], dtype := dtype.val, device := device } :=
  let b : Tensor { shape := #[1, channels, 1], dtype := dtype.val, device := device } := Tensor.reshape bias #[1, channels, 1]
  Tensor.add x (Tensor.broadcastTo b #[batch, channels, frames])

private def conv1dWithBias {batch inC outC inFrames kernel : Nat}
    (x : Tensor { shape := #[batch, inC, inFrames], dtype := dtype.val, device := device })
    (w : Tensor { shape := #[outC, inC, kernel], dtype := dtype.val, device := device })
    (b : Tensor { shape := #[outC], dtype := dtype.val, device := device })
    (stride pad dilation : Nat)
    (valid : Tensor.ConvValid inFrames kernel stride pad dilation := by conv_valid) :
    Tensor { shape := #[batch, outC, (inFrames + 2 * pad - (dilation * (kernel - 1) + 1)) / stride + 1], dtype := dtype.val, device := device } :=
  addBias3d (Tensor.conv1d x w stride pad dilation valid) b

private def affine3d {batch seq inDim outDim : Nat}
    (x : Tensor { shape := #[batch, seq, inDim], dtype := dtype.val, device := device })
    (w : Tensor { shape := #[outDim, inDim], dtype := dtype.val, device := device })
    (b : Tensor { shape := #[outDim], dtype := dtype.val, device := device }) : Tensor { shape := #[batch, seq, outDim], dtype := dtype.val, device := device } :=
  let x2 : Tensor { shape := #[batch * seq, inDim], dtype := dtype.val, device := device } := by simpa using (Tensor.reshape x #[batch * seq, inDim])
  let y0 : Tensor { shape := #[batch * seq, outDim], dtype := dtype.val, device := device } := Tensor.mm x2 (Tensor.transpose2d w)
  let y1 : Tensor { shape := #[batch * seq, outDim], dtype := dtype.val, device := device } := addBias2d y0 b
  Tensor.reshape y1 #[batch, seq, outDim]

private def scale3d {batch seq dim : Nat}
    (x : Tensor { shape := #[batch, seq, dim], dtype := dtype.val, device := device })
    (scale : Tensor { shape := #[dim], dtype := dtype.val, device := device }) : Tensor { shape := #[batch, seq, dim], dtype := dtype.val, device := device } :=
  let s : Tensor { shape := #[1, 1, dim], dtype := dtype.val, device := device } := Tensor.reshape scale #[1, 1, dim]
  Tensor.mul x (Tensor.broadcastTo s #[batch, seq, dim])

private def conv2Frames (melFrames : Nat) : Nat :=
  (melFrames + 1) / 2

private def vqFrames (melFrames : Nat) : Nat :=
  (conv2Frames melFrames) / 2

structure EncoderBlock25Hz (dtype : RealFloatingDType := float32DType) where
  attnLnWeight : Tensor { shape := #[1280], dtype := dtype.val, device := device }
  attnLnBias : Tensor { shape := #[1280], dtype := dtype.val, device := device }
  qWeight : Tensor { shape := #[1280, 1280], dtype := dtype.val, device := device }
  qBias : Tensor { shape := #[1280], dtype := dtype.val, device := device }
  kWeight : Tensor { shape := #[1280, 1280], dtype := dtype.val, device := device }
  kBias : Tensor { shape := #[1280], dtype := dtype.val, device := device }
  vWeight : Tensor { shape := #[1280, 1280], dtype := dtype.val, device := device }
  vBias : Tensor { shape := #[1280], dtype := dtype.val, device := device }
  oWeight : Tensor { shape := #[1280, 1280], dtype := dtype.val, device := device }
  oBias : Tensor { shape := #[1280], dtype := dtype.val, device := device }
  mlpLnWeight : Tensor { shape := #[1280], dtype := dtype.val, device := device }
  mlpLnBias : Tensor { shape := #[1280], dtype := dtype.val, device := device }
  fc1Weight : Tensor { shape := #[5120, 1280], dtype := dtype.val, device := device }
  fc1Bias : Tensor { shape := #[5120], dtype := dtype.val, device := device }
  fc2Weight : Tensor { shape := #[1280, 5120], dtype := dtype.val, device := device }
  fc2Bias : Tensor { shape := #[1280], dtype := dtype.val, device := device }
  deriving TensorStruct

structure SpeechTokenizer25HzEncoder (dtype : RealFloatingDType := float32DType) where
  conv1Weight : Tensor { shape := #[1280, 128, 3], dtype := dtype.val, device := device }
  conv1Bias : Tensor { shape := #[1280], dtype := dtype.val, device := device }
  conv2Weight : Tensor { shape := #[1280, 1280, 3], dtype := dtype.val, device := device }
  conv2Bias : Tensor { shape := #[1280], dtype := dtype.val, device := device }
  positionalEmbedding : Tensor { shape := #[1500, 1280], dtype := dtype.val, device := device }
  blocks : Array (EncoderBlock25Hz (device := device) (dtype := dtype))
  vqDownWeight : Tensor { shape := #[1280, 1280, 2], dtype := dtype.val, device := device }
  vqDownBias : Tensor { shape := #[1280], dtype := dtype.val, device := device }
  vqCodebook : Tensor { shape := #[32768, 1280], dtype := dtype.val, device := device }
  inputSampleRate : Static {rate : Nat // 0 < rate} := ⟨⟨24000, by decide⟩⟩
  encodeDownsampleRate : Nat := 1920
  deriving TensorStruct

namespace SpeechTokenizer25HzEncoder

private def loadBlock (weightsPath : String) (i : Nat) : IO (EncoderBlock25Hz (device := device) (dtype := dtype)) := do
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
def loadFromDir (speechTokenizerDir : String) (device : Device := Device.CPU) : IO (Σ dtype : RealFloatingDType, SpeechTokenizer25HzEncoder (device := device) dtype) := do
  let cfg ← SpeechTokenizer25HzEncoderConfig.loadFromFile s!"{speechTokenizerDir}/config.json"

  let weightsPath := s!"{speechTokenizerDir}/model.safetensors"
  let anchor ← (← safetensors.Source.open weightsPath).load "encoder.tokenizer.conv1.weight" #[1280, 128, 3] device
  let floating ← IO.ofExcept (DType.realFloating? anchor.1)
  let _ := floating.down
  let dtype := RealFloatingDType.ofDType anchor.1
  let conv1Weight ← loadFrozen weightsPath "encoder.tokenizer.conv1.weight" #[1280, 128, 3]
  let conv1Bias ← loadFrozen weightsPath "encoder.tokenizer.conv1.bias" #[1280]
  let conv2Weight ← loadFrozen weightsPath "encoder.tokenizer.conv2.weight" #[1280, 1280, 3]
  let conv2Bias ← loadFrozen weightsPath "encoder.tokenizer.conv2.bias" #[1280]
  let positionalEmbedding ← loadFrozen weightsPath "encoder.tokenizer.positional_embedding" #[1500, 1280]

  let mut blocks : Array (EncoderBlock25Hz (device := device) (dtype := dtype)) := #[]
  for i in [:6] do
    let b ← loadBlock weightsPath i
    blocks := blocks.push b

  let vqDownWeight ← loadFrozen weightsPath "encoder.tokenizer.audio_vq_downsample.weight" #[1280, 1280, 2]
  let vqDownBias ← loadFrozen weightsPath "encoder.tokenizer.audio_vq_downsample.bias" #[1280]
  let vqEmbedRaw ← loadFrozen (dtype := dtype) (device := device) weightsPath "encoder.tokenizer.audio_quantizer.rvqs.0.embed" #[1, 32768, 1280]
  let vqEmbed3 : Tensor { shape := #[1, 32768, 1280], dtype := dtype.val, device := device } := Tensor.slice vqEmbedRaw 0 0 1
  let vqCodebook : Tensor { shape := #[32768, 1280], dtype := dtype.val, device := device } := Tensor.reshape vqEmbed3 #[32768, 1280] (by simp)

  let enc : (SpeechTokenizer25HzEncoder (device := device) (dtype := dtype)) := {
    conv1Weight
    conv1Bias
    conv2Weight
    conv2Bias
    positionalEmbedding
    blocks
    vqDownWeight
    vqDownBias
    vqCodebook
    inputSampleRate := ⟨⟨cfg.inputSampleRate, cfg.ratesPos.1⟩⟩
    encodeDownsampleRate := cfg.encodeDownsampleRate
  }
  pure ⟨dtype, enc⟩

private def forwardAttention {seq : Nat}
    (blk : (EncoderBlock25Hz (device := device) (dtype := dtype)))
    (x : Tensor { shape := #[1, seq, 1280], dtype := dtype.val, device := device }) : Tensor { shape := #[1, seq, 1280], dtype := dtype.val, device := device } :=
  let q0 : Tensor { shape := #[1, seq, 1280], dtype := dtype.val, device := device } := affine3d x blk.qWeight blk.qBias
  let k0 : Tensor { shape := #[1, seq, 1280], dtype := dtype.val, device := device } := affine3d x blk.kWeight blk.kBias
  let v0 : Tensor { shape := #[1, seq, 1280], dtype := dtype.val, device := device } := affine3d x blk.vWeight blk.vBias

  let q : Tensor { shape := #[1, seq, 20, 64], dtype := dtype.val, device := device } := Tensor.reshape q0 #[1, seq, 20, 64] (by simp [Nat.mul_assoc])
  let k : Tensor { shape := #[1, seq, 20, 64], dtype := dtype.val, device := device } := Tensor.reshape k0 #[1, seq, 20, 64]
  let v : Tensor { shape := #[1, seq, 20, 64], dtype := dtype.val, device := device } := Tensor.reshape v0 #[1, seq, 20, 64]

  let qh : Tensor { shape := #[1, 20, seq, 64], dtype := dtype.val, device := device } := Tensor.splitHeads q
  let kh : Tensor { shape := #[1, 20, seq, 64], dtype := dtype.val, device := device } := Tensor.splitHeads k
  let vh : Tensor { shape := #[1, 20, seq, 64], dtype := dtype.val, device := device } := Tensor.splitHeads v

  let attn : Tensor { shape := #[1, 20, seq, 64], dtype := dtype.val, device := device } := Tensor.attentionGQA qh kh vh false
  let attn2 : Tensor { shape := #[1, seq, 20, 64], dtype := dtype.val, device := device } := Tensor.joinHeads attn
  let attn3 : Tensor { shape := #[1, seq, 1280], dtype := dtype.val, device := device } := Tensor.reshape attn2 #[1, seq, 1280]
  affine3d attn3 blk.oWeight blk.oBias

private def forwardBlock {seq : Nat}
    (blk : (EncoderBlock25Hz (device := device) (dtype := dtype)))
    (x : Tensor { shape := #[1, seq, 1280], dtype := dtype.val, device := device }) : Tensor { shape := #[1, seq, 1280], dtype := dtype.val, device := device } :=
  let h0 : Tensor { shape := #[1, seq, 1280], dtype := dtype.val, device := device } := Tensor.layerNorm x blk.attnLnWeight blk.attnLnBias 1e-5
  let a : Tensor { shape := #[1, seq, 1280], dtype := dtype.val, device := device } := forwardAttention blk h0
  let x1 : Tensor { shape := #[1, seq, 1280], dtype := dtype.val, device := device } := Tensor.add x a

  let h1 : Tensor { shape := #[1, seq, 1280], dtype := dtype.val, device := device } := Tensor.layerNorm x1 blk.mlpLnWeight blk.mlpLnBias 1e-5
  let m0 : Tensor { shape := #[1, seq, 5120], dtype := dtype.val, device := device } := affine3d h1 blk.fc1Weight blk.fc1Bias
  let m1 : Tensor { shape := #[1, seq, 5120], dtype := dtype.val, device := device } := Tensor.gelu m0
  let m2 : Tensor { shape := #[1, seq, 1280], dtype := dtype.val, device := device } := affine3d m1 blk.fc2Weight blk.fc2Bias
  Tensor.add x1 m2

private def encodeMelChunk {melFrames : Nat}
    (m : (SpeechTokenizer25HzEncoder (device := device) (dtype := dtype)))
    (mel : Tensor { shape := #[128, melFrames], dtype := dtype.val, device := device })
    (hMel : conv2Frames melFrames ≤ 1500) : Tensor { shape := #[vqFrames melFrames], dtype := .Int64, device := device } :=
  let c2Frames : Nat := conv2Frames melFrames
  let qFrames : Nat := vqFrames melFrames
  if hShort : melFrames < 2 ∨ qFrames = 0 then
    have hq : qFrames = 0 := by
      rcases hShort with h | h
      · simp only [qFrames, vqFrames, conv2Frames]; omega
      · exact h
    (Tensor.reshape (Tensor.fullInt #[0] 0) #[qFrames] (by simp [hq])).to device
  else
    have hm : 2 ≤ melFrames := by omega
    have hq : 1 ≤ qFrames := by omega
    Id.run do
      let x0 : Tensor { shape := #[1, 128, melFrames], dtype := dtype.val, device := device } := Tensor.reshape mel #[1, 128, melFrames]
      let h1 : Tensor { shape := #[1, 1280, melFrames], dtype := dtype.val, device := device } :=
        Tensor.gelu (Tensor.reshape (conv1dWithBias x0 m.conv1Weight m.conv1Bias 1 1 1) #[1, 1280, melFrames]
          (by have : (melFrames + 2 * 1 - (1 * (3 - 1) + 1)) / 1 + 1 = melFrames := by omega
              rw [this]))
      let h2 : Tensor { shape := #[1, 1280, c2Frames], dtype := dtype.val, device := device } :=
        Tensor.gelu (Tensor.reshape (conv1dWithBias h1 m.conv2Weight m.conv2Bias 2 1 1) #[1, 1280, c2Frames]
          (by have : (melFrames + 2 * 1 - (1 * (3 - 1) + 1)) / 2 + 1 = c2Frames := by
                simp only [c2Frames, conv2Frames]; omega
              rw [this]))

      let xSeq0 : Tensor { shape := #[1, c2Frames, 1280], dtype := dtype.val, device := device } := Tensor.reshape (Tensor.transpose h2 1 2) #[1, c2Frames, 1280]
      let pos : Tensor { shape := #[c2Frames, 1280], dtype := dtype.val, device := device } := Tensor.slice m.positionalEmbedding 0 0 c2Frames
      let pos3 : Tensor { shape := #[1, c2Frames, 1280], dtype := dtype.val, device := device } := Tensor.reshape pos #[1, c2Frames, 1280]
      let mut xSeq : Tensor { shape := #[1, c2Frames, 1280], dtype := dtype.val, device := device } := Tensor.add xSeq0 pos3

      for blk in m.blocks do
        xSeq := forwardBlock blk xSeq

      let vqIn : Tensor { shape := #[1, 1280, c2Frames], dtype := dtype.val, device := device } := Tensor.reshape (Tensor.transpose xSeq 1 2) #[1, 1280, c2Frames]
      have hc2 : 2 ≤ c2Frames := by simp only [qFrames, c2Frames, vqFrames, conv2Frames] at hq ⊢; omega
      let vqDown : Tensor { shape := #[1, 1280, qFrames], dtype := dtype.val, device := device } :=
        Tensor.reshape (conv1dWithBias vqIn m.vqDownWeight m.vqDownBias 2 0 1) #[1, 1280, qFrames]
          (by have : (c2Frames + 2 * 0 - (1 * (2 - 1) + 1)) / 2 + 1 = qFrames := by
                simp only [qFrames, c2Frames, vqFrames, conv2Frames] at hq ⊢; omega
              rw [this])
      let vqSeq : Tensor { shape := #[qFrames, 1280], dtype := dtype.val, device := device } :=
        Tensor.reshape (Tensor.transpose (Tensor.reshape vqDown #[1, 1280, qFrames]) 1 2) #[qFrames, 1280]

      let x2 : Tensor { shape := #[qFrames, 1], dtype := dtype.val, device := device } := by simpa using (Tensor.reshape (Tensor.sumDim (Tensor.mul vqSeq vqSeq) 1 true) #[qFrames, 1])
      let e2 : Tensor { shape := #[32768, 1], dtype := dtype.val, device := device } := by simpa using (Tensor.reshape (Tensor.sumDim (Tensor.mul m.vqCodebook m.vqCodebook) 1 true) #[32768, 1] (by simp))
      let dot : Tensor { shape := #[qFrames, 32768], dtype := dtype.val, device := device } := Tensor.mm vqSeq (Tensor.transpose2d m.vqCodebook)
      let x2e : Tensor { shape := #[qFrames, 32768], dtype := dtype.val, device := device } := Tensor.broadcastTo x2 #[qFrames, 32768]
      let e2Row : Tensor { shape := #[1, 32768], dtype := dtype.val, device := device } := Tensor.reshape (Tensor.transpose2d e2) #[1, 32768] (by simp)
      let e2e : Tensor { shape := #[qFrames, 32768], dtype := dtype.val, device := device } := Tensor.broadcastTo e2Row #[qFrames, 32768]
      let dist : Tensor { shape := #[qFrames, 32768], dtype := dtype.val, device := device } := Tensor.sub (Tensor.add x2e e2e) (Tensor.mulScalar dot 2.0)
      Tensor.reshape (Tensor.argmax (Tensor.mulScalar dist (-1.0)) 1) #[vqFrames melFrames] (by simp +zetaDelta)

private def appendAll (dst src : Array UInt64) : Array UInt64 :=
  Id.run do
    let mut out := dst
    for x in src do
      out := out.push x
    out

/-- Encode mono waveform to flat codec IDs for the 25Hz tokenizer architecture.
    Returns `(numCodeGroups, flatCodes)` where `numCodeGroups = 1`. -/
def encodeWaveToCodes
    (m : (SpeechTokenizer25HzEncoder (device := device) (dtype := dtype)))
    (wave : Array Float)
    : IO (Nat × Array UInt64) := do
  if wave.isEmpty then
    throw <| IO.userError "Audio encode input is empty"

  let preCfg : PreprocessorConfig := {
    featureSize := 128
    samplingRate := m.inputSampleRate.val.val
    samplingRatePos := m.inputSampleRate.val.property
    hopLength := 160
    nFft := 400
    paddingValue := 0.0
    returnAttentionMask := false
    doNormalize := false
    dither := 0.0
  }

  let ⟨melFrames, frontendOut⟩ ← waveformToWhisperFeaturesDynamic preCfg wave 0.0 none
  let feats : Tensor { shape := #[1, 128, melFrames], dtype := dtype.val, device := device } := Tensor.castLike m.conv1Weight (frontendOut.inputFeatures.to device)

  let chunkMel : Nat := 200
  let mut start : Nat := 0
  let mut outVals : Array UInt64 := #[]
  while start < melFrames do
    if hStart : start < melFrames then
      let remaining := melFrames - start
      let cur := if remaining < chunkMel then remaining else chunkMel
      if cur >= 2 then
        let melChunk3 : Tensor { shape := #[1, 128, cur], dtype := dtype.val, device := device } := Tensor.slice feats 2 start cur
          (_bounds := by simp only [cur, remaining]; split <;> simp <;> omega)
        let melChunk : Tensor { shape := #[128, cur], dtype := dtype.val, device := device } := Tensor.reshape melChunk3 #[128, cur] (by simp)
        let idx : Tensor { shape := #[vqFrames cur], dtype := .Int64, device := device } := encodeMelChunk m melChunk
          (by simp only [conv2Frames, cur, remaining, chunkMel]; split <;> omega)
        let vals ← Tensor.toUInt64Array idx
        outVals := appendAll outVals vals
      start := start + cur
    else
      break

  pure (1, outVals)

end SpeechTokenizer25HzEncoder

end torch.qwen3tts
