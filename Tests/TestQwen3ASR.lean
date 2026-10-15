import Tyr
import Tyr.Model.Qwen3ASR
import LeanTest

open torch
open torch.qwen3asr

private def tinyCfg : Qwen3ASRConfig :=
  { thinkerConfig := {
      audioConfig := {
        numMelBins := 8
        encoderLayers := 2
        encoderAttentionHeads := 2
        encoderFfnDim := 32
        dModel := 16
        outputDim := 16
        downsampleHiddenSize := 4
      }
      textConfig := {
        vocabSize := 96
        hiddenSize := 16
        intermediateSize := 32
        numHiddenLayers := 2
        numAttentionHeads := 4
        numKeyValueHeads := 2
        headDim := 4
        maxPositionEmbeddings := 1024
        ropeTheta := 10000.0
      }
      audioTokenId := 42
    }
    supportLanguages := #["English", "Chinese"]
  }

private def tinyForcedAlignerCfg : Qwen3ASRConfig :=
  { thinkerConfig := {
      audioConfig := {
        numMelBins := 8
        encoderLayers := 2
        encoderAttentionHeads := 2
        encoderFfnDim := 32
        dModel := 16
        outputDim := 16
        downsampleHiddenSize := 4
      }
      textConfig := {
        vocabSize := 96
        hiddenSize := 16
        intermediateSize := 32
        numHiddenLayers := 2
        numAttentionHeads := 4
        numKeyValueHeads := 2
        headDim := 4
        maxPositionEmbeddings := 1024
        ropeTheta := 10000.0
      }
      audioTokenId := 42
      modelType := "qwen3_forced_aligner_thinker"
      classifyNum := 7
    }
    supportLanguages := #["English"]
  }

private def tinyChunkedCfg : Qwen3ASRConfig :=
  { thinkerConfig := {
      audioConfig := {
        numMelBins := 8
        encoderLayers := 2
        encoderAttentionHeads := 2
        encoderFfnDim := 32
        dModel := 16
        outputDim := 16
        downsampleHiddenSize := 4
        nWindow := 4
        nWindowInfer := 8
        convChunkSize := 2
      }
      textConfig := {
        vocabSize := 96
        hiddenSize := 16
        intermediateSize := 32
        numHiddenLayers := 2
        numAttentionHeads := 4
        numKeyValueHeads := 2
        headDim := 4
        maxPositionEmbeddings := 1024
        ropeTheta := 10000.0
      }
      audioTokenId := 42
    }
    supportLanguages := #["English"]
  }

private def tinyPreprocessorCfg (melBins frames : UInt64) : PreprocessorConfig := {
  featureExtractorType := "WhisperFeatureExtractor"
  featureSize := melBins
  samplingRate := 16000
  hopLength := 160
  chunkLength := 1
  nFft := 400
  nSamples := frames * 160
  nbMaxFrames := frames
  paddingSide := "right"
  paddingValue := 0.0
  returnAttentionMask := true
  doNormalize := false
  dither := 0.0
}

private def withZeroLmHead {device : Device}
    (model : Qwen3ASRForConditionalGeneration (device := device) cfg) :
    Qwen3ASRForConditionalGeneration (device := device) cfg :=
  let zeroLmHead : Tensor { shape := #[ThinkerLmVocabSize cfg.thinkerConfig, cfg.thinkerConfig.textConfig.hiddenSize], device := device } :=
    (Tensor.zeros (device := device) #[ThinkerLmVocabSize cfg.thinkerConfig, cfg.thinkerConfig.textConfig.hiddenSize]).forgetDType
  { model with thinker := { model.thinker with lmHead := zeroLmHead } }

private def mkDeterministicWavSamples (n : Nat) : Array Float := Id.run do
  let mut out : Array Float := Array.mkEmpty n
  for i in [:n] do
    let raw := ((i * 37 + 13) % 201).toFloat
    let centered := (raw / 100.0) - 1.0
    out := out.push (0.2 * centered)
  out

private def writeDeterministicWavFixture
    (name : String)
    (sampleCount : UInt64 := 10240)
    (sampleRate : UInt64 := 16000)
    : IO String := do
  let path := s!"/tmp/tyr_qwen3asr_{name}.wav"
  let wave : Tensor { shape := #[sampleCount], dtype := .Float32, device := .CPU } :=
    Tensor.reshapeChecked (Tensor.fromFloatArray (mkDeterministicWavSamples sampleCount.toNat)) #[sampleCount]
  Tensor.saveWav wave path sampleRate
  pure path

private def mkAsciiStreamingTokenizer : tokenizer.qwen3.QwenTokenizer := Id.run do
  let mut idToToken : Array String := #[]
  let mut tokenToId : Std.HashMap String tokenizer.TokenId := {}
  let mut charToId : Std.HashMap Char tokenizer.TokenId := {}
  let mut next : Nat := 0
  for code in [32:127] do
    let c := Char.ofNat code
    let s := String.ofList [c]
    let id : tokenizer.TokenId := next.toUInt32
    idToToken := idToToken.push s
    tokenToId := tokenToId.insert s id
    charToId := charToId.insert c id
    next := next + 1
  let asrTag := "<asr_text>"
  let asrId : tokenizer.TokenId := next.toUInt32
  idToToken := idToToken.push asrTag
  tokenToId := tokenToId.insert asrTag asrId
  let specialTokens : Std.HashMap String tokenizer.TokenId :=
    ({} : Std.HashMap String tokenizer.TokenId).insert asrTag asrId
  let idToSpecial : Std.HashMap tokenizer.TokenId String :=
    ({} : Std.HashMap tokenizer.TokenId String).insert asrId asrTag
  {
    vocabSize := idToToken.size.toUInt32
    idToToken
    tokenToId
    charToId
    merges := #[]
    mergeLookup := {}
    mergePriority := {}
    specialTokens
    idToSpecial
    specialList := #[asrTag]
    unkToken := none
    padToken := 0
  }

private def runStreamingAudioSizesForMode
    (mode : StreamingDecodeMode)
    (audio : Array Float)
    : IO (Array Nat) := do
  let tok := mkAsciiStreamingTokenizer
  let chunkSec : Float := 2.0 / 16000.0
  let st0 ← initStreamingState
    tinyCfg.supportLanguages
    (context := "")
    (language := some "English")
    (unfixedChunkNum := 0)
    (unfixedTokenNum := 1)
    (chunkSizeSec := chunkSec)
    (stepSizeSec := chunkSec)
    (decodeMode := mode)
  let sizesRef ← IO.mkRef (#[] : Array Nat)
  let decodeFn : StreamingDecodeFn := fun _prompt audioAccum => do
    sizesRef.modify (fun xs => xs.push audioAccum.size)
    pure "x"
  let st1 ← streamingTranscribe tok decodeFn audio st0
  let _ ← finishStreamingTranscribe tok decodeFn st1
  sizesRef.get

@[test]
def testQwen3TokenizerDecodeTextSpecialToken : IO Unit := do
  let tok := mkAsciiStreamingTokenizer
  let aId := tok.tokenToId.getD "a" 0
  let bId := tok.tokenToId.getD "b" 0
  let cId := tok.tokenToId.getD "c" 0
  let asrId := tok.specialTokens.getD "<asr_text>" 0
  let decoded := tokenizer.qwen3.decodeText tok #[aId, bId, asrId, cId]
  LeanTest.assertEqual decoded "ab<asr_text>c" "decodeText should preserve special tokens in-place"

@[test]
def testQwen3ASRParseAsrOutput : IO Unit := do
  let (lang1, txt1) := parseAsrOutput "language Chinese<asr_text>hello"
  LeanTest.assertEqual lang1 "Chinese" "parser should extract language metadata"
  LeanTest.assertEqual txt1 "hello" "parser should extract text body after <asr_text>"

  let (lang2, txt2) := parseAsrOutput "plain transcription"
  LeanTest.assertEqual lang2 "" "plain text without metadata should have empty language"
  LeanTest.assertEqual txt2 "plain transcription" "plain text should pass through as transcription"

  let (lang3, txt3) := parseAsrOutput "forced output body" (userLanguage := some "English")
  LeanTest.assertEqual lang3 "English" "forced language should override parsed metadata"
  LeanTest.assertEqual txt3 "forced output body" "forced-language parse should treat raw as text-only"

  let (lang4, txt4) := parseAsrOutput "" (userLanguage := some "English")
  LeanTest.assertEqual lang4 "English" "forced language should survive an empty decoded output"
  LeanTest.assertEqual txt4 "" "empty decoded output should remain empty under a forced language"

@[test]
def testQwen3ASRAudioInputAutoDetect : IO Unit := do
  let autoUrl := ASRAudioInput.ofStringAuto "https://example.com/demo.wav"
  LeanTest.assertTrue
    (match autoUrl with | .url _ => true | _ => false)
    "audio source auto-detection should classify http/https strings as URL inputs"

  let autoDataUrl := ASRAudioInput.ofStringAuto "data:audio/wav;base64,AAAA"
  LeanTest.assertTrue
    (match autoDataUrl with | .base64 _ => true | _ => false)
    "audio source auto-detection should classify data:audio URLs as base64 inputs"

  let rawBase64 := String.ofList (List.replicate 128 'A')
  let autoRawBase64 := ASRAudioInput.ofStringAuto rawBase64
  LeanTest.assertTrue
    (match autoRawBase64 with | .base64 _ => true | _ => false)
    "audio source auto-detection should classify long raw base64 payloads as base64 inputs"

  let autoPath := ASRAudioInput.ofStringAuto "./fixtures/input.wav"
  LeanTest.assertTrue
    (match autoPath with | .wavPath _ => true | _ => false)
    "audio source auto-detection should classify local filesystem strings as wavPath inputs"

@[test]
def testQwen3ASRNormalizeAudioInputWaveform : IO Unit := do
  let direct ← normalizeAudioInputTo16k (.waveform #[2.0, -2.0, 0.5] 16000)
  LeanTest.assertEqual direct.size 3
    "waveform normalization at 16k should preserve sample count"
  LeanTest.assertTrue (direct.all (fun x => x <= 1.0 && x >= -1.0))
    "waveform normalization should clamp/scale all samples into [-1, 1]"

  let resampled ← normalizeAudioInputTo16k (.waveform #[0.1, -0.1, 0.2, -0.2] 8000)
  LeanTest.assertTrue (resampled.size > 0)
    "waveform normalization with non-16k source should return a non-empty resampled waveform"
  LeanTest.assertTrue (resampled.all (fun x => x <= 1.0 && x >= -1.0))
    "resampled waveform should remain in [-1, 1]"

@[test]
def testQwen3ASRProcessorPromptTemplateSurface : IO Unit := do
  let p : Qwen3ASRProcessor := {
    audioToken := "<AP>"
    audioBosToken := "<AB>"
    audioEosToken := "<AE>"
    imStartToken := "<S>"
    imEndToken := "</S>"
  }
  let expected :=
    "<S>system\nctx</S>\n"
    ++ "<S>user\n<AB><AP><AE></S>\n"
    ++ "<S>assistant\nlanguage English<asr_text>"
  let prompt := p.buildAsrPrompt "ctx" (some "English")
  LeanTest.assertEqual prompt expected
    "processor prompt construction should honor processor role/audio token markers"
  LeanTest.assertEqual (buildTextPrompt "ctx" (some "English") p) expected
    "streaming prompt builder should delegate to processor prompt construction"

@[test]
def testQwen3ASRStreamingDecodeModes : IO Unit := do
  let audio : Array Float := #[0.1, 0.2, 0.3, 0.4, 0.5]
  let rolling ← runStreamingAudioSizesForMode .rollingWindow audio
  let full ← runStreamingAudioSizesForMode .fullAccumulation audio
  LeanTest.assertEqual rolling #[2, 2, 2]
    "rolling-window decode mode should keep a fixed-size audio accumulation window"
  LeanTest.assertEqual full #[2, 4, 5]
    "full-accumulation decode mode should grow accumulation across chunks and tail"

@[test]
def testQwen3ASRForcedAlignerJpKrSegmentation : IO Unit := do
  let (jaWords, _) := Qwen3ForceAlignProcessor.encodeTimestampText "abcあい" "Japanese"
  LeanTest.assertEqual jaWords #["abc", "あ", "い"]
    "Japanese forced-alignment tokenization should split script chars while preserving latin runs"

  let (koWords, _) := Qwen3ForceAlignProcessor.encodeTimestampText "ab한글cd" "Korean"
  LeanTest.assertEqual koWords #["ab", "한", "글", "cd"]
    "Korean forced-alignment tokenization should split Hangul chars while preserving latin runs"

@[test]
def testQwen3ASRTimestampsRequireForcedAlignerOrHandle : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let tok := mkAsciiStreamingTokenizer
  let pre := tinyPreprocessorCfg cfg.thinkerConfig.audioConfig.numMelBins 32
  let audio := mkDeterministicWavSamples 12000

  let threwWithoutHandle ←
    try
      let _ ← model.transcribeWaveform
        tok
        pre
        audio
        (context := "")
        (language := some "English")
        (returnTimeStamps := true)
        (maxNewTokens := 2)
        (eosTokenIds := #[])
      pure false
    catch _ =>
      pure true
  LeanTest.assertTrue threwWithoutHandle
    "timestamps on non-forced-aligner ASR checkpoint should fail without explicit forced-aligner handle"

  let alignCfg := tinyForcedAlignerCfg
  let alignModel ← Qwen3ASRForConditionalGeneration.init alignCfg
  let aligner : Qwen3ForcedAligner := {
    device := .CPU
    cfg := alignCfg
    model := alignModel
    tok := tok
    preprocessor := tinyPreprocessorCfg alignCfg.thinkerConfig.audioConfig.numMelBins 32
  }
  let threwWithHandle ←
    try
      let _ ← model.transcribeWaveform
        tok
        pre
        audio
        (forcedAligner := some aligner)
        (context := "")
        (language := some "English")
        (returnTimeStamps := true)
        (maxNewTokens := 2)
        (eosTokenIds := #[])
      pure false
    catch _ =>
      pure true
  LeanTest.assertTrue (!threwWithHandle)
    "timestamps should succeed on non-forced-aligner ASR checkpoint when a forced-aligner handle is supplied"

@[test]
def testQwen3ASRTranscribeAudioInputAndBatchStepParity : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let tok := mkAsciiStreamingTokenizer
  let pre := tinyPreprocessorCfg cfg.thinkerConfig.audioConfig.numMelBins 32
  let audioA := mkDeterministicWavSamples 9000
  let audioB := mkDeterministicWavSamples 11000

  let outWaveform ← model.transcribeWaveform
    tok
    pre
    audioA
    (context := "")
    (language := some "English")
    (returnTimeStamps := false)
    (maxNewTokens := 2)
    (eosTokenIds := #[])
  let outAudioInput ← model.transcribeAudioInput
    tok
    pre
    (.waveform audioA 16000)
    (context := "")
    (language := some "English")
    (returnTimeStamps := false)
    (maxNewTokens := 2)
    (eosTokenIds := #[])
  LeanTest.assertEqual outAudioInput.language outWaveform.language
    "unified waveform+sample-rate audio input should match direct waveform transcription language"
  LeanTest.assertEqual outAudioInput.text outWaveform.text
    "unified waveform+sample-rate audio input should match direct waveform transcription text"

  let audios : Array (Array Float) := #[audioA, audioB]
  let outsUnbounded ← model.transcribeWaveforms
    tok
    pre
    audios
    (contexts := #["ctx"])
    (languages := #[some "English"])
    (maxInferenceBatchSize := -1)
    (returnTimeStamps := false)
    (maxNewTokens := 2)
    (eosTokenIds := #[])
  let outsChunked ← model.transcribeWaveforms
    tok
    pre
    audios
    (contexts := #["ctx"])
    (languages := #[some "English"])
    (maxInferenceBatchSize := 1)
    (returnTimeStamps := false)
    (maxNewTokens := 2)
    (eosTokenIds := #[])
  LeanTest.assertEqual (outsUnbounded.map (fun r => r.text)) (outsChunked.map (fun r => r.text))
    "maxInferenceBatchSize chunking should preserve transcription outputs"

@[test]
def testQwen3ASRMergeLanguages : IO Unit := do
  let merged := mergeLanguages #["Chinese", "English", "English", "", "Chinese"]
  LeanTest.assertEqual merged "Chinese,English,Chinese"
    "mergeLanguages should drop empties and consecutive duplicates while preserving order"

@[test]
def testQwen3ASRHubOfficialRepoIds : IO Unit := do
  LeanTest.assertTrue (hub.isQwen3ASRCollectionRepoId "Qwen/Qwen3-ASR-0.6B")
    "official ASR 0.6B repo id should be recognized"
  LeanTest.assertTrue (hub.isQwen3ASRCollectionRepoId "Qwen/Qwen3-ASR-1.7B")
    "official ASR 1.7B repo id should be recognized"
  LeanTest.assertTrue (!(hub.isQwen3ASRCollectionRepoId "Qwen/Qwen3-ASR-42B"))
    "unknown ASR repo ids should not be recognized as official collection members"

@[test]
def testQwen3ASRHubResolveLocalDir : IO Unit := do
  let localDir := "/tmp/qwen3asr_pretrained_local"
  IO.FS.createDirAll ⟨localDir⟩
  let resolved ← hub.resolvePretrainedDir localDir
  LeanTest.assertEqual resolved localDir
    "resolver should pass through existing local model directories unchanged"

@[test]
def testQwen3ASRInitStreamingStateLanguageValidation : IO Unit := do
  let st ← initStreamingState tinyCfg.supportLanguages (context := "") (language := some "english")
  LeanTest.assertEqual st.forceLanguage (some "English")
    "initStreamingState should normalize forced language to canonical name"

  let threw ←
    try
      let _ ← initStreamingState tinyCfg.supportLanguages (context := "") (language := some "Klingon")
      pure false
    catch _ =>
      pure true
  LeanTest.assertTrue threw "initStreamingState should reject unsupported forced language"

@[test]
def testQwen3ASRStreamingTranscribeAndFinish : IO Unit := do
  let tok := mkAsciiStreamingTokenizer
  let chunkSec : Float := 2.0 / 16000.0  -- 2 samples per chunk
  let st0 ← initStreamingState
    tinyCfg.supportLanguages
    (context := "")
    (language := some "English")
    (unfixedChunkNum := 0)
    (unfixedTokenNum := 1)
    (chunkSizeSec := chunkSec)
    (stepSizeSec := chunkSec)
    (decodeMode := .fullAccumulation)

  let promptsRef ← IO.mkRef (#[] : Array String)
  let decodeFn : StreamingDecodeFn := fun prompt audioAccum => do
    promptsRef.modify (fun xs => xs.push prompt)
    if audioAccum.size <= 2 then
      pure "abc"
    else if audioAccum.size <= 4 then
      pure "cd"
    else
      pure "de"

  let st1 ← streamingTranscribe tok decodeFn #[0.1, 0.2, 0.3, 0.4, 0.5] st0
  let prompts1 ← promptsRef.get

  LeanTest.assertEqual st1.chunkId 2 "streamingTranscribe should consume exactly two full chunks"
  LeanTest.assertEqual st1.buffer.size 1 "streamingTranscribe should retain one-sample tail in buffer"
  LeanTest.assertEqual st1.language "English" "forced language should be reflected in streaming state"
  LeanTest.assertEqual st1.text "abcd" "text should reflect prefix rollback + continuation behavior"
  LeanTest.assertEqual prompts1.size 2 "streamingTranscribe should call decode once per full chunk"

  let expectedPrefix2 :=
    let ids := tokenizer.qwen3.encodeText tok "abc"
    let endIdx := if ids.size > 1 then ids.size - 1 else 0
    if endIdx > 0 then tokenizer.qwen3.decodeText tok (ids.extract 0 endIdx) else ""
  LeanTest.assertEqual (prompts1.getD 0 "") st0.promptRaw "first chunk prompt should be raw base prompt"
  LeanTest.assertEqual (prompts1.getD 1 "") (st0.promptRaw ++ expectedPrefix2)
    "second chunk prompt should append rollback prefix"

  let st2 ← finishStreamingTranscribe tok decodeFn st1
  let prompts2 ← promptsRef.get
  LeanTest.assertEqual st2.chunkId 3 "finishStreamingTranscribe should decode one additional tail chunk"
  LeanTest.assertEqual st2.buffer.size 0 "finishStreamingTranscribe should flush remaining buffer"
  LeanTest.assertEqual st2.text "abcde" "finishStreamingTranscribe should update final text"
  LeanTest.assertEqual st2.language "English" "final language should remain forced language"
  LeanTest.assertEqual prompts2.size 3 "finishStreamingTranscribe should perform one extra decode call"

  let expectedPrefix3 :=
    let ids := tokenizer.qwen3.encodeText tok "abcd"
    let endIdxRaw := Nat.max 1 (ids.size - 1)
    let endIdx := Nat.min endIdxRaw ids.size
    tokenizer.qwen3.decodeText tok (ids.extract 0 endIdx)
  LeanTest.assertEqual (prompts2.getD 2 "") (st0.promptRaw ++ expectedPrefix3)
    "finish prompt should use max(1, len-k) rollback behavior"

@[test]
def testQwen3ASRStreamingPromptCacheLifecycle : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let tok := mkAsciiStreamingTokenizer
  let pre := tinyPreprocessorCfg cfg.thinkerConfig.audioConfig.numMelBins 32
  let chunkSec : Float := 2.0 / 16000.0
  let st0 ← initStreamingState
    cfg.supportLanguages
    (context := "")
    (language := some "English")
    (unfixedChunkNum := 0)
    (unfixedTokenNum := 1)
    (chunkSizeSec := chunkSec)
    (stepSizeSec := chunkSec)
    (decodeMode := .fullAccumulation)

  let (st1, cache1) ← streamingTranscribeWithModelCached
    model
    tok
    pre
    #[0.1, 0.2, 0.3, 0.4, 0.5]
    st0
    (maxNewTokens := 1)
    (eosTokenIds := #[])
  LeanTest.assertEqual st1.chunkId 2
    "cached streaming decode should process the same chunk count as the uncached path"
  LeanTest.assertTrue cache1.isSome
    "cached streaming decode should return a reusable prompt cache after decoding chunks"

  let (st2, cache2) ← finishStreamingTranscribeWithModelCached
    model
    tok
    pre
    st1
    (prefixCache := cache1)
    (maxNewTokens := 1)
    (eosTokenIds := #[])
  LeanTest.assertEqual st2.chunkId 3
    "cached finish decode should process one additional tail chunk"
  LeanTest.assertTrue cache2.isSome
    "cached finish decode should keep returning prompt cache state"

@[test]
def testQwen3ASRTranscribeWavOffline : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let tok := mkAsciiStreamingTokenizer
  let pre := tinyPreprocessorCfg cfg.thinkerConfig.audioConfig.numMelBins 64
  let wavPath ← writeDeterministicWavFixture "transcribe_offline"

  let out ← model.transcribeWav
    tok
    pre
    wavPath
    (context := "")
    (language := some "english")
    (returnTimeStamps := false)
    (maxNewTokens := 2)
    (eosTokenIds := #[])

  LeanTest.assertEqual out.language "English"
    "offline transcribe should normalize and preserve forced language"
  LeanTest.assertTrue out.timeStamps.isNone
    "offline transcribe should not return timestamps when returnTimeStamps=false"

@[test]
def testQwen3ASRTranscribeWaveformsBatchBroadcastAndMismatch : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let tok := mkAsciiStreamingTokenizer
  let pre := tinyPreprocessorCfg cfg.thinkerConfig.audioConfig.numMelBins 32
  let audios : Array (Array Float) := #[#[0.0, 0.1], #[0.2, -0.2, 0.3]]

  let outs ← model.transcribeWaveforms
    tok
    pre
    audios
    (contexts := #["ctx"])
    (languages := #[some "English"])
    (returnTimeStamps := false)
    (maxNewTokens := 1)
    (eosTokenIds := #[])

  LeanTest.assertEqual outs.size 2 "broadcasted offline transcribe should return one result per input audio"
  LeanTest.assertTrue (outs.all (fun r => r.language == "English"))
    "broadcasted forced language should apply to all batch entries"

  let threw ←
    try
      let _ ← model.transcribeWaveforms
        tok
        pre
        audios
        (contexts := #["a", "b", "c"])
        (languages := #[some "English"])
        (returnTimeStamps := false)
        (maxNewTokens := 1)
        (eosTokenIds := #[])
      pure false
    catch _ =>
      pure true
  LeanTest.assertTrue threw
    "offline transcribe should reject context batch size mismatch"

@[test]
def testQwen3ASRInitAndLanguages : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  LeanTest.assertEqual (model.getSupportedLanguages).size 2 "expected two configured languages"

@[test]
def testQwen3ASRAudioForward : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let batch : UInt64 := 2
  let frames : UInt64 := 32
  let mel ← Tensor.randn #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]
  let audio := model.thinker.encodeAudio mel.forgetDType
  let s := Tensor.item (Tensor.sumAll audio)
  LeanTest.assertTrue (Float.isFinite s) "audio encoder output sum should be finite"

@[test]
def testQwen3ASRForwardFromMel : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg

  let batch : UInt64 := 2
  let frames : UInt64 := 32
  let textSeq : UInt64 := 6
  let outSeq : UInt64 := AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames + textSeq

  let mel ← Tensor.randn #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]
  let textIds ← Tensor.randint 0 cfg.thinkerConfig.textConfig.vocabSize.toInt64 #[batch, textSeq]
  let logits := model.forwardFromMel mel.forgetDType textIds

  let s := Tensor.item (Tensor.sumAll logits)
  LeanTest.assertTrue (Float.isFinite s) "logit sum should be finite"

  let idxs := Tensor.argmax logits 2
  let flat : Tensor { shape := #[batch * outSeq], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked idxs #[batch * outSeq]
  let toks ← Tensor.toUInt64Array flat
  LeanTest.assertEqual toks.size (batch * outSeq).toNat "flattened argmax token count should match output length"

@[test]
def testQwen3ASRFaithfulForwardPlaceholderScatter : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg

  let batch : UInt64 := 1
  let frames : UInt64 := 32
  let audioSeq : UInt64 := AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames
  let seq : UInt64 := audioSeq + 4

  let aTok : Int64 := Int64.ofNat cfg.thinkerConfig.audioTokenId.toNat
  let idsVals : Array Int64 := #[aTok, aTok, aTok, aTok, 1, 2, 3, 4]
  let inputIds : Tensor { shape := #[batch, seq], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked (Tensor.fromInt64Array idsVals) #[batch, seq]

  let mel ← Tensor.randn #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]
  let featureMask : Tensor { shape := #[batch, frames], dtype := .Int64, device := .CPU } := Tensor.fullInt #[batch, frames] 1

  let logits ← model.forward inputIds (some mel.forgetDType) (some featureMask) none
  let s := Tensor.item (Tensor.sumAll logits)
  LeanTest.assertTrue (Float.isFinite s) "faithful forward logits should be finite"

@[test]
def testQwen3ASRFaithfulForwardPlaceholderMismatchThrows : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg

  let batch : UInt64 := 1
  let frames : UInt64 := 32
  let audioSeq : UInt64 := AudioEncoderConfig.featExtractOutputLength frames
  let seq : UInt64 := audioSeq + 2
  let aTok : Int64 := Int64.ofNat cfg.thinkerConfig.audioTokenId.toNat
  let mut idsVals : Array Int64 := #[]
  let placeholderCount := if audioSeq > 0 then audioSeq - 1 else 0
  for i in [:seq.toNat] do
    let v : Int64 := if i.toUInt64 < placeholderCount then aTok else (Int64.ofNat i + 5)
    idsVals := idsVals.push v
  let inputIds : Tensor { shape := #[batch, seq], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked (Tensor.fromInt64Array idsVals) #[batch, seq]
  let mel ← Tensor.randn #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]
  let featureMask : Tensor { shape := #[batch, frames], dtype := .Int64, device := .CPU } := Tensor.fullInt #[batch, frames] 1

  let threw ←
    try
      let _ ← model.forward inputIds (some mel.forgetDType) (some featureMask) none
      pure false
    catch _ =>
      pure true
  LeanTest.assertTrue threw "forward should fail when audio placeholder count does not match encoded audio length"

@[test]
def testQwen3ASRForwardFromOutputWav : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg

  let wavPath ← writeDeterministicWavFixture "forward_from_wav"

  let batch : UInt64 := 1
  let frames : UInt64 := 64
  let textSeq : UInt64 := 5
  let outSeq : UInt64 := AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames + textSeq

  let frontendCfg := tinyPreprocessorCfg cfg.thinkerConfig.audioConfig.numMelBins frames
  let frontendOut ← wavToWhisperFeatures frontendCfg wavPath
  let mel := frontendOut.inputFeatures
  let textIds ← Tensor.randint 0 cfg.thinkerConfig.textConfig.vocabSize.toInt64 #[batch, textSeq]
  let logits := model.forwardFromMel mel.forgetDType textIds

  let s := Tensor.item (Tensor.sumAll logits)
  LeanTest.assertTrue (Float.isFinite s) "logit sum from WAV-derived mel should be finite"

  let idxs := Tensor.argmax logits 2
  let flat : Tensor { shape := #[batch * outSeq], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked idxs #[batch * outSeq]
  let toks ← Tensor.toUInt64Array flat
  LeanTest.assertEqual toks.size (batch * outSeq).toNat "flattened argmax token count should match output length"

@[test]
def testQwen3ASRPreprocessorConfigLoad : IO Unit := do
  let path := "/tmp/qwen3asr_preprocessor_config_test.json"
  let json :=
    "{\n"
    ++ "  \"feature_extractor_type\": \"WhisperFeatureExtractor\",\n"
    ++ "  \"feature_size\": 8,\n"
    ++ "  \"sampling_rate\": 16000,\n"
    ++ "  \"hop_length\": 160,\n"
    ++ "  \"chunk_length\": 2,\n"
    ++ "  \"n_fft\": 400,\n"
    ++ "  \"padding_side\": \"right\",\n"
    ++ "  \"padding_value\": 0.0,\n"
    ++ "  \"return_attention_mask\": true,\n"
    ++ "  \"do_normalize\": false,\n"
    ++ "  \"dither\": 0.0\n"
    ++ "}\n"
  IO.FS.writeFile path json
  let cfg ← PreprocessorConfig.loadFromFile path
  LeanTest.assertEqual cfg.featureSize 8 "feature_size should be loaded"
  LeanTest.assertEqual cfg.samplingRate 16000 "sampling_rate should be loaded"
  LeanTest.assertEqual cfg.nFft 400 "n_fft should be loaded"
  LeanTest.assertEqual (PreprocessorConfig.expectedSampleCount cfg) 32000
    "n_samples should be derived from chunk_length * sampling_rate when missing"
  LeanTest.assertEqual (PreprocessorConfig.expectedFrames cfg) 200
    "nb_max_frames should be derived from n_samples / hop_length when missing"

@[test]
def testQwen3ASRProcessorReplaceSpecialTokens : IO Unit := do
  let p : Qwen3ASRProcessor := { audioToken := "<|AUDIO|>" }
  let text := #["x<|AUDIO|>y<|AUDIO|>z", "pre<|AUDIO|>post"]
  let lens := #[2, 3, 1]
  let out ←
    match p.replaceMultimodalSpecialTokens text lens with
    | .ok xs => pure xs
    | .error e => throw <| IO.userError s!"replaceMultimodalSpecialTokens failed: {e}"

  LeanTest.assertEqual (out.getD 0 "") "x<|AUDIO|><|AUDIO|>y<|AUDIO|><|AUDIO|><|AUDIO|>z"
    "first sample token replacement should match reference semantics"
  LeanTest.assertEqual (out.getD 1 "") "pre<|AUDIO|>post"
    "second sample should consume one audio length and keep token spelling"

@[test]
def testQwen3ASRProcessorChunkedIndex : IO Unit := do
  let idx := Qwen3ASRProcessor.getChunkedIndex #[0, 10, 999, 1000, 1300, 2200] 1000
  LeanTest.assertEqual idx #[(0, 3), (3, 5), (5, 6)]
    "chunk boundaries should follow token-range thresholds"

@[test]
def testQwen3ASRFrontendWavWhisperFeatures : IO Unit := do
  let wavPath ← writeDeterministicWavFixture "frontend_whisper_features"

  let cfg := tinyPreprocessorCfg 8 64
  let out ← wavToWhisperFeatures cfg wavPath
  let mel := out.inputFeatures
  let fmask := out.featureAttentionMask

  LeanTest.assertEqual mel.runtimeShape #[1, 8, 64] "frontend mel runtime shape should match config"
  LeanTest.assertEqual fmask.runtimeShape #[1, 64] "feature mask runtime shape should match config"

  let s := Tensor.item (Tensor.sumAll mel)
  LeanTest.assertTrue (Float.isFinite s) "Whisper frontend mel sum should be finite"

  let flatMask : Tensor { shape := #[64], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked fmask #[64]
  let maskVals ← Tensor.toUInt64Array (Tensor.toInt64 flatMask)
  LeanTest.assertEqual maskVals.size 64 "feature mask length should equal configured frames"
  LeanTest.assertTrue (maskVals.all (fun v => v == 1))
    "feature mask should be fully valid for full-length deterministic fixture"

@[test]
def testQwen3ASRForwardForcedAlignerHead : IO Unit := do
  let cfg := tinyForcedAlignerCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let batch : UInt64 := 2
  let seq : UInt64 := 5
  let textIds ← Tensor.randint 0 cfg.thinkerConfig.textConfig.vocabSize.toInt64 #[batch, seq]
  let logits := model.forwardText textIds
  let s := Tensor.item (Tensor.sumAll logits)
  LeanTest.assertTrue (Float.isFinite s) "forced-aligner logits should be finite"

  let idxs := Tensor.argmax logits 2
  let flat : Tensor { shape := #[batch * seq], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked idxs #[batch * seq]
  let ids ← Tensor.toUInt64Array flat
  LeanTest.assertTrue (ids.all (fun i => i < cfg.thinkerConfig.classifyNum))
    "argmax ids should be within forced-aligner classify_num range"

@[test]
def testQwen3ASRForwardWithAuxRopeDeltas : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let batch : UInt64 := 1
  let seq : UInt64 := 6
  let ids ← Tensor.randint 0 cfg.thinkerConfig.textConfig.vocabSize.toInt64 #[batch, seq]
  let attnVals : Array Int64 := #[0, 0, 1, 1, 1, 1]
  let attnMask : Tensor { shape := #[batch, seq], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked (Tensor.fromInt64Array attnVals) #[batch, seq]
  let out ← model.forwardWithAux
    ids
    (inputFeatures := (none : Option (Tensor { shape := #[batch, cfg.thinkerConfig.audioConfig.numMelBins, 1], device := .CPU })))
    (featureAttentionMask := (none : Option (Tensor { shape := #[batch, 1], dtype := .Int64, device := .CPU })))
    (attentionMask := some attnMask)
  match out.ropeDeltas with
  | none =>
    LeanTest.fail "forwardWithAux should populate ropeDeltas when attention mask is provided"
  | some d =>
    let flat : Tensor { shape := #[batch], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked d #[batch]
    let arr ← Tensor.toUInt64Array (Tensor.toInt64 flat)
    LeanTest.assertEqual arr.size batch.toNat "ropeDeltas batch size should match"

@[test]
def testQwen3ASRGreedyGenerateFromWav : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let wavPath ← writeDeterministicWavFixture "greedy_generate"

  let batch : UInt64 := 1
  let frames : UInt64 := 64
  let audioSeq : UInt64 := AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames
  let promptSeq : UInt64 := audioSeq + 2
  let aTok : Int64 := Int64.ofNat cfg.thinkerConfig.audioTokenId.toNat
  let mut idsVals : Array Int64 := #[]
  for _ in [:audioSeq.toNat] do
    idsVals := idsVals.push aTok
  idsVals := idsVals.push 5
  idsVals := idsVals.push 9
  let inputIds : Tensor { shape := #[batch, promptSeq], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked (Tensor.fromInt64Array idsVals) #[batch, promptSeq]

  let frontendCfg := tinyPreprocessorCfg cfg.thinkerConfig.audioConfig.numMelBins frames
  let frontendOut ← wavToWhisperFeatures frontendCfg wavPath
  let mel : Tensor { shape := #[1, cfg.thinkerConfig.audioConfig.numMelBins, frames], dtype := .Float32, device := .CPU } := frontendOut.inputFeatures
  let fmask : Tensor { shape := #[batch, frames], dtype := .Int64, device := .CPU } := frontendOut.featureAttentionMask

  let generated ← model.generateGreedy inputIds (some mel.forgetDType) (some fmask) 3 #[]
  let outSeq := generated.1
  let outIds := generated.2
  LeanTest.assertEqual outSeq (promptSeq + 3) "greedy generation should append max_new_tokens when eos set is empty"

  let flat : Tensor { shape := #[outSeq], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked outIds #[outSeq]
  let toks ← Tensor.toUInt64Array flat
  LeanTest.assertEqual toks.size outSeq.toNat "generated token tensor should match reported output sequence length"

@[test]
def testQwen3ASRGreedyCachedMatchesUncached : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let model := withZeroLmHead model
  let batch : UInt64 := 1
  let frames : UInt64 := 32
  let audioSeq : UInt64 := AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames
  let promptSeq : UInt64 := audioSeq + 2
  let aTok : Int64 := Int64.ofNat cfg.thinkerConfig.audioTokenId.toNat
  let mut idsVals : Array Int64 := #[]
  for _ in [:audioSeq.toNat] do
    idsVals := idsVals.push aTok
  idsVals := idsVals.push 7
  idsVals := idsVals.push 11
  let inputIds : Tensor { shape := #[batch, promptSeq], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked (Tensor.fromInt64Array idsVals) #[batch, promptSeq]
  let mel ← Tensor.randn #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]
  let fmask : Tensor { shape := #[batch, frames], dtype := .Int64, device := .CPU } := Tensor.fullInt #[batch, frames] 1

  let outCached ← model.generateGreedy inputIds (some mel.forgetDType) (some fmask) 4 #[]
  let outUncached ← model.generateGreedyUncached inputIds (some mel.forgetDType) (some fmask) 4 #[]

  let seqCached := outCached.1
  let idsCached := outCached.2
  let seqUncached := outUncached.1
  let idsUncached := outUncached.2

  LeanTest.assertEqual seqCached seqUncached "cached and uncached generation should produce same output length"
  let flatCached : Tensor { shape := #[seqCached], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked idsCached #[seqCached]
  let flatUncached : Tensor { shape := #[seqUncached], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked idsUncached #[seqUncached]
  let toksCached ← Tensor.toUInt64Array flatCached
  let toksUncached ← Tensor.toUInt64Array flatUncached
  LeanTest.assertEqual toksCached toksUncached "cached and uncached generation should produce identical token ids"

@[test]
def testQwen3ASRGreedyCachedMatchesUncachedWhenAudioTokenIsGenerated : IO Unit := do
  let cfg : Qwen3ASRConfig :=
    { tinyCfg with thinkerConfig := { tinyCfg.thinkerConfig with audioTokenId := 0 } }
  let model0 ← Qwen3ASRForConditionalGeneration.init cfg
  let model : Qwen3ASRForConditionalGeneration cfg := withZeroLmHead model0
  let batch : UInt64 := 1
  let frames : UInt64 := 32
  let audioSeq : UInt64 := AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames
  let promptSeq : UInt64 := audioSeq + 2
  let aTok : Int64 := Int64.ofNat cfg.thinkerConfig.audioTokenId.toNat
  let mut idsVals : Array Int64 := #[]
  for _ in [:audioSeq.toNat] do
    idsVals := idsVals.push aTok
  idsVals := idsVals.push 7
  idsVals := idsVals.push 11
  let inputIds : Tensor { shape := #[batch, promptSeq], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked (Tensor.fromInt64Array idsVals) #[batch, promptSeq]
  let mel ← Tensor.randn #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]
  let fmask : Tensor { shape := #[batch, frames], dtype := .Int64, device := .CPU } := Tensor.fullInt #[batch, frames] 1

  let outCached ← model.generateGreedy inputIds (some mel.forgetDType) (some fmask) 4 #[]
  let outUncached ← model.generateGreedyUncached inputIds (some mel.forgetDType) (some fmask) 4 #[]

  let seqCached := outCached.1
  let idsCached := outCached.2
  let seqUncached := outUncached.1
  let idsUncached := outUncached.2

  LeanTest.assertEqual seqCached seqUncached
    "cached and uncached generation should stay aligned even when the generated token equals the audio placeholder id"
  let flatCached : Tensor { shape := #[seqCached], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked idsCached #[seqCached]
  let flatUncached : Tensor { shape := #[seqUncached], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked idsUncached #[seqUncached]
  let toksCached ← Tensor.toUInt64Array flatCached
  let toksUncached ← Tensor.toUInt64Array flatUncached
  LeanTest.assertEqual toksCached toksUncached
    "cached and uncached generation should produce identical token ids even when the audio placeholder id is generated"
  LeanTest.assertTrue (toksCached.drop promptSeq.toNat |>.all (· == 0))
    s!"zero lmHead should force generated continuation tokens to the audio token id, got {reprStr toksCached}"

@[test]
def testQwen3ASRMaskedPositionsMatchTrimmedLeftPad : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let batch : UInt64 := 1
  let seq : UInt64 := 6
  let vocab := cfg.thinkerConfig.textConfig.vocabSize.toInt64
  let ids ← Tensor.randint 0 vocab #[batch, seq]
  let maskVals : Array Int64 := #[0, 0, 1, 1, 1, 1]
  let attnMask : Tensor { shape := #[batch, seq], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked (Tensor.fromInt64Array maskVals) #[batch, seq]

  let logitsMasked ← model.forward
    ids
    (inputFeatures := (none : Option (Tensor { shape := #[batch, cfg.thinkerConfig.audioConfig.numMelBins, 1], device := .CPU })))
    (featureAttentionMask := (none : Option (Tensor { shape := #[batch, 1], dtype := .Int64, device := .CPU })))
    (attentionMask := some attnMask)

  let idsTrim : Tensor { shape := #[batch, 4], dtype := .Int64, device := .CPU } := Tensor.sliceChecked ids 1 2 4 #[batch, 4]
  let logitsTrim ← model.forward
    idsTrim
    (inputFeatures := (none : Option (Tensor { shape := #[batch, cfg.thinkerConfig.audioConfig.numMelBins, 1], device := .CPU })))
    (featureAttentionMask := (none : Option (Tensor { shape := #[batch, 1], dtype := .Int64, device := .CPU })))
    (attentionMask := (none : Option (Tensor { shape := #[batch, 4], dtype := .Int64, device := .CPU })))

  let logitsMaskedValid : Tensor { shape := #[batch, 4, ThinkerLmVocabSize cfg.thinkerConfig], device := .CPU } := Tensor.sliceChecked logitsMasked 1 2 4 #[batch, 4, ThinkerLmVocabSize cfg.thinkerConfig]
  let a ← Tensor.toFloatArray (logitsMaskedValid)
  let b ← Tensor.toFloatArray (logitsTrim)
  let mut maxErr : Float := 0.0
  for i in [:a.size] do
    let err := Float.abs (a.getD i 0.0 - b.getD i 0.0)
    if err > maxErr then
      maxErr := err
  LeanTest.assertTrue (maxErr < 1e-4)
    s!"left-pad masked positions should match trimmed decode for valid tokens (max_err={maxErr})"

@[test]
def testQwen3ASRForwardVariableFeatureMask : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let batch : UInt64 := 1
  let frames : UInt64 := 64
  let validFrames : UInt64 := 32
  let audioLen : UInt64 := AudioEncoderConfig.featExtractOutputLength validFrames
  let seq : UInt64 := audioLen + 2
  let aTok : Int64 := Int64.ofNat cfg.thinkerConfig.audioTokenId.toNat
  let mut idsVals : Array Int64 := #[]
  for _ in [:audioLen.toNat] do
    idsVals := idsVals.push aTok
  idsVals := idsVals.push 3
  idsVals := idsVals.push 4
  let inputIds : Tensor { shape := #[batch, seq], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked (Tensor.fromInt64Array idsVals) #[batch, seq]
  let mel ← Tensor.randn #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]
  let mut mask : Array Int64 := #[]
  for i in [:frames.toNat] do
    mask := mask.push (if i.toUInt64 < validFrames then 1 else 0)
  let featureMask : Tensor { shape := #[batch, frames], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked (Tensor.fromInt64Array mask) #[batch, frames]

  let logits ← model.forward inputIds (some mel.forgetDType) (some featureMask) none
  let s := Tensor.item (Tensor.sumAll logits)
  LeanTest.assertTrue (Float.isFinite s) "variable feature mask forward should be finite"

@[test]
def testQwen3ASRAudioEncoderFeatureDType : IO Unit := do
  let cfg : AudioEncoderConfig :=
    { tinyChunkedCfg.thinkerConfig.audioConfig with nWindow := 8, nWindowInfer := 32 }
  let original ← AudioEncoder.init cfg
  let original := TensorStruct.detach original
  let features : Tensor { shape := #[3, cfg.numMelBins, 32], dtype := .Float32, device := .CPU } := Tensor.reshapeChecked
    (Tensor.sin (Tensor.toFloat32 (Tensor.arange (3 * cfg.numMelBins * 32)))) #[3, cfg.numMelBins, 32]
  for bf16Weights in #[false, true] do
    let encoder ← if bf16Weights then TensorStruct.castDType .BFloat16 original else pure original
    for bf16Features in #[false, true] do
      let mel := if bf16Features then (Tensor.toBFloat16 features).forgetDType else features.forgetDType
      let expectedInput := if bf16Weights then (Tensor.toBFloat16 mel).forgetDType else (Tensor.toFloat32 mel).forgetDType
      -- Full and partial convolution chunks, output padding, and an empty sample.
      let audio := encoder.forwardVarLen mel #[25, 13, 0]
      let expected := encoder.forwardVarLen expectedInput #[25, 13, 0]
      LeanTest.assertEqual audio.runtimeShape #[3, 4, cfg.outputDim] "Varlen encoder output shape"
      LeanTest.assertEqual audio.actualSpec.dtype encoder.conv2d1Weight.actualSpec.dtype "Varlen output uses model dtype"
      LeanTest.assertTrue (audio.device == encoder.conv2d1Weight.device) "Varlen output uses model device"
      let error := Tensor.item (Tensor.maxAll (Tensor.abs (Tensor.sub (Tensor.toFloat32 audio) (Tensor.toFloat32 expected))))
      LeanTest.assertEqual error 0.0 "Automatic feature conversion equals an explicit model-dtype cast"
      LeanTest.assertTrue (Float.isFinite (Tensor.item (Tensor.sumAll (Tensor.toFloat32 audio))))
        "Varlen audio features are finite"
      let empty : Tensor { shape := #[1, 4, cfg.outputDim], device := .CPU } := Tensor.sliceChecked audio 0 2 1 #[1, 4, cfg.outputDim]
      LeanTest.assertEqual (Tensor.item (Tensor.maxAllChecked (Tensor.abs empty))) 0.0 "Zero-length sample stays zero"
      let short : Tensor { shape := #[1, 4, cfg.outputDim], device := .CPU } := Tensor.sliceChecked audio 0 1 1 #[1, 4, cfg.outputDim]
      let padding : Tensor { shape := #[1, 2, cfg.outputDim], device := .CPU } := Tensor.sliceChecked short 1 2 2 #[1, 2, cfg.outputDim]
      LeanTest.assertEqual (Tensor.item (Tensor.maxAllChecked (Tensor.abs padding))) 0.0 "Short-sample output padding stays zero"
      let dense := encoder.forward mel
      let denseExpected := encoder.forward expectedInput
      LeanTest.assertEqual dense.actualSpec.dtype encoder.conv2d1Weight.actualSpec.dtype "Dense encoder output uses model dtype"
      LeanTest.assertEqual (Tensor.item (Tensor.maxAll (Tensor.abs (Tensor.sub (Tensor.toFloat32 dense) (Tensor.toFloat32 denseExpected))))) 0.0
        "Dense path also aligns features and positional embeddings"
    let noBatch : Tensor { shape := #[0, cfg.numMelBins, 32], dtype := .Float32, device := .CPU } := Tensor.zeros #[0, cfg.numMelBins, 32]
    let empty := encoder.forwardVarLen noBatch.forgetDType #[]
    LeanTest.assertEqual empty.runtimeShape #[0, 4, cfg.outputDim] "Empty batch has the declared shape"
    LeanTest.assertEqual empty.actualSpec.dtype encoder.conv2d1Weight.actualSpec.dtype "Empty batch keeps model dtype"

@[test]
def testQwen3ASRBFloat16VariableLengthGeneration : IO Unit := do
  let cfg : Qwen3ASRConfig := { tinyChunkedCfg with thinkerConfig := {
    tinyChunkedCfg.thinkerConfig with audioConfig := {
      tinyChunkedCfg.thinkerConfig.audioConfig with nWindow := 8, nWindowInfer := 32 } } }
  let original ← Qwen3ASRForConditionalGeneration.init cfg
  let original := TensorStruct.detach original
  let model ← TensorStruct.castDType .BFloat16 original
  let aTok := Int64.ofNat cfg.thinkerConfig.audioTokenId.toNat
  let ids : Tensor { shape := #[2, 6], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked (Tensor.fromInt64Array
    #[aTok, aTok, aTok, aTok, 7, 9, aTok, aTok, 3, 5, 7, 9]) #[2, 6]
  let mel : Tensor { shape := #[2, cfg.thinkerConfig.audioConfig.numMelBins, 32], dtype := .Float32, device := .CPU } := Tensor.reshapeChecked
    (Tensor.sin (Tensor.toFloat32 (Tensor.arange (2 * cfg.thinkerConfig.audioConfig.numMelBins * 32))))
    #[2, cfg.thinkerConfig.audioConfig.numMelBins, 32]
  let mask : Tensor { shape := #[2, 32], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked (Tensor.fromInt64Array
    ((List.range 32).toArray.map (fun i => if i < 25 then (1 : Int64) else 0) ++
     (List.range 32).toArray.map (fun i => if i < 13 then (1 : Int64) else 0))) #[2, 32]
  let logits ← model.forward ids (some mel.forgetDType) (some mask) none
  LeanTest.assertEqual logits.actualSpec.dtype DType.BFloat16 "Audio insertion and full decoder retain BF16"
  LeanTest.assertTrue (Float.isFinite (Tensor.item (Tensor.sumAll (Tensor.toFloat32 logits)))) "BF16 ASR logits are finite"
  -- Exercise actual masked audio insertion, prompt prefill, and three decoder steps.
  let generated ← model.generateGreedy ids (some mel.forgetDType) (some mask) 3 #[]
  let explicit ← model.generateGreedy ids (some (Tensor.toBFloat16 mel).forgetDType) (some mask) 3 #[]
  LeanTest.assertEqual generated.1 9 "Greedy decoding appends all requested tokens"
  LeanTest.assertEqual generated.2.runtimeShape #[2, 9] "Both variable-length rows complete decoding"
  LeanTest.assertEqual (← Tensor.toUInt64Array generated.2)
    (← Tensor.toUInt64Array explicit.2)
    "Float32 frontend features produce the same tokens as an explicit BF16 cast"
  let prefixIds : Tensor { shape := #[2, 6], dtype := .Int64, device := .CPU } := Tensor.sliceChecked generated.2 1 0 6 #[2, 6]
  LeanTest.assertEqual (← Tensor.toUInt64Array prefixIds)
    (← Tensor.toUInt64Array ids) "Generation preserves both prompts"

@[test]
def testQwen3ASRVarLenAudioEncoderChunkedPath : IO Unit := do
  let cfg := tinyChunkedCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let batch : UInt64 := 2
  let frames : UInt64 := 96
  let mel ← Tensor.randn #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]
  let featureLens : Array UInt64 := #[80, 56]
  let audio := model.thinker.encodeAudioVarLen mel.forgetDType featureLens
  let s := Tensor.item (Tensor.sumAll audio)
  LeanTest.assertTrue (Float.isFinite s) "varlen chunked audio encoder output should be finite"

@[test]
def testQwen3ASRForwardVariableFeatureMaskChunkedBatch : IO Unit := do
  let cfg := tinyChunkedCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let batch : UInt64 := 2
  let frames : UInt64 := 96
  let valid0 : UInt64 := 80
  let valid1 : UInt64 := 56
  let audioLen0 := AudioEncoderConfig.featExtractOutputLength valid0
  let audioLen1 := AudioEncoderConfig.featExtractOutputLength valid1
  let seq : UInt64 := 13
  let aTok : Int64 := Int64.ofNat cfg.thinkerConfig.audioTokenId.toNat

  let mut idVals : Array Int64 := #[]
  for i in [:seq.toNat] do
    idVals := idVals.push (if i.toUInt64 < audioLen0 then aTok else (Int64.ofNat i + 3))
  for i in [:seq.toNat] do
    idVals := idVals.push (if i.toUInt64 < audioLen1 then aTok else (Int64.ofNat i + 17))
  let inputIds : Tensor { shape := #[batch, seq], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked (Tensor.fromInt64Array idVals) #[batch, seq]

  let mel ← Tensor.randn #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]
  let mut maskVals : Array Int64 := #[]
  for i in [:frames.toNat] do
    maskVals := maskVals.push (if i.toUInt64 < valid0 then 1 else 0)
  for i in [:frames.toNat] do
    maskVals := maskVals.push (if i.toUInt64 < valid1 then 1 else 0)
  let featureMask : Tensor { shape := #[batch, frames], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked (Tensor.fromInt64Array maskVals) #[batch, frames]

  let logits ← model.forward inputIds (some mel.forgetDType) (some featureMask) none
  let s := Tensor.item (Tensor.sumAll logits)
  LeanTest.assertTrue (Float.isFinite s) "chunked varlen forward should be finite"

@[test]
def testQwen3ASRForcedAlignFromOutputIds : IO Unit := do
  let batch : UInt64 := 1
  let seq : UInt64 := 6
  let timestampTokenId : UInt64 := 99
  let inputVals : Array Int64 := #[1, 99, 99, 7, 99, 99]
  let outputVals : Array Int64 := #[0, 10, 20, 0, 30, 40]
  let inputIds : Tensor { shape := #[batch, seq], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked (Tensor.fromInt64Array inputVals) #[batch, seq]
  let outputIds : Tensor { shape := #[batch, seq], dtype := .Int64, device := .CPU } := Tensor.reshapeChecked (Tensor.fromInt64Array outputVals) #[batch, seq]
  let wordLists : Array (Array String) := #[#["a", "b"]]
  let results ← alignFromOutputIds inputIds outputIds wordLists timestampTokenId 10.0
  let r := results.getD 0 { items := #[] }
  LeanTest.assertEqual r.items.size 2 "forced aligner should return one span per word"
  let i0 := r.items.getD 0 default
  let i1 := r.items.getD 1 default
  LeanTest.assertTrue (Float.abs (i0.startTime - 0.1) < 1e-6) "first start_time should match converted timestamp"
  LeanTest.assertTrue (Float.abs (i0.endTime - 0.2) < 1e-6) "first end_time should match converted timestamp"
  LeanTest.assertTrue (Float.abs (i1.startTime - 0.3) < 1e-6) "second start_time should match converted timestamp"
  LeanTest.assertTrue (Float.abs (i1.endTime - 0.4) < 1e-6) "second end_time should match converted timestamp"

@[test]
def testQwen3ASRStreamModelPushAudioBoundsAccum : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let tok := mkAsciiStreamingTokenizer
  let pre := tinyPreprocessorCfg cfg.thinkerConfig.audioConfig.numMelBins 32
  let sm : StreamModel := {
    device := .CPU
    cfg := cfg
    model := model
    tok := tok
    preprocessor := pre
    modelDir := "."
  }
  let ss0 ← newSession sm (chunkSec := 0.5) (hopSec := 10.0)
  let pcm : Array Float := Array.replicate 128 0.1
  let (ss1, out) ← pushAudio sm ss0 pcm
  LeanTest.assertEqual ss1.asrState.buffer.size pcm.size
    "when hop threshold is not met, push should append audio into the streaming buffer"
  LeanTest.assertTrue (!out.didDecode)
    "push should not decode when hop threshold is not met"
  LeanTest.assertTrue ss1.promptCache.isNone
    "when no decode occurs, stream session should keep prompt cache unset"

@[test]
def testQwen3ASRStreamModelPersistsPromptCache : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let tok := mkAsciiStreamingTokenizer
  let pre := tinyPreprocessorCfg cfg.thinkerConfig.audioConfig.numMelBins 32
  let sm : StreamModel := {
    device := .CPU
    cfg := cfg
    model := model
    tok := tok
    preprocessor := pre
    modelDir := "."
  }
  let chunkSec : Float := 2.0 / 16000.0
  let ss0 ← newSession sm
    (chunkSec := chunkSec)
    (hopSec := chunkSec)
    (decodeMode := .fullAccumulation)

  let (ss1, out1) ← pushAudio sm ss0 #[0.1, 0.2] (maxNewTokens := 1)
  LeanTest.assertTrue out1.didDecode
    "push should decode when enough audio is buffered for one chunk"
  LeanTest.assertTrue ss1.promptCache.isSome
    "stream session should persist decoder prompt cache after first decode"

  let (ss2, out2) ← pushAudio sm ss1 #[0.3, 0.4] (maxNewTokens := 1)
  LeanTest.assertTrue out2.didDecode
    "subsequent pushes should continue decoding at hop cadence"
  LeanTest.assertTrue ss2.promptCache.isSome
    "stream session should keep prompt cache between decode hops"

@[test]
def testQwen3ASRStreamingDecodeCacheBenchmark : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let tok := mkAsciiStreamingTokenizer
  let pre := tinyPreprocessorCfg cfg.thinkerConfig.audioConfig.numMelBins 32
  let chunkSec : Float := 2.0 / 16000.0
  let stBase ← initStreamingState
    cfg.supportLanguages
    (context := "")
    (language := some "English")
    (unfixedChunkNum := 0)
    (unfixedTokenNum := 1)
    (chunkSizeSec := chunkSec)
    (stepSizeSec := chunkSec)
    (decodeMode := .fullAccumulation)
  let hopAudio : Array Float := #[0.1, 0.2]
  let iters : Nat := 20

  -- Warmup both paths once to reduce one-time kernel/setup noise.
  let (_warmSt1, warmPromptCache) ←
    streamingTranscribeWithModelCached
      model
      tok
      pre
      hopAudio
      stBase
      (maxNewTokens := 1)
      (eosTokenIds := #[])
  let (_warmSt2, _warmDecodeCache) ←
    streamingTranscribeWithModelStateCached
      model
      tok
      pre
      hopAudio
      stBase
      (cache := some { promptCache := warmPromptCache })
      (maxNewTokens := 1)
      (eosTokenIds := #[])

  let t0 ← IO.monoMsNow
  let mut stPrompt := stBase
  let mut promptCache : Option (StreamingPromptCache cfg) := none
  for _ in [:iters] do
    let (stNext, cacheNext) ←
      streamingTranscribeWithModelCached
        model
        tok
        pre
        hopAudio
        stPrompt
        (prefixCache := promptCache)
        (maxNewTokens := 1)
        (eosTokenIds := #[])
    stPrompt := stNext
    promptCache := cacheNext
  let t1 ← IO.monoMsNow

  let t2 ← IO.monoMsNow
  let mut stDecode := stBase
  let mut decodeCache : Option (StreamingDecodeCache cfg) := none
  for _ in [:iters] do
    let (stNext, cacheNext) ←
      streamingTranscribeWithModelStateCached
        model
        tok
        pre
        hopAudio
        stDecode
        (cache := decodeCache)
        (maxNewTokens := 1)
        (eosTokenIds := #[])
    stDecode := stNext
    decodeCache := cacheNext
  let t3 ← IO.monoMsNow

  LeanTest.assertEqual stPrompt.chunkId stDecode.chunkId
    "baseline and decode-cache streaming paths should decode the same number of chunks"
  LeanTest.assertEqual stPrompt.text stDecode.text
    "baseline and decode-cache streaming paths should produce identical text for deterministic tiny model"

  let baselineMs := (t1 - t0).toFloat
  let optimizedMs := (t3 - t2).toFloat
  let speedup :=
    if optimizedMs > 0.0 then baselineMs / optimizedMs else 0.0
  IO.println
    s!"[bench][qwen3asr_stream_decode_cache] iters={iters} baseline_ms={baselineMs} optimized_ms={optimizedMs} speedup={speedup}"
