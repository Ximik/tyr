import Tests.TestWhisperTyped
import Tests.TestFluxTyped
import Tests.TestVAETyped
import Tests.TestGemma4Typed
import Tests.TestShapePackedTensor
import Tests.TestKittenTTS
import Tests.TestQwen36Pretrained
import Tests.TestQwen36Model
import Tests.TestQwen35Pretrained
import Tests.TestQwen35Multimodal
import Tests.TestQwen35Model
import Tests.TestModelIO
import Tests.TestCheckpoint
import Tests.TestModdedGPT
import Tests.TestNanoChatCheckpoint
import Tests.TestCORETyped
import Tests.TestTTSTypedLoading
import Tests.TestQwen3TTS
import Tests.TestSpeechTokenizerConfig
import Tests.TestDataLoader
import Tests.TestQwen3ASR
import Tests.TestTypedNeural
import Tests.TestNanoChatTokens
import LeanTest
import Lean.Util.Path
import Tests.TestQwenTypedLoading
import Tests.TestQwen3Model
import Tests.TestSileroVAD
import Tests.TestTypedStandalone
import Tests.TestTensorStructFlatten
import Tests.TestWidgetTyped
import Tests.TestModuleTyped
import Tests.TestGPTTyped
import Tests.TestInferenceCache

/-- Native acceptance for the migrated model, AD, cache and solver boundaries. -/
unsafe def main : IO UInt32 := do
  Lean.initSearchPath (← Lean.findSysroot)
  let env ← Lean.importModules #[
    { module := `Tests.TestWhisperTyped },
    { module := `Tests.TestFluxTyped },
    { module := `Tests.TestVAETyped },
    { module := `Tests.TestGemma4Typed },
    { module := `Tests.TestShapePackedTensor },
    { module := `Tests.TestKittenTTS },
    { module := `Tests.TestQwen36Pretrained },
    { module := `Tests.TestQwen36Model },
    { module := `Tests.TestQwen35Pretrained },
    { module := `Tests.TestQwen35Multimodal },
    { module := `Tests.TestQwen35Model },
    { module := `Tests.TestModelIO },
    { module := `Tests.TestCheckpoint },
    { module := `Tests.TestModdedGPT },
    { module := `Tests.TestNanoChatCheckpoint },
    { module := `Tests.TestCORETyped },
    { module := `Tests.TestTTSTypedLoading },
    { module := `Tests.TestQwen3TTS },
    { module := `Tests.TestSpeechTokenizerConfig },

    { module := `Tests.TestDataLoader },
    { module := `Tests.TestQwen3ASR },
    { module := `Tests.TestTypedNeural },

    { module := `Tests.TestNanoChatTokens },

    { module := `Tests.TestQwenTypedLoading },
    { module := `Tests.TestQwen3Model },
    { module := `Tests.TestSileroVAD },
    { module := `Tests.TestTypedStandalone },
    { module := `Tests.TestTensorStructFlatten },
    { module := `Tests.TestWidgetTyped },
    { module := `Tests.TestModuleTyped },
    { module := `Tests.TestGPTTyped },
    { module := `Tests.TestInferenceCache }
  ] {}
  LeanTest.runTestsAndExit env {} { failFast := true }
