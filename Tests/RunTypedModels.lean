import Tests.TestModdedGPT
import Tests.TestNanoChatCheckpoint
import Tests.TestCORETyped
import Tests.TestTTSTypedLoading
import Tests.TestQwen3TTS
import Tests.TestSpeechTokenizerConfig
import Tests.TestFLM
import Tests.TestDataLoader
import Tests.TestQwen3ASR
import Tests.TestTypedNeural
import Tests.TestDiffEqSdeAdjoint
import Tests.TestDiffEqImplicitAdjointParity
import Tests.TestNanoChatTokens
import LeanTest
import Lean.Util.Path
import Tests.TestQwenTypedLoading
import Tests.TestQwen3Model
import Tests.TestSileroVAD
import Tests.TestTypedStandalone
import Tests.TestADFrontendSignature
import Tests.TestADTensorStructFlatten
import Tests.TestWidgetTyped
import Tests.TestModuleTyped
import Tests.TestGPTTyped
import Tests.TestInferenceCache
import Tests.TestDiffEqAdjoint
import Tests.TestDiffEqTyped
import Tests.TestDiffEqContainerBrownian

/-- Native acceptance for the migrated model, AD, cache and solver boundaries. -/
unsafe def main : IO UInt32 := do
  Lean.initSearchPath (← Lean.findSysroot)
  let env ← Lean.importModules #[
    { module := `Tests.TestModdedGPT },
    { module := `Tests.TestNanoChatCheckpoint },
    { module := `Tests.TestCORETyped },
    { module := `Tests.TestTTSTypedLoading },
    { module := `Tests.TestQwen3TTS },
    { module := `Tests.TestSpeechTokenizerConfig },

    { module := `Tests.TestFLM },
    { module := `Tests.TestDataLoader },
    { module := `Tests.TestQwen3ASR },
    { module := `Tests.TestTypedNeural },

    { module := `Tests.TestDiffEqSdeAdjoint },
    { module := `Tests.TestDiffEqImplicitAdjointParity },
    { module := `Tests.TestNanoChatTokens },

    { module := `Tests.TestQwenTypedLoading },
    { module := `Tests.TestQwen3Model },
    { module := `Tests.TestSileroVAD },
    { module := `Tests.TestTypedStandalone },
    { module := `Tests.TestADFrontendSignature },
    { module := `Tests.TestADTensorStructFlatten },
    { module := `Tests.TestWidgetTyped },
    { module := `Tests.TestModuleTyped },
    { module := `Tests.TestGPTTyped },
    { module := `Tests.TestInferenceCache },
    { module := `Tests.TestDiffEqAdjoint },
    { module := `Tests.TestDiffEqTyped },
    { module := `Tests.TestDiffEqContainerBrownian }
  ] {}
  LeanTest.runTestsAndExit env {} { failFast := true }
