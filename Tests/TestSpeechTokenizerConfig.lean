import Tyr.Model.Qwen3TTS.SpeechTokenizer
import LeanTest

open torch.qwen3tts

namespace Tests.SpeechTokenizerConfig

/-- Exact speech_tokenizer/config.json from Qwen/Qwen3-TTS-12Hz-0.6B-Base
revision 5d83992436eae1d760afd27aff78a71d676296fc, matching fixtures.json.
Git blob SHA-1: 06cc8dc4c5ec8a1929086b71b98c313020d9268b. -/
private def pinnedConfigPath := "Tests/fixtures/qwen3tts/tokenizer-12hz-0.6b-config.json"

@[test] def testPinnedSpeechTokenizerConfigLoads : IO Unit := do
  let cfg ← SpeechTokenizer12HzConfig.loadFromFile pinnedConfigPath
  LeanTest.assertEqual cfg.encoderNumQuantizers 32 "Available encoder codebooks"
  LeanTest.assertEqual cfg.encoderCodebookCounts (1, 15) "One semantic plus fifteen acoustic codebooks"

end Tests.SpeechTokenizerConfig
