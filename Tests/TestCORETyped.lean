import Examples.NanoChat.Eval.CORE
import LeanTest

namespace Tests.CORETyped
open torch torch.Eval.CORE

private def zeroTokenizer (s : String) : Array UInt64 := Array.replicate s.length 0
private def sevenTokenModel : ModelRunner (device := .CPU) 7 := fun {batch seq} _ =>
  pure (Tensor.ones #[batch, seq, 7])

@[test] def testCORETypedVocabularyAndLanguageModeling : IO Unit := do
  let result ← evaluateSingleExample (.lm { context := "a", continuation := "bc" })
    { taskType := .languageModeling } zeroTokenizer sevenTokenModel { bosToken := 0 }
  LeanTest.assertEqual result (some true)
    "Evaluation uses the backend's actual seven-token vocabulary and indexed sequence"

@[test] def testCORETypedMultipleChoiceAndEmptyChoices : IO Unit := do
  let taskMetadata : TaskMeta := { taskType := .multipleChoice }
  let result ← evaluateSingleExample (.mc { query := "q", choices := #["a", "b"], gold := 0 })
    taskMetadata zeroTokenizer sevenTokenModel { bosToken := 0 }
  LeanTest.assertEqual result (some true)
  let empty ← evaluateSingleExample (.mc { query := "q", choices := #[], gold := 0 })
    taskMetadata zeroTokenizer sevenTokenModel { bosToken := 0 }
  LeanTest.assertEqual empty none "Empty choices do not invoke a zero-batch model"

end Tests.CORETyped
