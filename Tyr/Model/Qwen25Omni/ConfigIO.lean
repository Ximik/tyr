/-
  Tyr/Model/Qwen25Omni/ConfigIO.lean

  HuggingFace `config.json` loader for Qwen2.5-Omni thinker text checkpoints.
-/
import Tyr.Model.Qwen25Omni.Config
import Tyr.Model.Qwen.ConfigIO

namespace torch.qwen25omni

open Lean ConfigJson

/-- The thinker's text decoder: `thinker_config.text_config`, `text_config`, or the root. -/
private def parseTextConfig (root : Json) (defaults : Config) : Except String Config := do
  let textJson ← match ← objectField? root "thinker_config" with
    | some thinker => pure ((← objectField? thinker "text_config").getD thinker)
    | none => pure ((← objectField? root "text_config").getD root)
  qwen.QwenConfig.parseJson textJson defaults (allowMRope := true)

namespace Config

/-- Load Qwen2.5-Omni thinker text config from a HuggingFace-style `config.json`. -/
def loadFromFile (path : String) (defaults : Config := Config.qwen25omni_3B) : IO Config := do
  IO.ofExcept (parseTextConfig (← ConfigJson.parseFile path) defaults)

/-- Load Qwen2.5-Omni thinker text config from `modelDir/config.json`. -/
def loadFromPretrainedDir (modelDir : String) (defaults : Config := Config.qwen25omni_3B) : IO Config :=
  loadFromFile s!"{modelDir}/config.json" defaults

end Config

end torch.qwen25omni
