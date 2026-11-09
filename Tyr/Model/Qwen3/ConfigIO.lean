/-
  Tyr/Model/Qwen3/ConfigIO.lean

  HuggingFace `config.json` loader for standalone Qwen3 causal-LM.
-/
import Tyr.Model.Qwen3.Config
import Tyr.Model.Qwen.ConfigIO

namespace torch.qwen3

namespace Config

/-- Load Qwen3 config from a HuggingFace-style `config.json` file. -/
def loadFromFile (path : String) (defaults : Config := Config.qwen3_4B) : IO Config := do
  IO.ofExcept (qwen.QwenConfig.parseJson (← ConfigJson.parseFile path) defaults)

/-- Load Qwen3 config from a model directory containing `config.json`. -/
def loadFromPretrainedDir (modelDir : String) (defaults : Config := Config.qwen3_4B) : IO Config :=
  loadFromFile s!"{modelDir}/config.json" defaults

end Config

end torch.qwen3
