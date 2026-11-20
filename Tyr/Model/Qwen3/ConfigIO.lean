/-
  Tyr/Model/Qwen3/ConfigIO.lean

  HuggingFace `config.json` loader for standalone Qwen3 causal-LM.
-/
import Tyr.Model.Qwen3.Config
import Tyr.Model.Qwen.ConfigIO

namespace torch.qwen3

namespace Config

/-- Load Qwen3 config from a HuggingFace-style `config.json` file, which must
declare `"model_type": "qwen3"`. -/
def loadFromFile (path : String) (defaults : Config := Config.qwen3_4B) : IO Config := do
  let json ← ConfigJson.parseFile path
  IO.ofExcept do
    let modelType : Option String ← ConfigJson.field? json "model_type"
    unless modelType == some "qwen3" do
      throw s!"unsupported 'model_type' = {modelType}; Qwen3ForCausalLM implements \"qwen3\""
    qwen.QwenConfig.parseJson json defaults

/-- Load Qwen3 config from a model directory containing `config.json`. -/
def loadFromPretrainedDir (modelDir : String) (defaults : Config := Config.qwen3_4B) : IO Config :=
  loadFromFile s!"{modelDir}/config.json" defaults

end Config

end torch.qwen3
