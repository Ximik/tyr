/-
  Tyr/Model/Qwen35/Qwen36/ConfigIO.lean

  Hugging Face `config.json` loaders for Qwen3.6 checkpoints.
  The wire format matches Tyr's shared Qwen3.5-MoE loaders; this module just
  supplies Qwen3.6-appropriate defaults and namespace.
-/
import Tyr.Model.Qwen35.VLConfigIO
import Tyr.Model.Qwen35.Qwen36.Config

namespace torch.qwen36

namespace Config

def loadFromFile (path : String) (defaults : Config := Config.qwen36_35B_A3B) : IO Config :=
  qwen35.Config.loadFromFile path defaults

def loadFromPretrainedDir (modelDir : String) (defaults : Config := Config.qwen36_35B_A3B) : IO Config :=
  qwen35.Config.loadFromPretrainedDir modelDir defaults

end Config

namespace VLConfig

def loadFromFile (path : String) (defaults : VLConfig := VLConfig.qwen36_35B_A3B) : IO VLConfig :=
  qwen35.VLConfig.loadFromFile path defaults

def loadFromPretrainedDir (modelDir : String) (defaults : VLConfig := VLConfig.qwen36_35B_A3B) : IO VLConfig :=
  qwen35.VLConfig.loadFromPretrainedDir modelDir defaults

end VLConfig

end torch.qwen36
