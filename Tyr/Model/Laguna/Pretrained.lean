/-
  Tyr/Model/Laguna/Pretrained.lean

  Resolve/load poolside Laguna checkpoints from either:
  - local model directory, or
  - HuggingFace repo id (download to local cache on demand).
-/
import Tyr.Hub
import Tyr.Model.Laguna.ConfigIO
import Tyr.Model.Laguna.Weights

namespace torch.laguna.hub

variable {device : Device}
open torch.Hub

/-- Public poolside Laguna repo ids explicitly covered by Tyr as of 2026-07-22. -/
def lagunaCollectionRepoIds : Array String := #[
  "poolside/Laguna-S-2.1-NVFP4"
]

def isLagunaCollectionRepoId (repoId : String) : Bool :=
  lagunaCollectionRepoIds.contains repoId

/-- Laguna tokenizer files to download. -/
def tokenizerFiles : Array String := #[
  "tokenizer.json",
  "tokenizer_config.json",
  "special_tokens_map.json",
  "chat_template.jinja"
]

export torch.Hub (resolvePretrainedDir detectWeightLayout findCachedSnapshot? shardFilesFromIndexFile)

end torch.laguna.hub

namespace torch.laguna

namespace LagunaForCausalLM

/-- Load a Laguna checkpoint from a local dir or HF repo id.
    Returns the resolved config and model as a dependent pair. -/
def loadFromPretrained
    (source : String)
    (defaults : Config := Config.laguna_s_2_1)
    (revision : String := "main")
    (cacheDir : String := Hub.defaultCacheDir)
    (device : Device := Device.CPU)
    : IO (Sigma (fun cfg => LagunaForCausalLM (device := device) cfg)) := do
  Hub.loadModelFromPretrained
    source
    revision
    cacheDir
    hub.tokenizerFiles
    (fun modelDir cfg => Config.loadFromPretrainedDir modelDir cfg)
    defaults
    (fun modelDir cfg => LagunaForCausalLM.loadSharded modelDir cfg device)
    (fun path cfg => LagunaForCausalLM.load path cfg device)

end LagunaForCausalLM

end torch.laguna
