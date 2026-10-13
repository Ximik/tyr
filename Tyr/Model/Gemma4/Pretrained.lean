/-
  Tyr/Model/Gemma4/Pretrained.lean

  Resolve/load Gemma 4 checkpoints from either:
  - local model directory, or
  - HuggingFace repo id (download to local cache on demand).
-/
import Tyr.Hub
import Tyr.Model.Gemma4.ConfigIO
import Tyr.Model.Gemma4.Weights

namespace torch.gemma4.hub

open torch.Hub

/-- Public Gemma 4 repo ids explicitly covered by Tyr as of 2026-04-02. -/
def gemma4CollectionRepoIds : Array String := #[
  "google/gemma-4-E2B",
  "google/gemma-4-E2B-it",
  "google/gemma-4-E4B",
  "google/gemma-4-E4B-it",
  "google/gemma-4-26B-A4B",
  "google/gemma-4-26B-A4B-it",
  "google/gemma-4-31B",
  "google/gemma-4-31B-it"
]

def isGemma4CollectionRepoId (repoId : String) : Bool :=
  gemma4CollectionRepoIds.contains repoId

/-- Gemma 4 tokenizer files to download. -/
def tokenizerFiles : Array String := #[
  "tokenizer.json",
  "tokenizer_config.json",
  "processor_config.json"
]

export torch.Hub (resolvePretrainedDir detectWeightLayout findCachedSnapshot?)

end torch.gemma4.hub

namespace torch.gemma4

namespace Gemma4ForCausalLM

/-- Load a text Gemma 4 checkpoint from local dir or HF repo id. -/
def loadFromPretrained
    (source : String)
    (defaults : Config := Config.gemma4_E4B)
    (revision : String := "main")
    (cacheDir : String := Hub.defaultCacheDir)
    : IO (Sigma (fun cfg => Gemma4ForCausalLM cfg)) := do
  Hub.loadModelFromPretrained
    source
    revision
    cacheDir
    hub.tokenizerFiles
    (fun modelDir cfg => Config.loadFromPretrainedDir modelDir cfg)
    defaults
    (fun modelDir cfg => Gemma4ForCausalLM.loadSharded modelDir cfg)
    (fun path cfg => Gemma4ForCausalLM.load path cfg)

end Gemma4ForCausalLM

end torch.gemma4
