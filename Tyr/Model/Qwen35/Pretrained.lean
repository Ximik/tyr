/-
  Tyr/Model/Qwen35/Pretrained.lean

  Resolve/load Qwen3.5 checkpoints from either:
  - local model directory, or
  - HuggingFace repo id (download to local cache on demand).
-/
import Tyr.Hub
import Tyr.Model.Qwen35.ConfigIO
import Tyr.Model.Qwen35.Weights
import Tyr.Model.Qwen35.VLConfigIO
import Tyr.Model.Qwen35.VLWeights

namespace torch.qwen35.hub

open torch.Hub

/-- Public Qwen3.5/Qwen3.6 repo ids explicitly covered by Tyr as of 2026-04-16. -/
def qwen35CollectionRepoIds : Array String := #[
  "Qwen/Qwen3.5-0.8B",
  "Qwen/Qwen3.5-0.8B-Base",
  "Qwen/Qwen3.5-397B-A17B",
  "Qwen/Qwen3.5-397B-A17B-FP8",
  "Qwen/Qwen3.5-122B-A14B",
  "Qwen/Qwen3.5-122B-A14B-FP8",
  "Qwen/Qwen3.5-35B-A3B",
  "Qwen/Qwen3.5-35B-A3B-FP8",
  "Qwen/Qwen3.5-35B-A3B-Base",
  "Qwen/Qwen3.5-27B",
  "Qwen/Qwen3.5-27B-FP8",
  "Qwen/Qwen3.6-35B-A3B"
]

def isQwen35CollectionRepoId (repoId : String) : Bool :=
  qwen35CollectionRepoIds.contains repoId

/-- Qwen tokenizer files to download. -/
def tokenizerFiles : Array String := #[
  "tokenizer.json",
  "tokenizer_config.json",
  "vocab.json",
  "merges.txt"
]

export torch.Hub (resolvePretrainedDir detectWeightLayout findCachedSnapshot? shardFilesFromIndexFile)

end torch.qwen35.hub

namespace torch.qwen35

variable {device : Device}

namespace Qwen35ForCausalLM

/-- Load a text Qwen3.5 checkpoint from local dir or HF repo id. -/
def loadFromPretrained
    (source : String)
    (defaults : Config := Config.qwen35_9B)
    (revision : String := "main")
    (cacheDir : String := Hub.defaultCacheDir)
    (device : Device := Device.CPU)
    : IO (Sigma (fun cfg => Σ dtype : RealFloatingDType, Qwen35ForCausalLM (device := device) cfg dtype.val)) := do
  Hub.loadModelFromPretrained
    source
    revision
    cacheDir
    hub.tokenizerFiles
    (fun modelDir cfg => Config.loadFromPretrainedDir modelDir cfg)
    defaults
    (fun modelDir cfg => Qwen35ForCausalLM.loadSharded modelDir cfg device)
    (fun path cfg => Qwen35ForCausalLM.load path cfg device)

end Qwen35ForCausalLM

namespace Qwen35ForConditionalGeneration

/-- Load a multimodal Qwen3.5 checkpoint from local dir or HF repo id. -/
def loadFromPretrained
    (source : String)
    (defaults : VLConfig := {})
    (revision : String := "main")
    (cacheDir : String := Hub.defaultCacheDir)
    (device : Device := Device.CPU)
    : IO (Sigma (fun cfg => Σ dtype : RealFloatingDType, Σ visionDType : RealFloatingDType, Qwen35ForConditionalGeneration (device := device) cfg dtype.val visionDType.val)) := do
  Hub.loadModelFromPretrained
    source
    revision
    cacheDir
    hub.tokenizerFiles
    (fun modelDir cfg => VLConfig.loadFromPretrainedDir modelDir cfg)
    defaults
    (fun modelDir cfg => Qwen35ForConditionalGeneration.loadSharded modelDir cfg device)
    (fun path cfg => Qwen35ForConditionalGeneration.load path cfg device)

end Qwen35ForConditionalGeneration

end torch.qwen35
