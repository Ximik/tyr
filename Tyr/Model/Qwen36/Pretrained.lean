/-
  Tyr/Model/Qwen36/Pretrained.lean

  Qwen3.6 convenience surface over the shared Qwen3.5-MoE implementation used
  by Tyr. The underlying architecture is the same family; this module exposes
  Qwen3.6-specific defaults, repo-id helpers, and user-facing names.
-/
import Tyr.Hub
import Tyr.Model.Qwen35.Pretrained
import Tyr.Model.Qwen36.ConfigIO

namespace torch.qwen36

variable {device : Device}

namespace hub

open torch.Hub

abbrev DownloadOptions := Hub.DownloadOptions

def qwen36CollectionRepoIds : Array String := #[
  "Qwen/Qwen3.6-35B-A3B"
]

def isQwen36CollectionRepoId (repoId : String) : Bool :=
  qwen36CollectionRepoIds.contains repoId

export torch.qwen35.hub (
  tokenizerFiles
)

export torch.Hub (
  shardFilesFromIndexFile
  findCachedSnapshot?
  resolvePretrainedDir
  detectWeightLayout
)

end hub

abbrev Qwen36ForCausalLM (cfg : Config) (dtype : DType := .Float32) := torch.qwen35.Qwen35ForCausalLM (device := device) cfg dtype
abbrev Qwen36ForConditionalGeneration (cfg : VLConfig) (dtype : DType := .Float32)
    (visionDType : DType := .Float32) :=
  torch.qwen35.Qwen35ForConditionalGeneration (device := device) cfg dtype visionDType

namespace Qwen36ForCausalLM

export torch.qwen35.Qwen35ForCausalLM (
  init
  embedTokens
  forwardEmbeds
  forward
  generateFromEmbeds
  generateFromEmbedsStream
  generate
  generateStream
  generateUncached
  generateGreedy
)

export torch.qwen35.Qwen35ForCausalLM (loadFromPretrained)

end Qwen36ForCausalLM

namespace Qwen36ForConditionalGeneration

export torch.qwen35.Qwen35ForConditionalGeneration (
  init
  getImageFeatures
  getVideoFeatures
  forwardText
  forwardWithImageFeatures
  forwardWithImageAndVideoFeatures
  forwardWithImagePatches
  forwardWithImageAndVideoPatches
  generate
  generateStream
  generateUncached
  loadFromPretrained
)

end Qwen36ForConditionalGeneration

end torch.qwen36
