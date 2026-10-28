/-
  Tyr/Model/Qwen3/Config.lean

  Config aliases/helpers for the standalone Qwen3 causal-LM model.
  This lives alongside (not inside) Flux-oriented Qwen usage.
-/
import Tyr.Model.Qwen.Config

namespace torch.qwen3

/-- Qwen3 causal-LM config (shared core schema with `torch.qwen`). -/
abbrev Config := qwen.QwenConfig

namespace Config

/-- Default Qwen3-4B config. -/
def qwen3_4B : Config := qwen.QwenConfig.qwen3_4B

/-- Compute head dimension from hidden size and attention heads. -/
def computeHeadDim (cfg : Config) : Nat :=
  qwen.QwenConfig.computeHeadDim cfg

/-- Compute key/value projection dimension. -/
def kvDim (cfg : Config) : Nat :=
  qwen.QwenConfig.kvDim cfg

/-- Number of query heads per KV group. -/
def numHeadsPerKVGroup (cfg : Config) : Nat :=
  qwen.QwenConfig.numHeadsPerKVGroup cfg

end Config

end torch.qwen3
