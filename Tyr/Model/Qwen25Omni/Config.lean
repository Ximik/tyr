/-  
  Tyr/Model/Qwen25Omni/Config.lean

  Text (thinker) configuration for Qwen2.5-Omni checkpoints.
  This model family is intentionally separate from Qwen35.
-/
import Tyr.Model.Qwen.Config

namespace torch.qwen25omni

/-- Qwen2.5-Omni thinker text config reuses the shared Qwen core schema. -/
abbrev Config := qwen.QwenConfig

namespace Config

/-- Qwen2.5-Omni-3B thinker text defaults (from HF config.json). -/
def qwen25omni_3B : Config := {
  vocab_size := 151936
  hidden_size := 2048
  intermediate_size := 11008
  num_hidden_layers := 36
  num_attention_heads := 16
  num_key_value_heads := 2
  half_head_dim := 64
  rope_theta := 1000000.0
  rms_norm_eps := 1e-6
}

/-- Qwen2.5-Omni-7B thinker text defaults (from HF config.json). -/
def qwen25omni_7B : Config := {
  vocab_size := 152064
  hidden_size := 3584
  intermediate_size := 18944
  num_hidden_layers := 28
  num_attention_heads := 28
  num_key_value_heads := 4
  half_head_dim := 64
  rope_theta := 1000000.0
  rms_norm_eps := 1e-6
}

end Config

end torch.qwen25omni

