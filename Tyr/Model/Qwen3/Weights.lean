/-
  Tyr/Model/Qwen3/Weights.lean

  Pretrained weight loading for standalone Qwen3 causal-LM.
-/
import Tyr.Typed
import Tyr.Log
import Tyr.SafeTensors.Load
import Tyr.Model.Utils
import Tyr.Model.Qwen.Weights
import Tyr.Model.Qwen3.Model

namespace torch.qwen3

open torch.Log
open torch.Model (reqGradFalse)
open torch.safetensors (tryLoadOptionalTensor tryLoadOptionalTensorSharded)

namespace Qwen3ForCausalLM

/-- Load Qwen3 causal-LM from a sharded HF SafeTensors directory.
    Falls back to tied embeddings if `lm_head.weight` is absent. -/
def loadSharded (modelDir : String) (cfg : Config := Config.qwen3_4B)
    (log : Handlers := {})
    : IO (Qwen3ForCausalLM cfg) := do
  log.onInfo s!"Loading Qwen3ForCausalLM from {modelDir}..."
  let model ← qwen.loadQwen3ModelSharded modelDir cfg false log

  let lmHeadOpt ← tryLoadOptionalTensorSharded modelDir "lm_head.weight" #[cfg.vocab_size, cfg.hidden_size]
  let (lmHead, tieWordEmbeddings) ←
    match lmHeadOpt with
    | some w => pure (reqGradFalse w.toFloat32, false)
    | none => do
      log.onInfo "  lm_head.weight not found; using tied embeddings."
      pure (model.embed_tokens, true)

  log.onInfo "Loaded Qwen3ForCausalLM weights."
  pure { model, lmHead, tieWordEmbeddings }

/-- Load Qwen3 causal-LM from a single SafeTensors file.
    Falls back to tied embeddings if `lm_head.weight` is absent. -/
def load (path : String) (cfg : Config := Config.qwen3_4B)
    (log : Handlers := {})
    : IO (Qwen3ForCausalLM cfg) := do
  log.onInfo s!"Loading Qwen3ForCausalLM from {path}..."
  let model ← qwen.loadQwen3Model path cfg log

  let lmHeadOpt ← tryLoadOptionalTensor path "lm_head.weight" #[cfg.vocab_size, cfg.hidden_size]
  let (lmHead, tieWordEmbeddings) ←
    match lmHeadOpt with
    | some w => pure (reqGradFalse w.toFloat32, false)
    | none => do
      log.onInfo "  lm_head.weight not found; using tied embeddings."
      pure (model.embed_tokens, true)

  log.onInfo "Loaded Qwen3ForCausalLM weights."
  pure { model, lmHead, tieWordEmbeddings }

end Qwen3ForCausalLM

end torch.qwen3
