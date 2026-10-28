/-  
  Tyr/Model/Qwen25Omni/Weights.lean

  Thinker text weight loading for Qwen2.5-Omni checkpoints.
  Supports both single-file and sharded SafeTensors layouts.
-/
import Tyr.Typed
import Tyr.Log
import Tyr.SafeTensors.Load
import Tyr.Model.Utils
import Tyr.Model.Qwen25Omni.Config
import Tyr.Model.Qwen3.Model
import Tyr.Model.Qwen.Model

namespace torch.qwen25omni

variable {device : Device}
open torch.Log
open torch.Model (reqGradFalse dequantizeFP8)
open torch.safetensors (pushUnique tryLoadTensor tryLoadTensorSharded
  loadTensorCandidates loadTensorShardedCandidates
  tryLoadTensorCandidates tryLoadTensorShardedCandidates)

private def baseNameCandidates (base : String) : Array String :=
  let out : Array String := #[]
  let out := pushUnique out base
  let out := pushUnique out s!"thinker.{base}"
  let out := pushUnique out s!"language_model.{base}"
  let out := pushUnique out s!"model.language_model.{base}"
  if base.startsWith "model." then
    let suffix := (base.drop 6).toString
    let out := pushUnique out suffix
    let out := pushUnique out s!"thinker.{suffix}"
    let out := pushUnique out s!"language_model.{suffix}"
    pushUnique out s!"model.language_model.{suffix}"
  else
    out

private def tensorNameCandidates (name : String) : Array String :=
  let out : Array String := #[]
  let out := pushUnique out name
  let out := pushUnique out s!"thinker.{name}"
  let out := pushUnique out s!"language_model.{name}"
  let out := pushUnique out s!"model.language_model.{name}"
  if name.startsWith "model." then
    let suffix := (name.drop 6).toString
    let out := pushUnique out suffix
    let out := pushUnique out s!"thinker.{suffix}"
    let out := pushUnique out s!"language_model.{suffix}"
    pushUnique out s!"model.language_model.{suffix}"
  else
    out

private def loadLinearWeightSharded (modelDir : String) (baseName : String) (outDim inDim : Nat)
    (device : Device := .CPU) : IO (Tensor { device := device, shape := #[outDim, inDim], dtype := .Float32 }) := do
  let bases := baseNameCandidates baseName
  for b in bases do
    let wName := s!"{b}.weight"
    if let some ⟨_, w⟩ ← tryLoadTensorSharded (device := device) modelDir wName #[outDim, inDim] then
      let scaleName := s!"{b}.weight_scale_inv"
      let scaleInvOpt ← tryLoadTensorSharded (device := device) modelDir scaleName #[outDim / 128, inDim / 128]
      return match scaleInvOpt with
        | some ⟨_, s⟩ => dequantizeFP8 w s
        | none => w.toFloat32
  throw <| IO.userError s!"Failed to load linear weight (sharded): {baseName}"

private def loadLinearWeight (path : String) (baseName : String) (outDim inDim : Nat)
    (device : Device := .CPU) : IO (Tensor { device := device, shape := #[outDim, inDim], dtype := .Float32 }) := do
  let bases := baseNameCandidates baseName
  for b in bases do
    let wName := s!"{b}.weight"
    if let some ⟨_, w⟩ ← tryLoadTensor (device := device) path wName #[outDim, inDim] then
      let scaleName := s!"{b}.weight_scale_inv"
      let scaleInvOpt ← tryLoadTensor (device := device) path scaleName #[outDim / 128, inDim / 128]
      return match scaleInvOpt with
        | some ⟨_, s⟩ => dequantizeFP8 w s
        | none => w.toFloat32
  throw <| IO.userError s!"Failed to load linear weight: {baseName}"

private def loadAttentionSharded (modelDir : String) (layerIdx : Nat)
    (hidden_size num_heads num_kv_heads head_dim : Nat)
    (device : Device := .CPU) : IO (qwen.QwenAttention (device := device) hidden_size num_heads num_kv_heads head_dim) := do
  let p := s!"model.layers.{layerIdx}.self_attn"
  let qProj ← loadLinearWeightSharded (device := device) modelDir s!"{p}.q_proj" (num_heads * head_dim) hidden_size
  let kProj ← loadLinearWeightSharded (device := device) modelDir s!"{p}.k_proj" (num_kv_heads * head_dim) hidden_size
  let vProj ← loadLinearWeightSharded (device := device) modelDir s!"{p}.v_proj" (num_kv_heads * head_dim) hidden_size
  let oProj ← loadLinearWeightSharded (device := device) modelDir s!"{p}.o_proj" hidden_size (num_heads * head_dim)
  let qNorm ← tryLoadTensorShardedCandidates (device := device) modelDir (tensorNameCandidates s!"{p}.q_norm.weight") #[head_dim]
  let kNorm ← tryLoadTensorShardedCandidates (device := device) modelDir (tensorNameCandidates s!"{p}.k_norm.weight") #[head_dim]
  pure {
    q_proj := reqGradFalse qProj
    k_proj := reqGradFalse kProj
    v_proj := reqGradFalse vProj
    o_proj := reqGradFalse oProj
    q_norm := qNorm.map (fun ⟨_, w⟩ => reqGradFalse w.toFloat32)
    k_norm := kNorm.map (fun ⟨_, w⟩ => reqGradFalse w.toFloat32)
  }

private def loadAttention (path : String) (layerIdx : Nat)
    (hidden_size num_heads num_kv_heads head_dim : Nat)
    (device : Device := .CPU) : IO (qwen.QwenAttention (device := device) hidden_size num_heads num_kv_heads head_dim) := do
  let p := s!"model.layers.{layerIdx}.self_attn"
  let qProj ← loadLinearWeight (device := device) path s!"{p}.q_proj" (num_heads * head_dim) hidden_size
  let kProj ← loadLinearWeight (device := device) path s!"{p}.k_proj" (num_kv_heads * head_dim) hidden_size
  let vProj ← loadLinearWeight (device := device) path s!"{p}.v_proj" (num_kv_heads * head_dim) hidden_size
  let oProj ← loadLinearWeight (device := device) path s!"{p}.o_proj" hidden_size (num_heads * head_dim)
  let qNorm ← tryLoadTensorCandidates (device := device) path (tensorNameCandidates s!"{p}.q_norm.weight") #[head_dim]
  let kNorm ← tryLoadTensorCandidates (device := device) path (tensorNameCandidates s!"{p}.k_norm.weight") #[head_dim]
  pure {
    q_proj := reqGradFalse qProj
    k_proj := reqGradFalse kProj
    v_proj := reqGradFalse vProj
    o_proj := reqGradFalse oProj
    q_norm := qNorm.map (fun ⟨_, w⟩ => reqGradFalse w.toFloat32)
    k_norm := kNorm.map (fun ⟨_, w⟩ => reqGradFalse w.toFloat32)
  }

private def loadMLPSharded (modelDir : String) (layerIdx : Nat)
    (hidden_size intermediate_size : Nat)
    (device : Device := .CPU) : IO (qwen.QwenMLP (device := device) hidden_size intermediate_size) := do
  let p := s!"model.layers.{layerIdx}.mlp"
  let gate ← loadLinearWeightSharded (device := device) modelDir s!"{p}.gate_proj" intermediate_size hidden_size
  let up ← loadLinearWeightSharded (device := device) modelDir s!"{p}.up_proj" intermediate_size hidden_size
  let down ← loadLinearWeightSharded (device := device) modelDir s!"{p}.down_proj" hidden_size intermediate_size
  pure {
    gate_proj := reqGradFalse gate
    up_proj := reqGradFalse up
    down_proj := reqGradFalse down
  }

private def loadMLP (path : String) (layerIdx : Nat)
    (hidden_size intermediate_size : Nat)
    (device : Device := .CPU) : IO (qwen.QwenMLP (device := device) hidden_size intermediate_size) := do
  let p := s!"model.layers.{layerIdx}.mlp"
  let gate ← loadLinearWeight (device := device) path s!"{p}.gate_proj" intermediate_size hidden_size
  let up ← loadLinearWeight (device := device) path s!"{p}.up_proj" intermediate_size hidden_size
  let down ← loadLinearWeight (device := device) path s!"{p}.down_proj" hidden_size intermediate_size
  pure {
    gate_proj := reqGradFalse gate
    up_proj := reqGradFalse up
    down_proj := reqGradFalse down
  }

private def loadRMSNormSharded (modelDir : String) (name : String) (dim : Nat)
    (device : Device := .CPU) : IO (RMSNorm (device := device) dim) := do
  let ⟨_, w⟩ ← loadTensorShardedCandidates (device := device) modelDir (tensorNameCandidates s!"{name}.weight") #[dim]
  pure { weight := reqGradFalse w.toFloat32, eps := ⟨1e-6⟩ }

private def loadRMSNorm (path : String) (name : String) (dim : Nat)
    (device : Device := .CPU) : IO (RMSNorm (device := device) dim) := do
  let ⟨_, w⟩ ← loadTensorCandidates (device := device) path (tensorNameCandidates s!"{name}.weight") #[dim]
  pure { weight := reqGradFalse w.toFloat32, eps := ⟨1e-6⟩ }

private def loadLayerSharded (modelDir : String) (layerIdx : Nat) (cfg : Config)
    (device : Device := .CPU) : IO (qwen.QwenLayer (device := device) cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim cfg.intermediate_size) := do
  let p := s!"model.layers.{layerIdx}"
  let inputNorm ← loadRMSNormSharded (device := device) modelDir s!"{p}.input_layernorm" cfg.hidden_size
  let selfAttn ← loadAttentionSharded (device := device) modelDir layerIdx cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim
  let postNorm ← loadRMSNormSharded (device := device) modelDir s!"{p}.post_attention_layernorm" cfg.hidden_size
  let mlp ← loadMLPSharded (device := device) modelDir layerIdx cfg.hidden_size cfg.intermediate_size
  pure {
    input_layernorm := inputNorm
    self_attn := selfAttn
    post_attention_layernorm := postNorm
    mlp := mlp
  }

private def loadLayer (path : String) (layerIdx : Nat) (cfg : Config)
    (device : Device := .CPU) : IO (qwen.QwenLayer (device := device) cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim cfg.intermediate_size) := do
  let p := s!"model.layers.{layerIdx}"
  let inputNorm ← loadRMSNorm (device := device) path s!"{p}.input_layernorm" cfg.hidden_size
  let selfAttn ← loadAttention (device := device) path layerIdx cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim
  let postNorm ← loadRMSNorm (device := device) path s!"{p}.post_attention_layernorm" cfg.hidden_size
  let mlp ← loadMLP (device := device) path layerIdx cfg.hidden_size cfg.intermediate_size
  pure {
    input_layernorm := inputNorm
    self_attn := selfAttn
    post_attention_layernorm := postNorm
    mlp := mlp
  }

private def loadThinkerModelSharded (modelDir : String) (cfg : Config) (log : Handlers := {})
    (device : Device := .CPU) : IO (qwen.Qwen3Model (device := device) cfg) := do
  log.onInfo s!"Loading Qwen2.5-Omni thinker model from {modelDir}..."
  let ⟨_, embed⟩ ← loadTensorShardedCandidates (device := device) modelDir (tensorNameCandidates "model.embed_tokens.weight") #[cfg.vocab_size, cfg.hidden_size]
  let embed := reqGradFalse embed.toFloat32

  let mut layers : Array (qwen.QwenLayer (device := device) cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim cfg.intermediate_size) := #[]
  for i in [:cfg.num_hidden_layers] do
    layers := layers.push (← loadLayerSharded (device := device) modelDir i cfg)
    if (i + 1) % 8 == 0 || i + 1 == cfg.num_hidden_layers then
      log.onInfo s!"  Loaded {i + 1}/{cfg.num_hidden_layers} layers"

  let norm ← loadRMSNormSharded (device := device) modelDir "model.norm" cfg.hidden_size
  log.onInfo "Loaded Qwen2.5-Omni thinker weights."
  pure { embed_tokens := embed, layers := layers, norm := norm }

private def loadThinkerModel (path : String) (cfg : Config) (log : Handlers := {})
    (device : Device := .CPU) : IO (qwen.Qwen3Model (device := device) cfg) := do
  log.onInfo s!"Loading Qwen2.5-Omni thinker model from {path}..."
  let ⟨_, embed⟩ ← loadTensorCandidates (device := device) path (tensorNameCandidates "model.embed_tokens.weight") #[cfg.vocab_size, cfg.hidden_size]
  let embed := reqGradFalse embed.toFloat32

  let mut layers : Array (qwen.QwenLayer (device := device) cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim cfg.intermediate_size) := #[]
  for i in [:cfg.num_hidden_layers] do
    layers := layers.push (← loadLayer (device := device) path i cfg)
    if (i + 1) % 8 == 0 || i + 1 == cfg.num_hidden_layers then
      log.onInfo s!"  Loaded {i + 1}/{cfg.num_hidden_layers} layers"

  let norm ← loadRMSNorm (device := device) path "model.norm" cfg.hidden_size
  log.onInfo "Loaded Qwen2.5-Omni thinker weights."
  pure { embed_tokens := embed, layers := layers, norm := norm }

/-- Qwen2.5-Omni thinker text model (causal LM). -/
abbrev Qwen25OmniForCausalLM := qwen3.Qwen3ForCausalLM (device := device)

namespace Qwen25OmniForCausalLM

private def lmHeadCandidates : Array String := #[
  "thinker.lm_head.weight",
  "lm_head.weight",
  "model.lm_head.weight"
]

/-- Load Qwen2.5-Omni thinker text model from sharded SafeTensors directory. -/
def loadSharded (modelDir : String) (cfg : Config := Config.qwen25omni_3B)
    (log : Handlers := {})
    (device : Device := .CPU) : IO (Qwen25OmniForCausalLM (device := device) cfg) := do
  let model ← loadThinkerModelSharded (device := device) modelDir cfg log
  let lmHeadOpt ← tryLoadTensorShardedCandidates (device := device) modelDir lmHeadCandidates #[cfg.vocab_size, cfg.hidden_size]
  let (lmHead, tieWordEmbeddings) ←
    match lmHeadOpt with
    | some ⟨_, w⟩ => pure (reqGradFalse w.toFloat32, false)
    | none => do
      log.onInfo "  lm_head.weight not found; using tied embeddings."
      pure (reqGradFalse model.embed_tokens, true)
  pure { model := model, lmHead := lmHead, tieWordEmbeddings := tieWordEmbeddings }

/-- Load Qwen2.5-Omni thinker text model from a single SafeTensors file. -/
def load (path : String) (cfg : Config := Config.qwen25omni_3B)
    (log : Handlers := {})
    (device : Device := .CPU) : IO (Qwen25OmniForCausalLM (device := device) cfg) := do
  let model ← loadThinkerModel (device := device) path cfg log
  let lmHeadOpt ← tryLoadTensorCandidates (device := device) path lmHeadCandidates #[cfg.vocab_size, cfg.hidden_size]
  let (lmHead, tieWordEmbeddings) ←
    match lmHeadOpt with
    | some ⟨_, w⟩ => pure (reqGradFalse w.toFloat32, false)
    | none => do
      log.onInfo "  lm_head.weight not found; using tied embeddings."
      pure (reqGradFalse model.embed_tokens, true)
  pure { model := model, lmHead := lmHead, tieWordEmbeddings := tieWordEmbeddings }

end Qwen25OmniForCausalLM

end torch.qwen25omni
