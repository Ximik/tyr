/-
  Tyr/Model/Qwen25Omni/Weights.lean

  Thinker text weight loading for Qwen2.5-Omni checkpoints.
  Supports both single-file and sharded SafeTensors layouts.
-/
import Tyr.Typed
import Tyr.Log
import Tyr.SafeTensors.Source
import Tyr.Model.Utils
import Tyr.Model.Qwen25Omni.Config
import Tyr.Model.Qwen3.Model
import Tyr.Model.Qwen.Model

namespace torch.qwen25omni

variable {device : Device}
open torch.Log
open torch.Model (reqGradFalse dequantizeFP8 checkFP8Blocks)
open torch.safetensors (pushUnique Source)

private def nameCandidates (name : String) : Array String :=
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

private def loadLinearWeight (src : Source) (baseName : String) (outDim inDim : Nat)
    (device : Device := .CPU) : IO (Tensor { device := device, shape := #[outDim, inDim], dtype := .Float32 }) := do
  let some b := (nameCandidates baseName).find? (fun b => src.contains s!"{b}.weight")
    | throw <| IO.userError s!"Failed to load linear weight: {baseName}"
  let ⟨_, w⟩ ← src.load s!"{b}.weight" #[outDim, inDim] device
  let scaleName := s!"{b}.weight_scale_inv"
  match ← src.load? scaleName #[outDim / 128, inDim / 128] device with
  | some ⟨_, s⟩ =>
    let ⟨hBlocks⟩ ← checkFP8Blocks scaleName outDim inDim
    return dequantizeFP8 w s hBlocks
  | none => return w.toFloat32

private def loadAttention (src : Source) (layerIdx : Nat)
    (hidden_size num_heads num_kv_heads head_dim : Nat)
    (device : Device := .CPU) : IO (qwen.QwenAttention (device := device) hidden_size num_heads num_kv_heads head_dim) := do
  let p := s!"model.layers.{layerIdx}.self_attn"
  let qProj ← loadLinearWeight (device := device) src s!"{p}.q_proj" (num_heads * head_dim) hidden_size
  let kProj ← loadLinearWeight (device := device) src s!"{p}.k_proj" (num_kv_heads * head_dim) hidden_size
  let vProj ← loadLinearWeight (device := device) src s!"{p}.v_proj" (num_kv_heads * head_dim) hidden_size
  let oProj ← loadLinearWeight (device := device) src s!"{p}.o_proj" hidden_size (num_heads * head_dim)
  let qNorm ← (src.find? (nameCandidates s!"{p}.q_norm.weight")).mapM (src.load · #[head_dim] device)
  let kNorm ← (src.find? (nameCandidates s!"{p}.k_norm.weight")).mapM (src.load · #[head_dim] device)
  pure {
    q_proj := reqGradFalse qProj
    k_proj := reqGradFalse kProj
    v_proj := reqGradFalse vProj
    o_proj := reqGradFalse oProj
    q_norm := qNorm.map (fun ⟨_, w⟩ => reqGradFalse w.toFloat32)
    k_norm := kNorm.map (fun ⟨_, w⟩ => reqGradFalse w.toFloat32)
  }

private def loadMLP (src : Source) (layerIdx : Nat)
    (hidden_size intermediate_size : Nat)
    (device : Device := .CPU) : IO (qwen.QwenMLP (device := device) hidden_size intermediate_size) := do
  let p := s!"model.layers.{layerIdx}.mlp"
  let gate ← loadLinearWeight (device := device) src s!"{p}.gate_proj" intermediate_size hidden_size
  let up ← loadLinearWeight (device := device) src s!"{p}.up_proj" intermediate_size hidden_size
  let down ← loadLinearWeight (device := device) src s!"{p}.down_proj" hidden_size intermediate_size
  pure {
    gate_proj := reqGradFalse gate
    up_proj := reqGradFalse up
    down_proj := reqGradFalse down
  }

private def loadRMSNorm (src : Source) (name : String) (dim : Nat)
    (device : Device := .CPU) : IO (RMSNorm (device := device) dim) := do
  let ⟨_, w⟩ ← src.load (← src.resolve (nameCandidates s!"{name}.weight")) #[dim] device
  pure { weight := reqGradFalse w.toFloat32, eps := ⟨1e-6⟩ }

private def loadLayer (src : Source) (layerIdx : Nat) (cfg : Config)
    (device : Device := .CPU) : IO (qwen.QwenLayer (device := device) cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim cfg.intermediate_size) := do
  let p := s!"model.layers.{layerIdx}"
  let inputNorm ← loadRMSNorm (device := device) src s!"{p}.input_layernorm" cfg.hidden_size
  let selfAttn ← loadAttention (device := device) src layerIdx cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim
  let postNorm ← loadRMSNorm (device := device) src s!"{p}.post_attention_layernorm" cfg.hidden_size
  let mlp ← loadMLP (device := device) src layerIdx cfg.hidden_size cfg.intermediate_size
  pure {
    input_layernorm := inputNorm
    self_attn := selfAttn
    post_attention_layernorm := postNorm
    mlp := mlp
  }

private def loadThinkerModel (src : Source) (cfg : Config) (log : Handlers := {})
    (device : Device := .CPU) : IO (qwen.Qwen3Model (device := device) cfg) := do
  log.onInfo s!"Loading Qwen2.5-Omni thinker model from {src.path}..."
  let ⟨_, embed⟩ ← src.load (← src.resolve (nameCandidates "model.embed_tokens.weight")) #[cfg.vocab_size, cfg.hidden_size] device
  let embed := reqGradFalse embed.toFloat32

  let mut layers : Array (qwen.QwenLayer (device := device) cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim cfg.intermediate_size) := #[]
  for i in [:cfg.num_hidden_layers] do
    layers := layers.push (← loadLayer (device := device) src i cfg)
    if (i + 1) % 8 == 0 || i + 1 == cfg.num_hidden_layers then
      log.onInfo s!"  Loaded {i + 1}/{cfg.num_hidden_layers} layers"

  let norm ← loadRMSNorm (device := device) src "model.norm" cfg.hidden_size
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

/-- Load Qwen2.5-Omni thinker text model from a SafeTensors file or a
    directory of shards. -/
def load (path : String) (cfg : Config := Config.qwen25omni_3B)
    (log : Handlers := {})
    (device : Device := .CPU) : IO (Qwen25OmniForCausalLM (device := device) cfg) := do
  let src ← Source.open path
  let model ← loadThinkerModel (device := device) src cfg log
  let lmHeadOpt ← (src.find? lmHeadCandidates).mapM (src.load · #[cfg.vocab_size, cfg.hidden_size] device)
  let (lmHead, tieWordEmbeddings) ←
    match lmHeadOpt with
    | some ⟨_, w⟩ => pure (reqGradFalse w.toFloat32, false)
    | none => do
      log.onInfo "  lm_head.weight not found; using tied embeddings."
      pure (reqGradFalse model.embed_tokens, true)
  pure { model := model, lmHead := lmHead, tieWordEmbeddings := tieWordEmbeddings }

end Qwen25OmniForCausalLM

end torch.qwen25omni
