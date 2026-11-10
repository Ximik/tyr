/-
  Tyr/Model/Qwen35/Weights.lean

  Pretrained weight loading for standalone Qwen3.5 text causal-LM.
  Supports dense and MoE checkpoints from HuggingFace SafeTensors
  (single-file and sharded).
-/
import Tyr.Typed.Neural
import Tyr.Log
import Tyr.SafeTensors.Source
import Tyr.Model.Utils
import Tyr.Model.Qwen35.Model

namespace torch.qwen35

variable {device : Device}

open torch.Log
open torch.Model (reqGradFalse dequantizeFP8 dequantizeFP8Experts checkFP8Blocks)
open torch.safetensors (pushUnique Source)

private def freezeStored {shape : Shape} (tensor : SomeDTypeTensor shape device) :
    SomeDTypeTensor shape device := ⟨tensor.1, tensor.2.detach⟩

/-- Equality admission only: never changes the tensor's dtype. -/
private def requireDType {shape : Shape} (dtype : DType)
    (tensor : SomeDTypeTensor shape device) : IO (Tensor ⟨shape, dtype, device⟩) :=
  IO.ofExcept (tensor.atDType? dtype)

private def finalizeDequantizedForDevice {s : Shape} (device : Device)
    (t : Tensor ⟨s, .Float32, device⟩) : SomeDTypeTensor s device :=
  match device.kind with
  | .CPU => ⟨.Float32, t⟩
  | .MPS => ⟨.BFloat16, t.toBFloat16⟩
  | .CUDA _ => ⟨.BFloat16, t.toBFloat16⟩

private def nameCandidates (name : String) : Array String :=
  let out : Array String := #[]
  let out := pushUnique out name
  let out := pushUnique out s!"language_model.{name}"
  let out := pushUnique out s!"model.language_model.{name}"
  let out :=
    if name.startsWith "model." then
      let suffix := name.drop 6
      let out := pushUnique out s!"model.language_model.{suffix}"
      pushUnique out s!"language_model.{suffix}"
    else
      out
  out

private def loadLinearWeight
    (src : Source)
    (baseName : String)
    (outDim inDim : Nat)
    (device : Device := Device.CPU)
    : IO (SomeDTypeTensor (#[outDim, inDim]) device) := do
  let bases := nameCandidates baseName
  for b in bases do
    let wName := s!"{b}.weight"
    if let some w ← src.load? wName #[outDim, inDim] device then
      let scaleName := s!"{b}.weight_scale_inv"
      let scaleInvOpt ← src.load? scaleName #[outDim / 128, inDim / 128] device
      match scaleInvOpt with
      | some s =>
        let ⟨hBlocks⟩ ← checkFP8Blocks scaleName outDim inDim
        return finalizeDequantizedForDevice device (dequantizeFP8 w.2 s.2 hBlocks)
      | none => return w
  throw <| IO.userError s!"Failed to load linear weight: {baseName}"

private def loadMoeExpertWeight
    (src : Source)
    (baseName : String)
    (experts outDim inDim : Nat)
    (device : Device := Device.CPU)
    : IO (SomeDTypeTensor (#[experts, outDim, inDim]) device) := do
  let bases := nameCandidates baseName
  for b in bases do
    -- BF16-style checkpoints often store experts tensors directly (no ".weight").
    if let some w ← src.load? b #[experts, outDim, inDim] device then
      let scaleName := s!"{b}.weight_scale_inv"
      let scaleInvOpt ← src.load? scaleName #[experts, outDim / 128, inDim / 128] device
      match scaleInvOpt with
      | some s =>
        let ⟨hBlocks⟩ ← checkFP8Blocks scaleName outDim inDim
        return finalizeDequantizedForDevice device (dequantizeFP8Experts w.2 s.2 hBlocks)
      | none => return w
    -- FP8 checkpoints store them as "<base>.weight" + optional weight_scale_inv.
    let wName := s!"{b}.weight"
    if let some w ← src.load? wName #[experts, outDim, inDim] device then
      let scaleName := s!"{b}.weight_scale_inv"
      let scaleInvOpt ← src.load? scaleName #[experts, outDim / 128, inDim / 128] device
      match scaleInvOpt with
      | some s =>
        let ⟨hBlocks⟩ ← checkFP8Blocks scaleName outDim inDim
        return finalizeDequantizedForDevice device (dequantizeFP8Experts w.2 s.2 hBlocks)
      | none => return w
  throw <| IO.userError s!"Failed to load MoE expert weight: {baseName}"

private def loadRMSNorm (src : Source) (baseName : String) (dim : Nat)
    (device : Device := Device.CPU)
    : IO (Qwen35RMSNorm (device := device) dim) := do
  let w ← src.load (← src.resolve (nameCandidates s!"{baseName}.weight")) #[dim] device
  pure (Qwen35RMSNorm.fromCheckpointWeight (freezeStored w) 1e-6)

private def loadRMSNormGated (src : Source) (baseName : String) (dim : Nat)
    (device : Device := Device.CPU)
    : IO (Qwen35RMSNormGated (device := device) dim) := do
  let w ← src.load (← src.resolve (nameCandidates s!"{baseName}.weight")) #[dim] device
  pure { weight := freezeStored w, eps := 1e-6 }

private def loadAttention (src : Source) (cfg : Config) (layerIdx : Nat)
    (device : Device := Device.CPU) (dtype : DType)
    : IO (Qwen35Attention (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.layers.{layerIdx}.self_attn"
  let qProj ← loadLinearWeight src s!"{p}.q_proj" (cfg.num_attention_heads * cfg.head_dim * 2) cfg.hidden_size device
  let kProj ← loadLinearWeight src s!"{p}.k_proj" (cfg.num_key_value_heads * cfg.head_dim) cfg.hidden_size device
  let vProj ← loadLinearWeight src s!"{p}.v_proj" (cfg.num_key_value_heads * cfg.head_dim) cfg.hidden_size device
  let oProj ← loadLinearWeight src s!"{p}.o_proj" cfg.hidden_size (cfg.num_attention_heads * cfg.head_dim) device

  let qNorm ← loadRMSNorm src s!"{p}.q_norm" cfg.head_dim device
  let kNorm ← loadRMSNorm src s!"{p}.k_norm" cfg.head_dim device

  pure {
    q_proj := reqGradFalse (← requireDType dtype qProj)
    k_proj := reqGradFalse (← requireDType dtype kProj)
    v_proj := reqGradFalse (← requireDType dtype vProj)
    o_proj := reqGradFalse (← requireDType dtype oProj)
    q_norm := qNorm
    k_norm := kNorm
  }

private def loadLinearMixer (src : Source) (cfg : Config) (layerIdx : Nat)
    (device : Device := Device.CPU) (dtype : DType)
    : IO (Qwen35GatedDeltaNet (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.layers.{layerIdx}.linear_attn"

  let inQKV ← loadLinearWeight src s!"{p}.in_proj_qkv" (Config.linearConvDim cfg) cfg.hidden_size device
  let inZ ← loadLinearWeight src s!"{p}.in_proj_z" (Config.linearValueDim cfg) cfg.hidden_size device
  let inB ← loadLinearWeight src s!"{p}.in_proj_b" cfg.linear_num_value_heads cfg.hidden_size device
  let inA ← loadLinearWeight src s!"{p}.in_proj_a" cfg.linear_num_value_heads cfg.hidden_size device

  let convW ← src.load (← src.resolve (nameCandidates s!"{p}.conv1d.weight")) #[Config.linearConvDim cfg, 1, cfg.linear_conv_kernel_dim] device
  let convBOpt ← ((src.find? (nameCandidates s!"{p}.conv1d.bias")).mapM (src.load · #[Config.linearConvDim cfg] device))
  let convB ←
    match convBOpt with
    | some b => pure b
    | none => do
      let support ← convW.1.acquireConstantSupport device
      let _ := support.down
      let bias := Tensor.zerosFor { shape := #[Config.linearConvDim cfg], dtype := convW.1, device }
      pure ⟨convW.1, bias⟩

  let dtBias ← src.load (← src.resolve (nameCandidates s!"{p}.dt_bias")) #[cfg.linear_num_value_heads] device
  let aLog ← src.load (← src.resolve (nameCandidates s!"{p}.A_log")) #[cfg.linear_num_value_heads] device

  let norm ← loadRMSNormGated src s!"{p}.norm" cfg.linear_value_head_dim device
  let outProj ← loadLinearWeight src s!"{p}.out_proj" cfg.hidden_size (Config.linearValueDim cfg) device

  pure {
    in_proj_qkv := reqGradFalse (← requireDType dtype inQKV)
    in_proj_z := reqGradFalse (← requireDType dtype inZ)
    in_proj_b := reqGradFalse (← requireDType dtype inB)
    in_proj_a := reqGradFalse (← requireDType dtype inA)
    conv1d_weight := reqGradFalse (← requireDType dtype convW)
    conv1d_bias := reqGradFalse (← requireDType dtype convB)
    dt_bias := reqGradFalse dtBias.2.toFloat32
    a_log := reqGradFalse aLog.2.toFloat32
    norm := norm
    out_proj := reqGradFalse (← requireDType dtype outProj)
  }

private def loadDenseMLP (src : Source) (cfg : Config) (layerIdx : Nat)
    (device : Device := Device.CPU) (dtype : DType)
    : IO (Qwen35MLP (dtype := dtype) (device := device) cfg.hidden_size cfg.intermediate_size) := do
  let p := s!"model.layers.{layerIdx}.mlp"
  let gate ← loadLinearWeight src s!"{p}.gate_proj" cfg.intermediate_size cfg.hidden_size device
  let up ← loadLinearWeight src s!"{p}.up_proj" cfg.intermediate_size cfg.hidden_size device
  let down ← loadLinearWeight src s!"{p}.down_proj" cfg.hidden_size cfg.intermediate_size device
  pure {
    gate_proj := reqGradFalse (← requireDType dtype gate)
    up_proj := reqGradFalse (← requireDType dtype up)
    down_proj := reqGradFalse (← requireDType dtype down)
  }

private def loadSparseMoe (src : Source) (cfg : Config) (layerIdx : Nat)
    (device : Device := Device.CPU) (dtype : DType)
    : IO (Qwen35SparseMoeBlock (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.layers.{layerIdx}.mlp"

  let routerW ← src.load (← src.resolve (nameCandidates s!"{p}.gate.weight")) #[cfg.num_experts, cfg.hidden_size] device
  let gu ← loadMoeExpertWeight
    src
    s!"{p}.experts.gate_up_proj"
    cfg.num_experts
    (2 * cfg.moe_intermediate_size)
    cfg.hidden_size
    device
  let down ← loadMoeExpertWeight
    src
    s!"{p}.experts.down_proj"
    cfg.num_experts
    cfg.hidden_size
    cfg.moe_intermediate_size
    device

  let sharedGateProj ← loadLinearWeight src s!"{p}.shared_expert.gate_proj" cfg.shared_expert_intermediate_size cfg.hidden_size device
  let sharedUpProj ← loadLinearWeight src s!"{p}.shared_expert.up_proj" cfg.shared_expert_intermediate_size cfg.hidden_size device
  let sharedDownProj ← loadLinearWeight src s!"{p}.shared_expert.down_proj" cfg.hidden_size cfg.shared_expert_intermediate_size device
  let sharedExpertGate ← loadLinearWeight src s!"{p}.shared_expert_gate" 1 cfg.hidden_size device

  let router : Qwen35TopKRouter (dtype := dtype) (device := device) cfg := { weight := reqGradFalse (← requireDType dtype routerW) }
  let experts : Qwen35MoeExperts (dtype := dtype) (device := device) cfg := {
    gate_up_proj := reqGradFalse (← requireDType dtype gu)
    down_proj := reqGradFalse (← requireDType dtype down)
  }
  let sharedExpert : Qwen35MLP (dtype := dtype) (device := device) cfg.hidden_size cfg.shared_expert_intermediate_size := {
    gate_proj := reqGradFalse (← requireDType dtype sharedGateProj)
    up_proj := reqGradFalse (← requireDType dtype sharedUpProj)
    down_proj := reqGradFalse (← requireDType dtype sharedDownProj)
  }

  pure {
    router := router
    experts := experts
    shared_expert := sharedExpert
    shared_expert_gate := reqGradFalse (← requireDType dtype sharedExpertGate)
  }

private def loadLayer (src : Source) (cfg : Config) (layerIdx : Nat)
    (device : Device := Device.CPU) (dtype : DType)
    : IO (Qwen35Layer (device := device) cfg dtype layerIdx) := do
  let p := s!"model.layers.{layerIdx}"

  let inputNorm ← loadRMSNorm src s!"{p}.input_layernorm" cfg.hidden_size device
  let postNorm ← loadRMSNorm src s!"{p}.post_attention_layernorm" cfg.hidden_size device

  let mixer ← Select.build (cfg.isFullAttentionLayer layerIdx)
    (loadLinearMixer src cfg layerIdx device dtype)
    (loadAttention src cfg layerIdx device dtype)

  let ffn ← Select.build cfg.isMoE
    (loadDenseMLP src cfg layerIdx device dtype)
    (loadSparseMoe src cfg layerIdx device dtype)

  pure {
    input_layernorm := inputNorm
    mixer := mixer
    post_attention_layernorm := postNorm
    ffn := ffn
  }

namespace Qwen35ForCausalLM

/-- Load Qwen3.5 model from a SafeTensors file or a directory of shards. -/
def load (path : String) (cfg : Config := Config.qwen35_9B)
    (device : Device := Device.CPU)
    (log : Handlers := {})
    : IO (Σ dtype : RealFloatingDType, Qwen35ForCausalLM (device := device) cfg dtype.val) := do
  log.onInfo s!"Loading Qwen35ForCausalLM (device := device) from {path}..."
  let src ← Source.open path

  let embedTokens ← src.load (← src.resolve (nameCandidates "model.embed_tokens.weight")) #[cfg.vocab_size, cfg.hidden_size] device
  let dtype := embedTokens.1
  let floating ← IO.ofExcept dtype.realFloating?
  let _ := floating.down

  let layers ← IndexedList.ofFnM (α := Qwen35Layer (device := device) cfg dtype) (fun i => do
    let layer ← loadLayer src cfg i device dtype
    if (i + 1) % 8 == 0 || i + 1 == cfg.num_hidden_layers then
      log.onInfo s!"  loaded layers {i + 1}/{cfg.num_hidden_layers}"
    pure layer) 0 cfg.num_hidden_layers

  let norm ← loadRMSNorm src "model.norm" cfg.hidden_size device

  let model : Qwen35Model (dtype := dtype) (device := device) cfg := {
    embed_tokens := reqGradFalse (← requireDType dtype embedTokens)
    layers := layers
    norm := norm
  }

  let lmHeadOpt ← (src.find? #["lm_head.weight", "language_model.lm_head.weight"]).mapM
    (src.load · #[cfg.vocab_size, cfg.hidden_size] device)

  let (lmHead, tieWordEmbeddings) ←
    match lmHeadOpt with
    | some w => pure (reqGradFalse (← requireDType dtype w), false)
    | none => do
      log.onInfo "  lm_head.weight not found; tying to embeddings."
      pure (reqGradFalse model.embed_tokens, true)

  log.onInfo "Loaded Qwen35ForCausalLM (device := device) weights."
  pure ⟨RealFloatingDType.ofDType dtype, { model := model, lmHead := lmHead, tieWordEmbeddings := tieWordEmbeddings }⟩

end Qwen35ForCausalLM

end torch.qwen35
