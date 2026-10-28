/-
  Tyr/Model/Qwen35/Weights.lean

  Pretrained weight loading for standalone Qwen3.5 text causal-LM.
  Supports dense and MoE checkpoints from HuggingFace SafeTensors
  (single-file and sharded).
-/
import Tyr.Typed.Neural
import Tyr.Log
import Tyr.SafeTensors.Load
import Tyr.Model.Utils
import Tyr.Model.Qwen35.Model

namespace torch.qwen35

variable {device : Device}

open torch.Log
open torch.Model (reqGradFalse dequantizeFP8 dequantizeFP8Experts)
open torch.safetensors (pushUnique tryLoadOptionalTensor tryLoadOptionalTensorSharded
  loadTensorCandidates loadTensorShardedCandidates
  tryLoadOptionalTensorCandidates tryLoadOptionalTensorShardedCandidates)

private def freezeStored {shape : Shape} (tensor : SomeDTypeTensor shape device) :
    SomeDTypeTensor shape device := ⟨tensor.1, tensor.2.detach⟩

/-- Equality admission only: loading never changes a checkpoint tensor's dtype. -/
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

private def loadLinearWeightSharded
    (modelDir : String)
    (baseName : String)
    (outDim inDim : Nat)
    (device : Device := Device.CPU)
    : IO (SomeDTypeTensor (#[outDim, inDim]) device) := do
  let bases := nameCandidates baseName
  for b in bases do
    let wName := s!"{b}.weight"
    if let some w ← tryLoadOptionalTensorSharded modelDir wName #[outDim, inDim] device then
      let scaleName := s!"{b}.weight_scale_inv"
      let scaleInvOpt ← tryLoadOptionalTensorSharded modelDir scaleName #[outDim / 128, inDim / 128] device
      return match scaleInvOpt with
        | some s => finalizeDequantizedForDevice device (dequantizeFP8 w.2 s.2)
        | none => w
  throw <| IO.userError s!"Failed to load linear weight (sharded): {baseName}"

private def loadLinearWeight
    (path : String)
    (baseName : String)
    (outDim inDim : Nat)
    (device : Device := Device.CPU)
    : IO (SomeDTypeTensor (#[outDim, inDim]) device) := do
  let bases := nameCandidates baseName
  for b in bases do
    let wName := s!"{b}.weight"
    if let some w ← tryLoadOptionalTensor path wName #[outDim, inDim] device then
      let scaleName := s!"{b}.weight_scale_inv"
      let scaleInvOpt ← tryLoadOptionalTensor path scaleName #[outDim / 128, inDim / 128] device
      return match scaleInvOpt with
        | some s => finalizeDequantizedForDevice device (dequantizeFP8 w.2 s.2)
        | none => w
  throw <| IO.userError s!"Failed to load linear weight: {baseName}"

private def loadMoeExpertWeightSharded
    (modelDir : String)
    (baseName : String)
    (experts outDim inDim : Nat)
    (device : Device := Device.CPU)
    : IO (SomeDTypeTensor (#[experts, outDim, inDim]) device) := do
  let bases := nameCandidates baseName
  for b in bases do
    -- BF16-style checkpoints often store experts tensors directly (no ".weight").
    if let some w ← tryLoadOptionalTensorSharded modelDir b #[experts, outDim, inDim] device then
      let scaleName := s!"{b}.weight_scale_inv"
      let scaleInvOpt ← tryLoadOptionalTensorSharded modelDir scaleName #[experts, outDim / 128, inDim / 128] device
      return match scaleInvOpt with
        | some s => finalizeDequantizedForDevice device (dequantizeFP8Experts w.2 s.2)
        | none => w
    -- FP8 checkpoints store them as "<base>.weight" + optional weight_scale_inv.
    let wName := s!"{b}.weight"
    if let some w ← tryLoadOptionalTensorSharded modelDir wName #[experts, outDim, inDim] device then
      let scaleName := s!"{b}.weight_scale_inv"
      let scaleInvOpt ← tryLoadOptionalTensorSharded modelDir scaleName #[experts, outDim / 128, inDim / 128] device
      return match scaleInvOpt with
        | some s => finalizeDequantizedForDevice device (dequantizeFP8Experts w.2 s.2)
        | none => w
  throw <| IO.userError s!"Failed to load MoE expert weight (sharded): {baseName}"

private def loadMoeExpertWeight
    (path : String)
    (baseName : String)
    (experts outDim inDim : Nat)
    (device : Device := Device.CPU)
    : IO (SomeDTypeTensor (#[experts, outDim, inDim]) device) := do
  let bases := nameCandidates baseName
  for b in bases do
    if let some w ← tryLoadOptionalTensor path b #[experts, outDim, inDim] device then
      let scaleName := s!"{b}.weight_scale_inv"
      let scaleInvOpt ← tryLoadOptionalTensor path scaleName #[experts, outDim / 128, inDim / 128] device
      return match scaleInvOpt with
        | some s => finalizeDequantizedForDevice device (dequantizeFP8Experts w.2 s.2)
        | none => w
    let wName := s!"{b}.weight"
    if let some w ← tryLoadOptionalTensor path wName #[experts, outDim, inDim] device then
      let scaleName := s!"{b}.weight_scale_inv"
      let scaleInvOpt ← tryLoadOptionalTensor path scaleName #[experts, outDim / 128, inDim / 128] device
      return match scaleInvOpt with
        | some s => finalizeDequantizedForDevice device (dequantizeFP8Experts w.2 s.2)
        | none => w
  throw <| IO.userError s!"Failed to load MoE expert weight: {baseName}"

private def loadRMSNormSharded (modelDir : String) (baseName : String) (dim : Nat)
    (device : Device := Device.CPU)
    : IO (Qwen35RMSNorm (device := device) dim) := do
  let w ← loadTensorShardedCandidates modelDir (nameCandidates s!"{baseName}.weight") #[dim] device
  pure (Qwen35RMSNorm.fromCheckpointWeight (freezeStored w) 1e-6)

private def loadRMSNorm (path : String) (baseName : String) (dim : Nat)
    (device : Device := Device.CPU)
    : IO (Qwen35RMSNorm (device := device) dim) := do
  let w ← loadTensorCandidates path (nameCandidates s!"{baseName}.weight") #[dim] device
  pure (Qwen35RMSNorm.fromCheckpointWeight (freezeStored w) 1e-6)

private def loadRMSNormGatedSharded (modelDir : String) (baseName : String) (dim : Nat)
    (device : Device := Device.CPU)
    : IO (Qwen35RMSNormGated (device := device) dim) := do
  let w ← loadTensorShardedCandidates modelDir (nameCandidates s!"{baseName}.weight") #[dim] device
  pure { weight := freezeStored w, eps := 1e-6 }

private def loadRMSNormGated (path : String) (baseName : String) (dim : Nat)
    (device : Device := Device.CPU)
    : IO (Qwen35RMSNormGated (device := device) dim) := do
  let w ← loadTensorCandidates path (nameCandidates s!"{baseName}.weight") #[dim] device
  pure { weight := freezeStored w, eps := 1e-6 }

private def loadAttentionSharded (modelDir : String) (cfg : Config) (layerIdx : Nat)
    (device : Device := Device.CPU) (dtype : DType)
    : IO (Qwen35Attention (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.layers.{layerIdx}.self_attn"
  let qProj ← loadLinearWeightSharded modelDir s!"{p}.q_proj" (cfg.num_attention_heads * cfg.head_dim * 2) cfg.hidden_size device
  let kProj ← loadLinearWeightSharded modelDir s!"{p}.k_proj" (cfg.num_key_value_heads * cfg.head_dim) cfg.hidden_size device
  let vProj ← loadLinearWeightSharded modelDir s!"{p}.v_proj" (cfg.num_key_value_heads * cfg.head_dim) cfg.hidden_size device
  let oProj ← loadLinearWeightSharded modelDir s!"{p}.o_proj" cfg.hidden_size (cfg.num_attention_heads * cfg.head_dim) device

  let qNorm ← loadRMSNormSharded modelDir s!"{p}.q_norm" cfg.head_dim device
  let kNorm ← loadRMSNormSharded modelDir s!"{p}.k_norm" cfg.head_dim device

  pure {
    q_proj := reqGradFalse (← requireDType dtype qProj)
    k_proj := reqGradFalse (← requireDType dtype kProj)
    v_proj := reqGradFalse (← requireDType dtype vProj)
    o_proj := reqGradFalse (← requireDType dtype oProj)
    q_norm := qNorm
    k_norm := kNorm
  }

private def loadAttention (path : String) (cfg : Config) (layerIdx : Nat)
    (device : Device := Device.CPU) (dtype : DType)
    : IO (Qwen35Attention (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.layers.{layerIdx}.self_attn"
  let qProj ← loadLinearWeight path s!"{p}.q_proj" (cfg.num_attention_heads * cfg.head_dim * 2) cfg.hidden_size device
  let kProj ← loadLinearWeight path s!"{p}.k_proj" (cfg.num_key_value_heads * cfg.head_dim) cfg.hidden_size device
  let vProj ← loadLinearWeight path s!"{p}.v_proj" (cfg.num_key_value_heads * cfg.head_dim) cfg.hidden_size device
  let oProj ← loadLinearWeight path s!"{p}.o_proj" cfg.hidden_size (cfg.num_attention_heads * cfg.head_dim) device

  let qNorm ← loadRMSNorm path s!"{p}.q_norm" cfg.head_dim device
  let kNorm ← loadRMSNorm path s!"{p}.k_norm" cfg.head_dim device

  pure {
    q_proj := reqGradFalse (← requireDType dtype qProj)
    k_proj := reqGradFalse (← requireDType dtype kProj)
    v_proj := reqGradFalse (← requireDType dtype vProj)
    o_proj := reqGradFalse (← requireDType dtype oProj)
    q_norm := qNorm
    k_norm := kNorm
  }

private def loadLinearMixerSharded (modelDir : String) (cfg : Config) (layerIdx : Nat)
    (device : Device := Device.CPU) (dtype : DType)
    : IO (Qwen35GatedDeltaNet (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.layers.{layerIdx}.linear_attn"

  let inQKV ← loadLinearWeightSharded modelDir s!"{p}.in_proj_qkv" (Config.linearConvDim cfg) cfg.hidden_size device
  let inZ ← loadLinearWeightSharded modelDir s!"{p}.in_proj_z" (Config.linearValueDim cfg) cfg.hidden_size device
  let inB ← loadLinearWeightSharded modelDir s!"{p}.in_proj_b" cfg.linear_num_value_heads cfg.hidden_size device
  let inA ← loadLinearWeightSharded modelDir s!"{p}.in_proj_a" cfg.linear_num_value_heads cfg.hidden_size device

  let convW ← loadTensorShardedCandidates modelDir (nameCandidates s!"{p}.conv1d.weight") #[Config.linearConvDim cfg, 1, cfg.linear_conv_kernel_dim] device
  let convBOpt ← tryLoadOptionalTensorShardedCandidates modelDir (nameCandidates s!"{p}.conv1d.bias") #[Config.linearConvDim cfg] device
  let convB ←
    match convBOpt with
    | some b => pure b
    | none => do
      let support ← convW.1.acquireConstantSupport device
      let _ := support.down
      let bias := Tensor.zerosFor { shape := #[Config.linearConvDim cfg], dtype := convW.1, device }
      pure ⟨convW.1, bias⟩

  let dtBias ← loadTensorShardedCandidates modelDir (nameCandidates s!"{p}.dt_bias") #[cfg.linear_num_value_heads] device
  let aLog ← loadTensorShardedCandidates modelDir (nameCandidates s!"{p}.A_log") #[cfg.linear_num_value_heads] device

  let norm ← loadRMSNormGatedSharded modelDir s!"{p}.norm" cfg.linear_value_head_dim device
  let outProj ← loadLinearWeightSharded modelDir s!"{p}.out_proj" cfg.hidden_size (Config.linearValueDim cfg) device

  pure {
    in_proj_qkv := reqGradFalse (← requireDType dtype inQKV)
    in_proj_z := reqGradFalse (← requireDType dtype inZ)
    in_proj_b := reqGradFalse (← requireDType dtype inB)
    in_proj_a := reqGradFalse (← requireDType dtype inA)
    conv1d_weight := reqGradFalse (← requireDType dtype convW)
    conv1d_bias := reqGradFalse (← requireDType dtype convB)
    dt_bias := freezeStored dtBias
    a_log := freezeStored aLog
    norm := norm
    out_proj := reqGradFalse (← requireDType dtype outProj)
  }

private def loadLinearMixer (path : String) (cfg : Config) (layerIdx : Nat)
    (device : Device := Device.CPU) (dtype : DType)
    : IO (Qwen35GatedDeltaNet (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.layers.{layerIdx}.linear_attn"

  let inQKV ← loadLinearWeight path s!"{p}.in_proj_qkv" (Config.linearConvDim cfg) cfg.hidden_size device
  let inZ ← loadLinearWeight path s!"{p}.in_proj_z" (Config.linearValueDim cfg) cfg.hidden_size device
  let inB ← loadLinearWeight path s!"{p}.in_proj_b" cfg.linear_num_value_heads cfg.hidden_size device
  let inA ← loadLinearWeight path s!"{p}.in_proj_a" cfg.linear_num_value_heads cfg.hidden_size device

  let convW ← loadTensorCandidates path (nameCandidates s!"{p}.conv1d.weight") #[Config.linearConvDim cfg, 1, cfg.linear_conv_kernel_dim] device
  let convBOpt ← tryLoadOptionalTensorCandidates path (nameCandidates s!"{p}.conv1d.bias") #[Config.linearConvDim cfg] device
  let convB ←
    match convBOpt with
    | some b => pure b
    | none => do
      let support ← convW.1.acquireConstantSupport device
      let _ := support.down
      let bias := Tensor.zerosFor { shape := #[Config.linearConvDim cfg], dtype := convW.1, device }
      pure ⟨convW.1, bias⟩

  let dtBias ← loadTensorCandidates path (nameCandidates s!"{p}.dt_bias") #[cfg.linear_num_value_heads] device
  let aLog ← loadTensorCandidates path (nameCandidates s!"{p}.A_log") #[cfg.linear_num_value_heads] device

  let norm ← loadRMSNormGated path s!"{p}.norm" cfg.linear_value_head_dim device
  let outProj ← loadLinearWeight path s!"{p}.out_proj" cfg.hidden_size (Config.linearValueDim cfg) device

  pure {
    in_proj_qkv := reqGradFalse (← requireDType dtype inQKV)
    in_proj_z := reqGradFalse (← requireDType dtype inZ)
    in_proj_b := reqGradFalse (← requireDType dtype inB)
    in_proj_a := reqGradFalse (← requireDType dtype inA)
    conv1d_weight := reqGradFalse (← requireDType dtype convW)
    conv1d_bias := reqGradFalse (← requireDType dtype convB)
    dt_bias := freezeStored dtBias
    a_log := freezeStored aLog
    norm := norm
    out_proj := reqGradFalse (← requireDType dtype outProj)
  }

private def loadDenseMLPSharded (modelDir : String) (cfg : Config) (layerIdx : Nat)
    (device : Device := Device.CPU) (dtype : DType)
    : IO (Qwen35MLP (dtype := dtype) (device := device) cfg.hidden_size cfg.intermediate_size) := do
  let p := s!"model.layers.{layerIdx}.mlp"
  let gate ← loadLinearWeightSharded modelDir s!"{p}.gate_proj" cfg.intermediate_size cfg.hidden_size device
  let up ← loadLinearWeightSharded modelDir s!"{p}.up_proj" cfg.intermediate_size cfg.hidden_size device
  let down ← loadLinearWeightSharded modelDir s!"{p}.down_proj" cfg.hidden_size cfg.intermediate_size device
  pure {
    gate_proj := reqGradFalse (← requireDType dtype gate)
    up_proj := reqGradFalse (← requireDType dtype up)
    down_proj := reqGradFalse (← requireDType dtype down)
  }

private def loadDenseMLP (path : String) (cfg : Config) (layerIdx : Nat)
    (device : Device := Device.CPU) (dtype : DType)
    : IO (Qwen35MLP (dtype := dtype) (device := device) cfg.hidden_size cfg.intermediate_size) := do
  let p := s!"model.layers.{layerIdx}.mlp"
  let gate ← loadLinearWeight path s!"{p}.gate_proj" cfg.intermediate_size cfg.hidden_size device
  let up ← loadLinearWeight path s!"{p}.up_proj" cfg.intermediate_size cfg.hidden_size device
  let down ← loadLinearWeight path s!"{p}.down_proj" cfg.hidden_size cfg.intermediate_size device
  pure {
    gate_proj := reqGradFalse (← requireDType dtype gate)
    up_proj := reqGradFalse (← requireDType dtype up)
    down_proj := reqGradFalse (← requireDType dtype down)
  }

private def loadSparseMoeSharded (modelDir : String) (cfg : Config) (layerIdx : Nat)
    (device : Device := Device.CPU) (dtype : DType)
    : IO (Qwen35SparseMoeBlock (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.layers.{layerIdx}.mlp"

  let routerW ← loadTensorShardedCandidates modelDir (nameCandidates s!"{p}.gate.weight") #[cfg.num_experts, cfg.hidden_size] device
  let gu ← loadMoeExpertWeightSharded
    modelDir
    s!"{p}.experts.gate_up_proj"
    cfg.num_experts
    (2 * cfg.moe_intermediate_size)
    cfg.hidden_size
    device
  let down ← loadMoeExpertWeightSharded
    modelDir
    s!"{p}.experts.down_proj"
    cfg.num_experts
    cfg.hidden_size
    cfg.moe_intermediate_size
    device

  let sharedGateProj ← loadLinearWeightSharded modelDir s!"{p}.shared_expert.gate_proj" cfg.shared_expert_intermediate_size cfg.hidden_size device
  let sharedUpProj ← loadLinearWeightSharded modelDir s!"{p}.shared_expert.up_proj" cfg.shared_expert_intermediate_size cfg.hidden_size device
  let sharedDownProj ← loadLinearWeightSharded modelDir s!"{p}.shared_expert.down_proj" cfg.hidden_size cfg.shared_expert_intermediate_size device
  let sharedExpertGate ← loadLinearWeightSharded modelDir s!"{p}.shared_expert_gate" 1 cfg.hidden_size device

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

private def loadSparseMoe (path : String) (cfg : Config) (layerIdx : Nat)
    (device : Device := Device.CPU) (dtype : DType)
    : IO (Qwen35SparseMoeBlock (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.layers.{layerIdx}.mlp"

  let routerW ← loadTensorCandidates path (nameCandidates s!"{p}.gate.weight") #[cfg.num_experts, cfg.hidden_size] device
  let gu ← loadMoeExpertWeight
    path
    s!"{p}.experts.gate_up_proj"
    cfg.num_experts
    (2 * cfg.moe_intermediate_size)
    cfg.hidden_size
    device
  let down ← loadMoeExpertWeight
    path
    s!"{p}.experts.down_proj"
    cfg.num_experts
    cfg.hidden_size
    cfg.moe_intermediate_size
    device

  let sharedGateProj ← loadLinearWeight path s!"{p}.shared_expert.gate_proj" cfg.shared_expert_intermediate_size cfg.hidden_size device
  let sharedUpProj ← loadLinearWeight path s!"{p}.shared_expert.up_proj" cfg.shared_expert_intermediate_size cfg.hidden_size device
  let sharedDownProj ← loadLinearWeight path s!"{p}.shared_expert.down_proj" cfg.hidden_size cfg.shared_expert_intermediate_size device
  let sharedExpertGate ← loadLinearWeight path s!"{p}.shared_expert_gate" 1 cfg.hidden_size device

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

private def loadLayerSharded (modelDir : String) (cfg : Config) (layerIdx : Nat)
    (device : Device := Device.CPU) (dtype : DType)
    : IO (Qwen35Layer (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.layers.{layerIdx}"
  let layerTypes := Config.normalizedLayerTypes cfg
  let lt := layerTypes.getD layerIdx .linearAttention

  let inputNorm ← loadRMSNormSharded modelDir s!"{p}.input_layernorm" cfg.hidden_size device
  let postNorm ← loadRMSNormSharded modelDir s!"{p}.post_attention_layernorm" cfg.hidden_size device

  let fullAttn ←
    match lt with
    | .fullAttention =>
      let a ← loadAttentionSharded modelDir cfg layerIdx device dtype
      pure (some a)
    | .linearAttention => pure none

  let linearAttn ←
    match lt with
    | .linearAttention =>
      let l ← loadLinearMixerSharded modelDir cfg layerIdx device dtype
      pure (some l)
    | .fullAttention => pure none

  let denseMlp ←
    if Config.isMoE cfg then
      pure none
    else
      let m ← loadDenseMLPSharded modelDir cfg layerIdx device dtype
      pure (some m)

  let sparseMoe ←
    if Config.isMoE cfg then
      let m ← loadSparseMoeSharded modelDir cfg layerIdx device dtype
      pure (some m)
    else
      pure none

  pure {
    layerType := lt
    input_layernorm := inputNorm
    full_attn := fullAttn
    linear_attn := linearAttn
    post_attention_layernorm := postNorm
    dense_mlp := denseMlp
    sparse_moe := sparseMoe
  }

private def loadLayer (path : String) (cfg : Config) (layerIdx : Nat)
    (device : Device := Device.CPU) (dtype : DType)
    : IO (Qwen35Layer (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.layers.{layerIdx}"
  let layerTypes := Config.normalizedLayerTypes cfg
  let lt := layerTypes.getD layerIdx .linearAttention

  let inputNorm ← loadRMSNorm path s!"{p}.input_layernorm" cfg.hidden_size device
  let postNorm ← loadRMSNorm path s!"{p}.post_attention_layernorm" cfg.hidden_size device

  let fullAttn ←
    match lt with
    | .fullAttention =>
      let a ← loadAttention path cfg layerIdx device dtype
      pure (some a)
    | .linearAttention => pure none

  let linearAttn ←
    match lt with
    | .linearAttention =>
      let l ← loadLinearMixer path cfg layerIdx device dtype
      pure (some l)
    | .fullAttention => pure none

  let denseMlp ←
    if Config.isMoE cfg then
      pure none
    else
      let m ← loadDenseMLP path cfg layerIdx device dtype
      pure (some m)

  let sparseMoe ←
    if Config.isMoE cfg then
      let m ← loadSparseMoe path cfg layerIdx device dtype
      pure (some m)
    else
      pure none

  pure {
    layerType := lt
    input_layernorm := inputNorm
    full_attn := fullAttn
    linear_attn := linearAttn
    post_attention_layernorm := postNorm
    dense_mlp := denseMlp
    sparse_moe := sparseMoe
  }

namespace Qwen35ForCausalLM

/-- Load Qwen3.5 model from sharded HF SafeTensors directory. -/
def loadSharded (modelDir : String) (cfg : Config := Config.qwen35_9B)
    (device : Device := Device.CPU)
    (log : Handlers := {})
    : IO (Σ dtype : RealFloatingDType, Qwen35ForCausalLM (device := device) cfg dtype.val) := do
  log.onInfo s!"Loading Qwen35ForCausalLM (device := device) from {modelDir}..."

  let embedTokens ← loadTensorShardedCandidates modelDir (nameCandidates "model.embed_tokens.weight") #[cfg.vocab_size, cfg.hidden_size] device
  let dtype := embedTokens.1
  let floating ← IO.ofExcept dtype.realFloating?
  let _ := floating.down

  let mut layers : Array (Qwen35Layer (dtype := dtype) (device := device) cfg) := #[]
  for i in [:cfg.num_hidden_layers] do
    let layer ← loadLayerSharded modelDir cfg i device dtype
    layers := layers.push layer
    if (i + 1) % 8 == 0 || i + 1 == cfg.num_hidden_layers then
      log.onInfo s!"  loaded layers {i + 1}/{cfg.num_hidden_layers}"

  let norm ← loadRMSNormSharded modelDir "model.norm" cfg.hidden_size device

  let model : Qwen35Model (dtype := dtype) (device := device) cfg := {
    embed_tokens := reqGradFalse (← requireDType dtype embedTokens)
    layers := layers
    norm := norm
  }

  let lmHeadOpt0 ← tryLoadOptionalTensorSharded modelDir "lm_head.weight" #[cfg.vocab_size, cfg.hidden_size] device
  let lmHeadOpt ←
    match lmHeadOpt0 with
    | some t => pure (some t)
    | none =>
      -- try multimodal-style prefixed key
      tryLoadOptionalTensorSharded modelDir "language_model.lm_head.weight" #[cfg.vocab_size, cfg.hidden_size] device

  let (lmHead, tieWordEmbeddings) ←
    match lmHeadOpt with
    | some w => pure (reqGradFalse (← requireDType dtype w), false)
    | none => do
      log.onInfo "  lm_head.weight not found; tying to embeddings."
      pure (reqGradFalse model.embed_tokens, true)

  log.onInfo "Loaded Qwen35ForCausalLM (device := device) weights."
  pure ⟨RealFloatingDType.ofDType dtype, { model := model, lmHead := lmHead, tieWordEmbeddings := tieWordEmbeddings }⟩

/-- Load Qwen3.5 model from a single HF SafeTensors file. -/
def load (path : String) (cfg : Config := Config.qwen35_9B)
    (device : Device := Device.CPU)
    (log : Handlers := {})
    : IO (Σ dtype : RealFloatingDType, Qwen35ForCausalLM (device := device) cfg dtype.val) := do
  log.onInfo s!"Loading Qwen35ForCausalLM (device := device) from {path}..."

  let embedTokens ← loadTensorCandidates path (nameCandidates "model.embed_tokens.weight") #[cfg.vocab_size, cfg.hidden_size] device
  let dtype := embedTokens.1
  let floating ← IO.ofExcept dtype.realFloating?
  let _ := floating.down

  let mut layers : Array (Qwen35Layer (dtype := dtype) (device := device) cfg) := #[]
  for i in [:cfg.num_hidden_layers] do
    let layer ← loadLayer path cfg i device dtype
    layers := layers.push layer
    if (i + 1) % 8 == 0 || i + 1 == cfg.num_hidden_layers then
      log.onInfo s!"  loaded layers {i + 1}/{cfg.num_hidden_layers}"

  let norm ← loadRMSNorm path "model.norm" cfg.hidden_size device

  let model : Qwen35Model (dtype := dtype) (device := device) cfg := {
    embed_tokens := reqGradFalse (← requireDType dtype embedTokens)
    layers := layers
    norm := norm
  }

  let lmHeadOpt0 ← tryLoadOptionalTensor path "lm_head.weight" #[cfg.vocab_size, cfg.hidden_size] device
  let lmHeadOpt ←
    match lmHeadOpt0 with
    | some t => pure (some t)
    | none => tryLoadOptionalTensor path "language_model.lm_head.weight" #[cfg.vocab_size, cfg.hidden_size] device

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
