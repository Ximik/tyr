/-
  Tyr/Model/Gemma4/Weights.lean

  Pretrained weight loading for standalone Gemma 4 text causal-LM.
  Supports:
  - single-file and sharded Hugging Face SafeTensors checkpoints
  - small-model per-layer input blocks (E2B/E4B)
  - dense and MoE feed-forward branches
-/
import Tyr.Log
import Tyr.SafeTensors.Load
import Tyr.Model.Utils
import Tyr.Model.Gemma4.Model

namespace torch.gemma4

open torch.Log
open torch.Model (reqGradFalse)
open torch.safetensors (pushUnique tryLoadTensor tryLoadTensorSharded
  loadTensorCandidates loadTensorShardedCandidates
  tryLoadTensorCandidates tryLoadTensorShardedCandidates)

private def tensorNameCandidates (name : String) : Array String :=
  Id.run do
    let modelPrefix := "model."
    let llmPrefix := "language_model."
    let modelLlmPrefix := "model.language_model."
    let mut out : Array String := #[]
    out := pushUnique out name
    if name.startsWith modelLlmPrefix then
      let suffix := (name.drop modelLlmPrefix.length).toString
      out := pushUnique out suffix
      out := pushUnique out s!"{modelPrefix}{suffix}"
      out := pushUnique out s!"{llmPrefix}{suffix}"
    else if name.startsWith llmPrefix then
      let suffix := (name.drop llmPrefix.length).toString
      out := pushUnique out suffix
      out := pushUnique out s!"{modelPrefix}{suffix}"
      out := pushUnique out s!"{modelLlmPrefix}{suffix}"
    else if name.startsWith modelPrefix then
      let suffix := (name.drop modelPrefix.length).toString
      out := pushUnique out suffix
      out := pushUnique out s!"{llmPrefix}{suffix}"
      out := pushUnique out s!"{modelLlmPrefix}{suffix}"
    else
      out := pushUnique out s!"{modelPrefix}{name}"
      out := pushUnique out s!"{llmPrefix}{name}"
      out := pushUnique out s!"{modelLlmPrefix}{name}"
    out

private def parameterNameCandidates (name : String) : Array String :=
  Id.run do
    let mut out := tensorNameCandidates name
    if !name.endsWith ".weight" then
      for cand in tensorNameCandidates s!"{name}.weight" do
        out := pushUnique out cand
    out

private def loadParameterSharded (modelDir : String) (name : String) (s : Shape) : IO (Tensor { shape := s }) :=
  loadTensorShardedCandidates modelDir (parameterNameCandidates name) s

private def tryLoadParameterSharded (modelDir : String) (name : String) (s : Shape) : IO (Option (Tensor { shape := s })) :=
  tryLoadTensorShardedCandidates modelDir (parameterNameCandidates name) s

private def loadParameter (path : String) (name : String) (s : Shape) : IO (Tensor { shape := s }) :=
  loadTensorCandidates path (parameterNameCandidates name) s

private def tryLoadParameter (path : String) (name : String) (s : Shape) : IO (Option (Tensor { shape := s })) :=
  tryLoadTensorCandidates path (parameterNameCandidates name) s

private def zeros1dOn {n : UInt64} (device : Device) : Tensor { shape := #[n] } :=
  qwen35.zerosDynamic #[n] false device

private def zeros2dOn {rows cols : UInt64} (device : Device) : Tensor { shape := #[rows, cols] } :=
  qwen35.zerosDynamic #[rows, cols] false device

private def pad1dTo {srcDim dstDim : UInt64}
    (x : Tensor { shape := #[srcDim] })
    : Tensor { shape := #[dstDim] } :=
  let base : Tensor { shape := #[dstDim] } := Tensor.castLike x (zeros1dOn x.device)
  Tensor.sliceScatter base 0 0 x

private def pad2dTo {srcRows srcCols dstRows dstCols : UInt64}
    (x : Tensor { shape := #[srcRows, srcCols] })
    : Tensor { shape := #[dstRows, dstCols] } :=
  let colsPadded : Tensor { shape := #[srcRows, dstCols] } :=
    Tensor.sliceScatter (Tensor.castLike x (zeros2dOn x.device : Tensor { shape := #[srcRows, dstCols] })) 1 0 x
  Tensor.sliceScatter (Tensor.castLike x (zeros2dOn x.device : Tensor { shape := #[dstRows, dstCols] })) 0 0 colsPadded

private def loadPadded1DSharded
    (modelDir : String)
    (name : String)
    (srcDim dstDim : UInt64)
    : IO (Tensor { shape := #[dstDim] }) := do
  let t ← loadParameterSharded modelDir name #[srcDim]
  pure (pad1dTo t)

private def loadPadded1D
    (path : String)
    (name : String)
    (srcDim dstDim : UInt64)
    : IO (Tensor { shape := #[dstDim] }) := do
  let t ← loadParameter path name #[srcDim]
  pure (pad1dTo t)

private def loadPadded2DSharded
    (modelDir : String)
    (name : String)
    (srcRows srcCols dstRows dstCols : UInt64)
    : IO (Tensor { shape := #[dstRows, dstCols] }) := do
  let t ← loadParameterSharded modelDir name #[srcRows, srcCols]
  pure (pad2dTo t)

private def loadPadded2D
    (path : String)
    (name : String)
    (srcRows srcCols dstRows dstCols : UInt64)
    : IO (Tensor { shape := #[dstRows, dstCols] }) := do
  let t ← loadParameter path name #[srcRows, srcCols]
  pure (pad2dTo t)

private def loadRMSNormSharded
    (modelDir : String)
    (name : String)
    (srcDim dstDim : UInt64)
    (eps : Float)
    : IO (Gemma4RMSNorm dstDim) := do
  let w ← loadPadded1DSharded modelDir name srcDim dstDim
  pure (Gemma4RMSNorm.fromCheckpointWeight (reqGradFalse w) eps)

private def loadRMSNorm
    (path : String)
    (name : String)
    (srcDim dstDim : UInt64)
    (eps : Float)
    : IO (Gemma4RMSNorm dstDim) := do
  let w ← loadPadded1D path name srcDim dstDim
  pure (Gemma4RMSNorm.fromCheckpointWeight (reqGradFalse w) eps)

private def layerHeadDim (cfg : Config) (layerType : LayerType) : UInt64 :=
  match layerType with
  | .slidingAttention => cfg.head_dim
  | .fullAttention => Config.fullHeadDim cfg

private def layerNumKVHeads (cfg : Config) (layerType : LayerType) : UInt64 :=
  match layerType with
  | .slidingAttention => cfg.num_key_value_heads
  | .fullAttention => Config.fullNumKVHeads cfg

private def layerAttentionDim (cfg : Config) (layerType : LayerType) : UInt64 :=
  cfg.num_attention_heads * layerHeadDim cfg layerType

private def layerKVProjDim (cfg : Config) (layerType : LayerType) : UInt64 :=
  layerNumKVHeads cfg layerType * layerHeadDim cfg layerType

private def loadAttentionSharded (modelDir : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Gemma4Attention cfg) := do
  let p := s!"model.language_model.layers.{layerIdx}.self_attn"
  let layerType := Config.layerTypeAt cfg layerIdx
  let headDim := layerHeadDim cfg layerType
  let attnDim := layerAttentionDim cfg layerType
  let kvProjDim := layerKVProjDim cfg layerType

  let qProj ←
    loadPadded2DSharded
      modelDir
      s!"{p}.q_proj"
      attnDim cfg.hidden_size
      (Config.maxAttentionDim cfg) cfg.hidden_size
  let kProj ←
    loadPadded2DSharded
      modelDir
      s!"{p}.k_proj"
      kvProjDim cfg.hidden_size
      (Config.maxKVProjDim cfg) cfg.hidden_size
  let vProj ←
    if cfg.attention_k_eq_v && layerType == .fullAttention then
      pure (zeros2dOn Device.CPU : Tensor { shape := #[Config.maxKVProjDim cfg, cfg.hidden_size] })
    else
      loadPadded2DSharded
        modelDir
        s!"{p}.v_proj"
        kvProjDim cfg.hidden_size
        (Config.maxKVProjDim cfg) cfg.hidden_size
  let oProj ←
    loadPadded2DSharded
      modelDir
      s!"{p}.o_proj"
      cfg.hidden_size attnDim
      cfg.hidden_size (Config.maxAttentionDim cfg)

  let qNorm ← loadRMSNormSharded modelDir s!"{p}.q_norm" headDim (Config.maxHeadDim cfg) cfg.rms_norm_eps
  let kNorm ← loadRMSNormSharded modelDir s!"{p}.k_norm" headDim (Config.maxHeadDim cfg) cfg.rms_norm_eps

  pure {
    q_proj := reqGradFalse qProj
    k_proj := reqGradFalse kProj
    v_proj := reqGradFalse vProj
    o_proj := reqGradFalse oProj
    q_norm := qNorm
    k_norm := kNorm
  }

private def loadAttention (path : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Gemma4Attention cfg) := do
  let p := s!"model.language_model.layers.{layerIdx}.self_attn"
  let layerType := Config.layerTypeAt cfg layerIdx
  let headDim := layerHeadDim cfg layerType
  let attnDim := layerAttentionDim cfg layerType
  let kvProjDim := layerKVProjDim cfg layerType

  let qProj ←
    loadPadded2D
      path
      s!"{p}.q_proj"
      attnDim cfg.hidden_size
      (Config.maxAttentionDim cfg) cfg.hidden_size
  let kProj ←
    loadPadded2D
      path
      s!"{p}.k_proj"
      kvProjDim cfg.hidden_size
      (Config.maxKVProjDim cfg) cfg.hidden_size
  let vProj ←
    if cfg.attention_k_eq_v && layerType == .fullAttention then
      pure (zeros2dOn Device.CPU : Tensor { shape := #[Config.maxKVProjDim cfg, cfg.hidden_size] })
    else
      loadPadded2D
        path
        s!"{p}.v_proj"
        kvProjDim cfg.hidden_size
        (Config.maxKVProjDim cfg) cfg.hidden_size
  let oProj ←
    loadPadded2D
      path
      s!"{p}.o_proj"
      cfg.hidden_size attnDim
      cfg.hidden_size (Config.maxAttentionDim cfg)

  let qNorm ← loadRMSNorm path s!"{p}.q_norm" headDim (Config.maxHeadDim cfg) cfg.rms_norm_eps
  let kNorm ← loadRMSNorm path s!"{p}.k_norm" headDim (Config.maxHeadDim cfg) cfg.rms_norm_eps

  pure {
    q_proj := reqGradFalse qProj
    k_proj := reqGradFalse kProj
    v_proj := reqGradFalse vProj
    o_proj := reqGradFalse oProj
    q_norm := qNorm
    k_norm := kNorm
  }

private def loadDenseMLPSharded (modelDir : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Gemma4MLP cfg) := do
  let p := s!"model.language_model.layers.{layerIdx}.mlp"
  let inter := Config.layerIntermediateSize cfg layerIdx
  let targetInter := Config.maxIntermediateSize cfg
  let gate ←
    loadPadded2DSharded
      modelDir
      s!"{p}.gate_proj"
      inter cfg.hidden_size
      targetInter cfg.hidden_size
  let up ←
    loadPadded2DSharded
      modelDir
      s!"{p}.up_proj"
      inter cfg.hidden_size
      targetInter cfg.hidden_size
  let down ←
    loadPadded2DSharded
      modelDir
      s!"{p}.down_proj"
      cfg.hidden_size inter
      cfg.hidden_size targetInter
  pure {
    gate_proj := reqGradFalse gate
    up_proj := reqGradFalse up
    down_proj := reqGradFalse down
  }

private def loadDenseMLP (path : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Gemma4MLP cfg) := do
  let p := s!"model.language_model.layers.{layerIdx}.mlp"
  let inter := Config.layerIntermediateSize cfg layerIdx
  let targetInter := Config.maxIntermediateSize cfg
  let gate ←
    loadPadded2D
      path
      s!"{p}.gate_proj"
      inter cfg.hidden_size
      targetInter cfg.hidden_size
  let up ←
    loadPadded2D
      path
      s!"{p}.up_proj"
      inter cfg.hidden_size
      targetInter cfg.hidden_size
  let down ←
    loadPadded2D
      path
      s!"{p}.down_proj"
      cfg.hidden_size inter
      cfg.hidden_size targetInter
  pure {
    gate_proj := reqGradFalse gate
    up_proj := reqGradFalse up
    down_proj := reqGradFalse down
  }

private def loadPerLayerInputBlockSharded (modelDir : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Gemma4PerLayerInputBlock cfg) := do
  let p := s!"model.language_model.layers.{layerIdx}"
  let gate ← loadParameterSharded modelDir s!"{p}.per_layer_input_gate" #[cfg.hidden_size_per_layer_input, cfg.hidden_size]
  let proj ← loadParameterSharded modelDir s!"{p}.per_layer_projection" #[cfg.hidden_size, cfg.hidden_size_per_layer_input]
  let norm ← loadRMSNormSharded modelDir s!"{p}.post_per_layer_input_norm" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  pure {
    per_layer_input_gate := reqGradFalse gate
    per_layer_projection := reqGradFalse proj
    post_per_layer_input_norm := norm
  }

private def loadPerLayerInputBlock (path : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Gemma4PerLayerInputBlock cfg) := do
  let p := s!"model.language_model.layers.{layerIdx}"
  let gate ← loadParameter path s!"{p}.per_layer_input_gate" #[cfg.hidden_size_per_layer_input, cfg.hidden_size]
  let proj ← loadParameter path s!"{p}.per_layer_projection" #[cfg.hidden_size, cfg.hidden_size_per_layer_input]
  let norm ← loadRMSNorm path s!"{p}.post_per_layer_input_norm" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  pure {
    per_layer_input_gate := reqGradFalse gate
    per_layer_projection := reqGradFalse proj
    post_per_layer_input_norm := norm
  }

private def loadMoeBranchSharded (modelDir : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Gemma4MoeBranch cfg) := do
  let p := s!"model.language_model.layers.{layerIdx}"
  let routerProj ← loadParameterSharded modelDir s!"{p}.router.proj" #[cfg.num_experts, cfg.hidden_size]
  let routerScale ← loadParameterSharded modelDir s!"{p}.router.scale" #[cfg.hidden_size]
  let perExpertScale ← loadParameterSharded modelDir s!"{p}.router.per_expert_scale" #[cfg.num_experts]
  let gateUp ←
    loadParameterSharded
      modelDir
      s!"{p}.experts.gate_up_proj"
      #[cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size]
  let down ←
    loadParameterSharded
      modelDir
      s!"{p}.experts.down_proj"
      #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size]
  let post1 ← loadRMSNormSharded modelDir s!"{p}.post_feedforward_layernorm_1" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  let post2 ← loadRMSNormSharded modelDir s!"{p}.post_feedforward_layernorm_2" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  let pre2 ← loadRMSNormSharded modelDir s!"{p}.pre_feedforward_layernorm_2" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  pure {
    router := {
      proj := reqGradFalse routerProj
      scale := reqGradFalse routerScale
      per_expert_scale := reqGradFalse perExpertScale
    }
    experts := {
      gate_up_proj := reqGradFalse gateUp
      down_proj := reqGradFalse down
    }
    post_feedforward_layernorm_1 := post1
    post_feedforward_layernorm_2 := post2
    pre_feedforward_layernorm_2 := pre2
  }

private def loadMoeBranch (path : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Gemma4MoeBranch cfg) := do
  let p := s!"model.language_model.layers.{layerIdx}"
  let routerProj ← loadParameter path s!"{p}.router.proj" #[cfg.num_experts, cfg.hidden_size]
  let routerScale ← loadParameter path s!"{p}.router.scale" #[cfg.hidden_size]
  let perExpertScale ← loadParameter path s!"{p}.router.per_expert_scale" #[cfg.num_experts]
  let gateUp ←
    loadParameter
      path
      s!"{p}.experts.gate_up_proj"
      #[cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size]
  let down ←
    loadParameter
      path
      s!"{p}.experts.down_proj"
      #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size]
  let post1 ← loadRMSNorm path s!"{p}.post_feedforward_layernorm_1" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  let post2 ← loadRMSNorm path s!"{p}.post_feedforward_layernorm_2" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  let pre2 ← loadRMSNorm path s!"{p}.pre_feedforward_layernorm_2" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  pure {
    router := {
      proj := reqGradFalse routerProj
      scale := reqGradFalse routerScale
      per_expert_scale := reqGradFalse perExpertScale
    }
    experts := {
      gate_up_proj := reqGradFalse gateUp
      down_proj := reqGradFalse down
    }
    post_feedforward_layernorm_1 := post1
    post_feedforward_layernorm_2 := post2
    pre_feedforward_layernorm_2 := pre2
  }

private def loadLayerSharded (modelDir : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Gemma4Layer cfg) := do
  let p := s!"model.language_model.layers.{layerIdx}"
  let layerType := Config.layerTypeAt cfg layerIdx
  let selfAttn ← loadAttentionSharded modelDir cfg layerIdx
  let mlp ← loadDenseMLPSharded modelDir cfg layerIdx
  let inputNorm ← loadRMSNormSharded modelDir s!"{p}.input_layernorm" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  let postAttnNorm ← loadRMSNormSharded modelDir s!"{p}.post_attention_layernorm" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  let preFfnNorm ← loadRMSNormSharded modelDir s!"{p}.pre_feedforward_layernorm" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  let postFfnNorm ← loadRMSNormSharded modelDir s!"{p}.post_feedforward_layernorm" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  let perLayerInput ←
    if Config.hasPerLayerInput cfg then
      some <$> loadPerLayerInputBlockSharded modelDir cfg layerIdx
    else
      pure none
  let moe ←
    if Config.isMoE cfg then
      some <$> loadMoeBranchSharded modelDir cfg layerIdx
    else
      pure none
  let layerScalar ← loadParameterSharded modelDir s!"{p}.layer_scalar" #[1]
  pure {
    layerIdx := layerIdx
    layerType := layerType
    self_attn := selfAttn
    mlp := mlp
    input_layernorm := inputNorm
    post_attention_layernorm := postAttnNorm
    pre_feedforward_layernorm := preFfnNorm
    post_feedforward_layernorm := postFfnNorm
    per_layer_input := perLayerInput
    moe := moe
    layer_scalar := reqGradFalse layerScalar
  }

private def loadLayer (path : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Gemma4Layer cfg) := do
  let p := s!"model.language_model.layers.{layerIdx}"
  let layerType := Config.layerTypeAt cfg layerIdx
  let selfAttn ← loadAttention path cfg layerIdx
  let mlp ← loadDenseMLP path cfg layerIdx
  let inputNorm ← loadRMSNorm path s!"{p}.input_layernorm" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  let postAttnNorm ← loadRMSNorm path s!"{p}.post_attention_layernorm" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  let preFfnNorm ← loadRMSNorm path s!"{p}.pre_feedforward_layernorm" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  let postFfnNorm ← loadRMSNorm path s!"{p}.post_feedforward_layernorm" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  let perLayerInput ←
    if Config.hasPerLayerInput cfg then
      some <$> loadPerLayerInputBlock path cfg layerIdx
    else
      pure none
  let moe ←
    if Config.isMoE cfg then
      some <$> loadMoeBranch path cfg layerIdx
    else
      pure none
  let layerScalar ← loadParameter path s!"{p}.layer_scalar" #[1]
  pure {
    layerIdx := layerIdx
    layerType := layerType
    self_attn := selfAttn
    mlp := mlp
    input_layernorm := inputNorm
    post_attention_layernorm := postAttnNorm
    pre_feedforward_layernorm := preFfnNorm
    post_feedforward_layernorm := postFfnNorm
    per_layer_input := perLayerInput
    moe := moe
    layer_scalar := reqGradFalse layerScalar
  }

namespace Gemma4ForCausalLM

/-- Load Gemma 4 model from a sharded HF SafeTensors directory. -/
def loadSharded (modelDir : String) (cfg : Config := Config.gemma4_E4B)
    (log : Handlers := {})
    : IO (Gemma4ForCausalLM cfg) := do
  log.onInfo s!"Loading Gemma4ForCausalLM from {modelDir}..."

  let embedTokens ←
    loadParameterSharded
      modelDir
      "model.language_model.embed_tokens"
      #[cfg.vocab_size, cfg.hidden_size]

  let embedTokensPerLayer ←
    if Config.hasPerLayerInput cfg then
      some <$> loadParameterSharded
        modelDir
        "model.language_model.embed_tokens_per_layer"
        #[cfg.vocab_size_per_layer_input, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input]
    else
      pure none

  let perLayerModelProjection ←
    if Config.hasPerLayerInput cfg then
      some <$> loadParameterSharded
        modelDir
        "model.language_model.per_layer_model_projection"
        #[cfg.num_hidden_layers * cfg.hidden_size_per_layer_input, cfg.hidden_size]
    else
      pure none

  let perLayerProjectionNorm ←
    if Config.hasPerLayerInput cfg then
      some <$> loadRMSNormSharded
        modelDir
        "model.language_model.per_layer_projection_norm"
        cfg.hidden_size_per_layer_input
        cfg.hidden_size_per_layer_input
        cfg.rms_norm_eps
    else
      pure none

  let mut layers : Array (Gemma4Layer cfg) := #[]
  for i in [:cfg.num_hidden_layers.toNat] do
    let layer ← loadLayerSharded modelDir cfg i.toUInt64
    layers := layers.push layer
    if (i + 1) % 8 == 0 || i + 1 == cfg.num_hidden_layers.toNat then
      log.onInfo s!"  loaded layers {i + 1}/{cfg.num_hidden_layers.toNat}"

  let norm ← loadRMSNormSharded modelDir "model.language_model.norm" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps

  let model : Gemma4Model cfg := {
    embed_tokens := reqGradFalse embedTokens
    embed_tokens_per_layer := embedTokensPerLayer.map reqGradFalse
    per_layer_model_projection := perLayerModelProjection.map reqGradFalse
    per_layer_projection_norm := perLayerProjectionNorm
    layers := layers
    norm := norm
  }

  let lmHeadOpt ← tryLoadParameterSharded modelDir "lm_head" #[cfg.vocab_size, cfg.hidden_size]
  let lmHead ←
    match lmHeadOpt with
    | some w => pure (reqGradFalse w)
    | none =>
      if cfg.tie_word_embeddings then
        pure (reqGradFalse model.embed_tokens)
      else
        throw <| IO.userError "lm_head not found and tie_word_embeddings=false"

  log.onInfo "Loaded Gemma4ForCausalLM weights."
  pure {
    model := model
    lmHead := lmHead
    tieWordEmbeddings := cfg.tie_word_embeddings
  }

/-- Load Gemma 4 model from a single HF SafeTensors file. -/
def load (path : String) (cfg : Config := Config.gemma4_E4B)
    (log : Handlers := {})
    : IO (Gemma4ForCausalLM cfg) := do
  log.onInfo s!"Loading Gemma4ForCausalLM from {path}..."

  let embedTokens ←
    loadParameter
      path
      "model.language_model.embed_tokens"
      #[cfg.vocab_size, cfg.hidden_size]

  let embedTokensPerLayer ←
    if Config.hasPerLayerInput cfg then
      some <$> loadParameter
        path
        "model.language_model.embed_tokens_per_layer"
        #[cfg.vocab_size_per_layer_input, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input]
    else
      pure none

  let perLayerModelProjection ←
    if Config.hasPerLayerInput cfg then
      some <$> loadParameter
        path
        "model.language_model.per_layer_model_projection"
        #[cfg.num_hidden_layers * cfg.hidden_size_per_layer_input, cfg.hidden_size]
    else
      pure none

  let perLayerProjectionNorm ←
    if Config.hasPerLayerInput cfg then
      some <$> loadRMSNorm
        path
        "model.language_model.per_layer_projection_norm"
        cfg.hidden_size_per_layer_input
        cfg.hidden_size_per_layer_input
        cfg.rms_norm_eps
    else
      pure none

  let mut layers : Array (Gemma4Layer cfg) := #[]
  for i in [:cfg.num_hidden_layers.toNat] do
    let layer ← loadLayer path cfg i.toUInt64
    layers := layers.push layer
    if (i + 1) % 8 == 0 || i + 1 == cfg.num_hidden_layers.toNat then
      log.onInfo s!"  loaded layers {i + 1}/{cfg.num_hidden_layers.toNat}"

  let norm ← loadRMSNorm path "model.language_model.norm" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps

  let model : Gemma4Model cfg := {
    embed_tokens := reqGradFalse embedTokens
    embed_tokens_per_layer := embedTokensPerLayer.map reqGradFalse
    per_layer_model_projection := perLayerModelProjection.map reqGradFalse
    per_layer_projection_norm := perLayerProjectionNorm
    layers := layers
    norm := norm
  }

  let lmHeadOpt ← tryLoadParameter path "lm_head" #[cfg.vocab_size, cfg.hidden_size]
  let lmHead ←
    match lmHeadOpt with
    | some w => pure (reqGradFalse w)
    | none =>
      if cfg.tie_word_embeddings then
        pure (reqGradFalse model.embed_tokens)
      else
        throw <| IO.userError "lm_head not found and tie_word_embeddings=false"

  log.onInfo "Loaded Gemma4ForCausalLM weights."
  pure {
    model := model
    lmHead := lmHead
    tieWordEmbeddings := cfg.tie_word_embeddings
  }

end Gemma4ForCausalLM

end torch.gemma4
