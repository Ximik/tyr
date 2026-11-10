/-
  Tyr/Model/Gemma4/Weights.lean

  Pretrained weight loading for standalone Gemma 4 text causal-LM.
  Supports:
  - single-file and sharded Hugging Face SafeTensors checkpoints
  - small-model per-layer input blocks (E2B/E4B)
  - dense and MoE feed-forward branches
-/
import Tyr.Log
import Tyr.SafeTensors.Source
import Tyr.Model.Utils
import Tyr.Model.Gemma4.Model

namespace torch.gemma4

variable {device : Device} {dtype : DType}

open torch.Log
open torch.Model (reqGradFalse)
open torch.safetensors (pushUnique Source)

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

private def loadParameter (src : Source) (name : String) (s : Shape) : IO (SomeDTypeTensor (s) device) := do
  src.load (← src.resolve (parameterNameCandidates name)) s device

private def tryLoadParameter (src : Source) (name : String) (s : Shape) : IO (Option (SomeDTypeTensor (s) device)) :=
  ((src.find? (parameterNameCandidates name)).mapM (src.load · s device))

private def freezeStored {shape : Shape} (tensor : SomeDTypeTensor shape device) :
    SomeDTypeTensor shape device := ⟨tensor.1, tensor.2.detach⟩

private def requireDType {shape : Shape} (dtype : DType)
    (tensor : SomeDTypeTensor shape device) : IO (Tensor ⟨shape, dtype, device⟩) :=
  IO.ofExcept (tensor.atDType? dtype)

private def pad1dTo {srcDim dstDim : Nat} {dt : DType}
    (x : Tensor ⟨#[srcDim], dt, device⟩) : IO (Tensor ⟨#[dstDim], dt, device⟩) := do
  let support ← dt.acquireConstantSupport device
  let _ := support.down
  let ⟨_hFit⟩ ← if h : srcDim ≤ dstDim then pure (PLift.up h) else
    throw <| IO.userError s!"Gemma4 weight of size {srcDim} does not fit its padded size {dstDim}"
  let base := Tensor.zerosFor ⟨#[dstDim], dt, device⟩
  pure (Tensor.sliceScatter base 0 0 x)

private def pad2dTo {srcRows srcCols dstRows dstCols : Nat} {dt : DType}
    (x : Tensor ⟨#[srcRows, srcCols], dt, device⟩) :
    IO (Tensor ⟨#[dstRows, dstCols], dt, device⟩) := do
  let support ← dt.acquireConstantSupport device
  let _ := support.down
  let ⟨_hFit⟩ ← if h : srcRows ≤ dstRows ∧ srcCols ≤ dstCols then pure (PLift.up h) else
    throw <| IO.userError s!"Gemma4 weight of shape {srcRows}x{srcCols} does not fit its padded shape {dstRows}x{dstCols}"
  let columns := Tensor.zerosFor ⟨#[srcRows, dstCols], dt, device⟩
  let rows := Tensor.zerosFor ⟨#[dstRows, dstCols], dt, device⟩
  let colsPadded := Tensor.sliceScatter columns 1 0 x
  pure (Tensor.sliceScatter rows 0 0 colsPadded)

private def loadPadded1D
    (src : Source)
    (name : String)
    (srcDim dstDim : Nat)
    : IO (SomeDTypeTensor (#[dstDim]) device) := do
  let t ← loadParameter src name #[srcDim]
  pure ⟨t.1, ← pad1dTo t.2⟩

private def loadPadded2D
    (src : Source)
    (name : String)
    (srcRows srcCols dstRows dstCols : Nat)
    : IO (SomeDTypeTensor (#[dstRows, dstCols]) device) := do
  let t ← loadParameter src name #[srcRows, srcCols]
  pure ⟨t.1, ← pad2dTo t.2⟩

private def loadRMSNorm
    (src : Source)
    (name : String)
    (srcDim dstDim : Nat)
    (eps : Float)
    : IO (Gemma4RMSNorm (device := device) dstDim) := do
  let w ← loadPadded1D src name srcDim dstDim
  pure (Gemma4RMSNorm.fromCheckpointWeight (freezeStored w) eps)

private def layerHeadDim (cfg : Config) (layerType : LayerType) : Nat :=
  match layerType with
  | .slidingAttention => cfg.head_dim
  | .fullAttention => Config.fullHeadDim cfg

private def layerNumKVHeads (cfg : Config) (layerType : LayerType) : Nat :=
  match layerType with
  | .slidingAttention => cfg.num_key_value_heads
  | .fullAttention => Config.fullNumKVHeads cfg

private def layerAttentionDim (cfg : Config) (layerType : LayerType) : Nat :=
  cfg.num_attention_heads * layerHeadDim cfg layerType

private def layerKVProjDim (cfg : Config) (layerType : LayerType) : Nat :=
  layerNumKVHeads cfg layerType * layerHeadDim cfg layerType

private def loadAttention (src : Source) (cfg : Config) (layerIdx : Nat)
    : IO (Gemma4Attention (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.language_model.layers.{layerIdx}.self_attn"
  let layerType := Config.layerTypeAt cfg layerIdx
  let headDim := layerHeadDim cfg layerType
  let attnDim := layerAttentionDim cfg layerType
  let kvProjDim := layerKVProjDim cfg layerType

  let qProj ←
    loadPadded2D
      src
      s!"{p}.q_proj"
      attnDim cfg.hidden_size
      (Config.maxAttentionDim cfg) cfg.hidden_size
  let kProj ←
    loadPadded2D
      src
      s!"{p}.k_proj"
      kvProjDim cfg.hidden_size
      (Config.maxKVProjDim cfg) cfg.hidden_size
  let vProj ←
    if cfg.attention_k_eq_v && layerType == .fullAttention then
      let support ← dtype.acquireConstantSupport device
      let _ := support.down
      pure ⟨dtype, Tensor.zerosFor ⟨#[Config.maxKVProjDim cfg, cfg.hidden_size], dtype, device⟩⟩
    else
      loadPadded2D
        src
        s!"{p}.v_proj"
        kvProjDim cfg.hidden_size
        (Config.maxKVProjDim cfg) cfg.hidden_size
  let oProj ←
    loadPadded2D
      src
      s!"{p}.o_proj"
      cfg.hidden_size attnDim
      cfg.hidden_size (Config.maxAttentionDim cfg)

  let qNorm ← loadRMSNorm src s!"{p}.q_norm" headDim (Config.maxHeadDim cfg) cfg.rms_norm_eps
  let kNorm ← loadRMSNorm src s!"{p}.k_norm" headDim (Config.maxHeadDim cfg) cfg.rms_norm_eps

  pure {
    q_proj := reqGradFalse (← requireDType dtype qProj)
    k_proj := reqGradFalse (← requireDType dtype kProj)
    v_proj := reqGradFalse (← requireDType dtype vProj)
    o_proj := reqGradFalse (← requireDType dtype oProj)
    q_norm := qNorm
    k_norm := kNorm
  }

private def loadDenseMLP (src : Source) (cfg : Config) (layerIdx : Nat)
    : IO (Gemma4MLP (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.language_model.layers.{layerIdx}.mlp"
  let inter := Config.layerIntermediateSize cfg layerIdx
  let targetInter := Config.maxIntermediateSize cfg
  let gate ←
    loadPadded2D
      src
      s!"{p}.gate_proj"
      inter cfg.hidden_size
      targetInter cfg.hidden_size
  let up ←
    loadPadded2D
      src
      s!"{p}.up_proj"
      inter cfg.hidden_size
      targetInter cfg.hidden_size
  let down ←
    loadPadded2D
      src
      s!"{p}.down_proj"
      cfg.hidden_size inter
      cfg.hidden_size targetInter
  pure {
    gate_proj := reqGradFalse (← requireDType dtype gate)
    up_proj := reqGradFalse (← requireDType dtype up)
    down_proj := reqGradFalse (← requireDType dtype down)
  }

private def loadPerLayerInputBlock (src : Source) (cfg : Config) (layerIdx : Nat)
    : IO (Gemma4PerLayerInputBlock (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.language_model.layers.{layerIdx}"
  let gate ← loadParameter src s!"{p}.per_layer_input_gate" #[cfg.hidden_size_per_layer_input, cfg.hidden_size]
  let proj ← loadParameter src s!"{p}.per_layer_projection" #[cfg.hidden_size, cfg.hidden_size_per_layer_input]
  let norm ← loadRMSNorm src s!"{p}.post_per_layer_input_norm" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  pure {
    per_layer_input_gate := reqGradFalse (← requireDType dtype gate)
    per_layer_projection := reqGradFalse (← requireDType dtype proj)
    post_per_layer_input_norm := norm
  }

private def loadMoeBranch (src : Source) (cfg : Config) (layerIdx : Nat)
    : IO (Gemma4MoeBranch (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.language_model.layers.{layerIdx}"
  let routerProj ← loadParameter src s!"{p}.router.proj" #[cfg.num_experts, cfg.hidden_size]
  let routerScale ← loadParameter src s!"{p}.router.scale" #[cfg.hidden_size]
  let perExpertScale ← loadParameter src s!"{p}.router.per_expert_scale" #[cfg.num_experts]
  let gateUp ←
    loadParameter
      src
      s!"{p}.experts.gate_up_proj"
      #[cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size]
  let down ←
    loadParameter
      src
      s!"{p}.experts.down_proj"
      #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size]
  let post1 ← loadRMSNorm src s!"{p}.post_feedforward_layernorm_1" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  let post2 ← loadRMSNorm src s!"{p}.post_feedforward_layernorm_2" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  let pre2 ← loadRMSNorm src s!"{p}.pre_feedforward_layernorm_2" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  pure {
    router := {
      proj := reqGradFalse routerProj.2.toFloat32
      scale := reqGradFalse routerScale.2.toFloat32
      per_expert_scale := reqGradFalse perExpertScale.2.toFloat32
    }
    experts := {
      gate_up_proj := reqGradFalse (← requireDType dtype gateUp)
      down_proj := reqGradFalse (← requireDType dtype down)
    }
    post_feedforward_layernorm_1 := post1
    post_feedforward_layernorm_2 := post2
    pre_feedforward_layernorm_2 := pre2
  }

private def loadLayer (src : Source) (cfg : Config) (layerIdx : Nat)
    : IO (Gemma4Layer (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.language_model.layers.{layerIdx}"
  let layerType := Config.layerTypeAt cfg layerIdx
  let selfAttn ← loadAttention src cfg layerIdx
  let mlp ← loadDenseMLP src cfg layerIdx
  let inputNorm ← loadRMSNorm src s!"{p}.input_layernorm" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  let postAttnNorm ← loadRMSNorm src s!"{p}.post_attention_layernorm" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  let preFfnNorm ← loadRMSNorm src s!"{p}.pre_feedforward_layernorm" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  let postFfnNorm ← loadRMSNorm src s!"{p}.post_feedforward_layernorm" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps
  let perLayerInput ←
    if Config.hasPerLayerInput cfg then
      some <$> loadPerLayerInputBlock src cfg layerIdx
    else
      pure none
  let moe ←
    if Config.isMoE cfg then
      some <$> loadMoeBranch src cfg layerIdx
    else
      pure none
  let layerScalar ← loadParameter src s!"{p}.layer_scalar" #[1]
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
    layer_scalar := reqGradFalse (← requireDType dtype layerScalar)
  }

namespace Gemma4ForCausalLM

/-- Load Gemma 4 model from a SafeTensors file or a directory of shards. -/
def load (path : String) (cfg : Config := Config.gemma4_E4B)
    (log : Handlers := {}) (device : Device := .CPU)
    : IO (Σ dtype : RealFloatingDType, Gemma4ForCausalLM (device := device) cfg dtype.val) := do
  log.onInfo s!"Loading Gemma4ForCausalLM (device := device) from {path}..."
  let src ← Source.open path

  let embedTokens ←
    loadParameter
      src
      "model.language_model.embed_tokens"
      #[cfg.vocab_size, cfg.hidden_size]
  let dtype := embedTokens.1
  let floating ← IO.ofExcept dtype.realFloating?
  let _ := floating.down

  let embedTokensPerLayer ←
    if Config.hasPerLayerInput cfg then
      some <$> loadParameter
        src
        "model.language_model.embed_tokens_per_layer"
        #[cfg.vocab_size_per_layer_input, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input]
    else
      pure none

  let perLayerModelProjection ←
    if Config.hasPerLayerInput cfg then
      some <$> loadParameter
        src
        "model.language_model.per_layer_model_projection"
        #[cfg.num_hidden_layers * cfg.hidden_size_per_layer_input, cfg.hidden_size]
    else
      pure none

  let perLayerProjectionNorm ←
    if Config.hasPerLayerInput cfg then
      some <$> loadRMSNorm
        src
        "model.language_model.per_layer_projection_norm"
        cfg.hidden_size_per_layer_input
        cfg.hidden_size_per_layer_input
        cfg.rms_norm_eps
    else
      pure none

  let indices : Vector cfg.num_hidden_layers Nat := ⟨Array.range cfg.num_hidden_layers, Array.size_range⟩
  let layers ← indices.mapM fun i => do
    let layer ← loadLayer src cfg i
    if (i + 1) % 8 == 0 || i + 1 == cfg.num_hidden_layers then
      log.onInfo s!"  loaded layers {i + 1}/{cfg.num_hidden_layers}"
    pure layer

  let norm ← loadRMSNorm src "model.language_model.norm" cfg.hidden_size cfg.hidden_size cfg.rms_norm_eps

  let model : Gemma4Model (dtype := dtype) (device := device) cfg := {
    embed_tokens := reqGradFalse (← requireDType dtype embedTokens)
    embed_tokens_per_layer := (← embedTokensPerLayer.mapM fun value => do pure (reqGradFalse (← requireDType dtype value)))
    per_layer_model_projection := (← perLayerModelProjection.mapM fun value => do pure (reqGradFalse (← requireDType dtype value)))
    per_layer_projection_norm := perLayerProjectionNorm
    layers := layers
    norm := norm
  }

  let lmHeadOpt ← tryLoadParameter src "lm_head" #[cfg.vocab_size, cfg.hidden_size]
  let lmHead ←
    match lmHeadOpt with
    | some w => pure (reqGradFalse (← requireDType dtype w))
    | none =>
      if cfg.tie_word_embeddings then
        pure (reqGradFalse model.embed_tokens)
      else
        throw <| IO.userError "lm_head not found and tie_word_embeddings=false"

  log.onInfo "Loaded Gemma4ForCausalLM (device := device) weights."
  pure ⟨RealFloatingDType.ofDType dtype, {
    model := model
    lmHead := lmHead
    tieWordEmbeddings := cfg.tie_word_embeddings
  }⟩

end Gemma4ForCausalLM

end torch.gemma4
