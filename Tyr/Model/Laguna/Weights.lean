/-
  Tyr/Model/Laguna/Weights.lean

  Pretrained weight loading for poolside Laguna text causal-LM
  (Laguna-S-2.1) from HuggingFace SafeTensors (single-file and sharded).

  Two checkpoint serializations are supported (detected per MoE layer):

  A. BF16 exports (e.g. `Tests/fixtures/laguna/tiny/model.safetensors`):
     everything is BF16 with standard `modeling_laguna.py` state_dict names,
     and the routed experts are FUSED dense 3-D banks
     `mlp.experts.gate_up_proj` `[E, 2*moeInt, hidden]` (dim 1: first half
     gate, second half up — HF `chunk(2, dim=-1)`) and
     `mlp.experts.down_proj` `[E, hidden, moeInt]`. These load into
     `LagunaDenseExperts` (see MoE.lean).

  B. Real NVFP4 (`poolside/Laguna-S-2.1-NVFP4`, 14 shards):
     non-expert tensors are BF16 with the same standard names; routed experts
     are PER-EXPERT NVFP4-packed
     `mlp.experts.{i}.{gate,up,down}_proj.{weight_packed,weight_scale,weight_global_scale}`
     (U8 `[out, in/2]`, F8_E4M3 `[out, in/16]`, F32 scalar). The 256 experts
     are loaded on CPU, stacked into the nine `[E, ...]` banks of
     `LagunaPackedExperts`, and moved to the target device once per layer
     (layer-by-layer, so peak memory stays ~1 per-layer bank set instead of a
     full-model CPU copy). Experts stay packed — dequantization happens
     per-forward in the MoE block.

  Ignored checkpoint tensors: `self_attn.{k,v}_scale` (BF16 scalars used by
  vLLM KV quantization) and `*.input_global_scale` (activation quantization
  scales) — the tyr runtime consumes neither.

  Router bias naming: transformers `modeling_laguna.py` stores the
  aux-loss-free selection bias as `mlp.gate.e_score_correction_bias`, while
  the vLLM-trained NVFP4 checkpoint stores it as
  `mlp.experts.e_score_correction_bias` (HF remaps it on load). Both names
  are tried; the bias is optional (absent ⇒ no bias).

  Sharded loading: `Source.open` reads `model.safetensors.index.json` and the
  header of each listed shard once per directory, so each per-tensor load
  below is a hash lookup plus one file read.
-/
import Tyr.Log
import Tyr.SafeTensors.Source
import Tyr.Model.Utils
import Tyr.Model.Laguna.Model

namespace torch.laguna

variable {device : Device}
open torch.Log
open torch.Model (reqGradFalse)
open torch.safetensors (pushUnique Source)

/-- Candidate names for one checkpoint tensor, tolerant of common
    wrapper prefixes (mirrors the Qwen35 loader). -/
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

private def freezeStored {shape : Shape} (tensor : SomeDTypeTensor shape device) :
    SomeDTypeTensor shape device := ⟨tensor.1, tensor.2.detach⟩

private def stackBank (dtype : DType) (tensors : Array (ShapePackedTensor dtype .CPU))
    (shape : Shape) (device : Device) : IO (SomeDTypeTensor shape device) := do
  let tensor ← IO.ofExcept (Tensor.catMany? tensors 0 shape)
  let raw ← RawTensor.convertIO shape tensor.raw device none
  let moved := Tensor.ofRaw! (σ := ⟨shape, dtype, device⟩) raw
  pure ⟨dtype, moved.detach⟩

/-- Load one attention block (`numHeads` is the per-layer query head count).
    `self_attn.{k,v}_scale` scalars are intentionally not loaded. -/
private def loadAttention (src : Source) (cfg : Config) (layerIdx numHeads : Nat)
    (device : Device) (dtype : DType) : IO (LagunaAttention (device := device) cfg numHeads dtype) := do
  let p := s!"model.layers.{layerIdx}.self_attn"
  let q ← src.loadAs (← src.resolve (nameCandidates s!"{p}.q_proj.weight"))
    ⟨#[numHeads * cfg.head_dim, cfg.hidden_size], dtype⟩ device
  let k ← src.loadAs (← src.resolve (nameCandidates s!"{p}.k_proj.weight"))
    ⟨#[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size], dtype⟩ device
  let v ← src.loadAs (← src.resolve (nameCandidates s!"{p}.v_proj.weight"))
    ⟨#[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size], dtype⟩ device
  let o ← src.loadAs (← src.resolve (nameCandidates s!"{p}.o_proj.weight"))
    ⟨#[cfg.hidden_size, numHeads * cfg.head_dim], dtype⟩ device
  let g ← src.loadAs (← src.resolve (nameCandidates s!"{p}.g_proj.weight"))
    ⟨#[numHeads, cfg.hidden_size], dtype⟩ device
  let qn ← src.load (← src.resolve (nameCandidates s!"{p}.q_norm.weight")) #[cfg.head_dim] device
  let kn ← src.load (← src.resolve (nameCandidates s!"{p}.k_norm.weight")) #[cfg.head_dim] device
  pure {
    q_proj := reqGradFalse q
    k_proj := reqGradFalse k
    v_proj := reqGradFalse v
    o_proj := reqGradFalse o
    g_proj := reqGradFalse g
    q_norm := freezeStored qn
    k_norm := freezeStored kn
  }

/-- Load the dense SwiGLU MLP of a `mlp_only_layers` layer. -/
private def loadDenseMLP (src : Source) (cfg : Config) (layerIdx : Nat)
    (device : Device) (dtype : DType) : IO (LagunaMLP (device := device) cfg dtype) := do
  let p := s!"model.layers.{layerIdx}.mlp"
  let gate ← src.loadAs (← src.resolve (nameCandidates s!"{p}.gate_proj.weight"))
    ⟨#[cfg.intermediate_size, cfg.hidden_size], dtype⟩ device
  let up ← src.loadAs (← src.resolve (nameCandidates s!"{p}.up_proj.weight"))
    ⟨#[cfg.intermediate_size, cfg.hidden_size], dtype⟩ device
  let down ← src.loadAs (← src.resolve (nameCandidates s!"{p}.down_proj.weight"))
    ⟨#[cfg.hidden_size, cfg.intermediate_size], dtype⟩ device
  pure {
    gate_proj := reqGradFalse gate
    up_proj := reqGradFalse up
    down_proj := reqGradFalse down
  }

/-- Unused placeholder packed banks for the dense-BF16 expert layout
    (`forward2d` reads `denseExperts` instead). -/
private def placeholderPackedExperts (cfg : Config) (device : Device) : LagunaPackedExperts (device := device) cfg :=
  let e := cfg.num_experts
  let mi := cfg.moe_intermediate_size
  let h := cfg.hidden_size
  -- Broadcast views keep the unused placeholders small while retaining true shapes.
  let z := qwen35.zerosDynamic #[] false device
  {
    gatePacked := ⟨.Float32, Tensor.broadcastTo z #[e, mi, h / 2]⟩
    gateScale := ⟨.Float32, Tensor.broadcastTo z #[e, mi, h / 16]⟩
    gateGlobal := ⟨.Float32, Tensor.broadcastTo z #[e]⟩
    upPacked := ⟨.Float32, Tensor.broadcastTo z #[e, mi, h / 2]⟩
    upScale := ⟨.Float32, Tensor.broadcastTo z #[e, mi, h / 16]⟩
    upGlobal := ⟨.Float32, Tensor.broadcastTo z #[e]⟩
    downPacked := ⟨.Float32, Tensor.broadcastTo z #[e, h, mi / 2]⟩
    downScale := ⟨.Float32, Tensor.broadcastTo z #[e, h, mi / 16]⟩
    downGlobal := ⟨.Float32, Tensor.broadcastTo z #[e]⟩
  }

/-- Load and stack the per-expert NVFP4-packed banks of one MoE layer
    (layout B). Expert tensors load on CPU, are stacked into `[E, ...]`
    banks with one `cat` per bank, and the nine banks move to `device` in
    one go — peak extra memory is one layer's banks (~1.4 GB for S-2.1). -/
private def loadPackedExpertBanks (src : Source) (p : String) (cfg : Config)
    (device : Device) : IO (LagunaPackedExperts (device := device) cfg) := do
  let e := cfg.num_experts
  let mi := cfg.moe_intermediate_size
  let h := cfg.hidden_size
  let mut gateP : Array (ShapePackedTensor .UInt8 .CPU) := Array.mkEmpty e
  let mut gateS : Array (ShapePackedTensor .Float8E4M3FN .CPU) := Array.mkEmpty e
  let mut gateG : Array (ShapePackedTensor .Float32 .CPU) := Array.mkEmpty e
  let mut upP : Array (ShapePackedTensor .UInt8 .CPU) := Array.mkEmpty e
  let mut upS : Array (ShapePackedTensor .Float8E4M3FN .CPU) := Array.mkEmpty e
  let mut upG : Array (ShapePackedTensor .Float32 .CPU) := Array.mkEmpty e
  let mut downP : Array (ShapePackedTensor .UInt8 .CPU) := Array.mkEmpty e
  let mut downS : Array (ShapePackedTensor .Float8E4M3FN .CPU) := Array.mkEmpty e
  let mut downG : Array (ShapePackedTensor .Float32 .CPU) := Array.mkEmpty e
  for i in [:e] do
    let ep := s!"{p}.experts.{i}"
    -- gate_proj / up_proj: packed [mi, h/2], scales [mi, h/16], global scalar.
    let gp ← src.loadAs s!"{ep}.gate_proj.weight_packed" ⟨#[mi, h / 2], .UInt8⟩
    let gs ← src.loadAs s!"{ep}.gate_proj.weight_scale" ⟨#[mi, h / 16], .Float8E4M3FN⟩
    let gg ← src.loadAs s!"{ep}.gate_proj.weight_global_scale" ⟨#[], .Float32⟩
    let up ← src.loadAs s!"{ep}.up_proj.weight_packed" ⟨#[mi, h / 2], .UInt8⟩
    let us ← src.loadAs s!"{ep}.up_proj.weight_scale" ⟨#[mi, h / 16], .Float8E4M3FN⟩
    let ug ← src.loadAs s!"{ep}.up_proj.weight_global_scale" ⟨#[], .Float32⟩
    -- down_proj: packed [h, mi/2], scales [h, mi/16], global scalar.
    let dp ← src.loadAs s!"{ep}.down_proj.weight_packed" ⟨#[h, mi / 2], .UInt8⟩
    let ds ← src.loadAs s!"{ep}.down_proj.weight_scale" ⟨#[h, mi / 16], .Float8E4M3FN⟩
    let dg ← src.loadAs s!"{ep}.down_proj.weight_global_scale" ⟨#[], .Float32⟩
    gateP := gateP.push ((Tensor.reshape gp #[1, mi, h / 2]).packShape)
    gateS := gateS.push ((Tensor.reshape gs #[1, mi, h / 16]).packShape)
    gateG := gateG.push (Tensor.reshape gg #[1]).packShape
    upP := upP.push ((Tensor.reshape up #[1, mi, h / 2]).packShape)
    upS := upS.push ((Tensor.reshape us #[1, mi, h / 16]).packShape)
    upG := upG.push (Tensor.reshape ug #[1]).packShape
    downP := downP.push ((Tensor.reshape dp #[1, h, mi / 2]).packShape)
    downS := downS.push ((Tensor.reshape ds #[1, h, mi / 16]).packShape)
    downG := downG.push (Tensor.reshape dg #[1]).packShape
  pure {
    gatePacked := (← stackBank .UInt8 gateP #[e, mi, h / 2] device)
    gateScale := (← stackBank .Float8E4M3FN gateS #[e, mi, h / 16] device)
    gateGlobal := (← stackBank .Float32 gateG #[e] device)
    upPacked := (← stackBank .UInt8 upP #[e, mi, h / 2] device)
    upScale := (← stackBank .Float8E4M3FN upS #[e, mi, h / 16] device)
    upGlobal := (← stackBank .Float32 upG #[e] device)
    downPacked := (← stackBank .UInt8 downP #[e, h, mi / 2] device)
    downScale := (← stackBank .Float8E4M3FN downS #[e, h, mi / 16] device)
    downGlobal := (← stackBank .Float32 downG #[e] device)
  }

/-- Load one sparse MoE block, auto-detecting the expert serialization:
    fused dense BF16 banks (layout A) when `mlp.experts.gate_up_proj`
    resolves, otherwise per-expert NVFP4-packed tensors (layout B). -/
private def loadSparseMoe (src : Source) (cfg : Config) (layerIdx : Nat)
    (device : Device) (dtype : DType) : IO (LagunaSparseMoeBlock (device := device) cfg dtype) := do
  let p := s!"model.layers.{layerIdx}.mlp"
  let e := cfg.num_experts
  let mi := cfg.moe_intermediate_size
  let h := cfg.hidden_size

  let routerW ← src.load (← src.resolve (nameCandidates s!"{p}.gate.weight")) #[e, h] device
  -- Optional selection bias; transformers and vLLM-trained checkpoints use
  -- different keys (see module doc).
  let biasOpt ← (src.find? (nameCandidates s!"{p}.gate.e_score_correction_bias"
      ++ nameCandidates s!"{p}.experts.e_score_correction_bias")).mapM
    (src.load · #[e] device)

  let sharedGate ← src.loadAs (← src.resolve (nameCandidates s!"{p}.shared_expert.gate_proj.weight"))
    ⟨#[cfg.shared_expert_intermediate_size, h], dtype⟩ device
  let sharedUp ← src.loadAs (← src.resolve (nameCandidates s!"{p}.shared_expert.up_proj.weight"))
    ⟨#[cfg.shared_expert_intermediate_size, h], dtype⟩ device
  let sharedDown ← src.loadAs (← src.resolve (nameCandidates s!"{p}.shared_expert.down_proj.weight"))
    ⟨#[h, cfg.shared_expert_intermediate_size], dtype⟩ device

  let guOpt ← (src.find? (nameCandidates s!"{p}.experts.gate_up_proj")).mapM
    (src.loadAs · ⟨#[e, 2 * mi, h], dtype⟩ device)
  let (experts, denseExperts) ←
    match guOpt with
    | some gu => do
      -- Layout A: fused dense banks; dim-1 first half gate, second half up.
      let down ← src.loadAs (← src.resolve (nameCandidates s!"{p}.experts.down_proj")) ⟨#[e, h, mi], dtype⟩ device
      let gate3 : Tensor { device := device, dtype := dtype, shape := #[e, mi, h] } := Tensor.reshape (Tensor.slice gu 1 0 mi) #[e, mi, h]
      let up3 : Tensor { device := device, dtype := dtype, shape := #[e, mi, h] } := Tensor.reshape (Tensor.slice gu 1 mi mi) #[e, mi, h]
      let dense : LagunaDenseExperts (device := device) cfg dtype := {
        gateProj := reqGradFalse gate3
        upProj := reqGradFalse up3
        downProj := reqGradFalse down
      }
      pure (placeholderPackedExperts cfg device, some dense)
    | none => do
      -- Layout B: per-expert NVFP4-packed tensors, stacked into banks.
      let packed ← loadPackedExpertBanks src p cfg device
      pure (packed, none)

  pure {
    router := { weight := freezeStored routerW, eScoreCorrectionBias := biasOpt.map freezeStored }
    experts := experts
    denseExperts := denseExperts
    sharedGateProj := reqGradFalse sharedGate
    sharedUpProj := reqGradFalse sharedUp
    sharedDownProj := reqGradFalse sharedDown
  }

/-- Load decoder layer `layerIdx`, with the attention and FFN variants its
    type selects. -/
private def loadLayer (src : Source) (cfg : Config) (layerIdx : Nat)
    (device : Device) (dtype : DType) : IO (LagunaLayer (device := device) cfg dtype layerIdx) := do
  let p := s!"model.layers.{layerIdx}"
  let inputNorm ← src.load (← src.resolve (nameCandidates s!"{p}.input_layernorm.weight"))
    #[cfg.hidden_size] device
  let postNorm ← src.load (← src.resolve (nameCandidates s!"{p}.post_attention_layernorm.weight"))
    #[cfg.hidden_size] device

  let attn ← Select.build (cfg.isFullAttentionLayer layerIdx)
    (loadAttention src cfg layerIdx cfg.num_attention_heads_sliding device dtype)
    (loadAttention src cfg layerIdx cfg.num_attention_heads device dtype)
  let ffn ← Select.build (cfg.isMoELayer layerIdx)
    (loadDenseMLP src cfg layerIdx device dtype)
    (loadSparseMoe src cfg layerIdx device dtype)

  pure {
    input_layernorm := freezeStored inputNorm
    attn := attn
    post_attention_layernorm := freezeStored postNorm
    ffn := ffn
  }

/-- Shared loader core over an abstract weight source. -/
private def loadCore (src : Source) (desc : String) (cfg : Config)
    (device : Device) (log : Handlers) : IO (Σ dtype : RealFloatingDType, LagunaForCausalLM (device := device) cfg dtype.val) := do
  log.onInfo s!"Loading LagunaForCausalLM from {desc}..."

  let embedTokens ← src.load (← src.resolve (nameCandidates "model.embed_tokens.weight"))
    #[cfg.vocab_size, cfg.hidden_size] device

  let dtype := embedTokens.1
  let floating ← IO.ofExcept (DType.realFloating? dtype)
  let layers ← IndexedList.ofFnM (α := LagunaLayer (device := device) cfg dtype) (fun i => do
    let layer ← loadLayer src cfg i device dtype
    if (i + 1) % 4 == 0 || i + 1 == cfg.num_hidden_layers then
      log.onInfo s!"  loaded layers {i + 1}/{cfg.num_hidden_layers}"
    pure layer) 0 cfg.num_hidden_layers

  let normW ← src.load (← src.resolve (nameCandidates "model.norm.weight")) #[cfg.hidden_size] device

  let model : LagunaModel (device := device) cfg dtype := {
    embed_tokens := reqGradFalse embedTokens.2
    layers := layers
    norm := freezeStored normW
  }

  let lmHeadOpt ← (src.find? (nameCandidates "lm_head.weight")).mapM
    (src.loadAs · ⟨#[cfg.vocab_size, cfg.hidden_size], dtype⟩ device)
  let (lmHead, tieWordEmbeddings) ←
    match lmHeadOpt with
    | some w => pure (reqGradFalse w, false)
    | none => do
      log.onInfo "  lm_head.weight not found; tying to embeddings."
      pure (reqGradFalse model.embed_tokens, true)

  log.onInfo "Loaded LagunaForCausalLM weights."
  pure ⟨⟨dtype, floating.down.valid⟩,
    { model := model, lmHead := lmHead, tieWordEmbeddings := tieWordEmbeddings }⟩

namespace LagunaForCausalLM

/-- Load a Laguna model from a SafeTensors file or a directory of shards.
    Routed experts are loaded on CPU and moved to `device` layer by layer. -/
def load (path : String) (cfg : Config := Config.laguna_s_2_1)
    (device : Device := Device.CPU)
    (log : Handlers := {})
    : IO (Σ dtype : RealFloatingDType, LagunaForCausalLM (device := device) cfg dtype.val) := do
  loadCore (← Source.open path) path cfg device log

end LagunaForCausalLM

end torch.laguna
