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

  Sharded-load performance: the C++ loader (`cc/src/tyr.cpp`
  `ShardedSafeTensorsDir`) parses all shard headers ONCE per directory and
  caches them (it never reads `model.safetensors.index.json`), so each
  per-tensor load below is an O(1) hash lookup plus one file read. The full
  NVFP4 checkpoint is ~145k tensors; loading is dominated by reading ~53 GB
  of packed expert bytes from disk, not by lookup overhead.
-/
import Tyr.Log
import Tyr.SafeTensors.Load
import Tyr.Model.Utils
import Tyr.Model.Laguna.Model

namespace torch.laguna

variable {device : Device}
open torch.Log
open torch.Model (reqGradFalse)
open torch.safetensors (pushUnique)

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

/-- Checkpoint tensor source: one SafeTensors file or a sharded directory.
    Unifies the single-file/sharded loader pairs so each component loader is
    written once (the Qwen35 loader duplicates them by hand). -/
private inductive WeightSource where
  | single (path : String)
  | sharded (dir : String)

namespace WeightSource

/-- Load `name` with expected shape `s` (numel-checked, reshaped in C++). -/
def load (src : WeightSource) (name : String) (s : Shape)
    (device : Device := Device.CPU) : IO (SomeDTypeTensor s device) :=
  match src with
  | .single path => torch.safetensors.loadTensorExact path name s device
  | .sharded dir => torch.safetensors.loadTensorShardedExact dir name s device

/-- Missing names are optional; malformed tensors and I/O failures remain errors. -/
def tryLoad (src : WeightSource) (name : String) (s : Shape)
    (device : Device := Device.CPU) : IO (Option (SomeDTypeTensor s device)) :=
  match src with
  | .single path => torch.safetensors.loadOptionalTensorExact path name s device
  | .sharded dir => torch.safetensors.loadOptionalTensorShardedExact dir name s device

/-- Global scales are scalar checkpoint values; normalize their single element
    to a length-one bank entry without changing its dtype. -/
def loadGlobal (src : WeightSource) (name : String) : IO (SomeDTypeTensor #[1] .CPU) := do
  let raw ← match src with
    | .single path => RawTensor.safeTensorsLoad path name #[1]
    | .sharded dir => RawTensor.safeTensorsLoadSharded dir name #[1]
  IO.ofExcept (SomeDTypeTensor.ofRaw raw)

/-- Try each of `names` in order; `none` if none loads. -/
def tryLoadCandidates (src : WeightSource) (names : Array String) (s : Shape)
    (device : Device := Device.CPU) : IO (Option (SomeDTypeTensor s device)) := do
  for n in names do
    if let some t ← src.tryLoad n s device then
      return some t
  pure none

/-- Load the first of `names` that resolves; throws listing all candidates
    if none loads. -/
def loadCandidates (src : WeightSource) (names : Array String) (s : Shape)
    (device : Device := Device.CPU) : IO (SomeDTypeTensor s device) := do
  match (← src.tryLoadCandidates names s device) with
  | some t => pure t
  | none => throw <| IO.userError s!"Failed to load tensor: {names}"

end WeightSource

private def freezeStored {shape : Shape} (tensor : SomeDTypeTensor shape device) :
    SomeDTypeTensor shape device := ⟨tensor.1, tensor.2.detach⟩

private def requireDType {shape : Shape} (dtype : DType)
    (tensor : SomeDTypeTensor shape device) : IO (Tensor ⟨shape, dtype, device⟩) :=
  IO.ofExcept (tensor.atDType? dtype)

private def stackBank (dtype : DType) (tensors : Array (ShapePackedTensor dtype .CPU))
    (shape : Shape) (device : Device) : IO (SomeDTypeTensor shape device) := do
  let tensor ← IO.ofExcept (Tensor.catMany? tensors 0 shape)
  let raw ← RawTensor.convertIO shape tensor.raw device none
  let moved ← IO.ofExcept (Tensor.ofRaw (σ := ⟨shape, dtype, device⟩) raw)
  pure ⟨dtype, moved.detach⟩

/-- Load one attention block (`numHeads` is the per-layer query head count).
    `self_attn.{k,v}_scale` scalars are intentionally not loaded. -/
private def loadAttention (src : WeightSource) (cfg : Config) (layerIdx numHeads : Nat)
    (device : Device) (dtype : DType) : IO (LagunaAttention (device := device) cfg numHeads dtype) := do
  let p := s!"model.layers.{layerIdx}.self_attn"
  let q ← src.loadCandidates (nameCandidates s!"{p}.q_proj.weight")
    #[numHeads * cfg.head_dim, cfg.hidden_size] device
  let k ← src.loadCandidates (nameCandidates s!"{p}.k_proj.weight")
    #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size] device
  let v ← src.loadCandidates (nameCandidates s!"{p}.v_proj.weight")
    #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size] device
  let o ← src.loadCandidates (nameCandidates s!"{p}.o_proj.weight")
    #[cfg.hidden_size, numHeads * cfg.head_dim] device
  let g ← src.loadCandidates (nameCandidates s!"{p}.g_proj.weight")
    #[numHeads, cfg.hidden_size] device
  let qn ← src.loadCandidates (nameCandidates s!"{p}.q_norm.weight") #[cfg.head_dim] device
  let kn ← src.loadCandidates (nameCandidates s!"{p}.k_norm.weight") #[cfg.head_dim] device
  pure {
    q_proj := reqGradFalse (← requireDType dtype q)
    k_proj := reqGradFalse (← requireDType dtype k)
    v_proj := reqGradFalse (← requireDType dtype v)
    o_proj := reqGradFalse (← requireDType dtype o)
    g_proj := reqGradFalse (← requireDType dtype g)
    q_norm := freezeStored qn
    k_norm := freezeStored kn
  }

/-- Load the dense SwiGLU MLP of a `mlp_only_layers` layer. -/
private def loadDenseMLP (src : WeightSource) (cfg : Config) (layerIdx : Nat)
    (device : Device) (dtype : DType) : IO (LagunaMLP (device := device) cfg dtype) := do
  let p := s!"model.layers.{layerIdx}.mlp"
  let gate ← src.loadCandidates (nameCandidates s!"{p}.gate_proj.weight")
    #[cfg.intermediate_size, cfg.hidden_size] device
  let up ← src.loadCandidates (nameCandidates s!"{p}.up_proj.weight")
    #[cfg.intermediate_size, cfg.hidden_size] device
  let down ← src.loadCandidates (nameCandidates s!"{p}.down_proj.weight")
    #[cfg.hidden_size, cfg.intermediate_size] device
  pure {
    gate_proj := reqGradFalse (← requireDType dtype gate)
    up_proj := reqGradFalse (← requireDType dtype up)
    down_proj := reqGradFalse (← requireDType dtype down)
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
    gatePacked := ⟨.Float32, Tensor.expand z #[e, mi, h / 2]⟩
    gateScale := ⟨.Float32, Tensor.expand z #[e, mi, h / 16]⟩
    gateGlobal := ⟨.Float32, Tensor.expand z #[e]⟩
    upPacked := ⟨.Float32, Tensor.expand z #[e, mi, h / 2]⟩
    upScale := ⟨.Float32, Tensor.expand z #[e, mi, h / 16]⟩
    upGlobal := ⟨.Float32, Tensor.expand z #[e]⟩
    downPacked := ⟨.Float32, Tensor.expand z #[e, h, mi / 2]⟩
    downScale := ⟨.Float32, Tensor.expand z #[e, h, mi / 16]⟩
    downGlobal := ⟨.Float32, Tensor.expand z #[e]⟩
  }

/-- Load and stack the per-expert NVFP4-packed banks of one MoE layer
    (layout B). Expert tensors load on CPU, are stacked into `[E, ...]`
    banks with one `cat` per bank, and the nine banks move to `device` in
    one go — peak extra memory is one layer's banks (~1.4 GB for S-2.1). -/
private def loadPackedExpertBanks (src : WeightSource) (p : String) (cfg : Config)
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
    let gp ← src.load s!"{ep}.gate_proj.weight_packed" #[mi, h / 2] Device.CPU
    let gp ← requireDType .UInt8 gp
    let gs ← src.load s!"{ep}.gate_proj.weight_scale" #[mi, h / 16] Device.CPU
    let gs ← requireDType .Float8E4M3FN gs
    let gg ← src.loadGlobal s!"{ep}.gate_proj.weight_global_scale"
    let gg ← requireDType .Float32 gg
    let up ← src.load s!"{ep}.up_proj.weight_packed" #[mi, h / 2] Device.CPU
    let up ← requireDType .UInt8 up
    let us ← src.load s!"{ep}.up_proj.weight_scale" #[mi, h / 16] Device.CPU
    let us ← requireDType .Float8E4M3FN us
    let ug ← src.loadGlobal s!"{ep}.up_proj.weight_global_scale"
    let ug ← requireDType .Float32 ug
    -- down_proj: packed [h, mi/2], scales [h, mi/16], global scalar.
    let dp ← src.load s!"{ep}.down_proj.weight_packed" #[h, mi / 2] Device.CPU
    let dp ← requireDType .UInt8 dp
    let ds ← src.load s!"{ep}.down_proj.weight_scale" #[h, mi / 16] Device.CPU
    let ds ← requireDType .Float8E4M3FN ds
    let dg ← src.loadGlobal s!"{ep}.down_proj.weight_global_scale"
    let dg ← requireDType .Float32 dg
    gateP := gateP.push ((Tensor.reshape gp #[1, mi, h / 2]).packShape)
    gateS := gateS.push ((Tensor.reshape gs #[1, mi, h / 16]).packShape)
    gateG := gateG.push gg.packShape
    upP := upP.push ((Tensor.reshape up #[1, mi, h / 2]).packShape)
    upS := upS.push ((Tensor.reshape us #[1, mi, h / 16]).packShape)
    upG := upG.push ug.packShape
    downP := downP.push ((Tensor.reshape dp #[1, h, mi / 2]).packShape)
    downS := downS.push ((Tensor.reshape ds #[1, h, mi / 16]).packShape)
    downG := downG.push dg.packShape
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
private def loadSparseMoe (src : WeightSource) (cfg : Config) (layerIdx : Nat)
    (device : Device) (dtype : DType) : IO (LagunaSparseMoeBlock (device := device) cfg dtype) := do
  let p := s!"model.layers.{layerIdx}.mlp"
  let e := cfg.num_experts
  let mi := cfg.moe_intermediate_size
  let h := cfg.hidden_size

  let routerW ← src.loadCandidates (nameCandidates s!"{p}.gate.weight") #[e, h] device
  -- Optional selection bias; transformers and vLLM-trained checkpoints use
  -- different keys (see module doc).
  let biasOpt ← src.tryLoadCandidates
    (nameCandidates s!"{p}.gate.e_score_correction_bias"
      ++ nameCandidates s!"{p}.experts.e_score_correction_bias")
    #[e] device

  let sharedGate ← src.loadCandidates (nameCandidates s!"{p}.shared_expert.gate_proj.weight")
    #[cfg.shared_expert_intermediate_size, h] device
  let sharedUp ← src.loadCandidates (nameCandidates s!"{p}.shared_expert.up_proj.weight")
    #[cfg.shared_expert_intermediate_size, h] device
  let sharedDown ← src.loadCandidates (nameCandidates s!"{p}.shared_expert.down_proj.weight")
    #[h, cfg.shared_expert_intermediate_size] device

  let guOpt ← src.tryLoadCandidates (nameCandidates s!"{p}.experts.gate_up_proj")
    #[e, 2 * mi, h] device
  let (experts, denseExperts) ←
    match guOpt with
    | some gu => do
      let gu ← requireDType dtype gu
      -- Layout A: fused dense banks; dim-1 first half gate, second half up.
      let down ← src.loadCandidates (nameCandidates s!"{p}.experts.down_proj") #[e, h, mi] device
      let gate3 : Tensor { device := device, dtype := dtype, shape := #[e, mi, h] } := Tensor.reshape (sliceTyped gu 1 0 mi) #[e, mi, h]
      let up3 : Tensor { device := device, dtype := dtype, shape := #[e, mi, h] } := Tensor.reshape (sliceTyped gu 1 mi mi) #[e, mi, h]
      let dense : LagunaDenseExperts (device := device) cfg dtype := {
        gateProj := reqGradFalse gate3
        upProj := reqGradFalse up3
        downProj := reqGradFalse (← requireDType dtype down)
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
    sharedGateProj := reqGradFalse (← requireDType dtype sharedGate)
    sharedUpProj := reqGradFalse (← requireDType dtype sharedUp)
    sharedDownProj := reqGradFalse (← requireDType dtype sharedDown)
  }

/-- Load one decoder layer: exactly one attention variant (by
    `Config.layerType`) and one FFN variant (by `Config.isDenseMlpLayer`). -/
private def loadLayer (src : WeightSource) (cfg : Config) (layerIdx : Nat)
    (device : Device) (dtype : DType) : IO (LagunaLayer (device := device) cfg dtype) := do
  let p := s!"model.layers.{layerIdx}"
  let inputNorm ← src.loadCandidates (nameCandidates s!"{p}.input_layernorm.weight")
    #[cfg.hidden_size] device
  let postNorm ← src.loadCandidates (nameCandidates s!"{p}.post_attention_layernorm.weight")
    #[cfg.hidden_size] device

  let attnFull ←
    match cfg.layerType layerIdx with
    | .fullAttention =>
      let a ← loadAttention src cfg layerIdx cfg.num_attention_heads device dtype
      pure (some a)
    | .slidingAttention => pure none
  let attnSliding ←
    match cfg.layerType layerIdx with
    | .slidingAttention =>
      let a ← loadAttention src cfg layerIdx cfg.num_attention_heads_sliding device dtype
      pure (some a)
    | .fullAttention => pure none

  let denseMlp ←
    if cfg.isDenseMlpLayer layerIdx || !cfg.isMoE then
      let m ← loadDenseMLP src cfg layerIdx device dtype
      pure (some m)
    else
      pure none
  let sparseMoe ←
    if cfg.isDenseMlpLayer layerIdx || !cfg.isMoE then
      pure none
    else
      let m ← loadSparseMoe src cfg layerIdx device dtype
      pure (some m)

  pure {
    input_layernorm := freezeStored inputNorm
    attnFull := attnFull
    attnSliding := attnSliding
    post_attention_layernorm := freezeStored postNorm
    denseMlp := denseMlp
    sparseMoe := sparseMoe
  }

/-- Shared loader core over an abstract weight source. -/
private def loadCore (src : WeightSource) (desc : String) (cfg : Config)
    (device : Device) (log : Handlers) : IO (Σ dtype : RealFloatingDType, LagunaForCausalLM (device := device) cfg dtype.val) := do
  log.onInfo s!"Loading LagunaForCausalLM from {desc}..."

  let embedTokens ← src.loadCandidates (nameCandidates "model.embed_tokens.weight")
    #[cfg.vocab_size, cfg.hidden_size] device

  let dtype := embedTokens.1
  let floating ← IO.ofExcept (DType.realFloating? dtype)
  let mut layers : Array (LagunaLayer (device := device) cfg dtype) := Array.mkEmpty cfg.num_hidden_layers
  for i in [:cfg.num_hidden_layers] do
    let layer ← loadLayer src cfg i device dtype
    layers := layers.push layer
    if (i + 1) % 4 == 0 || i + 1 == cfg.num_hidden_layers then
      log.onInfo s!"  loaded layers {i + 1}/{cfg.num_hidden_layers}"

  let normW ← src.loadCandidates (nameCandidates "model.norm.weight") #[cfg.hidden_size] device

  let model : LagunaModel (device := device) cfg dtype := {
    embed_tokens := reqGradFalse embedTokens.2
    layers := layers
    norm := freezeStored normW
  }

  let lmHeadOpt ← src.tryLoadCandidates (nameCandidates "lm_head.weight")
    #[cfg.vocab_size, cfg.hidden_size] device
  let (lmHead, tieWordEmbeddings) ←
    match lmHeadOpt with
    | some w => pure (reqGradFalse (← requireDType dtype w), false)
    | none => do
      log.onInfo "  lm_head.weight not found; tying to embeddings."
      pure (reqGradFalse model.embed_tokens, true)

  log.onInfo "Loaded LagunaForCausalLM weights."
  pure ⟨⟨dtype, floating.down.valid⟩,
    { model := model, lmHead := lmHead, tieWordEmbeddings := tieWordEmbeddings }⟩

namespace LagunaForCausalLM

/-- Load a Laguna model from a sharded HF SafeTensors directory.
    Routed experts are loaded on CPU and moved to `device` layer by layer. -/
def loadSharded (modelDir : String) (cfg : Config := Config.laguna_s_2_1)
    (device : Device := Device.CPU)
    (log : Handlers := {})
    : IO (Σ dtype : RealFloatingDType, LagunaForCausalLM (device := device) cfg dtype.val) :=
  loadCore (.sharded modelDir) modelDir cfg device log

/-- Load a Laguna model from a single HF SafeTensors file. -/
def load (path : String) (cfg : Config := Config.laguna_s_2_1)
    (device : Device := Device.CPU)
    (log : Handlers := {})
    : IO (Σ dtype : RealFloatingDType, LagunaForCausalLM (device := device) cfg dtype.val) :=
  loadCore (.single path) path cfg device log

end LagunaForCausalLM

end torch.laguna
