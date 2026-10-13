import Tyr.SafeTensors.Load
/-
  Tyr/Model/Flux/Weights.lean

  Weight loading for Flux Klein 4B from SafeTensors.
  Maps HuggingFace flux.safetensors weight names to Tyr structure.
-/
import Tyr.Model.Flux.TypedOps
import Tyr.Log
import Tyr.Model.Flux.Config
import Tyr.Model.Flux.Model

namespace torch.flux

open torch.Log

/-- Load RMSNorm weights from Flux2 format (uses .scale suffix) -/
def loadRMSNormFlux2 (path : String) (name : String) (dim : UInt64)
    : IO (RMSNorm dim (.Unknown "dynamic")) := do
  let weight ← safetensors.loadTensorCandidates path #[s!"{name}.scale"] #[dim]
  pure { weight := Tensor.setRequiresGrad weight false, eps := ⟨1e-6⟩ }

/-- Load QKNorm weights from Flux2 format -/
def loadQKNormFlux2 (path : String) (name : String) (head_dim : UInt64)
    : IO (QKNorm head_dim) := do
  let query_norm ← loadRMSNormFlux2 path s!"{name}.query_norm" head_dim
  let key_norm ← loadRMSNormFlux2 path s!"{name}.key_norm" head_dim
  pure { query_norm, key_norm }

/-- Load Modulation weights from Flux2 format -/
def loadModulationFlux2 (path : String) (name : String) (hidden_size : UInt64) (isDouble : Bool)
    : IO (Modulation hidden_size isDouble) := do
  let outDim := if isDouble then 6 * hidden_size else 3 * hidden_size
  let lin ← safetensors.loadTensorCandidates path #[s!"{name}.lin.weight"] #[outDim, hidden_size]
  pure { lin := Tensor.setRequiresGrad lin false }

/-- Load SelfAttention weights from Flux2 format -/
def loadSelfAttentionFlux2 (path : String) (name : String)
    (hidden_size num_heads head_dim : UInt64)
    : IO (SelfAttention hidden_size num_heads head_dim) := do
  let qkv ← safetensors.loadTensorCandidates path #[s!"{name}.qkv.weight"] #[num_heads * head_dim * 3, hidden_size]
  let norm ← loadQKNormFlux2 path s!"{name}.norm" head_dim
  let proj ← safetensors.loadTensorCandidates path #[s!"{name}.proj.weight"] #[hidden_size, num_heads * head_dim]
  pure {
    qkv := Tensor.setRequiresGrad qkv false
    norm
    proj := Tensor.setRequiresGrad proj false
  }

/-- Load SwiGLU MLP weights from Flux2 format (indexed layers: .0, .2) -/
def loadSwiGLUMLPFlux2 (path : String) (name : String)
    (hidden_size mlp_hidden : UInt64)
    : IO (SwiGLUMLP hidden_size mlp_hidden) := do
  -- Flux2 uses .0.weight for first layer (gate+up), .2.weight for down projection
  let w1 ← safetensors.loadTensorCandidates path #[s!"{name}.0.weight"] #[mlp_hidden * 2, hidden_size]
  let w2 ← safetensors.loadTensorCandidates path #[s!"{name}.2.weight"] #[hidden_size, mlp_hidden]
  pure {
    w1 := Tensor.setRequiresGrad w1 false
    w2 := Tensor.setRequiresGrad w2 false
  }

/-- Load DoubleStreamBlock weights from Flux2 format.
    Note: Flux2 uses elementwise_affine=False LayerNorms (no weights).
    Modulation is at model level, not per-block. -/
def loadDoubleStreamBlockFlux2 (path : String) (name : String) (cfg : FluxConfig)
    : IO (DoubleStreamBlock cfg.hidden_size cfg.num_heads cfg.head_dim (FluxConfig.mlpHiddenDim cfg)) := do
  let mlp_hidden := FluxConfig.mlpHiddenDim cfg

  -- Load attention (img_attn and txt_attn)
  let img_attn ← loadSelfAttentionFlux2 path s!"{name}.img_attn" cfg.hidden_size cfg.num_heads cfg.head_dim
  let txt_attn ← loadSelfAttentionFlux2 path s!"{name}.txt_attn" cfg.hidden_size cfg.num_heads cfg.head_dim

  -- Load MLP (indexed format)
  let img_mlp ← loadSwiGLUMLPFlux2 path s!"{name}.img_mlp" cfg.hidden_size mlp_hidden
  let txt_mlp ← loadSwiGLUMLPFlux2 path s!"{name}.txt_mlp" cfg.hidden_size mlp_hidden

  -- Flux2 uses elementwise_affine=False LayerNorms, create with ones
  let img_norm1 := initLayerNorm cfg.hidden_size 1e-6 false
  let img_norm2 := initLayerNorm cfg.hidden_size 1e-6 false
  let txt_norm1 := initLayerNorm cfg.hidden_size 1e-6 false
  let txt_norm2 := initLayerNorm cfg.hidden_size 1e-6 false

  pure {
    img_norm1, img_attn, img_norm2, img_mlp
    txt_norm1, txt_attn, txt_norm2, txt_mlp
  }

/-- Load SingleStreamBlock weights from Flux2 format -/
def loadSingleStreamBlockFlux2 (path : String) (name : String) (cfg : FluxConfig)
    : IO (SingleStreamBlock cfg.hidden_size cfg.num_heads cfg.head_dim (FluxConfig.mlpHiddenDim cfg)) := do
  let mlp_hidden := FluxConfig.mlpHiddenDim cfg
  let qkv_dim := cfg.num_heads * cfg.head_dim * 3
  let mlp_in := mlp_hidden * 2  -- SwiGLU takes 2x

  let linear1 ← safetensors.loadTensorCandidates path #[s!"{name}.linear1.weight"] #[qkv_dim + mlp_in, cfg.hidden_size]
  let linear2 ← safetensors.loadTensorCandidates path #[s!"{name}.linear2.weight"] #[cfg.hidden_size, cfg.num_heads * cfg.head_dim + mlp_hidden]
  let norm ← loadQKNormFlux2 path s!"{name}.norm" cfg.head_dim

  -- Flux2 uses elementwise_affine=False pre_norm
  let pre_norm := initLayerNorm cfg.hidden_size 1e-6 false

  pure {
    pre_norm
    linear1 := Tensor.setRequiresGrad linear1 false
    linear2 := Tensor.setRequiresGrad linear2 false
    norm
  }

/-- Load MLP embedder weights -/
def loadMLPEmbedder (path : String) (name : String) (in_dim hidden_dim : UInt64)
    : IO (MLPEmbedder in_dim hidden_dim) := do
  let in_layer ← safetensors.loadTensorCandidates path #[s!"{name}.in_layer.weight"] #[hidden_dim, in_dim]
  let out_layer ← safetensors.loadTensorCandidates path #[s!"{name}.out_layer.weight"] #[hidden_dim, hidden_dim]
  pure {
    in_layer := Tensor.setRequiresGrad in_layer false
    out_layer := Tensor.setRequiresGrad out_layer false
  }

/-- Load full Flux model from SafeTensors -/
def loadFluxModel (path : String) (cfg : FluxConfig := FluxConfig.klein4B)
    (log : Handlers := {})
    : IO (FluxModel cfg) := do
  log.onInfo s!"Loading Flux model from {path}..."

  -- Input projections
  let img_in ← safetensors.loadTensorCandidates path #["img_in.weight"] #[cfg.hidden_size, cfg.in_channels]
  let txt_in ← safetensors.loadTensorCandidates path #["txt_in.weight"] #[cfg.hidden_size, cfg.context_in_dim]
  log.onInfo "  Loaded input projections"

  -- Time embedder
  let time_in ← loadMLPEmbedder path "time_in" cfg.time_dim cfg.hidden_size
  log.onInfo "  Loaded time embedder"

  -- Model-level modulation (Flux2 style)
  let double_stream_modulation_img ← loadModulationFlux2 path "double_stream_modulation_img" cfg.hidden_size true
  let double_stream_modulation_txt ← loadModulationFlux2 path "double_stream_modulation_txt" cfg.hidden_size true
  let single_stream_modulation ← loadModulationFlux2 path "single_stream_modulation" cfg.hidden_size false

  -- Double blocks (using Flux2 loading)
  let mut double_blocks := #[]
  for i in [:cfg.num_double_layers.toNat] do
    let block ← loadDoubleStreamBlockFlux2 path s!"double_blocks.{i}" cfg
    double_blocks := double_blocks.push block
  log.onInfo s!"  Loaded {cfg.num_double_layers} double blocks"

  -- Single blocks (using Flux2 loading)
  let mut single_blocks := #[]
  for i in [:cfg.num_single_layers.toNat] do
    let block ← loadSingleStreamBlockFlux2 path s!"single_blocks.{i}" cfg
    single_blocks := single_blocks.push block
  log.onInfo s!"  Loaded {cfg.num_single_layers} single blocks"

  -- Final layer (Flux2 uses elementwise_affine=False for norm)
  let final_norm := initLayerNorm cfg.hidden_size 1e-6 false
  let final_linear ← safetensors.loadTensorCandidates path #["final_layer.linear.weight"] #[cfg.in_channels, cfg.hidden_size]
  let final_mod ← safetensors.loadTensorCandidates path #["final_layer.adaLN_modulation.1.weight"] #[cfg.hidden_size * 2, cfg.hidden_size]
  log.onInfo "  Loaded final layer"

  log.onInfo "Flux model loaded successfully!"
  pure {
    img_in := Tensor.setRequiresGrad img_in false
    txt_in := Tensor.setRequiresGrad txt_in false
    time_in
    double_blocks
    single_blocks
    double_stream_modulation_img
    double_stream_modulation_txt
    single_stream_modulation
    final_layer := {
      norm := final_norm
      linear := Tensor.setRequiresGrad final_linear false
      modulation := Tensor.setRequiresGrad final_mod false
    }
  }

end torch.flux
