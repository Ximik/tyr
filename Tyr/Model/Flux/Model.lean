/-
  Tyr/Model/Flux/Model.lean

  Full Flux diffusion model.
  Combines double-stream and single-stream transformer blocks.
-/
import Tyr.Model.Flux.TypedOps
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Module.Linear
import Tyr.Module.LayerNorm
import Tyr.Model.Flux.Config
import Tyr.Model.Flux.Modulation
import Tyr.Model.Flux.SingleStreamBlock
import Tyr.Model.Flux.DoubleStreamBlock

namespace torch.flux

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]
/-- MLP embedder for timestep -/
structure MLPEmbedder (in_dim hidden_dim : Nat) (dtype : DType := .Float32) where
  /-- First Tensor.linear layer -/
  in_layer : Tensor { dtype := dtype, device := device, shape := #[hidden_dim, in_dim] }
  /-- Second Tensor.linear layer -/
  out_layer : Tensor { dtype := dtype, device := device, shape := #[hidden_dim, hidden_dim] }
  deriving TensorStruct


namespace MLPEmbedder

/-- Initialize MLP embedder -/
def init (in_dim hidden_dim : Nat) (device : Device := .CPU) (dtype : DType := .Float32) : IO (MLPEmbedder (device := device) (dtype := dtype) in_dim hidden_dim) := do
  let floating ← IO.ofExcept (DType.realFloating? dtype)
  let _ := floating.down
  let std := Float.sqrt (2.0 / in_dim.toFloat)
  let in_l ← randomWeight (dtype := dtype) (device := device) #[hidden_dim, in_dim]
  let out_l ← randomWeight (dtype := dtype) (device := device) #[hidden_dim, hidden_dim]
  pure {
    in_layer := (← Tensor.parameter (Tensor.mulScalar in_l std))
    out_layer := (← Tensor.parameter (Tensor.mulScalar out_l std))
  }

/-- Forward pass: linear → SiLU → linear -/
def forward {batch in_dim hidden_dim : Nat}
    (emb : MLPEmbedder (device := device) (dtype := dtype) in_dim hidden_dim)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, in_dim] })
    : Tensor { dtype := dtype, device := device, shape := #[batch, hidden_dim] } :=
  let h := Tensor.linear x emb.in_layer
  let h := Tensor.silu h
  Tensor.linear h emb.out_layer

end MLPEmbedder

/-- Final output layer -/
structure LastLayer (hidden_size patch_size : Nat) (dtype : DType := .Float32) where
  /-- Layer norm -/
  norm : LayerNorm (device := device) hidden_size dtype
  /-- Output projection -/
  linear : Tensor { dtype := dtype, device := device, shape := #[patch_size, hidden_size] }
  /-- Modulation for final output -/
  modulation : Tensor { dtype := dtype, device := device, shape := #[hidden_size * 2, hidden_size] }
  deriving TensorStruct


/-- Full Flux model -/
structure FluxModel (cfg : FluxConfig) (dtype : DType := .Float32) where
  /-- Image input projection -/
  img_in : Tensor { dtype := dtype, device := device, shape := #[cfg.hidden_size, cfg.in_channels] }
  /-- Text input projection -/
  txt_in : Tensor { dtype := dtype, device := device, shape := #[cfg.hidden_size, cfg.context_in_dim] }
  /-- Timestep embedder -/
  time_in : MLPEmbedder (device := device) (dtype := dtype) cfg.time_dim cfg.hidden_size
  /-- Double-stream blocks (Flux Klein uses mlp_ratio=3.0) -/
  double_blocks : Array (DoubleStreamBlock (device := device) (dtype := dtype) cfg.hidden_size cfg.num_heads cfg.head_dim (FluxConfig.mlpHiddenDim cfg))
  /-- Single-stream blocks (Flux Klein uses mlp_ratio=3.0) -/
  single_blocks : Array (SingleStreamBlock (device := device) (dtype := dtype) cfg.hidden_size cfg.num_heads cfg.head_dim (FluxConfig.mlpHiddenDim cfg))
  /-- Model-level modulation for double-stream image path -/
  double_stream_modulation_img : Modulation (device := device) (dtype := dtype) cfg.hidden_size true
  /-- Model-level modulation for double-stream text path -/
  double_stream_modulation_txt : Modulation (device := device) (dtype := dtype) cfg.hidden_size true
  /-- Model-level modulation for single-stream blocks -/
  single_stream_modulation : Modulation (device := device) (dtype := dtype) cfg.hidden_size false
  /-- Final output layer -/
  final_layer : LastLayer (device := device) (dtype := dtype) cfg.hidden_size cfg.in_channels
  deriving TensorStruct


namespace FluxModel

/-- Initialize Flux model -/
def init (cfg : FluxConfig) (device : Device := .CPU) (dtype : DType := .Float32) : IO (FluxModel (device := device) (dtype := dtype) cfg) := do
  let floating ← IO.ofExcept (DType.realFloating? dtype)
  let _ := floating.down
  let std := Float.sqrt (2.0 / cfg.hidden_size.toFloat)

  -- Input projections
  let img_in ← randomWeight (dtype := dtype) (device := device) #[cfg.hidden_size, cfg.in_channels]
  let txt_in ← randomWeight (dtype := dtype) (device := device) #[cfg.hidden_size, cfg.context_in_dim]

  -- Timestep embedder
  let time_in ← MLPEmbedder.init (dtype := dtype) (device := device) cfg.time_dim cfg.hidden_size

  let mlp_hidden := FluxConfig.mlpHiddenDim cfg

  -- Double blocks (Flux Klein uses mlp_ratio=3.0)
  let mut double_blocks := #[]
  for _ in [:cfg.num_double_layers] do
    let block ← DoubleStreamBlock.init (dtype := dtype) (device := device) cfg.hidden_size cfg.num_heads cfg.head_dim mlp_hidden
    double_blocks := double_blocks.push block

  -- Single blocks (Flux Klein uses mlp_ratio=3.0)
  let mut single_blocks := #[]
  for _ in [:cfg.num_single_layers] do
    let block ← SingleStreamBlock.init (dtype := dtype) (device := device) cfg.hidden_size cfg.num_heads cfg.head_dim mlp_hidden
    single_blocks := single_blocks.push block

  -- Model-level modulation (Flux2 style)
  let double_stream_modulation_img ← Modulation.init (dtype := dtype) (device := device) cfg.hidden_size true
  let double_stream_modulation_txt ← Modulation.init (dtype := dtype) (device := device) cfg.hidden_size true
  let single_stream_modulation ← Modulation.init (dtype := dtype) (device := device) cfg.hidden_size false

  -- Final layer
  let norm ← initLayerNorm (dtype := dtype) (device := device) cfg.hidden_size
  let final_linear ← randomWeight (dtype := dtype) (device := device) #[cfg.in_channels, cfg.hidden_size]
  let final_mod ← randomWeight (dtype := dtype) (device := device) #[cfg.hidden_size * 2, cfg.hidden_size]

  pure {
    img_in := (← Tensor.parameter (Tensor.mulScalar img_in std))
    txt_in := (← Tensor.parameter (Tensor.mulScalar txt_in std))
    time_in
    double_blocks
    single_blocks
    double_stream_modulation_img
    double_stream_modulation_txt
    single_stream_modulation
    final_layer := {
      norm
      linear := (← Tensor.parameter (Tensor.mulScalar final_linear std))
      modulation := (← Tensor.parameter (Tensor.mulScalar final_mod std))
    }
  }

/-- Forward pass: predict noise/velocity.
    img: [batch, img_seq, in_channels] - latent image patches
    txt: [batch, txt_seq, context_dim] - text embeddings
    timesteps: [batch] - diffusion timesteps
    img_ids: [batch, img_seq, 4] - image position IDs for RoPE
    txt_ids: [batch, txt_seq, 4] - text position IDs for RoPE
    Returns: [batch, img_seq, in_channels] - predicted noise -/
-- Helper to process double blocks using fold (avoids Id.run array iteration bug)
private def processDoubleBlocks {batch img_seq txt_seq hidden_size num_heads head_dim mlp_hidden : Nat}
    (blocks : Array (DoubleStreamBlock (device := device) (dtype := dtype) hidden_size num_heads head_dim mlp_hidden))
    (img : Tensor { dtype := dtype, device := device, shape := #[batch, img_seq, hidden_size] })
    (txt : Tensor { dtype := dtype, device := device, shape := #[batch, txt_seq, hidden_size] })
    (img_pe : Tensor { device := device, shape := #[batch, 1, img_seq, head_dim / 2, 2, 2], dtype := .Float32 })
    (txt_pe : Tensor { device := device, shape := #[batch, 1, txt_seq, head_dim / 2, 2, 2], dtype := .Float32 })
    (mod_img : Tensor { dtype := dtype, device := device, shape := #[batch, 6, hidden_size] })
    (mod_txt : Tensor { dtype := dtype, device := device, shape := #[batch, 6, hidden_size] })
    : Tensor { dtype := dtype, device := device, shape := #[batch, img_seq, hidden_size] } × Tensor { dtype := dtype, device := device, shape := #[batch, txt_seq, hidden_size] } :=
  blocks.foldl (init := (img, txt)) fun (img, txt) block =>
    block.forward img txt img_pe txt_pe mod_img mod_txt

-- Helper to process single blocks using fold
private def processSingleBlocks {batch seq hidden_size num_heads head_dim mlp_hidden : Nat}
    (blocks : Array (SingleStreamBlock (device := device) (dtype := dtype) hidden_size num_heads head_dim mlp_hidden))
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, hidden_size] })
    (pe : Tensor { device := device, shape := #[batch, 1, seq, head_dim / 2, 2, 2], dtype := .Float32 })
    (mod : Tensor { dtype := dtype, device := device, shape := #[batch, 3, hidden_size] })
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, hidden_size] } :=
  blocks.foldl (init := x) fun x block =>
    block.forward x pe mod

def forward {batch img_seq txt_seq : Nat} (cfg : FluxConfig)
    (model : FluxModel (device := device) (dtype := dtype) cfg)
    (img : Tensor { dtype := dtype, device := device, shape := #[batch, img_seq, cfg.in_channels] })
    (txt : Tensor { dtype := dtype, device := device, shape := #[batch, txt_seq, cfg.context_in_dim] })
    (timesteps : Tensor { device := device, shape := #[batch], dtype := .Float32 })
    (img_ids : Tensor { device := device, shape := #[batch, img_seq, 4], dtype := .Int64 })
    (txt_ids : Tensor { device := device, shape := #[batch, txt_seq, 4], dtype := .Int64 })
    : Tensor { dtype := dtype, device := device, shape := #[batch, img_seq, cfg.in_channels] } :=
  -- Compute timestep embedding
  let t_emb := flux.timestepEmbedding timesteps cfg.time_dim
  let vec := model.time_in.forward (Tensor.castLike model.time_in.in_layer t_emb)  -- [batch, hidden_size]
  let mod_img := model.double_stream_modulation_img.forward true vec
  let mod_txt := model.double_stream_modulation_txt.forward true vec
  let mod_single := model.single_stream_modulation.forward false vec

  -- Project inputs to hidden dimension
  let img := Tensor.linear3d img model.img_in  -- [batch, img_seq, hidden_size]
  let txt := Tensor.linear3d txt model.txt_in  -- [batch, txt_seq, hidden_size]

  -- Compute RoPE embeddings
  let img_pe := flux.ropeEmbed (head_dim := cfg.head_dim) img_ids cfg.axes_dims cfg.theta
  let txt_pe := flux.ropeEmbed (head_dim := cfg.head_dim) txt_ids cfg.axes_dims cfg.theta

  -- Double-stream blocks (using fold to avoid Id.run bug)
  let (img, txt) := processDoubleBlocks model.double_blocks img txt img_pe txt_pe mod_img mod_txt

  -- Concatenate for single-stream
  let x : Tensor { dtype := dtype, device := device, shape := #[batch, txt_seq + img_seq, cfg.hidden_size] } := qwen35.typedCat txt img 1  -- [batch, txt_seq + img_seq, hidden_size]

  -- Compute combined RoPE for concatenated img+txt sequence
  let combined_ids : Tensor { device := device, shape := #[batch, txt_seq + img_seq, 4], dtype := .Int64 } := qwen35.typedCat txt_ids img_ids 1
  let combined_pe := flux.ropeEmbed (head_dim := cfg.head_dim) combined_ids cfg.axes_dims cfg.theta

  -- Single-stream blocks (using fold to avoid Id.run bug)
  let x := processSingleBlocks model.single_blocks x combined_pe mod_single

  -- Extract image tokens
  let img_slice := sliceTyped x 1 txt_seq img_seq
  let img_out := Tensor.reshapeChecked img_slice #[batch, img_seq, cfg.hidden_size]

  -- Final layer with modulation
  let mod := Tensor.silu vec
  let mod2 := Tensor.linear mod model.final_layer.modulation  -- [batch, hidden_size * 2]
  let shift_slice := sliceTyped mod2 1 0 cfg.hidden_size
  let scale_slice := sliceTyped mod2 1 cfg.hidden_size cfg.hidden_size
  let scale := Tensor.reshapeChecked scale_slice #[batch, cfg.hidden_size]
  let shift := Tensor.reshapeChecked shift_slice #[batch, cfg.hidden_size]

  let img_normed := model.final_layer.norm.forward3d img_out
  let img_modulated := applyModulation img_normed scale shift
  Tensor.linear3d img_modulated model.final_layer.linear

end FluxModel

end torch.flux
