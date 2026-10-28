/-
  Tyr/Model/VAE/Weights.lean

  Weight loading for Flux VAE decoder from SafeTensors (HuggingFace Diffusers format).
  Maps ae.safetensors weight names to Tyr structure.

  Diffusers format naming conventions:
  - post_quant_conv is at root level (not under decoder)
  - decoder.conv_norm_out instead of decoder.norm_out
  - decoder.mid_block.resnets.N instead of decoder.mid.block_N
  - decoder.mid_block.attentions.0.to_q instead of decoder.mid.attn_1.q
  - decoder.mid_block.attentions.0.to_out.0 instead of decoder.mid.attn_1.proj_out
  - decoder.up_blocks.N.resnets.M instead of decoder.up.N.block.M
  - decoder.up_blocks.N.upsamplers.0.conv instead of decoder.up.N.upsample.conv
  - conv_shortcut instead of nin_shortcut
  - up_blocks numbered 0→3 (low index = low res), opposite of BFL's 3→0
-/
import Tyr.Model.VAE.TypedOps
import Tyr.SafeTensors.Exact
import Tyr.Log
import Tyr.Model.VAE.Decoder

namespace torch.vae

variable {device : Device} {dtype : DType}
open torch.Log

private def loadWeight (path : String) (names : Array String) (shape : Shape)
    (device : Device := .CPU) : IO (Tensor ⟨shape, dtype, device⟩) := do
  let some name := names[0]? | throw (IO.userError "Empty weight name")
  let packed ← safetensors.loadTensorExact path name shape device
  match Tensor.ofRaw (σ := ⟨shape, dtype, device⟩) packed.2.raw with
  | .ok tensor => pure tensor
  | .error message => throw <| IO.userError s!"VAE tensor '{name}': {message}"

/-- Load GroupNorm parameters -/
def loadGroupNorm [DType.RealFloating dtype] (path : String) (name : String) (num_groups channels : Nat)
    (device : Device := .CPU) : IO (GroupNormParams (device := device) (dtype := dtype) num_groups channels) := do
  let weight ← loadWeight (dtype := dtype) (device := device) path #[s!"{name}.weight"] #[channels]
  let bias ← loadWeight (dtype := dtype) (device := device) path #[s!"{name}.bias"] #[channels]
  pure {
    weight := Tensor.detach weight
    bias := Tensor.detach bias
  }

/-- Load Conv2d parameters -/
def loadConv2dAtDType [DType.RealFloating dtype] (path : String) (name : String)
    (in_ch out_ch kernel_size padding : Nat)
    (device : Device := .CPU) : IO (Conv2dParams (device := device) (dtype := dtype) in_ch out_ch kernel_size padding) := do
  let weight ← loadWeight (dtype := dtype) (device := device) path #[s!"{name}.weight"] #[out_ch, in_ch, kernel_size, kernel_size]
  let bias ← loadWeight (dtype := dtype) (device := device) path #[s!"{name}.bias"] #[out_ch]
  pure {
    weight := Tensor.detach weight
    bias := Tensor.detach bias
  }

/-- Load a convolution with its stored dtype, checking that bias agrees. -/
def loadConv2d (path : String) (name : String)
    (in_ch out_ch kernel_size padding : Nat) (device : Device := .CPU) :
    IO (Σ dtype : RealFloatingDType, Conv2dParams (device := device) in_ch out_ch kernel_size padding dtype.val) := do
  let anchor ← safetensors.loadTensorExact path s!"{name}.weight"
    #[out_ch, in_ch, kernel_size, kernel_size] device
  let floating ← IO.ofExcept (DType.realFloating? anchor.1)
  let _ := floating.down
  let conv ← loadConv2dAtDType (dtype := anchor.1) path name in_ch out_ch kernel_size padding device
  pure ⟨RealFloatingDType.ofDType anchor.1, conv⟩

/-- Load ResnetBlock (same in/out channels) from Diffusers format -/
def loadResnetBlockSame [DType.RealFloating dtype] (path : String) (name : String) (ch : Nat)
    (device : Device := .CPU) : IO (ResnetBlock (device := device) (dtype := dtype) ch ch) := do
  let norm1 ← loadGroupNorm (dtype := dtype) (device := device) path s!"{name}.norm1" 32 ch
  let conv1 ← loadConv2dAtDType (dtype := dtype) (device := device) path s!"{name}.conv1" ch ch 3 1
  let norm2 ← loadGroupNorm (dtype := dtype) (device := device) path s!"{name}.norm2" 32 ch
  let conv2 ← loadConv2dAtDType (dtype := dtype) (device := device) path s!"{name}.conv2" ch ch 3 1
  pure { norm1, conv1, norm2, conv2, nin_shortcut := .identity rfl }

/-- Load ResnetBlock with channel change (has conv_shortcut) from Diffusers format -/
def loadResnetBlockChange [DType.RealFloating dtype] (path : String) (name : String) (in_ch out_ch : Nat)
    (device : Device := .CPU) : IO (ResnetBlock (device := device) (dtype := dtype) in_ch out_ch) := do
  let norm1 ← loadGroupNorm (dtype := dtype) (device := device) path s!"{name}.norm1" 32 in_ch
  let conv1 ← loadConv2dAtDType (dtype := dtype) (device := device) path s!"{name}.conv1" in_ch out_ch 3 1
  let norm2 ← loadGroupNorm (dtype := dtype) (device := device) path s!"{name}.norm2" 32 out_ch
  let conv2 ← loadConv2dAtDType (dtype := dtype) (device := device) path s!"{name}.conv2" out_ch out_ch 3 1
  -- Diffusers uses conv_shortcut instead of nin_shortcut
  let shortcut ← loadConv2dAtDType (dtype := dtype) (device := device) path s!"{name}.conv_shortcut" in_ch out_ch 1 0
  pure { norm1, conv1, norm2, conv2, nin_shortcut := .conv shortcut }

/-- Load AttnBlock from Diffusers format
    Diffusers uses: to_q, to_k, to_v, to_out.0, group_norm -/
def loadAttnBlock [DType.RealFloating dtype] (path : String) (name : String) (channels : Nat)
    (device : Device := .CPU) : IO (AttnBlock (device := device) (dtype := dtype) channels) := do
  let norm ← loadGroupNorm (dtype := dtype) (device := device) path s!"{name}.group_norm" 32 channels
  let q ← loadConv2dAtDType (dtype := dtype) (device := device) path s!"{name}.to_q" channels channels 1 0
  let k ← loadConv2dAtDType (dtype := dtype) (device := device) path s!"{name}.to_k" channels channels 1 0
  let v ← loadConv2dAtDType (dtype := dtype) (device := device) path s!"{name}.to_v" channels channels 1 0
  let proj_out ← loadConv2dAtDType (dtype := dtype) (device := device) path s!"{name}.to_out.0" channels channels 1 0
  pure { norm, q, k, v, proj_out }

/-- Load Upsample from Diffusers format -/
def loadUpsample [DType.RealFloating dtype] (path : String) (name : String) (channels : Nat)
    (device : Device := .CPU) : IO (Upsample (device := device) (dtype := dtype) channels) := do
  let conv ← loadConv2dAtDType (dtype := dtype) (device := device) path s!"{name}.conv" channels channels 3 1
  pure { conv }

/-- Load full VAE Decoder from SafeTensors (HuggingFace Diffusers format). -/
def loadDecoderAtDType [DType.RealFloating dtype] (path : String) (log : Handlers := {}) (device : Device := .CPU) : IO (Decoder (device := device) dtype) := do
  log.onInfo s!"Loading VAE Decoder from {path}..."

  -- Post-quant conv is at root level in Diffusers format
  let post_quant_conv ← loadConv2dAtDType (dtype := dtype) (device := device) path "post_quant_conv" 32 32 1 0

  -- Initial conv
  let conv_in ← loadConv2dAtDType (dtype := dtype) (device := device) path "decoder.conv_in" 32 512 3 1
  log.onInfo "  Loaded conv_in"

  -- Middle block: Diffusers uses mid_block.resnets.N and mid_block.attentions.0
  let mid_block_1 ← loadResnetBlockSame (dtype := dtype) (device := device) path "decoder.mid_block.resnets.0" 512
  let mid_attn ← loadAttnBlock (dtype := dtype) (device := device) path "decoder.mid_block.attentions.0" 512
  let mid_block_2 ← loadResnetBlockSame (dtype := dtype) (device := device) path "decoder.mid_block.resnets.1" 512
  log.onInfo "  Loaded mid block"

  -- Diffusers up_blocks are numbered opposite to BFL:
  -- up_blocks.0 = lowest res (our up_3), up_blocks.3 = highest res (our up_0)

  -- Up block 3 (our naming, lowest res): 512→512, 3 blocks, upsample
  -- = Diffusers up_blocks.0
  let up_3_block_0 ← loadResnetBlockSame (dtype := dtype) (device := device) path "decoder.up_blocks.0.resnets.0" 512
  let up_3_block_1 ← loadResnetBlockSame (dtype := dtype) (device := device) path "decoder.up_blocks.0.resnets.1" 512
  let up_3_block_2 ← loadResnetBlockSame (dtype := dtype) (device := device) path "decoder.up_blocks.0.resnets.2" 512
  let up_3_upsample ← loadUpsample (dtype := dtype) (device := device) path "decoder.up_blocks.0.upsamplers.0" 512
  log.onInfo "  Loaded up block 3 (up_blocks.0)"

  -- Up block 2: 512→512, 3 blocks, upsample
  -- = Diffusers up_blocks.1
  let up_2_block_0 ← loadResnetBlockSame (dtype := dtype) (device := device) path "decoder.up_blocks.1.resnets.0" 512
  let up_2_block_1 ← loadResnetBlockSame (dtype := dtype) (device := device) path "decoder.up_blocks.1.resnets.1" 512
  let up_2_block_2 ← loadResnetBlockSame (dtype := dtype) (device := device) path "decoder.up_blocks.1.resnets.2" 512
  let up_2_upsample ← loadUpsample (dtype := dtype) (device := device) path "decoder.up_blocks.1.upsamplers.0" 512
  log.onInfo "  Loaded up block 2 (up_blocks.1)"

  -- Up block 1: 512→256, 3 blocks, upsample
  -- = Diffusers up_blocks.2
  let up_1_block_0 ← loadResnetBlockChange (dtype := dtype) (device := device) path "decoder.up_blocks.2.resnets.0" 512 256
  let up_1_block_1 ← loadResnetBlockSame (dtype := dtype) (device := device) path "decoder.up_blocks.2.resnets.1" 256
  let up_1_block_2 ← loadResnetBlockSame (dtype := dtype) (device := device) path "decoder.up_blocks.2.resnets.2" 256
  let up_1_upsample ← loadUpsample (dtype := dtype) (device := device) path "decoder.up_blocks.2.upsamplers.0" 256
  log.onInfo "  Loaded up block 1 (up_blocks.2)"

  -- Up block 0 (highest res): 256→128, 3 blocks, no upsample
  -- = Diffusers up_blocks.3
  let up_0_block_0 ← loadResnetBlockChange (dtype := dtype) (device := device) path "decoder.up_blocks.3.resnets.0" 256 128
  let up_0_block_1 ← loadResnetBlockSame (dtype := dtype) (device := device) path "decoder.up_blocks.3.resnets.1" 128
  let up_0_block_2 ← loadResnetBlockSame (dtype := dtype) (device := device) path "decoder.up_blocks.3.resnets.2" 128
  log.onInfo "  Loaded up block 0 (up_blocks.3)"

  -- Final layers: Diffusers uses conv_norm_out instead of norm_out
  let norm_out ← loadGroupNorm (dtype := dtype) (device := device) path "decoder.conv_norm_out" 32 128
  let conv_out ← loadConv2dAtDType (dtype := dtype) (device := device) path "decoder.conv_out" 128 3 3 1
  log.onInfo "  Loaded final layers"

  log.onInfo "VAE Decoder loaded successfully!"
  pure {
    post_quant_conv
    conv_in
    mid_block_1, mid_attn, mid_block_2
    up_3_block_0, up_3_block_1, up_3_block_2, up_3_upsample
    up_2_block_0, up_2_block_1, up_2_block_2, up_2_upsample
    up_1_block_0, up_1_block_1, up_1_block_2, up_1_upsample
    up_0_block_0, up_0_block_1, up_0_block_2
    norm_out
    conv_out
  }

/-- Load the decoder without converting the checkpoint precision. -/
def loadDecoder (path : String) (log : Handlers := {}) (device : Device := .CPU) :
    IO (Σ dtype : RealFloatingDType, Decoder (device := device) dtype.val) := do
  let anchor ← safetensors.loadTensorExact path "post_quant_conv.weight" #[32, 32, 1, 1] device
  let floating ← IO.ofExcept (DType.realFloating? anchor.1)
  let _ := floating.down
  let decoder ← loadDecoderAtDType (dtype := anchor.1) path log device
  pure ⟨RealFloatingDType.ofDType anchor.1, decoder⟩

end torch.vae
