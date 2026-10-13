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
import Tyr.SafeTensors.Load
import Tyr.Log
import Tyr.Model.VAE.Decoder

namespace torch.vae

open torch.Log

/-- Load GroupNorm parameters -/
def loadGroupNorm (path : String) (name : String) (num_groups channels : UInt64)
    : IO (GroupNormParams num_groups channels) := do
  let weight ← safetensors.loadTensorCandidates path #[s!"{name}.weight"] #[channels]
  let bias ← safetensors.loadTensorCandidates path #[s!"{name}.bias"] #[channels]
  pure {
    weight := Tensor.setRequiresGrad weight false
    bias := Tensor.setRequiresGrad bias false
  }

/-- Load Conv2d parameters -/
def loadConv2d (path : String) (name : String)
    (in_ch out_ch kernel_size padding : UInt64)
    : IO (Conv2dParams in_ch out_ch kernel_size padding) := do
  let weight ← safetensors.loadTensorCandidates path #[s!"{name}.weight"] #[out_ch, in_ch, kernel_size, kernel_size]
  let bias ← safetensors.loadTensorCandidates path #[s!"{name}.bias"] #[out_ch]
  pure {
    weight := Tensor.setRequiresGrad weight false
    bias := Tensor.setRequiresGrad bias false
  }

/-- Load ResnetBlock (same in/out channels) from Diffusers format -/
def loadResnetBlockSame (path : String) (name : String) (ch : UInt64)
    : IO (ResnetBlock ch ch) := do
  let norm1 ← loadGroupNorm path s!"{name}.norm1" 32 ch
  let conv1 ← loadConv2d path s!"{name}.conv1" ch ch 3 1
  let norm2 ← loadGroupNorm path s!"{name}.norm2" 32 ch
  let conv2 ← loadConv2d path s!"{name}.conv2" ch ch 3 1
  pure { norm1, conv1, norm2, conv2, nin_shortcut := none }

/-- Load ResnetBlock with channel change (has conv_shortcut) from Diffusers format -/
def loadResnetBlockChange (path : String) (name : String) (in_ch out_ch : UInt64)
    : IO (ResnetBlock in_ch out_ch) := do
  let norm1 ← loadGroupNorm path s!"{name}.norm1" 32 in_ch
  let conv1 ← loadConv2d path s!"{name}.conv1" in_ch out_ch 3 1
  let norm2 ← loadGroupNorm path s!"{name}.norm2" 32 out_ch
  let conv2 ← loadConv2d path s!"{name}.conv2" out_ch out_ch 3 1
  -- Diffusers uses conv_shortcut instead of nin_shortcut
  let shortcut ← loadConv2d path s!"{name}.conv_shortcut" in_ch out_ch 1 0
  pure { norm1, conv1, norm2, conv2, nin_shortcut := some shortcut }

/-- Load AttnBlock from Diffusers format
    Diffusers uses: to_q, to_k, to_v, to_out.0, group_norm -/
def loadAttnBlock (path : String) (name : String) (channels : UInt64)
    : IO (AttnBlock channels) := do
  let norm ← loadGroupNorm path s!"{name}.group_norm" 32 channels
  let q ← loadConv2d path s!"{name}.to_q" channels channels 1 0
  let k ← loadConv2d path s!"{name}.to_k" channels channels 1 0
  let v ← loadConv2d path s!"{name}.to_v" channels channels 1 0
  let proj_out ← loadConv2d path s!"{name}.to_out.0" channels channels 1 0
  pure { norm, q, k, v, proj_out }

/-- Load Upsample from Diffusers format -/
def loadUpsample (path : String) (name : String) (channels : UInt64)
    : IO (Upsample channels) := do
  let conv ← loadConv2d path s!"{name}.conv" channels channels 3 1
  pure { conv }

/-- Load full VAE Decoder from SafeTensors (HuggingFace Diffusers format). -/
def loadDecoder (path : String) (log : Handlers := {}) : IO Decoder := do
  log.onInfo s!"Loading VAE Decoder from {path}..."

  -- Post-quant conv is at root level in Diffusers format
  let post_quant_conv ← loadConv2d path "post_quant_conv" 32 32 1 0

  -- Initial conv
  let conv_in ← loadConv2d path "decoder.conv_in" 32 512 3 1
  log.onInfo "  Loaded conv_in"

  -- Middle block: Diffusers uses mid_block.resnets.N and mid_block.attentions.0
  let mid_block_1 ← loadResnetBlockSame path "decoder.mid_block.resnets.0" 512
  let mid_attn ← loadAttnBlock path "decoder.mid_block.attentions.0" 512
  let mid_block_2 ← loadResnetBlockSame path "decoder.mid_block.resnets.1" 512
  log.onInfo "  Loaded mid block"

  -- Diffusers up_blocks are numbered opposite to BFL:
  -- up_blocks.0 = lowest res (our up_3), up_blocks.3 = highest res (our up_0)

  -- Up block 3 (our naming, lowest res): 512→512, 3 blocks, upsample
  -- = Diffusers up_blocks.0
  let up_3_block_0 ← loadResnetBlockSame path "decoder.up_blocks.0.resnets.0" 512
  let up_3_block_1 ← loadResnetBlockSame path "decoder.up_blocks.0.resnets.1" 512
  let up_3_block_2 ← loadResnetBlockSame path "decoder.up_blocks.0.resnets.2" 512
  let up_3_upsample ← loadUpsample path "decoder.up_blocks.0.upsamplers.0" 512
  log.onInfo "  Loaded up block 3 (up_blocks.0)"

  -- Up block 2: 512→512, 3 blocks, upsample
  -- = Diffusers up_blocks.1
  let up_2_block_0 ← loadResnetBlockSame path "decoder.up_blocks.1.resnets.0" 512
  let up_2_block_1 ← loadResnetBlockSame path "decoder.up_blocks.1.resnets.1" 512
  let up_2_block_2 ← loadResnetBlockSame path "decoder.up_blocks.1.resnets.2" 512
  let up_2_upsample ← loadUpsample path "decoder.up_blocks.1.upsamplers.0" 512
  log.onInfo "  Loaded up block 2 (up_blocks.1)"

  -- Up block 1: 512→256, 3 blocks, upsample
  -- = Diffusers up_blocks.2
  let up_1_block_0 ← loadResnetBlockChange path "decoder.up_blocks.2.resnets.0" 512 256
  let up_1_block_1 ← loadResnetBlockSame path "decoder.up_blocks.2.resnets.1" 256
  let up_1_block_2 ← loadResnetBlockSame path "decoder.up_blocks.2.resnets.2" 256
  let up_1_upsample ← loadUpsample path "decoder.up_blocks.2.upsamplers.0" 256
  log.onInfo "  Loaded up block 1 (up_blocks.2)"

  -- Up block 0 (highest res): 256→128, 3 blocks, no upsample
  -- = Diffusers up_blocks.3
  let up_0_block_0 ← loadResnetBlockChange path "decoder.up_blocks.3.resnets.0" 256 128
  let up_0_block_1 ← loadResnetBlockSame path "decoder.up_blocks.3.resnets.1" 128
  let up_0_block_2 ← loadResnetBlockSame path "decoder.up_blocks.3.resnets.2" 128
  log.onInfo "  Loaded up block 0 (up_blocks.3)"

  -- Final layers: Diffusers uses conv_norm_out instead of norm_out
  let norm_out ← loadGroupNorm path "decoder.conv_norm_out" 32 128
  let conv_out ← loadConv2d path "decoder.conv_out" 128 3 3 1
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

end torch.vae
