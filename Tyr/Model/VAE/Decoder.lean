/-
  Tyr/Model/VAE/Decoder.lean

  Full VAE decoder for Flux (BFL format).
  Decodes latent representations to images.

  BFL VAE structure:
  - conv_in: 32→512
  - mid: block_1, attn_1, block_2 (all 512 channels)
  - up.3: 3x ResnetBlock(512→512), upsample (32→64)
  - up.2: 3x ResnetBlock(512→512), upsample (64→128)
  - up.1: ResnetBlock(512→256), 2x ResnetBlock(256→256), upsample (128→256)
  - up.0: ResnetBlock(256→128), 2x ResnetBlock(128→128), no upsample
  - norm_out, conv_out: 128→3
-/
import Tyr.Model.VAE.TypedOps
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.VAE.ResnetBlock
import Tyr.Model.VAE.AttnBlock
import Tyr.Model.VAE.Upsample

namespace torch.vae

/-- Full VAE Decoder (BFL format).
    Decodes [batch, 32, H, W] latents to [batch, 3, 8*H, 8*W] images. -/
structure Decoder where
  /-- Post-quantization 1x1 conv -/
  post_quant_conv : Conv2dParams 32 32 1 0
  /-- Initial conv from z_channels to block_in -/
  conv_in : Conv2dParams 32 512 3 1

  /-- Middle block: resnet + attention + resnet -/
  mid_block_1 : ResnetBlock 512 512
  mid_attn : AttnBlock 512
  mid_block_2 : ResnetBlock 512 512

  /-- Up block 3 (lowest res): 512→512, 3 blocks, upsample -/
  up_3_block_0 : ResnetBlock 512 512
  up_3_block_1 : ResnetBlock 512 512
  up_3_block_2 : ResnetBlock 512 512
  up_3_upsample : Upsample 512

  /-- Up block 2: 512→512, 3 blocks, upsample -/
  up_2_block_0 : ResnetBlock 512 512
  up_2_block_1 : ResnetBlock 512 512
  up_2_block_2 : ResnetBlock 512 512
  up_2_upsample : Upsample 512

  /-- Up block 1: 512→256, 3 blocks, upsample -/
  up_1_block_0 : ResnetBlock 512 256  -- has nin_shortcut
  up_1_block_1 : ResnetBlock 256 256
  up_1_block_2 : ResnetBlock 256 256
  up_1_upsample : Upsample 256

  /-- Up block 0 (highest res): 256→128, 3 blocks, no upsample -/
  up_0_block_0 : ResnetBlock 256 128  -- has nin_shortcut
  up_0_block_1 : ResnetBlock 128 128
  up_0_block_2 : ResnetBlock 128 128

  /-- Final normalization -/
  norm_out : GroupNormParams 32 128

  /-- Final conv to RGB -/
  conv_out : Conv2dParams 128 3 3 1

  deriving TensorStruct

namespace Decoder

/-- Initialize decoder -/
def init : IO Decoder := do
  -- Post-quant conv
  let post_quant_conv ← Conv2dParams.init 32 32 1 0
  -- Initial conv
  let conv_in ← Conv2dParams.init 32 512 3 1

  -- Middle block
  let mid_block_1 ← ResnetBlock.init 512 512
  let mid_attn ← AttnBlock.init 512
  let mid_block_2 ← ResnetBlock.init 512 512

  -- Up block 3 (512→512)
  let up_3_block_0 ← ResnetBlock.init 512 512
  let up_3_block_1 ← ResnetBlock.init 512 512
  let up_3_block_2 ← ResnetBlock.init 512 512
  let up_3_upsample ← Upsample.init 512

  -- Up block 2 (512→512)
  let up_2_block_0 ← ResnetBlock.init 512 512
  let up_2_block_1 ← ResnetBlock.init 512 512
  let up_2_block_2 ← ResnetBlock.init 512 512
  let up_2_upsample ← Upsample.init 512

  -- Up block 1 (512→256)
  let up_1_block_0 ← ResnetBlock.init 512 256
  let up_1_block_1 ← ResnetBlock.init 256 256
  let up_1_block_2 ← ResnetBlock.init 256 256
  let up_1_upsample ← Upsample.init 256

  -- Up block 0 (256→128)
  let up_0_block_0 ← ResnetBlock.init 256 128
  let up_0_block_1 ← ResnetBlock.init 128 128
  let up_0_block_2 ← ResnetBlock.init 128 128

  -- Final layers
  let norm_out ← GroupNormParams.init 32 128
  let conv_out ← Conv2dParams.init 128 3 3 1

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

/-- Decode latents to image.
    Input: [batch, 32, H, W] latent
    Output: [batch, 3, 8*H, 8*W] RGB image -/
def forward {batch height width : UInt64}
    (dec : Decoder)
    (z : Tensor { shape := #[batch, 32, height, width] })
    : Tensor { shape := #[batch, 3, height * 2 * 2 * 2, width * 2 * 2 * 2] } :=
  -- Post-quant conv: [batch, 32, 32, 32] → [batch, 32, 32, 32]
  let z := Tensor.conv2dBias z dec.post_quant_conv.weight dec.post_quant_conv.bias #[1, 1] #[0, 0]
  let z := Tensor.reshapeChecked z #[batch, 32, height, width]
  -- Initial conv: [batch, 32, 32, 32] → [batch, 512, 32, 32]
  let h := Tensor.conv2dBias z dec.conv_in.weight dec.conv_in.bias #[1, 1] #[1, 1]
  let h := Tensor.reshapeChecked h #[batch, 512, height, width]
  -- Middle block
  let h := dec.mid_block_1.forward h
  let h := dec.mid_attn.forward h
  let h := dec.mid_block_2.forward h
  -- Up block 3: [batch, 512, 32, 32] → [batch, 512, 64, 64]
  let h := dec.up_3_block_0.forward h
  let h := dec.up_3_block_1.forward h
  let h := dec.up_3_block_2.forward h
  let h := dec.up_3_upsample.forward h
  -- Up block 2: [batch, 512, 64, 64] → [batch, 512, 128, 128]
  let h := dec.up_2_block_0.forward h
  let h := dec.up_2_block_1.forward h
  let h := dec.up_2_block_2.forward h
  let h := dec.up_2_upsample.forward h
  -- Up block 1: [batch, 512, 128, 128] → [batch, 256, 256, 256]
  let h := dec.up_1_block_0.forward h
  let h := dec.up_1_block_1.forward h
  let h := dec.up_1_block_2.forward h
  let h := dec.up_1_upsample.forward h
  -- Up block 0: [batch, 256, 256, 256] → [batch, 128, 256, 256]
  let h := dec.up_0_block_0.forward h
  let h := dec.up_0_block_1.forward h
  let h := dec.up_0_block_2.forward h
  -- Final: norm → swish → conv
  let h := dec.norm_out.forward h
  let h := swish h
  let h := Tensor.conv2dBias h dec.conv_out.weight dec.conv_out.bias #[1, 1] #[1, 1]
  Tensor.reshapeChecked h #[batch, 3, height * 2 * 2 * 2, width * 2 * 2 * 2]

/-- Decode latents to image (batch=1 specialized version).
    Input: [1, 32, 32, 32] latent
    Output: [1, 3, 256, 256] RGB image -/
def forward1 (dec : Decoder) (z : Tensor { shape := #[1, 32, 32, 32] }) : Tensor { shape := #[1, 3, 256, 256] } :=
  dec.forward z

end Decoder

end torch.vae
