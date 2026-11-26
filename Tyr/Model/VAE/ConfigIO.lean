/-
  Tyr/Model/VAE/ConfigIO.lean

  diffusers `vae/config.json` check for the Flux2 VAE decoder.
-/
import Tyr.IO.ConfigJson

namespace torch.vae

open Lean ConfigJson

private def requireField {α : Type} [FromJson α] [BEq α] [ToString α] (j : Json) (key : String)
    (expected : α) : Except String Unit := do
  let some (value : α) ← field? j key | throw s!"VAE config: missing '{key}'"
  unless value == expected do
    throw s!"VAE config: unsupported '{key}' = {value}; the decoder implements {expected}"

/-- The `batch_norm_eps` of an `AutoencoderKLFlux2` config, whose decoder layout
must be the one `Decoder` hard-codes. -/
def batchNormEps (j : Json) : Except String Float := do
  requireField j "_class_name" "AutoencoderKLFlux2"
  requireField j "block_out_channels" #[128, 256, 512, 512]
  requireField j "up_block_types" #["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"]
  requireField j "layers_per_block" 2
  requireField j "latent_channels" 32
  requireField j "out_channels" 3
  requireField j "norm_num_groups" 32
  requireField j "act_fn" "silu"
  requireField j "mid_block_add_attention" true
  requireField j "use_post_quant_conv" true
  requireField j "patch_size" #[2, 2]
  let some (eps : Float) ← field? j "batch_norm_eps" | throw "VAE config: missing 'batch_norm_eps'"
  pure eps

end torch.vae
