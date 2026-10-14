/-
  Tyr/Model/VAE.lean

  VAE (Variational Autoencoder) for Flux latent space.
  Re-exports all VAE components and provides full AutoEncoder.
-/

import Tyr.Model.VAE.ResnetBlock
import Tyr.Model.VAE.AttnBlock
import Tyr.Model.VAE.Upsample
import Tyr.Model.VAE.Decoder
import Tyr.Model.VAE.Weights
import Tyr.Log

namespace torch.vae

open torch.Log

/-- Full AutoEncoder with BatchNorm statistics for proper normalization.
    The encode/decode pipeline uses BN normalization on packed latents. -/
structure AutoEncoder where
  /-- The VAE decoder -/
  decoder : Decoder
  /-- BatchNorm running mean for latent normalization [128] -/
  bn_running_mean : Tensor { shape := #[128] }
  /-- BatchNorm running variance for latent normalization [128] -/
  bn_running_var : Tensor { shape := #[128] }
  deriving TensorStruct

namespace AutoEncoder

/-- BatchNorm epsilon for numerical stability -/
def bn_eps : Float := 1e-4

/-- Inverse normalize packed latents using BN statistics.
    z: [batch, 128, h, w] → [batch, 128, h, w]
    Formula: z * sqrt(running_var + eps) + running_mean -/
def invNormalize (ae : AutoEncoder) (z : Tensor { shape := #[1, 128, 16, 16] }) : Tensor { shape := #[1, 128, 16, 16] } :=
  -- Compute scale: sqrt(running_var + eps)
  let var_plus_eps := Tensor.Preserving.addScalar ae.bn_running_var bn_eps
  let scale := Tensor.Preserving.sqrt var_plus_eps
  -- Reshape and expand for element-wise ops: [128] → [1, 128, 1, 1] → [1, 128, 16, 16]
  let scale := Tensor.reshapeChecked scale #[1, 128, 1, 1]
  let mean := Tensor.reshapeChecked ae.bn_running_mean #[1, 128, 1, 1]
  -- Expand channel statistics over the spatial dimensions.
  let scale := Tensor.expand scale #[1, 128, 16, 16]
  let mean := Tensor.expand mean #[1, 128, 16, 16]
  z * scale + mean

/-- Unpack latents from packed format to VAE decoder format (16x16 packed → 32x32 unpacked).
    Packed: [1, 128, 16, 16]
    Unpacked: [1, 32, 32, 32]
    The reshape and permutation preserve the indexed spatial layout. -/
def unpackLatents16x16 (z : Tensor { shape := #[1, 128, 16, 16] }) : Tensor { shape := #[1, 32, 32, 32] } :=
  -- Reshape: [1, 128, 16, 16] → [1, 32, 2, 2, 16, 16]
  let z := Tensor.reshapeChecked z #[1, 32, 2, 2, 16, 16]
  -- Permute: [1, 32, 2, 2, 16, 16] → [1, 32, 16, 2, 16, 2]
  let z := Tensor.permute z #[0, 1, 4, 2, 5, 3]
  -- Reshape to final: [1, 32, 32, 32]
  Tensor.reshapeChecked z #[1, 32, 32, 32]

/-- Full decode pipeline: inv_normalize → unpack → decoder.
    Input: packed latents [1, 128, 16, 16]
    Output: RGB image [1, 3, 256, 256]
    The reshape and permutation preserve the indexed spatial layout. -/
def decode (ae : AutoEncoder) (z : Tensor { shape := #[1, 128, 16, 16] }) : Tensor { shape := #[1, 3, 256, 256] } :=
  -- Step 1: Inverse normalize using BN stats (uses broadcasting)
  let z := ae.invNormalize z
  -- Step 2: Unpack latents [1, 128, 16, 16] → [1, 32, 32, 32]
  let z := unpackLatents16x16 z
  -- Step 3: Decode through VAE decoder (batch=1 specialized version)
  ae.decoder.forward1 z

end AutoEncoder

/-- Load full AutoEncoder from SafeTensors including BN stats. -/
def loadAutoEncoder (path : String) (log : Handlers := {}) : IO AutoEncoder := do
  log.onInfo "Loading AutoEncoder..."

  -- Load decoder
  let decoder ← loadDecoder path log

  -- Load BN running stats
  let bn_running_mean ← safetensors.loadTensorCandidates path #["bn.running_mean"] #[128]
  let bn_running_var ← safetensors.loadTensorCandidates path #["bn.running_var"] #[128]
  log.onInfo "  Loaded BN running stats"

  log.onInfo "AutoEncoder loaded successfully!"
  pure {
    decoder
    bn_running_mean := Tensor.detach bn_running_mean
    bn_running_var := Tensor.detach bn_running_var
  }

end torch.vae
