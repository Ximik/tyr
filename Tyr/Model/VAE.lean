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
import Tyr.Model.VAE.DTypes
import Tyr.Model.VAE.ConfigIO

namespace torch.vae

variable {device : Device} {dtypes : AutoEncoderDTypes}
open torch.Log

/-- Full AutoEncoder with BatchNorm statistics for proper normalization.
    The encode/decode pipeline uses BN normalization on packed latents. -/
structure AutoEncoder (dtypes : AutoEncoderDTypes := float32AutoEncoderDTypes) where
  /-- The VAE decoder -/
  decoder : (Decoder (device := device) dtypes.compute.val)
  /-- BatchNorm running mean for latent normalization [128] -/
  bn_running_mean : Tensor { dtype := dtypes.mean, device := device, shape := #[128] }
  /-- BatchNorm running variance for latent normalization [128] -/
  bn_running_var : Tensor { dtype := dtypes.variance.val, device := device, shape := #[128] }
  /-- BatchNorm epsilon -/
  bn_eps : Static Float
  deriving TensorStruct


namespace AutoEncoder

/-- Inverse normalize packed latents using BN statistics.
    z: [batch, 128, h, w] → [batch, 128, h, w]
    Formula: z * sqrt(running_var + eps) + running_mean -/
def invNormalize {batch height width : Nat} (ae : (AutoEncoder (device := device) dtypes))
    (z : Tensor { dtype := dtypes.compute.val, device := device, shape := #[batch, 128, height, width] }) :
    Tensor { dtype := dtypes.compute.val, device := device, shape := #[batch, 128, height, width] } :=
  let scale := Tensor.sqrt (Tensor.addScalar ae.bn_running_var ae.bn_eps.val)
  let scale := Tensor.broadcastTo (Tensor.reshape scale #[1, 128, 1, 1]) #[batch, 128, height, width]
  let mean := Tensor.broadcastTo (Tensor.reshape ae.bn_running_mean #[1, 128, 1, 1]) #[batch, 128, height, width]
  letI := dtypes.scalePromotion
  letI := dtypes.offsetPromotion
  Tensor.add (Tensor.mul z scale) mean

/-- Unpack latents from the packed format to the decoder's: each of the 128
    packed channels is a channel `c` of 32 and a 2x2 patch position.
    Packed: [batch, 128, h, w]
    Unpacked: [batch, 32, 2h, 2w] -/
def unpackLatents {batch height width : Nat} {dtype : DType}
    (z : Tensor { dtype := dtype, device := device, shape := #[batch, 128, height, width] }) :
    Tensor { dtype := dtype, device := device, shape := #[batch, 32, height * 2, width * 2] } :=
  let z := Tensor.reshape z #[batch, 32, 2, 2, height, width] (by simp [Nat.mul_comm, Nat.mul_left_comm])
  let z := Tensor.permute z #[0, 1, 4, 2, 5, 3]
  Tensor.reshape z #[batch, 32, height * 2, width * 2] (by simp [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm])

/-- Full decode pipeline: inv_normalize → unpack → decoder.
    Input: packed latents [batch, 128, h, w]
    Output: RGB image [batch, 3, 16h, 16w] -/
def decode {batch height width : Nat} (ae : (AutoEncoder (device := device) dtypes))
    (z : Tensor { dtype := dtypes.compute.val, device := device, shape := #[batch, 128, height, width] })
    (hh : 0 < height) (hw : 0 < width) :
    Tensor { dtype := dtypes.compute.val, device := device, shape := #[batch, 3, height * 2 * 2 * 2 * 2, width * 2 * 2 * 2 * 2] } :=
  ae.decoder.forward (unpackLatents (ae.invNormalize z)) (by omega) (by omega)

end AutoEncoder

/-- Load the full AutoEncoder from a diffusers `vae/` directory: `config.json`,
    which must describe the decoder this module implements, and
    `diffusion_pytorch_model.safetensors`, retaining the checkpoint's decoder
    and statistic dtypes. -/
def loadAutoEncoder (dir : String) (log : Handlers := {}) (device : Device := .CPU) :
    IO (Σ dtypes : AutoEncoderDTypes, AutoEncoder (device := device) dtypes) := do
  log.onInfo "Loading AutoEncoder..."
  let configPath := s!"{dir}/config.json"
  let eps ← IO.ofExcept ((batchNormEps (← ConfigJson.parseFile configPath)).mapError (s!"{configPath}: " ++ ·))
  let path := s!"{dir}/diffusion_pytorch_model.safetensors"
  let compute ← (← safetensors.Source.open path).load "post_quant_conv.weight" #[32, 32, 1, 1] device
  let mean ← (← safetensors.Source.open path).load "bn.running_mean" #[128] device
  let variance ← (← safetensors.Source.open path).load "bn.running_var" #[128] device
  let dtypes ← IO.ofExcept (AutoEncoderDTypes.ofDTypes? compute.1 mean.1 variance.1)
  let decoder ← loadDecoderAtDType (dtype := dtypes.compute.val) path log device
  let bn_running_mean ← IO.ofExcept (mean.atDType? dtypes.mean)
  let bn_running_var ← IO.ofExcept (variance.atDType? dtypes.variance.val)
  log.onInfo "AutoEncoder loaded successfully!"
  pure ⟨dtypes, {
    decoder
    bn_running_mean := bn_running_mean.detach
    bn_running_var := bn_running_var.detach
    bn_eps := ⟨eps⟩ }⟩

end torch.vae
