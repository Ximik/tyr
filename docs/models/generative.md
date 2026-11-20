# Generative models

This chapter covers Tyr's **Flux + VAE** text-to-image stack (a Flux2 Klein 4B
rectified-flow diffusion transformer plus its BFL latent decoder). Use Flux for image
generation in a learned latent space.

Everything here is shape-typed Lean 4 over the libtorch FFI. Namespaces:
`torch.flux`, `torch.vae`.

## Architecture and main abstractions

### Flux (`Tyr/Model/Flux/`)

A Flux2 (Klein 4B) rectified-flow image diffusion transformer. `FluxConfig` (`Config.lean:13`)
carries the architecture dimensions; its field defaults *are* Klein 4B, also exposed as
`FluxConfig.klein4B`.

`FluxModel cfg dtype` is one flat `deriving TensorStruct` record sharing its
compute dtype across input
projections (`img_in`, `txt_in`), a timestep `MLPEmbedder`, arrays of
`DoubleStreamBlock`s (separate img/txt QKV+SwiGLU paths, joint attention) and
`SingleStreamBlock`s (fused QKV+MLP over the concatenated sequence), three model-level
adaLN `Modulation` layers, and a `LastLayer`. Q/K are RMS-normalized (`QKNorm`); positions
use 4-axis RoPE (`ropeEmbed`/`applyRope`, axes `#[t, h, w, l]`, `theta := 2000`), and the
timestep embedding is a C++ FFI extern (`@[extern "lean_torch_timestep_embedding"]`,
`Modulation.lean:19`). The forward (`Model.lean:165`) takes packed image latents, text
embeddings, timesteps, and both position-ID tensors, returning the velocity prediction:

```lean
variable {device : Device} {dtype : DType}

def forward {batch img_seq txt_seq : UInt64} (cfg : FluxConfig)
    (model : FluxModel (device := device) cfg dtype)
    (img : Tensor { shape := #[batch, img_seq, cfg.in_channels], dtype := dtype, device := device })
    (txt : Tensor { shape := #[batch, txt_seq, cfg.context_in_dim], dtype := dtype, device := device })
    (timesteps : Tensor { shape := #[batch], dtype := .Float32, device := device })
    (img_ids : Tensor { shape := #[batch, img_seq, 4], dtype := .Int64, device := device })
    (txt_ids : Tensor { shape := #[batch, txt_seq, 4], dtype := .Int64, device := device })
    : Tensor { shape := #[batch, img_seq, cfg.in_channels], dtype := dtype, device := device }
```

`Sampling.lean` implements the Flux2 schedule (`computeTimesteps image_seq_len num_steps`,
empirical-μ time-SNR shift) and Euler integration: `eulerStep` (`:61`), `denoise` (`:74`)
over an explicit noise tensor, `sample` (`:94`) drawing fresh noise.
`computeImagePositionIds height width` and `computeTextPositionIds seq_len` produce the
`[1, seq, 4]` RoPE inputs; `packLatents`/`unpackLatents` (`:156-187`) convert between VAE
spatial format `[batch, 32, H, W]` and the 2×2-patchified sequence format
`[batch, (H/2)·(W/2), 128]`. Weights load from a Flux2 HuggingFace SafeTensors file with
`loadFluxModel path (cfg := FluxConfig.klein4B)`. Its result is
`IO (Σ dtype : RealFloatingDType, FluxModel (device := device) cfg dtype.val)`:
the loader validates the input projection's floating dtype and checks connected
weights against it, retaining that evidence for forward computation. Timestep and RoPE calculations retain their explicit Float32 paths.
Text conditioning
is **not** part of this module: the model consumes pre-computed embeddings
(`context_in_dim := 7680`), produced in the demo by a Qwen encoder — see
[LLM model families](llms.md).

### VAE (`Tyr/Model/VAE.lean`, `Tyr/Model/VAE/`)

The BFL latent autoencoder decoder for Flux. `Decoder` (`Decoder.lean:28`) is the BFL
stack — `post_quant_conv`, `conv_in`, mid `ResnetBlock`/`AttnBlock`/`ResnetBlock`, four
up-groups with `Upsample`s, GroupNorm + swish + `conv_out` — decoding
`[batch, 32, H, W]` to `[batch, 3, 8H, 8W]`. Building blocks (`ResnetBlock.lean`):
`GroupNormParams`, `Conv2dParams`, `ResnetBlock in_ch out_ch` (with optional
`nin_shortcut`), `swish`.

`AutoEncoder dtypes` adds the BatchNorm running statistics used to normalize
packed latents. `AutoEncoderDTypes` retains independent exact `compute`, `mean`,
and `variance` dtypes, plus promotion witnesses ensuring that scaling and
offsetting latents return the decoder's compute dtype. Checkpoint loading
validates that relationship without casting stored statistics. The full decode
pipeline is:

```lean
variable {device : Device} {dtypes : AutoEncoderDTypes}

structure AutoEncoder (dtypes : AutoEncoderDTypes := float32AutoEncoderDTypes) where
  decoder : Decoder (device := device) dtypes.compute.val
  bn_running_mean : Tensor { shape := #[128], dtype := dtypes.mean.val, device := device }
  bn_running_var : Tensor { shape := #[128], dtype := dtypes.variance.val, device := device }
  deriving TensorStruct

def AutoEncoder.invNormalize (ae : AutoEncoder (device := device) dtypes)
    (z : Tensor ⟨#[1, 128, 16, 16], dtypes.compute.val, device⟩) :
    Tensor ⟨#[1, 128, 16, 16], dtypes.compute.val, device⟩
def AutoEncoder.decode (ae : AutoEncoder (device := device) dtypes)
    (z : Tensor ⟨#[1, 128, 16, 16], dtypes.compute.val, device⟩) :
    Tensor ⟨#[1, 3, 256, 256], dtypes.compute.val, device⟩
def loadAutoEncoder (path : String) (log : Handlers := {})
    (device : Device := .CPU) :
    IO (Σ dtypes : AutoEncoderDTypes, AutoEncoder (device := device) dtypes)
```

**Caveat.** `invNormalize`, `unpackLatents16x16`, `decode`, and `Decoder.forward1` are
hardcoded to the 256×256 path (concrete literal shapes to avoid an FFI issue with type
variables). Other output resolutions need new code — `Decoder.forward` is shape-generic
over `batch` only, not over spatial size.

## Key APIs

Flux + VAE:

| Entry point | Signature (abridged) | Location |
| --- | --- | --- |
| `loadFluxModel` | `(path : String) (cfg := FluxConfig.klein4B) … (device := .CPU) : IO (Σ dtype : RealFloatingDType, FluxModel (device := device) cfg dtype.val)` | `Flux/Weights.lean` |
| `computeImagePositionIds` / `computeTextPositionIds` | `[1, seq, 4]` RoPE ids | `Flux/Sampling.lean:109,131` |
| `denoise` | Euler rectified-flow loop, `num_steps := 4` | `Flux/Sampling.lean:74` |
| `sample` | fresh noise → `denoise` | `Flux/Sampling.lean:94` |
| `packLatents` / `unpackLatents` | VAE spatial ↔ packed sequence | `Flux/Sampling.lean:156,174` |
| `loadAutoEncoder` | `IO (Σ dtypes : AutoEncoderDTypes, AutoEncoder (device := device) dtypes)` | `VAE.lean` |
| `AutoEncoder.decode` | `Tensor ⟨#[1,128,16,16], dtypes.compute.val, device⟩ → Tensor ⟨#[1,3,256,256], dtypes.compute.val, device⟩` | `VAE.lean` |

## Usage examples

Reconstructed example (from `Examples/Flux/FluxDemo.lean`) — text-to-image with Flux
Klein 4B (256×256 path, 4-step distilled schedule):

```lean
import Tyr.Model.Qwen.Weights
import Tyr.Model.VAE
import Tyr.Model.Flux.Weights
import Tyr.Model.Flux
open torch torch.qwen torch.vae torch.flux
set_option synthInstance.maxSize 512 in
def fluxExample (qwenDir : String) (tokens : Tensor { shape := #[1, 512], dtype := .Int64, device := .CPU })
    (attnMask : Tensor { shape := #[1, 512], dtype := .Bool, device := .CPU }) : IO Unit := do

  -- text encoder (Qwen), diffusion model, and latent decoder
  let qwen ← loadQwenFluxEmbedderSharded qwenDir QwenConfig.fluxKleinTextEncoder 512 #[8, 17, 26]
  let ⟨_, storedFlux⟩ ← loadFluxModel "weights/flux.safetensors" FluxConfig.klein4B
  let flux ← TensorStruct.castTo (FluxModel (device := .CPU) FluxConfig.klein4B .Float32) storedFlux
  let ⟨_, storedVAE⟩ ← loadAutoEncoder "weights/ae.safetensors"
  let vae ← TensorStruct.castTo (AutoEncoder (device := .CPU) float32AutoEncoderDTypes) storedVAE

  -- encode prompt, prepare noise and RoPE position ids
  let ⟨hContext⟩ ← if h : qwen.outputLayers.size * QwenConfig.fluxKleinTextEncoder.hidden_size =
      FluxConfig.klein4B.context_in_dim then pure (PLift.up h) else
    throw <| IO.userError "text encoder width does not match Flux context_in_dim"
  let txtEmb := qwen.encodeMasked QwenConfig.fluxKleinTextEncoder 512 tokens attnMask
  let txtEmb : Tensor { dtype := .Float32, shape := #[1, 512, FluxConfig.klein4B.context_in_dim], device := .CPU } :=
    txtEmb.reshape #[1, 512, FluxConfig.klein4B.context_in_dim] (by simp [hContext])
  let noiseSeq := Tensor.permute (← randomWeight #[1, 128, 16, 16]) #[0, 2, 3, 1]
  let noise : Tensor { dtype := .Float32, shape := #[1, 256, 128], device := .CPU } := Tensor.reshape noiseSeq #[1, 256, 128] (by simp)
  let imgIds ← computeImagePositionIds 16 16
  let txtIds ← computeTextPositionIds 512

  -- denoise and decode
  let latents := denoise FluxConfig.klein4B flux noise txtEmb imgIds txtIds 4
  let packed : Tensor { dtype := .Float32, shape := #[1, 128, 16, 16], device := .CPU } :=
    (Tensor.permute (Tensor.reshape latents #[1, 16, 16, 128] (by simp [FluxConfig.klein4B]))
      #[0, 3, 1, 2]).reshape #[1, 128, 16, 16] (by simp)
  let image := vae.decoder.forward1 (AutoEncoder.unpackLatents16x16 (vae.invNormalize packed))
  saveImagePPM image "output.ppm"
```

## Related guides

- [core/tensors.md](../core/tensors.md) — `Tensor σ`, shapes, and indexed operations behind every forward here
- [core/tensorstruct.md](../core/tensorstruct.md) — parameter traversal, checkpointing, and target-type conversion
- [serialization.md](../serialization.md) — SafeTensors loading and checkpoint save/resume
- [llms.md](llms.md) — the Qwen text encoder used for Flux conditioning
- [examples-and-testing.md](../examples-and-testing.md) — the demo executables

Exhaustive symbol-level documentation for everything in this chapter is generated by doc-gen4 (see `docbuild/`); this guide covers the concepts and the main entry points only.
