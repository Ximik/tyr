# Generative models

This chapter covers Tyr's **Flux + VAE** text-to-image stack (a Flux2 Klein 4B
rectified-flow diffusion transformer plus the decoder of its diffusers
`AutoencoderKLFlux2`). Use Flux for image generation in a learned latent space.

Everything here is shape-typed Lean 4 over the libtorch FFI. Namespaces:
`torch.flux`, `torch.vae`.

## Architecture and main abstractions

### Flux (`Tyr/Model/Flux/`)

A Flux2 (Klein 4B) rectified-flow image diffusion transformer. `FluxConfig` carries the
architecture dimensions; `FluxConfig.loadFromPretrainedDir` reads them from a diffusers
checkout's `transformer/config.json` and rejects settings the model does not implement.

`FluxModel cfg dtype` is one flat `deriving TensorStruct` record sharing its
compute dtype across input
projections (`img_in`, `txt_in`), a timestep `MLPEmbedder`, arrays of
`DoubleStreamBlock`s (separate img/txt QKV+SwiGLU paths, joint attention) and
`SingleStreamBlock`s (fused QKV+MLP over the concatenated sequence), three model-level
adaLN `Modulation` layers, and a `LastLayer`. Q/K are RMS-normalized (`QKNorm`); positions
use 4-axis RoPE (`ropeEmbed`/`applyRope`, axes `#[t, h, w, l]`, `theta := 2000`), and the
timestep embedding is a C++ FFI extern (`@[extern "lean_torch_timestep_embedding"]`).
The forward takes packed image latents, text
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
empirical-μ time-SNR shift) and Euler integration: `eulerStep`, `denoise` over an
explicit noise tensor, `sample` drawing fresh noise. `FluxConfig.checkPipelineDir`
rejects a checkout whose `model_index.json` or `scheduler/scheduler_config.json` describes
another sampling.
`computeImagePositionIds height width` and `computeTextPositionIds seq_len` produce the
`[1, seq, 4]` RoPE inputs; `packLatents`/`unpackLatents` convert between VAE
spatial format `[batch, 32, H, W]` and the 2×2-patchified sequence format
`[batch, (H/2)·(W/2), 128]`. Weights load from the single-file checkpoint
(`flux-2-klein-4b.safetensors`) with `loadFluxModel path cfg`. Its result is
`IO (Σ dtype : RealFloatingDType, FluxModel (device := device) cfg dtype.val)`:
the loader validates the input projection's floating dtype and checks connected
weights against it, retaining that evidence for forward computation. Timestep and RoPE calculations retain their explicit Float32 paths.
Text conditioning
is **not** part of this module: the model consumes pre-computed embeddings
(`context_in_dim := 7680`), produced in the demo by a Qwen encoder — see
[LLM model families](llms.md).

### VAE (`Tyr/Model/VAE.lean`, `Tyr/Model/VAE/`)

The decoder of the diffusers `AutoencoderKLFlux2`. `Decoder` is the stack —
`post_quant_conv`, `conv_in`, mid `ResnetBlock`/`AttnBlock`/`ResnetBlock`, four
up-groups with `Upsample`s, GroupNorm + swish + `conv_out` — decoding
`[batch, 32, H, W]` to `[batch, 3, 8H, 8W]`. Building blocks (`ResnetBlock.lean`):
`GroupNormParams`, `Conv2dParams`, `ResnetBlock in_ch out_ch` (with optional
`nin_shortcut`), `swish`.

`AutoEncoder dtypes` adds the BatchNorm running statistics used to normalize
packed latents, and their epsilon. `AutoEncoderDTypes` retains independent exact `compute`, `mean`,
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
  bn_eps : Static Float
  deriving TensorStruct

def AutoEncoder.decode {batch height width : Nat} (ae : AutoEncoder (device := device) dtypes)
    (z : Tensor ⟨#[batch, 128, height, width], dtypes.compute.val, device⟩)
    (hh : 0 < height) (hw : 0 < width) :
    Tensor ⟨#[batch, 3, height * 2 * 2 * 2 * 2, width * 2 * 2 * 2 * 2], dtypes.compute.val, device⟩
def loadAutoEncoder (dir : String) (log : Handlers := {})
    (device : Device := .CPU) :
    IO (Σ dtypes : AutoEncoderDTypes, AutoEncoder (device := device) dtypes)
```

`loadAutoEncoder` reads a diffusers `vae/` directory: `config.json`, which must describe
the decoder layout above, and `diffusion_pytorch_model.safetensors`.

## Key APIs

Flux + VAE:

| Entry point | Signature (abridged) | Location |
| --- | --- | --- |
| `FluxConfig.loadFromPretrainedDir` / `checkPipelineDir` | transformer config / sampling check of a diffusers checkout | `Flux/ConfigIO.lean` |
| `loadFluxModel` | `(path : String) (cfg : FluxConfig) … (device := .CPU) : IO (Σ dtype : RealFloatingDType, FluxModel (device := device) cfg dtype.val)` | `Flux/Weights.lean` |
| `computeImagePositionIds` / `computeTextPositionIds` | `[1, seq, 4]` RoPE ids | `Flux/Sampling.lean` |
| `denoise` | Euler rectified-flow loop, `num_steps := 4` | `Flux/Sampling.lean` |
| `sample` | fresh noise → `denoise` | `Flux/Sampling.lean` |
| `packLatents` / `unpackLatents` | VAE spatial ↔ packed sequence | `Flux/Sampling.lean` |
| `loadAutoEncoder` | `(dir : String) … : IO (Σ dtypes : AutoEncoderDTypes, AutoEncoder (device := device) dtypes)` | `VAE.lean` |
| `AutoEncoder.decode` | packed latents `[batch, 128, h, w]` → image `[batch, 3, 16h, 16w]` | `VAE.lean` |

## Usage examples

`Examples/Flux/FluxDemo.lean` runs the whole pipeline: the Qwen3 text encoder, `denoise`
over 4 steps, and `AutoEncoder.decode`, from one FLUX.2-klein checkout.

## Related guides

- [core/tensors.md](../core/tensors.md) — `Tensor σ`, shapes, and indexed operations behind every forward here
- [core/tensorstruct.md](../core/tensorstruct.md) — parameter traversal, checkpointing, and target-type conversion
- [serialization.md](../serialization.md) — SafeTensors loading and checkpoint save/resume
- [llms.md](llms.md) — the Qwen text encoder used for Flux conditioning
- [examples-and-testing.md](../examples-and-testing.md) — the demo executables

Exhaustive symbol-level documentation for everything in this chapter is generated by doc-gen4 (see `docbuild/`); this guide covers the concepts and the main entry points only.
