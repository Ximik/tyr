/-
  Examples/Parity/FluxSample.lean

  Parity probe for `parity/tests/flux_sample.py`: text encoding, denoising and
  VAE decoding of a 256x256 image. `--weights` is a FLUX.2-klein directory with
  `model_index.json`, `scheduler/`, `text_encoder/`, `transformer/config.json`,
  `flux-2-klein-4b.safetensors` and `vae/`. Request:
  `{"dtype", "input_ids", "attention_mask", "steps", "noise": <safetensors path>}`,
  the ids and mask being 512 each and the file holding `noise` (`[1, 256, 128]`).
  Writes the denoised `latents` (`[1, 256, 128]`) and the decoder's `image`
  (`[1, 3, 256, 256]`), both Float32.
-/
import Examples.Parity.Common
import Tyr.Model.Qwen
import Tyr.Model.Qwen.Weights
import Tyr.Model.Qwen3.ConfigIO
import Tyr.Model.Flux
import Tyr.Model.Flux.Weights
import Tyr.Model.VAE
import Tyr.IO.SafeTensors.Source

namespace Examples.Parity.FluxSample

open Lean torch torch.qwen torch.flux torch.vae

structure Request where
  dtype : String
  input_ids : Array Nat
  attention_mask : Array Nat
  steps : Nat
  noise : String
  deriving FromJson

def maxSeqLen : Nat := 512
def side : Nat := 16

/-- The prompt embedding, with the text encoder released before the caller
loads the transformer. -/
def encodeText (dir : String) (device : Device) (request : Request) (contextDim : Nat) :
    IO (Tensor { shape := #[1, maxSeqLen, contextDim], dtype := .Float32, device }) := do
  let ids ← IO.ofExcept (Tensor.fromInt64ArrayAs? (request.input_ids.map (·.toInt64)) #[1, maxSeqLen])
  let mask ← IO.ofExcept (Tensor.fromInt64ArrayAs? (request.attention_mask.map (·.toInt64)) #[1, maxSeqLen])
  let cfg ← qwen3.Config.loadFromPretrainedDir s!"{dir}/text_encoder"
  let embedder ← loadQwenFluxEmbedder s!"{dir}/text_encoder" cfg maxSeqLen (device := device)
  let ⟨h⟩ ← if h : embedder.outputLayers.size * cfg.hidden_size = contextDim then pure (PLift.up h) else
    throw <| IO.userError
      s!"text encoder emits {embedder.outputLayers.size * cfg.hidden_size} features, Flux expects {contextDim}"
  let embeddings := embedder.encodeMasked (batch := 1) (seq := maxSeqLen) cfg maxSeqLen
    (ids.to device) (mask.to device)
  pure (embeddings.reshape #[1, maxSeqLen, contextDim] (by simp [h]))

set_option synthInstance.maxSize 512 in
def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let request : Request ← IO.ofExcept (fromJson? json)
  let some dtype := DType.ofString? request.dtype
    | throw <| IO.userError s!"unknown dtype '{request.dtype}'"
  let dtype ← IO.ofExcept (RealFloatingDType.ofDType? dtype)
  let dir := args.weights.toString
  let cfg ← FluxConfig.loadFromPretrainedDir dir
  FluxConfig.checkPipelineDir dir
  let ⟨hChannels⟩ ← if h : cfg.in_channels = 128 then pure (PLift.up h) else
    throw <| IO.userError s!"the VAE decodes 128 packed latent channels, the transformer has {cfg.in_channels}"
  let text ← encodeText dir device request cfg.context_in_dim
  let ⟨_, model⟩ ← loadFluxModel s!"{dir}/flux-2-klein-4b.safetensors" cfg (device := device)
  let model ← TensorStruct.castTo (FluxModel (device := device) cfg dtype.val) model
  let src ← safetensors.Source.open request.noise
  let noise ← src.loadAs "noise" ⟨#[1, side * side, cfg.in_channels], .Float32⟩ device
  let imgIds ← computeImagePositionIds side side device
  let txtIds ← computeTextPositionIds maxSeqLen device
  let latents := denoise cfg model (noise.castDType dtype.val) (text.castDType dtype.val) imgIds txtIds
    request.steps
  let ⟨_, vae⟩ ← loadAutoEncoder s!"{dir}/vae" (device := device)
  let vae ← TensorStruct.castTo (AutoEncoder (device := device) float32AutoEncoderDTypes) vae
  let packed : Tensor { shape := #[1, 128, side, side], dtype := .Float32, device } :=
    (Tensor.permute ((latents.castDType .Float32).reshape #[1, side, side, cfg.in_channels]
      (by simp [Nat.mul_assoc])) #[0, 3, 1, 2]).reshape #[1, 128, side, side] (by simp [hChannels])
  let image := vae.decode packed (by decide) (by decide)
  pure {
    json := Json.mkObj []
    tensors := #[
      ("latents", Tensor.toTensor (latents.castDType .Float32)),
      ("image", Tensor.toTensor image)]
  }

end Examples.Parity.FluxSample

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.FluxSample.probe
