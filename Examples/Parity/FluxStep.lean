/-
  Examples/Parity/FluxStep.lean

  Parity probe for `parity/tests/flux_step.py`. `--weights` is a FLUX.2-klein
  directory with `transformer/config.json` and
  `flux-2-klein-4b.safetensors`. Request:
  `{"dtype", "height", "width", "inputs": <safetensors path>}`, the file holding
  `latents` (`[1, height * width, 128]`), `text` (`[1, tokens, 7680]`) and
  `timestep` (`[1]`, in `[0, 1]`); `height` and `width` count packed latents.
  Writes one transformer prediction (`velocity`, Float32, the shape of `latents`).
-/
import Examples.Parity.Common
import Tyr.Model.Flux
import Tyr.Model.Flux.Weights
import Tyr.SafeTensors.Source

namespace Examples.Parity.FluxStep

open Lean torch torch.flux

structure Request where
  dtype : String
  height : Nat
  width : Nat
  inputs : String
  deriving FromJson

def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let request : Request ← IO.ofExcept (fromJson? json)
  let some dtype := DType.ofString? request.dtype
    | throw <| IO.userError s!"unknown dtype '{request.dtype}'"
  let dtype ← IO.ofExcept (RealFloatingDType.ofDType? dtype)
  let cfg ← FluxConfig.loadFromPretrainedDir args.weights.toString
  let ⟨_, model⟩ ← loadFluxModel s!"{args.weights}/flux-2-klein-4b.safetensors" cfg (device := device)
  let model ← TensorStruct.castTo (FluxModel (device := device) cfg dtype.val) model
  let src ← safetensors.Source.open request.inputs
  let some textSpec := src.spec? "text" | throw <| IO.userError s!"{request.inputs} has no 'text'"
  let some tokens := textSpec.shape[1]? | throw <| IO.userError s!"'text' has shape {textSpec.shape}"
  let imgSeq := request.height * request.width
  let latents ← src.loadAs "latents" ⟨#[1, imgSeq, cfg.in_channels], .Float32⟩ device
  let text ← src.loadAs "text" ⟨#[1, tokens, cfg.context_in_dim], .Float32⟩ device
  let timestep ← src.loadAs "timestep" ⟨#[1], .Float32⟩ device
  let imgIds ← computeImagePositionIds request.height request.width device
  let txtIds ← computeTextPositionIds tokens device
  let velocity := model.forward cfg (latents.castDType dtype.val) (text.castDType dtype.val) timestep
    imgIds txtIds
  pure {
    json := Json.mkObj []
    tensors := #[("velocity", Tensor.toTensor (velocity.castDType .Float32))]
  }

end Examples.Parity.FluxStep

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.FluxStep.probe
