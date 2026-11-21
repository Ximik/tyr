/-
  Examples/Parity/Gemma4VL.lean

  Parity probe for `parity/tests/gemma4_vl.py`. Request as in
  `Examples.Parity.LM`, plus `"inputs"` holding the processor's
  `image_position_ids` (`[1, patches, 2]`, `(x, y)` per patch, padding
  negative) and `"tensors"`: a safetensors path holding its `pixel_values`
  (`[1, patches, patch_dim]`). The patches must be in row-major order. Result
  as in `Examples.Parity.LM`.
-/
import Examples.Parity.LM
import Tyr.Model.Gemma4.VLConfigIO
import Tyr.Model.Gemma4.VLWeights
import Tyr.SafeTensors.Source

namespace Examples.Parity.Gemma4VL

open Lean torch torch.gemma4

structure ProcessorInputs where
  image_position_ids : Array (Array (Array Int))
  deriving FromJson

structure Images where
  inputs : ProcessorInputs
  tensors : String
  deriving FromJson

/-- Rows and columns of the patch grid, after checking that the valid patches
come first, in row-major order. -/
def gridSize (positions : Array (Array Int)) : IO (Nat × Nat) := do
  let valid := positions.filter fun p => p.all (0 ≤ ·)
  let cols := (valid.foldl (fun m p => max m (p[0]!.toNat + 1)) 0)
  let rows := (valid.foldl (fun m p => max m (p[1]!.toNat + 1)) 0)
  for i in [:valid.size] do
    let p := positions[i]!
    unless p[0]! == (i % cols : Nat) && p[1]! == (i / cols : Nat) do
      throw <| IO.userError s!"patch {i} is at {p}, not in row-major order of a {rows}x{cols} grid"
  unless valid.size == rows * cols do
    throw <| IO.userError s!"{valid.size} valid patches for a {rows}x{cols} grid"
  pure (rows, cols)

set_option synthInstance.maxSize 4096 in
def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let inputs ← LM.Inputs.parse json device
  let images : Images ← IO.ofExcept (fromJson? json)
  let dtype ← inputs.request.realDType
  let weights := args.weights.toString
  let cfg ← VLConfig.loadFromPretrainedDir weights
  let ⟨textDType, _, _, model⟩ ← Gemma4ForConditionalGeneration.load weights cfg (device := device) (textDType? := some dtype)
  let ⟨hText⟩ ← if h : textDType.val = dtype.val then pure (PLift.up h) else
    throw <| IO.userError s!"the text model loaded as {textDType.val}, not {dtype.val}"
  let model : Gemma4ForConditionalGeneration (device := device) cfg dtype.val dtype.val dtype.val := {
    vision_tower := ← TensorStruct.castTo (Gemma4VisionModel (dtype := dtype.val) (device := device) cfg) model.vision_tower
    embed_vision := ← TensorStruct.castTo (Gemma4MultimodalEmbedder (dtype := dtype.val) (device := device) cfg) model.embed_vision
    language_model := hText ▸ model.language_model }
  let some positions := images.inputs.image_position_ids[0]?
    | throw <| IO.userError "no image_position_ids"
  let (rows, cols) ← gridSize positions
  let ⟨hGrid⟩ ← if h : rows * cols ≤ positions.size then pure (PLift.up h) else
    throw <| IO.userError s!"a {rows}x{cols} grid from {positions.size} patches"
  let src ← safetensors.Source.open images.tensors
  let patchDim := VisionConfig.patchDim cfg.vision_config
  let pixels ← src.loadAs "pixel_values" ⟨#[1, positions.size, patchDim], .Float32⟩ device
  let flat : Tensor { device := device, shape := #[positions.size, patchDim], dtype := .Float32 } :=
    Tensor.reshape pixels #[positions.size, patchDim]
  let valid : Tensor { device := device, shape := #[rows * cols, patchDim], dtype := .Float32 } :=
    Tensor.slice flat 0 0 (rows * cols) (_bounds := by simpa using hGrid)
  let grid : Tensor { device := device, shape := #[rows, cols, patchDim], dtype := .Float32 } :=
    Tensor.reshape valid #[rows, cols, patchDim]
  let features := model.getImageFeatures cfg (grid.castDType dtype.val)
  let imageFeatures := some ⟨_, features⟩
  let ⟨_, generated⟩ ← model.generate cfg inputs.prompts inputs.request.max_new_tokens .greedy #[] imageFeatures
  LM.output inputs.seq generated (← model.forcedLogits cfg inputs.prompts inputs.forced imageFeatures)

end Examples.Parity.Gemma4VL

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.Gemma4VL.probe
