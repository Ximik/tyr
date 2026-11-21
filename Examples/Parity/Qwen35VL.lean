/-
  Examples/Parity/Qwen35VL.lean

  Parity probe for `parity/tests/qwen35_vl.py`. Request as in
  `Examples.Parity.LM`, plus `"inputs"` holding the processor's `image_grid_thw`
  and `"tensors"`: a safetensors path holding its `pixel_values`
  (`[patches, patch_dim]`). Result as in
  `Examples.Parity.LM`.
-/
import Examples.Parity.LM
import Tyr.Model.Qwen35.VLConfigIO
import Tyr.Model.Qwen35.VLWeights
import Tyr.SafeTensors.Source

namespace Examples.Parity.Qwen35VL

open Lean torch torch.qwen35

structure ProcessorInputs where
  image_grid_thw : Array (Array Nat)
  deriving FromJson

structure Images where
  inputs : ProcessorInputs
  tensors : String
  deriving FromJson

set_option synthInstance.maxSize 1024 in
def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let inputs ← LM.Inputs.parse json device
  let images : Images ← IO.ofExcept (fromJson? json)
  let dtype ← inputs.request.realDType
  let weights := args.weights.toString
  let cfg ← VLConfig.loadFromPretrainedDir weights
  let ⟨_, _, model⟩ ← Qwen35ForConditionalGeneration.load weights cfg device
  let model ← TensorStruct.castTo (Qwen35ForConditionalGeneration (device := device) cfg dtype.val dtype.val) model
  let src ← safetensors.Source.open images.tensors
  let some spec := src.spec? "pixel_values" | throw <| IO.userError s!"{images.tensors} has no 'pixel_values'"
  let some patches := spec.shape[0]? | throw <| IO.userError s!"'pixel_values' has shape {spec.shape}"
  let pixels ← src.loadAs "pixel_values" ⟨#[patches, VisionConfig.patchDim cfg.vision_config], .Float32⟩ device
  let grids ← images.inputs.image_grid_thw.mapM fun
    | #[1, h, w] => pure (h, w)
    | grid => throw <| IO.userError s!"image grid {grid} is not one frame"
  let features ← model.getImageFeatures cfg (pixels.castDType dtype.val) grids
  let ⟨_, generated⟩ ← model.generate cfg inputs.prompts inputs.request.max_new_tokens .greedy #[] (some features)
  LM.output inputs.seq generated (← model.forcedLogits cfg inputs.prompts inputs.forced (some features))

end Examples.Parity.Qwen35VL

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.Qwen35VL.probe
