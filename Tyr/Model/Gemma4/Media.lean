/-
  Tyr/Model/Gemma4/Media.lean

  Apple-only image preprocessing helpers for Gemma 4 multimodal inputs.
-/
import Tyr.Typed.Neural
import Tyr.RawTensor.Gemma4
import Tyr.Model.Gemma4.VLConfig

namespace torch.gemma4.media

open torch
open torch.gemma4

/-- Load one image and return a patch grid `[patchRows, patchCols, patchDim]`
    sized for the configured Gemma 4 image token budget. -/
def loadImagePatchGrid (cfg : VLConfig) (path : String)
    : IO (Sigma (fun patchRows =>
        Sigma (fun patchCols =>
          Tensor { device := .CPU, shape := #[patchRows, patchCols, VisionConfig.patchDim cfg.vision_config], dtype := .Float32 }))) := do
  let dyn ← RawTensor.gemma4LoadImage
    path
    cfg.vision_config.patch_size
    cfg.vision_config.pooling_kernel_size
    cfg.imageSoftTokenCount
    cfg.image_processor.rescale_factor
  let shp := dyn.runtimeShape
  if shp.size != 3 then
    throw <| IO.userError s!"Expected Gemma4 image patch grid rank 3, got shape={shp}"
  let patchRows := shp.getD 0 0
  let patchCols := shp.getD 1 0
  let patchDim := shp.getD 2 0
  if patchDim != VisionConfig.patchDim cfg.vision_config then
    throw <| IO.userError
      s!"Gemma4 image patch dim mismatch: expected {VisionConfig.patchDim cfg.vision_config}, got {patchDim}"
  match Tensor.ofRaw (σ := { device := .CPU, shape := #[patchRows, patchCols, VisionConfig.patchDim cfg.vision_config], dtype := .Float32 }) dyn with
  | .ok grid => pure ⟨patchRows, ⟨patchCols, grid⟩⟩
  | .error message => throw <| IO.userError s!"Gemma4 image tensor: {message}"

end torch.gemma4.media
