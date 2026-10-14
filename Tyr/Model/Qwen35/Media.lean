/-  
  Tyr/Model/Qwen35/Media.lean

  Apple-only media preprocessing helpers for Qwen3.5-VL:
  - image/video file -> patchified tensor `[nPatches, patchDim]`
-/
import Tyr.Typed.Neural
import Tyr.RawTensor.Qwen35
import Tyr.Model.Qwen35.VLConfig

namespace torch.qwen35.media

open torch
open torch.qwen35

private def parsePatchified {patchDim : UInt64} (dyn : RawTensor) : IO (Sigma (fun n => Tensor { shape := #[n, patchDim], dtype := .Float32 })) := do
  let shp := RawTensor.runtimeShape dyn
  if shp.size != 2 then
    throw <| IO.userError s!"Expected patchified tensor rank 2, got shape={shp}"
  let nPatches := shp.getD 0 0
  let gotPatchDim := shp.getD 1 0
  if gotPatchDim != patchDim then
    throw <| IO.userError
      s!"Patch dim mismatch: expected {patchDim}, got {gotPatchDim}"
  match Tensor.ofRaw (σ := ⟨#[nPatches, patchDim], .Float32, .any⟩) dyn with
  | .ok tensor => pure ⟨nPatches, tensor⟩
  | .error message => throw (IO.userError message)

/-- Load and patchify one image file for `cfg` (Apple-only path). -/
def loadImagePatches (cfg : VLConfig) (path : String)
    : IO (Sigma (fun n => Tensor { shape := #[n, VisionConfig.patchDim cfg.vision_config], dtype := .Float32 })) := do
  let dyn ← RawTensor.loadImagePatchified
    path
    cfg.vision_config.in_channels
    cfg.vision_config.patch_size
    cfg.vision_config.temporal_patch_size
  parsePatchified (patchDim := VisionConfig.patchDim cfg.vision_config) dyn

/-- Load and patchify one video file for `cfg` (Apple-only path). -/
def loadVideoPatches
    (cfg : VLConfig)
    (path : String)
    (maxFrames : UInt64 := 64)
    (frameStride : UInt64 := 1)
    : IO (Sigma (fun n => Tensor { shape := #[n, VisionConfig.patchDim cfg.vision_config], dtype := .Float32 })) := do
  let dyn ← RawTensor.loadVideoPatchified
    path
    cfg.vision_config.in_channels
    cfg.vision_config.patch_size
    cfg.vision_config.temporal_patch_size
    maxFrames
    frameStride
  parsePatchified (patchDim := VisionConfig.patchDim cfg.vision_config) dyn

end torch.qwen35.media
