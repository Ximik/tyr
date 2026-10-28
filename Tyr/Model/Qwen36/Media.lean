/-
  Tyr/Model/Qwen36/Media.lean

  Multimodal media helpers for Qwen3.6. The public Qwen3.6-35B-A3B checkpoint
  uses the same vision patchification path as Tyr's shared Qwen3.5 VL
  implementation, so this module exposes a Qwen3.6-local namespace over the
  shared helpers.
-/
import Tyr.Model.Qwen35.Media
import Tyr.Model.Qwen36.Config

namespace torch.qwen36.media

open torch

def loadImagePatches (cfg : VLConfig) (path : String)
    : IO (Sigma (fun n => Tensor { device := .CPU, shape := #[n, VisionConfig.patchDim cfg.vision_config], dtype := .Float32 })) :=
  qwen35.media.loadImagePatches cfg path

def loadVideoPatches
    (cfg : VLConfig)
    (path : String)
    (maxFrames : Nat := 64)
    (frameStride : Nat := 1)
    : IO (Sigma (fun n => Tensor { device := .CPU, shape := #[n, VisionConfig.patchDim cfg.vision_config], dtype := .Float32 })) :=
  qwen35.media.loadVideoPatches cfg path maxFrames frameStride

end torch.qwen36.media
