import Tyr.Core.FFI.RawTensor

namespace torch.RawTensor

@[extern "lean_torch_media_load_image_patchified"] opaque loadImagePatchified
    (path : @& String) (inChannels patchSize temporalPatchSize : @& Nat) : IO RawTensor
@[extern "lean_torch_media_load_video_patchified"] opaque loadVideoPatchified
    (path : @& String) (inChannels patchSize temporalPatchSize maxFrames frameStride : @& Nat) : IO RawTensor

end torch.RawTensor
