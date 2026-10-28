import Tyr.RawTensor

namespace torch.RawTensor

@[extern "lean_torch_qwen35_chunk_gated_delta_rule"] opaque qwen35Chunk

    (query key value g beta : @& RawTensor) : RawTensor × RawTensor
@[extern "lean_torch_qwen35_recurrent_gated_delta_rule"] opaque qwen35Recurrent

    (query key value g beta state : @& RawTensor) : RawTensor × RawTensor
@[extern "lean_torch_einsum2"] opaque einsum2
    (equation : String) (a b : @& RawTensor) : RawTensor
@[extern "lean_torch_topk_2d"] opaque topK2d
    (input : @& RawTensor) (k dim : @& Nat) : RawTensor × RawTensor
@[extern "lean_torch_tensor_softplus"] opaque softplus (input : @& RawTensor) : RawTensor

@[extern "lean_torch_media_load_image_patchified"] opaque loadImagePatchified
    (path : @& String) (inChannels patchSize temporalPatchSize : @& Nat) : IO RawTensor
@[extern "lean_torch_media_load_video_patchified"] opaque loadVideoPatchified
    (path : @& String) (inChannels patchSize temporalPatchSize maxFrames frameStride : @& Nat) : IO RawTensor

end torch.RawTensor
