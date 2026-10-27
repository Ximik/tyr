import Tyr.RawTensor

namespace torch.RawTensor

@[extern "lean_torch_qwen35_chunk_gated_delta_rule"] opaque qwen35Chunk
    (batch seq heads keyDim valueDim : UInt64)
    (query key value g beta : @& RawTensor) : RawTensor × RawTensor
@[extern "lean_torch_qwen35_recurrent_gated_delta_rule"] opaque qwen35Recurrent
    (batch seq heads keyDim valueDim : UInt64)
    (query key value g beta state : @& RawTensor) : RawTensor × RawTensor
@[extern "lean_torch_einsum2"] opaque einsum2 (shapeA shapeB : Shape)
    (equation : String) (a b : @& RawTensor) : RawTensor
@[extern "lean_torch_topk_2d"] opaque topK2d (rows cols : UInt64)
    (input : @& RawTensor) (k dim : UInt64) : RawTensor × RawTensor
@[extern "lean_torch_tensor_softplus"] opaque softplus (shape : Shape) (input : @& RawTensor) : RawTensor

@[extern "lean_torch_media_load_image_patchified"] opaque loadImagePatchified
    (path : @& String) (inChannels patchSize temporalPatchSize : UInt64) : IO RawTensor
@[extern "lean_torch_media_load_video_patchified"] opaque loadVideoPatchified
    (path : @& String) (inChannels patchSize temporalPatchSize maxFrames frameStride : UInt64) : IO RawTensor

end torch.RawTensor
