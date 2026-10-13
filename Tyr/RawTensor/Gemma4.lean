import Tyr.RawTensor

namespace torch.RawTensor

@[extern "lean_torch_gemma4_text_experts_forward"]
opaque gemma4TextExperts (tokens numExperts topK inter hidden : UInt64)
    (hiddenStates gateUpProj downProj topVals topIdx : @& RawTensor) : RawTensor

@[extern "lean_torch_sdpa_gqa_mask_qkv"]
opaque gemma4MaskedAttention (batch heads kvHeads qSeq kvSeq dim : UInt64)
    (query key value mask : @& RawTensor) (dropout : Float) (gqa : Bool) : RawTensor

@[extern "lean_torch_softcap"]
opaque gemma4Softcap (shape : Shape) (input : @& RawTensor) (cap : Float) : RawTensor

@[extern "lean_torch_media_load_image_patch_grid_gemma4"]
opaque gemma4LoadImage (path : @& String) (patchSize poolingKernelSize maxSoftTokens : UInt64)
    (rescaleFactor : Float) : IO RawTensor

@[extern "lean_torch_avg_pool2d"] opaque gemma4AvgPool2d (shape : Shape)
    (input : @& RawTensor) (kernel stride padding : Array UInt64) : RawTensor

end torch.RawTensor
