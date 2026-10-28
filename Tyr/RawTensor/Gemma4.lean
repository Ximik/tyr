import Tyr.RawTensor

namespace torch.RawTensor

@[extern "lean_torch_gemma4_text_experts_forward"]
opaque gemma4TextExperts (tokens numExperts topK inter hidden : @& Nat)
    (hiddenStates gateUpProj downProj topVals topIdx : @& RawTensor) : RawTensor

@[extern "lean_torch_sdpa_gqa_mask_qkv"]
opaque gemma4MaskedAttention
    (query key value mask : @& RawTensor) (gqa : Bool) : Except String RawTensor
@[extern "lean_torch_sdpa_gqa_mask_qkv_io"]
opaque gemma4MaskedAttentionIO
    (query key value mask : @& RawTensor) (dropout : Float) (gqa : Bool) : IO RawTensor

@[extern "lean_torch_softcap"]
opaque gemma4Softcap (input : @& RawTensor) (cap : Float) : RawTensor

@[extern "lean_torch_media_load_image_patch_grid_gemma4"]
opaque gemma4LoadImage (path : @& String) (patchSize poolingKernelSize maxSoftTokens : @& Nat)
    (rescaleFactor : Float) : IO RawTensor

@[extern "lean_torch_avg_pool2d"] opaque gemma4AvgPool2d
    (input : @& RawTensor) (kernel stride padding : Array Nat) : RawTensor

end torch.RawTensor
