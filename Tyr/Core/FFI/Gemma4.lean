import Tyr.Core.FFI.RawTensor

namespace torch.RawTensor

@[extern "lean_torch_gemma4_text_experts_forward"]
opaque gemma4TextExperts (tokens numExperts topK inter hidden : @& Nat)
    (hiddenStates gateUpProj downProj topVals topIdx : @& RawTensor) : RawTensor

@[extern "lean_torch_media_load_image_patch_grid_gemma4"]
opaque gemma4LoadImage (path : @& String) (patchSize poolingKernelSize maxSoftTokens : @& Nat)
    (rescaleFactor : Float) : IO RawTensor

end torch.RawTensor
