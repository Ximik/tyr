import Tyr.Core.FFI.RawTensor

namespace torch.RawTensor

@[extern "lean_torch_chunk_gated_delta_rule"] opaque chunkGatedDeltaRule
    (query key value g beta : @& RawTensor) : RawTensor × RawTensor
@[extern "lean_torch_recurrent_gated_delta_rule"] opaque recurrentGatedDeltaRule
    (query key value g beta state : @& RawTensor) : RawTensor × RawTensor

end torch.RawTensor
