import Tyr.RawTensor

namespace torch.RawTensor
@[extern "lean_torch_group_norm"]
opaque groupNorm (input : @& RawTensor) (groups : @& Nat)
    (weight bias : Option RawTensor) (eps : Float) : RawTensor
end torch.RawTensor
