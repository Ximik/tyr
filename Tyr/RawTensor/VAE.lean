import Tyr.RawTensor

namespace torch.RawTensor
@[extern "lean_torch_group_norm"]
opaque groupNorm (shape : Shape) (input : @& RawTensor) (groups : UInt64)
    (weight bias : Option RawTensor) (eps : Float) : RawTensor
end torch.RawTensor
