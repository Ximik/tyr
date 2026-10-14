import Tyr.RawTensor

namespace torch.RawTensor
@[extern "lean_torch_copy_inplace"] opaque copyInplace (shape : Shape)
    (dst : RawTensor) (src : @& RawTensor) : IO RawTensor
@[extern "lean_torch_touch"] opaque touch (shape : Shape) (input : @& RawTensor) : IO Unit
@[extern "lean_torch_mul_tensor_scalar"] opaque mulTensorScalar (shape : Shape)
    (input scalar : @& RawTensor) : RawTensor
@[extern "lean_torch_tensor_softplus"] opaque branchingSoftplus (shape : Shape) (input : @& RawTensor) : RawTensor
@[extern "lean_torch_sdpa_4d_bias"] opaque branchingAttentionBias (batch heads seq dim : UInt64)
    (q k v bias : @& RawTensor) (dropout : Float) (causal : Bool) : RawTensor
@[extern "lean_torch_binary_cross_entropy"] opaque branchingBCE (shape : Shape)
    (input target : @& RawTensor) (weight : Option RawTensor) (reduction : String) : RawTensor
end torch.RawTensor
