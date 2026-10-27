import Tyr.RawTensor

namespace torch.RawTensor
@[extern "lean_torch_copy_inplace"] opaque copyInplace
    (dst : RawTensor) (src : @& RawTensor) : IO RawTensor
@[extern "lean_torch_touch"] opaque touch (input : @& RawTensor) : IO Unit
@[extern "lean_torch_mul_tensor_scalar"] opaque mulTensorScalar
    (input scalar : @& RawTensor) : RawTensor
@[extern "lean_torch_tensor_softplus"] opaque branchingSoftplus (input : @& RawTensor) : RawTensor
@[extern "lean_torch_sdpa_4d_bias"] opaque branchingAttentionBias
    (q k v bias : @& RawTensor) (causal : Bool) : Except String RawTensor
@[extern "lean_torch_sdpa_4d_bias_io"] opaque branchingAttentionBiasIO
    (q k v bias : @& RawTensor) (dropout : Float) (causal : Bool) : IO RawTensor
@[extern "lean_torch_binary_cross_entropy"] opaque branchingBCE
    (input target : @& RawTensor) (weight : Option RawTensor) (reduction : String) : RawTensor
end torch.RawTensor

namespace torch.RawTensor
@[extern "lean_torch_mul_tensor_scalar_checked"] opaque mulTensorScalarChecked
    (input scalar : @& RawTensor) : Except String RawTensor
end torch.RawTensor
