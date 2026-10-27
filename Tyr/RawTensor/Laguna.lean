import Tyr.RawTensor

namespace torch.RawTensor

/-- Raw FFI binding. Call `torch.laguna.lagunaMoeFp4Forward` instead, which validates
    shapes/dtypes/devices first. Returns BF16 `[tokens, hidden]`. -/
@[extern "lean_torch_laguna_moe_fp4_forward"]
opaque lagunaMoeFp4Forward
    (x : @& RawTensor) (topIdx : @& RawTensor) (topW : @& RawTensor)
    (gatePacked : @& RawTensor) (gateScale : @& RawTensor) (gateGlobal : @& RawTensor)
    (upPacked : @& RawTensor) (upScale : @& RawTensor) (upGlobal : @& RawTensor)
    (downPacked : @& RawTensor) (downScale : @& RawTensor) (downGlobal : @& RawTensor)
    (numExperts : UInt64) (moeInt : UInt64) (hidden : UInt64) : IO RawTensor

@[extern "lean_torch_bmm"] opaque lagunaBmm
    (input mat2 : @& RawTensor) : RawTensor
@[extern "lean_torch_scatter_add"] opaque lagunaScatterAdd
    (input : @& RawTensor) (dim : Int64) (indices src : @& RawTensor) : RawTensor

@[extern "lean_torch_has_grad_fn"] opaque lagunaHasGradFn
    (input : @& RawTensor) : Bool

end torch.RawTensor
