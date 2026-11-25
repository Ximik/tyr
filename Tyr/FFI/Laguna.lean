import Tyr.FFI.RawTensor

namespace torch.RawTensor

/-- Raw FFI binding. Call `torch.laguna.lagunaMoeFp4Forward` instead, which validates
    shapes/dtypes/devices first. Returns BF16 `[tokens, hidden]`. -/
@[extern "lean_torch_laguna_moe_fp4_forward"]
opaque lagunaMoeFp4Forward
    (x : @& RawTensor) (topIdx : @& RawTensor) (topW : @& RawTensor)
    (gatePacked : @& RawTensor) (gateScale : @& RawTensor) (gateGlobal : @& RawTensor)
    (upPacked : @& RawTensor) (upScale : @& RawTensor) (upGlobal : @& RawTensor)
    (downPacked : @& RawTensor) (downScale : @& RawTensor) (downGlobal : @& RawTensor)
    (numExperts : @& Nat) (moeInt : @& Nat) (hidden : @& Nat) : IO RawTensor

end torch.RawTensor
