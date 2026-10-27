import Tyr.RawTensor

namespace torch.RawTensor
@[extern "lean_torch_leaky_relu"] opaque leakyRelu (input : @& RawTensor) (slope : Float) : RawTensor
@[extern "lean_torch_tensor_floor"] opaque floor (input : @& RawTensor) : RawTensor
@[extern "lean_torch_atan2"] opaque atan2 (y x : @& RawTensor) : RawTensor
@[extern "lean_torch_cumsum"] opaque cumsum (input : @& RawTensor) (dim : Int64) : RawTensor
@[extern "lean_torch_interpolate"] opaque interpolate (input : @& RawTensor)
    (size : Array UInt64) (mode : String) (alignCorners : Bool) : RawTensor
@[extern "lean_torch_istft_1d"] opaque istft (input : @& RawTensor)
    (nFft hopLength winLength : UInt64) (window : @& RawTensor)
    (center normalized : Bool) (length : UInt64) : RawTensor
end torch.RawTensor
