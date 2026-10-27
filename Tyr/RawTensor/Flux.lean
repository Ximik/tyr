import Tyr.RawTensor
namespace torch.RawTensor
@[extern "lean_torch_timestep_embedding"]
opaque timestepEmbedding (batch : UInt64) (t : @& RawTensor) (dim : UInt64)
    (maxPeriod timeFactor : Float) : RawTensor
@[extern "lean_torch_save_ppm"]
opaque savePPM (shape : Shape) (image : @& RawTensor) (path : @& String) : IO Unit
end torch.RawTensor
