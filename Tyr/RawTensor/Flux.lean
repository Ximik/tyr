import Tyr.RawTensor
namespace torch.RawTensor
@[extern "lean_torch_timestep_embedding"]
opaque timestepEmbedding (t : @& RawTensor) (dim : @& Nat)
    (maxPeriod timeFactor : Float) : RawTensor
@[extern "lean_torch_save_ppm"]
opaque savePPM (image : @& RawTensor) (path : @& String) : IO Unit
end torch.RawTensor
