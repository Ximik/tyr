import Tyr.Typed.Neural
import Tyr.RawTensor.VAE
import Tyr.RawTensor.KittenTTS
import Tyr.TensorStruct.DeriveCast

namespace torch.vae

variable {device : Device} {dtype : DType}
def groupNorm {b c h w : Nat} (x : Tensor { dtype := dtype, device := device, shape := #[b, c, h, w] })
    (groups : Nat) (weight bias : Tensor { dtype := dtype, device := device, shape := #[c] }) (eps : Float) : Tensor { dtype := dtype, device := device, shape := #[b, c, h, w] } :=
  Tensor.ofRaw! (RawTensor.groupNorm x.raw groups (some weight.raw) (some bias.raw) eps)

def upsample2 {b c h w : Nat} (x : Tensor { dtype := dtype, device := device, shape := #[b, c, h, w] }) : Tensor { dtype := dtype, device := device, shape := #[b, c, h * 2, w * 2] } :=
  (Tensor.ofRaw! (RawTensor.interpolate x.raw #[h * 2, w * 2] "nearest" false))

end torch.vae
