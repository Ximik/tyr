import Tyr.Typed.Neural
import Tyr.RawTensor.VAE
import Tyr.RawTensor.KittenTTS
import Tyr.Model.Qwen35.TypedOps
import Tyr.TensorStruct.DeriveCast

namespace torch.vae

variable {device : Device} {dtype : DType}
def groupNorm {b c h w : Nat} (x : Tensor { dtype := dtype, device := device, shape := #[b, c, h, w] })
    (groups : Nat) (weight bias : Tensor { dtype := dtype, device := device, shape := #[c] }) (eps : Float) : Tensor { dtype := dtype, device := device, shape := #[b, c, h, w] } :=
  x.checkedReplacement (RawTensor.groupNorm x.raw groups (some weight.raw) (some bias.raw) eps)

def upsample2 {b c h w : Nat} (x : Tensor { dtype := dtype, device := device, shape := #[b, c, h, w] }) : Tensor { dtype := dtype, device := device, shape := #[b, c, h * 2, w * 2] } :=
  match Tensor.ofRaw (RawTensor.interpolate x.raw #[h * 2, w * 2] "nearest" false) with
  | .ok y => y
  | .error message => nativeInvariantFailure s!"VAE upsample: {message}"

def bmm {b m k n : Nat} (x : Tensor { dtype := dtype, device := device, shape := #[b, m, k] }) (y : Tensor { dtype := dtype, device := device, shape := #[b, k, n] }) : Tensor { dtype := dtype, device := device, shape := #[b, m, n] } :=
  qwen35.einsumChecked "bmk,bkn->bmn" x y

end torch.vae
