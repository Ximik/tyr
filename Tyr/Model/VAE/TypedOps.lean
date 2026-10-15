import Tyr.Typed.Neural
import Tyr.RawTensor.VAE
import Tyr.RawTensor.KittenTTS
import Tyr.Model.Qwen35.TypedOps

namespace torch.vae

variable {device : Device}
def groupNorm {b c h w : UInt64} (x : Tensor { device := device, shape := #[b, c, h, w] })
    (groups : UInt64) (weight bias : Tensor { device := device, shape := #[c] }) (eps : Float) : Tensor { device := device, shape := #[b, c, h, w] } :=
  x.checkedReplacement (RawTensor.groupNorm x.shape x.raw groups (some weight.raw) (some bias.raw) eps)

def upsample2 {b c h w : UInt64} (x : Tensor { device := device, shape := #[b, c, h, w] }) : Tensor { device := device, shape := #[b, c, h * 2, w * 2] } :=
  match Tensor.ofRaw (RawTensor.interpolate x.shape x.raw #[h * 2, w * 2] "nearest" false) with
  | .ok y => y
  | .error message => panic! s!"VAE upsample: {message}"

def bmm {b m k n : UInt64} (x : Tensor { device := device, shape := #[b, m, k] }) (y : Tensor { device := device, shape := #[b, k, n] }) : Tensor { device := device, shape := #[b, m, n] } :=
  qwen35.einsumChecked "bmk,bkn->bmn" x y

end torch.vae
