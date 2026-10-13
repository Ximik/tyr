import Tyr.Typed.Neural
import Tyr.RawTensor.VAE
import Tyr.RawTensor.KittenTTS
import Tyr.Model.Qwen35.TypedOps

namespace torch.vae

def groupNorm {b c h w : UInt64} (x : STensor #[b, c, h, w])
    (groups : UInt64) (weight bias : STensor #[c]) (eps : Float) : STensor #[b, c, h, w] :=
  x.checkedReplacement (RawTensor.groupNorm x.shape x.raw groups (some weight.raw) (some bias.raw) eps)

def upsample2 {b c h w : UInt64} (x : STensor #[b, c, h, w]) : STensor #[b, c, h * 2, w * 2] :=
  match Tensor.ofRaw (RawTensor.interpolate x.shape x.raw #[h * 2, w * 2] "nearest" false) with
  | .ok y => y
  | .error message => panic! s!"VAE upsample: {message}"

def bmm {b m k n : UInt64} (x : STensor #[b, m, k]) (y : STensor #[b, k, n]) : STensor #[b, m, n] :=
  qwen35.einsumChecked "bmk,bkn->bmn" x y

end torch.vae
