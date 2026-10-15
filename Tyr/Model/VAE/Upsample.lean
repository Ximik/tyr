/-
  Tyr/Model/VAE/Upsample.lean

  Upsampling block for VAE decoder.
  Nearest-neighbor 2× upsample + 3×3 convolution.
-/
import Tyr.Model.VAE.TypedOps
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.VAE.ResnetBlock

namespace torch.vae

variable {device : Device}
/-- Upsampling block: 2× nearest-neighbor upsample + 3×3 conv -/
structure Upsample (channels : UInt64) where
  conv : Conv2dParams (device := device) channels channels 3 1
  deriving TensorStruct

namespace Upsample

/-- Initialize upsample block -/
def init (channels : UInt64) (device : Device := .CPU) : IO (Upsample (device := device) channels) := do
  let conv ← Conv2dParams.init (device := device) channels channels 3 1
  pure { conv }

/-- Forward pass: 2× upsample + conv.
    Input: [batch, channels, height, width]
    Output: [batch, channels, height*2, width*2] -/
def forward {batch channels height width : UInt64}
    (up : Upsample (device := device) channels)
    (x : Tensor { device := device, shape := #[batch, channels, height, width] })
    : Tensor { device := device, shape := #[batch, channels, height * 2, width * 2] } :=
  -- 2× nearest-neighbor upsample
  let x_up := upsample2 x
  let x_up := Tensor.reshapeChecked x_up #[batch, channels, height * 2, width * 2]

  -- 3×3 conv with padding=1 to maintain size
  let out := Tensor.conv2dBias x_up up.conv.weight up.conv.bias #[1, 1] #[1, 1]
  Tensor.reshapeChecked out #[batch, channels, height * 2, width * 2]

end Upsample

/-- Downsample block: 3×3 conv with stride 2 -/
structure Downsample (channels : UInt64) where
  conv : Conv2dParams (device := device) channels channels 3 1
  deriving TensorStruct

namespace Downsample

/-- Initialize downsample block -/
def init (channels : UInt64) (device : Device := .CPU) : IO (Downsample (device := device) channels) := do
  let conv ← Conv2dParams.init (device := device) channels channels 3 1
  pure { conv }

/-- Forward pass: strided conv for 2× downsample.
    Input: [batch, channels, height, width]
    Output: [batch, channels, ceil(height/2), ceil(width/2)] -/
def forward {batch channels height width : UInt64}
    (down : Downsample (device := device) channels)
    (x : Tensor { device := device, shape := #[batch, channels, height, width] })
    : Tensor { device := device, shape := #[batch, channels, Tensor.convExtent height 3 2 1 1, Tensor.convExtent width 3 2 1 1] } :=
  -- Pad input for proper strided conv
  -- Note: In VAE encoder, asymmetric padding may be needed
  let out := Tensor.conv2dBias x down.conv.weight (Tensor.castLike down.conv.weight (Tensor.zeros #[channels] false down.conv.weight.device)) #[2, 2] #[1, 1]
  Tensor.reshapeChecked out #[batch, channels, Tensor.convExtent height 3 2 1 1, Tensor.convExtent width 3 2 1 1]

end Downsample

end torch.vae
