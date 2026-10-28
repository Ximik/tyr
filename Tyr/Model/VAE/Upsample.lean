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

variable {device : Device} {dtype : DType}
/-- Upsampling block: 2× nearest-neighbor upsample + 3×3 conv -/
structure Upsample (channels : Nat) (dtype : DType := .Float32) where
  conv : Conv2dParams (device := device) (dtype := dtype) channels channels 3 1
  deriving TensorStruct


namespace Upsample

/-- Initialize upsample block -/
def init (channels : Nat) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Upsample (device := device) (dtype := dtype) channels) := do
  let conv ← Conv2dParams.init (dtype := dtype) (device := device) channels channels 3 1
  pure { conv }

/-- Forward pass: 2× upsample + conv.
    Input: [batch, channels, height, width]
    Output: [batch, channels, height*2, width*2] -/
def forward {batch channels height width : Nat}
    (up : Upsample (device := device) (dtype := dtype) channels)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, channels, height, width] })
    (hh : 0 < height) (hw : 0 < width)
    : Tensor { dtype := dtype, device := device, shape := #[batch, channels, height * 2, width * 2] } :=
  -- 2× nearest-neighbor upsample
  let x_up := upsample2 x
  let x_up := Tensor.reshape x_up #[batch, channels, height * 2, width * 2]

  -- 3×3 conv with padding=1 to maintain size
  let out := Tensor.conv2dBias x_up up.conv.weight up.conv.bias #[1, 1] #[1, 1]
  Tensor.reshape out #[batch, channels, height * 2, width * 2]
    (by simp [Tensor.convExtent_same (by omega : 0 < height * 2) (by decide : 2 * 1 + 1 = 3),
      Tensor.convExtent_same (by omega : 0 < width * 2) (by decide : 2 * 1 + 1 = 3)])

end Upsample

/-- Downsample block: 3×3 conv with stride 2 -/
structure Downsample (channels : Nat) (dtype : DType := .Float32) where
  conv : Conv2dParams (device := device) (dtype := dtype) channels channels 3 1
  deriving TensorStruct


namespace Downsample

/-- Initialize downsample block -/
def init (channels : Nat) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Downsample (device := device) (dtype := dtype) channels) := do
  let conv ← Conv2dParams.init (dtype := dtype) (device := device) channels channels 3 1
  pure { conv }

/-- Forward pass: strided conv for 2× downsample.
    Input: [batch, channels, height, width]
    Output: [batch, channels, ceil(height/2), ceil(width/2)] -/
def forward {batch channels height width : Nat}
    (down : Downsample (device := device) (dtype := dtype) channels)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, channels, height, width] })
    : Tensor { dtype := dtype, device := device, shape := #[batch, channels, Tensor.convExtent height 3 2 1 1, Tensor.convExtent width 3 2 1 1] } :=
  -- Pad input for proper strided conv
  -- Note: In VAE encoder, asymmetric padding may be needed
  let out := Tensor.conv2dBias x down.conv.weight (Tensor.castLike down.conv.weight (Tensor.zeros #[channels] false down.conv.weight.device)) #[2, 2] #[1, 1]
  Tensor.reshape out #[batch, channels, Tensor.convExtent height 3 2 1 1, Tensor.convExtent width 3 2 1 1] (by simp)

end Downsample

end torch.vae
