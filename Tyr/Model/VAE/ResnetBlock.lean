/-
  Tyr/Model/VAE/ResnetBlock.lean

  Residual block for VAE decoder.
  Uses GroupNorm and Swish activation.
-/
import Tyr.Model.VAE.TypedOps
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive

namespace torch.vae

variable {device : Device}
/-- GroupNorm parameters for VAE.
    Weight and bias for affine transformation. -/
structure GroupNormParams (num_groups channels : UInt64) where
  weight : Tensor { device := device, shape := #[channels] }
  bias : Tensor { device := device, shape := #[channels] }
  deriving TensorStruct

namespace GroupNormParams

/-- Initialize GroupNorm parameters (weight=1, bias=0) -/
def init (num_groups channels : UInt64) (device : Device := .CPU) : IO (GroupNormParams (device := device) num_groups channels) := do
  let weight ← Tensor.parameter ((Tensor.ones (device := device) #[channels]).forgetDType)
  let bias ← Tensor.parameter ((Tensor.zeros (device := device) #[channels]).forgetDType)
  pure { weight, bias }

/-- Apply GroupNorm to input tensor [batch, channels, height, width] -/
def forward {batch channels height width num_groups : UInt64}
    (gn : GroupNormParams (device := device) num_groups channels)
    (x : Tensor { device := device, shape := #[batch, channels, height, width] })
    (eps : Float := 1e-6)
    : Tensor { device := device, shape := #[batch, channels, height, width] } :=
  groupNorm x num_groups gn.weight gn.bias eps

end GroupNormParams

/-- Conv2d parameters for VAE.
    Weight and bias for standard conv2d. -/
structure Conv2dParams (in_ch out_ch kernel_size : UInt64) (padding : UInt64 := 0) where
  weight : Tensor { device := device, shape := #[out_ch, in_ch, kernel_size, kernel_size] }
  bias : Tensor { device := device, shape := #[out_ch] }
  deriving TensorStruct

namespace Conv2dParams

/-- Initialize Conv2d with Kaiming init -/
def init (in_ch out_ch kernel_size : UInt64) (padding : UInt64 := 0) (device : Device := .CPU) : IO (Conv2dParams (device := device) in_ch out_ch kernel_size padding) := do
  let std := Float.sqrt (2.0 / (in_ch * kernel_size * kernel_size).toFloat)
  let w ← Tensor.randn (device := device) #[out_ch, in_ch, kernel_size, kernel_size]
  let w := w.forgetDType
  let b := (Tensor.zeros (device := device) #[out_ch]).forgetDType
  pure {
    weight := (← Tensor.parameter (Tensor.Preserving.mulScalar w std))
    bias := (← Tensor.parameter b)
  }

/-- Apply convolution -/
def forward {batch in_ch out_ch kernel_size padding height width : UInt64}
    (conv : Conv2dParams (device := device) in_ch out_ch kernel_size padding)
    (x : Tensor { device := device, shape := #[batch, in_ch, height, width] })
    : Tensor { device := device, shape := #[batch, out_ch, Tensor.convExtent height kernel_size 1 padding 1,
        Tensor.convExtent width kernel_size 1 padding 1] } :=
  Tensor.conv2dBias x conv.weight (Tensor.castLike conv.weight (Tensor.zeros #[out_ch] false conv.weight.device)) #[1, 1] #[padding, padding]

end Conv2dParams

/-- Swish activation: x * sigmoid(x) -/
def swish {s : Shape} (x : Tensor { device := device, shape := s }) : Tensor { device := device, shape := s } :=
  Tensor.Promoting.mul x (Tensor.Preserving.sigmoid x)

/-- Residual block for VAE decoder.
    norm1 → swish → conv1 → norm2 → swish → conv2 (+ skip connection) -/
structure ResnetBlock (in_ch out_ch : UInt64) where
  /-- First normalization -/
  norm1 : GroupNormParams (device := device) 32 in_ch
  /-- First convolution 3×3 -/
  conv1 : Conv2dParams (device := device) in_ch out_ch 3 1
  /-- Second normalization -/
  norm2 : GroupNormParams (device := device) 32 out_ch
  /-- Second convolution 3×3 -/
  conv2 : Conv2dParams (device := device) out_ch out_ch 3 1
  /-- Skip connection (1×1 conv if in_ch ≠ out_ch) -/
  nin_shortcut : Option (Conv2dParams (device := device) in_ch out_ch 1 0)
  deriving TensorStruct

namespace ResnetBlock

/-- Initialize ResnetBlock -/
def init (in_ch out_ch : UInt64) (device : Device := .CPU) : IO (ResnetBlock (device := device) in_ch out_ch) := do
  let norm1 ← GroupNormParams.init (device := device) 32 in_ch
  let conv1 ← Conv2dParams.init (device := device) in_ch out_ch 3 1
  let norm2 ← GroupNormParams.init (device := device) 32 out_ch
  let conv2 ← Conv2dParams.init (device := device) out_ch out_ch 3 1
  let nin_shortcut ← if in_ch != out_ch then do
    let c ← Conv2dParams.init (device := device) in_ch out_ch 1 0
    pure (some c)
  else
    pure none
  pure { norm1, conv1, norm2, conv2, nin_shortcut }

/-- Forward pass for ResnetBlock.
    Input: [batch, in_ch, height, width]
    Output: [batch, out_ch, height, width] -/
def forward {batch in_ch out_ch height width : UInt64}
    (block : ResnetBlock (device := device) in_ch out_ch)
    (x : Tensor { device := device, shape := #[batch, in_ch, height, width] })
    : Tensor { device := device, shape := #[batch, out_ch, height, width] } :=
  -- Main path
  let h := block.norm1.forward x
  let h := swish h
  let h := Tensor.conv2dBias h block.conv1.weight block.conv1.bias #[1, 1] #[1, 1]
  let h := Tensor.reshapeChecked h #[batch, out_ch, height, width]

  let h := block.norm2.forward h
  let h := swish h
  let h := Tensor.conv2dBias h block.conv2.weight block.conv2.bias #[1, 1] #[1, 1]
  let h := Tensor.reshapeChecked h #[batch, out_ch, height, width]

  -- Skip connection
  let skip := match block.nin_shortcut with
    | some conv => Tensor.reshapeChecked (Tensor.conv2dBias x conv.weight conv.bias #[1, 1] #[0, 0]) #[batch, out_ch, height, width]
    | none => Tensor.reshapeChecked x #[batch, out_ch, height, width]

  Tensor.Promoting.add h skip

end ResnetBlock

end torch.vae
