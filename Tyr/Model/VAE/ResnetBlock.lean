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

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]
/-- GroupNorm parameters for VAE.
    Weight and bias for affine transformation. -/
structure GroupNormParams (num_groups channels : Nat) (dtype : DType := .Float32) where
  weight : Tensor { dtype := dtype, device := device, shape := #[channels] }
  bias : Tensor { dtype := dtype, device := device, shape := #[channels] }
  deriving TensorStruct


namespace GroupNormParams

/-- Initialize GroupNorm parameters (weight=1, bias=0) -/
def init (num_groups channels : Nat) (device : Device := .CPU) (dtype : DType := .Float32) : IO (GroupNormParams (device := device) (dtype := dtype) num_groups channels) := do
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  IO.ofExcept (Shape.validateAllocation #[channels])
  let weight ← Tensor.parameter (Tensor.onesFor { shape := #[channels], dtype, device })
  let bias ← Tensor.parameter (Tensor.zerosFor { shape := #[channels], dtype, device })
  pure { weight, bias }

/-- Apply GroupNorm to input tensor [batch, channels, height, width] -/
def forward {batch channels height width num_groups : Nat}
    (gn : GroupNormParams (device := device) (dtype := dtype) num_groups channels)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, channels, height, width] })
    (eps : Float := 1e-6)
    : Tensor { dtype := dtype, device := device, shape := #[batch, channels, height, width] } :=
  groupNorm x num_groups gn.weight gn.bias eps

end GroupNormParams

/-- Conv2d parameters for VAE.
    Weight and bias for standard conv2d. -/
structure Conv2dParams (in_ch out_ch kernel_size : Nat) (padding : Nat := 0) (dtype : DType := .Float32) where
  weight : Tensor { dtype := dtype, device := device, shape := #[out_ch, in_ch, kernel_size, kernel_size] }
  bias : Tensor { dtype := dtype, device := device, shape := #[out_ch] }
  deriving TensorStruct


namespace Conv2dParams

/-- Initialize Conv2d with Kaiming init -/
def init (in_ch out_ch kernel_size : Nat) (padding : Nat := 0) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Conv2dParams (device := device) (dtype := dtype) in_ch out_ch kernel_size padding) := do
  let floating ← IO.ofExcept (DType.realFloating? dtype)
  let _ := floating.down
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  let std := Float.sqrt (2.0 / (in_ch * kernel_size * kernel_size).toFloat)
  let w ← Tensor.randn (device := device) #[out_ch, in_ch, kernel_size, kernel_size]
  let w ← IO.ofExcept (w.castDType? dtype)
  let b := Tensor.zerosFor { shape := #[out_ch], dtype, device }
  pure {
    weight := (← Tensor.parameter (Tensor.mulScalar w std))
    bias := (← Tensor.parameter b)
  }

/-- Apply convolution -/
def forward {batch in_ch out_ch kernel_size padding height width : Nat}
    (conv : Conv2dParams (device := device) (dtype := dtype) in_ch out_ch kernel_size padding)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, in_ch, height, width] })
    : Tensor { dtype := dtype, device := device, shape := #[batch, out_ch, Tensor.convExtent height kernel_size 1 padding 1,
        Tensor.convExtent width kernel_size 1 padding 1] } :=
  Tensor.conv2dBias x conv.weight (Tensor.castLike conv.weight (Tensor.zeros #[out_ch] false conv.weight.device)) #[1, 1] #[padding, padding]

end Conv2dParams

/-- The skip path of a ResnetBlock: the input itself when channels match, else a 1×1 conv. -/
inductive Shortcut (in_ch out_ch : Nat) (dtype : DType := .Float32) where
  | identity (h : in_ch = out_ch)
  | conv (c : Conv2dParams (device := device) (dtype := dtype) in_ch out_ch 1 0)

instance {in_ch out_ch : Nat} {dtype : DType} :
    TensorStruct (Shortcut (device := device) in_ch out_ch dtype) where
  map f
    | .identity h => .identity h
    | .conv c => .conv (TensorStruct.map f c)
  mapM f
    | .identity h => pure (.identity h)
    | .conv c => do pure (.conv (← TensorStruct.mapM f c))
  zipWith f lhs rhs := match lhs, rhs with
    | .conv x, .conv y => .conv (TensorStruct.zipWith f x y)
    | lhs, _ => lhs
  fold f init
    | .identity _ => init
    | .conv c => TensorStruct.fold f init c

instance {P : StaticSpec → Prop} {in_ch out_ch : Nat} {dtype : DType}
    [TensorStructFor P (Conv2dParams (device := device) (dtype := dtype) in_ch out_ch 1 0)] :
    TensorStructFor P (Shortcut (device := device) in_ch out_ch dtype) where
  map f
    | .identity h => .identity h
    | .conv c => .conv (TensorStructFor.map f c)
  mapM f
    | .identity h => pure (.identity h)
    | .conv c => do pure (.conv (← TensorStructFor.mapM f c))
  zipWith f lhs rhs := match lhs, rhs with
    | .conv x, .conv y => .conv (TensorStructFor.zipWith f x y)
    | lhs, _ => lhs
  fold f init
    | .identity _ => init
    | .conv c => TensorStructFor.fold f init c

instance {in_ch out_ch : Nat} {d₁ d₂ : DType} {device₁ device₂ : Device}
    [TensorTreeCast (Conv2dParams (device := device₁) (dtype := d₁) in_ch out_ch 1 0)
      (Conv2dParams (device := device₂) (dtype := d₂) in_ch out_ch 1 0)] :
    TensorTreeCast (Shortcut (device := device₁) in_ch out_ch d₁) (Shortcut (device := device₂) in_ch out_ch d₂) where
  castAt path
    | .identity h => pure (.identity h)
    | .conv c => do pure (.conv (← TensorTreeCast.castAt path c))

/-- Swish activation: x * sigmoid(x) -/
def swish {s : Shape} (x : Tensor { dtype := dtype, device := device, shape := s }) : Tensor { dtype := dtype, device := device, shape := s } :=
  Tensor.mul x (Tensor.sigmoid x)

/-- Residual block for VAE decoder.
    norm1 → swish → conv1 → norm2 → swish → conv2 (+ skip connection) -/
structure ResnetBlock (in_ch out_ch : Nat) (dtype : DType := .Float32) where
  /-- First normalization -/
  norm1 : GroupNormParams (device := device) (dtype := dtype) 32 in_ch
  /-- First convolution 3×3 -/
  conv1 : Conv2dParams (device := device) (dtype := dtype) in_ch out_ch 3 1
  /-- Second normalization -/
  norm2 : GroupNormParams (device := device) (dtype := dtype) 32 out_ch
  /-- Second convolution 3×3 -/
  conv2 : Conv2dParams (device := device) (dtype := dtype) out_ch out_ch 3 1
  /-- Skip connection (1×1 conv if in_ch ≠ out_ch) -/
  nin_shortcut : Shortcut (device := device) in_ch out_ch dtype
  deriving TensorStruct


namespace ResnetBlock

/-- Initialize ResnetBlock -/
def init (in_ch out_ch : Nat) (device : Device := .CPU) (dtype : DType := .Float32) : IO (ResnetBlock (device := device) (dtype := dtype) in_ch out_ch) := do
  let norm1 ← GroupNormParams.init (dtype := dtype) (device := device) 32 in_ch
  let conv1 ← Conv2dParams.init (dtype := dtype) (device := device) in_ch out_ch 3 1
  let norm2 ← GroupNormParams.init (dtype := dtype) (device := device) 32 out_ch
  let conv2 ← Conv2dParams.init (dtype := dtype) (device := device) out_ch out_ch 3 1
  if h : in_ch = out_ch then
    pure { norm1, conv1, norm2, conv2, nin_shortcut := .identity h }
  else
    let c ← Conv2dParams.init (dtype := dtype) (device := device) in_ch out_ch 1 0
    pure { norm1, conv1, norm2, conv2, nin_shortcut := .conv c }

/-- Forward pass for ResnetBlock.
    Input: [batch, in_ch, height, width]
    Output: [batch, out_ch, height, width] -/
def forward {batch in_ch out_ch height width : Nat}
    (block : ResnetBlock (device := device) (dtype := dtype) in_ch out_ch)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, in_ch, height, width] })
    (hh : 0 < height) (hw : 0 < width)
    : Tensor { dtype := dtype, device := device, shape := #[batch, out_ch, height, width] } :=
  -- Main path
  let h := block.norm1.forward x
  let h := swish h
  let h := Tensor.conv2dBias h block.conv1.weight block.conv1.bias #[1, 1] #[1, 1]
  let h := Tensor.reshape h #[batch, out_ch, height, width] (by simp [Tensor.convExtent_same hh (by decide : 2 * 1 + 1 = 3), Tensor.convExtent_same hw (by decide : 2 * 1 + 1 = 3)])

  let h := block.norm2.forward h
  let h := swish h
  let h := Tensor.conv2dBias h block.conv2.weight block.conv2.bias #[1, 1] #[1, 1]
  let h := Tensor.reshape h #[batch, out_ch, height, width] (by simp [Tensor.convExtent_same hh (by decide : 2 * 1 + 1 = 3), Tensor.convExtent_same hw (by decide : 2 * 1 + 1 = 3)])

  -- Skip connection
  let skip := match block.nin_shortcut with
    | .conv conv => Tensor.reshape (Tensor.conv2dBias x conv.weight conv.bias #[1, 1] #[0, 0]) #[batch, out_ch, height, width]
        (by simp [Tensor.convExtent_same hh (by decide : 2 * 0 + 1 = 1), Tensor.convExtent_same hw (by decide : 2 * 0 + 1 = 1)])
    | .identity h => Tensor.reshape x #[batch, out_ch, height, width] (by simp [h])

  Tensor.add h skip

end ResnetBlock

end torch.vae
