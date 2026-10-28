/-
  Tyr/Model/VAE/AttnBlock.lean

  Self-attention block for VAE decoder.
  Single-head spatial attention.
-/
import Tyr.Model.VAE.TypedOps
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.VAE.ResnetBlock

namespace torch.vae

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]
/-- Self-attention block for VAE.
    Applies spatial attention using 1×1 convolutions. -/
structure AttnBlock (channels : Nat) (dtype : DType := .Float32) where
  /-- GroupNorm before attention -/
  norm : GroupNormParams (device := device) (dtype := dtype) 32 channels
  /-- Query projection (1×1 conv) -/
  q : Conv2dParams (device := device) (dtype := dtype) channels channels 1 0
  /-- Key projection (1×1 conv) -/
  k : Conv2dParams (device := device) (dtype := dtype) channels channels 1 0
  /-- Value projection (1×1 conv) -/
  v : Conv2dParams (device := device) (dtype := dtype) channels channels 1 0
  /-- Output projection (1×1 conv) -/
  proj_out : Conv2dParams (device := device) (dtype := dtype) channels channels 1 0
  deriving TensorStruct


namespace AttnBlock

/-- Initialize attention block -/
def init (channels : Nat) (device : Device := .CPU) (dtype : DType := .Float32) : IO (AttnBlock (device := device) (dtype := dtype) channels) := do
  let norm ← GroupNormParams.init (dtype := dtype) (device := device) 32 channels
  let q ← Conv2dParams.init (dtype := dtype) (device := device) channels channels 1 0
  let k ← Conv2dParams.init (dtype := dtype) (device := device) channels channels 1 0
  let v ← Conv2dParams.init (dtype := dtype) (device := device) channels channels 1 0
  let proj_out ← Conv2dParams.init (dtype := dtype) (device := device) channels channels 1 0
  pure { norm, q, k, v, proj_out }

/-- Forward pass for attention block.
    Input: [batch, channels, height, width]
    Output: [batch, channels, height, width]

    Applies single-head spatial self-attention. -/
def forward {batch channels height width : Nat}
    (attn : AttnBlock (device := device) (dtype := dtype) channels)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, channels, height, width] })
    (hh : 0 < height) (hw : 0 < width)
    : Tensor { dtype := dtype, device := device, shape := #[batch, channels, height, width] } :=
  let h := attn.norm.forward x

  -- Project to Q, K, V
  let q := Tensor.conv2dBias h attn.q.weight attn.q.bias #[1, 1] #[0, 0]
  let k := Tensor.conv2dBias h attn.k.weight attn.k.bias #[1, 1] #[0, 0]
  let v := Tensor.conv2dBias h attn.v.weight attn.v.bias #[1, 1] #[0, 0]

  -- Reshape to [batch, channels, height*width]
  let spatial := height * width
  let q := Tensor.reshape q #[batch, channels, spatial] (by simp +zetaDelta [Tensor.convExtent_same hh (by decide : 2 * 0 + 1 = 1), Tensor.convExtent_same hw (by decide : 2 * 0 + 1 = 1), Nat.mul_assoc])
  let k := Tensor.reshape k #[batch, channels, spatial] (by simp +zetaDelta [Tensor.convExtent_same hh (by decide : 2 * 0 + 1 = 1), Tensor.convExtent_same hw (by decide : 2 * 0 + 1 = 1), Nat.mul_assoc])
  let v := Tensor.reshape v #[batch, channels, spatial] (by simp +zetaDelta [Tensor.convExtent_same hh (by decide : 2 * 0 + 1 = 1), Tensor.convExtent_same hw (by decide : 2 * 0 + 1 = 1), Nat.mul_assoc])

  -- Compute attention: softmax(Qᵀ K / sqrt(channels)) V
  -- Q: [batch, channels, spatial], K: [batch, channels, spatial]
  -- Qᵀ K: [batch, spatial, spatial]
  let q_t := Tensor.transpose q 1 2  -- [batch, spatial, channels]
  let scale := Float.sqrt channels.toFloat
  let attn_weights := bmm q_t k  -- [batch, spatial, spatial]
  let attn_weights := Tensor.divScalar attn_weights scale
  let attn_weights := Tensor.softmax attn_weights (-1)

  -- V: [batch, channels, spatial]
  -- attn_weights @ Vᵀ = [batch, spatial, spatial] @ [batch, spatial, channels]
  let v_t := Tensor.transpose v 1 2  -- [batch, spatial, channels]
  let out := bmm attn_weights v_t  -- [batch, spatial, channels]
  let out := Tensor.transpose out 1 2  -- [batch, channels, spatial]

  -- Reshape back to spatial
  let out := Tensor.reshape out #[batch, channels, height, width] (by simp +zetaDelta [Nat.mul_assoc])

  -- Output projection + residual
  let out := Tensor.conv2dBias out attn.proj_out.weight attn.proj_out.bias #[1, 1] #[0, 0]
  let out := Tensor.reshape out #[batch, channels, height, width] (by simp +zetaDelta [Tensor.convExtent_same hh (by decide : 2 * 0 + 1 = 1), Tensor.convExtent_same hw (by decide : 2 * 0 + 1 = 1), Nat.mul_assoc])

  Tensor.add x out

end AttnBlock

end torch.vae
