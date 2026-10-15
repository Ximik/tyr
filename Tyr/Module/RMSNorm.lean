/-
  Tyr/Module/RMSNorm.lean

  RMS Layer Normalization module with TensorStruct and Module instances.
  Used in LLaMA, Qwen, and other modern transformer architectures.
-/
import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Widget

namespace torch

/-- RMS Layer Normalization: normalizes using root mean square.
    y = x / sqrt(mean(x^2) + eps) * weight
    Unlike LayerNorm, RMSNorm has no bias and doesn't subtract the mean.

    Normalize in Float32, restore the activation dtype, then apply the learned
    weight. This matches checkpoint inference rounding for BF16 models. -/
structure RMSNorm (dim : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  weight : Tensor ⟨#[dim], dtype, device⟩
  eps : Static Float := ⟨1e-6⟩
  deriving TensorStruct, ToModuleDisplay

namespace RMSNorm

/-- Initialize RMSNorm with ones for weight -/
def init (dim : UInt64) (eps : Float := 1e-6) (dtype : DType := .Float32)
    (device : Device := .CPU) : IO (RMSNorm dim dtype device) := do
  let weight ← Tensor.parameter ((Tensor.ones #[dim] (device := device)).castDType dtype)
  pure { weight, eps := ⟨eps⟩ }

/-- Forward pass for 2D input [seq, dim] -/
def forward2d {dim seq : UInt64} {d : DType} {dev : Device} (rn : RMSNorm dim d dev)
    (x : Tensor ⟨#[seq, dim], d, dev⟩)
    :
    Tensor ⟨#[seq, dim], d, dev⟩ :=
  Tensor.rmsNormPreserving x rn.weight rn.eps.val

/-- Forward pass for 3D input [batch, seq, dim] -/
def forward3d {dim batch seq : UInt64} {d : DType} {dev : Device} (rn : RMSNorm dim d dev)
    (x : Tensor ⟨#[batch, seq, dim], d, dev⟩)
    :
    Tensor ⟨#[batch, seq, dim], d, dev⟩ :=
  Tensor.rmsNormPreserving x rn.weight rn.eps.val

/-- Forward pass for 4D input [batch, seq, n_head, head_dim]
    Normalizes over the last dimension (head_dim). -/
def forward4d {batch seq n_head head_dim : UInt64} {d : DType} {dev : Device}
    (rn : RMSNorm head_dim d dev)
    (x : Tensor ⟨#[batch, seq, n_head, head_dim], d, dev⟩)
    :
    Tensor ⟨#[batch, seq, n_head, head_dim], d, dev⟩ :=
  Tensor.rmsNormPreserving x rn.weight rn.eps.val

/-- Forward pass for 4D input in attention format [batch, n_head, seq, head_dim]
    Normalizes over the last dimension (head_dim). -/
def forward5d {batch n_head seq head_dim : UInt64} {d : DType} {dev : Device}
    (rn : RMSNorm head_dim d dev)
    (x : Tensor ⟨#[batch, n_head, seq, head_dim], d, dev⟩)
    :
    Tensor ⟨#[batch, n_head, seq, head_dim], d, dev⟩ :=
  Tensor.rmsNormPreserving x rn.weight rn.eps.val

/-- HF-style normalization: accumulate in Float32, return the input dtype. -/
def forward3dPreserving {dim batch seq : UInt64} {d : DType} {dev : Device}
    (rn : RMSNorm dim d dev) (x : Tensor ⟨#[batch, seq, dim], d, dev⟩) :
    Tensor ⟨#[batch, seq, dim], d, dev⟩ :=
  Tensor.rmsNormPreserving x rn.weight rn.eps.val

end RMSNorm

/-- Module instance for 2D forward pass (Float32 layers; other dtypes can call
    the forward functions directly with an explicit `isFloating` proof). -/
instance {dim seq : UInt64} {dev : Device} :
    Module (RMSNorm dim .Float32 dev) (Tensor ⟨#[seq, dim], .Float32, dev⟩)
      (Tensor ⟨#[seq, dim], .Float32, dev⟩) where
  forward rn x := RMSNorm.forward2d rn x

/-- Module instance for 3D forward pass (Float32 layers). -/
instance {dim batch seq : UInt64} {dev : Device} :
    Module (RMSNorm dim .Float32 dev) (Tensor ⟨#[batch, seq, dim], .Float32, dev⟩)
      (Tensor ⟨#[batch, seq, dim], .Float32, dev⟩) where
  forward rn x := RMSNorm.forward3d rn x

end torch
