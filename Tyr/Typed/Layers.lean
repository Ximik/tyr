/-
  Tyr/Typed/Layers.lean

  Fully typed model layers — the migration template for writing model code
  against the graded `Tensor (σ : StaticSpec)` instead of raw handles. Mirrors
  `Tyr/Module/Linear.lean` and `Tyr/Module/RMSNorm.lean`, with shape, dtype,
  AND device tracked end to end: weights and activations must agree
  on dtype and device or the layer does not compile.

  Layers are generic over a floating dtype and a concrete device (CPU by
  default). The `isFloating`
  hypothesis is discharged by `decide` at concrete call sites and threaded
  through in generic code.
-/
import Tyr.Typed.Ops

namespace torch.Typed

/-- Linear layer: `y = x Wᵀ (+ b)`. Weight is `[outDim, inDim]` following
    PyTorch convention; dtype and device are shared by weight, bias,
    input, and output. -/
structure Linear (inDim outDim : UInt64) (d : DType)
    (dev : Device := .CPU) where
  weight : Tensor ⟨#[outDim, inDim], d, dev⟩
  bias : Option (Tensor ⟨#[outDim], d, dev⟩) := none

namespace Linear

/-- Kaiming-style random initialization (Float32, like the runtime default). -/
def init (inDim outDim : UInt64) (withBias : Bool := true) (device : Device := .CPU) :
    IO (Linear inDim outDim .Float32 device) := do
  let std := Float.sqrt (2.0 / inDim.toFloat)
  let w ← Tensor.randn #[outDim, inDim] (device := device)
  let scaled ← IO.ofExcept (Tensor.mulScalar? w std)
  let weight ← scaled.parameter
  let bias ←
    if withBias then
      pure (some (← (Tensor.zeros #[outDim] (device := device)).parameter))
    else
      pure none
  pure { weight, bias }

/-- Forward for 2D input `[batch, inDim] → [batch, outDim]`. -/
def forward2d {inDim outDim batch : UInt64} {d : DType} {dev : Device}
    [DType.Promotes d d d]
    (lin : Linear inDim outDim d dev) (x : Tensor ⟨#[batch, inDim], d, dev⟩) :
    Tensor ⟨#[batch, outDim], d, dev⟩ :=
  let y := Tensor.linear x lin.weight
  match lin.bias with
  | some b => Tensor.add y (b.expand #[batch, outDim])
  | none => y

/-- Forward for 3D input `[batch, seq, inDim] → [batch, seq, outDim]`. -/
def forward3d {inDim outDim batch seq : UInt64} {d : DType} {dev : Device}
    (lin : Linear inDim outDim d dev)
    (x : Tensor ⟨#[batch, seq, inDim], d, dev⟩) :
    Tensor ⟨#[batch, seq, outDim], d, dev⟩ :=
  match lin.bias with
  | some b => Tensor.affine3d x lin.weight b
  | none => Tensor.linear3d x lin.weight

end Linear

/-- RMS layer normalization: `y = x / sqrt(mean(x²) + eps) * weight`. -/
structure RMSNorm (dim : UInt64) (d : DType)
    (dev : Device := .CPU) where
  weight : Tensor ⟨#[dim], d, dev⟩
  eps : Float := 1e-6

namespace RMSNorm

/-- Initialize with unit weights (Float32). -/
def init (dim : UInt64) (eps : Float := 1e-6) (device : Device := .CPU) :
    IO (RMSNorm dim .Float32 device) := do
  IO.ofExcept (Shape.validateAllocation #[dim])
  pure { weight := (← (Tensor.ones #[dim] (device := device)).parameter), eps }

/-! The runtime normalizes in Float32 for stability, so forward results are
`promote Float32 d` (Float32 for f16/bf16 layers) — the dtype change is
visible in the type rather than hidden. Cast back explicitly for HF-style
dtype restoration. -/

/-- Forward for 2D input `[seq, dim]`. -/
def forward2d {dim seq : UInt64} {d result : DType} {dev : Device}
    [DType.Promotes .Float32 d result]
    (rn : RMSNorm dim d dev) (x : Tensor ⟨#[seq, dim], d, dev⟩)
    (float : d.isFloating = true := by decide) :
    Tensor ⟨#[seq, dim], DType.promote .Float32 d, dev⟩ :=
  letI : TensorSpec.Promotes ⟨#[seq, dim], .Float32⟩ ⟨#[dim], d⟩ result :=
    ⟨DType.Promotes.valid⟩
  Tensor.rmsNormWeighted x rn.weight rn.eps float

/-- Forward for 3D input `[batch, seq, dim]`. -/
def forward3d {dim batch seq : UInt64} {d result : DType} {dev : Device}
    [DType.Promotes .Float32 d result]
    (rn : RMSNorm dim d dev) (x : Tensor ⟨#[batch, seq, dim], d, dev⟩)
    (float : d.isFloating = true := by decide) :
    Tensor ⟨#[batch, seq, dim], DType.promote .Float32 d, dev⟩ :=
  letI : TensorSpec.Promotes ⟨#[batch, seq, dim], .Float32⟩ ⟨#[dim], d⟩ result :=
    ⟨DType.Promotes.valid⟩
  Tensor.rmsNormWeighted x rn.weight rn.eps float

/-- Forward for 4D input `[batch, seq, nHead, headDim]`, normalizing the
    last dimension. -/
def forward4d {batch seq nHead headDim : UInt64} {d result : DType} {dev : Device}
    [DType.Promotes .Float32 d result]
    (rn : RMSNorm headDim d dev)
    (x : Tensor ⟨#[batch, seq, nHead, headDim], d, dev⟩)
    (float : d.isFloating = true := by decide) :
    Tensor ⟨#[batch, seq, nHead, headDim], DType.promote .Float32 d, dev⟩ :=
  letI : TensorSpec.Promotes ⟨#[batch, seq, nHead, headDim], .Float32⟩ ⟨#[headDim], d⟩ result :=
    ⟨DType.Promotes.valid⟩
  Tensor.rmsNormWeighted x rn.weight rn.eps float

end RMSNorm

end torch.Typed
