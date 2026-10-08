/-
  Tyr/Typed/Ops.lean

  Typed operation surface over the untyped `torch.*` FFI, indexed by
  `StaticSpec` (shape, dtype, device policy).

  Every wrapper is a zero-cost phantom re-indexing of an existing extern:
  no FFI signature changes, no extra runtime parameters. Result specs follow
  PyTorch's semantics as encoded in the `StaticSpec`/`DType` algebra; the
  claims are validated against the live runtime by the parity tests in
  `Tests/TestTyped.lean`.

  Conventions:
  - Ops that preserve the whole spec are `Tensor σ → Tensor σ`.
  - Ops that need spec components use anonymous-constructor binders
    (`Tensor ⟨s, d, dev⟩`); the device policy variable flows through
    unchanged, so device-pinned tensors stay pinned.
  - Elementwise binary ops promote dtypes; true division and float-scalar
    arithmetic float integer results (`atLeastFloat`).
  - Float-only math requires an `isFloating` proof, discharged by `decide`
    for literal dtypes; integer tensors must be cast explicitly first.
  - Matmul-family ops require both operands to share dtype AND device
    policy — PyTorch throws on mixed operands; here it cannot compile.
    Generic `matmul` also requires `TensorSpec.matmulShape?` to accept
    the shape pair (auto-param, `rfl` for concrete shapes), so inner-dim
    mismatches fail to compile instead of throwing inside libtorch.
-/
import Tyr.Torch
import Tyr.Typed.Tensor

namespace torch

namespace Tensor

/-! ## Creation

The runtime defaults are Float32 for float factories and Int64 for
`full_int`; the indices state exactly that. The device policy stays `.any`:
the runtime `device` argument places the tensor, the policy only constrains
which placements the type admits. -/

def zeros (shape : Shape) (requiresGrad : Bool := false)
    (device : Device := Device.CPU) : Tensor { shape, dtype := .Float32 } :=
  .assumeSpec (torch.zeros shape requiresGrad device)

def ones (shape : Shape) (requiresGrad : Bool := false)
    (device : Device := Device.CPU) : Tensor { shape, dtype := .Float32 } :=
  .assumeSpec (torch.ones shape requiresGrad device)

def full (shape : Shape) (value : Float) (requiresGrad : Bool := false)
    (device : Device := Device.CPU) : Tensor { shape, dtype := .Float32 } :=
  .assumeSpec (torch.full shape value requiresGrad device)

def fullInt (shape : Shape) (value : Int64) : Tensor { shape, dtype := .Int64 } :=
  .assumeSpec (torch.full_int shape value)

def zerosLike {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  .assumeSpec (torch.zeros_like t.raw)

def onesLike {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  .assumeSpec (torch.ones_like t.raw)

def randn (shape : Shape) (requiresGrad : Bool := false)
    (device : Device := Device.CPU) : IO (Tensor { shape, dtype := .Float32 }) := do
  pure (.assumeSpec (← torch.randn shape requiresGrad device))

/-! ## Casts -/

def toFloat32 {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withDType .Float32) :=
  .assumeSpec (torch.toFloat' t.raw)

def toBFloat16 {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withDType .BFloat16) :=
  .assumeSpec (torch.toBFloat16' t.raw)

def toInt64 {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withDType .Int64) :=
  .assumeSpec (data.toLong t.raw)

/-! ## Elementwise binary (same shape and device, dtypes promote) -/

def add {s : Shape} {l r : DType} {dev : DevicePolicy}
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, DType.promote l r, dev⟩ :=
  .assumeSpec (torch.add a.raw b.raw)

def sub {s : Shape} {l r : DType} {dev : DevicePolicy}
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, DType.promote l r, dev⟩ :=
  .assumeSpec (torch.sub a.raw b.raw)

def mul {s : Shape} {l r : DType} {dev : DevicePolicy}
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, DType.promote l r, dev⟩ :=
  .assumeSpec (torch.mul a.raw b.raw)

def div {s : Shape} {l r : DType} {dev : DevicePolicy}
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, (DType.promote l r).atLeastFloat, dev⟩ :=
  .assumeSpec (nn.div a.raw b.raw)

instance {s : Shape} {l r : DType} {dev : DevicePolicy} :
    HAdd (Tensor ⟨s, l, dev⟩) (Tensor ⟨s, r, dev⟩) (Tensor ⟨s, DType.promote l r, dev⟩) :=
  ⟨add⟩

instance {s : Shape} {l r : DType} {dev : DevicePolicy} :
    HSub (Tensor ⟨s, l, dev⟩) (Tensor ⟨s, r, dev⟩) (Tensor ⟨s, DType.promote l r, dev⟩) :=
  ⟨sub⟩

instance {s : Shape} {l r : DType} {dev : DevicePolicy} :
    HMul (Tensor ⟨s, l, dev⟩) (Tensor ⟨s, r, dev⟩) (Tensor ⟨s, DType.promote l r, dev⟩) :=
  ⟨mul⟩

instance {s : Shape} {l r : DType} {dev : DevicePolicy} :
    HDiv (Tensor ⟨s, l, dev⟩) (Tensor ⟨s, r, dev⟩)
      (Tensor ⟨s, (DType.promote l r).atLeastFloat, dev⟩) :=
  ⟨div⟩

/-! Homogeneous instances (same full spec, dtype-preserving).
These matter for code that is *generic* over the spec: there
`DType.promote d d` is stuck (the equation `promote d d = d` is a theorem,
not a reduction), so the promoting instances would produce unusable result
types. `Add.add` is also the deterministic spelling for layer code.
No `Div` — true division changes integer dtypes. -/

instance {σ : StaticSpec} : Add (Tensor σ) :=
  ⟨fun a b => .assumeSpec (torch.add a.raw b.raw)⟩

instance {σ : StaticSpec} : Sub (Tensor σ) :=
  ⟨fun a b => .assumeSpec (torch.sub a.raw b.raw)⟩

instance {σ : StaticSpec} : Mul (Tensor σ) :=
  ⟨fun a b => .assumeSpec (torch.mul a.raw b.raw)⟩

/-! ## Scalar arithmetic (Float scalars are weak but still float a result) -/

def addScalar {σ : StaticSpec} (t : Tensor σ) (value : Float) :
    Tensor (σ.withDType σ.dtype.atLeastFloat) :=
  .assumeSpec (torch.add_scalar t.raw value)

def subScalar {σ : StaticSpec} (t : Tensor σ) (value : Float) :
    Tensor (σ.withDType σ.dtype.atLeastFloat) :=
  .assumeSpec (torch.sub_scalar t.raw value)

def mulScalar {σ : StaticSpec} (t : Tensor σ) (value : Float) :
    Tensor (σ.withDType σ.dtype.atLeastFloat) :=
  .assumeSpec (torch.mul_scalar t.raw value)

def divScalar {σ : StaticSpec} (t : Tensor σ) (value : Float) :
    Tensor (σ.withDType σ.dtype.atLeastFloat) :=
  .assumeSpec (torch.div_scalar t.raw value)

instance {σ : StaticSpec} :
    HAdd (Tensor σ) Float (Tensor (σ.withDType σ.dtype.atLeastFloat)) :=
  ⟨addScalar⟩

instance {σ : StaticSpec} :
    HSub (Tensor σ) Float (Tensor (σ.withDType σ.dtype.atLeastFloat)) :=
  ⟨subScalar⟩

instance {σ : StaticSpec} :
    HMul (Tensor σ) Float (Tensor (σ.withDType σ.dtype.atLeastFloat)) :=
  ⟨mulScalar⟩

instance {σ : StaticSpec} :
    HDiv (Tensor σ) Float (Tensor (σ.withDType σ.dtype.atLeastFloat)) :=
  ⟨divScalar⟩

/-! ## Elementwise unary, spec-preserving for all dtypes -/

def relu {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  .assumeSpec (torch.relu t.raw)

def abs {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  .assumeSpec (nn.abs t.raw)

/-! ## Float-only unary math

Requires `σ.dtype.isFloating`; the proof is found by `decide` when the
dtype is a literal. Integer tensors must cast explicitly (`.toFloat32`). -/

def exp {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (nn.exp t.raw)

def sqrt {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (nn.sqrt t.raw)

def rsqrt {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (torch.rsqrt t.raw)

def sigmoid {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (nn.sigmoid t.raw)

def tanh {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (nn.tanh t.raw)

def gelu {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (nn.gelu t.raw)

def silu {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (nn.silu t.raw)

def pow {σ : StaticSpec} (t : Tensor σ) (exponent : Float)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (nn.pow t.raw exponent)

def softmax {σ : StaticSpec} (t : Tensor σ) (dim : Int32 := -1)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (nn.softmax t.raw dim)

/-! ## Matmul family (operands must share dtype and device policy) -/

/-- Generic matmul (PyTorch semantics: 1D/2D special cases, batch dims
    broadcast). The auto-param proof — discharged by `rfl` for concrete
    shapes — guarantees the inner dims match and the batch dims broadcast
    (`TensorSpec.matmulShape?`), so mismatched operands fail to compile
    instead of throwing inside libtorch. Code generic over dimensions,
    where the proof cannot reduce, should use the checked
    `mm`/`linear`/`linear3d`/`matmul3d`/`bmm4d` family instead. -/
def matmul {s1 s2 : Shape} {d : DType} {dev : DevicePolicy}
    (a : Tensor ⟨s1, d, dev⟩) (b : Tensor ⟨s2, d, dev⟩)
    (_h : (TensorSpec.matmulShape? s1 s2).isSome = true := by rfl) :
    Tensor ⟨matmulShape s1 s2, d, dev⟩ :=
  .assumeSpec (nn.matmul a.raw b.raw)

def mm {m k n : UInt64} {d : DType} {dev : DevicePolicy}
    (a : Tensor ⟨#[m, k], d, dev⟩) (b : Tensor ⟨#[k, n], d, dev⟩) :
    Tensor ⟨#[m, n], d, dev⟩ :=
  .assumeSpec (nn.mm a.raw b.raw)

def linear {m n b : UInt64} {d : DType} {dev : DevicePolicy}
    (x : Tensor ⟨#[b, m], d, dev⟩) (weight : Tensor ⟨#[n, m], d, dev⟩) :
    Tensor ⟨#[b, n], d, dev⟩ :=
  .assumeSpec (torch.linear x.raw weight.raw)

def linear3d {batch seq inDim outDim : UInt64} {d : DType} {dev : DevicePolicy}
    (x : Tensor ⟨#[batch, seq, inDim], d, dev⟩)
    (weight : Tensor ⟨#[outDim, inDim], d, dev⟩) :
    Tensor ⟨#[batch, seq, outDim], d, dev⟩ :=
  .assumeSpec (torch.linear3d x.raw weight.raw)

def matmul3d {batch seq k n : UInt64} {d : DType} {dev : DevicePolicy}
    (a : Tensor ⟨#[batch, seq, k], d, dev⟩) (b : Tensor ⟨#[k, n], d, dev⟩) :
    Tensor ⟨#[batch, seq, n], d, dev⟩ :=
  .assumeSpec (nn.matmul3d a.raw b.raw)

def bmm4d {b h m k n : UInt64} {d : DType} {dev : DevicePolicy}
    (a : Tensor ⟨#[b, h, m, k], d, dev⟩) (x : Tensor ⟨#[b, h, k, n], d, dev⟩) :
    Tensor ⟨#[b, h, m, n], d, dev⟩ :=
  .assumeSpec (nn.bmm4d a.raw x.raw)

/-! ## Broadcasting elementwise ops

The result spec is `σ₁.pointwise σ₂` (NumPy/PyTorch rules; the device
policy of the left operand propagates), and the auto-param proof —
discharged by `rfl` for concrete shapes — guarantees the pair actually
broadcasts, so incompatible shape pairs fail to compile.

At runtime libtorch broadcasts natively; the operands are passed through a
`reshape #[]` type-erasure (a same-handle no-op in the C++ bridge), so no
expansion is materialized and the FFI surface is unchanged. -/

/-- Broadcasting elementwise addition (`a +ᵇ b`). -/
def addB {σ₁ σ₂ : StaticSpec} (a : Tensor σ₁) (b : Tensor σ₂)
    (_h : (TensorSpec.broadcastShapes? σ₁.shape σ₂.shape).isSome = true := by rfl) :
    Tensor (σ₁.pointwise σ₂) :=
  .assumeSpec (torch.reshape (torch.add (torch.reshape a.raw #[]) (torch.reshape b.raw #[]))
    (TensorSpec.broadcastShape σ₁.shape σ₂.shape))

/-- Broadcasting elementwise subtraction (`a -ᵇ b`). -/
def subB {σ₁ σ₂ : StaticSpec} (a : Tensor σ₁) (b : Tensor σ₂)
    (_h : (TensorSpec.broadcastShapes? σ₁.shape σ₂.shape).isSome = true := by rfl) :
    Tensor (σ₁.pointwise σ₂) :=
  .assumeSpec (torch.reshape (torch.sub (torch.reshape a.raw #[]) (torch.reshape b.raw #[]))
    (TensorSpec.broadcastShape σ₁.shape σ₂.shape))

/-- Broadcasting elementwise multiplication (`a *ᵇ b`). -/
def mulB {σ₁ σ₂ : StaticSpec} (a : Tensor σ₁) (b : Tensor σ₂)
    (_h : (TensorSpec.broadcastShapes? σ₁.shape σ₂.shape).isSome = true := by rfl) :
    Tensor (σ₁.pointwise σ₂) :=
  .assumeSpec (torch.reshape (torch.mul (torch.reshape a.raw #[]) (torch.reshape b.raw #[]))
    (TensorSpec.broadcastShape σ₁.shape σ₂.shape))

/-- Broadcasting elementwise division (`a /ᵇ b`). -/
def divB {σ₁ σ₂ : StaticSpec} (a : Tensor σ₁) (b : Tensor σ₂)
    (_h : (TensorSpec.broadcastShapes? σ₁.shape σ₂.shape).isSome = true := by rfl) :
    Tensor (σ₁.division σ₂) :=
  .assumeSpec (torch.reshape (nn.div (torch.reshape a.raw #[]) (torch.reshape b.raw #[]))
    (TensorSpec.broadcastShape σ₁.shape σ₂.shape))

/-- Asymmetric broadcast: stretch `t` to `target` (a libtorch `expand` view,
    no copy). The fixed-side proof `broadcastsTo s target` rules out targets
    that would themselves be stretched. -/
def broadcastTo {σ : StaticSpec} (t : Tensor σ) (target : Shape)
    (_h : TensorSpec.broadcastsTo σ.shape target := by rfl) :
    Tensor (σ.withShape target) :=
  .assumeSpec (nn.expand t.raw target)

/-- Unchecked `expand` view, for code generic over dimensions where the
    `broadcastTo` proof cannot reduce (e.g. `#[batch, outDim]` with symbolic
    dims). libtorch still validates the expansion at runtime. -/
def expand {σ : StaticSpec} (t : Tensor σ) (target : Shape) :
    Tensor (σ.withShape target) :=
  .assumeSpec (nn.expand t.raw target)

/-! ## Reductions -/

def sumAll {σ : StaticSpec} (t : Tensor σ) : Tensor σ.sumSpec :=
  .assumeSpec (nn.sumAll t.raw)

def meanAll {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by decide) :
    Tensor (σ.withShape #[]) :=
  .assumeSpec (nn.meanAll t.raw)

/-! ## Shape ops (dtype/device-preserving) -/

/-- Reshape without changing the element count. Unlike the legacy raw
    `torch.reshape`, an empty target creates a scalar. -/
def reshape {σ : StaticSpec} (t : Tensor σ) (shape' : Shape)
    (_h : TensorSpec.numelOfShape σ.shape = TensorSpec.numelOfShape shape' := by decide) :
    Tensor (σ.withShape shape') :=
  .assumeSpec (torch.reshapeExact t.raw shape')

def permute {σ : StaticSpec} (t : Tensor σ) (permutation : Array UInt64) :
    Tensor (σ.withShape (permuteShape σ.shape permutation)) :=
  .assumeSpec (torch.permute t.raw permutation)

def transpose {σ : StaticSpec} (t : Tensor σ) (dim0 dim1 : UInt64) :
    Tensor (σ.withShape (transposeShape σ.shape dim0.toNat dim1.toNat)) :=
  .assumeSpec (nn.transpose t.raw dim0 dim1)

def transpose2d {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Tensor ⟨#[n, m], d, dev⟩ :=
  .assumeSpec (nn.transpose2d t.raw)

def unsqueeze {σ : StaticSpec} (t : Tensor σ) (dim : Nat) :
    Tensor (σ.withShape (unsqueezeShape σ.shape dim)) :=
  .assumeSpec (nn.unsqueeze t.raw dim)

def squeeze {σ : StaticSpec} (t : Tensor σ) (dim : Nat) :
    Tensor (σ.withShape (squeezeShape σ.shape dim)) :=
  .assumeSpec (nn.squeeze t.raw dim)

def cat {s1 s2 : Shape} {d : DType} {dev : DevicePolicy}
    (t1 : Tensor ⟨s1, d, dev⟩) (t2 : Tensor ⟨s2, d, dev⟩) (dim : Nat) :
    Tensor ⟨nn.catShape s1 s2 dim, d, dev⟩ :=
  .assumeSpec (nn.cat t1.raw t2.raw dim)

/-! ## Autograd and fused layer primitives -/

/-- Set requires_grad (spec-preserving). -/
def setRequiresGrad {σ : StaticSpec} (t : Tensor σ) (requiresGrad : Bool) :
    Tensor σ :=
  .assumeSpec (autograd.set_requires_grad t.raw requiresGrad)

/-- Fused affine for 3D input: `x @ Wᵀ + b`. -/
def affine3d {batch seq inDim outDim : UInt64} {d : DType} {dev : DevicePolicy}
    (x : Tensor ⟨#[batch, seq, inDim], d, dev⟩)
    (weight : Tensor ⟨#[outDim, inDim], d, dev⟩)
    (bias : Tensor ⟨#[outDim], d, dev⟩) :
    Tensor ⟨#[batch, seq, outDim], d, dev⟩ :=
  .assumeSpec (torch.affine3d x.raw weight.raw bias.raw)

/-- Weighted RMS normalization over the last dimension.

    The runtime computes in Float32 for numerical stability (`nn.rmsNorm`
    upcasts via `toFloat'` and does not cast back), so the result dtype is
    `promote Float32 dtype` — Float32 for f16/bf16 inputs, Float64 for f64.
    Cast back explicitly (e.g. `.toBFloat16`) for HF-style dtype
    restoration. -/
def rmsNormWeighted {σ : StaticSpec} {w : Shape}
    (x : Tensor σ) (weight : Tensor (σ.withShape w)) (eps : Float := 1e-6)
    (_float : σ.dtype.isFloating = true := by decide) :
    Tensor (σ.withDType (DType.promote .Float32 σ.dtype)) :=
  .assumeSpec (nn.rmsNormWeighted x.raw weight.raw eps)

end Tensor

/-! ## Checked broadcasting operators

`+`/`-`/`*`/`/` stay same-shape (resolved by unification). Broadcasting
gets its own operators, which expand to `addB`/`subB`/`mulB`/`divB` —
unlike typeclass instances, notation produces a plain application, so the
broadcastability auto-param IS discharged and incompatible shapes fail to
compile. (A broadcasting `HAdd` instance cannot be guarded: instance
resolution never runs auto-param tactics.) -/

@[inherit_doc Tensor.addB] scoped infixl:65 " +ᵇ " => Tensor.addB
@[inherit_doc Tensor.subB] scoped infixl:65 " -ᵇ " => Tensor.subB
@[inherit_doc Tensor.mulB] scoped infixl:70 " *ᵇ " => Tensor.mulB
@[inherit_doc Tensor.divB] scoped infixl:70 " /ᵇ " => Tensor.divB

end torch
