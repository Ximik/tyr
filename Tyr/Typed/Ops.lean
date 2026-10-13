/-
  Tyr/Typed/Ops.lean

  Typed operation surface over the untyped `torch.*` FFI, indexed by
  `StaticSpec` (shape, dtype, device policy).

  Most wrappers re-index an existing extern without runtime metadata checks.
  The `Preserving` adapters check metadata where a generic dtype can make a
  native operation change its result type. Result specs follow
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
import Tyr.RawTensor
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
  .assumeSpec (RawTensor.zeros shape requiresGrad device)

def ones (shape : Shape) (requiresGrad : Bool := false)
    (device : Device := Device.CPU) : Tensor { shape, dtype := .Float32 } :=
  .assumeSpec (RawTensor.ones shape requiresGrad device)

def full (shape : Shape) (value : Float) (requiresGrad : Bool := false)
    (device : Device := Device.CPU) : Tensor { shape, dtype := .Float32 } :=
  .assumeSpec (RawTensor.full shape value requiresGrad device)

def fullInt (shape : Shape) (value : Int64) : Tensor { shape, dtype := .Int64 } :=
  .assumeSpec (RawTensor.fullInt shape value)

def zerosLike {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  .assumeSpec (RawTensor.zerosLike t.shape t.raw)

def onesLike {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  .assumeSpec (RawTensor.onesLike t.shape t.raw)

def randn (shape : Shape) (requiresGrad : Bool := false)
    (device : Device := Device.CPU) : IO (Tensor { shape, dtype := .Float32 }) := do
  pure (.assumeSpec (← RawTensor.randn shape requiresGrad device))

/-- Fresh uniform samples with the reference shape, floating dtype and actual
placement. Supports f16, bf16, f32 and f64; invalid bounds/dtypes return IO errors.
The result starts without gradient tracking, matching the random factories. -/
def uniformLike {σ : StaticSpec} (reference : Tensor σ) (lo : Float := 0.0)
    (hi : Float := 1.0) : IO (Tensor σ) := do
  let raw ← RawTensor.uniformLike reference.shape reference.raw lo hi
  match Tensor.ofRaw raw with
  | .ok result => pure result
  | .error message => throw (IO.userError s!"uniformLike: {message}")

/-- A valid zero value for a complete specification. Dynamic dtype uses the
native default; unconstrained placement chooses CPU. Pinned devices are respected. -/
def zerosFor (σ : StaticSpec) : Tensor σ :=
  let device := match σ.device with
    | .any => Device.CPU
    | .exact device => device
  let raw := RawTensor.zeros σ.shape false device
  let raw := match σ.dtype.safeTensorTag? with
    | some tag => RawTensor.castDType σ.shape raw tag
    | none => raw
  .assumeSpec raw

/-- A valid one value for a complete specification, mirroring `zerosFor`. -/
def onesFor (σ : StaticSpec) : Tensor σ :=
  let device := match σ.device with
    | .any => Device.CPU
    | .exact device => device
  let raw := RawTensor.ones σ.shape false device
  let raw := match σ.dtype.safeTensorTag? with
    | some tag => RawTensor.castDType σ.shape raw tag
    | none => raw
  .assumeSpec raw

instance {σ : StaticSpec} : Inhabited (Tensor σ) := ⟨zerosFor σ⟩

/-- Move without changing the static policy. A pinned policy must permit the
destination; unrestricted tensors may move to any device. -/
def to {σ : StaticSpec} (t : Tensor σ) (device : Device) : Tensor σ :=
  match σ.device.check device with
  | .error message =>
      letI : Inhabited (Tensor σ) := ⟨t⟩
      panic! message
  | .ok () => t.checkedReplacement (RawTensor.toDevice t.shape t.raw device)

def requiresGrad {σ : StaticSpec} (t : Tensor σ) : Bool :=
  RawTensor.requiresGrad t.shape t.raw

/-- Read a scalar as a host Float. Nonscalar inputs do not typecheck. -/
def item {d : DType} {dev : DevicePolicy} (t : Tensor ⟨#[], d, dev⟩) : Float :=
  RawTensor.item #[] t.raw

/-- Euclidean norm of an indexed vector. -/
def l2Norm {n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[n], d, dev⟩) : Float :=
  RawTensor.l2Norm n t.raw

def maximum {σ : StaticSpec} (a b : Tensor σ) : Tensor σ :=
  .assumeSpec (RawTensor.whereSelect a.shape (RawTensor.gt a.shape a.raw b.raw) a.raw b.raw)

/-- Diagonal of a square matrix, retaining dtype and device policy. -/
def diag {n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[n, n], d, dev⟩) : Tensor ⟨#[n], d, dev⟩ :=
  .assumeSpec (RawTensor.diag n t.raw)

/-- Frobenius norm of an indexed matrix. -/
def frobeniusNorm {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Float :=
  RawTensor.frobeniusNorm m n t.raw

/-- Row norms of a real floating-point matrix. -/
def rowNorms {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Tensor ⟨#[m], d, dev⟩ :=
  .assumeSpec (RawTensor.rowNorms m n t.raw)

/-- Maximum Euclidean row norm of a matrix. -/
def maxRowNorm {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Float :=
  RawTensor.maxRowNorm m n t.raw

/-- Apply dropout while retaining shape, dtype and device policy. -/
def dropout {σ : StaticSpec} (t : Tensor σ) (p : Float) (training : Bool := true) :
    IO (Tensor σ) := do
  pure (.assumeSpec (← RawTensor.dropout t.shape t.raw p training))

/-- Batched embedding lookup. Tokens must be Int64 and share the weight's
placement policy; libtorch checks token values against the vocabulary size. -/
def embedding {batch seq vocab embed : UInt64} {d : DType} {dev : DevicePolicy}
    (tokens : Tensor ⟨#[batch, seq], .Int64, dev⟩)
    (weight : Tensor ⟨#[vocab, embed], d, dev⟩) : Tensor ⟨#[batch, seq, embed], d, dev⟩ :=
  .assumeSpec (RawTensor.embedding batch seq vocab embed tokens.raw weight.raw)

/-- Vector embedding lookup with statically integral tokens. -/
def embedding1d {seq vocab embed : UInt64} {d : DType} {dev : DevicePolicy}
    (tokens : Tensor ⟨#[seq], .Int64, dev⟩)
    (weight : Tensor ⟨#[vocab, embed], d, dev⟩) : Tensor ⟨#[seq, embed], d, dev⟩ :=
  .assumeSpec (RawTensor.embedding1d seq vocab embed tokens.raw weight.raw)

/-- Matrix exponential with a checked dtype/device-preserving result. -/
def matrixExp {n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[n, n], d, dev⟩) : Tensor ⟨#[n, n], d, dev⟩ :=
  t.checkedReplacement (RawTensor.matrixExp n t.raw)

/-- Real matrix logarithm with a checked dtype/device-preserving result.
The native implementation returns NaNs if a significant imaginary part remains. -/
def matrixLog {n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[n, n], d, dev⟩) : Tensor ⟨#[n, n], d, dev⟩ :=
  t.checkedReplacement (RawTensor.matrixLog n t.raw)

/-- Inverse of a square matrix. Native code reports singular input errors. -/
def inv {n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[n, n], d, dev⟩) : Tensor ⟨#[n, n], d, dev⟩ :=
  t.checkedReplacement (RawTensor.inv n t.raw)

/-- Float32 identity matrix on the default placement. -/
def eye (n : UInt64) : Tensor ⟨#[n, n], .Float32, .any⟩ :=
  .assumeSpec (RawTensor.eye n)

/-- Identity matrix for a complete specification. The native factory produces
    Float32 on CPU; the value is cast and moved to match the specification. -/
def eyeFor {n : UInt64} (d : DType) (dev : DevicePolicy := .any) : Tensor ⟨#[n, n], d, dev⟩ :=
  let device := match dev with
    | .any => Device.CPU
    | .exact device => device
  let raw := RawTensor.eye n
  let raw := match d.safeTensorTag? with
    | some tag => RawTensor.castDType #[n, n] raw tag
    | none => raw
  let raw := RawTensor.toDevice #[n, n] raw device
  .assumeSpec raw

/-- Complete QR decomposition: `Q` is m×m orthogonal and `R` is m×n upper
    triangular, retaining dtype and device policy. -/
def qrComplete {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Tensor ⟨#[m, m], d, dev⟩ × Tensor ⟨#[m, n], d, dev⟩ :=
  let (q, r) := RawTensor.qr m n t.raw
  (.assumeSpec q, .assumeSpec r)

/-- Reduced QR decomposition of a tall or square matrix (`n ≤ m`): `Q` is m×n
    with orthonormal columns and `R` is n×n upper triangular. The shape contract
    is a precondition on the libtorch reduced factorization; wide inputs do not
    typecheck at concrete dimensions. -/
def qrReduced {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[m, n], d, dev⟩) (_tall : n.toNat ≤ m.toNat := by decide) :
    Tensor ⟨#[m, n], d, dev⟩ × Tensor ⟨#[n, n], d, dev⟩ :=
  let (q, r) := RawTensor.qrReduced m n t.raw
  (.assumeSpec q, .assumeSpec r)

/-- Reduced SVD of a tall or square matrix (`n ≤ m`): `U` is m×n, `S` holds the
    n singular values and `Vh` is n×n. The shape contract is a precondition on
    the libtorch reduced factorization; wide inputs do not typecheck at
    concrete dimensions. -/
def svdReduced {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[m, n], d, dev⟩) (_tall : n.toNat ≤ m.toNat := by decide) :
    Tensor ⟨#[m, n], d, dev⟩ × Tensor ⟨#[n], d, dev⟩ × Tensor ⟨#[n, n], d, dev⟩ :=
  let (u, s, vh) := RawTensor.svd m n t.raw
  (.assumeSpec u, .assumeSpec s, .assumeSpec vh)

/-- Singular values of a square matrix in descending order. -/
def svdvals {n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[n, n], d, dev⟩) : Tensor ⟨#[n], d, dev⟩ :=
  .assumeSpec (RawTensor.svdvals n n t.raw)

/-- Diagonal matrix built from a vector, retaining dtype and device policy. -/
def diagflat {n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[n], d, dev⟩) : Tensor ⟨#[n, n], d, dev⟩ :=
  .assumeSpec (RawTensor.diagflat n t.raw)

/-- Largest singular value of a matrix. -/
def spectralNorm {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Float :=
  RawTensor.spectralNorm m n t.raw

/-- Sum of the singular values of a matrix. -/
def nuclearNorm {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Float :=
  RawTensor.nuclearNorm m n t.raw

/-! ## Casts -/

def toFloat32 {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withDType .Float32) :=
  .assumeSpec (RawTensor.toFloat32 t.shape t.raw)

def toBFloat16 {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withDType .BFloat16) :=
  .assumeSpec (RawTensor.toBFloat16 t.shape t.raw)

def toInt64 {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withDType .Int64) :=
  .assumeSpec (RawTensor.toInt64 t.shape t.raw)

/-! ## Elementwise binary (same shape and device, dtypes promote) -/

def add {s : Shape} {l r : DType} {dev : DevicePolicy}
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, DType.promote l r, dev⟩ :=
  .assumeSpec (RawTensor.add a.shape a.raw b.raw)

def sub {s : Shape} {l r : DType} {dev : DevicePolicy}
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, DType.promote l r, dev⟩ :=
  .assumeSpec (RawTensor.sub a.shape a.raw b.raw)

def mul {s : Shape} {l r : DType} {dev : DevicePolicy}
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, DType.promote l r, dev⟩ :=
  .assumeSpec (RawTensor.mul a.shape a.raw b.raw)

def div {s : Shape} {l r : DType} {dev : DevicePolicy}
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, (DType.promote l r).atLeastFloat, dev⟩ :=
  .assumeSpec (RawTensor.div a.shape a.raw b.raw)

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
  ⟨fun a b => .assumeSpec (RawTensor.add a.shape a.raw b.raw)⟩

instance {σ : StaticSpec} : Sub (Tensor σ) :=
  ⟨fun a b => .assumeSpec (RawTensor.sub a.shape a.raw b.raw)⟩

instance {σ : StaticSpec} : Mul (Tensor σ) :=
  ⟨fun a b => .assumeSpec (RawTensor.mul a.shape a.raw b.raw)⟩

/-! ## Scalar arithmetic (Float scalars are weak but still float a result) -/

def addScalar {σ : StaticSpec} (t : Tensor σ) (value : Float) :
    Tensor (σ.withDType σ.dtype.atLeastFloat) :=
  .assumeSpec (RawTensor.addScalar t.shape t.raw value)

def subScalar {σ : StaticSpec} (t : Tensor σ) (value : Float) :
    Tensor (σ.withDType σ.dtype.atLeastFloat) :=
  .assumeSpec (RawTensor.subScalar t.shape t.raw value)

def mulScalar {σ : StaticSpec} (t : Tensor σ) (value : Float) :
    Tensor (σ.withDType σ.dtype.atLeastFloat) :=
  .assumeSpec (RawTensor.mulScalar t.shape t.raw value)

def divScalar {σ : StaticSpec} (t : Tensor σ) (value : Float) :
    Tensor (σ.withDType σ.dtype.atLeastFloat) :=
  .assumeSpec (RawTensor.divScalar t.shape t.raw value)

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
  .assumeSpec (RawTensor.relu t.shape t.raw)

def abs {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  .assumeSpec (RawTensor.abs t.shape t.raw)

/-! ## Float-only unary math

Requires `σ.dtype.isFloating`; the proof is found by `decide` when the
dtype is a literal. Integer tensors must cast explicitly (`.toFloat32`). -/

def exp {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (RawTensor.exp t.shape t.raw)

def sqrt {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (RawTensor.sqrt t.shape t.raw)

def rsqrt {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (RawTensor.rsqrt t.shape t.raw)

def sigmoid {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (RawTensor.sigmoid t.shape t.raw)

def tanh {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (RawTensor.tanh t.shape t.raw)

def gelu {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (RawTensor.gelu t.shape t.raw)

def silu {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (RawTensor.silu t.shape t.raw)

def pow {σ : StaticSpec} (t : Tensor σ) (exponent : Float)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (RawTensor.pow t.shape t.raw exponent)

def softmax {σ : StaticSpec} (t : Tensor σ) (dim : Int32 := -1)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (RawTensor.softmax t.shape t.raw dim)

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
  .assumeSpec (RawTensor.matmul s1 s2 a.raw b.raw)

def mm {m k n : UInt64} {d : DType} {dev : DevicePolicy}
    (a : Tensor ⟨#[m, k], d, dev⟩) (b : Tensor ⟨#[k, n], d, dev⟩) :
    Tensor ⟨#[m, n], d, dev⟩ :=
  .assumeSpec (RawTensor.mm m n k a.raw b.raw)

def linear {m n b : UInt64} {d : DType} {dev : DevicePolicy}
    (x : Tensor ⟨#[b, m], d, dev⟩) (weight : Tensor ⟨#[n, m], d, dev⟩) :
    Tensor ⟨#[b, n], d, dev⟩ :=
  .assumeSpec (RawTensor.linear m n b x.raw weight.raw)

def linear3d {batch seq inDim outDim : UInt64} {d : DType} {dev : DevicePolicy}
    (x : Tensor ⟨#[batch, seq, inDim], d, dev⟩)
    (weight : Tensor ⟨#[outDim, inDim], d, dev⟩) :
    Tensor ⟨#[batch, seq, outDim], d, dev⟩ :=
  .assumeSpec (RawTensor.linear3d batch seq inDim outDim x.raw weight.raw)

def matmul3d {batch seq k n : UInt64} {d : DType} {dev : DevicePolicy}
    (a : Tensor ⟨#[batch, seq, k], d, dev⟩) (b : Tensor ⟨#[k, n], d, dev⟩) :
    Tensor ⟨#[batch, seq, n], d, dev⟩ :=
  .assumeSpec (RawTensor.matmul3d batch seq k n a.raw b.raw)

def bmm4d {b h m k n : UInt64} {d : DType} {dev : DevicePolicy}
    (a : Tensor ⟨#[b, h, m, k], d, dev⟩) (x : Tensor ⟨#[b, h, k, n], d, dev⟩) :
    Tensor ⟨#[b, h, m, n], d, dev⟩ :=
  .assumeSpec (RawTensor.bmm4d b h m k n a.raw x.raw)

/-! ## Broadcasting elementwise ops

The result spec is `σ₁.pointwise σ₂` (NumPy/PyTorch rules; the device
policy of the left operand propagates), and the auto-param proof —
discharged by `rfl` for concrete shapes — guarantees the pair actually
broadcasts, so incompatible shape pairs fail to compile.

At runtime libtorch broadcasts the native handles directly; no shape erasure
or reshaping is required at the raw boundary. -/

/-- Broadcasting elementwise addition (`a +ᵇ b`). -/
def addB {σ₁ σ₂ : StaticSpec} (a : Tensor σ₁) (b : Tensor σ₂)
    (_h : (TensorSpec.broadcastShapes? σ₁.shape σ₂.shape).isSome = true := by rfl) :
    Tensor (σ₁.pointwise σ₂) :=
  .assumeSpec (RawTensor.add a.shape a.raw b.raw)

/-- Broadcasting elementwise subtraction (`a -ᵇ b`). -/
def subB {σ₁ σ₂ : StaticSpec} (a : Tensor σ₁) (b : Tensor σ₂)
    (_h : (TensorSpec.broadcastShapes? σ₁.shape σ₂.shape).isSome = true := by rfl) :
    Tensor (σ₁.pointwise σ₂) :=
  .assumeSpec (RawTensor.sub a.shape a.raw b.raw)

/-- Broadcasting elementwise multiplication (`a *ᵇ b`). -/
def mulB {σ₁ σ₂ : StaticSpec} (a : Tensor σ₁) (b : Tensor σ₂)
    (_h : (TensorSpec.broadcastShapes? σ₁.shape σ₂.shape).isSome = true := by rfl) :
    Tensor (σ₁.pointwise σ₂) :=
  .assumeSpec (RawTensor.mul a.shape a.raw b.raw)

/-- Broadcasting elementwise division (`a /ᵇ b`). -/
def divB {σ₁ σ₂ : StaticSpec} (a : Tensor σ₁) (b : Tensor σ₂)
    (_h : (TensorSpec.broadcastShapes? σ₁.shape σ₂.shape).isSome = true := by rfl) :
    Tensor (σ₁.division σ₂) :=
  .assumeSpec (RawTensor.div a.shape a.raw b.raw)

/-- Asymmetric broadcast: stretch `t` to `target` (a libtorch `expand` view,
    no copy). The fixed-side proof `broadcastsTo s target` rules out targets
    that would themselves be stretched. -/
def broadcastTo {σ : StaticSpec} (t : Tensor σ) (target : Shape)
    (_h : TensorSpec.broadcastsTo σ.shape target := by rfl) :
    Tensor (σ.withShape target) :=
  .assumeSpec (RawTensor.expand t.shape t.raw target)

/-- Unchecked `expand` view, for code generic over dimensions where the
    `broadcastTo` proof cannot reduce (e.g. `#[batch, outDim]` with symbolic
    dims). libtorch still validates the expansion at runtime. -/
def expand {σ : StaticSpec} (t : Tensor σ) (target : Shape) :
    Tensor (σ.withShape target) :=
  .assumeSpec (RawTensor.expand t.shape t.raw target)

/-! ## Reductions -/

def sumAll {σ : StaticSpec} (t : Tensor σ) : Tensor σ.sumSpec :=
  .assumeSpec (RawTensor.sumAll t.shape t.raw)

/-- Maximum over all elements: a scalar retaining dtype and device policy. -/
def maxAll {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withShape #[]) :=
  .assumeSpec (RawTensor.maxAll t.raw)

/-- Reduce one axis using libtorch's accumulation dtype. -/
def sumDim {σ : StaticSpec} (t : Tensor σ) (dim : Nat) (keepdim : Bool := false) :
    Tensor (σ.sumSpec.withShape (reduceShape σ.shape dim keepdim)) :=
  .assumeSpec (RawTensor.sum t.shape t.raw (some #[dim.toUInt64]) keepdim)

def meanAll {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by decide) :
    Tensor (σ.withShape #[]) :=
  .assumeSpec (RawTensor.meanAll t.shape t.raw)

/-! ## Shape ops (dtype/device-preserving) -/

/-- Reshape without changing the element count. Unlike the legacy raw
    `torch.reshape`, an empty target creates a scalar. -/
def reshape {σ : StaticSpec} (t : Tensor σ) (shape' : Shape)
    (_h : TensorSpec.numelOfShape σ.shape = TensorSpec.numelOfShape shape' := by decide) :
    Tensor (σ.withShape shape') :=
  .assumeSpec (RawTensor.reshape t.shape t.raw shape')

/-- Slice an exact number of entries along a valid axis. Bounds are checked in
natural arithmetic so `start + len` cannot wrap or silently truncate. -/
def slice {σ : StaticSpec} (t : Tensor σ) (dim : Nat) (start len : UInt64)
    (_axis : dim < σ.shape.size := by decide)
    (_bounds : start.toNat + len.toNat ≤ (σ.shape[dim]'_axis).toNat := by decide) :
    Tensor (σ.withShape (σ.shape.set! dim len)) :=
  .assumeSpec (RawTensor.slice t.shape t.raw dim.toUInt64 start len)

def permute {σ : StaticSpec} (t : Tensor σ) (permutation : Array UInt64) :
    Tensor (σ.withShape (TensorSpec.permuteShape σ.shape permutation)) :=
  .assumeSpec (RawTensor.permute t.shape t.raw permutation)

def transpose {σ : StaticSpec} (t : Tensor σ) (dim0 dim1 : UInt64) :
    Tensor (σ.withShape (transposeShape σ.shape dim0.toNat dim1.toNat)) :=
  .assumeSpec (RawTensor.transpose t.shape t.raw dim0 dim1)

def transpose2d {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Tensor ⟨#[n, m], d, dev⟩ :=
  .assumeSpec (RawTensor.transpose2d m n t.raw)

def unsqueeze {σ : StaticSpec} (t : Tensor σ) (dim : Nat) :
    Tensor (σ.withShape (unsqueezeShape σ.shape dim)) :=
  .assumeSpec (RawTensor.unsqueeze t.shape t.raw dim)

def squeeze {σ : StaticSpec} (t : Tensor σ) (dim : Nat) :
    Tensor (σ.withShape (squeezeShape σ.shape dim)) :=
  .assumeSpec (RawTensor.squeeze t.shape t.raw dim)

def cat {s1 s2 : Shape} {d : DType} {dev : DevicePolicy}
    (t1 : Tensor ⟨s1, d, dev⟩) (t2 : Tensor ⟨s2, d, dev⟩) (dim : Nat) :
    Tensor ⟨TensorSpec.catShape s1 s2 dim, d, dev⟩ :=
  .assumeSpec (RawTensor.cat s1 s2 t1.raw t2.raw dim)

/-- Stack a runtime list of equal vectors along a new leading axis; the result
    shape records the actual list length. -/
def stack1d {cols : UInt64} {d : DType} {dev : DevicePolicy}
    (tensors : Array (Tensor ⟨#[cols], d, dev⟩)) :
    Tensor ⟨#[tensors.size.toUInt64, cols], d, dev⟩ :=
  .assumeSpec (RawTensor.stack1d tensors.size.toUInt64 cols (tensors.map Tensor.toTensor) 0)

/-- Reshape to a runtime target, validating result metadata against the claimed
    spec and panicking on mismatch. For dynamically computed shapes where the
    static `reshape` numel proof cannot reduce; the runtime check keeps the
    conversion honest. -/
def reshapeChecked {σ : StaticSpec} (t : Tensor σ) (shape' : Shape) : Tensor (σ.withShape shape') :=
  match Tensor.ofRaw (RawTensor.reshape σ.shape t.raw shape') with
  | .ok r => r
  | .error message => panic! s!"Tensor.reshapeChecked: {message}"

/-- Slice `len` entries along `dim` at a runtime offset, validating the result
    against the claimed spec and panicking on mismatch. -/
def sliceChecked {σ : StaticSpec} (t : Tensor σ) (dim : Nat) (start len : UInt64)
    (shape' : Shape) : Tensor (σ.withShape shape') :=
  match Tensor.ofRaw (RawTensor.slice σ.shape t.raw dim.toUInt64 start len) with
  | .ok r => r
  | .error message => panic! s!"Tensor.sliceChecked: {message}"

/-- Vector-Jacobian product: for `y = f(x)` and output cotangent `dy`, returns
    the input cotangent, checked against `x`'s spec. -/
def grad {σ τ : StaticSpec} (y : Tensor τ) (x : Tensor σ) (dy : Tensor τ) : Tensor σ :=
  x.checkedReplacement (RawTensor.grad σ.shape τ.shape y.raw x.raw dy.raw)

/-- 1-D Float32 tensor from a host array; the result shape records the actual
    array length. -/
def fromFloatArray (arr : Array Float) : Tensor ⟨#[arr.size.toUInt64], .Float32, .any⟩ :=
  .assumeSpec (RawTensor.fromFloatArray arr)

/-- 1-D Int64 tensor from a host array; the result shape records the actual
    array length. -/
def fromInt64Array (arr : Array Int64) : Tensor ⟨#[arr.size.toUInt64], .Int64, .any⟩ :=
  .assumeSpec (RawTensor.fromInt64Array arr)

/-- Argmax along `dim`: Int64 indices with that dimension removed. -/
def argmax {s : Shape} {d : DType} {dev : DevicePolicy} (t : Tensor ⟨s, d, dev⟩) (dim : UInt64) :
    Tensor ⟨reduceShape s dim.toNat false, .Int64, dev⟩ :=
  .assumeSpec (RawTensor.argmax s t.raw dim)

/-- Top-k filter over logits (entries outside the top k become -inf), spec-preserving. -/
def topKFilter {σ : StaticSpec} (logits : Tensor σ) (k : UInt64) : Tensor σ :=
  .assumeSpec (RawTensor.topKFilter σ.shape logits.raw k)

/-- Top-p (nucleus) filter over logits, spec-preserving. -/
def topPFilter {σ : StaticSpec} (logits : Tensor σ) (p : Float) : Tensor σ :=
  .assumeSpec (RawTensor.topPFilter σ.shape logits.raw p)

/-- Elementwise equality with a scalar: Bool tensor of the same shape. -/
def eqScalar {s : Shape} {d : DType} {dev : DevicePolicy} (t : Tensor ⟨s, d, dev⟩) (scalar : Int64) :
    Tensor ⟨s, .Bool, dev⟩ :=
  .assumeSpec (RawTensor.eqScalar s t.raw scalar)

/-- Select entries of a 1-D tensor by Int64 indices. -/
def indexSelect {n k : UInt64} {d : DType} {dev : DevicePolicy}
    (data : Tensor ⟨#[n], d, dev⟩) (indices : Tensor ⟨#[k], .Int64, dev⟩) :
    Tensor ⟨#[k], d, dev⟩ :=
  .assumeSpec (RawTensor.indexSelect n k data.raw 0 indices.raw)

/-- Per-element cross entropy (no reduction) over a 2-D logits matrix with Int64 targets. -/
def crossEntropyNone {n c : UInt64} {d : DType} {dev : DevicePolicy}
    (logits : Tensor ⟨#[n, c], d, dev⟩) (targets : Tensor ⟨#[n], .Int64, dev⟩) :
    Tensor ⟨#[n], d, dev⟩ :=
  .assumeSpec (RawTensor.crossEntropyNone n c logits.raw targets.raw)

/-- Extract an Int64 scalar. -/
def itemInt {d : DType} {dev : DevicePolicy} (t : Tensor ⟨#[], d, dev⟩) : Int64 :=
  RawTensor.itemInt #[] t.raw

/-- Run an IO action with autograd graph construction disabled. -/
def noGrad {α : Type} (action : IO α) : IO α := do
  let wasEnabled ← RawTensor.isGradEnabled
  RawTensor.setGradEnabled false
  try
    action
  finally
    RawTensor.setGradEnabled wasEnabled

/-- Select between two same-spec tensors elementwise by a Bool condition. -/
def whereSelect {s : Shape} {d : DType} {dev : DevicePolicy}
    (condition : Tensor ⟨s, .Bool, dev⟩) (x y : Tensor ⟨s, d, dev⟩) :
    Tensor ⟨s, d, dev⟩ :=
  .assumeSpec (RawTensor.whereSelect s condition.raw x.raw y.raw)

/-- Elementwise logical negation of a Bool tensor. -/
def logicalNot {s : Shape} {dev : DevicePolicy} (t : Tensor ⟨s, .Bool, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  .assumeSpec (RawTensor.logicalNot s t.raw)

/-- Elementwise logical conjunction of two Bool tensors. -/
def logicalAnd {s : Shape} {dev : DevicePolicy} (a b : Tensor ⟨s, .Bool, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  .assumeSpec (RawTensor.logicalAnd s a.raw b.raw)

/-- Elementwise logical disjunction of two Bool tensors. -/
def logicalOr {s : Shape} {dev : DevicePolicy} (a b : Tensor ⟨s, .Bool, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  logicalNot (logicalAnd (logicalNot a) (logicalNot b))

/-- Cross-entropy loss over a 2-D logits matrix with Int64 targets. -/
def crossEntropy {n c : UInt64} {d : DType} {dev : DevicePolicy}
    (logits : Tensor ⟨#[n, c], d, dev⟩) (targets : Tensor ⟨#[n], .Int64, dev⟩) :
    Tensor ⟨#[], d, dev⟩ :=
  .assumeSpec (RawTensor.crossEntropy n c logits.raw targets.raw)

/-- Draw `numSamples` Int64 indices per row of a probability matrix. -/
def multinomial {n v : UInt64} {d : DType} {dev : DevicePolicy}
    (probs : Tensor ⟨#[n, v], d, dev⟩) (numSamples : UInt64) (replacement : Bool := false) :
    IO (Tensor ⟨#[n, numSamples], .Int64, dev⟩) := do
  pure (.assumeSpec (← RawTensor.multinomial #[n, v] probs.raw numSamples replacement))

/-- Scatter `src` into a clone of `input` along `dim` at Int64 `indices`;
    result preserves the input spec. -/
def scatter2d {batch seq k : UInt64} {d : DType} {dev : DevicePolicy}
    (input : Tensor ⟨#[batch, seq], d, dev⟩) (dim : Int64)
    (indices : Tensor ⟨#[batch, k], .Int64, dev⟩) (src : Tensor ⟨#[batch, k], d, dev⟩) :
    Tensor ⟨#[batch, seq], d, dev⟩ :=
  input.checkedReplacement (RawTensor.scatter2d batch seq k input.raw dim indices.raw src.raw)

/-- All-close predicate over two tensors of the same spec. -/
def allclose {σ : StaticSpec} (a b : Tensor σ) (rtol : Float := 1e-05) (atol : Float := 1e-08) :
    Bool :=
  RawTensor.allclose a.raw b.raw rtol atol

/-- Whether the tensor is an autograd leaf. -/
def isLeaf {σ : StaticSpec} (t : Tensor σ) : Bool :=
  RawTensor.isLeaf t.raw

/-- Log-softmax along a dimension, spec-preserving (float-only). -/
def logSoftmax {σ : StaticSpec} (t : Tensor σ) (dim : Int32 := -1)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ :=
  .assumeSpec (RawTensor.logSoftmax t.raw dim)

/-- Backward pass from a scalar loss (gradient = 1.0). -/
def backwardLoss {d : DType} {dev : DevicePolicy} (loss : Tensor ⟨#[], d, dev⟩) : IO Unit :=
  RawTensor.backwardLoss loss.raw

/-! ## Autograd and fused layer primitives -/

def detach {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  .assumeSpec (RawTensor.detach t.shape t.raw)

def gradOf {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  .assumeSpec (RawTensor.gradOf t.shape t.raw)

def zeroGrad {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  .assumeSpec (RawTensor.zeroGrad t.shape t.raw)

/-! Operations for generic parameter trees. Unlike the promoting arithmetic
surface, these require the native result to retain the input specification.
In particular, floating scalar arithmetic on an integer parameter fails the
dtype check instead of returning an incorrectly indexed tensor. -/
namespace Preserving

/-- Clamp while checking that floating bounds did not promote the dtype. -/
def clampFloat {σ : StaticSpec} (t : Tensor σ) (lo hi : Float) : Tensor σ :=
  t.checkedReplacement (RawTensor.clampFloat t.shape t.raw lo hi)

/-- Reduction for generic real-valued algorithms that require the input dtype.
Integer accumulation that changes dtype is rejected at this boundary. -/
def sumDim {σ : StaticSpec} (t : Tensor σ) (dim : Nat) (keepdim : Bool := false) :
    Tensor (σ.withShape (reduceShape σ.shape dim keepdim)) :=
  match Tensor.ofRaw (Tensor.sumDim t dim keepdim).raw with
  | .ok result => result
  | .error message => panic! s!"Tensor sumDim: {message}"

def mulScalar {σ : StaticSpec} (t : Tensor σ) (value : Float) : Tensor σ :=
  t.checkedReplacement (RawTensor.mulScalar t.shape t.raw value)

def divScalar {σ : StaticSpec} (t : Tensor σ) (value : Float) : Tensor σ :=
  t.checkedReplacement (RawTensor.divScalar t.shape t.raw value)

def addScalar {σ : StaticSpec} (t : Tensor σ) (value : Float) : Tensor σ :=
  t.checkedReplacement (RawTensor.addScalar t.shape t.raw value)

def sqrt {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  t.checkedReplacement (RawTensor.sqrt t.shape t.raw)

def atan {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  t.checkedReplacement (RawTensor.atan t.shape t.raw)

def div {σ : StaticSpec} (lhs rhs : Tensor σ) : Tensor σ :=
  lhs.checkedReplacement (RawTensor.div lhs.shape lhs.raw rhs.raw)

end Preserving

/-- Set requires_grad (spec-preserving). -/
def setRequiresGrad {σ : StaticSpec} (t : Tensor σ) (requiresGrad : Bool) :
    Tensor σ :=
  .assumeSpec (RawTensor.setRequiresGrad t.shape t.raw requiresGrad)

/-- Fused affine for 3D input: `x @ Wᵀ + b`. -/
def affine3d {batch seq inDim outDim : UInt64} {d : DType} {dev : DevicePolicy}
    (x : Tensor ⟨#[batch, seq, inDim], d, dev⟩)
    (weight : Tensor ⟨#[outDim, inDim], d, dev⟩)
    (bias : Tensor ⟨#[outDim], d, dev⟩) :
    Tensor ⟨#[batch, seq, outDim], d, dev⟩ :=
  .assumeSpec (RawTensor.affine3d batch seq inDim outDim x.raw weight.raw bias.raw)

/-- Fused layer normalization over the last dimension, preserving the spec. -/
def layerNorm {batch seq n : UInt64} {d : DType} {dev : DevicePolicy}
    (x : Tensor ⟨#[batch, seq, n], d, dev⟩)
    (weight : Tensor ⟨#[n], d, dev⟩)
    (bias : Tensor ⟨#[n], d, dev⟩) (eps : Float := 1e-5) :
    Tensor ⟨#[batch, seq, n], d, dev⟩ :=
  .assumeSpec (RawTensor.layerNorm batch seq n x.raw weight.raw bias.raw eps)

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
  .assumeSpec (RawTensor.rmsNormWeighted σ.shape w x.raw weight.raw eps)

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
