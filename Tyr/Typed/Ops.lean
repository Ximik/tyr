/-
  Tyr/Typed/Ops.lean

  Typed operation surface over the untyped `torch.*` FFI, indexed by
  `StaticSpec` (shape, dtype, device policy).

  Native operation results pass through checked construction before acquiring
  a static specification. The `Preserving` adapters retain an existing value as
  the panic fallback when a generic dtype changes unexpectedly. Result specs
  follow PyTorch's semantics as encoded in the `StaticSpec`/`DType` algebra;
  runtime metadata checks and the parity tests in `Tests/TestTyped.lean`
  validate those claims.

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

/-- Only return native results that satisfy their complete output specification.
A mismatch uses the valid default through Lean's panic mechanism. This also
catches wrapped UInt64 dimensions interpreted natively as negative axes or
inferred extents. -/
private def checkedResult {σ : StaticSpec} (operation : String)
    (result : Except String (Tensor σ)) : Tensor σ :=
  match result with
  | .ok result => result
  | .error message => panic! s!"Tensor.{operation}: {message}"

/-! ## Creation

The runtime defaults are Float32 for float factories and Int64 for
`full_int`; the indices state exactly that. The device policy stays `.any`:
the runtime `device` argument places the tensor, the policy only constrains
which placements the type admits. -/

def zeros (shape : Shape) (requiresGrad : Bool := false)
    (device : Device := Device.CPU) (noGrad : requiresGrad = false := by decide) : Tensor { shape, dtype := .Float32 } :=
  checkedResult "zeros" (Tensor.ofRaw (RawTensor.zeros shape requiresGrad device noGrad))

def ones (shape : Shape) (requiresGrad : Bool := false)
    (device : Device := Device.CPU) (noGrad : requiresGrad = false := by decide) : Tensor { shape, dtype := .Float32 } :=
  checkedResult "ones" (Tensor.ofRaw (RawTensor.ones shape requiresGrad device noGrad))

def full (shape : Shape) (value : Float) (requiresGrad : Bool := false)
    (device : Device := Device.CPU) (noGrad : requiresGrad = false := by decide) : Tensor { shape, dtype := .Float32 } :=
  checkedResult "full" (Tensor.ofRaw (RawTensor.full shape value requiresGrad device noGrad))

def fullInt (shape : Shape) (value : Int64) : Tensor { shape, dtype := .Int64 } :=
  checkedResult "fullInt" (Tensor.ofRaw (RawTensor.fullInt shape value))

def zerosLike {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  t.checkedReplacement (RawTensor.zerosLike t.shape t.raw)

def onesLike {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  t.checkedReplacement (RawTensor.onesLike t.shape t.raw)

def randn (shape : Shape) (requiresGrad : Bool := false)
    (device : Device := Device.CPU) : IO (Tensor { shape, dtype := .Float32 }) := do
  let raw ← RawTensor.randn shape requiresGrad device
  match Tensor.ofRaw raw with
  | .ok result => pure result
  | .error message => throw (IO.userError s!"Tensor.randn: {message}")

/-- Fresh uniform samples with the reference shape, floating dtype and actual
placement. Supports f16, bf16, f32 and f64; invalid bounds/dtypes return IO errors.
The result starts without gradient tracking, matching the random factories. -/
def uniformLike {σ : StaticSpec} (reference : Tensor σ) (lo : Float := 0.0)
    (hi : Float := 1.0) : IO (Tensor σ) := do
  let raw ← RawTensor.uniformLike reference.shape reference.raw lo hi
  match reference.checkedReplacement? raw with
  | .ok result => pure result
  | .error message => throw (IO.userError s!"uniformLike: {message}")

/-- A valid one value for a complete specification, mirroring `zerosFor`. -/
def onesFor (σ : StaticSpec) : Tensor σ :=
  let device := match σ.device with
    | .any => Device.CPU
    | .exact device => device
  let raw := RawTensor.ones σ.shape false device
  let raw := if σ.dtype.isKnown then
    RawTensor.castDType σ.shape raw σ.dtype.canonicalName else raw
  checkedResult "onesFor" (Tensor.ofRaw raw)

/-- Move without changing the static policy. A pinned policy must permit the
destination; unrestricted tensors may move to any device. -/
def to {σ : StaticSpec} (t : Tensor σ) (device : Device) : Tensor σ :=
  letI : Inhabited (Tensor σ) := ⟨t⟩
  let result : Except String (Tensor σ) := do
    σ.device.check device
    let raw := RawTensor.toDevice t.shape t.raw device
    TensorSpec.checkDType t.raw.dtype raw.dtype
    DevicePolicy.check (.exact device) raw.device
    Tensor.ofRaw raw
  match result with
  | .ok tensor => tensor
  | .error message => panic! s!"Tensor.to: {message}"

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
  a.checkedReplacement (RawTensor.whereSelect a.shape (RawTensor.gt a.shape a.raw b.raw) a.raw b.raw)

/-- Diagonal of a square matrix, retaining dtype and device policy. -/
def diag {n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[n, n], d, dev⟩) : Tensor ⟨#[n], d, dev⟩ :=
  checkedResult "diag" (Tensor.ofRaw (RawTensor.diag n t.raw))

/-- Frobenius norm of an indexed matrix. -/
def frobeniusNorm {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Float :=
  RawTensor.frobeniusNorm m n t.raw

/-- Row norms of a real floating-point matrix. -/
def rowNorms {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Tensor ⟨#[m], d, dev⟩ :=
  checkedResult "rowNorms" (Tensor.ofRaw (RawTensor.rowNorms m n t.raw))

/-- Maximum Euclidean row norm of a matrix. -/
def maxRowNorm {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Float :=
  RawTensor.maxRowNorm m n t.raw

/-- Apply dropout while retaining shape, dtype and device policy. -/
def dropout {σ : StaticSpec} (t : Tensor σ) (p : Float) (training : Bool := true) :
    IO (Tensor σ) := do
  let raw ← RawTensor.dropout t.shape t.raw p training
  match t.checkedReplacement? raw with
  | .ok result => pure result
  | .error message => throw (IO.userError s!"Tensor.dropout: {message}")

/-- Batched embedding lookup. Tokens must be Int64 and share the weight's
placement policy; libtorch checks token values against the vocabulary size. -/
def embedding {batch seq vocab embed : UInt64} {d : DType} {dev : DevicePolicy}
    (tokens : Tensor ⟨#[batch, seq], .Int64, dev⟩)
    (weight : Tensor ⟨#[vocab, embed], d, dev⟩) : Tensor ⟨#[batch, seq, embed], d, dev⟩ :=
  checkedResult "embedding" (Tensor.ofRaw (RawTensor.embedding batch seq vocab embed tokens.raw weight.raw))

/-- Vector embedding lookup with statically integral tokens. -/
def embedding1d {seq vocab embed : UInt64} {d : DType} {dev : DevicePolicy}
    (tokens : Tensor ⟨#[seq], .Int64, dev⟩)
    (weight : Tensor ⟨#[vocab, embed], d, dev⟩) : Tensor ⟨#[seq, embed], d, dev⟩ :=
  checkedResult "embedding1d" (Tensor.ofRaw (RawTensor.embedding1d seq vocab embed tokens.raw weight.raw))

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
  checkedResult "eye" (Tensor.ofRaw (RawTensor.eye n))

/-- Identity matrix for a complete specification. The native factory produces
    Float32 on CPU; the value is cast and moved to match the specification. -/
def eyeFor {n : UInt64} (d : DType) (dev : DevicePolicy := .any) : Tensor ⟨#[n, n], d, dev⟩ :=
  let device := match dev with
    | .any => Device.CPU
    | .exact device => device
  let raw := RawTensor.eye n
  let raw := if d == .Unknown "dynamic" then raw
    else RawTensor.castDType #[n, n] raw d.canonicalName
  let raw := RawTensor.toDevice #[n, n] raw device
  checkedResult "eyeFor" (Tensor.ofRaw raw)

/-- Complete QR decomposition: `Q` is m×m orthogonal and `R` is m×n upper
    triangular, retaining dtype and device policy. -/
def qrComplete {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Tensor ⟨#[m, m], d, dev⟩ × Tensor ⟨#[m, n], d, dev⟩ :=
  let (q, r) := RawTensor.qr m n t.raw
  (checkedResult "qr.Q" (Tensor.ofRaw q), checkedResult "qr.R" (Tensor.ofRaw r))

/-- Reduced QR decomposition of a tall or square matrix (`n ≤ m`): `Q` is m×n
    with orthonormal columns and `R` is n×n upper triangular. The shape contract
    is a precondition on the libtorch reduced factorization; wide inputs do not
    typecheck at concrete dimensions. -/
def qrReduced {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[m, n], d, dev⟩) (_tall : n.toNat ≤ m.toNat := by decide) :
    Tensor ⟨#[m, n], d, dev⟩ × Tensor ⟨#[n, n], d, dev⟩ :=
  let (q, r) := RawTensor.qrReduced m n t.raw
  (checkedResult "qr.Q" (Tensor.ofRaw q), checkedResult "qr.R" (Tensor.ofRaw r))

/-- Reduced SVD of a tall or square matrix (`n ≤ m`): `U` is m×n, `S` holds the
    n singular values and `Vh` is n×n. The shape contract is a precondition on
    the libtorch reduced factorization; wide inputs do not typecheck at
    concrete dimensions. -/
def svdReduced {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[m, n], d, dev⟩) (_tall : n.toNat ≤ m.toNat := by decide) :
    Tensor ⟨#[m, n], d, dev⟩ × Tensor ⟨#[n], d, dev⟩ × Tensor ⟨#[n, n], d, dev⟩ :=
  let (u, s, vh) := RawTensor.svd m n t.raw
  (checkedResult "svd.U" (Tensor.ofRaw u), checkedResult "svd.S" (Tensor.ofRaw s),
    checkedResult "svd.Vh" (Tensor.ofRaw vh))

/-- Singular values of a square matrix in descending order. -/
def svdvals {n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[n, n], d, dev⟩) : Tensor ⟨#[n], d, dev⟩ :=
  checkedResult "svdvals" (Tensor.ofRaw (RawTensor.svdvals n n t.raw))

/-- Diagonal matrix built from a vector, retaining dtype and device policy. -/
def diagflat {n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[n], d, dev⟩) : Tensor ⟨#[n, n], d, dev⟩ :=
  checkedResult "diagflat" (Tensor.ofRaw (RawTensor.diagflat n t.raw))

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
  checkedResult "toFloat32" (Tensor.ofRaw (RawTensor.toFloat32 t.shape t.raw))

def toBFloat16 {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withDType .BFloat16) :=
  checkedResult "toBFloat16" (Tensor.ofRaw (RawTensor.toBFloat16 t.shape t.raw))

def toInt64 {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withDType .Int64) :=
  checkedResult "toInt64" (Tensor.ofRaw (RawTensor.toInt64 t.shape t.raw))

/-! ## Elementwise binary (same shape and device, witnessed promotion)

The result dtype is an output parameter of `DType.Promotes`. Its proof ties
that dtype to a successful `promote?` result; unsupported pairs have no
instance. This witnesses dtype inference, not backend kernel availability.
-/

def add {s : Shape} {l r result : DType} {dev : DevicePolicy}
    [DType.Promotes l r result]
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, result, dev⟩ :=
  checkedResult "add" (Tensor.ofRaw (RawTensor.add a.shape a.raw b.raw))

def sub {s : Shape} {l r result : DType} {dev : DevicePolicy}
    [DType.Promotes l r result] [DType.NonBool l] [DType.NonBool r]
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, result, dev⟩ :=
  checkedResult "sub" (Tensor.ofRaw (RawTensor.sub a.shape a.raw b.raw))

def mul {s : Shape} {l r result : DType} {dev : DevicePolicy}
    [DType.Promotes l r result]
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, result, dev⟩ :=
  checkedResult "mul" (Tensor.ofRaw (RawTensor.mul a.shape a.raw b.raw))

def div {s : Shape} {l r result : DType} {dev : DevicePolicy}
    [DType.Promotes l r result]
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, result.atLeastFloat, dev⟩ :=
  checkedResult "div" (Tensor.ofRaw (RawTensor.div a.shape a.raw b.raw))

instance {s : Shape} {l r result : DType} {dev : DevicePolicy}
    [DType.Promotes l r result] :
    HAdd (Tensor ⟨s, l, dev⟩) (Tensor ⟨s, r, dev⟩) (Tensor ⟨s, result, dev⟩) := ⟨add⟩

instance {s : Shape} {l r result : DType} {dev : DevicePolicy}
    [DType.Promotes l r result] [DType.NonBool l] [DType.NonBool r] :
    HSub (Tensor ⟨s, l, dev⟩) (Tensor ⟨s, r, dev⟩) (Tensor ⟨s, result, dev⟩) := ⟨sub⟩

instance {s : Shape} {l r result : DType} {dev : DevicePolicy}
    [DType.Promotes l r result] :
    HMul (Tensor ⟨s, l, dev⟩) (Tensor ⟨s, r, dev⟩) (Tensor ⟨s, result, dev⟩) := ⟨mul⟩

instance {s : Shape} {l r result : DType} {dev : DevicePolicy}
    [DType.Promotes l r result] :
    HDiv (Tensor ⟨s, l, dev⟩) (Tensor ⟨s, r, dev⟩)
      (Tensor ⟨s, result.atLeastFloat, dev⟩) := ⟨div⟩

/-- Generic homogeneous arithmetic must also carry promotion evidence.
Runtime-typed parameter trees use the explicit `Preserving` adapters. -/
instance {σ : StaticSpec} [DType.Promotes σ.dtype σ.dtype σ.dtype] : Add (Tensor σ) :=
  ⟨add⟩

instance {σ : StaticSpec} [DType.Promotes σ.dtype σ.dtype σ.dtype]
    [DType.NonBool σ.dtype] : Sub (Tensor σ) := ⟨sub⟩

instance {σ : StaticSpec} [DType.Promotes σ.dtype σ.dtype σ.dtype] : Mul (Tensor σ) :=
  ⟨mul⟩

/-! ## Scalar arithmetic (host Float uses the Float32 default category) -/

def addScalar {σ : StaticSpec} {result : DType}
    [DType.ScalarPromotes σ.dtype .Float32 result] (t : Tensor σ) (value : Float) :
    Tensor (σ.withDType result) :=
  checkedResult "addScalar" (Tensor.ofRaw (RawTensor.addScalar t.shape t.raw value))

def subScalar {σ : StaticSpec} {result : DType}
    [DType.ScalarPromotes σ.dtype .Float32 result] [DType.NonBool σ.dtype]
    (t : Tensor σ) (value : Float) : Tensor (σ.withDType result) :=
  checkedResult "subScalar" (Tensor.ofRaw (RawTensor.subScalar t.shape t.raw value))

def mulScalar {σ : StaticSpec} {result : DType}
    [DType.ScalarPromotes σ.dtype .Float32 result] (t : Tensor σ) (value : Float) :
    Tensor (σ.withDType result) :=
  checkedResult "mulScalar" (Tensor.ofRaw (RawTensor.mulScalar t.shape t.raw value))

def divScalar {σ : StaticSpec} {result : DType}
    [DType.ScalarPromotes σ.dtype .Float32 result] (t : Tensor σ) (value : Float) :
    Tensor (σ.withDType result) :=
  checkedResult "divScalar" (Tensor.ofRaw (RawTensor.divScalar t.shape t.raw value))

instance {σ : StaticSpec} {result : DType} [DType.ScalarPromotes σ.dtype .Float32 result] :
    HAdd (Tensor σ) Float (Tensor (σ.withDType result)) := ⟨addScalar⟩

instance {σ : StaticSpec} {result : DType} [DType.ScalarPromotes σ.dtype .Float32 result]
    [DType.NonBool σ.dtype] : HSub (Tensor σ) Float (Tensor (σ.withDType result)) := ⟨subScalar⟩

instance {σ : StaticSpec} {result : DType} [DType.ScalarPromotes σ.dtype .Float32 result] :
    HMul (Tensor σ) Float (Tensor (σ.withDType result)) := ⟨mulScalar⟩

instance {σ : StaticSpec} {result : DType} [DType.ScalarPromotes σ.dtype .Float32 result] :
    HDiv (Tensor σ) Float (Tensor (σ.withDType result)) := ⟨divScalar⟩

/-! ## Elementwise unary -/

def relu {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  t.checkedReplacement (RawTensor.relu t.shape t.raw)

def abs {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withDType σ.dtype.absResult) :=
  checkedResult "abs" (Tensor.ofRaw (RawTensor.abs t.shape t.raw))

/-! ## Float-only unary math

Requires `σ.dtype.isFloating`; the proof is found by `decide` when the
dtype is a literal. Integer tensors must cast explicitly (`.toFloat32`). -/

def exp {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) : Tensor σ :=
  t.checkedReplacement (RawTensor.exp t.shape t.raw)

def sqrt {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) : Tensor σ :=
  t.checkedReplacement (RawTensor.sqrt t.shape t.raw)

def rsqrt {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) : Tensor σ :=
  t.checkedReplacement (RawTensor.rsqrt t.shape t.raw)

def sigmoid {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) : Tensor σ :=
  t.checkedReplacement (RawTensor.sigmoid t.shape t.raw)

def tanh {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) : Tensor σ :=
  t.checkedReplacement (RawTensor.tanh t.shape t.raw)

def gelu {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) : Tensor σ :=
  t.checkedReplacement (RawTensor.gelu t.shape t.raw)

def silu {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) : Tensor σ :=
  t.checkedReplacement (RawTensor.silu t.shape t.raw)

def pow {σ : StaticSpec} (t : Tensor σ) (exponent : Float)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) : Tensor σ :=
  t.checkedReplacement (RawTensor.pow t.shape t.raw exponent)

def softmax {σ : StaticSpec} (t : Tensor σ) (dim : Int32 := -1)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) : Tensor σ :=
  t.checkedReplacement (RawTensor.softmax t.shape t.raw dim)

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
  checkedResult "matmul" (Tensor.ofRaw (RawTensor.matmul s1 s2 a.raw b.raw))

def mm {m k n : UInt64} {d : DType} {dev : DevicePolicy}
    (a : Tensor ⟨#[m, k], d, dev⟩) (b : Tensor ⟨#[k, n], d, dev⟩) :
    Tensor ⟨#[m, n], d, dev⟩ :=
  checkedResult "mm" (Tensor.ofRaw (RawTensor.mm m n k a.raw b.raw))

def linear {m n b : UInt64} {d : DType} {dev : DevicePolicy}
    (x : Tensor ⟨#[b, m], d, dev⟩) (weight : Tensor ⟨#[n, m], d, dev⟩) :
    Tensor ⟨#[b, n], d, dev⟩ :=
  checkedResult "linear" (Tensor.ofRaw (RawTensor.linear m n b x.raw weight.raw))

def linear3d {batch seq inDim outDim : UInt64} {d : DType} {dev : DevicePolicy}
    (x : Tensor ⟨#[batch, seq, inDim], d, dev⟩)
    (weight : Tensor ⟨#[outDim, inDim], d, dev⟩) :
    Tensor ⟨#[batch, seq, outDim], d, dev⟩ :=
  checkedResult "linear3d" (Tensor.ofRaw (RawTensor.linear3d batch seq inDim outDim x.raw weight.raw))

def matmul3d {batch seq k n : UInt64} {d : DType} {dev : DevicePolicy}
    (a : Tensor ⟨#[batch, seq, k], d, dev⟩) (b : Tensor ⟨#[k, n], d, dev⟩) :
    Tensor ⟨#[batch, seq, n], d, dev⟩ :=
  checkedResult "matmul3d" (Tensor.ofRaw (RawTensor.matmul3d batch seq k n a.raw b.raw))

def bmm4d {b h m k n : UInt64} {d : DType} {dev : DevicePolicy}
    (a : Tensor ⟨#[b, h, m, k], d, dev⟩) (x : Tensor ⟨#[b, h, k, n], d, dev⟩) :
    Tensor ⟨#[b, h, m, n], d, dev⟩ :=
  checkedResult "bmm4d" (Tensor.ofRaw (RawTensor.bmm4d b h m k n a.raw x.raw))

/-! ## Broadcasting elementwise ops

The result spec is `σ₁.pointwise σ₂` (NumPy/PyTorch rules, including weak
scalar promotion and scalar placement), and the auto-param proof —
discharged by `rfl` for concrete shapes — guarantees the pair actually
broadcasts, so incompatible shape pairs fail to compile.

At runtime libtorch broadcasts the native handles directly; no shape erasure
or reshaping is required at the raw boundary. -/

/-- Broadcasting elementwise addition (`a +ᵇ b`). -/
def addB {σ₁ σ₂ : StaticSpec} {result : DType}
    [TensorSpec.Promotes σ₁.toTensorSpec σ₂.toTensorSpec result]
    (a : Tensor σ₁) (b : Tensor σ₂)
    (_h : (TensorSpec.broadcastShapes? σ₁.shape σ₂.shape).isSome = true := by rfl) :
    Tensor (σ₁.pointwise σ₂) :=
  checkedResult "add" (Tensor.ofRaw (RawTensor.add a.shape a.raw b.raw))

/-- Broadcasting elementwise subtraction (`a -ᵇ b`). -/
def subB {σ₁ σ₂ : StaticSpec} {result : DType}
    [TensorSpec.Promotes σ₁.toTensorSpec σ₂.toTensorSpec result] [DType.NonBool σ₁.dtype] [DType.NonBool σ₂.dtype]
    (a : Tensor σ₁) (b : Tensor σ₂)
    (_h : (TensorSpec.broadcastShapes? σ₁.shape σ₂.shape).isSome = true := by rfl) :
    Tensor (σ₁.pointwise σ₂) :=
  checkedResult "sub" (Tensor.ofRaw (RawTensor.sub a.shape a.raw b.raw))

/-- Broadcasting elementwise multiplication (`a *ᵇ b`). -/
def mulB {σ₁ σ₂ : StaticSpec} {result : DType}
    [TensorSpec.Promotes σ₁.toTensorSpec σ₂.toTensorSpec result]
    (a : Tensor σ₁) (b : Tensor σ₂)
    (_h : (TensorSpec.broadcastShapes? σ₁.shape σ₂.shape).isSome = true := by rfl) :
    Tensor (σ₁.pointwise σ₂) :=
  checkedResult "mul" (Tensor.ofRaw (RawTensor.mul a.shape a.raw b.raw))

/-- Broadcasting elementwise division (`a /ᵇ b`). -/
def divB {σ₁ σ₂ : StaticSpec} {result : DType}
    [TensorSpec.Promotes σ₁.toTensorSpec σ₂.toTensorSpec result]
    (a : Tensor σ₁) (b : Tensor σ₂)
    (_h : (TensorSpec.broadcastShapes? σ₁.shape σ₂.shape).isSome = true := by rfl) :
    Tensor (σ₁.division σ₂) :=
  checkedResult "div" (Tensor.ofRaw (RawTensor.div a.shape a.raw b.raw))

/-- Asymmetric broadcast: stretch `t` to `target` (a libtorch `expand` view,
    no copy). The fixed-side proof `broadcastsTo s target` rules out targets
    that would themselves be stretched. -/
def broadcastTo {σ : StaticSpec} (t : Tensor σ) (target : Shape)
    (_h : TensorSpec.broadcastsTo σ.shape target := by rfl) :
    Tensor (σ.withShape target) :=
  checkedResult "broadcastTo" (Tensor.ofRaw (RawTensor.expand t.shape t.raw target))

/-- Metadata-checked `expand` view, for code generic over dimensions where the
    `broadcastTo` proof cannot reduce (e.g. `#[batch, outDim]` with symbolic
    dims). libtorch still validates the expansion at runtime. -/
def expand? {σ : StaticSpec} (t : Tensor σ) (target : Shape) :
    Except String (Tensor (σ.withShape target)) :=
  Tensor.ofRaw (RawTensor.expand t.shape t.raw target)

/-- Expand to literal, nonnegative extents. Use `expand?` to handle a native
result-spec mismatch without invoking the panic fallback allocation. -/
def expand {σ : StaticSpec} (t : Tensor σ) (target : Shape) :
    Tensor (σ.withShape target) :=
  checkedResult "expand" (t.expand? target)

/-! ## Reductions -/

def sumAll {σ : StaticSpec} (t : Tensor σ) : Tensor σ.sumSpec :=
  checkedResult "sumAll" (Tensor.ofRaw (RawTensor.sumAll t.shape t.raw))

/-- Maximum over all elements: a scalar retaining dtype and device policy. -/
def maxAll {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withShape #[]) :=
  checkedResult "maxAll" (Tensor.ofRaw (RawTensor.maxAll t.raw))

/-- Reduce one axis using libtorch's accumulation dtype. -/
def sumDim {σ : StaticSpec} (t : Tensor σ) (dim : Nat) (keepdim : Bool := false) :
    Tensor (σ.sumSpec.withShape (reduceShape σ.shape dim keepdim)) :=
  checkedResult "sumDim" (Tensor.ofRaw (RawTensor.sum t.shape t.raw (some #[dim.toUInt64]) keepdim))

def meanAll {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) :
    Tensor (σ.withShape #[]) :=
  checkedResult "meanAll" (Tensor.ofRaw (RawTensor.meanAll t.shape t.raw))

/-! ## Shape ops (dtype/device-preserving) -/

/-- Reshape to literal, nonnegative extents. Invalid element counts and
wrapped signed extents are rejected before calling the native operation. -/
def reshapeChecked? {σ : StaticSpec} (t : Tensor σ) (shape' : Shape) :
    Except String (Tensor (σ.withShape shape')) := do
  if shape'.any (fun dim => dim > 9223372036854775807) then
    throw "reshape requires nonnegative extents representable as signed 64-bit integers"
  let actualNumel := TensorSpec.numelOfShape t.runtimeShape
  let targetNumel := TensorSpec.numelOfShape shape'
  if actualNumel != targetNumel then
    throw s!"reshape requires {actualNumel} elements, requested {targetNumel}"
  let raw := RawTensor.reshape t.shape t.raw shape'
  TensorSpec.checkDType t.raw.dtype raw.dtype
  DevicePolicy.check (.exact t.raw.device) raw.device
  Tensor.ofRaw raw

/-- Reshape without changing the element count. Unlike the legacy raw
    `torch.reshape`, an empty target creates a scalar. -/
def reshape {σ : StaticSpec} (t : Tensor σ) (shape' : Shape)
    (_h : TensorSpec.numelOfShape σ.shape = TensorSpec.numelOfShape shape' := by decide) :
    Tensor (σ.withShape shape') :=
  checkedResult "reshape" (t.reshapeChecked? shape')

/-- Slice an exact number of entries along a valid axis. Bounds are checked in
natural arithmetic so `start + len` cannot wrap or silently truncate. -/
def slice {σ : StaticSpec} (t : Tensor σ) (dim : Nat) (start len : UInt64)
    (_axis : dim < σ.shape.size := by decide)
    (_bounds : start.toNat + len.toNat ≤ (σ.shape[dim]'_axis).toNat := by decide) :
    Tensor (σ.withShape (σ.shape.set! dim len)) :=
  checkedResult "slice" (Tensor.ofRaw (RawTensor.slice t.shape t.raw dim.toUInt64 start len))

def permute {σ : StaticSpec} (t : Tensor σ) (permutation : Array UInt64) :
    Tensor (σ.withShape (TensorSpec.permuteShape σ.shape permutation)) :=
  checkedResult "permute" (Tensor.ofRaw (RawTensor.permute t.shape t.raw permutation))

/-- Transpose along nonnegative axes, returning native validation errors and
checking the result metadata. -/
def transpose? {σ : StaticSpec} (t : Tensor σ) (dim0 dim1 : UInt64) :
    Except String (Tensor (σ.withShape (transposeShape σ.shape dim0.toNat dim1.toNat))) := do
  Tensor.ofRaw (← RawTensor.transpose t.shape t.raw dim0 dim1)

def transpose {σ : StaticSpec} (t : Tensor σ) (dim0 dim1 : UInt64) :
    Tensor (σ.withShape (transposeShape σ.shape dim0.toNat dim1.toNat)) :=
  checkedResult "transpose" (t.transpose? dim0 dim1)

def transpose2d {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Tensor ⟨#[n, m], d, dev⟩ :=
  checkedResult "transpose2d" (Tensor.ofRaw (RawTensor.transpose2d m n t.raw))

def unsqueeze {σ : StaticSpec} (t : Tensor σ) (dim : Nat) :
    Tensor (σ.withShape (unsqueezeShape σ.shape dim)) :=
  checkedResult "unsqueeze" (Tensor.ofRaw (RawTensor.unsqueeze t.shape t.raw dim))

def squeeze {σ : StaticSpec} (t : Tensor σ) (dim : Nat) :
    Tensor (σ.withShape (squeezeShape σ.shape dim)) :=
  checkedResult "squeeze" (Tensor.ofRaw (RawTensor.squeeze t.shape t.raw dim))

def cat {s1 s2 : Shape} {d : DType} {dev : DevicePolicy}
    (t1 : Tensor ⟨s1, d, dev⟩) (t2 : Tensor ⟨s2, d, dev⟩) (dim : Nat) :
    Tensor ⟨TensorSpec.catShape s1 s2 dim, d, dev⟩ :=
  checkedResult "cat" (Tensor.ofRaw (RawTensor.cat s1 s2 t1.raw t2.raw dim))

/-- Stack a runtime list of equal vectors along a new leading axis; the result
    shape records the actual list length. -/
def stack1d {cols : UInt64} {d : DType} {dev : DevicePolicy}
    (tensors : Array (Tensor ⟨#[cols], d, dev⟩)) :
    Tensor ⟨#[tensors.size.toUInt64, cols], d, dev⟩ :=
  checkedResult "stack1d" (Tensor.ofRaw (RawTensor.stack1d tensors.size.toUInt64 cols (tensors.map Tensor.toTensor) 0))

/-- Reshape to a runtime target, validating result metadata against the claimed
    spec and panicking on mismatch. For dynamically computed shapes where the
    static `reshape` numel proof cannot reduce; the runtime check keeps the
    conversion honest. -/
def reshapeChecked {σ : StaticSpec} (t : Tensor σ) (shape' : Shape) : Tensor (σ.withShape shape') :=
  checkedResult "reshapeChecked" (t.reshapeChecked? shape')

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
  checkedResult "fromFloatArray" (Tensor.ofRaw (RawTensor.fromFloatArray arr))

/-- 1-D Int64 tensor from a host array; the result shape records the actual
    array length. -/
def fromInt64Array (arr : Array Int64) : Tensor ⟨#[arr.size.toUInt64], .Int64, .any⟩ :=
  checkedResult "fromInt64Array" (Tensor.ofRaw (RawTensor.fromInt64Array arr))

/-- Argmax along `dim`: Int64 indices with that dimension removed. -/
def argmax {s : Shape} {d : DType} {dev : DevicePolicy} (t : Tensor ⟨s, d, dev⟩) (dim : UInt64) :
    Tensor ⟨reduceShape s dim.toNat false, .Int64, dev⟩ :=
  checkedResult "argmax" (Tensor.ofRaw (RawTensor.argmax s t.raw dim))

/-- Top-k filter over logits (entries outside the top k become -inf), spec-preserving. -/
def topKFilter {σ : StaticSpec} (logits : Tensor σ) (k : UInt64) : Tensor σ :=
  logits.checkedReplacement (RawTensor.topKFilter σ.shape logits.raw k)

/-- Top-p (nucleus) filter over logits, spec-preserving. -/
def topPFilter {σ : StaticSpec} (logits : Tensor σ) (p : Float) : Tensor σ :=
  logits.checkedReplacement (RawTensor.topPFilter σ.shape logits.raw p)

/-- Elementwise equality with a scalar: Bool tensor of the same shape. -/
def eqScalar {s : Shape} {d : DType} {dev : DevicePolicy} (t : Tensor ⟨s, d, dev⟩) (scalar : Int64) :
    Tensor ⟨s, .Bool, dev⟩ :=
  checkedResult "eqScalar" (Tensor.ofRaw (RawTensor.eqScalar s t.raw scalar))

/-- Select entries of a 1-D tensor by Int64 indices. -/
def indexSelect {n k : UInt64} {d : DType} {dev : DevicePolicy}
    (data : Tensor ⟨#[n], d, dev⟩) (indices : Tensor ⟨#[k], .Int64, dev⟩) :
    Tensor ⟨#[k], d, dev⟩ :=
  checkedResult "indexSelect" (Tensor.ofRaw (RawTensor.indexSelect n k data.raw 0 indices.raw))

/-- Per-element cross entropy (no reduction) over a 2-D logits matrix with Int64 targets. -/
def crossEntropyNone {n c : UInt64} {d : DType} {dev : DevicePolicy}
    (logits : Tensor ⟨#[n, c], d, dev⟩) (targets : Tensor ⟨#[n], .Int64, dev⟩) :
    Tensor ⟨#[n], d, dev⟩ :=
  checkedResult "crossEntropyNone" (Tensor.ofRaw (RawTensor.crossEntropyNone n c logits.raw targets.raw))

/-- Extract an Int64 scalar. -/
def itemInt {d : DType} {dev : DevicePolicy} (t : Tensor ⟨#[], d, dev⟩) : Int64 :=
  RawTensor.itemInt #[] t.raw

/-- Select between two same-spec tensors elementwise by a Bool condition. -/
def whereSelect {s : Shape} {d : DType} {dev : DevicePolicy}
    (condition : Tensor ⟨s, .Bool, dev⟩) (x y : Tensor ⟨s, d, dev⟩) :
    Tensor ⟨s, d, dev⟩ :=
  x.checkedReplacement (RawTensor.whereSelect s condition.raw x.raw y.raw)

/-- Elementwise logical negation of a Bool tensor. -/
def logicalNot {s : Shape} {dev : DevicePolicy} (t : Tensor ⟨s, .Bool, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  t.checkedReplacement (RawTensor.logicalNot s t.raw)

/-- Elementwise logical conjunction of two Bool tensors. -/
def logicalAnd {s : Shape} {dev : DevicePolicy} (a b : Tensor ⟨s, .Bool, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  a.checkedReplacement (RawTensor.logicalAnd s a.raw b.raw)

/-- Elementwise logical disjunction of two Bool tensors. -/
def logicalOr {s : Shape} {dev : DevicePolicy} (a b : Tensor ⟨s, .Bool, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  logicalNot (logicalAnd (logicalNot a) (logicalNot b))

/-- Cross-entropy loss over a 2-D logits matrix with Int64 targets. -/
def crossEntropy {n c : UInt64} {d : DType} {dev : DevicePolicy}
    (logits : Tensor ⟨#[n, c], d, dev⟩) (targets : Tensor ⟨#[n], .Int64, dev⟩) :
    Tensor ⟨#[], d, dev⟩ :=
  checkedResult "crossEntropy" (Tensor.ofRaw (RawTensor.crossEntropy n c logits.raw targets.raw))

/-- Draw `numSamples` Int64 indices per row of a probability matrix. -/
def multinomial {n v : UInt64} {d : DType} {dev : DevicePolicy}
    (probs : Tensor ⟨#[n, v], d, dev⟩) (numSamples : UInt64) (replacement : Bool := false) :
    IO (Tensor ⟨#[n, numSamples], .Int64, dev⟩) := do
  let raw ← RawTensor.multinomial #[n, v] probs.raw numSamples replacement
  match Tensor.ofRaw raw with
  | .ok result => pure result
  | .error message => throw (IO.userError s!"Tensor.multinomial: {message}")

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
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) : Tensor σ :=
  t.checkedReplacement (RawTensor.logSoftmax t.raw dim)

/-- Backward pass from a scalar loss (gradient = 1.0). -/
def backwardLoss {d : DType} {dev : DevicePolicy} (loss : Tensor ⟨#[], d, dev⟩) : IO Unit :=
  RawTensor.backwardLoss loss.raw

/-! ## Autograd and fused layer primitives -/

def detach {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  t.checkedReplacement (RawTensor.detach t.shape t.raw)

/-- Read a detached gradient snapshot that remains stable across later backward,
    clipping, and zeroing operations. Missing gradients produce zeros. -/
def gradOf {σ : StaticSpec} (t : Tensor σ) : IO (Tensor σ) := do
  let raw ← RawTensor.gradOf t.shape t.raw
  match t.checkedReplacement? raw with
  | .ok gradient => pure gradient
  | .error message => throw (IO.userError s!"Tensor.gradOf: {message}")

/-- Clear the parameter's accumulated gradient in place. -/
def zeroGrad {σ : StaticSpec} (t : Tensor σ) : IO Unit :=
  RawTensor.zeroGrad t.shape t.raw

/-! Operations for generic parameter trees. Unlike the promoting arithmetic
surface, these require the native result to retain the input specification.
In particular, floating scalar arithmetic on an integer parameter fails the
dtype check instead of returning an incorrectly indexed tensor. -/
namespace Preserving

/-- Runtime preflight for algorithms whose tensor-tree leaf dtype is abstract.
The actual operand metadata must have a defined, dtype-preserving promotion.
This is deliberately an explicit checked path, not a promotion instance. -/
private def checkBinaryDType (operation : String) (lhs rhs : RawTensor) : Except String Unit := do
  let l := lhs.dtype
  let r := rhs.dtype
  if operation == "sub" && (l.isBool || r.isBool) then
    throw "Subtraction does not support Bool operands"
  match DType.promote? l r with
  | none => throw s!"Unsupported dtype promotion: {l} and {r}"
  | some result => DType.expectedEq l (if operation == "div" then result.divideResult else result)

/-- Checked dtype-preserving add for runtime-typed parameter trees. -/
def add? {σ : StaticSpec} (lhs rhs : Tensor σ) : Except String (Tensor σ) := do
  checkBinaryDType "add" lhs.raw rhs.raw
  lhs.checkedReplacement? (RawTensor.add lhs.shape lhs.raw rhs.raw)

/-- Dtype-preserving add; use `add?` for recoverable promotion rejection. -/
def add {σ : StaticSpec} (lhs rhs : Tensor σ) : Tensor σ :=
  match add? lhs rhs with
  | .ok result => result
  | .error message =>
      letI : Inhabited (Tensor σ) := ⟨lhs⟩
      panic! s!"Tensor.Preserving.add: {message}"

/-- Checked dtype-preserving sub for runtime-typed parameter trees. -/
def sub? {σ : StaticSpec} (lhs rhs : Tensor σ) : Except String (Tensor σ) := do
  checkBinaryDType "sub" lhs.raw rhs.raw
  lhs.checkedReplacement? (RawTensor.sub lhs.shape lhs.raw rhs.raw)

/-- Dtype-preserving sub; use `sub?` for recoverable promotion rejection. -/
def sub {σ : StaticSpec} (lhs rhs : Tensor σ) : Tensor σ :=
  match sub? lhs rhs with
  | .ok result => result
  | .error message =>
      letI : Inhabited (Tensor σ) := ⟨lhs⟩
      panic! s!"Tensor.Preserving.sub: {message}"

/-- Checked dtype-preserving mul for runtime-typed parameter trees. -/
def mul? {σ : StaticSpec} (lhs rhs : Tensor σ) : Except String (Tensor σ) := do
  checkBinaryDType "mul" lhs.raw rhs.raw
  lhs.checkedReplacement? (RawTensor.mul lhs.shape lhs.raw rhs.raw)

/-- Dtype-preserving mul; use `mul?` for recoverable promotion rejection. -/
def mul {σ : StaticSpec} (lhs rhs : Tensor σ) : Tensor σ :=
  match mul? lhs rhs with
  | .ok result => result
  | .error message =>
      letI : Inhabited (Tensor σ) := ⟨lhs⟩
      panic! s!"Tensor.Preserving.mul: {message}"



/-- Absolute value for generic algorithms that require dtype preservation.
Complex absolute value is real; use `Tensor.abs` to expose that result type. -/
def abs? {σ : StaticSpec} (t : Tensor σ) : Except String (Tensor σ) := do
  DType.expectedEq t.raw.dtype t.raw.dtype.absResult
  t.checkedReplacement? (RawTensor.abs t.shape t.raw)

def abs {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  match abs? t with
  | .ok result => result
  | .error message =>
      letI : Inhabited (Tensor σ) := ⟨t⟩
      panic! s!"Tensor.Preserving.abs: {message}"

/-- Clamp while checking that floating bounds did not promote the dtype. -/
def clampFloat {σ : StaticSpec} (t : Tensor σ) (lo hi : Float) : Tensor σ :=
  t.checkedReplacement (RawTensor.clampFloat t.shape t.raw lo hi)

/-- Reduction for generic algorithms that require the actual input dtype.
Integer accumulation cannot hide behind a dynamic dtype index. -/
def sumDim? {σ : StaticSpec} (t : Tensor σ) (dim : Nat) (keepdim : Bool := false) :
    Except String (Tensor (σ.withShape (reduceShape σ.shape dim keepdim))) := do
  let raw := RawTensor.sum t.shape t.raw (some #[dim.toUInt64]) keepdim
  TensorSpec.checkDType t.raw.dtype raw.dtype
  DevicePolicy.check (.exact t.raw.device) raw.device
  Tensor.ofRaw raw

/-- Dtype-preserving reduction; use `sumDim?` for recoverable rejection. -/
def sumDim {σ : StaticSpec} (t : Tensor σ) (dim : Nat) (keepdim : Bool := false) :
    Tensor (σ.withShape (reduceShape σ.shape dim keepdim)) :=
  match sumDim? t dim keepdim with
  | .ok result => result
  | .error message => panic! s!"Tensor sumDim: {message}"

/-- Host floating scalars use PyTorch's weak Float32 category. Validate the
actual result dtype before invoking a generic preserving operation. -/
private def checkScalarDType (tensor : RawTensor) : Except String Unit := do
  let dtype := tensor.dtype
  match DType.promoteScalar? dtype .Float32 with
  | none => throw s!"Unsupported scalar promotion for {dtype}"
  | some result => DType.expectedEq dtype result

def mulScalar? {σ : StaticSpec} (t : Tensor σ) (value : Float) : Except String (Tensor σ) := do
  checkScalarDType t.raw
  t.checkedReplacement? (RawTensor.mulScalar t.shape t.raw value)

def mulScalar {σ : StaticSpec} (t : Tensor σ) (value : Float) : Tensor σ :=
  match mulScalar? t value with
  | .ok result => result
  | .error message =>
      letI : Inhabited (Tensor σ) := ⟨t⟩
      panic! s!"Tensor.Preserving.mulScalar: {message}"

def divScalar? {σ : StaticSpec} (t : Tensor σ) (value : Float) : Except String (Tensor σ) := do
  checkScalarDType t.raw
  t.checkedReplacement? (RawTensor.divScalar t.shape t.raw value)

def divScalar {σ : StaticSpec} (t : Tensor σ) (value : Float) : Tensor σ :=
  match divScalar? t value with
  | .ok result => result
  | .error message =>
      letI : Inhabited (Tensor σ) := ⟨t⟩
      panic! s!"Tensor.Preserving.divScalar: {message}"

def addScalar? {σ : StaticSpec} (t : Tensor σ) (value : Float) : Except String (Tensor σ) := do
  checkScalarDType t.raw
  t.checkedReplacement? (RawTensor.addScalar t.shape t.raw value)

def addScalar {σ : StaticSpec} (t : Tensor σ) (value : Float) : Tensor σ :=
  match addScalar? t value with
  | .ok result => result
  | .error message =>
      letI : Inhabited (Tensor σ) := ⟨t⟩
      panic! s!"Tensor.Preserving.addScalar: {message}"

def sqrt {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  t.checkedReplacement (RawTensor.sqrt t.shape t.raw)

def atan {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  t.checkedReplacement (RawTensor.atan t.shape t.raw)

def div? {σ : StaticSpec} (lhs rhs : Tensor σ) : Except String (Tensor σ) := do
  checkBinaryDType "div" lhs.raw rhs.raw
  lhs.checkedReplacement? (RawTensor.div lhs.shape lhs.raw rhs.raw)

def div {σ : StaticSpec} (lhs rhs : Tensor σ) : Tensor σ :=
  match div? lhs rhs with
  | .ok result => result
  | .error message =>
      letI : Inhabited (Tensor σ) := ⟨lhs⟩
      panic! s!"Tensor.Preserving.div: {message}"

end Preserving

/-- Change the autograd flag without changing other aliases. Creating mutable
    gradient identity is effectful, even when the tensor values are equal. -/
def setRequiresGrad {σ : StaticSpec} (t : Tensor σ) (requiresGrad : Bool) :
    IO (Tensor σ) := do
  let dtype := t.raw.dtype
  let device := t.raw.device
  let raw ← RawTensor.setRequiresGrad σ.shape t.raw requiresGrad
  pure (checkedUpdate dtype device raw)

/-- Create an independent trainable leaf. Detaching first ensures fresh gradient
    identity even when the source is already a parameter. -/
def parameter {σ : StaticSpec} (t : Tensor σ) : IO (Tensor σ) :=
  setRequiresGrad (detach t) true

/-- Fused affine for 3D input: `x @ Wᵀ + b`. -/
def affine3d {batch seq inDim outDim : UInt64} {d : DType} {dev : DevicePolicy}
    (x : Tensor ⟨#[batch, seq, inDim], d, dev⟩)
    (weight : Tensor ⟨#[outDim, inDim], d, dev⟩)
    (bias : Tensor ⟨#[outDim], d, dev⟩) :
    Tensor ⟨#[batch, seq, outDim], d, dev⟩ :=
  checkedResult "affine3d" (Tensor.ofRaw (RawTensor.affine3d batch seq inDim outDim x.raw weight.raw bias.raw))

/-- Fused layer normalization over the last dimension, preserving the spec. -/
def layerNorm {batch seq n : UInt64} {d : DType} {dev : DevicePolicy}
    (x : Tensor ⟨#[batch, seq, n], d, dev⟩)
    (weight : Tensor ⟨#[n], d, dev⟩)
    (bias : Tensor ⟨#[n], d, dev⟩) (eps : Float := 1e-5) :
    Tensor ⟨#[batch, seq, n], d, dev⟩ :=
  x.checkedReplacement (RawTensor.layerNorm batch seq n x.raw weight.raw bias.raw eps)

/-- Weighted RMS normalization over the last dimension.

The runtime computes in Float32 before multiplying the weight. A non-scalar
Float64 weight promotes the result; a scalar weight follows weak scalar
promotion. The weight must not expand the input shape. -/
def rmsNormWeighted? {σ : StaticSpec} {w : Shape} {result : DType}
    [TensorSpec.Promotes ⟨σ.shape, .Float32⟩ ⟨w, σ.dtype⟩ result]
    (x : Tensor σ) (weight : Tensor (σ.withShape w)) (eps : Float := 1e-6)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) :
    Except String (Tensor (σ.withDType
      (TensorSpec.pointwiseDType ⟨σ.shape, .Float32⟩ ⟨w, σ.dtype⟩))) :=
  Tensor.ofRaw (RawTensor.rmsNormWeighted σ.shape w x.raw weight.raw eps)

/-- Weighted RMS normalization with metadata validation. Cast the result back
explicitly for HF-style dtype restoration; use `rmsNormWeighted?` to handle
a weight that broadcasts beyond the input shape. -/
def rmsNormWeighted {σ : StaticSpec} {w : Shape} {result : DType}
    [TensorSpec.Promotes ⟨σ.shape, .Float32⟩ ⟨w, σ.dtype⟩ result]
    (x : Tensor σ) (weight : Tensor (σ.withShape w)) (eps : Float := 1e-6)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide) :
    Tensor (σ.withDType (TensorSpec.pointwiseDType ⟨σ.shape, .Float32⟩ ⟨w, σ.dtype⟩)) :=
  checkedResult "rmsNormWeighted" (x.rmsNormWeighted? weight eps _float)

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

namespace torch.Tensor

/-- One-dimensional convolution with matching input channels, dtype and device.
The spatial result is computed using natural arithmetic and checked at the ABI
boundary, so overflowing UInt64 arithmetic cannot claim an incorrect shape. -/
def conv1d {batch inChannels outChannels length kernel : UInt64}
    {d : DType} {dev : DevicePolicy}
    (input : Tensor ⟨#[batch, inChannels, length], d, dev⟩)
    (weight : Tensor ⟨#[outChannels, inChannels, kernel], d, dev⟩)
    (stride : UInt64 := 1) (padding : UInt64 := 0) (dilation : UInt64 := 1) :
    Tensor ⟨#[batch, outChannels,
      ((length.toNat + 2 * padding.toNat - (dilation.toNat * (kernel.toNat - 1) + 1)) /
        stride.toNat + 1).toUInt64], d, dev⟩ :=
  match Tensor.ofRaw (RawTensor.conv1d input.shape weight.shape input.raw weight.raw
      stride padding dilation) with
  | .ok result => result
  | .error message => panic! s!"Tensor.conv1d: {message}"

/-- Copy tensor values to a host Float array through the native conversion. -/
def toFloatArray {σ : StaticSpec} (t : Tensor σ) : IO (Array Float) :=
  RawTensor.toFloatArray t.raw

/-- Whether any Boolean entry is true. -/
def any {s : Shape} {dev : DevicePolicy} (t : Tensor ⟨s, .Bool, dev⟩) : Bool :=
  RawTensor.any s t.raw
/-- Copy integral tensor values to host UInt64 values. -/
def toUInt64Array {s : Shape} {dev : DevicePolicy}
    (t : Tensor ⟨s, .Int64, dev⟩) : IO (Array UInt64) :=
  RawTensor.toUInt64Array t.raw
/-- Uniform integral samples in the half-open interval `[low, high)`. -/
def randint (low high : Int64) (shape : Shape) (device : Device := .CPU) :
    IO (Tensor { shape := shape, dtype := .Int64 }) := do
  let raw ← RawTensor.randint low high shape false device
  match Tensor.ofRaw raw with
  | .ok result => pure result
  | .error message => throw (IO.userError s!"Tensor.randint: {message}")

namespace Promoting

/-- Runtime-selected promotion tied to the actual input dtypes. The witness is
erased; the tensor retains the selected result dtype as an index. `resultDType`
accounts for operations such as true division that transform the promotion. -/
structure Promoted (lhs rhs : DType) (shape : Shape) (device : DevicePolicy)
    (resultDType : DType → DType := id) where
  dtype : DType
  witness : DType.Promotes lhs rhs dtype
  tensor : Tensor ⟨shape, resultDType dtype, device⟩

/-- Forget the selected dtype index without converting the result's values. -/
def Promoted.toDynamic? {lhs rhs : DType} {shape : Shape} {device : DevicePolicy}
    {resultDType : DType → DType} (result : Promoted lhs rhs shape device resultDType) :
    Except String (Tensor ⟨shape, .Unknown "dynamic", device⟩) :=
  Tensor.ofRaw result.tensor.raw

private def binary? {s : Shape} {l r : DType} {dev : DevicePolicy}
    (operation : String) (native : Shape → RawTensor → RawTensor → RawTensor)
    (resultDType : DType → DType)
    (left : Tensor ⟨s, l, dev⟩) (right : Tensor ⟨s, r, dev⟩) :
    Except String (Promoted left.actualSpec.dtype right.actualSpec.dtype s dev resultDType) := do
  let lhs := left.actualSpec.dtype
  let rhs := right.actualSpec.dtype
  if operation == "sub" && (lhs.isBool || rhs.isBool) then
    throw "Subtraction does not support Bool operands"
  match h : DType.promote? lhs rhs with
  | none => throw s!"Unsupported dtype promotion: {lhs} and {rhs}"
  | some dtype =>
    let tensor ← Tensor.ofRaw (σ := ⟨s, resultDType dtype, dev⟩)
      (native s left.raw right.raw)
    pure { dtype, witness := ⟨h⟩, tensor }

/-- Promote using actual dtypes and return the checked, witnessed result.
Promotion rejection is recoverable; backend kernel availability remains a
precondition of the underlying pure arithmetic operation. -/
def add? {s : Shape} {l r : DType} {dev : DevicePolicy}
    (left : Tensor ⟨s, l, dev⟩) (right : Tensor ⟨s, r, dev⟩) :
    Except String (Promoted left.actualSpec.dtype right.actualSpec.dtype s dev) :=
  binary? "add" RawTensor.add id left right

/-- Checked runtime promotion for subtraction, rejecting Bool operands first. -/
def sub? {s : Shape} {l r : DType} {dev : DevicePolicy}
    (left : Tensor ⟨s, l, dev⟩) (right : Tensor ⟨s, r, dev⟩) :
    Except String (Promoted left.actualSpec.dtype right.actualSpec.dtype s dev) :=
  binary? "sub" RawTensor.sub id left right

/-- Checked runtime promotion for multiplication. -/
def mul? {s : Shape} {l r : DType} {dev : DevicePolicy}
    (left : Tensor ⟨s, l, dev⟩) (right : Tensor ⟨s, r, dev⟩) :
    Except String (Promoted left.actualSpec.dtype right.actualSpec.dtype s dev) :=
  binary? "mul" RawTensor.mul id left right

/-- Checked runtime promotion for true division. The package records both the
ordinary promotion witness and its transformed result dtype. -/
def div? {s : Shape} {l r : DType} {dev : DevicePolicy}
    (left : Tensor ⟨s, l, dev⟩) (right : Tensor ⟨s, r, dev⟩) :
    Except String (Promoted left.actualSpec.dtype right.actualSpec.dtype s dev DType.divideResult) :=
  binary? "div" RawTensor.div DType.divideResult left right

/-- Runtime-promoting arithmetic with a dynamic result index. No operand or
result is cast back to an input dtype. Use `add?` to retain the promotion witness. -/
def add {s : Shape} {l r : DType} {dev : DevicePolicy}
    (left : Tensor ⟨s, l, dev⟩) (right : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, .Unknown "dynamic", dev⟩ :=
  checkedResult "Promoting.add" do (← add? left right).toDynamic?

def sub {s : Shape} {l r : DType} {dev : DevicePolicy}
    (left : Tensor ⟨s, l, dev⟩) (right : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, .Unknown "dynamic", dev⟩ :=
  checkedResult "Promoting.sub" do (← sub? left right).toDynamic?

def mul {s : Shape} {l r : DType} {dev : DevicePolicy}
    (left : Tensor ⟨s, l, dev⟩) (right : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, .Unknown "dynamic", dev⟩ :=
  checkedResult "Promoting.mul" do (← mul? left right).toDynamic?

def div {s : Shape} {l r : DType} {dev : DevicePolicy}
    (left : Tensor ⟨s, l, dev⟩) (right : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, .Unknown "dynamic", dev⟩ :=
  checkedResult "Promoting.div" do (← div? left right).toDynamic?

/-- Promotion by a host Float, using PyTorch's default Float32 scalar category.
The input dtype parameter is the actual metadata of the supplied tensor. -/
structure ScalarPromoted (input : DType) (shape : Shape) (device : DevicePolicy) where
  dtype : DType
  witness : DType.ScalarPromotes input .Float32 dtype
  tensor : Tensor ⟨shape, dtype, device⟩

def ScalarPromoted.toDynamic? {input : DType} {shape : Shape} {device : DevicePolicy}
    (result : ScalarPromoted input shape device) :
    Except String (Tensor ⟨shape, .Unknown "dynamic", device⟩) :=
  Tensor.ofRaw result.tensor.raw

private def scalar? {σ : StaticSpec} (operation : String)
    (native : Shape → RawTensor → Float → RawTensor) (input : Tensor σ) (value : Float) :
    Except String (ScalarPromoted input.actualSpec.dtype σ.shape σ.device) := do
  let dtype := input.actualSpec.dtype
  if operation == "subScalar" && dtype.isBool then
    throw "Subtraction does not support Bool operands"
  match h : DType.promoteScalar? dtype .Float32 with
  | none => throw s!"Unsupported scalar dtype promotion: {dtype} and Float32"
  | some result =>
    let tensor ← Tensor.ofRaw (σ := ⟨σ.shape, result, σ.device⟩)
      (native σ.shape input.raw value)
    pure { dtype := result, witness := ⟨h⟩, tensor }

def addScalar? {σ : StaticSpec} (input : Tensor σ) (value : Float) :
    Except String (ScalarPromoted input.actualSpec.dtype σ.shape σ.device) :=
  scalar? "addScalar" RawTensor.addScalar input value

def subScalar? {σ : StaticSpec} (input : Tensor σ) (value : Float) :
    Except String (ScalarPromoted input.actualSpec.dtype σ.shape σ.device) :=
  scalar? "subScalar" RawTensor.subScalar input value

def mulScalar? {σ : StaticSpec} (input : Tensor σ) (value : Float) :
    Except String (ScalarPromoted input.actualSpec.dtype σ.shape σ.device) :=
  scalar? "mulScalar" RawTensor.mulScalar input value

def divScalar? {σ : StaticSpec} (input : Tensor σ) (value : Float) :
    Except String (ScalarPromoted input.actualSpec.dtype σ.shape σ.device) :=
  scalar? "divScalar" RawTensor.divScalar input value

def addScalar {σ : StaticSpec} (input : Tensor σ) (value : Float) :
    Tensor ⟨σ.shape, .Unknown "dynamic", σ.device⟩ :=
  checkedResult "Promoting.addScalar" do (← addScalar? input value).toDynamic?

def subScalar {σ : StaticSpec} (input : Tensor σ) (value : Float) :
    Tensor ⟨σ.shape, .Unknown "dynamic", σ.device⟩ :=
  checkedResult "Promoting.subScalar" do (← subScalar? input value).toDynamic?

def mulScalar {σ : StaticSpec} (input : Tensor σ) (value : Float) :
    Tensor ⟨σ.shape, .Unknown "dynamic", σ.device⟩ :=
  checkedResult "Promoting.mulScalar" do (← mulScalar? input value).toDynamic?

def divScalar {σ : StaticSpec} (input : Tensor σ) (value : Float) :
    Tensor ⟨σ.shape, .Unknown "dynamic", σ.device⟩ :=
  checkedResult "Promoting.divScalar" do (← divScalar? input value).toDynamic?

end Promoting

end torch.Tensor
