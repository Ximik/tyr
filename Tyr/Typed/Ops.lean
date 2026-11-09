/-
  Tyr/Typed/Ops.lean

  Typed operation surface over the untyped `torch.*` FFI, indexed by
  `StaticSpec` (shape, dtype, device).

  Native operation results pass through checked construction before acquiring
  a static specification. Result specs
  follow PyTorch's semantics as encoded in the `StaticSpec`/`DType` algebra;
  runtime metadata checks and the parity tests in `Tests/TestTyped.lean`
  validate those claims.

  Conventions:
  - Ops that preserve the whole spec are `Tensor σ → Tensor σ`.
  - Ops that need spec components use anonymous-constructor binders
    (`Tensor ⟨s, d, dev⟩`); the device variable flows through unchanged.
  - Elementwise binary ops promote dtypes; true division and float-scalar
    arithmetic float integer results (`atLeastFloat`).
  - Float-only math requires an `isFloating` proof, discharged by `decide`
    for literal dtypes; integer tensors must be cast explicitly first.
  - Matmul-family ops require both operands to share dtype AND device — PyTorch throws on mixed operands; here it cannot compile.
    Generic `matmul` also requires `TensorSpec.matmulShape?` to accept
    the shape pair (auto-param, `rfl` for concrete shapes), so inner-dim
    mismatches fail to compile instead of throwing inside libtorch.
-/
import Tyr.RawTensor
import Tyr.ConstantSupport
import Tyr.Shape.Allocation
import Tyr.Typed.Tensor
import Tyr.Typed.Domain

namespace torch

@[simp] theorem StaticSpec.sumSpec_of_realFloating {σ : StaticSpec}
    [DType.RealFloating σ.dtype] : σ.sumSpec = σ.withShape #[] := by
  simp [StaticSpec.sumSpec, StaticSpec.withShape]

namespace Tensor

/-! ## Creation

The runtime defaults are Float32 for float factories and Int64 for
`fullInt`; the indices state exactly that. Factories with a device argument
index their result by that device, and host factories return CPU tensors. -/

def zeros (shape : Shape) (requiresGrad : Bool := false)
    (device : Device := Device.CPU) (noGrad : requiresGrad = false := by exact of_decide_eq_true rfl) : Tensor { shape, dtype := .Float32, device } :=
  checkedResult "zeros" (Tensor.ofRaw (RawTensor.zeros shape requiresGrad device noGrad))

def ones (shape : Shape) (requiresGrad : Bool := false)
    (device : Device := Device.CPU) (noGrad : requiresGrad = false := by exact of_decide_eq_true rfl) : Tensor { shape, dtype := .Float32, device } :=
  checkedResult "ones" (Tensor.ofRaw (RawTensor.ones shape requiresGrad device noGrad))

def full (shape : Shape) (value : Float) (requiresGrad : Bool := false)
    (device : Device := Device.CPU) (noGrad : requiresGrad = false := by exact of_decide_eq_true rfl) : Tensor { shape, dtype := .Float32, device } :=
  checkedResult "full" (Tensor.ofRaw (RawTensor.full shape value requiresGrad device noGrad))

def fullInt (shape : Shape) (value : Int64) : Tensor { shape, dtype := .Int64, device := .CPU } :=
  checkedResult "fullInt" (Tensor.ofRaw (RawTensor.fullInt shape value))

def zerosLike {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  t.checkedReplacement (RawTensor.zerosLike t.raw)

def onesLike {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  t.checkedReplacement (RawTensor.onesLike t.raw)

def randn (shape : Shape) (requiresGrad : Bool := false)
    (device : Device := Device.CPU) : IO (Tensor { shape, dtype := .Float32, device }) := do
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
def onesFor (σ : StaticSpec) [DType.ConstantSupported σ.dtype σ.device] : Tensor σ :=
  let raw := RawTensor.ones σ.shape false σ.device
  let raw := RawTensor.castDType raw σ.dtype
  checkedResult "onesFor" (Tensor.ofRaw raw)

/-- Transfer a tensor to the requested device, preserving its shape and dtype.
The result's device index is the destination, including a runtime-selected one. -/
def to {σ : StaticSpec} (t : Tensor σ) (device : Device) :
    Tensor { σ with device } :=
  checkedResult "to" do
    let raw := RawTensor.toDevice t.raw device
    TensorSpec.checkDType t.raw.dtype raw.dtype
    Device.check device raw.device
    Tensor.ofRaw raw

def requiresGrad {σ : StaticSpec} (t : Tensor σ) : Bool :=
  RawTensor.requiresGrad t.raw

/-- Read a scalar as a host Float. Nonscalar inputs do not typecheck. -/
def item? {d : DType} {dev : Device} (t : Tensor ⟨#[], d, dev⟩) : Except String Float :=
  RawTensor.itemChecked t.raw

/-- Read a scalar as a host Float; use `item?` to recover conversion failures. -/
def item {d : DType} {dev : Device} (t : Tensor ⟨#[], d, dev⟩) : Float :=
  match item? t with
  | .ok value => value
  | .error message => panic! s!"Tensor.item: {message}"

/-- Euclidean norm of an indexed vector. -/
def l2Norm! {n : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[n], d, dev⟩) : Float :=
  RawTensor.l2Norm! t.raw

def l2Norm {n : Nat} {d : DType} {dev : Device} [DType.FloatingOrComplex d]
    (t : Tensor ⟨#[n], d, dev⟩) : Float :=
  l2Norm! t

def maximum! {s : Shape} {l r result : DType} {dev : Device}
    [DType.Promotes l r result]
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) : Tensor ⟨s, result, dev⟩ :=
  checkedResult "maximum" (Tensor.ofRaw (RawTensor.maximum! a.raw b.raw))

def maximum {s : Shape} {l r result : DType} {dev : Device}
    [DType.Promotes l r result] [DType.Ordered l] [DType.Ordered r]
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) : Tensor ⟨s, result, dev⟩ :=
  maximum! a b

/-- Diagonal of a square matrix, retaining dtype and device. -/
def diag {n : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[n, n], d, dev⟩) : Tensor ⟨#[n], d, dev⟩ :=
  checkedResult "diag" (Tensor.ofRaw (RawTensor.diag t.raw))

/-- Frobenius norm of an indexed matrix. -/
def frobeniusNorm! {m n : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Float :=
  RawTensor.frobeniusNorm! t.raw

def frobeniusNorm {m n : Nat} {d : DType} {dev : Device} [DType.FloatingOrComplex d]
    (t : Tensor ⟨#[m, n], d, dev⟩) : Float :=
  frobeniusNorm! t

def rowNorms! {m n : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Tensor ⟨#[m], d.realResult, dev⟩ :=
  checkedResult "rowNorms" (Tensor.ofRaw (RawTensor.rowNorms! t.raw))

/-- Row norms are real, including for a complex input matrix. -/
def rowNorms {m n : Nat} {d : DType} {dev : Device} [DType.FloatingOrComplex d]
    (t : Tensor ⟨#[m, n], d, dev⟩) : Tensor ⟨#[m], d.realResult, dev⟩ :=
  rowNorms! t

/-- Maximum Euclidean row norm of a matrix. -/
def maxRowNorm! {m n : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Float :=
  RawTensor.maxRowNorm! t.raw

def maxRowNorm {m n : Nat} {d : DType} {dev : Device} [DType.FloatingOrComplex d]
    (t : Tensor ⟨#[m, n], d, dev⟩)
    (_rows : 0 < m := by first | exact of_decide_eq_true rfl | omega) : Float :=
  maxRowNorm! t

/-- Apply dropout while retaining shape, dtype and device. -/
def dropout {σ : StaticSpec} (t : Tensor σ) (p : Float) (training : Bool := true) :
    IO (Tensor σ) := do
  let raw ← RawTensor.dropout t.raw p training
  match t.checkedReplacement? raw with
  | .ok result => pure result
  | .error message => throw (IO.userError s!"Tensor.dropout: {message}")

/-- Batched embedding lookup. Tokens must be Int64 and share the weight's
placement; libtorch checks token values against the vocabulary size. -/
def embedding {batch seq vocab embed : Nat} {d : DType} {dev : Device}
    (tokens : Tensor ⟨#[batch, seq], .Int64, dev⟩)
    (weight : Tensor ⟨#[vocab, embed], d, dev⟩) : Tensor ⟨#[batch, seq, embed], d, dev⟩ :=
  checkedResult "embedding" (Tensor.ofRaw (RawTensor.embedding tokens.raw weight.raw))

/-- Vector embedding lookup with statically integral tokens. -/
def embedding1d {seq vocab embed : Nat} {d : DType} {dev : Device}
    (tokens : Tensor ⟨#[seq], .Int64, dev⟩)
    (weight : Tensor ⟨#[vocab, embed], d, dev⟩) : Tensor ⟨#[seq, embed], d, dev⟩ :=
  checkedResult "embedding1d" (Tensor.ofRaw (RawTensor.embedding1d tokens.raw weight.raw))

/-- Matrix exponential with a checked dtype/device-preserving result. -/
def matrixExp {n : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[n, n], d, dev⟩) : Tensor ⟨#[n, n], d, dev⟩ :=
  t.checkedReplacement (RawTensor.matrixExp t.raw)

/-- Checked real matrix logarithm. Dynamic dtypes are admitted only when their
actual metadata is a standard real floating dtype. Native failures, non-real
logarithms, and nonfinite results are reported as errors. The eigendecomposition
solver also rejects results that fail exponential reconstruction, including
unsupported defective or numerically unstable cases. -/
def matrixLog? {n : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[n, n], d, dev⟩) : Except String (Tensor ⟨#[n, n], d, dev⟩) := do
  if !t.raw.dtype.isRealFloating then
    throw s!"matrixLog requires a standard real floating dtype, got {t.raw.dtype}"
  t.checkedReplacement? (← RawTensor.matrixLogChecked t.raw)

/-- Real matrix logarithm. Its static domain excludes complex and shell types;
use `matrixLog?` for runtime-selected dtypes or recoverable numerical errors. -/
def matrixLog {n : Nat} {d : DType} {dev : Device}
    [DType.RealFloating d]
    (t : Tensor ⟨#[n, n], d, dev⟩) : Tensor ⟨#[n, n], d, dev⟩ :=
  match matrixLog? t with
  | .ok result => result
  | .error message =>
      letI : Inhabited (Tensor ⟨#[n, n], d, dev⟩) := ⟨t⟩
      panic! s!"Tensor.matrixLog: {message}"

/-- Inverse of a square matrix. Native code reports singular input errors. -/
def inv {n : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[n, n], d, dev⟩) : Tensor ⟨#[n, n], d, dev⟩ :=
  t.checkedReplacement (RawTensor.inv t.raw)

/-- Float32 identity matrix on CPU. -/
def eye (n : Nat) : Tensor ⟨#[n, n], .Float32, .CPU⟩ :=
  checkedResult "eye" (Tensor.ofRaw (RawTensor.eye n))

/-- Identity matrix for a complete specification. The native factory produces
    Float32 on CPU; the value is cast and moved to match the specification. -/
def eyeFor {n : Nat} (d : DType) (dev : Device := .CPU)
    [DType.ConstantSupported d dev] : Tensor ⟨#[n, n], d, dev⟩ :=
  let raw := RawTensor.eye n
  let raw := RawTensor.castDType raw d
  let raw := RawTensor.toDevice raw dev
  checkedResult "eyeFor" (Tensor.ofRaw raw)

/-- Complete QR decomposition: `Q` is m×m orthogonal and `R` is m×n upper
    triangular, retaining dtype and device. -/
def qrComplete {m n : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Tensor ⟨#[m, m], d, dev⟩ × Tensor ⟨#[m, n], d, dev⟩ :=
  let (q, r) := RawTensor.qr t.raw
  (checkedResult "qr.Q" (Tensor.ofRaw q), checkedResult "qr.R" (Tensor.ofRaw r))

/-- Reduced QR decomposition of a tall or square matrix (`n ≤ m`): `Q` is m×n
    with orthonormal columns and `R` is n×n upper triangular. The shape contract
    is a precondition on the libtorch reduced factorization; wide inputs do not
    typecheck at concrete dimensions. -/
def qrReduced {m n : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[m, n], d, dev⟩) (_tall : n ≤ m := by first | exact of_decide_eq_true rfl | omega | (simp <;> omega)) :
    Tensor ⟨#[m, n], d, dev⟩ × Tensor ⟨#[n, n], d, dev⟩ :=
  let (q, r) := RawTensor.qrReduced t.raw
  (checkedResult "qr.Q" (Tensor.ofRaw q), checkedResult "qr.R" (Tensor.ofRaw r))

/-- Reduced SVD of a tall or square matrix (`n ≤ m`). Singular values have
the real component dtype; `U` and `Vh` retain the input dtype. Native dtype,
backend, and factorization failures are recoverable. -/
def svdReduced? {m n : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[m, n], d, dev⟩) (_tall : n ≤ m := by first | exact of_decide_eq_true rfl | omega | (simp <;> omega)) :
    Except String (Tensor ⟨#[m, n], d, dev⟩ × Tensor ⟨#[n], d.realResult, dev⟩ ×
      Tensor ⟨#[n, n], d, dev⟩) := do
  let (u, s, vh) ← RawTensor.svdChecked t.raw
  let dtype := t.raw.dtype
  let device := t.raw.device
  return (← Tensor.checkedUpdate? dtype device u,
    ← Tensor.checkedUpdate? dtype.realResult device s,
    ← Tensor.checkedUpdate? dtype device vh)

def svdReduced {m n : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[m, n], d, dev⟩) (_tall : n ≤ m := by first | exact of_decide_eq_true rfl | omega | (simp <;> omega)) :
    Tensor ⟨#[m, n], d, dev⟩ × Tensor ⟨#[n], d.realResult, dev⟩ × Tensor ⟨#[n, n], d, dev⟩ :=
  match svdReduced? t _tall with
  | .ok result => result
  | .error message => nativeInvariantFailure s!"Tensor.svdReduced: {message}"

def svdvals! {n : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[n, n], d, dev⟩) : Tensor ⟨#[n], d.realResult, dev⟩ :=
  checkedResult "svdvals" (Tensor.ofRaw (RawTensor.svdvals! t.raw))

/-- Real singular values of a square matrix in descending order. -/
def svdvals {n : Nat} {d : DType} {dev : Device} [DType.FloatingOrComplex d]
    (t : Tensor ⟨#[n, n], d, dev⟩) : Tensor ⟨#[n], d.realResult, dev⟩ :=
  svdvals! t

/-- Diagonal matrix built from a vector, retaining dtype and device. -/
def diagflat {n : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[n], d, dev⟩) : Tensor ⟨#[n, n], d, dev⟩ :=
  checkedResult "diagflat" (Tensor.ofRaw (RawTensor.diagflat t.raw))

/-- Proves a shape has elements: literal shapes by evaluation, symbolic ones
    from a positivity hypothesis for every dimension. -/
macro "numel_pos" : tactic => `(tactic| first
  | exact of_decide_eq_true rfl
  | omega
  | (simp [Nat.pos_iff_ne_zero, Nat.mul_eq_zero] <;> omega))

def spectralNorm! {m n : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Float :=
  RawTensor.spectralNorm! t.raw

/-- Largest singular value of a nonempty floating or complex matrix. -/
def spectralNorm {m n : Nat} {d : DType} {dev : Device}
    [DType.FloatingOrComplex d] (t : Tensor ⟨#[m, n], d, dev⟩)
    (_nonempty : 0 < TensorSpec.numelOfShape #[m, n] := by numel_pos) : Float :=
  spectralNorm! t

def nuclearNorm! {m n : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Float :=
  RawTensor.nuclearNorm! t.raw

/-- Sum of the singular values of a floating or complex matrix. -/
def nuclearNorm {m n : Nat} {d : DType} {dev : Device}
    [DType.FloatingOrComplex d] (t : Tensor ⟨#[m, n], d, dev⟩) : Float :=
  nuclearNorm! t

/-! ## Casts -/

def toFloat32 {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withDType .Float32) :=
  checkedResult "toFloat32" (Tensor.ofRaw (RawTensor.toFloat32 t.raw))

def toBFloat16 {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withDType .BFloat16) :=
  checkedResult "toBFloat16" (Tensor.ofRaw (RawTensor.toBFloat16 t.raw))

def toInt64 {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withDType .Int64) :=
  checkedResult "toInt64" (Tensor.ofRaw (RawTensor.toInt64 t.raw))

/-! ## Elementwise binary (same shape and device, witnessed promotion)

The result dtype is an output parameter of `DType.Promotes`. Its proof ties
that dtype to a successful `promote?` result; unsupported pairs have no
instance. This witnesses dtype inference, not backend kernel availability.
-/

def add! {s : Shape} {l r result : DType} {dev : Device}
    [DType.Promotes l r result]
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, result, dev⟩ :=
  checkedResult "add" (Tensor.ofRaw (RawTensor.add! a.raw b.raw))

def add {s : Shape} {l r result : DType} {dev : Device}
    [DType.Promotes l r result]
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, result, dev⟩ :=
  add! a b

def sub! {s : Shape} {l r result : DType} {dev : Device}
    [DType.Promotes l r result]
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, result, dev⟩ :=
  checkedResult "sub" (Tensor.ofRaw (RawTensor.sub! a.raw b.raw))

def sub {s : Shape} {l r result : DType} {dev : Device}
    [DType.Promotes l r result] [DType.NonBool l] [DType.NonBool r]
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, result, dev⟩ :=
  sub! a b

def mul! {s : Shape} {l r result : DType} {dev : Device}
    [DType.Promotes l r result]
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, result, dev⟩ :=
  checkedResult "mul" (Tensor.ofRaw (RawTensor.mul! a.raw b.raw))

def mul {s : Shape} {l r result : DType} {dev : Device}
    [DType.Promotes l r result]
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, result, dev⟩ :=
  mul! a b

def div! {s : Shape} {l r result : DType} {dev : Device}
    [DType.Promotes l r result]
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, result.atLeastFloat, dev⟩ :=
  checkedResult "div" (Tensor.ofRaw (RawTensor.div! a.raw b.raw))

def div {s : Shape} {l r result : DType} {dev : Device}
    [DType.Promotes l r result]
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, result.atLeastFloat, dev⟩ :=
  div! a b

instance {s : Shape} {l r result : DType} {dev : Device}
    [DType.Promotes l r result] :
    HAdd (Tensor ⟨s, l, dev⟩) (Tensor ⟨s, r, dev⟩) (Tensor ⟨s, result, dev⟩) := ⟨add⟩

instance {s : Shape} {l r result : DType} {dev : Device}
    [DType.Promotes l r result] [DType.NonBool l] [DType.NonBool r] :
    HSub (Tensor ⟨s, l, dev⟩) (Tensor ⟨s, r, dev⟩) (Tensor ⟨s, result, dev⟩) := ⟨sub⟩

instance {s : Shape} {l r result : DType} {dev : Device}
    [DType.Promotes l r result] :
    HMul (Tensor ⟨s, l, dev⟩) (Tensor ⟨s, r, dev⟩) (Tensor ⟨s, result, dev⟩) := ⟨mul⟩

instance {s : Shape} {l r result : DType} {dev : Device}
    [DType.Promotes l r result] :
    HDiv (Tensor ⟨s, l, dev⟩) (Tensor ⟨s, r, dev⟩)
      (Tensor ⟨s, result.atLeastFloat, dev⟩) := ⟨div⟩

/-- Generic homogeneous arithmetic must also carry promotion evidence. -/
instance {σ : StaticSpec} [DType.Promotes σ.dtype σ.dtype σ.dtype] : Add (Tensor σ) :=
  ⟨add⟩

instance {σ : StaticSpec} [DType.Promotes σ.dtype σ.dtype σ.dtype]
    [DType.NonBool σ.dtype] : Sub (Tensor σ) := ⟨sub⟩

instance {σ : StaticSpec} [DType.Promotes σ.dtype σ.dtype σ.dtype] : Mul (Tensor σ) :=
  ⟨mul⟩

/-! ## Scalar arithmetic (host Float uses the Float32 default category) -/

def addScalar! {σ : StaticSpec} {result : DType}
    [DType.ScalarPromotes σ.dtype .Float32 result] (t : Tensor σ) (value : Float) :
    Tensor (σ.withDType result) :=
  checkedResult "addScalar" (Tensor.ofRaw (RawTensor.addScalar! t.raw value))

def addScalar {σ : StaticSpec} {result : DType}
    [DType.ScalarPromotes σ.dtype .Float32 result]
    (t : Tensor σ) (value : Float) : Tensor (σ.withDType result) :=
  addScalar! t value

def subScalar! {σ : StaticSpec} {result : DType}
    [DType.ScalarPromotes σ.dtype .Float32 result] (t : Tensor σ) (value : Float) :
    Tensor (σ.withDType result) :=
  checkedResult "subScalar" (Tensor.ofRaw (RawTensor.subScalar! t.raw value))

def subScalar {σ : StaticSpec} {result : DType}
    [DType.ScalarPromotes σ.dtype .Float32 result] [DType.NonBool σ.dtype]
    (t : Tensor σ) (value : Float) : Tensor (σ.withDType result) :=
  subScalar! t value

def mulScalar! {σ : StaticSpec} {result : DType}
    [DType.ScalarPromotes σ.dtype .Float32 result] (t : Tensor σ) (value : Float) :
    Tensor (σ.withDType result) :=
  checkedResult "mulScalar" (Tensor.ofRaw (RawTensor.mulScalar! t.raw value))

def mulScalar {σ : StaticSpec} {result : DType}
    [DType.ScalarPromotes σ.dtype .Float32 result]
    (t : Tensor σ) (value : Float) : Tensor (σ.withDType result) :=
  mulScalar! t value

def divScalar! {σ : StaticSpec} {result : DType}
    [DType.ScalarPromotes σ.dtype .Float32 result] (t : Tensor σ) (value : Float) :
    Tensor (σ.withDType result) :=
  checkedResult "divScalar" (Tensor.ofRaw (RawTensor.divScalar! t.raw value))

def divScalar {σ : StaticSpec} {result : DType}
    [DType.ScalarPromotes σ.dtype .Float32 result]
    (t : Tensor σ) (value : Float) : Tensor (σ.withDType result) :=
  divScalar! t value

instance {σ : StaticSpec} {result : DType} [DType.ScalarPromotes σ.dtype .Float32 result] :
    HAdd (Tensor σ) Float (Tensor (σ.withDType result)) := ⟨addScalar⟩

instance {σ : StaticSpec} {result : DType} [DType.ScalarPromotes σ.dtype .Float32 result]
    [DType.NonBool σ.dtype] : HSub (Tensor σ) Float (Tensor (σ.withDType result)) := ⟨subScalar⟩

instance {σ : StaticSpec} {result : DType} [DType.ScalarPromotes σ.dtype .Float32 result] :
    HMul (Tensor σ) Float (Tensor (σ.withDType result)) := ⟨mulScalar⟩

instance {σ : StaticSpec} {result : DType} [DType.ScalarPromotes σ.dtype .Float32 result] :
    HDiv (Tensor σ) Float (Tensor (σ.withDType result)) := ⟨divScalar⟩

/-- Floating clamp bounds follow weak scalar promotion, including integral
inputs becoming Float32. The result index records that native dtype. -/
def clampFloat {σ : StaticSpec} {result : DType}
    [DType.ScalarPromotes σ.dtype .Float32 result] [DType.Ordered σ.dtype]
    (t : Tensor σ) (lo hi : Float) : Tensor (σ.withDType result) :=
  checkedResult "clampFloat" (Tensor.ofRaw (RawTensor.clampFloat t.raw lo hi))

/-! ## Elementwise unary -/

def relu! {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  checkedResult "relu" (Tensor.ofRaw (RawTensor.relu! t.raw))

def relu {σ : StaticSpec} [DType.Relu σ.dtype] (t : Tensor σ) : Tensor σ :=
  relu! t

def abs! {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withDType σ.dtype.absResult) :=
  checkedResult "abs" (Tensor.ofRaw (RawTensor.abs! t.raw))

def abs {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withDType σ.dtype.absResult) :=
  abs! t

/-! ## Dtype-preserving unary math

Complex-capable operations accept inexact dtypes. Real-only activations use
standard real floating dtypes. The `!` variants omit the domain requirement;
libtorch rejects an unsupported dtype and the process aborts with its error.
Integer-to-floating promotion is not hidden behind a preserving index. -/

def exp! {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  checkedResult "exp" (Tensor.ofRaw (RawTensor.exp! t.raw))

def exp {σ : StaticSpec} (t : Tensor σ)
    (_domain : σ.dtype.isInexact = true := by first | rfl | exact DType.isInexact_of_realFloating | decide) : Tensor σ :=
  exp! t

def sqrt! {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  checkedResult "sqrt" (Tensor.ofRaw (RawTensor.sqrt! t.raw))

def sqrt {σ : StaticSpec} (t : Tensor σ)
    (_domain : σ.dtype.isInexact = true := by first | rfl | exact DType.isInexact_of_realFloating | decide) : Tensor σ :=
  sqrt! t

def rsqrt! {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  checkedResult "rsqrt" (Tensor.ofRaw (RawTensor.rsqrt! t.raw))

def rsqrt {σ : StaticSpec} (t : Tensor σ)
    (_domain : σ.dtype.isInexact = true := by first | rfl | exact DType.isInexact_of_realFloating | decide) : Tensor σ :=
  rsqrt! t

def sigmoid! {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  checkedResult "sigmoid" (Tensor.ofRaw (RawTensor.sigmoid! t.raw))

def sigmoid {σ : StaticSpec} (t : Tensor σ)
    (_domain : σ.dtype.isInexact = true := by first | rfl | exact DType.isInexact_of_realFloating | decide) : Tensor σ :=
  sigmoid! t

def tanh! {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  checkedResult "tanh" (Tensor.ofRaw (RawTensor.tanh! t.raw))

def tanh {σ : StaticSpec} (t : Tensor σ)
    (_domain : σ.dtype.isInexact = true := by first | rfl | exact DType.isInexact_of_realFloating | decide) : Tensor σ :=
  tanh! t

def atan! {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  checkedResult "atan" (Tensor.ofRaw (RawTensor.atan! t.raw))

def atan {σ : StaticSpec} (t : Tensor σ)
    (_inexact : σ.dtype.isInexact = true := by first | rfl | exact DType.isInexact_of_realFloating | decide) : Tensor σ :=
  atan! t

def gelu! {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  checkedResult "gelu" (Tensor.ofRaw (RawTensor.gelu! t.raw))

def gelu {σ : StaticSpec} (t : Tensor σ)
    (_domain : σ.dtype.isRealFloating = true := by first | rfl | exact DType.isRealFloating_of_realFloating | decide) : Tensor σ :=
  gelu! t

def silu! {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  checkedResult "silu" (Tensor.ofRaw (RawTensor.silu! t.raw))

def silu {σ : StaticSpec} (t : Tensor σ)
    (_domain : σ.dtype.isRealFloating = true := by first | rfl | exact DType.isRealFloating_of_realFloating | decide) : Tensor σ :=
  silu! t

def pow! {σ : StaticSpec} (t : Tensor σ) (exponent : Float) : Tensor σ :=
  checkedResult "pow" (Tensor.ofRaw (RawTensor.pow! t.raw exponent))

def pow {σ : StaticSpec} (t : Tensor σ) (exponent : Float)
    (_domain : σ.dtype.isInexact = true := by first | rfl | exact DType.isInexact_of_realFloating | decide) : Tensor σ :=
  pow! t exponent

def softmax! {σ : StaticSpec} (t : Tensor σ) (dim : Nat := σ.shape.size - 1) : Tensor σ :=
  checkedResult "softmax" (Tensor.ofRaw (RawTensor.softmax! t.raw dim))

def softmax {σ : StaticSpec} (t : Tensor σ) (dim : Nat := σ.shape.size - 1)
    (_domain : σ.dtype.isRealFloating = true := by first | rfl | exact DType.isRealFloating_of_realFloating | decide)
    (_axis : dim < max 1 σ.shape.size := by first | exact of_decide_eq_true rfl | omega | (simp <;> fail "axis out of range")) : Tensor σ :=
  softmax! t dim

/-! ## Matmul family (operands must share dtype and device) -/

/-- Matrix kernels require actual dtype and device equality, even when either
static component was erased. Elementwise promotion is not a matrix cast. -/
private def checkMatrixOperands {σ τ : StaticSpec} (a : Tensor σ) (b : Tensor τ) :
    Except String Unit := do
  DType.expectedEq a.raw.dtype b.raw.dtype
  Tensor.checkSameDevice a b

def matmul! {s1 s2 : Shape} {d : DType} {dev : Device}
    (a : Tensor ⟨s1, d, dev⟩) (b : Tensor ⟨s2, d, dev⟩) :
    Tensor ⟨matmulShape s1 s2, d, dev⟩ :=
  checkedResult "matmul" (Tensor.ofRaw (RawTensor.matmul! a.raw b.raw))

/-- Discharges `matmulShape?` compatibility for shapes with symbolic dimensions, including
    broadcast batch axes; fails hard on incompatible shapes. -/
macro "matmul_shape" : tactic => `(tactic| first
  | rfl
  | (simp [TensorSpec.broadcastShapes?, TensorSpec.broadcastShapes?.go, List.range_succ] <;>
      (repeat' split) <;> (repeat' split at *) <;> simp_all <;> fail "incompatible matmul shapes"))

def matmul {s1 s2 : Shape} {d : DType} {dev : Device} [DType.FloatingOrComplex d]
    (a : Tensor ⟨s1, d, dev⟩) (b : Tensor ⟨s2, d, dev⟩)
    (_h : (TensorSpec.matmulShape? s1 s2).isSome = true := by matmul_shape) :
    Tensor ⟨matmulShape s1 s2, d, dev⟩ :=
  matmul! a b

def mm! {m k n : Nat} {d : DType} {dev : Device}
    (a : Tensor ⟨#[m, k], d, dev⟩) (b : Tensor ⟨#[k, n], d, dev⟩) :
    Tensor ⟨#[m, n], d, dev⟩ :=
  checkedResult "mm" (Tensor.ofRaw (RawTensor.mm! a.raw b.raw))

def mm {m k n : Nat} {d : DType} {dev : Device} [DType.FloatingOrComplex d]
    (a : Tensor ⟨#[m, k], d, dev⟩) (b : Tensor ⟨#[k, n], d, dev⟩) :
    Tensor ⟨#[m, n], d, dev⟩ :=
  mm! a b

def linear! {m n b : Nat} {d : DType} {dev : Device}
    (x : Tensor ⟨#[b, m], d, dev⟩) (weight : Tensor ⟨#[n, m], d, dev⟩) :
    Tensor ⟨#[b, n], d, dev⟩ :=
  checkedResult "linear" (Tensor.ofRaw (RawTensor.linear! x.raw weight.raw))

def linear {m n b : Nat} {d : DType} {dev : Device} [DType.FloatingOrComplex d]
    (x : Tensor ⟨#[b, m], d, dev⟩) (weight : Tensor ⟨#[n, m], d, dev⟩) :
    Tensor ⟨#[b, n], d, dev⟩ :=
  linear! x weight

def linear3d! {batch seq inDim outDim : Nat} {d : DType} {dev : Device}
    (x : Tensor ⟨#[batch, seq, inDim], d, dev⟩)
    (weight : Tensor ⟨#[outDim, inDim], d, dev⟩) :
    Tensor ⟨#[batch, seq, outDim], d, dev⟩ :=
  checkedResult "linear3d" (Tensor.ofRaw (RawTensor.linear3d! x.raw weight.raw))

def linear3d {batch seq inDim outDim : Nat} {d : DType} {dev : Device} [DType.FloatingOrComplex d]
    (x : Tensor ⟨#[batch, seq, inDim], d, dev⟩)
    (weight : Tensor ⟨#[outDim, inDim], d, dev⟩) :
    Tensor ⟨#[batch, seq, outDim], d, dev⟩ :=
  linear3d! x weight

def matmul3d! {batch seq k n : Nat} {d : DType} {dev : Device}
    (a : Tensor ⟨#[batch, seq, k], d, dev⟩) (b : Tensor ⟨#[k, n], d, dev⟩) :
    Tensor ⟨#[batch, seq, n], d, dev⟩ :=
  checkedResult "matmul3d" (Tensor.ofRaw (RawTensor.matmul3d! a.raw b.raw))

def matmul3d {batch seq k n : Nat} {d : DType} {dev : Device} [DType.FloatingOrComplex d]
    (a : Tensor ⟨#[batch, seq, k], d, dev⟩) (b : Tensor ⟨#[k, n], d, dev⟩) :
    Tensor ⟨#[batch, seq, n], d, dev⟩ :=
  matmul3d! a b

def bmm4d! {b h m k n : Nat} {d : DType} {dev : Device}
    (a : Tensor ⟨#[b, h, m, k], d, dev⟩) (x : Tensor ⟨#[b, h, k, n], d, dev⟩) :
    Tensor ⟨#[b, h, m, n], d, dev⟩ :=
  checkedResult "bmm4d" (Tensor.ofRaw (RawTensor.bmm4d! a.raw x.raw))

def bmm4d {b h m k n : Nat} {d : DType} {dev : Device} [DType.FloatingOrComplex d]
    (a : Tensor ⟨#[b, h, m, k], d, dev⟩) (x : Tensor ⟨#[b, h, k, n], d, dev⟩) :
    Tensor ⟨#[b, h, m, n], d, dev⟩ :=
  bmm4d! a x

/-! ## Broadcasting elementwise ops

The result spec is `σ₁.pointwise σ₂` (NumPy/PyTorch rules, including weak
scalar promotion and scalar placement), and the auto-param proof —
discharged by `rfl` for concrete shapes — guarantees the pair actually
broadcasts, so incompatible shape pairs fail to compile.

At runtime libtorch broadcasts the native handles directly; no shape erasure
or reshaping is required at the raw boundary. -/

def addB! {σ₁ σ₂ : StaticSpec} {result : DType}
    [TensorSpec.Promotes σ₁.toTensorSpec σ₂.toTensorSpec result]
    (a : Tensor σ₁) (b : Tensor σ₂) : Tensor (σ₁.pointwise σ₂) :=
  checkedResult "add" (Tensor.ofRaw (RawTensor.add! a.raw b.raw))

/-- Broadcasting elementwise addition (`a +ᵇ b`). -/
def addB {σ₁ σ₂ : StaticSpec} {result : DType}
    [Device.PointwiseCompatible σ₁ σ₂]
    [TensorSpec.Promotes σ₁.toTensorSpec σ₂.toTensorSpec result]
    (a : Tensor σ₁) (b : Tensor σ₂)
    (_h : (TensorSpec.broadcastShapes? σ₁.shape σ₂.shape).isSome = true := by rfl) :
    Tensor (σ₁.pointwise σ₂) :=
  addB! a b

def subB! {σ₁ σ₂ : StaticSpec} {result : DType}
    [TensorSpec.Promotes σ₁.toTensorSpec σ₂.toTensorSpec result]
    (a : Tensor σ₁) (b : Tensor σ₂) : Tensor (σ₁.pointwise σ₂) :=
  checkedResult "sub" (Tensor.ofRaw (RawTensor.sub! a.raw b.raw))

/-- Broadcasting elementwise subtraction (`a -ᵇ b`). -/
def subB {σ₁ σ₂ : StaticSpec} {result : DType}
    [Device.PointwiseCompatible σ₁ σ₂]
    [TensorSpec.Promotes σ₁.toTensorSpec σ₂.toTensorSpec result] [DType.NonBool σ₁.dtype] [DType.NonBool σ₂.dtype]
    (a : Tensor σ₁) (b : Tensor σ₂)
    (_h : (TensorSpec.broadcastShapes? σ₁.shape σ₂.shape).isSome = true := by rfl) :
    Tensor (σ₁.pointwise σ₂) :=
  subB! a b

def mulB! {σ₁ σ₂ : StaticSpec} {result : DType}
    [TensorSpec.Promotes σ₁.toTensorSpec σ₂.toTensorSpec result]
    (a : Tensor σ₁) (b : Tensor σ₂) : Tensor (σ₁.pointwise σ₂) :=
  checkedResult "mul" (Tensor.ofRaw (RawTensor.mul! a.raw b.raw))

/-- Broadcasting elementwise multiplication (`a *ᵇ b`). -/
def mulB {σ₁ σ₂ : StaticSpec} {result : DType}
    [Device.PointwiseCompatible σ₁ σ₂]
    [TensorSpec.Promotes σ₁.toTensorSpec σ₂.toTensorSpec result]
    (a : Tensor σ₁) (b : Tensor σ₂)
    (_h : (TensorSpec.broadcastShapes? σ₁.shape σ₂.shape).isSome = true := by rfl) :
    Tensor (σ₁.pointwise σ₂) :=
  mulB! a b

def divB! {σ₁ σ₂ : StaticSpec} {result : DType}
    [TensorSpec.Promotes σ₁.toTensorSpec σ₂.toTensorSpec result]
    (a : Tensor σ₁) (b : Tensor σ₂) : Tensor (σ₁.division σ₂) :=
  checkedResult "div" (Tensor.ofRaw (RawTensor.div! a.raw b.raw))

/-- Broadcasting elementwise division (`a /ᵇ b`). -/
def divB {σ₁ σ₂ : StaticSpec} {result : DType}
    [Device.PointwiseCompatible σ₁ σ₂]
    [TensorSpec.Promotes σ₁.toTensorSpec σ₂.toTensorSpec result]
    (a : Tensor σ₁) (b : Tensor σ₂)
    (_h : (TensorSpec.broadcastShapes? σ₁.shape σ₂.shape).isSome = true := by rfl) :
    Tensor (σ₁.division σ₂) :=
  divB! a b

def broadcastTo! {σ : StaticSpec} (t : Tensor σ) (target : Shape) :
    Tensor (σ.withShape target) :=
  checkedResult "broadcastTo" (Tensor.ofRaw (RawTensor.expand! t.raw target))

/-- Asymmetric broadcast: stretch `t` to `target` (a libtorch `expand` view,
    no copy). The fixed-side proof `broadcastsTo s target` rules out targets
    that would themselves be stretched. -/
def broadcastTo {σ : StaticSpec} (t : Tensor σ) (target : Shape)
    (_h : TensorSpec.broadcastsTo σ.shape target := by rfl) :
    Tensor (σ.withShape target) :=
  broadcastTo! t target

/-- Metadata-checked `expand` view, for code generic over dimensions where the
    `broadcastTo` proof cannot reduce (e.g. `#[batch, outDim]` with symbolic
    dims). libtorch still validates the expansion at runtime. -/
def expand? {σ : StaticSpec} (t : Tensor σ) (target : Shape) :
    Except String (Tensor (σ.withShape target)) := do
  if TensorSpec.broadcastShapes? σ.shape target != some target then
    throw s!"Cannot expand shape {σ.shape} to {target}"
  Tensor.ofRaw (← RawTensor.expandChecked t.raw target)

/-- Expand to literal, nonnegative extents. Use `expand?` to handle a native
result-spec mismatch without invoking the panic fallback allocation. -/
def expand {σ : StaticSpec} (t : Tensor σ) (target : Shape) :
    Tensor (σ.withShape target) :=
  checkedResult "expand" (t.expand? target)

/-! ## Reductions -/

def sumAll! {σ : StaticSpec} (t : Tensor σ) : Tensor σ.sumSpec :=
  checkedResult "sumAll" (Tensor.ofRaw (RawTensor.sum! t.raw none false))

def sumAll {σ : StaticSpec} (t : Tensor σ) : Tensor σ.sumSpec :=
  sumAll! t

/-- Maximum of all elements, without precondition checks; libtorch rejects an
empty or unordered tensor and the process aborts with its error. -/
def maxAll! {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withShape #[]) :=
  checkedResult "maxAll" (Tensor.ofRaw (RawTensor.maxAll! t.raw))

/-- Maximum over a nonempty ordered tensor. Runtime shapes/dtypes use `maxAll!`. -/
def maxAll {σ : StaticSpec} [DType.Ordered σ.dtype] (t : Tensor σ)
    (_nonempty : 0 < TensorSpec.numelOfShape σ.shape := by numel_pos) :
    Tensor (σ.withShape #[]) :=
  maxAll! t

def sumDim! {σ : StaticSpec} (t : Tensor σ) (dim : Nat) (keepdim : Bool := false) :
    Tensor (σ.sumSpec.withShape (reduceShape σ.shape dim keepdim)) :=
  checkedResult "sumDim" (Tensor.ofRaw (RawTensor.sum! t.raw (some #[dim]) keepdim))

/-- Reduce one axis using libtorch's accumulation dtype. -/
def sumDim {σ : StaticSpec} (t : Tensor σ) (dim : Nat) (keepdim : Bool := false)
    (_axis : dim < max 1 σ.shape.size := by first | exact of_decide_eq_true rfl | (simp <;> fail "axis out of range")) :
    Tensor (σ.sumSpec.withShape (reduceShape σ.shape dim keepdim)) :=
  sumDim! t dim keepdim

def meanAll! {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withShape #[]) :=
  checkedResult "meanAll" (Tensor.ofRaw (RawTensor.mean! t.raw none false))

def meanAll {σ : StaticSpec} [DType.Mean σ.dtype] (t : Tensor σ) :
    Tensor (σ.withShape #[]) :=
  meanAll! t

/-! ## Shape ops (dtype/device-preserving) -/

/-- Reshape without changing the element count. Unlike the legacy raw
    `torch.reshape`, an empty target creates a scalar. Extents must be
    representable as signed 64-bit integers, so none acts as native inference. -/
def reshape {σ : StaticSpec} (t : Tensor σ) (shape' : Shape)
    (_h : TensorSpec.numelOfShape σ.shape = TensorSpec.numelOfShape shape' := by
      first
      | exact of_decide_eq_true rfl
      | (simp +zetaDelta [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm] <;> omega)) :
    Tensor (σ.withShape shape') :=
  checkedResult "reshape" do
    if shape'.any (fun dim => dim > 9223372036854775807) then
      throw "reshape requires nonnegative extents representable as signed 64-bit integers"
    let actualNumel := TensorSpec.numelOfShape t.runtimeShape
    let targetNumel := TensorSpec.numelOfShape shape'
    if actualNumel != targetNumel then
      throw s!"reshape requires {actualNumel} elements, requested {targetNumel}"
    let raw := RawTensor.reshape! t.raw shape'
    TensorSpec.checkDType t.raw.dtype raw.dtype
    Device.check t.raw.device raw.device
    Tensor.ofRaw raw

/-- Slice an exact number of entries along a valid axis. Bounds are checked in
natural arithmetic so `start + len` cannot wrap or silently truncate. -/
def slice {σ : StaticSpec} (t : Tensor σ) (dim : Nat) (start len : Nat)
    (_axis : dim < σ.shape.size := by first | exact of_decide_eq_true rfl | (simp <;> fail "axis out of range"))
    (_bounds : start + len ≤ σ.shape[dim]'_axis := by first | exact of_decide_eq_true rfl | omega | (simp <;> omega)) :
    Tensor (σ.withShape (σ.shape.set! dim len)) :=
  checkedResult "slice" (Tensor.ofRaw (RawTensor.slice t.raw dim start len))

def permute {σ : StaticSpec} (t : Tensor σ) (permutation : Array Nat)
    (_h : TensorSpec.isPermutation permutation σ.shape.size = true := by
      first | exact of_decide_eq_true rfl | (simp <;> fail "not a permutation of the axes")) :
    Tensor (σ.withShape (TensorSpec.permuteShape σ.shape permutation)) :=
  checkedResult "permute" (Tensor.ofRaw (RawTensor.permute t.raw permutation))

def transpose {σ : StaticSpec} (t : Tensor σ) (dim0 dim1 : Nat)
    (h0 : dim0 < max 1 σ.shape.size := by first | exact of_decide_eq_true rfl | (simp <;> fail "axis out of range"))
    (h1 : dim1 < max 1 σ.shape.size := by first | exact of_decide_eq_true rfl | (simp <;> fail "axis out of range")) :
    Tensor (σ.withShape (transposeShape σ.shape dim0 dim1 h0 h1)) :=
  checkedResult "transpose" do Tensor.ofRaw (← RawTensor.transpose t.raw dim0 dim1)

def transpose2d {m n : Nat} {d : DType} {dev : Device}
    (t : Tensor ⟨#[m, n], d, dev⟩) : Tensor ⟨#[n, m], d, dev⟩ :=
  checkedResult "transpose2d" (Tensor.ofRaw (RawTensor.transpose2d t.raw))

def unsqueeze {σ : StaticSpec} (t : Tensor σ) (dim : Nat)
    (h : dim ≤ σ.shape.size := by first | exact of_decide_eq_true rfl | (simp <;> fail "axis out of range")) :
    Tensor (σ.withShape (unsqueezeShape σ.shape dim h)) :=
  checkedResult "unsqueeze" (Tensor.ofRaw (RawTensor.unsqueeze t.raw dim))

def squeeze {σ : StaticSpec} (t : Tensor σ) (dim : Nat)
    (h : dim < max 1 σ.shape.size := by first | exact of_decide_eq_true rfl | (simp <;> fail "axis out of range")) :
    Tensor (σ.withShape (squeezeShape σ.shape dim h)) :=
  checkedResult "squeeze" (Tensor.ofRaw (RawTensor.squeeze t.raw dim))

/-- Discharges `catCompatible` for shapes with symbolic dimensions; fails hard on
    a rank mismatch, an out-of-range axis or disagreeing extents. -/
macro "cat_shape" : tactic => `(tactic| first
  | exact of_decide_eq_true rfl
  | (simp <;> fail "incompatible cat shapes"))

def cat! {s1 s2 : Shape} {d : DType} {dev : Device}
    (t1 : Tensor ⟨s1, d, dev⟩) (t2 : Tensor ⟨s2, d, dev⟩) (dim : Nat) :
    Tensor ⟨TensorSpec.catShape s1 s2 dim, d, dev⟩ :=
  checkedResult "cat" (Tensor.ofRaw (RawTensor.cat! t1.raw t2.raw dim))

def cat {s1 s2 : Shape} {d : DType} {dev : Device}
    (t1 : Tensor ⟨s1, d, dev⟩) (t2 : Tensor ⟨s2, d, dev⟩) (dim : Nat)
    (_h : TensorSpec.catCompatible s1 s2 dim = true := by cat_shape) :
    Tensor ⟨TensorSpec.catShape s1 s2 dim, d, dev⟩ :=
  cat! t1 t2 dim

/-- Stack a runtime list of equal vectors along a new leading axis; the result
    shape records the actual list length. -/
def stack1d {cols : Nat} {d : DType} {dev : Device}
    (tensors : Array (Tensor ⟨#[cols], d, dev⟩)) :
    Tensor ⟨#[tensors.size, cols], d, dev⟩ :=
  checkedResult "stack1d" (Tensor.ofRaw (RawTensor.stack1d (tensors.map Tensor.toTensor) 0))

/-- Slice `len` entries along `dim` at a runtime offset, validating the result
    against the claimed spec and panicking on mismatch. -/
def sliceChecked {σ : StaticSpec} (t : Tensor σ) (dim : Nat) (start len : Nat)
    (shape' : Shape) : Tensor (σ.withShape shape') :=
  match Tensor.ofRaw (RawTensor.slice t.raw dim start len) with
  | .ok r => r
  | .error message => nativeInvariantFailure s!"Tensor.sliceChecked: {message}"

/-- Vector-Jacobian product, checked against `x`'s spec. Saved intermediates
are released by default. As in PyTorch, omitted retention follows `createGraph`.
Unused inputs produce zero gradients; parameter gradient buffers are unchanged. -/
def grad {σ τ : StaticSpec} (y : Tensor τ) (x : Tensor σ) (dy : Tensor τ)
    (retainGraph : Option Bool := none) (createGraph : Bool := false) : IO (Tensor σ) := do
  let raw ← RawTensor.grad y.raw x.raw dy.raw
    (retainGraph.getD createGraph) createGraph
  match x.checkedReplacement? raw with
  | .ok gradient => pure gradient
  | .error message => throw (IO.userError s!"Tensor.grad: {message}")

/-- 1-D Float32 tensor from a host array; the result shape records the actual
    array length. -/
def fromFloatArray (arr : Array Float) : Tensor ⟨#[arr.size], .Float32, .CPU⟩ :=
  checkedResult "fromFloatArray" (Tensor.ofRaw (RawTensor.fromFloatArray arr))

/-- 1-D Int64 tensor from a host array; the result shape records the actual
    array length. -/
def fromInt64Array (arr : Array Int64) : Tensor ⟨#[arr.size], .Int64, .CPU⟩ :=
  checkedResult "fromInt64Array" (Tensor.ofRaw (RawTensor.fromInt64Array arr))

/-- Float32 tensor of `shape` from a host array whose length is known to match. -/
def fromFloatArrayShaped (arr : Array Float) (shape : Shape)
    (h : arr.size = TensorSpec.numelOfShape shape) : Tensor ⟨shape, .Float32, .CPU⟩ :=
  (fromFloatArray arr).reshape shape (by rw [← h]; simp)

/-- Int64 tensor of `shape` from a host array whose length is known to match. -/
def fromInt64ArrayShaped (arr : Array Int64) (shape : Shape)
    (h : arr.size = TensorSpec.numelOfShape shape) : Tensor ⟨shape, .Int64, .CPU⟩ :=
  (fromInt64Array arr).reshape shape (by rw [← h]; simp)

/-- Float32 tensor of `shape` from host data, rejecting a length mismatch where the data enters. -/
def fromFloatArrayAs? (arr : Array Float) (shape : Shape) : Except String (Tensor ⟨shape, .Float32, .CPU⟩) :=
  if h : arr.size = TensorSpec.numelOfShape shape then .ok (fromFloatArrayShaped arr shape h)
  else .error s!"expected {TensorSpec.numelOfShape shape} values for shape {shape}, got {arr.size}"

/-- Int64 tensor of `shape` from host data, rejecting a length mismatch where the data enters. -/
def fromInt64ArrayAs? (arr : Array Int64) (shape : Shape) : Except String (Tensor ⟨shape, .Int64, .CPU⟩) :=
  if h : arr.size = TensorSpec.numelOfShape shape then .ok (fromInt64ArrayShaped arr shape h)
  else .error s!"expected {TensorSpec.numelOfShape shape} values for shape {shape}, got {arr.size}"

/-- Argmax along `dim`, without precondition checks; libtorch rejects an
unordered dtype, an invalid axis or an empty axis and the process aborts. -/
def argmax! {s : Shape} {d : DType} {dev : Device} (t : Tensor ⟨s, d, dev⟩) (dim : Nat) :
    Tensor ⟨reduceShape s dim false, .Int64, dev⟩ :=
  checkedResult "argmax" (Tensor.ofRaw (RawTensor.argmax! t.raw dim))

/-- Argmax along a valid nonempty axis of an ordered tensor. Scalar input
permits axis zero. Runtime shapes and dtypes use `argmax!`. -/
def argmax {s : Shape} {d : DType} {dev : Device} [DType.Ordered d]
    (t : Tensor ⟨s, d, dev⟩) (dim : Nat)
    (_axis : dim < max 1 s.size := by first | exact of_decide_eq_true rfl | (simp <;> fail "axis out of range"))
    (_nonempty : 0 < s[dim]?.getD 1 := by first | exact of_decide_eq_true rfl | omega | (simp <;> omega)) :
    Tensor ⟨reduceShape s dim false, .Int64, dev⟩ :=
  argmax! t dim

/-- Top-k filter over logits (entries outside the top k become -inf), spec-preserving. -/
def topKFilter {σ : StaticSpec} (logits : Tensor σ) (k : Nat) : Tensor σ :=
  logits.checkedReplacement (RawTensor.topKFilter logits.raw k)

/-- Top-p (nucleus) filter over logits, spec-preserving. -/
def topPFilter {σ : StaticSpec} (logits : Tensor σ) (p : Float) : Tensor σ :=
  logits.checkedReplacement (RawTensor.topPFilter logits.raw p)

/-- Elementwise equality with a scalar: Bool tensor of the same shape. -/
def eqScalar {s : Shape} {d : DType} {dev : Device} (t : Tensor ⟨s, d, dev⟩) (scalar : Int64) :
    Tensor ⟨s, .Bool, dev⟩ :=
  checkedResult "eqScalar" (Tensor.ofRaw (RawTensor.eqScalar t.raw scalar))

def indexSelect! {n k : Nat} {d : DType} {dev : Device}
    (data : Tensor ⟨#[n], d, dev⟩) (indices : Tensor ⟨#[k], .Int64, dev⟩) :
    Tensor ⟨#[k], d, dev⟩ :=
  checkedResult "indexSelect" (Tensor.ofRaw (RawTensor.indexSelect! data.raw 0 indices.raw))

/-- Select entries of a 1-D tensor by Int64 indices. Index values are runtime
data: an out-of-range index aborts; `indexSelect?` returns it as an error. -/
def indexSelect {n k : Nat} {d : DType} {dev : Device}
    (data : Tensor ⟨#[n], d, dev⟩) (indices : Tensor ⟨#[k], .Int64, dev⟩) :
    Tensor ⟨#[k], d, dev⟩ :=
  indexSelect! data indices

/-- `indexSelect` for untrusted index values, returning an out-of-range index
as an error. Validation synchronizes one validity bit from the device. -/
def indexSelect? {n k : Nat} {d : DType} {dev : Device}
    (data : Tensor ⟨#[n], d, dev⟩) (indices : Tensor ⟨#[k], .Int64, dev⟩) :
    Except String (Tensor ⟨#[k], d, dev⟩) := do
  Tensor.ofRaw (← RawTensor.indexSelect? data.raw 0 indices.raw)

/-- Per-element cross entropy, without precondition checks; a non-floating dtype,
an invalid label or a backend error aborts the process. The default ignored label
is -100, matching the native operation. -/
def crossEntropyNone! {n c : Nat} {d : DType} {dev : Device}
    (logits : Tensor ⟨#[n, c], d, dev⟩) (targets : Tensor ⟨#[n], .Int64, dev⟩) :
    Tensor ⟨#[n], d, dev⟩ :=
  checkedResult "crossEntropyNone" (Tensor.ofRaw (RawTensor.crossEntropyNone! logits.raw targets.raw))

/-- Per-element cross entropy over real floating logits and Int64 targets. -/
def crossEntropyNone {n c : Nat} {d : DType} {dev : Device} [DType.RealFloating d]
    (logits : Tensor ⟨#[n, c], d, dev⟩) (targets : Tensor ⟨#[n], .Int64, dev⟩) :
    Tensor ⟨#[n], d, dev⟩ :=
  crossEntropyNone! logits targets

/-- Extract an Int64 scalar. -/
def itemInt {d : DType} {dev : Device} (t : Tensor ⟨#[], d, dev⟩) : Int64 :=
  RawTensor.itemInt t.raw

/-- Selection promotes the value operands; a Bool condition never changes dtype. -/
def whereSelect! {s : Shape} {l r result : DType} {dev : Device}
    [DType.Promotes l r result]
    (condition : Tensor ⟨s, .Bool, dev⟩) (x : Tensor ⟨s, l, dev⟩)
    (y : Tensor ⟨s, r, dev⟩) : Tensor ⟨s, result, dev⟩ :=
  checkedResult "whereSelect" (Tensor.ofRaw (RawTensor.whereSelect! condition.raw x.raw y.raw))

def whereSelect {s : Shape} {l r result : DType} {dev : Device}
    [DType.Promotes l r result]
    (condition : Tensor ⟨s, .Bool, dev⟩) (x : Tensor ⟨s, l, dev⟩)
    (y : Tensor ⟨s, r, dev⟩) : Tensor ⟨s, result, dev⟩ :=
  whereSelect! condition x y

/-- Elementwise logical negation of a Bool tensor. -/
def logicalNot {s : Shape} {dev : Device} (t : Tensor ⟨s, .Bool, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  t.checkedReplacement (RawTensor.logicalNot t.raw)

def logicalAnd! {s : Shape} {dev : Device} (a b : Tensor ⟨s, .Bool, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  a.checkedReplacement (RawTensor.logicalAnd! a.raw b.raw)

/-- Elementwise logical conjunction of two Bool tensors. -/
def logicalAnd {s : Shape} {dev : Device} (a b : Tensor ⟨s, .Bool, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  logicalAnd! a b

/-- Elementwise logical disjunction of two Bool tensors. -/
def logicalOr {s : Shape} {dev : Device} (a b : Tensor ⟨s, .Bool, dev⟩) :
    Tensor ⟨s, .Bool, dev⟩ :=
  logicalNot (logicalAnd (logicalNot a) (logicalNot b))

/-- Mean cross entropy, without precondition checks; a non-floating dtype, an
invalid label or a backend error aborts the process. -/
def crossEntropy! {n c : Nat} {d : DType} {dev : Device}
    (logits : Tensor ⟨#[n, c], d, dev⟩) (targets : Tensor ⟨#[n], .Int64, dev⟩) :
    Tensor ⟨#[], d, dev⟩ :=
  checkedResult "crossEntropy" (Tensor.ofRaw (RawTensor.crossEntropy! logits.raw targets.raw))

/-- Mean cross entropy over real floating logits and Int64 targets. -/
def crossEntropy {n c : Nat} {d : DType} {dev : Device} [DType.RealFloating d]
    (logits : Tensor ⟨#[n, c], d, dev⟩) (targets : Tensor ⟨#[n], .Int64, dev⟩) :
    Tensor ⟨#[], d, dev⟩ :=
  crossEntropy! logits targets

/-- Draw `numSamples` Int64 indices per row of a probability matrix. -/
def multinomial {n v : Nat} {d : DType} {dev : Device}
    (probs : Tensor ⟨#[n, v], d, dev⟩) (numSamples : Nat) (replacement : Bool := false) :
    IO (Tensor ⟨#[n, numSamples], .Int64, dev⟩) := do
  let raw ← RawTensor.multinomial probs.raw numSamples replacement
  match Tensor.ofRaw raw with
  | .ok result => pure result
  | .error message => throw (IO.userError s!"Tensor.multinomial: {message}")

/-- Scatter `src` into a clone of `input` along `dim` at Int64 `indices`;
    result preserves the input spec. -/
def scatter2d {batch seq k : Nat} {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, seq], d, dev⟩) (dim : Nat)
    (indices : Tensor ⟨#[batch, k], .Int64, dev⟩) (src : Tensor ⟨#[batch, k], d, dev⟩)
    (_axis : dim = 1 ∨ (dim = 0 ∧ k ≤ seq) := by
      first | exact of_decide_eq_true rfl | (simp <;> fail "scatter axis out of range")) :
    Tensor ⟨#[batch, seq], d, dev⟩ :=
  input.checkedReplacement (RawTensor.scatter2d input.raw dim indices.raw src.raw)

/-- All-close predicate over two tensors of the same spec. -/
def allclose {σ : StaticSpec} (a b : Tensor σ) (rtol : Float := 1e-05) (atol : Float := 1e-08) :
    Bool :=
  RawTensor.allclose a.raw b.raw rtol atol

/-- Whether the tensor is an autograd leaf. -/
def isLeaf {σ : StaticSpec} (t : Tensor σ) : Bool :=
  RawTensor.isLeaf t.raw

def logSoftmax! {σ : StaticSpec} (t : Tensor σ) (dim : Nat := σ.shape.size - 1) : Tensor σ :=
  checkedResult "logSoftmax" (Tensor.ofRaw (RawTensor.logSoftmax! t.raw dim))

/-- Log-softmax for standard real floating tensors. -/
def logSoftmax {σ : StaticSpec} (t : Tensor σ) (dim : Nat := σ.shape.size - 1)
    (_real : σ.dtype.isRealFloating = true := by first | rfl | exact DType.isRealFloating_of_realFloating | decide)
    (_axis : dim < max 1 σ.shape.size := by first | exact of_decide_eq_true rfl | omega | (simp <;> fail "axis out of range")) : Tensor σ :=
  logSoftmax! t dim

/-- Backward from a scalar loss (gradient = 1.0). Saved intermediates are released
by default; omitted retention follows `createGraph`, as in PyTorch. -/
def backwardLoss {d : DType} {dev : Device} (loss : Tensor ⟨#[], d, dev⟩)
    (retainGraph : Option Bool := none) (createGraph : Bool := false) : IO Unit :=
  RawTensor.backwardLoss loss.raw (retainGraph.getD createGraph) createGraph

/-! ## Autograd and fused layer primitives -/

def detach {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  t.checkedReplacement (RawTensor.detach t.raw)

/-- Read a detached gradient snapshot that remains stable across later backward,
    clipping, and zeroing operations. Missing gradients produce zeros. -/
def gradOf {σ : StaticSpec} (t : Tensor σ) : IO (Tensor σ) := do
  let raw ← RawTensor.gradOf t.raw
  match t.checkedReplacement? raw with
  | .ok gradient => pure gradient
  | .error message => throw (IO.userError s!"Tensor.gradOf: {message}")

/-- Clear accumulated gradients. The default releases the gradient and breaks
cycles created by higher-order backward. Pass false to zero an existing buffer. -/
def zeroGrad {σ : StaticSpec} (t : Tensor σ) (setToNone : Bool := true) : IO Unit :=
  RawTensor.zeroGrad t.raw setToNone


/-- Change the autograd flag without changing other aliases. Creating mutable
    gradient identity is effectful, even when the tensor values are equal. -/
def setRequiresGrad {σ : StaticSpec} (t : Tensor σ) (requiresGrad : Bool) :
    IO (Tensor σ) := do
  let dtype := t.raw.dtype
  let device := t.raw.device
  let raw ← RawTensor.setRequiresGrad t.raw requiresGrad
  pure (checkedUpdate dtype device raw)

/-- Create an independent trainable leaf. Detaching first ensures fresh gradient
    identity even when the source is already a parameter. -/
def parameter {σ : StaticSpec} (t : Tensor σ) : IO (Tensor σ) :=
  setRequiresGrad (detach t) true

def affine3d! {batch seq inDim outDim : Nat} {d : DType} {dev : Device}
    (x : Tensor ⟨#[batch, seq, inDim], d, dev⟩)
    (weight : Tensor ⟨#[outDim, inDim], d, dev⟩)
    (bias : Tensor ⟨#[outDim], d, dev⟩) :
    Tensor ⟨#[batch, seq, outDim], d, dev⟩ :=
  checkedResult "affine3d" (Tensor.ofRaw (RawTensor.affine3d! x.raw weight.raw bias.raw))

/-- Fused affine for 3D input: `x @ Wᵀ + b`. -/
def affine3d {batch seq inDim outDim : Nat} {d : DType} {dev : Device} [DType.FloatingOrComplex d]
    (x : Tensor ⟨#[batch, seq, inDim], d, dev⟩)
    (weight : Tensor ⟨#[outDim, inDim], d, dev⟩)
    (bias : Tensor ⟨#[outDim], d, dev⟩) :
    Tensor ⟨#[batch, seq, outDim], d, dev⟩ :=
  affine3d! x weight bias

/-- Fused layer normalization over the last dimension, preserving the spec. -/
def layerNorm {batch seq n : Nat} {d : DType} {dev : Device}
    (x : Tensor ⟨#[batch, seq, n], d, dev⟩)
    (weight : Tensor ⟨#[n], d, dev⟩)
    (bias : Tensor ⟨#[n], d, dev⟩) (eps : Float := 1e-5) :
    Tensor ⟨#[batch, seq, n], d, dev⟩ :=
  x.checkedReplacement (RawTensor.layerNorm x.raw weight.raw bias.raw eps)

/-- Discharges `broadcastsTo` for shapes with symbolic dimensions. -/
macro "broadcast_shape" : tactic => `(tactic| first
  | rfl
  | (simp [TensorSpec.broadcastsTo, TensorSpec.broadcastShapes?, TensorSpec.broadcastShapes?.go,
      List.range_succ] <;> (repeat' split) <;> simp_all <;> fail "shapes do not broadcast"))

def rmsNormWeighted! {σ : StaticSpec} {w : Shape} {result : DType}
    [TensorSpec.Promotes ⟨σ.shape, .Float32⟩ ⟨w, σ.dtype⟩ result]
    (x : Tensor σ) (weight : Tensor (σ.withShape w)) (eps : Float := 1e-6) :
    Tensor (σ.withDType (TensorSpec.pointwiseDType ⟨σ.shape, .Float32⟩ ⟨w, σ.dtype⟩)) :=
  checkedResult "rmsNormWeighted" (Tensor.ofRaw (RawTensor.rmsNormWeighted! x.raw weight.raw eps))

/-- Weighted RMS normalization over the last dimension.

The runtime computes in Float32 before multiplying the weight. A non-scalar
Float64 weight promotes the result; a scalar weight follows weak scalar
promotion. The weight must broadcast to the input shape without expanding it.
Cast the result back explicitly for HF-style dtype restoration. -/
def rmsNormWeighted {σ : StaticSpec} {w : Shape} {result : DType}
    [TensorSpec.Promotes ⟨σ.shape, .Float32⟩ ⟨w, σ.dtype⟩ result]
    (x : Tensor σ) (weight : Tensor (σ.withShape w)) (eps : Float := 1e-6)
    (_float : σ.dtype.isFloating = true := by first | rfl | decide)
    (_weight : TensorSpec.broadcastsTo w σ.shape := by broadcast_shape) :
    Tensor (σ.withDType (TensorSpec.pointwiseDType ⟨σ.shape, .Float32⟩ ⟨w, σ.dtype⟩)) :=
  rmsNormWeighted! x weight eps

end Tensor

/-- Top-k sampling filter size; `none` keeps every token. -/
abbrev TopK := Option PosNat

namespace TopK

/-- The `k` largest logits, or all of them for `none`. -/
def filter {σ : StaticSpec} (topK : TopK) (logits : Tensor σ) : Tensor σ :=
  match topK with
  | none => logits
  | some k => logits.topKFilter k.val

end TopK

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

/-- Positive stride, dilation and kernel, with the dilated window fitting the
padded input, as native convolution requires. -/
def ConvValid (length kernel stride padding dilation : Nat) : Prop :=
  0 < stride ∧ 0 < dilation ∧ 0 < kernel ∧ dilation * (kernel - 1) + 1 ≤ length + 2 * padding

instance (length kernel stride padding dilation : Nat) :
    Decidable (ConvValid length kernel stride padding dilation) := by
  unfold ConvValid; infer_instance

/-- Discharges `ConvValid` for literal parameters, or from facts in scope. -/
macro "conv_valid" : tactic => `(tactic| first
  | exact of_decide_eq_true rfl
  | assumption
  | (refine ⟨?_, ?_, ?_, ?_⟩ <;> omega))

def conv1d! {batch inChannels outChannels length kernel : Nat}
    {d : DType} {dev : Device}
    (input : Tensor ⟨#[batch, inChannels, length], d, dev⟩)
    (weight : Tensor ⟨#[outChannels, inChannels, kernel], d, dev⟩)
    (stride : Nat := 1) (padding : Nat := 0) (dilation : Nat := 1) :
    Tensor ⟨#[batch, outChannels,
      ((length + 2 * padding - (dilation * (kernel - 1) + 1)) /
        stride + 1)], d, dev⟩ :=
  checkedResult "conv1d" (Tensor.ofRaw (RawTensor.conv1d! input.raw weight.raw stride padding dilation))

/-- One-dimensional convolution without bias. -/
def conv1d {batch inChannels outChannels length kernel : Nat}
    {d : DType} {dev : Device} [DType.FloatingOrComplex d]
    (input : Tensor ⟨#[batch, inChannels, length], d, dev⟩)
    (weight : Tensor ⟨#[outChannels, inChannels, kernel], d, dev⟩)
    (stride : Nat := 1) (padding : Nat := 0) (dilation : Nat := 1)
    (_valid : ConvValid length kernel stride padding dilation := by conv_valid) :
    Tensor ⟨#[batch, outChannels,
      ((length + 2 * padding - (dilation * (kernel - 1) + 1)) /
        stride + 1)], d, dev⟩ :=
  conv1d! input weight stride padding dilation

/-- Copy tensor values to a host Float array through the native conversion. -/
def toFloatArray {σ : StaticSpec} (t : Tensor σ) : IO (Array Float) :=
  RawTensor.toFloatArray t.raw

/-- Whether any Boolean entry is true. -/
def any {s : Shape} {dev : Device} (t : Tensor ⟨s, .Bool, dev⟩) : Bool :=
  RawTensor.any t.raw
/-- Copy integral tensor values to host UInt64 values. -/
def toUInt64Array {s : Shape} {dev : Device}
    (t : Tensor ⟨s, .Int64, dev⟩) : IO (Array UInt64) :=
  RawTensor.toUInt64Array t.raw
/-- Uniform integral samples in the half-open interval `[low, high)`. -/
def randint (low high : Int64) (shape : Shape) (device : Device := .CPU) :
    IO (Tensor { shape := shape, dtype := .Int64, device }) := do
  let raw ← RawTensor.randint low high shape device
  match Tensor.ofRaw raw with
  | .ok result => pure result
  | .error message => throw (IO.userError s!"Tensor.randint: {message}")

end torch.Tensor
