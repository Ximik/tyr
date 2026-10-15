import Tyr.Basic

/-!
# PyTorch dtype algebra

These rules describe the pinned PyTorch 2.10 scalar types, including types
with restricted kernel support. Promotion is partial: `none` means PyTorch
has no result dtype for the pair. `Tyr.Typed.Promotion` turns successful
results into erased, type-level witnesses used by typed arithmetic.
-/

namespace torch.DType

@[reducible] def isKnown : DType → _root_.Bool
  | .Unknown _ => false
  | _ => true

@[reducible] def isBool : DType → _root_.Bool
  | .Bool => true
  | _ => false

/-- Integral classification follows c10; sub-byte shell types are separate. -/
@[reducible] def isUnsignedIntegral : DType → _root_.Bool
  | .UInt8 | .UInt16 | .UInt32 | .UInt64 => true
  | _ => false

@[reducible] def isSignedIntegral : DType → _root_.Bool
  | .Int8 | .Int16 | .Int32 | .Int64 => true
  | _ => false

@[reducible] def isIntegral (dtype : DType) : _root_.Bool :=
  dtype.isUnsignedIntegral || dtype.isSignedIntegral

@[reducible] def isFloat8 : DType → _root_.Bool
  | .Float8E4M3FN | .Float8E5M2 | .Float8E4M3FNUZ | .Float8E5M2FNUZ
  | .Float8E8M0FNU => true
  | _ => false

/-- Floating classification does not imply that a backend implements math
kernels for this dtype. In particular Float8 and packed Float4 are shell types. -/
@[reducible] def isFloating (dtype : DType) : _root_.Bool :=
  match dtype with
  | .Float16 | .BFloat16 | .Float32 | .Float64 | .Float4E2M1FNX2 => true
  | _ => dtype.isFloat8

/-- Standard real floating dtypes, excluding complex and shell formats. This
domain does not imply that a particular backend implements every operation. -/
@[reducible] def isRealFloating : DType → _root_.Bool
  | .Float16 | .BFloat16 | .Float32 | .Float64 => true
  | _ => false

/-- Erased evidence that a dtype belongs to the standard real floating domain. -/
class RealFloating (dtype : DType) : Prop where
  valid : dtype.isRealFloating = true

instance : RealFloating .Float16 := ⟨rfl⟩
instance : RealFloating .BFloat16 := ⟨rfl⟩
instance : RealFloating .Float32 := ⟨rfl⟩
instance : RealFloating .Float64 := ⟨rfl⟩

/-- Validate a runtime-selected dtype before using real floating algorithms. -/
def realFloating? (dtype : DType) : Except String (PLift (RealFloating dtype)) :=
  if valid : dtype.isRealFloating = true then .ok ⟨⟨valid⟩⟩
  else .error s!"Expected a standard real floating dtype, got {dtype}"

@[simp] theorem isRealFloating_of_realFloating {dtype : DType} [h : RealFloating dtype] :
    dtype.isRealFloating = true := h.valid

@[reducible] def isComplex : DType → _root_.Bool
  | .Complex32 | .Complex64 | .Complex128 => true
  | _ => false

/-- Standard real or complex floating domain. Kernel availability remains
backend-specific and checked native APIs report unsupported implementations. -/
class FloatingOrComplex (dtype : DType) : Prop where
  valid : (dtype.isRealFloating || dtype.isComplex) = true

instance {dtype : DType} [h : RealFloating dtype] : FloatingOrComplex dtype :=
  ⟨by rw [h.valid]; rfl⟩
instance : FloatingOrComplex .Complex32 := ⟨rfl⟩
instance : FloatingOrComplex .Complex64 := ⟨rfl⟩
instance : FloatingOrComplex .Complex128 := ⟨rfl⟩

@[reducible] def isQuantized : DType → _root_.Bool
  | .QInt8 | .QUInt8 | .QInt32 | .QUInt4x2 | .QUInt2x4 => true
  | _ => false

@[reducible] def isBits : DType → _root_.Bool
  | .Bits1x8 | .Bits2x4 | .Bits4x2 | .Bits8 | .Bits16 => true
  | _ => false

@[reducible] def isBarebonesUnsigned : DType → _root_.Bool
  | .UInt1 | .UInt2 | .UInt3 | .UInt4 | .UInt5 | .UInt6 | .UInt7
  | .UInt16 | .UInt32 | .UInt64 => true
  | _ => false

/-- The 13 scalar types in PyTorch's ordinary promotion table. -/
@[reducible] def isStandard (dtype : DType) : _root_.Bool :=
  match dtype with
  | .Bool | .UInt8 | .Int8 | .Int16 | .Int32 | .Int64
  | .Float16 | .BFloat16 | .Float32 | .Float64
  | .Complex32 | .Complex64 | .Complex128 => true
  | _ => false

@[reducible] def isInexact (dtype : DType) : _root_.Bool :=
  dtype.isFloating || dtype.isComplex

@[reducible] def isDifferentiable : DType → _root_.Bool := isInexact

@[reducible] def isIndex : DType → _root_.Bool
  | .UInt8 | .Int32 | .Int64 => true
  | _ => false

/-- Logical precision in bits, rather than the size of a packed storage cell.
For opaque bits containers the width describes the whole container. -/
def bitWidth? : DType → Option Nat
  | .Bool | .UInt1 | .Int1 => some 1
  | .UInt2 | .Int2 | .QUInt2x4 => some 2
  | .UInt3 | .Int3 => some 3
  | .UInt4 | .Int4 | .QUInt4x2 | .Float4E2M1FNX2 => some 4
  | .UInt5 | .Int5 => some 5
  | .UInt6 | .Int6 => some 6
  | .UInt7 | .Int7 => some 7
  | .UInt8 | .Int8 | .QInt8 | .QUInt8
  | .Float8E4M3FN | .Float8E5M2 | .Float8E4M3FNUZ | .Float8E5M2FNUZ
  | .Float8E8M0FNU | .Bits1x8 | .Bits2x4 | .Bits4x2 | .Bits8 => some 8
  | .Int16 | .UInt16 | .Float16 | .BFloat16 | .Bits16 => some 16
  | .Int32 | .UInt32 | .Float32 | .Complex32 | .QInt32 => some 32
  | .Int64 | .UInt64 | .Float64 | .Complex64 => some 64
  | .Complex128 => some 128
  | .Unknown _ => none

/-- Integer true division uses Tyr's Float32 default. Floating and complex
inputs retain their dtype; kernel availability is a separate precondition. -/
@[reducible] def atLeastFloat : DType → DType
  | .Bool | .UInt8 | .UInt16 | .UInt32 | .UInt64
  | .Int8 | .Int16 | .Int32 | .Int64 => .Float32
  | dtype => dtype

@[reducible] private def promoteIntegral : DType → DType → DType
  | .Int64, _ | _, .Int64 => .Int64
  | .Int32, _ | _, .Int32 => .Int32
  | .Int16, _ | _, .Int16 => .Int16
  | .UInt8, .Int8 | .Int8, .UInt8 => .Int16
  | .Int8, _ | _, .Int8 => .Int8
  | .UInt8, _ | _, .UInt8 => .UInt8
  | _, _ => .Bool

@[reducible] private def promoteStandard : DType → DType → DType
  | .Complex128, _ | _, .Complex128 => .Complex128
  | .Complex64, .Float64 | .Float64, .Complex64
  | .Complex32, .Float64 | .Float64, .Complex32 => .Complex128
  | .Complex64, _ | _, .Complex64 => .Complex64
  | .Complex32, .Float32 | .Float32, .Complex32
  | .Complex32, .BFloat16 | .BFloat16, .Complex32 => .Complex64
  | .Complex32, _ | _, .Complex32 => .Complex32
  | .Float64, _ | _, .Float64 => .Float64
  | .Float32, _ | _, .Float32 => .Float32
  | .Float16, .BFloat16 | .BFloat16, .Float16 => .Float32
  | .Float16, _ | _, .Float16 => .Float16
  | .BFloat16, _ | _, .BFloat16 => .BFloat16
  | lhs, rhs => promoteIntegral lhs rhs

/-- Constructor equality that reduces during promotion instance synthesis. -/
@[reducible] private def sameKnownDType : DType → DType → _root_.Bool
  | .UInt8, .UInt8 => true
  | .Int8, .Int8 => true
  | .Int16, .Int16 => true
  | .Int32, .Int32 => true
  | .Int64, .Int64 => true
  | .Float16, .Float16 => true
  | .BFloat16, .BFloat16 => true
  | .Float32, .Float32 => true
  | .Float64, .Float64 => true
  | .Bool, .Bool => true
  | .Float8E4M3FN, .Float8E4M3FN => true
  | .Float8E5M2, .Float8E5M2 => true
  | .Complex32, .Complex32 => true
  | .Complex64, .Complex64 => true
  | .Complex128, .Complex128 => true
  | .QInt8, .QInt8 => true
  | .QUInt8, .QUInt8 => true
  | .QInt32, .QInt32 => true
  | .QUInt4x2, .QUInt4x2 => true
  | .QUInt2x4, .QUInt2x4 => true
  | .Bits1x8, .Bits1x8 => true
  | .Bits2x4, .Bits2x4 => true
  | .Bits4x2, .Bits4x2 => true
  | .Bits8, .Bits8 => true
  | .Bits16, .Bits16 => true
  | .Float8E5M2FNUZ, .Float8E5M2FNUZ => true
  | .Float8E4M3FNUZ, .Float8E4M3FNUZ => true
  | .UInt16, .UInt16 => true
  | .UInt32, .UInt32 => true
  | .UInt64, .UInt64 => true
  | .UInt1, .UInt1 => true
  | .UInt2, .UInt2 => true
  | .UInt3, .UInt3 => true
  | .UInt4, .UInt4 => true
  | .UInt5, .UInt5 => true
  | .UInt6, .UInt6 => true
  | .UInt7, .UInt7 => true
  | .Int1, .Int1 => true
  | .Int2, .Int2 => true
  | .Int3, .Int3 => true
  | .Int4, .Int4 => true
  | .Int5, .Int5 => true
  | .Int6, .Int6 => true
  | .Int7, .Int7 => true
  | .Float8E8M0FNU, .Float8E8M0FNU => true
  | .Float4E2M1FNX2, .Float4E2M1FNX2 => true
  | _, _ => false

/-- Exact result of `torch.promote_types` for every known dtype in PyTorch
2.10. Undefined results and native rejections are both `none`. Unknown tokens,
including the dynamic wildcard, never establish a static promotion.
Constructor matches keep this reducible during typeclass synthesis.
No arithmetic kernel availability is implied by a successful result. -/
@[reducible] def promote? (lhs rhs : DType) : Option DType :=
  match lhs, rhs with
  | .Unknown _, _ | _, .Unknown _ => none
  | _, _ =>
    match sameKnownDType lhs rhs with
    | true => some lhs
    | false =>
      match lhs.isQuantized, rhs.isQuantized, lhs.isBits, rhs.isBits,
          lhs.isFloat8, rhs.isFloat8 with
      | true, _, _, _, _, _ | _, true, _, _, _, _
      | _, _, true, _, _, _ | _, _, _, true, _, _
      | _, _, _, _, true, _ | _, _, _, _, _, true => none
      | _, _, _, _, _, _ =>
        match lhs.isBarebonesUnsigned, rhs.isBarebonesUnsigned with
        | false, false =>
          match lhs.isStandard, rhs.isStandard with
          | true, true => some (promoteStandard lhs rhs)
          | _, _ => none
        | _, _ =>
          match lhs.isFloating, rhs.isFloating with
          | true, _ => some lhs
          | _, true => some rhs
          | _, _ => none

@[reducible] def toComplex? : DType → Option DType
  | .Float16 | .Complex32 => some .Complex32
  | .BFloat16 | .Float32 | .Complex64 => some .Complex64
  | .Float64 | .Complex128 => some .Complex128
  | _ => none

/-- Result-type combination for a dimensioned tensor and a zero-dimensional
(non-wrapped) tensor. The dimensioned operand determines precision within its
category. This is distinct from `promote?`, and deliberately does not claim
that an arithmetic kernel exists for the result. -/
@[reducible] def promoteScalar? (tensor scalar : DType) : Option DType :=
  match tensor, scalar with
  | .Unknown _, _ | _, .Unknown _ => none
  | _, _ =>
    match tensor.isComplex, scalar.isComplex with
    | true, _ => some tensor
    | false, true =>
      match tensor.isFloating with
      | true => tensor.toComplex?
      | false => some scalar
    | false, false =>
      match tensor.isFloating, tensor.isBool, scalar.isFloating with
      | true, _, _ => some tensor
      | false, true, _ | false, _, true => promote? tensor scalar
      | _, _, _ => some tensor

/-- The real component dtype of a complex tensor. Magnitudes, norms, and
singular values use this dtype; non-complex dtypes are unchanged. -/
@[reducible] def realResult : DType → DType
  | .Complex32 => .Float16
  | .Complex64 => .Float32
  | .Complex128 => .Float64
  | dtype => dtype

/-- Complex absolute values are real. -/
@[reducible] def absResult : DType → DType := realResult

@[reducible] def sumResult : DType → DType
  | .Bool | .UInt8 | .UInt16 | .UInt32 | .UInt64
  | .Int8 | .Int16 | .Int32 | .Int64 => .Int64
  | dtype => dtype

@[reducible] def prodResult : DType → DType := sumResult

/-- Mean does not implicitly convert integer tensors; such inputs must be
rejected by the operation or converted explicitly. -/
@[reducible] def meanResult (dtype : DType) : DType := dtype

@[reducible] def divideResult (dtype : DType) : DType := dtype.atLeastFloat

def canRepresentSafeTensors (dtype : DType) : _root_.Bool :=
  dtype.safeTensorTag?.isSome

def expectedEq (expected actual : DType) : Except String Unit :=
  if actual == expected then .ok ()
  else .error s!"Expected dtype {expected}, got {actual}"


/-! Real floating computations retain their dtype where the natural result
rule says so. These equalities only rewrite indices; they never cast values. -/

@[simp] theorem isFloating_of_realFloating {dtype : DType} [h : RealFloating dtype] :
    dtype.isFloating = true := by
  cases dtype <;> first | rfl | cases h.valid

@[simp] theorem isInexact_of_realFloating {dtype : DType} [h : RealFloating dtype] :
    dtype.isInexact = true := by
  cases dtype <;> first | rfl | cases h.valid

@[simp] theorem isDifferentiable_of_realFloating {dtype : DType} [h : RealFloating dtype] :
    dtype.isDifferentiable = true := by
  cases dtype <;> first | rfl | cases h.valid

@[simp] theorem isComplex_of_realFloating {dtype : DType} [h : RealFloating dtype] :
    dtype.isComplex = false := by
  cases dtype <;> first | rfl | cases h.valid

@[simp] theorem isBool_of_realFloating {dtype : DType} [h : RealFloating dtype] :
    dtype.isBool = false := by
  cases dtype <;> first | rfl | cases h.valid

@[simp] theorem atLeastFloat_of_realFloating {dtype : DType} [h : RealFloating dtype] :
    dtype.atLeastFloat = dtype := by
  cases dtype <;> first | rfl | cases h.valid

@[simp] theorem realResult_of_realFloating {dtype : DType} [h : RealFloating dtype] :
    dtype.realResult = dtype := by
  cases dtype <;> first | rfl | cases h.valid

@[simp] theorem absResult_of_realFloating {dtype : DType} [h : RealFloating dtype] :
    dtype.absResult = dtype := by
  cases dtype <;> first | rfl | cases h.valid

@[simp] theorem sumResult_of_realFloating {dtype : DType} [h : RealFloating dtype] :
    dtype.sumResult = dtype := by
  cases dtype <;> first | rfl | cases h.valid

@[simp] theorem prodResult_of_realFloating {dtype : DType} [h : RealFloating dtype] :
    dtype.prodResult = dtype := by
  cases dtype <;> first | rfl | cases h.valid

@[simp] theorem divideResult_of_realFloating {dtype : DType} [h : RealFloating dtype] :
    dtype.divideResult = dtype := by
  cases dtype <;> first | rfl | cases h.valid


instance {dtype : DType} [h : RealFloating dtype] : RealFloating dtype.atLeastFloat := by
  simpa only [atLeastFloat_of_realFloating] using h

instance {dtype : DType} [h : RealFloating dtype] : RealFloating dtype.realResult := by
  simpa only [realResult_of_realFloating] using h

instance {dtype : DType} [h : RealFloating dtype] : RealFloating dtype.absResult := by
  simpa only [absResult_of_realFloating] using h

instance {dtype : DType} [h : RealFloating dtype] : RealFloating dtype.sumResult := by
  simpa only [sumResult_of_realFloating] using h

instance {dtype : DType} [h : RealFloating dtype] : RealFloating dtype.prodResult := by
  simpa only [prodResult_of_realFloating] using h

instance {dtype : DType} [h : RealFloating dtype] : RealFloating dtype.divideResult := by
  simpa only [divideResult_of_realFloating] using h

end torch.DType
