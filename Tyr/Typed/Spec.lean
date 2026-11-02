import Tyr.Typed.Promotion

/-!
# Typed tensor specs

A small, LeanMLX-inspired tensor spec layer for Tyr: shape plus dtype, with
pure result-spec algebra for common tensor operations.
-/

namespace torch

structure TensorSpec where
  shape : Shape
  dtype : DType
deriving Repr, BEq, DecidableEq

namespace TensorSpec

/-- libtorch's `gather` shape rule: `dim` is an axis, the index has the input's
rank, and it is no larger than the input on every other axis. -/
@[simp] def gatherCompatible (input indices : Shape) (dim : Nat) : Bool :=
  dim < input.size && input.size == indices.size &&
    (input.toList.zip indices.toList).zipIdx.all fun ((a, b), i) => i == dim || decide (b ≤ a)

/-- `source` can replace a slice of `input` along `dim`: `dim` is an axis and
the shapes agree on every other axis. The slice bound is not part of this. -/
@[simp] def sliceScatterCompatible (input source : Shape) (dim : Nat) : Bool :=
  dim < input.size && input.size == source.size &&
    (input.toList.zip source.toList).zipIdx.all fun ((a, b), i) => i == dim || a == b

/-- `permutation` lists every axis below `rank` exactly once. -/
@[simp] def isPermutation (permutation : Array Nat) (rank : Nat) : Bool :=
  permutation.size == rank && (List.range rank).all permutation.toList.contains

/-- Reorder dimensions for a permutation. Like `catShape`, it takes no proof;
`Tensor.permute` requires `isPermutation`. -/
@[simp] def permuteShape (s : Shape) (permutation : Array Nat) : Shape :=
  permutation.map fun p => s.getD p 0

/-- Shapes concatenate along `dim` when it is an axis of both, and they agree on
every other axis. libtorch's legacy skipping of an empty 1-D operand is excluded. -/
@[simp] def catCompatible (s1 s2 : Shape) (dim : Nat) : Bool :=
  dim < s1.size && s1.size == s2.size &&
    (s1.toList.zip s2.toList).zipIdx.all fun ((a, b), i) => i == dim || a == b

/-- Shape after concatenation along `dim`. It takes no compatibility proof:
`Tensor.cat` requires one, and a proof term in the result type makes failing
`decide` attempts on later shape goals several times more expensive. -/
@[simp] def catShape (s1 s2 : Shape) (dim : Nat) : Shape :=
  s1.set! dim (s1.getD dim 0 + s2.getD dim 0)

def scalar (dtype : DType) : TensorSpec :=
  { shape := #[], dtype }

def rank (spec : TensorSpec) : Nat :=
  spec.shape.size

@[simp] def numelOfShape (shape : Shape) : Nat :=
  shape.foldl (fun acc dim => acc * dim) 1

def numel (spec : TensorSpec) : Nat :=
  numelOfShape spec.shape

def withShape (spec : TensorSpec) (shape : Shape) : TensorSpec :=
  { spec with shape }

def withDType (spec : TensorSpec) (dtype : DType) : TensorSpec :=
  { spec with dtype }

@[simp] def dimFromRight (shape : Shape) (offset : Nat) : Nat :=
  if offset < shape.size then
    shape.getD (shape.size - 1 - offset) 1
  else
    1

def broadcastShapes? (lhs rhs : Shape) : Option Shape :=
  if lhs.isEmpty then some rhs
  else if rhs.isEmpty then some lhs
  else
    let maxLen := Nat.max lhs.size rhs.size
    let rec go : List Nat → List Nat → Option Shape
      | [], revOut => some revOut.toArray
      | offset :: rest, revOut =>
          let ldim := dimFromRight lhs offset
          let rdim := dimFromRight rhs offset
          if ldim == rdim then
            go rest (ldim :: revOut)
          else if ldim == 1 then
            go rest (rdim :: revOut)
          else if rdim == 1 then
            go rest (ldim :: revOut)
          else
            none
    go (List.range maxLen) []

def broadcastsTo (src dst : Shape) : Prop :=
  broadcastShapes? src dst = some dst

/-- n-ary mutual broadcast: fold the binary broadcast over all shapes.
    `none` for the empty list or any incompatible pair. -/
def broadcastList? : List Shape → Option Shape
  | [] => none
  | shape :: rest => rest.foldlM broadcastShapes? shape

/-- Total broadcast-shape function for use in type indices (`#[]` junk on
    incompatible shapes — callers rule that out with a
    `(broadcastShapes? s1 s2).isSome` proof). Scalar broadcasting retains the
    other shape directly, so subsequent operations can infer its dimensions. -/
@[reducible] def broadcastShape (lhs rhs : Shape) : Shape :=
  match lhs.toList, rhs.toList with
  | [], _ => rhs
  | _, [] => lhs
  | _, _ => (broadcastShapes? lhs rhs).getD #[]

/-- Dtype promotion distinguishes zero-dimensional tensor operands from
non-scalars, whose precision takes precedence within a dtype category.
Unsupported dtype combinations have no result. -/
@[reducible] def pointwiseDType? (lhs rhs : TensorSpec) : Option DType :=
  match lhs.shape.toList, rhs.shape.toList with
  | [], _ :: _ => DType.promoteScalar? rhs.dtype lhs.dtype
  | _ :: _, [] => DType.promoteScalar? lhs.dtype rhs.dtype
  | _, _ => DType.promote? lhs.dtype rhs.dtype

/-- Evidence of the pointwise dtype result, including weak scalar promotion.
This does not establish shape compatibility or native kernel availability. -/
class Promotes (lhs rhs : TensorSpec) (result : outParam DType) : Prop where
  valid : pointwiseDType? lhs rhs = some result

instance {lhs rhs : TensorSpec} {result : DType}
    [h : SomeResult (pointwiseDType? lhs rhs) result] : Promotes lhs rhs result :=
  ⟨h.valid⟩

/-- A pointwise dtype index backed by a successful promotion witness. -/
@[reducible] def pointwiseDType (lhs rhs : TensorSpec) {result : DType}
    [Promotes lhs rhs result] : DType := result

def pointwise? (lhs rhs : TensorSpec) : Option TensorSpec := do
  let shape ← broadcastShapes? lhs.shape rhs.shape
  let dtype ← pointwiseDType? lhs rhs
  pure { shape, dtype }

def division? (lhs rhs : TensorSpec) : Option TensorSpec := do
  let shape ← broadcastShapes? lhs.shape rhs.shape
  let dtype ← pointwiseDType? lhs rhs
  pure { shape, dtype := DType.divideResult dtype }

@[simp] def matmulShape? (lhs rhs : Shape) : Option Shape :=
  match lhs.size, rhs.size with
  | 0, _ => none
  | _, 0 => none
  | 1, 1 =>
      if lhs.getD 0 0 == rhs.getD 0 0 then some #[] else none
  | 1, 2 =>
      if lhs.getD 0 0 == rhs.getD 0 0 then some #[rhs.getD 1 0] else none
  | 2, 1 =>
      if lhs.getD 1 0 == rhs.getD 0 0 then some #[lhs.getD 0 0] else none
  | 2, 2 =>
      if lhs.getD 1 0 == rhs.getD 0 0 then some #[lhs.getD 0 0, rhs.getD 1 0] else none
  | 1, n₂ =>
      if lhs.getD 0 0 == rhs.getD (n₂ - 2) 0 then
        some <| rhs.extract 0 (n₂ - 2) ++ #[rhs.getD (n₂ - 1) 0]
      else
        none
  | n₁, 1 =>
      if lhs.getD (n₁ - 1) 0 == rhs.getD 0 0 then
        some <| lhs.extract 0 (n₁ - 2) ++ #[lhs.getD (n₁ - 2) 0]
      else
        none
  | n₁, n₂ =>
      if lhs.getD (n₁ - 1) 0 != rhs.getD (n₂ - 2) 0 then
        none
      else
        match broadcastShapes? (lhs.extract 0 (n₁ - 2)) (rhs.extract 0 (n₂ - 2)) with
        | some batch => some <| batch ++ #[lhs.getD (n₁ - 2) 0, rhs.getD (n₂ - 1) 0]
        | none => none

/-- Matrix multiplication requires equal input dtypes. A valid result spec
does not imply the selected backend provides a kernel for that dtype. -/
def matmul? (lhs rhs : TensorSpec) : Option TensorSpec := do
  let shape ← matmulShape? lhs.shape rhs.shape
  if lhs.dtype != rhs.dtype then none
  else
    let dtype ← DType.promote? lhs.dtype rhs.dtype
    pure { shape, dtype }

def sum (spec : TensorSpec) : TensorSpec :=
  { shape := #[], dtype := DType.sumResult spec.dtype }

def mean (spec : TensorSpec) : TensorSpec :=
  { shape := #[], dtype := DType.meanResult spec.dtype }

def prod (spec : TensorSpec) : TensorSpec :=
  { shape := #[], dtype := DType.prodResult spec.dtype }

def checkShape (expected actual : Shape) : Except String Unit :=
  if actual == expected then
    .ok ()
  else
    .error s!"Expected shape {reprStr expected}, got {reprStr actual}"

def checkDType (expected actual : DType) : Except String Unit :=
  DType.expectedEq expected actual

def checkCompatible (expected actual : TensorSpec) : Except String Unit := do
  checkShape expected.shape actual.shape
  checkDType expected.dtype actual.dtype

end TensorSpec

end torch
