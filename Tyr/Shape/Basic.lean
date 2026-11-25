import Tyr.FFI.Basic.Types

/-! Result shapes of tensor operations. -/

namespace torch.Shape

/-- Insert an axis of extent 1 at `dim`. -/
@[simp] def unsqueeze (s : Shape) (dim : Nat) (h : dim ≤ s.size) : Shape :=
  s.insertIdx dim 1 h

/-- The axis `dim` is removed if it has extent 1. -/
@[simp] def squeeze (s : Shape) (dim : Nat) : Shape :=
  if s[dim]? = some 1 then s.eraseIdxIfInBounds dim else s

/-- The extents at `dim0` and `dim1` are swapped. -/
@[simp] def transpose (s : Shape) (dim0 dim1 : Nat) : Shape :=
  (s.setIfInBounds dim0 (s.getD dim1 0)).setIfInBounds dim1 (s.getD dim0 0)

/-- Reduction along `dim`: the axis becomes 1 with `keepdim`, else it is removed. -/
@[simp] def reduce (s : Shape) (dim : Nat) (keepdim : Bool) : Shape :=
  if keepdim then s.setIfInBounds dim 1 else s.eraseIdxIfInBounds dim

/-- The axis `dim` is removed. -/
@[simp] def unbind (s : Shape) (dim : Nat) (h : dim < s.size) : Shape :=
  s.eraseIdx dim h

/-- Extent of the axis `offset` places from the right; 1 beyond the leading axis. -/
@[simp] def dimFromRight (shape : Shape) (offset : Nat) : Nat :=
  if offset < shape.size then shape.getD (shape.size - 1 - offset) 1 else 1

/-- Aligned from the right, every axis agrees or is 1 on one side. -/
@[simp] def BroadcastCompatible (lhs rhs : Shape) : Prop :=
  ∀ offset ∈ List.range (max lhs.size rhs.size),
    dimFromRight lhs offset = dimFromRight rhs offset ∨
      dimFromRight lhs offset = 1 ∨ dimFromRight rhs offset = 1

instance (lhs rhs : Shape) : Decidable (BroadcastCompatible lhs rhs) := by
  unfold BroadcastCompatible; infer_instance

/-- Mutual broadcast. A scalar operand yields the other shape unchanged, so later
operations can infer its dimensions. -/
@[reducible] def broadcast (lhs rhs : Shape) : Shape :=
  match lhs.toList, rhs.toList with
  | [], _ => rhs
  | _, [] => lhs
  | _, _ => ((List.range (max lhs.size rhs.size)).reverse.map fun offset =>
      if dimFromRight lhs offset = 1 then dimFromRight rhs offset else dimFromRight lhs offset).toArray

/-- `src` stretches to `target` without stretching `target`. -/
@[simp] def BroadcastToCompatible (src target : Shape) : Prop :=
  src.size ≤ target.size ∧ ∀ offset ∈ List.range src.size,
    dimFromRight src offset = dimFromRight target offset ∨ dimFromRight src offset = 1

instance (src target : Shape) : Decidable (BroadcastToCompatible src target) := by
  unfold BroadcastToCompatible; infer_instance

/-- Both operands have rank at least 2, the inner extents agree, and the batch
axes broadcast. -/
@[simp] def MatmulCompatible (s1 s2 : Shape) : Prop :=
  2 ≤ s1.size ∧ 2 ≤ s2.size ∧ s1.getD (s1.size - 1) 0 = s2.getD (s2.size - 2) 0 ∧
    BroadcastCompatible (s1.extract 0 (s1.size - 2)) (s2.extract 0 (s2.size - 2))

instance (s1 s2 : Shape) : Decidable (MatmulCompatible s1 s2) := by
  unfold MatmulCompatible; infer_instance

/-- Matrix multiplication of two operands of rank at least 2:
    `[..., m, k] @ [..., k, n] -> [broadcast(...), m, n]`. Other ranks give `#[]`. -/
@[simp] def matmul (s1 s2 : Shape) : Shape :=
  match s1.size, s2.size with
  | 2, 2 => #[s1.getD 0 0, s2.getD 1 0]
  | n1, n2 =>
    if n1 < 2 || n2 < 2 then #[]
    else
      let batchOut := broadcast (s1.extract 0 (n1 - 2)) (s2.extract 0 (n2 - 2))
      batchOut ++ #[s1.getD (n1 - 2) 0, s2.getD (n2 - 1) 0]

/-- libtorch's `gather` shape rule: `dim` is an axis, the index has the input's
rank, and it is no larger than the input on every other axis. -/
@[simp] def GatherCompatible (input indices : Shape) (dim : Nat) : Prop :=
  dim < input.size ∧ input.size = indices.size ∧
    ∀ p ∈ (input.toList.zip indices.toList).zipIdx, p.2 = dim ∨ p.1.2 ≤ p.1.1

instance (input indices : Shape) (dim : Nat) : Decidable (GatherCompatible input indices dim) := by
  unfold GatherCompatible; infer_instance

/-- `source` can replace a slice of `input` along `dim`: `dim` is an axis and
the shapes agree on every other axis. The slice bound is not part of this. -/
@[simp] def SliceScatterCompatible (input source : Shape) (dim : Nat) : Prop :=
  dim < input.size ∧ input.size = source.size ∧
    ∀ p ∈ (input.toList.zip source.toList).zipIdx, p.2 = dim ∨ p.1.1 = p.1.2

instance (input source : Shape) (dim : Nat) : Decidable (SliceScatterCompatible input source dim) := by
  unfold SliceScatterCompatible; infer_instance

/-- `permutation` lists every axis below `rank` exactly once. -/
@[simp] def IsPermutation (permutation : Array Nat) (rank : Nat) : Prop :=
  permutation.size = rank ∧ ∀ i ∈ List.range rank, i ∈ permutation.toList

instance (permutation : Array Nat) (rank : Nat) : Decidable (IsPermutation permutation rank) := by
  unfold IsPermutation; infer_instance

/-- Shapes concatenate along `dim` when it is an axis of both, and they agree on
every other axis. libtorch's legacy skipping of an empty 1-D operand is excluded. -/
@[simp] def CatCompatible (s1 s2 : Shape) (dim : Nat) : Prop :=
  dim < s1.size ∧ s1.size = s2.size ∧
    ∀ p ∈ (s1.toList.zip s2.toList).zipIdx, p.2 = dim ∨ p.1.1 = p.1.2

instance (s1 s2 : Shape) (dim : Nat) : Decidable (CatCompatible s1 s2 dim) := by
  unfold CatCompatible; infer_instance

/-- Concatenation along `dim`. It takes no compatibility proof: a proof term in
the result type makes failing `decide` attempts on later shape goals several
times more expensive. -/
@[simp] def cat (s1 s2 : Shape) (dim : Nat) : Shape :=
  s1.set! dim (s1.getD dim 0 + s2.getD dim 0)

/-- Dimensions reordered by `permutation`. Like `cat`, it takes no proof. -/
@[simp] def permute (s : Shape) (permutation : Array Nat) : Shape :=
  permutation.map fun p => s.getD p 0

@[simp] def numel (shape : Shape) : Nat :=
  shape.foldl (fun acc dim => acc * dim) 1

/-- Output length of a convolution along one spatial axis. -/
def convLength (input kernel : Nat) (stride : Nat := 1) (padding : Nat := 0)
    (dilation : Nat := 1) : Nat :=
  (input + 2 * padding - (dilation * (kernel - 1) + 1)) / stride + 1

/-- A stride-1 convolution padded by half its odd kernel keeps any nonempty extent. -/
theorem convLength_same {n kernel padding : Nat} (hn : 0 < n) (hk : 2 * padding + 1 = kernel) :
    convLength n kernel 1 padding 1 = n := by
  unfold convLength; omega

/-- Output length of a transposed convolution along one spatial axis; the
padding is subtracted last so a short input is not truncated early. -/
def convTransposeLength (input kernel : Nat) (stride : Nat := 1) (padding : Nat := 0)
    (outputPadding : Nat := 0) (dilation : Nat := 1) : Nat :=
  (input - 1) * stride + dilation * (kernel - 1) + outputPadding + 1 - 2 * padding

end torch.Shape
