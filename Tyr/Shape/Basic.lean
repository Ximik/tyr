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

/-- Broadcast two batch shapes (everything except the last 2 axes) following PyTorch rules. -/
private def broadcastBatch (s1 s2 : Shape) : Shape :=
  let n1 := s1.size
  let n2 := s2.size
  let maxLen := max n1 n2
  Array.ofFn fun (i : Fin maxLen) =>
    let idx1 := if i.val < maxLen - n1 then none else some (i.val - (maxLen - n1))
    let idx2 := if i.val < maxLen - n2 then none else some (i.val - (maxLen - n2))
    match idx1, idx2 with
    | none, none => 1
    | some j, none => s1.getD j 1
    | none, some j => s2.getD j 1
    | some j1, some j2 =>
      let d1 := s1.getD j1 1
      let d2 := s2.getD j2 1
      if d1 = 1 then d2 else if d2 = 1 then d1 else max d1 d2

/-- Matrix multiplication of two operands of rank at least 2:
    `[..., m, k] @ [..., k, n] -> [broadcast(...), m, n]`. Other ranks give `#[]`. -/
@[simp] def matmul (s1 s2 : Shape) : Shape :=
  match s1.size, s2.size with
  | 2, 2 => #[s1.getD 0 0, s2.getD 1 0]
  | n1, n2 =>
    if n1 < 2 || n2 < 2 then #[]
    else
      let batchOut := broadcastBatch (s1.extract 0 (n1 - 2)) (s2.extract 0 (n2 - 2))
      batchOut ++ #[s1.getD (n1 - 2) 0, s2.getD (n2 - 1) 0]

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
