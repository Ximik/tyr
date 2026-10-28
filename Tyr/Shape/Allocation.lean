import Tyr.Basic.Types

namespace torch.Shape

/-- Validate runtime dimensions before entering pure native construction.
The native contiguous layout uses signed 64-bit extents and suffix strides;
zero element counts do not exempt the nonzero suffix strides from overflow. -/
def validateAllocation (shape : Shape) : Except String Unit := do
  let limit : Nat := 9223372036854775807
  let mut stride := 1
  for dimension in shape.toList.reverse do
    if dimension > limit then
      throw s!"Tensor extent {dimension} exceeds the native signed 64-bit range"
    if stride > limit then
      throw "Tensor contiguous stride exceeds the native signed 64-bit range"
    stride := stride * max dimension 1
  let elements := shape.foldl (fun count dimension => count * dimension) 1
  if elements > limit then
    throw "Tensor element count exceeds the native signed 64-bit range"

end torch.Shape
