/-!
# Tyr.Basic.Types

The pure part of `Tyr.Basic`: `Shape`, `DType`, `Device`, the opaque tensor
carrier `T`, and shape-transform utilities. It declares no native functions, so
the GPU code generator can use it without linking `libTyrC`. `Tyr.Basic` adds
the tensor metadata functions implemented in `libTyrC`.
-/

namespace torch

abbrev Shape := Array UInt64

inductive DType where
| UInt8
| Int8
| Int16
| Int32
| Int64
| Float16
| BFloat16
| Float32
| Float64
| Bool
| Float8E4M3FN
| Float8E5M2
| Unknown (raw : String)
deriving Repr, Inhabited, BEq, DecidableEq

private def normalizeDTypeToken (raw : String) : String :=
  String.ofList <| raw.toList.map Char.toLower

/-- Parse a dtype token used by Tyr, PyTorch metadata, or SafeTensors headers. -/
def DType.ofString? (raw : String) : Option DType :=
  match normalizeDTypeToken raw with
  | "u8" | "uint8" => some .UInt8
  | "i8" | "int8" => some .Int8
  | "i16" | "int16" => some .Int16
  | "i32" | "int32" => some .Int32
  | "i64" | "int64" | "long" => some .Int64
  | "f16" | "float16" | "half" => some .Float16
  | "bf16" | "bfloat16" => some .BFloat16
  | "f32" | "float32" | "float" => some .Float32
  | "f64" | "float64" | "double" => some .Float64
  | "bool" => some .Bool
  | "f8_e4m3" | "float8_e4m3fn" => some .Float8E4M3FN
  | "f8_e5m2" | "float8_e5m2" => some .Float8E5M2
  | _ => none

/-- Parse a dtype token, preserving unknown tokens in `DType.Unknown`. -/
def DType.parse (raw : String) : DType :=
  (DType.ofString? raw).getD (.Unknown raw)

/-- Canonical name for a dtype in Tyr metadata. -/
def DType.canonicalName : DType → String
  | .UInt8 => "UInt8"
  | .Int8 => "Int8"
  | .Int16 => "Int16"
  | .Int32 => "Int32"
  | .Int64 => "Int64"
  | .Float16 => "Float16"
  | .BFloat16 => "BFloat16"
  | .Float32 => "Float32"
  | .Float64 => "Float64"
  | .Bool => "Bool"
  | .Float8E4M3FN => "Float8E4M3FN"
  | .Float8E5M2 => "Float8E5M2"
  | .Unknown raw =>
      if raw.isEmpty then "Unknown" else s!"Unknown({raw})"

/-- SafeTensors dtype tag for this dtype, if representable in the format. -/
def DType.safeTensorTag? : DType → Option String
  | .UInt8 => some "U8"
  | .Int8 => some "I8"
  | .Int16 => some "I16"
  | .Int32 => some "I32"
  | .Int64 => some "I64"
  | .Float16 => some "F16"
  | .BFloat16 => some "BF16"
  | .Float32 => some "F32"
  | .Float64 => some "F64"
  | .Bool => some "BOOL"
  | .Float8E4M3FN => some "F8_E4M3"
  | .Float8E5M2 => some "F8_E5M2"
  | .Unknown _ => none

instance : ToString DType where
  toString := DType.canonicalName

inductive Device where
| CUDA : UInt64 → Device
| CPU
| MPS
deriving Repr, Inhabited, BEq

opaque TSpec : NonemptyType
def T (_ : Shape) : Type :=  TSpec.type

/-! ## Shape Manipulation Helpers -/

/-- Compute output shape for unsqueeze: insert a dimension of size 1 at position `dim` -/
def unsqueezeShape (s : Shape) (dim : Nat) : Shape :=
  if dim > s.size then s
  else s[:dim].toArray ++ #[1] ++ s[dim:].toArray

/-- Compute output shape for squeeze: remove dimension at `dim` if it has size 1 -/
def squeezeShape (s : Shape) (dim : Nat) : Shape :=
  if h : dim < s.size then
    if s[dim]'h = 1 then s[:dim].toArray ++ s[dim+1:].toArray else s
  else s

/-- Compute output shape for transpose: swap dimensions dim0 and dim1 -/
def transposeShape (s : Shape) (dim0 dim1 : Nat) : Shape :=
  if h0 : dim0 < s.size then
    if h1 : dim1 < s.size then
      let v0 := s[dim0]'h0
      let v1 := s[dim1]'h1
      let s' := s.set (Fin.mk dim0 h0) v1
      have h1' : dim1 < s'.size := by rw [Array.size_set]; exact h1
      s'.set (Fin.mk dim1 h1') v0
    else s
  else s

/-- Compute output shape for reduction along a dimension -/
def reduceShape (s : Shape) (dim : Nat) (keepdim : Bool) : Shape :=
  if h : dim < s.size then
    if keepdim then s.set (Fin.mk dim h) 1
    else s[:dim].toArray ++ s[dim+1:].toArray
  else s

def replaceAtDim (s : Shape) (dim : Nat) (newSize : UInt64) : Shape :=
  if h : dim < s.size then s.set (Fin.mk dim h) newSize else s

/-- Compute output shape for stacking: insert a new dimension of size `n` at `dim` -/
def stackShape (s : Shape) (n : Nat) (dim : Nat) : Shape :=
  if dim > s.size then s
  else s[:dim].toArray ++ #[n.toUInt64] ++ s[dim:].toArray

/-- Compute output shape for unbinding: remove dimension at `dim` -/
def unbindShape (s : Shape) (dim : Nat) : Shape :=
  if dim < s.size then
    s[:dim].toArray ++ s[dim+1:].toArray
  else s

/-- Broadcast two batch shapes (everything except last 2 dims) following PyTorch rules -/
private def broadcastBatchShapes (s1 s2 : Shape) : Shape :=
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

/-- Compute output shape for matrix multiplication following PyTorch broadcasting rules.
    - 1D @ 1D: dot product -> scalar []
    - 2D @ 2D: [m,k] @ [k,n] -> [m,n]
    - 1D @ 2D: [k] @ [k,n] -> [n]
    - 2D @ 1D: [m,k] @ [k] -> [m]
    - ND @ ND: broadcast batch dims, matmul last 2 dims -/
def matmulShape (s1 s2 : Shape) : Shape :=
  match s1.size, s2.size with
  | 0, _ => #[]  -- scalar, invalid but return empty
  | _, 0 => #[]  -- scalar, invalid but return empty
  | 1, 1 => #[]  -- [k] @ [k] -> scalar (dot product)
  | 1, 2 => #[s2.getD 1 0]  -- [k] @ [k,n] -> [n]
  | 2, 1 => #[s1.getD 0 0]  -- [m,k] @ [k] -> [m]
  | 2, 2 => #[s1.getD 0 0, s2.getD 1 0]  -- [m,k] @ [k,n] -> [m,n]
  | 1, n2 =>
    -- [k] @ [..., k, n] -> [..., n]
    let batch := s2[:n2-2].toArray
    batch ++ #[s2.getD (n2 - 1) 0]
  | n1, 1 =>
    -- [..., m, k] @ [k] -> [..., m]
    let batch := s1[:n1-2].toArray
    batch ++ #[s1.getD (n1 - 2) 0]
  | n1, n2 =>
    -- General case: broadcast batch dims, matmul last 2
    let batch1 := s1[:n1-2].toArray
    let batch2 := s2[:n2-2].toArray
    let batchOut := broadcastBatchShapes batch1 batch2
    let m := s1.getD (n1 - 2) 0
    let n := s2.getD (n2 - 1) 0
    batchOut ++ #[m, n]
instance (s : Shape) : Nonempty (T s) :=
  TSpec.property
