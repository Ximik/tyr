/-
  Examples/Parity/LM.lean

  The request and result of the parity probes for models that emit tokens
  (`parity/lib/lm.py`). Request:
  `{"dtype", "prompts": [[token]], "max_new_tokens", "forced": [[token]]}`.
  Result: the greedy continuation of each prompt (`tokens`, without the prompt)
  and the Float32 logits of feeding `forced` through the KV cache (`logits`,
  `[batch, steps, vocab]`).
-/
import Examples.Parity.Common
import Tyr.Core.Typed.Neural

namespace Examples.Parity.LM

open Lean torch

structure Request where
  dtype : String
  prompts : Array (Array Nat)
  max_new_tokens : Nat
  forced : Array (Array Nat)
  deriving FromJson

/-- The request's dtype, which must be a real floating dtype. -/
def Request.realDType (request : Request) : IO RealFloatingDType := do
  let some dtype := DType.ofString? request.dtype
    | throw <| IO.userError s!"unknown dtype '{request.dtype}'"
  IO.ofExcept (RealFloatingDType.ofDType? dtype)

/-- `batch` rows of `width` tokens as one tensor on `device`. -/
def tokenMatrix (rows : Array (Array Nat)) (batch width : Nat) (device : Device) :
    IO (Tensor { device := device, shape := #[batch, width], dtype := .Int64 }) := do
  if rows.size != batch then
    throw <| IO.userError s!"{rows.size} rows where {batch} are expected"
  if let some row := rows.find? (·.size != width) then
    throw <| IO.userError s!"rows of different lengths ({row.size} and {width}) need a padding mask, which cached decoding does not take"
  let flat := rows.flatMap (·.map Int64.ofNat)
  pure ((← IO.ofExcept (Tensor.fromInt64ArrayAs? flat #[batch, width])).to device)

/-- The parsed request with its prompts and forced tokens as tensors. -/
structure Inputs (device : Device) where
  request : Request
  batch : Nat
  seq : Nat
  steps : Nat
  prompts : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }
  forced : Tensor { device := device, shape := #[batch, steps], dtype := .Int64 }

def Inputs.parse (json : Json) (device : Device) : IO (Inputs device) := do
  let request : Request ← IO.ofExcept (fromJson? json)
  let some first := request.prompts[0]?
    | throw <| IO.userError "request has no prompts"
  let batch := request.prompts.size
  let seq := first.size
  let steps := request.forced[0]?.map (·.size) |>.getD 0
  pure {
    request, batch, seq, steps
    prompts := ← tokenMatrix request.prompts batch seq device
    forced := ← tokenMatrix request.forced batch steps device
  }

/-- The result for generated ids `[batch, outSeq]` that start with the `seq`
prompt tokens, and the forced-run logits. -/
def output {batch outSeq steps vocab : Nat} {dtype : DType} {device : Device} (seq : Nat)
    (generated : Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })
    (logits : Tensor { device := device, shape := #[batch, steps, vocab], dtype := dtype }) :
    IO Output := do
  let flat ← generated.toUInt64Array
  let tokens := (Array.range batch).map fun row =>
    (flat.extract (row * outSeq + seq) ((row + 1) * outSeq)).map (·.toNat)
  pure {
    json := Json.mkObj [("tokens", toJson tokens)]
    tensors := #[("logits", Tensor.toTensor (Tensor.castDType logits .Float32))]
  }

end Examples.Parity.LM
