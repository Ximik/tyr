/-
  Examples/Parity/Qwen3.lean

  Parity probe for `parity/tests/qwen3.py`. Request:
  `{"dtype", "prompts": [[token]], "max_new_tokens", "forced": [[token]]}`.
  Writes the greedy continuation of each prompt (`tokens`, without the prompt)
  and the logits of feeding `forced` through the KV cache (`logits`,
  `[batch, steps, vocab]`).
-/
import Examples.Parity.Common
import Tyr.Model.Qwen3

namespace Examples.Parity.Qwen3

open Lean torch torch.qwen3

structure Request where
  dtype : String
  prompts : Array (Array Nat)
  max_new_tokens : Nat
  forced : Array (Array Nat)
  deriving FromJson

/-- `batch` rows of `width` tokens as one tensor on `device`. -/
private def tokenMatrix (rows : Array (Array Nat)) (batch width : Nat) (device : Device) :
    IO (Tensor { device := device, shape := #[batch, width], dtype := .Int64 }) := do
  if rows.size != batch then
    throw <| IO.userError s!"{rows.size} rows where {batch} are expected"
  if let some row := rows.find? (·.size != width) then
    throw <| IO.userError s!"rows of different lengths ({row.size} and {width}) need a padding mask, which Qwen3ForCausalLM's cached decoding does not take"
  let flat := rows.flatMap (·.map Int64.ofNat)
  pure ((← IO.ofExcept (Tensor.fromInt64ArrayAs? flat #[batch, width])).to device)

def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let request : Request ← IO.ofExcept (fromJson? json)
  match DType.ofString? request.dtype with
  | some .Float32 => pure ()
  | _ => throw <| IO.userError s!"Qwen3ForCausalLM computes in float32 only; the request asks for '{request.dtype}'"
  let some first := request.prompts[0]?
    | throw <| IO.userError "request has no prompts"
  let batch := request.prompts.size
  let seq := first.size
  let steps := request.forced[0]?.map (·.size) |>.getD 0
  let inputIds ← tokenMatrix request.prompts batch seq device
  let forced ← tokenMatrix request.forced batch steps device

  let weights := args.weights.toString
  let cfg ← Config.loadFromPretrainedDir weights
  let model ← Qwen3ForCausalLM.load weights cfg {} device

  let ⟨outSeq, generated⟩ ← model.generateGreedy inputIds request.max_new_tokens
  let flat ← generated.toUInt64Array
  let tokens := (Array.range batch).map fun row =>
    (flat.extract (row * outSeq + seq) ((row + 1) * outSeq)).map (·.toNat)

  let logits ← model.forcedLogits inputIds forced
  pure {
    json := Json.mkObj [("tokens", toJson tokens)]
    tensors := #[("logits", logits.raw)]
  }

end Examples.Parity.Qwen3

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.Qwen3.probe
