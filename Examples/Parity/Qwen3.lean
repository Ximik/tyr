/-
  Examples/Parity/Qwen3.lean

  Parity probe for `parity/tests/qwen3.py`; request and result as in
  `Examples.Parity.LM`.
-/
import Examples.Parity.LM
import Tyr.Model.Qwen3

namespace Examples.Parity.Qwen3

open Lean torch torch.qwen3

def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let inputs ← LM.Inputs.parse json device
  let dtype ← inputs.request.realDType
  unless dtype.val == .Float32 do
    throw <| IO.userError s!"Qwen3ForCausalLM computes in float32 only; the request asks for '{inputs.request.dtype}'"
  let weights := args.weights.toString
  let cfg ← Config.loadFromPretrainedDir weights
  let model ← Qwen3ForCausalLM.load weights cfg {} device
  let ⟨_, generated⟩ ← model.generateGreedy inputs.prompts inputs.request.max_new_tokens
  LM.output inputs.seq generated (← model.forcedLogits inputs.prompts inputs.forced)

end Examples.Parity.Qwen3

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.Qwen3.probe
