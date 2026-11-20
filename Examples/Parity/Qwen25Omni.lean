/-
  Examples/Parity/Qwen25Omni.lean

  Parity probe for `parity/tests/qwen25omni.py`, the thinker's text model only;
  request and result as in `Examples.Parity.LM`.
-/
import Examples.Parity.LM
import Tyr.Model.Qwen25Omni

namespace Examples.Parity.Qwen25Omni

open Lean torch torch.qwen25omni

def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let inputs ← LM.Inputs.parse json device
  let dtype ← inputs.request.realDType
  unless dtype.val == .Float32 do
    throw <| IO.userError s!"Qwen25OmniForCausalLM computes in float32 only; the request asks for '{inputs.request.dtype}'"
  let weights := args.weights.toString
  let cfg ← Config.loadFromPretrainedDir weights
  let model ← Qwen25OmniForCausalLM.load weights cfg {} device
  let ⟨_, generated⟩ ← model.generateGreedy inputs.prompts inputs.request.max_new_tokens
  LM.output inputs.seq generated (← model.forcedLogits inputs.prompts inputs.forced)

end Examples.Parity.Qwen25Omni

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.Qwen25Omni.probe
