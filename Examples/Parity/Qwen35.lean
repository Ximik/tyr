/-
  Examples/Parity/Qwen35.lean

  Parity probe for `parity/tests/qwen35.py`, the text model only; request and
  result as in `Examples.Parity.LM`.
-/
import Examples.Parity.LM
import Tyr.Model.Qwen35.ConfigIO
import Tyr.Model.Qwen35.Weights

namespace Examples.Parity.Qwen35

open Lean torch torch.qwen35

def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let inputs ← LM.Inputs.parse json device
  let dtype ← inputs.request.realDType
  let weights := args.weights.toString
  let cfg ← Config.loadFromPretrainedDir weights
  let ⟨dtype, model⟩ ← Qwen35ForCausalLM.load weights cfg device {} (some dtype)
  have : DType.RealFloating dtype.val := inferInstance
  let ⟨_, generated⟩ ← model.generateGreedy cfg inputs.prompts inputs.request.max_new_tokens
  LM.output inputs.seq generated (← model.forcedLogits cfg inputs.prompts inputs.forced)

end Examples.Parity.Qwen35

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.Qwen35.probe
