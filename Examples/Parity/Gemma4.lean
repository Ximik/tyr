/-
  Examples/Parity/Gemma4.lean

  Parity probe for `parity/tests/gemma4.py` and `gemma4_moe.py`, the text model
  only; request and result as in `Examples.Parity.LM`.
-/
import Examples.Parity.LM
import Tyr.Model.Gemma4.ConfigIO
import Tyr.Model.Gemma4.Weights

namespace Examples.Parity.Gemma4

open Lean torch torch.gemma4

def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let inputs ← LM.Inputs.parse json device
  let dtype ← inputs.request.realDType
  let weights := args.weights.toString
  let cfg ← Config.loadFromPretrainedDir weights
  let ⟨dtype, model⟩ ← Gemma4ForCausalLM.load weights cfg {} device (some dtype)
  have : DType.RealFloating dtype.val := inferInstance
  let ⟨_, generated⟩ ← model.generate cfg inputs.prompts inputs.request.max_new_tokens .greedy
  LM.output inputs.seq generated (← model.forcedLogits cfg inputs.prompts inputs.forced)

end Examples.Parity.Gemma4

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.Gemma4.probe
