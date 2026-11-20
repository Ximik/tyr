/-
  Examples/Parity/Laguna.lean

  Parity probe for `parity/tests/laguna.py`; request and result as in
  `Examples.Parity.LM`. The model runs in the checkpoint's dtype.
-/
import Examples.Parity.LM
import Tyr.Model.Laguna.ConfigIO
import Tyr.Model.Laguna.Weights

namespace Examples.Parity.Laguna

open Lean torch torch.laguna

def probe (args : Args) (device : Device) (json : Json) : IO Output := do
  let inputs ← LM.Inputs.parse json device
  let weights := args.weights.toString
  let cfg ← Config.loadFromPretrainedDir weights
  let ⟨dtype, model⟩ ← LagunaForCausalLM.load weights cfg device
  unless DType.ofString? inputs.request.dtype == some dtype.val do
    throw <| IO.userError s!"the checkpoint loads as {dtype.val}; the request asks for '{inputs.request.dtype}'"
  have : DType.RealFloating dtype.val := inferInstance
  let ⟨_, generated⟩ ← model.generate cfg inputs.prompts inputs.request.max_new_tokens .greedy #[]
  LM.output inputs.seq generated (← model.forcedLogits cfg inputs.prompts inputs.forced)

end Examples.Parity.Laguna

def main (argv : List String) : IO UInt32 :=
  Examples.Parity.run argv Examples.Parity.Laguna.probe
