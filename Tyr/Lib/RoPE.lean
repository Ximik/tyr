/-
  Tyr/Lib/RoPE.lean

  Rotary position embedding tables: the cosine and sine tables of
  `Tensor.rotaryFreqs` for `max_seq` positions, computed once on a device.
-/
import Tyr.Core.Typed.Neural

namespace torch

variable {device : Device}

/-- Float32 cosine and sine tables, `[max_seq, half]` each. -/
structure RoPECache (max_seq : Nat) (half : Nat) where
  cos : Tensor { device := device, shape := #[max_seq, half], dtype := .Float32 }
  sin : Tensor { device := device, shape := #[max_seq, half], dtype := .Float32 }

namespace RoPECache

def init (max_seq half : Nat) (theta : Float := 10000.0) (device : Device := .CPU) : IO (RoPECache (device := device) max_seq half) := do
  let (cos, sin) := Tensor.rotaryFreqs max_seq half theta
  pure { cos := cos.to device, sin := sin.to device }

end RoPECache

end torch
