import Tyr.Core.FFI.Runtime

/-! Runtime services and shape utilities for Tyr. Tensor operations are exposed
through `Tyr.Core.Typed` and `Tyr.Core.Typed.Neural`, using `torch.Tensor` indices. -/

namespace torch

namespace Device

/-- Acquire the placement `kind` describes; throws if its backend is unavailable. -/
def acquire : DeviceKind → IO Device
  | .CPU => pure .CPU
  | .CUDA index => cuda index
  | .MPS => mps

/-- Acquire the placement a `DeviceKind.ofString?` spelling names. -/
def acquireNamed (name : String) : IO Device :=
  match DeviceKind.ofString? name with
  | some kind => acquire kind
  | none => throw (IO.userError s!"unknown device '{name}' (expected cpu, mps, cuda or cuda:<ordinal>)")

end Device

/--
Probe MPS with a short retry budget.

On some macOS/libtorch combinations the first process-local probe can race MPS
backend startup and transiently return `false`. Retrying here keeps device
selection stable for downstream callers.
-/
def mps_is_available_stable (retries : Nat := 4) (delayMs : Nat := 250) : IO Bool := do
  let rec loop (remaining : Nat) : IO Bool := do
    let available ← mps_is_available
    if available then
      return true
    match remaining with
    | 0 => return false
    | remaining + 1 =>
        IO.sleep delayMs.toUInt32
        loop remaining
  loop retries

end torch
