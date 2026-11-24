import Tyr.RawTensor.Runtime

/-! Runtime services and shape utilities for Tyr. Tensor operations are exposed
through `Tyr.Typed` and `Tyr.Typed.Neural`, using `torch.Tensor` indices. -/

namespace torch

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

/-- Output length of a 1D convolution, in the form the typed `conv1d` result uses. -/
@[simp] def convOutputSize (input_size kernel_size : Nat) (stride : Nat := 1) (padding : Nat := 0) (dilation : Nat := 1) : Nat :=
  (input_size + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1

/-- Output length of a 1D transposed convolution, in the form the typed `convTranspose1d` result
    uses; the padding is subtracted last so a short input is not truncated early. -/
@[simp] def convTransposeOutputSize (input_size kernel_size : Nat)
    (stride : Nat := 1) (padding : Nat := 0) (output_padding : Nat := 0)
    (dilation : Nat := 1) : Nat :=
  (input_size - 1) * stride + dilation * (kernel_size - 1) + output_padding + 1 - 2 * padding

end torch
