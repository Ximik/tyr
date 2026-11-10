/-
  Tyr/Model/Qwen/RoPE.lean

  Rotary Position Embeddings for Qwen3.
  Standard 2D rotation (different from Flux's 4-axis RoPE).
-/
import Tyr.Typed.Neural

namespace torch.qwen

variable {device : Device}
/-- Typed adapter over the `lean_torch_apply_rotary_emb` extern.
    x: [batch, seq, n_head, 2 * half]; cos/sin: [seq, half].
    The runtime moves cos/sin onto x's device if needed and casts the result
    back to x's dtype, so the output honestly re-asserts x's spec. -/
def applyRotaryEmb {batch seq n_head half : Nat} {d dc : DType} {dev : Device}
    (x : Tensor ⟨#[batch, seq, n_head, 2 * half], d, dev⟩)
    (cos : Tensor ⟨#[seq, half], dc, dev⟩)
    (sin : Tensor ⟨#[seq, half], dc, dev⟩)
    : Tensor ⟨#[batch, seq, n_head, 2 * half], d, dev⟩ :=
  Tensor.applyRotary x cos sin

/-- Precomputed RoPE cos/sin embeddings for a given sequence length.
    These are cached and reused across forward passes.
    The runtime computes them in Float32. -/
structure RoPECache (max_seq : Nat) (half : Nat) where
  /-- Cosine values: [max_seq, half] -/
  cos : Tensor { device := device, shape := #[max_seq, half], dtype := .Float32 }
  /-- Sine values: [max_seq, half] -/
  sin : Tensor { device := device, shape := #[max_seq, half], dtype := .Float32 }

namespace RoPECache

/-- Initialize RoPE cache with precomputed cos/sin values.
    Uses theta base for frequency computation. -/
def init (max_seq half : Nat) (theta : Float := 10000.0) (device : Device := .CPU) : IO (RoPECache (device := device) max_seq half) := do
  let (cos, sin) := Tensor.rotaryFreqs max_seq half theta
  pure { cos := cos.to device, sin := sin.to device }

end RoPECache

/-- Pure typed frequency table used by full and cached decoding. -/
def computeFreqsOnDevice (seq half : Nat) (theta : Float := 10000.0)
    (device : Device := .CPU) :
    Tensor { device := device, shape := #[seq, half], dtype := .Float32 } × Tensor { device := device, shape := #[seq, half], dtype := .Float32 } :=
  Tensor.rotaryFreqs seq half theta device

end torch.qwen
