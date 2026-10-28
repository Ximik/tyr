/-
  Tyr/Model/Qwen/RoPE.lean

  Rotary Position Embeddings for Qwen3.
  Standard 2D rotation (different from Flux's 4-axis RoPE).
-/
import Tyr.Typed.Neural

namespace torch.qwen

variable {device : Device}
/-- Typed adapter over the `lean_torch_apply_rotary_emb` extern.
    x: [batch, seq, n_head, head_dim]; cos/sin: [seq, head_dim/2].
    The runtime moves cos/sin onto x's device if needed and casts the result
    back to x's dtype, so the output honestly re-asserts x's spec. -/
def applyRotaryEmb {batch seq n_head head_dim : Nat} {d dc : DType} {dev : Device}
    (x : Tensor ⟨#[batch, seq, n_head, head_dim], d, dev⟩)
    (cos : Tensor ⟨#[seq, head_dim / 2], dc, dev⟩)
    (sin : Tensor ⟨#[seq, head_dim / 2], dc, dev⟩)
    : Tensor ⟨#[batch, seq, n_head, head_dim], d, dev⟩ :=
  Tensor.applyRotary x cos sin

/-- Precomputed RoPE cos/sin embeddings for a given sequence length.
    These are cached and reused across forward passes.
    The runtime computes them in Float32. -/
structure RoPECache (max_seq : Nat) (head_dim : Nat) where
  /-- Cosine values: [max_seq, head_dim/2] -/
  cos : Tensor { device := device, shape := #[max_seq, head_dim / 2], dtype := .Float32 }
  /-- Sine values: [max_seq, head_dim/2] -/
  sin : Tensor { device := device, shape := #[max_seq, head_dim / 2], dtype := .Float32 }

namespace RoPECache

/-- Initialize RoPE cache with precomputed cos/sin values.
    Uses theta base for frequency computation. -/
def init (max_seq head_dim : Nat) (theta : Float := 10000.0) (device : Device := .CPU) : IO (RoPECache (device := device) max_seq head_dim) := do
  let (cos, sin) := Tensor.rotaryFreqs max_seq head_dim theta
  pure { cos := cos.to device, sin := sin.to device }

end RoPECache

/-- Apply rotary position embeddings to query/key tensors.
    x: [batch, seq, n_head, head_dim]
    Returns: [batch, seq, n_head, head_dim] with positions encoded -/
def applyRoPE {batch seq n_head head_dim max_seq : Nat}
    (x : Tensor { device := device, shape := #[batch, seq, n_head, head_dim], dtype := .Float32 })
    (cache : RoPECache (device := device) max_seq head_dim)
    (start_pos : Nat := 0)
    : Tensor { device := device, shape := #[batch, seq, n_head, head_dim], dtype := .Float32 } :=
  -- Slice cos/sin to current sequence length (runtime bounds, validated)
  let cos := cache.cos.sliceChecked 0 start_pos seq #[seq, head_dim / 2]
  let sin := cache.sin.sliceChecked 0 start_pos seq #[seq, head_dim / 2]
  -- Apply rotation
  applyRotaryEmb x cos sin

/-- Pure typed frequency table used by full and cached decoding. -/
def computeFreqsOnDevice (seq headDim : Nat) (theta : Float := 10000.0)
    (device : Device := .CPU) :
    Tensor { device := device, shape := #[seq, headDim / 2], dtype := .Float32 } × Tensor { device := device, shape := #[seq, headDim / 2], dtype := .Float32 } :=
  Tensor.rotaryFreqs seq headDim theta device

end torch.qwen
