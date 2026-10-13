/-
  Tyr/Model/Qwen/RoPE.lean

  Rotary Position Embeddings for Qwen3.
  Standard 2D rotation (different from Flux's 4-axis RoPE).
-/
import Tyr.Typed.Neural

namespace torch.qwen

/-- Typed adapter over the `lean_torch_apply_rotary_emb` extern.
    x: [batch, seq, n_head, head_dim]; cos/sin: [seq, head_dim/2].
    The runtime moves cos/sin onto x's device if needed and casts the result
    back to x's dtype, so the output honestly re-asserts x's spec. -/
def applyRotaryEmb {batch seq n_head head_dim : UInt64} {d dc : DType} {dev : DevicePolicy}
    (x : Tensor ⟨#[batch, seq, n_head, head_dim], d, dev⟩)
    (cos : Tensor ⟨#[seq, head_dim / 2], dc, dev⟩)
    (sin : Tensor ⟨#[seq, head_dim / 2], dc, dev⟩)
    : Tensor ⟨#[batch, seq, n_head, head_dim], d, dev⟩ :=
  Tensor.applyRotary x cos sin

/-- Precomputed RoPE cos/sin embeddings for a given sequence length.
    These are cached and reused across forward passes.
    The runtime computes them in Float32. -/
structure RoPECache (max_seq : UInt64) (head_dim : UInt64) where
  /-- Cosine values: [max_seq, head_dim/2] -/
  cos : DTensor #[max_seq, head_dim / 2] .Float32
  /-- Sine values: [max_seq, head_dim/2] -/
  sin : DTensor #[max_seq, head_dim / 2] .Float32

namespace RoPECache

/-- Initialize RoPE cache with precomputed cos/sin values.
    Uses theta base for frequency computation. -/
def init (max_seq head_dim : UInt64) (theta : Float := 10000.0) : IO (RoPECache max_seq head_dim) := do
  let (cos, sin) := Tensor.rotaryFreqs max_seq head_dim theta
  pure { cos, sin }

end RoPECache

/-- Apply rotary position embeddings to query/key tensors.
    x: [batch, seq, n_head, head_dim]
    Returns: [batch, seq, n_head, head_dim] with positions encoded -/
def applyRoPE {batch seq n_head head_dim max_seq : UInt64}
    (x : DTensor #[batch, seq, n_head, head_dim] .Float32)
    (cache : RoPECache max_seq head_dim)
    (start_pos : UInt64 := 0)
    : DTensor #[batch, seq, n_head, head_dim] .Float32 :=
  -- Slice cos/sin to current sequence length (runtime bounds, validated)
  let cos := cache.cos.sliceChecked 0 start_pos seq #[seq, head_dim / 2]
  let sin := cache.sin.sliceChecked 0 start_pos seq #[seq, head_dim / 2]
  -- Apply rotation
  applyRotaryEmb x cos sin

/-- Pure typed frequency table used by full and cached decoding. -/
def computeFreqsOnDevice (seq headDim : UInt64) (theta : Float := 10000.0)
    (device : Device := .CPU) :
    DTensor #[seq, headDim / 2] .Float32 × DTensor #[seq, headDim / 2] .Float32 :=
  Tensor.rotaryFreqs seq headDim theta device

end torch.qwen
