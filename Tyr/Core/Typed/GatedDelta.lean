/-
  Tyr/Core/Typed/GatedDelta.lean

  The gated delta rule, a linear attention with a per-head state
  `S : [kdim, vdim]`. Per head, `query` and `key` are L2-normalized over their
  last dimension (eps 1e-6), `query` is scaled by `1 / sqrt kdim`, and for each
  position `t`:
    `S ← exp(g_t) • S`
    `S ← S + k_t ⊗ (beta_t • (v_t - Sᵀ k_t))`
    `o_t = Sᵀ q_t`
  computed in Float32; the output is cast back to the input dtype and the final
  state stays Float32.
-/
import Tyr.Core.Typed.Neural
import Tyr.Core.FFI.GatedDelta

namespace torch.Tensor

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]

/-- The gated delta rule from a zero state, evaluated in chunks of 64 positions.
    Returns the output and the final state. -/
def chunkGatedDeltaRule {batch seq heads kdim vdim : Nat}
    (query key : Tensor { device := device, shape := #[batch, seq, heads, kdim], dtype := dtype })
    (value : Tensor { device := device, shape := #[batch, seq, heads, vdim], dtype := dtype })
    (g : Tensor { device := device, shape := #[batch, seq, heads], dtype := .Float32 })
    (beta : Tensor { device := device, shape := #[batch, seq, heads], dtype := dtype }) :
    Tensor { device := device, shape := #[batch, seq, heads, vdim], dtype := dtype } ×
      Tensor { device := device, shape := #[batch, heads, kdim, vdim], dtype := .Float32 } :=
  let (out, state) := RawTensor.chunkGatedDeltaRule query.raw key.raw value.raw g.raw beta.raw
  (Tensor.ofRaw! out, Tensor.ofRaw! state)

/-- The gated delta rule from `initialState`, one position at a time.
    Returns the output and the final state. -/
def recurrentGatedDeltaRule {batch seq heads kdim vdim : Nat}
    (query key : Tensor { device := device, shape := #[batch, seq, heads, kdim], dtype := dtype })
    (value : Tensor { device := device, shape := #[batch, seq, heads, vdim], dtype := dtype })
    (g : Tensor { device := device, shape := #[batch, seq, heads], dtype := .Float32 })
    (beta : Tensor { device := device, shape := #[batch, seq, heads], dtype := dtype })
    (initialState : Tensor { device := device, shape := #[batch, heads, kdim, vdim], dtype := .Float32 }) :
    Tensor { device := device, shape := #[batch, seq, heads, vdim], dtype := dtype } ×
      Tensor { device := device, shape := #[batch, heads, kdim, vdim], dtype := .Float32 } :=
  let (out, state) := RawTensor.recurrentGatedDeltaRule
    query.raw key.raw value.raw g.raw beta.raw initialState.raw
  (Tensor.ofRaw! out, Tensor.ofRaw! state)

end torch.Tensor
