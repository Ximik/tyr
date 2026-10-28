/-
  Tyr/Model/Generation.lean

  Shared generation infrastructure for causal language models:
  sampling strategies, logits filtering, and token-selection helpers.

  Typed throughout: logits are Float32, sampled tokens are Int64, and the
  EOS/finished masks are honest `Bool` tensors (`Tensor.eqScalar` returns
  `.Bool`, matching what the runtime always produced).
-/
import Tyr.Torch
import Tyr.Typed

namespace torch.Model

variable {device : Device}
open torch

/-! ## Sampling Strategy -/

inductive SamplingStrategy where
  | greedy
  | multinomial (temperature : Float := 1.0) (topK : Nat := 0) (topP : Float := 1.0)
  deriving Repr, Inhabited

/-- Callback invoked on every generated token during streaming decode.
    Tokens are Int64 vocabulary indices. -/
abbrev StreamCallback (batch : Nat) := Nat → Tensor { device := device, shape := #[batch], dtype := .Int64 } → IO Unit

/-- All-false Bool mask on the same placement as `reference`. -/
private def falseMaskLike {n : Nat} {dev : Device}
    (reference : Tensor ⟨#[n], .Int64, dev⟩) : Tensor ⟨#[n], .Bool, dev⟩ :=
  Tensor.eqScalar (Tensor.zerosLike reference) 1

/-! ## Token Selection -/

/-- Sample a batch of next tokens from logits according to a strategy. -/
def sampleFromLogits {batch vocab : Nat}
    (logits : Tensor { device := device, shape := #[batch, vocab], dtype := .Float32 })
    (strategy : SamplingStrategy)
    : IO (Tensor { device := device, shape := #[batch], dtype := .Int64 }) := do
  match strategy with
  | .greedy =>
    let indices ← IO.ofExcept (Tensor.argmax? logits 1)
    pure (indices.reshape #[batch])
  | .multinomial temperature topK topP =>
    if temperature <= 0.0 then
      throw <| IO.userError s!"multinomial sampling requires temperature > 0, got {temperature}"
    let scaled :=
      if temperature == 1.0 then logits
      else Tensor.mulScalar logits (1.0 / temperature)
    let filtered :=
      if topK == 0 then scaled
      else Tensor.topKFilter scaled topK
    let filtered :=
      if topP >= 1.0 then filtered
      else Tensor.topPFilter filtered topP
    let probs := Tensor.softmax filtered (-1) rfl
    let sampled ← Tensor.multinomial probs 1 false
    pure (Tensor.reshape sampled #[batch])

/-! ## EOS / Finish Helpers -/

/-- Return a Bool mask of tokens that belong to `values`. -/
def tokenInSet {n : Nat}
    (tokens : Tensor { device := device, shape := #[n], dtype := .Int64 })
    (values : Array UInt64)
    : Tensor { device := device, shape := #[n], dtype := .Bool } :=
  Id.run do
    let mut mask := falseMaskLike tokens
    for value in values do
      let hit : Tensor { device := device, shape := #[n], dtype := .Bool } := Tensor.eqScalar tokens (Int64.ofNat value.toNat)
      mask := Tensor.logicalOr mask hit
    return mask

/-- Replace tokens with `eosToken` wherever `finished` is true. -/
def applyFinishedEos {n : Nat}
    (tokens : Tensor { device := device, shape := #[n], dtype := .Int64 })
    (finished : Tensor { device := device, shape := #[n], dtype := .Bool })
    (eosToken : UInt64)
    : Tensor { device := device, shape := #[n], dtype := .Int64 } :=
  let eos : Tensor { device := device, shape := #[n], dtype := .Int64 } :=
    (Tensor.fullInt #[n] (Int64.ofNat eosToken.toNat)).to tokens.device
  Tensor.whereSelect finished eos tokens

end torch.Model
