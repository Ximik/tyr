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

open torch

/-! ## Sampling Strategy -/

inductive SamplingStrategy where
  | greedy
  | multinomial (temperature : Float := 1.0) (topK : UInt64 := 0) (topP : Float := 1.0)
  deriving Repr, Inhabited

/-- Callback invoked on every generated token during streaming decode.
    Tokens are Int64 vocabulary indices. -/
abbrev StreamCallback (batch : UInt64) := UInt64 → DTensor #[batch] .Int64 → IO Unit

/-! ## Mask adapters

`Tyr.Typed.Ops` has no Bool combinators or `where` wrapper yet; these
private adapters re-assert the spec over the existing externs (the same
pattern as the flash-attention adapter in `Tyr/Inference/KVCache.lean`). -/

/-- Boolean OR for Bool masks (De Morgan over the legacy externs). -/
private def orMask {s : Shape} {dev : DevicePolicy}
    (a b : Tensor ⟨s, .Bool, dev⟩) : Tensor ⟨s, .Bool, dev⟩ :=
  Tensor.assumeSpec (torch.logical_not (s := s)
    (torch.logical_and (s := s)
      (torch.logical_not (s := s) a.raw) (torch.logical_not (s := s) b.raw)))

/-- `where` with a Bool condition and both branches sharing one spec. -/
private def whereSelect {s : Shape} {d : DType} {dev : DevicePolicy}
    (condition : Tensor ⟨s, .Bool, dev⟩) (x y : Tensor ⟨s, d, dev⟩) : Tensor ⟨s, d, dev⟩ :=
  Tensor.assumeSpec (RawTensor.whereSelect s condition.raw x.raw y.raw)

/-- All-false Bool mask on the same placement as `reference`. -/
private def falseMaskLike {n : UInt64} {dev : DevicePolicy}
    (reference : Tensor ⟨#[n], .Int64, dev⟩) : Tensor ⟨#[n], .Bool, dev⟩ :=
  let zeros := RawTensor.toDevice #[n] (RawTensor.fullInt #[n] 0) reference.raw.device
  Tensor.assumeSpec (RawTensor.eqScalar #[n] zeros 1)

/-! ## Token Selection -/

/-- Sample a batch of next tokens from logits according to a strategy. -/
def sampleFromLogits {batch vocab : UInt64}
    (logits : DTensor #[batch, vocab] .Float32)
    (strategy : SamplingStrategy)
    : IO (DTensor #[batch] .Int64) := do
  match strategy with
  | .greedy =>
    -- `argmax` removes dim 1 via `reduceShape`, which does not reduce at
    -- symbolic dims; `reshapeChecked` re-establishes `#[batch]` with a
    -- runtime check.
    pure (Tensor.reshapeChecked (Tensor.argmax logits 1) #[batch])
  | .multinomial temperature topK topP =>
    if temperature <= 0.0 then
      throw <| IO.userError s!"multinomial sampling requires temperature > 0, got {temperature}"
    let scaled :=
      if temperature == 1.0 then logits
      else Tensor.Preserving.mulScalar logits (1.0 / temperature)
    let filtered :=
      if topK == 0 then scaled
      else Tensor.topKFilter scaled topK
    let filtered :=
      if topP >= 1.0 then filtered
      else Tensor.topPFilter filtered topP
    let probs := Tensor.softmax filtered (-1) rfl
    let sampled ← Tensor.multinomial probs 1 false
    pure (Tensor.reshapeChecked sampled #[batch])

/-! ## EOS / Finish Helpers -/

/-- Return a Bool mask of tokens that belong to `values`. -/
def tokenInSet {n : UInt64}
    (tokens : DTensor #[n] .Int64)
    (values : Array UInt64)
    : DTensor #[n] .Bool :=
  Id.run do
    let mut mask := falseMaskLike tokens
    for value in values do
      let hit : DTensor #[n] .Bool := Tensor.eqScalar tokens (Int64.ofNat value.toNat)
      mask := orMask mask hit
    return mask

/-- Replace tokens with `eosToken` wherever `finished` is true. -/
def applyFinishedEos {n : UInt64}
    (tokens : DTensor #[n] .Int64)
    (finished : DTensor #[n] .Bool)
    (eosToken : UInt64)
    : DTensor #[n] .Int64 :=
  let eos : DTensor #[n] .Int64 :=
    Tensor.assumeSpec (RawTensor.toDevice #[n]
      (RawTensor.fullInt #[n] (Int64.ofNat eosToken.toNat)) tokens.raw.device)
  whereSelect finished eos tokens

end torch.Model
