/-
  Tyr/Model/Qwen3/Model.lean

  Standalone Qwen3 causal-LM built on shared Qwen transformer blocks.
  Includes:
  - LM head
  - masked/unmasked forward
  - incremental KV-cache decode
  - greedy generation (cached + uncached)
-/
import Tyr.Typed
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Qwen
import Tyr.Model.Qwen3.Config
import Tyr.Model.Utils

namespace torch.qwen3

variable {device : Device}
open torch
open torch.Model

private def tokenInSetMask {batch : UInt64}
    (tokens : Tensor { device := device, shape := #[batch], dtype := .Int64 })
    (allow : Array UInt64)
    : Tensor { device := device, shape := #[batch], dtype := .Bool } :=
  Id.run do
    let mut out : Tensor { device := device, shape := #[batch], dtype := .Bool } := Tensor.eqScalar tokens (-1)
    for tok in allow do
      let isTok : Tensor { device := device, shape := #[batch], dtype := .Bool } := Tensor.eqScalar tokens (Int64.ofNat tok.toNat)
      out := Tensor.logicalOr out isTok
    out

private def eosVectorOnDevice {batch : UInt64}
    (eosTokenIds : Array UInt64)
    (device : Device)
    : Option (Tensor { device := device, shape := #[batch], dtype := .Int64 }) :=
  match eosTokenIds[0]? with
  | none => none
  | some eosTok =>
    let eosCpu : Tensor { device := .CPU, shape := #[batch], dtype := .Int64 } := Tensor.fullInt #[batch] (Int64.ofNat eosTok.toNat)
    let eosDev : Tensor { device := device, shape := #[batch], dtype := .Int64 } := eosCpu.to device
    some eosDev

/-- Causal-LM output container. -/
structure CausalLMOutput (cfg : Config) (batch seq : UInt64) where
  logits : Tensor { device := device, shape := #[batch, seq, cfg.vocab_size], dtype := .Float32 }
  hiddenStates : Option (Array (Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := .Float32 })) := none

/-- Full standalone Qwen3 causal language model. -/
structure Qwen3ForCausalLM (cfg : Config) where
  model : qwen.Qwen3Model (device := device) cfg
  lmHead : Tensor { device := device, shape := #[cfg.vocab_size, cfg.hidden_size], dtype := .Float32 }
  tieWordEmbeddings : Bool := true
  deriving TensorStruct

namespace Qwen3ForCausalLM

/-- Initialize model + LM head. By default, LM head is tied to token embeddings. -/
def init (cfg : Config) (tieWordEmbeddings : Bool := true) (device : Device := .CPU) : IO (Qwen3ForCausalLM (device := device) cfg) := do
  let model ← qwen.Qwen3Model.init (device := device) cfg
  let lmHead ←
    if tieWordEmbeddings then
      pure model.embed_tokens
    else
      initWeight (device := device) #[cfg.vocab_size, cfg.hidden_size] cfg.hidden_size
  pure { model, lmHead, tieWordEmbeddings }

/-- Token embedding lookup. -/
def embedTokens {batch seq : UInt64}
    (m : Qwen3ForCausalLM (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := .Float32 } :=
  Tensor.embedding inputIds m.model.embed_tokens

/-- Forward from pre-computed input embeddings. -/
def forwardEmbeds {batch seq : UInt64}
    (m : Qwen3ForCausalLM (device := device) cfg)
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := .Float32 })
    (attnMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Bool }) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.vocab_size], dtype := .Float32 } :=
  let (cos, sin) := qwen.computeFreqsOnDevice seq cfg.head_dim cfg.rope_theta device
  let hidden :=
    match attnMask with
    | some mask =>
      m.model.layers.foldl
        (fun h layer => layer.forwardMasked h cos sin mask true)
        inputsEmbeds
    | none =>
      m.model.layers.foldl
        (fun h layer => layer.forward h cos sin true)
        inputsEmbeds
  let hidden := m.model.norm.forward3d hidden
  Tensor.linear3d hidden m.lmHead

/-- Standard forward pass from token IDs. -/
def forward {batch seq : UInt64}
    (m : Qwen3ForCausalLM (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (attnMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Bool }) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.vocab_size], dtype := .Float32 } :=
  m.forwardEmbeds (m.embedTokens inputIds) attnMask

/-- Forward pass that also returns per-layer hidden states. -/
def forwardWithHiddenStates {batch seq : UInt64}
    (m : Qwen3ForCausalLM (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (attnMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Bool }) := none)
    : CausalLMOutput (device := device) cfg batch seq :=
  let (cos, sin) := qwen.computeFreqsOnDevice seq cfg.head_dim cfg.rope_theta device
  let hiddenStates :=
    match attnMask with
    | some mask => m.model.forwardWithHiddenStatesMasked cfg inputIds cos sin mask
    | none => m.model.forwardWithHiddenStates cfg inputIds cos sin
  let finalHidden : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := .Float32 } :=
    hiddenStates.getD (hiddenStates.size - 1) (Tensor.zeros (device := device) #[batch, seq, cfg.hidden_size])
  let logits := Tensor.linear3d finalHidden m.lmHead
  { logits, hiddenStates := some hiddenStates }

/-- One-layer KV cache type for cached decoding. -/
abbrev LayerKVCache (cfg : Config) (batch : UInt64) :=
  qwen.QwenAttention.KVCache (device := device) batch cfg.num_key_value_heads cfg.head_dim

private def initLayerKVCaches {batch : UInt64}
    (m : Qwen3ForCausalLM (device := device) cfg)
    (maxLen : UInt64)
    (device : Device)
    : Array (LayerKVCache (device := device) cfg batch) :=
  m.model.layers.map (fun _ =>
    qwen.QwenAttention.initKVCache
      maxLen
      (batch := batch)
      (num_kv_heads := cfg.num_key_value_heads)
      (head_dim := cfg.head_dim)
      device)

private def precomputeDecodeRotary {maxLen : UInt64}
    (_m : Qwen3ForCausalLM (device := device) cfg)
    (device : Device)
    : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 } × Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 } :=
  qwen.computeFreqsOnDevice
    maxLen
    cfg.head_dim
    cfg.rope_theta
    device

private def decodeStepFromEmbedWithCache {batch maxLen : UInt64}
    (m : Qwen3ForCausalLM (device := device) cfg)
    (cosAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (sinAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (tokenEmbed : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := .Float32 })
    (position : UInt64)
    (caches : Array (LayerKVCache (device := device) cfg batch))
    : IO (Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := .Float32 } × Array (LayerKVCache (device := device) cfg batch)) := do
  let cos : Tensor { device := device, shape := #[1, cfg.head_dim / 2], dtype := .Float32 } := Tensor.sliceChecked cosAll 0 position 1 #[1, cfg.head_dim / 2]
  let sin : Tensor { device := device, shape := #[1, cfg.head_dim / 2], dtype := .Float32 } := Tensor.sliceChecked sinAll 0 position 1 #[1, cfg.head_dim / 2]

  let mut hidden : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := .Float32 } := tokenEmbed
  let mut nextCaches := caches

  for i in [:m.model.layers.size] do
    let layer ←
      match m.model.layers[i]? with
      | some l => pure l
      | none => throw <| IO.userError s!"missing Qwen3 layer at index {i}"
    let cache ←
      match nextCaches[i]? with
      | some c => pure c
      | none => throw <| IO.userError s!"missing Qwen3 KV cache at index {i}"
    let (hNext, cNext) := layer.forwardStep hidden cos sin cache
    hidden := hNext
    nextCaches := nextCaches.set! i cNext

  let hiddenNorm := m.model.norm.forward3d hidden
  let logits3 : Tensor { device := device, shape := #[batch, 1, cfg.vocab_size], dtype := .Float32 } := Tensor.linear3d hiddenNorm m.lmHead
  let logits2 : Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := .Float32 } := Tensor.reshapeChecked logits3 #[batch, cfg.vocab_size]
  pure (logits2, nextCaches)

private partial def prefillCachesFromEmbeds {batch seq maxLen : UInt64}
    (m : Qwen3ForCausalLM (device := device) cfg)
    (cosAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (sinAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := .Float32 })
    (caches : Array (LayerKVCache (device := device) cfg batch))
    (position : Nat)
    (lastLogits : Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := .Float32 })
    : IO (Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := .Float32 } × Array (LayerKVCache (device := device) cfg batch)) := do
  if position >= seq.toNat then
    pure (lastLogits, caches)
  else
    let tok : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := .Float32 } := Tensor.sliceChecked inputsEmbeds 1 position.toUInt64 1 #[batch, 1, cfg.hidden_size]
    let (logits, caches') ← decodeStepFromEmbedWithCache m cosAll sinAll tok position.toUInt64 caches
    prefillCachesFromEmbeds m cosAll sinAll inputsEmbeds caches' (position + 1) logits

private partial def greedyLoopCached {batch maxLen : UInt64}
    (m : Qwen3ForCausalLM (device := device) cfg)
    (cosAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (sinAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (eosTokenIds : Array UInt64)
    (eosVector : Option (Tensor { device := device, shape := #[batch], dtype := .Int64 }))
    (remaining : Nat)
    (caches : Array (LayerKVCache (device := device) cfg batch))
    (lastLogits : Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := .Float32 })
    (finished : Option (Tensor { device := device, shape := #[batch], dtype := .Bool }))
    {curSeq : UInt64}
    (curIds : Tensor { device := device, shape := #[batch, curSeq], dtype := .Int64 })
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩

  let nextTokRaw : Tensor { device := device, shape := #[batch], dtype := .Int64 } := (Tensor.argmax lastLogits 1).reshapeChecked #[batch]
  let nextTok : Tensor { device := device, shape := #[batch], dtype := .Int64 } :=
    match eosVector, finished with
    | some eosTok, some doneMask =>
      let activeMask : Tensor { device := device, shape := #[batch], dtype := .Bool } := Tensor.logicalNot doneMask
      Tensor.whereSelect activeMask nextTokRaw eosTok
    | _, _ => nextTokRaw
  let nextCol : Tensor { device := device, shape := #[batch, 1], dtype := .Int64 } := Tensor.reshapeChecked nextTok #[batch, 1]
  let appended : Tensor { device := device, shape := #[batch, curSeq + 1], dtype := .Int64 } := Tensor.cat curIds nextCol 1

  match eosVector with
  | none =>
    let nextEmb : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := .Float32 } := m.embedTokens nextCol
    let (nextLogits, caches') ← decodeStepFromEmbedWithCache m cosAll sinAll nextEmb curSeq caches
    greedyLoopCached m cosAll sinAll eosTokenIds none (remaining - 1) caches' nextLogits none appended
  | some _ =>
    let reachedEos : Tensor { device := device, shape := #[batch], dtype := .Bool } := tokenInSetMask nextTok eosTokenIds
    let finished' : Tensor { device := device, shape := #[batch], dtype := .Bool } :=
      match finished with
      | some doneMask => Tensor.logicalOr doneMask reachedEos
      | none => reachedEos
    let hasActiveRows : Bool := Tensor.any (Tensor.logicalNot finished')
    if !hasActiveRows then
      return ⟨curSeq + 1, appended⟩
    else
      let nextEmb : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := .Float32 } := m.embedTokens nextCol
      let (nextLogits, caches') ← decodeStepFromEmbedWithCache m cosAll sinAll nextEmb curSeq caches
      greedyLoopCached m cosAll sinAll eosTokenIds eosVector (remaining - 1) caches' nextLogits (some finished') appended

private partial def greedyLoopUncached {batch : UInt64}
    (m : Qwen3ForCausalLM (device := device) cfg)
    (eosTokenIds : Array UInt64)
    (eosVector : Option (Tensor { device := device, shape := #[batch], dtype := .Int64 }))
    (remaining : Nat)
    (finished : Option (Tensor { device := device, shape := #[batch], dtype := .Bool }))
    {curSeq : UInt64}
    (curIds : Tensor { device := device, shape := #[batch, curSeq], dtype := .Int64 })
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩
  if curSeq == 0 then
    throw <| IO.userError "generateGreedy requires non-empty prompt sequence"

  let logits := m.forward curIds none
  let lastPos := curSeq - 1
  let last3 : Tensor { device := device, shape := #[batch, 1, cfg.vocab_size], dtype := .Float32 } :=
    Tensor.reshapeChecked (Tensor.sliceChecked logits 1 lastPos 1 #[batch, 1, cfg.vocab_size]) #[batch, 1, cfg.vocab_size]
  let last2 : Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := .Float32 } := Tensor.reshapeChecked last3 #[batch, cfg.vocab_size]
  let nextTokRaw : Tensor { device := device, shape := #[batch], dtype := .Int64 } := (Tensor.argmax last2 1).reshapeChecked #[batch]
  let nextTok : Tensor { device := device, shape := #[batch], dtype := .Int64 } :=
    match eosVector, finished with
    | some eosTok, some doneMask =>
      let activeMask : Tensor { device := device, shape := #[batch], dtype := .Bool } := Tensor.logicalNot doneMask
      Tensor.whereSelect activeMask nextTokRaw eosTok
    | _, _ => nextTokRaw
  let nextCol : Tensor { device := device, shape := #[batch, 1], dtype := .Int64 } := Tensor.reshapeChecked nextTok #[batch, 1]
  let appended : Tensor { device := device, shape := #[batch, curSeq + 1], dtype := .Int64 } := Tensor.cat curIds nextCol 1

  match eosVector with
  | none =>
    greedyLoopUncached m eosTokenIds none (remaining - 1) none appended
  | some _ =>
    let reachedEos : Tensor { device := device, shape := #[batch], dtype := .Bool } := tokenInSetMask nextTok eosTokenIds
    let finished' : Tensor { device := device, shape := #[batch], dtype := .Bool } :=
      match finished with
      | some doneMask => Tensor.logicalOr doneMask reachedEos
      | none => reachedEos
    let hasActiveRows : Bool := Tensor.any (Tensor.logicalNot finished')
    if !hasActiveRows then
      return ⟨curSeq + 1, appended⟩
    else
      greedyLoopUncached m eosTokenIds eosVector (remaining - 1) (some finished') appended

/-- Greedy generation with static per-layer KV caches. -/
def generateGreedy {batch seq : UInt64}
    (m : Qwen3ForCausalLM (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  if seq == 0 then
    throw <| IO.userError "generateGreedy requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩

  let inputsEmbeds := m.embedTokens inputIds
  let cacheDevice := inputsEmbeds.device
  let cacheMaxLen : UInt64 := seq + maxNewTokens
  let eosVector := eosVectorOnDevice (batch := batch) eosTokenIds cacheDevice
  let (cosAll, sinAll) := precomputeDecodeRotary (maxLen := cacheMaxLen) m cacheDevice
  let caches0 := initLayerKVCaches m cacheMaxLen cacheDevice
  let tok0 : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := .Float32 } := Tensor.sliceChecked inputsEmbeds 1 0 1 #[batch, 1, cfg.hidden_size]
  let (logits0, caches1) ← decodeStepFromEmbedWithCache m cosAll sinAll tok0 0 caches0
  let (lastLogits, cachesPrefill) ←
    prefillCachesFromEmbeds m cosAll sinAll inputsEmbeds caches1 1 logits0
  greedyLoopCached m cosAll sinAll eosTokenIds eosVector maxNewTokens.toNat cachesPrefill lastLogits none inputIds

/-- Reference greedy generation by full re-forward on each decode step. -/
def generateGreedyUncached {batch seq : UInt64}
    (m : Qwen3ForCausalLM (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  if seq == 0 then
    throw <| IO.userError "generateGreedy requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩
  let eosVector := eosVectorOnDevice (batch := batch) eosTokenIds inputIds.device
  greedyLoopUncached m eosTokenIds eosVector maxNewTokens.toNat none inputIds

end Qwen3ForCausalLM

end torch.qwen3
