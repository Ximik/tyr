/-
  Tyr/Model/Qwen/ConfigIO.lean

  HuggingFace `config.json` fields shared by Qwen3 and Qwen2.5 text decoders.
-/
import Tyr.Model.Qwen.Config
import Tyr.Model.ConfigJson

namespace torch.qwen

open Lean ConfigJson

namespace QwenConfig

/-- Parse the text decoder from a `config.json` node. Behavior the shared layer
hard-codes (SiLU MLP, no attention bias, no sliding window, plain RoPE) is
required of the file; `allowMRope` additionally admits multimodal RoPE (type
`mrope`, or type `default` with an `mrope_section`), recorded as `mrope`: its
sections must cover the rotary frequency pairs, and `rope_scaling` may hold no
other key. Only a decoder whose tokens all have equal positions on the three
axes (text and audio, no image grid) may set `allowMRope`: there every layout
equals the plain RoPE the decoder computes. A missing `head_dim` is
`hidden_size / num_attention_heads` when `deriveHeadDim`, else `d.head_dim`. -/
def parseJson (j : Json) (d : QwenConfig) (allowMRope : Bool := false) (deriveHeadDim : Bool := true) :
    Except String QwenConfig := do
  requireOneOf j "hidden_act" ["silu"]
  requireEq j "attention_bias" false
  requireEq j "use_sliding_window" false
  let mrope : Option MRopeLayout ← match ← member? j "rope_scaling" with
  | none | some .null => pure none
  | some scaling =>
    let unsupported := s!"unsupported 'rope_scaling' = {scaling.compress}; this model implements plain RoPE"
    let kind : Option String ← firstFieldD? scaling ["type", "rope_type"] none
    let sections : Option (Array Nat) ← field? scaling "mrope_section"
    let some sections := sections | throw unsupported
    unless allowMRope && (kind == some "mrope" || kind == some "default") do throw unsupported
    let known := ["type", "rope_type", "mrope_section", "interleaved", "mrope_interleaved"]
    let .obj fields := scaling | throw unsupported
    if fields.toArray.any (fun (key, _) => !known.contains key) then throw unsupported
    let interleaved : Bool ← fieldD? scaling "mrope_interleaved" false
    let alsoInterleaved : Bool ← fieldD? scaling "interleaved" interleaved
    unless alsoInterleaved == interleaved do throw unsupported
    pure (some ({ sections, interleaved } : MRopeLayout))
  let hidden ← fieldD? j "hidden_size" d.hidden_size
  let heads ← fieldD? j "num_attention_heads" d.num_attention_heads
  let kvHeads ← fieldD? j "num_key_value_heads" d.num_key_value_heads
  let ⟨hHeads⟩ ← if h : 0 < heads ∧ 0 < kvHeads ∧ kvHeads ∣ heads then pure (PLift.up h) else
    throw s!"Invalid Qwen config: num_key_value_heads={kvHeads} must be positive and divide num_attention_heads={heads}"
  let headDim ← match ← field? j "head_dim" with
    | some headDim => pure headDim
    | none =>
      if !deriveHeadDim then pure d.head_dim
      else if heads ∣ hidden then pure (hidden / heads)
      else throw s!"Invalid Qwen config: without head_dim, num_attention_heads={heads} must divide hidden_size={hidden}"
  let ⟨hHeadDim⟩ ← if h : 0 < headDim ∧ 2 ∣ headDim then pure (PLift.up h) else
    throw s!"Invalid Qwen config: head_dim={headDim} must be positive and even"
  if let some layout := mrope then
    unless layout.sections.sum == headDim / 2 do
      throw s!"Invalid Qwen config: mrope_section {layout.sections} must sum to head_dim / 2 = {headDim / 2}"
  let vocab ← fieldD? j "vocab_size" d.vocab_size
  let ⟨hVocab⟩ ← if h : 0 < vocab then pure (PLift.up h) else
    throw "Invalid Qwen config: vocab_size must be positive"
  let theta ← fieldD? j "rope_theta" d.rope_theta
  let ⟨hTheta⟩ ← if h : 0 < theta then pure (PLift.up h) else
    throw s!"Invalid Qwen config: rope_theta={theta} must be positive"
  let eps ← fieldD? j "rms_norm_eps" d.rms_norm_eps
  let ⟨hEps⟩ ← if h : 0 < eps then pure (PLift.up h) else
    throw s!"Invalid Qwen config: rms_norm_eps={eps} must be positive"
  pure {
    vocab_size := vocab
    hidden_size := hidden
    intermediate_size := ← fieldD? j "intermediate_size" d.intermediate_size
    num_hidden_layers := ← fieldD? j "num_hidden_layers" d.num_hidden_layers
    num_attention_heads := heads
    num_key_value_heads := kvHeads
    half_head_dim := headDim / 2
    rope_theta := theta
    rms_norm_eps := eps
    mrope := mrope
    headsValid := hHeads
    halfHeadDimPos := by omega
    vocabPos := hVocab
    ropeThetaPos := hTheta
    rmsNormEpsPos := hEps
  }

end QwenConfig

end torch.qwen
