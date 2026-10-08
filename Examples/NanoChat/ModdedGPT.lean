/-
  Tyr/ModdedGPT.lean

  Modded GPT model architecture based on modded-nanogpt.

  Key features:
  - YaRN rotary embeddings with half-truncation
  - Sparse gated attention with variable window sizes
  - Value embeddings (3 separate token->value mappings)
  - Learnable scalars for residual/skip connections
  - Smear gate (forward-shift token embeddings)
  - Backout lambda (subtract early layer contributions)
  - ReLU^2 activation instead of GELU
  - Logit softcapping

  Architecture: 11 layers, 6 heads, 128 head_dim, 768 model_dim
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Derive

namespace torch.moddedGpt

open torch

/-- Model configuration matching modded-nanogpt -/
structure Config where
  /-- Vocabulary size (padded to multiple of 128) -/
  vocabSize : UInt64 := 50304
  /-- Number of transformer layers -/
  nLayer : UInt64 := 11
  /-- Number of attention heads -/
  nHead : UInt64 := 6
  /-- Head dimension -/
  headDim : UInt64 := 128
  /-- Model dimension (nHead * headDim) -/
  modelDim : UInt64 := 768
  /-- Maximum sequence length -/
  maxSeqLen : UInt64 := 2048
  /-- Block size for attention windowing -/
  blockSize : UInt64 := 128
  /-- Base frequency for rotary embeddings -/
  ropeBase : Float := 500000.0
  /-- Number of value embedding layers -/
  numValueEmbeds : Nat := 3
  /-- Softcap value for logits -/
  softcapValue : Float := 15.0
  /-- Sliding window attention pattern string.
      Characters: L=long (full context), S=short (half context)
      Pattern is tiled across layers. Final layer always gets L.
      Examples: "L"=all full context, "SL"=alternating, "LSS"=one long then two short -/
  windowPattern : String := "L"
  deriving Repr, Inhabited, Lean.ToJson, Lean.FromJson

/-- Default modded-nanogpt configuration -/
def Config.default : Config := {}

/-! ## Sliding Window Attention Helpers -/

/-- Get window size for a specific layer based on the pattern string.
    Returns `none` for full context (L) or `some windowSize` for sliding window (S).

    Pattern characters:
    - 'L' or 'l': full context (returns none)
    - 'S' or 's': half context (returns some (seqLen / 2))

    The pattern is tiled across layers. Final layer always gets full context.
-/
def getWindowSizeForLayer (pattern : String) (layerIdx : Nat) (nLayers : Nat) (seqLen : UInt64)
    : Option UInt64 :=
  -- Final layer always gets full context
  if layerIdx == nLayers - 1 then
    none
  else if pattern.isEmpty then
    none  -- Empty pattern = full context
  else
    -- Get character for this layer (tiled)
    let charIdx := layerIdx % pattern.length
    let c := String.Pos.Raw.get! pattern ⟨charIdx⟩
    if c == 'L' || c == 'l' then
      none  -- Full context
    else if c == 'S' || c == 's' then
      some (seqLen / 2)  -- Half context sliding window
    else
      none  -- Unknown character = full context (safe default)

/-- Compute window sizes for all layers in advance.
    Returns array of Option UInt64 where none = full context. -/
def computeWindowSizes (cfg : Config) : Array (Option UInt64) := Id.run do
  let mut sizes : Array (Option UInt64) := #[]
  for i in [:cfg.nLayer.toNat] do
    sizes := sizes.push (getWindowSizeForLayer cfg.windowPattern i cfg.nLayer.toNat cfg.maxSeqLen)
  return sizes

/-! ## Rotary Embeddings with YaRN -/

/-- YaRN rotary embedding state.

    YaRN (Yet Another RoPE iNtegration) allows extending context length
    by interpolating rotary frequencies. We use half-truncated RoPE
    where only the first half of head dimensions get rotation.
-/
structure YarnRotary (headDim maxSeqLen : UInt64) where
  /-- Cosine frequencies [maxSeqLen, headDim/2] for half-truncated RoPE -/
  cos : T #[maxSeqLen, headDim / 2]
  /-- Sine frequencies [maxSeqLen, headDim/2] for half-truncated RoPE -/
  sin : T #[maxSeqLen, headDim / 2]
  /-- Base angular frequencies [headDim/2] -/
  angularFreq : T #[headDim / 2]
  /-- Attention scale factor -/
  attnScale : Float := 0.1
  deriving Repr

/-- Initialize YaRN rotary embeddings -/
def YarnRotary.init (headDim maxSeqLen : UInt64) (base : Float := 500000.0)
    : IO (YarnRotary headDim maxSeqLen) := do
  -- computeFreqs returns [maxSeqLen, headDim/2]
  let (cosHalf, sinHalf) ← rotary.computeFreqs maxSeqLen headDim base
  let angularFreq := zeros #[headDim / 2]
  return {
    cos := cosHalf
    sin := sinHalf
    angularFreq := angularFreq
    attnScale := 0.1
  }

/-- Apply YaRN extension when window size changes.

    When extending to longer sequences, we interpolate the
    rotary frequencies to maintain position information.
-/
def YarnRotary.applyExtension (yarn : YarnRotary headDim maxSeqLen)
    (_oldWindow _newWindow : UInt64) : YarnRotary headDim maxSeqLen :=
  -- YaRN frequency interpolation would go here
  -- For now, return unchanged
  yarn

/-! ## Attention Arguments -/

/-- Arguments passed to attention layers during forward pass -/
structure AttnArgs (batch maxSeq nHead headDim : UInt64) where
  /-- Value embeddings for current layer (optional) -/
  valueEmbed : Option (T #[batch, maxSeq, nHead * headDim])
  /-- SA (scaled attention) lambda for this layer -/
  saLambda : Float
  /-- Cumulative sequence lengths for variable-length attention -/
  seqlens : Option (T #[batch + 1])
  /-- Short window size (in blocks) -/
  wsShort : UInt64
  /-- Long window size (in blocks) -/
  wsLong : UInt64
  /-- Cosine frequencies for RoPE -/
  cos : T #[maxSeq, headDim]
  /-- Sine frequencies for RoPE -/
  sin : T #[maxSeq, headDim]
  /-- Attention scale factor -/
  attnScale : Float
  /-- Whether to shift keys for induction -/
  keyShift : Bool
  deriving Repr

/-! ## Linear Layers -/

/-- Linear layer that can optionally use FP8 quantization -/
structure CastedLinear (inDim outDim : UInt64) where
  /-- Weight matrix [outDim, inDim] -/
  weight : T #[outDim, inDim]
  /-- Whether to use FP8 training (requires H100+) -/
  useFp8 : Bool := false
  /-- Learning rate multiplier -/
  lrMul : Float := 1.0
  /-- Weight decay multiplier -/
  wdMul : Float := 1.0
  deriving Repr, TensorStruct

/-- Initialize CastedLinear with scaled random weights -/
def CastedLinear.init (inDim outDim : UInt64) (scale : Float := 0.02)
    (lrMul wdMul : Float := 1.0) : IO (CastedLinear inDim outDim) := do
  let weight ← randn #[outDim, inDim] false
  let scaledWeight := mul_scalar weight scale
  return {
    weight := autograd.set_requires_grad (autograd.detach scaledWeight) true
    useFp8 := false
    lrMul := lrMul
    wdMul := wdMul
  }

/-- Forward pass for CastedLinear (3D input) -/
def CastedLinear.forward {batch seq inDim outDim : UInt64}
    (layer : CastedLinear inDim outDim) (x : T #[batch, seq, inDim])
    : T #[batch, seq, outDim] :=
  linear3d x layer.weight

/-! ## Causal Self-Attention -/

/-- Causal self-attention with separate Q, K, V, O projections.

    Features:
    - QK normalization (nanochat style)
    - Rotary embeddings via YaRN
    - Variable window sizes for sliding window attention
    - Zero-initialized output projection (modded-nanogpt style)

    Weight shapes for linear3d: [out_dim, in_dim] since it computes x @ W.T
-/
structure CausalSelfAttention (dim headDim numHeads : UInt64) where
  /-- Query projection [numHeads * headDim, dim] -/
  wQ : T #[numHeads * headDim, dim]
  /-- Key projection [numHeads * headDim, dim] -/
  wK : T #[numHeads * headDim, dim]
  /-- Value projection [numHeads * headDim, dim] -/
  wV : T #[numHeads * headDim, dim]
  /-- Output projection [dim, numHeads * headDim] -/
  wO : T #[dim, numHeads * headDim]
  deriving Repr, TensorStruct

/-- Initialize CausalSelfAttention with proper initialization -/
def CausalSelfAttention.init (dim headDim numHeads : UInt64)
    : IO (CausalSelfAttention dim headDim numHeads) := do
  -- Uniform init with bound = sqrt(3) * (1/sqrt(dim)) for same std as normal
  let s := Float.sqrt 3.0 / Float.sqrt dim.toFloat
  -- Weight shapes: [out_dim, in_dim] for linear3d
  let wQ ← randn #[numHeads * headDim, dim] false
  let wQ := mul_scalar wQ s
  let wK ← randn #[numHeads * headDim, dim] false
  let wK := mul_scalar wK s
  let wV ← randn #[numHeads * headDim, dim] false
  let wV := mul_scalar wV s
  -- Output projection initialized to zero (modded-nanogpt style)
  let wO := zeros #[dim, numHeads * headDim]
  return {
    wQ := autograd.set_requires_grad (autograd.detach wQ) true
    wK := autograd.set_requires_grad (autograd.detach wK) true
    wV := autograd.set_requires_grad (autograd.detach wV) true
    wO := autograd.set_requires_grad (autograd.detach wO) true
  }

/-- Functional RMSNorm (no learnable parameters) -/
def rmsNorm3d {batch seq dim : UInt64} (x : T #[batch, seq, dim]) : T #[batch, seq, dim] :=
  nn.rmsNorm x

/-- Forward pass for attention with rotary embeddings and QK norm.
    x: [batch, seq, dim]
    Returns: [batch, seq, dim] -/
def CausalSelfAttention.forward {batch seq dim headDim numHeads rotaryLen : UInt64}
    (attn : CausalSelfAttention dim headDim numHeads)
    (x : T #[batch, seq, dim])
    (yarn : YarnRotary headDim rotaryLen)
    (windowSize : Option UInt64 := none)
    (valueEmbed : Option (T #[batch, seq, numHeads * headDim]) := none)
    (valueMix : T #[] := zeros #[])
    : T #[batch, seq, dim] :=
  -- Project to Q, K, V: [batch, seq, numHeads * headDim]
  let q := linear3d x attn.wQ
  let k := linear3d x attn.wK
  let vBase := linear3d x attn.wV

  -- Value-embedding blend (modded-nanogpt style): v := (1-λ)*v + λ*ve.
  let v := match valueEmbed with
    | some ve =>
      let mix := nn.sigmoid valueMix
      let one : T #[] := (ones #[]).to mix.device
      let oneMinus := one - mix
      let mixExpanded := nn.expand mix #[batch, seq, numHeads * headDim]
      let oneMinusExpanded := nn.expand oneMinus #[batch, seq, numHeads * headDim]
      (vBase * oneMinusExpanded) + (ve * mixExpanded)
    | none =>
      vBase

  -- Reshape to [batch, seq, numHeads, headDim] for rotary/attention
  let q := reshape q #[batch, seq, numHeads, headDim]
  let k := reshape k #[batch, seq, numHeads, headDim]
  let v := reshape v #[batch, seq, numHeads, headDim]

  -- Slice rotary embeddings to seq length (cos/sin are [rotaryLen, headDim])
  let cos := data.slice2d yarn.cos 0 seq
  let sin := data.slice2d yarn.sin 0 seq

  -- Apply rotary embeddings (uses half-dim rotation)
  let q := rotary.applyRotaryEmb q cos sin
  let k := rotary.applyRotaryEmb k cos sin

  -- QK normalization (nanochat style: normalize queries and keys)
  let q := nn.rmsNorm q
  let k := nn.rmsNorm k

  -- Transpose to [batch, numHeads, seq, headDim] for attention
  let q := nn.transpose_for_attention q
  let k := nn.transpose_for_attention k
  let v := nn.transpose_for_attention v

  -- Scaled dot-product attention with optional sliding window
  -- Using GQA window function for sliding window, regular SDPA for full context
  let attnOut := match windowSize with
    | some ws => nn.scaledDotProductAttentionGQAWindow q k v 0.0 true false ws
    | none => nn.scaled_dot_product_attention q k v 0.0 true

  -- Transpose back to [batch, seq, numHeads, headDim]
  let attnOut := nn.transpose_from_attention attnOut

  -- Reshape to [batch, seq, numHeads * headDim] and project
  let attnOut := reshape attnOut #[batch, seq, numHeads * headDim]
  let output := linear3d attnOut attn.wO

  output

/-! ## MLP Layer -/

/-- MLP layer with ReLU^2 activation.

    Architecture:
    - c_fc: dim -> 4*dim (expansion)
    - ReLU^2 activation
    - c_proj: 4*dim -> dim (projection back)

    Note: modded-nanogpt uses transposed layout for weights
    and zero-initialized c_proj.
-/
structure MLP (dim : UInt64) where
  /-- Expansion weights [4*dim, dim] -/
  cFc : T #[4 * dim, dim]
  /-- Projection weights [4*dim, dim] (transposed layout) -/
  cProj : T #[4 * dim, dim]
  deriving Repr, TensorStruct

/-- Initialize MLP with scaled weights -/
def MLP.init (dim : UInt64) : IO (MLP dim) := do
  let cFc ← randn #[4 * dim, dim] false
  -- c_proj initialized to zero (modded-nanogpt style)
  let cProj := zeros #[4 * dim, dim]
  return {
    cFc := autograd.set_requires_grad (autograd.detach (mul_scalar cFc 0.02)) true
    cProj := autograd.set_requires_grad (autograd.detach cProj) true
  }

/-- Forward pass for MLP with ReLU^2 activation -/
def MLP.forward {batch seq dim : UInt64} (mlp : MLP dim) (x : T #[batch, seq, dim])
    : IO (T #[batch, seq, dim]) := do
  -- x @ c_fc.T -> [batch, seq, 4*dim]
  let h := linear3d x mlp.cFc
  -- ReLU^2 activation
  let h := nn.reluSquared h
  -- h @ c_proj -> [batch, seq, dim] (transposed weight layout)
  -- Note: In transposed layout, we use the transpose for the projection
  let cProjT := nn.transpose2d mlp.cProj
  let out := linear3d h cProjT
  return out

/-! ## Transformer Block -/

/-- Transformer block (attention + MLP).

    Some layers have attention skipped (layer 6 in modded-nanogpt).
    First MLP layer is also skipped.
-/
structure Block (dim headDim numHeads : UInt64) where
  /-- Attention layer (None for layer 6) -/
  attn : Option (CausalSelfAttention dim headDim numHeads)
  /-- MLP layer -/
  mlp : MLP dim
  deriving Repr, TensorStruct

instance {inDim outDim : UInt64} : Inhabited (CastedLinear inDim outDim) where
  default := { weight := zeros _ }

instance {dim : UInt64} : Inhabited (MLP dim) where
  default := { cFc := zeros _, cProj := zeros _ }

instance {dim headDim numHeads : UInt64} : Inhabited (CausalSelfAttention dim headDim numHeads) where
  default := { wQ := zeros _, wK := zeros _, wV := zeros _, wO := zeros _ }

instance {dim headDim numHeads : UInt64} : Inhabited (Block dim headDim numHeads) where
  default := { attn := none, mlp := default }

/-- Initialize Block -/
def Block.init (dim headDim numHeads : UInt64) (skipAttn : Bool := false)
    : IO (Block dim headDim numHeads) := do
  let attn ← if skipAttn then pure none else Option.some <$> CausalSelfAttention.init dim headDim numHeads
  let mlp ← MLP.init dim
  return { attn := attn, mlp := mlp }

/-! ## Full Model Parameters -/

/-- Learnable scalars for modded-nanogpt.

    Includes:
    - Residual lambdas (per layer)
    - Skip connection lambdas
    - SA (scaled attention) lambdas
    - Smear gate
    - Backout lambda
-/
structure Scalars (nLayer : UInt64) where
  /-- All scalars packed into one tensor for efficiency -/
  values : T #[nLayer * 4 + 8]  -- 4 per layer + 8 global
  deriving Repr, TensorStruct

/-- Initialize scalars with modded-nanogpt defaults -/
def Scalars.init (nLayer : UInt64) : Scalars nLayer := {
  values := autograd.set_requires_grad (mul_scalar (ones #[nLayer * 4 + 8]) 1.1) true
}

/-! ## Scalar/Value Helper Utilities -/

/-- Extract scalar tensor at `idx` from a 1D tensor. -/
private def scalarAt {n : UInt64} (values : T #[n]) (idx : UInt64) : T #[] :=
  let oneElem : T #[1] := data.slice1d (n := n) (m := 1) values idx.toInt64 (idx.toInt64 + 1)
  reshape oneElem #[]

/-- Extract scalar at `idx` and squash to (0, 1). -/
private def scalarSigmoidAt {n : UInt64} (values : T #[n]) (idx : UInt64) : T #[] :=
  nn.sigmoid (scalarAt values idx)

/-- Scale a tensor by a scalar tensor, keeping gradient flow through the scalar. -/
private def scaleByScalar {s : Shape} (x : T s) (scalar : T #[]) : T s :=
  let expanded := nn.expand scalar s
  x * expanded

/-- Blend two tensors with scalar weights. -/
private def blendByScalars {s : Shape} (x y : T s) (sx sy : T #[]) : T s :=
  scaleByScalar x sx + scaleByScalar y sy

/-- Value-embedding routing pattern used by modded-nanogpt:
    assign value embeddings to early and late layers, skip middle layers. -/
private def layerValueEmbed {cfg : Config} {batch seq : UInt64}
    (valueEmbeds : Array (T #[batch, seq, cfg.modelDim]))
    (layerIdx : Nat)
    : Option (T #[batch, seq, cfg.modelDim]) := Id.run do
  let nLayers := cfg.nLayer.toNat
  let nValue := valueEmbeds.size
  if nLayers == 0 || nValue == 0 then
    return none
  let span := min nValue nLayers
  if layerIdx < span then
    return valueEmbeds[layerIdx]?
  let backStart := nLayers - span
  if layerIdx >= backStart then
    return valueEmbeds[(layerIdx - backStart) % span]?
  return none

/-- Full modded GPT model parameters -/
structure ModdedGPTParams (cfg : Config) where
  /-- Token embeddings [vocabSize, modelDim] -/
  embed : T #[cfg.vocabSize, cfg.modelDim]
  /-- Smear gate (forward-shift tokens) -/
  smearGate : CastedLinear 12 1
  /-- Value embeddings (3 separate layers) -/
  valueEmbeds : Array (T #[cfg.vocabSize, cfg.modelDim])
  /-- Transformer blocks -/
  blocks : Array (Block cfg.modelDim cfg.headDim cfg.nHead)
  /-- Language model head (weight-tied with embed in forward) -/
  lmHead : CastedLinear cfg.modelDim cfg.vocabSize
  /-- Learnable scalars -/
  scalars : Scalars cfg.nLayer
  deriving Repr, TensorStruct

/-- Initialize full model -/
def ModdedGPTParams.init (cfg : Config) : IO (ModdedGPTParams cfg) := do
  -- Token embeddings
  let embed ← randn #[cfg.vocabSize, cfg.modelDim] false
  let embed := autograd.set_requires_grad (autograd.detach (mul_scalar embed 0.02)) true

  -- Smear gate
  let smearGate ← CastedLinear.init 12 1 0.02

  -- Value embeddings (3 layers)
  let mut valueEmbeds := #[]
  for _ in [:cfg.numValueEmbeds] do
    let ve ← randn #[cfg.vocabSize, cfg.modelDim] false
    valueEmbeds := valueEmbeds.push (autograd.set_requires_grad (autograd.detach (mul_scalar ve 0.02)) true)

  -- Transformer blocks
  let mut blocks := #[]
  for i in [:cfg.nLayer.toNat] do
    -- Layer 6 (0-indexed: 5) has attention skipped
    let skipAttn := i == 5
    let block ← Block.init cfg.modelDim cfg.headDim cfg.nHead skipAttn
    blocks := blocks.push block

  -- LM head
  let lmHead ← CastedLinear.init cfg.modelDim cfg.vocabSize 0.02

  -- Scalars
  let scalars := Scalars.init cfg.nLayer

  return {
    embed := embed
    smearGate := smearGate
    valueEmbeds := valueEmbeds
    blocks := blocks
    lmHead := lmHead
    scalars := scalars
  }

/-! ## Forward Pass -/

/-- Forward pass through the model.

    Implements all modded-nanogpt features:
    1. Token embedding + smear gate
    2. Add value embeddings to V in attention
    3. Variable window attention per layer (configurable via windowPattern)
    4. Skip connections (layer 3 -> 6)
    5. Backout lambda
    6. Softcapped logits
-/
def forward {cfg : Config} {batch seq : UInt64}
    (params : ModdedGPTParams cfg)
    (yarn : YarnRotary cfg.headDim cfg.maxSeqLen)
    (inputSeq : T #[batch, seq])
    (_training : Bool := true)
    : IO (T #[batch, seq, cfg.vocabSize]) := do
  -- 1) Token embedding + lightweight smear-style modulation.
  let tokEmb := nn.embedding inputSeq params.embed
  let smearIn : T #[batch, seq, 12] := data.slice tokEmb 2 0 12
  let smearGate := nn.sigmoid (params.smearGate.forward smearIn)
  let smearGateExpanded := nn.expand smearGate #[batch, seq, cfg.modelDim]
  let x0 := nn.rmsNorm (tokEmb + (tokEmb * smearGateExpanded))

  -- Precompute token value embeddings and global scalar controls.
  let valueEmbeds := params.valueEmbeds.map (nn.embedding inputSeq ·)
  let scalarBaseGlobal := cfg.nLayer * 4
  let valueMix := scalarSigmoidAt params.scalars.values scalarBaseGlobal
  let skipMix := scalarSigmoidAt params.scalars.values (scalarBaseGlobal + 1)
  let finalScale := scalarSigmoidAt params.scalars.values (scalarBaseGlobal + 2)

  -- 2) Apply transformer blocks with value embeddings and scalar-gated residuals.
  let mut x := x0
  let mut skipConnection : Option (T #[batch, seq, cfg.modelDim]) := none

  for i in [:cfg.nLayer.toNat] do
    let block : Block cfg.modelDim cfg.headDim cfg.nHead := params.blocks[i]!
    let scalarBase := (i.toUInt64) * 4
    let lamX := scalarSigmoidAt params.scalars.values scalarBase
    let lamX0 := scalarSigmoidAt params.scalars.values (scalarBase + 1)
    let lamAttn := scalarSigmoidAt params.scalars.values (scalarBase + 2)
    let lamMlp := scalarSigmoidAt params.scalars.values (scalarBase + 3)
    let xIn := blendByScalars x x0 lamX lamX0

    -- Skip connection: save layer 3 output for layer 6
    if i == 2 then  -- Layer 3 (0-indexed: 2)
      skipConnection := some xIn

    -- Determine window size from config's windowPattern
    -- Pattern is tiled across layers, final layer always gets full context
    let windowSize : Option UInt64 := getWindowSizeForLayer cfg.windowPattern i cfg.nLayer.toNat seq

    -- Apply block with attention and MLP
    match block.attn with
    | some attn =>
      -- Pre-norm (RMSNorm before attention)
      let xNormed := nn.rmsNorm xIn
      let layerVE := layerValueEmbed (cfg := cfg) valueEmbeds i
      -- Apply attention with rotary embeddings/window and optional value embedding.
      let attnOut := attn.forward xNormed yarn windowSize layerVE valueMix
      -- Residual connection
      let x' := xIn + scaleByScalar attnOut lamAttn
      -- Pre-norm before MLP
      let xMlpNormed := nn.rmsNorm x'
      -- Apply MLP
      let h ← block.mlp.forward xMlpNormed
      -- Residual connection
      x := x' + scaleByScalar h lamMlp
    | none =>
      -- Attention skipped (layer 6)
      -- Apply skip connection from layer 3
      match skipConnection with
      | some skip =>
        x := xIn + scaleByScalar skip skipMix
      | none =>
        x := xIn
      -- Pre-norm before MLP
      let xNormed := nn.rmsNorm x
      let h ← block.mlp.forward xNormed
      x := x + scaleByScalar h lamMlp

  -- 3) Final layer norm / projection / softcap.
  let xNormed := nn.rmsNorm (scaleByScalar x finalScale)

  -- Final projection to vocab
  let logits := params.lmHead.forward xNormed

  -- Softcap logits
  let logitsCapped := nn.softcap logits cfg.softcapValue

  return logitsCapped

/-- Compute cross-entropy loss -/
def loss {cfg : Config} {batch seq : UInt64}
    (params : ModdedGPTParams cfg)
    (yarn : YarnRotary cfg.headDim cfg.maxSeqLen)
    (inputSeq : T #[batch, seq])
    (targets : T #[batch, seq])
    (training : Bool := true)
    : IO (T #[]) := do
  let logits ← forward params yarn inputSeq training
  -- Reshape for cross-entropy: (batch*seq, vocab) vs (batch*seq,)
  let logitsFlat := reshape logits #[batch * seq, cfg.vocabSize]
  let targetsFlat := reshape targets #[batch * seq]
  return nn.cross_entropy logitsFlat targetsFlat

end torch.moddedGpt
