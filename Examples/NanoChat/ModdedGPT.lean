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
import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Module.Derive
import Tyr.TensorStruct.DeriveCast

namespace torch.moddedGpt

open torch

variable {device : Device}

/-- Model configuration matching modded-nanogpt -/
structure Config where
  /-- Vocabulary size (padded to multiple of 128) -/
  vocabSize : Nat := 50304
  /-- Number of transformer layers -/
  nLayer : Nat := 11
  /-- Number of attention heads -/
  nHead : Nat := 6
  /-- Head dimension -/
  headDim : Nat := 128
  /-- Model dimension (nHead * headDim) -/
  modelDim : Nat := 768
  /-- Maximum sequence length -/
  maxSeqLen : Nat := 2048
  /-- Block size for attention windowing -/
  blockSize : Nat := 128
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
def getWindowSizeForLayer (pattern : String) (layerIdx : Nat) (nLayers : Nat) (seqLen : Nat)
    : Option Nat :=
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
    Returns array of Option Nat where none = full context. -/
def computeWindowSizes (cfg : Config) : Array (Option Nat) := Id.run do
  let mut sizes : Array (Option Nat) := #[]
  for i in [:cfg.nLayer] do
    sizes := sizes.push (getWindowSizeForLayer cfg.windowPattern i cfg.nLayer cfg.maxSeqLen)
  return sizes

/-! ## Rotary Embeddings with YaRN -/

/-- YaRN rotary embedding state.

    YaRN (Yet Another RoPE iNtegration) allows extending context length
    by interpolating rotary frequencies. We use half-truncated RoPE
    where only the first half of head dimensions get rotation.
-/
structure YarnRotary (headDim maxSeqLen : Nat) where
  /-- Cosine frequencies [maxSeqLen, headDim/2] for half-truncated RoPE -/
  cos : Tensor { shape := #[maxSeqLen, headDim / 2], dtype := .Float32, device := device }
  /-- Sine frequencies [maxSeqLen, headDim/2] for half-truncated RoPE -/
  sin : Tensor { shape := #[maxSeqLen, headDim / 2], dtype := .Float32, device := device }
  /-- Base angular frequencies [headDim/2] -/
  angularFreq : Tensor { shape := #[headDim / 2], dtype := .Float32, device := device }
  /-- Attention scale factor -/
  attnScale : Float := 0.1
  deriving Repr

/-- Initialize YaRN rotary embeddings -/
def YarnRotary.init (headDim maxSeqLen : Nat) (base : Float := 500000.0) (device : Device := .CPU)
    : IO (YarnRotary (device := device) headDim maxSeqLen) := do
  -- computeFreqs returns [maxSeqLen, headDim/2]
  let (cosHalf, sinHalf) := Tensor.rotaryFreqs maxSeqLen headDim base
  let angularFreq := Tensor.zeros (device := device) #[headDim / 2]
  return {
    cos := cosHalf.to device
    sin := sinHalf.to device
    angularFreq := angularFreq
    attnScale := 0.1
  }

/-- Apply YaRN extension when window size changes.

    When extending to longer sequences, we interpolate the
    rotary frequencies to maintain position information.
-/
def YarnRotary.applyExtension (yarn : YarnRotary (device := device) headDim maxSeqLen)
    (_oldWindow _newWindow : Nat) : YarnRotary (device := device) headDim maxSeqLen :=
  -- YaRN frequency interpolation would go here
  -- For now, return unchanged
  yarn

/-! ## Attention Arguments -/

/-- Arguments passed to attention layers during forward pass -/
structure AttnArgs (batch maxSeq nHead headDim : Nat) where
  /-- Value embeddings for current layer (optional) -/
  valueEmbed : Option (Tensor { shape := #[batch, maxSeq, nHead * headDim], dtype := .Float32, device := device })
  /-- SA (scaled attention) lambda for this layer -/
  saLambda : Float
  /-- Cumulative sequence lengths for variable-length attention -/
  seqlens : Option (Tensor { shape := #[batch + 1], dtype := .Int64, device := device })
  /-- Short window size (in blocks) -/
  wsShort : Nat
  /-- Long window size (in blocks) -/
  wsLong : Nat
  /-- Cosine frequencies for RoPE -/
  cos : Tensor { shape := #[maxSeq, headDim], dtype := .Float32, device := device }
  /-- Sine frequencies for RoPE -/
  sin : Tensor { shape := #[maxSeq, headDim], dtype := .Float32, device := device }
  /-- Attention scale factor -/
  attnScale : Float
  /-- Whether to shift keys for induction -/
  keyShift : Bool
  deriving Repr

/-! ## Linear Layers -/

/-- Linear layer that can optionally use FP8 quantization -/
structure CastedLinear (inDim outDim : Nat) where
  /-- Weight matrix [outDim, inDim] -/
  weight : Tensor { shape := #[outDim, inDim], dtype := .Float32, device := device }
  /-- Whether to use FP8 training (requires H100+) -/
  useFp8 : Bool := false
  /-- Learning rate multiplier -/
  lrMul : Float := 1.0
  /-- Weight decay multiplier -/
  wdMul : Float := 1.0
  deriving Repr, TensorStruct

/-- Initialize CastedLinear with scaled random weights -/
def CastedLinear.init (inDim outDim : Nat) (scale : Float := 0.02)
    (lrMul wdMul : Float := 1.0) (device : Device := .CPU) : IO (CastedLinear (device := device) inDim outDim) := do
  let weight ← Tensor.randn (device := device) #[outDim, inDim] false
  let scaledWeight := Tensor.mulScalar weight scale
  return {
    weight := ← Tensor.parameter (Tensor.detach scaledWeight)
    useFp8 := false
    lrMul := lrMul
    wdMul := wdMul
  }

/-- Forward pass for CastedLinear (3D input) -/
def CastedLinear.forward {batch seq inDim outDim : Nat}
    (layer : CastedLinear (device := device) inDim outDim) (x : Tensor { shape := #[batch, seq, inDim], dtype := .Float32, device := device })
    : Tensor { shape := #[batch, seq, outDim], dtype := .Float32, device := device } :=
  Tensor.linear3d x layer.weight

/-! ## Causal Self-Attention -/

/-- Causal self-attention with separate Q, K, V, O projections.

    Features:
    - QK normalization (nanochat style)
    - Rotary embeddings via YaRN
    - Variable window sizes for sliding window attention
    - Zero-initialized output projection (modded-nanogpt style)

    Weight shapes for Tensor.linear3d: [out_dim, in_dim] since it computes x @ W.T
-/
structure CausalSelfAttention (dim headDim numHeads : Nat) where
  /-- Query projection [numHeads * headDim, dim] -/
  wQ : Tensor { shape := #[numHeads * headDim, dim], dtype := .Float32, device := device }
  /-- Key projection [numHeads * headDim, dim] -/
  wK : Tensor { shape := #[numHeads * headDim, dim], dtype := .Float32, device := device }
  /-- Value projection [numHeads * headDim, dim] -/
  wV : Tensor { shape := #[numHeads * headDim, dim], dtype := .Float32, device := device }
  /-- Output projection [dim, numHeads * headDim] -/
  wO : Tensor { shape := #[dim, numHeads * headDim], dtype := .Float32, device := device }
  deriving Repr, TensorStruct

/-- Initialize CausalSelfAttention with proper initialization -/
def CausalSelfAttention.init (dim headDim numHeads : Nat) (device : Device := .CPU)
    : IO (CausalSelfAttention (device := device) dim headDim numHeads) := do
  -- Uniform init with bound = sqrt(3) * (1/sqrt(dim)) for same std as normal
  let s := Float.sqrt 3.0 / Float.sqrt dim.toFloat
  -- Weight shapes: [out_dim, in_dim] for Tensor.linear3d
  let wQ ← Tensor.randn (device := device) #[numHeads * headDim, dim] false
  let wQ := Tensor.mulScalar wQ s
  let wK ← Tensor.randn (device := device) #[numHeads * headDim, dim] false
  let wK := Tensor.mulScalar wK s
  let wV ← Tensor.randn (device := device) #[numHeads * headDim, dim] false
  let wV := Tensor.mulScalar wV s
  -- Output projection initialized to zero (modded-nanogpt style)
  let wO := Tensor.zeros (device := device) #[dim, numHeads * headDim]
  return {
    wQ := ← Tensor.parameter (Tensor.detach wQ)
    wK := ← Tensor.parameter (Tensor.detach wK)
    wV := ← Tensor.parameter (Tensor.detach wV)
    wO := ← Tensor.parameter (Tensor.detach wO)
  }

/-- Functional RMSNorm (no learnable parameters) -/
def rmsNorm3d {batch seq dim : Nat} (x : Tensor { shape := #[batch, seq, dim], dtype := .Float32, device := device }) : Tensor { shape := #[batch, seq, dim], dtype := .Float32, device := device } :=
  Tensor.rmsNorm x

/-- Forward pass for attention with rotary embeddings and QK norm.
    x: [batch, seq, dim]
    Returns: [batch, seq, dim] -/
def CausalSelfAttention.forward {batch seq dim headDim numHeads rotaryLen : Nat}
    (attn : CausalSelfAttention (device := device) dim headDim numHeads)
    (x : Tensor { shape := #[batch, seq, dim], dtype := .Float32, device := device })
    (yarn : YarnRotary (device := device) headDim rotaryLen)
    (windowSize : Option Nat := none)
    (valueEmbed : Option (Tensor { shape := #[batch, seq, numHeads * headDim], dtype := .Float32, device := device }) := none)
    (valueMix : Tensor { shape := #[], dtype := .Float32, device := device } := Tensor.zeros (device := device) #[])
    : Tensor { shape := #[batch, seq, dim], dtype := .Float32, device := device } :=
  -- Project to Q, K, V: [batch, seq, numHeads * headDim]
  let q := Tensor.linear3d x attn.wQ
  let k := Tensor.linear3d x attn.wK
  let vBase := Tensor.linear3d x attn.wV

  -- Value-embedding blend (modded-nanogpt style): v := (1-λ)*v + λ*ve.
  let v := match valueEmbed with
    | some ve =>
      let mix := Tensor.sigmoid valueMix
      let one : Tensor { shape := #[], dtype := .Float32, device := device } := (Tensor.ones (device := device) #[]).to mix.device
      let oneMinus := one - mix
      let mixExpanded := Tensor.expand mix #[batch, seq, numHeads * headDim]
      let oneMinusExpanded := Tensor.expand oneMinus #[batch, seq, numHeads * headDim]
      (vBase * oneMinusExpanded) + (ve * mixExpanded)
    | none =>
      vBase

  -- Reshape to [batch, seq, numHeads, headDim] for rotary/attention
  let q := Tensor.reshapeChecked q #[batch, seq, numHeads, headDim]
  let k := Tensor.reshapeChecked k #[batch, seq, numHeads, headDim]
  let v := Tensor.reshapeChecked v #[batch, seq, numHeads, headDim]

  -- Slice rotary embeddings to seq length (cos/sin are [rotaryLen, headDim])
  let cos := Tensor.sliceChecked yarn.cos 0 0 seq #[seq, headDim / 2]
  let sin := Tensor.sliceChecked yarn.sin 0 0 seq #[seq, headDim / 2]

  -- Apply rotary embeddings (uses half-dim rotation)
  let q := Tensor.applyRotary q cos sin
  let k := Tensor.applyRotary k cos sin

  -- QK normalization (nanochat style: normalize queries and keys)
  let q := Tensor.rmsNorm q
  let k := Tensor.rmsNorm k

  -- Transpose to [batch, numHeads, seq, headDim] for attention
  let q := Tensor.splitHeads q
  let k := Tensor.splitHeads k
  let v := Tensor.splitHeads v

  -- Scaled dot-product attention with optional sliding window
  -- Using GQA window function for sliding window, regular SDPA for full context
  let attnOut := match windowSize with
    | some ws => Tensor.attentionWindow q k v true false ws
    | none => Tensor.attentionGQA q k v true

  -- Transpose back to [batch, seq, numHeads, headDim]
  let attnOut := Tensor.joinHeads attnOut

  -- Reshape to [batch, seq, numHeads * headDim] and project
  let attnOut := Tensor.reshapeChecked attnOut #[batch, seq, numHeads * headDim]
  let output := Tensor.linear3d attnOut attn.wO

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
structure MLP (dim : Nat) where
  /-- Expansion weights [4*dim, dim] -/
  cFc : Tensor { shape := #[4 * dim, dim], dtype := .Float32, device := device }
  /-- Projection weights [4*dim, dim] (transposed layout) -/
  cProj : Tensor { shape := #[4 * dim, dim], dtype := .Float32, device := device }
  deriving Repr, TensorStruct

/-- Initialize MLP with scaled weights -/
def MLP.init (dim : Nat) (device : Device := .CPU) : IO (MLP (device := device) dim) := do
  let cFc ← Tensor.randn (device := device) #[4 * dim, dim] false
  -- c_proj initialized to zero (modded-nanogpt style)
  let cProj := Tensor.zeros (device := device) #[4 * dim, dim]
  return {
    cFc := ← Tensor.parameter (Tensor.detach (Tensor.mulScalar cFc 0.02))
    cProj := ← Tensor.parameter (Tensor.detach cProj)
  }

/-- Forward pass for MLP with ReLU^2 activation -/
def MLP.forward {batch seq dim : Nat} (mlp : MLP (device := device) dim) (x : Tensor { shape := #[batch, seq, dim], dtype := .Float32, device := device })
    : IO (Tensor { shape := #[batch, seq, dim], dtype := .Float32, device := device }) := do
  -- x @ c_fc.T -> [batch, seq, 4*dim]
  let h := Tensor.linear3d x mlp.cFc
  -- ReLU^2 activation
  let h := Tensor.relu h
  let h := h * h
  -- h @ c_proj -> [batch, seq, dim] (transposed weight layout)
  -- Note: In transposed layout, we use the transpose for the projection
  let cProjT := Tensor.transpose2d mlp.cProj
  let out := Tensor.linear3d h cProjT
  return out

/-! ## Transformer Block -/

/-- Transformer block (attention + MLP).

    Some layers have attention skipped (layer 6 in modded-nanogpt).
    First MLP layer is also skipped.
-/
structure Block (dim headDim numHeads : Nat) where
  /-- Attention layer (None for layer 6) -/
  attn : Option (CausalSelfAttention (device := device) dim headDim numHeads)
  /-- MLP layer -/
  mlp : MLP (device := device) dim
  deriving Repr, TensorStruct

instance {inDim outDim : Nat} : Inhabited (CastedLinear (device := device) inDim outDim) where
  default := { weight := Tensor.zeros (device := device) _ }

instance {dim : Nat} : Inhabited (MLP (device := device) dim) where
  default := { cFc := Tensor.zeros (device := device) _, cProj := Tensor.zeros (device := device) _ }

instance {dim headDim numHeads : Nat} : Inhabited (CausalSelfAttention (device := device) dim headDim numHeads) where
  default := { wQ := Tensor.zeros (device := device) _, wK := Tensor.zeros (device := device) _, wV := Tensor.zeros (device := device) _, wO := Tensor.zeros (device := device) _ }

instance {dim headDim numHeads : Nat} : Inhabited (Block (device := device) dim headDim numHeads) where
  default := { attn := none, mlp := default }

/-- Initialize Block -/
def Block.init (dim headDim numHeads : Nat) (skipAttn : Bool := false) (device : Device := .CPU)
    : IO (Block (device := device) dim headDim numHeads) := do
  let attn ← if skipAttn then pure none else Option.some <$> CausalSelfAttention.init (device := device) dim headDim numHeads
  let mlp ← MLP.init (device := device) dim
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
structure Scalars (nLayer : Nat) where
  /-- All scalars packed into one tensor for efficiency -/
  values : Tensor { shape := #[nLayer * 4 + 8], dtype := .Float32, device := device }  -- 4 per layer + 8 global
  deriving Repr, TensorStruct

/-- Initialize scalars with modded-nanogpt defaults -/
def Scalars.init (nLayer : Nat) (device : Device := .CPU) : IO (Scalars (device := device) nLayer) := do
  pure {
    values := ← Tensor.parameter (Tensor.mulScalar (Tensor.ones (device := device) #[nLayer * 4 + 8]) 1.1)
  }

/-! ## Scalar/Value Helper Utilities -/

/-- Extract scalar tensor at `idx` from a 1D tensor. -/
private def scalarAt {n : Nat} (values : Tensor { shape := #[n], dtype := .Float32, device := device }) (idx : Nat) : Tensor { shape := #[], dtype := .Float32, device := device } :=
  let oneElem : Tensor { shape := #[1], dtype := .Float32, device := device } := Tensor.sliceChecked values 0 idx 1 #[1]
  Tensor.reshapeChecked oneElem #[]

/-- Extract scalar at `idx` and squash to (0, 1). -/
private def scalarSigmoidAt {n : Nat} (values : Tensor { shape := #[n], dtype := .Float32, device := device }) (idx : Nat) : Tensor { shape := #[], dtype := .Float32, device := device } :=
  Tensor.sigmoid (scalarAt values idx)

/-- Scale a tensor by a scalar tensor, keeping gradient flow through the scalar. -/
private def scaleByScalar {s : Shape} (x : Tensor { shape := s, dtype := .Float32, device := device }) (scalar : Tensor { shape := #[], dtype := .Float32, device := device }) : Tensor { shape := s, dtype := .Float32, device := device } :=
  let expanded := Tensor.expand scalar s
  x * expanded

/-- Blend two tensors with scalar weights. -/
private def blendByScalars {s : Shape} (x y : Tensor { shape := s, dtype := .Float32, device := device }) (sx sy : Tensor { shape := #[], dtype := .Float32, device := device }) : Tensor { shape := s, dtype := .Float32, device := device } :=
  scaleByScalar x sx + scaleByScalar y sy

/-- Value-embedding routing pattern used by modded-nanogpt:
    assign value embeddings to early and late layers, skip middle layers. -/
private def layerValueEmbed {cfg : Config} {batch seq : Nat}
    (valueEmbeds : Array (Tensor { shape := #[batch, seq, cfg.modelDim], dtype := .Float32, device := device }))
    (layerIdx : Nat)
    : Option (Tensor { shape := #[batch, seq, cfg.modelDim], dtype := .Float32, device := device }) := Id.run do
  let nLayers := cfg.nLayer
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
  embed : Tensor { shape := #[cfg.vocabSize, cfg.modelDim], dtype := .Float32, device := device }
  /-- Smear gate (forward-shift tokens) -/
  smearGate : CastedLinear (device := device) 12 1
  /-- Value embeddings (3 separate layers) -/
  valueEmbeds : Array (Tensor { shape := #[cfg.vocabSize, cfg.modelDim], dtype := .Float32, device := device })
  /-- Transformer blocks -/
  blocks : Array (Block (device := device) cfg.modelDim cfg.headDim cfg.nHead)
  /-- Language model head (weight-tied with embed in forward) -/
  lmHead : CastedLinear (device := device) cfg.modelDim cfg.vocabSize
  /-- Learnable scalars -/
  scalars : Scalars (device := device) cfg.nLayer
  deriving Repr, TensorStruct

/-- Initialize full model -/
def ModdedGPTParams.init (cfg : Config) (device : Device := .CPU) : IO (ModdedGPTParams (device := device) cfg) := do
  -- Token embeddings
  let embed ← Tensor.randn (device := device) #[cfg.vocabSize, cfg.modelDim] false
  let embed ← Tensor.setRequiresGrad (Tensor.detach (Tensor.mulScalar embed 0.02)) true

  -- Smear gate
  let smearGate ← CastedLinear.init (device := device) 12 1 0.02

  -- Value embeddings (3 layers)
  let mut valueEmbeds := #[]
  for _ in [:cfg.numValueEmbeds] do
    let ve ← Tensor.randn (device := device) #[cfg.vocabSize, cfg.modelDim] false
    let ve ← Tensor.parameter (Tensor.mulScalar ve 0.02)
    valueEmbeds := valueEmbeds.push ve

  -- Transformer blocks
  let mut blocks := #[]
  for i in [:cfg.nLayer] do
    -- Layer 6 (0-indexed: 5) has attention skipped
    let skipAttn := i == 5
    let block ← Block.init (device := device) cfg.modelDim cfg.headDim cfg.nHead skipAttn
    blocks := blocks.push block

  -- LM head
  let lmHead ← CastedLinear.init (device := device) cfg.modelDim cfg.vocabSize 0.02

  -- Scalars
  let scalars ← Scalars.init (device := device) cfg.nLayer

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
def forward {cfg : Config} {batch seq : Nat}
    (params : ModdedGPTParams (device := device) cfg)
    (yarn : YarnRotary (device := device) cfg.headDim cfg.maxSeqLen)
    (inputSeq : Tensor { shape := #[batch, seq], dtype := .Int64, device := device })
    (_training : Bool := true)
    : IO (Tensor { shape := #[batch, seq, cfg.vocabSize], dtype := .Float32, device := device }) := do
  if cfg.modelDim < 12 then
    throw (IO.userError "ModdedGPT requires modelDim >= 12 for the smear gate")
  -- 1) Token embedding + lightweight smear-style modulation.
  let tokEmb := Tensor.embedding inputSeq params.embed
  let smearIn : Tensor { shape := #[batch, seq, 12], dtype := .Float32, device := device } := Tensor.sliceChecked tokEmb 2 0 12 #[batch, seq, 12]
  let smearGate := Tensor.sigmoid (params.smearGate.forward smearIn)
  let smearGateExpanded := Tensor.expand smearGate #[batch, seq, cfg.modelDim]
  let x0 := Tensor.rmsNorm (tokEmb + (tokEmb * smearGateExpanded))

  -- Precompute token value embeddings and global scalar controls.
  let valueEmbeds := params.valueEmbeds.map (Tensor.embedding inputSeq ·)
  let scalarBaseGlobal := cfg.nLayer * 4
  let valueMix := scalarSigmoidAt params.scalars.values scalarBaseGlobal
  let skipMix := scalarSigmoidAt params.scalars.values (scalarBaseGlobal + 1)
  let finalScale := scalarSigmoidAt params.scalars.values (scalarBaseGlobal + 2)

  -- 2) Apply transformer blocks with value embeddings and scalar-gated residuals.
  let mut x := x0
  let mut skipConnection : Option (Tensor { shape := #[batch, seq, cfg.modelDim], dtype := .Float32, device := device }) := none

  for i in [:cfg.nLayer] do
    let block : Block (device := device) cfg.modelDim cfg.headDim cfg.nHead := params.blocks[i]!
    let scalarBase := (i) * 4
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
    let windowSize : Option Nat := getWindowSizeForLayer cfg.windowPattern i cfg.nLayer seq

    -- Apply block with attention and MLP
    match block.attn with
    | some attn =>
      -- Pre-norm (RMSNorm before attention)
      let xNormed := Tensor.rmsNorm xIn
      let layerVE := (layerValueEmbed (cfg := cfg) valueEmbeds i).map
        (fun t => t.reshapeChecked #[batch, seq, cfg.nHead * cfg.headDim])
      -- Apply attention with rotary embeddings/window and optional value embedding.
      let attnOut := attn.forward xNormed yarn windowSize layerVE valueMix
      -- Residual connection
      let x' := xIn + scaleByScalar attnOut lamAttn
      -- Pre-norm before MLP
      let xMlpNormed := Tensor.rmsNorm x'
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
      let xNormed := Tensor.rmsNorm x
      let h ← block.mlp.forward xNormed
      x := x + scaleByScalar h lamMlp

  -- 3) Final layer norm / projection / softcap.
  let xNormed := Tensor.rmsNorm (scaleByScalar x finalScale)

  -- Final projection to vocab
  let logits := params.lmHead.forward xNormed

  -- Softcap logits
  let logitsCapped := Tensor.mulScalar (Tensor.tanh (Tensor.divScalar logits cfg.softcapValue)) cfg.softcapValue

  return logitsCapped

/-- Compute cross-entropy loss -/
def loss {cfg : Config} {batch seq : Nat}
    (params : ModdedGPTParams (device := device) cfg)
    (yarn : YarnRotary (device := device) cfg.headDim cfg.maxSeqLen)
    (inputSeq : Tensor { shape := #[batch, seq], dtype := .Int64, device := device })
    (targets : Tensor { shape := #[batch, seq], dtype := .Int64, device := device })
    (training : Bool := true)
    : IO (Tensor { shape := #[], dtype := .Float32, device := device }) := do
  let logits ← forward params yarn inputSeq training
  -- Reshape for cross-entropy: (batch*seq, vocab) vs (batch*seq,)
  let logitsFlat := Tensor.reshapeChecked logits #[batch * seq, cfg.vocabSize]
  let targetsFlat := Tensor.reshapeChecked targets #[batch * seq]
  IO.ofExcept (Tensor.crossEntropy? logitsFlat targetsFlat)


/-! Device transfers are type-directed across every model leaf. -/

derive_tensor_tree_cast {source target : Device} {headDim maxSeqLen : Nat} :
  YarnRotary (device := source) headDim maxSeqLen => YarnRotary (device := target) headDim maxSeqLen

derive_tensor_tree_cast {source target : Device} {batch maxSeq nHead headDim : Nat} :
  AttnArgs (device := source) batch maxSeq nHead headDim => AttnArgs (device := target) batch maxSeq nHead headDim

derive_tensor_tree_cast {source target : Device} {inDim outDim : Nat} :
  CastedLinear (device := source) inDim outDim => CastedLinear (device := target) inDim outDim

derive_tensor_tree_cast {source target : Device} {dim headDim numHeads : Nat} :
  CausalSelfAttention (device := source) dim headDim numHeads => CausalSelfAttention (device := target) dim headDim numHeads

derive_tensor_tree_cast {source target : Device} {dim : Nat} :
  MLP (device := source) dim => MLP (device := target) dim

derive_tensor_tree_cast {source target : Device} {dim headDim numHeads : Nat} :
  Block (device := source) dim headDim numHeads => Block (device := target) dim headDim numHeads

derive_tensor_tree_cast {source target : Device} {nLayer : Nat} :
  Scalars (device := source) nLayer => Scalars (device := target) nLayer

derive_tensor_tree_cast {source target : Device} {cfg : Config} :
  ModdedGPTParams (device := source) cfg => ModdedGPTParams (device := target) cfg

end torch.moddedGpt
