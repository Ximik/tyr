/-
  Discrete Diffusion Transformer for Tyr

  A masked discrete diffusion model following tiny-diffusion architecture.
  Key differences from GPT:
  - Bidirectional attention (is_causal=false)
  - Time step conditioning via embeddings
  - RMSNorm without learnable parameters
  - ReLU² activation in MLP
  - Rotary positional embeddings
  - No bias in linear layers
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Derive
import Examples.Diffusion.Parameters

namespace torch.diffusion

open torch
open TensorStruct

/-- RMSNorm without learnable parameters (functional) -/
def norm {s : Shape} (x : Activation s) : Activation s :=
  Tensor.rmsNorm x

/-- Diffusion model configuration -/
structure Config where
  seq_len : UInt64 := 256         -- sequence length for generation
  vocab_size : UInt64 := 128      -- ASCII vocab (0-127), where 0 = [MASK]
  mask_token_id : UInt64 := 0     -- NUL character as mask token
  n_layer : UInt64 := 4           -- transformer blocks (smaller for testing)
  n_head : UInt64 := 4            -- attention heads
  n_embd : UInt64 := 256          -- embedding dimension (smaller for testing)
  diffusion_steps : UInt64 := 64  -- timesteps for diffusion (smaller for testing)
  context_len : UInt64 := 16      -- prefix tokens never masked
  deriving Repr, Inhabited

/-- Compute head dimension from config -/
def Config.headDim (cfg : Config) : UInt64 := cfg.n_embd / cfg.n_head

/-- Tiny config for unit tests -/
def Config.tiny : Config :=
  { seq_len := 64, vocab_size := 128, n_layer := 2, n_head := 2, n_embd := 64, diffusion_steps := 16 }

/-- Small config for testing (no context for simpler sampling) -/
def Config.small : Config :=
  { seq_len := 128, vocab_size := 128, n_layer := 4, n_head := 4, n_embd := 256, diffusion_steps := 64, context_len := 0 }

/-- Full config matching tiny-diffusion paper -/
def Config.full : Config :=
  { seq_len := 256, vocab_size := 128, n_layer := 6, n_head := 6, n_embd := 384, diffusion_steps := 128 }

/-- Full diffusion model parameters -/
structure DiffusionParams (cfg : Config) where
  -- Token embedding
  token_emb : Activation #[cfg.vocab_size, cfg.n_embd]
  -- Time embedding for timesteps
  time_emb : Activation #[cfg.diffusion_steps, cfg.n_embd]
  -- Transformer blocks with equal query and KV head counts.
  blocks : Array (BlockParams cfg.n_embd cfg.n_head cfg.n_head)
  -- Output head (NOT tied to token_emb)
  output_head : Activation #[cfg.vocab_size, cfg.n_embd]

  deriving TensorStruct

/-- Initialize diffusion model parameters -/
def DiffusionParams.init (cfg : Config) : IO (DiffusionParams cfg) := do
  if cfg.n_head == 0 || cfg.n_embd == 0 || cfg.n_embd % cfg.n_head != 0 ||
      cfg.headDim % 2 != 0 || cfg.seq_len == 0 || cfg.diffusion_steps == 0 ||
      cfg.context_len > cfg.seq_len || cfg.mask_token_id >= cfg.vocab_size then
    throw <| IO.userError "Invalid diffusion dimensions, rotary head width, context, or mask token"
  let scale := 1.0 / Float.sqrt cfg.n_embd.toFloat

  -- Token and time embeddings
  let token_emb ← Tensor.randn #[cfg.vocab_size, cfg.n_embd] false
  let time_emb ← Tensor.randn #[cfg.diffusion_steps, cfg.n_embd] false

  -- Transformer blocks
  let mut blocks := #[]
  for _ in [:cfg.n_layer.toNat] do
    let block ← BlockParams.init cfg.n_embd cfg.n_head cfg.n_head
    blocks := blocks.push block

  -- Output head (zero-initialized like tiny-diffusion)
  let output_head := Tensor.zeros #[cfg.vocab_size, cfg.n_embd]

  return {
    token_emb := ← makeLeafParam ((token_emb * scale))
    time_emb := ← makeLeafParam time_emb  -- standard normal init for embeddings
    blocks := blocks
    output_head := ← makeLeafParam output_head
  }

/-- Bidirectional attention forward pass (is_causal=false) -/
def bidirectionalAttentionForward {batch seq n_embd n_head : UInt64}
    (params : AttentionParams n_embd n_head n_head)
    (x : Activation #[batch, seq, n_embd])
    (rotaryCache : RotaryCache rotaryLen (n_embd / n_head))
    : Activation #[batch, seq, n_embd] :=
  let head_dim := n_embd / n_head

  -- Project to Q, K, V (no bias)
  let q := Tensor.linear3d x params.c_q
  let k := Tensor.linear3d x params.c_k
  let v := Tensor.linear3d x params.c_v

  -- Reshape: [batch, seq, n*d] -> [batch, seq, n, d]
  let q := Tensor.reshapeChecked q #[batch, seq, n_head, head_dim]
  let k := Tensor.reshapeChecked k #[batch, seq, n_head, head_dim]
  let v := Tensor.reshapeChecked v #[batch, seq, n_head, head_dim]

  -- Apply rotary embeddings (use full cache - no slicing needed when rotaryLen == seq)
  let cos := rotaryCache.cos.sliceChecked 0 0 seq #[seq, head_dim / 2]
  let sin := rotaryCache.sin.sliceChecked 0 0 seq #[seq, head_dim / 2]
  let q := Tensor.applyRotary q cos sin
  let k := Tensor.applyRotary k cos sin

  -- QK normalization
  let q := norm q
  let k := norm k

  -- Transpose for attention: [batch, seq, n, d] -> [batch, n, seq, d]
  let q := Tensor.splitHeads q
  let k := Tensor.splitHeads k
  let v := Tensor.splitHeads v

  -- Bidirectional attention (is_causal=false)
  let attn := Tensor.attentionGQA q k v false

  -- Reshape back: [batch, n, seq, d] -> [batch, seq, n*d]
  let attn := Tensor.joinHeads attn
  let attn := Tensor.reshapeChecked attn #[batch, seq, n_embd]

  -- Output projection (no bias)
  Tensor.linear3d attn params.c_proj

/-- MLP forward pass with ReLU². -/
def mlpForward {batch seq n_embd : UInt64}
    (params : MLPParams n_embd)
    (x : Activation #[batch, seq, n_embd])
    : Activation #[batch, seq, n_embd] :=
  let h := Tensor.linear3d x params.c_fc      -- Expand to 4x
  let h := h.relu * h.relu            -- ReLU² activation
  Tensor.linear3d h params.c_proj             -- Project back

/-- Block forward pass (pre-norm with bidirectional attention) -/
def blockForward {batch seq n_embd n_head : UInt64}
    (params : BlockParams n_embd n_head n_head)
    (x : Activation #[batch, seq, n_embd])
    (rotaryCache : RotaryCache rotaryLen (n_embd / n_head))
    : Activation #[batch, seq, n_embd] :=
  -- Attention with residual (pre-norm)
  let x := x + bidirectionalAttentionForward params.attn (norm x) rotaryCache
  -- MLP with residual (pre-norm)
  x + mlpForward params.mlp (norm x)

/-- Full diffusion model forward pass -/
def forward {cfg : Config} {batch seq : UInt64}
    (params : DiffusionParams cfg)
    (x_t : Tokens #[batch, seq])              -- Noisy tokens (with masks)
    (t : Tokens #[batch])                     -- Timestep indices
    (rotaryCache : RotaryCache rotaryLen cfg.headDim)
    : Activation #[batch, seq, cfg.vocab_size] :=
  -- Token embedding
  let x := Tensor.embedding x_t params.token_emb  -- [batch, seq, n_embd]

  -- Time embedding: lookup and broadcast to all positions
  let t_emb := Tensor.embedding1d t params.time_emb  -- [batch, n_embd]
  let t_emb := Tensor.unsqueeze t_emb 1              -- [batch, 1, n_embd]
  let t_emb := Tensor.expand t_emb #[batch, seq, cfg.n_embd]  -- [batch, seq, n_embd]
  let x := x + t_emb
  -- Norm after combining embeddings (matching tiny-diffusion)
  let x := norm x

  -- Transformer blocks
  let x := params.blocks.foldl (fun x block =>
    blockForward block x rotaryCache
  ) x

  -- Final norm and output projection
  let x := norm x
  Tensor.linear3d x params.output_head

/-- Compute diffusion loss (cross-entropy on all positions, masking done separately) -/
def loss {cfg : Config} {batch seq : UInt64}
    (logits : Activation #[batch, seq, cfg.vocab_size])
    (targets : Tokens #[batch, seq])
    : Activation #[] :=
  let logitsFlat := Tensor.reshapeChecked logits #[batch * seq, cfg.vocab_size]
  let targetsFlat := Tensor.reshapeChecked targets #[batch * seq]
  Tensor.crossEntropy logitsFlat targetsFlat

/-- Helper to get mask of positions that are still masked -/
def getMaskedPositions' {batch seq : UInt64}
    (x : Tokens #[batch, seq])
    (mask_token_id : UInt64)
    : Mask #[batch, seq] :=
  Tensor.eqScalar x mask_token_id.toInt64

private def samplingMask {batch seq : UInt64} (x : Tokens #[batch, seq])
    (maskToken protectedPrefix : UInt64) : Mask #[batch, seq] :=
  let protectedMask := ((Tensor.arange seq).to x.device).ltScalar protectedPrefix.toFloat
    |>.reshapeChecked #[1, seq] |>.expand #[batch, seq]
  Tensor.logicalAnd (getMaskedPositions' x maskToken) protectedMask.logicalNot

private def validateSampling (cfg : Config) (batch : UInt64) (temperature : Float) : IO Unit := do
  if batch == 0 || cfg.seq_len == 0 || cfg.diffusion_steps == 0 || cfg.context_len > cfg.seq_len ||
      temperature <= 0 || !temperature.isFinite then
    throw <| IO.userError "Diffusion sampling requires valid dimensions and finite positive temperature"

/-- Sample using confidence-aware parallel decoding.
    At each step, decode all tokens whose confidence exceeds a threshold.
    Ensures at least one token is decoded per step if any remain masked.
-/
def sampleConfidence {cfg : Config} {batch : UInt64}
    (params : DiffusionParams cfg)
    (rotaryCache : RotaryCache rotaryLen cfg.headDim)
    (confidenceThreshold : Float := 0.5)
    (temperature : Float := 1.0)
    (maxSteps : Nat := 256)
    (contextTokens : Option (Tokens #[batch, cfg.context_len]) := none)
    : IO (Tokens #[batch, cfg.seq_len]) := do
  validateSampling cfg batch temperature
  -- Start with all mask tokens
  let mut x := Tensor.fullInt #[batch, cfg.seq_len] cfg.mask_token_id.toInt64

  -- If context tokens provided, set them in first context_len positions
  if let some ctx := contextTokens then
    -- Create position mask: true for positions < context_len
    let positions := Tensor.arange cfg.seq_len
    let contextMask := Tensor.ltScalar positions cfg.context_len.toFloat
    let contextMaskBatch := Tensor.expand (Tensor.unsqueeze contextMask 0) #[batch, cfg.seq_len]
    -- Pad ctx to full seq_len with mask tokens
    let padding := Tensor.fullInt #[batch, cfg.seq_len - cfg.context_len] cfg.mask_token_id.toInt64
    let ctxPadded := (Tensor.cat ctx padding 1).reshapeChecked #[batch, cfg.seq_len]
    -- Blend: context positions get ctx, rest stays masked
    x := Tensor.whereSelect contextMaskBatch ctxPadded x

  -- Track which positions are still masked
  let mut maskedPositions := samplingMask x cfg.mask_token_id (if contextTokens.isSome then cfg.context_len else 0)

  for step in [:maxSteps] do
    -- Check if all tokens are decoded
    let anyMasked := Tensor.any maskedPositions
    if !anyMasked then
      if step < 10 then
        IO.println s!"  All tokens decoded at step {step}"
      break

    -- Create timestep tensor: start at T-1 (fully masked) and decrease toward 0
    -- During training: t=T-1 means high masking, t=0 means low masking
    -- During sampling: we start fully masked, so use t=T-1 and decrease
    let tVal := (cfg.diffusion_steps.toNat - 1 - step).max 0
    let t := Tensor.fullInt #[batch] tVal.toUInt64.toInt64

    -- Predict tokens
    let logits := forward params x t rotaryCache

    -- Apply temperature and softmax
    let scaledLogits := logits / temperature
    let probs := Tensor.softmax scaledLogits  -- [batch, seq, vocab]

    -- Get max probability and predicted token per position
    let (confidences, predictedTokens) ← IO.ofExcept (Tensor.maxLast3d? probs)  -- along vocab dim

    if step < 3 then
      -- Debug: show first few predicted tokens and their confidences
      let conf0 := (confidences.reshapeChecked #[batch * cfg.seq_len]).sliceChecked 0 0 1 #[1] |>.reshapeChecked #[]
      let tok0 := (predictedTokens.reshapeChecked #[batch * cfg.seq_len]).sliceChecked 0 0 1 #[1] |>.reshapeChecked #[]
      IO.println s!"  step {step}: conf[0]={Tensor.item conf0}, tok[0]={Tensor.itemInt tok0}"

    -- Select positions above threshold AND still masked
    let aboveThreshold := Tensor.ge confidences (Tensor.full #[batch, cfg.seq_len] confidenceThreshold)
    let selectMask := Tensor.logicalAnd aboveThreshold maskedPositions

    -- Ensure at least one token decoded per step if any remain
    -- If no positions above threshold, use top-k fallback (decode k=1 position)
    let anyAbove := Tensor.any selectMask
    let selectMask ← if anyAbove then
      pure selectMask
    else do
      -- Fallback: decode top-1 most confident masked position using scatter
      let negInf := Tensor.full #[batch, cfg.seq_len] (-1e10 : Float)
      let maskedConf := Tensor.whereSelect maskedPositions confidences negInf
      let (_, topkIdx) ← IO.ofExcept (Tensor.topKLast2d? maskedConf 1)  -- [batch, 1]
      let topkIdxLong := topkIdx
      let oneHot := Tensor.zeros #[batch, cfg.seq_len]
      let onesK := Tensor.ones #[batch, 1]
      let scattered := Tensor.scatter2d oneHot 1 topkIdxLong onesK
      pure (Tensor.logicalAnd (Tensor.lt (Tensor.zeros #[batch, cfg.seq_len]) scattered) maskedPositions)

    -- Update x where selectMask is true
    let predictedLong := predictedTokens
    x := Tensor.whereSelect selectMask predictedLong x

    -- Update masked positions
    maskedPositions := samplingMask x cfg.mask_token_id (if contextTokens.isSome then cfg.context_len else 0)

  return x

/-- Sample using top-K parallel decoding.
    At each step, decode exactly K tokens with highest confidence.
-/
def sampleTopK {cfg : Config} {batch : UInt64}
    (params : DiffusionParams cfg)
    (rotaryCache : RotaryCache rotaryLen cfg.headDim)
    (k : UInt64)
    (temperature : Float := 1.0)
    (maxSteps : Nat := 256)
    (contextTokens : Option (Tokens #[batch, cfg.context_len]) := none)
    : IO (Tokens #[batch, cfg.seq_len]) := do
  validateSampling cfg batch temperature
  if k == 0 || k > cfg.seq_len then
    throw <| IO.userError "Diffusion top-k must lie between 1 and sequence length"
  -- Start with all mask tokens
  let mut x := Tensor.fullInt #[batch, cfg.seq_len] cfg.mask_token_id.toInt64

  -- If context tokens provided, set them in first context_len positions
  if let some ctx := contextTokens then
    let positions := Tensor.arange cfg.seq_len
    let contextMask := Tensor.ltScalar positions cfg.context_len.toFloat
    let contextMaskBatch := Tensor.expand (Tensor.unsqueeze contextMask 0) #[batch, cfg.seq_len]
    let padding := Tensor.fullInt #[batch, cfg.seq_len - cfg.context_len] cfg.mask_token_id.toInt64
    let ctxPadded := (Tensor.cat ctx padding 1).reshapeChecked #[batch, cfg.seq_len]
    x := Tensor.whereSelect contextMaskBatch ctxPadded x

  -- Track which positions are still masked
  let mut maskedPositions := samplingMask x cfg.mask_token_id (if contextTokens.isSome then cfg.context_len else 0)

  for step in [:maxSteps] do
    -- Check if all tokens are decoded
    if !(Tensor.any maskedPositions) then
      break

    -- Create timestep tensor: start at T-1 (fully masked) and decrease toward 0
    let tVal := (cfg.diffusion_steps.toNat - 1 - step).max 0
    let t := Tensor.fullInt #[batch] tVal.toUInt64.toInt64

    -- Predict tokens
    let logits := forward params x t rotaryCache

    -- Apply temperature and softmax
    let scaledLogits := logits / temperature
    let probs := Tensor.softmax scaledLogits

    -- Get max probability and predicted token per position
    let (confidences, predictedTokens) ← IO.ofExcept (Tensor.maxLast3d? probs)

    -- Mask out already-decoded positions (set confidence to -inf)
    let negInf := Tensor.full #[batch, cfg.seq_len] (-1e10)
    let maskedConfidences := Tensor.whereSelect maskedPositions confidences negInf

    -- Get top-k positions per batch (properly typed)
    let (_, topkIndices) ← IO.ofExcept (Tensor.topKLast2d? maskedConfidences k)  -- [batch, k]

    -- Create selection mask using scatter: scatter 1s at topk positions
    let selectMask := Tensor.zeros #[batch, cfg.seq_len]
    let selectedOnes := Tensor.ones #[batch, k]
    let topkIndicesLong := topkIndices  -- scatter expects int64 indices
    let selectMaskScattered := Tensor.scatter2d selectMask 1 topkIndicesLong selectedOnes
    let selectMaskBool := Tensor.lt (Tensor.zeros #[batch, cfg.seq_len]) selectMaskScattered

    -- Only update positions in top-k AND still masked
    let finalMask := Tensor.logicalAnd selectMaskBool maskedPositions

    -- Update x
    let predictedLong := predictedTokens
    x := Tensor.whereSelect finalMask predictedLong x

    -- Update masked positions
    maskedPositions := samplingMask x cfg.mask_token_id (if contextTokens.isSome then cfg.context_len else 0)

  return x

/-- Generate multiple blocks in parallel using batched inference.
    Each block is an independent sample from the diffusion model.
    Returns tensor of shape [numBlocks, seq_len].
-/
def sampleMultiple {cfg : Config} (numBlocks : UInt64)
    (params : DiffusionParams cfg)
    (rotaryCache : RotaryCache rotaryLen cfg.headDim)
    (k : UInt64)
    (temperature : Float := 1.0)
    (maxSteps : Nat := 256)
    : IO (Tokens #[numBlocks, cfg.seq_len]) := do
  sampleTopK (batch := numBlocks) params rotaryCache k temperature maxSteps none

/-- Generate multiple blocks with shared context prefix.
    Each block starts with the same context tokens but generates different continuations.
    contextTokens: [numBlocks, context_len] - can be same or different per block
    Returns tensor of shape [numBlocks, seq_len].
-/
def sampleMultipleWithContext {cfg : Config} (numBlocks : UInt64)
    (params : DiffusionParams cfg)
    (rotaryCache : RotaryCache rotaryLen cfg.headDim)
    (contextTokens : Tokens #[numBlocks, cfg.context_len])
    (k : UInt64)
    (temperature : Float := 1.0)
    (maxSteps : Nat := 256)
    : IO (Tokens #[numBlocks, cfg.seq_len]) := do
  sampleTopK (batch := numBlocks) params rotaryCache k temperature maxSteps (some contextTokens)

/-- Generate multiple blocks with the same prompt replicated across all blocks.
    prompt: [context_len] - single prompt tensor replicated to all blocks
    Returns tensor of shape [numBlocks, seq_len].
-/
def sampleMultipleWithPrompt {cfg : Config} (numBlocks : UInt64)
    (params : DiffusionParams cfg)
    (rotaryCache : RotaryCache rotaryLen cfg.headDim)
    (prompt : Tokens #[cfg.context_len])
    (k : UInt64)
    (temperature : Float := 1.0)
    (maxSteps : Nat := 256)
    : IO (Tokens #[numBlocks, cfg.seq_len]) := do
  -- Replicate prompt across all blocks: [context_len] -> [numBlocks, context_len]
  let promptExpanded := Tensor.expand (Tensor.unsqueeze prompt 0) #[numBlocks, cfg.context_len]
  sampleTopK (batch := numBlocks) params rotaryCache k temperature maxSteps (some promptExpanded)

end torch.diffusion
