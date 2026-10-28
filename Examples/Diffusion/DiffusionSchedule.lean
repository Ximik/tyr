/-
  Masked Diffusion Schedule for discrete diffusion training

  Implements a linear mask schedule where:
  - mask_prob = linspace(1/T, 1.0, T)
  - At timestep t: probability of masking = mask_probs[t]
  - Context tokens (first context_len) are never masked
-/
import Examples.Diffusion.Parameters

namespace torch.diffusion

open torch

/-- Masked diffusion schedule for training -/
structure MaskedDiffusionSchedule (num_timesteps : Nat) where
  mask_token_id : UInt64
  context_len : Nat
  /-- mask_probs[t] = probability of masking at timestep t -/
  mask_probs : Activation #[num_timesteps]

/-- Initialize a linear mask schedule -/
def MaskedDiffusionSchedule.init (num_timesteps : Nat) (mask_token_id : UInt64) (context_len : Nat)
    : MaskedDiffusionSchedule num_timesteps :=
  let start := 1.0 / num_timesteps.toFloat
  let stop := 1.0
  {
    mask_token_id := mask_token_id
    context_len := context_len
    mask_probs := Tensor.linspace start stop num_timesteps
  }

/-- Add masks to clean tokens for training.
    Given clean tokens x_0 and timestep t (per sample),
    returns masked tokens x_t where random positions are replaced with mask_token_id.

    NOTE: Context tokens (first context_len positions) are never masked.
    The mask probability is timestep-dependent: higher timesteps = more masking.
-/
def MaskedDiffusionSchedule.addMasks {num_timesteps batch seq : Nat}
    (schedule : MaskedDiffusionSchedule num_timesteps)
    (x_0 : Tokens #[batch, seq])     -- Clean tokens
    (t : Tokens #[batch])            -- Timestep per sample (int64)
    : IO (Tokens #[batch, seq]) := do
  -- Get mask probability for each sample based on timestep
  -- mask_probs[t] gives the probability of masking at timestep t
  -- Higher t = more masking (linear schedule from 1/T to 1.0)
  let maskProbPerSample : Activation #[batch] := Tensor.indexSelect schedule.mask_probs t  -- [batch]

  -- Expand to [batch, seq] for comparison
  let maskProbExpanded := Tensor.unsqueeze maskProbPerSample 1  -- [batch, 1]
  let maskProbBroadcast := Tensor.expand maskProbExpanded #[batch, seq]  -- [batch, seq]

  -- Generate random values in [0, 1)
  let randTensor ← Tensor.uniformLike (Tensor.zeros #[batch, seq])

  -- Create mask: true where we should mask (Tensor.rand < mask_prob)
  let shouldMask := Tensor.lt randTensor maskProbBroadcast  -- [batch, seq]

  -- Protect context tokens: create position mask
  let positionValues := Tensor.linspace 0.0 (seq.toFloat - 1.0) seq
  let contextMask := Tensor.ltScalar positionValues schedule.context_len.toFloat  -- [seq], true for context

  -- Expand context mask to [batch, seq]
  let contextMask_unsq := Tensor.unsqueeze contextMask 0  -- [1, seq]
  let contextMask_exp := Tensor.expand contextMask_unsq #[batch, seq]  -- [batch, seq]

  -- Final mask: should mask AND not in context
  let notContext := Tensor.logicalNot contextMask_exp
  let finalMask := Tensor.logicalAnd shouldMask notContext

  -- Apply mask: where finalMask is true, use mask_token_id, else original
  let maskTokens := Tensor.fullInt #[batch, seq] schedule.mask_token_id.toInt64
  let x_t := Tensor.whereSelect finalMask maskTokens x_0

  return x_t

/-- Get a mask indicating which positions are masked in x_t -/
def getMaskedPositions {batch seq : Nat}
    (x_t : Tokens #[batch, seq])
    (mask_token_id : UInt64)
    : Mask #[batch, seq] :=
  Tensor.eqScalar x_t mask_token_id.toInt64

/-- Compute masked cross-entropy loss.
    Only computes loss on positions that were masked (x_t == mask_token_id).
    Returns the mean loss over masked positions.
-/
def maskedCrossEntropyLoss {batch seq vocab : Nat}
    (logits : Activation #[batch, seq, vocab])   -- Model output logits
    (targets : Tokens #[batch, seq])          -- Original clean tokens (x_0)
    (x_t : Tokens #[batch, seq])              -- Masked input tokens
    (mask_token_id : UInt64)
    : Activation #[] :=
  -- Flatten for cross-entropy
  let logitsFlat := Tensor.reshape logits #[batch * seq, vocab]
  let targetsFlat := Tensor.reshape targets #[batch * seq]

  -- Compute per-token loss (reduction=none)
  let lossPerToken := Tensor.crossEntropyNone logitsFlat targetsFlat

  -- Get mask of which positions were masked
  let mask := Tensor.eqScalar x_t mask_token_id.toInt64  -- [batch, seq]
  let maskFlat := Tensor.reshape mask #[batch * seq]
  let maskFloat := maskFlat.toFloat32  -- Convert bool to float for multiplication

  -- Apply mask and compute mean over masked positions
  let maskedLoss := lossPerToken * maskFloat
  let totalLoss := Tensor.sumAll maskedLoss
  let numMasked := Tensor.sumAll maskFloat

  -- Return mean (total / count)
  -- Add small epsilon to avoid division by zero
  let numMaskedSafe := numMasked + (1e-8 : Float)
  totalLoss / numMaskedSafe

end torch.diffusion
