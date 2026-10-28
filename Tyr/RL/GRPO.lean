/-
  Tyr/RL/GRPO.lean

  Group Relative Policy Optimization (GRPO) for RL fine-tuning.

  Based on nanochat's chat_rl.py - a simplified GRPO that is
  "a lot simpler and more similar to just REINFORCE":

  Key simplifications from PPO/GRPO:
  - No trust region (no KL penalty to reference model)
  - No importance weights (on-policy only)
  - No PPO clipping
  - Simple advantage: A = r - mean(r), not z-score normalization
  - Token-level uniform advantage application

  Algorithm:
  1. Generate num_samples completions per prompt
  2. Compute binary reward (correct/incorrect)
  3. Compute advantages: A = r - mean(r) per batch
  4. Compute policy gradient: sum(log_p * A) / num_valid_tokens
  5. Loss = -pg_objective (negate to minimize)

  Current limitations:
  - `temperature`/`topK`/`topP` are plumbed through `GRPOConfig` and
    `GenerationConfig` but never applied: token sampling is delegated
    entirely to the `generateOneFn` callback, which receives no sampling
    parameters.
  - `grpoStepWithModelUpdate` requires batch size `b = 1` and throws
    otherwise; gradients are accumulated sample-by-sample under that
    constraint.
-/
import Tyr.TensorStruct
import Tyr.Optim
import Tyr.Distributed
import Tyr.Train.RunLedger

namespace torch.RL.GRPO

open torch

variable {device : Device}
open torch.dist
open torch.Train.RunLedger

/-! ## Configuration -/

/-- Configuration for GRPO training.
    Hyperparameters from nanochat's chat_rl.py -/
structure GRPOConfig where
  /-- Number of samples to generate per prompt -/
  numSamples : Nat := 16
  /-- Maximum new tokens to generate -/
  maxNewTokens : Nat := 256
  /-- Temperature for sampling (1.0 = no scaling) -/
  temperature : Float := 1.0
  /-- Top-k sampling (0 = disabled) -/
  topK : Nat := 50
  /-- Examples per training step (across all ranks) -/
  examplesPerStep : Nat := 16
  /-- Max batch size per forward pass -/
  deviceBatchSize : Nat := 8
  /-- Learning rates (scaled by batch size in practice) -/
  embeddingLr : Float := 0.2
  unembeddingLr : Float := 0.004
  matrixLr : Float := 0.02
  /-- Initial LR fraction (5% of base) -/
  initLrFrac : Float := 0.05
  /-- Weight decay (disabled in nanochat) -/
  weightDecay : Float := 0.0
  /-- Evaluation frequency (steps) -/
  evalEvery : Nat := 100
  /-- Logging frequency (steps) -/
  logEvery : Nat := 10
  /-- Padding token ID -/
  padToken : Nat := 0
  /-- EOS token ID -/
  eosToken : Nat := 2
  deriving Repr, Inhabited

/-! ## Learning Rate Schedule -/

/-- Linear decay LR schedule (from nanochat).
    LR decays linearly from initLrFrac to 0 over training. -/
def getLrMultiplier (step totalSteps : Nat) (initLrFrac : Float := 0.05) : Float :=
  let progress := step.toFloat / totalSteps.toFloat
  initLrFrac * (1.0 - progress)

/-! ## Advantage Computation -/

/-- Compute advantages by simple mean subtraction (from nanochat).
    A = r - mean(r), NOT z-score normalization (r - mean) / std

    This is applied uniformly to all tokens in a sequence. -/
def computeAdvantages (rewards : Array Float) : Array Float :=
  if rewards.isEmpty then #[]
  else
    let sum := rewards.foldl (· + ·) 0.0
    let mu := sum / rewards.size.toFloat
    rewards.map (· - mu)

/-- Compute advantages from a tensor of rewards -/
def computeAdvantagesTensor (rewards : Tensor { shape := #[b], dtype := .Float32, device := device }) : Tensor { shape := #[b], dtype := .Float32, device := device } :=
  let mu := Tensor.meanAll rewards
  let muExpanded := Tensor.expand mu #[b]
  rewards - muExpanded

/-! ## Reward Functions -/

/-- Result of reward computation -/
structure RewardResult where
  /-- Reward value (typically 0.0 or 1.0 for correctness) -/
  reward : Float
  /-- Whether the response was valid -/
  isValid : Bool := true
  deriving Repr

/-- Binary correctness reward for math problems (GSM8K style).
    Returns 1.0 if extracted answer matches target, 0.0 otherwise.

    GSM8K format: Response ends with "#### <answer>" -/
def mathReward (expectedAnswer : String) (response : String) : RewardResult :=
  -- Look for #### marker and extract answer
  let parts := response.splitOn "####"
  if parts.length < 2 then
    { reward := 0.0, isValid := true }
  else
    let extracted := parts[1]!.trimAscii.toString
    -- Simple string comparison (could be more sophisticated)
    let isCorrect := extracted == expectedAnswer.trimAscii.toString
    { reward := if isCorrect then 1.0 else 0.0, isValid := true }

/-- Exact match reward -/
def exactMatchReward (expected : String) (response : String) : RewardResult :=
  let isCorrect := response.trimAscii.toString == expected.trimAscii.toString
  { reward := if isCorrect then 1.0 else 0.0, isValid := true }

/-! ## Rollout Data Structures -/

/-- A single generated sample with its data -/
structure Sample where
  /-- Generated response tokens -/
  responseTokens : Array UInt64
  /-- Response text (decoded) -/
  responseText : String
  /-- Reward received -/
  reward : Float
  deriving Repr, Inhabited

/-- Batch of samples for one prompt -/
structure RolloutBatch where
  /-- Prompt tokens -/
  promptTokens : Array UInt64
  /-- Prompt text -/
  promptText : String
  /-- Generated samples -/
  samples : Array Sample
  /-- Expected answer (for reward computation) -/
  expectedAnswer : String
  deriving Repr

/-! ## Policy Gradient Computation -/

/-- Compute policy gradient loss following nanochat exactly.

    From chat_rl.py lines 272-282:
    ```python
    # logp = -model(inputs, targets, loss_reduction='none')  # (B, T)
    # pg_obj = (logp * advantages.unsqueeze(-1)).sum()
    # num_valid = (targets >= 0).sum().clamp(min=1)
    # pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
    # loss = -pg_obj
    ```

    Parameters:
    - logProbs: Per-token log probabilities, shape [B, T]
      (obtained by negating cross-entropy with reduction='none')
    - advantages: Per-sample advantages, shape [B]
    - mask: Valid token mask, shape [B, T]
      (1 for valid tokens, 0 for padding/ignored)
    - numPasses: Number of forward passes per example
    - examplesPerRank: Examples processed per rank per step

    Returns the loss (negated pg_objective). -/
def computePGLoss
    (logProbs : Tensor { shape := #[b, t], dtype := .Float32, device := device })
    (advantages : Tensor { shape := #[b], dtype := .Float32, device := device })
    (mask : Tensor { shape := #[b, t], dtype := .Float32, device := device })
    (numPasses : Nat := 1)
    (examplesPerRank : Nat := 1)
    : Tensor { shape := #[], dtype := .Float32, device := device } :=

  -- Expand advantages to [B, T] for broadcasting
  let advExpanded := Tensor.unsqueeze advantages 1  -- [B, 1]
  let advExpanded := Tensor.expand advExpanded #[b, t]  -- [B, T]

  -- Compute per-token policy gradient: logp * A
  let perTokenPG := logProbs * advExpanded  -- [B, T]

  -- Apply mask (invalid tokens contribute 0)
  let maskedPG := perTokenPG * mask  -- [B, T]

  -- Sum to get policy gradient objective
  let pgObj := Tensor.sumAll maskedPG

  -- Normalize by valid tokens, passes, and examples
  -- Add small epsilon to avoid division by zero (equivalent to clamp(min=1))
  let numValid := Tensor.sumAll mask
  let numValidSafe := Tensor.addScalar numValid 1e-8
  let pgObjNorm := Tensor.div pgObj numValidSafe
  let pgObjNorm := Tensor.divScalar pgObjNorm (numPasses.toFloat * examplesPerRank.toFloat)

  -- Return negative (we minimize loss, not maximize objective)
  Tensor.mulScalar pgObjNorm (-1.0)

/-- Get log probabilities from model cross-entropy output.
    In nanochat: logp = -model(inputs, targets, loss_reduction='none')

    The model returns NLL = -log(p), so we negate to get log(p). -/
def negateToLogProb {σ : StaticSpec} [DType.RealFloating σ.dtype] (nllPerToken : Tensor σ) : Tensor σ :=
  Tensor.mulScalar nllPerToken (-1.0)

/-! ## Training State -/

/-- Training state for GRPO -/
structure GRPOState where
  /-- Current training step -/
  step : Nat := 0
  /-- Total steps in training -/
  totalSteps : Nat := 1000
  /-- Running mean of rewards (exponential moving average) -/
  runningMeanReward : Float := 0.0
  /-- Best pass@1 seen -/
  bestPass1 : Float := 0.0
  /-- Total examples processed -/
  examplesProcessed : Nat := 0
  deriving Repr, Inhabited

/-- Update learning rate based on linear decay schedule -/
def GRPOState.getLr (state : GRPOState) (baseLr : Float) (initLrFrac : Float := 0.05) : Float :=
  let mult := getLrMultiplier state.step state.totalSteps initLrFrac
  baseLr * mult

/-! ## Training Step -/

/-- Result of a training step -/
structure StepResult where
  /-- Mean reward across all samples -/
  meanReward : Float
  /-- Policy gradient loss value -/
  pgLoss : Float
  /-- Number of valid samples -/
  numValidSamples : Nat
  /-- Mean generation length -/
  meanGenLen : Float
  deriving Repr

/-! ## Sequence Padding Utilities -/

/-- Pad a single sequence to a target length -/
def padSequence (seq : Array UInt64) (targetLen : Nat) (padValue : UInt64) : Array UInt64 :=
  if seq.size >= targetLen then seq
  else seq ++ (List.replicate (targetLen - seq.size) padValue).toArray

/-- Create input and target sequences from a full sequence (shift by 1) -/
def createInputTarget (fullSeq : Array UInt64) (padToken : UInt64 := 0)
    : Array UInt64 × Array UInt64 :=
  -- Keep a valid one-step pair for degenerate sequences.
  if fullSeq.size <= 1 then
    (#[padToken], #[padToken])
  else
    -- Input: all tokens except last (seq[:-1])
    let input := fullSeq.extract 0 (fullSeq.size - 1)
    -- Target: all tokens except first (seq[1:])
    let target := fullSeq.extract 1 fullSeq.size
    (input, target)

/-- Create mask for valid tokens (1.0 for valid, 0.0 for padding) -/
def createValidMask (seqLen : Nat) (paddedLen : Nat) : Array Float :=
  (List.replicate seqLen 1.0).toArray ++ (List.replicate (paddedLen - seqLen) 0.0).toArray

/-- Batch rollout data for training -/
structure BatchedRollout where
  /-- All sequences (prompt + response), padded to same length [B, T] -/
  sequences : Array (Array UInt64)
  /-- Rewards per sequence [B] -/
  rewards : Array Float
  /-- Valid token masks [B, T] -/
  masks : Array (Array Float)
  /-- Batch size -/
  batchSize : Nat
  /-- Sequence length (after padding) -/
  seqLen : Nat
  deriving Repr

/-- Prepare batched data from rollouts -/
def prepareBatchedRollouts (rollouts : Array RolloutBatch) (padToken : UInt64 := 0)
    : BatchedRollout := Id.run do
  -- Collect all sequences and rewards
  let mut allSeqs : Array (Array UInt64) := #[]
  let mut allRewards : Array Float := #[]

  for batch in rollouts do
    for sample in batch.samples do
      -- Combine prompt and response
      let fullSeq := batch.promptTokens ++ sample.responseTokens
      allSeqs := allSeqs.push fullSeq
      allRewards := allRewards.push sample.reward

  if allSeqs.isEmpty then
    return { sequences := #[], rewards := #[], masks := #[], batchSize := 0, seqLen := 0 }

  -- Find max length
  let maxLen := allSeqs.foldl (fun acc seq => max acc seq.size) 0

  -- Pad all sequences and create masks
  let mut paddedSeqs : Array (Array UInt64) := #[]
  let mut masks : Array (Array Float) := #[]

  for seq in allSeqs do
    paddedSeqs := paddedSeqs.push (padSequence seq maxLen padToken)
    masks := masks.push (createValidMask seq.size maxLen)

  {
    sequences := paddedSeqs
    rewards := allRewards
    masks := masks
    batchSize := allSeqs.size
    seqLen := maxLen
  }

/-- Perform one GRPO training step (simple version without model).

    This variant computes rollout statistics and a proxy PG score.
    Use grpoStepWithModel/grpoStepWithModelUpdate for gradient updates. -/
def grpoStep
    (rollouts : Array RolloutBatch)
    (_config : GRPOConfig)
    (state : GRPOState)
    : IO (StepResult × GRPOState) := do
  -- Collect all rewards
  let mut allRewards : Array Float := #[]
  let mut totalGenLen := 0

  for batch in rollouts do
    for sample in batch.samples do
      allRewards := allRewards.push sample.reward
      totalGenLen := totalGenLen + sample.responseTokens.size

  -- Compute advantages (nanochat: A = r - mean(r))
  let advantages := computeAdvantages allRewards

  -- Compute mean reward
  let meanReward := if allRewards.isEmpty then 0.0
    else allRewards.foldl (· + ·) 0.0 / allRewards.size.toFloat

  -- Mean generation length
  let meanGenLen := if allRewards.isEmpty then 0.0
    else totalGenLen.toFloat / allRewards.size.toFloat

  -- Proxy PG objective for logging in non-gradient mode.
  let mut pgObj : Float := 0.0
  for i in [:allRewards.size] do
    pgObj := pgObj + allRewards[i]! * advantages[i]!
  let pgLoss := if allRewards.isEmpty then 0.0 else -(pgObj / allRewards.size.toFloat)

  -- Update state
  let newState := { state with
    step := state.step + 1
    examplesProcessed := state.examplesProcessed + rollouts.size
    runningMeanReward := 0.9 * state.runningMeanReward + 0.1 * meanReward
  }

  return ({
    meanReward
    pgLoss
    numValidSamples := allRewards.size
    meanGenLen
  }, newState)

/-- Perform one GRPO training step with model forward/backward.

    Following nanochat's training loop exactly:
    1. Pad all sequences to same length
    2. Create input/target tensors (shifted by 1)
    3. Forward pass: get per-token NLL
    4. Convert to log prob: logp = -nll
    5. Compute PG loss: -sum(logp * A) / num_valid
    6. Backward pass

    Parameters:
    - rollouts: Generated samples with rewards
    - forwardFn: Model forward pass, returns per-token NLL [B, T]
    - config: GRPO configuration
    - state: Current training state
    - padToken: Padding token ID (default 0) -/
def grpoStepWithModel (b t : Nat)
    (rollouts : Array RolloutBatch)
    (forwardFn : Tensor { shape := #[b, t], dtype := .Int64, device := device } → Tensor { shape := #[b, t], dtype := .Int64, device := device } → IO (Tensor { shape := #[b, t], dtype := .Float32, device := device }))  -- (input, target) → nll_per_token
    (config : GRPOConfig)
    (state : GRPOState)
    (padToken : UInt64 := 0)
    : IO (StepResult × GRPOState) := do

  -- Prepare batched data
  let batched := prepareBatchedRollouts rollouts padToken

  if batched.batchSize == 0 then
    return ({
      meanReward := 0.0
      pgLoss := 0.0
      numValidSamples := 0
      meanGenLen := 0.0
    }, state)

  -- Compute advantages: A = r - mean(r)
  let advantages := computeAdvantages batched.rewards

  -- Create input/target tensors (shift sequences by 1)
  let mut inputSeqs : Array (Array UInt64) := #[]
  let mut targetSeqs : Array (Array UInt64) := #[]
  let mut shiftedMasks : Array (Array Float) := #[]

  for i in [:batched.batchSize] do
    let seq := batched.sequences[i]!
    let mask := batched.masks[i]!
    -- Input: seq[:-1], Target: seq[1:]
    let inputSeq := seq.extract 0 (seq.size - 1)
    let targetSeq := seq.extract 1 seq.size
    -- Shift mask accordingly (drop first element)
    let shiftedMask := mask.extract 1 mask.size
    inputSeqs := inputSeqs.push inputSeq
    targetSeqs := targetSeqs.push targetSeq
    shiftedMasks := shiftedMasks.push shiftedMask

  -- Convert to host arrays. Token IDs are staged through Float because the
  -- typed surface has no Int64 host-array factory; values below 2^24
  -- round-trip exactly, which covers every vocab in use.
  let inputFlat : Array Float := inputSeqs.foldl (fun acc seq => acc ++ seq.map (·.toFloat)) #[]
  let targetFlat : Array Float := targetSeqs.foldl (fun acc seq => acc ++ seq.map (·.toFloat)) #[]
  let maskFlat : Array Float := shiftedMasks.foldl (fun acc m => acc ++ m) #[]

  -- Create tensors, validating the runtime shape against the claimed [B, T]
  let inputTensor : Tensor { shape := #[b, t], dtype := .Int64, device := device } :=
    (Tensor.reshapeChecked (Tensor.toInt64 (Tensor.fromFloatArray inputFlat)) #[b, t]).to device
  let targetTensor : Tensor { shape := #[b, t], dtype := .Int64, device := device } :=
    (Tensor.reshapeChecked (Tensor.toInt64 (Tensor.fromFloatArray targetFlat)) #[b, t]).to device

  let maskTensor : Tensor { shape := #[b, t], dtype := .Float32, device := device } :=
    (Tensor.reshapeChecked (Tensor.fromFloatArray maskFlat) #[b, t]).to device

  let advTensor : Tensor { shape := #[b], dtype := .Float32, device := device } :=
    (Tensor.reshapeChecked (Tensor.fromFloatArray advantages) #[b]).to device

  -- Forward pass: get per-token NLL
  let nllPerToken ← forwardFn inputTensor targetTensor  -- [B, T-1]

  -- Convert NLL to log prob: logp = -nll
  let logProbs := negateToLogProb nllPerToken

  -- Compute policy gradient loss
  let pgLoss := computePGLoss logProbs advTensor maskTensor config.examplesPerStep 1

  -- Backward pass
  Tensor.backwardLoss pgLoss

  -- Compute statistics
  let meanReward := batched.rewards.foldl (· + ·) 0.0 / batched.batchSize.toFloat
  let totalGenLen := batched.sequences.foldl (fun acc seq => acc + seq.size) 0
  let meanGenLen := totalGenLen.toFloat / batched.batchSize.toFloat

  -- Get loss value for logging
  let lossValue := Tensor.item pgLoss

  -- Update state
  let newState := { state with
    step := state.step + 1
    examplesProcessed := state.examplesProcessed + rollouts.size
    runningMeanReward := 0.9 * state.runningMeanReward + 0.1 * meanReward
  }

  return ({
    meanReward
    pgLoss := lossValue
    numValidSamples := batched.batchSize
    meanGenLen
  }, newState)

/-! ## Evaluation -/

/-- Pass@k evaluation result -/
structure PassKResult where
  /-- Pass@1 accuracy -/
  pass1 : Float
  /-- Pass@k accuracy (any of k samples correct) -/
  passK : Float
  /-- k value used -/
  k : Nat
  deriving Repr

/-- Compute pass@k metric.
    pass@k = fraction of problems where at least one of k samples is correct -/
def computePassK (rollouts : Array RolloutBatch) (k : Nat) : PassKResult := Id.run do
  let mut numCorrectAtLeast1 := 0
  let mut numCorrectFirst := 0

  for batch in rollouts do
    -- Check if first sample is correct (pass@1)
    if batch.samples.size > 0 && batch.samples[0]!.reward > 0.5 then
      numCorrectFirst := numCorrectFirst + 1

    -- Check if any of first k samples is correct (pass@k)
    let kSamples := batch.samples.extract 0 (min k batch.samples.size)
    let anyCorrect := kSamples.any (·.reward > 0.5)
    if anyCorrect then
      numCorrectAtLeast1 := numCorrectAtLeast1 + 1

  let numProblems := rollouts.size
  let pass1 := if numProblems > 0 then
    numCorrectFirst.toFloat / numProblems.toFloat
  else 0.0

  let passK := if numProblems > 0 then
    numCorrectAtLeast1.toFloat / numProblems.toFloat
  else 0.0

  { pass1, passK, k }

/-! ## Logging -/

/-- Render a human-readable GRPO training update. -/
def formatTrainUpdate (result : StepResult) (state : GRPOState) : String :=
  s!"Step {state.step}/{state.totalSteps}: " ++
  s!"reward={result.meanReward} " ++
  s!"loss={result.pgLoss} " ++
  s!"valid={result.numValidSamples} " ++
  s!"gen_len={result.meanGenLen} " ++
  s!"running_reward={state.runningMeanReward}"

/-- Render a human-readable GRPO evaluation update. -/
def formatEvalUpdate (passK : PassKResult) (step : Nat) : String :=
  s!"Eval @ step {step}: " ++
  s!"pass@1={passK.pass1} " ++
  s!"pass@{passK.k}={passK.passK}"

/-- Hook set for GRPO training progress and evaluation. -/
structure Callbacks where
  onTrainStep : StepResult → GRPOState → IO Unit := fun _ _ => pure ()
  onEval : PassKResult → Nat → IO Unit := fun _ _ => pure ()

/-- Compose two callback sets. -/
def Callbacks.combine (lhs rhs : Callbacks) : Callbacks := {
  onTrainStep := fun result state => do
    lhs.onTrainStep result state
    rhs.onTrainStep result state
  onEval := fun passK step => do
    lhs.onEval passK step
    rhs.onEval passK step
}

/-- Emit GRPO metrics to the run ledger. -/
def artifactCallbacks
    (artifacts : RunArtifacts)
    (scopePrefix : String := "grpo") : Callbacks := {
  onTrainStep := fun result state => do
    appendMetricEvent artifacts {
      scope := s!"{scopePrefix}/train"
      step := some state.step
      metrics := [
        metricFloat "mean_reward" result.meanReward,
        metricFloat "pg_loss" result.pgLoss,
        metricNat "num_valid_samples" result.numValidSamples,
        metricFloat "mean_gen_len" result.meanGenLen,
        metricFloat "running_mean_reward" state.runningMeanReward
      ]
    }
  onEval := fun passK step => do
    appendMetricEvent artifacts {
      scope := s!"{scopePrefix}/eval"
      step := some step
      metrics := [
        metricFloat "pass_at_1" passK.pass1,
        metricFloat s!"pass_at_{passK.k}" passK.passK
      ]
    }
}

/-! ## Generation for Rollouts -/

/-- Configuration for generation during RL -/
structure GenerationConfig where
  /-- Maximum new tokens to generate -/
  maxNewTokens : Nat := 256
  /-- Temperature for sampling -/
  temperature : Float := 1.0
  /-- Top-k sampling (0 = disabled) -/
  topK : Nat := 50
  /-- Top-p (nucleus) sampling (1.0 = disabled) -/
  topP : Float := 1.0
  /-- End-of-sequence token -/
  eosToken : UInt64 := 2
  /-- Pad token -/
  padToken : UInt64 := 0
  deriving Repr, Inhabited

/-- Generate a single sample from a prompt.
    Returns generated tokens (excluding prompt). -/
def generateSample
    (promptTokens : Array UInt64)
    (generateOneFn : Array UInt64 → IO UInt64)  -- Given context, returns next token
    (config : GenerationConfig)
    : IO (Array UInt64) := do
  let mut tokens := promptTokens
  let mut newTokens : Array UInt64 := #[]

  for _ in [:config.maxNewTokens] do
    let nextToken ← generateOneFn tokens
    newTokens := newTokens.push nextToken
    tokens := tokens.push nextToken

    -- Stop on either terminal token.
    if nextToken == config.eosToken || nextToken == config.padToken then
      break

  return newTokens

/-- Generate multiple samples from a prompt (for pass@k evaluation).
    Returns array of generated token sequences. -/
def generateMultipleSamples
    (promptTokens : Array UInt64)
    (generateOneFn : Array UInt64 → IO UInt64)
    (numSamples : Nat)
    (config : GenerationConfig)
    : IO (Array (Array UInt64)) := do
  let mut samples : Array (Array UInt64) := #[]
  for _ in [:numSamples] do
    let sample ← generateSample promptTokens generateOneFn config
    samples := samples.push sample
  return samples

/-- Create a rollout batch from a prompt by generating samples and computing rewards. -/
def createRolloutBatch
    (promptTokens : Array UInt64)
    (promptText : String)
    (expectedAnswer : String)
    (generateOneFn : Array UInt64 → IO UInt64)
    (decodeFn : Array UInt64 → String)
    (rewardFn : String → String → Float)  -- expected → response → reward
    (numSamples : Nat)
    (genConfig : GenerationConfig)
    : IO RolloutBatch := do
  let mut samples : Array Sample := #[]

  for _ in [:numSamples] do
    let responseTokens ← generateSample promptTokens generateOneFn genConfig
    let responseText := decodeFn responseTokens
    let reward := rewardFn expectedAnswer responseText
    samples := samples.push { responseTokens, responseText, reward }

  return { promptTokens, promptText, samples, expectedAnswer }

/-! ## Full Training Loop -/

/-- Training loop state -/
structure TrainLoopState where
  /-- GRPO state -/
  grpoState : GRPOState
  /-- Total steps completed -/
  stepsCompleted : Nat := 0
  /-- Best pass@1 seen during training -/
  bestPass1 : Float := 0.0
  /-- Steps without improvement (for early stopping) -/
  stepsWithoutImprovement : Nat := 0
  deriving Repr, Inhabited

/-- Result of full training run -/
structure TrainResult where
  /-- Final GRPO state -/
  finalState : GRPOState
  /-- Final pass@1 on eval set -/
  finalPass1 : Float
  /-- Best pass@1 seen during training -/
  bestPass1 : Float
  /-- Total steps trained -/
  totalSteps : Nat
  deriving Repr

/-- Run a full GRPO training loop.

    This orchestrates:
    1. Sample generation from prompts
    2. Reward computation
    3. Policy gradient update

    Parameters:
    - getPromptFn: Function to get (promptTokens, promptText, expectedAnswer) for step i
    - generateOneFn: Model generation function (context → next token)
    - forwardFn: Model forward pass for NLL computation
    - decodeFn: Token decoder
    - rewardFn: Reward function (expected → response → reward)
    - evalFn: Optional evaluation function
    - config: GRPO configuration
    - numSteps: Number of training steps -/
def trainLoop (b t : Nat)
    (getPromptFn : Nat → IO (Array UInt64 × String × String))  -- step → (tokens, text, answer)
    (generateOneFn : Array UInt64 → IO UInt64)
    (forwardFn : Tensor { shape := #[b, t], dtype := .Int64, device := device } → Tensor { shape := #[b, t], dtype := .Int64, device := device } → IO (Tensor { shape := #[b, t], dtype := .Float32, device := device }))
    (decodeFn : Array UInt64 → String)
    (rewardFn : String → String → Float)
    (config : GRPOConfig)
    (numSteps : Nat)
    (callbacks : Callbacks := {})
    : IO TrainResult := do

  let genConfig : GenerationConfig := {
    maxNewTokens := config.maxNewTokens
    temperature := config.temperature
    topK := config.topK
    eosToken := config.eosToken.toUInt64
    padToken := config.padToken.toUInt64
  }

  let initialState : GRPOState := {
    totalSteps := numSteps
  }

  let mut state := initialState
  let mut bestPass1 : Float := 0.0
  let logEvery := max 1 config.logEvery

  for step in [:numSteps] do
    -- Generate rollouts for this step
    let mut rollouts : Array RolloutBatch := #[]

    for _ in [:config.examplesPerStep] do
      let (promptTokens, promptText, expectedAnswer) ← getPromptFn step
      let rollout ← createRolloutBatch
        promptTokens promptText expectedAnswer
        generateOneFn decodeFn rewardFn
        config.numSamples genConfig
      rollouts := rollouts.push rollout

    -- Training step
    let (result, newState) ← grpoStepWithModel b t rollouts forwardFn config state config.padToken.toUInt64

    state := newState

    -- Log progress
    if step % logEvery == 0 || step + 1 == numSteps then
      callbacks.onTrainStep result state

    -- Evaluation
    if step > 0 && step % config.evalEvery == 0 then
      let passK := computePassK rollouts config.numSamples
      callbacks.onEval passK step
      if passK.pass1 > bestPass1 then
        bestPass1 := passK.pass1

  return {
    finalState := state
    finalPass1 := state.runningMeanReward  -- Approximate
    bestPass1
    totalSteps := numSteps
  }

/-- Perform one GRPO training step with parameter updates.

    This variant accumulates gradients across all generated samples and applies
    a single AdamW update to the model parameters. -/
def grpoStepWithModelUpdate (b t : Nat) [TensorStruct P] [FloatingTensorStruct P]
    (params : P)
    (optState : Optim.AdamWState P)
    (rollouts : Array RolloutBatch)
    (forwardFn : P → Tensor { shape := #[b, t], dtype := .Int64, device := device } → Tensor { shape := #[b, t], dtype := .Int64, device := device } → IO (Tensor { shape := #[b, t], dtype := .Float32, device := device }))  -- (params, input, target) → nll_per_token
    (config : GRPOConfig)
    (state : GRPOState)
    (padToken : UInt64 := 0)
    : IO (P × Optim.AdamWState P × StepResult × GRPOState) := do
  if b != 1 then
    throw <| IO.userError s!"grpoStepWithModelUpdate currently requires b=1, got b={b}"

  -- Flatten samples and rewards for statistics/advantages.
  let mut rewards : Array Float := #[]
  let mut totalGenLen := 0
  for batch in rollouts do
    for sample in batch.samples do
      rewards := rewards.push sample.reward
      totalGenLen := totalGenLen + sample.responseTokens.size

  if rewards.isEmpty then
    let result : StepResult := {
      meanReward := 0.0
      pgLoss := 0.0
      numValidSamples := 0
      meanGenLen := 0.0
    }
    return (params, optState, result, state)

  let advantages := computeAdvantages rewards
  let tNat := t
  let mut sampleIdx := 0
  let mut totalLoss? : Option (Tensor { shape := #[], dtype := .Float32, device := device }) := none

  -- Accumulate gradients across all rollout samples.
  let workingParams ← TensorStruct.makeLeafParams params
  TensorStruct.zeroGrads workingParams

  for batch in rollouts do
    for sample in batch.samples do
      let fullSeqRaw := batch.promptTokens ++ sample.responseTokens
      let fullSeq :=
        if fullSeqRaw.size >= tNat + 1 then
          fullSeqRaw.extract 0 (tNat + 1)
        else
          padSequence fullSeqRaw (tNat + 1) padToken

      let validTargets :=
        if fullSeqRaw.isEmpty then
          0
        else
          min tNat (fullSeqRaw.size - 1)

      let inputSeq := fullSeq.extract 0 tNat
      let targetSeq := fullSeq.extract 1 (tNat + 1)
      let maskSeq := createValidMask validTargets tNat

      -- Token IDs are staged through Float (see grpoStepWithModel); values
      -- below 2^24 round-trip exactly, which covers every vocab in use.
      let inputTensor : Tensor { shape := #[b, t], dtype := .Int64, device := device } :=
        (Tensor.reshapeChecked (Tensor.toInt64 (Tensor.fromFloatArray (inputSeq.map (·.toFloat)))) #[b, t]).to device
      let targetTensor : Tensor { shape := #[b, t], dtype := .Int64, device := device } :=
        (Tensor.reshapeChecked (Tensor.toInt64 (Tensor.fromFloatArray (targetSeq.map (·.toFloat)))) #[b, t]).to device
      let maskTensor : Tensor { shape := #[b, t], dtype := .Float32, device := device } :=
        (Tensor.reshapeChecked (Tensor.fromFloatArray maskSeq) #[b, t]).to device

      let adv := advantages[sampleIdx]!
      let advTensor : Tensor { shape := #[b], dtype := .Float32, device := device } :=
        (Tensor.reshapeChecked (Tensor.fromFloatArray #[adv]) #[b]).to device

      let nllPerToken ← forwardFn workingParams inputTensor targetTensor
      let logProbs := negateToLogProb nllPerToken
      let pgLoss := computePGLoss logProbs advTensor maskTensor 1 1
      totalLoss? := some <| match totalLoss? with
        | some lossSoFar => lossSoFar + pgLoss
        | none => pgLoss
      sampleIdx := sampleIdx + 1

  let totalLoss ← match totalLoss? with
    | some loss => pure loss
    | none => throw <| IO.userError "grpoStepWithModelUpdate: no losses accumulated"

  let meanLossTensor := Tensor.divScalar totalLoss rewards.size.toFloat
  Tensor.backwardLoss meanLossTensor

  -- Synchronize gradients in distributed runs.
  let isDistributed ← dist.isInitialized
  let grads ← TensorStruct.grads workingParams
  let grads ← if isDistributed then dist.allReduceGrads grads .avg else pure grads
  let lr := state.getLr config.matrixLr config.initLrFrac
  let opt := Optim.adamw (lr := lr) (b1 := 0.9) (b2 := 0.999) (weight_decay := config.weightDecay)
  let (newParams, newOptState) ← Optim.step opt workingParams grads optState

  let meanReward := rewards.foldl (· + ·) 0.0 / rewards.size.toFloat
  let meanGenLen := totalGenLen.toFloat / rewards.size.toFloat
  let meanLoss := Tensor.item meanLossTensor

  let newState := { state with
    step := state.step + 1
    examplesProcessed := state.examplesProcessed + rollouts.size
    runningMeanReward := 0.9 * state.runningMeanReward + 0.1 * meanReward
  }

  let result : StepResult := {
    meanReward := meanReward
    pgLoss := meanLoss
    numValidSamples := rewards.size
    meanGenLen := meanGenLen
  }

  return (newParams, newOptState, result, newState)

/-- GRPO training loop that updates model parameters each step. -/
def trainLoopWithUpdates (b t : Nat) [TensorStruct P] [FloatingTensorStruct P]
    (getPromptFn : Nat → IO (Array UInt64 × String × String))  -- step → (tokens, text, answer)
    (generateOneFn : P → Array UInt64 → IO UInt64)
    (forwardFn : P → Tensor { shape := #[b, t], dtype := .Int64, device := device } → Tensor { shape := #[b, t], dtype := .Int64, device := device } → IO (Tensor { shape := #[b, t], dtype := .Float32, device := device }))
    (params : P)
    (optState : Optim.AdamWState P)
    (decodeFn : Array UInt64 → String)
    (rewardFn : String → String → Float)
    (config : GRPOConfig)
    (numSteps : Nat)
    (callbacks : Callbacks := {})
    : IO (P × Optim.AdamWState P × TrainResult) := do

  let genConfig : GenerationConfig := {
    maxNewTokens := config.maxNewTokens
    temperature := config.temperature
    topK := config.topK
    eosToken := config.eosToken.toUInt64
    padToken := config.padToken.toUInt64
  }

  let initialState : GRPOState := {
    totalSteps := numSteps
  }

  let mut state := initialState
  let mut bestPass1 : Float := 0.0
  let mut currentParams := params
  let mut currentOptState := optState
  let logEvery := max 1 config.logEvery

  for step in [:numSteps] do
    let mut rollouts : Array RolloutBatch := #[]

    for _ in [:config.examplesPerStep] do
      let (promptTokens, promptText, expectedAnswer) ← getPromptFn step
      let rollout ← createRolloutBatch
        promptTokens promptText expectedAnswer
        (fun context => generateOneFn currentParams context)
        decodeFn rewardFn
        config.numSamples genConfig
      rollouts := rollouts.push rollout

    let (newParams, newOptState, result, newState) ←
      grpoStepWithModelUpdate b t currentParams currentOptState rollouts forwardFn config state config.padToken.toUInt64

    currentParams := newParams
    currentOptState := newOptState
    state := newState

    if step % logEvery == 0 || step + 1 == numSteps then
      callbacks.onTrainStep result state

    if step > 0 && step % config.evalEvery == 0 then
      let passK := computePassK rollouts config.numSamples
      callbacks.onEval passK step
      if passK.pass1 > bestPass1 then
        bestPass1 := passK.pass1

  let trainResult : TrainResult := {
    finalState := state
    finalPass1 := state.runningMeanReward
    bestPass1 := bestPass1
    totalSteps := numSteps
  }

  return (currentParams, currentOptState, trainResult)

/-- Simplified training interface for when you have pre-generated prompts. -/
def trainOnPrompts (b t : Nat)
    (prompts : Array (Array UInt64 × String × String))  -- (tokens, text, answer)
    (generateOneFn : Array UInt64 → IO UInt64)
    (forwardFn : Tensor { shape := #[b, t], dtype := .Int64, device := device } → Tensor { shape := #[b, t], dtype := .Int64, device := device } → IO (Tensor { shape := #[b, t], dtype := .Float32, device := device }))
    (decodeFn : Array UInt64 → String)
    (rewardFn : String → String → Float)
    (config : GRPOConfig)
    (numEpochs : Nat := 1)
    (callbacks : Callbacks := {})
    : IO TrainResult := do

  let numSteps := numEpochs * (prompts.size / config.examplesPerStep)

  -- Create prompt getter that cycles through prompts
  let getPromptFn := fun step => do
    let idx := step % prompts.size
    return prompts[idx]!

  trainLoop b t getPromptFn generateOneFn forwardFn decodeFn rewardFn config numSteps callbacks

/-- Prompt-array convenience wrapper for the update-enabled GRPO loop. -/
def trainOnPromptsWithUpdates (b t : Nat) [TensorStruct P] [FloatingTensorStruct P]
    (prompts : Array (Array UInt64 × String × String))  -- (tokens, text, answer)
    (generateOneFn : P → Array UInt64 → IO UInt64)
    (forwardFn : P → Tensor { shape := #[b, t], dtype := .Int64, device := device } → Tensor { shape := #[b, t], dtype := .Int64, device := device } → IO (Tensor { shape := #[b, t], dtype := .Float32, device := device }))
    (params : P)
    (optState : Optim.AdamWState P)
    (decodeFn : Array UInt64 → String)
    (rewardFn : String → String → Float)
    (config : GRPOConfig)
    (numEpochs : Nat := 1)
    (callbacks : Callbacks := {})
    : IO (P × Optim.AdamWState P × TrainResult) := do

  let numSteps := numEpochs * (prompts.size / config.examplesPerStep)

  let getPromptFn := fun step => do
    let idx := step % prompts.size
    return prompts[idx]!

  trainLoopWithUpdates b t getPromptFn generateOneFn forwardFn params optState decodeFn rewardFn config numSteps callbacks

end torch.RL.GRPO
