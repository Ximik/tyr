import Examples.AlphaGradPort.Tasks
import Tyr.Module.Linear
import Tyr.Optim
import Tyr.Mctx

/-!
  Examples/AlphaGradPort/PolicyTrain.lean

  Full network training for AlphaGrad elimination:
  - PPO actor-critic training
  - AlphaZero-style MCTS self-play training

  Both paths use real trainable neural parameters (`Linear` layers),
  autograd backprop, and `Optim.adamw` updates.
-/

namespace Examples.AlphaGradPort

open torch
open torch.mctx
open Tyr.AD.Elim

private abbrev OBS_DIM : UInt64 := 8
private abbrev HIDDEN_DIM : UInt64 := 64

private def lcgA : UInt64 := 6364136223846793005
private def lcgC : UInt64 := 1442695040888963407

private def mix (x : UInt64) : UInt64 :=
  x * lcgA + lcgC

private def uniform01 (x : UInt64) : Float :=
  let mant := (x >>> 11).toNat
  let denom : Float := Float.ofNat (Nat.pow 2 53)
  Float.ofNat mant / denom

private def maskInvalidLogits (logits : Array Float) (invalid : Array Bool) : Array Float :=
  (Array.range logits.size).map fun i =>
    if invalid.getD i false then -1.0e30 else logits.getD i (-1.0e30)

private def sampleCategorical?
    (probs : Array Float)
    (seed : UInt64) :
    Except String (ActionId0 × UInt64) := do
  if probs.isEmpty then
    throw "Cannot sample from empty probability vector."

  let seed' := mix seed
  let u := uniform01 seed'
  let mut cdf := 0.0
  let mut chosen? : Option Nat := none

  for i in [:probs.size] do
    let pRaw := probs.getD i 0.0
    let p := if pRaw < 0.0 then 0.0 else pRaw
    cdf := cdf + p
    if chosen?.isNone && u <= cdf then
      chosen? := some i

  let chosen :=
    match chosen? with
    | some a => a
    | none => Id.run do
      let mut best := 0
      let mut bestP := probs.getD 0 0.0
      for i in [1:probs.size] do
        let p := probs.getD i 0.0
        if p > bestP then
          best := i
          bestP := p
      return best

  pure (chosen, mix (seed' + 0x9e3779b97f4a7c15))

private def graphEdgeCount (g : ElimGraph) : Nat := Id.run do
  let mut total := 0
  for v in vertices g do
    total := total + (outNeighbors g v).size
  return total

private def feasibleActionCount
    (cfg : AlphaGradMctxConfig)
    (s : AlphaGradState) :
    Nat :=
  (invalidActionMask cfg s).foldl (init := 0) fun acc isInvalid =>
    if isInvalid then acc else acc + 1

private def stateFeatures
    (cfg : AlphaGradMctxConfig)
    (s : AlphaGradState) :
    Array Float :=
  let nNat := s.numVertices
  let n := Float.ofNat nNat
  let remainingNat := s.numEliminableActions - s.eliminatedEliminableCount
  let remaining := Float.ofNat remainingNat
  let feasible := Float.ofNat (feasibleActionCount cfg s)
  let edgeCount := Float.ofNat (graphEdgeCount s.graph)
  let denomEdges :=
    if nNat = 0 then 1.0 else Float.ofNat (nNat * nNat)
  let unresolvedSoft :=
    unresolvedSoftPenalty cfg.constraints (fun v => s.isVertexEliminated v)
  #[
    if nNat = 0 then 0.0 else Float.ofNat s.stepCount / n,
    if nNat = 0 then 0.0 else remaining / n,
    if nNat = 0 then 0.0 else feasible / n,
    edgeCount / denomEdges,
    unresolvedSoft,
    s.cumulativeReward,
    if s.violation?.isSome then 1.0 else 0.0,
    1.0
  ]

structure AlphaGradNet (actionDim : UInt64) where
  fc1 : Linear OBS_DIM HIDDEN_DIM
  policyHead : Linear HIDDEN_DIM actionDim
  valueHead : Linear HIDDEN_DIM 1
  deriving TensorStruct

namespace AlphaGradNet

def init (actionDim : UInt64) : IO (AlphaGradNet actionDim) := do
  let fc1 ← Linear.init OBS_DIM HIDDEN_DIM true
  let policyHead ← Linear.init HIDDEN_DIM actionDim true
  let valueHead ← Linear.init HIDDEN_DIM 1 true
  pure { fc1, policyHead, valueHead }

def forward {batch actionDim : UInt64}
    (net : AlphaGradNet actionDim)
    (x : Tensor { shape := #[batch, OBS_DIM], dtype := .Float32 }) :
    Tensor { shape := #[batch, actionDim], dtype := .Float32 } × Tensor { shape := #[batch, 1], dtype := .Float32 } :=
  let h := Tensor.relu (Linear.forward2d net.fc1 x)
  let logits := Linear.forward2d net.policyHead h
  let value := Linear.forward2d net.valueHead h
  (logits, value)

end AlphaGradNet

private structure NetEval (actionDim : UInt64) where
  fc1Weight : Array Float
  fc1Bias : Array Float
  policyWeight : Array Float
  policyBias : Array Float
  valueWeight : Array Float
  valueBias : Float
  deriving Repr

private def extractBiasOrZeros
    {n : UInt64}
    (b? : Option (Tensor { shape := #[n], dtype := .Float32 })) :
    IO (Array Float) := do
  match b? with
  | some b => b.toFloatArray
  | none => pure (Array.replicate n.toNat 0.0)

private def buildEvalNet
    {actionDim : UInt64}
    (net : AlphaGradNet actionDim) :
    IO (NetEval actionDim) := do
  let fc1Weight ← net.fc1.weight.toFloatArray
  let fc1Bias ← extractBiasOrZeros net.fc1.bias
  let policyWeight ← net.policyHead.weight.toFloatArray
  let policyBias ← extractBiasOrZeros net.policyHead.bias
  let valueWeight ← net.valueHead.weight.toFloatArray
  let valueBiasArr ← extractBiasOrZeros net.valueHead.bias
  let valueBias := valueBiasArr.getD 0 0.0
  pure {
    fc1Weight := fc1Weight
    fc1Bias := fc1Bias
    policyWeight := policyWeight
    policyBias := policyBias
    valueWeight := valueWeight
    valueBias := valueBias
  }

private def evalStatePolicyValue
    {actionDim : UInt64}
    (net : NetEval actionDim)
    (envCfg : AlphaGradMctxConfig)
    (s : AlphaGradState) :
    Array Float × Float :=
  Id.run do
    let feats := stateFeatures envCfg s
    let obsNat := OBS_DIM.toNat
    let hiddenNat := HIDDEN_DIM.toNat
    let actionNat := actionDim.toNat

    let mut hidden : Array Float := Array.replicate hiddenNat 0.0
    for j in [:hiddenNat] do
      let mut acc := net.fc1Bias.getD j 0.0
      for i in [:obsNat] do
        let w := net.fc1Weight.getD (j * obsNat + i) 0.0
        acc := acc + w * feats.getD i 0.0
      hidden := hidden.set! j (if acc > 0.0 then acc else 0.0)

    let mut rawLogits : Array Float := Array.replicate actionNat 0.0
    for a in [:actionNat] do
      let mut acc := net.policyBias.getD a 0.0
      for j in [:hiddenNat] do
        let w := net.policyWeight.getD (a * hiddenNat + j) 0.0
        acc := acc + w * hidden.getD j 0.0
      rawLogits := rawLogits.set! a acc

    let mut value := net.valueBias
    for j in [:hiddenNat] do
      let w := net.valueWeight.getD j 0.0
      value := value + w * hidden.getD j 0.0

    let invalid := invalidActionMask envCfg s
    let logits := maskInvalidLogits rawLogits invalid
    return (logits, value)

private structure PPORolloutStep where
  features : Array Float
  action : ActionId0
  reward : Float
  value : Float
  nextValue : Float
  oldLogProb : Float
  done : Bool
  deriving Repr

private def computeGAE
    (steps : Array PPORolloutStep)
    (gamma gaeLambda : Float) :
    Array Float × Array Float := Id.run do
  let n := steps.size
  let mut advantages := Array.replicate n 0.0
  let mut returns := Array.replicate n 0.0
  let mut gae := 0.0
  for k in [:n] do
    let i := n - 1 - k
    let st := steps.getD i {
      features := #[]
      action := 0
      reward := 0.0
      value := 0.0
      nextValue := 0.0
      oldLogProb := 0.0
      done := true
    }
    let nonTerminal := if st.done then 0.0 else 1.0
    let delta := st.reward + gamma * st.nextValue * nonTerminal - st.value
    gae := delta + gamma * gaeLambda * nonTerminal * gae
    advantages := advantages.set! i gae
    returns := returns.set! i (gae + st.value)
  return (advantages, returns)

private def meanArray (xs : Array Float) : Float :=
  if xs.isEmpty then
    0.0
  else
    xs.foldl (init := 0.0) (· + ·) / Float.ofNat xs.size

private def stdArray (xs : Array Float) (eps : Float := 1e-8) : Float :=
  if xs.isEmpty then
    1.0
  else
    let m := meanArray xs
    let var :=
      xs.foldl (init := 0.0) fun acc x =>
        let d := x - m
        acc + d * d
    Float.sqrt (var / Float.ofNat xs.size + eps)

private def normalizeArray (xs : Array Float) : Array Float :=
  if xs.isEmpty then
    #[]
  else
    let m := meanArray xs
    let s := stdArray xs
    xs.map (fun x => (x - m) / s)

private def rowsToTensor?
    (rows : Array (Array Float))
    (cols : UInt64) :
    Except String (Σ n : UInt64, Tensor { shape := #[n, cols], dtype := .Float32 }) := do
  let colsNat := cols.toNat
  for i in [:rows.size] do
    let row := rows.getD i #[]
    if row.size != colsNat then
      throw s!"Row {i} width mismatch: expected {colsNat}, got {row.size}."
  let mut flat : Array Float := #[]
  for row in rows do
    for x in row do
      flat := flat.push x
  let n : UInt64 := rows.size.toUInt64
  let tDyn := Tensor.fromFloatArray flat
  let t : Tensor { shape := #[n, cols], dtype := .Float32 } := Tensor.reshapeChecked tDyn #[n, cols]
  pure ⟨n, t⟩

private def rolloutPPOEpisode
    {actionDim : UInt64}
    (task : TaskSpec)
    (net : NetEval actionDim)
    (seed : UInt64) :
    Except String (Array PPORolloutStep × Float × UInt64) := do
  if actionDim.toNat != task.numVertices then
    throw s!"Network actionDim={actionDim} does not match task vertices={task.numVertices}."

  let s0 ← initAlphaGradState? task.graph task.numVertices
  let mut s := s0
  let mut key := seed
  let mut steps : Array PPORolloutStep := #[]
  let maxSteps := task.envCfg.maxEpisodeSteps.getD task.numEliminableVertices
  let mut iters := 0

  while iters < maxSteps && !(isTerminal task.envCfg s) do
    let features := stateFeatures task.envCfg s
    let (logits, value) := evalStatePolicyValue net task.envCfg s
    let probs := softmax logits
    let (action, key') ← sampleCategorical? probs key
    key := key'
    let t := transition task.envCfg s action
    let nextValue :=
      if t.done then
        0.0
      else
        (evalStatePolicyValue net task.envCfg t.nextState).2
    let oldLogProb := logSafe (probs.getD action 0.0)
    steps := steps.push {
      features := features
      action := action
      reward := t.reward
      value := value
      nextValue := nextValue
      oldLogProb := oldLogProb
      done := t.done
    }
    s := t.nextState
    iters := iters + 1

  if !(isTerminal task.envCfg s) then
    throw s!"PPO rollout did not terminate within {maxSteps} steps."

  pure (steps, s.cumulativeReward, key)

private def evalGreedyReward?
    {actionDim : UInt64}
    (task : TaskSpec)
    (net : NetEval actionDim) :
    Except String Float := do
  if actionDim.toNat != task.numVertices then
    throw s!"Network actionDim={actionDim} does not match task vertices={task.numVertices}."
  let s0 ← initAlphaGradState? task.graph task.numVertices
  let mut s := s0
  let maxSteps := task.envCfg.maxEpisodeSteps.getD task.numEliminableVertices
  let mut iters := 0
  while iters < maxSteps && !(isTerminal task.envCfg s) do
    let (logits, _value) := evalStatePolicyValue net task.envCfg s
    let action := argmax logits
    let t := transition task.envCfg s action
    s := t.nextState
    iters := iters + 1
  pure s.cumulativeReward

structure PPOTrainConfig where
  epochs : Nat := 64
  episodesPerEpoch : Nat := 16
  updateEpochs : Nat := 4
  gamma : Float := 0.99
  gaeLambda : Float := 0.95
  clipEps : Float := 0.2
  valueCoef : Float := 0.5
  entropyCoef : Float := 0.01
  learningRate : Float := 3.0e-4
  weightDecay : Float := 1.0e-4
  seed : UInt64 := 250197
  logEvery : Nat := 4
  deriving Repr

structure PPOSummary where
  taskName : String
  epochs : Nat
  episodesPerEpoch : Nat
  initialGreedyReward : Float
  lastGreedyReward : Float
  bestGreedyReward : Float
  bestGreedyEpoch : Nat
  bestEpochAvgRolloutReward : Float
  deriving Repr

private def ppoUpdate
    {actionDim : UInt64}
    (net : AlphaGradNet actionDim)
    (optState : Optim.AdamWState (AlphaGradNet actionDim))
    (steps : Array PPORolloutStep)
    (cfg : PPOTrainConfig) :
    IO (Except String
      (AlphaGradNet actionDim × Optim.AdamWState (AlphaGradNet actionDim) × Float × Float × Float)) := do
  if steps.isEmpty then
    return .error "PPO update received an empty rollout batch."

  let rows := steps.map (·.features)
  let actions : Array ActionId0 := steps.map (·.action)
  let oldLogProbs := steps.map (·.oldLogProb)
  let (advantagesRaw, returnsRaw) := computeGAE steps cfg.gamma cfg.gaeLambda
  let advantages := normalizeArray advantagesRaw

  match rowsToTensor? rows OBS_DIM with
  | .error msg =>
    return .error msg
  | .ok ⟨n, obs⟩ =>
    if n.toNat != steps.size then
      return .error s!"Obs batch size mismatch: n={n}, steps={steps.size}."
    if actions.size != steps.size || oldLogProbs.size != steps.size || advantages.size != steps.size || returnsRaw.size != steps.size then
      return .error "PPO target arrays are not aligned with rollout size."

    let actionNat := actionDim.toNat
    let mut oneHotFlat : Array Float := Array.replicate (steps.size * actionNat) 0.0
    for i in [:steps.size] do
      let a := actions.getD i 0
      if a >= actionNat then
        return .error s!"PPO action index out of range: action={a}, actionDim={actionNat}."
      oneHotFlat := oneHotFlat.set! (i * actionNat + a) 1.0
    let oneHot : Tensor { shape := #[n, actionDim], dtype := .Float32 } :=
      Tensor.reshapeChecked (Tensor.fromFloatArray oneHotFlat) #[n, actionDim]

    let oldLogT : Tensor { shape := #[n, 1], dtype := .Float32 } :=
      Tensor.reshapeChecked (Tensor.fromFloatArray oldLogProbs) #[n, 1]
    let advT : Tensor { shape := #[n, 1], dtype := .Float32 } :=
      Tensor.reshapeChecked (Tensor.fromFloatArray advantages) #[n, 1]
    let retT : Tensor { shape := #[n, 1], dtype := .Float32 } :=
      Tensor.reshapeChecked (Tensor.fromFloatArray returnsRaw) #[n, 1]

    let mut netCur := net
    let mut stateCur := optState
    let mut policyLossV := 0.0
    let mut valueLossV := 0.0
    let mut entropyV := 0.0

    for _ in [:cfg.updateEpochs] do
      let working ← TensorStruct.makeLeafParams netCur
      TensorStruct.zeroGrads working
      let (logits, values) := AlphaGradNet.forward working obs
      let logProbsAll := Tensor.logSoftmax logits (-1) rfl
      let selectedLogProbs : Tensor { shape := #[n, 1], dtype := .Float32 } :=
        Tensor.sumDim (oneHot * logProbsAll) 1 true

      let ratio := Tensor.exp (selectedLogProbs - oldLogT) rfl
      let surr1 := ratio * advT
      let clipped := Tensor.Preserving.clampFloat ratio (1.0 - cfg.clipEps) (1.0 + cfg.clipEps)
      let surr2 := clipped * advT
      let minSurr := Tensor.Preserving.mulScalar
        (Tensor.maximum (Tensor.Preserving.mulScalar surr1 (-1.0))
          (Tensor.Preserving.mulScalar surr2 (-1.0)))
        (-1.0)
      let policyLoss := Tensor.Preserving.mulScalar (Tensor.meanAll minSurr rfl) (-1.0)

      let valueErr := values - retT
      let valueLoss := Tensor.meanAll (valueErr * valueErr) rfl

      let probs := Tensor.softmax logits (-1) rfl
      let entropyTerms := probs * logProbsAll
      let entropyPer := Tensor.sumDim entropyTerms 1 false
      let entropy := Tensor.Preserving.mulScalar (Tensor.meanAll entropyPer rfl) (-1.0)

      let totalLoss :=
        policyLoss +
        Tensor.Preserving.mulScalar valueLoss cfg.valueCoef -
        Tensor.Preserving.mulScalar entropy cfg.entropyCoef

      Tensor.backwardLoss totalLoss
      let grads ← TensorStruct.grads working
      let opt := Optim.adamw (lr := cfg.learningRate) (weight_decay := cfg.weightDecay)
      let (netNext, stateNext) ← Optim.step opt working grads stateCur

      netCur := netNext
      stateCur := stateNext
      policyLossV := Tensor.item policyLoss
      valueLossV := Tensor.item valueLoss
      entropyV := Tensor.item entropy

    return .ok (netCur, stateCur, policyLossV, valueLossV, entropyV)

def trainPPO
    (task : TaskSpec)
    (cfg : PPOTrainConfig := {}) :
    IO (Except String PPOSummary) := do
  let actionDim : UInt64 := task.numVertices.toUInt64
  let net0 ← AlphaGradNet.init actionDim
  let net0 ← TensorStruct.makeLeafParams net0
  let opt := Optim.adamw (lr := cfg.learningRate) (weight_decay := cfg.weightDecay)
  let optState0 := opt.init net0
  let behavior0Eval ← buildEvalNet (TensorStruct.detach net0)

  let initialGreedyReward ←
    match evalGreedyReward? task behavior0Eval with
    | .ok r => pure r
    | .error msg =>
      return .error s!"PPO initial greedy eval failed: {msg}"

  IO.println s!"[AlphaGradPPO] task={task.name} epochs={cfg.epochs} episodes/epoch={cfg.episodesPerEpoch} vertices={task.numVertices} eliminable={task.numEliminableVertices} edges={task.edges.size}"
  IO.println s!"[AlphaGradPPO] initial_greedy_reward={initialGreedyReward}"

  let mut net := net0
  let mut optState := optState0
  let mut key := cfg.seed
  let mut bestGreedy := initialGreedyReward
  let mut bestGreedyEpoch : Nat := 0
  let mut bestAvgRollout := -1.0e30
  let mut lastGreedy := initialGreedyReward

  for epoch in [:cfg.epochs] do
    let behaviorEval ← buildEvalNet (TensorStruct.detach net)
    let mut allSteps : Array PPORolloutStep := #[]
    let mut rewardSum := 0.0
    let mut episodesDone : Nat := 0

    for _ in [:cfg.episodesPerEpoch] do
      match rolloutPPOEpisode task behaviorEval key with
      | .error msg =>
        return .error s!"PPO rollout failed at epoch {epoch + 1}: {msg}"
      | .ok (steps, totalReward, key') =>
        key := key'
        allSteps := allSteps ++ steps
        rewardSum := rewardSum + totalReward
        episodesDone := episodesDone + 1

    let avgRollout :=
      if episodesDone = 0 then 0.0 else rewardSum / Float.ofNat episodesDone
    if avgRollout > bestAvgRollout then
      bestAvgRollout := avgRollout

    match (← ppoUpdate net optState allSteps cfg) with
    | .error msg =>
      return .error s!"PPO update failed at epoch {epoch + 1}: {msg}"
    | .ok (net', optState', policyLoss, valueLoss, entropy) =>
      net := net'
      optState := optState'
      let evalNet ← buildEvalNet (TensorStruct.detach net)
      match evalGreedyReward? task evalNet with
      | .error msg =>
        return .error s!"PPO greedy eval failed at epoch {epoch + 1}: {msg}"
      | .ok greedyReward =>
        lastGreedy := greedyReward
        if greedyReward > bestGreedy then
          bestGreedy := greedyReward
          bestGreedyEpoch := epoch + 1
        let shouldLog :=
          cfg.logEvery > 0 &&
          (((epoch + 1) % cfg.logEvery = 0) || (epoch + 1 = cfg.epochs))
        if shouldLog then
          IO.println s!"[AlphaGradPPO] epoch={epoch + 1}/{cfg.epochs} avg_rollout={avgRollout} greedy={greedyReward} best_greedy={bestGreedy} policy_loss={policyLoss} value_loss={valueLoss} entropy={entropy}"

  let summary : PPOSummary := {
    taskName := task.name
    epochs := cfg.epochs
    episodesPerEpoch := cfg.episodesPerEpoch
    initialGreedyReward := initialGreedyReward
    lastGreedyReward := lastGreedy
    bestGreedyReward := bestGreedy
    bestGreedyEpoch := bestGreedyEpoch
    bestEpochAvgRolloutReward := bestAvgRollout
  }
  IO.println s!"[AlphaGradPPO] summary={reprStr summary}"
  pure (.ok summary)

private structure AZSearchParams (actionDim : UInt64) where
  envCfg : AlphaGradMctxConfig
  net : NetEval actionDim

private def recurrentFromNet
    {actionDim : UInt64} :
    RecurrentFn (AZSearchParams actionDim) AlphaGradState :=
  fun p _rng action s =>
    let t := transition p.envCfg s action
    let (priors, value) := evalStatePolicyValue p.net p.envCfg t.nextState
    let discount := if t.done then 0.0 else p.envCfg.discount
    ({ reward := t.reward, discount := discount, priorLogits := priors, value := value }, t.nextState)

private structure AZSample where
  features : Array Float
  policyTarget : Array Float
  valueTarget : Float
  deriving Repr

structure AlphaZeroTrainConfig where
  epochs : Nat := 48
  episodesPerEpoch : Nat := 8
  updateEpochs : Nat := 8
  gamma : Float := 1.0
  valueCoef : Float := 0.5
  learningRate : Float := 3.0e-4
  weightDecay : Float := 1.0e-4
  numSimulations : Nat := 48
  maxDepth : Option Nat := none
  maxNodes : Option Nat := none
  dirichletFraction : Float := 0.25
  temperature : Float := 1.0
  seed : UInt64 := 991337
  logEvery : Nat := 4
  deriving Repr

structure AlphaZeroSummary where
  taskName : String
  epochs : Nat
  episodesPerEpoch : Nat
  bestEpochAvgReward : Float
  finalGreedyReward : Float
  deriving Repr

private def discountedReturns (rewards : Array Float) (gamma : Float) : Array Float := Id.run do
  let n := rewards.size
  let mut out := Array.replicate n 0.0
  let mut acc := 0.0
  for k in [:n] do
    let i := n - 1 - k
    acc := rewards.getD i 0.0 + gamma * acc
    out := out.set! i acc
  return out

private def rolloutAlphaZeroEpisode
    {actionDim : UInt64}
    (task : TaskSpec)
    (net : NetEval actionDim)
    (cfg : AlphaZeroTrainConfig)
    (seed : UInt64) :
    Except String (Array AZSample × Float × UInt64) := do
  if actionDim.toNat != task.numVertices then
    throw s!"Network actionDim={actionDim} does not match task vertices={task.numVertices}."
  let s0 ← initAlphaGradState? task.graph task.numVertices
  let searchParams : AZSearchParams actionDim := {
    envCfg := task.envCfg
    net := net
  }

  let mut s := s0
  let mut key := seed
  let mut featRows : Array (Array Float) := #[]
  let mut policyTargets : Array (Array Float) := #[]
  let mut rewards : Array Float := #[]
  let mut tree? : Option (Tree AlphaGradState Unit) := none
  let maxSteps := task.envCfg.maxEpisodeSteps.getD task.numEliminableVertices
  let mut steps : Nat := 0

  while steps < maxSteps && !(isTerminal task.envCfg s) do
    let features := stateFeatures task.envCfg s
    let (priors, value) := evalStatePolicyValue net task.envCfg s
    let invalid := invalidActionMask task.envCfg s
    if invalid.all (fun b => b) then
      throw s!"AlphaZero root has no feasible actions at step {s.stepCount}."

    let root : RootFnOutput AlphaGradState := {
      priorLogits := priors
      value := value
      embedding := s
    }
    let out := alphazeroPolicy
      (params := searchParams)
      (rngKey := key)
      (root := root)
      (recurrentFn := recurrentFromNet)
      (numSimulations := cfg.numSimulations)
      (searchTree := tree?)
      (maxNodes := cfg.maxNodes)
      (invalidActions := some invalid)
      (maxDepth := cfg.maxDepth)
      (dirichletFraction := cfg.dirichletFraction)
      (temperature := cfg.temperature)

    let t := transition task.envCfg s out.action
    featRows := featRows.push features
    policyTargets := policyTargets.push out.actionWeights
    rewards := rewards.push t.reward
    s := t.nextState
    key := mix (key + UInt64.ofNat (steps + 1))
    tree? :=
      if t.done then
        none
      else
        some (getSubtree out.searchTree out.action)
    steps := steps + 1

  if !(isTerminal task.envCfg s) then
    throw s!"AlphaZero rollout did not terminate within {maxSteps} steps."

  let returns := discountedReturns rewards cfg.gamma
  let mut samples : Array AZSample := #[]
  for i in [:featRows.size] do
    samples := samples.push {
      features := featRows.getD i #[]
      policyTarget := policyTargets.getD i #[]
      valueTarget := returns.getD i 0.0
    }
  pure (samples, s.cumulativeReward, key)

private def alphaZeroUpdate
    {actionDim : UInt64}
    (net : AlphaGradNet actionDim)
    (optState : Optim.AdamWState (AlphaGradNet actionDim))
    (samples : Array AZSample)
    (cfg : AlphaZeroTrainConfig) :
    IO (Except String
      (AlphaGradNet actionDim × Optim.AdamWState (AlphaGradNet actionDim) × Float × Float)) := do
  if samples.isEmpty then
    return .error "AlphaZero update received an empty sample batch."

  let obsRows := samples.map (·.features)
  let policyRows := samples.map (·.policyTarget)
  let valueTargets := samples.map (·.valueTarget)

  match rowsToTensor? obsRows OBS_DIM with
  | .error msg =>
    return .error msg
  | .ok ⟨n, obs⟩ =>
    if policyRows.size != n.toNat then
      return .error s!"AlphaZero policy rows mismatch: expected {n.toNat}, got {policyRows.size}."
    let actionNat := actionDim.toNat
    let mut flatPolicy : Array Float := #[]
    for i in [:policyRows.size] do
      let row := policyRows.getD i #[]
      if row.size != actionNat then
        return .error s!"AlphaZero policy row {i} width mismatch: expected {actionNat}, got {row.size}."
      for x in row do
        flatPolicy := flatPolicy.push x
    let policyT : Tensor { shape := #[n, actionDim], dtype := .Float32 } :=
      Tensor.reshapeChecked (Tensor.fromFloatArray flatPolicy) #[n, actionDim]

    if valueTargets.size != n.toNat then
      return .error s!"AlphaZero value targets mismatch: expected {n.toNat}, got {valueTargets.size}."
    let valueT : Tensor { shape := #[n, 1], dtype := .Float32 } :=
      Tensor.reshapeChecked (Tensor.fromFloatArray valueTargets) #[n, 1]

    let mut netCur := net
    let mut stateCur := optState
    let mut policyLossV := 0.0
    let mut valueLossV := 0.0

    for _ in [:cfg.updateEpochs] do
      let working ← TensorStruct.makeLeafParams netCur
      TensorStruct.zeroGrads working
      let (logits, values) := AlphaGradNet.forward working obs
      let logProbs := Tensor.logSoftmax logits (-1) rfl
      let policyPer := Tensor.sumDim (policyT * logProbs) 1 false
      let policyLoss := Tensor.Preserving.mulScalar (Tensor.meanAll policyPer rfl) (-1.0)
      let valueErr := values - valueT
      let valueLoss := Tensor.meanAll (valueErr * valueErr) rfl
      let totalLoss := policyLoss + Tensor.Preserving.mulScalar valueLoss cfg.valueCoef

      Tensor.backwardLoss totalLoss
      let grads ← TensorStruct.grads working
      let opt := Optim.adamw (lr := cfg.learningRate) (weight_decay := cfg.weightDecay)
      let (netNext, stateNext) ← Optim.step opt working grads stateCur
      netCur := netNext
      stateCur := stateNext
      policyLossV := Tensor.item policyLoss
      valueLossV := Tensor.item valueLoss

    return .ok (netCur, stateCur, policyLossV, valueLossV)

def trainAlphaZero
    (task : TaskSpec)
    (cfg : AlphaZeroTrainConfig := {}) :
    IO (Except String AlphaZeroSummary) := do
  let actionDim : UInt64 := task.numVertices.toUInt64
  let net0 ← AlphaGradNet.init actionDim
  let net0 ← TensorStruct.makeLeafParams net0
  let opt := Optim.adamw (lr := cfg.learningRate) (weight_decay := cfg.weightDecay)
  let optState0 := opt.init net0

  IO.println s!"[AlphaGradAZ] task={task.name} epochs={cfg.epochs} episodes/epoch={cfg.episodesPerEpoch} sims={cfg.numSimulations} vertices={task.numVertices} eliminable={task.numEliminableVertices} edges={task.edges.size}"

  let mut net := net0
  let mut optState := optState0
  let mut key := cfg.seed
  let mut bestAvgReward := -1.0e30

  for epoch in [:cfg.epochs] do
    let behavior ← buildEvalNet (TensorStruct.detach net)
    let mut samples : Array AZSample := #[]
    let mut rewardSum := 0.0
    let mut episodesDone : Nat := 0

    for _ in [:cfg.episodesPerEpoch] do
      match rolloutAlphaZeroEpisode task behavior cfg key with
      | .error msg =>
        return .error s!"AlphaZero rollout failed at epoch {epoch + 1}: {msg}"
      | .ok (epSamples, totalReward, key') =>
        key := key'
        samples := samples ++ epSamples
        rewardSum := rewardSum + totalReward
        episodesDone := episodesDone + 1

    let avgReward :=
      if episodesDone = 0 then 0.0 else rewardSum / Float.ofNat episodesDone
    if avgReward > bestAvgReward then
      bestAvgReward := avgReward

    match (← alphaZeroUpdate net optState samples cfg) with
    | .error msg =>
      return .error s!"AlphaZero update failed at epoch {epoch + 1}: {msg}"
    | .ok (net', optState', policyLoss, valueLoss) =>
      net := net'
      optState := optState'
      let shouldLog :=
        cfg.logEvery > 0 &&
          (((epoch + 1) % cfg.logEvery = 0) || (epoch + 1 = cfg.epochs))
      if shouldLog then
        IO.println s!"[AlphaGradAZ] epoch={epoch + 1}/{cfg.epochs} avg_reward={avgReward} best_avg={bestAvgReward} policy_loss={policyLoss} value_loss={valueLoss}"

  let evalNet ← buildEvalNet (TensorStruct.detach net)
  let finalGreedyReward ←
    match evalGreedyReward? task evalNet with
    | .ok r => pure r
    | .error msg =>
      return .error s!"AlphaZero final greedy eval failed: {msg}"

  let summary : AlphaZeroSummary := {
    taskName := task.name
    epochs := cfg.epochs
    episodesPerEpoch := cfg.episodesPerEpoch
    bestEpochAvgReward := bestAvgReward
    finalGreedyReward := finalGreedyReward
  }
  IO.println s!"[AlphaGradAZ] summary={reprStr summary}"
  pure (.ok summary)

inductive TrainMode where
  | ppo
  | alphazero
  deriving Repr, Inhabited

instance : ToString TrainMode where
  toString
    | .ppo => "ppo"
    | .alphazero => "alphazero"

private def parseMode? (s : String) : Option TrainMode :=
  match s.trimAscii.toString.toLower with
  | "ppo" => some .ppo
  | "az" => some .alphazero
  | "alphazero" => some .alphazero
  | _ => none

private def parseNatArg? (s : String) : Option Nat :=
  s.toNat?

private def usage : String :=
  String.intercalate "\n" <| ([
    "Usage:",
    "  lake exe AlphaGradPolicyTrain",
    "  lake exe AlphaGradPolicyTrain <mode>",
    "  lake exe AlphaGradPolicyTrain <mode> <task-name>",
    "  lake exe AlphaGradPolicyTrain <mode> <task-name> <epochs>",
    "  lake exe AlphaGradPolicyTrain <mode> <task-name> <epochs> <episodes-per-epoch>",
    "Modes: ppo, alphazero (or az)",
    s!"Tasks: {taskNamesCsv}"
  ] : List String)

private def runTrain
    (mode : TrainMode)
    (taskName : TaskName)
    (epochs? : Option Nat := none)
    (episodesPerEpoch? : Option Nat := none) :
    IO UInt32 := do
  match (← materializeTask taskName) with
  | .error msg =>
    IO.eprintln s!"[AlphaGradTrain] task materialization failed: {msg}"
    pure 1
  | .ok task =>
    match mode with
    | .ppo =>
      let cfg : PPOTrainConfig := {
        epochs := epochs?.getD 64
        episodesPerEpoch := episodesPerEpoch?.getD 16
      }
      match (← trainPPO task cfg) with
      | .error msg =>
        IO.eprintln s!"[AlphaGradTrain][ppo] failed: {msg}"
        pure 1
      | .ok _ =>
        pure 0
    | .alphazero =>
      let cfg : AlphaZeroTrainConfig := {
        epochs := epochs?.getD 48
        episodesPerEpoch := episodesPerEpoch?.getD 8
        numSimulations := task.mctsCfg.numSimulations
      }
      match (← trainAlphaZero task cfg) with
      | .error msg =>
        IO.eprintln s!"[AlphaGradTrain][alphazero] failed: {msg}"
        pure 1
      | .ok _ =>
        pure 0

def policyTrainMain (args : List String) : IO UInt32 := do
  match args with
  | [] =>
    runTrain .ppo .roeFlux1d
  | [modeArg] =>
    match parseMode? modeArg with
    | none =>
      IO.eprintln s!"Unknown mode: {modeArg}"
      IO.eprintln usage
      pure 1
    | some mode =>
      runTrain mode .roeFlux1d
  | [modeArg, taskArg] =>
    match parseMode? modeArg, parseTaskName? taskArg with
    | some mode, some taskName =>
      runTrain mode taskName
    | _, _ =>
      IO.eprintln s!"Invalid arguments: {modeArg} {taskArg}"
      IO.eprintln usage
      pure 1
  | [modeArg, taskArg, epochsArg] =>
    match parseMode? modeArg, parseTaskName? taskArg, parseNatArg? epochsArg with
    | some mode, some taskName, some epochs =>
      runTrain mode taskName (some epochs)
    | _, _, _ =>
      IO.eprintln s!"Invalid arguments: {modeArg} {taskArg} {epochsArg}"
      IO.eprintln usage
      pure 1
  | [modeArg, taskArg, epochsArg, epPerEpochArg] =>
    match parseMode? modeArg, parseTaskName? taskArg, parseNatArg? epochsArg, parseNatArg? epPerEpochArg with
    | some mode, some taskName, some epochs, some episodesPerEpoch =>
      runTrain mode taskName (some epochs) (some episodesPerEpoch)
    | _, _, _, _ =>
      IO.eprintln s!"Invalid arguments: {modeArg} {taskArg} {epochsArg} {epPerEpochArg}"
      IO.eprintln usage
      pure 1
  | _ =>
    IO.eprintln usage
    pure 1

end Examples.AlphaGradPort
