import Tyr
import Tyr.Checkpoint
import Tyr.Optim
import Tyr.Model.BranchingFlows.MoleculeTransformer
import Tyr.Model.BranchingFlows.QM9

/-!
  Dataset-backed BranchingFlows molecule training and generation.

  This executable trains the molecule transformer on freshly sampled
  BranchingFlows bridge batches, evaluates the trained model on held-out bridge
  samples, then runs the learned model through the forward molecule generator
  and writes XYZ artifacts.
-/

namespace Examples.BranchingFlows

open torch
open torch.branching

private def defaultJsonl : String :=
  "{\"name\":\"water\",\"smiles\":\"O\",\"atoms\":[{\"label\":8,\"coord\":[0.0,0.0,0.0]},{\"label\":1,\"coord\":[0.95,0.0,0.0]},{\"label\":1,\"coord\":[-0.24,0.93,0.0]}]}\n" ++
  "{\"name\":\"ammonia\",\"smiles\":\"N\",\"atoms\":[{\"label\":7,\"coord\":[0.0,0.0,0.0]},{\"label\":1,\"coord\":[0.94,0.0,0.0]},{\"label\":1,\"coord\":[-0.31,0.89,0.0]},{\"label\":1,\"coord\":[-0.31,-0.89,0.0]}]}\n" ++
  "{\"name\":\"methane\",\"smiles\":\"C\",\"atoms\":[{\"label\":6,\"coord\":[0.0,0.0,0.0]},{\"label\":1,\"coord\":[0.63,0.63,0.63]},{\"label\":1,\"coord\":[-0.63,-0.63,0.63]},{\"label\":1,\"coord\":[-0.63,0.63,-0.63]},{\"label\":1,\"coord\":[0.63,-0.63,-0.63]}]}\n" ++
  "{\"name\":\"methanol\",\"smiles\":\"CO\",\"atoms\":[{\"label\":6,\"coord\":[0.0,0.0,0.0]},{\"label\":8,\"coord\":[1.43,0.0,0.0]},{\"label\":1,\"coord\":[-0.54,0.94,0.0]},{\"label\":1,\"coord\":[-0.54,-0.47,0.81]},{\"label\":1,\"coord\":[-0.54,-0.47,-0.81]},{\"label\":1,\"coord\":[1.76,0.9,0.0]}]}\n"

structure RunOptions where
  dataPath? : Option String := none
  requireData : Bool := false
  outputPrefix : String := "examples_branching_molecule_trained"
  checkpointDir : String := "checkpoints/branchingflows_molecule_train_generate"
  resumeCheckpoint? : Option String := none
  generateOnly : Bool := false
  saveCheckpoint : Bool := true
  saveOptimizer : Bool := true
  resumeOptimizer : Bool := true
  steps : Nat := 260
  totalSteps : Nat := 260
  warmupSteps : Nat := 0
  cooldownSteps : Nat := 0
  batchSize : Nat := 4
  vocabSize : Nat := 10
  maskToken : Nat := 0
  fixedLabels : Bool := false
  maxLen : UInt64 := 16
  fullArchitecture : Bool := false
  hiddenDim : UInt64 := 16
  heads : UInt64 := 2
  headDim : UInt64 := 8
  mlp : UInt64 := 48
  rffDim : UInt64 := 8
  layers : UInt64 := 2
  coordUpdateLayers : Nat := 1
  lr : Float := 8.0e-3
  lrEnd : Float := 0.0
  weightDecay : Float := 0.01
  gradClip : Float := 0.0
  coordWeight : Float := 10.0
  labelWeight : Float := 0.3333333333333333
  splitsWeight : Float := 1.0
  delWeight : Float := 1.0
  useBranchingTimeProb : Float := 0.5
  deletionPad : Float := 1.2
  logEvery : Nat := 50
  sampleSteps : Nat := 7
  sampleCount : Nat := 1
  generate : Bool := true
  device : Device := Device.CPU
  bf16 : Bool := false
  cudaGraph : Bool := true
  timePhases : Bool := false
  parallelSampling : Nat := 1
  splitLogitCap? : Option Float := none
  coordTargetCap? : Option Float := none
  seed : UInt64 := 20260618
  deriving Repr

private def usage : String :=
  "usage: lake exe BranchingFlowsMoleculeTrainGenerate [--data molecules.jsonl|json] " ++
  "[--profile smoke|paper-qm9-main|paper-qm9-appendix] [--out-prefix prefix] " ++
  "[--checkpoint-dir dir] [--resume-checkpoint dir] [--generate-only] [--no-checkpoint] " ++
  "[--steps n] [--total-steps n] [--batch-size n] [--max-len n] " ++
  "[--architecture compact|full] [--hidden-dim n] [--heads n] [--head-dim n] " ++
  "[--mlp n] [--rff-dim n] [--layers n] [--coord-update-layers n] [--coord-target-cap x] " ++
  "[--coord-weight x] [--label-weight x] [--splits-weight x] [--del-weight x] " ++
  "[--branching-time-prob x] [--split-logit-cap x] " ++
  "[--fixed-labels] [--device cpu|cuda] [--lr x] [--seed n] [--time-phases] [--parallel-sampling n]"

private def parseNatArg (name value : String) : IO Nat := do
  match value.toNat? with
  | some n => pure n
  | none => throw (IO.userError s!"{name} expects a natural number, got '{value}'")

private def parseUInt64Arg (name value : String) : IO UInt64 := do
  pure (← parseNatArg name value).toUInt64

private def parseSeedArg (value : String) : IO UInt64 := do
  let n ← parseNatArg "--seed" value
  pure n.toUInt64

private def parseFloatLit? (s : String) : Option Float :=
  let trimmed := s.trimAscii.toString
  if trimmed.isEmpty then
    none
  else
    let negative := trimmed.startsWith "-"
    let body := if negative then (trimmed.drop 1).toString else trimmed
    let unsigned? :=
      match body.splitOn "." with
      | [whole] =>
        whole.toNat?.map Nat.toFloat
      | [whole, frac] =>
        match whole.toNat?, frac.toNat? with
        | some w, some f =>
          let denom : Float := (Nat.pow 10 frac.length).toFloat
          some (w.toFloat + f.toFloat / denom)
        | _, _ => none
      | _ => none
    unsigned?.map fun x => if negative then -x else x

private def parseFloatArg (name value : String) : IO Float := do
  match parseFloatLit? value with
  | some x => pure x
  | none => throw (IO.userError s!"{name} expects a decimal number, got '{value}'")

private def applyProfile (name : String) (opts : RunOptions) : IO RunOptions := do
  match name with
  | "smoke" => pure opts
  | "paper-qm9-main" | "paper-qm9-unconditional" =>
      pure { opts with
        requireData := true
        steps := 800000
        totalSteps := 800000
        batchSize := 128
        maxLen := 64
        fullArchitecture := true
        hiddenDim := 384
        heads := 12
        headDim := 64
        mlp := 1536
        rffDim := 64
        layers := 12
        coordUpdateLayers := 6
        lr := 0.005
        lrEnd := 0.0
        warmupSteps := 0
        cooldownSteps := 50000
        deletionPad := 1.2
        logEvery := 100
        sampleSteps := 1000
        sampleCount := 10000
        generate := true
      }
  | "paper-qm9-appendix" =>
      pure { opts with
        requireData := true
        steps := 500000
        totalSteps := 500000
        batchSize := 128
        maxLen := 64
        fullArchitecture := true
        hiddenDim := 384
        heads := 12
        headDim := 64
        mlp := 1536
        rffDim := 64
        layers := 12
        coordUpdateLayers := 6
        lr := 0.005
        lrEnd := 0.0
        warmupSteps := 0
        cooldownSteps := 50000
        deletionPad := 1.2
        logEvery := 100
        sampleSteps := 1000
        sampleCount := 10000
        generate := true
      }
  | _ =>
      throw (IO.userError s!"unknown profile '{name}'\n{usage}")

partial def parseArgsLoop (args : List String) (opts : RunOptions) : IO RunOptions := do
  match args with
  | [] => pure opts
  | "--help" :: _ => throw (IO.userError usage)
  | "--profile" :: value :: rest =>
      parseArgsLoop rest (← applyProfile value opts)
  | "--data" :: value :: rest =>
      parseArgsLoop rest { opts with dataPath? := some value }
  | "--require-data" :: rest =>
      parseArgsLoop rest { opts with requireData := true }
  | "--out-prefix" :: value :: rest =>
      parseArgsLoop rest { opts with outputPrefix := value }
  | "--checkpoint-dir" :: value :: rest =>
      parseArgsLoop rest { opts with checkpointDir := value }
  | "--resume-checkpoint" :: value :: rest =>
      parseArgsLoop rest { opts with resumeCheckpoint? := some value }
  | "--generate-only" :: rest =>
      parseArgsLoop rest { opts with generateOnly := true }
  | "--no-checkpoint" :: rest =>
      parseArgsLoop rest { opts with saveCheckpoint := false }
  | "--no-optimizer-checkpoint" :: rest =>
      parseArgsLoop rest { opts with saveOptimizer := false }
  | "--no-resume-optimizer" :: rest =>
      parseArgsLoop rest { opts with resumeOptimizer := false }
  | "--steps" :: value :: rest =>
      parseArgsLoop rest { opts with steps := (← parseNatArg "--steps" value) }
  | "--total-steps" :: value :: rest =>
      parseArgsLoop rest { opts with totalSteps := (← parseNatArg "--total-steps" value) }
  | "--warmup-steps" :: value :: rest =>
      parseArgsLoop rest { opts with warmupSteps := (← parseNatArg "--warmup-steps" value) }
  | "--cooldown-steps" :: value :: rest =>
      parseArgsLoop rest { opts with cooldownSteps := (← parseNatArg "--cooldown-steps" value) }
  | "--batch-size" :: value :: rest =>
      parseArgsLoop rest { opts with batchSize := (← parseNatArg "--batch-size" value) }
  | "--vocab-size" :: value :: rest =>
      parseArgsLoop rest { opts with vocabSize := (← parseNatArg "--vocab-size" value) }
  | "--mask-token" :: value :: rest =>
      parseArgsLoop rest { opts with maskToken := (← parseNatArg "--mask-token" value) }
  | "--fixed-labels" :: rest =>
      parseArgsLoop rest { opts with fixedLabels := true }
  | "--max-len" :: value :: rest =>
      parseArgsLoop rest { opts with maxLen := (← parseUInt64Arg "--max-len" value) }
  | "--architecture" :: value :: rest =>
      match value with
      | "compact" => parseArgsLoop rest { opts with fullArchitecture := false }
      | "full" => parseArgsLoop rest { opts with fullArchitecture := true }
      | _ => throw (IO.userError s!"--architecture expects compact or full, got '{value}'")
  | "--full-architecture" :: rest =>
      parseArgsLoop rest { opts with fullArchitecture := true }
  | "--compact-architecture" :: rest =>
      parseArgsLoop rest { opts with fullArchitecture := false }
  | "--hidden-dim" :: value :: rest =>
      parseArgsLoop rest { opts with hiddenDim := (← parseUInt64Arg "--hidden-dim" value) }
  | "--heads" :: value :: rest =>
      parseArgsLoop rest { opts with heads := (← parseUInt64Arg "--heads" value) }
  | "--head-dim" :: value :: rest =>
      parseArgsLoop rest { opts with headDim := (← parseUInt64Arg "--head-dim" value) }
  | "--mlp" :: value :: rest =>
      parseArgsLoop rest { opts with mlp := (← parseUInt64Arg "--mlp" value) }
  | "--rff-dim" :: value :: rest =>
      parseArgsLoop rest { opts with rffDim := (← parseUInt64Arg "--rff-dim" value) }
  | "--layers" :: value :: rest =>
      parseArgsLoop rest { opts with layers := (← parseUInt64Arg "--layers" value) }
  | "--coord-update-layers" :: value :: rest =>
      parseArgsLoop rest { opts with coordUpdateLayers := (← parseNatArg "--coord-update-layers" value) }
  | "--lr" :: value :: rest =>
      parseArgsLoop rest { opts with lr := (← parseFloatArg "--lr" value) }
  | "--lr-end" :: value :: rest =>
      parseArgsLoop rest { opts with lrEnd := (← parseFloatArg "--lr-end" value) }
  | "--weight-decay" :: value :: rest =>
      parseArgsLoop rest { opts with weightDecay := (← parseFloatArg "--weight-decay" value) }
  | "--grad-clip" :: value :: rest =>
      parseArgsLoop rest { opts with gradClip := (← parseFloatArg "--grad-clip" value) }
  | "--coord-weight" :: value :: rest =>
      parseArgsLoop rest { opts with coordWeight := (← parseFloatArg "--coord-weight" value) }
  | "--label-weight" :: value :: rest =>
      parseArgsLoop rest { opts with labelWeight := (← parseFloatArg "--label-weight" value) }
  | "--splits-weight" :: value :: rest =>
      parseArgsLoop rest { opts with splitsWeight := (← parseFloatArg "--splits-weight" value) }
  | "--del-weight" :: value :: rest =>
      parseArgsLoop rest { opts with delWeight := (← parseFloatArg "--del-weight" value) }
  | "--branching-time-prob" :: value :: rest =>
      parseArgsLoop rest { opts with useBranchingTimeProb := (← parseFloatArg "--branching-time-prob" value) }
  | "--deletion-pad" :: value :: rest =>
      parseArgsLoop rest { opts with deletionPad := (← parseFloatArg "--deletion-pad" value) }
  | "--log-every" :: value :: rest =>
      parseArgsLoop rest { opts with logEvery := (← parseNatArg "--log-every" value) }
  | "--sample-steps" :: value :: rest =>
      parseArgsLoop rest { opts with sampleSteps := (← parseNatArg "--sample-steps" value) }
  | "--sample-count" :: value :: rest =>
      parseArgsLoop rest { opts with sampleCount := (← parseNatArg "--sample-count" value) }
  | "--device" :: value :: rest =>
      match value with
      | "cpu" => parseArgsLoop rest { opts with device := Device.CPU }
      | "cuda" | "cuda:0" => parseArgsLoop rest { opts with device := ← Device.cuda 0 }
      | _ => throw (IO.userError s!"--device expects cpu or cuda, got '{value}'")
  | "--dtype" :: value :: rest =>
      match value with
      | "fp32" => parseArgsLoop rest { opts with bf16 := false }
      | "bf16" => parseArgsLoop rest { opts with bf16 := true }
      | _ => throw (IO.userError s!"--dtype expects fp32 or bf16, got '{value}'")
  | "--no-generate" :: rest =>
      parseArgsLoop rest { opts with generate := false }
  | "--time-phases" :: rest =>
      parseArgsLoop rest { opts with timePhases := true }
  | "--cuda-graph" :: rest =>
      parseArgsLoop rest { opts with cudaGraph := true }
  | "--no-cuda-graph" :: rest =>
      parseArgsLoop rest { opts with cudaGraph := false }
  | "--parallel-sampling" :: value :: rest =>
      parseArgsLoop rest { opts with parallelSampling := (← parseNatArg "--parallel-sampling" value) }
  | "--split-logit-cap" :: value :: rest =>
      parseArgsLoop rest { opts with splitLogitCap? := some (← parseFloatArg "--split-logit-cap" value) }
  | "--no-split-logit-cap" :: rest =>
      parseArgsLoop rest { opts with splitLogitCap? := none }
  | "--coord-target-cap" :: value :: rest =>
      parseArgsLoop rest { opts with coordTargetCap? := some (← parseFloatArg "--coord-target-cap" value) }
  | "--seed" :: value :: rest =>
      parseArgsLoop rest { opts with seed := (← parseSeedArg value) }
  | flag :: rest =>
      if flag.startsWith "--" then
        throw (IO.userError s!"unknown option '{flag}'\n{usage}")
      else
        match opts.dataPath? with
        | none => parseArgsLoop rest { opts with dataPath? := some flag }
        | some _ => throw (IO.userError s!"unexpected positional argument '{flag}'\n{usage}")

private def parseOptions (args : List String) : IO RunOptions :=
  parseArgsLoop args {}

private def exceptToIO {α : Type} (context : String) (x : Except String α) : IO α :=
  match x with
  | .ok value => pure value
  | .error e => throw (IO.userError s!"{context}: {e}")

private def parseDatasetRaw (raw : String) : Except String (Array QM9MoleculeRecord) :=
  match parseQM9MoleculeJsonl raw with
  | .ok records =>
      if records.isEmpty then
        parseQM9MoleculeDatasetJson raw
      else
        .ok records
  | .error jsonlErr =>
      match parseQM9MoleculeDatasetJson raw with
      | .ok records => .ok records
      | .error jsonErr => .error s!"as JSONL: {jsonlErr}; as JSON: {jsonErr}"

private def loadRecords (opts : RunOptions) : IO (Array QM9MoleculeRecord) := do
  match opts.dataPath? with
  | none => exceptToIO "parse embedded molecule fixture" (parseDatasetRaw defaultJsonl)
  | some "-" => exceptToIO "parse embedded molecule fixture" (parseDatasetRaw defaultJsonl)
  | some path =>
      let raw ← IO.FS.readFile ⟨path⟩
      exceptToIO s!"parse molecule dataset {path}" (parseDatasetRaw raw)

private def shuffleStates
    (states : Array (BranchingState MoleculeAtom))
    (seed : UInt64) : Array (BranchingState MoleculeAtom) := Id.run do
  let mut shuffled := states
  let mut rng : Rng := { state := seed }
  for offset in [:states.size] do
    let i := states.size - 1 - offset
    if i > 0 then
      let (j, rng') := randNat rng (i + 1)
      rng := rng'
      let left := shuffled[i]!
      let right := shuffled[j]!
      shuffled := shuffled.set! i right
      shuffled := shuffled.set! j left
  return shuffled

private def splitTrainEval
    (states : Array (BranchingState MoleculeAtom))
    (seed : UInt64) :
    Array (BranchingState MoleculeAtom) × Array (BranchingState MoleculeAtom) :=
  if states.size <= 1 then
    (states, states)
  else
    let shuffled := shuffleStates states seed
    let evalCount := Nat.max 1 (states.size / 10)
    let split := states.size - evalCount
    (shuffled.extract 0 split, shuffled.extract split shuffled.size)

private def countSplits (events : Array (Array BranchingStepEvent)) : Nat :=
  events.foldl (init := 0) (fun acc step =>
    step.foldl (init := acc) (fun acc ev => acc + ev.splitCount))

private def countDeletes (events : Array (Array BranchingStepEvent)) : Nat :=
  events.foldl (init := 0) (fun acc step =>
    step.foldl (init := acc) (fun acc ev => if ev.deleted then acc + 1 else acc))

private def labelSymbol (maskToken label : Nat) : String :=
  if label == maskToken then "X" else qm9AtomicNumberSymbol label

private def scheduledLr (opts : RunOptions) (step : Nat) : Float :=
  if opts.warmupSteps > 0 && step < opts.warmupSteps then
    opts.lr * (step.toFloat / opts.warmupSteps.toFloat)
  else if opts.cooldownSteps > 0 && opts.totalSteps > opts.cooldownSteps &&
      step >= opts.totalSteps - opts.cooldownSteps then
    let cooldownStart := opts.totalSteps - opts.cooldownSteps
    let progress := (step - cooldownStart).toFloat / opts.cooldownSteps.toFloat
    opts.lr - progress * (opts.lr - opts.lrEnd)
  else
    opts.lr

private def cosineGenerationSchedule (steps : Nat) : Array Float := Id.run do
  if steps == 0 then
    return #[0.0, 1.0]
  let pi : Float := 3.14159265358979323846
  let denom := steps.toFloat
  let mut out : Array Float := #[]
  for i in [:steps + 1] do
    let s := i.toFloat / denom
    out := out.push (1.0 - ((Float.cos (pi * s) + 1.0) / 2.0))
  return out

private def generatedPath (outPrefix : String) (sampleCount i : Nat) : String :=
  if sampleCount == 1 then
    outPrefix ++ "_generated.xyz"
  else
    outPrefix ++ "_generated_" ++ toString i ++ ".xyz"

private def trajectoryPath (outPrefix : String) (sampleCount sample step : Nat) : String :=
  if sampleCount == 1 then
    s!"{outPrefix}_step_{step}.xyz"
  else
    s!"{outPrefix}_sample_{sample}_step_{step}.xyz"

private def lineagePath (outPrefix : String) (sampleCount sample : Nat) : String :=
  if sampleCount == 1 then
    s!"{outPrefix}_trajectory.jsonl"
  else
    s!"{outPrefix}_sample_{sample}_trajectory.jsonl"

private def architectureName (opts : RunOptions) : String :=
  if opts.fullArchitecture then "full" else "compact"

private def deviceName (device : Device) : String :=
  match device.kind with
  | .CPU => "cpu"
  | .MPS => "mps"
  | .CUDA i => s!"cuda:{i}"

private def muonCountPath (dir : String) : String :=
  dir ++ "/optim_muon_count.txt"

private def muonOptimStateExists (dir : String) : IO Bool :=
  data.fileExists (muonCountPath dir)

private def saveMuonOptimState [TensorStruct Params]
    (state : MoleculeMuonState Params) (dir : String) : IO Unit := do
  _root_.torch.checkpoint.saveParams state.momentum dir "optim_muon_momentum"
  IO.FS.writeFile (muonCountPath dir) (toString state.step)

private def loadMuonOptimState [TensorStruct Params]
    (template : Params) (dir : String) : IO (MoleculeMuonState Params) := do
  let momentum ←
    _root_.torch.checkpoint.loadParams template dir "optim_muon_momentum" (asParameters := false)
  let countRaw ← IO.FS.readFile (muonCountPath dir)
  let count ← match countRaw.trimAscii.toString.toNat? with
    | some n => pure n
    | none => throw (IO.userError s!"invalid Muon optimizer count in {muonCountPath dir}")
  pure { momentum := TensorStruct.map Tensor.detach momentum, step := count }

private def runWithModel {dtype : DType} [DType.RealFloating dtype]
    [DType.Promotes dtype .Float32 .Float32] [DType.Promotes .Float32 dtype .Float32]
    {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params] {maxLen vocab : UInt64}
    (opts : RunOptions)
    (recordsCount usableCount : Nat)
    (trainStates evalStates : Array (BranchingState MoleculeAtom))
    (bridgeCfg : MoleculeBridgeConfig)
    (labelDFM : Option DistNoisyDiscreteConfig)
    (trainCfg : BranchingTrainConfig)
    (model : BranchingMoleculeModel (dtype := dtype) (device := opts.device) maxLen vocab Params)
    (initParams : Params)
    (makeLearnedModel : Params → Float → BranchingState MoleculeAtom → IO MoleculeModelPrediction) :
    IO Unit := do
  -- Pin one immutable directory before reading model, metadata, and optimizer
  -- state. Legacy flat checkpoint directories resolve to themselves.
  let resumeSnapshot? ← match opts.resumeCheckpoint? with
    | none => pure none
    | some dir => some <$> _root_.torch.checkpoint.resolveSnapshot dir
  let (initParams, startIteration, resumedMeta?) ←
    match resumeSnapshot? with
    | none => pure (initParams, 0, none)
    | some checkpointDir => do
        let checkpointMeta ←
          _root_.torch.checkpoint.loadCheckpointMeta (checkpointDir ++ "/meta.txt")
        let loadedParams ← _root_.torch.checkpoint.loadParams initParams checkpointDir "param"
        IO.println s!"loaded molecule checkpoint from {checkpointDir} at iteration={checkpointMeta.iteration} trainLoss={checkpointMeta.trainLoss}"
        pure (loadedParams, checkpointMeta.iteration, some checkpointMeta)
  let mut params := initParams
  let mut rng : Rng := { state := opts.seed + 17 }

  let (fixedTrainBridge, rng') ←
    sampleMoleculeBridgeBatch bridgeCfg trainStates opts.batchSize rng
      (useBranchingTimeProb := opts.useBranchingTimeProb)
      (maxLen := some opts.maxLen.toNat) (deletionPad := opts.deletionPad) (parallelism := opts.parallelSampling)
  rng := rng'
  let initTrain ←
    evalMoleculeLoss (maxLen := maxLen) (vocab := vocab) trainCfg model params fixedTrainBridge
      labelDFM (device := opts.device)
  let mut finalTrain := initTrain
  let mut lastReport := initTrain

  if !opts.generateOnly then
    let initOptState := initMoleculeMuonState params
    let initOptState ←
      match resumeSnapshot? with
      | some checkpointDir =>
          if opts.resumeOptimizer then
            let expectedCount := (resumedMeta?.map (fun m => m.optimCount)).getD 0
            let hasOptim ← muonOptimStateExists checkpointDir
            if hasOptim then
              let loaded ← loadMuonOptimState params checkpointDir
              if expectedCount > 0 && loaded.step != expectedCount then
                throw (IO.userError s!"Muon optimizer count mismatch in {checkpointDir}: metadata={expectedCount}, optimizer={loaded.step}")
              IO.println s!"loaded molecule Muon optimizer checkpoint from {checkpointDir} at count={loaded.step}"
              pure loaded
            else
              if expectedCount > 0 then
                throw (IO.userError s!"missing Muon optimizer checkpoint in {checkpointDir}: metadata declares count={expectedCount}")
              IO.eprintln s!"warning: no Muon optimizer checkpoint in {checkpointDir}; resuming parameters with fresh optimizer state"
              pure initOptState
          else
            pure initOptState
      | none => pure initOptState
    let mut optState := initOptState
    let mut sampleMs : Nat := 0
    let mut trainMs : Nat := 0

    -- Pipelined sampling: the next batch is sampled on a worker thread while
    -- the current step trains. `sampleMoleculeBridgeBatchPure` is
    -- deterministic given its rng, and rng chaining preserves the sequential
    -- draw order, so results are identical to the non-pipelined loop.
    let sampleNext := fun (rng : Rng) =>
      sampleMoleculeBridgeBatchPure bridgeCfg trainStates opts.batchSize rng
        (useBranchingTimeProb := opts.useBranchingTimeProb)
        (maxLen := some opts.maxLen.toNat) (deletionPad := opts.deletionPad)
        (parallelism := opts.parallelSampling)
    let mut pendingSample := Task.spawn (fun () => sampleNext rng)

    -- Value updates may replace storage; graph replay requires a separate
    -- mutable buffer interface with stable addresses.
    if opts.cudaGraph then
      IO.eprintln "CUDA graph training is unavailable with value-preserving tensor updates; using eager training."

    for step in [:opts.steps] do
      let globalStep := startIteration + step
      let lr := scheduledLr opts globalStep
      let t0 ← IO.monoMsNow
      let (bridgeBatch, rng') := pendingSample.get
      rng := rng'
      pendingSample := Task.spawn (fun () => sampleNext rng)
      let t1 ← IO.monoMsNow
      let (params', optState', report) ←
        trainStepMoleculeMuon (maxLen := maxLen) (vocab := vocab) trainCfg model
          params optState bridgeBatch lr labelDFM (device := opts.device) (timePhases := opts.timePhases)
      params := params'
      optState := optState'
      lastReport := report
      let t2 ← IO.monoMsNow
      sampleMs := sampleMs + (t1 - t0)
      trainMs := trainMs + (t2 - t1)
      if opts.logEvery > 0 && step % opts.logEvery == 0 then
        IO.println s!"molecule_train step={globalStep} lr={lr} total={lastReport.total} coord={lastReport.coord} label={lastReport.label} splits={lastReport.splits} del={lastReport.del}"
        if opts.timePhases && step > 0 then
          IO.println s!"molecule_phases step={globalStep} sample_ms_avg={(sampleMs / (step + 1))} train_ms_avg={(trainMs / (step + 1))}"

    finalTrain ←
      evalMoleculeLoss (maxLen := maxLen) (vocab := vocab) trainCfg model params fixedTrainBridge
        labelDFM (device := opts.device)
    if !(Float.isFinite finalTrain.total) then
      throw (IO.userError "final train loss is not finite")
    if !(finalTrain.total < initTrain.total) then
      match resumedMeta? with
      | some _ =>
          IO.eprintln s!"warning: resumed training did not reduce fixed train loss over this short run: initial={initTrain.total}, final={finalTrain.total}"
      | none =>
          throw (IO.userError s!"training did not reduce fixed train loss: initial={initTrain.total}, final={finalTrain.total}")

    if opts.saveCheckpoint then
      let checkpointIteration := startIteration + opts.steps
      _root_.torch.checkpoint.publishSnapshot opts.checkpointDir "CURRENT" fun snapshot => do
        _root_.torch.checkpoint.saveParams params snapshot "param"
        if opts.saveOptimizer then
          saveMuonOptimState optState snapshot
        _root_.torch.checkpoint.saveCheckpointMeta {
          iteration := checkpointIteration
          bestValLoss := finalTrain.total
          trainLoss := finalTrain.total
          optimCount := if opts.saveOptimizer then optState.step else 0
        } (snapshot ++ "/meta.txt")
      IO.println s!"wrote molecule checkpoint {opts.checkpointDir}"

  let (evalBridge, rng') ←
    sampleMoleculeBridgeBatch bridgeCfg evalStates opts.batchSize rng
      (useBranchingTimeProb := opts.useBranchingTimeProb)
      (maxLen := some opts.maxLen.toNat) (deletionPad := opts.deletionPad) (parallelism := opts.parallelSampling)
  rng := rng'
  let evalReport ←
    evalMoleculeLoss (maxLen := maxLen) (vocab := vocab) trainCfg model params evalBridge
      labelDFM (device := opts.device)

  let target := evalStates[0]!
  writeMoleculeXYZ ⟨opts.outputPrefix ++ "_target.xyz"⟩ target
    "target molecule from evaluation split" (labelSymbol opts.maskToken)

  let mut totalGeneratedSplits := 0
  let mut totalGeneratedDeletes := 0
  let mut lastSourceAtoms := 0
  let mut lastGeneratedAtoms := 0
  if opts.generate then
    let flow : CoalescentFlow MoleculeBridgeConfig MoleculeAtom :=
      CoalescentFlow.mkDefault bridgeCfg TimeDist.betaOneThreeHalves TimeDist.uniform
    let learnedModel := makeLearnedModel params
    let schedule := cosineGenerationSchedule opts.sampleSteps
    for i in [:opts.sampleCount] do
      let (x0Atom, rng') := MoleculeBridgeConfig.sampleInitialAtom bridgeCfg rng
      rng := rng'
      let x0 : BranchingState MoleculeAtom := BranchingState.mkDefault #[x0Atom] #[0]
      if i == 0 then
        writeMoleculeXYZ ⟨opts.outputPrefix ++ "_source.xyz"⟩ x0
          "masked source molecule for learned generation" (labelSymbol opts.maskToken)
      let (generated, rng') ←
        moleculeBranchingGenerateIO flow x0 learnedModel schedule
          (maxStateLen? := some opts.maxLen.toNat) (rng := rng)
      rng := rng'
      if generated.finalState.state.isEmpty then
        throw (IO.userError s!"learned molecule generation produced an empty state for sample {i}")
      totalGeneratedSplits := totalGeneratedSplits + countSplits generated.events
      totalGeneratedDeletes := totalGeneratedDeletes + countDeletes generated.events
      lastSourceAtoms := x0.state.size
      lastGeneratedAtoms := generated.finalState.state.size
      writeMoleculeXYZ ⟨generatedPath opts.outputPrefix opts.sampleCount i⟩ generated.finalState
        s!"learned BranchingFlows molecule generation sample={i}" (labelSymbol opts.maskToken)
      writeMoleculeTrajectoryJsonl ⟨lineagePath opts.outputPrefix opts.sampleCount i⟩ generated
      for step in [:generated.trajectory.size] do
        let state := generated.trajectory[step]!
        let time := generated.times.getD step 0.0
        writeMoleculeXYZ ⟨trajectoryPath opts.outputPrefix opts.sampleCount i step⟩ state
          s!"learned branching sample={i} step={step} t={time}" (labelSymbol opts.maskToken)
        if i == 0 then
          IO.println s!"molecule_trajectory step={step} t={time} atoms={state.state.size}"
          if step > 0 then
            for event in generated.events.getD (step - 1) #[] do
              IO.println s!"molecule_event step={step} source_id={event.sourceId} splits={event.splitCount} deleted={event.deleted} interval=[{event.t0}, {event.t1}]"

  IO.println s!"molecule_dataset records={recordsCount} usable={usableCount} train={trainStates.size} eval={evalStates.size}"
  let labelProcess := if opts.fixedLabels then "fixed" else "dfm"
  let splitCap := match opts.splitLogitCap? with | some cap => toString cap | none => "none"
  IO.println s!"molecule_config architecture={architectureName opts} optimizer=muon device={deviceName opts.device} dtype={if opts.bf16 then "bf16" else "fp32"} max_len={opts.maxLen} vocab_size={opts.vocabSize} mask_token={opts.maskToken} label_process={labelProcess} hidden_dim={opts.hiddenDim} heads={opts.heads} head_dim={opts.headDim} mlp={opts.mlp} rff_dim={opts.rffDim} layers={opts.layers} coord_update_layers={opts.coordUpdateLayers} batch_size={opts.batchSize} steps={opts.steps} total_steps={opts.totalSteps} coord_weight={opts.coordWeight} label_weight={opts.labelWeight} splits_weight={opts.splitsWeight} del_weight={opts.delWeight} branching_time_prob={opts.useBranchingTimeProb} split_logit_cap={splitCap}"
  if opts.generateOnly then
    IO.println s!"molecule_eval checkpoint_train_total={finalTrain.total} heldout_total={evalReport.total}"
  else
    IO.println s!"molecule_train fixed_initial_total={initTrain.total} fixed_final_total={finalTrain.total}"
    IO.println s!"molecule_train last_batch_total={lastReport.total} eval_total={evalReport.total}"
    if opts.saveCheckpoint then
      IO.println s!"molecule_train checkpoint_dir={opts.checkpointDir}"
  if opts.generate then
    IO.println s!"molecule_generate samples={opts.sampleCount} sample_steps={opts.sampleSteps} target_atoms={target.state.size} last_source_atoms={lastSourceAtoms} last_generated_atoms={lastGeneratedAtoms}"
    IO.println s!"molecule_generate splits={totalGeneratedSplits} deletes={totalGeneratedDeletes}"
  IO.println s!"wrote {opts.outputPrefix}_target.xyz"
  if opts.generate then
    IO.println s!"wrote {opts.outputPrefix}_source.xyz"
    IO.println s!"wrote generated samples under {opts.outputPrefix}_generated*.xyz"
    IO.println s!"wrote lineage-aware raw trajectories under {opts.outputPrefix}_*trajectory.jsonl"
    IO.println s!"wrote learned branching trajectory frames under {opts.outputPrefix}_*step*.xyz"

private def runAt (dtype : DType) [DType.RealFloating dtype]
    [DType.Promotes dtype .Float32 .Float32] [DType.Promotes .Float32 dtype .Float32]
    [DType.NonBool dtype] (opts : RunOptions) : IO Unit := do
  if opts.generateOnly && opts.resumeCheckpoint?.isNone then
    throw (IO.userError "--generate-only requires --resume-checkpoint")
  if opts.generateOnly && !opts.generate then
    throw (IO.userError "--generate-only cannot be combined with --no-generate")
  if !opts.generateOnly && opts.steps == 0 then
    throw (IO.userError "training steps must be positive")
  if opts.batchSize == 0 then
    throw (IO.userError "batch size must be positive")
  if opts.totalSteps == 0 then
    throw (IO.userError "total steps must be positive")
  if opts.sampleCount == 0 then
    throw (IO.userError "sample count must be positive")
  if opts.vocabSize == 0 then
    throw (IO.userError "vocab size must be positive")
  if opts.maskToken >= opts.vocabSize then
    throw (IO.userError s!"mask token {opts.maskToken} is outside vocab size {opts.vocabSize}")
  if opts.maxLen == 0 then
    throw (IO.userError "max length must be positive")
  match opts.coordTargetCap? with
  | some cap =>
      if cap <= 0.0 then
        throw (IO.userError "--coord-target-cap must be positive")
  | none => pure ()
  if opts.useBranchingTimeProb < 0.0 || opts.useBranchingTimeProb > 1.0 then
    throw (IO.userError "--branching-time-prob must be between 0 and 1")
  if opts.coordWeight < 0.0 || opts.labelWeight < 0.0 ||
      opts.splitsWeight < 0.0 || opts.delWeight < 0.0 then
    throw (IO.userError "molecule loss weights must be nonnegative")
  if opts.weightDecay < 0.0 then
    throw (IO.userError "--weight-decay must be nonnegative")
  match opts.splitLogitCap? with
  | some cap =>
      if cap > 11.0 then
        throw (IO.userError "--split-logit-cap cannot exceed the core Julia-compatible cap of 11")
  | none => pure ()
  if opts.heads == 0 || opts.headDim == 0 || opts.mlp == 0 then
    throw (IO.userError "heads, head-dim, and mlp must be positive")
  if opts.fullArchitecture then
    if opts.hiddenDim == 0 || opts.rffDim == 0 || opts.layers == 0 then
      throw (IO.userError "hidden-dim, rff-dim, and layers must be positive for full architecture")
    if opts.coordUpdateLayers == 0 then
      throw (IO.userError "coord-update-layers must be positive for full architecture")
    if opts.headDim % 2 != 0 then
      throw (IO.userError "full architecture requires an even head-dim for RoPE")
  if opts.requireData then
    match opts.dataPath? with
    | some path =>
        if path == "-" then
          throw (IO.userError "--require-data cannot use embedded fixture data")
    | none =>
        throw (IO.userError "--require-data requires --data")

  let vocab : UInt64 := opts.vocabSize.toUInt64
  let bridgeCfgBase := MoleculeBridgeConfig.qm9 opts.vocabSize opts.maskToken
  let labelDFM :=
    if opts.fixedLabels then none
    else some (DistNoisyDiscreteConfig.qm9 opts.vocabSize opts.maskToken)
  let bridgeCfg := { bridgeCfgBase with labelDFM }
  let records ← loadRecords opts
  let allStates ← exceptToIO "convert molecule records"
    (qm9RecordsToBranchingStates records { vocabSize? := some opts.vocabSize, maskToken? := some opts.maskToken })
  let states := allStates.filter (fun state => state.state.size <= opts.maxLen.toNat)
  if states.isEmpty then
    throw (IO.userError s!"no molecules fit maxLen={opts.maxLen}")
  let (trainStates, evalStates) := splitTrainEval states opts.seed
  let evalStates := if evalStates.isEmpty then trainStates else evalStates

  let trainCfg : BranchingTrainConfig := {
    maxLen := opts.maxLen
    padToken := 0
    anchorWeight := 1.0
    coordWeight := opts.coordWeight
    labelWeight := opts.labelWeight
    splitsWeight := opts.splitsWeight
    delWeight := opts.delWeight
    weightDecay := opts.weightDecay
    gradClip := opts.gradClip
  }

  torch.manualSeed opts.seed
  if opts.fullArchitecture then
    let model :=
      fullMoleculeTransformerModel (dtype := dtype) (device := opts.device) (maxLen := opts.maxLen) (vocab := vocab)
        (hidden := opts.hiddenDim) (heads := opts.heads) (headDim := opts.headDim)
        (mlp := opts.mlp) (rff := opts.rffDim) (layers := opts.layers)
        (coordUpdateLayers := opts.coordUpdateLayers)
        (coordTargetCap? := opts.coordTargetCap?)
    let initParams ←
      FullMoleculeTransformerParams.init vocab opts.hiddenDim opts.heads opts.headDim
        opts.mlp opts.rffDim opts.layers (device := opts.device)
    let initParams ← TensorStruct.castTo
      (FullMoleculeTransformerParams (device := opts.device) vocab opts.hiddenDim opts.heads opts.headDim
        opts.mlp opts.rffDim opts.layers dtype) initParams
    let makeLearnedModel := fun params =>
      fullMoleculeTransformerIOModel (maxLen := opts.maxLen) (vocab := vocab)
        (hidden := opts.hiddenDim) (heads := opts.heads) (headDim := opts.headDim)
        (mlp := opts.mlp) (rff := opts.rffDim) (layers := opts.layers)
        trainCfg.padToken params (coordUpdateLayers := opts.coordUpdateLayers)
        (splitLogitCap? := opts.splitLogitCap?)
        (coordTargetCap? := opts.coordTargetCap?)
        (device := opts.device)
    runWithModel (maxLen := opts.maxLen) (vocab := vocab)
      opts records.size states.size trainStates evalStates bridgeCfg labelDFM trainCfg
      model initParams makeLearnedModel
  else
    let model :=
      moleculeTransformerModel (dtype := dtype) (device := opts.device) (maxLen := opts.maxLen) (vocab := vocab)
        (heads := opts.heads) (headDim := opts.headDim) (mlp := opts.mlp)
        (coordTargetCap? := opts.coordTargetCap?)
    let initParams ← MoleculeTransformerParams.init vocab opts.heads opts.headDim opts.mlp (device := opts.device)
    let initParams ← TensorStruct.castTo
      (MoleculeTransformerParams (device := opts.device) vocab opts.heads opts.headDim opts.mlp dtype) initParams
    let makeLearnedModel := fun params =>
      moleculeTransformerIOModel (maxLen := opts.maxLen) (vocab := vocab)
        (heads := opts.heads) (headDim := opts.headDim) (mlp := opts.mlp)
        trainCfg.padToken params (splitLogitCap? := opts.splitLogitCap?)
        (coordTargetCap? := opts.coordTargetCap?)
        (device := opts.device)
    runWithModel (maxLen := opts.maxLen) (vocab := vocab)
      opts records.size states.size trainStates evalStates bridgeCfg labelDFM trainCfg
      model initParams makeLearnedModel

def run (opts : RunOptions) : IO Unit :=
  if opts.bf16 then runAt .BFloat16 opts else runAt .Float32 opts

def _root_.main (args : List String) : IO UInt32 := do
  let opts ← parseOptions args
  run opts
  pure 0

end Examples.BranchingFlows
