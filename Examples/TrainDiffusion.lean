/-
  Diffusion Training Script

  Train a tiny discrete diffusion model on Shakespeare text.
  Uses masked diffusion with bidirectional attention.
-/
import Tyr
import Examples.Diffusion.Diffusion
import Examples.Diffusion.DiffusionSchedule
import Examples.Diffusion.DiffusionTrain
import Examples.Diffusion.DiffusionCheckpoint

namespace Examples.TrainDiffusion

open torch
open torch.diffusion
open torch.diffusion.train
open torch.diffusion.checkpoint

-- ANSI escape codes for TUI animation
def ansiReset : String := "\x1b[0m"
def ansiGreen : String := "\x1b[32m"
def ansiYellow : String := "\x1b[33m"
def ansiRed : String := "\x1b[31m"
def ansiBgGreen : String := "\x1b[42m"
def ansiBgYellow : String := "\x1b[43m"
def ansiBgRed : String := "\x1b[41m"
def ansiDim : String := "\x1b[2m"
def ansiClearLine : String := "\x1b[2K"
def ansiMoveUp (n : Nat) : String := s!"\x1b[{n}A"
def ansiHideCursor : String := "\x1b[?25l"
def ansiShowCursor : String := "\x1b[?25h"

-- ASCII encoding/decoding (simple, like tiny-diffusion)
-- Vocab is 0-127 where 0 is reserved as [MASK] token
def asciiEncode (s : String) : Array Int64 :=
  s.toList.toArray.map fun c =>
    let code := c.toNat
    if code < 128 then code.toInt64 else 127  -- clamp to ASCII

def asciiDecode (tokens : Array Int64) : String :=
  String.ofList <| tokens.toList.map fun t =>
    let code := t.toUInt64.toNat
    if code == 0 then '[' -- Show mask token as [
    else if code < 128 then Char.ofNat code
    else '?'

/-- Convert tensor to array for decoding -/
def tensorToArray {n : UInt64} (t : Tokens #[n]) : IO (Array Int64) := do
  return (← t.toUInt64Array).map UInt64.toInt64

/-- Extract tensor values as Float array (for confidence display). -/
def tensorToFloatArray {n : UInt64} (t : Activation #[n]) : IO (Array Float) :=
  t.toFloatArray

/-- Decode multiple blocks from a batched tensor [numBlocks, seqLen] -/
def decodeMultipleBlocks {numBlocks seqLen : UInt64}
    (tensor : Tokens #[numBlocks, seqLen]) : IO (Array String) := do
  let mut results := #[]
  for i in [:numBlocks.toNat] do
    -- Extract row i: slice along batch dimension
    let start := i.toUInt64
    let row : Tokens #[1, seqLen] := tensor.sliceChecked 0 start 1 #[1, seqLen]
    let rowFlat := Tensor.reshapeChecked row #[seqLen]
    let arr ← tensorToArray rowFlat
    results := results.push (asciiDecode arr)
  return results

/-- Render mask with confidence-based background color -/
def renderMask (conf : Float) : String :=
  let bg := if conf > 0.7 then ansiBgGreen
            else if conf > 0.4 then ansiBgYellow
            else ansiBgRed
  s!"{bg}{ansiDim}█{ansiReset}"

/-- Render decoded char with confidence-based foreground color -/
def renderChar (c : Char) (conf : Float) : String :=
  let fg := if conf > 0.7 then ansiGreen
            else if conf > 0.4 then ansiYellow
            else ansiRed
  s!"{fg}{c}{ansiReset}"

/-- Progress bar -/
def progressBar (current total : Nat) (width : Nat := 30) : String :=
  let filled := current * width / total
  let bar := String.ofList (List.replicate filled '=') ++ ">" ++
             String.ofList (List.replicate (width - filled) ' ')
  s!"[{bar}]"

/-- Move cursor to absolute position (1-indexed) -/
def ansiMoveTo (row col : Nat) : String := s!"\x1b[{row};{col}H"

/-- Save cursor position -/
def ansiSaveCursor : String := "\x1b[s"

/-- Restore cursor position -/
def ansiRestoreCursor : String := "\x1b[u"

/-- Animated sampling with optional prompt conditioning -/
def sampleAnimated {cfg : Config}
    (params : DiffusionParams cfg)
    (rotaryCache : RotaryCache cfg.seq_len cfg.headDim)
    (k : UInt64)
    (temperature : Float := 0.8)
    (frameDelay : Nat := 100)
    (prompt : String := "")
    (displayWidth : Nat := 70)  -- Keep narrow to avoid wrapping
    : IO (Tokens #[1, cfg.seq_len]) := do

  -- Calculate display dimensions
  let textLen := min cfg.seq_len.toNat displayWidth
  let numTextLines := (textLen + displayWidth - 1) / displayWidth  -- ceiling division
  let totalLines := 1 + numTextLines  -- header + text lines

  -- Reserve space by printing blank lines
  for _ in [:totalLines] do
    IO.println ""

  IO.print ansiHideCursor
  (← IO.getStdout).flush

  -- Start with all masks
  let mut x := Tensor.fullInt #[1, cfg.seq_len] cfg.mask_token_id.toInt64

  -- Set prompt tokens at the beginning (these won't be masked)
  let promptTokens := asciiEncode prompt
  let promptLen := min promptTokens.size cfg.seq_len.toNat
  for i in [:promptLen] do
    let promptTensor := Tensor.fullInt #[1, cfg.seq_len] (promptTokens[i]!)
    let positions := Tensor.arange cfg.seq_len
    let posMask := Tensor.eqScalar positions i.toInt64
    let posMaskBatch := Tensor.expand (Tensor.unsqueeze posMask 0) #[1, cfg.seq_len]
    x := Tensor.whereSelect posMaskBatch promptTensor x

  let mut maskedPositions := getMaskedPositions' x cfg.mask_token_id

  for step in [:cfg.diffusion_steps.toNat] do
    if !(Tensor.any maskedPositions) then break

    let tVal := (cfg.diffusion_steps.toNat - 1 - step).max 0
    let t := Tensor.fullInt #[1] tVal.toUInt64.toInt64

    let logits := forward params x t rotaryCache
    let probs := Tensor.softmax (logits / temperature)
    let (confidences, predictedTokens) ← IO.ofExcept (Tensor.maxLast3d? probs)

    -- Extract arrays for display
    let tokensArr ← tensorToArray (Tensor.reshapeChecked x #[cfg.seq_len])
    let confArr ← tensorToFloatArray (Tensor.reshapeChecked confidences #[cfg.seq_len])

    -- Build display text with colors
    let remaining := tokensArr.foldl (fun acc t =>
      if t == cfg.mask_token_id.toInt64 then acc + 1 else acc) 0

    let mut textChars := ""
    for i in [:textLen] do
      let tok := tokensArr[i]!
      let conf := confArr[i]!
      if tok == cfg.mask_token_id.toInt64 then
        textChars := textChars ++ renderMask conf
      else
        let c := Char.ofNat tok.toUInt64.toNat
        -- Show printable chars, replace control chars with dot
        if c.toNat >= 32 && c.toNat < 127 then
          textChars := textChars ++ renderChar c conf
        else if c.toNat == 10 then  -- newline
          textChars := textChars ++ s!"{ansiDim}↵{ansiReset}"
        else
          textChars := textChars ++ s!"{ansiDim}·{ansiReset}"

    -- Move up to start of our reserved area and render
    IO.print (ansiMoveUp totalLines)

    -- Header line
    let header := s!"{ansiClearLine}Step {step}/{cfg.diffusion_steps}  " ++
                  progressBar step cfg.diffusion_steps.toNat 20 ++
                  s!"  {remaining} remaining"
    IO.println header

    -- Text line(s)
    IO.print ansiClearLine
    IO.println textChars

    -- Fill remaining reserved lines if any
    for _ in [:totalLines - 2] do
      IO.print ansiClearLine
      IO.println ""

    (← IO.getStdout).flush
    IO.sleep frameDelay.toUInt32

    -- Update tokens (top-k decoding)
    let negInf := Tensor.full #[1, cfg.seq_len] (-1e10)
    let maskedConf := Tensor.whereSelect maskedPositions confidences negInf
    let (_, topkIdx) ← IO.ofExcept (Tensor.topKLast2d? maskedConf k)
    let selectMask := Tensor.scatter2d (Tensor.zeros #[1, cfg.seq_len]) 1
                        (topkIdx) (Tensor.ones #[1, k])
    let finalMask := Tensor.logicalAnd (Tensor.lt (Tensor.zeros #[1, cfg.seq_len]) selectMask) maskedPositions
    x := Tensor.whereSelect finalMask (predictedTokens) x
    maskedPositions := getMaskedPositions' x cfg.mask_token_id

  IO.print ansiShowCursor
  (← IO.getStdout).flush
  return x

/-- Animated generation of multiple blocks in sequence with continuity.
    Each block uses the last `overlap` tokens from the previous block as context,
    creating a continuous stream of text across blocks.
-/
def sampleAnimatedMultiple {cfg : Config}
    (params : DiffusionParams cfg)
    (rotaryCache : RotaryCache cfg.seq_len cfg.headDim)
    (numBlocks : Nat)
    (k : UInt64)
    (temperature : Float := 0.8)
    (frameDelay : Nat := 100)
    (prompt : String := "")
    (displayWidth : Nat := 70)
    (overlap : Nat := 32)  -- tokens carried from previous block for continuity
    : IO (Array String) := do
  let mut results := #[]
  let mut carryoverTokens : Array Int64 := #[]  -- tokens to carry to next block

  -- Encode initial prompt
  let initialPrompt := asciiEncode prompt

  for blockIdx in [:numBlocks] do
    -- Print block header
    IO.println ""
    IO.println s!"{ansiGreen}━━━ Generating Block {blockIdx + 1}/{numBlocks} ━━━{ansiReset}"
    if blockIdx > 0 then
      IO.println s!"{ansiDim}(continuing from previous block with {carryoverTokens.size} tokens){ansiReset}"
    IO.println ""

    -- Calculate display dimensions
    let textLen := min cfg.seq_len.toNat displayWidth
    let numTextLines := (textLen + displayWidth - 1) / displayWidth
    let totalLines := 1 + numTextLines

    -- Reserve space
    for _ in [:totalLines] do
      IO.println ""

    IO.print ansiHideCursor
    (← IO.getStdout).flush

    -- Start with all masks
    let mut x := Tensor.fullInt #[1, cfg.seq_len] cfg.mask_token_id.toInt64

    -- Determine context tokens for this block
    let contextTokens := if blockIdx == 0 then
      initialPrompt  -- First block uses the user prompt
    else
      carryoverTokens  -- Subsequent blocks use carryover from previous

    -- Set context tokens at the beginning (these provide continuity)
    let contextLen := min contextTokens.size cfg.seq_len.toNat
    for i in [:contextLen] do
      let tokenTensor := Tensor.fullInt #[1, cfg.seq_len] (contextTokens[i]!)
      let positions := Tensor.arange cfg.seq_len
      let posMask := Tensor.eqScalar positions i.toInt64
      let posMaskBatch := Tensor.expand (Tensor.unsqueeze posMask 0) #[1, cfg.seq_len]
      x := Tensor.whereSelect posMaskBatch tokenTensor x

    let mut maskedPositions := getMaskedPositions' x cfg.mask_token_id

    for step in [:cfg.diffusion_steps.toNat] do
      if !(Tensor.any maskedPositions) then break

      let tVal := (cfg.diffusion_steps.toNat - 1 - step).max 0
      let t := Tensor.fullInt #[1] tVal.toUInt64.toInt64

      let logits := forward params x t rotaryCache
      let probs := Tensor.softmax (logits / temperature)
      let (confidences, predictedTokens) ← IO.ofExcept (Tensor.maxLast3d? probs)

      -- Extract arrays for display
      let tokensArr ← tensorToArray (Tensor.reshapeChecked x #[cfg.seq_len])
      let confArr ← tensorToFloatArray (Tensor.reshapeChecked confidences #[cfg.seq_len])

      let remaining := tokensArr.foldl (fun acc t =>
        if t == cfg.mask_token_id.toInt64 then acc + 1 else acc) 0

      let mut textChars := ""
      for i in [:textLen] do
        let tok := tokensArr[i]!
        let conf := confArr[i]!
        -- Highlight context tokens differently
        let isContext := i < contextLen
        if tok == cfg.mask_token_id.toInt64 then
          textChars := textChars ++ renderMask conf
        else
          let c := Char.ofNat tok.toUInt64.toNat
          if c.toNat >= 32 && c.toNat < 127 then
            if isContext then
              textChars := textChars ++ s!"{ansiDim}{c}{ansiReset}"  -- dim for context
            else
              textChars := textChars ++ renderChar c conf
          else if c.toNat == 10 then
            textChars := textChars ++ s!"{ansiDim}↵{ansiReset}"
          else
            textChars := textChars ++ s!"{ansiDim}·{ansiReset}"

      -- Move up and render
      IO.print (ansiMoveUp totalLines)

      let header := s!"{ansiClearLine}Block {blockIdx + 1}/{numBlocks} | Step {step}/{cfg.diffusion_steps}  " ++
                    progressBar step cfg.diffusion_steps.toNat 20 ++
                    s!"  {remaining} remaining"
      IO.println header

      IO.print ansiClearLine
      IO.println textChars

      for _ in [:totalLines - 2] do
        IO.print ansiClearLine
        IO.println ""

      (← IO.getStdout).flush
      IO.sleep frameDelay.toUInt32

      -- Update tokens
      let negInf := Tensor.full #[1, cfg.seq_len] (-1e10)
      let maskedConf := Tensor.whereSelect maskedPositions confidences negInf
      let (_, topkIdx) ← IO.ofExcept (Tensor.topKLast2d? maskedConf k)
      let selectMask := Tensor.scatter2d (Tensor.zeros #[1, cfg.seq_len]) 1
                          (topkIdx) (Tensor.ones #[1, k])
      let finalMask := Tensor.logicalAnd (Tensor.lt (Tensor.zeros #[1, cfg.seq_len]) selectMask) maskedPositions
      x := Tensor.whereSelect finalMask (predictedTokens) x
      maskedPositions := getMaskedPositions' x cfg.mask_token_id

    IO.print ansiShowCursor
    (← IO.getStdout).flush

    -- Decode result
    let genFlat := Tensor.reshapeChecked x #[cfg.seq_len]
    let genArr ← tensorToArray genFlat

    -- Extract the NEW content (excluding context that came from previous block)
    let newContent := if blockIdx == 0 then
      genArr  -- First block: include everything
    else
      genArr.extract contextLen genArr.size  -- Skip the carryover context

    let text := asciiDecode newContent
    results := results.push text

    -- Extract last `overlap` tokens for next block's context
    let overlapStart := if genArr.size > overlap then genArr.size - overlap else 0
    carryoverTokens := genArr.extract overlapStart genArr.size

    -- Show final result for this block
    IO.println ""
    IO.println s!"{ansiDim}Completed block {blockIdx + 1} ({newContent.size} new tokens):{ansiReset}"
    IO.println text

  return results

/-- Concatenate all blocks into a single continuous string -/
def joinBlocks (blocks : Array String) : String :=
  blocks.foldl (· ++ ·) ""

/-- Train diffusion model with validation -/
def runTrainingWithVal {nTrain nVal : UInt64}
    (modelCfg : Config) (trainCfg : TrainConfig)
    (trainData : Tokens #[nTrain]) (valData : Tokens #[nVal])
    : IO (DiffusionParams modelCfg) := do
  IO.println ""
  IO.println "Initializing diffusion model..."
  let params ← DiffusionParams.init modelCfg

  -- Initialize optimizer state
  let opt := Optim.adamw (lr := trainCfg.learningRate) (weight_decay := trainCfg.weightDecay)
  let optState := opt.init params

  -- Initialize mask schedule
  let schedule := MaskedDiffusionSchedule.init modelCfg.diffusion_steps modelCfg.mask_token_id modelCfg.context_len

  -- Initialize rotary cache (exact seq_len to avoid slice2d issues)
  let rotaryCache ← RotaryCache.init modelCfg.seq_len modelCfg.headDim

  IO.println s!"Model initialized with {modelCfg.n_layer} layers, {modelCfg.n_embd} embedding dim"
  IO.println s!"Diffusion steps: {modelCfg.diffusion_steps}, context_len: {modelCfg.context_len}"
  IO.println ""

  -- Train with validation
  let (finalParams, _bestValLoss) ← trainLoop trainCfg params optState schedule trainData valData rotaryCache

  IO.println ""
  return finalParams

/-- Train on random data for testing -/
def runTrainingRandom (modelCfg : Config) (trainCfg : TrainConfig) : IO (DiffusionParams modelCfg) := do
  IO.println ""
  IO.println "Using random tokens for testing..."
  let numTokens : UInt64 := 50000
  -- Generate random tokens (1-127, avoiding 0 which is mask)
  let trainData ← Tensor.randint 1 127 #[numTokens]
  let valData ← Tensor.randint 1 127 #[numTokens / 10]

  runTrainingWithVal modelCfg trainCfg trainData valData

/-- Load text file and convert to ASCII tokens -/
def loadTextFile (path : String) : IO (Array Int64) := do
  let contents ← IO.FS.readFile path
  return asciiEncode contents

/-- Parse command line arguments -/
structure Args where
  generateOnly : Bool := false
  checkpointDir : String := "checkpoints/diffusion"
  prompt : String := ""
  numBlocks : Nat := 4
  temperature : Float := 0.9
  deriving Repr

def parseArgs (args : List String) : Args := Id.run do
  let mut result : Args := {}
  let mut i := 0
  while i < args.length do
    let arg := args[i]!
    if arg == "--generate" || arg == "-g" then
      result := { result with generateOnly := true }
      -- Check if next arg is a path (doesn't start with -)
      if i + 1 < args.length && !args[i+1]!.startsWith "-" then
        result := { result with checkpointDir := args[i+1]! }
        i := i + 1
    else if arg == "--checkpoint" || arg == "-c" then
      if i + 1 < args.length then
        result := { result with checkpointDir := args[i+1]! }
        i := i + 1
    else if arg == "--prompt" || arg == "-p" then
      if i + 1 < args.length then
        result := { result with prompt := args[i+1]! }
        i := i + 1
    else if arg == "--blocks" || arg == "-n" then
      if i + 1 < args.length then
        result := { result with numBlocks := args[i+1]!.toNat?.getD 4 }
        i := i + 1
    else if arg == "--temperature" || arg == "-t" then
      if i + 1 < args.length then
        -- Simple float parsing
        let s := args[i+1]!
        let temp := if s.contains '.' then
          let parts := s.splitOn "."
          let intPart := parts[0]!.toNat?.getD 0
          let fracPart := parts[1]!.toNat?.getD 0
          let fracLen := parts[1]!.length
          intPart.toFloat + fracPart.toFloat / (10.0 ^ fracLen.toFloat)
        else s.toNat?.getD 1 |>.toFloat
        result := { result with temperature := temp }
        i := i + 1
    i := i + 1
  return result

def printUsage : IO Unit := do
  IO.println "Usage: TrainDiffusion [options]"
  IO.println ""
  IO.println "Options:"
  IO.println "  --generate, -g [path]   Load checkpoint and generate (skip training)"
  IO.println "  --checkpoint, -c <path> Checkpoint directory (default: checkpoints/diffusion)"
  IO.println "  --prompt, -p <text>     Prompt for generation (default: empty)"
  IO.println "  --blocks, -n <num>      Number of blocks to generate (default: 4)"
  IO.println "  --temperature, -t <val> Sampling temperature (default: 0.9)"
  IO.println ""
  IO.println "Examples:"
  IO.println "  TrainDiffusion                           # Train from scratch"
  IO.println "  TrainDiffusion --generate                # Generate from default checkpoint"
  IO.println "  TrainDiffusion -g checkpoints/best -n 8  # Generate 8 blocks from checkpoint"
  IO.println "  TrainDiffusion -g -p \"ROMEO:\" -t 0.7     # Generate with prompt"

def runGeneration (modelCfg : Config) (params : DiffusionParams modelCfg)
    (prompt : String) (numBlocks : Nat) (temperature : Float) : IO Unit := do
  -- Initialize rotary cache for generation
  let rotaryCache ← RotaryCache.init modelCfg.seq_len modelCfg.headDim

  -- Generate multiple blocks with continuity
  IO.println ""
  IO.println s!"=== Generating {numBlocks} Continuous Blocks (Animated) ==="
  if prompt != "" then
    IO.println s!"Starting prompt: \"{prompt}\""

  let blocks ← sampleAnimatedMultiple params rotaryCache
      (numBlocks := numBlocks) (k := 4) (temperature := temperature) (frameDelay := 60)
      (prompt := prompt) (displayWidth := 80) (overlap := 32)

  -- Show continuous joined text
  IO.println ""
  IO.println "=== Complete Continuous Text ==="
  IO.println "---"
  IO.println (joinBlocks blocks)
  IO.println "---"
  IO.println ""
  IO.println "Done!"

def main (args : List String) : IO Unit := do
  -- Check for help
  if args.contains "--help" || args.contains "-h" then
    printUsage
    return

  let parsedArgs := parseArgs args

  IO.println "Starting..."
  IO.println "=== Tyr Discrete Diffusion ==="
  IO.println ""
  (← IO.getStdout).flush

  -- Small model config (matches nanoGPT style)
  let modelCfg : Config := Config.small
  IO.println s!"Model config: vocab={modelCfg.vocab_size}, seq_len={modelCfg.seq_len}"
  IO.println s!"  n_layer={modelCfg.n_layer}, n_head={modelCfg.n_head}, n_embd={modelCfg.n_embd}"
  IO.println s!"  diffusion_steps={modelCfg.diffusion_steps}, context_len={modelCfg.context_len}"

  -- Check if we're in generate-only mode
  if parsedArgs.generateOnly then
    IO.println ""
    IO.println s!"Loading checkpoint from: {parsedArgs.checkpointDir}"

    -- Check if checkpoint exists
    let ckptExists ← checkpointExists parsedArgs.checkpointDir
    if !ckptExists then
      IO.println s!"Error: Checkpoint not found at {parsedArgs.checkpointDir}"
      IO.println "Train a model first or specify a valid checkpoint path."
      return

    let params ← loadDiffusionParams modelCfg parsedArgs.checkpointDir
    runGeneration modelCfg params parsedArgs.prompt parsedArgs.numBlocks parsedArgs.temperature
    return

  -- Training mode
  let trainCfg : TrainConfig := {
    maxIters := 5000
    evalInterval := 500
    logInterval := 100
    learningRate := 1e-2
    minLr := 1e-4
    warmupIters := 200
    lrDecayIters := 5000
    gradClip := 1.0
    batchSize := 64
    weightDecay := 0.0
  }

  -- Try to load Shakespeare data
  IO.println ""
  IO.println "Loading data..."
  (← IO.getStdout).flush

  let shakespeareExists ← do
    try
      let _ ← IO.FS.readFile "data/shakespeare_char/input.txt"
      pure true
    catch _ =>
      pure false

  let trainedParams ← if shakespeareExists then
    IO.println "Found Shakespeare text file..."
    let tokens ← loadTextFile "data/shakespeare_char/input.txt"
    IO.println s!"Loaded {tokens.size} tokens"

    -- Split 90/10 for train/val
    let splitIdx := tokens.size * 9 / 10
    let trainTokens := tokens.extract 0 splitIdx
    let valTokens := tokens.extract splitIdx tokens.size

    let nTrain : UInt64 := trainTokens.size.toUInt64
    let nVal : UInt64 := valTokens.size.toUInt64

    IO.println s!"Training set: {nTrain} tokens"
    IO.println s!"Validation set: {nVal} tokens"

    -- Convert to tensors
    let trainData : Tokens #[nTrain] := Tensor.reshapeChecked (Tensor.fromInt64Array trainTokens) #[nTrain]
    let valData : Tokens #[nVal] := Tensor.reshapeChecked (Tensor.fromInt64Array valTokens) #[nVal]

    runTrainingWithVal modelCfg trainCfg trainData valData
  else
    IO.println "Shakespeare data not found, using random tokens for testing..."
    runTrainingRandom modelCfg trainCfg

  -- Generate after training
  runGeneration modelCfg trainedParams parsedArgs.prompt parsedArgs.numBlocks parsedArgs.temperature

end Examples.TrainDiffusion

def main (args : List String) : IO Unit := Examples.TrainDiffusion.main args
