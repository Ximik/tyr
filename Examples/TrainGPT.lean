/-
  GPT Training Script

  Train a tiny GPT model on Shakespeare character data.
  Uses the same data format as nanoGPT.
-/
import Tyr
import Tyr.Checkpoint
import Examples.GPT.GPT
import Examples.GPT.Train
import Examples.GPT.GPTDataLoader

namespace Examples.TrainGPT

open torch
open torch.gpt
open torch.train
open torch.checkpoint

-- Shakespeare character vocabulary (65 chars)
-- This must match the prepare.py encoding
def shakespeareChars : String :=
  "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def charToInt (c : Char) : Int64 :=
  match shakespeareChars.toList.findIdx? (· == c) with
  | some idx => idx.toInt64
  | none => 0  -- fallback to newline

def intToChar (i : Int64) : Char :=
  shakespeareChars.toList.getD i.toUInt64.toNat '\n'

def encode (s : String) : Array Int64 :=
  s.toList.toArray.map charToInt

def decode (tokens : Array Int64) : String :=
  String.ofList (tokens.toList.map intToChar)

/-- Train with data of known size (legacy, no validation) -/
def runTraining {n : UInt64} (modelCfg : Config) (trainCfg : TrainConfig)
    (trainData : Tokens #[n]) : IO (GPTParams modelCfg) := do
  IO.println ""
  IO.println "Initializing model..."
  -- Create model directly on target device
  let params ← GPTParams.init modelCfg trainCfg.device
  -- Initialize optimizer state using Optax-style API
  let opt := Optim.adamw (lr := trainCfg.learningRate)
  let optState := opt.init params

  IO.println s!"Model initialized with {modelCfg.n_layer} layers"
  IO.println ""

  -- Train
  let finalParams ← trainLoop trainCfg params optState trainData

  IO.println ""
  return finalParams

/-- Train with validation data and checkpointing -/
def runTrainingWithVal {nTrain nVal : UInt64} (modelCfg : Config) (trainCfg : TrainConfig)
    (trainData : Tokens #[nTrain]) (valData : Tokens #[nVal])
    (checkpointDir : Option String := none) : IO (GPTParams modelCfg) := do
  IO.println ""
  IO.println "Initializing model..."
  -- Create model directly on target device
  let params ← GPTParams.init modelCfg trainCfg.device
  -- Initialize optimizer state using Optax-style API
  let opt := Optim.adamw (lr := trainCfg.learningRate)
  let optState := opt.init params

  IO.println s!"Model initialized with {modelCfg.n_layer} layers"
  IO.println ""

  -- Train with validation
  let (finalParams, bestValLoss) ← trainLoopWithVal trainCfg params optState trainData valData

  -- Save final checkpoint if directory specified
  if let some dir := checkpointDir then
    saveCheckpoint finalParams trainCfg.maxIters bestValLoss 0.0 dir

  IO.println ""
  return finalParams

def main : IO Unit := do
  IO.println "Starting..."
  IO.println "=== Tyr GPT Training ==="
  IO.println ""
  (← IO.getStdout).flush

  -- Default to CPU for stability on heterogeneous cluster nodes.
  -- Override with TYR_DEVICE=auto|cpu|cuda|mps.
  let requestedDevice? := (← IO.getEnv "TYR_DEVICE")
  let device ←
    match requestedDevice?.map String.toLower with
    | some "auto" => getBestDevice
    | some "cuda" => pure (Device.CUDA 0)
    | some "mps" => pure Device.MPS
    | some "cpu" => pure Device.CPU
    | some _ => pure Device.CPU
    | none => pure Device.CPU
  let deviceName := match device with
    | Device.MPS => "MPS (Apple Silicon)"
    | Device.CUDA n => s!"CUDA:{n}"
    | Device.CPU => "CPU"
  IO.println s!"Using device: {deviceName}"
  IO.println ""

  -- nanoGPT CPU configuration for Shakespeare character-level training
  -- https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py
  -- CPU command: python train.py config/train_shakespeare_char.py --device=cpu --compile=False
  --              --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12
  --              --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
  let modelCfg := Config.nanogpt_cpu_shakespeare
  IO.println s!"Model config: vocab={modelCfg.vocab_size}, block={modelCfg.block_size}, embd={modelCfg.n_embd}, heads={modelCfg.n_head}, layers={modelCfg.n_layer}, dropout={modelCfg.dropout}"

  let trainCfg : TrainConfig := {
    maxIters := 5000
    evalInterval := 500
    logInterval := 50
    learningRate := 1e-3
    minLr := 1e-4
    warmupIters := 100
    lrDecayIters := 2000
    gradClip := 1.0      -- Enable gradient clipping
    batchSize := 12
    blockSize := modelCfg.block_size
    device := device     -- Use selected device (MPS/CUDA/CPU)
  }

  -- Try to load Shakespeare data, fall back to random data
  IO.println ""
  IO.println "Loading data..."
  (← IO.getStdout).flush

  -- Check if training data exists
  let trainFileExists ← do
    try
      let _ ← data.binFileTokenCount "data/shakespeare_char/train.bin"
      pure true
    catch _ =>
      pure false

  -- Check if validation data exists
  let valFileExists ← do
    try
      let _ ← data.binFileTokenCount "data/shakespeare_char/val.bin"
      pure true
    catch _ =>
      pure false

  let trainedParams ← if trainFileExists then
    IO.println "Found Shakespeare data..."

    -- Load training data
    let nTrain ← data.binFileTokenCount "data/shakespeare_char/train.bin"
    IO.println s!"Training set: {nTrain} tokens"
    let trainData ← Tensor.loadU16Bin nTrain "data/shakespeare_char/train.bin"

    -- Use validation if available
    if valFileExists then
      let nVal ← data.binFileTokenCount "data/shakespeare_char/val.bin"
      IO.println s!"Validation set: {nVal} tokens"
      let valData ← Tensor.loadU16Bin nVal "data/shakespeare_char/val.bin"
      IO.println "Training with validation..."
      runTrainingWithVal modelCfg trainCfg trainData valData (some "checkpoints/gpt")
    else
      IO.println "No validation data found, training without validation..."
      runTraining modelCfg trainCfg trainData
  else
    IO.println "Shakespeare data not found, using random tokens for testing..."
    let numTokens : UInt64 := 10000
    let tokens ← Tensor.randint 0 modelCfg.vocab_size.toInt64 #[numTokens]
    runTraining modelCfg trainCfg tokens

  -- Generate some text
  IO.println ""
  IO.println "=== Generating Shakespeare ==="
  IO.println ""

  let prompt := "ROMEO:"
  let promptTokens := encode prompt
  IO.println s!"Prompt: \"{prompt}\""

  let generated ← generate trainedParams promptTokens 200 1.0
  let text := decode generated

  IO.println "Generated text:"
  IO.println "---"
  IO.println text
  IO.println "---"
  IO.println ""
  IO.println "Done!"

end Examples.TrainGPT

def main : IO Unit := Examples.TrainGPT.main
