/-
  Examples/ModelRunner.lean

  Shared CLI parsing, batching, decoding, and streaming utilities for
  model RunHF examples.
-/
import Tyr.Torch
import Tyr.Hub
import Tyr.Model.Generation
import Tyr.TensorStruct

open torch
open torch.Model

namespace Examples.ModelRunner

/-! ## CLI parsing -/

def parseNatArg (name : String) (v : String) : IO UInt64 := do
  match v.toNat? with
  | some n => pure n.toUInt64
  | none => throw <| IO.userError s!"Invalid {name}: {v}"

/-! ## Device resolution -/

def deviceToString : Device → String
  | Device.MPS => "MPS"
  | Device.CPU => "CPU"
  | Device.CUDA n => s!"CUDA:{n}"

def resolveDevice (arg : String) : IO (Device × Option String) := do
  let requested := arg.trimAscii.toString.toLower
  match requested with
  | "auto" => return (← getBestDevice, none)
  | "cpu" => pure (.CPU, none)
  | "mps" => pure (.MPS, none)
  | "cuda" =>
      if ← cuda_is_available then
        pure (.CUDA 0, none)
      else
        pure (.CPU, some "Warning: --device cuda requested but CUDA is unavailable; falling back to CPU.")
  | _ =>
      if requested.startsWith "cuda:" then
        match (requested.drop 5).toNat? with
        | some idx =>
            if ← cuda_is_available then
              pure (.CUDA idx.toUInt64, none)
            else
              pure (.CPU, some s!"Warning: --device cuda:{idx} requested but CUDA is unavailable; falling back to CPU.")
        | none => pure (.CPU, some s!"Warning: invalid device selector '{arg}'; falling back to CPU.")
      else
        pure (.CPU, some s!"Warning: invalid device selector '{arg}'; falling back to CPU.")

/-! ## Prompt loading -/

def loadPrompts (promptFile : Option String) (defaultPrompt : String) : IO (Array String) := do
  match promptFile with
  | some path =>
    let contents ← IO.FS.readFile path
    let lines := contents.splitOn "\n"
    let prompts := lines.foldl
      (init := #[])
      (fun acc line =>
        let s := line.trimAscii.toString
        if s.isEmpty then acc else acc.push s)
    if prompts.isEmpty then
      throw <| IO.userError s!"No prompts found in {path}"
    pure prompts
  | none =>
    pure #[defaultPrompt]

/-! ## Model device movement -/

def moveModelToDevice [TensorStruct α] (device : Device) (x : α) : IO α :=
  TensorStruct.mapM (fun t => pure (t.to device)) x

def moveSigmaTensorTo {d : UInt64} {dtype : DType} (device : Device)
    (x : Sigma (fun n => DTensor #[n, d] dtype)) : Sigma (fun n => DTensor #[n, d] dtype) :=
  match x with
  | ⟨n, t⟩ => ⟨n, t.to device⟩

/-! ## Tensor batching -/

def buildBatchInputWithEncoder
    (padToken : UInt64)
    (prompts : Array String)
    (encode : String → IO (Array UInt64))
    : IO (Sigma (fun batch => Sigma (fun seq => DTensor #[batch, seq] .Int64 × Array Nat))) := do
  let mut encoded : Array (Array UInt64) := #[]
  for prompt in prompts do
    encoded := encoded.push (← encode prompt)
  let batch := encoded.size.toUInt64
  if batch == 0 then
    throw <| IO.userError "buildBatchInputWithEncoder requires at least one prompt"

  let maxLenNat := encoded.foldl (fun m ids => Nat.max m ids.size) 0
  if maxLenNat == 0 then
    throw <| IO.userError "Prompt tokenization produced empty input."
  let seq := maxLenNat.toUInt64

  let mut flat : Array Int64 := #[]
  let mut promptLens : Array Nat := #[]
  for ids in encoded do
    promptLens := promptLens.push ids.size
    let mut row : Array Int64 := ids.map (fun x => x.toInt64)
    while row.size < maxLenNat do
      row := row.push padToken.toInt64
    flat := flat ++ row

  let inputIds : DTensor #[batch, seq] .Int64 := Tensor.reshapeChecked (Tensor.fromInt64Array flat) #[batch, seq]
  pure ⟨batch, ⟨seq, (inputIds, promptLens)⟩⟩

def buildBatchInput
    (padToken : UInt64)
    (prompts : Array String)
    (encode : String → Array UInt64)
    : IO (Sigma (fun batch => Sigma (fun seq => DTensor #[batch, seq] .Int64 × Array Nat))) := do
  buildBatchInputWithEncoder padToken prompts (fun s => pure (encode s))

/-! ## Generation output formatting -/

def generatedIdsFromBatch
    (promptLens : Array Nat)
    {batch outSeq : UInt64}
    (ids : DTensor #[batch, outSeq] .Int64)
    : IO (Array (Array UInt64)) := do
  let mut out : Array (Array UInt64) := #[]
  for i in [:batch.toNat] do
    let row2 : DTensor #[1, outSeq] .Int64 := ids.sliceChecked 0 i.toUInt64 1 #[1, outSeq]
    let row1 : DTensor #[outSeq] .Int64 := Tensor.reshapeChecked row2 #[outSeq]
    let vals ← Tensor.toUInt64Array row1
    let promptLen := promptLens.getD i 0
    let gen :=
      if vals.size <= promptLen then
        #[]
      else
        vals.extract promptLen vals.size
    out := out.push gen
  pure out

def printDecodedBatch
    (chunkStart : Nat)
    (decoded : Array String)
    (singleOnly : Bool := false)
    : IO Unit := do
  if singleOnly && decoded.size == 1 && chunkStart == 0 then
    IO.println "GEN_BEGIN"
    IO.println decoded[0]!
    IO.println "GEN_END"
  else
    for i in [:decoded.size] do
      let idx := chunkStart + i
      IO.println s!"GEN[{idx}]_BEGIN"
      IO.println decoded[i]!
      IO.println s!"GEN[{idx}]_END"

def printGeneratedIds
    (chunkStart : Nat)
    (generatedIds : Array (Array UInt64))
    (singleOnly : Bool := false)
    : IO Unit := do
  if singleOnly && generatedIds.size == 1 && chunkStart == 0 then
    IO.println s!"GEN_IDS={generatedIds[0]!}"
  else
    for i in [:generatedIds.size] do
      let idx := chunkStart + i
      IO.println s!"GEN_IDS[{idx}]={generatedIds[i]!}"

def decodeGeneratedBatch
    (promptLens : Array Nat)
    {batch outSeq : UInt64}
    (ids : DTensor #[batch, outSeq] .Int64)
    (decodeText : Array UInt32 → String)
    : IO (Array String) := do
  let generatedIds ← generatedIdsFromBatch promptLens ids
  pure <| generatedIds.map (fun xs => decodeText (xs.map (fun x => x.toUInt32)))

/-! ## Streaming callback -/

def makeStreamCallback
    (decodeOne : UInt32 → String)
    {batch : UInt64}
    (chunkStart : Nat)
    : StreamCallback batch := fun _step nextTok => do
  let flat : DTensor #[batch] .Int64 := Tensor.reshapeChecked nextTok #[batch]
  let vals ← Tensor.toUInt64Array flat
  if batch == 1 then
    match vals[0]? with
    | some v => IO.print (decodeOne v.toUInt32)
    | none => pure ()
  else
    for i in [:vals.size] do
      let idx := chunkStart + i
      IO.println s!"STREAM[{idx}] {decodeOne vals[i]!.toUInt32}"

/-! ## Runner orchestration -/

def runGenerationBatches
    (prompts : Array String)
    (batchSize : UInt64)
    (_maxNewTokens : UInt64)
    (stream : Bool)
    (debugIds : Bool)
    (singleOnly : Bool)
    (buildBatch : Array String → IO (Sigma (fun b => Sigma (fun s => DTensor #[b, s] .Int64 × Array Nat))))
    (runGenerate : (b : UInt64) → {s : UInt64} → DTensor #[b, s] .Int64 → IO (Sigma (fun os => DTensor #[b, os] .Int64)))
    (runGenerateStream : (b : UInt64) → {s : UInt64} → DTensor #[b, s] .Int64 → StreamCallback b → IO (Sigma (fun os => DTensor #[b, os] .Int64)))
    (decodeBatch : Array Nat → {b os : UInt64} → DTensor #[b, os] .Int64 → IO (Array String))
    (makeStreamCallback : {b : UInt64} → Nat → StreamCallback b)
    : IO Unit := do
  let chunkSize := Nat.max 1 batchSize.toNat
  let mut start : Nat := 0
  while start < prompts.size do
    let stop := Nat.min prompts.size (start + chunkSize)
    let chunk := prompts.extract start stop
    let ⟨batch, ⟨_seq, (inputIds, promptLens)⟩⟩ ← buildBatch chunk
    let ⟨_outSeq, outIds⟩ ←
      if stream then
        runGenerateStream batch inputIds (makeStreamCallback start)
      else
        runGenerate batch inputIds
    if stream && chunk.size == 1 then
      IO.println ""
    if !stream then
      let decoded ← decodeBatch promptLens outIds
      if debugIds then
        let generatedIds ← generatedIdsFromBatch promptLens outIds
        printGeneratedIds start generatedIds (singleOnly := singleOnly)
      printDecodedBatch start decoded (singleOnly := singleOnly)
    start := stop

/-- Alias for text-only generation batches. -/
abbrev runTextBatches := @runGenerationBatches

/-- Alias for multimodal generation batches. -/
abbrev runMultimodalBatches := @runGenerationBatches

/-- Greedy generation without streaming support (for models that only expose generateGreedy). -/
def runGreedyBatches
    (prompts : Array String)
    (batchSize : UInt64)
    (_maxNewTokens : UInt64)
    (singleOnly : Bool)
    (buildBatch : Array String → IO (Sigma (fun b => Sigma (fun s => DTensor #[b, s] .Int64 × Array Nat))))
    (runGenerate : (b : UInt64) → {s : UInt64} → DTensor #[b, s] .Int64 → IO (Sigma (fun os => DTensor #[b, os] .Int64)))
    (decodeBatch : Array Nat → {b os : UInt64} → DTensor #[b, os] .Int64 → IO (Array String))
    : IO Unit := do
  let chunkSize := Nat.max 1 batchSize.toNat
  let mut start : Nat := 0
  while start < prompts.size do
    let stop := Nat.min prompts.size (start + chunkSize)
    let chunk := prompts.extract start stop
    let ⟨_batch, ⟨_seq, (inputIds, promptLens)⟩⟩ ← buildBatch chunk
    let ⟨_outSeq, outIds⟩ ← runGenerate _batch inputIds
    let decoded ← decodeBatch promptLens outIds
    printDecodedBatch start decoded (singleOnly := singleOnly)
    start := stop

end Examples.ModelRunner
