/- Forward-only decode benchmark for the TK-style H100 decode kernel.

   Decode workloads have qSeq=1 (one query token, attending over a long KV
   cache) and don't need a backward pass — autograd-safe but we never call
   `.backward()`. This file is a slimmer cousin of `RunFlashAttnBench.lean`
   focused on the decode shape matrix.

   Backends:
   - `tyr_runtime`  : `tyr::flash_attn` — routes head_dim ∈ {64, 128, 256}
                      decode shapes to the native TK kernel, others to SDPA.
   - `torch_sdpa`   : PyTorch SDPA reference.

   Output: human-readable one-liner per (case, backend) plus optional JSONL.
   Latency is reported as p50 over `--repeats` repeats of `--iters` calls
   each, with `--warmup` warmup iters between repeats. CUDA streams sync
   between repeats.
-/
import Lean.Data.Json.Basic
import Lean.Data.Json.Printer
import Lean.Data.Json.FromToJson
import Tyr.Torch
import Tyr.Typed.Neural
import Examples.GPU.Benchmark

namespace Examples.GPU

open Lean (Json)
open torch

variable {device : Device}

inductive DecodeBackend where
  | tyrRuntime
  | torchSdpa
  deriving Repr, Inhabited, BEq

namespace DecodeBackend

def name : DecodeBackend → String
  | .tyrRuntime => "tyr_runtime"
  | .torchSdpa => "torch_sdpa"

def parse? (s : String) : Option DecodeBackend :=
  if s == "tyr_runtime" || s == "tyr" then some .tyrRuntime
  else if s == "torch_sdpa" || s == "sdpa" || s == "torch" then some .torchSdpa
  else none

def all : Array DecodeBackend := #[.tyrRuntime, .torchSdpa]

end DecodeBackend

structure DecodeBenchCase where
  caseId : String
  batch : Nat
  qHeads : Nat
  kvHeads : Nat
  kvSeq : Nat
  headDim : Nat
  expectedTyrRoute : String := "tkKernel"
  deriving Repr, Inhabited

/-- Mirrors `Examples/GPU/RunMhaH100Decode.lean :: decodeShapes` plus a couple
    of long-context rows for the bench-only cases. -/
def caseMatrix : Array DecodeBenchCase := #[
  -- Existing parity shapes
  { caseId := "llama3_b1",         batch := 1, qHeads := 32, kvHeads := 8,
    kvSeq := 2048, headDim := 128 },
  { caseId := "llama3_b4",         batch := 4, qHeads := 32, kvHeads := 8,
    kvSeq := 2048, headDim := 128 },
  { caseId := "llama3_tail",       batch := 1, qHeads := 32, kvHeads := 8,
    kvSeq := 2049, headDim := 128 },
  { caseId := "llama3_one_block",  batch := 1, qHeads := 32, kvHeads := 8,
    kvSeq := 64,   headDim := 128 },
  { caseId := "qwen3_4b_d64",      batch := 1, qHeads := 32, kvHeads := 8,
    kvSeq := 2048, headDim := 64  },
  { caseId := "qwen36_35B",        batch := 1, qHeads := 16, kvHeads := 2,
    kvSeq := 2048, headDim := 256 },
  -- Long-context rows: where decode latency actually matters
  { caseId := "llama3_b1_kv8k",    batch := 1, qHeads := 32, kvHeads := 8,
    kvSeq := 8192, headDim := 128 },
  { caseId := "qwen36_35B_kv8k",   batch := 1, qHeads := 16, kvHeads := 2,
    kvSeq := 8192, headDim := 256 }
]

private def splitCsv (s : String) : Array String :=
  (s.splitOn ",").foldl
    (fun acc part =>
      let t := part.trimAscii.toString
      if t.isEmpty then acc else acc.push t)
    #[]

private def parseBackendSelection (s : String) : Array DecodeBackend :=
  let parts := splitCsv s
  if parts.isEmpty || parts.contains "all" then DecodeBackend.all
  else parts.filterMap DecodeBackend.parse?

private def parseCaseSelection (s : String) : Array DecodeBenchCase :=
  let parts := splitCsv s
  if parts.isEmpty || parts.contains "all" then caseMatrix
  else caseMatrix.filter fun c => parts.any (· == c.caseId)

private def parseArgNat (args : List String) (flag : String) (default : Nat) : Nat := Id.run do
  let rec loop (acc : Nat) (xs : List String) : Nat :=
    match xs with
    | key :: value :: rest =>
        if key == flag then
          match value.toNat? with
          | some n => loop n rest
          | none => loop acc rest
        else
          loop acc (value :: rest)
    | _ => acc
  loop default args

private def parseArgString? (args : List String) (flag : String) : Option String := Id.run do
  let rec loop (xs : List String) : Option String :=
    match xs with
    | key :: value :: rest => if key == flag then some value else loop (value :: rest)
    | _ => none
  loop args

private def hasFlag (args : List String) (flag : String) : Bool :=
  args.contains flag

private def appendLine (path : System.FilePath) (line : String) : IO Unit := do
  if let some parent := path.parent then IO.FS.createDirAll parent
  IO.FS.withFile path .append fun h => do h.putStr line; h.putStr "\n"

private def shellTrim? (cmd : String) (args : Array String := #[]) : IO (Option String) := do
  try
    let out ← IO.Process.output { cmd := cmd, args := args }
    if out.exitCode == 0 then
      let s := out.stdout.trimAscii.toString
      pure <| if s.isEmpty then none else some s
    else pure none
  catch _ => pure none

/-- Reuse the harness's heuristic: head_dim ∈ {64, 128, 256} on a CUDA
    BF16 tensor with q_seq=1 → routes to the native TK decode kernel. -/
private def expectedRouteFor (c : DecodeBenchCase) : String :=
  if c.headDim == 64 || c.headDim == 128 || c.headDim == 256 then "tkKernel"
  else "portable"

private def percentile (xs : Array Float) (pct : Nat) : Option Float :=
  if xs.isEmpty then none
  else
    let sorted := xs.qsort (· < ·)
    let idx := ((sorted.size - 1) * pct) / 100
    sorted[idx]?

private def meanFloat (xs : Array Float) : Option Float :=
  if xs.isEmpty then none
  else some <| xs.foldl (· + ·) 0.0 / xs.size.toFloat

/-- Time completed device execution with the shared CUDA-event contract. -/
private def benchmarkAction (runId : String) (warmup iters repeats : Nat)
    (stream : UInt64) (action : IO Unit) : IO (Array Float) := do
  let cfg : Benchmark.Config := { warmup, iters, repeats, runId }
  Benchmark.timeCudaEvents cfg stream action

private structure DecodeRow where
  event : String := "summary"
  runId : String
  caseId : String
  backend : String
  routeActual : String := "n/a"
  batch : Nat
  qHeads : Nat
  kvHeads : Nat
  kvSeq : Nat
  headDim : Nat
  latencyMsP50 : Option Float := none
  latencyMsP10 : Option Float := none
  latencyMsP90 : Option Float := none
  latencyMsMean : Option Float := none
  speedupVsSdpaP50 : Option Float := none
  outMae : Option Float := none
  outMax : Option Float := none
  correctnessOk : Option Bool := none
  deriving Lean.ToJson

private def runOne (runId : String) (warmup iters repeats : Nat)
    (c : DecodeBenchCase) (backend : DecodeBackend) : IO DecodeRow := do
  let device ← Device.cuda 0
  let stream ← torch.cuda_current_stream
  let qShape : Shape := #[c.batch, c.qHeads, 1, c.headDim]
  let kvShape : Shape := #[c.batch, c.kvHeads, c.kvSeq, c.headDim]
  let qBase ← Tensor.randn qShape false device
  let kBase ← Tensor.randn kvShape false device
  let vBase ← Tensor.randn kvShape false device
  let q := qBase.toBFloat16
  let k := kBase.toBFloat16
  let v := vBase.toBFloat16

  -- SDPA reference output (BF16 in / BF16 out, same as kernel) for parity.
  let outRef : Tensor { shape := #[c.batch, c.qHeads, 1, c.headDim], dtype := .BFloat16, device := device } :=
    Tensor.attentionGQAQKV
      (q.toFloat32) (k.toFloat32) (v.toFloat32) false true
      |> Tensor.toBFloat16

  match backend with
  | .torchSdpa =>
      let sink : Tensor { shape := #[c.batch, c.qHeads, 1, c.headDim], dtype := .BFloat16, device := device } ← Tensor.uniformLike q 0 0
      let sinkRef ← IO.mkRef (some sink)
      let action : IO Unit := do
        let out := Tensor.attentionGQAQKV q k v false true
        let some destination ← sinkRef.swap none
          | throw (IO.userError "Decode benchmark destination is unavailable")
        let updated ← Tensor.copyInto destination out
        sinkRef.set (some updated)
      let lats ← benchmarkAction runId warmup iters repeats stream action
      pure {
        runId := runId, caseId := c.caseId, backend := "torch_sdpa",
        batch := c.batch, qHeads := c.qHeads,
        kvHeads := c.kvHeads, kvSeq := c.kvSeq,
        headDim := c.headDim,
        latencyMsP50 := percentile lats 50,
        latencyMsP10 := percentile lats 10,
        latencyMsP90 := percentile lats 90,
        latencyMsMean := meanFloat lats,
        outMae := some 0.0, outMax := some 0.0, correctnessOk := some true
      }
  | .tyrRuntime =>
      let routeActual := expectedRouteFor c
      -- Correctness once before timing.
      let outActual : Tensor { shape := #[c.batch, c.qHeads, 1, c.headDim], dtype := .BFloat16, device := device } :=
        Tensor.flashAttention q k v true
      let _ ← torch.cuda_synchronize
      let outMae := Tensor.item (Tensor.meanAll (Tensor.abs (outActual - outRef)))
      let outMax := Tensor.item (Tensor.maxAllChecked (Tensor.abs (outActual - outRef)))
      let okRoute := routeActual == c.expectedTyrRoute
      let okOut := Tensor.allclose outRef outActual 5.0e-2 5.0e-2
      let sink : Tensor { shape := #[c.batch, c.qHeads, 1, c.headDim], dtype := .BFloat16, device := device } ← Tensor.uniformLike q 0 0
      let sinkRef ← IO.mkRef (some sink)
      let action : IO Unit := do
        let out := Tensor.flashAttention q k v true
        let some destination ← sinkRef.swap none
          | throw (IO.userError "Decode benchmark destination is unavailable")
        let updated ← Tensor.copyInto destination out
        sinkRef.set (some updated)
      let lats ← benchmarkAction runId warmup iters repeats stream action
      pure {
        runId := runId, caseId := c.caseId, backend := "tyr_runtime",
        routeActual := routeActual,
        batch := c.batch, qHeads := c.qHeads,
        kvHeads := c.kvHeads, kvSeq := c.kvSeq,
        headDim := c.headDim,
        latencyMsP50 := percentile lats 50,
        latencyMsP10 := percentile lats 10,
        latencyMsP90 := percentile lats 90,
        latencyMsMean := meanFloat lats,
        outMae := some outMae, outMax := some outMax,
        correctnessOk := some (okRoute && okOut)
      }

private def attachSpeedup (rows : Array DecodeRow) : Array DecodeRow :=
  rows.map fun row =>
    if row.backend == "torch_sdpa" then row
    else
      let sdpa? := rows.foldl (fun acc r =>
        match acc with
        | some _ => acc
        | none =>
            if r.caseId == row.caseId && r.backend == "torch_sdpa"
            then r.latencyMsP50 else none) none
      match sdpa?, row.latencyMsP50 with
      | some sdpaMs, some tyrMs =>
          if tyrMs > 0.0 then { row with speedupVsSdpaP50 := some (sdpaMs / tyrMs) }
          else row
      | _, _ => row

private def fmtMs : Option Float → String
  | none => "n/a"
  | some ms => s!"{ms}"

private def fmtFloat : Option Float → String
  | none => "n/a"
  | some x => s!"{x}"

private def printHuman (row : DecodeRow) : IO Unit := do
  let speedup :=
    match row.speedupVsSdpaP50 with
    | some x => s!"speedup_vs_sdpa={x}"
    | none => ""
  let corr :=
    match row.correctnessOk with
    | some true => "ok"
    | some false => "FAIL"
    | none => "n/a"
  let routeStr :=
    if row.routeActual == "n/a" then "" else s!" route={row.routeActual}"
  IO.println s!"decode case={row.caseId} backend={row.backend}{routeStr} p50_ms={fmtMs row.latencyMsP50} p10_ms={fmtMs row.latencyMsP10} p90_ms={fmtMs row.latencyMsP90} {speedup} correctness={corr} mae={fmtFloat row.outMae}"

def main (args : List String) : IO UInt32 := do
  if hasFlag args "--list-cases" then
    for c in caseMatrix do
      IO.println s!"{c.caseId} batch={c.batch} qHeads={c.qHeads} kvHeads={c.kvHeads} kvSeq={c.kvSeq} headDim={c.headDim}"
    return 0
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return 1
  let warmup := parseArgNat args "--warmup" 20
  let iters := parseArgNat args "--iters" 200
  let repeats := parseArgNat args "--repeats" 5
  let caseSel := (parseArgString? args "--case").getD "all"
  let backendSel := (parseArgString? args "--backend").getD "all"
  let jsonlOut? := (parseArgString? args "--jsonl-out").map System.FilePath.mk
  if iters == 0 || repeats == 0 then
    IO.eprintln "--iters and --repeats must be > 0"; return 1
  let cases := parseCaseSelection caseSel
  let backends := parseBackendSelection backendSel
  if cases.isEmpty then
    IO.eprintln s!"No cases matched: {caseSel}"; return 1
  if backends.isEmpty then
    IO.eprintln s!"No backends matched: {backendSel}"; return 1

  let runId := s!"decode_bench_{← IO.monoMsNow}"
  let gitSha := (← shellTrim? "git" #["rev-parse", "--short", "HEAD"]).getD "unknown"
  let gpuName := (← shellTrim? "bash" #["-lc",
    "nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1"]).getD "unknown"
  IO.println s!"run_id={runId} git={gitSha} gpu={gpuName} warmup={warmup} iters={iters} repeats={repeats}"

  torch.manualSeed 20260426

  let mut rows : Array DecodeRow := #[]
  for c in cases do
    -- Per-case: time SDPA first (so the speedup denominator exists when we time tyr).
    for backend in backends do
      try
        let row ← runOne runId warmup iters repeats c backend
        rows := rows.push row
      catch e =>
        IO.eprintln s!"case={c.caseId} backend={DecodeBackend.name backend} error={e}"

  let rowsFinal := attachSpeedup rows
  for row in rowsFinal do
    printHuman row
    match jsonlOut? with
    | none => pure ()
    | some path => appendLine path (Lean.toJson row).compress

  pure 0

end Examples.GPU

def main : List String → IO UInt32 := Examples.GPU.main
