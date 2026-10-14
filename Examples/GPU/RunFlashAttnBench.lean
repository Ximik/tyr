/- One-H100 benchmark scaffold for the current `tyr::flash_attn` bring-up.

   The current executable focuses on:

   - a fixed case matrix covering native-now and portable-control shapes,
   - structured JSONL output suitable for later performance claims,
   - a stable backend naming scheme that already includes a FlashAttention slot.

   Today, the runnable backends are:

   - `tyr_runtime`  : `tyr::flash_attn` (native TK route or portable fallback)
   - `torch_sdpa`   : PyTorch SDPA reference
   - `flash_attention` : reserved baseline slot, currently reported as
       `unsupported` until the repo-local FA3 generated CUDA matches the
       vendored ThunderKittens TMA surface.
-/
import Lean.Data.Json.Basic
import Lean.Data.Json.Printer
import Lean.Data.Json.FromToJson
import Tyr.Torch
import Examples.GPU.FlashBenchReference
import Tyr.GPU.Ops.FlashAttn

namespace Examples.GPU

open Lean (Json)
open torch
open Examples.GPU.FlashBenchReference
open Tyr.GPU.Ops.FlashAttn

inductive Backend where
  | tyrRuntime
  | torchSdpa
  | flashAttention
  deriving Repr, Inhabited, BEq

namespace Backend

def name : Backend → String
  | .tyrRuntime => "tyr_runtime"
  | .torchSdpa => "torch_sdpa"
  | .flashAttention => "flash_attention"

def parse? (s : String) : Option Backend :=
  if s == "tyr_runtime" || s == "tyr" then
    some .tyrRuntime
  else if s == "torch_sdpa" || s == "sdpa" || s == "torch" then
    some .torchSdpa
  else if s == "flash_attention" || s == "flash" || s == "fa" then
    some .flashAttention
  else
    none

def all : Array Backend :=
  #[.tyrRuntime, .torchSdpa, .flashAttention]

end Backend

structure BenchCase where
  caseId : String
  caseGroup : String
  mode : String
  batch : UInt64
  qHeads : UInt64
  kvHeads : UInt64
  qSeq : UInt64
  kvSeq : UInt64
  headDim : UInt64
  dtype : String := "bf16"
  isCausal : Bool := false
  enableGqa : Bool := false
  expectedTyrRoute : String := "portable"
  deriving Repr, Inhabited

def caseMatrix : Array BenchCase := #[
  {
    caseId := "native_dense_128x64"
    caseGroup := "native_now"
    mode := "forward_backward"
    batch := 1
    qHeads := 1
    kvHeads := 1
    qSeq := 128
    kvSeq := 128
    headDim := 64
    expectedTyrRoute := "tkKernel"
  },
  {
    caseId := "native_dense_768x64"
    caseGroup := "native_now"
    mode := "forward_backward"
    batch := 1
    qHeads := 1
    kvHeads := 1
    qSeq := 768
    kvSeq := 768
    headDim := 64
    expectedTyrRoute := "tkKernel"
  },
  {
    caseId := "portable_dense_96x64"
    caseGroup := "portable_controls"
    mode := "forward_backward"
    batch := 1
    qHeads := 1
    kvHeads := 1
    qSeq := 96
    kvSeq := 96
    headDim := 64
    expectedTyrRoute := "portable"
  },
  {
    caseId := "portable_causal_128x64"
    caseGroup := "portable_controls"
    mode := "forward_backward"
    batch := 1
    qHeads := 1
    kvHeads := 1
    qSeq := 128
    kvSeq := 128
    headDim := 64
    isCausal := true
    expectedTyrRoute := "portable"
  },
  {
    caseId := "portable_gqa_96x64"
    caseGroup := "portable_controls"
    mode := "forward_backward"
    batch := 1
    qHeads := 4
    kvHeads := 2
    qSeq := 96
    kvSeq := 96
    headDim := 64
    enableGqa := true
    expectedTyrRoute := "portable"
  },
  {
    caseId := "future_flash_256x64"
    caseGroup := "flash_aux"
    mode := "forward_only"
    batch := 1
    qHeads := 1
    kvHeads := 1
    qSeq := 256
    kvSeq := 256
    headDim := 64
    expectedTyrRoute := "portable"
  }
]

private def caseSelectionMatches (selector : String) (c : BenchCase) : Bool :=
  selector == "all" || selector == c.caseGroup || selector == c.caseId

private def splitCsv (s : String) : Array String :=
  (s.splitOn ",").foldl
    (fun acc part =>
      let t := part.trimAscii.toString
      if t.isEmpty then acc else acc.push t)
    #[]

private def parseBackendSelection (s : String) : Array Backend :=
  let parts := splitCsv s
  if parts.isEmpty || parts.contains "all" then
    Backend.all
  else
    parts.filterMap Backend.parse?

private def parseCaseSelection (s : String) : Array BenchCase :=
  let parts := splitCsv s
  if parts.isEmpty || parts.contains "all" then
    caseMatrix
  else
    caseMatrix.filter fun c => parts.any (fun part => caseSelectionMatches part c)

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
    | key :: value :: rest =>
        if key == flag then some value else loop (value :: rest)
    | _ => none
  loop args

private def hasFlag (args : List String) (flag : String) : Bool :=
  args.contains flag

private def appendLine (path : System.FilePath) (line : String) : IO Unit := do
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  IO.FS.withFile path .append fun h => do
    h.putStr line
    h.putStr "\n"

private def shellTrim? (cmd : String) (args : Array String := #[]) : IO (Option String) := do
  try
    let out ← IO.Process.output { cmd := cmd, args := args }
    if out.exitCode == 0 then
      let s := out.stdout.trimAscii.toString
      pure <| if s.isEmpty then none else some s
    else
      pure none
  catch _ =>
    pure none

private def routeName : DispatchRoute → String
  | .tkKernel => "tkKernel"
  | .portable => "portable"

private def meanFloat (xs : Array Float) : Option Float :=
  if xs.isEmpty then
    none
  else
    some <| xs.foldl (· + ·) 0.0 / xs.size.toFloat

private def stdFloat (xs : Array Float) : Option Float :=
  match meanFloat xs with
  | none => none
  | some μ =>
      if xs.size < 2 then
        some 0.0
      else
        let var := xs.foldl (fun acc x => acc + ((x - μ) * (x - μ))) 0.0 / xs.size.toFloat
        some (Float.sqrt var)

private def percentile (xs : Array Float) (pct : Nat) : Option Float :=
  if xs.isEmpty then
    none
  else
    let sorted := xs.qsort (· < ·)
    let idx := ((sorted.size - 1) * pct) / 100
    sorted[idx]?

private def qTokensPerSec (c : BenchCase) (latencyMs : Float) : Float :=
  if latencyMs <= 0.0 then
    0.0
  else
    (c.batch.toFloat * c.qSeq.toFloat) / (latencyMs / 1000.0)

private structure Correctness where
  routeActual : String := "n/a"
  routeOk : Option Bool := none
  outOk : Option Bool := none
  dqOk : Option Bool := none
  dkOk : Option Bool := none
  dvOk : Option Bool := none
  correctnessOk : Option Bool := none
  outMae : Option Float := none
  dqMae : Option Float := none
  dkMae : Option Float := none
  dvMae : Option Float := none
  outMax : Option Float := none
  dqMax : Option Float := none
  dkMax : Option Float := none
  dvMax : Option Float := none
  outRelL2 : Option Float := none
  dqRelL2 : Option Float := none
  dkRelL2 : Option Float := none
  dvRelL2 : Option Float := none
  deriving Inhabited, Repr

private structure RepeatRow where
  event : String := "repeat"
  runId : String
  caseId : String
  caseGroup : String
  backendRequested : String
  backendExecuted : String
  repeatIdx : Nat
  latencyMs : Float
  deriving Lean.ToJson

private structure SummaryRow where
  event : String := "summary"
  runId : String
  gitSha : String
  hostname : String
  gpuName : String
  cudaVisibleDevices : String
  warmupIters : Nat
  timedIters : Nat
  repeats : Nat
  seed : Nat

  caseId : String
  caseGroup : String
  mode : String
  batch : Nat
  qHeads : Nat
  kvHeads : Nat
  gqaRatio : Nat
  qSeq : Nat
  kvSeq : Nat
  headDim : Nat
  dtype : String
  isCausal : Bool
  enableGqa : Bool
  expectedTyrRoute : String

  backendRequested : String
  backendExecuted : String
  supportStatus : String
  supportReason : Option String := none
  routeActual : String := "n/a"
  routeOk : Option Bool := none

  latencyMsP10 : Option Float := none
  latencyMsP50 : Option Float := none
  latencyMsP90 : Option Float := none
  latencyMsMean : Option Float := none
  latencyMsStd : Option Float := none
  qTokensPerSec : Option Float := none
  speedupVsSdpaP50 : Option Float := none

  outOk : Option Bool := none
  dqOk : Option Bool := none
  dkOk : Option Bool := none
  dvOk : Option Bool := none
  correctnessOk : Option Bool := none

  outMae : Option Float := none
  dqMae : Option Float := none
  dkMae : Option Float := none
  dvMae : Option Float := none
  outMax : Option Float := none
  dqMax : Option Float := none
  dkMax : Option Float := none
  dvMax : Option Float := none
  outRelL2 : Option Float := none
  dqRelL2 : Option Float := none
  dkRelL2 : Option Float := none
  dvRelL2 : Option Float := none
  deriving Lean.ToJson

private structure MetaRow where
  event : String := "meta"
  runId : String
  gitSha : String
  hostname : String
  gpuName : String
  cudaVisibleDevices : String
  warmupIters : Nat
  timedIters : Nat
  repeats : Nat
  seed : Nat
  selectedCases : Array String
  selectedBackends : Array String
  deriving Lean.ToJson

private structure RunMeta where
  runId : String
  gitSha : String
  hostname : String
  gpuName : String
  cudaVisibleDevices : String
  warmupIters : Nat
  timedIters : Nat
  repeats : Nat
  seed : UInt64
  deriving Inhabited

private def computeGqaRatio (qHeads kvHeads : UInt64) : Nat :=
  if kvHeads == 0 then 0 else (qHeads / kvHeads).toNat

private def emitJsonl?
    (jsonlOut? : Option System.FilePath)
    (emitStdout : Bool)
    (payload : Json) : IO Unit := do
  let line := payload.compress
  if emitStdout then
    IO.println line
  match jsonlOut? with
  | none => pure ()
  | some path => appendLine path line

private def mkMetaRow (runMeta : RunMeta) (cases : Array BenchCase) (backends : Array Backend) : MetaRow :=
  {
    runId := runMeta.runId
    gitSha := runMeta.gitSha
    hostname := runMeta.hostname
    gpuName := runMeta.gpuName
    cudaVisibleDevices := runMeta.cudaVisibleDevices
    warmupIters := runMeta.warmupIters
    timedIters := runMeta.timedIters
    repeats := runMeta.repeats
    seed := runMeta.seed.toNat
    selectedCases := cases.map (·.caseId)
    selectedBackends := backends.map Backend.name
  }

private def mkUnsupportedSummary
    (runMeta : RunMeta)
    (c : BenchCase)
    (backend : Backend)
    (reason : String)
    : SummaryRow :=
  {
    runId := runMeta.runId
    gitSha := runMeta.gitSha
    hostname := runMeta.hostname
    gpuName := runMeta.gpuName
    cudaVisibleDevices := runMeta.cudaVisibleDevices
    warmupIters := runMeta.warmupIters
    timedIters := runMeta.timedIters
    repeats := runMeta.repeats
    seed := runMeta.seed.toNat
    caseId := c.caseId
    caseGroup := c.caseGroup
    mode := c.mode
    batch := c.batch.toNat
    qHeads := c.qHeads.toNat
    kvHeads := c.kvHeads.toNat
    gqaRatio := computeGqaRatio c.qHeads c.kvHeads
    qSeq := c.qSeq.toNat
    kvSeq := c.kvSeq.toNat
    headDim := c.headDim.toNat
    dtype := c.dtype
    isCausal := c.isCausal
    enableGqa := c.enableGqa
    expectedTyrRoute := c.expectedTyrRoute
    backendRequested := Backend.name backend
    backendExecuted := Backend.name backend
    supportStatus := "unsupported"
    supportReason := some reason
  }

private def mkErrorSummary
    (runMeta : RunMeta)
    (c : BenchCase)
    (backend : Backend)
    (reason : String)
    : SummaryRow :=
  {
    runId := runMeta.runId
    gitSha := runMeta.gitSha
    hostname := runMeta.hostname
    gpuName := runMeta.gpuName
    cudaVisibleDevices := runMeta.cudaVisibleDevices
    warmupIters := runMeta.warmupIters
    timedIters := runMeta.timedIters
    repeats := runMeta.repeats
    seed := runMeta.seed.toNat
    caseId := c.caseId
    caseGroup := c.caseGroup
    mode := c.mode
    batch := c.batch.toNat
    qHeads := c.qHeads.toNat
    kvHeads := c.kvHeads.toNat
    gqaRatio := computeGqaRatio c.qHeads c.kvHeads
    qSeq := c.qSeq.toNat
    kvSeq := c.kvSeq.toNat
    headDim := c.headDim.toNat
    dtype := c.dtype
    isCausal := c.isCausal
    enableGqa := c.enableGqa
    expectedTyrRoute := c.expectedTyrRoute
    backendRequested := Backend.name backend
    backendExecuted := Backend.name backend
    supportStatus := "error"
    supportReason := some reason
  }

private def makeSummary
    (runMeta : RunMeta)
    (c : BenchCase)
    (backend : Backend)
    (backendExecuted supportStatus : String)
    (latencies : Array Float)
    (corr : Correctness)
    : SummaryRow :=
  {
    runId := runMeta.runId
    gitSha := runMeta.gitSha
    hostname := runMeta.hostname
    gpuName := runMeta.gpuName
    cudaVisibleDevices := runMeta.cudaVisibleDevices
    warmupIters := runMeta.warmupIters
    timedIters := runMeta.timedIters
    repeats := runMeta.repeats
    seed := runMeta.seed.toNat
    caseId := c.caseId
    caseGroup := c.caseGroup
    mode := c.mode
    batch := c.batch.toNat
    qHeads := c.qHeads.toNat
    kvHeads := c.kvHeads.toNat
    gqaRatio := computeGqaRatio c.qHeads c.kvHeads
    qSeq := c.qSeq.toNat
    kvSeq := c.kvSeq.toNat
    headDim := c.headDim.toNat
    dtype := c.dtype
    isCausal := c.isCausal
    enableGqa := c.enableGqa
    expectedTyrRoute := c.expectedTyrRoute
    backendRequested := Backend.name backend
    backendExecuted := backendExecuted
    supportStatus := supportStatus
    routeActual := corr.routeActual
    routeOk := corr.routeOk
    latencyMsP10 := percentile latencies 10
    latencyMsP50 := percentile latencies 50
    latencyMsP90 := percentile latencies 90
    latencyMsMean := meanFloat latencies
    latencyMsStd := stdFloat latencies
    qTokensPerSec := (percentile latencies 50).map (qTokensPerSec c)
    outOk := corr.outOk
    dqOk := corr.dqOk
    dkOk := corr.dkOk
    dvOk := corr.dvOk
    correctnessOk := corr.correctnessOk
    outMae := corr.outMae
    dqMae := corr.dqMae
    dkMae := corr.dkMae
    dvMae := corr.dvMae
    outMax := corr.outMax
    dqMax := corr.dqMax
    dkMax := corr.dkMax
    dvMax := corr.dvMax
    outRelL2 := corr.outRelL2
    dqRelL2 := corr.dqRelL2
    dkRelL2 := corr.dkRelL2
    dvRelL2 := corr.dvRelL2
  }

private def printHumanSummary (row : SummaryRow) : IO Unit := do
  let latencyStr :=
    match row.latencyMsP50 with
    | some ms => s!"p50_ms={ms}"
    | none => "p50_ms=n/a"
  let speedupStr :=
    match row.speedupVsSdpaP50 with
    | some x => s!"speedup_vs_sdpa={x}"
    | none => "speedup_vs_sdpa=n/a"
  let correctnessStr :=
    match row.correctnessOk with
    | some ok => s!"correctness_ok={ok}"
    | none => "correctness_ok=n/a"
  let routeStr :=
    if row.routeActual == "n/a" then
      "route=n/a"
    else
      s!"route={row.routeActual}"
  let reasonStr :=
    match row.supportReason with
    | some reason => s!" reason={reason}"
    | none => ""
  IO.println
    s!"bench case={row.caseId} backend={row.backendExecuted} status={row.supportStatus} {routeStr} {correctnessStr} {latencyStr} {speedupStr}{reasonStr}"

private def flashAttentionUnsupportedReason : String :=
  "repo-local FA3 baseline is disabled: generated FlashAttn3 CUDA requires kittens::tma::fence_view_async_shared, which is absent in the vendored ThunderKittens checkout"

private def benchmarkAction (warmup iters repeats : Nat) (action : IO Unit) : IO (Array Float) := do
  let mut latencies : Array Float := #[]
  for _ in [:repeats] do
    for _ in [:warmup] do
      action
    let _ ← torch.cuda_synchronize
    let t0 ← IO.monoNanosNow
    for _ in [:iters] do
      action
    let _ ← torch.cuda_synchronize
    let t1 ← IO.monoNanosNow
    let avgMs := (t1 - t0).toFloat / 1000000.0 / iters.toFloat
    latencies := latencies.push avgMs
  pure latencies

private def sdpaEqualRef
    {batch heads seq headDim : UInt64}
    (qBase kBase vBase dOBase : Tensor { shape := #[batch, heads, seq, headDim], dtype := .BFloat16 })
    (isCausal : Bool)
    : IO (Tensor { shape := #[batch, heads, seq, headDim], dtype := .BFloat16 } × Tensor { shape := #[batch, heads, seq, headDim], dtype := .Float32 } × Tensor { shape := #[batch, heads, seq, headDim], dtype := .Float32 } × Tensor { shape := #[batch, heads, seq, headDim], dtype := .Float32 }) := do
  let q ← makeLeaf qBase
  let k ← makeLeaf kBase
  let v ← makeLeaf vBase
  let out := Tensor.attentionGQA q k v 0.0 isCausal
  backwardWithCotangent out dOBase
  let _ ← torch.cuda_synchronize
  pure (out, (← Tensor.gradOf q).toFloat32, (← Tensor.gradOf k).toFloat32, (← Tensor.gradOf v).toFloat32)

private def sdpaGqaRef
    {batch qHeads kvHeads seq headDim : UInt64}
    (qBase : Tensor { shape := #[batch, qHeads, seq, headDim], dtype := .BFloat16 })
    (kBase vBase : Tensor { shape := #[batch, kvHeads, seq, headDim], dtype := .BFloat16 })
    (dOBase : Tensor { shape := #[batch, qHeads, seq, headDim], dtype := .BFloat16 })
    (isCausal : Bool)
    : IO (Tensor { shape := #[batch, qHeads, seq, headDim], dtype := .BFloat16 } × Tensor { shape := #[batch, qHeads, seq, headDim], dtype := .Float32 } × Tensor { shape := #[batch, kvHeads, seq, headDim], dtype := .Float32 } × Tensor { shape := #[batch, kvHeads, seq, headDim], dtype := .Float32 }) := do
  let q ← makeLeaf qBase
  let k ← makeLeaf kBase
  let v ← makeLeaf vBase
  let out := Tensor.attentionGQAQKV q k v 0.0 isCausal true
  backwardWithCotangent out dOBase
  let _ ← torch.cuda_synchronize
  pure (out, (← Tensor.gradOf q).toFloat32, (← Tensor.gradOf k).toFloat32, (← Tensor.gradOf v).toFloat32)

private def tyrEqualOnce
    {batch heads seq headDim : UInt64}
    (qBase kBase vBase dOBase : Tensor { shape := #[batch, heads, seq, headDim], dtype := .BFloat16 })
    (isCausal : Bool)
    : IO (DispatchRoute × Tensor { shape := #[batch, heads, seq, headDim], dtype := .BFloat16 } × Tensor { shape := #[batch, heads, seq, headDim], dtype := .Float32 } × Tensor { shape := #[batch, heads, seq, headDim], dtype := .Float32 } × Tensor { shape := #[batch, heads, seq, headDim], dtype := .Float32 }) := do
  let q ← makeLeaf qBase
  let k ← makeLeaf kBase
  let v ← makeLeaf vBase
  let (route, out) := flashAttnWithRoute q k v none 0.0 isCausal none false
  backwardWithCotangent out dOBase
  let _ ← torch.cuda_synchronize
  pure (route, out, (← Tensor.gradOf q).toFloat32, (← Tensor.gradOf k).toFloat32, (← Tensor.gradOf v).toFloat32)

private def tyrGqaOnce
    {batch qHeads kvHeads seq headDim : UInt64}
    (qBase : Tensor { shape := #[batch, qHeads, seq, headDim], dtype := .BFloat16 })
    (kBase vBase : Tensor { shape := #[batch, kvHeads, seq, headDim], dtype := .BFloat16 })
    (dOBase : Tensor { shape := #[batch, qHeads, seq, headDim], dtype := .BFloat16 })
    (isCausal : Bool)
    : IO (DispatchRoute × Tensor { shape := #[batch, qHeads, seq, headDim], dtype := .BFloat16 } × Tensor { shape := #[batch, qHeads, seq, headDim], dtype := .Float32 } × Tensor { shape := #[batch, kvHeads, seq, headDim], dtype := .Float32 } × Tensor { shape := #[batch, kvHeads, seq, headDim], dtype := .Float32 }) := do
  let q ← makeLeaf qBase
  let k ← makeLeaf kBase
  let v ← makeLeaf vBase
  let (route, out) := flashAttnWithRoute q k v none 0.0 isCausal none true
  backwardWithCotangent out dOBase
  let _ ← torch.cuda_synchronize
  pure (route, out, (← Tensor.gradOf q).toFloat32, (← Tensor.gradOf k).toFloat32, (← Tensor.gradOf v).toFloat32)

private def sdpaEqualForward {batch heads seq headDim : UInt64}
    (qBase kBase vBase : Tensor { shape := #[batch, heads, seq, headDim], dtype := .BFloat16 })
    (isCausal : Bool)
    : Tensor { shape := #[batch, heads, seq, headDim], dtype := .BFloat16 } :=
  Tensor.attentionGQA qBase kBase vBase 0.0 isCausal

private def tyrEqualForward {batch heads seq headDim : UInt64}
    (qBase kBase vBase : Tensor { shape := #[batch, heads, seq, headDim], dtype := .BFloat16 })
    (isCausal : Bool)
    : DispatchRoute × Tensor { shape := #[batch, heads, seq, headDim], dtype := .BFloat16 } :=
  flashAttnWithRoute qBase kBase vBase none 0.0 isCausal none false

private def runEqualHeadsFwdBwd
    {batch heads seq headDim : UInt64}
    (runMeta : RunMeta)
    (c : BenchCase)
    (backend : Backend)
    (jsonlOut? : Option System.FilePath)
    (emitJsonlStdout : Bool)
    : IO SummaryRow := do
  let device := Device.CUDA 0
  let qBase := (← Tensor.randn #[batch, heads, seq, headDim] false device).toBFloat16
  let kBase := (← Tensor.randn #[batch, heads, seq, headDim] false device).toBFloat16
  let vBase := (← Tensor.randn #[batch, heads, seq, headDim] false device).toBFloat16
  let dOBase := (← Tensor.randn #[batch, heads, seq, headDim] false device).toBFloat16

  let (outRef, dqRef, dkRef, dvRef) ← sdpaEqualRef qBase kBase vBase dOBase c.isCausal

  match backend with
  | .flashAttention =>
      pure <| mkUnsupportedSummary runMeta c backend flashAttentionUnsupportedReason
  | .torchSdpa =>
      let corr : Correctness := {
        routeActual := "n/a"
        outOk := some true
        dqOk := some true
        dkOk := some true
        dvOk := some true
        correctnessOk := some true
        outMae := some 0.0
        dqMae := some 0.0
        dkMae := some 0.0
        dvMae := some 0.0
        outMax := some 0.0
        dqMax := some 0.0
        dkMax := some 0.0
        dvMax := some 0.0
        outRelL2 := some 0.0
        dqRelL2 := some 0.0
        dkRelL2 := some 0.0
        dvRelL2 := some 0.0
      }
      let action : IO Unit := do
        let q ← makeLeaf qBase
        let k ← makeLeaf kBase
        let v ← makeLeaf vBase
        let out := Tensor.attentionGQA q k v 0.0 c.isCausal
        backwardWithCotangent out dOBase
        let _ ← torch.cuda_synchronize
      let latencies ← benchmarkAction runMeta.warmupIters runMeta.timedIters runMeta.repeats action
      for idx in [:latencies.size] do
        emitJsonl? jsonlOut? emitJsonlStdout (Lean.toJson {
          runId := runMeta.runId
          caseId := c.caseId
          caseGroup := c.caseGroup
          backendRequested := Backend.name backend
          backendExecuted := "torch_sdpa"
          repeatIdx := idx
          latencyMs := latencies[idx]!
          : RepeatRow
        })
      pure <| makeSummary runMeta c backend "torch_sdpa" "reference" latencies corr
  | .tyrRuntime =>
      let (route, out, dq, dk, dv) ← tyrEqualOnce qBase kBase vBase dOBase c.isCausal
      let routeActual := routeName route
      let routeOk := routeActual == c.expectedTyrRoute
      let outOk := Tensor.allclose outRef out 3e-2 3e-2
      let dqOk := Tensor.allclose dqRef dq 3e-2 3e-2
      let dkOk := Tensor.allclose dkRef dk 3e-2 3e-2
      let dvOk := Tensor.allclose dvRef dv 3e-2 3e-2
      let corr : Correctness := {
        routeActual := routeActual
        routeOk := some routeOk
        outOk := some outOk
        dqOk := some dqOk
        dkOk := some dkOk
        dvOk := some dvOk
        correctnessOk := some (routeOk && outOk && dqOk && dkOk && dvOk)
        outMae := some <| Tensor.item (Tensor.meanAll (Tensor.abs (out - outRef)))
        dqMae := some <| Tensor.item (Tensor.meanAll (Tensor.abs (dq - dqRef)))
        dkMae := some <| Tensor.item (Tensor.meanAll (Tensor.abs (dk - dkRef)))
        dvMae := some <| Tensor.item (Tensor.meanAll (Tensor.abs (dv - dvRef)))
        outMax := some <| Tensor.item (Tensor.maxAll (Tensor.abs (out - outRef)))
        dqMax := some <| Tensor.item (Tensor.maxAll (Tensor.abs (dq - dqRef)))
        dkMax := some <| Tensor.item (Tensor.maxAll (Tensor.abs (dk - dkRef)))
        dvMax := some <| Tensor.item (Tensor.maxAll (Tensor.abs (dv - dvRef)))
        outRelL2 := some <| relL2 out outRef
        dqRelL2 := some <| relL2 dq dqRef
        dkRelL2 := some <| relL2 dk dkRef
        dvRelL2 := some <| relL2 dv dvRef
      }
      let action : IO Unit := do
        let q ← makeLeaf qBase
        let k ← makeLeaf kBase
        let v ← makeLeaf vBase
        let (_, outTimed) := flashAttnWithRoute q k v none 0.0 c.isCausal none false
        backwardWithCotangent outTimed dOBase
        let _ ← torch.cuda_synchronize
      let latencies ← benchmarkAction runMeta.warmupIters runMeta.timedIters runMeta.repeats action
      for idx in [:latencies.size] do
        emitJsonl? jsonlOut? emitJsonlStdout (Lean.toJson {
          runId := runMeta.runId
          caseId := c.caseId
          caseGroup := c.caseGroup
          backendRequested := Backend.name backend
          backendExecuted := if routeActual == "tkKernel" then "tyr_tk_runtime_native" else "tyr_tk_runtime_portable"
          repeatIdx := idx
          latencyMs := latencies[idx]!
          : RepeatRow
        })
      pure <|
        makeSummary runMeta c backend
          (if routeActual == "tkKernel" then "tyr_tk_runtime_native" else "tyr_tk_runtime_portable")
          (if routeActual == "tkKernel" then "native" else "portable")
          latencies corr

private def runGqaFwdBwd
    {batch qHeads kvHeads seq headDim : UInt64}
    (runMeta : RunMeta)
    (c : BenchCase)
    (backend : Backend)
    (jsonlOut? : Option System.FilePath)
    (emitJsonlStdout : Bool)
    : IO SummaryRow := do
  let device := Device.CUDA 0
  let qBase := (← Tensor.randn #[batch, qHeads, seq, headDim] false device).toBFloat16
  let kBase := (← Tensor.randn #[batch, kvHeads, seq, headDim] false device).toBFloat16
  let vBase := (← Tensor.randn #[batch, kvHeads, seq, headDim] false device).toBFloat16
  let dOBase := (← Tensor.randn #[batch, qHeads, seq, headDim] false device).toBFloat16

  let (outRef, dqRef, dkRef, dvRef) ← sdpaGqaRef qBase kBase vBase dOBase c.isCausal

  match backend with
  | .flashAttention =>
      pure <| mkUnsupportedSummary runMeta c backend flashAttentionUnsupportedReason
  | .torchSdpa =>
      let corr : Correctness := {
        routeActual := "n/a"
        outOk := some true
        dqOk := some true
        dkOk := some true
        dvOk := some true
        correctnessOk := some true
        outMae := some 0.0
        dqMae := some 0.0
        dkMae := some 0.0
        dvMae := some 0.0
        outMax := some 0.0
        dqMax := some 0.0
        dkMax := some 0.0
        dvMax := some 0.0
        outRelL2 := some 0.0
        dqRelL2 := some 0.0
        dkRelL2 := some 0.0
        dvRelL2 := some 0.0
      }
      let action : IO Unit := do
        let q ← makeLeaf qBase
        let k ← makeLeaf kBase
        let v ← makeLeaf vBase
        let out := Tensor.attentionGQAQKV q k v 0.0 c.isCausal true
        backwardWithCotangent out dOBase
        let _ ← torch.cuda_synchronize
      let latencies ← benchmarkAction runMeta.warmupIters runMeta.timedIters runMeta.repeats action
      for idx in [:latencies.size] do
        emitJsonl? jsonlOut? emitJsonlStdout (Lean.toJson {
          runId := runMeta.runId
          caseId := c.caseId
          caseGroup := c.caseGroup
          backendRequested := Backend.name backend
          backendExecuted := "torch_sdpa"
          repeatIdx := idx
          latencyMs := latencies[idx]!
          : RepeatRow
        })
      pure <| makeSummary runMeta c backend "torch_sdpa" "reference" latencies corr
  | .tyrRuntime =>
      let (route, out, dq, dk, dv) ← tyrGqaOnce qBase kBase vBase dOBase c.isCausal
      let routeActual := routeName route
      let routeOk := routeActual == c.expectedTyrRoute
      let outOk := Tensor.allclose outRef out 1e-5 1e-5
      let dqOk := Tensor.allclose dqRef dq 1e-5 1e-5
      let dkOk := Tensor.allclose dkRef dk 1e-5 1e-5
      let dvOk := Tensor.allclose dvRef dv 1e-5 1e-5
      let corr : Correctness := {
        routeActual := routeActual
        routeOk := some routeOk
        outOk := some outOk
        dqOk := some dqOk
        dkOk := some dkOk
        dvOk := some dvOk
        correctnessOk := some (routeOk && outOk && dqOk && dkOk && dvOk)
        outMae := some <| Tensor.item (Tensor.meanAll (Tensor.abs (out - outRef)))
        dqMae := some <| Tensor.item (Tensor.meanAll (Tensor.abs (dq - dqRef)))
        dkMae := some <| Tensor.item (Tensor.meanAll (Tensor.abs (dk - dkRef)))
        dvMae := some <| Tensor.item (Tensor.meanAll (Tensor.abs (dv - dvRef)))
        outMax := some <| Tensor.item (Tensor.maxAll (Tensor.abs (out - outRef)))
        dqMax := some <| Tensor.item (Tensor.maxAll (Tensor.abs (dq - dqRef)))
        dkMax := some <| Tensor.item (Tensor.maxAll (Tensor.abs (dk - dkRef)))
        dvMax := some <| Tensor.item (Tensor.maxAll (Tensor.abs (dv - dvRef)))
        outRelL2 := some <| relL2 out outRef
        dqRelL2 := some <| relL2 dq dqRef
        dkRelL2 := some <| relL2 dk dkRef
        dvRelL2 := some <| relL2 dv dvRef
      }
      let action : IO Unit := do
        let q ← makeLeaf qBase
        let k ← makeLeaf kBase
        let v ← makeLeaf vBase
        let (_, outTimed) := flashAttnWithRoute q k v none 0.0 c.isCausal none true
        backwardWithCotangent outTimed dOBase
        let _ ← torch.cuda_synchronize
      let latencies ← benchmarkAction runMeta.warmupIters runMeta.timedIters runMeta.repeats action
      for idx in [:latencies.size] do
        emitJsonl? jsonlOut? emitJsonlStdout (Lean.toJson {
          runId := runMeta.runId
          caseId := c.caseId
          caseGroup := c.caseGroup
          backendRequested := Backend.name backend
          backendExecuted := if routeActual == "tkKernel" then "tyr_tk_runtime_native" else "tyr_tk_runtime_portable"
          repeatIdx := idx
          latencyMs := latencies[idx]!
          : RepeatRow
        })
      pure <|
        makeSummary runMeta c backend
          (if routeActual == "tkKernel" then "tyr_tk_runtime_native" else "tyr_tk_runtime_portable")
          (if routeActual == "tkKernel" then "native" else "portable")
          latencies corr

private def runEqualHeadsForwardOnly
    {batch heads seq headDim : UInt64}
    (runMeta : RunMeta)
    (c : BenchCase)
    (backend : Backend)
    (jsonlOut? : Option System.FilePath)
    (emitJsonlStdout : Bool)
    : IO SummaryRow := do
  let device := Device.CUDA 0
  let qBase := (← Tensor.randn #[batch, heads, seq, headDim] false device).toBFloat16
  let kBase := (← Tensor.randn #[batch, heads, seq, headDim] false device).toBFloat16
  let vBase := (← Tensor.randn #[batch, heads, seq, headDim] false device).toBFloat16
  let outRef := sdpaEqualForward qBase kBase vBase c.isCausal
  match backend with
  | .flashAttention =>
      pure <| mkUnsupportedSummary runMeta c backend flashAttentionUnsupportedReason
  | .torchSdpa =>
      let corr : Correctness := {
        routeActual := "n/a"
        outOk := some true
        correctnessOk := some true
        outMae := some 0.0
        outMax := some 0.0
        outRelL2 := some 0.0
      }
      let action : IO Unit := do
        let _out := Tensor.attentionGQA qBase kBase vBase 0.0 c.isCausal
        let _ ← torch.cuda_synchronize
        pure ()
      let latencies ← benchmarkAction runMeta.warmupIters runMeta.timedIters runMeta.repeats action
      for idx in [:latencies.size] do
        emitJsonl? jsonlOut? emitJsonlStdout (Lean.toJson {
          runId := runMeta.runId
          caseId := c.caseId
          caseGroup := c.caseGroup
          backendRequested := Backend.name backend
          backendExecuted := "torch_sdpa"
          repeatIdx := idx
          latencyMs := latencies[idx]!
          : RepeatRow
        })
      pure <| makeSummary runMeta c backend "torch_sdpa" "reference" latencies corr
  | .tyrRuntime =>
      let (route, out) := tyrEqualForward qBase kBase vBase c.isCausal
      let routeActual := routeName route
      let routeOk := routeActual == c.expectedTyrRoute
      let outOk := Tensor.allclose outRef out 3e-2 3e-2
      let corr : Correctness := {
        routeActual := routeActual
        routeOk := some routeOk
        outOk := some outOk
        correctnessOk := some (routeOk && outOk)
        outMae := some <| Tensor.item (Tensor.meanAll (Tensor.abs (out - outRef)))
        outMax := some <| Tensor.item (Tensor.maxAll (Tensor.abs (out - outRef)))
        outRelL2 := some <| relL2 out outRef
      }
      let action : IO Unit := do
        let _ := tyrEqualForward qBase kBase vBase c.isCausal
        let _ ← torch.cuda_synchronize
        pure ()
      let latencies ← benchmarkAction runMeta.warmupIters runMeta.timedIters runMeta.repeats action
      for idx in [:latencies.size] do
        emitJsonl? jsonlOut? emitJsonlStdout (Lean.toJson {
          runId := runMeta.runId
          caseId := c.caseId
          caseGroup := c.caseGroup
          backendRequested := Backend.name backend
          backendExecuted := if routeActual == "tkKernel" then "tyr_tk_runtime_native" else "tyr_tk_runtime_portable"
          repeatIdx := idx
          latencyMs := latencies[idx]!
          : RepeatRow
        })
      pure <|
        makeSummary runMeta c backend
          (if routeActual == "tkKernel" then "tyr_tk_runtime_native" else "tyr_tk_runtime_portable")
          (if routeActual == "tkKernel" then "native" else "portable")
          latencies corr

private def runCase
    (runMeta : RunMeta)
    (c : BenchCase)
    (backend : Backend)
    (jsonlOut? : Option System.FilePath)
    (emitJsonlStdout : Bool)
    : IO SummaryRow := do
  try
    match c.caseId, backend with
    | "native_dense_128x64", _ =>
        runEqualHeadsFwdBwd
          (batch := 1) (heads := 1) (seq := 128) (headDim := 64)
          runMeta c backend jsonlOut? emitJsonlStdout
    | "native_dense_768x64", _ =>
        runEqualHeadsFwdBwd
          (batch := 1) (heads := 1) (seq := 768) (headDim := 64)
          runMeta c backend jsonlOut? emitJsonlStdout
    | "portable_dense_96x64", _ =>
        runEqualHeadsFwdBwd
          (batch := 1) (heads := 1) (seq := 96) (headDim := 64)
          runMeta c backend jsonlOut? emitJsonlStdout
    | "portable_causal_128x64", _ =>
        runEqualHeadsFwdBwd
          (batch := 1) (heads := 1) (seq := 128) (headDim := 64)
          runMeta c backend jsonlOut? emitJsonlStdout
    | "portable_gqa_96x64", _ =>
        runGqaFwdBwd
          (batch := 1) (qHeads := 4) (kvHeads := 2) (seq := 96) (headDim := 64)
          runMeta c backend jsonlOut? emitJsonlStdout
    | "future_flash_256x64", .flashAttention =>
        pure <| mkUnsupportedSummary runMeta c backend flashAttentionUnsupportedReason
    | "future_flash_256x64", _ =>
        runEqualHeadsForwardOnly
          (batch := 1) (heads := 1) (seq := 256) (headDim := 64)
          runMeta c backend jsonlOut? emitJsonlStdout
    | _, .flashAttention =>
        pure <| mkUnsupportedSummary runMeta c backend flashAttentionUnsupportedReason
    | _, _ =>
        pure <| mkErrorSummary runMeta c backend s!"unhandled case dispatch: {c.caseId}"
  catch e =>
    pure <| mkErrorSummary runMeta c backend e.toString

private def sdpaP50ForCase (rows : Array SummaryRow) (caseId : String) : Option Float :=
  rows.foldl
    (fun acc candidate =>
      match acc with
      | some _ => acc
      | none =>
          if candidate.caseId == caseId && candidate.backendExecuted == "torch_sdpa" then
            candidate.latencyMsP50
          else
            none)
    none

private def withSpeedups (rows : Array SummaryRow) : Array SummaryRow :=
  rows.map fun row =>
    match row.latencyMsP50 with
    | none => row
    | some ms =>
        let sdpa? := sdpaP50ForCase rows row.caseId
        let speedup? :=
          match sdpa? with
          | some sdpaMs =>
              if ms > 0.0 then some (sdpaMs / ms) else none
          | none => none
        { row with speedupVsSdpaP50 := speedup? }

private def printCaseList : IO Unit := do
  for c in caseMatrix do
    IO.println s!"{c.caseId} group={c.caseGroup} mode={c.mode} batch={c.batch} q_heads={c.qHeads} kv_heads={c.kvHeads} q_seq={c.qSeq} kv_seq={c.kvSeq} head_dim={c.headDim} causal={c.isCausal} gqa={c.enableGqa} expected_tyr_route={c.expectedTyrRoute}"

private def printBackendList : IO Unit := do
  for b in Backend.all do
    IO.println (Backend.name b)

def main (args : List String) : IO UInt32 := do
  if hasFlag args "--list-cases" then
    printCaseList
    return 0

  if hasFlag args "--list-backends" then
    printBackendList
    return 0

  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return 1

  let warmup := parseArgNat args "--warmup" 20
  let iters := parseArgNat args "--iters" 200
  let repeats := parseArgNat args "--repeats" 5
  let seedNat := parseArgNat args "--seed" 20260422
  let caseSel := (parseArgString? args "--case").getD "all"
  let backendSel := (parseArgString? args "--backend").getD "all"
  let jsonlOut? := (parseArgString? args "--jsonl-out").map System.FilePath.mk
  let emitJsonlStdout := hasFlag args "--jsonl-stdout"
  let strict := hasFlag args "--strict"

  if iters == 0 || repeats == 0 then
    IO.eprintln "--iters and --repeats must be > 0"
    return 1

  let cases := parseCaseSelection caseSel
  let backends := parseBackendSelection backendSel
  if cases.isEmpty then
    IO.eprintln s!"No benchmark cases matched selector: {caseSel}"
    return 1
  if backends.isEmpty then
    IO.eprintln s!"No backends matched selector: {backendSel}"
    return 1

  let runId := s!"flash_attn_bench_{← IO.monoMsNow}"
  let gitSha := (← shellTrim? "git" #["rev-parse", "--short", "HEAD"]).getD "unknown"
  let hostname := (← shellTrim? "hostname").getD "unknown"
  let gpuName := (← shellTrim? "bash" #["-lc", "nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1"]).getD "unknown"
  let cudaVisibleDevices := (← IO.getEnv "CUDA_VISIBLE_DEVICES").getD "0"
  let runMeta : RunMeta := {
    runId := runId
    gitSha := gitSha
    hostname := hostname
    gpuName := gpuName
    cudaVisibleDevices := cudaVisibleDevices
    warmupIters := warmup
    timedIters := iters
    repeats := repeats
    seed := seedNat.toUInt64
  }

  torch.manualSeed runMeta.seed
  emitJsonl? jsonlOut? emitJsonlStdout (Lean.toJson (mkMetaRow runMeta cases backends))

  let mut rows : Array SummaryRow := #[]
  let mut hadError := false
  let mut hadIncorrect := false

  for c in cases do
    for backend in backends do
      let row ← runCase runMeta c backend jsonlOut? emitJsonlStdout
      rows := rows.push row

  let rowsFinal := withSpeedups rows
  for row in rowsFinal do
    emitJsonl? jsonlOut? emitJsonlStdout (Lean.toJson row)
    printHumanSummary row
    if row.supportStatus == "error" then
      hadError := true
    match row.correctnessOk with
    | some false => hadIncorrect := true
    | _ => pure ()

  pure <| if strict && (hadError || hadIncorrect) then 1 else 0

end Examples.GPU

def main : List String → IO UInt32 := Examples.GPU.main
