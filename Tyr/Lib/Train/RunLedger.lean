import Lean.Data.Json.Basic
import Lean.Data.Json.Printer
import Lean.Data.Json.FromToJson

/-!
# Tyr.Lib.Train.RunLedger

Structured run-artifact helpers shared by Tyr training loops.

This module provides a Tinker-style local artifact ledger:
- `config.json`
- `metrics.jsonl`
- `checkpoints.jsonl`
- `checkpoints/`

It also defines simple evaluator scheduling utilities that can be reused by
recipe-style training entrypoints.
-/

namespace torch.Train.RunLedger

open Lean (Json)

/-- How to handle an existing run directory. -/
inductive ExistingRunPolicy where
  | resume
  | overwrite
  | failIfExists
  deriving Repr, BEq, Lean.ToJson, Lean.FromJson

/-- Filesystem layout for a single training run. -/
structure RunArtifacts where
  baseDir : String
  configPath : System.FilePath
  metricsPath : System.FilePath
  checkpointsPath : System.FilePath
  checkpointsDir : System.FilePath
  reportPath : System.FilePath
  deriving Repr

/-- Construct the standard artifact layout for a base run directory. -/
def RunArtifacts.ofBaseDir (baseDir : String) : RunArtifacts :=
  {
    baseDir := baseDir
    configPath := ⟨s!"{baseDir}/config.json"⟩
    metricsPath := ⟨s!"{baseDir}/metrics.jsonl"⟩
    checkpointsPath := ⟨s!"{baseDir}/checkpoints.jsonl"⟩
    checkpointsDir := ⟨s!"{baseDir}/checkpoints"⟩
    reportPath := ⟨s!"{baseDir}/report.md"⟩
  }

/-- Append one line to a text file, creating parents when needed. -/
private def appendLine (path : System.FilePath) (line : String) : IO Unit := do
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  IO.FS.withFile path .append fun h => do
    h.putStr line
    h.putStr "\n"

/-- Check whether any standard run-artifact file already exists. -/
private def hasExistingArtifacts (artifacts : RunArtifacts) : IO Bool := do
  let configExists := ← System.FilePath.pathExists artifacts.configPath
  let metricsExists := ← System.FilePath.pathExists artifacts.metricsPath
  let checkpointsExists := ← System.FilePath.pathExists artifacts.checkpointsPath
  let reportExists := ← System.FilePath.pathExists artifacts.reportPath
  pure <| configExists || metricsExists || checkpointsExists || reportExists

/-- Overwrite the run configuration file. -/
def writeConfig [Lean.ToJson α] (artifacts : RunArtifacts) (config : α) : IO Unit := do
  IO.FS.writeFile artifacts.configPath (Lean.toJson config).pretty

/-- Ensure the standard run layout exists and write `config.json` according to policy. -/
def prepare [Lean.ToJson α]
    (artifacts : RunArtifacts)
    (config : α)
    (policy : ExistingRunPolicy := .resume) : IO Unit := do
  IO.FS.createDirAll ⟨artifacts.baseDir⟩
  IO.FS.createDirAll artifacts.checkpointsDir
  match policy with
  | .failIfExists =>
      if ← hasExistingArtifacts artifacts then
        throw <| IO.userError s!"Run directory already contains artifacts: {artifacts.baseDir}"
      writeConfig artifacts config
      IO.FS.writeFile artifacts.metricsPath ""
      IO.FS.writeFile artifacts.checkpointsPath ""
  | .overwrite =>
      writeConfig artifacts config
      IO.FS.writeFile artifacts.metricsPath ""
      IO.FS.writeFile artifacts.checkpointsPath ""
  | .resume =>
      if !(← System.FilePath.pathExists artifacts.configPath) then
        writeConfig artifacts config
      if !(← System.FilePath.pathExists artifacts.metricsPath) then
        IO.FS.writeFile artifacts.metricsPath ""
      if !(← System.FilePath.pathExists artifacts.checkpointsPath) then
        IO.FS.writeFile artifacts.checkpointsPath ""

/-- Free-form metrics payload attached to one logged event. -/
abbrev MetricFields := List (String × Json)

/-- Common metric constructors for JSONL event payloads. -/
def metricStr (name value : String) : String × Json := (name, .str value)
def metricFloat (name : String) (value : Float) : String × Json := (name, Lean.toJson value)
def metricNat (name : String) (value : Nat) : String × Json := (name, Lean.toJson value)
/-- One structured metrics row in `metrics.jsonl`. -/
structure MetricEvent where
  scope : String
  step : Option Nat := none
  metrics : MetricFields := []
  timestampMs : Nat := 0

instance : Lean.ToJson MetricEvent where
  toJson ev := .mkObj [
    ("scope", .str ev.scope),
    ("step", match ev.step with | some step => Lean.toJson step | none => .null),
    ("timestampMs", Lean.toJson ev.timestampMs),
    ("metrics", .mkObj ev.metrics)
  ]

/-- One structured checkpoint row in `checkpoints.jsonl`. -/
structure CheckpointEvent where
  name : String
  path : String
  kind : String
  step : Option Nat := none
  metadata : MetricFields := []
  timestampMs : Nat := 0

instance : Lean.ToJson CheckpointEvent where
  toJson ev := .mkObj [
    ("name", .str ev.name),
    ("path", .str ev.path),
    ("kind", .str ev.kind),
    ("step", match ev.step with | some step => Lean.toJson step | none => .null),
    ("timestampMs", Lean.toJson ev.timestampMs),
    ("metadata", .mkObj ev.metadata)
  ]

/-- Append one metrics row to `metrics.jsonl`, filling timestamp when omitted. -/
def appendMetricEvent (artifacts : RunArtifacts) (event : MetricEvent) : IO Unit := do
  let timestampMs ← if event.timestampMs == 0 then IO.monoMsNow else pure event.timestampMs
  appendLine artifacts.metricsPath (Lean.toJson { event with timestampMs := timestampMs }).compress

/-- Append one checkpoint row to `checkpoints.jsonl`, filling timestamp when omitted. -/
def appendCheckpointEvent (artifacts : RunArtifacts) (event : CheckpointEvent) : IO Unit := do
  let timestampMs ← if event.timestampMs == 0 then IO.monoMsNow else pure event.timestampMs
  appendLine artifacts.checkpointsPath (Lean.toJson { event with timestampMs := timestampMs }).compress

/-- Schedule for recipe-style evaluators. -/
structure EvalSchedule where
  every : Nat := 0
  runAtStart : Bool := false
  runAtEnd : Bool := true
  deriving Repr, BEq, Lean.ToJson, Lean.FromJson

/-- Determine whether an evaluator should run at this step. -/
def EvalSchedule.shouldRun (schedule : EvalSchedule) (step : Nat) (isLastStep : Bool := false) : Bool :=
  (schedule.runAtStart && step == 0) ||
  (schedule.every > 0 && step > 0 && step % schedule.every == 0) ||
  (schedule.runAtEnd && isLastStep)

/-- Generic evaluator with its own schedule and metric prefix. -/
structure Evaluator (α : Type) where
  name : String
  schedule : EvalSchedule := {}
  run : α → IO MetricFields

/-- Run due evaluators and prefix their metrics with `name/`. -/
def runDueEvaluators (evaluators : Array (Evaluator α)) (ctx : α) (step : Nat)
    (isLastStep : Bool := false) : IO MetricFields := do
  let mut metrics : MetricFields := []
  for evaluator in evaluators do
    if evaluator.schedule.shouldRun step isLastStep then
      let result ← evaluator.run ctx
      metrics := metrics ++ result.map (fun (k, v) => (s!"{evaluator.name}/{k}", v))
  pure metrics

end torch.Train.RunLedger
