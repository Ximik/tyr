import Tyr.Lib.Train.RunLedger
import LeanTest
import Lean.Data.Json.Basic
import Lean.Data.Json.Parser

namespace Tests.RunLedger

open torch.Train.RunLedger

@[test]
def testPrepareWritesConfigAndLedgers : IO Unit := do
  let ts ← IO.monoMsNow
  let baseDir := s!"/tmp/tyr_run_ledger_{ts}"
  let artifacts := RunArtifacts.ofBaseDir baseDir
  let cfg := [("model", "toy"), ("mode", "test")]
  prepare artifacts cfg .overwrite
  LeanTest.assertTrue (← System.FilePath.pathExists artifacts.configPath) "config.json should exist"
  LeanTest.assertTrue (← System.FilePath.pathExists artifacts.metricsPath) "metrics.jsonl should exist"
  LeanTest.assertTrue (← System.FilePath.pathExists artifacts.checkpointsPath) "checkpoints.jsonl should exist"
  let configText ← IO.FS.readFile artifacts.configPath
  LeanTest.assertTrue (configText.containsSubstr "\"model\"") "config file should contain serialized keys"

@[test]
def testAppendMetricAndCheckpointEvents : IO Unit := do
  let ts ← IO.monoMsNow
  let baseDir := s!"/tmp/tyr_run_ledger_events_{ts}"
  let artifacts := RunArtifacts.ofBaseDir baseDir
  prepare artifacts [("run", "events")] .overwrite
  appendMetricEvent artifacts {
    scope := "sft/train"
    step := some 12
    metrics := [metricFloat "loss" 1.25, metricNat "num_tokens" 32]
  }
  appendCheckpointEvent artifacts {
    name := "latest"
    path := s!"{baseDir}/checkpoints/latest.ckpt"
    kind := "model"
    step := some 12
    metadata := [metricStr "stage" "sft"]
  }
  let metricLine := (← IO.FS.readFile artifacts.metricsPath).trimAscii.toString
  let checkpointLine := (← IO.FS.readFile artifacts.checkpointsPath).trimAscii.toString
  match Lean.Json.parse metricLine with
  | .error err => LeanTest.fail s!"failed to parse metric jsonl row: {err}"
  | .ok json =>
      let scope := (json.getObjValAs? String "scope").toOption.getD ""
      LeanTest.assertEqual scope "sft/train"
  match Lean.Json.parse checkpointLine with
  | .error err => LeanTest.fail s!"failed to parse checkpoint jsonl row: {err}"
  | .ok json =>
      let kind := (json.getObjValAs? String "kind").toOption.getD ""
      LeanTest.assertEqual kind "model"

@[test]
def testEvalScheduleAndEvaluatorPrefixing : IO Unit := do
  let schedule : EvalSchedule := { every := 3, runAtStart := true, runAtEnd := true }
  LeanTest.assertTrue (schedule.shouldRun 0 false) "should run at start"
  LeanTest.assertTrue (schedule.shouldRun 3 false) "should run on cadence"
  LeanTest.assertTrue (schedule.shouldRun 5 true) "should run at end"
  LeanTest.assertTrue (!(schedule.shouldRun 2 false)) "should skip off-cadence"
  let evaluators : Array (Evaluator Nat) := #[
    {
      name := "fast"
      schedule := { every := 2 }
      run := fun n => pure [metricNat "step_seen" n]
    },
    {
      name := "final"
      schedule := { every := 0, runAtEnd := true }
      run := fun n => pure [metricNat "step_seen" n]
    }
  ]
  let midMetrics ← runDueEvaluators evaluators 4 4 false
  LeanTest.assertEqual midMetrics.length 1
  LeanTest.assertEqual midMetrics[0]!.fst "fast/step_seen"
  let finalMetrics ← runDueEvaluators evaluators 7 7 true
  LeanTest.assertEqual finalMetrics.length 2
  LeanTest.assertEqual finalMetrics[0]!.fst "fast/step_seen"
  LeanTest.assertEqual finalMetrics[1]!.fst "final/step_seen"

end Tests.RunLedger
