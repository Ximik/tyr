/-
  Tests/TestPipeline.lean

  Tests for Pipeline orchestration (Tyr/Pipeline.lean).
-/
import Tyr.Pipeline
import LeanTest

namespace Tests.Pipeline

open torch.Pipeline

/-! ## Float Assertion Helpers -/

/-- Assert two floats are approximately equal within epsilon -/
def assertFloatEq (actual expected : Float) (epsilon : Float := 0.001)
    (msg : String := "") : IO Unit := do
  let diff := Float.abs (actual - expected)
  if diff > epsilon then
    LeanTest.fail s!"{msg}: expected {expected}, got {actual} (diff {diff} > epsilon {epsilon})"

/-- Assert a float is less than a threshold -/
def assertFloatLt (actual threshold : Float) (msg : String := "") : IO Unit := do
  if actual >= threshold then
    LeanTest.fail s!"{msg}: expected {actual} < {threshold}"

/-- Assert a float is greater than a threshold -/
def assertFloatGt (actual threshold : Float) (msg : String := "") : IO Unit := do
  if actual <= threshold then
    LeanTest.fail s!"{msg}: expected {actual} > {threshold}"

/-- Assert a float is in range [lo, hi] -/
def assertFloatInRange (actual lo hi : Float) (msg : String := "") : IO Unit := do
  if actual < lo || actual > hi then
    LeanTest.fail s!"{msg}: expected {actual} in [{lo}, {hi}]"

/-! ## RetryPolicy Tests -/

@[test]
def testRetryDelayExponentialBackoff : IO Unit := do
  let policy : RetryPolicy := { initialDelayMs := 100, backoffMultiplier := 2.0 }
  -- 100 * 2^0 = 100, 100 * 2^1 = 200, 100 * 2^2 = 400
  LeanTest.assertEqual (policy.delayForAttempt 0) 100
  LeanTest.assertEqual (policy.delayForAttempt 1) 200
  LeanTest.assertEqual (policy.delayForAttempt 2) 400

@[test]
def testRetryDelayMaxCap : IO Unit := do
  let policy : RetryPolicy := { initialDelayMs := 1000, maxDelayMs := 5000, backoffMultiplier := 2.0 }
  -- At attempt 10: 1000 * 2^10 = 1024000, but should be capped at 5000
  let delay := policy.delayForAttempt 10
  LeanTest.assertTrue (delay <= 5000) s!"delay {delay} should be capped at maxDelayMs 5000"

/-! ## StageStatus Serialization Tests -/

@[test]
def testStageStatusJsonRoundTrip : IO Unit := do
  -- Test all variants
  for status in [StageStatus.pending, .running, .completed, .failed "test error"] do
    let json := toJson status
    match (fromJson? json : Except String StageStatus) with
    | .ok status' => LeanTest.assertTrue (status == status') "status should round-trip"
    | .error e => LeanTest.fail s!"Failed to parse: {e}"

/-! ## StageInfo Tests -/

@[test]
def testStageInfoJsonRoundTrip : IO Unit := do
  let info : StageInfo := {
    name := "test-stage"
    status := .completed
    startTime := some 1000
    endTime := some 2000
    metrics := [("loss", "0.5"), ("tokens", "1024")]
    retryCount := 1
  }
  let json := toJson info
  match (fromJson? json : Except String StageInfo) with
  | .ok info' =>
    LeanTest.assertEqual info.name info'.name
    LeanTest.assertEqual info.metrics info'.metrics
    LeanTest.assertEqual info.retryCount info'.retryCount
  | .error e => LeanTest.fail s!"Failed to parse StageInfo: {e}"

@[test]
def testStageInfoDurationMs : IO Unit := do
  let info : StageInfo := {
    name := "test"
    status := .completed
    startTime := some 1000
    endTime := some 5000
  }
  LeanTest.assertEqual info.durationMs (some 4000)

/-! ## PipelineCheckpoint Serialization Tests -/

@[test]
def testPipelineCheckpointJsonRoundTrip : IO Unit := do
  let checkpoint : PipelineCheckpoint := {
    stages := #[
      { name := "stage1", status := .completed, startTime := some 100, endTime := some 200,
        metrics := [("loss", "0.25")] },
      { name := "stage2", status := .failed "error", startTime := some 300, endTime := some 400,
        metrics := [("status", "failed")] }
    ]
    timestamp := 12345
    runId := "test-run-123"
  }
  let json := toJson checkpoint
  match (fromJson? json : Except String PipelineCheckpoint) with
  | .ok checkpoint' =>
    LeanTest.assertEqual checkpoint.stages.size checkpoint'.stages.size
    LeanTest.assertEqual checkpoint.runId checkpoint'.runId
    LeanTest.assertEqual checkpoint.stages[0]!.metrics checkpoint'.stages[0]!.metrics
  | .error e => LeanTest.fail s!"Failed to parse PipelineCheckpoint: {e}"

@[test]
def testStageFailFastFalseContinues : IO Unit := do
  let ts ← IO.monoMsNow
  let baseDir := s!"/tmp/tyr_pipeline_failfast_{ts}"
  let cfg : PipelineConfig := {
    baseDir := baseDir
    checkpointAfterStage := false
    resumeFromCheckpoint := false
    failFast := false
  }
  let initial ← initPipeline cfg
  let (_, finalState) ← (do
    stage "fails" (throw <| IO.userError "boom")
    stage "continues" (pure ())
  ).run initial

  LeanTest.assertEqual finalState.stages.size 2
  match finalState.stages[0]!.status with
  | .failed err => LeanTest.assertTrue (err.containsSubstr "boom") "first stage should fail"
  | _ => LeanTest.fail "first stage should be marked failed"
  match finalState.stages[1]!.status with
  | .completed => pure ()
  | _ => LeanTest.fail "second stage should complete when failFast is false"

/-! ## Duration Formatting Tests -/

@[test]
def testFormatDuration : IO Unit := do
  LeanTest.assertEqual (formatDuration 5000) "5s"
  LeanTest.assertEqual (formatDuration 65000) "1m5s"
  LeanTest.assertEqual (formatDuration 3665000) "1h1m5s"

/-! ## Report Section Markdown Tests -/

@[test]
def testReportSectionMarkdown : IO Unit := do
  let header := ReportSection.header "content"
  LeanTest.assertTrue (header.toMarkdown.containsSubstr "# Training Report") "header format"

  let stage := ReportSection.stage "pretrain" "Duration: 10m"
  LeanTest.assertTrue (stage.toMarkdown.containsSubstr "## pretrain") "stage format"

  let metrics := ReportSection.metrics "loss" [("train", "0.5")]
  LeanTest.assertTrue (metrics.toMarkdown.containsSubstr "train: 0.5") "metrics format"

end Tests.Pipeline
