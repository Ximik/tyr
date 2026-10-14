import Tyr.Torch
import Tyr.Typed.Ops

/-! Shared helpers for GPU end-to-end parity checks. -/

namespace Examples.GPU

open torch

private def normalizedEnvOr (name fallback : String) (normalize : String → String) : IO String := do
  match (← IO.getEnv name) with
  | some value =>
      let trimmed := value.trimAscii.toString
      if trimmed.isEmpty then
        pure fallback
      else
        pure (normalize trimmed)
  | none => pure fallback

/-- GPU target selected by the surrounding shell harness, if any. -/
def gpuTarget : IO String :=
  normalizedEnvOr "TYR_GPU_TARGET" "H100" String.toUpper

/-- Logical GPU family selected by the surrounding shell harness, if any. -/
def gpuFamily : IO String :=
  normalizedEnvOr "TYR_GPU_FAMILY" "hopper" String.toLower

/-- True when the current target is one of the provided architecture labels. -/
def gpuTargetIsAny (targets : Array String) : IO Bool := do
  let target ← gpuTarget
  pure <| targets.any fun candidate => target == candidate

/-- True when the suite is running in Blackwell-family mode. -/
def isBlackwellFamily : IO Bool := do
  let family ← gpuFamily
  pure (family == "blackwell")

/-- Materialized tensor comparison metrics for one parity check. -/
structure TensorCheck where
  label : String
  ok : Bool
  mae : Float
  maxErr : Float
  rtol : Float
  atol : Float
  deriving Inhabited

/-- Deterministically seed tensor generation for fixture regeneration. -/
def seedFixtures (suite : String) (seed : UInt64) : IO Unit := do
  torch.manualSeed seed
  IO.println s!"[{suite}] seed={seed}"

/-- Return `true` when CUDA is available, otherwise print a consistent error. -/
def requireCuda (suite : String) : IO Bool := do
  if !(← torch.cuda_is_available) then
    IO.eprintln s!"[{suite}] CUDA is not available on this host."
    return false
  pure true

/-- Compare equal shapes across storage dtypes, computing GPU parity metrics in Float32. -/
def compareTensors {s : Shape} {expectedDtype actualDtype : DType}
    (label : String) (expected : DTensor s expectedDtype) (actual : DTensor s actualDtype)
    (rtol atol : Float) : TensorCheck :=
  let expected := expected.toFloat32
  let actual := actual.toFloat32
  let ok := Tensor.allclose expected actual rtol atol
  let diff := (actual - expected).abs
  let mae := diff.meanAll.item
  let maxErr := diff.maxAll.item
  { label, ok, mae, maxErr, rtol, atol }

/-- Print one tensor comparison in a stable machine-readable format. -/
def logTensorCheck (check : TensorCheck) : IO Unit := do
  IO.println
    s!"{check.label} ok={check.ok} mae={check.mae} max={check.maxErr} rtol={check.rtol} atol={check.atol}"

/-- Report allclose plus mean/max absolute error for one tensor comparison. -/
def reportTensorComparison {s : Shape} {expectedDtype actualDtype : DType}
    (label : String) (expected : DTensor s expectedDtype) (actual : DTensor s actualDtype) (rtol atol : Float) : IO Bool := do
  let check := compareTensors label expected actual rtol atol
  logTensorCheck check
  pure check.ok

/-- Print a compact one-line summary for a suite of checks. -/
def reportCheckSummary (suite : String) (checks : Array (String × Bool)) : IO Bool := do
  let ok := checks.foldl (init := true) fun acc (_, passed) => acc && passed
  let rendered := String.intercalate " " <| checks.toList.map fun (label, passed) =>
    s!"{label}={passed}"
  IO.println s!"[{suite}] summary ok={ok} {rendered}"
  pure ok

/-- Optional hook for a vendored ThunderKittens reference runner.

The external command is discovered via `TYR_GPU_VENDORED_REF_RUNNER` and is invoked as:

`$TYR_GPU_VENDORED_REF_RUNNER <suite-name> <fixture-dir>`

It should exit with status 0 on success and nonzero on mismatch/failure.
-/
private def resolveVendoredRunner : IO (Option System.FilePath) := do
  match (← IO.getEnv "TYR_GPU_VENDORED_REF_RUNNER") with
  | some runner => pure (some ⟨runner⟩)
  | none =>
    let defaultRunner : System.FilePath := ⟨"scripts/gpu/run_vendored_reference.sh"⟩
    if ← defaultRunner.pathExists then
      pure (some defaultRunner)
    else
      pure none

def runVendoredReferenceIfConfigured
    (suite : String) (fixtureDir : System.FilePath) : IO Bool := do
  match (← resolveVendoredRunner) with
  | none =>
    IO.println s!"[{suite}] vendored_ref configured=false"
    pure true
  | some runner =>
    let out ← IO.Process.output {
      cmd := runner.toString
      args := #[suite, fixtureDir.toString]
    }
    let stdout := out.stdout.trimAscii.toString
    let unsupported := stdout.contains "vendored_ref unsupported=true"
    if !stdout.isEmpty then
      IO.println stdout
    if unsupported then
      IO.println s!"[{suite}] vendored_ref skipped=true"
      pure true
    else if out.exitCode ≠ 0 then
      let stderr := out.stderr.trimAscii.toString
      if !stderr.isEmpty then
        IO.eprintln stderr
      IO.eprintln s!"[{suite}] vendored_ref ok=false exit_code={out.exitCode}"
      pure false
    else
      IO.println s!"[{suite}] vendored_ref ok=true"
      pure true

end Examples.GPU
